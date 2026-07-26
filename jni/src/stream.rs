// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use jni::objects::{GlobalRef, JByteArray, JObject, JObjectArray, JValue};
use jni::JavaVM;
use paimon_vindex_core::io::{ReadRequest, SeekRead, SeekReadCapabilities};
use std::io;
use std::sync::Arc;

/// JNI-backed input stream that delegates to Java's VectorIndexInput.
#[derive(Clone)]
pub struct JniSeekableStream {
    jvm: Arc<JavaVM>,
    stream_ref: Arc<GlobalRef>,
    capabilities: SeekReadCapabilities,
}

impl JniSeekableStream {
    pub fn new(jvm: JavaVM, stream_ref: GlobalRef, capabilities: SeekReadCapabilities) -> Self {
        JniSeekableStream {
            jvm: Arc::new(jvm),
            stream_ref: Arc::new(stream_ref),
            capabilities,
        }
    }
}

const MAX_JNI_RANGES_PER_CALL: usize = 256;

impl SeekRead for JniSeekableStream {
    /// Positional reads via VectorIndexInput.pread(long[] positions, byte[][] buffers).
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        if ranges.is_empty() {
            return Ok(());
        }

        let mut env = self
            .jvm
            .attach_current_thread()
            .map_err(|e| io::Error::other(format!("JNI attach: {}", e)))?;

        for chunk in ranges.chunks_mut(MAX_JNI_RANGES_PER_CALL) {
            let result = env
                .with_local_frame(16 + chunk.len() as i32, |env| {
                    Ok::<_, jni::errors::Error>(pread_chunk(env, self.stream_ref.as_obj(), chunk))
                })
                .map_err(|e| io::Error::other(format!("JNI local frame: {e}")))?;
            result?;
        }
        Ok(())
    }

    fn try_clone_reader(&self) -> io::Result<Option<Self>> {
        Ok(Some(self.clone()))
    }

    fn read_capabilities(&self) -> SeekReadCapabilities {
        self.capabilities
    }
}

pub fn read_capabilities(
    env: &mut jni::JNIEnv<'_>,
    stream: &JObject<'_>,
) -> Result<SeekReadCapabilities, String> {
    let mut read_hint = |name: &str| -> Result<usize, String> {
        let value = env
            .call_method(stream, name, "()J", &[])
            .map_err(|error| format!("{name}: {error}"))?
            .j()
            .map_err(|error| format!("{name}: {error}"))?;
        usize::try_from(value)
            .map_err(|_| format!("{name} must be non-negative and fit in usize, got {value}"))
    };
    Ok(SeekReadCapabilities {
        preferred_alignment_bytes: read_hint("preferredReadAlignmentBytes")?,
        preferred_window_bytes: read_hint("preferredReadWindowBytes")?,
        max_ranges_per_pread: read_hint("maxRangesPerRead")?,
    })
}

fn pread_chunk(
    env: &mut jni::JNIEnv<'_>,
    stream: &JObject<'_>,
    ranges: &mut [ReadRequest<'_>],
) -> io::Result<()> {
    let positions = env
        .new_long_array(ranges.len() as i32)
        .map_err(|e| io::Error::other(format!("JNI alloc positions: {}", e)))?;
    let position_values: Vec<i64> = ranges.iter().map(|range| range.pos as i64).collect();
    env.set_long_array_region(&positions, 0, &position_values)
        .map_err(|e| io::Error::other(format!("JNI set positions: {}", e)))?;

    let byte_array_class = env
        .find_class("[B")
        .map_err(|e| io::Error::other(format!("JNI find byte[] class: {}", e)))?;
    let buffers = env
        .new_object_array(ranges.len() as i32, byte_array_class, JObject::null())
        .map_err(|e| io::Error::other(format!("JNI alloc buffers: {}", e)))?;
    for (idx, range) in ranges.iter().enumerate() {
        let jbuf = env
            .new_byte_array(range.buf.len() as i32)
            .map_err(|e| io::Error::other(format!("JNI alloc range buffer: {}", e)))?;
        env.set_object_array_element(&buffers, idx as i32, &jbuf)
            .map_err(|e| io::Error::other(format!("JNI set buffer: {}", e)))?;
        env.delete_local_ref(jbuf)
            .map_err(|e| io::Error::other(format!("JNI delete range buffer ref: {}", e)))?;
    }

    env.call_method(
        stream,
        "pread",
        "([J[[B)V",
        &[JValue::Object(&positions), JValue::Object(&buffers)],
    )
    .map_err(|e| io::Error::other(format!("JNI pread: {}", e)))?;

    copy_java_buffers(env, &buffers, ranges)
}

fn copy_java_buffers(
    env: &mut jni::JNIEnv<'_>,
    buffers: &JObjectArray<'_>,
    ranges: &mut [ReadRequest<'_>],
) -> io::Result<()> {
    for (idx, range) in ranges.iter_mut().enumerate() {
        let obj = env
            .get_object_array_element(buffers, idx as i32)
            .map_err(|e| io::Error::other(format!("JNI get buffer: {}", e)))?;
        let jbuf = JByteArray::from(obj);
        let len = env
            .get_array_length(&jbuf)
            .map_err(|e| io::Error::other(format!("JNI get buffer length: {}", e)))?
            as usize;
        if len != range.buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Java pread returned buffer length {} != {}",
                    len,
                    range.buf.len()
                ),
            ));
        }
        if len > 0 {
            let mut signed_buf = vec![0i8; range.buf.len()];
            env.get_byte_array_region(&jbuf, 0, &mut signed_buf)
                .map_err(|e| io::Error::other(format!("JNI get_region: {}", e)))?;

            for (i, &b) in signed_buf.iter().enumerate() {
                range.buf[i] = b as u8;
            }
        }
        env.delete_local_ref(jbuf)
            .map_err(|e| io::Error::other(format!("JNI delete returned buffer ref: {}", e)))?;
    }
    Ok(())
}

/// JNI-backed output stream that delegates to Java's PositionOutputStream.
pub struct JniOutputStream {
    jvm: Arc<JavaVM>,
    stream_ref: Arc<GlobalRef>,
    pos: u64,
}

impl JniOutputStream {
    pub fn new(jvm: JavaVM, stream_ref: GlobalRef) -> Self {
        JniOutputStream {
            jvm: Arc::new(jvm),
            stream_ref: Arc::new(stream_ref),
            pos: 0,
        }
    }
}

impl paimon_vindex_core::io::SeekWrite for JniOutputStream {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut env = self
            .jvm
            .attach_current_thread()
            .map_err(|e| io::Error::other(format!("JNI attach: {}", e)))?;

        let jbuf = env
            .new_byte_array(buf.len() as i32)
            .map_err(|e| io::Error::other(format!("JNI alloc: {}", e)))?;

        let signed: Vec<i8> = buf.iter().map(|&b| b as i8).collect();
        env.set_byte_array_region(&jbuf, 0, &signed)
            .map_err(|e| io::Error::other(format!("JNI set_region: {}", e)))?;

        env.call_method(
            self.stream_ref.as_obj(),
            "write",
            "([B)V",
            &[jni::objects::JValue::Object(&jbuf)],
        )
        .map_err(|e| io::Error::other(format!("JNI write: {}", e)))?;

        self.pos += buf.len() as u64;
        Ok(())
    }

    fn pos(&self) -> u64 {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jni_seekable_stream_is_cloneable_for_parallel_diskann_batch() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<JniSeekableStream>();
    }

    #[test]
    fn jni_range_calls_are_bounded_to_one_local_reference_frame() {
        let range_count = 50_000;
        let chunk_sizes = (0..range_count)
            .collect::<Vec<_>>()
            .chunks(MAX_JNI_RANGES_PER_CALL)
            .map(<[usize]>::len)
            .collect::<Vec<_>>();

        assert_eq!(chunk_sizes.iter().sum::<usize>(), range_count);
        assert!(chunk_sizes
            .iter()
            .all(|&size| size <= MAX_JNI_RANGES_PER_CALL));
        assert!(chunk_sizes.len() > 1);
    }
}
