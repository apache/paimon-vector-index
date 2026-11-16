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
const JNI_WRITE_CHUNK_BYTES: usize = 1024 * 1024;

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
    let mut read_hint = |name: &str| -> Result<i64, String> {
        let value = env
            .call_method(stream, name, "()J", &[])
            .map_err(|error| format!("{name}: {error}"))?
            .j()
            .map_err(|error| format!("{name}: {error}"))?;
        if value < 0 {
            return Err(format!("{name} must be non-negative, got {value}"));
        }
        Ok(value)
    };
    Ok(SeekReadCapabilities {
        estimated_random_read_latency_nanos: read_hint("estimatedRandomReadLatencyNanos")? as u64,
        preferred_window_bytes: usize::try_from(read_hint("preferredReadWindowBytes")?)
            .map_err(|_| "preferredReadWindowBytes exceeds usize".to_string())?,
        max_ranges_per_pread: usize::try_from(read_hint("maxRangesPerRead")?)
            .map_err(|_| "maxRangesPerRead exceeds usize".to_string())?,
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
        if buf.is_empty() {
            return Ok(());
        }
        let mut env = self
            .jvm
            .attach_current_thread()
            .map_err(|e| io::Error::other(format!("JNI attach: {}", e)))?;

        let result = env
            .with_local_frame(8, |env| {
                let capacity = buf.len().min(JNI_WRITE_CHUNK_BYTES);
                let jbuf = env.new_byte_array(capacity as i32)?;
                Ok::<_, jni::errors::Error>(write_chunks(
                    buf,
                    &mut self.pos,
                    JNI_WRITE_CHUNK_BYTES,
                    |chunk| {
                        if chunk.len() == capacity {
                            write_chunk(env, self.stream_ref.as_obj(), &jbuf, chunk)
                        } else {
                            // The Java contract is write(byte[]), so a short final
                            // chunk needs its exact length to avoid trailing bytes.
                            let tail = env
                                .new_byte_array(chunk.len() as i32)
                                .map_err(|e| io::Error::other(format!("JNI alloc: {e}")))?;
                            let result = write_chunk(env, self.stream_ref.as_obj(), &tail, chunk);
                            // Local-frame cleanup also works with a pending Java exception.
                            result
                        }
                    },
                ))
            })
            .map_err(|e| io::Error::other(format!("JNI output local frame: {e}")))?;
        result?;

        Ok(())
    }

    fn pos(&self) -> u64 {
        self.pos
    }
}

fn write_chunks(
    buf: &[u8],
    pos: &mut u64,
    chunk_bytes: usize,
    mut write: impl FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<()> {
    if chunk_bytes == 0 || chunk_bytes > i32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid JNI write chunk size",
        ));
    }
    pos.checked_add(buf.len() as u64)
        .ok_or_else(|| io::Error::other("JNI output position overflow"))?;
    for chunk in buf.chunks(chunk_bytes) {
        write(chunk)?;
        *pos += chunk.len() as u64;
    }
    Ok(())
}

fn write_chunk(
    env: &mut jni::JNIEnv<'_>,
    stream: &JObject<'_>,
    jbuf: &JByteArray<'_>,
    buf: &[u8],
) -> io::Result<()> {
    // u8 and JNI jbyte (i8) have identical size/alignment and all bit patterns
    // are valid. Borrow the bytes rather than copying into a second Rust Vec.
    let signed = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<i8>(), buf.len()) };
    env.set_byte_array_region(jbuf, 0, signed)
        .map_err(|e| io::Error::other(format!("JNI set_region: {e}")))?;
    env.call_method(stream, "write", "([B)V", &[JValue::Object(jbuf)])
        .map_err(|e| io::Error::other(format!("JNI write: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jni_chunked_output_preserves_bytes_and_position() {
        let input = [0, 127, 128, 255, 4, 5, 6, 7];
        for limit in [1, 3, 4, 8, 20] {
            let mut pos = 17;
            let mut output = Vec::new();
            let mut sizes = Vec::new();
            write_chunks(&input, &mut pos, limit, |chunk| {
                sizes.push(chunk.len());
                output.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
            assert_eq!(output, input);
            assert_eq!(pos, 25);
            assert_eq!(sizes.len(), input.len().div_ceil(limit));
            assert!(sizes.iter().all(|&len| len <= limit));
        }
    }

    #[test]
    fn jni_chunked_output_stops_on_error_and_counts_only_successful_chunks() {
        let mut pos = 17;
        let mut calls = 0;
        let mut output = Vec::new();
        let error = write_chunks(&[0, 1, 2, 128, 255, 5, 6, 7], &mut pos, 3, |chunk| {
            calls += 1;
            if calls == 2 {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "Java write failed",
                ));
            }
            output.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap_err();
        assert_eq!(calls, 2);
        assert_eq!(output, [0, 1, 2]);
        assert_eq!(pos, 20);
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        assert_eq!(error.to_string(), "Java write failed");
        write_chunks(&[], &mut pos, 3, |_| panic!("empty write")).unwrap();
        assert_eq!(pos, 20);
        assert!(write_chunks(&[0], &mut pos, 0, |_| panic!("invalid limit")).is_err());
        let mut full_pos = u64::MAX;
        assert!(write_chunks(&[0], &mut full_pos, 1, |_| panic!("position overflow")).is_err());
    }

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
