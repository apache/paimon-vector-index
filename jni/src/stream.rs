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
use paimon_vindex_core::io::{ReadRequest, SeekRead};
use std::io;
use std::sync::Arc;

/// JNI-backed input stream that delegates to Java's VectorIndexInput.
pub struct JniSeekableStream {
    jvm: Arc<JavaVM>,
    stream_ref: Arc<GlobalRef>,
}

impl JniSeekableStream {
    pub fn new(jvm: JavaVM, stream_ref: GlobalRef) -> Self {
        JniSeekableStream {
            jvm: Arc::new(jvm),
            stream_ref: Arc::new(stream_ref),
        }
    }
}

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
            env.set_object_array_element(&buffers, idx as i32, jbuf)
                .map_err(|e| io::Error::other(format!("JNI set buffer: {}", e)))?;
        }

        env.call_method(
            self.stream_ref.as_obj(),
            "pread",
            "([J[[B)V",
            &[JValue::Object(&positions), JValue::Object(&buffers)],
        )
        .map_err(|e| io::Error::other(format!("JNI pread: {}", e)))?;

        copy_java_buffers(&mut env, &buffers, ranges)
    }
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
