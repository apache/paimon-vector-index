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

use jni::objects::GlobalRef;
use jni::JavaVM;
use paimon_vindex_core::io::SeekRead;
use std::io;
use std::sync::{Arc, Mutex};

/// JNI-backed seekable stream that delegates to Java's SeekableInputStream.
///
/// If the Java stream also implements VectoredReadable, pread() is used for
/// thread-safe positional reads without changing the stream cursor.
pub struct JniSeekableStream {
    jvm: Arc<JavaVM>,
    stream_ref: Arc<GlobalRef>,
    stream_lock: Arc<Mutex<()>>,
    /// Whether the Java stream supports pread (implements VectoredReadable)
    has_pread: bool,
}

impl JniSeekableStream {
    pub fn new(jvm: JavaVM, stream_ref: GlobalRef) -> Self {
        let jvm = Arc::new(jvm);
        let has_pread = check_has_pread(&jvm, &stream_ref);
        JniSeekableStream {
            jvm,
            stream_ref: Arc::new(stream_ref),
            stream_lock: Arc::new(Mutex::new(())),
            has_pread,
        }
    }
}

/// Check if the Java object implements VectoredReadable (has pread method).
fn check_has_pread(jvm: &JavaVM, stream_ref: &GlobalRef) -> bool {
    let mut env = match jvm.attach_current_thread() {
        Ok(e) => e,
        Err(_) => return false,
    };
    // Try to find the pread method — if it exists, the stream supports positional reads
    let class = match env.get_object_class(stream_ref.as_obj()) {
        Ok(c) => c,
        Err(_) => return false,
    };
    env.get_method_id(&class, "pread", "(J[BII)I").is_ok()
}

impl SeekRead for JniSeekableStream {
    fn seek(&mut self, pos: u64) -> io::Result<()> {
        let _guard = self
            .stream_lock
            .lock()
            .map_err(|e| io::Error::other(format!("Lock poisoned: {}", e)))?;

        let mut env = self
            .jvm
            .attach_current_thread()
            .map_err(|e| io::Error::other(format!("JNI attach: {}", e)))?;

        env.call_method(
            self.stream_ref.as_obj(),
            "seek",
            "(J)V",
            &[jni::objects::JValue::Long(pos as i64)],
        )
        .map_err(|e| io::Error::other(format!("JNI seek: {}", e)))?;

        Ok(())
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let _guard = self
            .stream_lock
            .lock()
            .map_err(|e| io::Error::other(format!("Lock poisoned: {}", e)))?;

        read_bytes_from_stream(&self.jvm, &self.stream_ref, buf)
    }

    /// Positional read via Java's VectoredReadable.pread(position, buffer, offset, length).
    /// Thread-safe: does not change the stream cursor position.
    fn pread(&mut self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        if !self.has_pread {
            // Fallback: seek + read with lock
            let _guard = self
                .stream_lock
                .lock()
                .map_err(|e| io::Error::other(format!("Lock poisoned: {}", e)))?;

            let mut env = self
                .jvm
                .attach_current_thread()
                .map_err(|e| io::Error::other(format!("JNI attach: {}", e)))?;

            env.call_method(
                self.stream_ref.as_obj(),
                "seek",
                "(J)V",
                &[jni::objects::JValue::Long(pos as i64)],
            )
            .map_err(|e| io::Error::other(format!("JNI seek: {}", e)))?;

            drop(env);
            return read_bytes_from_stream(&self.jvm, &self.stream_ref, buf);
        }

        // Use pread — no lock needed, thread-safe positional read
        let mut env = self
            .jvm
            .attach_current_thread()
            .map_err(|e| io::Error::other(format!("JNI attach: {}", e)))?;

        let jbuf = env
            .new_byte_array(buf.len() as i32)
            .map_err(|e| io::Error::other(format!("JNI alloc: {}", e)))?;

        let mut total_read = 0i32;
        let length = buf.len() as i32;

        while total_read < length {
            let remaining = length - total_read;
            let n = env
                .call_method(
                    self.stream_ref.as_obj(),
                    "pread",
                    "(J[BII)I",
                    &[
                        jni::objects::JValue::Long(pos as i64 + total_read as i64),
                        jni::objects::JValue::Object(&jbuf),
                        jni::objects::JValue::Int(total_read),
                        jni::objects::JValue::Int(remaining),
                    ],
                )
                .map_err(|e| io::Error::other(format!("JNI pread: {}", e)))?
                .i()
                .map_err(|e| io::Error::other(format!("JNI pread return: {}", e)))?;

            if n <= 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("pread EOF: read {} of {} bytes", total_read, length),
                ));
            }
            total_read += n;
        }

        let mut signed_buf = vec![0i8; buf.len()];
        env.get_byte_array_region(&jbuf, 0, &mut signed_buf)
            .map_err(|e| io::Error::other(format!("JNI get_region: {}", e)))?;

        for (i, &b) in signed_buf.iter().enumerate() {
            buf[i] = b as u8;
        }

        Ok(())
    }

    fn supports_concurrent_pread(&self) -> bool {
        self.has_pread
    }
}

/// Helper: read bytes from the Java stream (after seek, under lock).
fn read_bytes_from_stream(jvm: &JavaVM, stream_ref: &GlobalRef, buf: &mut [u8]) -> io::Result<()> {
    let mut env = jvm
        .attach_current_thread()
        .map_err(|e| io::Error::other(format!("JNI attach: {}", e)))?;

    let jbuf = env
        .new_byte_array(buf.len() as i32)
        .map_err(|e| io::Error::other(format!("JNI alloc: {}", e)))?;

    let mut total_read = 0i32;
    let length = buf.len() as i32;

    while total_read < length {
        let remaining = length - total_read;
        let n = env
            .call_method(
                stream_ref.as_obj(),
                "read",
                "([BII)I",
                &[
                    jni::objects::JValue::Object(&jbuf),
                    jni::objects::JValue::Int(total_read),
                    jni::objects::JValue::Int(remaining),
                ],
            )
            .map_err(|e| io::Error::other(format!("JNI read: {}", e)))?
            .i()
            .map_err(|e| io::Error::other(format!("JNI read return: {}", e)))?;

        if n <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("EOF: read {} of {} bytes", total_read, length),
            ));
        }
        total_read += n;
    }

    let mut signed_buf = vec![0i8; buf.len()];
    env.get_byte_array_region(&jbuf, 0, &mut signed_buf)
        .map_err(|e| io::Error::other(format!("JNI get_region: {}", e)))?;

    for (i, &b) in signed_buf.iter().enumerate() {
        buf[i] = b as u8;
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
