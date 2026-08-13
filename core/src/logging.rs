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

//! Pluggable diagnostic log sink.
//!
//! By default runtime diagnostics are written to stderr, which keeps the
//! historical behavior for the C FFI and Python consumers. Embedders (such as
//! the JNI layer) may install a process-wide sink once to redirect records to
//! their own logging system (e.g. SLF4J/log4j in Spark executors).

use std::io::Write;
use std::sync::OnceLock;

/// Severity of a diagnostic record emitted by the core library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
}

/// Process-wide log sink. Receives the level and a message without a trailing
/// newline. Implementations must never panic.
pub type LogSink = Box<dyn Fn(LogLevel, &str) + Send + Sync>;

static LOG_SINK: OnceLock<LogSink> = OnceLock::new();

/// Installs the process-wide sink. The first caller wins; returns
/// `Err(sink)` if a sink is already installed.
pub fn set_log_sink(sink: LogSink) -> Result<(), LogSink> {
    LOG_SINK.set(sink)
}

/// Emits one record through the installed sink, or falls back to stderr.
pub(crate) fn emit_log(level: LogLevel, message: &str) {
    match LOG_SINK.get() {
        Some(sink) => sink(level, message),
        None => {
            let _ = writeln!(std::io::stderr().lock(), "{message}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // The sink is a process-global OnceLock shared by every test in this
    // binary, so install/deliver/reject must run inside one test.
    #[test]
    fn sink_install_deliver_and_reject_second_install() {
        let captured: Arc<Mutex<Vec<(LogLevel, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink_capture = Arc::clone(&captured);
        set_log_sink(Box::new(move |level, message| {
            sink_capture
                .lock()
                .unwrap()
                .push((level, message.to_string()));
        }))
        .unwrap_or_else(|_| panic!("first install must succeed"));

        emit_log(LogLevel::Info, "hello");
        emit_log(LogLevel::Warn, "watch out");

        let records = captured.lock().unwrap();
        assert_eq!(
            *records,
            vec![
                (LogLevel::Info, "hello".to_string()),
                (LogLevel::Warn, "watch out".to_string()),
            ]
        );
        drop(records);

        assert!(set_log_sink(Box::new(|_, _| {})).is_err());
    }
}
