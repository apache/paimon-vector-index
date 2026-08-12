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

//! Installs a core log sink that forwards diagnostic records to
//! `org.apache.paimon.index.vector.NativeLogBridge`, so native output reaches
//! SLF4J/log4j instead of the raw process stderr (which Spark's
//! System.out-to-log4j redirect cannot capture).

use jni::objects::{GlobalRef, JStaticMethodID, JValue};
use jni::signature::{Primitive, ReturnType};
use jni::sys::{jint, JNI_ERR, JNI_VERSION_1_8};
use jni::{JNIEnv, JavaVM};
use paimon_vindex_core::logging::{set_log_sink, LogLevel};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

const BRIDGE_CLASS: &str = "org/apache/paimon/index/vector/NativeLogBridge";

struct LogBridge {
    vm: JavaVM,
    // Pins NativeLogBridge (and its classloader) so the method id stays valid.
    class: GlobalRef,
    log_method: JStaticMethodID,
}

/// Called synchronously by `System.load` (`NativeLibraryLoader.load`). Never
/// leaves a pending exception and never fails the library load just because
/// logging is unavailable (e.g. slf4j-api absent outside Spark).
#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: *mut jni::sys::JavaVM, _reserved: *mut c_void) -> jint {
    let vm = match unsafe { JavaVM::from_raw(vm) } {
        Ok(vm) => vm,
        Err(_) => return JNI_ERR,
    };
    if let Err(error) = install_log_bridge(&vm) {
        eprintln!("[paimon-vindex] native log bridge disabled, keeping stderr logging: {error}");
    }
    JNI_VERSION_1_8
}

fn install_log_bridge(vm: &JavaVM) -> Result<(), jni::errors::Error> {
    // The JNI_OnLoad thread is already attached; FindClass here resolves
    // through NativeLibraryLoader's classloader (JNI spec), which also loads
    // NativeLogBridge.
    let mut env = vm.get_env()?;
    let found = env.find_class(BRIDGE_CLASS);
    let class = clear_on_err(&mut env, found)?;
    let method = env.get_static_method_id(&class, "log", "(ILjava/lang/String;)V");
    let log_method = clear_on_err(&mut env, method)?;
    let class = env.new_global_ref(&class)?;
    let bridge = LogBridge {
        vm: unsafe { JavaVM::from_raw(vm.get_java_vm_pointer())? },
        class,
        log_method,
    };
    let _ = set_log_sink(Box::new(move |level, message| {
        // Contract: never panic, never leave a pending exception behind.
        let delivered =
            catch_unwind(AssertUnwindSafe(|| forward(&bridge, level, message))).unwrap_or(false);
        if !delivered {
            eprintln!("{message}");
        }
    }));
    Ok(())
}

fn clear_on_err<T>(env: &mut JNIEnv, result: jni::errors::Result<T>) -> jni::errors::Result<T> {
    // A failed FindClass/GetStaticMethodID leaves a pending exception which
    // would make System.load throw; clear it before degrading gracefully.
    if result.is_err() && env.exception_check().unwrap_or(false) {
        let _ = env.exception_clear();
    }
    result
}

fn forward(bridge: &LogBridge, level: LogLevel, message: &str) -> bool {
    // No-op for already-attached threads; daemon-attaches Rayon workers so
    // they never block DestroyJavaVM. Auto-detach happens at thread exit.
    let Ok(mut env) = bridge.vm.attach_current_thread_as_daemon() else {
        return false;
    };
    // Daemon-attached worker threads have no Java frame to release locals.
    env.with_local_frame(4, |env| -> jni::errors::Result<bool> {
        let jmsg = env.new_string(message)?;
        let args = [
            JValue::Int(level as jint).as_jni(),
            JValue::Object(&jmsg).as_jni(),
        ];
        let ok = unsafe {
            env.call_static_method_unchecked(
                &bridge.class,
                bridge.log_method,
                ReturnType::Primitive(Primitive::Void),
                &args,
            )
        }
        .is_ok();
        if !ok || env.exception_check()? {
            let _ = env.exception_clear();
            return Ok(false);
        }
        Ok(true)
    })
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_bridge_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LogBridge>();
    }
}
