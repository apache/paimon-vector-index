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

mod stream;

use jni::objects::{JByteArray, JClass, JFloatArray, JLongArray, JObject, JValue};
use jni::sys::{jint, jlong, jobject, jobjectArray};
use jni::JNIEnv;
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexMetadata, VectorIndexReader, VectorIndexWriter,
    VectorSearchParams,
};
use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use stream::{JniOutputStream, JniSeekableStream};

fn throw_and_return<T: Default>(env: &mut JNIEnv, msg: &str) -> T {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
    T::default()
}

fn jni_call<T, F>(mut env: JNIEnv, f: F) -> T
where
    T: Default,
    F: FnOnce(&mut JNIEnv) -> T,
{
    match catch_unwind(AssertUnwindSafe(|| f(&mut env))) {
        Ok(value) => value,
        Err(payload) => throw_panic_and_return(&mut env, &*payload),
    }
}

fn jni_call_void<F>(env: JNIEnv, f: F)
where
    F: FnOnce(&mut JNIEnv),
{
    jni_call(env, |env| f(env))
}

fn throw_panic_and_return<T: Default>(env: &mut JNIEnv, payload: &(dyn Any + Send)) -> T {
    let payload = if let Some(message) = payload.downcast_ref::<&str>() {
        *message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "unknown panic payload"
    };
    throw_and_return(env, &format!("Rust panic in JNI call: {}", payload))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterStage {
    NotTrained,
    CollectingTraining,
    Trained,
    AddingOrWritten,
    Failed,
}

struct JniVectorIndexWriter {
    writer: VectorIndexWriter,
    training_data: Vec<f32>,
    training_vector_count: usize,
    stage: WriterStage,
}

impl JniVectorIndexWriter {
    fn new(writer: VectorIndexWriter) -> Self {
        Self {
            writer,
            training_data: Vec::new(),
            training_vector_count: 0,
            stage: WriterStage::NotTrained,
        }
    }

    fn dimension(&self) -> usize {
        self.writer.dimension()
    }

    fn release_training_data(&mut self) {
        self.training_data = Vec::new();
        self.training_vector_count = 0;
    }
}

enum FinishTrainingFailure {
    Error(std::io::Error),
    Panic(Box<dyn Any + Send>),
}

fn finish_staged_training<F>(
    writer: &mut JniVectorIndexWriter,
    train: F,
) -> Result<(), FinishTrainingFailure>
where
    F: FnOnce(&mut VectorIndexWriter, &[f32], usize) -> std::io::Result<()>,
{
    let result = catch_unwind(AssertUnwindSafe(|| {
        train(
            &mut writer.writer,
            &writer.training_data,
            writer.training_vector_count,
        )
    }));
    writer.release_training_data();
    match result {
        Ok(Ok(())) => {
            writer.stage = WriterStage::Trained;
            Ok(())
        }
        Ok(Err(e)) => {
            writer.stage = WriterStage::Failed;
            Err(FinishTrainingFailure::Error(e))
        }
        Err(payload) => {
            writer.stage = WriterStage::Failed;
            Err(FinishTrainingFailure::Panic(payload))
        }
    }
}

fn deref_writer(ptr: jlong) -> Option<&'static mut JniVectorIndexWriter> {
    if ptr == 0 {
        None
    } else {
        Some(unsafe { &mut *(ptr as *mut JniVectorIndexWriter) })
    }
}

fn deref_reader(ptr: jlong) -> Option<&'static mut VectorIndexReader<JniSeekableStream>> {
    if ptr == 0 {
        None
    } else {
        Some(unsafe { &mut *(ptr as *mut VectorIndexReader<JniSeekableStream>) })
    }
}

fn build_config_from_options(
    env: &mut JNIEnv,
    keys: jobjectArray,
    values: jobjectArray,
) -> Option<VectorIndexConfig> {
    let keys = unsafe { jni::objects::JObjectArray::from_raw(keys) };
    let values = unsafe { jni::objects::JObjectArray::from_raw(values) };
    let key_len = match env.get_array_length(&keys) {
        Ok(len) => len,
        Err(e) => {
            throw_and_return::<()>(env, &format!("get_array_length(keys): {}", e));
            return None;
        }
    };
    let value_len = match env.get_array_length(&values) {
        Ok(len) => len,
        Err(e) => {
            throw_and_return::<()>(env, &format!("get_array_length(values): {}", e));
            return None;
        }
    };
    if key_len != value_len {
        throw_and_return::<()>(
            env,
            &format!(
                "options key/value array length mismatch: {} != {}",
                key_len, value_len
            ),
        );
        return None;
    }

    let mut options = HashMap::with_capacity(key_len as usize);
    for idx in 0..key_len {
        let key = match env.get_object_array_element(&keys, idx) {
            Ok(key) => key,
            Err(e) => {
                throw_and_return::<()>(env, &format!("get options key {}: {}", idx, e));
                return None;
            }
        };
        let value = match env.get_object_array_element(&values, idx) {
            Ok(value) => value,
            Err(e) => {
                throw_and_return::<()>(env, &format!("get options value {}: {}", idx, e));
                return None;
            }
        };
        let key = match java_string(env, key) {
            Ok(key) => key,
            Err(e) => {
                throw_and_return::<()>(env, &format!("read options key {}: {}", idx, e));
                return None;
            }
        };
        let value = match java_string(env, value) {
            Ok(value) => value,
            Err(e) => {
                throw_and_return::<()>(env, &format!("read options value {}: {}", idx, e));
                return None;
            }
        };
        options.insert(key, value);
    }

    match VectorIndexConfig::from_options(&options) {
        Ok(config) => Some(config),
        Err(e) => {
            throw_and_return::<()>(env, &format!("invalid vector index options: {}", e));
            None
        }
    }
}

fn java_string(env: &mut JNIEnv, object: JObject) -> Result<String, String> {
    let string = jni::objects::JString::from(object);
    env.get_string(&string)
        .map(|value| value.into())
        .map_err(|e| format!("get_string: {}", e))
}

fn read_byte_array(env: &mut JNIEnv, array: JByteArray) -> Result<Vec<u8>, String> {
    if array.as_raw().is_null() {
        return Err("filter byte array is null".to_string());
    }

    env.convert_byte_array(array)
        .map_err(|e| format!("convert_byte_array: {}", e))
}

fn read_float_array(env: &mut JNIEnv, array: &JFloatArray, name: &str) -> Result<Vec<f32>, String> {
    if array.as_raw().is_null() {
        return Err(format!("{} float array is null", name));
    }
    let len = env
        .get_array_length(array)
        .map_err(|e| format!("get_array_length({}): {}", name, e))? as usize;
    let mut buf = vec![0.0f32; len];
    env.get_float_array_region(array, 0, &mut buf)
        .map_err(|e| format!("get_float_array_region({}): {}", name, e))?;
    Ok(buf)
}

fn read_long_array(env: &mut JNIEnv, array: &JLongArray, name: &str) -> Result<Vec<i64>, String> {
    if array.as_raw().is_null() {
        return Err(format!("{} long array is null", name));
    }
    let len = env
        .get_array_length(array)
        .map_err(|e| format!("get_array_length({}): {}", name, e))? as usize;
    let mut buf = vec![0i64; len];
    env.get_long_array_region(array, 0, &mut buf)
        .map_err(|e| format!("get_long_array_region({}): {}", name, e))?;
    Ok(buf)
}

fn build_result(env: &mut JNIEnv, ids: Vec<i64>, dists: Vec<f32>) -> jobject {
    let id_array = match env.new_long_array(ids.len() as i32) {
        Ok(a) => a,
        Err(e) => return throw_and_return(env, &format!("new_long_array: {}", e)),
    };
    let _ = env.set_long_array_region(&id_array, 0, &ids);

    let dist_array = match env.new_float_array(dists.len() as i32) {
        Ok(a) => a,
        Err(e) => return throw_and_return(env, &format!("new_float_array: {}", e)),
    };
    let _ = env.set_float_array_region(&dist_array, 0, &dists);

    let result_class = match env.find_class("org/apache/paimon/index/vector/VectorSearchResult") {
        Ok(c) => c,
        Err(e) => return throw_and_return(env, &format!("find_class: {}", e)),
    };

    let result = match env.new_object(
        result_class,
        "([J[F)V",
        &[JValue::Object(&id_array), JValue::Object(&dist_array)],
    ) {
        Ok(r) => r,
        Err(e) => return throw_and_return(env, &format!("new_object: {}", e)),
    };

    result.into_raw()
}

fn build_batch_result(
    env: &mut JNIEnv,
    ids: Vec<i64>,
    dists: Vec<f32>,
    nq: usize,
    k: usize,
) -> jobject {
    let id_array = match env.new_long_array((nq * k) as i32) {
        Ok(a) => a,
        Err(e) => return throw_and_return(env, &format!("new_long_array: {}", e)),
    };
    let _ = env.set_long_array_region(&id_array, 0, &ids);

    let dist_array = match env.new_float_array((nq * k) as i32) {
        Ok(a) => a,
        Err(e) => return throw_and_return(env, &format!("new_float_array: {}", e)),
    };
    let _ = env.set_float_array_region(&dist_array, 0, &dists);

    let result_class =
        match env.find_class("org/apache/paimon/index/vector/VectorSearchBatchResult") {
            Ok(c) => c,
            Err(e) => return throw_and_return(env, &format!("find_class: {}", e)),
        };

    let result = match env.new_object(
        result_class,
        "([J[FII)V",
        &[
            JValue::Object(&id_array),
            JValue::Object(&dist_array),
            JValue::Int(nq as jint),
            JValue::Int(k as jint),
        ],
    ) {
        Ok(r) => r,
        Err(e) => return throw_and_return(env, &format!("new_object: {}", e)),
    };

    result.into_raw()
}

fn build_metadata(env: &mut JNIEnv, metadata: VectorIndexMetadata) -> jobject {
    let class = match env.find_class("org/apache/paimon/index/vector/VectorIndexMetadata") {
        Ok(c) => c,
        Err(e) => return throw_and_return(env, &format!("find_class: {}", e)),
    };
    let index_type = match env.new_string(metadata.index_type.as_str()) {
        Ok(value) => JObject::from(value),
        Err(e) => return throw_and_return(env, &format!("new_string(index_type): {}", e)),
    };
    let metric = match env.new_string(metadata.metric.as_str()) {
        Ok(value) => JObject::from(value),
        Err(e) => return throw_and_return(env, &format!("new_string(metric): {}", e)),
    };
    let (hnsw_m, ef_construction, max_level) = metadata
        .hnsw
        .map(|h| (h.m as jint, h.ef_construction as jint, h.max_level as jint))
        .unwrap_or((0, 0, 0));
    let result = match env.new_object(
        class,
        "(Ljava/lang/String;IILjava/lang/String;JIIII)V",
        &[
            JValue::Object(&index_type),
            JValue::Int(metadata.dimension as jint),
            JValue::Int(metadata.nlist as jint),
            JValue::Object(&metric),
            JValue::Long(metadata.total_vectors),
            JValue::Int(metadata.pq_m.unwrap_or(0) as jint),
            JValue::Int(hnsw_m),
            JValue::Int(ef_construction),
            JValue::Int(max_level),
        ],
    ) {
        Ok(r) => r,
        Err(e) => return throw_and_return(env, &format!("new_object: {}", e)),
    };
    result.into_raw()
}

fn search_params(k: jint, nprobe: jint, ef_search: jint) -> Option<VectorSearchParams> {
    if k < 0 || nprobe < 0 || ef_search < 0 {
        None
    } else {
        Some(VectorSearchParams::with_ef_search(
            k as usize,
            nprobe as usize,
            ef_search as usize,
        ))
    }
}

fn validate_vectors(
    data: &[f32],
    n: usize,
    dimension: usize,
    value_name: &str,
) -> Result<(), String> {
    if n == 0 {
        return Err("vector count must be greater than 0".to_string());
    }
    let expected_len = n
        .checked_mul(dimension)
        .ok_or_else(|| "vector count * dimension overflows usize".to_string())?;
    if data.len() != expected_len {
        return Err(format!(
            "{} length {} does not match vector count * dimension {}",
            value_name,
            data.len(),
            expected_len
        ));
    }
    for (idx, value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{} contains non-finite value at offset {}: {}",
                value_name,
                idx,
                format_non_finite(*value)
            ));
        }
    }
    Ok(())
}

fn format_non_finite(value: f32) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_sign_positive() {
        "inf".to_string()
    } else {
        "-inf".to_string()
    }
}

// --- Unified Writer API ---

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_createWriter(
    env: JNIEnv,
    _class: JClass,
    keys: jobjectArray,
    values: jobjectArray,
) -> jlong {
    jni_call(env, |env| {
        let config = match build_config_from_options(env, keys, values) {
            Some(config) => config,
            None => return 0,
        };

        let writer = match VectorIndexWriter::new(config) {
            Ok(writer) => writer,
            Err(e) => return throw_and_return(env, &format!("create writer: {}", e)),
        };
        Box::into_raw(Box::new(JniVectorIndexWriter::new(writer))) as jlong
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_train(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    data: JFloatArray,
    n: jint,
) {
    jni_call_void(env, |env| {
        let writer = match deref_writer(ptr) {
            Some(writer) => writer,
            None => return throw_and_return(env, "null native pointer (writer already freed?)"),
        };
        match writer.stage {
            WriterStage::NotTrained => {}
            WriterStage::CollectingTraining => {
                return throw_and_return(
                    env,
                    "cannot call train after staged training has started; call finishTraining",
                )
            }
            WriterStage::Trained | WriterStage::AddingOrWritten => {
                return throw_and_return(env, "cannot train writer after training has completed")
            }
            WriterStage::Failed => {
                return throw_and_return(env, "writer is unusable after failed training")
            }
        }
        if n < 0 {
            return throw_and_return(env, &format!("invalid vector count: {}", n));
        }
        let n = n as usize;
        let data_buf = match read_float_array(env, &data, "data") {
            Ok(buf) => buf,
            Err(e) => return throw_and_return(env, &e),
        };
        if let Err(e) = writer.writer.train(&data_buf, n) {
            throw_and_return::<()>(env, &format!("train: {}", e));
        } else {
            writer.stage = WriterStage::Trained;
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_addTrainingVectors(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    data: JFloatArray,
    n: jint,
) {
    jni_call_void(env, |env| {
        let writer = match deref_writer(ptr) {
            Some(writer) => writer,
            None => return throw_and_return(env, "null native pointer (writer already freed?)"),
        };
        match writer.stage {
            WriterStage::NotTrained | WriterStage::CollectingTraining => {}
            WriterStage::Trained => {
                return throw_and_return(
                    env,
                    "cannot add training vectors after training is complete",
                )
            }
            WriterStage::AddingOrWritten => {
                return throw_and_return(
                    env,
                    "cannot add training vectors after vectors have been added or index written",
                )
            }
            WriterStage::Failed => {
                return throw_and_return(env, "writer is unusable after failed training")
            }
        }
        if n < 0 {
            return throw_and_return(env, &format!("invalid vector count: {}", n));
        }
        let n = n as usize;
        let data_buf = match read_float_array(env, &data, "data") {
            Ok(buf) => buf,
            Err(e) => return throw_and_return(env, &e),
        };
        if let Err(e) = validate_vectors(&data_buf, n, writer.dimension(), "training data") {
            return throw_and_return(env, &e);
        }
        let training_vector_count = match writer.training_vector_count.checked_add(n) {
            Some(count) => count,
            None => return throw_and_return(env, "training vector count overflows usize"),
        };
        writer.training_data.extend_from_slice(&data_buf);
        writer.training_vector_count = training_vector_count;
        writer.stage = WriterStage::CollectingTraining;
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_finishTraining(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    jni_call_void(env, |env| {
        let writer = match deref_writer(ptr) {
            Some(writer) => writer,
            None => return throw_and_return(env, "null native pointer (writer already freed?)"),
        };
        match writer.stage {
            WriterStage::CollectingTraining => {}
            WriterStage::NotTrained => {
                return throw_and_return(
                    env,
                    "no training vectors added; call addTrainingVectors before finishTraining",
                )
            }
            WriterStage::Trained | WriterStage::AddingOrWritten => {
                return throw_and_return(env, "training is already complete")
            }
            WriterStage::Failed => {
                return throw_and_return(env, "writer is unusable after failed training")
            }
        }
        if writer.training_vector_count == 0 || writer.training_data.is_empty() {
            writer.release_training_data();
            return throw_and_return(env, "no training vectors added");
        }

        match finish_staged_training(writer, |writer, data, count| writer.train(data, count)) {
            Ok(()) => {}
            Err(FinishTrainingFailure::Error(e)) => {
                throw_and_return(env, &format!("finishTraining: {}", e))
            }
            Err(FinishTrainingFailure::Panic(payload)) => {
                resume_unwind(payload);
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_writerDimension(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jint {
    jni_call(env, |env| {
        let writer = match deref_writer(ptr) {
            Some(writer) => writer,
            None => return throw_and_return(env, "null native pointer (writer already freed?)"),
        };
        writer.dimension() as jint
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_addVectors(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    ids: JLongArray,
    data: JFloatArray,
    n: jint,
) {
    jni_call_void(env, |env| {
        let writer = match deref_writer(ptr) {
            Some(writer) => writer,
            None => return throw_and_return(env, "null native pointer (writer already freed?)"),
        };
        match writer.stage {
            WriterStage::Trained | WriterStage::AddingOrWritten => {}
            WriterStage::NotTrained => {
                return throw_and_return(env, "cannot add vectors before training is complete")
            }
            WriterStage::CollectingTraining => {
                return throw_and_return(env, "cannot add vectors before finishTraining is called")
            }
            WriterStage::Failed => {
                return throw_and_return(env, "writer is unusable after failed training")
            }
        }
        if n < 0 {
            return throw_and_return(env, &format!("invalid vector count: {}", n));
        }
        let n = n as usize;
        let id_buf = match read_long_array(env, &ids, "ids") {
            Ok(buf) => buf,
            Err(e) => return throw_and_return(env, &e),
        };
        let data_buf = match read_float_array(env, &data, "data") {
            Ok(buf) => buf,
            Err(e) => return throw_and_return(env, &e),
        };
        if let Err(e) = writer.writer.add_vectors(&id_buf, &data_buf, n) {
            throw_and_return::<()>(env, &format!("add_vectors: {}", e));
        } else {
            writer.stage = WriterStage::AddingOrWritten;
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_writeIndex(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    stream_output: JObject,
) {
    jni_call_void(env, |env| {
        let writer = match deref_writer(ptr) {
            Some(writer) => writer,
            None => return throw_and_return(env, "null native pointer (writer already freed?)"),
        };

        match writer.stage {
            WriterStage::Trained | WriterStage::AddingOrWritten => {}
            WriterStage::NotTrained => {
                return throw_and_return(env, "cannot write index before training is complete")
            }
            WriterStage::CollectingTraining => {
                return throw_and_return(env, "cannot write index before finishTraining is called")
            }
            WriterStage::Failed => {
                return throw_and_return(env, "writer is unusable after failed training")
            }
        }

        let jvm = match env.get_java_vm() {
            Ok(vm) => vm,
            Err(e) => return throw_and_return(env, &format!("get_java_vm: {}", e)),
        };
        let global_ref = match env.new_global_ref(stream_output) {
            Ok(r) => r,
            Err(e) => return throw_and_return(env, &format!("new_global_ref: {}", e)),
        };

        let mut output = JniOutputStream::new(jvm, global_ref);
        if let Err(e) = writer.writer.write(&mut output) {
            throw_and_return::<()>(env, &format!("write index: {}", e));
        } else {
            writer.stage = WriterStage::AddingOrWritten;
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_freeWriter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    jni_call_void(env, |_env| {
        if ptr != 0 {
            unsafe {
                drop(Box::from_raw(ptr as *mut JniVectorIndexWriter));
            }
        }
    })
}

// --- Unified Reader API ---

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_openReader(
    env: JNIEnv,
    _class: JClass,
    stream_input: JObject,
) -> jlong {
    jni_call(env, |env| {
        let jvm = match env.get_java_vm() {
            Ok(vm) => vm,
            Err(e) => return throw_and_return(env, &format!("get_java_vm: {}", e)),
        };
        let global_ref = match env.new_global_ref(stream_input) {
            Ok(r) => r,
            Err(e) => return throw_and_return(env, &format!("new_global_ref: {}", e)),
        };

        let stream = JniSeekableStream::new(jvm, global_ref);
        let reader = match VectorIndexReader::open(stream) {
            Ok(reader) => reader,
            Err(e) => return throw_and_return(env, &format!("open reader: {}", e)),
        };
        Box::into_raw(Box::new(reader)) as jlong
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_metadata(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jobject {
    jni_call(env, |env| {
        let reader = match deref_reader(ptr) {
            Some(reader) => reader,
            None => return throw_and_return(env, "null native pointer (reader already freed?)"),
        };
        build_metadata(env, reader.metadata())
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_optimizeForSearch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    jni_call_void(env, |env| {
        let reader = match deref_reader(ptr) {
            Some(reader) => reader,
            None => return throw_and_return(env, "null native pointer (reader already freed?)"),
        };
        if let Err(e) = reader.optimize_for_search() {
            throw_and_return::<()>(env, &format!("optimize_for_search: {}", e));
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_search(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    query: JFloatArray,
    k: jint,
    nprobe: jint,
    ef_search: jint,
) -> jobject {
    jni_call(env, |env| {
        let reader = match deref_reader(ptr) {
            Some(reader) => reader,
            None => return throw_and_return(env, "null native pointer (reader already freed?)"),
        };
        let params = match search_params(k, nprobe, ef_search) {
            Some(params) => params,
            None => {
                return throw_and_return(
                    env,
                    &format!(
                        "invalid search parameters: k={}, nprobe={}, efSearch={}",
                        k, nprobe, ef_search
                    ),
                )
            }
        };
        let query_buf = match read_float_array(env, &query, "query") {
            Ok(buf) => buf,
            Err(e) => return throw_and_return(env, &e),
        };
        let (ids, dists) = match reader.search(&query_buf, params) {
            Ok(result) => result,
            Err(e) => return throw_and_return(env, &format!("search: {}", e)),
        };
        build_result(env, ids, dists)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_searchWithRoaringFilter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    query: JFloatArray,
    k: jint,
    nprobe: jint,
    ef_search: jint,
    roaring_filter: JByteArray,
) -> jobject {
    jni_call(env, |env| {
        let reader = match deref_reader(ptr) {
            Some(reader) => reader,
            None => return throw_and_return(env, "null native pointer (reader already freed?)"),
        };
        let params = match search_params(k, nprobe, ef_search) {
            Some(params) => params,
            None => {
                return throw_and_return(
                    env,
                    &format!(
                        "invalid search parameters: k={}, nprobe={}, efSearch={}",
                        k, nprobe, ef_search
                    ),
                )
            }
        };
        let query_buf = match read_float_array(env, &query, "query") {
            Ok(buf) => buf,
            Err(e) => return throw_and_return(env, &e),
        };
        let filter_bytes = match read_byte_array(env, roaring_filter) {
            Ok(bytes) => bytes,
            Err(e) => return throw_and_return(env, &e),
        };
        let (ids, dists) =
            match reader.search_with_roaring_filter(&query_buf, params, &filter_bytes) {
                Ok(result) => result,
                Err(e) => return throw_and_return(env, &format!("search_with_filter: {}", e)),
            };
        build_result(env, ids, dists)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_searchBatch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    query_count: jint,
    k: jint,
    nprobe: jint,
    ef_search: jint,
) -> jobject {
    jni_call(env, |env| {
        let reader = match deref_reader(ptr) {
            Some(reader) => reader,
            None => return throw_and_return(env, "null native pointer (reader already freed?)"),
        };
        if query_count < 0 {
            return throw_and_return(env, &format!("invalid query count: {}", query_count));
        }
        let params = match search_params(k, nprobe, ef_search) {
            Some(params) => params,
            None => {
                return throw_and_return(
                    env,
                    &format!(
                        "invalid search parameters: k={}, nprobe={}, efSearch={}",
                        k, nprobe, ef_search
                    ),
                )
            }
        };
        let nq = query_count as usize;
        let query_buf = match read_float_array(env, &queries, "queries") {
            Ok(buf) => buf,
            Err(e) => return throw_and_return(env, &e),
        };
        let (ids, dists) = match reader.search_batch(&query_buf, nq, params) {
            Ok(result) => result,
            Err(e) => return throw_and_return(env, &format!("search_batch: {}", e)),
        };
        build_batch_result(env, ids, dists, nq, params.top_k)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_searchBatchWithRoaringFilter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    query_count: jint,
    k: jint,
    nprobe: jint,
    ef_search: jint,
    roaring_filter: JByteArray,
) -> jobject {
    jni_call(env, |env| {
        let reader = match deref_reader(ptr) {
            Some(reader) => reader,
            None => return throw_and_return(env, "null native pointer (reader already freed?)"),
        };
        if query_count < 0 {
            return throw_and_return(env, &format!("invalid query count: {}", query_count));
        }
        let params = match search_params(k, nprobe, ef_search) {
            Some(params) => params,
            None => {
                return throw_and_return(
                    env,
                    &format!(
                        "invalid search parameters: k={}, nprobe={}, efSearch={}",
                        k, nprobe, ef_search
                    ),
                )
            }
        };
        let nq = query_count as usize;
        let query_buf = match read_float_array(env, &queries, "queries") {
            Ok(buf) => buf,
            Err(e) => return throw_and_return(env, &e),
        };
        let filter_bytes = match read_byte_array(env, roaring_filter) {
            Ok(bytes) => bytes,
            Err(e) => return throw_and_return(env, &e),
        };
        let (ids, dists) =
            match reader.search_batch_with_roaring_filter(&query_buf, nq, params, &filter_bytes) {
                Ok(result) => result,
                Err(e) => {
                    return throw_and_return(env, &format!("search_batch_with_filter: {}", e))
                }
            };
        build_batch_result(env, ids, dists, nq, params.top_k)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_freeReader(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    jni_call_void(env, |_env| {
        if ptr != 0 {
            unsafe {
                drop(Box::from_raw(
                    ptr as *mut VectorIndexReader<JniSeekableStream>,
                ));
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use paimon_vindex_core::distance::MetricType;

    fn staged_test_writer() -> JniVectorIndexWriter {
        let writer = VectorIndexWriter::new(VectorIndexConfig::IvfFlat {
            dimension: 1,
            nlist: 1,
            metric: MetricType::L2,
        })
        .expect("test writer");
        let mut writer = JniVectorIndexWriter::new(writer);
        writer.training_data = vec![0.0, 1.0];
        writer.training_vector_count = 2;
        writer.stage = WriterStage::CollectingTraining;
        writer
    }

    #[test]
    fn finish_staged_training_releases_training_data_after_panic() {
        let mut writer = staged_test_writer();

        let result = finish_staged_training(&mut writer, |_writer, _data, _count| {
            panic!("injected staged training panic");
        });

        match result {
            Err(FinishTrainingFailure::Panic(_)) => {}
            _ => panic!("expected staged training panic"),
        }
        assert_eq!(WriterStage::Failed, writer.stage);
        assert!(writer.training_data.is_empty());
        assert_eq!(0, writer.training_data.capacity());
        assert_eq!(0, writer.training_vector_count);
    }
}
