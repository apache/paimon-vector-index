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

use jni::objects::{JClass, JFloatArray, JLongArray, JObject, JValue};
use jni::sys::{jboolean, jint, jlong, jobject};
use jni::JNIEnv;
use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::io::{write_index, IVFPQIndexReader};
use paimon_vindex_core::ivfpq::IVFPQIndex;
use stream::{JniOutputStream, JniSeekableStream};

fn throw_and_return<T: Default>(env: &mut JNIEnv, msg: &str) -> T {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
    T::default()
}

fn deref_writer(ptr: jlong) -> Option<&'static mut IVFPQIndex> {
    if ptr == 0 {
        None
    } else {
        Some(unsafe { &mut *(ptr as *mut IVFPQIndex) })
    }
}

fn deref_reader(ptr: jlong) -> Option<&'static mut IVFPQIndexReader<JniSeekableStream>> {
    if ptr == 0 {
        None
    } else {
        Some(unsafe { &mut *(ptr as *mut IVFPQIndexReader<JniSeekableStream>) })
    }
}

// --- Writer API ---

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_createWriter(
    mut env: JNIEnv,
    _class: JClass,
    d: jint,
    nlist: jint,
    m: jint,
    metric: jint,
    use_opq: jboolean,
) -> jlong {
    if d <= 0 || nlist <= 0 || m <= 0 {
        return throw_and_return(
            &mut env,
            &format!("invalid parameters: d={}, nlist={}, m={}", d, nlist, m),
        );
    }
    if d % m != 0 {
        return throw_and_return(&mut env, &format!("d={} must be divisible by m={}", d, m));
    }

    let metric_type = match MetricType::from_code(metric as u32) {
        Some(m) => m,
        None => return throw_and_return(&mut env, &format!("Unknown metric: {}", metric)),
    };

    let index = Box::new(IVFPQIndex::new(
        d as usize,
        nlist as usize,
        m as usize,
        metric_type,
        use_opq != 0,
    ));
    Box::into_raw(index) as jlong
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_train(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    data: JFloatArray,
    n: jint,
) {
    let index = match deref_writer(ptr) {
        Some(i) => i,
        None => return throw_and_return(&mut env, "null native pointer (writer already freed?)"),
    };

    if n <= 0 {
        return throw_and_return(&mut env, &format!("invalid n: {}", n));
    }
    let n = n as usize;

    let len = match env.get_array_length(&data) {
        Ok(l) => l as usize,
        Err(e) => return throw_and_return(&mut env, &format!("get_array_length: {}", e)),
    };

    if len < n * index.d {
        return throw_and_return(
            &mut env,
            &format!(
                "data array too short: {} < n*d={}*{}={}",
                len,
                n,
                index.d,
                n * index.d
            ),
        );
    }

    let mut buf = vec![0.0f32; len];
    if let Err(e) = env.get_float_array_region(&data, 0, &mut buf) {
        return throw_and_return(&mut env, &format!("get_float_array_region: {}", e));
    }

    index.train(&buf, n);
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_addVectors(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    ids: JLongArray,
    data: JFloatArray,
    n: jint,
) {
    let index = match deref_writer(ptr) {
        Some(i) => i,
        None => return throw_and_return(&mut env, "null native pointer (writer already freed?)"),
    };

    if n <= 0 {
        return throw_and_return(&mut env, &format!("invalid n: {}", n));
    }
    let n = n as usize;

    let id_len = match env.get_array_length(&ids) {
        Ok(l) => l as usize,
        Err(e) => return throw_and_return(&mut env, &format!("get_array_length: {}", e)),
    };
    if id_len < n {
        return throw_and_return(
            &mut env,
            &format!("ids array too short: {} < n={}", id_len, n),
        );
    }

    let mut id_buf = vec![0i64; n];
    if let Err(e) = env.get_long_array_region(&ids, 0, &mut id_buf) {
        return throw_and_return(&mut env, &format!("get_long_array_region: {}", e));
    }

    let data_len = match env.get_array_length(&data) {
        Ok(l) => l as usize,
        Err(e) => return throw_and_return(&mut env, &format!("get_array_length: {}", e)),
    };
    if data_len < n * index.d {
        return throw_and_return(
            &mut env,
            &format!("data array too short: {} < n*d={}", data_len, n * index.d),
        );
    }

    let mut data_buf = vec![0.0f32; data_len];
    if let Err(e) = env.get_float_array_region(&data, 0, &mut data_buf) {
        return throw_and_return(&mut env, &format!("get_float_array_region: {}", e));
    }

    index.add(&data_buf, &id_buf, n);
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_writeIndex(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    stream_output: JObject,
) {
    let index = match deref_writer(ptr) {
        Some(i) => i,
        None => return throw_and_return(&mut env, "null native pointer (writer already freed?)"),
    };

    let jvm = match env.get_java_vm() {
        Ok(vm) => vm,
        Err(e) => return throw_and_return(&mut env, &format!("get_java_vm: {}", e)),
    };

    let global_ref = match env.new_global_ref(stream_output) {
        Ok(r) => r,
        Err(e) => return throw_and_return(&mut env, &format!("new_global_ref: {}", e)),
    };

    let mut writer = JniOutputStream::new(jvm, global_ref);
    if let Err(e) = write_index(index, &mut writer) {
        throw_and_return::<()>(&mut env, &format!("write_index: {}", e));
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_freeWriter(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    if ptr != 0 {
        unsafe {
            drop(Box::from_raw(ptr as *mut IVFPQIndex));
        }
    }
}

// --- Reader API ---

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_openReader(
    mut env: JNIEnv,
    _class: JClass,
    stream_input: JObject,
) -> jlong {
    let jvm = match env.get_java_vm() {
        Ok(vm) => vm,
        Err(e) => return throw_and_return(&mut env, &format!("get_java_vm: {}", e)),
    };

    let global_ref = match env.new_global_ref(stream_input) {
        Ok(r) => r,
        Err(e) => return throw_and_return(&mut env, &format!("new_global_ref: {}", e)),
    };

    let stream = JniSeekableStream::new(jvm, global_ref);
    let reader = match IVFPQIndexReader::open(stream) {
        Ok(r) => r,
        Err(e) => return throw_and_return(&mut env, &format!("open: {}", e)),
    };

    Box::into_raw(Box::new(reader)) as jlong
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_search(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    query: JFloatArray,
    k: jint,
    nprobe: jint,
) -> jobject {
    let reader = match deref_reader(ptr) {
        Some(r) => r,
        None => return throw_and_return(&mut env, "null native pointer (reader already freed?)"),
    };

    if k <= 0 || nprobe <= 0 {
        return throw_and_return(
            &mut env,
            &format!("invalid parameters: k={}, nprobe={}", k, nprobe),
        );
    }

    let d = reader.d;
    let query_len = match env.get_array_length(&query) {
        Ok(l) => l as usize,
        Err(e) => return throw_and_return(&mut env, &format!("get_array_length: {}", e)),
    };
    if query_len < d {
        return throw_and_return(
            &mut env,
            &format!("query array too short: {} < d={}", query_len, d),
        );
    }

    let mut query_buf = vec![0.0f32; d];
    if let Err(e) = env.get_float_array_region(&query, 0, &mut query_buf) {
        return throw_and_return(&mut env, &format!("get_float_array_region: {}", e));
    }

    let (ids, dists) = match reader.search(&query_buf, k as usize, nprobe as usize) {
        Ok(r) => r,
        Err(e) => return throw_and_return(&mut env, &format!("search: {}", e)),
    };

    let id_array = match env.new_long_array(ids.len() as i32) {
        Ok(a) => a,
        Err(e) => return throw_and_return(&mut env, &format!("new_long_array: {}", e)),
    };
    let _ = env.set_long_array_region(&id_array, 0, &ids);

    let dist_array = match env.new_float_array(dists.len() as i32) {
        Ok(a) => a,
        Err(e) => return throw_and_return(&mut env, &format!("new_float_array: {}", e)),
    };
    let _ = env.set_float_array_region(&dist_array, 0, &dists);

    let result_class = match env.find_class("org/apache/paimon/index/ivfpq/IVFPQResult") {
        Ok(c) => c,
        Err(e) => return throw_and_return(&mut env, &format!("find_class: {}", e)),
    };

    let result = match env.new_object(
        result_class,
        "([J[F)V",
        &[JValue::Object(&id_array), JValue::Object(&dist_array)],
    ) {
        Ok(r) => r,
        Err(e) => return throw_and_return(&mut env, &format!("new_object: {}", e)),
    };

    result.into_raw()
}

// --- Reader metadata ---

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_getDimension(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jint {
    let reader = match deref_reader(ptr) {
        Some(r) => r,
        None => return throw_and_return(&mut env, "null native pointer (reader already freed?)"),
    };
    reader.d as jint
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_getTotalVectors(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jlong {
    let reader = match deref_reader(ptr) {
        Some(r) => r,
        None => return throw_and_return(&mut env, "null native pointer (reader already freed?)"),
    };
    reader.total_vectors
}

// --- Batch search ---

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_searchBatch(
    mut env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    nq: jint,
    k: jint,
    nprobe: jint,
) -> jobject {
    let reader = match deref_reader(ptr) {
        Some(r) => r,
        None => return throw_and_return(&mut env, "null native pointer (reader already freed?)"),
    };

    if nq <= 0 || k <= 0 || nprobe <= 0 {
        return throw_and_return(
            &mut env,
            &format!("invalid parameters: nq={}, k={}, nprobe={}", nq, k, nprobe),
        );
    }

    let d = reader.d;
    let nq = nq as usize;
    let k = k as usize;

    let query_len = match env.get_array_length(&queries) {
        Ok(l) => l as usize,
        Err(e) => return throw_and_return(&mut env, &format!("get_array_length: {}", e)),
    };
    if query_len < nq * d {
        return throw_and_return(
            &mut env,
            &format!("queries array too short: {} < nq*d={}", query_len, nq * d),
        );
    }

    let mut query_buf = vec![0.0f32; nq * d];
    if let Err(e) = env.get_float_array_region(&queries, 0, &mut query_buf) {
        return throw_and_return(&mut env, &format!("get_float_array_region: {}", e));
    }

    let mut all_ids = vec![-1i64; nq * k];
    let mut all_dists = vec![f32::MAX; nq * k];

    for qi in 0..nq {
        let query = &query_buf[qi * d..(qi + 1) * d];
        match reader.search(query, k, nprobe as usize) {
            Ok((ids, dists)) => {
                let base = qi * k;
                for (i, (&id, &dist)) in ids.iter().zip(dists.iter()).enumerate() {
                    all_ids[base + i] = id;
                    all_dists[base + i] = dist;
                }
            }
            Err(e) => return throw_and_return(&mut env, &format!("search: {}", e)),
        }
    }

    let id_array = match env.new_long_array((nq * k) as i32) {
        Ok(a) => a,
        Err(e) => return throw_and_return(&mut env, &format!("new_long_array: {}", e)),
    };
    let _ = env.set_long_array_region(&id_array, 0, &all_ids);

    let dist_array = match env.new_float_array((nq * k) as i32) {
        Ok(a) => a,
        Err(e) => return throw_and_return(&mut env, &format!("new_float_array: {}", e)),
    };
    let _ = env.set_float_array_region(&dist_array, 0, &all_dists);

    let result_class = match env.find_class("org/apache/paimon/index/ivfpq/IVFPQBatchResult") {
        Ok(c) => c,
        Err(e) => return throw_and_return(&mut env, &format!("find_class: {}", e)),
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
        Err(e) => return throw_and_return(&mut env, &format!("new_object: {}", e)),
    };

    result.into_raw()
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_ivfpq_IVFPQNative_freeReader(
    _env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    if ptr != 0 {
        unsafe {
            drop(Box::from_raw(
                ptr as *mut IVFPQIndexReader<JniSeekableStream>,
            ));
        }
    }
}
