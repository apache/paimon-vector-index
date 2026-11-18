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

use super::*;
use paimon_vindex_core::index::CentroidRouter;

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_routerNprobe(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    params: JObject,
    total_vectors: jlong,
    matching_vectors: jlong,
) -> jint {
    jni_call(env, |env| {
        if total_vectors < 0 || matching_vectors < -1 {
            return throw_and_return(env, "invalid vector count");
        }
        let params = match search_params(env, params) {
            Ok(p) => p,
            Err(e) => return throw_and_return(env, &e),
        };
        match router(ptr).and_then(|t| {
            t.resolve_nprobe(
                params,
                total_vectors as usize,
                if matching_vectors < 0 {
                    None
                } else {
                    Some(matching_vectors as usize)
                },
            )
            .map_err(|e| e.to_string())
        }) {
            Ok(nprobe) => nprobe as jint,
            Err(e) => throw_and_return(env, &e),
        }
    })
}

fn router(ptr: jlong) -> Result<&'static CentroidRouter, String> {
    if ptr == 0 {
        return Err("null router pointer".into());
    }
    Ok(unsafe { &*(ptr as *const CentroidRouter) })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_serializeRouter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jobject {
    jni_call(env, |env| {
        let bytes = match router(ptr).map(CentroidRouter::serialize) {
            Ok(bytes) => bytes,
            Err(e) => return throw_and_return(env, &e),
        };
        match env.byte_array_from_slice(&bytes) {
            Ok(array) => array.into_raw(),
            Err(e) => throw_and_return(env, &e.to_string()),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_deserializeRouter(
    env: JNIEnv,
    _class: JClass,
    payload: JByteArray,
) -> jlong {
    jni_call(env, |env| {
        let size = match env.get_array_length(&payload) {
            Ok(size) => size as usize,
            Err(e) => return throw_and_return(env, &e.to_string()),
        };
        let mut header = [0i8; CentroidRouter::HEADER_BYTES];
        if !(header.len()..=CentroidRouter::MAX_SERIALIZED_BYTES).contains(&size) {
            return throw_and_return(env, "invalid centroid router length");
        }
        if let Err(e) = env.get_byte_array_region(&payload, 0, &mut header) {
            return throw_and_return(env, &e.to_string());
        }
        match CentroidRouter::serialized_size(&header.map(|v| v as u8)) {
            Ok(expected) if expected == size => {}
            Ok(_) => return throw_and_return(env, "invalid centroid router length"),
            Err(e) => return throw_and_return(env, &e.to_string()),
        }
        let bytes = match read_byte_array(env, payload) {
            Ok(bytes) => bytes,
            Err(e) => return throw_and_return(env, &e),
        };
        match CentroidRouter::deserialize(&bytes) {
            Ok(t) => Box::into_raw(Box::new(t)) as jlong,
            Err(e) => throw_and_return(env, &e.to_string()),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_routerNlist(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jint {
    jni_call(env, |env| match router(ptr).map(CentroidRouter::nlist) {
        Ok(nlist) => nlist as jint,
        Err(e) => throw_and_return(env, &e),
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_assignCentroid(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    vector: JFloatArray,
) -> jint {
    jni_call(env, |env| {
        let vector = match read_float_array(env, &vector, "vector") {
            Ok(vector) => vector,
            Err(e) => return throw_and_return(env, &e),
        };
        match router(ptr).and_then(|t| t.assign_centroid(&vector).map_err(|e| e.to_string())) {
            Ok(centroid) => centroid as jint,
            Err(e) => throw_and_return(env, &e),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_findNearestCentroids(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    query: JFloatArray,
    nprobe: jint,
) -> jobject {
    jni_call(env, |env| {
        if nprobe <= 0 {
            return throw_and_return(env, "nprobe must be positive");
        }
        let query = match read_float_array(env, &query, "query") {
            Ok(q) => q,
            Err(e) => return throw_and_return(env, &e),
        };
        let centroids = match router(ptr).and_then(|t| {
            t.find_nearest_centroids(&query, nprobe as usize)
                .map_err(|e| e.to_string())
        }) {
            Ok(c) => c.into_iter().map(|c| c as jint).collect::<Vec<_>>(),
            Err(e) => return throw_and_return(env, &e),
        };
        let array = match env.new_int_array(centroids.len() as jint) {
            Ok(a) => a,
            Err(e) => return throw_and_return(env, &e.to_string()),
        };
        match env.set_int_array_region(&array, 0, &centroids) {
            Ok(()) => array.into_raw(),
            Err(e) => throw_and_return(env, &e.to_string()),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_routerFromTraining(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jlong {
    jni_call(env, |env| {
        if ptr == 0 {
            return throw_and_return(env, "null training pointer");
        }
        let mut handle = unsafe { Box::from_raw(ptr as *mut JniVectorIndexTraining) };
        match handle.take().and_then(|training| {
            CentroidRouter::from_training(training).map_err(|error| error.to_string())
        }) {
            Ok(router) => Box::into_raw(Box::new(router)) as jlong,
            Err(error) => throw_and_return(env, &error),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_freeRouter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    jni_call_void(env, |_env| {
        if ptr != 0 {
            unsafe {
                drop(Box::from_raw(ptr as *mut CentroidRouter));
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_maxRouterBytes(
    _env: JNIEnv,
    _class: JClass,
) -> jint {
    CentroidRouter::MAX_SERIALIZED_BYTES as jint
}
