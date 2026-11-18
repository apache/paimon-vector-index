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

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_trainingNprobe(
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
        match training(ptr).and_then(|t| {
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

fn training(ptr: jlong) -> Result<&'static VectorIndexTraining, String> {
    if ptr == 0 {
        return Err("null training pointer".into());
    }
    unsafe { &*(ptr as *const JniVectorIndexTraining) }
        .training
        .as_ref()
        .ok_or_else(|| "training has already been consumed".into())
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_serializeTraining(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jobject {
    jni_call(env, |env| {
        let bytes = match training(ptr).and_then(|t| t.serialize().map_err(|e| e.to_string())) {
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
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_deserializeTraining(
    env: JNIEnv,
    _class: JClass,
    payload: JByteArray,
) -> jlong {
    jni_call(env, |env| {
        let bytes = match read_byte_array(env, payload) {
            Ok(bytes) => bytes,
            Err(e) => return throw_and_return(env, &e),
        };
        match VectorIndexTraining::deserialize(&bytes) {
            Ok(t) => Box::into_raw(Box::new(JniVectorIndexTraining::new(t))) as jlong,
            Err(e) => throw_and_return(env, &e.to_string()),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_trainingIndexType(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jobject {
    jni_call(env, |env| {
        let name = match training(ptr) {
            Ok(t) => t.index_type().as_str(),
            Err(e) => return throw_and_return(env, &e),
        };
        match env.new_string(name) {
            Ok(s) => s.into_raw(),
            Err(e) => throw_and_return(env, &e.to_string()),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_trainingNlist(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jint {
    jni_call(env, |env| {
        match training(ptr).and_then(|t| t.nlist().map_err(|e| e.to_string())) {
            Ok(nlist) => nlist as jint,
            Err(e) => throw_and_return(env, &e),
        }
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
        match training(ptr).and_then(|t| t.assign_centroid(&vector).map_err(|e| e.to_string())) {
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
        let centroids = match training(ptr).and_then(|t| {
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
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_createCentroidWriter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    centroid: jint,
) -> jlong {
    jni_call(env, |env| {
        if centroid < 0 {
            return throw_and_return(env, "centroid must be nonnegative");
        }
        match training(ptr).and_then(|t| {
            t.create_centroid_writer(centroid as usize)
                .map_err(|e| e.to_string())
        }) {
            Ok(writer) => Box::into_raw(Box::new(JniVectorIndexWriter::new(writer))) as jlong,
            Err(e) => throw_and_return(env, &e),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_addRoutedIvfCentroidVectors(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    ids: JLongArray,
    data: JFloatArray,
    n: jint,
) {
    jni_call_void(env, |env| {
        if n <= 0 {
            return throw_and_return(env, "vector count must be positive");
        }
        let Some(writer) = deref_writer(ptr) else {
            return throw_and_return(env, "null writer pointer");
        };
        let ids = match read_long_array(env, &ids, "ids") {
            Ok(v) => v,
            Err(e) => return throw_and_return(env, &e),
        };
        let data = match read_float_array(env, &data, "data") {
            Ok(v) => v,
            Err(e) => return throw_and_return(env, &e),
        };
        if let Err(e) = writer
            .writer
            .add_routed_ivf_centroid_vectors(&ids, &data, n as usize)
        {
            throw_and_return::<()>(env, &e.to_string());
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_searchRoutedIvfShard(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    query: JFloatArray,
    params: JObject,
    centroid: jint,
    filter: JByteArray,
) -> jobject {
    jni_call(env, |env| {
        if centroid < 0 {
            return throw_and_return(env, "centroid must be nonnegative");
        }
        let Some(reader) = deref_reader(ptr) else {
            return throw_and_return(env, "null reader pointer");
        };
        let params = match search_params(env, params) {
            Ok(v) => v,
            Err(e) => return throw_and_return(env, &e),
        };
        let query = match read_float_array(env, &query, "query") {
            Ok(v) => v,
            Err(e) => return throw_and_return(env, &e),
        };
        let filter = if filter.is_null() {
            None
        } else {
            match read_byte_array(env, filter) {
                Ok(v) => Some(v),
                Err(e) => return throw_and_return(env, &e),
            }
        };
        match reader.search_routed_ivf_shard(&query, params, centroid as usize, filter.as_deref()) {
            Ok((ids, distances)) => build_result(env, ids, distances),
            Err(e) => throw_and_return(env, &e.to_string()),
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_searchRoutedIvfShardBatch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    query_count: jint,
    params: JObject,
    centroid: jint,
    filter: JByteArray,
) -> jobject {
    jni_call(env, |env| {
        if centroid < 0 || query_count <= 0 {
            return throw_and_return(env, "centroid must be nonnegative and query count positive");
        }
        let Some(reader) = deref_reader(ptr) else {
            return throw_and_return(env, "null reader pointer");
        };
        let params = match search_params(env, params) {
            Ok(v) => v,
            Err(e) => return throw_and_return(env, &e),
        };
        let queries = match read_float_array(env, &queries, "queries") {
            Ok(v) => v,
            Err(e) => return throw_and_return(env, &e),
        };
        let filter = if filter.is_null() {
            None
        } else {
            match read_byte_array(env, filter) {
                Ok(v) => Some(v),
                Err(e) => return throw_and_return(env, &e),
            }
        };
        match reader.search_routed_ivf_shard_batch(
            &queries,
            query_count as usize,
            params,
            centroid as usize,
            filter.as_deref(),
        ) {
            Ok((ids, distances)) => {
                build_batch_result(env, ids, distances, query_count as usize, params.top_k)
            }
            Err(e) => throw_and_return(env, &e.to_string()),
        }
    })
}
