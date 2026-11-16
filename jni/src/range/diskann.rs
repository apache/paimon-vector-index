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
use paimon_vindex_core::range::DiskAnnRangeSearchParams;

fn params(env: &mut JNIEnv, object: JObject) -> Result<DiskAnnRangeSearchParams, String> {
    if object.is_null() {
        return Err("params is null".into());
    }
    let band = read_band(env, &object)?;
    let starting_l = usize::try_from(call_int_method(env, &object, "startingL")?)
        .map_err(|_| "startingL must be positive")?;
    let initial_slack = env
        .call_method(&object, "initialSlack", "()F", &[])
        .and_then(|v| v.f())
        .map_err(|e| e.to_string())?;
    let range_slack = env
        .call_method(&object, "rangeSlack", "()F", &[])
        .and_then(|v| v.f())
        .map_err(|e| e.to_string())?;
    let bytes = env
        .call_method(&object, "maxWorkingBytes", "()J", &[])
        .and_then(|v| v.j())
        .map_err(|e| e.to_string())?;
    let max_working_bytes = usize::try_from(bytes).map_err(|_| "invalid maxWorkingBytes")?;
    let exhaustive = env
        .call_method(&object, "exhaustive", "()Z", &[])
        .and_then(|v| v.z())
        .map_err(|e| e.to_string())?;
    Ok(DiskAnnRangeSearchParams {
        band,
        starting_l,
        initial_slack,
        range_slack,
        max_working_bytes,
        exhaustive,
    })
}

fn run(
    env: &mut JNIEnv,
    ptr: jlong,
    queries: JFloatArray,
    count: Option<jint>,
    object: JObject,
    filter: Option<JByteArray>,
) -> Result<jobject, String> {
    let reader = deref_reader(ptr).ok_or("null native reader")?;
    let count = count
        .map(|n| usize::try_from(n).map_err(|_| "invalid query count"))
        .transpose()?;
    let params = params(env, object)?;
    let queries = read_float_array(env, &queries, "queries")?;
    let filter = filter.map(|f| read_byte_array(env, f)).transpose()?;
    let result = match (count, filter) {
        (None, None) => reader.diskann_range_search(&queries, params),
        (None, Some(f)) => reader.diskann_range_search_with_roaring_filter(&queries, params, &f),
        (Some(n), None) => reader.diskann_range_search_batch(&queries, n, params),
        (Some(n), Some(f)) => {
            reader.diskann_range_search_batch_with_roaring_filter(&queries, n, params, &f)
        }
    }
    .map_err(|e| format!("DiskANN range search: {e}"))?;
    build_range_result(env, result)
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_supportsDiskAnnRangeSearch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jboolean {
    range_call(env, |_| {
        let reader = deref_reader(ptr).ok_or("null native reader")?;
        Ok(jboolean::from(reader.supports_diskann_range_search()))
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_diskAnnRangeSearch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    params: JObject,
) -> jobject {
    range_call(env, |env| run(env, ptr, queries, None, params, None))
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_diskAnnRangeSearchWithRoaringFilter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    params: JObject,
    filter: JByteArray,
) -> jobject {
    range_call(env, |env| {
        run(env, ptr, queries, None, params, Some(filter))
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_diskAnnRangeSearchBatch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    count: jint,
    params: JObject,
) -> jobject {
    range_call(env, |env| run(env, ptr, queries, Some(count), params, None))
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_diskAnnRangeSearchBatchWithRoaringFilter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    count: jint,
    params: JObject,
    filter: JByteArray,
) -> jobject {
    range_call(env, |env| {
        run(env, ptr, queries, Some(count), params, Some(filter))
    })
}
