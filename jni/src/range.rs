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

use jni::objects::{
    AutoLocal, JByteArray, JClass, JFloatArray, JLongArray, JObject, JString, JValue,
};
use jni::sys::{jboolean, jint, jlong, jobject};
use jni::JNIEnv;
use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::range::{
    Bound, CutOperator, DistanceBand, DistanceEndpoint, RangeSearchResult, VectorRangeSearchParams,
};

use crate::{
    call_int_method, deref_reader, jni_call, read_byte_array, read_float_array, throw_and_return,
};

fn range_call<T: Default>(env: JNIEnv, action: impl FnOnce(&mut JNIEnv) -> Result<T, String>) -> T {
    jni_call(env, |env| match action(env) {
        Ok(result) => result,
        Err(error) => {
            if env.exception_check().unwrap_or(true) {
                T::default()
            } else {
                throw_and_return(env, &error)
            }
        }
    })
}

fn checked_array_length(value: usize, name: &str) -> Result<jint, String> {
    jint::try_from(value).map_err(|_| format!("{name} exceeds Java array length limit"))
}

fn checked_counter(value: usize, name: &str) -> Result<jlong, String> {
    jlong::try_from(value).map_err(|_| format!("{name} exceeds Java long limit"))
}

fn parse_metric(value: &str) -> Result<MetricType, String> {
    match value {
        "l2" => Ok(MetricType::L2),
        "inner_product" => Ok(MetricType::InnerProduct),
        "cosine" => Ok(MetricType::Cosine),
        _ => Err(format!("unknown range metric: {value}")),
    }
}

fn read_metric(env: &mut JNIEnv, metric: JObject) -> Result<MetricType, String> {
    if metric.is_null() {
        return Err("metric is null".to_string());
    }
    let metric = env.auto_local(JString::from(metric));
    let value: String = env
        .get_string(&metric)
        .map_err(|error| error.to_string())?
        .into();
    parse_metric(&value)
}

fn cut_operator(code: jint) -> Result<CutOperator, String> {
    match code {
        0 => Ok(CutOperator::Ge),
        1 => Ok(CutOperator::Gt),
        2 => Ok(CutOperator::Le),
        3 => Ok(CutOperator::Lt),
        _ => Err(format!("invalid endpoint operator: {code}")),
    }
}

fn read_endpoint(
    env: &mut JNIEnv,
    value: JObject,
    code: jint,
) -> Result<Option<DistanceEndpoint>, String> {
    if value.is_null() {
        if code != -1 {
            return Err("unbounded endpoint must not have an operator".to_string());
        }
        return Ok(None);
    }
    let op = cut_operator(code)?;
    let value = env
        .call_method(value, "doubleValue", "()D", &[])
        .and_then(|value| value.d())
        .map_err(|error| error.to_string())?;
    Ok(Some(DistanceEndpoint { value, op }))
}

fn read_bound(env: &mut JNIEnv, band: &JObject, name: &str) -> Result<Bound, String> {
    let value = env
        .call_method(band, name, "()Ljava/lang/Float;", &[])
        .and_then(|value| value.l())
        .map_err(|error| error.to_string())?;
    let value = env.auto_local(value);
    if value.is_null() {
        Ok(Bound::Unbounded)
    } else {
        env.call_method(&value, "floatValue", "()F", &[])
            .and_then(|value| value.f())
            .map(Bound::Finite)
            .map_err(|error| error.to_string())
    }
}

fn range_params(env: &mut JNIEnv, params: JObject) -> Result<VectorRangeSearchParams, String> {
    if params.is_null() {
        return Err("params is null".to_string());
    }
    let nprobe = call_int_method(env, &params, "nprobe")?;
    let nprobe = usize::try_from(nprobe).map_err(|_| "nprobe must be positive".to_string())?;
    let band = env
        .call_method(
            &params,
            "band",
            "()Lorg/apache/paimon/index/vector/VectorDistanceBand;",
            &[],
        )
        .and_then(|value| value.l())
        .map_err(|error| error.to_string())?;
    let band = env.auto_local(band);
    if band.is_null() {
        return Err("band is null".to_string());
    }
    let metric = env
        .call_method(&band, "metric", "()Ljava/lang/String;", &[])
        .and_then(|value| value.l())
        .map_err(|error| error.to_string())?;
    let metric = read_metric(env, metric)?;
    let raw_lower = read_bound(env, &band, "rawLower")?;
    let raw_upper = read_bound(env, &band, "rawUpper")?;
    let band =
        DistanceBand::from_raw(raw_lower, raw_upper, metric).map_err(|error| error.to_string())?;
    Ok(VectorRangeSearchParams::new(band, nprobe))
}

fn boxed_bound<'local>(
    env: &mut JNIEnv<'local>,
    bound: Bound,
) -> Result<AutoLocal<'local, JObject<'local>>, String> {
    let object = match bound {
        Bound::Unbounded => JObject::null(),
        Bound::Finite(value) => env
            .new_object("java/lang/Float", "(F)V", &[JValue::Float(value)])
            .map_err(|error| error.to_string())?,
    };
    Ok(env.auto_local(object))
}

fn long_array<'local>(
    env: &mut JNIEnv<'local>,
    values: &[jlong],
    name: &str,
) -> Result<AutoLocal<'local, JLongArray<'local>>, String> {
    let array = env
        .new_long_array(checked_array_length(values.len(), name)?)
        .map_err(|error| error.to_string())?;
    let array = env.auto_local(array);
    env.set_long_array_region(&array, 0, values)
        .map_err(|error| error.to_string())?;
    Ok(array)
}

fn counter_array<'local>(
    env: &mut JNIEnv<'local>,
    len: usize,
    name: &str,
    value_at: impl Fn(usize) -> usize,
) -> Result<AutoLocal<'local, JLongArray<'local>>, String> {
    let array = env
        .new_long_array(checked_array_length(len, name)?)
        .map_err(|error| error.to_string())?;
    let array = env.auto_local(array);
    let mut scratch = [0; 1024];
    for offset in (0..len).step_by(scratch.len()) {
        let count = (len - offset).min(scratch.len());
        for (index, value) in scratch[..count].iter_mut().enumerate() {
            *value = checked_counter(value_at(offset + index), name)?;
        }
        env.set_long_array_region(
            &array,
            checked_array_length(offset, name)?,
            &scratch[..count],
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(array)
}

fn build_range_result(env: &mut JNIEnv, result: RangeSearchResult) -> Result<jobject, String> {
    checked_array_length(result.query_count(), "query count")?;
    checked_array_length(result.lims().len(), "lims")?;
    checked_array_length(result.labels().len(), "labels")?;
    let raw_distance_count = checked_array_length(result.raw_distances().len(), "rawDistances")?;
    let list_reads = checked_counter(result.call_stats().list_reads(), "listReads")?;
    let labels = long_array(env, result.labels(), "labels")?;
    let raw_distances = env
        .new_float_array(raw_distance_count)
        .map_err(|error| error.to_string())?;
    let raw_distances = env.auto_local(raw_distances);
    env.set_float_array_region(&raw_distances, 0, result.raw_distances())
        .map_err(|error| error.to_string())?;
    let lims = counter_array(env, result.lims().len(), "lims", |index| {
        result.lims()[index]
    })?;
    let lists_probed = counter_array(env, result.query_count(), "listsProbed", |index| {
        result.query(index).stats.lists_probed()
    })?;
    let rows_scanned = counter_array(env, result.query_count(), "rowsScanned", |index| {
        result.query(index).stats.rows_scanned()
    })?;
    let rows_committed = counter_array(env, result.query_count(), "rowsCommitted", |index| {
        result.query(index).stats.rows_committed()
    })?;
    let early_abandoned = counter_array(env, result.query_count(), "earlyAbandoned", |index| {
        result.query(index).stats.early_abandoned()
    })?;
    env.call_static_method(
        "org/apache/paimon/index/vector/VectorRangeSearchResult",
        "fromNative",
        "([J[F[J[J[J[J[JJ)Lorg/apache/paimon/index/vector/VectorRangeSearchResult;",
        &[
            JValue::Object(&labels),
            JValue::Object(&raw_distances),
            JValue::Object(&lims),
            JValue::Object(&lists_probed),
            JValue::Object(&rows_scanned),
            JValue::Object(&rows_committed),
            JValue::Object(&early_abandoned),
            JValue::Long(list_reads),
        ],
    )
    .and_then(|value| value.l())
    .map(|object| object.into_raw())
    .map_err(|error| error.to_string())
}

fn range_search(
    env: &mut JNIEnv,
    ptr: jlong,
    queries: JFloatArray,
    query_count: Option<jint>,
    params: JObject,
    filter: Option<JByteArray>,
) -> Result<jobject, String> {
    let reader = deref_reader(ptr)
        .ok_or_else(|| "null native pointer (reader already freed?)".to_string())?;
    let query_count = query_count
        .map(|value| usize::try_from(value).map_err(|_| format!("invalid query count: {value}")))
        .transpose()?;
    let params = range_params(env, params)?;
    let queries = read_float_array(env, &queries, "queries")?;
    let filter = filter
        .map(|filter| read_byte_array(env, filter))
        .transpose()?;
    let result = match (query_count, filter) {
        (None, None) => reader.range_search(&queries, params),
        (None, Some(filter)) => reader.range_search_with_roaring_filter(&queries, params, &filter),
        (Some(count), None) => reader.range_search_batch(&queries, count, params),
        (Some(count), Some(filter)) => {
            reader.range_search_batch_with_roaring_filter(&queries, count, params, &filter)
        }
    }
    .map_err(|error| format!("range_search: {error}"))?;
    build_range_result(env, result)
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_distanceBandFromEndpoints(
    env: JNIEnv,
    _class: JClass,
    metric: JObject,
    lower: JObject,
    lower_operator: jint,
    upper: JObject,
    upper_operator: jint,
) -> jobject {
    range_call(env, |env| {
        let metric = read_metric(env, metric)?;
        let lower = read_endpoint(env, lower, lower_operator)?;
        let upper = read_endpoint(env, upper, upper_operator)?;
        let band = DistanceBand::from_endpoints(lower, upper, metric)
            .map_err(|error| error.to_string())?;
        let raw_lower = boxed_bound(env, band.raw_lower())?;
        let raw_upper = boxed_bound(env, band.raw_upper())?;
        let metric = env
            .new_string(metric.as_str())
            .map_err(|error| error.to_string())?;
        let metric = env.auto_local(metric);
        env.new_object(
            "org/apache/paimon/index/vector/VectorDistanceBand",
            "(Ljava/lang/String;Ljava/lang/Float;Ljava/lang/Float;)V",
            &[
                JValue::Object(&metric),
                JValue::Object(&raw_lower),
                JValue::Object(&raw_upper),
            ],
        )
        .map(|object| object.into_raw())
        .map_err(|error| error.to_string())
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_supportsRangeSearch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) -> jboolean {
    range_call(env, |_env| {
        let reader = deref_reader(ptr)
            .ok_or_else(|| "null native pointer (reader already freed?)".to_string())?;
        Ok(jboolean::from(reader.supports_range_search()))
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_rangeSearch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    query: JFloatArray,
    params: JObject,
) -> jobject {
    range_call(env, |env| range_search(env, ptr, query, None, params, None))
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_rangeSearchWithRoaringFilter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    query: JFloatArray,
    params: JObject,
    filter: JByteArray,
) -> jobject {
    range_call(env, |env| {
        range_search(env, ptr, query, None, params, Some(filter))
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_rangeSearchBatch(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    query_count: jint,
    params: JObject,
) -> jobject {
    range_call(env, |env| {
        range_search(env, ptr, queries, Some(query_count), params, None)
    })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_rangeSearchBatchWithRoaringFilter(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    queries: JFloatArray,
    query_count: jint,
    params: JObject,
    filter: JByteArray,
) -> jobject {
    range_call(env, |env| {
        range_search(env, ptr, queries, Some(query_count), params, Some(filter))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_lengths_and_counters_are_checked() {
        assert_eq!(checked_array_length(0, "labels").unwrap(), 0);
        assert_eq!(
            checked_array_length(jint::MAX as usize, "labels").unwrap(),
            jint::MAX
        );
        assert!(checked_array_length(jint::MAX as usize + 1, "labels").is_err());
        assert_eq!(checked_counter(123, "rowsScanned").unwrap(), 123);
        if usize::BITS == 64 {
            assert_eq!(
                checked_counter(jlong::MAX as usize, "listReads").unwrap(),
                jlong::MAX
            );
            assert!(checked_counter(usize::MAX, "listReads").is_err());
        }
    }

    #[test]
    fn external_metric_and_operator_codes_are_explicit() {
        assert_eq!(parse_metric("l2").unwrap(), MetricType::L2);
        assert_eq!(parse_metric("cosine").unwrap(), MetricType::Cosine);
        assert_eq!(
            parse_metric("inner_product").unwrap(),
            MetricType::InnerProduct
        );
        assert!(parse_metric("unknown").is_err());
        for (code, expected) in [
            (0, CutOperator::Ge),
            (1, CutOperator::Gt),
            (2, CutOperator::Le),
            (3, CutOperator::Lt),
        ] {
            assert_eq!(cut_operator(code).unwrap(), expected);
        }
        assert!(cut_operator(-1).is_err());
        assert!(cut_operator(4).is_err());
    }
}
