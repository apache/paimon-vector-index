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
use paimon_vindex_core::range::{
    Bound, CutOperator, DistanceBand, DistanceEndpoint, RangeSearchResult, VectorRangeSearchParams,
};

pub const PAIMON_VINDEX_BOUND_UNBOUNDED: u32 = 0;
pub const PAIMON_VINDEX_BOUND_FINITE: u32 = 1;
pub const PAIMON_VINDEX_CUT_GE: u32 = 0;
pub const PAIMON_VINDEX_CUT_GT: u32 = 1;
pub const PAIMON_VINDEX_CUT_LE: u32 = 2;
pub const PAIMON_VINDEX_CUT_LT: u32 = 3;

/// Explicitly raw half-open distance band: squared L2, 1-cosine, or negative
/// inner product. Prefer paimon_vindex_distance_band_from_endpoints for public
/// predicates. Raw cuts are not the original endpoints; inner product reverses
/// their sides. Unbounded sides ignore their value; finite sides must be finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PaimonVindexRawDistanceBand {
    pub metric: u32,
    pub raw_lower_kind: u32,
    pub raw_lower: f32,
    pub raw_upper_kind: u32,
    pub raw_upper: f32,
}

/// Public-distance predicate endpoint. Values are passed to core without rounding.
/// Lower endpoints accept GE/GT; upper endpoints accept LE/LT.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PaimonVindexDistanceEndpoint {
    pub value: f64,
    pub op: u32,
}

/// Fixed IVF probe count, greater than zero and clamped by core to nlist.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PaimonVindexRangeSearchParams {
    pub band: PaimonVindexRawDistanceBand,
    pub nprobe: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PaimonVindexRangeSearchStats {
    pub lists_probed: usize,
    pub rows_scanned: usize,
    pub rows_committed: usize,
    pub early_abandoned: usize,
}

/// Borrowed CSR view; all pointers remain valid until the owning result is destroyed.
/// lims has query_count + 1 entries; labels/raw_distances have hit_count entries;
/// stats has query_count entries. Empty arrays must not be dereferenced.
/// raw_distances retain squared-L2, cosine-distance, or negative-inner-product
/// units; they cannot be compared directly with public L2 or IP endpoints.
#[repr(C)]
pub struct PaimonVindexRangeSearchResultView {
    pub query_count: usize,
    pub hit_count: usize,
    pub lims: *const usize,
    pub labels: *const i64,
    pub raw_distances: *const f32,
    pub stats: *const PaimonVindexRangeSearchStats,
    pub list_reads: usize,
}

/// Owns all range result buffers independently of the reader and input buffers.
pub struct PaimonVindexRangeSearchResult {
    inner: RangeSearchResult,
    stats: Vec<PaimonVindexRangeSearchStats>,
}

fn range_metric(code: u32) -> Result<MetricType, String> {
    match code {
        PAIMON_VINDEX_METRIC_L2 => Ok(MetricType::L2),
        PAIMON_VINDEX_METRIC_COSINE => Ok(MetricType::Cosine),
        PAIMON_VINDEX_METRIC_INNER_PRODUCT => Ok(MetricType::InnerProduct),
        _ => Err(format!("invalid range metric: {code}")),
    }
}

fn range_bound(kind: u32, value: f32) -> Result<Bound, String> {
    match kind {
        PAIMON_VINDEX_BOUND_UNBOUNDED => Ok(Bound::Unbounded),
        PAIMON_VINDEX_BOUND_FINITE => Ok(Bound::Finite(value)),
        _ => Err(format!("invalid bound kind: {kind}")),
    }
}

fn range_params(params: PaimonVindexRangeSearchParams) -> Result<VectorRangeSearchParams, String> {
    let band = DistanceBand::from_raw(
        range_bound(params.band.raw_lower_kind, params.band.raw_lower)?,
        range_bound(params.band.raw_upper_kind, params.band.raw_upper)?,
        range_metric(params.band.metric)?,
    )
    .map_err(|error| format!("range band: {error}"))?;
    Ok(VectorRangeSearchParams::new(band, params.nprobe))
}

unsafe fn range_endpoint(
    endpoint: *const PaimonVindexDistanceEndpoint,
) -> Result<Option<DistanceEndpoint>, String> {
    if endpoint.is_null() {
        return Ok(None);
    }
    let endpoint = unsafe { &*endpoint };
    let op = match endpoint.op {
        PAIMON_VINDEX_CUT_GE => CutOperator::Ge,
        PAIMON_VINDEX_CUT_GT => CutOperator::Gt,
        PAIMON_VINDEX_CUT_LE => CutOperator::Le,
        PAIMON_VINDEX_CUT_LT => CutOperator::Lt,
        code => return Err(format!("invalid endpoint operator: {code}")),
    };
    Ok(Some(DistanceEndpoint {
        value: endpoint.value,
        op,
    }))
}

fn range_band_to_ffi(band: DistanceBand) -> PaimonVindexRawDistanceBand {
    let encode = |bound| match bound {
        Bound::Unbounded => (PAIMON_VINDEX_BOUND_UNBOUNDED, 0.0),
        Bound::Finite(value) => (PAIMON_VINDEX_BOUND_FINITE, value),
    };
    let (raw_lower_kind, raw_lower) = encode(band.raw_lower());
    let (raw_upper_kind, raw_upper) = encode(band.raw_upper());
    PaimonVindexRawDistanceBand {
        metric: metric_code(band.metric()),
        raw_lower_kind,
        raw_lower,
        raw_upper_kind,
        raw_upper,
    }
}

/// Converts public-distance endpoints through core, including f64 boundary rounding
/// and inner-product side reversal. NULL endpoints mean structurally unbounded.
/// Returns zero on success or -1 with last_error set; out is unchanged on error.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_distance_band_from_endpoints(
    metric: u32,
    lower: *const PaimonVindexDistanceEndpoint,
    upper: *const PaimonVindexDistanceEndpoint,
    out: *mut PaimonVindexRawDistanceBand,
) -> c_int {
    ffi_status(|| {
        if out.is_null() {
            return Err("out band pointer is null".to_string());
        }
        let band = DistanceBand::from_endpoints(
            unsafe { range_endpoint(lower) }?,
            unsafe { range_endpoint(upper) }?,
            range_metric(metric)?,
        )
        .map_err(|error| format!("range endpoints: {error}"))?;
        unsafe { *out = range_band_to_ffi(band) };
        Ok(())
    })
}

/// Queries core capability without issuing a search; out receives zero or one.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_supports_range_search(
    handle: *const PaimonVindexReaderHandle,
    out: *mut c_int,
) -> c_int {
    ffi_status(|| {
        if out.is_null() {
            return Err("out capability pointer is null".to_string());
        }
        let handle = unsafe { reader_ref(handle) }?;
        unsafe { *out = c_int::from(handle.inner.supports_range_search()) };
        Ok(())
    })
}

fn range_len<T>(len: usize, name: &str) -> Result<(), String> {
    if len
        .checked_mul(size_of::<T>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(format!("{name} byte length overflow"));
    }
    Ok(())
}

unsafe fn range_slice<'a, T>(data: *const T, len: usize, name: &str) -> Result<&'a [T], String> {
    range_len::<T>(len, name)?;
    if len > 0 && !data.is_aligned() {
        return Err(format!("{name} pointer is not aligned"));
    }
    unsafe { const_slice(data, len, name) }
}

impl PaimonVindexRangeSearchResult {
    fn new(inner: RangeSearchResult) -> Result<Self, String> {
        let mut stats = Vec::new();
        stats
            .try_reserve_exact(inner.query_count())
            .map_err(|error| format!("range result allocation: {error}"))?;
        for query in 0..inner.query_count() {
            let counters = inner.query(query).stats;
            stats.push(PaimonVindexRangeSearchStats {
                lists_probed: counters.lists_probed(),
                rows_scanned: counters.rows_scanned(),
                rows_committed: counters.rows_committed(),
                early_abandoned: counters.early_abandoned(),
            });
        }
        Ok(Self { inner, stats })
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn range_search(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    queries_len: usize,
    query_count: usize,
    params: PaimonVindexRangeSearchParams,
    filter: Option<(*const u8, usize)>,
    batch: bool,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    ffi_status(|| {
        if out.is_null() {
            return Err("out result pointer is null".to_string());
        }
        unsafe { *out = ptr::null_mut() };
        let handle = unsafe { reader_mut(handle) }?;
        let expected_len = checked_len(query_count, handle.inner.dimension(), "range queries")?;
        if queries_len != expected_len {
            return Err(format!(
                "range query length {queries_len} does not match required {expected_len}"
            ));
        }
        let lims_len = query_count
            .checked_add(1)
            .ok_or_else(|| "range offsets length overflow".to_string())?;
        range_len::<usize>(lims_len, "range offsets")?;
        range_len::<PaimonVindexRangeSearchStats>(query_count, "range statistics")?;
        let queries = unsafe { range_slice(queries, queries_len, "range queries") }?;
        let params = range_params(params)?;
        let filter = filter
            .map(|(data, len)| unsafe { range_slice(data, len, "range filter") })
            .transpose()?;
        let result = match (batch, filter) {
            (false, None) => handle.inner.range_search(queries, params),
            (false, Some(filter)) => handle
                .inner
                .range_search_with_roaring_filter(queries, params, filter),
            (true, None) => handle
                .inner
                .range_search_batch(queries, query_count, params),
            (true, Some(filter)) => handle.inner.range_search_batch_with_roaring_filter(
                queries,
                query_count,
                params,
                filter,
            ),
        }
        .map_err(|error| format!("range search: {error}"))?;
        let result = Box::new(PaimonVindexRangeSearchResult::new(result)?);
        unsafe { *out = Box::into_raw(result) };
        Ok(())
    })
}

/// Returns an owned variable-length result; no top-K limit or padding is applied.
/// Inputs are borrowed for this call only. A valid out pointer is set to NULL
/// before any work; on failure last_error is set and no result is transferred.
/// Query length is in floats and must equal the reader dimension.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_range_search(
    handle: *mut PaimonVindexReaderHandle,
    query: *const f32,
    query_len: usize,
    params: PaimonVindexRangeSearchParams,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    unsafe { range_search(handle, query, query_len, 1, params, None, false, out) }
}

/// Filtered single-query variant. The filter must be a serialized Roaring
/// bitmap/treemap; zero bytes are malformed, not equivalent to no filter.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_range_search_with_roaring_filter(
    handle: *mut PaimonVindexReaderHandle,
    query: *const f32,
    query_len: usize,
    params: PaimonVindexRangeSearchParams,
    filter: *const u8,
    filter_len: usize,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    unsafe {
        range_search(
            handle,
            query,
            query_len,
            1,
            params,
            Some((filter, filter_len)),
            false,
            out,
        )
    }
}

/// Batched CSR variant. queries_len must equal query_count * dimension.
/// The query count must be positive, as required by core.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_range_search_batch(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    queries_len: usize,
    query_count: usize,
    params: PaimonVindexRangeSearchParams,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    unsafe {
        range_search(
            handle,
            queries,
            queries_len,
            query_count,
            params,
            None,
            true,
            out,
        )
    }
}

/// Batched CSR variant sharing one serialized Roaring allow-list across queries.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_range_search_batch_with_roaring_filter(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    queries_len: usize,
    query_count: usize,
    params: PaimonVindexRangeSearchParams,
    filter: *const u8,
    filter_len: usize,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    unsafe {
        range_search(
            handle,
            queries,
            queries_len,
            query_count,
            params,
            Some((filter, filter_len)),
            true,
            out,
        )
    }
}

/// Obtains a borrowed view without transferring ownership or copying buffers.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_range_search_result_view(
    result: *const PaimonVindexRangeSearchResult,
    out: *mut PaimonVindexRangeSearchResultView,
) -> c_int {
    ffi_status(|| {
        if result.is_null() || out.is_null() {
            return Err("range result or view pointer is null".to_string());
        }
        let result = unsafe { &*result };
        unsafe {
            *out = PaimonVindexRangeSearchResultView {
                query_count: result.inner.query_count(),
                hit_count: result.inner.labels().len(),
                lims: result.inner.lims().as_ptr(),
                labels: result.inner.labels().as_ptr(),
                raw_distances: result.inner.raw_distances().as_ptr(),
                stats: result.stats.as_ptr(),
                list_reads: result.inner.call_stats().list_reads(),
            }
        };
        Ok(())
    })
}

/// Destroys a result exactly once, invalidating every borrowed view. NULL is a no-op.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_range_search_result_destroy(
    result: *mut PaimonVindexRangeSearchResult,
) {
    if !result.is_null() {
        unsafe { drop(Box::from_raw(result)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_query_byte_limits_are_checked_before_slice_construction() {
        let maximum = isize::MAX as usize / size_of::<f32>();
        assert!(range_len::<f32>(maximum, "queries").is_ok());
        for len in [maximum + 1, usize::MAX] {
            let error = unsafe { range_slice::<f32>(ptr::dangling(), len, "queries") }.unwrap_err();
            assert_eq!(error, "queries byte length overflow");
        }
    }
}
