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

/// Additive DiskANN ABI. Does not change the existing IVF range parameter layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PaimonVindexDiskAnnRangeSearchParams {
    pub band: PaimonVindexRawDistanceBand,
    pub starting_l: usize,
    pub initial_slack: f32,
    pub range_slack: f32,
    pub max_working_bytes: usize,
    /// Zero for graph search, one for exhaustive persisted-vector scanning.
    pub exhaustive: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PaimonVindexDiskAnnRangeSearchStats {
    pub nodes_expanded: usize,
    pub pq_evaluations: usize,
    pub vector_evaluations: usize,
    pub read_calls: usize,
    pub read_bytes: usize,
    pub second_phase: u32,
    pub exhaustive: u32,
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_supports_diskann_range_search(
    handle: *const PaimonVindexReaderHandle,
    out: *mut c_int,
) -> c_int {
    ffi_status(|| {
        if out.is_null() {
            return Err("out capability pointer is null".into());
        }
        let handle = unsafe { reader_ref(handle) }?;
        unsafe { *out = c_int::from(handle.inner.supports_diskann_range_search()) };
        Ok(())
    })
}

/// Borrows result-owned graph diagnostics. IVF results return count zero.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_range_search_result_diskann_stats(
    result: *const PaimonVindexRangeSearchResult,
    out: *mut *const PaimonVindexDiskAnnRangeSearchStats,
    count: *mut usize,
) -> c_int {
    ffi_status(|| {
        if result.is_null() || out.is_null() || count.is_null() {
            return Err("null DiskANN range stats pointer".into());
        }
        let stats = &unsafe { &*result }.diskann_stats;
        unsafe {
            *out = stats.as_ptr();
            *count = stats.len();
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn run(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    len: usize,
    count: usize,
    params: PaimonVindexDiskAnnRangeSearchParams,
    filter: Option<(*const u8, usize)>,
    batch: bool,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    ffi_status(|| {
        if out.is_null() {
            return Err("out result pointer is null".into());
        }
        unsafe { *out = ptr::null_mut() };
        let handle = unsafe { reader_mut(handle) }?;
        if count == 0 || checked_len(count, handle.inner.dimension(), "range queries")? != len {
            return Err("invalid DiskANN range query length or count".into());
        }
        let lims = count.checked_add(1).ok_or("range offsets overflow")?;
        range_len::<usize>(lims, "range offsets")?;
        range_len::<PaimonVindexRangeSearchStats>(count, "range statistics")?;
        range_len::<PaimonVindexDiskAnnRangeSearchStats>(count, "DiskANN statistics")?;
        let queries = unsafe { range_slice(queries, len, "range queries") }?;
        let band = range_params(PaimonVindexRangeSearchParams {
            band: params.band,
            nprobe: 1,
        })?
        .band();
        if params.exhaustive > 1 {
            return Err("exhaustive must be zero or one".into());
        }
        let params = DiskAnnRangeSearchParams {
            band,
            starting_l: params.starting_l,
            initial_slack: params.initial_slack,
            range_slack: params.range_slack,
            max_working_bytes: params.max_working_bytes,
            exhaustive: params.exhaustive == 1,
        };
        let filter = filter
            .map(|(p, n)| unsafe { range_slice(p, n, "range filter") })
            .transpose()?;
        let result = match (batch, filter) {
            (false, None) => handle.inner.diskann_range_search(queries, params),
            (false, Some(filter)) => handle
                .inner
                .diskann_range_search_with_roaring_filter(queries, params, filter),
            (true, None) => handle
                .inner
                .diskann_range_search_batch(queries, count, params),
            (true, Some(filter)) => handle
                .inner
                .diskann_range_search_batch_with_roaring_filter(queries, count, params, filter),
        }
        .map_err(|e| format!("DiskANN range search: {e}"))?;
        let result = Box::new(PaimonVindexRangeSearchResult::new(result)?);
        unsafe { *out = Box::into_raw(result) };
        Ok(())
    })
}

/// Returns an owned uncapped CSR result. Same ownership/error rules as IVF range.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_diskann_range_search(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    queries_len: usize,
    params: PaimonVindexDiskAnnRangeSearchParams,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    unsafe { run(handle, queries, queries_len, 1, params, None, false, out) }
}

/// Returns an owned uncapped CSR result. Same ownership/error rules as IVF range.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_diskann_range_search_with_roaring_filter(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    queries_len: usize,
    params: PaimonVindexDiskAnnRangeSearchParams,
    filter: *const u8,
    filter_len: usize,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    unsafe {
        run(
            handle,
            queries,
            queries_len,
            1,
            params,
            Some((filter, filter_len)),
            false,
            out,
        )
    }
}

/// Returns an owned uncapped CSR result. Same ownership/error rules as IVF range.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_diskann_range_search_batch(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    queries_len: usize,
    query_count: usize,
    params: PaimonVindexDiskAnnRangeSearchParams,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    unsafe {
        run(
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

/// Returns an owned uncapped CSR result. Same ownership/error rules as IVF range.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_diskann_range_search_batch_with_roaring_filter(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    queries_len: usize,
    query_count: usize,
    params: PaimonVindexDiskAnnRangeSearchParams,
    filter: *const u8,
    filter_len: usize,
    out: *mut *mut PaimonVindexRangeSearchResult,
) -> c_int {
    unsafe {
        run(
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
