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
    unsafe {
        range_search(
            handle,
            queries,
            len,
            count,
            || diskann_params(params).map(Into::into),
            filter,
            batch,
            out,
        )
    }
}

pub(super) fn diskann_params(
    params: PaimonVindexDiskAnnRangeSearchParams,
) -> Result<DiskAnnRangeSearchParams, String> {
    let band = range_params(PaimonVindexRangeSearchParams {
        band: params.band,
        nprobe: 1,
    })?
    .band();
    if params.exhaustive > 1 {
        return Err("exhaustive must be zero or one".into());
    }
    Ok(DiskAnnRangeSearchParams {
        band,
        starting_l: params.starting_l,
        initial_slack: params.initial_slack,
        range_slack: params.range_slack,
        max_working_bytes: params.max_working_bytes,
        exhaustive: params.exhaustive == 1,
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
