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
