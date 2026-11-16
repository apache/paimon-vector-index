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

pub const PAIMON_VINDEX_RANGE_OPTIONS_IVF: u32 = 0;
pub const PAIMON_VINDEX_RANGE_OPTIONS_DISKANN: u32 = 1;

/// Only the member selected by PaimonVindexRangeSearchOptions.kind is read.
#[repr(C)]
#[derive(Clone, Copy)]
pub union PaimonVindexRangeSearchOptionsValue {
    pub ivf: PaimonVindexRangeSearchParams,
    pub diskann: PaimonVindexDiskAnnRangeSearchParams,
}

/// Tagged family-specific parameters for the unified range entry point.
/// The existing IVF and DiskANN by-value parameter ABIs are unchanged.
#[repr(C)]
pub struct PaimonVindexRangeSearchOptions {
    pub kind: u32,
    pub params: PaimonVindexRangeSearchOptionsValue,
}

/// Unified CSR search for one or more queries. query_count must be positive and
/// queries_len must equal query_count * dimension. Options must match the index
/// family. NULL filter with length zero means no filter; a non-NULL empty filter
/// is malformed. Output ownership and failure rules match the legacy range APIs.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_range_search_with_options(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    queries_len: usize,
    query_count: usize,
    options: *const PaimonVindexRangeSearchOptions,
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
            || {
                let options = &range_slice(options, 1, "range options")?[0];
                match options.kind {
                    PAIMON_VINDEX_RANGE_OPTIONS_IVF => {
                        range_params(options.params.ivf).map(Into::into)
                    }
                    PAIMON_VINDEX_RANGE_OPTIONS_DISKANN => {
                        diskann::diskann_params(options.params.diskann).map(Into::into)
                    }
                    kind => Err(format!("invalid range options kind: {kind}")),
                }
            },
            if filter.is_null() && filter_len == 0 {
                None
            } else {
                Some((filter, filter_len))
            },
            query_count != 1,
            out,
        )
    }
}
