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

//! Standalone JNI fixture with the old range exports and no result-budget capability.
//! Build with rustc --crate-type cdylib; no JVM headers or third-party crates are needed.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

type Object = *mut c_void;
static RANGE_CALLS: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_freeReader(
    _env: Object,
    _class: Object,
    _reader: i64,
) {
}

macro_rules! range_export {
    ($name:ident ($($arg:ident: $ty:ty),*) -> $result:ty = $value:expr) => {
        #[no_mangle]
        pub extern "system" fn $name(_env: Object, _class: Object, _reader: i64, $($arg: $ty),*) -> $result {
            RANGE_CALLS.fetch_add(1, Ordering::Relaxed);
            $value
        }
    };
}

range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_supportsRangeSearch() -> u8 = 1);
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_supportsDiskAnnRangeSearch() -> u8 = 1);
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_rangeSearch(_q: Object, _p: Object) -> Object = std::ptr::null_mut());
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_rangeSearchWithRoaringFilter(_q: Object, _p: Object, _f: Object) -> Object = std::ptr::null_mut());
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_rangeSearchBatch(_q: Object, _n: i32, _p: Object) -> Object = std::ptr::null_mut());
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_rangeSearchBatchWithRoaringFilter(_q: Object, _n: i32, _p: Object, _f: Object) -> Object = std::ptr::null_mut());
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_diskAnnRangeSearch(_q: Object, _p: Object) -> Object = std::ptr::null_mut());
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_diskAnnRangeSearchWithRoaringFilter(_q: Object, _p: Object, _f: Object) -> Object = std::ptr::null_mut());
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_diskAnnRangeSearchBatch(_q: Object, _n: i32, _p: Object) -> Object = std::ptr::null_mut());
range_export!(Java_org_apache_paimon_index_vector_VectorIndexNative_diskAnnRangeSearchBatchWithRoaringFilter(_q: Object, _n: i32, _p: Object, _f: Object) -> Object = std::ptr::null_mut());

#[cfg(result_api_v2)]
#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexNative_rangeResultApiVersion(
    _env: Object,
    _class: Object,
) -> i32 {
    2
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_index_vector_VectorIndexRangeNativeCompatibilityTest_legacyRangeCalls(
    _env: Object,
    _class: Object,
) -> i32 {
    RANGE_CALLS.load(Ordering::Relaxed) as i32
}
