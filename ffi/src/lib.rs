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

#![allow(clippy::missing_safety_doc)]

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexMetadata, VectorIndexReader, VectorIndexTrainer,
    VectorIndexTraining, VectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::{ReadRequest, SeekRead, SeekWrite};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::io;
use std::os::raw::{c_char, c_int, c_void};
use std::panic::{self, AssertUnwindSafe};
use std::{ptr, slice};

pub const PAIMON_VINDEX_INDEX_TYPE_IVF_FLAT: u32 = 0;
pub const PAIMON_VINDEX_INDEX_TYPE_IVF_PQ: u32 = 1;
pub const PAIMON_VINDEX_INDEX_TYPE_IVF_HNSW_FLAT: u32 = 2;
pub const PAIMON_VINDEX_INDEX_TYPE_IVF_HNSW_SQ: u32 = 3;

pub const PAIMON_VINDEX_METRIC_L2: u32 = 0;
pub const PAIMON_VINDEX_METRIC_INNER_PRODUCT: u32 = 1;
pub const PAIMON_VINDEX_METRIC_COSINE: u32 = 2;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<String>) {
    let msg = msg.into().replace('\0', "\\0");
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

fn panic_message(e: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        format!("native panic: {}", s)
    } else if let Some(s) = e.downcast_ref::<&str>() {
        format!("native panic: {}", s)
    } else {
        "native panic: unknown".to_string()
    }
}

fn ffi_status<F>(f: F) -> c_int
where
    F: FnOnce() -> Result<(), String>,
{
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_error(e);
            -1
        }
        Err(e) => {
            set_error(panic_message(&e));
            -1
        }
    }
}

fn ffi_ptr<T, F>(f: F) -> *mut T
where
    F: FnOnce() -> Result<*mut T, String>,
{
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => {
            set_error(e);
            ptr::null_mut()
        }
        Err(e) => {
            set_error(panic_message(&e));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn paimon_vindex_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(msg) => msg.as_ptr(),
        None => ptr::null(),
    })
}

// ======================== IO callbacks ========================

#[repr(C)]
pub struct PaimonVindexOutputFile {
    pub ctx: *mut c_void,
    pub write_fn: Option<unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int>,
    pub flush_fn: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    pub get_pos_fn: Option<unsafe extern "C" fn(*mut c_void) -> i64>,
}

struct FfiOutputFile {
    raw: PaimonVindexOutputFile,
    pos: u64,
}

unsafe impl Send for FfiOutputFile {}

impl FfiOutputFile {
    fn flush(&mut self) -> io::Result<()> {
        if let Some(flush_fn) = self.raw.flush_fn {
            let result = unsafe { flush_fn(self.raw.ctx) };
            if result != 0 {
                return Err(io::Error::other("flush callback failed"));
            }
        }
        Ok(())
    }
}

impl SeekWrite for FfiOutputFile {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        if let Some(write_fn) = self.raw.write_fn {
            let result = unsafe { write_fn(self.raw.ctx, buf.as_ptr(), buf.len()) };
            if result != 0 {
                return Err(io::Error::other("write callback failed"));
            }
            self.pos = self
                .pos
                .checked_add(buf.len() as u64)
                .ok_or_else(|| io::Error::other("output position overflow"))?;
            Ok(())
        } else {
            Err(io::Error::other("write_fn is null"))
        }
    }

    fn pos(&self) -> u64 {
        if let Some(get_pos_fn) = self.raw.get_pos_fn {
            let pos = unsafe { get_pos_fn(self.raw.ctx) };
            if pos >= 0 {
                return pos as u64;
            }
        }
        self.pos
    }
}

#[repr(C)]
pub struct PaimonVindexInputFile {
    pub ctx: *mut c_void,
    pub read_at_fn: Option<unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> c_int>,
}

struct FfiInputFile {
    raw: PaimonVindexInputFile,
}

unsafe impl Send for FfiInputFile {}

impl SeekRead for FfiInputFile {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        if let Some(read_at_fn) = self.raw.read_at_fn {
            for range in ranges {
                let result = unsafe {
                    read_at_fn(
                        self.raw.ctx,
                        range.pos,
                        range.buf.as_mut_ptr(),
                        range.buf.len(),
                    )
                };
                if result != 0 {
                    return Err(io::Error::other(format!(
                        "read_at callback failed at offset {} length {}",
                        range.pos,
                        range.buf.len()
                    )));
                }
            }
            Ok(())
        } else {
            Err(io::Error::other("read_at_fn is null"))
        }
    }
}

// ======================== Common structs ========================

#[repr(C)]
pub struct PaimonVindexMetadata {
    pub index_type: u32,
    pub dimension: usize,
    pub nlist: usize,
    pub metric: u32,
    pub total_vectors: i64,
    pub pq_m: usize,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_max_level: usize,
}

pub struct PaimonVindexTrainerHandle {
    inner: Option<VectorIndexTrainer>,
}

pub struct PaimonVindexTrainingHandle {
    inner: Option<VectorIndexTraining>,
}

pub struct PaimonVindexWriterHandle {
    inner: VectorIndexWriter,
}

pub struct PaimonVindexReaderHandle {
    inner: VectorIndexReader<FfiInputFile>,
}

fn metadata_to_ffi(metadata: VectorIndexMetadata) -> PaimonVindexMetadata {
    let (hnsw_m, hnsw_ef_construction, hnsw_max_level) = metadata
        .hnsw
        .map(|h| (h.m, h.ef_construction, h.max_level))
        .unwrap_or((0, 0, 0));
    PaimonVindexMetadata {
        index_type: metadata.index_type as u32,
        dimension: metadata.dimension,
        nlist: metadata.nlist,
        metric: metric_code(metadata.metric),
        total_vectors: metadata.total_vectors,
        pq_m: metadata.pq_m.unwrap_or(0),
        hnsw_m,
        hnsw_ef_construction,
        hnsw_max_level,
    }
}

fn metric_code(metric: MetricType) -> u32 {
    match metric {
        MetricType::L2 => PAIMON_VINDEX_METRIC_L2,
        MetricType::InnerProduct => PAIMON_VINDEX_METRIC_INNER_PRODUCT,
        MetricType::Cosine => PAIMON_VINDEX_METRIC_COSINE,
    }
}

unsafe fn options_from_raw(
    keys: *const *const c_char,
    values: *const *const c_char,
    len: usize,
) -> Result<HashMap<String, String>, String> {
    if len > 0 && (keys.is_null() || values.is_null()) {
        return Err("option keys or values pointer is null".to_string());
    }
    let key_ptrs = if len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(keys, len) }
    };
    let value_ptrs = if len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(values, len) }
    };
    let mut options = HashMap::with_capacity(len);
    for idx in 0..len {
        let key_ptr = key_ptrs[idx];
        let value_ptr = value_ptrs[idx];
        if key_ptr.is_null() {
            return Err(format!("option key {} is null", idx));
        }
        if value_ptr.is_null() {
            return Err(format!("option value {} is null", idx));
        }
        let key = unsafe { CStr::from_ptr(key_ptr) }
            .to_str()
            .map_err(|_| format!("option key {} contains invalid UTF-8", idx))?
            .to_string();
        let value = unsafe { CStr::from_ptr(value_ptr) }
            .to_str()
            .map_err(|_| format!("option value {} contains invalid UTF-8", idx))?
            .to_string();
        if options.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option key '{}'", key));
        }
    }
    Ok(options)
}

unsafe fn writer_mut<'a>(
    handle: *mut PaimonVindexWriterHandle,
) -> Result<&'a mut PaimonVindexWriterHandle, String> {
    if handle.is_null() {
        Err("null writer handle".to_string())
    } else {
        Ok(unsafe { &mut *handle })
    }
}

unsafe fn trainer_mut<'a>(
    handle: *mut PaimonVindexTrainerHandle,
) -> Result<&'a mut PaimonVindexTrainerHandle, String> {
    if handle.is_null() {
        Err("null trainer handle".to_string())
    } else {
        Ok(unsafe { &mut *handle })
    }
}

unsafe fn trainer_ref<'a>(
    handle: *const PaimonVindexTrainerHandle,
) -> Result<&'a PaimonVindexTrainerHandle, String> {
    if handle.is_null() {
        Err("null trainer handle".to_string())
    } else {
        Ok(unsafe { &*handle })
    }
}

unsafe fn training_mut<'a>(
    handle: *mut PaimonVindexTrainingHandle,
) -> Result<&'a mut PaimonVindexTrainingHandle, String> {
    if handle.is_null() {
        Err("null training handle".to_string())
    } else {
        Ok(unsafe { &mut *handle })
    }
}

unsafe fn reader_mut<'a>(
    handle: *mut PaimonVindexReaderHandle,
) -> Result<&'a mut PaimonVindexReaderHandle, String> {
    if handle.is_null() {
        Err("null reader handle".to_string())
    } else {
        Ok(unsafe { &mut *handle })
    }
}

unsafe fn reader_ref<'a>(
    handle: *const PaimonVindexReaderHandle,
) -> Result<&'a PaimonVindexReaderHandle, String> {
    if handle.is_null() {
        Err("null reader handle".to_string())
    } else {
        Ok(unsafe { &*handle })
    }
}

unsafe fn writer_ref<'a>(
    handle: *const PaimonVindexWriterHandle,
) -> Result<&'a PaimonVindexWriterHandle, String> {
    if handle.is_null() {
        Err("null writer handle".to_string())
    } else {
        Ok(unsafe { &*handle })
    }
}

fn checked_len(a: usize, b: usize, name: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("{} length overflow", name))
}

unsafe fn const_slice<'a, T>(ptr: *const T, len: usize, name: &str) -> Result<&'a [T], String> {
    if len == 0 {
        Ok(&[])
    } else if ptr.is_null() {
        Err(format!("{} pointer is null", name))
    } else {
        Ok(unsafe { slice::from_raw_parts(ptr, len) })
    }
}

unsafe fn mut_slice<'a, T>(ptr: *mut T, len: usize, name: &str) -> Result<&'a mut [T], String> {
    if len == 0 {
        Ok(&mut [])
    } else if ptr.is_null() {
        Err(format!("{} pointer is null", name))
    } else {
        Ok(unsafe { slice::from_raw_parts_mut(ptr, len) })
    }
}

fn copy_search_result(
    ids: &[i64],
    distances: &[f32],
    out_ids: *mut i64,
    out_distances: *mut f32,
    result_len: usize,
    expected_len: usize,
) -> Result<(), String> {
    if ids.len() != expected_len || distances.len() != expected_len {
        return Err(format!(
            "native result length mismatch: ids={}, distances={}, expected={}",
            ids.len(),
            distances.len(),
            expected_len
        ));
    }
    if result_len < expected_len {
        return Err(format!(
            "result buffers length {} is smaller than required {}",
            result_len, expected_len
        ));
    }
    let out_ids = unsafe { mut_slice(out_ids, expected_len, "out_ids") }?;
    let out_distances = unsafe { mut_slice(out_distances, expected_len, "out_distances") }?;
    out_ids.copy_from_slice(ids);
    out_distances.copy_from_slice(distances);
    Ok(())
}

// ======================== Trainer / Writer ========================

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_trainer_open(
    keys: *const *const c_char,
    values: *const *const c_char,
    num_options: usize,
) -> *mut PaimonVindexTrainerHandle {
    ffi_ptr(|| {
        let options = unsafe { options_from_raw(keys, values, num_options) }?;
        let config = VectorIndexConfig::from_options(&options)
            .map_err(|e| format!("invalid vector index options: {}", e))?;
        let trainer =
            VectorIndexTrainer::new(config).map_err(|e| format!("create trainer: {}", e))?;
        Ok(Box::into_raw(Box::new(PaimonVindexTrainerHandle {
            inner: Some(trainer),
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_trainer_free(handle: *mut PaimonVindexTrainerHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_trainer_dimension(
    handle: *const PaimonVindexTrainerHandle,
    out: *mut usize,
) -> c_int {
    ffi_status(|| {
        if out.is_null() {
            return Err("out pointer is null".to_string());
        }
        let handle = unsafe { trainer_ref(handle) }?;
        let trainer = handle
            .inner
            .as_ref()
            .ok_or_else(|| "trainer has already finished".to_string())?;
        unsafe {
            *out = trainer.dimension();
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_trainer_add_training_vectors(
    handle: *mut PaimonVindexTrainerHandle,
    data: *const f32,
    vector_count: usize,
) -> c_int {
    ffi_status(|| {
        let handle = unsafe { trainer_mut(handle) }?;
        let trainer = handle
            .inner
            .as_mut()
            .ok_or_else(|| "trainer has already finished".to_string())?;
        let len = checked_len(vector_count, trainer.dimension(), "training data")?;
        let data = unsafe { const_slice(data, len, "data") }?;
        trainer
            .add_training_vectors_mut(data, vector_count)
            .map(|_| ())
            .map_err(|e| format!("add training vectors: {}", e))
    })
}

/// Finishes training and consumes the trainer's internal state, but does not free `handle`.
/// Callers must still call `paimon_vindex_trainer_free(handle)` after this returns.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_trainer_finish(
    handle: *mut PaimonVindexTrainerHandle,
) -> *mut PaimonVindexTrainingHandle {
    ffi_ptr(|| {
        let handle = unsafe { trainer_mut(handle) }?;
        let trainer = handle
            .inner
            .take()
            .ok_or_else(|| "trainer has already finished".to_string())?;
        let training = trainer
            .finish()
            .map_err(|e| format!("finish training: {}", e))?;
        Ok(Box::into_raw(Box::new(PaimonVindexTrainingHandle {
            inner: Some(training),
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_training_free(handle: *mut PaimonVindexTrainingHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Opens a writer by consuming the training state inside `training`, but does not free the handle.
/// Callers must still call `paimon_vindex_training_free(training)` after this returns.
#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_writer_open(
    training: *mut PaimonVindexTrainingHandle,
) -> *mut PaimonVindexWriterHandle {
    ffi_ptr(|| {
        let training = unsafe { training_mut(training) }?;
        let training = training
            .inner
            .take()
            .ok_or_else(|| "training has already been consumed".to_string())?;
        Ok(Box::into_raw(Box::new(PaimonVindexWriterHandle {
            inner: VectorIndexWriter::new(training),
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_writer_free(handle: *mut PaimonVindexWriterHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_writer_dimension(
    handle: *const PaimonVindexWriterHandle,
    out: *mut usize,
) -> c_int {
    ffi_status(|| {
        if out.is_null() {
            return Err("out pointer is null".to_string());
        }
        let handle = unsafe { writer_ref(handle) }?;
        unsafe {
            *out = handle.inner.dimension();
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_writer_add_vectors(
    handle: *mut PaimonVindexWriterHandle,
    ids: *const i64,
    data: *const f32,
    vector_count: usize,
) -> c_int {
    ffi_status(|| {
        let handle = unsafe { writer_mut(handle) }?;
        let len = checked_len(vector_count, handle.inner.dimension(), "vector data")?;
        let ids = unsafe { const_slice(ids, vector_count, "ids") }?;
        let data = unsafe { const_slice(data, len, "data") }?;
        handle
            .inner
            .add_vectors(ids, data, vector_count)
            .map_err(|e| format!("add_vectors: {}", e))
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_writer_write_index(
    handle: *mut PaimonVindexWriterHandle,
    output_file: PaimonVindexOutputFile,
) -> c_int {
    ffi_status(|| {
        let handle = unsafe { writer_mut(handle) }?;
        let mut output = FfiOutputFile {
            raw: output_file,
            pos: 0,
        };
        handle
            .inner
            .write(&mut output)
            .map_err(|e| format!("write index: {}", e))?;
        output.flush().map_err(|e| format!("flush index: {}", e))
    })
}

// ======================== Reader ========================

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_open(
    input_file: PaimonVindexInputFile,
) -> *mut PaimonVindexReaderHandle {
    ffi_ptr(|| {
        let input = FfiInputFile { raw: input_file };
        let reader = VectorIndexReader::open(input).map_err(|e| format!("open reader: {}", e))?;
        Ok(Box::into_raw(Box::new(PaimonVindexReaderHandle {
            inner: reader,
        })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_free(handle: *mut PaimonVindexReaderHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_metadata(
    handle: *const PaimonVindexReaderHandle,
    out: *mut PaimonVindexMetadata,
) -> c_int {
    ffi_status(|| {
        if out.is_null() {
            return Err("out pointer is null".to_string());
        }
        let handle = unsafe { reader_ref(handle) }?;
        unsafe {
            *out = metadata_to_ffi(handle.inner.metadata());
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_optimize_for_search(
    handle: *mut PaimonVindexReaderHandle,
) -> c_int {
    ffi_status(|| {
        let handle = unsafe { reader_mut(handle) }?;
        handle
            .inner
            .optimize_for_search()
            .map_err(|e| format!("optimize_for_search: {}", e))
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_search(
    handle: *mut PaimonVindexReaderHandle,
    query: *const f32,
    top_k: usize,
    nprobe: usize,
    ef_search: usize,
    out_ids: *mut i64,
    out_distances: *mut f32,
    result_len: usize,
) -> c_int {
    ffi_status(|| {
        let handle = unsafe { reader_mut(handle) }?;
        let query = unsafe { const_slice(query, handle.inner.dimension(), "query") }?;
        let params = VectorSearchParams::with_ef_search(top_k, nprobe, ef_search);
        let (ids, distances) = handle
            .inner
            .search(query, params)
            .map_err(|e| format!("search: {}", e))?;
        copy_search_result(&ids, &distances, out_ids, out_distances, result_len, top_k)
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_search_with_roaring_filter(
    handle: *mut PaimonVindexReaderHandle,
    query: *const f32,
    top_k: usize,
    nprobe: usize,
    ef_search: usize,
    roaring_filter: *const u8,
    roaring_filter_len: usize,
    out_ids: *mut i64,
    out_distances: *mut f32,
    result_len: usize,
) -> c_int {
    ffi_status(|| {
        let handle = unsafe { reader_mut(handle) }?;
        let query = unsafe { const_slice(query, handle.inner.dimension(), "query") }?;
        let filter = unsafe { const_slice(roaring_filter, roaring_filter_len, "roaring_filter") }?;
        let params = VectorSearchParams::with_ef_search(top_k, nprobe, ef_search);
        let (ids, distances) = handle
            .inner
            .search_with_roaring_filter(query, params, filter)
            .map_err(|e| format!("search_with_roaring_filter: {}", e))?;
        copy_search_result(&ids, &distances, out_ids, out_distances, result_len, top_k)
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_search_batch(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    query_count: usize,
    top_k: usize,
    nprobe: usize,
    ef_search: usize,
    out_ids: *mut i64,
    out_distances: *mut f32,
    result_len: usize,
) -> c_int {
    ffi_status(|| {
        let handle = unsafe { reader_mut(handle) }?;
        let query_len = checked_len(query_count, handle.inner.dimension(), "queries")?;
        let queries = unsafe { const_slice(queries, query_len, "queries") }?;
        let expected_len = checked_len(query_count, top_k, "batch result")?;
        let params = VectorSearchParams::with_ef_search(top_k, nprobe, ef_search);
        let (ids, distances) = handle
            .inner
            .search_batch(queries, query_count, params)
            .map_err(|e| format!("search_batch: {}", e))?;
        copy_search_result(
            &ids,
            &distances,
            out_ids,
            out_distances,
            result_len,
            expected_len,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn paimon_vindex_reader_search_batch_with_roaring_filter(
    handle: *mut PaimonVindexReaderHandle,
    queries: *const f32,
    query_count: usize,
    top_k: usize,
    nprobe: usize,
    ef_search: usize,
    roaring_filter: *const u8,
    roaring_filter_len: usize,
    out_ids: *mut i64,
    out_distances: *mut f32,
    result_len: usize,
) -> c_int {
    ffi_status(|| {
        let handle = unsafe { reader_mut(handle) }?;
        let query_len = checked_len(query_count, handle.inner.dimension(), "queries")?;
        let queries = unsafe { const_slice(queries, query_len, "queries") }?;
        let filter = unsafe { const_slice(roaring_filter, roaring_filter_len, "roaring_filter") }?;
        let expected_len = checked_len(query_count, top_k, "batch result")?;
        let params = VectorSearchParams::with_ef_search(top_k, nprobe, ef_search);
        let (ids, distances) = handle
            .inner
            .search_batch_with_roaring_filter(queries, query_count, params, filter)
            .map_err(|e| format!("search_batch_with_roaring_filter: {}", e))?;
        copy_search_result(
            &ids,
            &distances,
            out_ids,
            out_distances,
            result_len,
            expected_len,
        )
    })
}
