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

#![allow(clippy::useless_conversion)]

use numpy::{PyArray1, PyReadonlyArray1};
use paimon_vindex_core::io::{IVFPQIndexReader, SeekRead};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io;

/// Python file object wrapper implementing SeekRead.
struct PyFileStream {
    file: PyObject,
}

impl SeekRead for PyFileStream {
    fn seek(&mut self, pos: u64) -> io::Result<()> {
        Python::with_gil(|py| {
            self.file
                .call_method1(py, "seek", (pos as i64,))
                .map_err(|e| io::Error::other(format!("seek: {}", e)))?;
            Ok(())
        })
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        Python::with_gil(|py| {
            let result = self
                .file
                .call_method1(py, "read", (buf.len(),))
                .map_err(|e| io::Error::other(format!("read: {}", e)))?;

            let bytes: &Bound<PyBytes> = result
                .downcast_bound(py)
                .map_err(|e| io::Error::other(format!("downcast: {}", e)))?;

            let data = bytes.as_bytes();
            if data.len() != buf.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("read {} of {} bytes", data.len(), buf.len()),
                ));
            }
            buf.copy_from_slice(data);
            Ok(())
        })
    }
}

#[pyclass]
struct IVFPQReader {
    inner: IVFPQIndexReader<PyFileStream>,
}

#[pymethods]
impl IVFPQReader {
    #[new]
    fn new(file: PyObject) -> PyResult<Self> {
        let stream = PyFileStream { file };
        let reader = IVFPQIndexReader::open(stream)
            .map_err(|e| PyIOError::new_err(format!("Failed to open index: {}", e)))?;
        Ok(IVFPQReader { inner: reader })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.d
    }

    #[getter]
    fn nlist(&self) -> usize {
        self.inner.nlist
    }

    #[getter]
    fn m(&self) -> usize {
        self.inner.m
    }

    #[getter]
    fn total_vectors(&self) -> i64 {
        self.inner.total_vectors
    }

    #[allow(clippy::type_complexity)]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        top_k: usize,
        nprobe: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        let query_slice = query.as_slice()?;

        if query_slice.len() != self.inner.d {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query length {} != index dimension {}",
                query_slice.len(),
                self.inner.d
            )));
        }
        if top_k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("top_k must be > 0"));
        }
        if nprobe == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nprobe must be > 0",
            ));
        }

        let (ids, dists) = self
            .inner
            .search(query_slice, top_k, nprobe)
            .map_err(|e| PyIOError::new_err(format!("Search failed: {}", e)))?;

        let id_array = PyArray1::from_vec_bound(py, ids);
        let dist_array = PyArray1::from_vec_bound(py, dists);

        Ok((id_array, dist_array))
    }

    fn close(&mut self) -> PyResult<()> {
        Ok(())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, pyo3::types::PyType>>,
        _exc_val: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

#[pymodule]
fn paimon_vindex(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<IVFPQReader>()?;
    Ok(())
}
