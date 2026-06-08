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

use numpy::{
    PyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::io::{write_index, IVFPQIndexReader, SeekRead};
use paimon_vindex_core::ivfpq::{
    search_batch_reader, search_batch_reader_roaring_filter, IVFPQIndex,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
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

/// Python file object wrapper implementing SeekWrite.
struct PyOutputStream {
    file: PyObject,
    pos: u64,
}

impl paimon_vindex_core::io::SeekWrite for PyOutputStream {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        Python::with_gil(|py| {
            let bytes = PyBytes::new_bound(py, buf);
            let written = self
                .file
                .call_method1(py, "write", (bytes,))
                .map_err(|e| io::Error::other(format!("write: {}", e)))?
                .extract::<usize>(py)
                .map_err(|e| io::Error::other(format!("write return value: {}", e)))?;
            if written != buf.len() {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    format!("write accepted {} of {} bytes", written, buf.len()),
                ));
            }
            self.pos += buf.len() as u64;
            Ok(())
        })
    }

    fn pos(&self) -> u64 {
        self.pos
    }
}

fn parse_metric(metric: &str) -> PyResult<MetricType> {
    match metric.to_ascii_lowercase().as_str() {
        "l2" => Ok(MetricType::L2),
        "inner_product" | "ip" => Ok(MetricType::InnerProduct),
        "cosine" => Ok(MetricType::Cosine),
        _ => Err(PyValueError::new_err(format!(
            "unknown metric '{}'; expected 'l2', 'inner_product', or 'cosine'",
            metric
        ))),
    }
}

fn validate_positive(value: usize, name: &str) -> PyResult<()> {
    if value == 0 {
        Err(PyValueError::new_err(format!("{} must be > 0", name)))
    } else {
        Ok(())
    }
}

fn decode_filter_bytes<'a>(
    filter_bytes: Option<&'a Bound<'_, PyAny>>,
) -> PyResult<Option<&'a [u8]>> {
    if let Some(filter_obj) = filter_bytes {
        let bytes: &Bound<PyBytes> = filter_obj
            .downcast()
            .map_err(|_| PyValueError::new_err("filter_bytes must be bytes"))?;
        Ok(Some(bytes.as_bytes()))
    } else {
        Ok(None)
    }
}

fn pyarray2_from_flat<'py, T: numpy::Element + Clone>(
    py: Python<'py>,
    data: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let matrix = data
        .chunks(cols)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    debug_assert_eq!(matrix.len(), rows);
    PyArray::from_vec2_bound(py, &matrix)
        .map_err(|e| PyValueError::new_err(format!("reshape batch result: {}", e)))
}

fn validate_matrix_shape(
    shape: &[usize],
    dimension: usize,
    value_name: &str,
    dimension_name: &str,
) -> PyResult<usize> {
    let row_count = shape[0];
    let actual_dimension = shape[1];
    if actual_dimension != dimension {
        return Err(PyValueError::new_err(format!(
            "{} dimension {} != {} {}",
            value_name, actual_dimension, dimension_name, dimension
        )));
    }
    if row_count == 0 {
        return Err(PyValueError::new_err(format!(
            "{} must contain at least one row",
            value_name
        )));
    }
    Ok(row_count)
}

#[pyclass]
struct IVFPQReader {
    inner: IVFPQIndexReader<PyFileStream>,
}

#[pyclass]
struct IVFPQWriter {
    index: Option<IVFPQIndex>,
    dimension: usize,
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
    #[pyo3(signature = (query, top_k, nprobe, filter_bytes=None))]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        top_k: usize,
        nprobe: usize,
        filter_bytes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        let query_slice = query.as_slice()?;

        if query_slice.len() != self.inner.d {
            return Err(PyValueError::new_err(format!(
                "query length {} != index dimension {}",
                query_slice.len(),
                self.inner.d
            )));
        }
        validate_positive(top_k, "top_k")?;
        validate_positive(nprobe, "nprobe")?;

        let (ids, dists) = if let Some(bytes) = decode_filter_bytes(filter_bytes)? {
            self.inner
                .search_with_roaring_filter(query_slice, top_k, nprobe, bytes)
                .map_err(|e| PyIOError::new_err(format!("Search failed: {}", e)))?
        } else {
            self.inner
                .search(query_slice, top_k, nprobe)
                .map_err(|e| PyIOError::new_err(format!("Search failed: {}", e)))?
        };

        let id_array = PyArray1::from_vec_bound(py, ids);
        let dist_array = PyArray1::from_vec_bound(py, dists);

        Ok((id_array, dist_array))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (queries, top_k, nprobe, filter_bytes=None))]
    fn search_batch<'py>(
        &mut self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        top_k: usize,
        nprobe: usize,
        filter_bytes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let shape = queries.shape();
        let query_count = validate_matrix_shape(shape, self.inner.d, "query", "index dimension")?;
        validate_positive(top_k, "top_k")?;
        validate_positive(nprobe, "nprobe")?;

        let query_slice = queries.as_slice().map_err(|_| {
            PyValueError::new_err("queries must be a contiguous two-dimensional float32 array")
        })?;

        let (ids, dists) = if let Some(bytes) = decode_filter_bytes(filter_bytes)? {
            search_batch_reader_roaring_filter(
                &mut self.inner,
                query_slice,
                query_count,
                top_k,
                nprobe,
                bytes,
            )
            .map_err(|e| PyIOError::new_err(format!("Batch search failed: {}", e)))?
        } else {
            search_batch_reader(&mut self.inner, query_slice, query_count, top_k, nprobe)
                .map_err(|e| PyIOError::new_err(format!("Batch search failed: {}", e)))?
        };

        Ok((
            pyarray2_from_flat(py, ids, query_count, top_k)?,
            pyarray2_from_flat(py, dists, query_count, top_k)?,
        ))
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

#[pymethods]
impl IVFPQWriter {
    #[new]
    #[pyo3(signature = (dimension, nlist, m, metric="l2", use_opq=false))]
    fn new(
        dimension: usize,
        nlist: usize,
        m: usize,
        metric: &str,
        use_opq: bool,
    ) -> PyResult<Self> {
        validate_positive(dimension, "dimension")?;
        validate_positive(nlist, "nlist")?;
        validate_positive(m, "m")?;
        if !dimension.is_multiple_of(m) {
            return Err(PyValueError::new_err(format!(
                "dimension {} must be divisible by m {}",
                dimension, m
            )));
        }
        let metric = parse_metric(metric)?;
        Ok(IVFPQWriter {
            index: Some(IVFPQIndex::new(dimension, nlist, m, metric, use_opq)),
            dimension,
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn train(&mut self, data: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = data.shape();
        let row_count = validate_matrix_shape(shape, self.dimension, "data", "writer dimension")?;
        let data_slice = data.as_slice().map_err(|_| {
            PyValueError::new_err("data must be a contiguous two-dimensional float32 array")
        })?;
        self.index_mut()?.train(data_slice, row_count);
        Ok(())
    }

    fn add_vectors(
        &mut self,
        ids: PyReadonlyArray1<i64>,
        data: PyReadonlyArray2<f32>,
    ) -> PyResult<()> {
        let shape = data.shape();
        let row_count = validate_matrix_shape(shape, self.dimension, "data", "writer dimension")?;
        let id_slice = ids.as_slice()?;
        if id_slice.len() != row_count {
            return Err(PyValueError::new_err(format!(
                "ids length {} != vector count {}",
                id_slice.len(),
                row_count
            )));
        }
        let data_slice = data.as_slice().map_err(|_| {
            PyValueError::new_err("data must be a contiguous two-dimensional float32 array")
        })?;
        self.index_mut()?.add(data_slice, id_slice, row_count);
        Ok(())
    }

    fn write(&mut self, file: PyObject) -> PyResult<()> {
        let mut stream = PyOutputStream { file, pos: 0 };
        write_index(self.index_ref()?, &mut stream)
            .map_err(|e| PyIOError::new_err(format!("Failed to write index: {}", e)))?;
        Ok(())
    }

    fn close(&mut self) -> PyResult<()> {
        self.index = None;
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

impl IVFPQWriter {
    fn index_ref(&self) -> PyResult<&IVFPQIndex> {
        self.index
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("IVFPQWriter is closed"))
    }

    fn index_mut(&mut self) -> PyResult<&mut IVFPQIndex> {
        self.index
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("IVFPQWriter is closed"))
    }
}

#[pymodule]
fn paimon_vindex(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<IVFPQReader>()?;
    m.add_class::<IVFPQWriter>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray, PyArrayMethods};
    use paimon_vindex_core::distance::MetricType;
    use paimon_vindex_core::io::{write_index, PosWriter};
    use paimon_vindex_core::ivfpq::IVFPQIndex;
    use pyo3::types::PyBytes;
    use roaring::RoaringTreemap;

    fn generate_clustered_data(n: usize, d: usize, clusters: usize) -> Vec<f32> {
        let mut data = vec![0.0; n * d];
        for i in 0..n {
            let cluster = i % clusters;
            for j in 0..d {
                data[i * d + j] = cluster as f32 * 10.0 + j as f32 * 0.01 + i as f32 * 0.0001;
            }
        }
        data
    }

    fn build_test_index_bytes() -> Vec<u8> {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;
        let data = generate_clustered_data(n, d, 4);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();
        buf
    }

    #[test]
    fn python_batch_search_returns_2d_numpy_arrays() {
        Python::with_gil(|py| {
            let io = py.import_bound("io").unwrap();
            let file = io
                .getattr("BytesIO")
                .unwrap()
                .call1((PyBytes::new_bound(py, &build_test_index_bytes()),))
                .unwrap();
            let mut reader = IVFPQReader::new(file.unbind()).unwrap();
            let queries = generate_clustered_data(3, reader.dimension(), 4);
            let query_array = PyArray::from_vec2_bound(
                py,
                &queries
                    .chunks(reader.dimension())
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
            .unwrap();

            let (ids, dists) = reader
                .search_batch(py, query_array.readonly(), 5, 2, None)
                .unwrap();

            assert_eq!(ids.shape(), &[3, 5]);
            assert_eq!(dists.shape(), &[3, 5]);
            assert_eq!(ids.readonly().as_slice().unwrap()[0], 0);
        });
    }

    #[test]
    fn python_batch_search_accepts_roaring_filter_bytes() {
        Python::with_gil(|py| {
            let io = py.import_bound("io").unwrap();
            let file = io
                .getattr("BytesIO")
                .unwrap()
                .call1((PyBytes::new_bound(py, &build_test_index_bytes()),))
                .unwrap();
            let mut reader = IVFPQReader::new(file.unbind()).unwrap();
            let queries = generate_clustered_data(3, reader.dimension(), 4);
            let query_array = PyArray::from_vec2_bound(
                py,
                &queries
                    .chunks(reader.dimension())
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
            .unwrap();

            let mut allowed = RoaringTreemap::new();
            for id in (0..500u64).filter(|id| id % 7 == 0) {
                allowed.insert(id);
            }
            let mut filter_bytes = Vec::new();
            allowed.serialize_into(&mut filter_bytes).unwrap();
            let filter = PyBytes::new_bound(py, &filter_bytes);

            let (ids, _) = reader
                .search_batch(py, query_array.readonly(), 5, 2, Some(filter.as_any()))
                .unwrap();

            assert_eq!(ids.shape(), &[3, 5]);
            for &id in ids.readonly().as_slice().unwrap() {
                if id >= 0 {
                    assert_eq!(id % 7, 0);
                }
            }
        });
    }

    #[test]
    fn python_writer_can_build_an_index_for_reader() {
        Python::with_gil(|py| {
            let io = py.import_bound("io").unwrap();
            let output = io.getattr("BytesIO").unwrap().call0().unwrap();
            let mut writer = IVFPQWriter::new(16, 4, 4, "l2", false).unwrap();
            let data = generate_clustered_data(500, 16, 4);
            let ids: Vec<i64> = (0..500).collect();

            let train = PyArray::from_vec2_bound(
                py,
                &data
                    .chunks(16)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let id_array = PyArray1::from_vec_bound(py, ids);

            writer.train(train.readonly()).unwrap();
            writer
                .add_vectors(id_array.readonly(), train.readonly())
                .unwrap();
            writer.write(output.as_any().clone().unbind()).unwrap();

            output.call_method1("seek", (0,)).unwrap();
            let mut reader = IVFPQReader::new(output.unbind()).unwrap();
            let query = PyArray1::from_vec_bound(py, data[0..16].to_vec());

            let (result_ids, _) = reader.search(py, query.readonly(), 5, 2, None).unwrap();

            assert_eq!(result_ids.len(), 5);
            assert_eq!(result_ids.readonly().as_slice().unwrap()[0], 0);
        });
    }

    #[test]
    fn python_batch_search_validates_query_shape() {
        Python::with_gil(|py| {
            let io = py.import_bound("io").unwrap();
            let file = io
                .getattr("BytesIO")
                .unwrap()
                .call1((PyBytes::new_bound(py, &build_test_index_bytes()),))
                .unwrap();
            let mut reader = IVFPQReader::new(file.unbind()).unwrap();
            let wrong_dim = PyArray::from_vec2_bound(py, &[vec![0.0f32; 15]]).unwrap();

            let err = reader
                .search_batch(py, wrong_dim.readonly(), 5, 2, None)
                .unwrap_err();

            assert!(err
                .to_string()
                .contains("query dimension 15 != index dimension 16"));
        });
    }

    #[test]
    fn python_writer_rejects_short_writes() {
        Python::with_gil(|py| {
            let locals = pyo3::types::PyDict::new_bound(py);
            py.run_bound(
                r#"
class ShortWriter:
    def write(self, data):
        return max(0, len(data) - 1)
"#,
                None,
                Some(&locals),
            )
            .unwrap();
            let output = locals
                .get_item("ShortWriter")
                .unwrap()
                .unwrap()
                .call0()
                .unwrap();
            let mut writer = IVFPQWriter::new(16, 4, 4, "l2", false).unwrap();
            let data = generate_clustered_data(500, 16, 4);
            let train = PyArray::from_vec2_bound(
                py,
                &data
                    .chunks(16)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
            .unwrap();

            writer.train(train.readonly()).unwrap();
            let err = writer.write(output.unbind()).unwrap_err();

            assert!(err.to_string().contains("write accepted"));
        });
    }
}
