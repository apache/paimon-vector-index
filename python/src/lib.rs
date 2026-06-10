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
use paimon_vindex_core::index::{
    IndexType, VectorIndexConfig, VectorIndexReader as CoreVectorIndexReader,
    VectorIndexWriter as CoreVectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::{ReadRequest, SeekRead, SeekWrite};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::io;

struct PyVectorIndexInput {
    input: PyObject,
}

impl SeekRead for PyVectorIndexInput {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        Python::with_gil(|py| {
            if !self
                .input
                .bind(py)
                .hasattr("pread_many")
                .map_err(|e| io::Error::other(format!("hasattr(pread_many): {}", e)))?
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Python input must define pread_many(ranges)",
                ));
            }

            let request_list = PyList::empty_bound(py);
            for range in ranges.iter() {
                request_list
                    .append((range.pos, range.buf.len()))
                    .map_err(|e| io::Error::other(format!("build pread_many request: {}", e)))?;
            }
            let result = self
                .input
                .call_method1(py, "pread_many", (request_list,))
                .map_err(|e| io::Error::other(format!("pread_many: {}", e)))?;
            let result_list: &Bound<PyList> = result
                .downcast_bound(py)
                .map_err(|e| io::Error::other(format!("pread_many result: {}", e)))?;
            if result_list.len() != ranges.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "pread_many returned {} buffers for {} ranges",
                        result_list.len(),
                        ranges.len()
                    ),
                ));
            }
            for (idx, range) in ranges.iter_mut().enumerate() {
                let item = result_list
                    .get_item(idx)
                    .map_err(|e| io::Error::other(format!("pread_many item: {}", e)))?;
                copy_py_bytes(&item, range.buf)?;
            }
            Ok(())
        })
    }
}

fn copy_py_bytes(value: &Bound<'_, PyAny>, buf: &mut [u8]) -> io::Result<()> {
    let bytes: &Bound<PyBytes> = value
        .downcast()
        .map_err(|e| io::Error::other(format!("downcast bytes: {}", e)))?;
    let data = bytes.as_bytes();
    if data.len() != buf.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("pread returned {} of {} bytes", data.len(), buf.len()),
        ));
    }
    buf.copy_from_slice(data);
    Ok(())
}

struct PyOutputStream {
    file: PyObject,
    pos: u64,
}

impl SeekWrite for PyOutputStream {
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

fn metric_name(metric: MetricType) -> &'static str {
    match metric {
        MetricType::L2 => "l2",
        MetricType::InnerProduct => "inner_product",
        MetricType::Cosine => "cosine",
    }
}

fn index_type_name(index_type: IndexType) -> &'static str {
    index_type.as_str()
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
struct VectorIndexMetadata {
    #[pyo3(get)]
    index_type: String,
    #[pyo3(get)]
    dimension: usize,
    #[pyo3(get)]
    nlist: usize,
    #[pyo3(get)]
    metric: String,
    #[pyo3(get)]
    total_vectors: i64,
    #[pyo3(get)]
    pq_m: Option<usize>,
    #[pyo3(get)]
    hnsw_m: Option<usize>,
    #[pyo3(get)]
    hnsw_ef_construction: Option<usize>,
    #[pyo3(get)]
    hnsw_max_level: Option<usize>,
}

fn options_from_py(options: &Bound<'_, PyAny>) -> PyResult<HashMap<String, String>> {
    let dict: &Bound<PyDict> = options
        .downcast()
        .map_err(|_| PyValueError::new_err("options must be a dict[str, str]"))?;
    let mut result = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| PyValueError::new_err("option keys must be strings"))?;
        let value = value
            .extract::<String>()
            .map_err(|_| PyValueError::new_err("option values must be strings"))?;
        result.insert(key, value);
    }
    Ok(result)
}

fn config_from_options(options: &Bound<'_, PyAny>) -> PyResult<VectorIndexConfig> {
    VectorIndexConfig::from_options(&options_from_py(options)?)
        .map_err(|e| PyValueError::new_err(format!("invalid vector index options: {}", e)))
}

#[pyclass]
struct VectorIndexWriter {
    index: Option<CoreVectorIndexWriter>,
    dimension: usize,
}

#[pymethods]
impl VectorIndexWriter {
    #[new]
    fn new(options: &Bound<'_, PyAny>) -> PyResult<Self> {
        let config = config_from_options(options)?;
        let dimension = config.dimension();
        let index = CoreVectorIndexWriter::new(config)
            .map_err(|e| PyValueError::new_err(format!("failed to create writer: {}", e)))?;
        Ok(Self {
            index: Some(index),
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
        self.index_mut()?
            .train(data_slice, row_count)
            .map_err(|e| PyValueError::new_err(format!("train failed: {}", e)))?;
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
        self.index_mut()?
            .add_vectors(id_slice, data_slice, row_count)
            .map_err(|e| PyValueError::new_err(format!("add_vectors failed: {}", e)))?;
        Ok(())
    }

    fn write(&mut self, file: PyObject) -> PyResult<()> {
        let mut stream = PyOutputStream { file, pos: 0 };
        self.index_mut()?
            .write(&mut stream)
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

impl VectorIndexWriter {
    fn index_mut(&mut self) -> PyResult<&mut CoreVectorIndexWriter> {
        self.index
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("VectorIndexWriter is closed"))
    }
}

#[pyclass]
struct VectorIndexReader {
    inner: CoreVectorIndexReader<PyVectorIndexInput>,
}

#[pymethods]
impl VectorIndexReader {
    #[new]
    fn new(input: PyObject) -> PyResult<Self> {
        let stream = PyVectorIndexInput { input };
        let reader = CoreVectorIndexReader::open(stream)
            .map_err(|e| PyIOError::new_err(format!("Failed to open index: {}", e)))?;
        Ok(Self { inner: reader })
    }

    #[getter]
    fn index_type(&self) -> String {
        index_type_name(self.inner.index_type()).to_string()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.metadata().dimension
    }

    #[getter]
    fn nlist(&self) -> usize {
        self.inner.metadata().nlist
    }

    #[getter]
    fn total_vectors(&self) -> i64 {
        self.inner.metadata().total_vectors
    }

    fn metadata(&self) -> VectorIndexMetadata {
        let metadata = self.inner.metadata();
        VectorIndexMetadata {
            index_type: index_type_name(metadata.index_type).to_string(),
            dimension: metadata.dimension,
            nlist: metadata.nlist,
            metric: metric_name(metadata.metric).to_string(),
            total_vectors: metadata.total_vectors,
            pq_m: metadata.pq_m,
            hnsw_m: metadata.hnsw.map(|h| h.m),
            hnsw_ef_construction: metadata.hnsw.map(|h| h.ef_construction),
            hnsw_max_level: metadata.hnsw.map(|h| h.max_level),
        }
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (query, top_k, nprobe, ef_search=0, filter_bytes=None))]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        top_k: usize,
        nprobe: usize,
        ef_search: usize,
        filter_bytes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        let query_slice = query.as_slice()?;
        let dimension = self.inner.metadata().dimension;
        if query_slice.len() != dimension {
            return Err(PyValueError::new_err(format!(
                "query length {} != index dimension {}",
                query_slice.len(),
                dimension
            )));
        }
        validate_positive(top_k, "top_k")?;
        validate_positive(nprobe, "nprobe")?;
        let params = VectorSearchParams::with_ef_search(top_k, nprobe, ef_search);

        let (ids, dists) = if let Some(bytes) = decode_filter_bytes(filter_bytes)? {
            self.inner
                .search_with_roaring_filter(query_slice, params, bytes)
                .map_err(|e| PyIOError::new_err(format!("Search failed: {}", e)))?
        } else {
            self.inner
                .search(query_slice, params)
                .map_err(|e| PyIOError::new_err(format!("Search failed: {}", e)))?
        };

        Ok((
            PyArray1::from_vec_bound(py, ids),
            PyArray1::from_vec_bound(py, dists),
        ))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (queries, top_k, nprobe, ef_search=0, filter_bytes=None))]
    fn search_batch<'py>(
        &mut self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        top_k: usize,
        nprobe: usize,
        ef_search: usize,
        filter_bytes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let dimension = self.inner.metadata().dimension;
        let shape = queries.shape();
        let query_count = validate_matrix_shape(shape, dimension, "query", "index dimension")?;
        validate_positive(top_k, "top_k")?;
        validate_positive(nprobe, "nprobe")?;
        let query_slice = queries.as_slice().map_err(|_| {
            PyValueError::new_err("queries must be a contiguous two-dimensional float32 array")
        })?;
        let params = VectorSearchParams::with_ef_search(top_k, nprobe, ef_search);

        let (ids, dists) = if let Some(bytes) = decode_filter_bytes(filter_bytes)? {
            self.inner
                .search_batch_with_roaring_filter(query_slice, query_count, params, bytes)
                .map_err(|e| PyIOError::new_err(format!("Batch search failed: {}", e)))?
        } else {
            self.inner
                .search_batch(query_slice, query_count, params)
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

#[pymodule]
fn paimon_vindex(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<VectorIndexMetadata>()?;
    m.add_class::<VectorIndexReader>()?;
    m.add_class::<VectorIndexWriter>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray, PyArrayMethods};
    use pyo3::types::PyBytes;
    use roaring::RoaringTreemap;

    fn generate_clustered_data(n: usize, d: usize, clusters: usize) -> Vec<f32> {
        let mut data = vec![0.0; n * d];
        for i in 0..n {
            let cluster = i % clusters;
            for j in 0..d {
                data[i * d + j] = cluster as f32 * 20.0 + j as f32 * 0.01 + i as f32 * 0.0001;
            }
        }
        data
    }

    fn write_index_bytes<'py>(
        py: Python<'py>,
        config: &Bound<'py, PyAny>,
        d: usize,
    ) -> Bound<'py, PyAny> {
        let io = py.import_bound("io").unwrap();
        let output = io.getattr("BytesIO").unwrap().call0().unwrap();
        let mut writer = VectorIndexWriter::new(config).unwrap();
        let data = generate_clustered_data(500, d, 4);
        let ids: Vec<i64> = (0..500).collect();
        let train = PyArray::from_vec2_bound(
            py,
            &data
                .chunks(d)
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
        output
    }

    fn vector_index_input<'py>(py: Python<'py>, output: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        let data = output
            .call_method0("getvalue")
            .unwrap()
            .downcast_into::<PyBytes>()
            .unwrap();
        Py::new(
            py,
            PyBytesVectorIndexInput {
                data: data.as_bytes().to_vec(),
            },
        )
        .unwrap()
        .into_bound(py)
        .into_any()
    }

    fn options<'py>(py: Python<'py>, values: &[(&str, &str)]) -> Bound<'py, PyAny> {
        let dict = PyDict::new_bound(py);
        for (key, value) in values {
            dict.set_item(*key, *value).unwrap();
        }
        dict.into_any()
    }

    #[pyclass]
    struct PyBytesVectorIndexInput {
        data: Vec<u8>,
    }

    #[pymethods]
    impl PyBytesVectorIndexInput {
        fn pread_many<'py>(
            &self,
            py: Python<'py>,
            ranges: &Bound<'_, PyList>,
        ) -> PyResult<Bound<'py, PyList>> {
            let result = PyList::empty_bound(py);
            for item in ranges.iter() {
                let (pos, len): (usize, usize) = item.extract()?;
                let end = pos
                    .checked_add(len)
                    .ok_or_else(|| PyIOError::new_err("pread_many range position overflow"))?;
                if end > self.data.len() {
                    return Err(PyIOError::new_err(format!(
                        "pread_many range {}..{} out of bounds {}",
                        pos,
                        end,
                        self.data.len()
                    )));
                }
                result.append(PyBytes::new_bound(py, &self.data[pos..end]))?;
            }
            Ok(result)
        }
    }

    #[test]
    fn python_unified_writer_reader_roundtrips_supported_indexes() {
        Python::with_gil(|py| {
            let configs: Vec<(Bound<'_, PyAny>, usize, &str)> = vec![
                (
                    options(
                        py,
                        &[
                            ("index.type", "ivf_flat"),
                            ("dimension", "16"),
                            ("nlist", "4"),
                            ("metric", "l2"),
                        ],
                    ),
                    16,
                    "ivf_flat",
                ),
                (
                    options(
                        py,
                        &[
                            ("index.type", "ivf_pq"),
                            ("dimension", "16"),
                            ("nlist", "4"),
                            ("pq.m", "4"),
                            ("metric", "l2"),
                            ("use-opq", "false"),
                        ],
                    ),
                    16,
                    "ivf_pq",
                ),
                (
                    options(
                        py,
                        &[
                            ("index.type", "ivf_hnsw_flat"),
                            ("dimension", "16"),
                            ("nlist", "4"),
                            ("metric", "l2"),
                        ],
                    ),
                    16,
                    "ivf_hnsw_flat",
                ),
                (
                    options(
                        py,
                        &[
                            ("index.type", "ivf_hnsw_sq"),
                            ("dimension", "16"),
                            ("nlist", "4"),
                            ("metric", "l2"),
                            ("hnsw.m", "12"),
                        ],
                    ),
                    16,
                    "ivf_hnsw_sq",
                ),
            ];

            for (config, d, expected_type) in configs {
                let output = write_index_bytes(py, &config, d);
                let input = vector_index_input(py, &output);
                let mut reader = VectorIndexReader::new(input.unbind()).unwrap();
                assert_eq!(reader.index_type(), expected_type);
                assert_eq!(reader.dimension(), d);
                assert_eq!(reader.metadata().index_type, expected_type);

                let data = generate_clustered_data(1, d, 1);
                let query = PyArray1::from_vec_bound(py, data[0..d].to_vec());
                let (result_ids, _) = reader.search(py, query.readonly(), 5, 4, 32, None).unwrap();
                assert_eq!(result_ids.len(), 5);
                assert_eq!(result_ids.readonly().as_slice().unwrap()[0], 0);
            }
        });
    }

    #[test]
    fn python_batch_search_accepts_roaring_filter_bytes() {
        Python::with_gil(|py| {
            let config = options(
                py,
                &[
                    ("index.type", "ivf_flat"),
                    ("dimension", "2"),
                    ("nlist", "1"),
                    ("metric", "l2"),
                ],
            );
            let io = py.import_bound("io").unwrap();
            let output = io.getattr("BytesIO").unwrap().call0().unwrap();
            let mut writer = VectorIndexWriter::new(&config).unwrap();
            let train = PyArray::from_vec2_bound(
                py,
                &[vec![0.0f32, 0.0], vec![0.1, 0.0], vec![10.0, 10.0]],
            )
            .unwrap();
            let id_array = PyArray1::from_vec_bound(py, vec![10i64, 11, 12]);

            writer.train(train.readonly()).unwrap();
            writer
                .add_vectors(id_array.readonly(), train.readonly())
                .unwrap();
            writer.write(output.as_any().clone().unbind()).unwrap();

            let mut allowed = RoaringTreemap::new();
            allowed.insert(12);
            let mut filter_bytes = Vec::new();
            allowed.serialize_into(&mut filter_bytes).unwrap();
            let filter = PyBytes::new_bound(py, &filter_bytes);

            let input = vector_index_input(py, &output);
            let mut reader = VectorIndexReader::new(input.unbind()).unwrap();
            let queries =
                PyArray::from_vec2_bound(py, &[vec![0.0f32, 0.0], vec![10.0, 10.0]]).unwrap();
            let (result_ids, result_dists) = reader
                .search_batch(py, queries.readonly(), 2, 1, 0, Some(filter.as_any()))
                .unwrap();

            assert_eq!(result_ids.shape(), &[2, 2]);
            assert_eq!(result_ids.readonly().as_slice().unwrap(), &[12, -1, 12, -1]);
            assert_eq!(result_dists.readonly().as_slice().unwrap()[1], f32::MAX);
        });
    }

    #[test]
    fn python_batch_search_validates_query_shape() {
        Python::with_gil(|py| {
            let config = options(
                py,
                &[
                    ("index.type", "ivf_pq"),
                    ("dimension", "16"),
                    ("nlist", "4"),
                    ("pq.m", "4"),
                    ("metric", "l2"),
                ],
            );
            let output = write_index_bytes(py, &config, 16);
            let input = vector_index_input(py, &output);
            let mut reader = VectorIndexReader::new(input.unbind()).unwrap();
            let wrong_dim = PyArray::from_vec2_bound(py, &[vec![0.0f32; 15]]).unwrap();

            let err = reader
                .search_batch(py, wrong_dim.readonly(), 5, 2, 0, None)
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
            let config = options(
                py,
                &[
                    ("index.type", "ivf_pq"),
                    ("dimension", "16"),
                    ("nlist", "4"),
                    ("pq.m", "4"),
                    ("metric", "l2"),
                ],
            );
            let mut writer = VectorIndexWriter::new(&config).unwrap();
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
