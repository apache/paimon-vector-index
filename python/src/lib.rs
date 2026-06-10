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
use paimon_vindex_core::hnsw::HnswBuildParams;
use paimon_vindex_core::index::{
    IndexType, VectorIndexConfig, VectorIndexReader as CoreVectorIndexReader,
    VectorIndexWriter as CoreVectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::{ReadRequest, SeekRead, SeekWrite};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyList};
use std::io;

struct PyFileStream {
    file: PyObject,
}

impl SeekRead for PyFileStream {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        Python::with_gil(|py| {
            if self
                .file
                .bind(py)
                .hasattr("pread_many")
                .map_err(|e| io::Error::other(format!("hasattr(pread_many): {}", e)))?
            {
                let request_list = PyList::empty_bound(py);
                for range in ranges.iter() {
                    request_list
                        .append((range.pos, range.buf.len()))
                        .map_err(|e| io::Error::other(format!("build pread_many request: {}", e)))?;
                }
                let result = self
                    .file
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
                return Ok(());
            }

            if !self
                .file
                .bind(py)
                .hasattr("pread")
                .map_err(|e| io::Error::other(format!("hasattr(pread): {}", e)))?
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Python input must define pread_many(ranges) or pread(position, length)",
                ));
            }

            for range in ranges.iter_mut() {
                let result = self
                    .file
                    .call_method1(py, "pread", (range.pos, range.buf.len()))
                    .map_err(|e| io::Error::other(format!("pread: {}", e)))?;
                copy_py_bytes(result.bind(py), range.buf)?;
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

fn hnsw_params(hnsw: Option<&HnswConfig>) -> HnswBuildParams {
    hnsw.map(|h| h.to_core())
        .unwrap_or_else(HnswBuildParams::default)
}

#[pyclass]
#[derive(Clone)]
struct HnswConfig {
    #[pyo3(get)]
    m: usize,
    #[pyo3(get)]
    ef_construction: usize,
    #[pyo3(get)]
    max_level: usize,
}

#[pymethods]
impl HnswConfig {
    #[new]
    #[pyo3(signature = (m=20, ef_construction=150, max_level=7))]
    fn new(m: usize, ef_construction: usize, max_level: usize) -> PyResult<Self> {
        validate_positive(m, "m")?;
        validate_positive(ef_construction, "ef_construction")?;
        validate_positive(max_level, "max_level")?;
        Ok(Self {
            m,
            ef_construction,
            max_level,
        })
    }
}

impl HnswConfig {
    fn to_core(&self) -> HnswBuildParams {
        HnswBuildParams {
            m: self.m,
            ef_construction: self.ef_construction,
            max_level: self.max_level,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct IvfFlatConfig {
    #[pyo3(get)]
    dimension: usize,
    #[pyo3(get)]
    nlist: usize,
    #[pyo3(get)]
    metric: String,
}

#[pymethods]
impl IvfFlatConfig {
    #[new]
    #[pyo3(signature = (dimension, nlist, metric="l2"))]
    fn new(dimension: usize, nlist: usize, metric: &str) -> PyResult<Self> {
        validate_positive(dimension, "dimension")?;
        validate_positive(nlist, "nlist")?;
        parse_metric(metric)?;
        Ok(Self {
            dimension,
            nlist,
            metric: metric.to_string(),
        })
    }
}

#[pyclass]
#[derive(Clone)]
struct IvfPqConfig {
    #[pyo3(get)]
    dimension: usize,
    #[pyo3(get)]
    nlist: usize,
    #[pyo3(get)]
    m: usize,
    #[pyo3(get)]
    metric: String,
    #[pyo3(get)]
    use_opq: bool,
}

#[pymethods]
impl IvfPqConfig {
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
        parse_metric(metric)?;
        Ok(Self {
            dimension,
            nlist,
            m,
            metric: metric.to_string(),
            use_opq,
        })
    }
}

#[pyclass]
#[derive(Clone)]
struct IvfHnswFlatConfig {
    #[pyo3(get)]
    dimension: usize,
    #[pyo3(get)]
    nlist: usize,
    #[pyo3(get)]
    metric: String,
    hnsw: HnswConfig,
}

#[pymethods]
impl IvfHnswFlatConfig {
    #[new]
    #[pyo3(signature = (dimension, nlist, metric="l2", hnsw=None))]
    fn new(
        dimension: usize,
        nlist: usize,
        metric: &str,
        hnsw: Option<&HnswConfig>,
    ) -> PyResult<Self> {
        validate_positive(dimension, "dimension")?;
        validate_positive(nlist, "nlist")?;
        parse_metric(metric)?;
        Ok(Self {
            dimension,
            nlist,
            metric: metric.to_string(),
            hnsw: hnsw.cloned().unwrap_or_else(|| HnswConfig {
                m: 20,
                ef_construction: 150,
                max_level: 7,
            }),
        })
    }

    #[getter]
    fn hnsw(&self) -> HnswConfig {
        self.hnsw.clone()
    }
}

#[pyclass]
#[derive(Clone)]
struct IvfHnswSqConfig {
    #[pyo3(get)]
    dimension: usize,
    #[pyo3(get)]
    nlist: usize,
    #[pyo3(get)]
    metric: String,
    hnsw: HnswConfig,
}

#[pymethods]
impl IvfHnswSqConfig {
    #[new]
    #[pyo3(signature = (dimension, nlist, metric="l2", hnsw=None))]
    fn new(
        dimension: usize,
        nlist: usize,
        metric: &str,
        hnsw: Option<&HnswConfig>,
    ) -> PyResult<Self> {
        validate_positive(dimension, "dimension")?;
        validate_positive(nlist, "nlist")?;
        parse_metric(metric)?;
        Ok(Self {
            dimension,
            nlist,
            metric: metric.to_string(),
            hnsw: hnsw.cloned().unwrap_or_else(|| HnswConfig {
                m: 20,
                ef_construction: 150,
                max_level: 7,
            }),
        })
    }

    #[getter]
    fn hnsw(&self) -> HnswConfig {
        self.hnsw.clone()
    }
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
    hnsw: Option<HnswConfig>,
}

#[pymethods]
impl VectorIndexMetadata {
    #[getter]
    fn hnsw(&self) -> Option<HnswConfig> {
        self.hnsw.clone()
    }
}

fn config_from_py(config: &Bound<'_, PyAny>) -> PyResult<VectorIndexConfig> {
    if let Ok(config) = config.extract::<PyRef<'_, IvfFlatConfig>>() {
        return Ok(VectorIndexConfig::IvfFlat {
            dimension: config.dimension,
            nlist: config.nlist,
            metric: parse_metric(&config.metric)?,
        });
    }
    if let Ok(config) = config.extract::<PyRef<'_, IvfPqConfig>>() {
        return Ok(VectorIndexConfig::IvfPq {
            dimension: config.dimension,
            nlist: config.nlist,
            m: config.m,
            metric: parse_metric(&config.metric)?,
            use_opq: config.use_opq,
        });
    }
    if let Ok(config) = config.extract::<PyRef<'_, IvfHnswFlatConfig>>() {
        return Ok(VectorIndexConfig::IvfHnswFlat {
            dimension: config.dimension,
            nlist: config.nlist,
            metric: parse_metric(&config.metric)?,
            hnsw: hnsw_params(Some(&config.hnsw)),
        });
    }
    if let Ok(config) = config.extract::<PyRef<'_, IvfHnswSqConfig>>() {
        return Ok(VectorIndexConfig::IvfHnswSq {
            dimension: config.dimension,
            nlist: config.nlist,
            metric: parse_metric(&config.metric)?,
            hnsw: hnsw_params(Some(&config.hnsw)),
        });
    }
    Err(PyValueError::new_err(
        "config must be IvfFlatConfig, IvfPqConfig, IvfHnswFlatConfig, or IvfHnswSqConfig",
    ))
}

#[pyclass]
struct VectorIndexWriter {
    index: Option<CoreVectorIndexWriter>,
    dimension: usize,
}

#[pymethods]
impl VectorIndexWriter {
    #[new]
    fn new(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let config = config_from_py(config)?;
        let dimension = config.dimension();
        let index = CoreVectorIndexWriter::new(config)
            .map_err(|e| PyValueError::new_err(format!("failed to create writer: {}", e)))?;
        Ok(Self {
            index: Some(index),
            dimension,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (dimension, nlist, metric="l2"))]
    fn ivf_flat(dimension: usize, nlist: usize, metric: &str) -> PyResult<Self> {
        let config = VectorIndexConfig::IvfFlat {
            dimension,
            nlist,
            metric: parse_metric(metric)?,
        };
        let index = CoreVectorIndexWriter::new(config)
            .map_err(|e| PyValueError::new_err(format!("failed to create writer: {}", e)))?;
        Ok(Self {
            index: Some(index),
            dimension,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (dimension, nlist, m, metric="l2", use_opq=false))]
    fn ivf_pq(
        dimension: usize,
        nlist: usize,
        m: usize,
        metric: &str,
        use_opq: bool,
    ) -> PyResult<Self> {
        let config = IvfPqConfig::new(dimension, nlist, m, metric, use_opq)?;
        let core = VectorIndexConfig::IvfPq {
            dimension: config.dimension,
            nlist: config.nlist,
            m: config.m,
            metric: parse_metric(&config.metric)?,
            use_opq: config.use_opq,
        };
        let index = CoreVectorIndexWriter::new(core)
            .map_err(|e| PyValueError::new_err(format!("failed to create writer: {}", e)))?;
        Ok(Self {
            index: Some(index),
            dimension,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (dimension, nlist, metric="l2", hnsw=None))]
    fn ivf_hnsw_flat(
        dimension: usize,
        nlist: usize,
        metric: &str,
        hnsw: Option<&HnswConfig>,
    ) -> PyResult<Self> {
        let config = VectorIndexConfig::IvfHnswFlat {
            dimension,
            nlist,
            metric: parse_metric(metric)?,
            hnsw: hnsw_params(hnsw),
        };
        let index = CoreVectorIndexWriter::new(config)
            .map_err(|e| PyValueError::new_err(format!("failed to create writer: {}", e)))?;
        Ok(Self {
            index: Some(index),
            dimension,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (dimension, nlist, metric="l2", hnsw=None))]
    fn ivf_hnsw_sq(
        dimension: usize,
        nlist: usize,
        metric: &str,
        hnsw: Option<&HnswConfig>,
    ) -> PyResult<Self> {
        let config = VectorIndexConfig::IvfHnswSq {
            dimension,
            nlist,
            metric: parse_metric(metric)?,
            hnsw: hnsw_params(hnsw),
        };
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
    inner: CoreVectorIndexReader<PyFileStream>,
}

#[pymethods]
impl VectorIndexReader {
    #[new]
    fn new(file: PyObject) -> PyResult<Self> {
        let stream = PyFileStream { file };
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
            hnsw: metadata.hnsw.map(|h| HnswConfig {
                m: h.m,
                ef_construction: h.ef_construction,
                max_level: h.max_level,
            }),
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
    m.add_class::<HnswConfig>()?;
    m.add_class::<IvfFlatConfig>()?;
    m.add_class::<IvfPqConfig>()?;
    m.add_class::<IvfHnswFlatConfig>()?;
    m.add_class::<IvfHnswSqConfig>()?;
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
        output.call_method1("seek", (0,)).unwrap();
        output
    }

    #[test]
    fn python_unified_writer_reader_roundtrips_supported_indexes() {
        Python::with_gil(|py| {
            let configs: Vec<(Bound<'_, PyAny>, usize, &str)> = vec![
                (
                    Py::new(py, IvfFlatConfig::new(16, 4, "l2").unwrap())
                        .unwrap()
                        .into_bound(py)
                        .into_any(),
                    16,
                    "ivf_flat",
                ),
                (
                    Py::new(py, IvfPqConfig::new(16, 4, 4, "l2", false).unwrap())
                        .unwrap()
                        .into_bound(py)
                        .into_any(),
                    16,
                    "ivf_pq",
                ),
                (
                    Py::new(py, IvfHnswFlatConfig::new(16, 4, "l2", None).unwrap())
                        .unwrap()
                        .into_bound(py)
                        .into_any(),
                    16,
                    "ivf_hnsw_flat",
                ),
                (
                    Py::new(py, IvfHnswSqConfig::new(16, 4, "l2", None).unwrap())
                        .unwrap()
                        .into_bound(py)
                        .into_any(),
                    16,
                    "ivf_hnsw_sq",
                ),
            ];

            for (config, d, expected_type) in configs {
                let output = write_index_bytes(py, &config, d);
                let mut reader = VectorIndexReader::new(output.unbind()).unwrap();
                assert_eq!(reader.index_type(), expected_type);
                assert_eq!(reader.dimension(), d);
                assert_eq!(reader.metadata().index_type, expected_type);

                let data = generate_clustered_data(1, d, 1);
                let query = PyArray1::from_vec_bound(py, data[0..d].to_vec());
                let (result_ids, _) = reader
                    .search(py, query.readonly(), 5, 4, 32, None)
                    .unwrap();
                assert_eq!(result_ids.len(), 5);
                assert_eq!(result_ids.readonly().as_slice().unwrap()[0], 0);
            }
        });
    }

    #[test]
    fn python_batch_search_accepts_roaring_filter_bytes() {
        Python::with_gil(|py| {
            let config = Py::new(py, IvfFlatConfig::new(2, 1, "l2").unwrap())
                .unwrap()
                .into_bound(py)
                .into_any();
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

            output.call_method1("seek", (0,)).unwrap();
            let mut reader = VectorIndexReader::new(output.unbind()).unwrap();
            let queries =
                PyArray::from_vec2_bound(py, &[vec![0.0f32, 0.0], vec![10.0, 10.0]]).unwrap();
            let (result_ids, result_dists) = reader
                .search_batch(py, queries.readonly(), 2, 1, 0, Some(filter.as_any()))
                .unwrap();

            assert_eq!(result_ids.shape(), &[2, 2]);
            assert_eq!(
                result_ids.readonly().as_slice().unwrap(),
                &[12, -1, 12, -1]
            );
            assert_eq!(result_dists.readonly().as_slice().unwrap()[1], f32::MAX);
        });
    }

    #[test]
    fn python_batch_search_validates_query_shape() {
        Python::with_gil(|py| {
            let config = Py::new(py, IvfPqConfig::new(16, 4, 4, "l2", false).unwrap())
                .unwrap()
                .into_bound(py)
                .into_any();
            let output = write_index_bytes(py, &config, 16);
            let mut reader = VectorIndexReader::new(output.unbind()).unwrap();
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
            let config = Py::new(py, IvfPqConfig::new(16, 4, 4, "l2", false).unwrap())
                .unwrap()
                .into_bound(py)
                .into_any();
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
