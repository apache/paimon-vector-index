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

use crate::distance::MetricType;
use crate::hnsw::HnswBuildParams;
use crate::io::{write_index, IVFPQIndexReader, SeekRead, SeekWrite, MAGIC};
use crate::ivfflat::IVFFlatIndex;
use crate::ivfflat_io::{
    search_batch_ivfflat_reader, search_batch_ivfflat_reader_roaring_filter, write_ivfflat_index,
    IVFFlatIndexReader, IVFFLAT_MAGIC,
};
use crate::ivfhnswflat::IVFHNSWFlatIndex;
use crate::ivfhnswflat_io::{
    search_batch_ivfhnswflat_reader, search_batch_ivfhnswflat_reader_roaring_filter,
    write_ivfhnswflat_index, IVFHNSWFlatIndexReader, IVF_HNSW_FLAT_MAGIC,
};
use crate::ivfhnswsq::IVFHNSWSQIndex;
use crate::ivfhnswsq_io::{
    search_batch_ivfhnswsq_reader, search_batch_ivfhnswsq_reader_roaring_filter,
    write_ivfhnswsq_index, IVFHNSWSQIndexReader, IVF_HNSW_SQ_MAGIC,
};
use crate::ivfpq::{
    search_batch_reader, search_batch_reader_roaring_filter, search_with_reader,
    search_with_reader_roaring_filter, IVFPQIndex,
};
use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IndexType {
    IvfFlat = 0,
    IvfPq = 1,
    IvfHnswFlat = 2,
    IvfHnswSq = 3,
}

impl IndexType {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::IvfFlat),
            1 => Some(Self::IvfPq),
            2 => Some(Self::IvfHnswFlat),
            3 => Some(Self::IvfHnswSq),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::IvfFlat => "ivf_flat",
            Self::IvfPq => "ivf_pq",
            Self::IvfHnswFlat => "ivf_hnsw_flat",
            Self::IvfHnswSq => "ivf_hnsw_sq",
        }
    }
}

#[derive(Debug, Clone)]
pub enum VectorIndexConfig {
    IvfFlat {
        dimension: usize,
        nlist: usize,
        metric: MetricType,
    },
    IvfPq {
        dimension: usize,
        nlist: usize,
        m: usize,
        metric: MetricType,
        use_opq: bool,
    },
    IvfHnswFlat {
        dimension: usize,
        nlist: usize,
        metric: MetricType,
        hnsw: HnswBuildParams,
    },
    IvfHnswSq {
        dimension: usize,
        nlist: usize,
        metric: MetricType,
        hnsw: HnswBuildParams,
    },
}

impl VectorIndexConfig {
    pub fn index_type(&self) -> IndexType {
        match self {
            Self::IvfFlat { .. } => IndexType::IvfFlat,
            Self::IvfPq { .. } => IndexType::IvfPq,
            Self::IvfHnswFlat { .. } => IndexType::IvfHnswFlat,
            Self::IvfHnswSq { .. } => IndexType::IvfHnswSq,
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::IvfFlat { dimension, .. }
            | Self::IvfPq { dimension, .. }
            | Self::IvfHnswFlat { dimension, .. }
            | Self::IvfHnswSq { dimension, .. } => *dimension,
        }
    }

    pub fn nlist(&self) -> usize {
        match self {
            Self::IvfFlat { nlist, .. }
            | Self::IvfPq { nlist, .. }
            | Self::IvfHnswFlat { nlist, .. }
            | Self::IvfHnswSq { nlist, .. } => *nlist,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VectorSearchParams {
    pub top_k: usize,
    pub nprobe: usize,
    pub ef_search: usize,
}

impl VectorSearchParams {
    pub fn new(top_k: usize, nprobe: usize) -> Self {
        Self {
            top_k,
            nprobe,
            ef_search: 0,
        }
    }

    pub fn with_ef_search(top_k: usize, nprobe: usize, ef_search: usize) -> Self {
        Self {
            top_k,
            nprobe,
            ef_search,
        }
    }

    fn hnsw_ef_search(self) -> usize {
        if self.ef_search == 0 {
            self.top_k.max(32)
        } else {
            self.ef_search
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorIndexMetadata {
    pub index_type: IndexType,
    pub dimension: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub pq_m: Option<usize>,
    pub hnsw: Option<HnswBuildParams>,
}

pub enum VectorIndexWriter {
    IvfFlat(IVFFlatIndex),
    IvfPq(IVFPQIndex),
    IvfHnswFlat(IVFHNSWFlatIndex),
    IvfHnswSq(IVFHNSWSQIndex),
}

impl VectorIndexWriter {
    pub fn new(config: VectorIndexConfig) -> io::Result<Self> {
        validate_config(&config)?;
        Ok(match config {
            VectorIndexConfig::IvfFlat {
                dimension,
                nlist,
                metric,
            } => Self::IvfFlat(IVFFlatIndex::new(dimension, nlist, metric)),
            VectorIndexConfig::IvfPq {
                dimension,
                nlist,
                m,
                metric,
                use_opq,
            } => Self::IvfPq(IVFPQIndex::new(dimension, nlist, m, metric, use_opq)),
            VectorIndexConfig::IvfHnswFlat {
                dimension,
                nlist,
                metric,
                hnsw,
            } => Self::IvfHnswFlat(IVFHNSWFlatIndex::new(
                dimension,
                nlist,
                metric,
                hnsw.sanitized(),
            )),
            VectorIndexConfig::IvfHnswSq {
                dimension,
                nlist,
                metric,
                hnsw,
            } => Self::IvfHnswSq(IVFHNSWSQIndex::new(
                dimension,
                nlist,
                metric,
                hnsw.sanitized(),
            )),
        })
    }

    pub fn index_type(&self) -> IndexType {
        match self {
            Self::IvfFlat(_) => IndexType::IvfFlat,
            Self::IvfPq(_) => IndexType::IvfPq,
            Self::IvfHnswFlat(_) => IndexType::IvfHnswFlat,
            Self::IvfHnswSq(_) => IndexType::IvfHnswSq,
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::IvfFlat(index) => index.d,
            Self::IvfPq(index) => index.d,
            Self::IvfHnswFlat(index) => index.flat.d,
            Self::IvfHnswSq(index) => index.d,
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) -> io::Result<()> {
        validate_vectors(data, n, self.dimension(), "training data")?;
        match self {
            Self::IvfFlat(index) => index.train(data, n),
            Self::IvfPq(index) => index.train(data, n),
            Self::IvfHnswFlat(index) => index.train(data, n),
            Self::IvfHnswSq(index) => index.train(data, n),
        }
        Ok(())
    }

    pub fn add_vectors(&mut self, ids: &[i64], data: &[f32], n: usize) -> io::Result<()> {
        validate_vectors(data, n, self.dimension(), "vector data")?;
        if ids.len() < n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "ids length {} is shorter than vector count {}",
                    ids.len(),
                    n
                ),
            ));
        }
        match self {
            Self::IvfFlat(index) => index.add(data, ids, n),
            Self::IvfPq(index) => index.add(data, ids, n),
            Self::IvfHnswFlat(index) => index.add(data, ids, n),
            Self::IvfHnswSq(index) => index.add(data, ids, n),
        }
        Ok(())
    }

    pub fn write(&mut self, out: &mut dyn SeekWrite) -> io::Result<()> {
        match self {
            Self::IvfFlat(index) => write_ivfflat_index(index, out),
            Self::IvfPq(index) => write_index(index, out),
            Self::IvfHnswFlat(index) => {
                index.build_graphs()?;
                write_ivfhnswflat_index(index, out)
            }
            Self::IvfHnswSq(index) => {
                index.build_graphs()?;
                write_ivfhnswsq_index(index, out)
            }
        }
    }
}

pub enum VectorIndexReader<R: SeekRead> {
    IvfFlat(IVFFlatIndexReader<R>),
    IvfPq(IVFPQIndexReader<R>),
    IvfHnswFlat(IVFHNSWFlatIndexReader<R>),
    IvfHnswSq(IVFHNSWSQIndexReader<R>),
}

impl<R: SeekRead> VectorIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        reader.seek(0)?;
        let mut magic_buf = [0u8; 4];
        reader.read_exact(&mut magic_buf)?;
        let magic = u32::from_le_bytes(magic_buf);
        reader.seek(0)?;

        match magic {
            IVFFLAT_MAGIC => Ok(Self::IvfFlat(IVFFlatIndexReader::open(reader)?)),
            MAGIC => Ok(Self::IvfPq(IVFPQIndexReader::open(reader)?)),
            IVF_HNSW_FLAT_MAGIC => Ok(Self::IvfHnswFlat(IVFHNSWFlatIndexReader::open(reader)?)),
            IVF_HNSW_SQ_MAGIC => Ok(Self::IvfHnswSq(IVFHNSWSQIndexReader::open(reader)?)),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown vector index magic: 0x{:08X}", magic),
            )),
        }
    }

    pub fn index_type(&self) -> IndexType {
        match self {
            Self::IvfFlat(_) => IndexType::IvfFlat,
            Self::IvfPq(_) => IndexType::IvfPq,
            Self::IvfHnswFlat(_) => IndexType::IvfHnswFlat,
            Self::IvfHnswSq(_) => IndexType::IvfHnswSq,
        }
    }

    pub fn metadata(&self) -> VectorIndexMetadata {
        match self {
            Self::IvfFlat(reader) => VectorIndexMetadata {
                index_type: IndexType::IvfFlat,
                dimension: reader.d,
                nlist: reader.nlist,
                metric: reader.metric,
                total_vectors: reader.total_vectors,
                pq_m: None,
                hnsw: None,
            },
            Self::IvfPq(reader) => VectorIndexMetadata {
                index_type: IndexType::IvfPq,
                dimension: reader.d,
                nlist: reader.nlist,
                metric: reader.metric,
                total_vectors: reader.total_vectors,
                pq_m: Some(reader.m),
                hnsw: None,
            },
            Self::IvfHnswFlat(reader) => VectorIndexMetadata {
                index_type: IndexType::IvfHnswFlat,
                dimension: reader.d,
                nlist: reader.nlist,
                metric: reader.metric,
                total_vectors: reader.total_vectors,
                pq_m: None,
                hnsw: Some(reader.hnsw_params),
            },
            Self::IvfHnswSq(reader) => VectorIndexMetadata {
                index_type: IndexType::IvfHnswSq,
                dimension: reader.d,
                nlist: reader.nlist,
                metric: reader.metric,
                total_vectors: reader.total_vectors,
                pq_m: None,
                hnsw: Some(reader.hnsw_params),
            },
        }
    }

    pub fn dimension(&self) -> usize {
        self.metadata().dimension
    }

    pub fn total_vectors(&self) -> i64 {
        self.metadata().total_vectors
    }

    pub fn search(
        &mut self,
        query: &[f32],
        params: VectorSearchParams,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        match self {
            Self::IvfFlat(reader) => reader.search(query, params.top_k, params.nprobe),
            Self::IvfPq(reader) => search_with_reader(reader, query, params.top_k, params.nprobe),
            Self::IvfHnswFlat(reader) => {
                reader.search(query, params.top_k, params.nprobe, params.hnsw_ef_search())
            }
            Self::IvfHnswSq(reader) => {
                reader.search(query, params.top_k, params.nprobe, params.hnsw_ef_search())
            }
        }
    }

    pub fn search_with_roaring_filter(
        &mut self,
        query: &[f32],
        params: VectorSearchParams,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        match self {
            Self::IvfFlat(reader) => reader.search_with_roaring_filter(
                query,
                params.top_k,
                params.nprobe,
                roaring_filter_bytes,
            ),
            Self::IvfPq(reader) => search_with_reader_roaring_filter(
                reader,
                query,
                params.top_k,
                params.nprobe,
                roaring_filter_bytes,
            ),
            Self::IvfHnswFlat(reader) => reader.search_with_roaring_filter(
                query,
                params.top_k,
                params.nprobe,
                params.hnsw_ef_search(),
                roaring_filter_bytes,
            ),
            Self::IvfHnswSq(reader) => reader.search_with_roaring_filter(
                query,
                params.top_k,
                params.nprobe,
                params.hnsw_ef_search(),
                roaring_filter_bytes,
            ),
        }
    }

    pub fn search_batch(
        &mut self,
        queries: &[f32],
        query_count: usize,
        params: VectorSearchParams,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        match self {
            Self::IvfFlat(reader) => search_batch_ivfflat_reader(
                reader,
                queries,
                query_count,
                params.top_k,
                params.nprobe,
            ),
            Self::IvfPq(reader) => {
                search_batch_reader(reader, queries, query_count, params.top_k, params.nprobe)
            }
            Self::IvfHnswFlat(reader) => search_batch_ivfhnswflat_reader(
                reader,
                queries,
                query_count,
                params.top_k,
                params.nprobe,
                params.hnsw_ef_search(),
            ),
            Self::IvfHnswSq(reader) => search_batch_ivfhnswsq_reader(
                reader,
                queries,
                query_count,
                params.top_k,
                params.nprobe,
                params.hnsw_ef_search(),
            ),
        }
    }

    pub fn search_batch_with_roaring_filter(
        &mut self,
        queries: &[f32],
        query_count: usize,
        params: VectorSearchParams,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        match self {
            Self::IvfFlat(reader) => search_batch_ivfflat_reader_roaring_filter(
                reader,
                queries,
                query_count,
                params.top_k,
                params.nprobe,
                roaring_filter_bytes,
            ),
            Self::IvfPq(reader) => search_batch_reader_roaring_filter(
                reader,
                queries,
                query_count,
                params.top_k,
                params.nprobe,
                roaring_filter_bytes,
            ),
            Self::IvfHnswFlat(reader) => search_batch_ivfhnswflat_reader_roaring_filter(
                reader,
                queries,
                query_count,
                params.top_k,
                params.nprobe,
                params.hnsw_ef_search(),
                roaring_filter_bytes,
            ),
            Self::IvfHnswSq(reader) => search_batch_ivfhnswsq_reader_roaring_filter(
                reader,
                queries,
                query_count,
                params.top_k,
                params.nprobe,
                params.hnsw_ef_search(),
                roaring_filter_bytes,
            ),
        }
    }
}

fn validate_config(config: &VectorIndexConfig) -> io::Result<()> {
    validate_positive(config.dimension(), "dimension")?;
    validate_positive(config.nlist(), "nlist")?;
    match config {
        VectorIndexConfig::IvfPq { dimension, m, .. } => {
            validate_positive(*m, "m")?;
            if dimension % m != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("dimension {} must be divisible by m {}", dimension, m),
                ));
            }
        }
        VectorIndexConfig::IvfHnswFlat { hnsw, .. } | VectorIndexConfig::IvfHnswSq { hnsw, .. } => {
            validate_hnsw_params(*hnsw)?
        }
        _ => {}
    }
    Ok(())
}

fn validate_hnsw_params(params: HnswBuildParams) -> io::Result<()> {
    validate_positive(params.m, "hnsw m")?;
    validate_positive(params.ef_construction, "hnsw ef_construction")?;
    validate_positive(params.max_level, "hnsw max_level")
}

fn validate_positive(value: usize, name: &str) -> io::Result<()> {
    if value == 0 {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} must be greater than 0", name),
        ))
    } else {
        Ok(())
    }
}

fn validate_vectors(data: &[f32], n: usize, dimension: usize, value_name: &str) -> io::Result<()> {
    validate_positive(n, "vector count")?;
    let expected_len = n.checked_mul(dimension).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "vector count * dimension overflows usize",
        )
    })?;
    if data.len() < expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} length {} is shorter than vector count * dimension {}",
                value_name,
                data.len(),
                expected_len
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::PosWriter;
    use std::io::Cursor;

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

    fn roundtrip(config: VectorIndexConfig) {
        let d = config.dimension();
        let nlist = config.nlist();
        let n = 512;
        let data = generate_clustered_data(n, d, nlist);
        let ids = (0..n as i64).collect::<Vec<_>>();

        let mut writer = VectorIndexWriter::new(config.clone()).unwrap();
        assert_eq!(writer.index_type(), config.index_type());
        writer.train(&data, n).unwrap();
        writer.add_vectors(&ids, &data, n).unwrap();

        let mut buf = Vec::new();
        writer.write(&mut PosWriter::new(&mut buf)).unwrap();

        let mut reader = VectorIndexReader::open(Cursor::new(buf)).unwrap();
        let metadata = reader.metadata();
        assert_eq!(metadata.index_type, config.index_type());
        assert_eq!(metadata.dimension, d);
        assert_eq!(metadata.nlist, nlist);
        assert_eq!(metadata.total_vectors, n as i64);

        let params = VectorSearchParams::with_ef_search(5, nlist, 32);
        let (result_ids, result_dists) = reader.search(&data[0..d], params).unwrap();
        assert_eq!(result_ids.len(), 5);
        assert_eq!(result_dists.len(), 5);
        assert_eq!(result_ids[0], 0);
    }

    #[test]
    fn unified_reader_writer_roundtrips_all_index_types() {
        roundtrip(VectorIndexConfig::IvfFlat {
            dimension: 8,
            nlist: 4,
            metric: MetricType::L2,
        });
        roundtrip(VectorIndexConfig::IvfPq {
            dimension: 16,
            nlist: 4,
            m: 4,
            metric: MetricType::L2,
            use_opq: false,
        });
        roundtrip(VectorIndexConfig::IvfHnswFlat {
            dimension: 8,
            nlist: 4,
            metric: MetricType::L2,
            hnsw: HnswBuildParams::default(),
        });
        roundtrip(VectorIndexConfig::IvfHnswSq {
            dimension: 8,
            nlist: 4,
            metric: MetricType::L2,
            hnsw: HnswBuildParams::default(),
        });
    }

    #[test]
    fn unified_reader_rejects_unknown_magic() {
        let err = match VectorIndexReader::open(Cursor::new(vec![0xFF; 8])) {
            Ok(_) => panic!("unknown magic should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("unknown vector index magic"));
    }

    #[test]
    fn unified_config_rejects_invalid_pq_m() {
        let err = match VectorIndexWriter::new(VectorIndexConfig::IvfPq {
            dimension: 10,
            nlist: 4,
            m: 3,
            metric: MetricType::L2,
            use_opq: false,
        }) {
            Ok(_) => panic!("invalid PQ config should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("must be divisible"));
    }
}
