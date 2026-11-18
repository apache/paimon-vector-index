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
use crate::distance::{fvec_l2sqr, fvec_normalize};
use crate::io::{PosWriter, FLAG_HAS_OPQ, HEADER_SIZE};
use crate::ivfpq::{
    search_batch_reader_filter_with_centroid, search_with_reader_filter_and_centroid,
};
use crate::kmeans::find_topk_checked;
use std::borrow::Cow;

// Versioned independently from the regular IVF-PQ file nested in this envelope.
const MODEL_MAGIC: &[u8; 8] = b"PIVFTM01";
const MODEL_HEADER: usize = 20;
#[path = "centroid_typed.rs"]
mod typed;

fn invalid_model(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

// Call after checking that the model length includes the complete list table.
fn validate_empty_list_table(payload: &[u8], nlist: usize) -> io::Result<()> {
    let eof = (payload.len() as u64).to_le_bytes();
    for entry in payload[payload.len() - nlist * 16..].chunks_exact(16) {
        if entry[..8] != eof || entry[8..] != [0; 8] {
            return Err(invalid_model("invalid empty centroid list metadata"));
        }
    }
    Ok(())
}

impl VectorIndexTraining {
    fn centroid_model(&self) -> io::Result<&IVFPQIndex> {
        match &self.inner {
            VectorIndexWriter::IvfPq(index) => Ok(index),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "centroid sharding currently requires IVF-PQ",
            )),
        }
    }

    /// Serialize complete, data-free IVF state or a routing-only DiskANN model.
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        if !matches!(self.inner, VectorIndexWriter::IvfPq(_)) {
            return self.serialize_typed();
        }
        let model = self.centroid_model()?;
        let mut index_bytes = Vec::new();
        write_index(model, &mut PosWriter::new(&mut index_bytes))?;
        let mut bytes = Vec::with_capacity(MODEL_HEADER + index_bytes.len());
        bytes.extend_from_slice(MODEL_MAGIC);
        bytes.extend_from_slice(&model.training_flags().to_le_bytes());
        bytes.extend_from_slice(&(index_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&index_bytes);
        Ok(bytes)
    }

    /// Restore a model, validating the envelope and all lengths before allocating metadata.
    pub fn deserialize(bytes: &[u8]) -> io::Result<Self> {
        if bytes.get(..8) == Some(b"PIVFTM02") {
            return Self::deserialize_typed(bytes);
        }
        if bytes.len() < MODEL_HEADER + HEADER_SIZE || &bytes[..8] != MODEL_MAGIC {
            return Err(invalid_model(
                "invalid or unsupported training model envelope",
            ));
        }
        let flags = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let size = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        if flags & !3 != 0 || size != (bytes.len() - MODEL_HEADER) as u64 {
            return Err(invalid_model("invalid training model flags or length"));
        }
        let payload = &bytes[MODEL_HEADER..];
        let mut reader = IVFPQIndexReader::open(Cursor::new(payload))?;
        let native_flags = u32::from_le_bytes(payload[40..44].try_into().unwrap());
        let expected = (|| {
            let rotation = if native_flags & FLAG_HAS_OPQ != 0 {
                reader.d.checked_mul(reader.d)?
            } else {
                0
            };
            let floats = rotation
                .checked_add(reader.nlist.checked_mul(reader.d)?)?
                .checked_add(
                    reader
                        .m
                        .checked_mul(reader.ksub)?
                        .checked_mul(reader.dsub)?,
                )?;
            HEADER_SIZE
                .checked_add(floats.checked_mul(4)?)?
                .checked_add(reader.nlist.checked_mul(16)?)
        })();
        if reader.total_vectors != 0 || expected != Some(payload.len()) {
            return Err(invalid_model(
                "training model must contain complete metadata and no vectors",
            ));
        }
        validate_empty_list_table(payload, reader.nlist)?;
        reader.ensure_loaded()?;
        if reader
            .quantizer_centroids
            .iter()
            .chain(reader.pq.centroids())
            .any(|v| !v.is_finite())
            || reader
                .opq
                .as_ref()
                .is_some_and(|opq| opq.rotation.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid_model("non-finite training model value"));
        }
        let mut index = IVFPQIndex::with_nbits(
            reader.d,
            reader.nlist,
            reader.m,
            reader.pq.nbits(),
            reader.metric,
            false,
        );
        index.by_residual = reader.by_residual;
        index.set_quantizer_centroids(reader.quantizer_centroids);
        index.pq = reader.pq;
        index.opq = reader.opq;
        index.set_approximate_coarse_assignment(flags & 1 != 0);
        index.set_canonical_pq_encoding(flags & 2 != 0);
        Ok(Self {
            inner: VectorIndexWriter::IvfPq(index),
            centroid_router: None,
        })
    }

    pub fn nlist(&self) -> io::Result<usize> {
        if let VectorIndexWriter::DiskAnn(_) = &self.inner {
            return self
                .centroid_router
                .as_ref()
                .map(|r| r.nlist)
                .ok_or_else(|| invalid_input("DiskANN requires a centroid trainer"));
        }
        Ok(self.inner.nlist_for_centroid())
    }

    /// Resolve outer routing with the same automatic width policy as ordinary IVF search.
    pub fn resolve_nprobe(
        &self,
        params: VectorSearchParams,
        total_vectors: usize,
        matching_vectors: Option<usize>,
    ) -> io::Result<usize> {
        params.validate()?;
        params.resolve_ivf_nprobe(self.nlist()?, total_vectors, matching_vectors)
    }

    fn routing_vector<'a>(&self, query: &'a [f32]) -> io::Result<Cow<'a, [f32]>> {
        let d = self.dimension();
        validate_query(query, d)?;
        let mut processed = Cow::Borrowed(query);
        if self.inner.model_metric() == MetricType::Cosine {
            fvec_normalize(processed.to_mut());
        }
        if let VectorIndexWriter::IvfPq(index) = &self.inner {
            if let Some(opq) = &index.opq {
                let mut rotated = vec![0.0; d];
                opq.apply(&processed, &mut rotated);
                processed = Cow::Owned(rotated);
            }
        }
        if let Cow::Owned(ref values) = processed {
            validate_query(values, d)?;
        }
        Ok(processed)
    }

    /// Exact coarse routing, with the same cosine normalization and OPQ transform as search.
    pub fn find_nearest_centroids(&self, query: &[f32], nprobe: usize) -> io::Result<Vec<usize>> {
        validate_positive(nprobe, "nprobe")?;
        let processed = self.routing_vector(query)?;
        find_topk_checked(
            &processed,
            self.coarse_centroids()?,
            self.nlist()?,
            self.dimension(),
            nprobe,
        )
        .map(|centroids| {
            centroids
                .into_iter()
                .map(|(_, centroid)| centroid)
                .collect()
        })
        .map_err(|_| invalid_input("non-finite coarse distance"))
    }

    pub fn assign_centroid(&self, vector: &[f32]) -> io::Result<usize> {
        let point = self.routing_vector(vector)?;
        let mut nearest = 0;
        let mut best = f32::INFINITY;
        for (id, center) in self
            .coarse_centroids()?
            .chunks_exact(self.dimension())
            .enumerate()
        {
            let distance = fvec_l2sqr(&point, center);
            if !distance.is_finite() {
                return Err(invalid_input("non-finite coarse distance"));
            }
            if distance < best {
                best = distance;
                nearest = id;
            }
        }
        Ok(nearest)
    }

    /// Derive an independent writer without consuming or mutating this model.
    pub fn create_centroid_writer(&self, centroid: usize) -> io::Result<VectorIndexWriter> {
        if centroid >= self.nlist()? {
            return Err(invalid_input("centroid is outside nlist"));
        }
        let mut writer = self.empty_writer()?;
        match &mut writer {
            VectorIndexWriter::IvfFlat(i) => i.routed_centroid = Some(centroid),
            VectorIndexWriter::IvfSq(i) => i.routed_centroid = Some(centroid),
            VectorIndexWriter::IvfRq(i) => i.routed_centroid = Some(centroid),
            VectorIndexWriter::IvfPq(i) => i.routed_centroid = Some(centroid),
            VectorIndexWriter::DiskAnn(i) => i.routed_centroid = Some(centroid),
        }
        Ok(writer)
    }
}

impl VectorIndexWriter {
    /// Trust the caller's routing decision and encode against this writer's fixed centroid.
    /// IDs are nonnegative absolute row IDs. No second assignment is performed.
    pub fn add_routed_ivf_centroid_vectors(
        &mut self,
        ids: &[i64],
        data: &[f32],
        n: usize,
    ) -> io::Result<()> {
        self.add_centroid_vectors(ids, data, n)
    }

    /// Append pre-routed absolute IDs for any supported vector index.
    pub fn add_centroid_vectors(&mut self, ids: &[i64], data: &[f32], n: usize) -> io::Result<()> {
        validate_vectors(data, n, self.dimension(), "vector data")?;
        if ids.len() != n || ids.iter().any(|id| *id < 0) {
            return Err(invalid_input(
                "routed vectors require one nonnegative absolute row ID per vector",
            ));
        }
        let c = self
            .routed_centroid()
            .ok_or_else(|| invalid_input("writer was not created for a centroid shard"))?;
        match self {
            Self::IvfFlat(i) => i.add_routed(data, ids, n, c),
            Self::IvfSq(i) => i.add_routed(data, ids, n, c),
            Self::IvfRq(i) => i.add_routed(data, ids, n, c),
            Self::IvfPq(i) => i.add_routed(data, ids, n, c),
            Self::DiskAnn(i) => i.add(data, ids),
        }
        Ok(())
    }
}

/// A single shard selected by an external centroid router.
pub struct CentroidShardReader<R: SeekRead> {
    reader: VectorIndexReader<R>,
    ivf_list: usize,
}

impl<R: SeekRead> CentroidShardReader<R> {
    pub fn open(input: R) -> io::Result<Self> {
        Self::open_with_options(input, VectorIndexReaderOptions::default())
    }

    /// Search the only populated IVF list, or the local DiskANN graph.
    /// Empty IVF files are valid shards.
    pub fn open_with_options(input: R, options: VectorIndexReaderOptions) -> io::Result<Self> {
        let mut reader = VectorIndexReader::open_with_options(input, options)?;
        let counts: &[i32] = match &mut reader {
            VectorIndexReader::IvfPq(r) => {
                r.ensure_loaded()?;
                &r.list_counts
            }
            VectorIndexReader::IvfFlat(r) => {
                r.ensure_loaded()?;
                &r.list_counts
            }
            VectorIndexReader::IvfSq(r) => {
                r.ensure_loaded()?;
                &r.list_counts
            }
            VectorIndexReader::IvfRq(r) => {
                r.ensure_loaded()?;
                &r.list_counts
            }
            VectorIndexReader::DiskAnn(_) => &[],
        };
        let mut populated = counts.iter().enumerate().filter(|(_, count)| **count > 0);
        // List zero is empty in an empty IVF shard. DiskANN ignores the list.
        let ivf_list = populated.next().map_or(0, |(list, _)| list);
        if populated.next().is_some() {
            return Err(invalid_input(
                "centroid shard contains multiple populated IVF lists",
            ));
        }
        Ok(Self { reader, ivf_list })
    }

    pub fn metadata(&self) -> VectorIndexMetadata {
        self.reader.metadata()
    }

    pub fn optimize_for_search(&mut self) -> io::Result<()> {
        self.reader.optimize_for_search()
    }

    pub fn search_stats(&self) -> SearchStats {
        self.reader.search_stats()
    }

    pub fn search(
        &mut self,
        query: &[f32],
        params: VectorSearchParams,
        filter: Option<&[u8]>,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let centroid = self.ivf_list;
        self.reader.observed_search(1, |reader| {
            let VectorIndexReader::IvfPq(pq) = reader else {
                return reader.search_typed_centroid_batch(query, 1, params, centroid, filter);
            };
            validate_pq_params(params)?;
            validate_query(query, pq.d)?;
            let filter = filter.map(decode_roaring_filter).transpose()?;
            let mut result = search_with_reader_filter_and_centroid(
                pq,
                query,
                params.top_k,
                1,
                filter.as_ref().map(|f| f as &dyn crate::ivfpq::RowIdFilter),
                Some(centroid),
            )?;
            result.0.resize(params.top_k, -1);
            result.1.resize(params.top_k, f32::MAX);
            Ok(result)
        })
    }

    pub fn search_batch(
        &mut self,
        queries: &[f32],
        query_count: usize,
        params: VectorSearchParams,
        filter: Option<&[u8]>,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let centroid = self.ivf_list;
        self.reader.observed_search(query_count, |reader| {
            let VectorIndexReader::IvfPq(pq) = reader else {
                return reader.search_typed_centroid_batch(
                    queries,
                    query_count,
                    params,
                    centroid,
                    filter,
                );
            };
            validate_pq_params(params)?;
            validate_vectors(queries, query_count, pq.d, "queries")?;
            let filter = filter.map(decode_roaring_filter).transpose()?;
            search_batch_reader_filter_with_centroid(
                pq,
                queries,
                query_count,
                params.top_k,
                0,
                1,
                &[],
                &[],
                filter.as_ref().map(|f| f as &dyn crate::ivfpq::RowIdFilter),
                params.ivfpq_batch_table_reuse,
                params.ivfpq_batch_table_reuse_max_bytes,
                |_| {},
                #[cfg(test)]
                None,
                Some(centroid),
            )
        })
    }
}

fn validate_pq_params(params: VectorSearchParams) -> io::Result<()> {
    params.validate()?;
    if matches!(params.search_width, SearchWidth::DiskAnnLSearch) {
        return Err(invalid_input(
            "centroid shard search does not accept DiskANN search width",
        ));
    }
    Ok(())
}
