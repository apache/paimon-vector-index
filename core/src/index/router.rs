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
use crate::kmeans::find_topk_checked;
use std::borrow::Cow;

const MAGIC: &[u8; 8] = b"PVROUT01";
const HEADER: usize = 20;

/// Exact coarse routing shared by all shard index types. Contains no shard training state.
pub struct CentroidRouter {
    dimension: usize,
    metric: MetricType,
    centers: Vec<f32>,
}

impl CentroidRouter {
    fn payload_len(d: usize, nlist: usize) -> io::Result<usize> {
        if d == 0 || nlist == 0 || d > i32::MAX as usize || nlist > i32::MAX as usize {
            return Err(invalid_input("invalid centroid router dimensions"));
        }
        // Bound both center storage and the per-list allocations of an IVF-FLAT trainer.
        crate::ivfflat_io::flat_metadata_bytes(d, nlist)?;
        Ok(d * nlist * 4)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn nlist(&self) -> usize {
        self.centers.len() / self.dimension
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER + self.centers.len() * 4);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(self.dimension as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.nlist() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.metric as u32).to_le_bytes());
        for v in &self.centers {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Largest supported serialized router, including its fixed header.
    pub const MAX_SERIALIZED_BYTES: usize =
        HEADER + crate::index_io_util::MAX_READER_METADATA_BYTES / 2;
    pub const HEADER_BYTES: usize = HEADER;

    /// Check the fixed header and return the exact serialized size before allocating its body.
    pub fn serialized_size(header: &[u8]) -> io::Result<usize> {
        let (dimension, nlist, _) = Self::header(header)?;
        Ok(HEADER + Self::payload_len(dimension, nlist)?)
    }

    fn header(bytes: &[u8]) -> io::Result<(usize, usize, MetricType)> {
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            return Err(invalid_input("invalid centroid router header"));
        }
        let number = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let dimension = number(8) as usize;
        let nlist = number(12) as usize;
        let metric = MetricType::from_code(number(16))
            .ok_or_else(|| invalid_input("invalid routing metric"))?;
        Ok((dimension, nlist, metric))
    }

    pub fn deserialize(bytes: &[u8]) -> io::Result<Self> {
        let (dimension, nlist, metric) = Self::header(bytes)?;
        if Self::payload_len(dimension, nlist)? != bytes.len() - HEADER {
            return Err(invalid_input("invalid centroid router length"));
        }
        let centers = bytes[HEADER..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Self::from_centers(dimension, metric, centers)
    }

    fn from_centers(dimension: usize, metric: MetricType, centers: Vec<f32>) -> io::Result<Self> {
        if centers.iter().any(|v| !v.is_finite()) {
            return Err(invalid_input("non-finite routing centroid"));
        }
        Ok(Self {
            dimension,
            metric,
            centers,
        })
    }

    /// Resolve outer routing with the same automatic width policy as ordinary IVF search.
    pub fn resolve_nprobe(
        &self,
        params: VectorSearchParams,
        total_vectors: usize,
        matching_vectors: Option<usize>,
    ) -> io::Result<usize> {
        params.validate()?;
        params.resolve_ivf_nprobe(self.nlist(), total_vectors, matching_vectors)
    }

    fn routing_vector<'a>(&self, query: &'a [f32]) -> io::Result<Cow<'a, [f32]>> {
        let d = self.dimension();
        validate_query(query, d)?;
        let mut processed = Cow::Borrowed(query);
        if self.metric == MetricType::Cosine {
            fvec_normalize(processed.to_mut());
        }
        if let Cow::Owned(ref values) = processed {
            validate_query(values, d)?;
        }
        Ok(processed)
    }

    /// Exact coarse routing, for a query in the same metric space as training.
    pub fn find_nearest_centroids(&self, query: &[f32], nprobe: usize) -> io::Result<Vec<usize>> {
        validate_positive(nprobe, "nprobe")?;
        let processed = self.routing_vector(query)?;
        find_topk_checked(
            &processed,
            &self.centers,
            self.nlist(),
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
        for (id, center) in self.centers.chunks_exact(self.dimension()).enumerate() {
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

    /// Take the coarse centers from a trained IVF-FLAT model.
    pub fn from_training(training: VectorIndexTraining) -> io::Result<Self> {
        let VectorIndexWriter::IvfFlat(flat) = training.inner else {
            return Err(invalid_input(
                "a centroid router requires IVF-FLAT training",
            ));
        };
        Self::payload_len(flat.d, flat.nlist)?;
        Self::from_centers(flat.d, flat.metric, flat.into_quantizer_centroids())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_payload_is_independent_of_shard_type_and_rejects_corruption() {
        // Frozen routing format: two 2D cosine centers, including a tie on [1, 1].
        let mut bytes = b"PVROUT01".to_vec();
        for n in [2u32, 2, 2] {
            bytes.extend(n.to_le_bytes());
        }
        for f in [1.0f32, 0.0, 0.0, 1.0] {
            bytes.extend(f.to_le_bytes());
        }
        let router = CentroidRouter::deserialize(&bytes).unwrap();
        assert_eq!(router.serialize(), bytes);
        assert_eq!(router.assign_centroid(&[0., 30.]).unwrap(), 1);
        assert_eq!(router.assign_centroid(&[1., 1.]).unwrap(), 0);
        assert_eq!(router.find_nearest_centroids(&[1., 1.], 9).unwrap(), [0, 1]);
        assert!(router.assign_centroid(&[f32::NAN, 0.]).is_err());
        assert!(router.find_nearest_centroids(&[0.], 1).is_err());
        assert!(router.find_nearest_centroids(&[1., 0.], 0).is_err());
        for (offset, word) in [
            (0, 0u32),
            (8, 0),
            (12, u32::MAX),
            (16, 7),
            (20, f32::INFINITY.to_bits()),
        ] {
            let mut bad = bytes.clone();
            bad[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            assert!(CentroidRouter::deserialize(&bad).is_err());
        }
        assert!(CentroidRouter::deserialize(&bytes[..bytes.len() - 1]).is_err());
        bytes.push(0);
        assert!(CentroidRouter::deserialize(&bytes).is_err());
    }
}
