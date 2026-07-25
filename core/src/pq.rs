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

use crate::blas::sgemm_a_bt;
use crate::distance::{
    fvec_ip_batch, fvec_l2sqr_batch, fvec_norm_l2sqr, pq_distance_from_table, MetricType,
};
use crate::kmeans::{self, KMeansConfig};
use rayon::prelude::*;

/// Product Quantizer aligned with Faiss's ProductQuantizer.
///
/// Splits D-dimensional vectors into M contiguous chunks and independently
/// quantizes each chunk with `ksub` centroids. Uniform chunks remain the
/// default for IVF-PQ compatibility; DiskANN may use near-equal chunks when
/// `D` is not divisible by `M`.
///
/// Centroids are chunk-major. Chunk `m` starts at
/// `chunk_offsets[m] * ksub`, and each of its `ksub` centroids contains
/// `chunk_offsets[m + 1] - chunk_offsets[m]` contiguous components.
pub struct ProductQuantizer {
    pub d: usize,
    pub m: usize,
    pub nbits: usize,
    /// Uniform chunk width for legacy formats, or the largest chunk width for
    /// a balanced non-uniform layout.
    pub dsub: usize,
    pub ksub: usize,
    pub chunk_offsets: Vec<usize>,
    pub centroids: Vec<f32>,
    /// Pre-computed squared norms of each centroid: [M * ksub].
    /// Avoids recomputing per query for L2 distance table.
    pub centroid_norms_cache: Vec<f32>,
}

impl ProductQuantizer {
    pub fn new(d: usize, m: usize) -> Self {
        Self::with_nbits(d, m, 8)
    }

    pub fn with_nbits(d: usize, m: usize, nbits: usize) -> Self {
        assert!(
            d.is_multiple_of(m),
            "dimension {} must be divisible by m={}",
            d,
            m
        );
        let dsub = d / m;
        let chunk_offsets = (0..=m).map(|chunk| chunk * dsub).collect();
        Self::with_validated_chunk_offsets(d, nbits, chunk_offsets)
    }

    /// Create a quantizer whose chunks differ in width by at most one.
    ///
    /// The first `d % m` chunks contain one additional component. This is the
    /// DiskANN layout and supports every `1 <= m <= d`.
    pub fn with_nbits_balanced(d: usize, m: usize, nbits: usize) -> Self {
        assert!(d > 0, "dimension must be greater than zero");
        assert!(m > 0 && m <= d, "m must be in 1..=dimension");
        let base = d / m;
        let remainder = d % m;
        let mut chunk_offsets = Vec::with_capacity(m + 1);
        chunk_offsets.push(0);
        let mut offset = 0usize;
        for chunk in 0..m {
            offset += base + usize::from(chunk < remainder);
            chunk_offsets.push(offset);
        }
        Self::with_validated_chunk_offsets(d, nbits, chunk_offsets)
    }

    /// Restore a persisted chunk plan after the caller has decoded it.
    pub fn try_with_chunk_offsets(
        d: usize,
        nbits: usize,
        chunk_offsets: Vec<usize>,
    ) -> Result<Self, &'static str> {
        if d == 0 || !matches!(nbits, 4 | 8) || chunk_offsets.len() < 2 {
            return Err("invalid PQ shape");
        }
        if chunk_offsets[0] != 0 || chunk_offsets.last().copied() != Some(d) {
            return Err("invalid PQ chunk bounds");
        }
        if chunk_offsets
            .windows(2)
            .any(|bounds| bounds[0] >= bounds[1])
        {
            return Err("PQ chunk offsets must be strictly increasing");
        }
        Ok(Self::with_validated_chunk_offsets(d, nbits, chunk_offsets))
    }

    fn with_validated_chunk_offsets(d: usize, nbits: usize, chunk_offsets: Vec<usize>) -> Self {
        assert!(
            nbits == 4 || nbits == 8,
            "nbits must be 4 or 8, got {}",
            nbits
        );
        let m = chunk_offsets.len() - 1;
        let dsub = chunk_offsets
            .windows(2)
            .map(|bounds| bounds[1] - bounds[0])
            .max()
            .expect("validated non-empty PQ chunk plan");
        let ksub = 1 << nbits;
        ProductQuantizer {
            d,
            m,
            nbits,
            dsub,
            ksub,
            chunk_offsets,
            centroids: Vec::new(),
            centroid_norms_cache: Vec::new(),
        }
    }

    #[inline]
    pub fn chunk_range(&self, sub: usize) -> std::ops::Range<usize> {
        self.chunk_offsets[sub]..self.chunk_offsets[sub + 1]
    }

    #[inline]
    pub fn chunk_dim(&self, sub: usize) -> usize {
        self.chunk_offsets[sub + 1] - self.chunk_offsets[sub]
    }

    #[inline]
    pub fn centroid_chunk_base(&self, sub: usize) -> usize {
        self.chunk_offsets[sub] * self.ksub
    }

    pub fn has_valid_layout(&self) -> bool {
        Self::try_with_chunk_offsets(self.d, self.nbits, self.chunk_offsets.clone()).is_ok_and(
            |layout| {
                layout.m == self.m
                    && layout.dsub == self.dsub
                    && layout.ksub == self.ksub
                    && self.centroids.len() == self.d * self.ksub
            },
        )
    }

    /// Train the codebooks from training data.
    /// data: flat [n * d], n training vectors.
    pub fn train(&mut self, data: &[f32], n: usize) {
        self.train_with_config(data, n, &KMeansConfig::default());
    }

    pub fn train_with_config(&mut self, data: &[f32], n: usize, km_config: &KMeansConfig) {
        self.train_hot_start(data, n, km_config, false);
    }

    /// Train with optional hot-start: reuse existing centroids as k-means initial values.
    /// Parallelizes across M sub-quantizers with rayon.
    pub fn train_hot_start(
        &mut self,
        data: &[f32],
        n: usize,
        km_config: &KMeansConfig,
        hot_start: bool,
    ) {
        let prev_centroids = if hot_start && !self.centroids.is_empty() {
            Some(self.centroids.clone())
        } else {
            None
        };

        let m = self.m;
        let d = self.d;
        let ksub = self.ksub;
        let chunk_offsets = &self.chunk_offsets;

        // Train all M sub-quantizers in parallel
        let sub_results: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|sub| {
                let start = chunk_offsets[sub];
                let stop = chunk_offsets[sub + 1];
                let chunk_dim = stop - start;

                let mut sub_data = vec![0.0f32; n * chunk_dim];
                for i in 0..n {
                    sub_data[i * chunk_dim..(i + 1) * chunk_dim]
                        .copy_from_slice(&data[i * d + start..i * d + stop]);
                }

                let init: Option<Vec<f32>> = prev_centroids.as_ref().map(|pc| {
                    let src = start * ksub;
                    pc[src..src + ksub * chunk_dim].to_vec()
                });

                kmeans::kmeans_train_with_init(
                    km_config,
                    &sub_data,
                    n,
                    chunk_dim,
                    ksub,
                    init.as_deref(),
                )
            })
            .collect();

        self.centroids = vec![0.0f32; d * ksub];
        for (sub, sub_centroids) in sub_results.into_iter().enumerate() {
            let chunk_dim = self.chunk_dim(sub);
            let dst_offset = self.centroid_chunk_base(sub);
            self.centroids[dst_offset..dst_offset + ksub * chunk_dim]
                .copy_from_slice(&sub_centroids);
        }
        self.rebuild_norms_cache();
    }

    /// Rebuild the centroid norms cache. Called after training or loading centroids.
    pub fn rebuild_norms_cache(&mut self) {
        self.try_rebuild_norms_cache()
            .expect("PQ centroid norms allocation failed");
    }

    pub fn try_rebuild_norms_cache(&mut self) -> Result<(), std::collections::TryReserveError> {
        let mut norms = Vec::new();
        norms.try_reserve_exact(self.m * self.ksub)?;
        norms.resize(self.m * self.ksub, 0.0f32);
        for sub in 0..self.m {
            let chunk_dim = self.chunk_dim(sub);
            let c_base = self.centroid_chunk_base(sub);
            for j in 0..self.ksub {
                let c_off = c_base + j * chunk_dim;
                norms[sub * self.ksub + j] =
                    fvec_norm_l2sqr(&self.centroids[c_off..c_off + chunk_dim]);
            }
        }
        self.centroid_norms_cache = norms;
        Ok(())
    }

    /// Bytes per encoded vector.
    pub fn code_size(&self) -> usize {
        if self.nbits == 4 {
            self.m.div_ceil(2)
        } else {
            self.m
        }
    }

    /// Encode a single vector into PQ codes.
    /// For nbits=8: codes has length M (one byte per sub-quantizer).
    /// For nbits=4: codes has length ceil(M/2). If M is odd, the final high
    /// nibble is canonical zero padding.
    pub fn encode(&self, x: &[f32], codes: &mut [u8]) {
        let mut distances = vec![0.0f32; self.ksub];
        self.encode_with_distances(x, codes, &mut distances);
    }

    fn encode_with_distances(&self, x: &[f32], codes: &mut [u8], distances: &mut [f32]) {
        debug_assert!(distances.len() >= self.ksub);
        if self.nbits == 4 {
            self.encode_4bit(x, codes, distances);
        } else {
            self.encode_8bit(x, codes, distances);
        }
    }

    fn encode_8bit(&self, x: &[f32], codes: &mut [u8], distances: &mut [f32]) {
        for sub in 0..self.m {
            self.compute_sub_l2_distances(x, sub, distances);
            codes[sub] = argmin_code(&distances[..self.ksub]);
        }
    }

    fn encode_4bit(&self, x: &[f32], codes: &mut [u8], distances: &mut [f32]) {
        for pair in 0..self.m.div_ceil(2) {
            let sub_lo = pair * 2;
            let sub_hi = pair * 2 + 1;

            self.compute_sub_l2_distances(x, sub_lo, distances);
            let best_lo = argmin_code(&distances[..self.ksub]);

            let best_hi = if sub_hi < self.m {
                self.compute_sub_l2_distances(x, sub_hi, distances);
                argmin_code(&distances[..self.ksub])
            } else {
                0
            };

            // Pack: low nibble + high nibble
            codes[pair] = best_lo | (best_hi << 4);
        }
    }

    fn compute_sub_l2_distances(&self, x: &[f32], sub: usize, distances: &mut [f32]) {
        let range = self.chunk_range(sub);
        let chunk_dim = range.len();
        let c_base = self.centroid_chunk_base(sub);
        let query_sub = &x[range];
        let centroids = &self.centroids[c_base..c_base + self.ksub * chunk_dim];

        if chunk_dim >= 4 && self.ksub >= 8 {
            fvec_ip_batch(query_sub, centroids, chunk_dim, self.ksub, distances);
            let q_norm = fvec_norm_l2sqr(query_sub);
            let norms_base = sub * self.ksub;
            for j in 0..self.ksub {
                let c_norm = if !self.centroid_norms_cache.is_empty() {
                    self.centroid_norms_cache[norms_base + j]
                } else {
                    let c_off = j * chunk_dim;
                    fvec_norm_l2sqr(&centroids[c_off..c_off + chunk_dim])
                };
                distances[j] = (q_norm + c_norm - 2.0 * distances[j]).max(0.0);
            }
        } else {
            fvec_l2sqr_batch(query_sub, centroids, chunk_dim, self.ksub, distances);
        }
    }

    /// Encode multiple vectors in parallel.
    pub fn encode_batch(&self, data: &[f32], n: usize, codes: &mut [u8]) {
        let d = self.d;
        let cs = self.code_size();

        codes.par_chunks_mut(cs).enumerate().for_each_init(
            || vec![0.0f32; self.ksub],
            |distances, (i, code_chunk)| {
                if i < n {
                    self.encode_with_distances(&data[i * d..(i + 1) * d], code_chunk, distances);
                }
            },
        );
    }

    /// Decode PQ codes back to an approximate vector.
    pub fn decode(&self, codes: &[u8], x: &mut [f32]) {
        for sub in 0..self.m {
            let code = if self.nbits == 4 {
                let byte = codes[sub / 2];
                if sub.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                }
            } else {
                codes[sub]
            } as usize;
            let range = self.chunk_range(sub);
            let chunk_dim = range.len();
            let c_off = self.centroid_chunk_base(sub) + code * chunk_dim;
            x[range].copy_from_slice(&self.centroids[c_off..c_off + chunk_dim]);
        }
    }

    /// Precompute the distance table from a query to all PQ centroids.
    /// Uses sgemm for chunks of at least four components
    /// (L2: ||q-c||²=||q||²+||c||²-2q·cᵀ).
    pub fn compute_distance_table(&self, query: &[f32], metric: MetricType, table: &mut [f32]) {
        for sub in 0..self.m {
            let range = self.chunk_range(sub);
            let chunk_dim = range.len();
            let c_base = self.centroid_chunk_base(sub);
            let t_base = sub * self.ksub;
            let query_chunk = &query[range];
            let centroids = &self.centroids[c_base..c_base + self.ksub * chunk_dim];

            if chunk_dim >= 4 {
                sgemm_a_bt(
                    1,
                    self.ksub,
                    chunk_dim,
                    1.0,
                    query_chunk,
                    centroids,
                    0.0,
                    &mut table[t_base..t_base + self.ksub],
                );
            } else {
                fvec_ip_batch(
                    query_chunk,
                    centroids,
                    chunk_dim,
                    self.ksub,
                    &mut table[t_base..t_base + self.ksub],
                );
            }

            match metric {
                MetricType::L2 | MetricType::Cosine => {
                    // ||q-c||² = ||q||² + ||c||² - 2·q·c
                    // Use pre-cached centroid norms (avoids recomputing per query)
                    let q_norm = fvec_norm_l2sqr(query_chunk);
                    let norms_base = sub * self.ksub;
                    for j in 0..self.ksub {
                        let c_norm = if !self.centroid_norms_cache.is_empty() {
                            self.centroid_norms_cache[norms_base + j]
                        } else {
                            let c_off = c_base + j * chunk_dim;
                            fvec_norm_l2sqr(&self.centroids[c_off..c_off + chunk_dim])
                        };
                        table[t_base + j] = (q_norm + c_norm - 2.0 * table[t_base + j]).max(0.0);
                    }
                }
                MetricType::InnerProduct => {
                    for j in 0..self.ksub {
                        table[t_base + j] = -table[t_base + j];
                    }
                }
            }
        }
    }

    /// Compute inner product table: ip_table[m * ksub + j] = <query_m, centroid_m_j>.
    pub fn compute_inner_product_table(&self, query: &[f32], table: &mut [f32]) {
        for sub in 0..self.m {
            let range = self.chunk_range(sub);
            let chunk_dim = range.len();
            let c_base = self.centroid_chunk_base(sub);
            let t_base = sub * self.ksub;

            fvec_ip_batch(
                &query[range],
                &self.centroids[c_base..c_base + self.ksub * chunk_dim],
                chunk_dim,
                self.ksub,
                &mut table[t_base..t_base + self.ksub],
            );
        }
    }

    /// Compute the approximate distance from a distance table.
    #[inline]
    pub fn distance_from_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        if self.nbits == 4 {
            self.distance_from_table_4bit(table, codes)
        } else {
            pq_distance_from_table(table, codes, self.m, self.ksub)
        }
    }

    /// 4-bit PQ distance: unpack nibbles and accumulate from 16-entry tables.
    #[inline]
    fn distance_from_table_4bit(&self, table: &[f32], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for sub in 0..self.m {
            let byte = codes[sub / 2];
            let code = if sub.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            dist += table[sub * self.ksub + code as usize];
        }
        dist
    }

    /// Compute squared norms of all PQ centroids.
    /// Uses cache if available, otherwise computes from scratch.
    pub fn compute_centroid_norms(&self) -> Vec<f32> {
        if !self.centroid_norms_cache.is_empty() {
            return self.centroid_norms_cache.clone();
        }
        let mut norms = vec![0.0f32; self.m * self.ksub];
        for sub in 0..self.m {
            let chunk_dim = self.chunk_dim(sub);
            let c_base = self.centroid_chunk_base(sub);
            for j in 0..self.ksub {
                let c_off = c_base + j * chunk_dim;
                norms[sub * self.ksub + j] =
                    fvec_norm_l2sqr(&self.centroids[c_off..c_off + chunk_dim]);
            }
        }
        norms
    }
}

#[inline]
fn argmin_code(distances: &[f32]) -> u8 {
    debug_assert!(distances.len() <= 256);

    let mut best = 0usize;
    let mut best_dist = f32::MAX;
    for (j, &dist) in distances.iter().enumerate() {
        if dist < best_dist {
            best_dist = dist;
            best = j;
        }
    }
    best as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::fvec_l2sqr_sub;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_encode_decode_roundtrip() {
        let d = 8;
        let m = 2;
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);

        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();

        let mut pq = ProductQuantizer::new(d, m);
        pq.train(&data, n);

        let original = &data[0..d];
        let mut codes = vec![0u8; m];
        pq.encode(original, &mut codes);

        let mut decoded = vec![0.0f32; d];
        pq.decode(&codes, &mut decoded);

        // Decoded should be a reasonable approximation
        let error = fvec_l2sqr_sub(original, 0, &decoded, 0, d);
        assert!(error < 10.0); // PQ introduces quantization error
    }

    #[test]
    fn test_distance_table() {
        let d = 8;
        let m = 2;
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);

        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();

        let mut pq = ProductQuantizer::new(d, m);
        pq.train(&data, n);

        let query = &data[0..d];
        let mut table = vec![0.0f32; m * pq.ksub];
        pq.compute_distance_table(query, MetricType::L2, &mut table);

        let mut codes = vec![0u8; m];
        pq.encode(query, &mut codes);

        let dist = pq.distance_from_table(&table, &codes);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_4bit_encode_decode() {
        let d = 8;
        let m = 4; // must be even for 4-bit
        let n = 200;
        let mut rng = StdRng::seed_from_u64(42);

        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();

        let mut pq = ProductQuantizer::with_nbits(d, m, 4);
        assert_eq!(pq.ksub, 16);
        assert_eq!(pq.code_size(), 2); // m/2 = 2 bytes per vector

        pq.train(&data, n);

        let original = &data[0..d];
        let mut codes = vec![0u8; pq.code_size()];
        pq.encode(original, &mut codes);

        // Verify codes are non-trivial (not all zeros)
        assert!(codes.iter().any(|&b| b != 0));

        let mut decoded = vec![0.0f32; d];
        pq.decode(&codes, &mut decoded);

        // Should be a reasonable approximation
        let error = fvec_l2sqr_sub(original, 0, &decoded, 0, d);
        assert!(error < 20.0); // 4-bit has higher error than 8-bit

        // Distance table
        let mut table = vec![0.0f32; m * pq.ksub];
        pq.compute_distance_table(original, MetricType::L2, &mut table);
        let dist = pq.distance_from_table(&table, &codes);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_4bit_batch_encode() {
        let d = 16;
        let m = 8;
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);

        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();

        let mut pq = ProductQuantizer::with_nbits(d, m, 4);
        pq.train(&data, n);

        let cs = pq.code_size(); // m/2 = 4
        let mut codes = vec![0u8; n * cs];
        pq.encode_batch(&data, n, &mut codes);

        // Verify codes are non-trivial (not all zeros)
        assert!(codes.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_balanced_chunks_encode_decode_and_distance_table() {
        let d = 7;
        let m = 3;
        let n = 100;
        let mut rng = StdRng::seed_from_u64(73);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
        let mut pq = ProductQuantizer::with_nbits_balanced(d, m, 8);

        assert_eq!(pq.chunk_offsets, vec![0, 3, 5, 7]);
        assert_eq!(pq.dsub, 3);
        assert_eq!(pq.code_size(), 3);

        pq.train(&data, n);
        assert!(pq.has_valid_layout());
        let query = &data[d..2 * d];
        let mut codes = vec![0u8; pq.code_size()];
        pq.encode(query, &mut codes);
        let mut decoded = vec![0.0f32; d];
        pq.decode(&codes, &mut decoded);
        let mut table = vec![0.0f32; m * pq.ksub];
        pq.compute_distance_table(query, MetricType::L2, &mut table);

        let table_distance = pq.distance_from_table(&table, &codes);
        let decoded_distance = fvec_l2sqr_sub(query, 0, &decoded, 0, d);
        assert!((table_distance - decoded_distance).abs() < 1e-4);
    }

    #[test]
    fn test_odd_4bit_chunk_count_uses_canonical_padding_nibble() {
        let d = 7;
        let m = 3;
        let n = 100;
        let mut rng = StdRng::seed_from_u64(74);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
        let mut pq = ProductQuantizer::with_nbits_balanced(d, m, 4);
        pq.train(&data, n);

        assert_eq!(pq.chunk_offsets, vec![0, 3, 5, 7]);
        assert_eq!(pq.code_size(), 2);
        let mut codes = vec![0xff; pq.code_size()];
        pq.encode(&data[..d], &mut codes);
        assert_eq!(codes[1] & 0xf0, 0);

        let mut decoded = vec![0.0f32; d];
        pq.decode(&codes, &mut decoded);
        let mut table = vec![0.0f32; m * pq.ksub];
        pq.compute_distance_table(&data[..d], MetricType::L2, &mut table);
        let table_distance = pq.distance_from_table(&table, &codes);
        let decoded_distance = fvec_l2sqr_sub(&data[..d], 0, &decoded, 0, d);
        assert!((table_distance - decoded_distance).abs() < 1e-4);
    }

    #[test]
    fn test_sgemm_distance_table_clamps_negative_self_distance() {
        let d = 128;
        let m = 1;
        let ksub = 256;
        let mut pq = ProductQuantizer::new(d, m);
        pq.centroids = vec![0.0; ksub * d];

        let mut query = vec![0.0; d];
        for i in 0..d {
            let value = if i.is_multiple_of(2) {
                1.0e10_f32 + i as f32
            } else {
                -1.0e10_f32 + i as f32
            };
            query[i] = value;
            pq.centroids[i] = value;
        }
        pq.rebuild_norms_cache();

        let mut table = vec![0.0; m * ksub];
        pq.compute_distance_table(&query, MetricType::L2, &mut table);

        assert_eq!(table[0], 0.0);
    }
}
