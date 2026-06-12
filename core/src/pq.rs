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
/// Splits D-dimensional vectors into M sub-vectors of dimension dsub = D/M,
/// and independently quantizes each sub-vector with ksub centroids.
///
/// Centroid layout: flat [M * ksub * dsub], row-major.
/// centroids[m][j][d] is at index: m * ksub * dsub + j * dsub + d
pub struct ProductQuantizer {
    pub d: usize,
    pub m: usize,
    pub nbits: usize,
    pub dsub: usize,
    pub ksub: usize,
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
        assert!(
            nbits == 4 || nbits == 8,
            "nbits must be 4 or 8, got {}",
            nbits
        );
        if nbits == 4 {
            assert!(
                m.is_multiple_of(2),
                "m must be even for 4-bit PQ, got {}",
                m
            );
        }
        let dsub = d / m;
        let ksub = 1 << nbits;
        ProductQuantizer {
            d,
            m,
            nbits,
            dsub,
            ksub,
            centroids: Vec::new(),
            centroid_norms_cache: Vec::new(),
        }
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
        let dsub = self.dsub;
        let ksub = self.ksub;

        // Train all M sub-quantizers in parallel
        let sub_results: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|sub| {
                let offset = sub * dsub;

                let mut sub_data = vec![0.0f32; n * dsub];
                for i in 0..n {
                    sub_data[i * dsub..(i + 1) * dsub]
                        .copy_from_slice(&data[i * d + offset..i * d + offset + dsub]);
                }

                let init: Option<Vec<f32>> = prev_centroids.as_ref().map(|pc| {
                    let src = sub * ksub * dsub;
                    pc[src..src + ksub * dsub].to_vec()
                });

                kmeans::kmeans_train_with_init(km_config, &sub_data, n, dsub, ksub, init.as_deref())
            })
            .collect();

        self.centroids = vec![0.0f32; m * ksub * dsub];
        for (sub, sub_centroids) in sub_results.into_iter().enumerate() {
            let dst_offset = sub * ksub * dsub;
            self.centroids[dst_offset..dst_offset + ksub * dsub].copy_from_slice(&sub_centroids);
        }
        self.rebuild_norms_cache();
    }

    /// Rebuild the centroid norms cache. Called after training or loading centroids.
    pub fn rebuild_norms_cache(&mut self) {
        self.centroid_norms_cache = vec![0.0f32; self.m * self.ksub];
        for sub in 0..self.m {
            let c_base = sub * self.ksub * self.dsub;
            for j in 0..self.ksub {
                let c_off = c_base + j * self.dsub;
                self.centroid_norms_cache[sub * self.ksub + j] =
                    fvec_norm_l2sqr(&self.centroids[c_off..c_off + self.dsub]);
            }
        }
    }

    /// Bytes per encoded vector.
    pub fn code_size(&self) -> usize {
        if self.nbits == 4 {
            self.m / 2
        } else {
            self.m
        }
    }

    /// Encode a single vector into PQ codes.
    /// For nbits=8: codes has length M (one byte per sub-quantizer).
    /// For nbits=4: codes has length M/2 (two nibbles per byte).
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
        for pair in 0..self.m / 2 {
            let sub_lo = pair * 2;
            let sub_hi = pair * 2 + 1;

            self.compute_sub_l2_distances(x, sub_lo, distances);
            let best_lo = argmin_code(&distances[..self.ksub]);

            self.compute_sub_l2_distances(x, sub_hi, distances);
            let best_hi = argmin_code(&distances[..self.ksub]);

            // Pack: low nibble + high nibble
            codes[pair] = best_lo | (best_hi << 4);
        }
    }

    fn compute_sub_l2_distances(&self, x: &[f32], sub: usize, distances: &mut [f32]) {
        let x_off = sub * self.dsub;
        let c_base = sub * self.ksub * self.dsub;
        let query_sub = &x[x_off..x_off + self.dsub];
        let centroids = &self.centroids[c_base..c_base + self.ksub * self.dsub];

        if self.dsub >= 4 && self.ksub >= 8 {
            fvec_ip_batch(query_sub, centroids, self.dsub, self.ksub, distances);
            let q_norm = fvec_norm_l2sqr(query_sub);
            let norms_base = sub * self.ksub;
            for j in 0..self.ksub {
                let c_norm = if !self.centroid_norms_cache.is_empty() {
                    self.centroid_norms_cache[norms_base + j]
                } else {
                    let c_off = j * self.dsub;
                    fvec_norm_l2sqr(&centroids[c_off..c_off + self.dsub])
                };
                distances[j] = (q_norm + c_norm - 2.0 * distances[j]).max(0.0);
            }
        } else {
            fvec_l2sqr_batch(query_sub, centroids, self.dsub, self.ksub, distances);
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
        if self.nbits == 4 {
            for pair in 0..self.m / 2 {
                let byte = codes[pair];
                let code_lo = (byte & 0x0F) as usize;
                let code_hi = ((byte >> 4) & 0x0F) as usize;

                let sub_lo = pair * 2;
                let sub_hi = pair * 2 + 1;

                let c_off_lo = sub_lo * self.ksub * self.dsub + code_lo * self.dsub;
                let x_off_lo = sub_lo * self.dsub;
                x[x_off_lo..x_off_lo + self.dsub]
                    .copy_from_slice(&self.centroids[c_off_lo..c_off_lo + self.dsub]);

                let c_off_hi = sub_hi * self.ksub * self.dsub + code_hi * self.dsub;
                let x_off_hi = sub_hi * self.dsub;
                x[x_off_hi..x_off_hi + self.dsub]
                    .copy_from_slice(&self.centroids[c_off_hi..c_off_hi + self.dsub]);
            }
        } else {
            for sub in 0..self.m {
                let c_off = sub * self.ksub * self.dsub + (codes[sub] as usize) * self.dsub;
                let x_off = sub * self.dsub;
                x[x_off..x_off + self.dsub]
                    .copy_from_slice(&self.centroids[c_off..c_off + self.dsub]);
            }
        }
    }

    /// Precompute the distance table from a query to all PQ centroids.
    /// Uses sgemm for dsub >= 4 (L2: ||q-c||²=||q||²+||c||²-2q·cᵀ).
    pub fn compute_distance_table(&self, query: &[f32], metric: MetricType, table: &mut [f32]) {
        if self.dsub >= 4 {
            self.compute_distance_table_sgemm(query, metric, table);
        } else {
            self.compute_distance_table_loop(query, metric, table);
        }
    }

    fn compute_distance_table_sgemm(&self, query: &[f32], metric: MetricType, table: &mut [f32]) {
        for sub in 0..self.m {
            let q_off = sub * self.dsub;
            let c_base = sub * self.ksub * self.dsub;
            let t_base = sub * self.ksub;

            // Inner product: ip[ksub] = query_sub[1×dsub] · centroids_sub[ksub×dsub]ᵀ
            sgemm_a_bt(
                1,
                self.ksub,
                self.dsub,
                1.0,
                &query[q_off..q_off + self.dsub],
                &self.centroids[c_base..c_base + self.ksub * self.dsub],
                0.0,
                &mut table[t_base..t_base + self.ksub],
            );

            match metric {
                MetricType::L2 | MetricType::Cosine => {
                    // ||q-c||² = ||q||² + ||c||² - 2·q·c
                    // Use pre-cached centroid norms (avoids recomputing per query)
                    let q_norm = fvec_norm_l2sqr(&query[q_off..q_off + self.dsub]);
                    let norms_base = sub * self.ksub;
                    for j in 0..self.ksub {
                        let c_norm = if !self.centroid_norms_cache.is_empty() {
                            self.centroid_norms_cache[norms_base + j]
                        } else {
                            let c_off = c_base + j * self.dsub;
                            fvec_norm_l2sqr(&self.centroids[c_off..c_off + self.dsub])
                        };
                        table[t_base + j] = q_norm + c_norm - 2.0 * table[t_base + j];
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

    fn compute_distance_table_loop(&self, query: &[f32], metric: MetricType, table: &mut [f32]) {
        for sub in 0..self.m {
            let q_off = sub * self.dsub;
            let c_base = sub * self.ksub * self.dsub;
            let t_base = sub * self.ksub;

            match metric {
                MetricType::L2 | MetricType::Cosine => {
                    fvec_l2sqr_batch(
                        &query[q_off..q_off + self.dsub],
                        &self.centroids[c_base..c_base + self.ksub * self.dsub],
                        self.dsub,
                        self.ksub,
                        &mut table[t_base..t_base + self.ksub],
                    );
                }
                MetricType::InnerProduct => {
                    fvec_ip_batch(
                        &query[q_off..q_off + self.dsub],
                        &self.centroids[c_base..c_base + self.ksub * self.dsub],
                        self.dsub,
                        self.ksub,
                        &mut table[t_base..t_base + self.ksub],
                    );
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
            let q_off = sub * self.dsub;
            let c_base = sub * self.ksub * self.dsub;
            let t_base = sub * self.ksub;

            fvec_ip_batch(
                &query[q_off..q_off + self.dsub],
                &self.centroids[c_base..c_base + self.ksub * self.dsub],
                self.dsub,
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
        for pair in 0..self.m / 2 {
            let byte = codes[pair];
            let code_lo = (byte & 0x0F) as usize;
            let code_hi = ((byte >> 4) & 0x0F) as usize;

            let sub_lo = pair * 2;
            let sub_hi = pair * 2 + 1;

            dist += table[sub_lo * self.ksub + code_lo];
            dist += table[sub_hi * self.ksub + code_hi];
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
            let c_base = sub * self.ksub * self.dsub;
            for j in 0..self.ksub {
                let c_off = c_base + j * self.dsub;
                norms[sub * self.ksub + j] =
                    fvec_norm_l2sqr(&self.centroids[c_off..c_off + self.dsub]);
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
}
