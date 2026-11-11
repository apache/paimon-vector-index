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
use std::sync::OnceLock;

/// Product Quantizer aligned with Faiss's ProductQuantizer.
///
/// Splits D-dimensional vectors into M contiguous chunks and independently
/// quantizes each chunk with `ksub` centroids. Uniform chunks remain the
/// default for IVF-PQ compatibility; DiskANN may use near-equal chunks when
/// `D` is not divisible by `M`.
///
/// Centroids are chunk-major. Chunk `m` starts at
/// `chunk_offsets[m] * ksub`, and each of its `ksub` centroids contains
/// `chunk_offsets[m + 1] - chunk_offsets[m]` contiguous components. Use
/// [`Self::set_centroids`] to replace them so derived caches stay synchronized.
/// Layout is immutable after construction so derived caches stay valid.
///
/// ```compile_fail,E0616
/// use paimon_vindex_core::pq::ProductQuantizer;
/// let mut pq = ProductQuantizer::new(12, 2);
/// pq.chunk_offsets = vec![0, 4, 12];
/// ```
pub struct ProductQuantizer {
    d: usize,
    m: usize,
    nbits: usize,
    /// Uniform chunk width for legacy formats, or the largest chunk width for
    /// a balanced non-uniform layout.
    dsub: usize,
    ksub: usize,
    chunk_offsets: Vec<usize>,
    centroids: Vec<f32>,
    centroids_are_finite: bool,
    /// Pre-computed squared norms of each centroid: [M * ksub].
    /// Avoids recomputing per query for L2 distance table.
    centroid_norms_cache: Vec<f32>,
    transposed_codebook_cache: OnceLock<Vec<f32>>,
}

impl ProductQuantizer {
    /// Vector dimension.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Number of subquantizers.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Bits per PQ code.
    pub fn nbits(&self) -> usize {
        self.nbits
    }

    /// Largest subvector dimension.
    pub fn dsub(&self) -> usize {
        self.dsub
    }

    /// Centroids per subquantizer.
    pub fn ksub(&self) -> usize {
        self.ksub
    }

    /// Contiguous subvector boundaries.
    pub fn chunk_offsets(&self) -> &[usize] {
        &self.chunk_offsets
    }

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
            centroids_are_finite: true,
            centroid_norms_cache: Vec::new(),
            transposed_codebook_cache: OnceLock::new(),
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

    /// Return the chunk-major PQ codebook.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Replace the PQ codebook and refresh its derived norm and transpose caches.
    pub fn set_centroids(&mut self, centroids: Vec<f32>) {
        self.try_set_centroids(centroids)
            .expect("PQ centroid norms allocation failed");
    }

    pub(crate) fn try_set_centroids(
        &mut self,
        centroids: Vec<f32>,
    ) -> Result<(), std::collections::TryReserveError> {
        assert_eq!(
            centroids.len(),
            self.d * self.ksub,
            "PQ centroids must hold d * ksub values"
        );
        let norms = self.try_compute_centroid_norms(&centroids)?;
        self.centroids_are_finite = centroids.iter().all(|value| value.is_finite());
        self.centroids = centroids;
        self.centroid_norms_cache = norms;
        self.transposed_codebook_cache.take();
        Ok(())
    }

    /// Copy trained state while leaving the writer-local transpose lazy.
    pub(crate) fn clone_without_transposed_cache(&self) -> Self {
        Self {
            d: self.d,
            m: self.m,
            nbits: self.nbits,
            dsub: self.dsub,
            ksub: self.ksub,
            chunk_offsets: self.chunk_offsets.clone(),
            centroids: self.centroids.clone(),
            centroids_are_finite: self.centroids_are_finite,
            centroid_norms_cache: self.centroid_norms_cache.clone(),
            transposed_codebook_cache: OnceLock::new(),
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
        self.train_hot_start_with_parallelism(
            data,
            n,
            km_config,
            hot_start,
            rayon::current_num_threads().max(1),
        );
    }

    pub(crate) fn train_hot_start_with_parallelism(
        &mut self,
        data: &[f32],
        n: usize,
        km_config: &KMeansConfig,
        hot_start: bool,
        max_parallelism: usize,
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

        let train_subquantizers = || {
            (0..m)
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
                .collect::<Vec<Vec<f32>>>()
        };
        let current_parallelism = rayon::current_num_threads().max(1);
        let sub_results = if max_parallelism >= current_parallelism {
            train_subquantizers()
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_parallelism.max(1))
                .build()
                .expect("bounded PQ training thread pool should be constructible")
                .install(train_subquantizers)
        };

        let mut centroids = vec![0.0f32; d * ksub];
        for (sub, sub_centroids) in sub_results.into_iter().enumerate() {
            let chunk_dim = self.chunk_dim(sub);
            let dst_offset = self.centroid_chunk_base(sub);
            centroids[dst_offset..dst_offset + ksub * chunk_dim].copy_from_slice(&sub_centroids);
        }
        self.set_centroids(centroids);
    }

    fn try_compute_centroid_norms(
        &self,
        centroids: &[f32],
    ) -> Result<Vec<f32>, std::collections::TryReserveError> {
        let mut norms = Vec::new();
        norms.try_reserve_exact(self.m * self.ksub)?;
        norms.resize(self.m * self.ksub, 0.0f32);
        for sub in 0..self.m {
            let chunk_dim = self.chunk_dim(sub);
            let c_base = self.centroid_chunk_base(sub);
            for j in 0..self.ksub {
                let c_off = c_base + j * chunk_dim;
                norms[sub * self.ksub + j] = fvec_norm_l2sqr(&centroids[c_off..c_off + chunk_dim]);
            }
        }
        Ok(norms)
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
    ///
    /// This is the canonical path: results are bit-identical to the
    /// per-vector [`Self::encode`] on the same runtime backend, which golden
    /// storage fixtures rely on (DiskANN serializes these codes). IVF-PQ's add
    /// path uses [`Self::encode_batch_blocked`] instead.
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

    /// Automatic batch encode for the IVF-PQ add path.
    ///
    /// The backend depends only on shape and CPU features, never on `n`:
    /// - 8-bit shapes with `dsub >= 4` use the transposed direct-L2 encoder
    ///   when this CPU has a fast fused kernel.
    /// - The same finite shape uses blocked SGEMM otherwise, avoiding billions
    ///   of software `fmaf` calls on x86 CPUs without FMA.
    /// - Other shapes, and non-finite codebooks on the SGEMM fallback, use the
    ///   canonical encoder.
    ///
    /// Each backend is deterministic across thread counts and batch splits,
    /// but their floating-point contracts differ. The transposed path treats
    /// NaN distances as losing, lets infinite distances lose, returns code 0
    /// when no distance is below `f32::MAX`, and computes direct squared L2.
    /// The SGEMM and canonical paths use the expanded form
    /// `|q|² + |c|² - 2q·c`, including its existing NaN and large-offset
    /// behavior. Canonical mode reproduces the per-vector encoder on the same
    /// CPU and runtime backend; neither mode promises cross-CPU code identity.
    pub(crate) fn encode_batch_blocked(&self, data: &[f32], n: usize, codes: &mut [u8]) {
        if self.nbits == 8 && self.ksub == 256 && (0..self.m).all(|sub| self.chunk_dim(sub) >= 4) {
            if supports_transposed_pq_encoding() {
                self.encode_batch_8bit_transposed(data, n, codes);
            } else if self.centroids_are_finite {
                self.encode_batch_8bit_sgemm(data, n, codes);
            } else {
                self.encode_batch(data, n, codes);
            }
            return;
        }
        self.encode_batch(data, n, codes);
    }

    /// Blocked SGEMM fallback for CPUs without a fast transposed kernel.
    /// Unlike the former main path, this runs for every `n`; block sizes only
    /// divide work and never select another encoder.
    fn encode_batch_8bit_sgemm(&self, data: &[f32], n: usize, codes: &mut [u8]) {
        let d = self.d;
        let m = self.m;
        let ksub = self.ksub;
        let cs = self.code_size();
        debug_assert_eq!(cs, m);
        debug_assert_eq!(ksub, 256);
        debug_assert!((0..m).all(|sub| self.chunk_dim(sub) >= 4));
        debug_assert!(self.centroids_are_finite);

        let max_dsub = (0..m).map(|sub| self.chunk_dim(sub)).max().unwrap_or(0);
        let computed_norms = self
            .centroid_norms_cache
            .is_empty()
            .then(|| self.compute_centroid_norms());
        let centroid_norms = computed_norms
            .as_ref()
            .unwrap_or(&self.centroid_norms_cache);
        let block_rows = encode_block_rows(n, rayon::current_num_threads());
        codes[..n * cs]
            .par_chunks_mut(block_rows * cs)
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0.0f32; block_rows * max_dsub],
                        vec![0.0f32; block_rows * ksub],
                    )
                },
                |(queries, distances), (block_idx, block_codes)| {
                    let row0 = block_idx * block_rows;
                    let rows = block_rows.min(n - row0);
                    let block_data = &data[row0 * d..(row0 + rows) * d];

                    for sub in 0..m {
                        let range = self.chunk_range(sub);
                        let dsub = range.len();
                        for r in 0..rows {
                            queries[r * dsub..(r + 1) * dsub].copy_from_slice(
                                &block_data[r * d + range.start..r * d + range.end],
                            );
                        }
                        let c_base = self.centroid_chunk_base(sub);
                        sgemm_a_bt(
                            rows,
                            ksub,
                            dsub,
                            1.0,
                            &queries[..rows * dsub],
                            &self.centroids[c_base..c_base + ksub * dsub],
                            0.0,
                            &mut distances[..rows * ksub],
                        );
                        for r in 0..rows {
                            let q_norm = fvec_norm_l2sqr(&queries[r * dsub..(r + 1) * dsub]);
                            let row_distances = &mut distances[r * ksub..(r + 1) * ksub];
                            for j in 0..ksub {
                                row_distances[j] = (q_norm + centroid_norms[sub * ksub + j]
                                    - 2.0 * row_distances[j])
                                    .max(0.0);
                            }
                            block_codes[r * cs + sub] = argmin_code(row_distances);
                        }
                    }
                },
            );
    }

    /// Compact per-sub `[dsub][ksub]` blocks at `chunk_offsets[sub] * ksub`.
    /// Even irregular layouts need only as much space as the original codebook.
    fn build_transposed_codebook(&self) -> Vec<f32> {
        let m = self.m;
        let ksub = self.ksub;
        let mut transposed = vec![0.0f32; self.centroids.len()];
        for sub in 0..m {
            let dsub = self.chunk_dim(sub);
            let c_base = self.centroid_chunk_base(sub);
            let dst = &mut transposed[c_base..c_base + dsub * ksub];
            for j in 0..ksub {
                for k in 0..dsub {
                    dst[k * ksub + j] = self.centroids[c_base + j * dsub + k];
                }
            }
        }
        transposed
    }

    /// Transposed-codebook encode for nbits=8. Rows are split into blocks
    /// only for parallelism; each row is encoded independently.
    fn encode_batch_8bit_transposed(&self, data: &[f32], n: usize, codes: &mut [u8]) {
        let d = self.d;
        let m = self.m;
        let ksub = self.ksub;
        let cs = self.code_size();
        debug_assert_eq!(cs, m);

        let transposed = self
            .transposed_codebook_cache
            .get_or_init(|| self.build_transposed_codebook());
        let kernels: Vec<ScoreArgminKernel> = (0..m)
            .map(|sub| score_argmin_kernel(self.chunk_dim(sub), ksub))
            .collect();

        let block_rows = encode_block_rows(n, rayon::current_num_threads());
        codes[..n * cs]
            .par_chunks_mut(block_rows * cs)
            .enumerate()
            .for_each_init(
                || vec![0.0f32; ksub],
                |scores, (block_idx, block_codes)| {
                    let row0 = block_idx * block_rows;
                    let rows = block_rows.min(n - row0);
                    let block_data = &data[row0 * d..(row0 + rows) * d];

                    for r in 0..rows {
                        let row = &block_data[r * d..(r + 1) * d];
                        for sub in 0..m {
                            let range = self.chunk_range(sub);
                            let dsub = range.len();
                            let q = &row[range];
                            let base = self.centroid_chunk_base(sub);
                            let t = &transposed[base..base + dsub * ksub];
                            block_codes[r * cs + sub] =
                                score_argmin(kernels[sub], q, t, ksub, scores);
                        }
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
    /// Reuses the encoder's transposed codebook for single-query SIMD across
    /// centroids, avoiding a separate SGEMM packing operation for every chunk.
    /// L2 retains the expanded form ||q-c||²=||q||²+||c||²-2q·cᵀ.
    pub fn compute_distance_table(&self, query: &[f32], metric: MetricType, table: &mut [f32]) {
        self.compute_inner_product_table(query, table);
        for sub in 0..self.m {
            let range = self.chunk_range(sub);
            let chunk_dim = range.len();
            let c_base = self.centroid_chunk_base(sub);
            let t_base = sub * self.ksub;
            let query_chunk = &query[range];

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
        let transposed = self
            .transposed_codebook_cache
            .get_or_init(|| self.build_transposed_codebook());
        for sub in 0..self.m {
            let range = self.chunk_range(sub);
            let chunk_dim = range.len();
            let t_base = sub * self.ksub;
            let c_base = self.centroid_chunk_base(sub);
            transposed_inner_products(
                &query[range],
                &transposed[c_base..c_base + self.ksub * chunk_dim],
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

/// Independent centroid lanes keep the query broadcasts and partial sums in
/// registers. The fixed lane count lets LLVM vectorize without horizontal
/// reductions or a BLAS packing/scratch allocation. Both PQ codebook sizes
/// are multiples of 16.
fn transposed_inner_products(query: &[f32], transposed: &[f32], ksub: usize, table: &mut [f32]) {
    assert_eq!(transposed.len(), query.len() * ksub);
    assert_eq!(table.len(), ksub);
    for (block, output) in table.chunks_exact_mut(16).enumerate() {
        let mut sums = [0.0f32; 16];
        for (dim, &q) in query.iter().enumerate() {
            let values: &[f32; 16] = transposed[dim * ksub + block * 16..][..16]
                .try_into()
                .unwrap();
            for lane in 0..16 {
                sums[lane] += q * values[lane];
            }
        }
        output.copy_from_slice(&sums);
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn supports_transposed_pq_encoding() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn supports_transposed_pq_encoding() -> bool {
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn supports_transposed_pq_encoding() -> bool {
    false
}

/// Row block for the automatic batch-encode paths. 512 rows keeps scratch
/// buffers hot and yields enough blocks to occupy the worker pool. Blocks
/// only divide work; they never select another encoder.
const MAX_ENCODE_BLOCK_ROWS: usize = 512;

fn encode_block_rows(rows: usize, workers: usize) -> usize {
    rows.div_ceil(workers.max(1))
        .clamp(1, MAX_ENCODE_BLOCK_ROWS)
}

/// Squared-L2 argmin kernel over a transposed sub-codebook `t` laid out as
/// `[dsub][ksub]` (stride-1 over `j`). `scores` is a reusable `ksub`-sized
/// scratch buffer that some kernels do not touch.
type ScoreArgminKernel = fn(&[f32], &[f32], usize, &mut [f32]) -> u8;

/// Pick the kernel for one sub-quantizer. The choice depends only on `dsub`
/// and CPU features, never on the batch size, which is what makes
/// [`ProductQuantizer::encode_batch_blocked`] batch invariant. Every kernel
/// returned here is byte-identical to [`score_argmin_scalar`].
fn score_argmin_kernel(dsub: usize, ksub: usize) -> ScoreArgminKernel {
    #[cfg(target_arch = "aarch64")]
    {
        if dsub == 4 && ksub.is_multiple_of(4) {
            return score_argmin_neon_d4_entry;
        }
        if dsub >= 4 && ksub.is_multiple_of(16) {
            return score_argmin_neon_generic_entry;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            if dsub == 4 && ksub.is_multiple_of(8) {
                return score_argmin_avx2_d4_entry;
            }
            return score_argmin_avx2_generic_entry;
        }
    }
    let _ = (dsub, ksub);
    score_argmin_scalar
}

/// Run `kernel` after checking the slice contract shared by all kernels.
#[inline]
fn score_argmin(
    kernel: ScoreArgminKernel,
    q: &[f32],
    t: &[f32],
    ksub: usize,
    scores: &mut [f32],
) -> u8 {
    assert_eq!(q.len().checked_mul(ksub), Some(t.len()));
    assert!(scores.len() >= ksub);
    kernel(q, t, ksub, scores)
}

/// Direct squared-difference accumulation, one `mul_add` per dimension so
/// every kernel rounds once per dimension in the same order.
#[inline(always)]
fn score_argmin_accumulate(q: &[f32], t: &[f32], ksub: usize, scores: &mut [f32]) -> u8 {
    let scores = &mut scores[..ksub];
    scores.fill(0.0);
    for (k, &qv) in q.iter().enumerate() {
        let tk = &t[k * ksub..(k + 1) * ksub];
        for (s, &tv) in scores.iter_mut().zip(tk) {
            let diff = qv - tv;
            *s = diff.mul_add(diff, *s);
        }
    }
    argmin_code(scores)
}

/// Scalar oracle for every SIMD kernel.
fn score_argmin_scalar(q: &[f32], t: &[f32], ksub: usize, scores: &mut [f32]) -> u8 {
    score_argmin_accumulate(q, t, ksub, scores)
}

/// Generic AVX2+FMA kernel for any `dsub`: the same loop as the scalar
/// oracle compiled with the target features enabled so the stride-1 inner
/// loop vectorizes with `vfmadd`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn score_argmin_avx2_generic(q: &[f32], t: &[f32], ksub: usize, scores: &mut [f32]) -> u8 {
    score_argmin_accumulate(q, t, ksub, scores)
}

#[cfg(target_arch = "x86_64")]
fn score_argmin_avx2_generic_entry(q: &[f32], t: &[f32], ksub: usize, scores: &mut [f32]) -> u8 {
    // SAFETY: only handed out by `score_argmin_kernel` after AVX2 and FMA were
    // detected on this CPU.
    unsafe { score_argmin_avx2_generic(q, t, ksub, scores) }
}

#[cfg(target_arch = "x86_64")]
fn score_argmin_avx2_d4_entry(q: &[f32], t: &[f32], ksub: usize, _scores: &mut [f32]) -> u8 {
    debug_assert_eq!(q.len(), 4);
    debug_assert!(ksub.is_multiple_of(8));
    // SAFETY: only handed out by `score_argmin_kernel` after AVX2 and FMA were
    // detected on this CPU; slice bounds are checked by `score_argmin`.
    unsafe { score_argmin_avx2_d4(q, t, ksub) }
}

#[cfg(target_arch = "aarch64")]
fn score_argmin_neon_generic_entry(q: &[f32], t: &[f32], ksub: usize, _scores: &mut [f32]) -> u8 {
    debug_assert!(q.len() >= 4);
    debug_assert!(ksub.is_multiple_of(16));
    // SAFETY: NEON is baseline on aarch64; slice bounds are checked by
    // `score_argmin`.
    unsafe { score_argmin_neon_generic(q, t, ksub) }
}

#[cfg(target_arch = "aarch64")]
fn score_argmin_neon_d4_entry(q: &[f32], t: &[f32], ksub: usize, _scores: &mut [f32]) -> u8 {
    debug_assert_eq!(q.len(), 4);
    debug_assert!(ksub.is_multiple_of(4));
    // SAFETY: NEON is baseline on aarch64; slice bounds are checked by
    // `score_argmin`.
    unsafe { score_argmin_neon_d4(q, t, ksub) }
}

/// Generic NEON kernel: score 16 centroids together, then update the running
/// SIMD argmin without materializing the score table.
#[cfg(target_arch = "aarch64")]
unsafe fn score_argmin_neon_generic(q: &[f32], t: &[f32], ksub: usize) -> u8 {
    use std::arch::aarch64::*;

    unsafe {
        let mut min_val = vdupq_n_f32(f32::MAX);
        let mut min_idx = vdupq_n_u32(0);
        let lane0: [u32; 4] = [0, 1, 2, 3];
        let mut cur_idx = vld1q_u32(lane0.as_ptr());
        let four = vdupq_n_u32(4);

        for j in (0..ksub).step_by(16) {
            let q0 = vdupq_n_f32(q[0]);
            let t0 = t.as_ptr().add(j);
            let d0 = vsubq_f32(q0, vld1q_f32(t0));
            let d1 = vsubq_f32(q0, vld1q_f32(t0.add(4)));
            let d2 = vsubq_f32(q0, vld1q_f32(t0.add(8)));
            let d3 = vsubq_f32(q0, vld1q_f32(t0.add(12)));
            let mut s0 = vmulq_f32(d0, d0);
            let mut s1 = vmulq_f32(d1, d1);
            let mut s2 = vmulq_f32(d2, d2);
            let mut s3 = vmulq_f32(d3, d3);

            for (k, &qv) in q.iter().enumerate().skip(1) {
                let qk = vdupq_n_f32(qv);
                let tk = t.as_ptr().add(k * ksub + j);
                let d0 = vsubq_f32(qk, vld1q_f32(tk));
                let d1 = vsubq_f32(qk, vld1q_f32(tk.add(4)));
                let d2 = vsubq_f32(qk, vld1q_f32(tk.add(8)));
                let d3 = vsubq_f32(qk, vld1q_f32(tk.add(12)));
                s0 = vfmaq_f32(s0, d0, d0);
                s1 = vfmaq_f32(s1, d1, d1);
                s2 = vfmaq_f32(s2, d2, d2);
                s3 = vfmaq_f32(s3, d3, d3);
            }

            for score in [s0, s1, s2, s3] {
                let mask = vcltq_f32(score, min_val);
                min_val = vbslq_f32(mask, score, min_val);
                min_idx = vbslq_u32(mask, cur_idx, min_idx);
                cur_idx = vaddq_u32(cur_idx, four);
            }
        }

        let mut vals = [0.0f32; 4];
        let mut idxs = [0u32; 4];
        vst1q_f32(vals.as_mut_ptr(), min_val);
        vst1q_u32(idxs.as_mut_ptr(), min_idx);
        let mut best = idxs[0];
        let mut best_val = vals[0];
        for lane in 1..4 {
            if vals[lane] < best_val || (vals[lane] == best_val && idxs[lane] < best) {
                best_val = vals[lane];
                best = idxs[lane];
            }
        }
        best as u8
    }
}

/// dsub=4 NEON kernel: 4 broadcast-FMA rows, SIMD min+index tracking,
/// horizontal reduce with smallest-index tie-break.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn score_argmin_neon_d4(q: &[f32], t: &[f32], ksub: usize) -> u8 {
    use std::arch::aarch64::*;

    let t0 = t.as_ptr();
    let t1 = unsafe { t0.add(ksub) };
    let t2 = unsafe { t0.add(2 * ksub) };
    let t3 = unsafe { t0.add(3 * ksub) };

    unsafe {
        let q0 = vdupq_n_f32(q[0]);
        let q1 = vdupq_n_f32(q[1]);
        let q2 = vdupq_n_f32(q[2]);
        let q3 = vdupq_n_f32(q[3]);

        let mut min_val = vdupq_n_f32(f32::MAX);
        let mut min_idx = vdupq_n_u32(0);
        let lane0: [u32; 4] = [0, 1, 2, 3];
        let mut cur_idx = vld1q_u32(lane0.as_ptr());
        let step = vdupq_n_u32(4);

        for j in (0..ksub).step_by(4) {
            let d0 = vsubq_f32(q0, vld1q_f32(t0.add(j)));
            let d1 = vsubq_f32(q1, vld1q_f32(t1.add(j)));
            let d2 = vsubq_f32(q2, vld1q_f32(t2.add(j)));
            let d3 = vsubq_f32(q3, vld1q_f32(t3.add(j)));
            let mut s = vmulq_f32(d0, d0);
            s = vfmaq_f32(s, d1, d1);
            s = vfmaq_f32(s, d2, d2);
            s = vfmaq_f32(s, d3, d3);

            // Strictly-smaller keeps the earliest index on equal scores.
            let mask = vcltq_f32(s, min_val);
            min_val = vbslq_f32(mask, s, min_val);
            min_idx = vbslq_u32(mask, cur_idx, min_idx);
            cur_idx = vaddq_u32(cur_idx, step);
        }

        let mut vals = [0.0f32; 4];
        let mut idxs = [0u32; 4];
        vst1q_f32(vals.as_mut_ptr(), min_val);
        vst1q_u32(idxs.as_mut_ptr(), min_idx);
        let mut best = idxs[0];
        let mut best_val = vals[0];
        for l in 1..4 {
            if vals[l] < best_val || (vals[l] == best_val && idxs[l] < best) {
                best_val = vals[l];
                best = idxs[l];
            }
        }
        best as u8
    }
}

/// dsub=4 AVX2 kernel: mirrors the NEON version 8-wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn score_argmin_avx2_d4(q: &[f32], t: &[f32], ksub: usize) -> u8 {
    use std::arch::x86_64::*;

    let t0 = t.as_ptr();
    let t1 = unsafe { t0.add(ksub) };
    let t2 = unsafe { t0.add(2 * ksub) };
    let t3 = unsafe { t0.add(3 * ksub) };

    unsafe {
        let q0 = _mm256_set1_ps(q[0]);
        let q1 = _mm256_set1_ps(q[1]);
        let q2 = _mm256_set1_ps(q[2]);
        let q3 = _mm256_set1_ps(q[3]);

        let mut min_val = _mm256_set1_ps(f32::MAX);
        let mut min_idx = _mm256_setzero_si256();
        let mut cur_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let step = _mm256_set1_epi32(8);

        for j in (0..ksub).step_by(8) {
            let d0 = _mm256_sub_ps(q0, _mm256_loadu_ps(t0.add(j)));
            let d1 = _mm256_sub_ps(q1, _mm256_loadu_ps(t1.add(j)));
            let d2 = _mm256_sub_ps(q2, _mm256_loadu_ps(t2.add(j)));
            let d3 = _mm256_sub_ps(q3, _mm256_loadu_ps(t3.add(j)));
            let mut s = _mm256_mul_ps(d0, d0);
            s = _mm256_fmadd_ps(d1, d1, s);
            s = _mm256_fmadd_ps(d2, d2, s);
            s = _mm256_fmadd_ps(d3, d3, s);

            // Strictly-smaller keeps the earliest index on equal scores.
            let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(s, min_val);
            min_val = _mm256_blendv_ps(min_val, s, mask);
            min_idx = _mm256_blendv_epi8(min_idx, cur_idx, _mm256_castps_si256(mask));
            cur_idx = _mm256_add_epi32(cur_idx, step);
        }

        let mut vals = [0.0f32; 8];
        let mut idxs = [0i32; 8];
        _mm256_storeu_ps(vals.as_mut_ptr(), min_val);
        _mm256_storeu_si256(idxs.as_mut_ptr().cast(), min_idx);
        let mut best = idxs[0] as u32;
        let mut best_val = vals[0];
        for l in 1..8 {
            let idx = idxs[l] as u32;
            if vals[l] < best_val || (vals[l] == best_val && idx < best) {
                best_val = vals[l];
                best = idx;
            }
        }
        best as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::fvec_l2sqr_sub;
    use rand::rngs::StdRng;

    #[test]
    fn query_codebook_cache_stays_compact_for_irregular_chunks() {
        let offsets: Vec<usize> = std::iter::once(0).chain(32..=64).collect();
        let mut pq = ProductQuantizer::try_with_chunk_offsets(64, 8, offsets).unwrap();
        pq.set_centroids(vec![1.0; 64 * 256]);
        let mut table = vec![0.0; pq.m() * pq.ksub()];
        pq.compute_inner_product_table(&[2.0; 64], &mut table);
        assert_eq!(pq.transposed_codebook_cache.get().unwrap().len(), 64 * 256);
        for sub in 0..pq.m() {
            assert!(table[sub * 256..(sub + 1) * 256]
                .iter()
                .all(|&dot| dot == 2.0 * pq.chunk_dim(sub) as f32));
        }
    }

    #[test]
    fn query_tables_match_independent_distances_and_refresh_after_centroid_replacement() {
        let mut rng = StdRng::seed_from_u64(20261111);
        for bits in [4, 8] {
            for width in [1, 2, 3, 4, 5, 8, 12, 48, 64] {
                let d = width * 3 + 1;
                let mut pq = ProductQuantizer::with_nbits_balanced(d, 3, bits);
                let query: Vec<f32> = (0..d).map(|_| rng.gen_range(-2.0..2.0)).collect();
                // Exercise a warmed cache followed by a different codebook.
                for _ in 0..2 {
                    pq.set_centroids(
                        (0..d * pq.ksub())
                            .map(|_| rng.gen_range(-2.0..2.0))
                            .collect(),
                    );
                    let mut dot = vec![f32::NAN; pq.m() * pq.ksub()];
                    pq.compute_inner_product_table(&query, &mut dot);
                    for metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
                        let mut table = vec![f32::NAN; dot.len()];
                        pq.compute_distance_table(&query, metric, &mut table);
                        for sub in 0..pq.m() {
                            for code in 0..pq.ksub() {
                                let base = pq.centroid_chunk_base(sub) + code * pq.chunk_dim(sub);
                                let centroid = &pq.centroids()[base..base + pq.chunk_dim(sub)];
                                let mut expected = 0.0f64;
                                let mut expected_dot = 0.0f64;
                                for (&q, &c) in query[pq.chunk_range(sub)].iter().zip(centroid) {
                                    expected_dot += f64::from(q) * f64::from(c);
                                    expected += (f64::from(q) - f64::from(c)).powi(2);
                                }
                                if metric == MetricType::InnerProduct {
                                    expected = -expected_dot;
                                }
                                let offset = sub * pq.ksub() + code;
                                let tolerance = 2e-5 * pq.chunk_dim(sub) as f64;
                                assert!((f64::from(table[offset]) - expected).abs() < tolerance,
                                    "bits={bits} width={width} sub={sub} code={code} metric={metric:?}");
                                assert!((f64::from(dot[offset]) - expected_dot).abs() < tolerance);
                            }
                        }
                    }
                }
            }
        }
    }
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
        let mut canonical = vec![0u8; n * cs];
        pq.encode_batch(&data, n, &mut canonical);
        let mut blocked = vec![0u8; n * cs];
        pq.encode_batch_blocked(&data, n, &mut blocked);

        assert_eq!(blocked, canonical);
        assert!(blocked.iter().any(|&b| b != 0));
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

    /// Scalar direct-L2 oracle for the transposed path: same transposed
    /// table, always the scalar kernel, no threads.
    fn encode_scalar_oracle(pq: &ProductQuantizer, data: &[f32], n: usize) -> Vec<u8> {
        let cs = pq.code_size();
        assert_eq!(cs, pq.m);
        let transposed = pq.build_transposed_codebook();
        let mut scores = vec![0.0f32; pq.ksub];
        let mut codes = vec![0u8; n * cs];
        for r in 0..n {
            let row = &data[r * pq.d..(r + 1) * pq.d];
            for sub in 0..pq.m {
                let range = pq.chunk_range(sub);
                let dsub = range.len();
                let base = pq.centroid_chunk_base(sub);
                let t = &transposed[base..base + dsub * pq.ksub];
                codes[r * cs + sub] =
                    score_argmin(score_argmin_scalar, &row[range], t, pq.ksub, &mut scores);
            }
        }
        codes
    }

    const BATCH_SIZES: [usize; 11] = [1, 2, 7, 31, 32, 33, 511, 512, 513, 2048, 4096];

    fn trained_pq(d: usize, m: usize, seed: u64) -> (ProductQuantizer, StdRng) {
        let mut rng = StdRng::seed_from_u64(seed);
        let train: Vec<f32> = (0..3000 * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let mut pq = ProductQuantizer::new(d, m);
        pq.train(&train, 3000);
        (pq, rng)
    }

    #[test]
    fn test_encode_batch_8bit_transposed_matches_scalar_oracle() {
        // dsub = 4 (d4 kernel), 5 and 8 (generic kernel).
        for (d, m) in [(32, 8), (25, 5), (40, 5)] {
            let (pq, mut rng) = trained_pq(d, m, 20260820 + d as u64);
            for n in BATCH_SIZES {
                let data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let oracle = encode_scalar_oracle(&pq, &data, n);
                let mut batch = vec![0u8; n * pq.code_size()];
                pq.encode_batch_8bit_transposed(&data, n, &mut batch);
                assert_eq!(batch, oracle, "d={d} m={m} n={n}");
            }
        }
    }

    #[test]
    fn test_centroids_finite_cache_updates_and_clones() {
        let mut pq = ProductQuantizer::new(4, 1);
        for value in [0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0] {
            let mut centroids = vec![0.0; 4 * 256];
            centroids[4 * 256 - 1] = value;
            pq.set_centroids(centroids);
            assert_eq!(pq.centroids_are_finite, value.is_finite());
            let cloned = pq.clone_without_transposed_cache();
            assert_eq!(cloned.centroids_are_finite, value.is_finite());
            if !supports_transposed_pq_encoding() {
                let data = [1.0; 4];
                let mut expected = [0];
                if value.is_finite() {
                    pq.encode_batch_8bit_sgemm(&data, 1, &mut expected);
                } else {
                    pq.encode_batch(&data, 1, &mut expected);
                }
                for encoder in [&pq, &cloned] {
                    let mut actual = [0];
                    encoder.encode_batch_blocked(&data, 1, &mut actual);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn test_transposed_codebook_cache_reuses_and_invalidates() {
        let (mut pq, mut rng) = trained_pq(32, 8, 20260908);
        let data: Vec<f32> = (0..32).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let mut expected = vec![0u8; pq.code_size()];

        assert!(pq.transposed_codebook_cache.get().is_none());
        pq.encode_batch_8bit_transposed(&data, 1, &mut expected);
        let cached = pq.transposed_codebook_cache.get().unwrap().as_ptr();

        let mut actual = vec![0u8; pq.code_size()];
        pq.encode_batch_8bit_transposed(&data, 1, &mut actual);
        assert_eq!(actual, expected);
        assert_eq!(pq.transposed_codebook_cache.get().unwrap().as_ptr(), cached);

        let centroids = pq.centroids().to_vec();
        pq.set_centroids(centroids);
        assert!(pq.transposed_codebook_cache.get().is_none());
        pq.encode_batch_8bit_transposed(&data, 1, &mut actual);
        assert_eq!(actual, expected);
        assert!(pq.transposed_codebook_cache.get().is_some());
    }

    #[test]
    fn test_encode_batch_8bit_transposed_non_uniform_chunks() {
        let d = 13;
        let m = 3;
        let mut rng = StdRng::seed_from_u64(20260821);
        let train: Vec<f32> = (0..2000 * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let mut pq = ProductQuantizer::with_nbits_balanced(d, m, 8);
        pq.train(&train, 2000);
        // Chunks of 5, 4 and 4: generic kernel next to the d4 kernel.
        assert_eq!(pq.chunk_offsets, vec![0, 5, 9, 13]);

        let n = MAX_ENCODE_BLOCK_ROWS + 13;
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let oracle = encode_scalar_oracle(&pq, &data, n);
        let mut batch = vec![0u8; n * pq.code_size()];
        pq.encode_batch_8bit_transposed(&data, n, &mut batch);
        assert_eq!(batch, oracle);
    }

    #[test]
    fn test_encode_batch_8bit_transposed_large_offset_is_correct() {
        // The expanded form cancels here; the direct form must still pick the
        // exact centroid, for every batch size.
        let mut pq = ProductQuantizer::new(4, 1);
        let mut centroids = vec![100_000_016.0; pq.d * pq.ksub];
        centroids[0..4].fill(100_000_008.0);
        centroids[4..8].fill(100_000_000.0);
        pq.set_centroids(centroids);

        for n in BATCH_SIZES {
            let data = vec![100_000_000.0; n * pq.d];
            let mut codes = vec![0; n];
            pq.encode_batch_8bit_transposed(&data, n, &mut codes);
            assert!(codes.iter().all(|&code| code == 1), "n={n}");
        }
    }

    #[test]
    fn test_encode_batch_8bit_transposed_is_batch_invariant() {
        let mut pq = ProductQuantizer::new(4, 1);
        let mut centroids = vec![100.0; pq.d * pq.ksub];
        centroids[0..4].copy_from_slice(&[0.3658799, 0.06077051, -0.46501994, -0.31766486]);
        centroids[4..8].copy_from_slice(&[0.3658799, 0.06077051, -0.46501994, -0.31766483]);
        pq.set_centroids(centroids);

        let max_n = *BATCH_SIZES.last().unwrap();
        let mut largest = vec![0; max_n];
        pq.encode_batch_8bit_transposed(&vec![0.0; max_n * 4], max_n, &mut largest);
        for n in BATCH_SIZES {
            let mut codes = vec![0; n];
            pq.encode_batch_8bit_transposed(&vec![0.0; n * 4], n, &mut codes);
            assert_eq!(codes.as_slice(), &largest[..n], "n={n}");
        }
    }

    #[test]
    fn test_encode_batch_8bit_transposed_non_finite_semantics_are_batch_invariant() {
        let mut pq = ProductQuantizer::new(4, 1);
        let mut centroids = vec![1.0; pq.d * pq.ksub];
        centroids[0..4].fill(f32::NAN);
        centroids[4..8].fill(0.0);
        centroids[8..12].fill(f32::INFINITY);
        centroids[12..16].fill(f32::NEG_INFINITY);
        pq.set_centroids(centroids);

        for n in BATCH_SIZES {
            let mut codes = vec![0; n];
            pq.encode_batch_8bit_transposed(&vec![0.0; n * pq.d], n, &mut codes);
            assert!(codes.iter().all(|&code| code == 1), "n={n}");
        }

        pq.set_centroids(vec![f32::NAN; pq.d * pq.ksub]);
        for n in BATCH_SIZES {
            let mut codes = vec![u8::MAX; n];
            pq.encode_batch_8bit_transposed(&vec![0.0; n * pq.d], n, &mut codes);
            assert!(codes.iter().all(|&code| code == 0), "n={n}");
        }
    }

    #[test]
    fn test_encode_batch_blocked_small_dsub_uses_canonical_path() {
        for dsub in [1, 2, 3] {
            let m = 4;
            let d = m * dsub;
            let (pq, mut rng) = trained_pq(d, m, 20260903 + dsub as u64);
            for n in [1, 33, 513] {
                let data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let mut canonical = vec![0u8; n * m];
                pq.encode_batch(&data, n, &mut canonical);
                let mut blocked = vec![0u8; n * m];
                pq.encode_batch_blocked(&data, n, &mut blocked);
                assert_eq!(blocked, canonical, "dsub={dsub} n={n}");
            }
        }
    }

    #[test]
    fn test_encode_batch_8bit_transposed_is_thread_and_split_invariant() {
        let d = 32;
        let m = 8;
        let (pq, mut rng) = trained_pq(d, m, 20260904);
        let n = 3 * MAX_ENCODE_BLOCK_ROWS + 5;
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let mut whole = vec![0u8; n * m];
        pq.encode_batch_8bit_transposed(&data, n, &mut whole);

        for threads in [1, 2, 11, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut codes = vec![0u8; n * m];
            pool.install(|| pq.encode_batch_8bit_transposed(&data, n, &mut codes));
            assert_eq!(codes, whole, "threads={threads}");
        }

        for batch in [1, 7, MAX_ENCODE_BLOCK_ROWS] {
            let mut codes = vec![0u8; n * m];
            for start in (0..n).step_by(batch) {
                let rows = batch.min(n - start);
                pq.encode_batch_8bit_transposed(
                    &data[start * d..(start + rows) * d],
                    rows,
                    &mut codes[start * m..(start + rows) * m],
                );
            }
            assert_eq!(codes, whole, "batch={batch}");
        }
    }

    #[test]
    fn test_encode_batch_8bit_sgemm_is_thread_and_split_invariant() {
        let d = 32;
        let m = 8;
        let (pq, mut rng) = trained_pq(d, m, 20260905);
        let n = 3 * MAX_ENCODE_BLOCK_ROWS + 5;
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let mut whole = vec![0u8; n * m];
        pq.encode_batch_8bit_sgemm(&data, n, &mut whole);

        for threads in [1, 2, 11, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut codes = vec![0u8; n * m];
            pool.install(|| pq.encode_batch_8bit_sgemm(&data, n, &mut codes));
            assert_eq!(codes, whole, "threads={threads}");
        }

        for batch in [1, 7, MAX_ENCODE_BLOCK_ROWS] {
            let mut codes = vec![0u8; n * m];
            for start in (0..n).step_by(batch) {
                let rows = batch.min(n - start);
                pq.encode_batch_8bit_sgemm(
                    &data[start * d..(start + rows) * d],
                    rows,
                    &mut codes[start * m..(start + rows) * m],
                );
            }
            assert_eq!(codes, whole, "batch={batch}");
        }
    }

    #[test]
    fn test_encode_block_rows_clamps() {
        assert_eq!(encode_block_rows(2730, 12), 228);
        assert_eq!(encode_block_rows(32768, 12), MAX_ENCODE_BLOCK_ROWS);
        assert_eq!(encode_block_rows(5, 16), 1);
        assert_eq!(encode_block_rows(0, 16), 1);
        assert_eq!(encode_block_rows(100, 0), 100);
    }

    /// Every kernel `score_argmin_kernel` can hand out on this machine, plus
    /// the scalar oracle.
    fn kernels_under_test(dsub: usize, ksub: usize) -> Vec<(&'static str, ScoreArgminKernel)> {
        let mut kernels: Vec<(&'static str, ScoreArgminKernel)> = vec![
            ("scalar", score_argmin_scalar),
            ("dispatched", score_argmin_kernel(dsub, ksub)),
        ];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                kernels.push(("avx2_generic", score_argmin_avx2_generic_entry));
                if dsub == 4 && ksub.is_multiple_of(8) {
                    kernels.push(("avx2_d4", score_argmin_avx2_d4_entry));
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if dsub >= 4 && ksub.is_multiple_of(16) {
                kernels.push(("neon_generic", score_argmin_neon_generic_entry));
            }
            if dsub == 4 && ksub.is_multiple_of(4) {
                kernels.push(("neon_d4", score_argmin_neon_d4_entry));
            }
        }
        kernels
    }

    #[test]
    fn test_score_argmin_kernels_match_scalar_oracle() {
        let ksub = 256;
        let mut rng = StdRng::seed_from_u64(20260905);
        for dsub in [4, 5, 8, 12, 16] {
            let mut t: Vec<f32> = (0..dsub * ksub)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect();
            // Inject exact duplicates so some queries hit exact ties.
            for j in (0..ksub).step_by(37) {
                for k in 0..dsub {
                    t[k * ksub + (j + 1) % ksub] = t[k * ksub + j];
                }
            }
            let mut scores = vec![0.0f32; ksub];
            for _ in 0..500 {
                let q: Vec<f32> = (0..dsub).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let expected = score_argmin(score_argmin_scalar, &q, &t, ksub, &mut scores);
                for (name, kernel) in kernels_under_test(dsub, ksub) {
                    let got = score_argmin(kernel, &q, &t, ksub, &mut scores);
                    assert_eq!(got, expected, "kernel {name} dsub {dsub}");
                }
            }
            // Queries equal to a centroid resolve to its smallest duplicate.
            for j in (0..ksub).step_by(37) {
                let q: Vec<f32> = (0..dsub).map(|k| t[k * ksub + j]).collect();
                for (name, kernel) in kernels_under_test(dsub, ksub) {
                    assert_eq!(
                        score_argmin(kernel, &q, &t, ksub, &mut scores) as usize,
                        j,
                        "kernel {name} dsub {dsub} j {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_score_argmin_non_finite_semantics() {
        let ksub = 256;
        for dsub in [4, 5] {
            let q = vec![0.0f32; dsub];
            let mut t = vec![1.0f32; dsub * ksub];
            for k in 0..dsub {
                t[k * ksub] = f32::NAN;
                t[k * ksub + 1] = 0.0;
                t[k * ksub + 2] = f32::INFINITY;
                t[k * ksub + 3] = f32::NEG_INFINITY;
            }
            let mut scores = vec![0.0f32; ksub];
            for (name, kernel) in kernels_under_test(dsub, ksub) {
                assert_eq!(
                    score_argmin(kernel, &q, &t, ksub, &mut scores),
                    1,
                    "kernel {name} dsub {dsub}"
                );
            }

            t.fill(f32::NAN);
            for (name, kernel) in kernels_under_test(dsub, ksub) {
                assert_eq!(
                    score_argmin(kernel, &q, &t, ksub, &mut scores),
                    0,
                    "all NaN: kernel {name} dsub {dsub}"
                );
            }
        }
    }

    #[test]
    fn test_score_argmin_tie_prefers_smallest_index() {
        let ksub = 256;
        let dsub = 4;
        let q = [0.25f32, -0.5, 0.75, -0.125];
        let mut scores = vec![0.0f32; ksub];

        // All centroids identical -> every score ties -> index 0 wins.
        let all_equal = vec![0.5f32; dsub * ksub];
        for (name, kernel) in kernels_under_test(dsub, ksub) {
            assert_eq!(
                score_argmin(kernel, &q, &all_equal, ksub, &mut scores),
                0,
                "{name}"
            );
        }

        // Exact ties at indices 1 and 8 (different AVX2 lanes) and at 8 and 9
        // (the same lane group): the smallest index must win each time.
        for (tie_a, tie_b) in [(1usize, 8usize), (8, 9), (7, 8), (255, 1)] {
            let mut t = vec![3.0f32; dsub * ksub];
            for k in 0..dsub {
                t[k * ksub + tie_a] = q[k];
                t[k * ksub + tie_b] = q[k];
            }
            let expected = tie_a.min(tie_b);
            for (name, kernel) in kernels_under_test(dsub, ksub) {
                assert_eq!(
                    score_argmin(kernel, &q, &t, ksub, &mut scores) as usize,
                    expected,
                    "{name} tie ({tie_a}, {tie_b})"
                );
            }
        }
    }

    #[test]
    fn test_score_argmin_scalar_matches_squared_distance() {
        let q = [0.4f32, -0.7, 0.2, 0.9, -0.3];
        let ksub = 17;
        let t: Vec<f32> = (0..q.len() * ksub)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) / 7.0)
            .collect();
        let distances: Vec<f32> = (0..ksub)
            .map(|j| {
                q.iter()
                    .enumerate()
                    .map(|(k, value)| (value - t[k * ksub + j]).powi(2))
                    .sum()
            })
            .collect();
        let mut scores = vec![0.0; ksub];
        assert_eq!(
            score_argmin(score_argmin_scalar, &q, &t, ksub, &mut scores),
            argmin_code(&distances)
        );
    }

    #[test]
    #[should_panic]
    fn test_score_argmin_rejects_short_transposed_codebook() {
        let mut scores = vec![0.0; 256];
        score_argmin(
            score_argmin_scalar,
            &[0.0; 4],
            &[0.0; 4 * 256 - 1],
            256,
            &mut scores,
        );
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
        let mut centroids = vec![0.0; ksub * d];
        let mut query = vec![0.0; d];
        for i in 0..d {
            let value = if i.is_multiple_of(2) {
                1.0e10_f32 + i as f32
            } else {
                -1.0e10_f32 + i as f32
            };
            query[i] = value;
            centroids[i] = value;
        }
        pq.set_centroids(centroids);

        let mut table = vec![0.0; m * ksub];
        pq.compute_distance_table(&query, MetricType::L2, &mut table);

        assert_eq!(table[0], 0.0);
    }
}
