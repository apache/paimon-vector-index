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

use crate::distance::{
    fvec_madd, fvec_normalize, pq_distance_four_codes, pq_distance_from_table, MetricType,
};
use crate::io::{IVFPQIndexReader, SeekRead};
use crate::kmeans::{self, KMeansConfig};
use crate::opq::OPQMatrix;
use crate::pq::ProductQuantizer;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io;

/// IVF-PQ index aligned with Faiss's IndexIVFPQ.
pub struct IVFPQIndex {
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub by_residual: bool,

    pub quantizer_centroids: Vec<f32>,
    pub pq: ProductQuantizer,
    pub opq: Option<OPQMatrix>,

    pub ids: Vec<Vec<i64>>,
    pub codes: Vec<Vec<u8>>,

    /// Precomputed table [nlist * M * ksub] for L2+by_residual mode.
    /// Avoids recomputing distance table per list during search.
    precomputed_table: Vec<f32>,
    /// Block-layout packed codes for 4-bit FastScan. One per list.
    fastscan_codes: Vec<Vec<u8>>,
}

impl IVFPQIndex {
    pub fn new(d: usize, nlist: usize, m: usize, metric: MetricType, use_opq: bool) -> Self {
        Self::with_nbits(d, nlist, m, 8, metric, use_opq)
    }

    pub fn with_nbits(
        d: usize,
        nlist: usize,
        m: usize,
        nbits: usize,
        metric: MetricType,
        use_opq: bool,
    ) -> Self {
        let by_residual = metric == MetricType::L2;
        IVFPQIndex {
            d,
            nlist,
            metric,
            by_residual,
            quantizer_centroids: Vec::new(),
            pq: ProductQuantizer::with_nbits(d, m, nbits),
            opq: if use_opq {
                Some(OPQMatrix::new(d, m))
            } else {
                None
            },
            ids: vec![Vec::new(); nlist],
            codes: vec![Vec::new(); nlist],
            precomputed_table: Vec::new(),
            fastscan_codes: Vec::new(),
        }
    }

    /// Create an index with automatic nlist based on target partition size.
    /// nlist = max(1, n / target_partition_size), clamped to reasonable bounds.
    pub fn with_target_partition_size(
        d: usize,
        n: usize,
        target_partition_size: usize,
        m: usize,
        metric: MetricType,
        use_opq: bool,
    ) -> Self {
        let nlist = (n / target_partition_size.max(1)).clamp(1, 65536);
        Self::new(d, nlist, m, metric, use_opq)
    }

    /// Create an index from an already-trained index, copying centroids, codebooks, and OPQ.
    /// The new index has empty inverted lists — call `add()` to populate.
    /// Used for distributed build: train once globally, then each worker creates from_trained.
    pub fn from_trained(trained: &IVFPQIndex) -> Self {
        IVFPQIndex {
            d: trained.d,
            nlist: trained.nlist,
            metric: trained.metric,
            by_residual: trained.by_residual,
            quantizer_centroids: trained.quantizer_centroids.clone(),
            pq: ProductQuantizer {
                d: trained.pq.d,
                m: trained.pq.m,
                nbits: trained.pq.nbits,
                dsub: trained.pq.dsub,
                ksub: trained.pq.ksub,
                centroids: trained.pq.centroids.clone(),
                centroid_norms_cache: trained.pq.centroid_norms_cache.clone(),
            },
            opq: trained.opq.as_ref().map(|o| OPQMatrix {
                d: o.d,
                m: o.m,
                niter: 0,
                niter_pq: 0,
                niter_pq_0: 0,
                max_train_points: 0,
                rotation: o.rotation.clone(),
                is_trained: true,
            }),
            ids: vec![Vec::new(); trained.nlist],
            codes: vec![Vec::new(); trained.nlist],
            precomputed_table: Vec::new(),
            fastscan_codes: Vec::new(),
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        let d = self.d;

        let train_data = if self.metric == MetricType::Cosine {
            let mut normalized = data[..n * d].to_vec();
            for i in 0..n {
                fvec_normalize(&mut normalized[i * d..(i + 1) * d]);
            }
            normalized
        } else {
            data[..n * d].to_vec()
        };

        // When OPQ is enabled, jointly train rotation + PQ, then project data.
        // IVF centroids must be trained on projected (rotated) data since
        // add() and search() assign rotated vectors via preprocess_queries().
        let effective_data = if let Some(ref mut opq) = self.opq {
            opq.train(&train_data, n, &mut self.pq);
            let mut projected = vec![0.0f32; n * d];
            opq.apply_batch(&train_data, &mut projected, n);
            projected
        } else {
            train_data
        };

        let km_config = KMeansConfig::default();
        self.quantizer_centroids =
            kmeans::kmeans_train(&km_config, &effective_data, n, d, self.nlist);

        // Retrain PQ on the exact distribution that add/search will encode.
        // For OPQ: opq.train() trained PQ on centered data, but add/search
        // encode uncentered vectors, so we must retrain here for all metrics.
        let pq_train_data = if self.by_residual {
            compute_residuals(&effective_data, n, d, &self.quantizer_centroids, self.nlist)
        } else {
            effective_data
        };
        self.pq.train(&pq_train_data, n);
    }

    /// Add vectors in batches (Faiss-style: batch assign → batch residual → batch encode).
    pub fn add(&mut self, data: &[f32], ids: &[i64], n: usize) {
        const BATCH_SIZE: usize = 32768;
        let mut offset = 0;
        while offset < n {
            let batch_n = (n - offset).min(BATCH_SIZE);
            self.add_batch(
                &data[offset * self.d..(offset + batch_n) * self.d],
                &ids[offset..offset + batch_n],
                batch_n,
            );
            offset += batch_n;
        }
    }

    fn add_batch(&mut self, data: &[f32], ids: &[i64], n: usize) {
        let d = self.d;

        // Step 1: Preprocess (normalize + OPQ rotate)
        let processed = self.preprocess_queries(data, n);

        // Step 2: Batch assign to coarse centroids (uses sgemm)
        let assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                kmeans::find_nearest(
                    &processed[i * d..(i + 1) * d],
                    &self.quantizer_centroids,
                    self.nlist,
                    d,
                )
            })
            .collect();

        // Step 3: Batch compute residuals (parallel)
        let to_encode = if self.by_residual {
            let mut residuals = vec![0.0f32; n * d];
            residuals
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(i, res)| {
                    let list_id = assignments[i];
                    for j in 0..d {
                        res[j] = processed[i * d + j] - self.quantizer_centroids[list_id * d + j];
                    }
                });
            residuals
        } else {
            processed
        };

        // Step 4: Batch PQ encode (parallel)
        let cs = self.pq.code_size();
        let mut codes = vec![0u8; n * cs];
        self.pq.encode_batch(&to_encode, n, &mut codes);

        // Step 5: Distribute to inverted lists
        for i in 0..n {
            let list_id = assignments[i];
            self.ids[list_id].push(ids[i]);
            self.codes[list_id].extend_from_slice(&codes[i * cs..(i + 1) * cs]);
        }

        // Invalidate stale precomputed structures (must rebuild after all adds)
        if !self.fastscan_codes.is_empty() {
            self.fastscan_codes.clear();
        }
        if !self.precomputed_table.is_empty() {
            self.precomputed_table.clear();
        }
    }

    /// Build fastscan block codes for 4-bit search acceleration.
    /// Call after all vectors are added. Lightweight — only reorganizes existing codes.
    pub fn build_search_structures(&mut self) {
        if self.pq.nbits == 4 {
            let cs = self.pq.code_size();
            self.fastscan_codes = self
                .codes
                .iter()
                .enumerate()
                .map(|(list_id, codes)| {
                    let count = self.ids[list_id].len();
                    if count == 0 {
                        Vec::new()
                    } else {
                        crate::fastscan::pack_codes_block_layout(codes, count, cs)
                    }
                })
                .collect();
        }
    }

    /// Build precomputed distance tables for faster repeated queries.
    /// Only useful for long-running services with many queries on the same index.
    /// Costs ~10ms to build and uses nlist * M * ksub * 4 bytes of memory.
    pub fn build_precomputed_table(&mut self) {
        let d = self.d;
        let m = self.pq.m;
        let ksub = self.pq.ksub;
        let nlist = self.nlist;

        if self.metric != MetricType::L2 || !self.by_residual {
            return;
        }
        {
            let pq_norms = self.pq.compute_centroid_norms();
            let mut table = vec![0.0f32; nlist * m * ksub];

            for i in 0..nlist {
                let centroid = &self.quantizer_centroids[i * d..(i + 1) * d];
                let tab_base = i * m * ksub;

                for sub in 0..m {
                    let sub_centroid = &centroid[sub * self.pq.dsub..(sub + 1) * self.pq.dsub];
                    let pq_base = sub * ksub * self.pq.dsub;

                    for j in 0..ksub {
                        let pq_off = pq_base + j * self.pq.dsub;
                        let mut ip = 0.0f32;
                        for dd in 0..self.pq.dsub {
                            ip += sub_centroid[dd] * self.pq.centroids[pq_off + dd];
                        }
                        table[tab_base + sub * ksub + j] = pq_norms[sub * ksub + j] + 2.0 * ip;
                    }
                }
            }
            self.precomputed_table = table;
        }
    }

    /// Search for top-k nearest neighbors.
    /// Uses rayon to parallelize across queries.
    pub fn search(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
        nprobe: usize,
        result_distances: &mut [f32],
        result_labels: &mut [i64],
    ) {
        self.search_with_filter(
            queries,
            nq,
            k,
            nprobe,
            None,
            result_distances,
            result_labels,
        );
    }

    /// Search with optional ID filter.
    pub fn search_with_filter(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
        nprobe: usize,
        filter: Option<&HashSet<i64>>,
        result_distances: &mut [f32],
        result_labels: &mut [i64],
    ) {
        let d = self.d;
        let m = self.pq.m;
        let ksub = self.pq.ksub;

        let processed_queries = self.preprocess_queries(queries, nq);

        let (all_probe_indices, all_coarse_dists) = kmeans::find_topk_batch(
            &processed_queries,
            nq,
            &self.quantizer_centroids,
            self.nlist,
            d,
            nprobe,
        );

        let use_precomputed = !self.precomputed_table.is_empty();
        let use_fastscan = !self.fastscan_codes.is_empty() && self.pq.nbits == 4;

        let results: Vec<Vec<(f32, i64)>> = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let query = &processed_queries[qi * d..(qi + 1) * d];
                let probe_indices = &all_probe_indices[qi];
                let coarse_dists = &all_coarse_dists[qi];

                let mut heap = TopKHeap::new(k);
                let mut sim_table = vec![0.0f32; m * ksub];

                let ip_table = if use_precomputed {
                    let mut t = vec![0.0f32; m * ksub];
                    self.pq.compute_inner_product_table(query, &mut t);
                    t
                } else {
                    Vec::new()
                };

                for (probe_rank, &list_id) in probe_indices.iter().enumerate() {
                    let count = self.ids[list_id].len();
                    if count == 0 {
                        continue;
                    }

                    // Precomputed sim_table omits ||q-c||²; add it as dis0.
                    // Non-precomputed path computes from residual_query, already full distance.
                    let dis0 = if use_precomputed {
                        coarse_dists[probe_rank]
                    } else {
                        0.0
                    };

                    if use_precomputed {
                        let tab_base = list_id * m * ksub;
                        fvec_madd(
                            &self.precomputed_table[tab_base..tab_base + m * ksub],
                            &ip_table,
                            -2.0,
                            &mut sim_table,
                        );
                    } else {
                        self.compute_list_table(query, list_id, &mut sim_table);
                    }

                    if use_fastscan {
                        let mut dists = vec![0.0f32; count];
                        crate::fastscan::fastscan_4bit(
                            &sim_table,
                            &self.fastscan_codes[list_id],
                            count,
                            m,
                            &mut dists,
                        );
                        for i in 0..count {
                            if let Some(f) = filter {
                                if !f.contains(&self.ids[list_id][i]) {
                                    continue;
                                }
                            }
                            heap.push(dis0 + dists[i], self.ids[list_id][i]);
                        }
                    } else if self.pq.nbits == 4 {
                        scan_codes_4bit(
                            &sim_table,
                            &self.codes[list_id],
                            &self.ids[list_id],
                            count,
                            m,
                            ksub,
                            dis0,
                            filter,
                            &mut heap,
                        );
                    } else {
                        scan_codes_batched(
                            &sim_table,
                            &self.codes[list_id],
                            &self.ids[list_id],
                            count,
                            m,
                            ksub,
                            dis0,
                            filter,
                            &mut heap,
                        );
                    }
                }

                heap.into_sorted()
            })
            .collect();

        for (qi, result) in results.into_iter().enumerate() {
            let out_base = qi * k;
            for (i, &(dist, id)) in result.iter().enumerate() {
                result_distances[out_base + i] = dist;
                result_labels[out_base + i] = id;
            }
            for i in result.len()..k {
                result_distances[out_base + i] = f32::MAX;
                result_labels[out_base + i] = -1;
            }
        }
    }

    fn preprocess_queries(&self, queries: &[f32], nq: usize) -> Vec<f32> {
        let d = self.d;
        let mut processed = queries[..nq * d].to_vec();

        if self.metric == MetricType::Cosine {
            for i in 0..nq {
                fvec_normalize(&mut processed[i * d..(i + 1) * d]);
            }
        }

        if let Some(ref opq) = self.opq {
            let mut rotated = vec![0.0f32; nq * d];
            opq.apply_batch(&processed, &mut rotated, nq);
            return rotated;
        }

        processed
    }

    fn compute_list_table(&self, query: &[f32], list_id: usize, sim_table: &mut [f32]) {
        let d = self.d;
        if self.by_residual {
            let mut residual_query = vec![0.0f32; d];
            for j in 0..d {
                residual_query[j] = query[j] - self.quantizer_centroids[list_id * d + j];
            }
            self.pq
                .compute_distance_table(&residual_query, self.metric, sim_table);
        } else {
            self.pq
                .compute_distance_table(query, self.metric, sim_table);
        }
    }

    /// Search with max_codes budget: stop scanning when total scanned codes exceeds limit.
    /// Useful for bounding worst-case latency when some inverted lists are very large.
    pub fn search_with_max_codes(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
        nprobe: usize,
        max_codes: usize,
        result_distances: &mut [f32],
        result_labels: &mut [i64],
    ) {
        let d = self.d;
        let m = self.pq.m;
        let ksub = self.pq.ksub;

        let processed_queries = self.preprocess_queries(queries, nq);
        let (all_probe_indices, all_coarse_dists) = kmeans::find_topk_batch(
            &processed_queries,
            nq,
            &self.quantizer_centroids,
            self.nlist,
            d,
            nprobe,
        );

        let use_precomputed = !self.precomputed_table.is_empty();
        let use_fastscan = !self.fastscan_codes.is_empty() && self.pq.nbits == 4;

        let results: Vec<Vec<(f32, i64)>> = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let query = &processed_queries[qi * d..(qi + 1) * d];
                let probe_indices = &all_probe_indices[qi];
                let coarse_dists = &all_coarse_dists[qi];

                let mut heap = TopKHeap::new(k);
                let mut sim_table = vec![0.0f32; m * ksub];
                let mut total_scanned = 0usize;

                let ip_table = if use_precomputed {
                    let mut t = vec![0.0f32; m * ksub];
                    self.pq.compute_inner_product_table(query, &mut t);
                    t
                } else {
                    Vec::new()
                };

                for (probe_rank, &list_id) in probe_indices.iter().enumerate() {
                    let count = self.ids[list_id].len();
                    if count == 0 {
                        continue;
                    }

                    if total_scanned >= max_codes {
                        break;
                    }
                    let scan_count = count.min(max_codes - total_scanned);

                    let dis0 = if use_precomputed {
                        coarse_dists[probe_rank]
                    } else {
                        0.0
                    };

                    if use_precomputed {
                        let tab_base = list_id * m * ksub;
                        fvec_madd(
                            &self.precomputed_table[tab_base..tab_base + m * ksub],
                            &ip_table,
                            -2.0,
                            &mut sim_table,
                        );
                    } else {
                        self.compute_list_table(query, list_id, &mut sim_table);
                    }

                    if use_fastscan {
                        let mut dists = vec![0.0f32; scan_count];
                        crate::fastscan::fastscan_4bit(
                            &sim_table,
                            &self.fastscan_codes[list_id],
                            scan_count,
                            m,
                            &mut dists,
                        );
                        for i in 0..scan_count {
                            heap.push(dis0 + dists[i], self.ids[list_id][i]);
                        }
                    } else if self.pq.nbits == 4 {
                        scan_codes_4bit(
                            &sim_table,
                            &self.codes[list_id],
                            &self.ids[list_id],
                            scan_count,
                            m,
                            ksub,
                            dis0,
                            None,
                            &mut heap,
                        );
                    } else {
                        scan_codes_batched(
                            &sim_table,
                            &self.codes[list_id],
                            &self.ids[list_id],
                            scan_count,
                            m,
                            ksub,
                            dis0,
                            None,
                            &mut heap,
                        );
                    }

                    total_scanned += scan_count;
                }

                heap.into_sorted()
            })
            .collect();

        for (qi, result) in results.into_iter().enumerate() {
            let out_base = qi * k;
            for (i, &(dist, id)) in result.iter().enumerate() {
                result_distances[out_base + i] = dist;
                result_labels[out_base + i] = id;
            }
            for i in result.len()..k {
                result_distances[out_base + i] = f32::MAX;
                result_labels[out_base + i] = -1;
            }
        }
    }

    /// Merge another index's inverted lists into this one.
    /// Both indexes must have the same centroids and codebooks (trained from the same data).
    /// Used for compaction: merging multiple small index files into one.
    pub fn merge_from(&mut self, other: &IVFPQIndex) {
        assert_eq!(self.d, other.d, "Dimension mismatch");
        assert_eq!(self.nlist, other.nlist, "nlist mismatch");
        assert_eq!(self.pq.m, other.pq.m, "PQ M mismatch");
        assert_eq!(self.pq.nbits, other.pq.nbits, "PQ nbits mismatch");

        for list_id in 0..self.nlist {
            self.ids[list_id].extend_from_slice(&other.ids[list_id]);
            self.codes[list_id].extend_from_slice(&other.codes[list_id]);
        }

        // Invalidate precomputed structures (need to rebuild after merge)
        self.fastscan_codes.clear();
        self.precomputed_table.clear();
    }
}

/// Scan 4-bit packed codes using u8-domain accumulation.
fn scan_codes_4bit(
    sim_table: &[f32],
    codes: &[u8],
    ids: &[i64],
    count: usize,
    m: usize,
    _ksub: usize,
    dis0: f32,
    filter: Option<&HashSet<i64>>,
    heap: &mut TopKHeap,
) {
    let mut dists = vec![0.0f32; count];
    crate::distance::scan_4bit_simd(sim_table, codes, count, m, &mut dists);

    for i in 0..count {
        if let Some(f) = filter {
            if !f.contains(&ids[i]) {
                continue;
            }
        }
        heap.push(dis0 + dists[i], ids[i]);
    }
}

/// Scan 4-bit transposed codes: layout [M/2][n].
/// Each sub-quantizer pair's codes are contiguous — ideal for SIMD.
fn scan_codes_4bit_transposed(
    sim_table: &[f32],
    codes: &[u8],
    ids: &[i64],
    count: usize,
    m: usize,
    dis0: f32,
    filter: Option<&HashSet<i64>>,
    heap: &mut TopKHeap,
) {
    let cs = m / 2;

    const FLAT_NUM: usize = 200;
    let flat_end = count.min(FLAT_NUM);

    let mut dists = vec![0.0f32; count];

    for i in 0..flat_end {
        let mut d = 0.0f32;
        for pair in 0..cs {
            let byte = codes[pair * count + i];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            d += sim_table[(pair * 2) * 16 + lo];
            d += sim_table[(pair * 2 + 1) * 16 + hi];
        }
        dists[i] = d;
    }

    if count > FLAT_NUM {
        let qmin = sim_table.iter().cloned().fold(f32::INFINITY, f32::min);
        let qmax = dists[..flat_end].iter().cloned().fold(f32::MIN, f32::max);
        let range = (qmax - qmin).max(1e-10);
        let factor = 255.0 / range;

        let qtable: Vec<u8> = sim_table
            .iter()
            .map(|&d| ((d - qmin) * factor).clamp(0.0, 255.0) as u8)
            .collect();

        let mut q_dists = vec![0u16; count];
        for pair in 0..cs {
            let qtab_lo = &qtable[(pair * 2) * 16..(pair * 2 + 1) * 16];
            let qtab_hi = &qtable[(pair * 2 + 1) * 16..(pair * 2 + 2) * 16];
            let col = &codes[pair * count..];

            for i in flat_end..count {
                let byte = col[i];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                q_dists[i] += qtab_lo[lo] as u16 + qtab_hi[hi] as u16;
            }
        }

        let inv_factor = range / 255.0;
        let base_dist = qmin * m as f32;
        for i in flat_end..count {
            dists[i] = q_dists[i] as f32 * inv_factor + base_dist;
        }
    }

    for i in 0..count {
        if let Some(f) = filter {
            if !f.contains(&ids[i]) {
                continue;
            }
        }
        heap.push(dis0 + dists[i], ids[i]);
    }
}

/// Scan transposed (column-major) codes: layout is [M][n].
/// The distance table sub-slice stays in L1 cache for the entire inner loop.
fn scan_codes_transposed(
    sim_table: &[f32],
    codes: &[u8],
    ids: &[i64],
    count: usize,
    m: usize,
    ksub: usize,
    dis0: f32,
    filter: Option<&HashSet<i64>>,
    heap: &mut TopKHeap,
) {
    let mut dists = vec![dis0; count];
    for sub in 0..m {
        let tab_base = sub * ksub;
        let col_base = sub * count;
        for i in 0..count {
            dists[i] += sim_table[tab_base + codes[col_base + i] as usize];
        }
    }

    for i in 0..count {
        if let Some(f) = filter {
            if !f.contains(&ids[i]) {
                continue;
            }
        }
        heap.push(dists[i], ids[i]);
    }
}

/// Scan inverted list codes with 4-code batching for ILP (row-major layout).
fn scan_codes_batched(
    sim_table: &[f32],
    codes: &[u8],
    ids: &[i64],
    count: usize,
    m: usize,
    ksub: usize,
    dis0: f32,
    filter: Option<&HashSet<i64>>,
    heap: &mut TopKHeap,
) {
    let mut i = 0;

    while i + 4 <= count {
        let dists = pq_distance_four_codes(
            sim_table,
            codes,
            m,
            ksub,
            [i * m, (i + 1) * m, (i + 2) * m, (i + 3) * m],
        );

        for j in 0..4 {
            let idx = i + j;
            let id = ids[idx];
            if let Some(f) = filter {
                if !f.contains(&id) {
                    continue;
                }
            }
            heap.push(dis0 + dists[j], id);
        }
        i += 4;
    }

    while i < count {
        let code = &codes[i * m..(i + 1) * m];
        let dist = dis0 + pq_distance_from_table(sim_table, code, m, ksub);
        let id = ids[i];
        if let Some(f) = filter {
            if !f.contains(&id) {
                i += 1;
                continue;
            }
        }
        heap.push(dist, id);
        i += 1;
    }
}

struct PreReadList {
    list_id: usize,
    count: usize,
    dis0: f32,
    ids: Vec<i64>,
    codes: Vec<u8>,
}

struct ReaderSearchContext<'a> {
    q: &'a [f32],
    ip_table: &'a [f32],
    use_precomputed: bool,
    filter: Option<&'a HashSet<i64>>,
    d: usize,
    m: usize,
    ksub: usize,
    metric: MetricType,
    by_residual: bool,
    transposed_codes: bool,
    pq: &'a crate::pq::ProductQuantizer,
    quantizer_centroids: &'a [f32],
    precomputed_table: &'a [f32],
}

/// Search using a lazy reader (reads inverted lists on demand).
pub fn search_with_reader<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    query: &[f32],
    k: usize,
    nprobe: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_with_reader_filter(reader, query, k, nprobe, None)
}

/// Search with optional ID filter using a lazy reader.
pub fn search_with_reader_filter<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    query: &[f32],
    k: usize,
    nprobe: usize,
    filter: Option<&HashSet<i64>>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    reader.ensure_loaded()?;
    let d = reader.d;
    if query.len() != d {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "query length {} does not match index dimension {}",
                query.len(),
                d
            ),
        ));
    }
    if k == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "k must be greater than 0",
        ));
    }
    if nprobe == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "nprobe must be greater than 0",
        ));
    }

    let m = reader.m;
    let ksub = reader.ksub;
    let metric = reader.metric;
    let by_residual = reader.by_residual;

    let mut q = query.to_vec();
    if metric == MetricType::Cosine {
        fvec_normalize(&mut q);
    }

    if let Some(ref opq) = reader.opq {
        let mut rotated = vec![0.0f32; d];
        opq.apply(&q, &mut rotated);
        q = rotated;
    }

    let (probe_indices, coarse_dists) =
        kmeans::find_topk(&q, &reader.quantizer_centroids, reader.nlist, d, nprobe);

    let use_precomputed =
        metric == MetricType::L2 && by_residual && !reader.precomputed_table.is_empty();
    let ip_table = if use_precomputed {
        let mut t = vec![0.0f32; m * ksub];
        reader.pq.compute_inner_product_table(&q, &mut t);
        t
    } else {
        Vec::new()
    };

    let mut heap = TopKHeap::new(k);

    if reader.supports_concurrent_pread() {
        // Pre-read all inverted lists upfront so we can scan them in parallel.
        let mut list_data: Vec<PreReadList> = Vec::new();
        for (probe_idx, &list_id) in probe_indices.iter().enumerate() {
            let count = reader.list_counts[list_id] as usize;
            if count == 0 {
                continue;
            }
            let dis0 = if use_precomputed {
                coarse_dists[probe_idx]
            } else {
                0.0
            };
            let (ids, codes) = reader.read_inverted_list(list_id)?;
            list_data.push(PreReadList {
                list_id,
                count,
                dis0,
                ids,
                codes,
            });
        }

        let ctx = ReaderSearchContext {
            q: &q,
            ip_table: &ip_table,
            use_precomputed,
            filter,
            d,
            m,
            ksub,
            metric,
            by_residual,
            transposed_codes: reader.transposed_codes,
            pq: &reader.pq,
            quantizer_centroids: &reader.quantizer_centroids,
            precomputed_table: &reader.precomputed_table,
        };
        let per_list_results: Vec<Vec<(f32, i64)>> = list_data
            .par_iter()
            .map(|entry| {
                let mut local_heap = TopKHeap::new(k);
                scan_reader_list(entry, &ctx, &mut local_heap);
                local_heap.into_sorted()
            })
            .collect();

        for results in per_list_results {
            for (dist, id) in results {
                heap.push(dist, id);
            }
        }
    } else {
        for (probe_idx, &list_id) in probe_indices.iter().enumerate() {
            let count = reader.list_counts[list_id] as usize;
            if count == 0 {
                continue;
            }
            let dis0 = if use_precomputed {
                coarse_dists[probe_idx]
            } else {
                0.0
            };
            let (ids, codes) = reader.read_inverted_list(list_id)?;
            let entry = PreReadList {
                list_id,
                count,
                dis0,
                ids,
                codes,
            };
            let ctx = ReaderSearchContext {
                q: &q,
                ip_table: &ip_table,
                use_precomputed,
                filter,
                d,
                m,
                ksub,
                metric,
                by_residual,
                transposed_codes: reader.transposed_codes,
                pq: &reader.pq,
                quantizer_centroids: &reader.quantizer_centroids,
                precomputed_table: &reader.precomputed_table,
            };
            scan_reader_list(&entry, &ctx, &mut heap);
        }
    }

    let sorted = heap.into_sorted();
    let result_ids: Vec<i64> = sorted.iter().map(|&(_, id)| id).collect();
    let result_dists: Vec<f32> = sorted.iter().map(|&(d, _)| d).collect();

    Ok((result_ids, result_dists))
}

fn scan_reader_list(entry: &PreReadList, ctx: &ReaderSearchContext<'_>, heap: &mut TopKHeap) {
    let d = ctx.d;
    let m = ctx.m;
    let ksub = ctx.ksub;
    let metric = ctx.metric;
    let mut sim_table = vec![0.0f32; m * ksub];

    if ctx.use_precomputed {
        let tab_base = entry.list_id * m * ksub;
        fvec_madd(
            &ctx.precomputed_table[tab_base..tab_base + m * ksub],
            ctx.ip_table,
            -2.0,
            &mut sim_table,
        );
    } else if ctx.by_residual {
        let mut residual_query = vec![0.0f32; d];
        for j in 0..d {
            residual_query[j] = ctx.q[j] - ctx.quantizer_centroids[entry.list_id * d + j];
        }
        ctx.pq
            .compute_distance_table(&residual_query, metric, &mut sim_table);
    } else {
        ctx.pq.compute_distance_table(ctx.q, metric, &mut sim_table);
    }

    let is_4bit = ctx.pq.nbits == 4;
    if is_4bit && ctx.transposed_codes {
        scan_codes_4bit_transposed(
            &sim_table,
            &entry.codes,
            &entry.ids,
            entry.count,
            m,
            entry.dis0,
            ctx.filter,
            heap,
        );
    } else if is_4bit {
        scan_codes_4bit(
            &sim_table,
            &entry.codes,
            &entry.ids,
            entry.count,
            m,
            ksub,
            entry.dis0,
            ctx.filter,
            heap,
        );
    } else if ctx.transposed_codes {
        scan_codes_transposed(
            &sim_table,
            &entry.codes,
            &entry.ids,
            entry.count,
            m,
            ksub,
            entry.dis0,
            ctx.filter,
            heap,
        );
    } else {
        scan_codes_batched(
            &sim_table,
            &entry.codes,
            &entry.ids,
            entry.count,
            m,
            ksub,
            entry.dis0,
            ctx.filter,
            heap,
        );
    }
}

/// Big batch search: batch queries share list reads.
/// Instead of nq*nprobe I/O ops, reads each unique list once and scans for all queries.
pub fn search_batch_reader<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    reader.ensure_loaded()?;
    let d = reader.d;
    if nq == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "nq must be greater than 0",
        ));
    }
    let expected_query_len = nq.checked_mul(d).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "nq * dimension overflows usize",
        )
    })?;
    if queries.len() != expected_query_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "queries length {} does not match nq * dimension {}",
                queries.len(),
                expected_query_len
            ),
        ));
    }
    if k == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "k must be greater than 0",
        ));
    }
    if nprobe == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "nprobe must be greater than 0",
        ));
    }

    let m = reader.m;
    let ksub = reader.ksub;
    let metric = reader.metric;
    let by_residual = reader.by_residual;

    // Step 1: Preprocess all queries
    let mut processed = queries[..nq * d].to_vec();
    if metric == MetricType::Cosine {
        for i in 0..nq {
            fvec_normalize(&mut processed[i * d..(i + 1) * d]);
        }
    }
    if let Some(ref opq) = reader.opq {
        let mut rotated = vec![0.0f32; nq * d];
        opq.apply_batch(&processed, &mut rotated, nq);
        processed = rotated;
    }

    // Step 2: Batch coarse search (one sgemm)
    let (all_probe_indices, all_coarse_dists) = kmeans::find_topk_batch(
        &processed,
        nq,
        &reader.quantizer_centroids,
        reader.nlist,
        d,
        nprobe,
    );

    // Step 3: Group (query_idx, probe_rank) pairs by probed list_id only.
    let mut list_to_queries: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    let mut unique_lists = Vec::new();
    for qi in 0..nq {
        for (rank, &list_id) in all_probe_indices[qi].iter().enumerate() {
            let coarse_dist = all_coarse_dists[qi][rank];
            let entry = list_to_queries.entry(list_id).or_insert_with(|| {
                unique_lists.push(list_id);
                Vec::new()
            });
            entry.push((qi, coarse_dist));
        }
    }

    // Step 4: For each unique list that has queries, read once and scan for all
    let use_precomputed =
        metric == MetricType::L2 && by_residual && !reader.precomputed_table.is_empty();

    let all_ip_tables: Vec<Vec<f32>> = if use_precomputed {
        (0..nq)
            .map(|qi| {
                let mut t = vec![0.0f32; m * ksub];
                reader
                    .pq
                    .compute_inner_product_table(&processed[qi * d..(qi + 1) * d], &mut t);
                t
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut heaps: Vec<TopKHeap> = (0..nq).map(|_| TopKHeap::new(k)).collect();

    for list_id in unique_lists {
        let count = reader.list_counts[list_id] as usize;
        if count == 0 {
            continue;
        }

        // Read list once (shared across all queries that probe it)
        let (ids, codes) = reader.read_inverted_list(list_id)?;
        let mut entry = PreReadList {
            list_id,
            count,
            dis0: 0.0,
            ids,
            codes,
        };

        for &(qi, coarse_dist) in &list_to_queries[&list_id] {
            let query = &processed[qi * d..(qi + 1) * d];
            let dis0 = if use_precomputed { coarse_dist } else { 0.0 };
            let ctx = ReaderSearchContext {
                q: query,
                ip_table: if use_precomputed {
                    &all_ip_tables[qi]
                } else {
                    &[]
                },
                use_precomputed,
                filter: None,
                d,
                m,
                ksub,
                metric,
                by_residual,
                transposed_codes: reader.transposed_codes,
                pq: &reader.pq,
                quantizer_centroids: &reader.quantizer_centroids,
                precomputed_table: &reader.precomputed_table,
            };
            entry.dis0 = dis0;
            scan_reader_list(&entry, &ctx, &mut heaps[qi]);
        }
    }

    // Collect results
    let mut result_ids = vec![-1i64; nq * k];
    let mut result_dists = vec![f32::MAX; nq * k];
    for qi in 0..nq {
        let sorted = std::mem::replace(&mut heaps[qi], TopKHeap::new(0)).into_sorted();
        let base = qi * k;
        for (i, &(dist, id)) in sorted.iter().enumerate() {
            result_ids[base + i] = id;
            result_dists[base + i] = dist;
        }
    }

    Ok((result_ids, result_dists))
}

// --- Top-K Heap ---

struct TopKHeap {
    k: usize,
    data: Vec<(f32, i64)>,
    built: bool,
}

impl TopKHeap {
    fn new(k: usize) -> Self {
        TopKHeap {
            k,
            data: Vec::with_capacity(k),
            built: false,
        }
    }

    #[inline]
    fn push(&mut self, dist: f32, id: i64) {
        if self.k == 0 {
            return;
        }
        if self.data.len() < self.k {
            self.data.push((dist, id));
            if self.data.len() == self.k {
                build_max_heap(&mut self.data);
                self.built = true;
            }
        } else if dist < self.data[0].0 {
            self.data[0] = (dist, id);
            sift_down(&mut self.data, 0);
        }
    }

    fn into_sorted(mut self) -> Vec<(f32, i64)> {
        self.data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.data
    }
}

// --- Utilities ---

fn compute_residuals(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    nlist: usize,
) -> Vec<f32> {
    let mut residuals = vec![0.0f32; n * d];
    for i in 0..n {
        let point = &data[i * d..(i + 1) * d];
        let list_id = kmeans::find_nearest(point, centroids, nlist, d);
        for j in 0..d {
            residuals[i * d + j] = point[j] - centroids[list_id * d + j];
        }
    }
    residuals
}

fn build_max_heap(heap: &mut [(f32, i64)]) {
    let n = heap.len();
    for i in (0..n / 2).rev() {
        sift_down(heap, i);
    }
}

fn sift_down(heap: &mut [(f32, i64)], mut i: usize) {
    let n = heap.len();
    loop {
        let mut largest = i;
        let left = 2 * i + 1;
        let right = 2 * i + 2;

        if left < n && heap[left].0 > heap[largest].0 {
            largest = left;
        }
        if right < n && heap[right].0 > heap[largest].0 {
            largest = right;
        }
        if largest == i {
            break;
        }
        heap.swap(i, largest);
        i = largest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::SeekRead;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct ReaderStats {
        pread_calls: usize,
        max_pread_len: usize,
    }

    struct NonConcurrentPreadCursor {
        inner: Cursor<Vec<u8>>,
        stats: Arc<Mutex<ReaderStats>>,
    }

    impl NonConcurrentPreadCursor {
        fn new(data: Vec<u8>, stats: Arc<Mutex<ReaderStats>>) -> Self {
            NonConcurrentPreadCursor {
                inner: Cursor::new(data),
                stats,
            }
        }
    }

    impl SeekRead for NonConcurrentPreadCursor {
        fn seek(&mut self, pos: u64) -> io::Result<()> {
            io::Seek::seek(&mut self.inner, io::SeekFrom::Start(pos))?;
            Ok(())
        }

        fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
            io::Read::read_exact(&mut self.inner, buf)
        }

        fn pread(&mut self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
            {
                let mut stats = self.stats.lock().unwrap();
                stats.pread_calls += 1;
                stats.max_pread_len = stats.max_pread_len.max(buf.len());
            }
            io::Seek::seek(&mut self.inner, io::SeekFrom::Start(pos))?;
            io::Read::read_exact(&mut self.inner, buf)
        }
    }

    fn generate_clustered_data(n: usize, d: usize, num_clusters: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut centers = vec![0.0f32; num_clusters * d];
        for i in 0..num_clusters * d {
            centers[i] = rng.gen::<f32>() * 100.0;
        }

        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            let cluster = i % num_clusters;
            for j in 0..d {
                data[i * d + j] = centers[cluster * d + j] + rng.gen::<f32>() * 2.0 - 1.0;
            }
        }
        data
    }

    #[test]
    fn test_build_and_search_l2() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;
        let k = 5;
        let nprobe = 2;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let query = &data[0..d];
        let mut dists = vec![0.0f32; k];
        let mut labels = vec![0i64; k];
        index.search(query, 1, k, nprobe, &mut dists, &mut labels);

        assert_eq!(labels[0], 0);
        for i in 1..k {
            assert!(dists[i] >= dists[i - 1]);
        }
    }

    #[test]
    fn test_build_and_search_ip() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;

        let data = generate_clustered_data(n, d, 4, 123);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::InnerProduct, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut dists = vec![0.0f32; 5];
        let mut labels = vec![0i64; 5];
        index.search(&data[0..d], 1, 5, 2, &mut dists, &mut labels);

        for i in 1..5 {
            assert!(dists[i] >= dists[i - 1]);
        }
    }

    #[test]
    fn test_search_with_filter() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;
        let k = 5;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let filter: HashSet<i64> = (0..n as i64).filter(|id| id % 2 == 0).collect();
        let mut dists = vec![0.0f32; k];
        let mut labels = vec![0i64; k];
        index.search_with_filter(&data[0..d], 1, k, 4, Some(&filter), &mut dists, &mut labels);

        for &label in &labels[..k] {
            if label >= 0 {
                assert!(label % 2 == 0, "Filter violated: got odd ID {}", label);
            }
        }
    }

    #[test]
    fn test_batch_search() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;
        let k = 5;
        let nq = 10;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let queries: Vec<f32> = data[..nq * d].to_vec();
        let mut dists = vec![0.0f32; nq * k];
        let mut labels = vec![0i64; nq * k];
        index.search(&queries, nq, k, 2, &mut dists, &mut labels);

        for qi in 0..nq {
            assert_eq!(labels[qi * k], qi as i64);
        }
    }

    #[test]
    fn test_4bit_ivfpq() {
        let d = 16;
        let nlist = 4;
        let m = 8;
        let n = 1000;
        let k = 5;
        let nprobe = 2;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::with_nbits(d, nlist, m, 4, MetricType::L2, false);
        assert_eq!(index.pq.ksub, 16);
        assert_eq!(index.pq.code_size(), 4);

        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut dists = vec![0.0f32; k];
        let mut labels = vec![0i64; k];
        index.search(&data[0..d], 1, k, nprobe, &mut dists, &mut labels);

        assert_eq!(labels[0], 0);
        for i in 1..k {
            assert!(dists[i] >= dists[i - 1]);
        }

        let codes_8bit_size = n * m;
        let codes_4bit_size: usize = index.codes.iter().map(|c| c.len()).sum();
        assert!(
            codes_4bit_size < codes_8bit_size,
            "4-bit ({}) should be smaller than 8-bit ({})",
            codes_4bit_size,
            codes_8bit_size,
        );
    }

    #[test]
    fn test_max_codes_early_termination() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;
        let k = 5;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut dists_limited = vec![0.0f32; k];
        let mut labels_limited = vec![0i64; k];
        index.search_with_max_codes(
            &data[0..d],
            1,
            k,
            4,
            50,
            &mut dists_limited,
            &mut labels_limited,
        );

        let valid = labels_limited.iter().filter(|&&id| id >= 0).count();
        assert!(valid > 0, "max_codes search returned no results");

        let mut dists_full = vec![0.0f32; k];
        let mut labels_full = vec![0i64; k];
        index.search(&data[0..d], 1, k, 4, &mut dists_full, &mut labels_full);

        assert!(dists_full[0] <= dists_limited[0] + 1e-6);
    }

    #[test]
    fn test_from_trained_and_merge() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;

        let data = generate_clustered_data(n * 2, d, 4, 42);
        let ids_a: Vec<i64> = (0..n as i64).collect();
        let ids_b: Vec<i64> = (n as i64..2 * n as i64).collect();

        let mut trainer = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        trainer.train(&data[..n * d], n);

        let mut worker_a = IVFPQIndex::from_trained(&trainer);
        worker_a.add(&data[..n * d], &ids_a, n);

        let mut worker_b = IVFPQIndex::from_trained(&trainer);
        worker_b.add(&data[n * d..], &ids_b, n);

        let total_a: usize = worker_a.ids.iter().map(|l| l.len()).sum();
        let total_b: usize = worker_b.ids.iter().map(|l| l.len()).sum();
        assert_eq!(total_a + total_b, n * 2);

        let mut merged = IVFPQIndex::from_trained(&trainer);
        merged.merge_from(&worker_a);
        merged.merge_from(&worker_b);

        let total_merged: usize = merged.ids.iter().map(|l| l.len()).sum();
        assert_eq!(total_merged, n * 2);

        let mut dists = vec![0.0f32; 5];
        let mut labels = vec![0i64; 5];
        merged.search(&data[0..d], 1, 5, 4, &mut dists, &mut labels);
        assert_eq!(labels[0], 0);

        merged.search(&data[n * d..(n + 1) * d], 1, 5, 4, &mut dists, &mut labels);
        assert_eq!(labels[0], n as i64);
    }

    #[test]
    fn test_opq_ip() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;
        let k = 5;

        let data = generate_clustered_data(n, d, 4, 55);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::InnerProduct, true);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut dists = vec![0.0f32; k];
        let mut labels = vec![0i64; k];
        index.search(&data[0..d], 1, k, 4, &mut dists, &mut labels);

        let valid = labels.iter().filter(|&&id| id >= 0).count();
        assert!(valid > 0, "OPQ+IP should return results");
        for i in 1..valid {
            assert!(dists[i] >= dists[i - 1]);
        }
    }

    #[test]
    fn test_opq_cosine() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;
        let k = 5;

        let data = generate_clustered_data(n, d, 4, 77);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::Cosine, true);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut dists = vec![0.0f32; k];
        let mut labels = vec![0i64; k];
        index.search(&data[0..d], 1, k, 4, &mut dists, &mut labels);

        let valid = labels.iter().filter(|&&id| id >= 0).count();
        assert!(valid > 0, "OPQ+Cosine should return results");
        for i in 1..valid {
            assert!(dists[i] >= dists[i - 1]);
        }
    }

    #[test]
    fn test_opq_4bit() {
        let d = 16;
        let nlist = 4;
        let m = 8;
        let n = 1000;
        let k = 5;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::with_nbits(d, nlist, m, 4, MetricType::L2, true);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut dists = vec![0.0f32; k];
        let mut labels = vec![0i64; k];
        index.search(&data[0..d], 1, k, 4, &mut dists, &mut labels);

        assert_eq!(labels[0], 0, "OPQ+4bit should recall query vector itself");
        for i in 1..k {
            assert!(dists[i] >= dists[i - 1]);
        }
    }

    #[test]
    fn test_precomputed_table_matches_normal_search() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;
        let k = 10;
        let nprobe = 4;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        // Normal search
        let mut dists_normal = vec![0.0f32; k];
        let mut labels_normal = vec![0i64; k];
        index.search(
            &data[0..d],
            1,
            k,
            nprobe,
            &mut dists_normal,
            &mut labels_normal,
        );

        // Enable precomputed table and search again
        index.build_precomputed_table();
        let mut dists_precomp = vec![0.0f32; k];
        let mut labels_precomp = vec![0i64; k];
        index.search(
            &data[0..d],
            1,
            k,
            nprobe,
            &mut dists_precomp,
            &mut labels_precomp,
        );

        // Same top-k ranking
        assert_eq!(
            labels_normal, labels_precomp,
            "precomputed table should produce identical ranking"
        );
        for i in 0..k {
            assert!(
                (dists_normal[i] - dists_precomp[i]).abs() < 1e-2,
                "distance mismatch at rank {}: normal={}, precomp={}",
                i,
                dists_normal[i],
                dists_precomp[i]
            );
        }
    }

    #[test]
    fn test_fastscan_invalidated_after_add() {
        let d = 16;
        let nlist = 4;
        let m = 8;
        let n = 500;
        let k = 5;

        let data = generate_clustered_data(n * 2, d, 4, 42);
        let ids_a: Vec<i64> = (0..n as i64).collect();
        let ids_b: Vec<i64> = (n as i64..2 * n as i64).collect();

        let mut index = IVFPQIndex::with_nbits(d, nlist, m, 4, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data[..n * d], &ids_a, n);

        // Build fastscan, then add more vectors
        index.build_search_structures();
        assert!(!index.fastscan_codes.is_empty());

        index.add(&data[n * d..], &ids_b, n);
        assert!(
            index.fastscan_codes.is_empty(),
            "fastscan_codes must be cleared after add()"
        );

        // Rebuild and search — should find vectors from both batches
        index.build_search_structures();
        let mut dists = vec![0.0f32; k];
        let mut labels = vec![0i64; k];
        index.search(&data[0..d], 1, k, 4, &mut dists, &mut labels);
        assert_eq!(labels[0], 0);

        index.search(&data[n * d..(n + 1) * d], 1, k, 4, &mut dists, &mut labels);
        assert_eq!(labels[0], n as i64);
    }

    #[test]
    fn test_precomputed_table_invalidated_after_add() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;

        let data = generate_clustered_data(n * 2, d, 4, 42);
        let ids_a: Vec<i64> = (0..n as i64).collect();
        let ids_b: Vec<i64> = (n as i64..2 * n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data[..n * d], n);
        index.add(&data[..n * d], &ids_a, n);

        index.build_precomputed_table();
        assert!(!index.precomputed_table.is_empty());

        index.add(&data[n * d..], &ids_b, n);
        assert!(
            index.precomputed_table.is_empty(),
            "precomputed_table must be cleared after add()"
        );

        // Rebuild and search — should find vectors from both batches
        index.build_precomputed_table();
        let k = 5;
        let mut dists = vec![0.0f32; k];
        let mut labels = vec![0i64; k];
        index.search(&data[0..d], 1, k, 4, &mut dists, &mut labels);
        assert_eq!(labels[0], 0);

        index.search(&data[n * d..(n + 1) * d], 1, k, 4, &mut dists, &mut labels);
        assert_eq!(labels[0], n as i64);
    }

    #[test]
    fn test_write_read_search() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;
        let k = 10;

        let data = generate_clustered_data(n, d, 4, 789);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let mut cursor = Cursor::new(buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();

        let (result_ids, result_dists) = reader.search(&data[0..d], k, 4).unwrap();

        assert!(!result_ids.is_empty());
        assert!(result_ids.contains(&0));
        for i in 1..result_dists.len() {
            assert!(result_dists[i] >= result_dists[i - 1]);
        }
    }

    #[test]
    fn test_reader_search_works_without_concurrent_pread() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 8;
        let m = 4;
        let n = 800;
        let k = 5;
        let nprobe = 4;

        let data = generate_clustered_data(n, d, 8, 789);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let mut baseline_reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let (baseline_ids, baseline_dists) =
            baseline_reader.search(&data[0..d], k, nprobe).unwrap();

        let stats = Arc::new(Mutex::new(ReaderStats::default()));
        let stream = NonConcurrentPreadCursor::new(buf, Arc::clone(&stats));
        let mut reader = IVFPQIndexReader::open(stream).unwrap();
        assert!(!reader.supports_concurrent_pread());

        let (ids, dists) = reader.search(&data[0..d], k, nprobe).unwrap();

        assert_eq!(ids, baseline_ids);
        assert_eq!(dists, baseline_dists);
        assert!(
            stats.lock().unwrap().pread_calls > 0,
            "search should still read inverted lists through pread fallback"
        );
    }

    #[test]
    fn test_reader_search_validates_inputs() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;

        let data = generate_clustered_data(n, d, 4, 789);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let mut reader = IVFPQIndexReader::open(Cursor::new(buf)).unwrap();

        let err = reader.search(&data[0..d - 1], 5, 2).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let err = reader.search(&data[0..d + 1], 5, 2).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let err = reader.search(&data[0..d], 0, 2).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let err = reader.search(&data[0..d], 5, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_write_read_search_with_filter() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;
        let k = 5;

        let data = generate_clustered_data(n, d, 4, 789);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let mut cursor = Cursor::new(buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();

        let filter: HashSet<i64> = (0..n as i64).filter(|id| id % 3 == 0).collect();
        let (result_ids, _) =
            search_with_reader_filter(&mut reader, &data[0..d], k, 4, Some(&filter)).unwrap();

        for &id in &result_ids {
            assert!(id % 3 == 0, "Filter violated: got ID {}", id);
        }
    }

    #[test]
    fn test_big_batch_search() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};
        use std::io::Cursor;

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 1000;
        let k = 5;
        let nq = 20;
        let nprobe = 2;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let mut cursor = Cursor::new(&buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();

        let queries = &data[..nq * d];
        let (batch_ids, batch_dists) =
            search_batch_reader(&mut reader, queries, nq, k, nprobe).unwrap();

        for qi in 0..nq {
            let base = qi * k;
            assert_eq!(batch_ids[base], qi as i64);
            for i in 1..k {
                if batch_ids[base + i] >= 0 {
                    assert!(batch_dists[base + i] >= batch_dists[base + i - 1]);
                }
            }
        }
    }

    #[test]
    fn test_batch_reader_matches_single_reader_search() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};
        use std::io::Cursor;

        let d = 16;
        let nlist = 8;
        let m = 4;
        let n = 1000;
        let k = 5;
        let nq = 12;
        let nprobe = 3;

        let data = generate_clustered_data(n, d, 8, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let queries = &data[..nq * d];
        let mut batch_reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let (batch_ids, batch_dists) =
            search_batch_reader(&mut batch_reader, queries, nq, k, nprobe).unwrap();

        for qi in 0..nq {
            let mut single_reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
            let query = &queries[qi * d..(qi + 1) * d];
            let (single_ids, single_dists) = single_reader.search(query, k, nprobe).unwrap();
            let base = qi * k;

            assert_eq!(&batch_ids[base..base + k], &single_ids[..]);
            assert_eq!(&batch_dists[base..base + k], &single_dists[..]);
        }
    }

    #[test]
    fn test_batch_reader_validates_inputs() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};
        use std::io::Cursor;

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;
        let nq = 4;
        let k = 5;
        let nprobe = 2;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let queries = &data[..nq * d];

        let mut reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let err = search_batch_reader(&mut reader, &queries[..queries.len() - 1], nq, k, nprobe)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let mut longer_queries = queries.to_vec();
        longer_queries.push(0.0);
        let mut reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let err = search_batch_reader(&mut reader, &longer_queries, nq, k, nprobe).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let mut reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let err = search_batch_reader(&mut reader, queries, 0, k, nprobe).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let mut reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let err = search_batch_reader(&mut reader, queries, nq, 0, nprobe).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let mut reader = IVFPQIndexReader::open(Cursor::new(buf)).unwrap();
        let err = search_batch_reader(&mut reader, queries, nq, k, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
