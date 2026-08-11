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
    fvec_inner_product, fvec_madd, fvec_normalize, pq_distance_four_codes, pq_distance_from_table,
    MetricType,
};
use crate::index_io_util::ivf_payload_is_oversized;
use crate::io::{IVFPQIndexReader, InvertedListPayload, SeekRead};
use crate::kmeans::{self, KMeansConfig};
use crate::logging::{emit_log, LogLevel};
use crate::opq::OPQMatrix;
use crate::pq::ProductQuantizer;
use rayon::prelude::*;
use roaring::RoaringTreemap;
use std::borrow::Cow;
use std::collections::HashSet;
use std::io;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub trait RowIdFilter: Sync {
    fn contains(&self, id: i64) -> bool;
}

impl RowIdFilter for HashSet<i64> {
    fn contains(&self, id: i64) -> bool {
        HashSet::contains(self, &id)
    }
}

impl RowIdFilter for RoaringTreemap {
    fn contains(&self, id: i64) -> bool {
        id >= 0 && RoaringTreemap::contains(self, id as u64)
    }
}

fn decode_roaring_filter(bytes: &[u8]) -> io::Result<RoaringTreemap> {
    RoaringTreemap::deserialize_from(bytes).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid RoaringTreemap filter: {}", e),
        )
    })
}

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
        assert!(
            nbits != 4 || m.is_multiple_of(2),
            "4-bit IVF-PQ requires even m, got {m}"
        );
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
                chunk_offsets: trained.pq.chunk_offsets.clone(),
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

        // L2/IP without OPQ borrows the caller's batch instead of copying it.
        let processed = self.preprocess_queries(data, n);
        let assignments =
            kmeans::find_nearest_batch(&processed, n, &self.quantizer_centroids, self.nlist, d);

        let to_encode = if self.by_residual {
            let mut residuals = vec![0.0f32; n * d];
            residuals
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(i, res)| {
                    let list_id = assignments[i];
                    fvec_madd(
                        &processed[i * d..(i + 1) * d],
                        &self.quantizer_centroids[list_id * d..(list_id + 1) * d],
                        -1.0,
                        res,
                    );
                });
            Cow::Owned(residuals)
        } else {
            processed
        };

        let code_size = self.pq.code_size();
        let mut codes = vec![0u8; n * code_size];
        self.pq.encode_batch(&to_encode, n, &mut codes);

        for i in 0..n {
            let list_id = assignments[i];
            self.ids[list_id].push(ids[i]);
            self.codes[list_id].extend_from_slice(&codes[i * code_size..(i + 1) * code_size]);
        }

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

            table
                .par_chunks_mut(m * ksub)
                .enumerate()
                .for_each(|(i, list_table)| {
                    let centroid = &self.quantizer_centroids[i * d..(i + 1) * d];
                    for sub in 0..m {
                        let sub_centroid = &centroid[sub * self.pq.dsub..(sub + 1) * self.pq.dsub];
                        let pq_base = sub * ksub * self.pq.dsub;

                        for j in 0..ksub {
                            let pq_off = pq_base + j * self.pq.dsub;
                            let ip = fvec_inner_product(
                                sub_centroid,
                                &self.pq.centroids[pq_off..pq_off + self.pq.dsub],
                            );
                            list_table[sub * ksub + j] = pq_norms[sub * ksub + j] + 2.0 * ip;
                        }
                    }
                });
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
        filter: Option<&dyn RowIdFilter>,
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
        let matching_rows_by_list = filter.map(|filter| {
            let mut probed_lists = vec![false; self.nlist];
            for probe_indices in &all_probe_indices {
                for &list_id in probe_indices {
                    probed_lists[list_id] = true;
                }
            }
            self.ids
                .iter()
                .zip(probed_lists)
                .map(|(ids, probed)| {
                    if probed {
                        matching_rows(ids, Some(filter)).unwrap()
                    } else {
                        MatchingRows::Sparse(Vec::new())
                    }
                })
                .collect::<Vec<_>>()
        });

        let results: Vec<Vec<(f32, i64)>> = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let query = &processed_queries[qi * d..(qi + 1) * d];
                let probe_indices = &all_probe_indices[qi];
                let coarse_dists = &all_coarse_dists[qi];

                let mut heap = TopKHeap::new(k);
                let mut sim_table = Vec::new();
                let mut non_residual_table_ready = false;
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
                    let matching_rows = matching_rows_by_list.as_ref().map(|rows| &rows[list_id]);
                    if matching_rows.is_some_and(MatchingRows::is_empty) {
                        continue;
                    }

                    if sim_table.is_empty() {
                        sim_table.resize(m * ksub, 0.0);
                    }
                    if !self.by_residual && !non_residual_table_ready {
                        self.pq
                            .compute_distance_table(query, self.metric, &mut sim_table);
                        non_residual_table_ready = true;
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
                    } else if self.by_residual {
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
                        if let Some(rows) = matching_rows {
                            for position in rows.positions() {
                                heap.push(dis0 + dists[position], self.ids[list_id][position]);
                            }
                        } else {
                            for i in 0..count {
                                heap.push(dis0 + dists[i], self.ids[list_id][i]);
                            }
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
                            matching_rows,
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
                            matching_rows,
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

    fn preprocess_queries<'a>(&self, queries: &'a [f32], nq: usize) -> Cow<'a, [f32]> {
        let d = self.d;
        let processed = match self.metric {
            MetricType::Cosine => {
                let mut normalized = queries[..nq * d].to_vec();
                for vector in normalized.chunks_exact_mut(d) {
                    fvec_normalize(vector);
                }
                Cow::Owned(normalized)
            }
            MetricType::L2 | MetricType::InnerProduct => Cow::Borrowed(&queries[..nq * d]),
        };

        if let Some(ref opq) = self.opq {
            let mut rotated = vec![0.0f32; nq * d];
            opq.apply_batch(&processed, &mut rotated, nq);
            return Cow::Owned(rotated);
        }

        processed
    }

    fn compute_list_table(&self, query: &[f32], list_id: usize, sim_table: &mut [f32]) {
        let d = self.d;
        if self.by_residual {
            let mut residual_query = vec![0.0f32; d];
            fvec_madd(
                query,
                &self.quantizer_centroids[list_id * d..(list_id + 1) * d],
                -1.0,
                &mut residual_query,
            );
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
    /// Both indexes must have identical training state: metric, residual mode,
    /// OPQ rotation, coarse centroids, and PQ codebooks.
    /// Used for compaction: merging multiple small index files into one.
    pub fn merge_from(&mut self, other: &IVFPQIndex) -> io::Result<()> {
        self.ensure_merge_compatible(other)?;

        for list_id in 0..self.nlist {
            self.ids[list_id].extend_from_slice(&other.ids[list_id]);
            self.codes[list_id].extend_from_slice(&other.codes[list_id]);
        }

        // Invalidate precomputed structures (need to rebuild after merge)
        self.fastscan_codes.clear();
        self.precomputed_table.clear();
        Ok(())
    }

    fn ensure_merge_compatible(&self, other: &IVFPQIndex) -> io::Result<()> {
        if self.d != other.d {
            return Err(invalid_merge_input(format!(
                "dimension mismatch: self={}, other={}",
                self.d, other.d
            )));
        }
        if self.nlist != other.nlist {
            return Err(invalid_merge_input(format!(
                "nlist mismatch: self={}, other={}",
                self.nlist, other.nlist
            )));
        }
        if self.metric != other.metric {
            return Err(invalid_merge_input(format!(
                "metric mismatch: self={:?}, other={:?}",
                self.metric, other.metric
            )));
        }
        if self.by_residual != other.by_residual {
            return Err(invalid_merge_input(format!(
                "residual mode mismatch: self={}, other={}",
                self.by_residual, other.by_residual
            )));
        }
        if self.pq.d != other.pq.d
            || self.pq.m != other.pq.m
            || self.pq.nbits != other.pq.nbits
            || self.pq.dsub != other.pq.dsub
            || self.pq.ksub != other.pq.ksub
        {
            return Err(invalid_merge_input(format!(
                "PQ layout mismatch: self=(d={}, m={}, nbits={}, dsub={}, ksub={}), other=(d={}, m={}, nbits={}, dsub={}, ksub={})",
                self.pq.d,
                self.pq.m,
                self.pq.nbits,
                self.pq.dsub,
                self.pq.ksub,
                other.pq.d,
                other.pq.m,
                other.pq.nbits,
                other.pq.dsub,
                other.pq.ksub
            )));
        }
        if self.opq.is_some() != other.opq.is_some() {
            return Err(invalid_merge_input("OPQ configuration mismatch"));
        }
        if let (Some(self_opq), Some(other_opq)) = (&self.opq, &other.opq) {
            if self_opq.d != other_opq.d || self_opq.m != other_opq.m {
                return Err(invalid_merge_input(format!(
                    "OPQ layout mismatch: self=(d={}, m={}), other=(d={}, m={})",
                    self_opq.d, self_opq.m, other_opq.d, other_opq.m
                )));
            }
            if self_opq.rotation != other_opq.rotation {
                return Err(invalid_merge_input("OPQ rotation mismatch"));
            }
        }
        if self.quantizer_centroids != other.quantizer_centroids {
            return Err(invalid_merge_input("coarse centroids mismatch"));
        }
        if self.pq.centroids != other.pq.centroids {
            return Err(invalid_merge_input("PQ codebooks mismatch"));
        }

        Ok(())
    }
}

fn invalid_merge_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

enum MatchingRows {
    Sparse(Vec<usize>),
    Bitmap { words: Vec<u64>, len: usize },
}

impl MatchingRows {
    fn len(&self) -> usize {
        match self {
            Self::Sparse(positions) => positions.len(),
            Self::Bitmap { len, .. } => *len,
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn contains(&self, position: usize) -> bool {
        match self {
            Self::Sparse(positions) => positions.binary_search(&position).is_ok(),
            Self::Bitmap { words, .. } => words
                .get(position / 64)
                .is_some_and(|word| word & (1u64 << (position % 64)) != 0),
        }
    }

    fn positions(&self) -> MatchingRowIter<'_> {
        match self {
            Self::Sparse(positions) => MatchingRowIter::Sparse {
                positions,
                index: 0,
            },
            Self::Bitmap { words, .. } => MatchingRowIter::Bitmap {
                words,
                word_index: 0,
                word: 0,
                word_base: 0,
            },
        }
    }

    #[cfg(test)]
    fn storage_bytes(&self) -> usize {
        match self {
            Self::Sparse(positions) => positions.capacity() * std::mem::size_of::<usize>(),
            Self::Bitmap { words, .. } => words.capacity() * std::mem::size_of::<u64>(),
        }
    }
}

enum MatchingRowIter<'a> {
    Sparse {
        positions: &'a [usize],
        index: usize,
    },
    Bitmap {
        words: &'a [u64],
        word_index: usize,
        word: u64,
        word_base: usize,
    },
}

impl Iterator for MatchingRowIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Sparse { positions, index } => {
                let position = positions.get(*index).copied();
                *index += usize::from(position.is_some());
                position
            }
            Self::Bitmap {
                words,
                word_index,
                word,
                word_base,
            } => loop {
                if *word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    *word &= *word - 1;
                    return Some(*word_base + bit);
                }
                let next_word = words.get(*word_index).copied()?;
                *word = next_word;
                *word_base = *word_index * 64;
                *word_index += 1;
            },
        }
    }
}

fn matching_rows(ids: &[i64], filter: Option<&dyn RowIdFilter>) -> Option<MatchingRows> {
    filter.map(|filter| {
        let bitmap_words = ids.len().div_ceil(64);
        let sparse_limit =
            bitmap_words.saturating_mul(std::mem::size_of::<u64>()) / std::mem::size_of::<usize>();
        let mut positions = Vec::new();
        let mut bitmap = None::<Vec<u64>>;
        let mut matching_count = 0usize;

        for (position, &id) in ids.iter().enumerate() {
            if !filter.contains(id) {
                continue;
            }
            matching_count += 1;
            if let Some(words) = bitmap.as_mut() {
                words[position / 64] |= 1u64 << (position % 64);
            } else if positions.len() < sparse_limit {
                positions.push(position);
            } else {
                let mut words = vec![0u64; bitmap_words];
                for previous in positions.drain(..) {
                    words[previous / 64] |= 1u64 << (previous % 64);
                }
                words[position / 64] |= 1u64 << (position % 64);
                bitmap = Some(words);
            }
        }

        match bitmap {
            Some(words) => MatchingRows::Bitmap {
                words,
                len: matching_count,
            },
            None => MatchingRows::Sparse(positions),
        }
    })
}

// Sparse row-major scans give up four-code ILP, while sparse transposed scans
// replace sequential column reads with random row lookups. Keep separate,
// conservative crossover points instead of applying one threshold to every
// kernel. Packed 4-bit and FastScan paths retain their normal distance kernel
// to preserve score semantics.
const ROW_MAJOR_SPARSE_SCAN_DIVISOR: usize = 4;
const TRANSPOSED_SPARSE_SCAN_DIVISOR: usize = 8;

fn should_scan_sparse(count: usize, matching_rows: &MatchingRows, divisor: usize) -> bool {
    matching_rows.len().saturating_mul(divisor) <= count
}

fn has_matching_rows(matching_rows: Option<&MatchingRows>) -> bool {
    match matching_rows {
        Some(rows) => !rows.is_empty(),
        None => true,
    }
}

// Below this size, table construction and Rayon scheduling dominate the saved
// per-query/list distance-table work.
const MIN_EPHEMERAL_PRECOMPUTE_QUERIES: usize = 64;
pub const DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IvfPqBatchTableReuseMode {
    Off = 0,
    On = 1,
    Auto = 2,
}

fn should_use_ephemeral_precomputation(
    matching_list_count: usize,
    active_query_count: usize,
    probe_count: usize,
) -> bool {
    let setup_tables = matching_list_count.saturating_add(active_query_count);
    // Require at least 2x reuse over the list-table and query-table setup work.
    setup_tables > 0 && probe_count >= setup_tables.saturating_mul(2)
}

fn ephemeral_precomputed_table_fits_budget(
    matching_list_count: usize,
    query_scratch_count: usize,
    m: usize,
    ksub: usize,
    max_bytes: usize,
) -> bool {
    if matching_list_count == 0 {
        return false;
    }
    matching_list_count
        .checked_add(1)
        .and_then(|tables| tables.checked_add(query_scratch_count))
        .and_then(|tables| tables.checked_mul(m))
        .and_then(|values| values.checked_mul(ksub))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
        .is_some_and(|bytes| bytes <= max_bytes)
}

#[cfg(test)]
fn fill_list_precomputed_table(
    coarse_centroid: &[f32],
    pq: &ProductQuantizer,
    pq_norms: &[f32],
    table: &mut Vec<f32>,
) {
    debug_assert_eq!(coarse_centroid.len(), pq.d);
    debug_assert_eq!(pq_norms.len(), pq.m * pq.ksub);
    table.resize(pq.m * pq.ksub, 0.0);
    for sub in 0..pq.m {
        let range = pq.chunk_range(sub);
        let chunk_dim = range.len();
        let pq_base = range.start * pq.ksub;
        for code in 0..pq.ksub {
            let pq_offset = pq_base + code * chunk_dim;
            let mut inner_product = 0.0f32;
            for dimension in 0..chunk_dim {
                inner_product +=
                    coarse_centroid[range.start + dimension] * pq.centroids[pq_offset + dimension];
            }
            let table_offset = sub * pq.ksub + code;
            table[table_offset] = pq_norms[table_offset] + 2.0 * inner_product;
        }
    }
}

fn compute_stable_ephemeral_pq_norms(pq: &ProductQuantizer) -> Vec<f64> {
    let mut norms = vec![0.0f64; pq.m * pq.ksub];
    for sub in 0..pq.m {
        let range = pq.chunk_range(sub);
        let chunk_dim = range.len();
        let pq_base = range.start * pq.ksub;
        for code in 0..pq.ksub {
            let pq_offset = pq_base + code * chunk_dim;
            norms[sub * pq.ksub + code] = (0..chunk_dim)
                .map(|dimension| {
                    let value = f64::from(pq.centroids[pq_offset + dimension]);
                    value * value
                })
                .sum();
        }
    }
    norms
}

fn fill_stable_ephemeral_list_table(
    coarse_centroid: &[f32],
    pq: &ProductQuantizer,
    pq_norms: &[f64],
    table: &mut Vec<f64>,
) {
    table.resize(pq.m * pq.ksub, 0.0);
    for sub in 0..pq.m {
        let range = pq.chunk_range(sub);
        let chunk_dim = range.len();
        let pq_base = range.start * pq.ksub;
        for code in 0..pq.ksub {
            let pq_offset = pq_base + code * chunk_dim;
            let mut inner_product = 0.0f64;
            for dimension in 0..chunk_dim {
                let pq_value = f64::from(pq.centroids[pq_offset + dimension]);
                inner_product += f64::from(coarse_centroid[range.start + dimension]) * pq_value;
            }
            let offset = sub * pq.ksub + code;
            table[offset] = pq_norms[offset] + 2.0 * inner_product;
        }
    }
}

fn fill_stable_ephemeral_query_table(query: &[f32], pq: &ProductQuantizer, table: &mut Vec<f64>) {
    table.resize(pq.m * pq.ksub, 0.0);
    for sub in 0..pq.m {
        let range = pq.chunk_range(sub);
        let chunk_dim = range.len();
        let pq_base = range.start * pq.ksub;
        for code in 0..pq.ksub {
            let pq_offset = pq_base + code * chunk_dim;
            let mut inner_product = 0.0f64;
            for dimension in 0..chunk_dim {
                inner_product += f64::from(query[range.start + dimension])
                    * f64::from(pq.centroids[pq_offset + dimension]);
            }
            table[sub * pq.ksub + code] = inner_product;
        }
    }
}

fn combine_stable_ephemeral_tables(
    list_table: &[f64],
    query_table: &[f64],
    query: &[f32],
    coarse_centroid: &[f32],
    pq: &ProductQuantizer,
    sim_table: &mut Vec<f32>,
) {
    sim_table.resize(pq.m * pq.ksub, 0.0);
    for sub in 0..pq.m {
        let range = pq.chunk_range(sub);
        let mut residual_norm = 0.0f64;
        for dimension in range {
            let residual = f64::from(query[dimension]) - f64::from(coarse_centroid[dimension]);
            residual_norm += residual * residual;
        }
        let table_base = sub * pq.ksub;
        for code in 0..pq.ksub {
            let offset = table_base + code;
            sim_table[offset] =
                (residual_norm + list_table[offset] - 2.0 * query_table[offset]).max(0.0) as f32;
        }
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
    matching_rows: Option<&MatchingRows>,
    heap: &mut TopKHeap,
) {
    let mut dists = vec![0.0f32; count];
    crate::distance::scan_4bit_simd(sim_table, codes, count, m, &mut dists);

    if let Some(rows) = matching_rows {
        for position in rows.positions() {
            heap.push(dis0 + dists[position], ids[position]);
        }
    } else {
        for i in 0..count {
            heap.push(dis0 + dists[i], ids[i]);
        }
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
    matching_rows: Option<&MatchingRows>,
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

    if let Some(rows) = matching_rows {
        for position in rows.positions() {
            heap.push(dis0 + dists[position], ids[position]);
        }
    } else {
        for i in 0..count {
            heap.push(dis0 + dists[i], ids[i]);
        }
    }
}

/// Scan transposed (column-major) codes: layout is [M][n].
/// The distance table sub-slice stays in L1 cache for the entire inner loop.
#[allow(clippy::too_many_arguments)]
fn scan_codes_transposed_with_scratch(
    sim_table: &[f32],
    codes: &[u8],
    ids: &[i64],
    count: usize,
    m: usize,
    ksub: usize,
    dis0: f32,
    matching_rows: Option<&MatchingRows>,
    heap: &mut TopKHeap,
    dists: &mut Vec<f32>,
) {
    debug_assert!(m > 0);
    if let Some(rows) =
        matching_rows.filter(|rows| should_scan_sparse(count, rows, TRANSPOSED_SPARSE_SCAN_DIVISOR))
    {
        for row in rows.positions() {
            let mut distance = dis0;
            for sub in 0..m {
                distance += sim_table[sub * ksub + codes[sub * count + row] as usize];
            }
            heap.push(distance, ids[row]);
        }
        return;
    }

    dists.resize(count, 0.0);
    let table = &sim_table[..ksub];
    let column = &codes[..count];
    let mut row = 0usize;
    while row + 8 <= count {
        dists[row] = dis0 + table[column[row] as usize];
        dists[row + 1] = dis0 + table[column[row + 1] as usize];
        dists[row + 2] = dis0 + table[column[row + 2] as usize];
        dists[row + 3] = dis0 + table[column[row + 3] as usize];
        dists[row + 4] = dis0 + table[column[row + 4] as usize];
        dists[row + 5] = dis0 + table[column[row + 5] as usize];
        dists[row + 6] = dis0 + table[column[row + 6] as usize];
        dists[row + 7] = dis0 + table[column[row + 7] as usize];
        row += 8;
    }
    while row < count {
        dists[row] = dis0 + table[column[row] as usize];
        row += 1;
    }
    for sub in 1..m {
        let table = &sim_table[sub * ksub..(sub + 1) * ksub];
        let col_base = sub * count;
        let column = &codes[col_base..col_base + count];
        let mut row = 0usize;
        while row + 8 <= count {
            dists[row] += table[column[row] as usize];
            dists[row + 1] += table[column[row + 1] as usize];
            dists[row + 2] += table[column[row + 2] as usize];
            dists[row + 3] += table[column[row + 3] as usize];
            dists[row + 4] += table[column[row + 4] as usize];
            dists[row + 5] += table[column[row + 5] as usize];
            dists[row + 6] += table[column[row + 6] as usize];
            dists[row + 7] += table[column[row + 7] as usize];
            row += 8;
        }
        while row < count {
            dists[row] += table[column[row] as usize];
            row += 1;
        }
    }

    if let Some(rows) = matching_rows {
        for position in rows.positions() {
            heap.push(dists[position], ids[position]);
        }
    } else {
        for i in 0..count {
            heap.push(dists[i], ids[i]);
        }
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
    matching_rows: Option<&MatchingRows>,
    heap: &mut TopKHeap,
) {
    if let Some(rows) =
        matching_rows.filter(|rows| should_scan_sparse(count, rows, ROW_MAJOR_SPARSE_SCAN_DIVISOR))
    {
        for position in rows.positions() {
            let code = &codes[position * m..(position + 1) * m];
            let distance = dis0 + pq_distance_from_table(sim_table, code, m, ksub);
            heap.push(distance, ids[position]);
        }
        return;
    }

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
            if matching_rows.is_none_or(|rows| rows.contains(idx)) {
                heap.push(dis0 + dists[j], ids[idx]);
            }
        }
        i += 4;
    }

    while i < count {
        if matching_rows.is_none_or(|rows| rows.contains(i)) {
            let code = &codes[i * m..(i + 1) * m];
            let dist = dis0 + pq_distance_from_table(sim_table, code, m, ksub);
            heap.push(dist, ids[i]);
        }
        i += 1;
    }
}

struct ReaderSearchContext<'a> {
    q: &'a [f32],
    ip_table: &'a [f32],
    use_precomputed: bool,
    shared_sim_table: Option<&'a OnceLock<Vec<f32>>>,
    #[cfg(test)]
    distance_table_builds: Option<&'a std::sync::atomic::AtomicUsize>,
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

#[derive(Default)]
struct ReaderScanScratch {
    sim_table: Vec<f32>,
    ip_table: Vec<f64>,
    distances: Vec<f32>,
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
    filter: Option<&dyn RowIdFilter>,
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
    let shared_sim_table = OnceLock::new();

    let mut heap = TopKHeap::new(k);

    let mut lists_to_read = Vec::new();
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
        lists_to_read.push((list_id, count, dis0));
    }
    lists_to_read.sort_unstable_by_key(|&(list_id, _, _)| reader.list_offsets[list_id]);

    let read_list_ids = lists_to_read
        .iter()
        .map(|&(list_id, _, _)| list_id)
        .collect::<Vec<_>>();
    let mut batch_start = 0usize;
    while batch_start < read_list_ids.len() {
        let first_list = read_list_ids[batch_start];
        if ivf_payload_is_oversized(reader.list_payload_len(first_list)?) {
            let (_, _, dis0) = lists_to_read[batch_start];
            let sim_table = by_residual
                .then(|| reader_sim_table(reader, first_list, &q, &ip_table, use_precomputed));
            let pq_nbits = reader.pq.nbits;
            let transposed_codes = reader.transposed_codes;
            let mut scratch = ReaderScanScratch::default();
            reader.for_each_streamed_list_chunk(first_list, |pq, ids, codes| {
                let positions = matching_rows(ids, filter);
                if positions.as_ref().is_some_and(MatchingRows::is_empty) {
                    return;
                }
                let sim_table = sim_table.as_deref().unwrap_or_else(|| {
                    shared_sim_table.get_or_init(|| {
                        let mut table = vec![0.0f32; m * ksub];
                        pq.compute_distance_table(&q, metric, &mut table);
                        table
                    })
                });
                scan_reader_codes(
                    sim_table,
                    codes,
                    ids,
                    m,
                    ksub,
                    pq_nbits,
                    transposed_codes,
                    dis0,
                    positions.as_ref(),
                    &mut scratch.distances,
                    &mut heap,
                );
            })?;
            batch_start += 1;
            continue;
        }
        let count = reader.batch_read_end(&read_list_ids[batch_start..])?.max(1);
        let batch_end = (batch_start + count).min(read_list_ids.len());
        let read_lists =
            reader.read_inverted_list_payloads(&read_list_ids[batch_start..batch_end])?;
        let mut list_data = Vec::with_capacity(read_lists.len());
        for (&(list_id, expected_count, dis0), read_list) in
            lists_to_read[batch_start..batch_end].iter().zip(read_lists)
        {
            if list_id != read_list.list_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "batched inverted list read returned lists out of order",
                ));
            }
            if expected_count != read_list.ids.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "batched inverted list read returned an unexpected row count",
                ));
            }
            list_data.push((read_list, dis0));
        }

        let ctx = ReaderSearchContext {
            q: &q,
            ip_table: &ip_table,
            use_precomputed,
            shared_sim_table: (!by_residual).then_some(&shared_sim_table),
            #[cfg(test)]
            distance_table_builds: None,
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
        let per_list_results = list_data
            .par_iter()
            .map_init(ReaderScanScratch::default, |scratch, (entry, dis0)| {
                let mut local_heap = TopKHeap::new(k);
                let positions = matching_rows(&entry.ids, filter);
                scan_reader_list(
                    entry,
                    *dis0,
                    &ctx,
                    positions.as_ref(),
                    scratch,
                    &mut local_heap,
                );
                local_heap.into_sorted()
            })
            .collect::<Vec<_>>();

        for results in per_list_results {
            for (dist, id) in results {
                heap.push(dist, id);
            }
        }
        batch_start = batch_end;
    }

    let sorted = heap.into_sorted();
    let result_ids: Vec<i64> = sorted.iter().map(|&(_, id)| id).collect();
    let result_dists: Vec<f32> = sorted.iter().map(|&(d, _)| d).collect();

    Ok((result_ids, result_dists))
}

/// Search with a cross-language serialized RoaringTreemap row-id filter.
pub fn search_with_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    query: &[f32],
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_with_reader_filter(reader, query, k, nprobe, Some(&filter))
}

fn scan_reader_list(
    entry: &InvertedListPayload,
    dis0: f32,
    ctx: &ReaderSearchContext<'_>,
    matching_rows: Option<&MatchingRows>,
    scratch: &mut ReaderScanScratch,
    heap: &mut TopKHeap,
) {
    if matching_rows.is_some_and(MatchingRows::is_empty) {
        return;
    }
    let sim_table = if let Some(table) = ctx.shared_sim_table {
        table.get_or_init(|| {
            let mut sim_table = Vec::new();
            fill_reader_sim_table(entry.list_id, ctx, &mut sim_table);
            sim_table
        })
    } else {
        fill_reader_sim_table(entry.list_id, ctx, &mut scratch.sim_table);
        &scratch.sim_table
    };
    scan_reader_codes(
        sim_table,
        entry.codes(),
        &entry.ids,
        ctx.m,
        ctx.ksub,
        ctx.pq.nbits,
        ctx.transposed_codes,
        dis0,
        matching_rows,
        &mut scratch.distances,
        heap,
    );
}

fn fill_reader_sim_table(list_id: usize, ctx: &ReaderSearchContext<'_>, sim_table: &mut Vec<f32>) {
    let d = ctx.d;
    let m = ctx.m;
    let ksub = ctx.ksub;
    sim_table.resize(m * ksub, 0.0);
    #[cfg(test)]
    if !ctx.use_precomputed {
        if let Some(builds) = ctx.distance_table_builds {
            builds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if ctx.use_precomputed {
        let tab_base = list_id * m * ksub;
        fvec_madd(
            &ctx.precomputed_table[tab_base..tab_base + m * ksub],
            ctx.ip_table,
            -2.0,
            sim_table,
        );
    } else if ctx.by_residual {
        let mut residual_query = vec![0.0f32; d];
        fvec_madd(
            ctx.q,
            &ctx.quantizer_centroids[list_id * d..(list_id + 1) * d],
            -1.0,
            &mut residual_query,
        );
        ctx.pq
            .compute_distance_table(&residual_query, ctx.metric, sim_table);
    } else {
        ctx.pq.compute_distance_table(ctx.q, ctx.metric, sim_table);
    }
}

fn reader_sim_table<R: SeekRead>(
    reader: &IVFPQIndexReader<R>,
    list_id: usize,
    query: &[f32],
    ip_table: &[f32],
    use_precomputed: bool,
) -> Vec<f32> {
    let ctx = ReaderSearchContext {
        q: query,
        ip_table,
        use_precomputed,
        shared_sim_table: None,
        #[cfg(test)]
        distance_table_builds: None,
        d: reader.d,
        m: reader.m,
        ksub: reader.ksub,
        metric: reader.metric,
        by_residual: reader.by_residual,
        transposed_codes: reader.transposed_codes,
        pq: &reader.pq,
        quantizer_centroids: &reader.quantizer_centroids,
        precomputed_table: &reader.precomputed_table,
    };
    let mut sim_table = Vec::new();
    fill_reader_sim_table(list_id, &ctx, &mut sim_table);
    sim_table
}

#[allow(clippy::too_many_arguments)]
fn scan_reader_codes(
    sim_table: &[f32],
    codes: &[u8],
    ids: &[i64],
    m: usize,
    ksub: usize,
    pq_nbits: usize,
    transposed_codes: bool,
    dis0: f32,
    matching_rows: Option<&MatchingRows>,
    distances: &mut Vec<f32>,
    heap: &mut TopKHeap,
) {
    if matching_rows.is_some_and(MatchingRows::is_empty) {
        return;
    }
    let is_4bit = pq_nbits == 4;
    let count = ids.len();
    if is_4bit && transposed_codes {
        scan_codes_4bit_transposed(sim_table, codes, ids, count, m, dis0, matching_rows, heap);
    } else if is_4bit {
        scan_codes_4bit(
            sim_table,
            codes,
            ids,
            count,
            m,
            ksub,
            dis0,
            matching_rows,
            heap,
        );
    } else if transposed_codes {
        scan_codes_transposed_with_scratch(
            sim_table,
            codes,
            ids,
            count,
            m,
            ksub,
            dis0,
            matching_rows,
            heap,
            distances,
        );
    } else {
        scan_codes_batched(
            sim_table,
            codes,
            ids,
            count,
            m,
            ksub,
            dis0,
            matching_rows,
            heap,
        );
    }
}

#[derive(Default)]
struct IvfpqBatchTiming {
    load: Duration,
    preprocess: Duration,
    coarse: Duration,
    prepare: Duration,
    io_read: Duration,
    decode: Duration,
    filter: Duration,
    scan: Duration,
    finalize: Duration,
    read_calls: usize,
    requested_bytes: usize,
    unique_list_rows: usize,
    query_list_pairs: usize,
    pq_codes_evaluated: usize,
    matched_rows: usize,
    queries_below_k: usize,
    min_hits_per_query: usize,
}

impl IvfpqBatchTiming {
    fn record_scan_work(&mut self, rows: usize, matched_rows: usize, query_uses: usize) {
        self.unique_list_rows = self.unique_list_rows.saturating_add(rows);
        self.matched_rows = self.matched_rows.saturating_add(matched_rows);
        self.pq_codes_evaluated = self
            .pq_codes_evaluated
            .saturating_add(matched_rows.saturating_mul(query_uses));
    }

    fn write_to<W: io::Write>(
        &self,
        mut output: W,
        total: Duration,
        nq: usize,
        nprobe: usize,
        pq_bits: usize,
        topk: usize,
        unique_lists: usize,
        filtered: bool,
    ) -> io::Result<()> {
        let millis = |duration: Duration| duration.as_secs_f64() * 1_000.0;
        writeln!(
            output,
            "[paimon-vindex] ivfpq_batch_timing nq={nq} nprobe={nprobe} pq_bits={pq_bits} \
             topk={topk} unique_lists={unique_lists} unique_list_rows={} query_list_pairs={} \
             pq_codes_evaluated={} matched_rows={} read_calls={} requested_bytes={} \
             queries_below_k={} min_hits_per_query={} filtered={filtered} total_ms={:.3} \
             load_ms={:.3} preprocess_ms={:.3} coarse_ms={:.3} prepare_ms={:.3} \
             io_read_ms={:.3} decode_ms={:.3} filter_ms={:.3} scan_ms={:.3} finalize_ms={:.3}",
            self.unique_list_rows,
            self.query_list_pairs,
            self.pq_codes_evaluated,
            self.matched_rows,
            self.read_calls,
            self.requested_bytes,
            self.queries_below_k,
            self.min_hits_per_query,
            millis(total),
            millis(self.load),
            millis(self.preprocess),
            millis(self.coarse),
            millis(self.prepare),
            millis(self.io_read),
            millis(self.decode),
            millis(self.filter),
            millis(self.scan),
            millis(self.finalize),
        )
    }
}

#[inline]
fn elapsed_since(started: Option<Instant>) -> Duration {
    started.map_or(Duration::ZERO, |started| started.elapsed())
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
    search_batch_reader_with_reuse_mode(
        reader,
        queries,
        nq,
        k,
        nprobe,
        IvfPqBatchTableReuseMode::Auto,
    )
}

pub fn search_batch_reader_with_reuse_mode<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    reuse_mode: IvfPqBatchTableReuseMode,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_reader_with_reuse_mode_and_budget(
        reader,
        queries,
        nq,
        k,
        nprobe,
        reuse_mode,
        DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
    )
}

pub fn search_batch_reader_with_reuse_mode_and_budget<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    reuse_mode: IvfPqBatchTableReuseMode,
    reuse_max_bytes: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_reader_filter_with_reuse_mode_and_budget(
        reader,
        queries,
        nq,
        k,
        nprobe,
        None,
        reuse_mode,
        reuse_max_bytes,
    )
}

/// Big batch search with an optional row-id filter.
pub fn search_batch_reader_filter<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_reader_filter_with_reuse_mode(
        reader,
        queries,
        nq,
        k,
        nprobe,
        filter,
        IvfPqBatchTableReuseMode::Auto,
    )
}

pub fn search_batch_reader_filter_with_reuse_mode<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
    reuse_mode: IvfPqBatchTableReuseMode,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_reader_filter_with_reuse_mode_and_budget(
        reader,
        queries,
        nq,
        k,
        nprobe,
        filter,
        reuse_mode,
        DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
    )
}

pub fn search_batch_reader_filter_with_reuse_mode_and_budget<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
    reuse_mode: IvfPqBatchTableReuseMode,
    reuse_max_bytes: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_reader_filter_with_reuse_mode_and_observer(
        reader,
        queries,
        nq,
        k,
        nprobe,
        filter,
        reuse_mode,
        reuse_max_bytes,
        |_| {},
        #[cfg(test)]
        None,
    )
}

#[cfg(test)]
fn search_batch_reader_filter_with_observer<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
    mut observe_ephemeral_precomputed_lists: impl FnMut(usize),
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_reader_filter_with_reuse_mode_and_observer(
        reader,
        queries,
        nq,
        k,
        nprobe,
        filter,
        IvfPqBatchTableReuseMode::Auto,
        DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
        &mut observe_ephemeral_precomputed_lists,
        None,
    )
}

fn search_batch_reader_filter_with_reuse_mode_and_observer<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
    reuse_mode: IvfPqBatchTableReuseMode,
    reuse_max_bytes: usize,
    mut observe_ephemeral_precomputed_lists: impl FnMut(usize),
    #[cfg(test)] distance_table_builds: Option<&std::sync::atomic::AtomicUsize>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let timing_enabled = std::env::var_os("PAIMON_VINDEX_LOG_IVFPQ_TIMING").is_some();
    let total_started = timing_enabled.then(Instant::now);
    let mut timing = IvfpqBatchTiming::default();
    let load_started = timing_enabled.then(Instant::now);
    reader.ensure_loaded()?;
    timing.load = elapsed_since(load_started);
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
    let preprocess_started = timing_enabled.then(Instant::now);
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
    timing.preprocess = elapsed_since(preprocess_started);

    // Step 2: Batch coarse search (one sgemm)
    let coarse_started = timing_enabled.then(Instant::now);
    let (all_probe_indices, all_coarse_dists) = kmeans::find_topk_batch(
        &processed,
        nq,
        &reader.quantizer_centroids,
        reader.nlist,
        d,
        nprobe,
    );
    timing.coarse = elapsed_since(coarse_started);

    // Step 3: Read every probed list once. Queries share the decoded list
    // payloads, then scan independently in parallel.
    let prepare_started = timing_enabled.then(Instant::now);
    let mut seen = vec![false; reader.nlist];
    let mut unique_lists = Vec::new();
    let mut query_uses_by_list = timing_enabled.then(|| vec![0usize; reader.nlist]);
    for probe_indices in &all_probe_indices {
        for &list_id in probe_indices {
            if let Some(query_uses) = query_uses_by_list.as_mut() {
                query_uses[list_id] = query_uses[list_id].saturating_add(1);
                timing.query_list_pairs = timing.query_list_pairs.saturating_add(1);
            }
            if !seen[list_id] && reader.list_counts[list_id] > 0 {
                seen[list_id] = true;
                unique_lists.push(list_id);
            }
        }
    }
    unique_lists.sort_unstable_by_key(|&list_id| reader.list_offsets[list_id]);

    let reuse_required_bytes = nq
        .checked_mul(m)
        .and_then(|values| values.checked_mul(ksub))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()));
    let reused_query_tables_fit_budget =
        reuse_required_bytes.is_some_and(|bytes| bytes <= reuse_max_bytes);
    let use_precomputed = reuse_mode != IvfPqBatchTableReuseMode::Off
        && metric == MetricType::L2
        && by_residual
        && !reader.precomputed_table.is_empty()
        && reused_query_tables_fit_budget;
    let allow_ephemeral_precomputed = reader.pq.nbits == 8
        && metric == MetricType::L2
        && by_residual
        && !use_precomputed
        && match reuse_mode {
            IvfPqBatchTableReuseMode::Off => false,
            IvfPqBatchTableReuseMode::On => true,
            IvfPqBatchTableReuseMode::Auto => nq >= MIN_EPHEMERAL_PRECOMPUTE_QUERIES,
        };
    let all_ip_tables: Vec<Vec<f32>> = if use_precomputed {
        (0..nq)
            .into_par_iter()
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
    // Non-residual tables depend only on the query and PQ codebook, so every
    // probed list for that query can share one table.
    let reuse_non_residual_tables = reader.pq.nbits == 8
        && !by_residual
        && nprobe > 1
        && match reuse_mode {
            IvfPqBatchTableReuseMode::Off => false,
            IvfPqBatchTableReuseMode::On => true,
            IvfPqBatchTableReuseMode::Auto => {
                let (active_query_count, probe_count) = all_probe_indices
                    .iter()
                    .map(|probes| {
                        probes
                            .iter()
                            .filter(|&&list_id| reader.list_counts[list_id] > 0)
                            .count()
                    })
                    .fold((0usize, 0usize), |(active, total), probes| {
                        (active + usize::from(probes > 0), total + probes)
                    });
                nq >= MIN_EPHEMERAL_PRECOMPUTE_QUERIES
                    && should_use_ephemeral_precomputation(0, active_query_count, probe_count)
            }
        }
        && reused_query_tables_fit_budget;
    let shared_sim_tables = if reuse_non_residual_tables {
        (0..nq).map(|_| OnceLock::new()).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut stable_pq_norms = None;
    timing.prepare = elapsed_since(prepare_started);

    let mut heaps = (0..nq).map(|_| TopKHeap::new(k)).collect::<Vec<_>>();
    if timing_enabled {
        reader.begin_read_metrics();
    }
    let mut batch_start = 0usize;
    while batch_start < unique_lists.len() {
        let first_list = unique_lists[batch_start];
        if ivf_payload_is_oversized(reader.list_payload_len(first_list)?) {
            let prepare_started = timing_enabled.then(Instant::now);
            let query_tables = (0..nq)
                .filter_map(|query_index| {
                    all_probe_indices[query_index]
                        .iter()
                        .position(|&list_id| list_id == first_list)
                        .map(|probe_rank| {
                            let query = &processed[query_index * d..(query_index + 1) * d];
                            let sim_table = (!reuse_non_residual_tables).then(|| {
                                reader_sim_table(
                                    reader,
                                    first_list,
                                    query,
                                    if use_precomputed {
                                        &all_ip_tables[query_index]
                                    } else {
                                        &[]
                                    },
                                    use_precomputed,
                                )
                            });
                            let dis0 = if use_precomputed {
                                all_coarse_dists[query_index][probe_rank]
                            } else {
                                0.0
                            };
                            (query_index, dis0, sim_table)
                        })
                })
                .collect::<Vec<_>>();
            let pq_nbits = reader.pq.nbits;
            let transposed_codes = reader.transposed_codes;
            // The loop is sequential across queries. Reuse one chunk-sized
            // distance buffer instead of retaining one per query.
            let mut distances = Vec::new();
            timing.prepare += elapsed_since(prepare_started);
            let streamed_started = timing_enabled.then(Instant::now);
            let mut streamed_filter = Duration::ZERO;
            let mut streamed_scan = Duration::ZERO;
            reader.for_each_streamed_list_chunk(first_list, |pq, ids, codes| {
                let filter_started = timing_enabled.then(Instant::now);
                let positions = matching_rows(ids, filter);
                streamed_filter += elapsed_since(filter_started);
                if timing_enabled {
                    let matched_rows = positions.as_ref().map_or(ids.len(), MatchingRows::len);
                    timing.record_scan_work(ids.len(), matched_rows, query_tables.len());
                }
                if positions.as_ref().is_some_and(MatchingRows::is_empty) {
                    return;
                }
                let scan_started = timing_enabled.then(Instant::now);
                for (query_index, dis0, sim_table) in &query_tables {
                    let sim_table = sim_table.as_deref().unwrap_or_else(|| {
                        shared_sim_tables[*query_index].get_or_init(|| {
                            let mut table = vec![0.0f32; m * ksub];
                            pq.compute_distance_table(
                                &processed[*query_index * d..(*query_index + 1) * d],
                                metric,
                                &mut table,
                            );
                            #[cfg(test)]
                            if let Some(builds) = distance_table_builds {
                                builds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            table
                        })
                    });
                    scan_reader_codes(
                        sim_table,
                        codes,
                        ids,
                        m,
                        ksub,
                        pq_nbits,
                        transposed_codes,
                        *dis0,
                        positions.as_ref(),
                        &mut distances,
                        &mut heaps[*query_index],
                    );
                }
                streamed_scan += elapsed_since(scan_started);
            })?;
            let streamed_total = elapsed_since(streamed_started);
            timing.filter += streamed_filter;
            timing.scan += streamed_scan;
            timing.decode += streamed_total.saturating_sub(streamed_filter + streamed_scan);
            batch_start += 1;
            continue;
        }
        let count = reader.batch_read_end(&unique_lists[batch_start..])?.max(1);
        let batch_end = (batch_start + count).min(unique_lists.len());
        let read_decode_started = timing_enabled.then(Instant::now);
        let loaded_lists =
            reader.read_inverted_list_payloads(&unique_lists[batch_start..batch_end])?;
        timing.decode += elapsed_since(read_decode_started);
        let prepare_started = timing_enabled.then(Instant::now);
        let mut list_positions = vec![usize::MAX; reader.nlist];
        for (position, list) in loaded_lists.iter().enumerate() {
            list_positions[list.list_id] = position;
        }
        timing.prepare += elapsed_since(prepare_started);
        let filter_started = timing_enabled.then(Instant::now);
        let matching_rows_by_list = loaded_lists
            .iter()
            .map(|list| matching_rows(&list.ids, filter))
            .collect::<Vec<_>>();
        timing.filter += elapsed_since(filter_started);
        if let Some(query_uses) = query_uses_by_list.as_ref() {
            for (list, matching_rows) in loaded_lists.iter().zip(&matching_rows_by_list) {
                timing.record_scan_work(
                    list.ids.len(),
                    matching_rows
                        .as_ref()
                        .map_or(list.ids.len(), MatchingRows::len),
                    query_uses[list.list_id],
                );
            }
        }
        let scan_started = timing_enabled.then(Instant::now);
        let matching_list_count = matching_rows_by_list
            .iter()
            .filter(|rows| has_matching_rows(rows.as_ref()))
            .count();
        let (active_query_count, probe_count) = if allow_ephemeral_precomputed {
            let mut probe_count = 0usize;
            let mut active_query_count = 0usize;
            for probe_indices in &all_probe_indices {
                let matching_probe_count = probe_indices
                    .iter()
                    .filter(|&&list_id| {
                        let position = list_positions[list_id];
                        position != usize::MAX
                            && has_matching_rows(matching_rows_by_list[position].as_ref())
                    })
                    .count();
                probe_count += matching_probe_count;
                active_query_count += usize::from(matching_probe_count > 0);
            }
            (active_query_count, probe_count)
        } else {
            (0, 0)
        };
        let query_scratch_count = active_query_count.min(rayon::current_num_threads());
        let use_ephemeral_precomputed = allow_ephemeral_precomputed
            && ephemeral_precomputed_table_fits_budget(
                matching_list_count,
                query_scratch_count,
                m,
                ksub,
                reuse_max_bytes,
            )
            && (reuse_mode == IvfPqBatchTableReuseMode::On
                || should_use_ephemeral_precomputation(
                    matching_list_count,
                    active_query_count,
                    probe_count,
                ));
        let ephemeral_precomputed_tables = if use_ephemeral_precomputed {
            let pq_norms = stable_pq_norms
                .get_or_insert_with(|| compute_stable_ephemeral_pq_norms(&reader.pq));
            loaded_lists
                .par_iter()
                .zip(&matching_rows_by_list)
                .map(|(list, rows)| {
                    let mut table = Vec::new();
                    if has_matching_rows(rows.as_ref()) {
                        fill_stable_ephemeral_list_table(
                            &reader.quantizer_centroids[list.list_id * d..(list.list_id + 1) * d],
                            &reader.pq,
                            pq_norms,
                            &mut table,
                        );
                    }
                    table
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        observe_ephemeral_precomputed_lists(
            ephemeral_precomputed_tables
                .iter()
                .filter(|table| !table.is_empty())
                .count(),
        );

        let rows = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let query = &processed[qi * d..(qi + 1) * d];
                let ctx = ReaderSearchContext {
                    q: query,
                    ip_table: if use_precomputed {
                        &all_ip_tables[qi]
                    } else {
                        &[]
                    },
                    use_precomputed,
                    shared_sim_table: reuse_non_residual_tables.then(|| &shared_sim_tables[qi]),
                    #[cfg(test)]
                    distance_table_builds,
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
                let mut heap = TopKHeap::new(k);
                let mut scratch = ReaderScanScratch::default();
                let query_uses_ephemeral_precomputed = use_ephemeral_precomputed
                    && all_probe_indices[qi].iter().any(|&list_id| {
                        let position = list_positions[list_id];
                        position != usize::MAX && !ephemeral_precomputed_tables[position].is_empty()
                    });
                if query_uses_ephemeral_precomputed {
                    fill_stable_ephemeral_query_table(query, &reader.pq, &mut scratch.ip_table);
                }
                for (probe_rank, &list_id) in all_probe_indices[qi].iter().enumerate() {
                    let position = list_positions[list_id];
                    if position == usize::MAX {
                        continue;
                    }
                    let use_ephemeral_list = query_uses_ephemeral_precomputed
                        && !ephemeral_precomputed_tables[position].is_empty();
                    let dis0 = if use_ephemeral_list {
                        0.0
                    } else if use_precomputed {
                        all_coarse_dists[qi][probe_rank]
                    } else {
                        0.0
                    };
                    if use_ephemeral_list {
                        combine_stable_ephemeral_tables(
                            &ephemeral_precomputed_tables[position],
                            &scratch.ip_table,
                            query,
                            &reader.quantizer_centroids[list_id * d..(list_id + 1) * d],
                            &reader.pq,
                            &mut scratch.sim_table,
                        );
                        scan_reader_codes(
                            &scratch.sim_table,
                            loaded_lists[position].codes(),
                            &loaded_lists[position].ids,
                            m,
                            ksub,
                            reader.pq.nbits,
                            reader.transposed_codes,
                            dis0,
                            matching_rows_by_list[position].as_ref(),
                            &mut scratch.distances,
                            &mut heap,
                        );
                    } else {
                        scan_reader_list(
                            &loaded_lists[position],
                            dis0,
                            &ctx,
                            matching_rows_by_list[position].as_ref(),
                            &mut scratch,
                            &mut heap,
                        );
                    }
                }
                heap.into_sorted()
            })
            .collect::<Vec<_>>();
        for (qi, row) in rows.into_iter().enumerate() {
            for (distance, row_id) in row {
                heaps[qi].push(distance, row_id);
            }
        }
        timing.scan += elapsed_since(scan_started);
        batch_start = batch_end;
    }
    if timing_enabled {
        let read_metrics = reader.end_read_metrics();
        timing.io_read = read_metrics.elapsed;
        timing.decode = timing.decode.saturating_sub(timing.io_read);
        timing.read_calls = read_metrics.calls;
        timing.requested_bytes = read_metrics.requested_bytes;
    }

    let finalize_started = timing_enabled.then(Instant::now);
    let mut result_ids = vec![-1i64; nq * k];
    let mut result_dists = vec![f32::MAX; nq * k];
    timing.min_hits_per_query = k;
    for (qi, heap) in heaps.into_iter().enumerate() {
        let sorted = heap.into_sorted();
        if timing_enabled {
            timing.queries_below_k = timing
                .queries_below_k
                .saturating_add(usize::from(sorted.len() < k));
            timing.min_hits_per_query = timing.min_hits_per_query.min(sorted.len());
        }
        let base = qi * k;
        for (i, &(dist, id)) in sorted.iter().enumerate() {
            result_ids[base + i] = id;
            result_dists[base + i] = dist;
        }
    }
    timing.finalize = elapsed_since(finalize_started);

    if timing_enabled {
        let mut buf = Vec::with_capacity(256);
        let _ = timing.write_to(
            &mut buf,
            elapsed_since(total_started),
            nq,
            nprobe,
            reader.pq.nbits,
            k,
            unique_lists.len(),
            filter.is_some(),
        );
        emit_log(LogLevel::Info, String::from_utf8_lossy(&buf).trim_end());
    }

    if !by_residual && std::env::var_os("PAIMON_VINDEX_LOG_IVFPQ_BATCH_REUSE").is_some() {
        let tables_built = shared_sim_tables
            .iter()
            .filter(|table| table.get().is_some())
            .count();
        let message = format!(
            "[paimon-vindex] ivfpq_batch_table_reuse strategy=non_residual_query_table \
             mode={reuse_mode:?} enabled={reuse_non_residual_tables} used={} metric={} \
             pq_bits={} nq={nq} nprobe={nprobe} unique_lists={} filtered={} required_bytes={:?} \
             budget_bytes={reuse_max_bytes} tables_built={tables_built}",
            tables_built > 0,
            metric.as_str(),
            reader.pq.nbits,
            unique_lists.len(),
            filter.is_some(),
            reuse_required_bytes,
        );
        emit_log(LogLevel::Info, &message);
    }

    Ok((result_ids, result_dists))
}

/// Big batch search with a cross-language serialized RoaringTreemap row-id filter.
pub fn search_batch_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_reader_roaring_filter_with_reuse_mode(
        reader,
        queries,
        nq,
        k,
        nprobe,
        roaring_filter_bytes,
        IvfPqBatchTableReuseMode::Auto,
    )
}

pub fn search_batch_reader_roaring_filter_with_reuse_mode<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
    reuse_mode: IvfPqBatchTableReuseMode,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_reader_roaring_filter_with_reuse_mode_and_budget(
        reader,
        queries,
        nq,
        k,
        nprobe,
        roaring_filter_bytes,
        reuse_mode,
        DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
    )
}

pub fn search_batch_reader_roaring_filter_with_reuse_mode_and_budget<R: SeekRead>(
    reader: &mut IVFPQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
    reuse_mode: IvfPqBatchTableReuseMode,
    reuse_max_bytes: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_batch_reader_filter_with_reuse_mode_and_budget(
        reader,
        queries,
        nq,
        k,
        nprobe,
        Some(&filter),
        reuse_mode,
        reuse_max_bytes,
    )
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
    let assignments = kmeans::find_nearest_batch(data, n, centroids, nlist, d);
    residuals
        .par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, residual)| {
            let list_id = assignments[i];
            fvec_madd(
                &data[i * d..(i + 1) * d],
                &centroids[list_id * d..(list_id + 1) * d],
                -1.0,
                residual,
            );
        });
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
    use crate::io::{ReadRequest, SeekRead};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    struct CountingFilter {
        contains_calls: AtomicUsize,
    }

    impl CountingFilter {
        fn new() -> Self {
            Self {
                contains_calls: AtomicUsize::new(0),
            }
        }
    }

    impl RowIdFilter for CountingFilter {
        fn contains(&self, id: i64) -> bool {
            self.contains_calls.fetch_add(1, Ordering::Relaxed);
            id % 7 == 0
        }
    }

    #[derive(Default)]
    struct ReaderStats {
        pread_calls: usize,
        pread_batches: usize,
        max_ranges_per_batch: usize,
        max_pread_len: usize,
        last_positions: Vec<u64>,
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
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            {
                let mut stats = self.stats.lock().unwrap();
                stats.pread_batches += 1;
                stats.max_ranges_per_batch = stats.max_ranges_per_batch.max(ranges.len());
                stats.last_positions = ranges.iter().map(|range| range.pos).collect();
            }
            for range in ranges {
                {
                    let mut stats = self.stats.lock().unwrap();
                    stats.pread_calls += 1;
                    stats.max_pread_len = stats.max_pread_len.max(range.buf.len());
                }
                io::Seek::seek(&mut self.inner, io::SeekFrom::Start(range.pos))?;
                io::Read::read_exact(&mut self.inner, range.buf)?;
            }
            Ok(())
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

    fn observed_ephemeral_precomputed_lists(
        nq: usize,
        nprobe: usize,
        filter_step: Option<usize>,
        apply_filter: bool,
        seed: u64,
        reuse_mode: IvfPqBatchTableReuseMode,
    ) -> usize {
        observed_ephemeral_precomputed_lists_with_nbits(
            8,
            nq,
            nprobe,
            filter_step,
            apply_filter,
            seed,
            reuse_mode,
        )
    }

    fn observed_ephemeral_precomputed_lists_with_nbits(
        nbits: usize,
        nq: usize,
        nprobe: usize,
        filter_step: Option<usize>,
        apply_filter: bool,
        seed: u64,
        reuse_mode: IvfPqBatchTableReuseMode,
    ) -> usize {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 600;
        let k = 5;
        let data = generate_clustered_data(n, d, nlist, seed);
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFPQIndex::with_nbits(d, nlist, m, nbits, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let filter = match filter_step {
            Some(step) => ids.iter().copied().step_by(step).collect::<HashSet<_>>(),
            None => HashSet::new(),
        };
        let filter = if apply_filter {
            Some(&filter as &dyn RowIdFilter)
        } else {
            None
        };
        let precomputed_lists = AtomicUsize::new(0);
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();

        search_batch_reader_filter_with_reuse_mode_and_observer(
            &mut reader,
            &data[..nq * d],
            nq,
            k,
            nprobe,
            filter,
            reuse_mode,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
            |count| {
                precomputed_lists.fetch_add(count, Ordering::Relaxed);
            },
            None,
        )
        .unwrap();

        precomputed_lists.load(Ordering::Relaxed)
    }

    fn assert_invalid_merge(base: &IVFPQIndex, other: &IVFPQIndex, expected_message: &str) {
        let mut target = IVFPQIndex::from_trained(base);
        let before_ids = target.ids.clone();
        let before_codes = target.codes.clone();

        let err = target.merge_from(other).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains(expected_message),
            "merge error `{}` does not contain `{}`",
            err,
            expected_message
        );
        assert_eq!(target.ids, before_ids);
        assert_eq!(target.codes, before_codes);
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
    fn in_memory_batch_filter_only_evaluates_probed_lists() {
        let d = 16;
        let nlist = 8;
        let m = 4;
        let n = 800;
        let nq = 4;
        let k = 5;
        let nprobe = 2;
        let data = generate_clustered_data(n, d, nlist, 51);
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let queries = &data[..nq * d];
        let processed_queries = index.preprocess_queries(queries, nq);
        let (probe_indices, _) = kmeans::find_topk_batch(
            &processed_queries,
            nq,
            &index.quantizer_centroids,
            nlist,
            d,
            nprobe,
        );
        let mut probed_lists = vec![false; nlist];
        for query_probes in probe_indices {
            for list_id in query_probes {
                probed_lists[list_id] = true;
            }
        }
        let expected_calls = index
            .ids
            .iter()
            .zip(probed_lists)
            .filter_map(|(ids, probed)| probed.then_some(ids.len()))
            .sum::<usize>();

        let filter = CountingFilter::new();
        let mut distances = vec![0.0f32; nq * k];
        let mut labels = vec![-1i64; nq * k];
        index.search_with_filter(
            queries,
            nq,
            k,
            nprobe,
            Some(&filter),
            &mut distances,
            &mut labels,
        );

        assert!(
            labels.iter().filter(|&&id| id >= 0).all(|id| id % 7 == 0),
            "filtered search returned a disallowed row ID"
        );
        assert_eq!(
            filter.contains_calls.load(Ordering::Relaxed),
            expected_calls,
            "the filter should only evaluate rows from lists probed by this batch"
        );
        assert!(
            expected_calls < n,
            "the test must leave at least one inverted list unprobed"
        );
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
        merged.merge_from(&worker_a).unwrap();
        merged.merge_from(&worker_b).unwrap();

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
    fn test_merge_rejects_incompatible_training_state() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut trainer = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        trainer.train(&data, n);

        let mut base = IVFPQIndex::from_trained(&trainer);
        base.add(&data, &ids, n);

        let mut mismatched_metric = IVFPQIndex::from_trained(&trainer);
        mismatched_metric.metric = MetricType::InnerProduct;
        mismatched_metric.by_residual = false;
        assert_invalid_merge(&base, &mismatched_metric, "metric mismatch");

        let mut mismatched_residual = IVFPQIndex::from_trained(&trainer);
        mismatched_residual.by_residual = false;
        assert_invalid_merge(&base, &mismatched_residual, "residual mode mismatch");

        let mut mismatched_centroids = IVFPQIndex::from_trained(&trainer);
        mismatched_centroids.quantizer_centroids[0] += 1.0;
        assert_invalid_merge(&base, &mismatched_centroids, "coarse centroids mismatch");

        let mut mismatched_codebooks = IVFPQIndex::from_trained(&trainer);
        mismatched_codebooks.pq.centroids[0] += 1.0;
        assert_invalid_merge(&base, &mismatched_codebooks, "PQ codebooks mismatch");

        let mismatched_opq = IVFPQIndex::new(d, nlist, m, MetricType::L2, true);
        assert_invalid_merge(&base, &mismatched_opq, "OPQ configuration mismatch");
    }

    #[test]
    fn test_merge_rejects_incompatible_opq_rotation() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;

        let data = generate_clustered_data(n, d, 4, 55);

        let mut trainer = IVFPQIndex::new(d, nlist, m, MetricType::L2, true);
        trainer.train(&data, n);

        let base = IVFPQIndex::from_trained(&trainer);
        let mut mismatched_rotation = IVFPQIndex::from_trained(&trainer);
        mismatched_rotation.opq.as_mut().unwrap().rotation[0] += 1.0;

        assert_invalid_merge(&base, &mismatched_rotation, "OPQ rotation mismatch");
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
    fn ivfpq_only_allocates_preprocessed_vectors_when_required() {
        let data = vec![3.0, 4.0, 1.0, 2.0];
        let l2 = IVFPQIndex::new(2, 1, 1, MetricType::L2, false);
        assert!(matches!(l2.preprocess_queries(&data, 2), Cow::Borrowed(_)));

        let cosine = IVFPQIndex::new(2, 1, 1, MetricType::Cosine, false);
        assert!(matches!(cosine.preprocess_queries(&data, 2), Cow::Owned(_)));
    }

    #[test]
    fn transposed_scan_matches_scalar_distance_table() {
        let count = 37;
        let m = 7;
        let ksub = 256;
        let dis0 = 3.25;
        let ids = (1_000..1_000 + count as i64).collect::<Vec<_>>();
        let codes = (0..m * count)
            .map(|index| ((index * 73 + 19) % ksub) as u8)
            .collect::<Vec<_>>();
        let table = (0..m * ksub)
            .map(|index| (index % 101) as f32 * 0.03125)
            .collect::<Vec<_>>();

        let mut heap = TopKHeap::new(count);
        let mut scratch = Vec::new();
        scan_codes_transposed_with_scratch(
            &table,
            &codes,
            &ids,
            count,
            m,
            ksub,
            dis0,
            None,
            &mut heap,
            &mut scratch,
        );

        let mut expected = (0..count)
            .map(|row| {
                let distance = (0..m).fold(dis0, |distance, sub| {
                    distance + table[sub * ksub + codes[sub * count + row] as usize]
                });
                (distance, ids[row])
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| left.0.total_cmp(&right.0));
        assert_eq!(heap.into_sorted(), expected);

        let matching_positions = (0..count).step_by(5).collect::<Vec<_>>();
        let matching_rows = MatchingRows::Sparse(matching_positions);
        let mut filtered_heap = TopKHeap::new(matching_rows.len());
        scan_codes_transposed_with_scratch(
            &table,
            &codes,
            &ids,
            count,
            m,
            ksub,
            dis0,
            Some(&matching_rows),
            &mut filtered_heap,
            &mut scratch,
        );
        let filtered_expected = expected
            .into_iter()
            .filter(|(_, id)| (id - ids[0]) % 5 == 0)
            .collect::<Vec<_>>();
        assert_eq!(filtered_heap.into_sorted(), filtered_expected);
    }

    #[test]
    fn row_major_sparse_scan_matches_exact_distances() {
        let count = 400;
        let m = 8;
        let ksub = 256;
        let dis0 = 1.25;
        let ids = (10_000..10_000 + count as i64).collect::<Vec<_>>();
        let codes = (0..count * m)
            .map(|index| ((index * 37 + 11) % ksub) as u8)
            .collect::<Vec<_>>();
        let table = (0..m * ksub)
            .map(|index| ((index * 29 + 7) % 113) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let matching_positions = (0..count).step_by(3).collect::<Vec<_>>();
        let mut expected = matching_positions
            .iter()
            .map(|&position| {
                let code = &codes[position * m..(position + 1) * m];
                (
                    dis0 + pq_distance_from_table(&table, code, m, ksub),
                    ids[position],
                )
            })
            .collect::<Vec<_>>();
        expected.sort_unstable_by_key(|&(_, id)| id);

        let mut filtered_heap = TopKHeap::new(matching_positions.len());
        let matching_rows = MatchingRows::Sparse(matching_positions);
        scan_codes_batched(
            &table,
            &codes,
            &ids,
            count,
            m,
            ksub,
            dis0,
            Some(&matching_rows),
            &mut filtered_heap,
        );

        let mut actual = filtered_heap.into_sorted();
        actual.sort_unstable_by_key(|&(_, id)| id);
        assert_eq!(actual, expected);
    }

    #[test]
    fn ivfpq_batch_timing_output_names_search_phases() {
        let timing = IvfpqBatchTiming {
            load: std::time::Duration::from_millis(1),
            preprocess: std::time::Duration::from_millis(2),
            coarse: std::time::Duration::from_millis(3),
            prepare: std::time::Duration::from_millis(4),
            io_read: std::time::Duration::from_millis(5),
            decode: std::time::Duration::from_millis(6),
            filter: std::time::Duration::from_millis(7),
            scan: std::time::Duration::from_millis(8),
            finalize: std::time::Duration::from_millis(9),
            read_calls: 10,
            requested_bytes: 4096,
            unique_list_rows: 120,
            query_list_pairs: 512,
            pq_codes_evaluated: 240,
            matched_rows: 30,
            queries_below_k: 2,
            min_hits_per_query: 1,
        };
        let mut output = Vec::new();

        timing
            .write_to(
                &mut output,
                std::time::Duration::from_millis(45),
                64,
                8,
                4,
                3,
                12,
                true,
            )
            .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "[paimon-vindex] ivfpq_batch_timing nq=64 nprobe=8 pq_bits=4 \
             topk=3 unique_lists=12 unique_list_rows=120 query_list_pairs=512 \
             pq_codes_evaluated=240 matched_rows=30 read_calls=10 requested_bytes=4096 \
             queries_below_k=2 min_hits_per_query=1 filtered=true total_ms=45.000 load_ms=1.000 \
             preprocess_ms=2.000 coarse_ms=3.000 prepare_ms=4.000 \
             io_read_ms=5.000 decode_ms=6.000 filter_ms=7.000 scan_ms=8.000 \
             finalize_ms=9.000\n"
        );
    }

    #[test]
    fn ivfpq_batch_timing_accumulates_actual_scan_work() {
        let mut timing = IvfpqBatchTiming::default();

        timing.record_scan_work(100, 25, 4);
        timing.record_scan_work(40, 10, 2);

        assert_eq!(timing.unique_list_rows, 140);
        assert_eq!(timing.matched_rows, 35);
        assert_eq!(timing.pq_codes_evaluated, 120);
    }

    #[test]
    fn matching_rows_adapts_sparse_positions_to_bounded_bitmap() {
        let ids = (0..1024i64).collect::<Vec<_>>();
        let sparse_filter = [3i64, 511, 900].into_iter().collect::<HashSet<_>>();
        let sparse = matching_rows(&ids, Some(&sparse_filter)).unwrap();
        assert!(matches!(sparse, MatchingRows::Sparse(_)));
        assert_eq!(sparse.positions().collect::<Vec<_>>(), vec![3, 511, 900]);

        let dense_filter = ids.iter().copied().collect::<HashSet<_>>();
        let dense = matching_rows(&ids, Some(&dense_filter)).unwrap();
        assert!(matches!(dense, MatchingRows::Bitmap { .. }));
        assert_eq!(dense.len(), ids.len());
        assert_eq!(
            dense.positions().collect::<Vec<_>>(),
            (0..ids.len()).collect::<Vec<_>>()
        );
        assert!(
            dense.storage_bytes() <= ids.len().div_ceil(64) * size_of::<u64>(),
            "dense match storage must be bounded to one bit per row"
        );
    }

    #[test]
    fn row_major_dense_bitmap_scan_matches_exact_distances() {
        let count = 1024;
        let m = 8;
        let ksub = 256;
        let dis0 = 2.5;
        let ids = (20_000..20_000 + count as i64).collect::<Vec<_>>();
        let codes = (0..count * m)
            .map(|index| ((index * 73 + 19) % ksub) as u8)
            .collect::<Vec<_>>();
        let table = (0..m * ksub)
            .map(|index| ((index * 31 + 5) % 127) as f32 * 0.015625)
            .collect::<Vec<_>>();
        let filter = ids
            .iter()
            .copied()
            .filter(|id| id % 4 != 0)
            .collect::<HashSet<_>>();
        let matching_rows = matching_rows(&ids, Some(&filter)).unwrap();
        assert!(matches!(matching_rows, MatchingRows::Bitmap { .. }));

        let mut expected = matching_rows
            .positions()
            .map(|position| {
                let code = &codes[position * m..(position + 1) * m];
                (
                    dis0 + pq_distance_from_table(&table, code, m, ksub),
                    ids[position],
                )
            })
            .collect::<Vec<_>>();
        expected.sort_unstable_by_key(|&(_, id)| id);

        let mut heap = TopKHeap::new(matching_rows.len());
        scan_codes_batched(
            &table,
            &codes,
            &ids,
            count,
            m,
            ksub,
            dis0,
            Some(&matching_rows),
            &mut heap,
        );
        let mut actual = heap.into_sorted();
        actual.sort_unstable_by_key(|&(_, id)| id);
        assert_eq!(actual, expected);
    }

    #[test]
    fn reader_list_precomputed_table_matches_index_table() {
        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 600;
        let data = generate_clustered_data(n, d, nlist, 44);
        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.build_precomputed_table();
        let pq_norms = index.pq.compute_centroid_norms();

        for list_id in 0..nlist {
            let mut actual = Vec::new();
            fill_list_precomputed_table(
                &index.quantizer_centroids[list_id * d..(list_id + 1) * d],
                &index.pq,
                &pq_norms,
                &mut actual,
            );
            let table_size = m * index.pq.ksub;
            assert_eq!(
                actual,
                index.precomputed_table[list_id * table_size..(list_id + 1) * table_size]
            );
        }
    }

    #[test]
    fn ephemeral_precomputation_requires_matching_probe_work() {
        assert!(!should_use_ephemeral_precomputation(0, 0, 0));
    }

    #[test]
    fn ephemeral_precomputation_respects_batch_memory_budget() {
        let max_values = DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES / std::mem::size_of::<f64>();
        let max_list_values = max_values / 3;
        assert!(ephemeral_precomputed_table_fits_budget(
            1,
            1,
            1,
            max_list_values,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
        ));
        assert!(!ephemeral_precomputed_table_fits_budget(
            1,
            1,
            1,
            max_list_values + 1,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
        ));
        assert!(!ephemeral_precomputed_table_fits_budget(
            0,
            1,
            1,
            1,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
        ));
        assert!(!ephemeral_precomputed_table_fits_budget(
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
        ));
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

        let (ids, dists) = reader.search(&data[0..d], k, nprobe).unwrap();

        assert_eq!(ids, baseline_ids);
        assert_eq!(dists, baseline_dists);
        assert!(
            stats.lock().unwrap().pread_calls > 0,
            "search should still read inverted lists through pread fallback"
        );
    }

    #[test]
    fn test_reader_search_batches_multiple_list_preads() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 8;
        let m = 4;
        let n = 800;
        let k = 5;
        let nprobe = 4;

        let data = generate_clustered_data(n, d, 8, 987);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut buf)).unwrap();

        let stats = Arc::new(Mutex::new(ReaderStats::default()));
        let stream = NonConcurrentPreadCursor::new(buf, Arc::clone(&stats));
        let mut reader = IVFPQIndexReader::open(stream).unwrap();
        reader.ensure_loaded().unwrap();

        {
            let mut stats = stats.lock().unwrap();
            *stats = ReaderStats::default();
        }

        let (_ids, _dists) = reader.search(&data[0..d], k, nprobe).unwrap();

        let stats = stats.lock().unwrap();
        assert!(
            stats.max_ranges_per_batch > 1,
            "multiple probed IVF-PQ lists should share one batched pread"
        );
        assert!(
            stats
                .last_positions
                .windows(2)
                .all(|positions| positions[0] <= positions[1]),
            "fallback readers should receive inverted-list ranges in physical order"
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
    fn test_reader_search_with_roaring_filter_bytes() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};
        use roaring::RoaringTreemap;

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

        let mut allowed = RoaringTreemap::new();
        for id in (0..n as u64).filter(|id| id % 5 == 0) {
            allowed.insert(id);
        }
        let mut filter_bytes = Vec::new();
        allowed.serialize_into(&mut filter_bytes).unwrap();

        let mut reader = IVFPQIndexReader::open(Cursor::new(buf)).unwrap();
        let (result_ids, _) =
            search_with_reader_roaring_filter(&mut reader, &data[0..d], k, 4, &filter_bytes)
                .unwrap();

        for &id in &result_ids {
            assert_eq!(id % 5, 0, "Roaring filter violated: got ID {}", id);
        }
    }

    #[test]
    fn test_reader_search_rejects_invalid_roaring_filter_bytes() {
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
        let err = search_with_reader_roaring_filter(&mut reader, &data[0..d], 5, 4, b"not roaring")
            .unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
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
    fn inner_product_batch_table_reuse_modes_preserve_results_and_control_table_builds() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 600;
        let nq = 8;
        let k = 5;
        let nprobe = nlist;
        let data = generate_clustered_data(n, d, nlist, 43);
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::InnerProduct, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let distance_table_builds = AtomicUsize::new(0);
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();

        let (batch_ids, batch_distances) = search_batch_reader_filter_with_reuse_mode_and_observer(
            &mut reader,
            &data[..nq * d],
            nq,
            k,
            nprobe,
            None,
            IvfPqBatchTableReuseMode::On,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
            |_| {},
            Some(&distance_table_builds),
        )
        .unwrap();

        assert_eq!(distance_table_builds.load(Ordering::Relaxed), nq);

        let distance_table_builds = AtomicUsize::new(0);
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        search_batch_reader_filter_with_reuse_mode_and_observer(
            &mut reader,
            &data[..nq * d],
            nq,
            k,
            nprobe,
            None,
            IvfPqBatchTableReuseMode::On,
            nq * m * 256 * std::mem::size_of::<f32>() - 1,
            |_| {},
            Some(&distance_table_builds),
        )
        .unwrap();
        assert_eq!(
            distance_table_builds.load(Ordering::Relaxed),
            nq * nprobe,
            "the direct path should be used when reused tables exceed the configured budget"
        );

        let mut direct_reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let (direct_ids, direct_distances) = search_batch_reader_with_reuse_mode(
            &mut direct_reader,
            &data[..nq * d],
            nq,
            k,
            nprobe,
            IvfPqBatchTableReuseMode::Off,
        )
        .unwrap();
        assert_eq!(batch_ids, direct_ids);
        assert_eq!(batch_distances, direct_distances);

        for query_index in 0..nq {
            let mut scalar_reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
            let query = &data[query_index * d..(query_index + 1) * d];
            let (scalar_ids, scalar_distances) =
                search_with_reader(&mut scalar_reader, query, k, nprobe).unwrap();
            let result = query_index * k..(query_index + 1) * k;
            assert_eq!(&batch_ids[result.clone()], scalar_ids.as_slice());
            assert_eq!(&batch_distances[result], scalar_distances.as_slice());
        }

        let large_nq = MIN_EPHEMERAL_PRECOMPUTE_QUERIES;
        let distance_table_builds = AtomicUsize::new(0);
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        search_batch_reader_filter_with_reuse_mode_and_observer(
            &mut reader,
            &data[..large_nq * d],
            large_nq,
            k,
            nprobe,
            None,
            IvfPqBatchTableReuseMode::Auto,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
            |_| {},
            Some(&distance_table_builds),
        )
        .unwrap();
        assert_eq!(
            distance_table_builds.load(Ordering::Relaxed),
            large_nq,
            "Auto should reuse tables when the batch and probe work amortize them"
        );

        let distance_table_builds = AtomicUsize::new(0);
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        search_batch_reader_filter_with_reuse_mode_and_observer(
            &mut reader,
            &data[..nq * d],
            nq,
            k,
            nprobe,
            None,
            IvfPqBatchTableReuseMode::Auto,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
            |_| {},
            Some(&distance_table_builds),
        )
        .unwrap();
        assert_eq!(
            distance_table_builds.load(Ordering::Relaxed),
            nq * nprobe,
            "Auto should keep the direct path for small batches"
        );

        let distance_table_builds = AtomicUsize::new(0);
        let empty_filter = HashSet::new();
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();
        search_batch_reader_filter_with_reuse_mode_and_observer(
            &mut reader,
            &data[..nq * d],
            nq,
            k,
            nprobe,
            Some(&empty_filter),
            IvfPqBatchTableReuseMode::On,
            DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
            |_| {},
            Some(&distance_table_builds),
        )
        .unwrap();
        assert_eq!(
            distance_table_builds.load(Ordering::Relaxed),
            0,
            "empty filters should not build query distance tables"
        );
    }

    #[test]
    fn test_batch_reader_search_with_roaring_filter_bytes() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};
        use roaring::RoaringTreemap;
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

        let mut allowed = RoaringTreemap::new();
        for id in (0..n as u64).filter(|id| id % 7 == 0) {
            allowed.insert(id);
        }
        let mut filter_bytes = Vec::new();
        allowed.serialize_into(&mut filter_bytes).unwrap();

        let queries = &data[..nq * d];
        let mut batch_reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let (batch_ids, batch_dists) = search_batch_reader_roaring_filter(
            &mut batch_reader,
            queries,
            nq,
            k,
            nprobe,
            &filter_bytes,
        )
        .unwrap();

        for qi in 0..nq {
            let base = qi * k;
            for &id in &batch_ids[base..base + k] {
                if id >= 0 {
                    assert_eq!(id % 7, 0, "Roaring filter violated: got ID {}", id);
                }
            }

            let mut single_reader = IVFPQIndexReader::open(Cursor::new(buf.clone())).unwrap();
            let query = &queries[qi * d..(qi + 1) * d];
            let (single_ids, single_dists) = search_with_reader_roaring_filter(
                &mut single_reader,
                query,
                k,
                nprobe,
                &filter_bytes,
            )
            .unwrap();

            assert_eq!(&batch_ids[base..base + k], &single_ids[..]);
            assert_eq!(&batch_dists[base..base + k], &single_dists[..]);
        }
    }

    #[test]
    fn test_batch_reader_evaluates_filter_once_per_loaded_row() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 1;
        let m = 4;
        let n = 500;
        let k = 5;
        let nq = 4;
        let nprobe = 1;

        let data = generate_clustered_data(n, d, 1, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let filter = CountingFilter::new();
        let queries = &data[..nq * d];
        let mut reader = IVFPQIndexReader::open(Cursor::new(buf)).unwrap();
        let (result_ids, _) =
            search_batch_reader_filter(&mut reader, queries, nq, k, nprobe, Some(&filter)).unwrap();

        assert!(
            result_ids
                .iter()
                .filter(|&&id| id >= 0)
                .all(|id| id % 7 == 0),
            "filtered batch search returned a disallowed row ID"
        );
        assert_eq!(
            filter.contains_calls.load(Ordering::Relaxed),
            n,
            "the shared filter should be evaluated once per loaded row, not once per query"
        );
    }

    #[test]
    fn filtered_batch_reader_uses_ephemeral_list_precomputation() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};
        use std::sync::atomic::AtomicUsize;

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 600;
        let nq = 64;
        let k = 5;
        let nprobe = nlist;
        let data = generate_clustered_data(n, d, nlist, 45);
        let ids = (0..n as i64).map(|id| 50_000 + id * 3).collect::<Vec<_>>();
        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let filter = ids.iter().copied().step_by(5).collect::<HashSet<_>>();
        let queries = &data[..nq * d];
        let precomputed_lists = AtomicUsize::new(0);
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();

        let (batch_ids, batch_dists) = search_batch_reader_filter_with_observer(
            &mut reader,
            queries,
            nq,
            k,
            nprobe,
            Some(&filter),
            |count| {
                precomputed_lists.fetch_add(count, Ordering::Relaxed);
            },
        )
        .unwrap();

        assert_eq!(precomputed_lists.load(Ordering::Relaxed), nlist);
        assert!(
            reader.precomputed_table.is_empty(),
            "batch-local precomputation must not remain resident on the reader"
        );
        for query_index in 0..nq {
            let mut single_reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
            let query = &queries[query_index * d..(query_index + 1) * d];
            let (single_ids, single_dists) =
                search_with_reader_filter(&mut single_reader, query, k, nprobe, Some(&filter))
                    .unwrap();
            assert_eq!(
                &batch_ids[query_index * k..(query_index + 1) * k],
                single_ids.as_slice()
            );
            for (batch, single) in batch_dists[query_index * k..(query_index + 1) * k]
                .iter()
                .zip(&single_dists)
            {
                // The algebraically equivalent precomputed formula changes
                // floating-point accumulation order. Allow a small absolute
                // floor near zero plus a few ULPs for large distances.
                let tolerance = 1e-4 + 4.0 * f32::EPSILON * single.abs();
                assert!(
                    (batch - single).abs() <= tolerance,
                    "ephemeral precomputation distance {batch} should match direct residual distance {single} within {tolerance}"
                );
            }
        }
    }

    #[test]
    fn unfiltered_batch_reader_uses_ephemeral_list_precomputation() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};
        use std::sync::atomic::AtomicUsize;

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 600;
        let nq = 64;
        let k = 5;
        let nprobe = nlist;
        let data = generate_clustered_data(n, d, nlist, 49);
        let ids = (0..n as i64).map(|id| 70_000 + id * 3).collect::<Vec<_>>();
        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let queries = &data[..nq * d];
        let precomputed_lists = AtomicUsize::new(0);
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();

        let (batch_ids, batch_dists) = search_batch_reader_filter_with_observer(
            &mut reader,
            queries,
            nq,
            k,
            nprobe,
            None,
            |count| {
                precomputed_lists.fetch_add(count, Ordering::Relaxed);
            },
        )
        .unwrap();

        assert_eq!(precomputed_lists.load(Ordering::Relaxed), nlist);
        assert!(
            reader.precomputed_table.is_empty(),
            "batch-local precomputation must not remain resident on the reader"
        );
        for query_index in 0..nq {
            let mut single_reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
            let query = &queries[query_index * d..(query_index + 1) * d];
            let (single_ids, single_dists) =
                search_with_reader_filter(&mut single_reader, query, k, nprobe, None).unwrap();
            assert_eq!(
                &batch_ids[query_index * k..(query_index + 1) * k],
                single_ids.as_slice()
            );
            for (batch, single) in batch_dists[query_index * k..(query_index + 1) * k]
                .iter()
                .zip(&single_dists)
            {
                let tolerance = 1e-4 + 4.0 * f32::EPSILON * single.abs();
                assert!(
                    (batch - single).abs() <= tolerance,
                    "ephemeral precomputation distance {batch} should match direct residual distance {single} within {tolerance}"
                );
            }
        }
    }

    #[test]
    fn forced_ephemeral_reuse_is_stable_for_8bit_large_offsets() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 4;
        let nlist = 4;
        let m = 1;
        let n = 512;
        let nq = MIN_EPHEMERAL_PRECOMPUTE_QUERIES;
        let k = 8;
        let common = 1_000_000.0f32;
        let spread = 500_000.0f32;
        let data = (0..n)
            .flat_map(|row| {
                let cluster = row % nlist;
                let point = row / nlist;
                (0..d).map(move |dimension| {
                    common
                        + cluster as f32 * spread
                        + (((point * 17 + dimension * 13) % 31) as f32 - 15.0) * spread / 16.0
                })
            })
            .collect::<Vec<_>>();
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let queries = &data[..nq * d];
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();
        let (result_ids, result_distances) = search_batch_reader_with_reuse_mode(
            &mut reader,
            queries,
            nq,
            k,
            nlist,
            IvfPqBatchTableReuseMode::On,
        )
        .unwrap();

        let code_size = index.pq.code_size();
        let mut decoded = vec![0.0f32; d];
        for query_index in 0..nq {
            let query = &queries[query_index * d..(query_index + 1) * d];
            for rank in 0..k {
                let offset = query_index * k + rank;
                let id = result_ids[offset];
                let reported = result_distances[offset];
                assert!(
                    reported >= 0.0,
                    "query {query_index} rank {rank} produced negative squared L2 distance {reported}"
                );

                let (list_id, position) = index
                    .ids
                    .iter()
                    .enumerate()
                    .find_map(|(list_id, list_ids)| {
                        list_ids
                            .iter()
                            .position(|candidate| *candidate == id)
                            .map(|position| (list_id, position))
                    })
                    .unwrap();
                let code_offset = position * code_size;
                index.pq.decode(
                    &index.codes[list_id][code_offset..code_offset + code_size],
                    &mut decoded,
                );
                let centroid = &index.quantizer_centroids[list_id * d..(list_id + 1) * d];
                let exact = (0..d)
                    .map(|dimension| {
                        let delta = f64::from(query[dimension])
                            - f64::from(centroid[dimension])
                            - f64::from(decoded[dimension]);
                        delta * delta
                    })
                    .sum::<f64>();
                let tolerance = 1e-3 + 8.0 * f64::from(f32::EPSILON) * exact.abs().max(1.0);
                assert!(
                    (f64::from(reported) - exact).abs() <= tolerance,
                    "query {query_index} rank {rank} reported {reported}, decoded oracle {exact}, tolerance {tolerance}"
                );
            }
        }
    }

    #[test]
    fn small_filtered_batch_reader_skips_ephemeral_list_precomputation() {
        assert_eq!(
            observed_ephemeral_precomputed_lists(
                4,
                4,
                Some(5),
                true,
                46,
                IvfPqBatchTableReuseMode::Auto,
            ),
            0,
            "small batches should keep the direct residual-table path"
        );
    }

    #[test]
    fn single_probe_filtered_batch_reader_skips_ephemeral_list_precomputation() {
        assert_eq!(
            observed_ephemeral_precomputed_lists(
                MIN_EPHEMERAL_PRECOMPUTE_QUERIES,
                1,
                Some(5),
                true,
                47,
                IvfPqBatchTableReuseMode::Auto,
            ),
            0,
            "single-probe batches cannot amortize list precomputation"
        );
    }

    #[test]
    fn empty_filtered_batch_reader_skips_ephemeral_list_precomputation() {
        assert_eq!(
            observed_ephemeral_precomputed_lists(
                MIN_EPHEMERAL_PRECOMPUTE_QUERIES,
                4,
                None,
                true,
                48,
                IvfPqBatchTableReuseMode::Auto,
            ),
            0,
            "lists without matching rows should not be precomputed"
        );
    }

    #[test]
    fn small_unfiltered_batch_reader_skips_ephemeral_list_precomputation() {
        assert_eq!(
            observed_ephemeral_precomputed_lists(
                4,
                4,
                None,
                false,
                50,
                IvfPqBatchTableReuseMode::Auto,
            ),
            0,
            "small unfiltered batches should keep the direct residual-table path"
        );
    }

    #[test]
    fn batch_table_reuse_off_never_precomputes_list_tables() {
        assert_eq!(
            observed_ephemeral_precomputed_lists(
                MIN_EPHEMERAL_PRECOMPUTE_QUERIES,
                4,
                Some(5),
                true,
                51,
                IvfPqBatchTableReuseMode::Off,
            ),
            0,
            "off mode must keep the direct residual-table path"
        );
    }

    #[test]
    fn resident_precomputed_tables_respect_batch_reuse_mode_and_budget() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 600;
        let nq = 4;
        let k = 5;
        let data = generate_clustered_data(n, d, nlist, 53);
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let queries = &data[..nq * d];
        let mut direct_reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let expected = search_batch_reader_with_reuse_mode(
            &mut direct_reader,
            queries,
            nq,
            k,
            nlist,
            IvfPqBatchTableReuseMode::Off,
        )
        .unwrap();

        let mut optimized_reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();
        optimized_reader.optimize_for_search().unwrap();
        assert!(!optimized_reader.precomputed_table.is_empty());
        optimized_reader.precomputed_table.fill(1_000_000_000.0);
        let actual = search_batch_reader_with_reuse_mode(
            &mut optimized_reader,
            queries,
            nq,
            k,
            nlist,
            IvfPqBatchTableReuseMode::Off,
        )
        .unwrap();

        assert_eq!(actual, expected, "Off must ignore resident reuse tables");

        let distance_table_builds = AtomicUsize::new(0);
        let actual = search_batch_reader_filter_with_reuse_mode_and_observer(
            &mut optimized_reader,
            queries,
            nq,
            k,
            nlist,
            None,
            IvfPqBatchTableReuseMode::On,
            1,
            |_| {},
            Some(&distance_table_builds),
        )
        .unwrap();
        assert_eq!(
            actual, expected,
            "resident reuse tables must be ignored when query tables exceed the budget"
        );
        assert_eq!(distance_table_builds.load(Ordering::Relaxed), nq * nlist);
    }

    #[test]
    fn batch_table_reuse_on_precomputes_for_small_batches() {
        assert!(
            observed_ephemeral_precomputed_lists(
                4,
                4,
                Some(5),
                true,
                52,
                IvfPqBatchTableReuseMode::On,
            ) > 0,
            "on mode must bypass the automatic batch-size heuristic"
        );
    }

    #[test]
    fn four_bit_batch_table_reuse_modes_skip_ephemeral_precomputation() {
        for reuse_mode in [IvfPqBatchTableReuseMode::Auto, IvfPqBatchTableReuseMode::On] {
            assert_eq!(
                observed_ephemeral_precomputed_lists_with_nbits(
                    4,
                    MIN_EPHEMERAL_PRECOMPUTE_QUERIES,
                    4,
                    None,
                    false,
                    54,
                    reuse_mode,
                ),
                0,
                "4-bit {reuse_mode:?} must keep the existing scan path"
            );
        }
    }

    #[test]
    fn four_bit_auto_batch_table_reuse_matches_off() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 600;
        let nq = MIN_EPHEMERAL_PRECOMPUTE_QUERIES;
        let k = 10;
        let nprobe = nlist;
        let data = generate_clustered_data(n, d, nlist, 55);
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFPQIndex::with_nbits(d, nlist, m, 4, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let queries = &data[..nq * d];

        let mut off_reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let expected = search_batch_reader_with_reuse_mode(
            &mut off_reader,
            queries,
            nq,
            k,
            nprobe,
            IvfPqBatchTableReuseMode::Off,
        )
        .unwrap();

        let mut auto_reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();
        let actual = search_batch_reader_with_reuse_mode(
            &mut auto_reader,
            queries,
            nq,
            k,
            nprobe,
            IvfPqBatchTableReuseMode::Auto,
        )
        .unwrap();

        assert_eq!(
            actual, expected,
            "4-bit Auto must preserve the Off path results"
        );
    }

    #[test]
    fn test_batch_reader_empty_roaring_filter_returns_empty_results() {
        use crate::io::{write_index, IVFPQIndexReader, PosWriter};
        use roaring::RoaringTreemap;
        use std::io::Cursor;

        let d = 16;
        let nlist = 4;
        let m = 4;
        let n = 500;
        let k = 5;
        let nq = 4;
        let nprobe = 2;

        let data = generate_clustered_data(n, d, 4, 42);
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let empty = RoaringTreemap::new();
        let mut filter_bytes = Vec::new();
        empty.serialize_into(&mut filter_bytes).unwrap();

        let queries = &data[..nq * d];
        let mut reader = IVFPQIndexReader::open(Cursor::new(buf)).unwrap();
        let (batch_ids, batch_dists) =
            search_batch_reader_roaring_filter(&mut reader, queries, nq, k, nprobe, &filter_bytes)
                .unwrap();

        assert!(batch_ids.iter().all(|&id| id == -1));
        assert!(batch_dists.iter().all(|&dist| dist == f32::MAX));
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

    #[test]
    #[should_panic(expected = "4-bit IVF-PQ requires even m")]
    fn ivfpq_rejects_odd_4bit_subquantizer_count_at_construction() {
        let _ = IVFPQIndex::with_nbits(12, 4, 3, 4, MetricType::L2, false);
    }
}
