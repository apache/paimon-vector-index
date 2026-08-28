// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::distance::{fvec_madd, fvec_norm_l2sqr, preprocess_vectors, MetricType};
use crate::ivfpq::RowIdFilter;
use crate::kmeans::{self, KMeansConfig};
use crate::logging::{emit_log, LogLevel};
use crate::rq::{
    RQEncodeScratch, RQQueryContext, RQRotation, RQVectorFactors, RaBitQuantizer, DEFAULT_RQ_BITS,
    DEFAULT_RQ_ROTATION_ROUNDS, DEFAULT_RQ_ROTATION_SEED,
};
use crate::topk::TopKHeap;
use rayon::prelude::*;
use std::borrow::Cow;
use std::time::{Duration, Instant};

const APPROX_ASSIGN_SEARCH_LIST: usize = 15;
const APPROX_ASSIGN_MIN_CENTROID_VALUES: usize = 1_000_000;

fn use_approximate_assignment(d: usize, nlist: usize) -> bool {
    d.saturating_mul(nlist) >= APPROX_ASSIGN_MIN_CENTROID_VALUES
}

pub(crate) fn build_timing_enabled() -> bool {
    std::env::var_os("PAIMON_LOG_VECTOR_INDEX_BUILD_TIMING").is_some()
}

pub(crate) fn log_build_timing(enabled: bool, stage: &str, started: Instant) {
    log_build_elapsed(enabled, stage, started.elapsed());
}

pub(crate) fn log_build_elapsed(enabled: bool, stage: &str, elapsed: Duration) {
    if enabled {
        emit_log(
            LogLevel::Info,
            &format!(
                "ivf_rq_build stage={stage} elapsed_ms={:.3}",
                elapsed.as_secs_f64() * 1_000.0
            ),
        );
    }
}

pub struct IVFRQIndex {
    pub d: usize,
    pub padded_d: usize,
    pub nlist: usize,
    pub bits: usize,
    pub metric: MetricType,
    pub quantizer_centroids: Vec<f32>,
    pub quantizer_centroid_norms: Vec<f32>,
    pub rotated_centroids: Vec<f32>,
    pub rotation_seed: u64,
    pub rotation_rounds: u32,
    pub ids: Vec<Vec<i64>>,
    pub codes: Vec<Vec<u8>>,
    pub factors: Vec<Vec<RQVectorFactors>>,
    quantizer: RaBitQuantizer,
    rotation: RQRotation,
    assign_graph: Option<crate::vamana::VamanaGraph>,
}

impl IVFRQIndex {
    pub fn new(d: usize, nlist: usize, metric: MetricType) -> Self {
        Self::with_options(
            d,
            nlist,
            DEFAULT_RQ_BITS,
            metric,
            DEFAULT_RQ_ROTATION_SEED,
            DEFAULT_RQ_ROTATION_ROUNDS,
        )
    }

    pub fn with_bits(d: usize, nlist: usize, bits: usize, metric: MetricType) -> Self {
        Self::with_options(
            d,
            nlist,
            bits,
            metric,
            DEFAULT_RQ_ROTATION_SEED,
            DEFAULT_RQ_ROTATION_ROUNDS,
        )
    }

    pub fn with_options(
        d: usize,
        nlist: usize,
        bits: usize,
        metric: MetricType,
        rotation_seed: u64,
        rotation_rounds: u32,
    ) -> Self {
        let quantizer = RaBitQuantizer::new(d, bits);
        let padded_d = quantizer.padded_dimension();
        Self {
            d,
            padded_d,
            nlist,
            bits,
            metric,
            quantizer_centroids: Vec::new(),
            quantizer_centroid_norms: Vec::new(),
            rotated_centroids: Vec::new(),
            rotation_seed,
            rotation_rounds,
            ids: vec![Vec::new(); nlist],
            codes: vec![Vec::new(); nlist],
            factors: vec![Vec::new(); nlist],
            quantizer,
            rotation: RQRotation::new(d, rotation_seed, rotation_rounds),
            assign_graph: None,
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        let timing = build_timing_enabled();
        let total_started = Instant::now();
        let phase_started = Instant::now();
        let processed = self.preprocess_vectors(data, n);
        log_build_timing(timing, "train.preprocess", phase_started);

        let phase_started = Instant::now();
        self.quantizer_centroids =
            kmeans::kmeans_train(&KMeansConfig::default(), &processed, n, self.d, self.nlist);
        log_build_timing(timing, "train.kmeans", phase_started);

        let phase_started = Instant::now();
        self.quantizer_centroid_norms = self
            .quantizer_centroids
            .chunks_exact(self.d)
            .map(fvec_norm_l2sqr)
            .collect();
        log_build_timing(timing, "train.centroid_norms", phase_started);

        let phase_started = Instant::now();
        self.rotated_centroids = vec![0.0; self.nlist * self.padded_d];
        let mut scratch = vec![0.0; self.padded_d];
        for list_id in 0..self.nlist {
            let centroid = &self.quantizer_centroids[list_id * self.d..(list_id + 1) * self.d];
            self.rotation.rotate(
                centroid,
                &mut self.rotated_centroids[list_id * self.padded_d..(list_id + 1) * self.padded_d],
                &mut scratch,
            );
        }
        log_build_timing(timing, "train.rotate_centroids", phase_started);

        let phase_started = Instant::now();
        self.assign_graph = None;
        if use_approximate_assignment(self.d, self.nlist) {
            let params = crate::diskann::DiskAnnBuildParams {
                max_degree: 12,
                build_search_list_size: APPROX_ASSIGN_SEARCH_LIST,
                alpha: 1.2,
                seed: 42,
                memory_budget_bytes: 1024 * 1024 * 1024,
                storage_layout: crate::diskann::DiskAnnStorageLayout::Compact,
                raw_vector_encoding: crate::diskann::DiskAnnRawVectorEncoding::F32,
                build_distance: crate::diskann::DiskAnnBuildDistance::FullPrecision,
            };
            match crate::vamana::VamanaGraph::build(
                &self.quantizer_centroids,
                self.nlist,
                self.d,
                params,
            ) {
                Ok(graph) => self.assign_graph = Some(graph),
                Err(error) => emit_log(
                    LogLevel::Warn,
                    &format!("automatic IVF-RQ approximate assignment disabled: {error}"),
                ),
            }
        }
        log_build_timing(timing, "train.assign_graph", phase_started);
        log_build_timing(timing, "train.total", total_started);
    }

    pub fn add(&mut self, data: &[f32], ids: &[i64], n: usize) {
        let timing = build_timing_enabled();
        let total_started = Instant::now();
        let phase_started = Instant::now();
        let processed = self.preprocess_vectors(data, n);
        log_build_timing(timing, "add.preprocess", phase_started);

        let phase_started = Instant::now();
        let list_ids = match &self.assign_graph {
            Some(graph) => {
                let mut list_ids = vec![0usize; n];
                let chunk = (n / (rayon::current_num_threads() * 4).max(1)).clamp(16, 1024);
                list_ids.par_chunks_mut(chunk).enumerate().for_each_init(
                    || graph.search_scratch(APPROX_ASSIGN_SEARCH_LIST),
                    |scratch, (chunk_idx, chunk_ids)| {
                        let row0 = chunk_idx * chunk;
                        for (i, list_id) in chunk_ids.iter_mut().enumerate() {
                            let row = row0 + i;
                            *list_id = graph
                                .greedy_search_best_with_scratch(
                                    &self.quantizer_centroids,
                                    self.d,
                                    &processed[row * self.d..(row + 1) * self.d],
                                    APPROX_ASSIGN_SEARCH_LIST,
                                    scratch,
                                )
                                .map(|node| node.id as usize)
                                .unwrap_or(0);
                        }
                    },
                );
                list_ids
            }
            None => kmeans::find_nearest_batch(
                &processed,
                n,
                &self.quantizer_centroids,
                self.nlist,
                self.d,
            ),
        };
        log_build_timing(timing, "add.assign", phase_started);

        let phase_started = Instant::now();
        let mut list_rows = vec![Vec::new(); self.nlist];
        for (row, list_id) in list_ids.into_iter().enumerate() {
            list_rows[list_id].push(row);
        }
        log_build_timing(timing, "add.group", phase_started);

        let phase_started = Instant::now();
        let d = self.d;
        let padded_d = self.padded_d;
        let metric = self.metric;
        let centroids = &self.quantizer_centroids;
        let rotated_centroids = &self.rotated_centroids;
        let quantizer = &self.quantizer;
        let rotation = &self.rotation;
        let output_ids = &mut self.ids;
        let output_codes = &mut self.codes;
        let output_factors = &mut self.factors;
        if n > 1_000 && self.nlist > 1 {
            output_ids
                .par_iter_mut()
                .zip(output_codes.par_iter_mut())
                .zip(output_factors.par_iter_mut())
                .zip(list_rows.into_par_iter())
                .enumerate()
                .for_each_init(
                    || IVFRQEncodeScratch::new(d, padded_d, quantizer.code_size()),
                    |scratch, (list_id, (((list_ids, list_codes), list_factors), rows))| {
                        append_encoded_rows(
                            &processed,
                            ids,
                            &rows,
                            d,
                            &centroids[list_id * d..(list_id + 1) * d],
                            &rotated_centroids[list_id * padded_d..(list_id + 1) * padded_d],
                            metric,
                            quantizer,
                            rotation,
                            list_ids,
                            list_codes,
                            list_factors,
                            scratch,
                        );
                    },
                );
        } else {
            let mut scratch = IVFRQEncodeScratch::new(d, padded_d, quantizer.code_size());
            for (list_id, (((list_ids, list_codes), list_factors), rows)) in output_ids
                .iter_mut()
                .zip(output_codes.iter_mut())
                .zip(output_factors.iter_mut())
                .zip(list_rows)
                .enumerate()
            {
                append_encoded_rows(
                    &processed,
                    ids,
                    &rows,
                    d,
                    &centroids[list_id * d..(list_id + 1) * d],
                    &rotated_centroids[list_id * padded_d..(list_id + 1) * padded_d],
                    metric,
                    quantizer,
                    rotation,
                    list_ids,
                    list_codes,
                    list_factors,
                    &mut scratch,
                );
            }
        }
        log_build_timing(timing, "add.encode", phase_started);
        log_build_timing(timing, "add.total", total_started);
    }

    pub fn total_vectors(&self) -> usize {
        self.ids.iter().map(Vec::len).sum()
    }

    pub fn code_size(&self) -> usize {
        self.quantizer.code_size()
    }

    pub fn plane_size(&self) -> usize {
        self.quantizer.plane_size()
    }

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
        let processed_queries = self.preprocess_vectors(queries, nq);
        let (all_probe_indices, all_probe_distances) = kmeans::find_topk_batch_with_centroid_norms(
            &processed_queries,
            nq,
            &self.quantizer_centroids,
            &self.quantizer_centroid_norms,
            self.nlist,
            self.d,
            nprobe,
        );
        let mut rotated_query = vec![0.0; self.padded_d];
        let mut rotation_scratch = vec![0.0; self.padded_d];

        for qi in 0..nq {
            let query = &processed_queries[qi * self.d..(qi + 1) * self.d];
            self.rotation
                .rotate(query, &mut rotated_query, &mut rotation_scratch);
            let query_context = self.quantizer.prepare_query(rotated_query.clone());
            let query_norm_sqr = fvec_norm_l2sqr(query);
            let mut heap = TopKHeap::new(k);
            for (&list_id, &coarse_distance) in
                all_probe_indices[qi].iter().zip(&all_probe_distances[qi])
            {
                let query_terms = self.quantizer.query_terms_from_coarse_distance(
                    coarse_distance,
                    query_norm_sqr,
                    self.quantizer_centroid_norms[list_id],
                    self.metric,
                );
                self.scan_list(&query_context, query_terms, list_id, filter, &mut heap);
            }

            let sorted = heap.into_sorted();
            let out_base = qi * k;
            for (i, &(dist, id)) in sorted.iter().enumerate() {
                result_distances[out_base + i] = dist;
                result_labels[out_base + i] = id;
            }
            for i in sorted.len()..k {
                result_distances[out_base + i] = f32::MAX;
                result_labels[out_base + i] = -1;
            }
        }
    }

    pub(crate) fn preprocess_vectors<'a>(&self, data: &'a [f32], n: usize) -> Cow<'a, [f32]> {
        match self.metric {
            MetricType::Cosine => {
                Cow::Owned(preprocess_vectors(data, n, self.d, MetricType::Cosine))
            }
            MetricType::L2 | MetricType::InnerProduct => Cow::Borrowed(&data[..n * self.d]),
        }
    }

    fn scan_list(
        &self,
        query_context: &RQQueryContext,
        query_terms: crate::rq::RQQueryTerms,
        list_id: usize,
        filter: Option<&dyn RowIdFilter>,
        heap: &mut TopKHeap,
    ) {
        let code_size = self.code_size();
        for (local_idx, &id) in self.ids[list_id].iter().enumerate() {
            if filter.map(|f| !f.contains(id)).unwrap_or(false) {
                continue;
            }
            let code = &self.codes[list_id][local_idx * code_size..(local_idx + 1) * code_size];
            let factors = self.factors[list_id][local_idx];
            if self.bits == 1 {
                let distance = self.quantizer.estimate(
                    self.quantizer.coarse_inner_product(query_context, code),
                    factors.coarse,
                    query_terms,
                );
                if heap.should_consider(distance) {
                    heap.push(distance, id);
                }
                continue;
            }

            let coarse = self.quantizer.estimate(
                self.quantizer.coarse_inner_product(query_context, code),
                factors.coarse,
                query_terms,
            );
            let lower = self
                .quantizer
                .lower_bound(coarse, factors.coarse, query_terms);
            if heap.should_consider(lower) {
                let distance = self.quantizer.estimate(
                    self.quantizer.full_inner_product(query_context, code),
                    factors.full,
                    query_terms,
                );
                if heap.should_consider(distance) {
                    heap.push(distance, id);
                }
            }
        }
    }
}

struct IVFRQEncodeScratch {
    residual: Vec<f32>,
    rotated_residual: Vec<f32>,
    rotation: Vec<f32>,
    code: Vec<u8>,
    encode: RQEncodeScratch,
}

impl IVFRQEncodeScratch {
    fn new(d: usize, padded_d: usize, code_size: usize) -> Self {
        Self {
            residual: vec![0.0; d],
            rotated_residual: vec![0.0; padded_d],
            rotation: vec![0.0; padded_d],
            code: vec![0; code_size],
            encode: RQEncodeScratch::new(padded_d),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_encoded_rows(
    data: &[f32],
    input_ids: &[i64],
    rows: &[usize],
    d: usize,
    centroid: &[f32],
    rotated_centroid: &[f32],
    metric: MetricType,
    quantizer: &RaBitQuantizer,
    rotation: &RQRotation,
    output_ids: &mut Vec<i64>,
    output_codes: &mut Vec<u8>,
    output_factors: &mut Vec<RQVectorFactors>,
    scratch: &mut IVFRQEncodeScratch,
) {
    let code_size = quantizer.code_size();
    output_ids.reserve(rows.len());
    output_codes.reserve(rows.len().saturating_mul(code_size));
    output_factors.reserve(rows.len());
    for &row in rows {
        let vector = &data[row * d..(row + 1) * d];
        fvec_madd(vector, centroid, -1.0, &mut scratch.residual);
        rotation.rotate(
            &scratch.residual,
            &mut scratch.rotated_residual,
            &mut scratch.rotation,
        );
        let factors = quantizer.encode_with_scratch(
            &scratch.rotated_residual,
            rotated_centroid,
            metric,
            &mut scratch.code,
            &mut scratch.encode,
        );
        output_ids.push(input_ids[row]);
        output_codes.extend_from_slice(&scratch.code);
        output_factors.push(factors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivfrq_vamana_assignment_matches_exact_on_connected_graph() {
        assert!(use_approximate_assignment(768, 4096));
        assert!(!use_approximate_assignment(768, 1024));

        let d = 4;
        let nlist = 4;
        let n = 64;
        let data = (0..n)
            .flat_map(|row| {
                (0..d).map(move |dimension| {
                    (row % nlist) as f32 * 10.0 + row as f32 * 0.01 + dimension as f32 * 0.1
                })
            })
            .collect::<Vec<_>>();
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, n);
        let expected = kmeans::find_nearest_batch(&data, n, &index.quantizer_centroids, nlist, d);
        let adjacency = (0..nlist)
            .map(|node| {
                (0..nlist)
                    .filter(|&neighbor| neighbor != node)
                    .map(|neighbor| neighbor as u32)
                    .collect()
            })
            .collect();
        index.assign_graph = Some(crate::vamana::VamanaGraph::from_adjacency(0, adjacency));
        index.add(&data, &ids, n);

        let mut actual = vec![usize::MAX; n];
        for (list_id, list_ids) in index.ids.iter().enumerate() {
            for &id in list_ids {
                actual[id as usize] = list_id;
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn ivfrq_only_allocates_preprocessed_vectors_for_cosine() {
        let data = vec![3.0, 4.0, 1.0, 2.0];
        let l2 = IVFRQIndex::new(2, 1, MetricType::L2);
        let ip = IVFRQIndex::new(2, 1, MetricType::InnerProduct);
        let cosine = IVFRQIndex::new(2, 1, MetricType::Cosine);

        assert!(matches!(l2.preprocess_vectors(&data, 2), Cow::Borrowed(_)));
        assert!(matches!(ip.preprocess_vectors(&data, 2), Cow::Borrowed(_)));
        assert!(matches!(cosine.preprocess_vectors(&data, 2), Cow::Owned(_)));
    }

    #[test]
    fn ivfrq_four_bit_recalls_query_vector_without_dimension_alignment_requirement() {
        let d = 13;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                (0..d).map(move |dim| cluster + i as f32 * 0.01 + dim as f32)
            })
            .collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut distances = vec![0.0; 5];
        let mut labels = vec![0; 5];
        index.search(
            &data[7 * d..8 * d],
            1,
            5,
            nlist,
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels[0], ids[7]);
        assert!(distances[0] <= 1e-3);
        assert_eq!(index.padded_d, 64);
    }

    #[test]
    fn ivfrq_inner_product_recalls_query_vector() {
        let d = 64;
        let n = d;
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            data[i * d + i] = 1.0;
        }
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let mut index = IVFRQIndex::with_bits(d, 1, 4, MetricType::InnerProduct);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let query_id = 37;
        let mut distances = vec![0.0; 5];
        let mut labels = vec![0; 5];
        index.search(
            &data[query_id * d..(query_id + 1) * d],
            1,
            5,
            1,
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels[0], ids[query_id]);
    }

    #[test]
    fn ivfrq_parallel_add_matches_incremental_serial_add() {
        let d = 65;
        let nlist = 8;
        let n = 2_048;
        let data = (0..n)
            .flat_map(|row| {
                (0..d).map(move |dimension| {
                    (row % nlist) as f32 * 100.0 + row as f32 * 0.003 + dimension as f32 * 0.07
                })
            })
            .collect::<Vec<_>>();
        let ids = (10_000..10_000 + n as i64).collect::<Vec<_>>();
        let mut parallel = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        let mut serial = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        parallel.train(&data, n);
        serial.train(&data, n);

        parallel.add(&data, &ids, n);
        for (data_chunk, id_chunk) in data.chunks_exact(512 * d).zip(ids.chunks_exact(512)) {
            serial.add(data_chunk, id_chunk, 512);
        }

        assert_eq!(parallel.ids, serial.ids);
        assert_eq!(parallel.codes, serial.codes);
        assert_eq!(parallel.factors, serial.factors);
    }
}
