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

//! IVF with per-dimension 8-bit residual scalar quantization.

use crate::coarse::CoarseAssignment;
use crate::distance::{preprocess_vectors, MetricType};
use crate::ivfpq::RowIdFilter;
use crate::kmeans::{self, KMeansConfig};
use crate::sq::ScalarQuantizer;
use crate::topk::TopKHeap;
use rayon::prelude::*;
use std::borrow::Cow;

pub struct IVFSQIndex {
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    quantizer_centroids: Vec<f32>,
    pub sq: ScalarQuantizer,
    pub list_sqs: Vec<ScalarQuantizer>,
    pub ids: Vec<Vec<i64>>,
    pub codes: Vec<Vec<u8>>,
    coarse_assignment: CoarseAssignment,
}

impl IVFSQIndex {
    pub fn new(d: usize, nlist: usize, metric: MetricType) -> Self {
        Self {
            d,
            nlist,
            metric,
            quantizer_centroids: Vec::new(),
            sq: ScalarQuantizer::new(d),
            list_sqs: vec![ScalarQuantizer::new(d); nlist],
            ids: vec![Vec::new(); nlist],
            codes: vec![Vec::new(); nlist],
            coarse_assignment: CoarseAssignment::default(),
        }
    }

    pub fn quantizer_centroids(&self) -> &[f32] {
        &self.quantizer_centroids
    }

    /// Enables automatic Vamana coarse assignment for large centroid matrices.
    /// Disable it to keep vector assignment exact.
    pub(crate) fn set_approximate_coarse_assignment(&mut self, enabled: bool) {
        assert!(
            self.ids.iter().all(Vec::is_empty),
            "cannot change coarse assignment after vectors have been added"
        );
        self.coarse_assignment.set_approximate_enabled(enabled);
    }

    pub fn set_quantizer_centroids(&mut self, centroids: Vec<f32>) {
        assert_eq!(
            centroids.len(),
            self.nlist * self.d,
            "quantizer centroids must hold nlist * d values"
        );
        assert!(
            self.ids.iter().all(Vec::is_empty),
            "cannot replace quantizer centroids after vectors have been added"
        );
        self.quantizer_centroids = centroids;
        self.coarse_assignment.reset();
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        self.train_with_config(data, n, &KMeansConfig::default())
    }

    pub fn train_with_config(&mut self, data: &[f32], n: usize, config: &KMeansConfig) {
        let processed = self.preprocess_vectors(data, n);
        self.quantizer_centroids = kmeans::kmeans_train(config, &processed, n, self.d, self.nlist);
        self.coarse_assignment.reset();
        let list_ids = self.coarse_assignment.assign(
            &processed,
            n,
            &self.quantizer_centroids,
            self.nlist,
            self.d,
        );
        self.train_list_sqs(&processed, &list_ids);
    }

    pub fn add(&mut self, data: &[f32], ids: &[i64], n: usize) {
        let processed = self.preprocess_vectors(data, n);
        let list_ids = self.coarse_assignment.assign(
            &processed,
            n,
            &self.quantizer_centroids,
            self.nlist,
            self.d,
        );
        let mut list_rows = vec![Vec::new(); self.nlist];
        for (row, list_id) in list_ids.into_iter().enumerate() {
            list_rows[list_id].push(row);
        }

        let d = self.d;
        let centroids = &self.quantizer_centroids;
        let list_sqs = &self.list_sqs;
        let output_ids = &mut self.ids;
        let output_codes = &mut self.codes;
        if n > 1_000 && self.nlist > 1 {
            output_ids
                .par_iter_mut()
                .zip(output_codes.par_iter_mut())
                .zip(list_rows.into_par_iter())
                .enumerate()
                .for_each(|(list_id, ((list_ids, list_codes), rows))| {
                    append_encoded_rows(
                        &processed,
                        ids,
                        &rows,
                        d,
                        &centroids[list_id * d..(list_id + 1) * d],
                        &list_sqs[list_id],
                        list_ids,
                        list_codes,
                    );
                });
        } else {
            for (list_id, ((list_ids, list_codes), rows)) in output_ids
                .iter_mut()
                .zip(output_codes.iter_mut())
                .zip(list_rows)
                .enumerate()
            {
                append_encoded_rows(
                    &processed,
                    ids,
                    &rows,
                    d,
                    &centroids[list_id * d..(list_id + 1) * d],
                    &list_sqs[list_id],
                    list_ids,
                    list_codes,
                );
            }
        }
    }

    pub fn total_vectors(&self) -> usize {
        self.ids.iter().map(Vec::len).sum()
    }

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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
        let (all_probe_indices, _) = kmeans::find_topk_batch(
            &processed_queries,
            nq,
            &self.quantizer_centroids,
            self.nlist,
            self.d,
            nprobe,
        );

        for qi in 0..nq {
            let query = &processed_queries[qi * self.d..(qi + 1) * self.d];
            let mut heap = TopKHeap::new(k);
            for &list_id in &all_probe_indices[qi] {
                self.scan_list(query, list_id, filter, &mut heap);
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
        query: &[f32],
        list_id: usize,
        filter: Option<&dyn RowIdFilter>,
        heap: &mut TopKHeap,
    ) {
        let sq = self.list_sq(list_id);
        let context = sq.distance_context(query, self.metric);
        let centroid = self.list_centroid(list_id);
        let code_size = self.code_size();
        for (local_id, &row_id) in self.ids[list_id].iter().enumerate() {
            if filter.map(|f| !f.contains(row_id)).unwrap_or(false) {
                continue;
            }
            let code = &self.codes[list_id][local_id * code_size..(local_id + 1) * code_size];
            let distance =
                sq.distance_to_code_with_offset_with_context(query, code, centroid, context);
            if heap.should_consider(distance) {
                heap.push(distance, row_id);
            }
        }
    }

    fn train_list_sqs(&mut self, data: &[f32], list_ids: &[usize]) {
        if list_ids.is_empty() {
            self.sq = ScalarQuantizer::new(self.d);
            self.list_sqs = vec![self.sq.clone(); self.nlist];
            return;
        }
        let mut list_rows = vec![Vec::new(); self.nlist];
        for (row, &list_id) in list_ids.iter().enumerate() {
            list_rows[list_id].push(row);
        }
        let trained = list_rows
            .par_iter()
            .enumerate()
            .map(|(list_id, rows)| {
                (!rows.is_empty()).then(|| {
                    ScalarQuantizer::train_residual_rows(data, rows, self.list_centroid(list_id))
                })
            })
            .collect::<Vec<_>>();
        let mut mins = vec![f32::INFINITY; self.d];
        let mut maxs = vec![f32::NEG_INFINITY; self.d];
        for sq in trained.iter().flatten() {
            for dim in 0..self.d {
                mins[dim] = mins[dim].min(sq.mins[dim]);
                maxs[dim] = maxs[dim].max(sq.maxs[dim]);
            }
        }
        self.sq = ScalarQuantizer::with_dimension_bounds(self.d, mins, maxs);
        // A training sample may contain only a handful of rows in a partition.
        // Its extrema severely clip unseen residuals (including constant sample
        // dimensions). Pool the observed residual bounds across partitions;
        // retain per-list metadata so existing v1 files keep their own bounds.
        self.list_sqs = vec![self.sq.clone(); self.nlist];
    }

    pub(crate) fn list_centroid(&self, list_id: usize) -> &[f32] {
        &self.quantizer_centroids[list_id * self.d..(list_id + 1) * self.d]
    }

    pub(crate) fn list_sq(&self, list_id: usize) -> &ScalarQuantizer {
        self.list_sqs.get(list_id).unwrap_or(&self.sq)
    }

    pub(crate) fn code_size(&self) -> usize {
        self.sq.code_size()
    }
}

#[allow(clippy::too_many_arguments)]
fn append_encoded_rows(
    data: &[f32],
    input_ids: &[i64],
    rows: &[usize],
    d: usize,
    centroid: &[f32],
    sq: &ScalarQuantizer,
    output_ids: &mut Vec<i64>,
    output_codes: &mut Vec<u8>,
) {
    output_ids.reserve(rows.len());
    if rows.is_empty() {
        return;
    }
    let start = output_codes.len();
    output_codes.resize(start + rows.len() * d, 0);
    sq.encode_residual_rows(data, rows, centroid, &mut output_codes[start..]);
    output_ids.extend(rows.iter().map(|&row| input_ids[row]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::fvec_madd;
    use std::collections::HashSet;

    #[test]
    fn sparse_partition_bounds_do_not_collapse_unseen_residuals() {
        let mut index = IVFSQIndex::new(2, 3, MetricType::L2);
        index.set_quantizer_centroids(vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0]);
        index.train_list_sqs(&[-2.0, -2.0, 10.0, 10.0, 12.0, 12.0], &[0, 1, 1]);
        // Partition zero saw only one residual and partition two was empty.
        // Neither should freeze future vectors at a sample's constant value.
        index.add(&[0.0, 0.0, 20.0, 20.0], &[42, 43], 2);
        let mut distances = [0.0; 2];
        let mut labels = [0; 2];
        index.search(
            &[0.0, 0.0, 20.0, 20.0],
            2,
            1,
            3,
            &mut distances,
            &mut labels,
        );
        assert_eq!(labels, [42, 43]);
        assert!(
            distances.iter().all(|&distance| distance < 0.001),
            "{distances:?}"
        );
    }

    #[test]
    fn ivfsq_full_scan_recalls_added_vector() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [
                    cluster + i as f32 * 2.0,
                    10.0 + i as f32,
                    20.0 + i as f32,
                    30.0 + i as f32,
                ]
            })
            .collect();
        let ids: Vec<i64> = (10_000..10_000 + n as i64).collect();
        let mut index = IVFSQIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut labels = vec![-1; 5];
        let mut distances = vec![f32::MAX; 5];
        let query_id = 23;
        index.search(
            &data[query_id * d..(query_id + 1) * d],
            1,
            5,
            nlist,
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels[0], ids[query_id]);
        assert!(distances[0].is_finite());
    }

    #[test]
    fn ivfsq_filter_is_applied_during_scan() {
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];
        let mut index = IVFSQIndex::new(2, 1, MetricType::L2);
        index.train(&data, 3);
        index.add(&data, &ids, 3);
        let allowed = HashSet::from([12]);

        let mut labels = vec![-1; 2];
        let mut distances = vec![f32::MAX; 2];
        index.search_with_filter(
            &[0.0, 0.0],
            1,
            2,
            1,
            Some(&allowed),
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels, vec![12, -1]);
    }

    #[test]
    fn ivfsq_preprocessing_borrows_non_cosine_input() {
        let data = vec![3.0, 4.0, 1.0, 2.0];
        let l2 = IVFSQIndex::new(2, 1, MetricType::L2);
        let ip = IVFSQIndex::new(2, 1, MetricType::InnerProduct);
        let cosine = IVFSQIndex::new(2, 1, MetricType::Cosine);

        assert!(matches!(l2.preprocess_vectors(&data, 2), Cow::Borrowed(_)));
        assert!(matches!(ip.preprocess_vectors(&data, 2), Cow::Borrowed(_)));
        assert!(matches!(cosine.preprocess_vectors(&data, 2), Cow::Owned(_)));
    }

    #[test]
    fn ivfsq_parallel_incremental_add_keeps_ids_aligned_with_codes() {
        let d = 8;
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
        let mut index = IVFSQIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data[..1_024 * d], &ids[..1_024], 1_024);
        index.add(&data[1_024 * d..], &ids[1_024..], 1_024);

        let mut residual = vec![0.0f32; d];
        let mut expected_code = vec![0u8; d];
        for list_id in 0..nlist {
            for (position, &row_id) in index.ids[list_id].iter().enumerate() {
                let row = (row_id - 10_000) as usize;
                fvec_madd(
                    &data[row * d..(row + 1) * d],
                    index.list_centroid(list_id),
                    -1.0,
                    &mut residual,
                );
                index.list_sq(list_id).encode(&residual, &mut expected_code);
                assert_eq!(
                    &index.codes[list_id][position * d..(position + 1) * d],
                    expected_code,
                    "row ID {row_id} was paired with another row's code"
                );
            }
        }
    }
}
