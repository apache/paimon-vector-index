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

//! IVF with per-list, per-dimension 8-bit residual scalar quantization.

use crate::coarse::CoarseAssignment;
use crate::distance::{fvec_madd, preprocess_vectors, MetricType};
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
        let processed = self.preprocess_vectors(data, n);
        self.quantizer_centroids =
            kmeans::kmeans_train(&KMeansConfig::default(), &processed, n, self.d, self.nlist);
        self.coarse_assignment.reset();
        let (list_ids, residuals) = self.assign_residuals(&processed, n);
        self.sq.train(&residuals, n);
        self.train_list_sqs(&list_ids, &residuals);
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

    fn assign_residuals(&mut self, processed: &[f32], n: usize) -> (Vec<usize>, Vec<f32>) {
        let list_ids = self.coarse_assignment.assign(
            processed,
            n,
            &self.quantizer_centroids,
            self.nlist,
            self.d,
        );
        let mut residuals = vec![0.0f32; n * self.d];
        for i in 0..n {
            let vector = &processed[i * self.d..(i + 1) * self.d];
            self.write_residual(
                vector,
                list_ids[i],
                &mut residuals[i * self.d..(i + 1) * self.d],
            );
        }
        (list_ids, residuals)
    }

    fn train_list_sqs(&mut self, list_ids: &[usize], residuals: &[f32]) {
        let mut list_residuals = vec![Vec::new(); self.nlist];
        for (i, &list_id) in list_ids.iter().enumerate() {
            let residual = &residuals[i * self.d..(i + 1) * self.d];
            list_residuals[list_id].extend_from_slice(residual);
        }
        self.list_sqs = vec![self.sq.clone(); self.nlist];
        for (list_id, values) in list_residuals.iter().enumerate() {
            if !values.is_empty() {
                let mut sq = ScalarQuantizer::new(self.d);
                sq.train(values, values.len() / self.d);
                self.list_sqs[list_id] = sq;
            }
        }
    }

    fn write_residual(&self, vector: &[f32], list_id: usize, out: &mut [f32]) {
        fvec_madd(vector, self.list_centroid(list_id), -1.0, out);
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
    output_codes.reserve(rows.len().saturating_mul(d));
    let mut residual = vec![0.0f32; d];
    let mut code = vec![0u8; d];
    for &row in rows {
        let vector = &data[row * d..(row + 1) * d];
        fvec_madd(vector, centroid, -1.0, &mut residual);
        sq.encode(&residual, &mut code);
        output_ids.push(input_ids[row]);
        output_codes.extend_from_slice(&code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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
