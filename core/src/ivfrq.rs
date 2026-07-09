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

use crate::distance::{fvec_madd, fvec_norm_l2sqr, preprocess_vectors, MetricType};
use crate::ivfpq::RowIdFilter;
use crate::kmeans::{self, KMeansConfig};
use crate::rq::{
    RQCodeFactors, RQRotation, RaBitQuantizer, DEFAULT_RQ_QUERY_BITS, DEFAULT_RQ_ROTATION_ROUNDS,
    DEFAULT_RQ_ROTATION_SEED, RQ_BYTE_LUT_MIN_LIST_SIZE,
};
use crate::topk::TopKHeap;

pub struct IVFRQIndex {
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub quantizer_centroids: Vec<f32>,
    pub rotation_seed: u64,
    pub rotation_rounds: u32,
    pub ids: Vec<Vec<i64>>,
    pub codes: Vec<Vec<u8>>,
    pub factors: Vec<Vec<RQCodeFactors>>,
    quantizer: RaBitQuantizer,
    rotation: RQRotation,
}

impl IVFRQIndex {
    pub fn new(d: usize, nlist: usize, metric: MetricType) -> Self {
        Self::with_rotation(
            d,
            nlist,
            metric,
            DEFAULT_RQ_ROTATION_SEED,
            DEFAULT_RQ_ROTATION_ROUNDS,
        )
    }

    pub fn with_rotation(
        d: usize,
        nlist: usize,
        metric: MetricType,
        rotation_seed: u64,
        rotation_rounds: u32,
    ) -> Self {
        Self {
            d,
            nlist,
            metric,
            quantizer_centroids: Vec::new(),
            rotation_seed,
            rotation_rounds,
            ids: vec![Vec::new(); nlist],
            codes: vec![Vec::new(); nlist],
            factors: vec![Vec::new(); nlist],
            quantizer: RaBitQuantizer::new(d),
            rotation: RQRotation::new(d, rotation_seed, rotation_rounds),
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        let processed = self.preprocess_vectors(data, n);
        self.quantizer_centroids =
            kmeans::kmeans_train(&KMeansConfig::default(), &processed, n, self.d, self.nlist);
    }

    pub fn add(&mut self, data: &[f32], ids: &[i64], n: usize) {
        let processed = self.preprocess_vectors(data, n);
        let list_ids = kmeans::find_nearest_batch(
            &processed,
            n,
            &self.quantizer_centroids,
            self.nlist,
            self.d,
        );
        let code_size = self.code_size();
        let mut residual = vec![0.0f32; self.d];
        let mut code = vec![0u8; code_size];

        for i in 0..n {
            let list_id = list_ids[i];
            let vector = &processed[i * self.d..(i + 1) * self.d];
            self.write_rotated_residual(vector, list_id, &mut residual);
            let factors = self
                .quantizer
                .encode(&residual, fvec_norm_l2sqr(vector), &mut code);
            self.ids[list_id].push(ids[i]);
            self.codes[list_id].extend_from_slice(&code);
            self.factors[list_id].push(factors);
        }
    }

    pub fn total_vectors(&self) -> usize {
        self.ids.iter().map(Vec::len).sum()
    }

    pub fn code_size(&self) -> usize {
        self.quantizer.code_size()
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
            DEFAULT_RQ_QUERY_BITS,
            result_distances,
            result_labels,
        );
    }

    pub fn search_with_query_bits(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
        nprobe: usize,
        query_bits: usize,
        result_distances: &mut [f32],
        result_labels: &mut [i64],
    ) {
        self.search_with_filter(
            queries,
            nq,
            k,
            nprobe,
            None,
            query_bits,
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
        query_bits: usize,
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
                self.scan_list(query, list_id, filter, query_bits, &mut heap);
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

    pub(crate) fn preprocess_vectors(&self, data: &[f32], n: usize) -> Vec<f32> {
        preprocess_vectors(data, n, self.d, self.metric)
    }

    pub(crate) fn list_centroid(&self, list_id: usize) -> &[f32] {
        &self.quantizer_centroids[list_id * self.d..(list_id + 1) * self.d]
    }

    pub(crate) fn rotated_query_residual(&self, query: &[f32], list_id: usize) -> Vec<f32> {
        let mut residual = vec![0.0f32; self.d];
        self.write_rotated_residual(query, list_id, &mut residual);
        residual
    }

    fn scan_list(
        &self,
        query: &[f32],
        list_id: usize,
        filter: Option<&dyn RowIdFilter>,
        query_bits: usize,
        heap: &mut TopKHeap,
    ) {
        let rotated_query_residual = self.rotated_query_residual(query, list_id);
        let distance_context = self.quantizer.prepare_distance_context_with_query_bits(
            rotated_query_residual,
            query,
            self.ids[list_id].len() >= RQ_BYTE_LUT_MIN_LIST_SIZE,
            query_bits,
        );
        let code_size = self.code_size();
        for (local_idx, &id) in self.ids[list_id].iter().enumerate() {
            if filter.map(|f| !f.contains(id)).unwrap_or(false) {
                continue;
            }
            let code = &self.codes[list_id][local_idx * code_size..(local_idx + 1) * code_size];
            let dist = self.quantizer.distance_to_code_prepared(
                &distance_context,
                code,
                self.factors[list_id][local_idx],
                self.metric,
            );
            heap.push(dist, id);
        }
    }

    fn write_rotated_residual(&self, vector: &[f32], list_id: usize, out: &mut [f32]) {
        fvec_madd(vector, self.list_centroid(list_id), -1.0, out);
        self.rotation.apply(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivfrq_recalls_query_vector() {
        let d = 8;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            })
            .collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let mut index = IVFRQIndex::new(d, nlist, MetricType::L2);
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
        assert!(distances[0] <= 1e-4);
    }

    #[test]
    fn ivfrq_inner_product_recalls_query_vector() {
        let d = 8;
        let nlist = 1;
        let n = 8;
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            data[i * d + i] = 1.0;
        }
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let mut index = IVFRQIndex::new(d, nlist, MetricType::InnerProduct);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let query_id = 7;
        let mut distances = vec![0.0; 5];
        let mut labels = vec![0; 5];
        index.search(
            &data[query_id * d..(query_id + 1) * d],
            1,
            5,
            nlist,
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels[0], ids[query_id]);
    }
}
