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

use crate::coarse::CoarseAssignment;
use crate::distance::{preprocess_vectors, MetricType, QueryDistance};
use crate::ivfpq::RowIdFilter;
use crate::kmeans::{self, KMeansConfig};

/// IVF-FLAT index. Stores raw vectors in each IVF list for exact per-list scan.
pub struct IVFFlatIndex {
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    quantizer_centroids: Vec<f32>,
    pub ids: Vec<Vec<i64>>,
    pub vectors: Vec<Vec<f32>>,
    coarse_assignment: CoarseAssignment,
}

impl IVFFlatIndex {
    pub fn new(d: usize, nlist: usize, metric: MetricType) -> Self {
        IVFFlatIndex {
            d,
            nlist,
            metric,
            quantizer_centroids: Vec::new(),
            ids: vec![Vec::new(); nlist],
            vectors: vec![Vec::new(); nlist],
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
        let train_data = self.preprocess_vectors(data, n);
        self.quantizer_centroids = kmeans::kmeans_train(config, &train_data, n, self.d, self.nlist);
        self.coarse_assignment.reset();
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
        for i in 0..n {
            let vector = &processed[i * self.d..(i + 1) * self.d];
            let list_id = list_ids[i];
            self.ids[list_id].push(ids[i]);
            self.vectors[list_id].extend_from_slice(vector);
        }
    }

    pub fn total_vectors(&self) -> usize {
        self.ids.iter().map(Vec::len).sum()
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
            let distance = QueryDistance::new(query, self.metric);
            let mut heap = FlatTopKHeap::new(k);

            for &list_id in &all_probe_indices[qi] {
                let ids = &self.ids[list_id];
                let vectors = &self.vectors[list_id];
                for (local_idx, &id) in ids.iter().enumerate() {
                    if let Some(f) = filter {
                        if !f.contains(id) {
                            continue;
                        }
                    }
                    let vector = &vectors[local_idx * self.d..(local_idx + 1) * self.d];
                    heap.push(distance.distance_to(vector, None), id);
                }
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
}

pub(crate) struct FlatTopKHeap {
    k: usize,
    data: Vec<(f32, i64)>,
    worst_index: Option<usize>,
    // Large K uses an indirect max heap. Data slots never move, preserving the
    // existing last-slot eviction and stable ordering of equal distances.
    heap_indices: Vec<usize>,
}

impl FlatTopKHeap {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            data: Vec::with_capacity(k),
            worst_index: None,
            heap_indices: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, dist: f32, id: i64) {
        if self.k == 0 || dist.is_nan() {
            return;
        }
        if self.data.len() < self.k {
            self.data.push((dist, id));
            if self.data.len() == self.k {
                self.refresh_worst();
            }
            return;
        }
        let worst = self
            .worst_index
            .expect("a full IVF-FLAT heap has a worst entry");
        if dist.total_cmp(&self.data[worst].0).is_lt() {
            self.data[worst] = (dist, id);
            self.refresh_worst();
        }
    }

    fn refresh_worst(&mut self) {
        if self.k < 32 {
            self.worst_index = self
                .data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.0.total_cmp(&b.0))
                .map(|(index, _)| index);
        } else {
            if self.heap_indices.is_empty() {
                self.heap_indices.extend(0..self.data.len());
                for parent in (0..self.heap_indices.len() / 2).rev() {
                    self.sift_worst_down(parent);
                }
            } else {
                self.sift_worst_down(0);
            }
            self.worst_index = Some(self.heap_indices[0]);
        }
    }

    fn worse(&self, a: usize, b: usize) -> bool {
        self.data[a]
            .0
            .total_cmp(&self.data[b].0)
            .then_with(|| a.cmp(&b))
            .is_gt()
    }

    fn sift_worst_down(&mut self, mut position: usize) {
        let entry = self.heap_indices[position];
        while position * 2 + 1 < self.heap_indices.len() {
            let mut child = position * 2 + 1;
            if child + 1 < self.heap_indices.len()
                && self.worse(self.heap_indices[child + 1], self.heap_indices[child])
            {
                child += 1;
            }
            if !self.worse(self.heap_indices[child], entry) {
                break;
            }
            self.heap_indices[position] = self.heap_indices[child];
            position = child;
        }
        self.heap_indices[position] = entry;
    }

    pub(crate) fn into_sorted(mut self) -> Vec<(f32, i64)> {
        self.data.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.data
    }
}

impl crate::collect::Collector for FlatTopKHeap {
    const BATCH_SHORT_L2: bool = true;

    #[inline]
    fn cutoff(&self) -> f32 {
        self.worst_index
            .map_or(f32::INFINITY, |index| self.data[index].0)
    }

    #[inline]
    fn push(&mut self, id: i64, value: f32) -> std::io::Result<()> {
        FlatTopKHeap::push(self, value, id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::MetricType;

    #[test]
    fn flat_topk_preserves_legacy_slots_ties_and_nonfinite_values() {
        use crate::collect::Collector;

        for k in [0, 1, 10, 31, 32, 33, 100, 1000] {
            let mut actual = FlatTopKHeap::new(k);
            let mut expected: Vec<(f32, i64)> = Vec::new();
            for i in 0..4097 {
                let distance = match i % 31 {
                    0 => f32::NAN,
                    1 => -f32::NAN,
                    2 => f32::INFINITY,
                    3 => f32::NEG_INFINITY,
                    4 => -0.0,
                    5 => 0.0,
                    _ => ((4097 - i) % 127) as f32 - 64.0,
                };
                let id = i as i64 % 133;
                if k > 0 && !distance.is_nan() {
                    if expected.len() < k {
                        expected.push((distance, id));
                    } else {
                        let worst = expected
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.0.total_cmp(&b.0))
                            .unwrap()
                            .0;
                        if distance.total_cmp(&expected[worst].0).is_lt() {
                            expected[worst] = (distance, id);
                        }
                    }
                }
                actual.push(distance, id);
                let cutoff = if expected.len() < k || k == 0 {
                    f32::INFINITY
                } else {
                    expected
                        .iter()
                        .map(|pair| pair.0)
                        .max_by(f32::total_cmp)
                        .unwrap()
                };
                assert_eq!(actual.cutoff().to_bits(), cutoff.to_bits(), "k={k} row={i}");
            }
            expected.sort_by(|a, b| a.0.total_cmp(&b.0));
            let actual = actual.into_sorted();
            assert_eq!(
                actual
                    .iter()
                    .map(|&(d, id)| (d.to_bits(), id))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|&(d, id)| (d.to_bits(), id))
                    .collect::<Vec<_>>(),
                "k={k}"
            );
        }
    }

    #[test]
    fn test_ivfflat_add_assigns_all_vectors() {
        let d = 4;
        let nlist = 2;
        let n = 16;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| [i as f32, 0.0, i as f32 + 0.5, 1.0])
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        assert_eq!(index.total_vectors(), n);
        for list_id in 0..nlist {
            assert_eq!(index.vectors[list_id].len(), index.ids[list_id].len() * d);
        }
    }

    #[test]
    fn test_ivfflat_recalls_query_vector() {
        let d = 4;
        let nlist = 4;
        let n = 64;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
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
        assert_eq!(distances[0], 0.0);
        for i in 1..5 {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_ivfflat_search_with_filter() {
        use std::collections::HashSet;

        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.train(&data, 3);
        index.add(&data, &ids, 3);

        let filter: HashSet<i64> = [12].into_iter().collect();
        let mut distances = vec![0.0; 2];
        let mut labels = vec![0; 2];
        index.search_with_filter(
            &[0.0, 0.0],
            1,
            2,
            1,
            Some(&filter),
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels, vec![12, -1]);
        assert!(distances[0] > 0.0);
        assert_eq!(distances[1], f32::MAX);
    }

    #[test]
    fn test_flat_topk_heap_skips_both_nan_signs() {
        let expected = [(0.25, 5), (0.5, 3), (1.0, 1), (2.0, 4)];
        for k in [0, 1, 3, 5] {
            let mut heap = FlatTopKHeap::new(k);
            for (distance, id) in [
                (f32::NAN, 10),
                (-f32::NAN, 11),
                (1.0, 1),
                (f32::NAN, 12),
                (0.5, 3),
                (-f32::NAN, 13),
                (2.0, 4),
                (f32::NAN, 14),
                (-f32::NAN, 15),
                (0.25, 5),
            ] {
                heap.push(distance, id);
            }
            assert_eq!(
                heap.into_sorted(),
                expected[..k.min(expected.len())],
                "k={k}"
            );
        }
    }

    #[test]
    fn test_ivfflat_inner_product_search_skips_both_nan_signs() {
        for nan in [f32::NAN, -f32::NAN] {
            for (data, ids) in [
                ([nan, 1.0, 2.0], [99, 1, 2]),
                ([1.0, nan, 2.0], [1, 99, 2]),
                ([1.0, 2.0, nan], [1, 2, 99]),
            ] {
                let mut index = IVFFlatIndex::new(1, 1, MetricType::InnerProduct);
                index.set_quantizer_centroids(vec![0.0]);
                index.add(&data, &ids, 3);
                for k in 1..=3 {
                    let mut distances = vec![0.0; k];
                    let mut labels = vec![0; k];
                    index.search(&[1.0], 1, k, 1, &mut distances, &mut labels);
                    assert_eq!(labels, [2, 1, -1][..k]);
                    assert_eq!(distances, [-2.0, -1.0, f32::MAX][..k]);
                }
            }
        }
    }

    #[test]
    fn test_ivfflat_search_with_only_nan_returns_missing_neighbors() {
        for metric in [MetricType::L2, MetricType::InnerProduct] {
            let mut index = IVFFlatIndex::new(1, 1, metric);
            index.set_quantizer_centroids(vec![0.0]);
            index.add(&[f32::NAN, -f32::NAN], &[1, 2], 2);
            let mut distances = [0.0; 3];
            let mut labels = [0; 3];
            index.search(&[1.0], 1, 3, 1, &mut distances, &mut labels);
            assert_eq!(labels, [-1; 3]);
            assert_eq!(distances, [f32::MAX; 3]);
        }
    }
}
