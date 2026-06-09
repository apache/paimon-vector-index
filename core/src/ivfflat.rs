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

use crate::distance::{fvec_distance, fvec_normalize, MetricType};
use crate::ivfpq::RowIdFilter;
use crate::kmeans::{self, KMeansConfig};

/// IVF-FLAT index. Stores raw vectors in each IVF list for exact per-list scan.
pub struct IVFFlatIndex {
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub quantizer_centroids: Vec<f32>,
    pub ids: Vec<Vec<i64>>,
    pub vectors: Vec<Vec<f32>>,
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
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        let train_data = self.preprocess_vectors(data, n);
        self.quantizer_centroids =
            kmeans::kmeans_train(&KMeansConfig::default(), &train_data, n, self.d, self.nlist);
    }

    pub fn add(&mut self, data: &[f32], ids: &[i64], n: usize) {
        let processed = self.preprocess_vectors(data, n);
        for i in 0..n {
            let vector = &processed[i * self.d..(i + 1) * self.d];
            let list_id =
                kmeans::find_nearest(vector, &self.quantizer_centroids, self.nlist, self.d);
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
                    heap.push(fvec_distance(query, vector, self.metric), id);
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
        let mut processed = data[..n * self.d].to_vec();
        if self.metric == MetricType::Cosine {
            for i in 0..n {
                fvec_normalize(&mut processed[i * self.d..(i + 1) * self.d]);
            }
        }
        processed
    }
}

struct FlatTopKHeap {
    k: usize,
    data: Vec<(f32, i64)>,
}

impl FlatTopKHeap {
    fn new(k: usize) -> Self {
        Self {
            k,
            data: Vec::with_capacity(k),
        }
    }

    fn push(&mut self, dist: f32, id: i64) {
        if self.k == 0 {
            return;
        }
        if self.data.len() < self.k {
            self.data.push((dist, id));
            return;
        }
        if let Some((worst_idx, _)) = self
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
        {
            if dist < self.data[worst_idx].0 {
                self.data[worst_idx] = (dist, id);
            }
        }
    }

    fn into_sorted(mut self) -> Vec<(f32, i64)> {
        self.data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::MetricType;

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
}
