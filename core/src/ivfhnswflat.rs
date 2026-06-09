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

use crate::distance::{fvec_distance, MetricType};
use crate::hnsw::{HnswBuildParams, HnswGraph};
use crate::ivfflat::IVFFlatIndex;
use crate::ivfpq::RowIdFilter;
use crate::kmeans;
use std::io;

pub struct IVFHNSWFlatIndex {
    pub flat: IVFFlatIndex,
    pub graphs: Vec<Option<HnswGraph>>,
    pub hnsw_params: HnswBuildParams,
}

impl IVFHNSWFlatIndex {
    pub fn new(d: usize, nlist: usize, metric: MetricType, hnsw_params: HnswBuildParams) -> Self {
        IVFHNSWFlatIndex {
            flat: IVFFlatIndex::new(d, nlist, metric),
            graphs: vec![None; nlist],
            hnsw_params,
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        self.flat.train(data, n);
    }

    pub fn add(&mut self, data: &[f32], ids: &[i64], n: usize) {
        self.flat.add(data, ids, n);
        self.graphs.fill(None);
    }

    pub fn build_graphs(&mut self) -> io::Result<()> {
        for list_id in 0..self.flat.nlist {
            let count = self.flat.ids[list_id].len();
            self.graphs[list_id] = if count == 0 {
                None
            } else {
                Some(HnswGraph::build(
                    &self.flat.vectors[list_id],
                    count,
                    self.flat.d,
                    self.flat.metric,
                    self.hnsw_params,
                )?)
            };
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
        nprobe: usize,
        ef_search: usize,
        result_distances: &mut [f32],
        result_labels: &mut [i64],
    ) {
        self.search_with_filter(
            queries,
            nq,
            k,
            nprobe,
            ef_search,
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
        ef_search: usize,
        filter: Option<&dyn RowIdFilter>,
        result_distances: &mut [f32],
        result_labels: &mut [i64],
    ) {
        let processed_queries = self.flat.preprocess_vectors(queries, nq);
        let (all_probe_indices, _) = kmeans::find_topk_batch(
            &processed_queries,
            nq,
            &self.flat.quantizer_centroids,
            self.flat.nlist,
            self.flat.d,
            nprobe,
        );

        for qi in 0..nq {
            let query = &processed_queries[qi * self.flat.d..(qi + 1) * self.flat.d];
            let mut heap = HnswFlatTopKHeap::new(k);
            let force_flat_scan = filter
                .map(|f| self.count_filtered(&all_probe_indices[qi], f) <= ef_search.max(k))
                .unwrap_or(false);

            for &list_id in &all_probe_indices[qi] {
                if force_flat_scan {
                    self.scan_flat_list(query, list_id, filter, &mut heap);
                    continue;
                }
                if let Some(ref graph) = self.graphs[list_id] {
                    let local_results = graph.search(query, ef_search.max(k), ef_search.max(k));
                    for (local_id, dist) in local_results {
                        let row_id = self.flat.ids[list_id][local_id];
                        if let Some(f) = filter {
                            if !f.contains(row_id) {
                                continue;
                            }
                        }
                        heap.push(dist, row_id);
                    }
                } else {
                    self.scan_flat_list(query, list_id, filter, &mut heap);
                }
            }
            if filter.is_some() && heap.len() < k {
                for &list_id in &all_probe_indices[qi] {
                    self.scan_flat_list(query, list_id, filter, &mut heap);
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

    fn count_filtered(&self, probe_indices: &[usize], filter: &dyn RowIdFilter) -> usize {
        probe_indices
            .iter()
            .map(|&list_id| {
                self.flat.ids[list_id]
                    .iter()
                    .filter(|&&id| filter.contains(id))
                    .count()
            })
            .sum()
    }

    fn scan_flat_list(
        &self,
        query: &[f32],
        list_id: usize,
        filter: Option<&dyn RowIdFilter>,
        heap: &mut HnswFlatTopKHeap,
    ) {
        for (local_id, &row_id) in self.flat.ids[list_id].iter().enumerate() {
            if let Some(f) = filter {
                if !f.contains(row_id) {
                    continue;
                }
            }
            let vector =
                &self.flat.vectors[list_id][local_id * self.flat.d..(local_id + 1) * self.flat.d];
            heap.push(fvec_distance(query, vector, self.flat.metric), row_id);
        }
    }
}

struct HnswFlatTopKHeap {
    k: usize,
    data: Vec<(f32, i64)>,
}

impl HnswFlatTopKHeap {
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
        if let Some((existing_dist, _)) = self
            .data
            .iter_mut()
            .find(|(_, existing_id)| *existing_id == id)
        {
            if dist < *existing_dist {
                *existing_dist = dist;
            }
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
            .max_by(|(_, a), (_, b)| a.0.total_cmp(&b.0))
        {
            if dist < self.data[worst_idx].0 {
                self.data[worst_idx] = (dist, id);
            }
        }
    }

    fn into_sorted(mut self) -> Vec<(f32, i64)> {
        self.data.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::MetricType;
    use crate::hnsw::HnswBuildParams;

    #[test]
    fn test_ivfhnswflat_recalls_query_vector() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let query_id = 23;
        let mut distances = vec![0.0; 5];
        let mut labels = vec![0; 5];
        index.search(
            &data[query_id * d..(query_id + 1) * d],
            1,
            5,
            nlist,
            32,
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels[0], ids[query_id]);
        assert_eq!(distances[0], 0.0);
    }

    #[test]
    fn test_ivfhnswflat_without_built_graphs_falls_back_to_flat_scan() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);

        let query_id = 23;
        let mut distances = vec![0.0; 5];
        let mut labels = vec![0; 5];
        index.search(
            &data[query_id * d..(query_id + 1) * d],
            1,
            5,
            nlist,
            32,
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels[0], ids[query_id]);
        assert_eq!(distances[0], 0.0);
    }

    #[test]
    fn test_ivfhnswflat_selective_filter_uses_exact_results() {
        use std::collections::HashSet;

        let d = 2;
        let nlist = 1;
        let n = 64;
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            data.push(i as f32);
            data.push(0.0);
        }
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let filter: HashSet<i64> = [63].into_iter().collect();
        let mut distances = vec![0.0; 1];
        let mut labels = vec![0; 1];
        index.search_with_filter(
            &[0.0, 0.0],
            1,
            1,
            1,
            4,
            Some(&filter),
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels[0], 63);
        assert_eq!(distances[0], 63.0 * 63.0);
    }

    #[test]
    fn test_ivfhnswflat_filter_backfills_when_graph_returns_too_few_matches() {
        use std::collections::HashSet;

        let d = 2;
        let nlist = 1;
        let n = 128;
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            data.push(i as f32);
            data.push(0.0);
        }
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let filter: HashSet<i64> = (0..n as i64).filter(|id| id % 2 == 0).collect();
        let mut distances = vec![0.0; 10];
        let mut labels = vec![0; 10];
        index.search_with_filter(
            &[127.0, 0.0],
            1,
            10,
            1,
            1,
            Some(&filter),
            &mut distances,
            &mut labels,
        );

        assert_eq!(
            labels,
            vec![126, 124, 122, 120, 118, 116, 114, 112, 110, 108]
        );
        assert!(labels.iter().all(|id| id % 2 == 0));
    }

    #[test]
    fn test_topk_heap_keeps_closest_duplicate_id() {
        let mut heap = HnswFlatTopKHeap::new(2);

        heap.push(10.0, 7);
        heap.push(5.0, 8);
        heap.push(1.0, 7);

        assert_eq!(heap.into_sorted(), vec![(1.0, 7), (5.0, 8)]);
    }
}
