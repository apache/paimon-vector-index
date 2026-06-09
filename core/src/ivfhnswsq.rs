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

use crate::distance::{preprocess_vectors, MetricType};
use crate::hnsw::{HnswBuildParams, HnswGraph};
use crate::hnsw_search::{search_hnsw_lists, HnswSearchList};
use crate::ivfpq::RowIdFilter;
use crate::kmeans::{self, KMeansConfig};
use crate::sq::ScalarQuantizer;
use crate::topk::TopKHeap;
use rayon::prelude::*;
use std::io;

pub struct IVFHNSWSQIndex {
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub quantizer_centroids: Vec<f32>,
    pub sq: ScalarQuantizer,
    pub list_sqs: Vec<ScalarQuantizer>,
    pub ids: Vec<Vec<i64>>,
    pub codes: Vec<Vec<u8>>,
    pub graphs: Vec<Option<HnswGraph>>,
    pub hnsw_params: HnswBuildParams,
}

impl IVFHNSWSQIndex {
    pub fn new(d: usize, nlist: usize, metric: MetricType, hnsw_params: HnswBuildParams) -> Self {
        Self {
            d,
            nlist,
            metric,
            quantizer_centroids: Vec::new(),
            sq: ScalarQuantizer::new(d),
            list_sqs: vec![ScalarQuantizer::new(d); nlist],
            ids: vec![Vec::new(); nlist],
            codes: vec![Vec::new(); nlist],
            graphs: vec![None; nlist],
            hnsw_params,
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        let processed = self.preprocess_vectors(data, n);
        self.quantizer_centroids =
            kmeans::kmeans_train(&KMeansConfig::default(), &processed, n, self.d, self.nlist);
        let (list_ids, residuals) = self.assign_residuals(&processed, n);
        self.sq.train(&residuals, n);
        self.train_list_sqs(&list_ids, &residuals);
    }

    pub fn add(&mut self, data: &[f32], ids: &[i64], n: usize) {
        let processed = self.preprocess_vectors(data, n);
        let code_size = self.code_size();
        let (list_ids, residuals) = self.assign_residuals(&processed, n);

        let mut code = vec![0u8; code_size];
        for i in 0..n {
            let list_id = list_ids[i];
            self.list_sqs[list_id].encode(&residuals[i * self.d..(i + 1) * self.d], &mut code);
            self.ids[list_id].push(ids[i]);
            self.codes[list_id].extend_from_slice(&code);
        }
        self.graphs.fill(None);
    }

    pub fn total_vectors(&self) -> usize {
        self.ids.iter().map(Vec::len).sum()
    }

    pub fn build_graphs(&mut self) -> io::Result<()> {
        self.graphs = (0..self.nlist)
            .into_par_iter()
            .map(|list_id| {
                let count = self.ids[list_id].len();
                if count == 0 {
                    Ok(None)
                } else {
                    let vectors = self.decode_list_vectors(list_id, count);
                    HnswGraph::build_owned(vectors, count, self.d, self.metric, self.hnsw_params)
                        .map(Some)
                }
            })
            .collect::<io::Result<Vec<_>>>()?;
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
            let lists: Vec<_> = all_probe_indices[qi]
                .iter()
                .map(|&list_id| HnswSearchList {
                    ids: self.ids[list_id].as_slice(),
                    graph: self.graphs[list_id].as_ref(),
                    payload: list_id,
                })
                .collect();
            let sorted = search_hnsw_lists(query, &lists, k, ef_search, filter, |list, heap| {
                self.scan_sq_list(query, list.payload, filter, heap);
            });
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

    fn scan_sq_list(
        &self,
        query: &[f32],
        list_id: usize,
        filter: Option<&dyn RowIdFilter>,
        heap: &mut TopKHeap,
    ) {
        let context = self.sq.distance_context(query, self.metric);
        let sq = self.list_sq(list_id);
        let code_size = self.code_size();
        let centroid = self.list_centroid(list_id);
        for (local_id, &row_id) in self.ids[list_id].iter().enumerate() {
            if filter.map(|f| !f.contains(row_id)).unwrap_or(false) {
                continue;
            }
            let code = &self.codes[list_id][local_id * code_size..(local_id + 1) * code_size];
            heap.push(
                sq.distance_to_code_with_offset_with_context(query, code, centroid, context),
                row_id,
            );
        }
    }

    pub(crate) fn decode_list_vectors(&self, list_id: usize, count: usize) -> Vec<f32> {
        let mut vectors = vec![0.0f32; count * self.d];
        self.list_sq(list_id)
            .decode_batch(&self.codes[list_id], count, &mut vectors);
        let centroid = self.list_centroid(list_id);
        for vector in vectors.chunks_exact_mut(self.d) {
            for i in 0..self.d {
                vector[i] += centroid[i];
            }
        }
        vectors
    }

    fn assign_residuals(&self, processed: &[f32], n: usize) -> (Vec<usize>, Vec<f32>) {
        let mut list_ids = Vec::with_capacity(n);
        let mut residuals = vec![0.0f32; n * self.d];
        for i in 0..n {
            let vector = &processed[i * self.d..(i + 1) * self.d];
            let list_id =
                kmeans::find_nearest(vector, &self.quantizer_centroids, self.nlist, self.d);
            list_ids.push(list_id);
            self.write_residual(
                vector,
                list_id,
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
        let centroid = self.list_centroid(list_id);
        for i in 0..self.d {
            out[i] = vector[i] - centroid[i];
        }
    }

    fn list_centroid(&self, list_id: usize) -> &[f32] {
        &self.quantizer_centroids[list_id * self.d..(list_id + 1) * self.d]
    }

    pub(crate) fn list_sq(&self, list_id: usize) -> &ScalarQuantizer {
        self.list_sqs.get(list_id).unwrap_or(&self.sq)
    }

    pub(crate) fn code_size(&self) -> usize {
        self.sq.code_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswBuildParams;
    use std::collections::HashSet;

    #[test]
    fn test_ivfhnswsq_recalls_query_vector() {
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
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
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
        assert!(distances[0].is_finite());
    }

    #[test]
    fn test_ivfhnswsq_without_built_graphs_falls_back_to_sq_scan() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];
        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 3);
        index.add(&data, &ids, 3);

        let mut distances = vec![0.0; 2];
        let mut labels = vec![0; 2];
        index.search(&[0.0, 0.0], 1, 2, nlist, 8, &mut distances, &mut labels);

        assert_eq!(labels[0], 10);
    }

    #[test]
    fn test_ivfhnswsq_search_with_filter() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];
        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 3);
        index.add(&data, &ids, 3);
        index.build_graphs().unwrap();

        let filter: HashSet<i64> = [12].into_iter().collect();
        let mut distances = vec![0.0; 2];
        let mut labels = vec![0; 2];
        index.search_with_filter(
            &[0.0, 0.0],
            1,
            2,
            nlist,
            8,
            Some(&filter),
            &mut distances,
            &mut labels,
        );

        assert_eq!(labels[0], 12);
        assert_eq!(labels[1], -1);
    }
}
