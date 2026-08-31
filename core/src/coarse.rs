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

use crate::diskann::{
    DiskAnnBuildDistance, DiskAnnBuildParams, DiskAnnRawVectorEncoding, DiskAnnStorageLayout,
};
use crate::kmeans;
use crate::logging::{emit_log, LogLevel};
use crate::vamana::VamanaGraph;
use rayon::prelude::*;

const APPROX_ASSIGN_SEARCH_LIST: usize = 15;
const APPROX_ASSIGN_MIN_CENTROID_VALUES: usize = 1_000_000;

fn use_approximate_assignment(d: usize, nlist: usize) -> bool {
    d.saturating_mul(nlist) >= APPROX_ASSIGN_MIN_CENTROID_VALUES
}

#[derive(Default)]
pub(crate) struct CoarseAssignment {
    graph: Option<VamanaGraph>,
    build_attempted: bool,
}

impl CoarseAssignment {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn prepare(&mut self, centroids: &[f32], nlist: usize, d: usize) {
        if self.build_attempted {
            return;
        }
        self.build_attempted = true;
        if !use_approximate_assignment(d, nlist) {
            return;
        }

        let params = DiskAnnBuildParams {
            max_degree: 12,
            build_search_list_size: APPROX_ASSIGN_SEARCH_LIST,
            alpha: 1.2,
            seed: 42,
            memory_budget_bytes: 1024 * 1024 * 1024,
            storage_layout: DiskAnnStorageLayout::Compact,
            raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
            build_distance: DiskAnnBuildDistance::FullPrecision,
        };
        match VamanaGraph::build(centroids, nlist, d, params) {
            Ok(graph) => self.graph = Some(graph),
            Err(error) => emit_log(
                LogLevel::Warn,
                &format!("automatic approximate coarse assignment disabled: {error}"),
            ),
        }
    }

    pub(crate) fn assign(
        &mut self,
        data: &[f32],
        n: usize,
        centroids: &[f32],
        nlist: usize,
        d: usize,
    ) -> Vec<usize> {
        self.prepare(centroids, nlist, d);
        let Some(graph) = &self.graph else {
            return kmeans::find_nearest_batch(data, n, centroids, nlist, d);
        };

        let mut assignments = vec![0usize; n];
        let chunk = (n / (rayon::current_num_threads() * 4).max(1)).clamp(16, 1024);
        assignments.par_chunks_mut(chunk).enumerate().for_each_init(
            || graph.search_scratch(APPROX_ASSIGN_SEARCH_LIST),
            |scratch, (chunk_idx, chunk_assignments)| {
                let row0 = chunk_idx * chunk;
                for (i, assignment) in chunk_assignments.iter_mut().enumerate() {
                    let row = row0 + i;
                    *assignment = graph
                        .greedy_search_best_with_scratch(
                            centroids,
                            d,
                            &data[row * d..(row + 1) * d],
                            APPROX_ASSIGN_SEARCH_LIST,
                            scratch,
                        )
                        .map(|node| node.id as usize)
                        .unwrap_or(0);
                }
            },
        );
        assignments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approximate_assignment_depends_on_centroid_values() {
        assert!(!use_approximate_assignment(768, 1024));
        assert!(use_approximate_assignment(768, 4096));
    }

    #[test]
    fn vamana_coarse_assignment_matches_exact_on_connected_graph() {
        let d = 4;
        let nlist = 4;
        let n = 64;
        let centroids = (0..nlist)
            .flat_map(|list| (0..d).map(move |dimension| list as f32 * 10.0 + dimension as f32))
            .collect::<Vec<_>>();
        let data = (0..n)
            .flat_map(|row| {
                (0..d).map(move |dimension| {
                    (row % nlist) as f32 * 10.0 + row as f32 * 0.01 + dimension as f32
                })
            })
            .collect::<Vec<_>>();
        let expected = kmeans::find_nearest_batch(&data, n, &centroids, nlist, d);
        let adjacency = (0..nlist)
            .map(|node| {
                (0..nlist)
                    .filter(|&neighbor| neighbor != node)
                    .map(|neighbor| neighbor as u32)
                    .collect()
            })
            .collect();
        let mut assignment = CoarseAssignment {
            graph: Some(VamanaGraph::from_adjacency(0, adjacency)),
            build_attempted: true,
        };

        assert_eq!(assignment.assign(&data, n, &centroids, nlist, d), expected);
        assignment.reset();
        assert!(assignment.graph.is_none());
        assert!(!assignment.build_attempted);
    }
}
