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

use crate::diskann::DiskAnnBuildParams;
use crate::distance::{
    fvec_distance, fvec_l2sqr, fvec_l2sqr_four, fvec_l2sqr_scaled_exceeds, MetricType,
};
use crate::kmeans::{self, KMeansConfig};
use crate::pq::ProductQuantizer;
use crate::sparse_table::{estimated_memory_bytes as sparse_table_memory_bytes, SparseTable};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::io;
use std::ops::Index;
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

const PARALLEL_ADJACENCY_NODES_PER_SHARD: usize = 256;
const PARALLEL_BUILD_BATCH_NODES: usize = 128;
const CONNECTIVITY_SOURCE_SAMPLE_SIZE: usize = 64;
const SPARSE_BUILD_VISITED_MIN_MEMORY_SAVINGS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredNode {
    pub id: u32,
    pub distance: f32,
}

impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        scored_node_order(self, other)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VamanaGraph {
    pub entry_node: u32,
    pub(crate) adjacency: CompactAdjacency,
}

pub(crate) struct VamanaMemoryEstimate {
    pub(crate) build_peak_bytes: usize,
    pub(crate) remap_peak_bytes: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct ConnectivityRepairStats {
    full_reachability_traversals: usize,
    source_distance_evaluations: usize,
    edges_added: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VamanaBuildStats {
    pub(crate) initialization: Duration,
    pub(crate) pass_one: Duration,
    pub(crate) pass_two: Duration,
    pub(crate) connectivity_repair: Duration,
}

pub(crate) fn estimate_vamana_memory_bytes(
    node_count: usize,
    max_degree: usize,
    search_list_size: usize,
    workers: usize,
) -> Option<VamanaMemoryEstimate> {
    let edge_bytes = max_degree.checked_mul(size_of::<u32>())?;
    let builder_edges = node_count.checked_mul(edge_bytes)?;
    let builder_degrees = node_count.checked_mul(size_of::<u16>())?;
    let builder_shards = node_count
        .div_ceil(PARALLEL_ADJACENCY_NODES_PER_SHARD)
        .checked_mul(size_of::<RwLock<AdjacencyShard>>())?;
    let builder_graph = builder_edges
        .checked_add(builder_degrees)?
        .checked_add(builder_shards)?;
    let build_order = node_count.checked_mul(size_of::<usize>())?;
    let expected_visited = search_list_size
        .checked_mul(max_degree)?
        .checked_add(1)?
        .min(node_count);
    let dense_worker_states = node_count
        .checked_mul(size_of::<u8>())?
        .checked_add(expected_visited.checked_mul(size_of::<u32>())?)?;
    let sparse_worker_states = sparse_table_memory_bytes(expected_visited, size_of::<u8>())?;
    let worker_states = if sparse_worker_states
        .checked_mul(SPARSE_BUILD_VISITED_MIN_MEMORY_SAVINGS)
        .is_some_and(|threshold| threshold < dense_worker_states)
    {
        sparse_worker_states
    } else {
        dense_worker_states
    };
    let worker_candidates = search_list_size
        .checked_mul(3)?
        .checked_mul(size_of::<ScoredNode>())?;
    let prune_candidates = search_list_size.checked_add(max_degree)?;
    let worker_prune =
        prune_candidates.checked_mul(size_of::<ScoredNode>().checked_add(size_of::<u32>())?)?;
    let worker_candidate_ids = search_list_size.checked_mul(size_of::<u32>())?;
    let worker_neighbors = max_degree.checked_mul(2 * size_of::<u32>())?;
    let worker_scratch = workers.max(1).checked_mul(
        worker_states
            .checked_add(worker_candidates)?
            .checked_add(worker_prune)?
            .checked_add(worker_candidate_ids)?
            .checked_add(worker_neighbors)?,
    )?;
    let reverse_edge_batch = PARALLEL_BUILD_BATCH_NODES
        .checked_mul(max_degree)?
        .checked_mul(size_of::<(u32, u32)>() + size_of::<u32>())?;
    let build_peak_bytes = builder_graph
        .checked_add(build_order)?
        .checked_add(worker_scratch)?
        .checked_add(reverse_edge_batch)?;

    let final_shards = node_count
        .div_ceil(PARALLEL_ADJACENCY_NODES_PER_SHARD)
        .checked_mul(size_of::<AdjacencyShard>())?;
    let compact_graph = builder_edges
        .checked_add(builder_degrees)?
        .checked_add(final_shards)?;
    let permutations = node_count.checked_mul(2 * size_of::<u32>())?;
    let permutation_visited = node_count.checked_mul(size_of::<bool>())?;
    let permutation_scratch = edge_bytes.checked_add(size_of::<u16>())?;
    let remap_peak_bytes = compact_graph
        .checked_add(permutations)?
        .checked_add(permutation_visited)?
        .checked_add(permutation_scratch)?;
    Some(VamanaMemoryEstimate {
        build_peak_bytes,
        remap_peak_bytes,
    })
}

pub(crate) fn estimate_sharded_vamana_memory_bytes(
    node_count: usize,
    dimension: usize,
    max_degree: usize,
    shard_count: usize,
    pq_code_size: usize,
) -> Option<usize> {
    if node_count == 0 || shard_count < 2 {
        return None;
    }
    let edge_bytes = max_degree.checked_mul(size_of::<u32>())?;
    let compact_graph = node_count.checked_mul(edge_bytes.checked_add(size_of::<u16>())?)?;
    let assignments = node_count.checked_mul(2 * size_of::<usize>())?;
    let local_count = overlapping_shard_capacity(node_count, shard_count)?;
    let memberships = shard_count
        .checked_mul(local_count)?
        .checked_mul(size_of::<u32>())?
        .checked_add(shard_count.checked_mul(size_of::<Vec<u32>>())?)?;
    let centroids = shard_count
        .checked_mul(dimension)?
        .checked_mul(size_of::<f32>())?;
    let kmeans_train_count = node_count.min(shard_count.checked_mul(256)?);
    let kmeans_training_copy = kmeans_train_count
        .checked_mul(dimension)?
        .checked_mul(size_of::<f32>())?;
    let kmeans_assignments = kmeans_train_count.checked_mul(size_of::<usize>())?;
    let kmeans_initialization = kmeans_train_count.checked_mul(size_of::<f32>())?;
    let kmeans_score_matrix = kmeans_train_count
        .checked_mul(shard_count)?
        .min(4 * 1024 * 1024)
        .checked_mul(size_of::<f32>())?;
    let kmeans_centroid_scratch = centroids.checked_mul(3)?;
    let kmeans_peak = [
        kmeans_training_copy,
        kmeans_assignments,
        kmeans_initialization,
        kmeans_score_matrix,
        kmeans_centroid_scratch,
    ]
    .into_iter()
    .try_fold(0usize, |total, value| total.checked_add(value))?;
    let local_vectors = local_count
        .checked_mul(dimension)?
        .checked_mul(size_of::<f32>())?;
    let local_ids = local_count.checked_mul(size_of::<u32>())?;
    let local_pq_codes = local_count.checked_mul(pq_code_size)?;
    // Sequential local construction briefly holds nested and compact
    // adjacency plus its order/visited vectors.
    let local_graph = local_count.checked_mul(
        edge_bytes
            .checked_mul(2)?
            .checked_add(size_of::<Vec<u32>>())?
            .checked_add(size_of::<usize>())?
            .checked_add(2 * size_of::<bool>())?,
    )?;
    let build_peak = [
        compact_graph,
        assignments,
        memberships,
        centroids,
        kmeans_peak,
        local_vectors,
        local_ids,
        local_pq_codes,
        local_graph,
    ]
    .into_iter()
    .try_fold(0usize, |total, value| total.checked_add(value))?;
    let remap_peak = estimate_vamana_memory_bytes(node_count, max_degree, max_degree.max(1), 1)?
        .remap_peak_bytes;
    Some(build_peak.max(remap_peak))
}

impl VamanaGraph {
    pub(crate) fn search_scratch(&self, search_list_size: usize) -> GreedySearchScratch {
        GreedySearchScratch::new(
            self.adjacency.len(),
            self.adjacency.max_degree,
            search_list_size.min(self.adjacency.len()),
        )
    }

    pub fn from_adjacency(entry_node: u32, adjacency: Vec<Vec<u32>>) -> Self {
        let max_degree = adjacency.iter().map(Vec::len).max().unwrap_or(0);
        Self {
            entry_node,
            adjacency: CompactAdjacency::from_nested(adjacency, max_degree),
        }
    }

    pub fn build(
        vectors: &[f32],
        count: usize,
        dimension: usize,
        params: DiskAnnBuildParams,
    ) -> io::Result<Self> {
        Self::build_with_stats(vectors, count, dimension, MetricType::L2, params)
            .map(|(graph, _)| graph)
    }

    pub(crate) fn build_with_stats(
        vectors: &[f32],
        count: usize,
        dimension: usize,
        metric: MetricType,
        params: DiskAnnBuildParams,
    ) -> io::Result<(Self, VamanaBuildStats)> {
        Self::build_with_search_distance(
            vectors,
            count,
            dimension,
            metric,
            params,
            BuildSearchDistance::FullPrecision {
                vectors,
                dimension,
                metric,
            },
        )
    }

    pub(crate) fn build_with_pq_stats(
        vectors: &[f32],
        pq: &ProductQuantizer,
        pq_codes: &[u8],
        count: usize,
        dimension: usize,
        metric: MetricType,
        params: DiskAnnBuildParams,
    ) -> io::Result<(Self, VamanaBuildStats)> {
        validate_build_inputs(vectors, count, dimension, params)?;
        if pq.d != dimension
            || !matches!(pq.nbits, 4 | 8)
            || !pq.has_valid_layout()
            || pq_codes.len() != count.saturating_mul(pq.code_size())
        {
            return Err(invalid_input(
                "Vamana PQ-guided build received an invalid codebook or code buffer",
            ));
        }
        let distance_started = Instant::now();
        let distance = PqBuildDistance::new(pq, pq_codes, count, metric)?;
        let distance_initialization = distance_started.elapsed();
        let (graph, mut stats) = Self::build_with_search_distance(
            vectors,
            count,
            dimension,
            metric,
            params,
            BuildSearchDistance::ProductQuantized(distance),
        )?;
        stats.initialization = stats.initialization.saturating_add(distance_initialization);
        Ok((graph, stats))
    }

    pub(crate) fn build_sharded_with_stats(
        vectors: &[f32],
        count: usize,
        dimension: usize,
        metric: MetricType,
        params: DiskAnnBuildParams,
        shard_count: usize,
    ) -> io::Result<(Self, VamanaBuildStats)> {
        Self::build_sharded_with_optional_pq(
            vectors,
            None,
            count,
            dimension,
            metric,
            params,
            shard_count,
        )
    }

    pub(crate) fn build_sharded_with_pq_stats(
        vectors: &[f32],
        pq: &ProductQuantizer,
        pq_codes: &[u8],
        count: usize,
        dimension: usize,
        metric: MetricType,
        params: DiskAnnBuildParams,
        shard_count: usize,
    ) -> io::Result<(Self, VamanaBuildStats)> {
        Self::build_sharded_with_optional_pq(
            vectors,
            Some((pq, pq_codes)),
            count,
            dimension,
            metric,
            params,
            shard_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_sharded_with_optional_pq(
        vectors: &[f32],
        pq_build: Option<(&ProductQuantizer, &[u8])>,
        count: usize,
        dimension: usize,
        metric: MetricType,
        params: DiskAnnBuildParams,
        shard_count: usize,
    ) -> io::Result<(Self, VamanaBuildStats)> {
        validate_build_inputs(vectors, count, dimension, params)?;
        if shard_count < 2 || shard_count > count {
            return Err(invalid_input(
                "Vamana shard count must be between 2 and the vector count",
            ));
        }
        if let Some((pq, pq_codes)) = pq_build {
            if pq.d != dimension
                || !matches!(pq.nbits, 4 | 8)
                || !pq.has_valid_layout()
                || pq_codes.len() != count.saturating_mul(pq.code_size())
            {
                return Err(invalid_input(
                    "sharded Vamana PQ-guided build received an invalid codebook or code buffer",
                ));
            }
        }
        let initialization_started = Instant::now();
        let cluster_config = KMeansConfig {
            niter: 8,
            nredo: 1,
            max_points_per_centroid: 256,
            seed: params.seed,
            balance_factor: 0.1,
        };
        let centroids =
            kmeans::kmeans_train(&cluster_config, vectors, count, dimension, shard_count);
        let mut assignments = (0..count)
            .into_par_iter()
            .map(|node| {
                nearest_two_centroids(
                    &vectors[node * dimension..(node + 1) * dimension],
                    &centroids,
                    shard_count,
                    dimension,
                )
            })
            .collect::<Vec<_>>();
        let membership_capacity = overlapping_shard_capacity(count, shard_count)
            .ok_or_else(|| invalid_input("Vamana overlapping-shard capacity overflows"))?;
        let mut memberships = (0..shard_count)
            .map(|_| Vec::with_capacity(membership_capacity))
            .collect::<Vec<_>>();
        for (node, [first, second]) in assignments.iter().copied().enumerate() {
            memberships[first].push(node as u32);
            if second != first {
                memberships[second].push(node as u32);
            }
        }
        rebalance_overlapping_shards(
            vectors,
            dimension,
            &centroids,
            &mut assignments,
            &mut memberships,
        )?;
        let initialization = initialization_started.elapsed();

        let entry_node = centroid_entry(vectors, count, dimension, metric) as u32;
        let degree = params.max_degree.min(count.saturating_sub(1));
        let mut graph = Self {
            entry_node,
            adjacency: CompactAdjacency::empty(count, degree),
        };
        let mut mapped = Vec::with_capacity(degree);
        let mut pass_one = Duration::ZERO;
        let mut pass_two = Duration::ZERO;
        for (shard, members) in memberships.iter().enumerate() {
            if members.len() < 2 {
                continue;
            }
            let local_started = Instant::now();
            let mut local_vectors = Vec::new();
            local_vectors
                .try_reserve_exact(members.len().saturating_mul(dimension))
                .map_err(|_| invalid_input("Vamana shard vector allocation failed"))?;
            for &node in members {
                let node = node as usize;
                local_vectors.extend_from_slice(&vectors[node * dimension..(node + 1) * dimension]);
            }
            let local_pq_codes = pq_build.map(|(pq, pq_codes)| {
                let code_size = pq.code_size();
                let mut local_codes = Vec::with_capacity(members.len() * code_size);
                for &node in members {
                    let node = node as usize;
                    local_codes
                        .extend_from_slice(&pq_codes[node * code_size..(node + 1) * code_size]);
                }
                local_codes
            });
            let local_degree = params.max_degree.min(members.len() - 1);
            let local_params = DiskAnnBuildParams {
                max_degree: local_degree,
                build_search_list_size: params
                    .build_search_list_size
                    .min(members.len())
                    .max(local_degree),
                seed: derived_seed(params.seed, shard as u64),
                ..params
            };
            let local_graph = if let Some((pq, _)) = pq_build {
                Self::build_with_pq_stats(
                    &local_vectors,
                    pq,
                    local_pq_codes
                        .as_deref()
                        .expect("PQ-guided shard has local codes"),
                    members.len(),
                    dimension,
                    metric,
                    local_params,
                )?
                .0
            } else {
                Self::build_sequential_with_metric(
                    &local_vectors,
                    members.len(),
                    dimension,
                    metric,
                    local_params,
                )?
            };
            pass_one = pass_one.saturating_add(local_started.elapsed());
            let merge_started = Instant::now();
            for (local_node, &global_node) in members.iter().enumerate() {
                mapped.clear();
                mapped.extend(
                    local_graph.adjacency[local_node]
                        .iter()
                        .map(|&neighbor| members[neighbor as usize]),
                );
                let selected = robust_prune_candidates(
                    vectors,
                    dimension,
                    global_node as usize,
                    &mapped,
                    &graph.adjacency[global_node as usize],
                    count,
                    degree,
                    params.alpha,
                    metric,
                );
                graph.adjacency.replace(global_node as usize, &selected);
            }
            pass_two = pass_two.saturating_add(merge_started.elapsed());
        }
        let connectivity_started = Instant::now();
        graph.repair_connectivity(vectors, dimension, degree, metric)?;
        let connectivity_repair = connectivity_started.elapsed();
        graph.validate(degree)?;
        if !graph.is_fully_reachable() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sharded Vamana graph is not fully reachable from its entry node",
            ));
        }
        Ok((
            graph,
            VamanaBuildStats {
                initialization,
                pass_one,
                pass_two,
                connectivity_repair,
            },
        ))
    }

    fn build_with_search_distance(
        vectors: &[f32],
        count: usize,
        dimension: usize,
        metric: MetricType,
        params: DiskAnnBuildParams,
        search_distance: BuildSearchDistance<'_>,
    ) -> io::Result<(Self, VamanaBuildStats)> {
        validate_build_inputs(vectors, count, dimension, params)?;
        let entry_node = centroid_entry(vectors, count, dimension, metric) as u32;
        let degree = params.max_degree.min(count.saturating_sub(1));
        let initialization_started = Instant::now();
        let adjacency = ParallelAdjacency::new_random(count, degree, params.seed);
        let initialization = initialization_started.elapsed();
        let mut rng = StdRng::seed_from_u64(derived_seed(params.seed, u64::MAX));
        let builder = ParallelVamanaBuilder {
            vectors,
            dimension,
            metric,
            entry_node,
            adjacency,
            search_distance,
        };
        let pass_one_started = Instant::now();
        builder.run_pass(params, 1.0, &mut rng);
        let pass_one = pass_one_started.elapsed();
        let pass_two_started = Instant::now();
        builder.run_pass(params, params.alpha, &mut rng);
        let mut graph = builder.finish()?;
        let pass_two = pass_two_started.elapsed();
        let connectivity_started = Instant::now();
        graph.repair_connectivity(vectors, dimension, params.max_degree, metric)?;
        let connectivity_repair = connectivity_started.elapsed();
        graph.validate(params.max_degree)?;
        if !graph.is_fully_reachable() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "parallel Vamana graph is not fully reachable from its entry node",
            ));
        }
        Ok((
            graph,
            VamanaBuildStats {
                initialization,
                pass_one,
                pass_two,
                connectivity_repair,
            },
        ))
    }

    pub fn build_sequential(
        vectors: &[f32],
        count: usize,
        dimension: usize,
        params: DiskAnnBuildParams,
    ) -> io::Result<Self> {
        Self::build_sequential_with_metric(vectors, count, dimension, MetricType::L2, params)
    }

    fn build_sequential_with_metric(
        vectors: &[f32],
        count: usize,
        dimension: usize,
        metric: MetricType,
        params: DiskAnnBuildParams,
    ) -> io::Result<Self> {
        validate_build_inputs(vectors, count, dimension, params)?;
        let entry_node = centroid_entry(vectors, count, dimension, metric) as u32;
        let mut rng = StdRng::seed_from_u64(params.seed);
        let degree = params.max_degree.min(count.saturating_sub(1));
        let adjacency = (0..count)
            .map(|node| random_neighbors(&mut rng, count, node, degree))
            .collect::<Vec<_>>();
        let mut graph = Self {
            entry_node,
            adjacency: CompactAdjacency::from_nested(adjacency, degree),
        };

        graph.run_sequential_pass(vectors, dimension, metric, params, 1.0, &mut rng);
        graph.run_sequential_pass(vectors, dimension, metric, params, params.alpha, &mut rng);
        graph.repair_connectivity(vectors, dimension, params.max_degree, metric)?;
        graph.validate(params.max_degree)?;
        if !graph.is_fully_reachable() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Vamana graph is not fully reachable from its entry node",
            ));
        }
        Ok(graph)
    }

    fn run_sequential_pass(
        &mut self,
        vectors: &[f32],
        dimension: usize,
        metric: MetricType,
        params: DiskAnnBuildParams,
        alpha: f32,
        rng: &mut StdRng,
    ) {
        let mut order = (0..self.adjacency.len()).collect::<Vec<_>>();
        order.shuffle(rng);
        for node in order {
            let query_start = node * dimension;
            let candidates = self.greedy_search_with_metric(
                vectors,
                dimension,
                metric,
                &vectors[query_start..query_start + dimension],
                params.build_search_list_size,
            );
            let candidate_ids = candidates
                .into_iter()
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>();
            let selected = self.robust_prune_with_metric(
                vectors,
                dimension,
                metric,
                node,
                &candidate_ids,
                params.max_degree,
                alpha,
            );
            self.adjacency.replace(node, &selected);
            for neighbor in selected {
                self.insert_reverse_edge(
                    vectors,
                    dimension,
                    metric,
                    neighbor as usize,
                    node as u32,
                    params.max_degree,
                    alpha,
                );
            }
        }
    }

    fn insert_reverse_edge(
        &mut self,
        vectors: &[f32],
        dimension: usize,
        metric: MetricType,
        node: usize,
        neighbor: u32,
        max_degree: usize,
        alpha: f32,
    ) {
        if self.adjacency[node].contains(&neighbor) {
            return;
        }
        if self.adjacency[node].len() < max_degree {
            self.adjacency.push(node, neighbor);
            return;
        }
        let mut candidates = self.adjacency[node].to_vec();
        candidates.push(neighbor);
        let selected = self.robust_prune_with_metric(
            vectors,
            dimension,
            metric,
            node,
            &candidates,
            max_degree,
            alpha,
        );
        self.adjacency.replace(node, &selected);
    }

    pub fn greedy_search(
        &self,
        vectors: &[f32],
        dimension: usize,
        query: &[f32],
        search_list_size: usize,
    ) -> Vec<ScoredNode> {
        self.greedy_search_with_metric(vectors, dimension, MetricType::L2, query, search_list_size)
    }

    fn greedy_search_with_metric(
        &self,
        vectors: &[f32],
        dimension: usize,
        metric: MetricType,
        query: &[f32],
        search_list_size: usize,
    ) -> Vec<ScoredNode> {
        let mut scratch = self.search_scratch(search_list_size);
        self.greedy_search_with_scratch(
            vectors,
            dimension,
            metric,
            query,
            search_list_size,
            &mut scratch,
        );
        scratch.results.into_sorted_vec()
    }

    pub(crate) fn greedy_search_best_with_scratch(
        &self,
        vectors: &[f32],
        dimension: usize,
        query: &[f32],
        search_list_size: usize,
        scratch: &mut GreedySearchScratch,
    ) -> Option<ScoredNode> {
        self.greedy_search_with_scratch(
            vectors,
            dimension,
            MetricType::L2,
            query,
            search_list_size,
            scratch,
        );
        scratch.results.iter().min().copied()
    }

    fn greedy_search_with_scratch(
        &self,
        vectors: &[f32],
        dimension: usize,
        metric: MetricType,
        query: &[f32],
        search_list_size: usize,
        scratch: &mut GreedySearchScratch,
    ) {
        scratch.begin_search();
        if search_list_size == 0 || self.adjacency.is_empty() {
            return;
        }

        let entry = self.entry_node as usize;
        scratch.insert_candidate(
            ScoredNode {
                id: self.entry_node,
                distance: node_distance(vectors, dimension, entry, query, metric),
            },
            search_list_size,
        );

        while let Some(current) = scratch.pop_nearest_unexpanded() {
            scratch.mark_expanded(current.id as usize);
            for &neighbor in &self.adjacency[current.id as usize] {
                let neighbor = neighbor as usize;
                if neighbor >= self.adjacency.len() || scratch.is_visited(neighbor) {
                    continue;
                }
                scratch.insert_candidate(
                    ScoredNode {
                        id: neighbor as u32,
                        distance: node_distance(vectors, dimension, neighbor, query, metric),
                    },
                    search_list_size,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn robust_prune(
        &self,
        vectors: &[f32],
        dimension: usize,
        node: usize,
        candidates: &[u32],
        max_degree: usize,
        alpha: f32,
    ) -> Vec<u32> {
        self.robust_prune_with_metric(
            vectors,
            dimension,
            MetricType::L2,
            node,
            candidates,
            max_degree,
            alpha,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn robust_prune_with_metric(
        &self,
        vectors: &[f32],
        dimension: usize,
        metric: MetricType,
        node: usize,
        candidates: &[u32],
        max_degree: usize,
        alpha: f32,
    ) -> Vec<u32> {
        robust_prune_candidates(
            vectors,
            dimension,
            node,
            candidates,
            &self.adjacency[node],
            self.adjacency.len(),
            max_degree,
            alpha,
            metric,
        )
    }

    pub fn is_fully_reachable(&self) -> bool {
        if self.adjacency.is_empty() || self.entry_node as usize >= self.adjacency.len() {
            return false;
        }
        let mut visited = vec![false; self.adjacency.len()];
        let mut queue = VecDeque::from([self.entry_node as usize]);
        visited[self.entry_node as usize] = true;
        while let Some(node) = queue.pop_front() {
            for &neighbor in &self.adjacency[node] {
                let neighbor = neighbor as usize;
                if neighbor < visited.len() && !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        visited.into_iter().all(|value| value)
    }

    fn repair_connectivity(
        &mut self,
        vectors: &[f32],
        dimension: usize,
        max_degree: usize,
        metric: MetricType,
    ) -> io::Result<()> {
        self.repair_connectivity_with_stats(vectors, dimension, max_degree, metric)
            .map(|_| ())
    }

    fn repair_connectivity_with_stats(
        &mut self,
        vectors: &[f32],
        dimension: usize,
        max_degree: usize,
        metric: MetricType,
    ) -> io::Result<ConnectivityRepairStats> {
        let (mut visited, mut parent) = self.reachability_tree();
        let mut stats = ConnectivityRepairStats {
            full_reachability_traversals: 1,
            ..ConnectivityRepairStats::default()
        };
        let mut eligible_sources = visited
            .iter()
            .enumerate()
            .filter_map(|(node, &reachable)| {
                (reachable && self.is_repair_source(node, max_degree, &parent)).then_some(node)
            })
            .collect::<Vec<_>>();
        let mut queue = VecDeque::new();
        let mut newly_reachable = Vec::new();
        let mut target_cursor = 0usize;

        loop {
            while target_cursor < visited.len() && visited[target_cursor] {
                target_cursor += 1;
            }
            if target_cursor == visited.len() {
                return Ok(stats);
            }
            let target = target_cursor;
            let target_vector = &vectors[target * dimension..target * dimension + dimension];
            let (source_index, source) = self.select_repair_source(
                vectors,
                dimension,
                target,
                target_vector,
                max_degree,
                &parent,
                &eligible_sources,
                &mut stats,
                metric,
            )?;
            eligible_sources.swap_remove(source_index);

            if self.adjacency[source].len() == max_degree {
                let removable = self.adjacency[source]
                    .iter()
                    .enumerate()
                    .filter(|(_, neighbor)| parent[**neighbor as usize] != Some(source))
                    .map(|(slot, &neighbor)| {
                        (
                            slot,
                            node_distance(
                                vectors,
                                dimension,
                                neighbor as usize,
                                &vectors[source * dimension..source * dimension + dimension],
                                metric,
                            ),
                        )
                    })
                    .max_by(|left, right| {
                        left.1
                            .total_cmp(&right.1)
                            .then_with(|| left.0.cmp(&right.0))
                    })
                    .map(|(slot, _)| slot)
                    .expect("eligible full source has a non-tree edge");
                self.adjacency.swap_remove(source, removable);
            }
            self.adjacency.push(source, target as u32);
            stats.edges_added += 1;

            visited[target] = true;
            parent[target] = Some(source);
            queue.push_back(target);
            newly_reachable.clear();
            newly_reachable.push(target);
            while let Some(node) = queue.pop_front() {
                for &neighbor in &self.adjacency[node] {
                    let neighbor = neighbor as usize;
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        parent[neighbor] = Some(node);
                        queue.push_back(neighbor);
                        newly_reachable.push(neighbor);
                    }
                }
            }
            if self.is_repair_source(source, max_degree, &parent) {
                eligible_sources.push(source);
            }
            for &node in &newly_reachable {
                if self.is_repair_source(node, max_degree, &parent) {
                    eligible_sources.push(node);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn select_repair_source(
        &self,
        vectors: &[f32],
        dimension: usize,
        target: usize,
        target_vector: &[f32],
        max_degree: usize,
        parent: &[Option<usize>],
        eligible_sources: &[usize],
        stats: &mut ConnectivityRepairStats,
        metric: MetricType,
    ) -> io::Result<(usize, usize)> {
        if eligible_sources.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Vamana connectivity repair found no replaceable edge",
            ));
        }
        let sample_size = CONNECTIVITY_SOURCE_SAMPLE_SIZE.min(eligible_sources.len());
        let start =
            derived_seed(target as u64, stats.edges_added as u64) as usize % eligible_sources.len();
        let mut best: Option<(usize, usize, f32)> = None;
        for offset in 0..sample_size {
            let index = (start + offset) % eligible_sources.len();
            let source = eligible_sources[index];
            if !self.is_repair_source(source, max_degree, parent) {
                continue;
            }
            stats.source_distance_evaluations += 1;
            let distance = node_distance(vectors, dimension, source, target_vector, metric);
            let candidate = (index, source, distance);
            if best.is_none_or(|(_, best_source, best_distance)| {
                distance
                    .total_cmp(&best_distance)
                    .then_with(|| source.cmp(&best_source))
                    .is_lt()
            }) {
                best = Some(candidate);
            }
        }
        if let Some((index, source, _)) = best {
            return Ok((index, source));
        }
        eligible_sources
            .iter()
            .enumerate()
            .find_map(|(index, &source)| {
                self.is_repair_source(source, max_degree, parent)
                    .then_some((index, source))
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Vamana connectivity repair found no replaceable edge",
                )
            })
    }

    fn is_repair_source(&self, source: usize, max_degree: usize, parent: &[Option<usize>]) -> bool {
        self.adjacency[source].len() < max_degree
            || self.adjacency[source]
                .iter()
                .any(|neighbor| parent[*neighbor as usize] != Some(source))
    }

    fn reachability_tree(&self) -> (Vec<bool>, Vec<Option<usize>>) {
        let mut visited = vec![false; self.adjacency.len()];
        let mut parent = vec![None; self.adjacency.len()];
        let entry = self.entry_node as usize;
        let mut queue = VecDeque::from([entry]);
        visited[entry] = true;
        while let Some(node) = queue.pop_front() {
            for &neighbor in &self.adjacency[node] {
                let neighbor = neighbor as usize;
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = Some(node);
                    queue.push_back(neighbor);
                }
            }
        }
        (visited, parent)
    }

    fn validate(&self, max_degree: usize) -> io::Result<()> {
        for (node, neighbors) in self.adjacency.iter().enumerate() {
            if neighbors.len() > max_degree {
                return Err(invalid_input(format!(
                    "Vamana node {} degree {} exceeds maximum {}",
                    node,
                    neighbors.len(),
                    max_degree
                )));
            }
            let mut sorted = neighbors.to_vec();
            sorted.sort_unstable();
            if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(invalid_input(format!(
                    "Vamana node {} contains duplicate neighbors",
                    node
                )));
            }
            if sorted.iter().any(|&neighbor| {
                neighbor as usize >= self.adjacency.len() || neighbor as usize == node
            }) {
                return Err(invalid_input(format!(
                    "Vamana node {} contains invalid neighbor",
                    node
                )));
            }
        }
        Ok(())
    }
}

fn overlapping_shard_capacity(node_count: usize, shard_count: usize) -> Option<usize> {
    if shard_count < 2 {
        return None;
    }
    if shard_count == 2 {
        return Some(node_count);
    }
    // For k > 2, ceil(2N / (k - 1)) leaves enough aggregate capacity even
    // when one shard cannot accept a node's second, distinct membership. This
    // lets deterministic overflow repair preserve exactly two memberships
    // without an unbounded degenerate cluster.
    node_count
        .checked_mul(2)?
        .checked_add(shard_count - 2)
        .map(|value| value / (shard_count - 1))
}

fn rebalance_overlapping_shards(
    vectors: &[f32],
    dimension: usize,
    centroids: &[f32],
    assignments: &mut [[usize; 2]],
    memberships: &mut [Vec<u32>],
) -> io::Result<()> {
    let shard_count = memberships.len();
    let capacity = overlapping_shard_capacity(assignments.len(), shard_count)
        .ok_or_else(|| invalid_input("Vamana overlapping-shard capacity overflows"))?;
    for shard in 0..shard_count {
        while memberships[shard].len() > capacity {
            let node = memberships[shard]
                .pop()
                .expect("an overflowing Vamana shard is non-empty") as usize;
            let slot = if assignments[node][0] == shard {
                0
            } else {
                debug_assert_eq!(assignments[node][1], shard);
                1
            };
            let other = assignments[node][1 - slot];
            let vector = &vectors[node * dimension..(node + 1) * dimension];
            let mut replacement = None;
            let mut replacement_distance = f32::INFINITY;
            for candidate in 0..shard_count {
                if candidate == other || memberships[candidate].len() >= capacity {
                    continue;
                }
                let start = candidate * dimension;
                let distance = fvec_l2sqr(vector, &centroids[start..start + dimension]);
                if distance < replacement_distance
                    || (distance == replacement_distance
                        && replacement.is_none_or(|current| candidate < current))
                {
                    replacement = Some(candidate);
                    replacement_distance = distance;
                }
            }
            let replacement = replacement.ok_or_else(|| {
                invalid_input("Vamana overlapping shards cannot be capacity-balanced")
            })?;
            assignments[node][slot] = replacement;
            memberships[replacement].push(node as u32);
        }
    }
    debug_assert!(memberships.iter().all(|members| members.len() <= capacity));
    Ok(())
}

struct ParallelVamanaBuilder<'a> {
    vectors: &'a [f32],
    dimension: usize,
    metric: MetricType,
    entry_node: u32,
    adjacency: ParallelAdjacency,
    search_distance: BuildSearchDistance<'a>,
}

enum BuildSearchDistance<'a> {
    FullPrecision {
        vectors: &'a [f32],
        dimension: usize,
        metric: MetricType,
    },
    ProductQuantized(PqBuildDistance<'a>),
}

impl BuildSearchDistance<'_> {
    #[inline]
    fn between(&self, left: usize, right: usize) -> f32 {
        match self {
            Self::FullPrecision {
                vectors,
                dimension,
                metric,
            } => distance_between(vectors, *dimension, left, right, *metric),
            Self::ProductQuantized(distance) => distance.between(left, right),
        }
    }
}

struct PqBuildDistance<'a> {
    codes: &'a [u8],
    code_size: usize,
    m: usize,
    ksub: usize,
    nbits: usize,
    centroid_distances: Vec<f32>,
}

impl<'a> PqBuildDistance<'a> {
    fn new(
        pq: &ProductQuantizer,
        codes: &'a [u8],
        count: usize,
        metric: MetricType,
    ) -> io::Result<Self> {
        let table_len = pq
            .m
            .checked_mul(pq.ksub)
            .and_then(|value| value.checked_mul(pq.ksub))
            .ok_or_else(|| invalid_input("Vamana PQ build-distance table size overflows usize"))?;
        let mut centroid_distances = Vec::new();
        centroid_distances
            .try_reserve_exact(table_len)
            .map_err(|_| invalid_input("Vamana PQ build-distance table allocation failed"))?;
        centroid_distances.resize(table_len, 0.0);
        for sub in 0..pq.m {
            let chunk_dim = pq.chunk_dim(sub);
            let sub_base = pq.centroid_chunk_base(sub);
            let table_base = sub * pq.ksub * pq.ksub;
            for left in 0..pq.ksub {
                let left_start = sub_base + left * chunk_dim;
                for right in left..pq.ksub {
                    let right_start = sub_base + right * chunk_dim;
                    let distance = fvec_distance(
                        &pq.centroids[left_start..left_start + chunk_dim],
                        &pq.centroids[right_start..right_start + chunk_dim],
                        metric,
                    );
                    centroid_distances[table_base + left * pq.ksub + right] = distance;
                    centroid_distances[table_base + right * pq.ksub + left] = distance;
                }
            }
        }
        let code_size = pq.code_size();
        if codes.len() != count.saturating_mul(code_size) {
            return Err(invalid_input(
                "Vamana PQ code buffer does not match vector count",
            ));
        }
        Ok(Self {
            codes,
            code_size,
            m: pq.m,
            ksub: pq.ksub,
            nbits: pq.nbits,
            centroid_distances,
        })
    }

    #[inline]
    fn code(&self, node: usize, sub: usize) -> usize {
        let codes = &self.codes[node * self.code_size..(node + 1) * self.code_size];
        if self.nbits == 4 {
            let byte = codes[sub / 2];
            usize::from(if sub.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            })
        } else {
            usize::from(codes[sub])
        }
    }

    #[inline]
    fn between(&self, left: usize, right: usize) -> f32 {
        let mut distance = 0.0;
        for sub in 0..self.m {
            let left_code = self.code(left, sub);
            let right_code = self.code(right, sub);
            distance += self.centroid_distances
                [sub * self.ksub * self.ksub + left_code * self.ksub + right_code];
        }
        distance
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdjacencyShard {
    start_node: usize,
    node_count: usize,
    slots: Vec<u32>,
    degrees: Vec<u16>,
}

impl AdjacencyShard {
    fn neighbors(&self, node: usize, max_degree: usize) -> &[u32] {
        let local_node = node - self.start_node;
        let start = local_node * max_degree;
        &self.slots[start..start + self.degrees[local_node] as usize]
    }

    fn neighbors_mut(&mut self, node: usize, max_degree: usize) -> &mut [u32] {
        let local_node = node - self.start_node;
        let start = local_node * max_degree;
        let degree = self.degrees[local_node] as usize;
        &mut self.slots[start..start + degree]
    }

    fn replace(&mut self, node: usize, max_degree: usize, neighbors: &[u32]) {
        debug_assert!(neighbors.len() <= max_degree);
        let local_node = node - self.start_node;
        let start = local_node * max_degree;
        self.slots[start..start + neighbors.len()].copy_from_slice(neighbors);
        self.degrees[local_node] = neighbors.len() as u16;
    }

    fn push(&mut self, node: usize, max_degree: usize, neighbor: u32) {
        let local_node = node - self.start_node;
        let degree = self.degrees[local_node] as usize;
        assert!(degree < max_degree, "compact adjacency node is full");
        self.slots[local_node * max_degree + degree] = neighbor;
        self.degrees[local_node] += 1;
    }

    fn swap_remove(&mut self, node: usize, max_degree: usize, slot: usize) {
        let local_node = node - self.start_node;
        let degree = self.degrees[local_node] as usize;
        assert!(slot < degree, "compact adjacency removal slot is invalid");
        let start = local_node * max_degree;
        self.slots[start + slot] = self.slots[start + degree - 1];
        self.degrees[local_node] -= 1;
    }

    fn swap_node_with_buffer(
        &mut self,
        node: usize,
        max_degree: usize,
        slots: &mut [u32],
        degree: &mut u16,
    ) {
        let local_node = node - self.start_node;
        let start = local_node * max_degree;
        for (stored, buffered) in self.slots[start..start + max_degree].iter_mut().zip(slots) {
            std::mem::swap(stored, buffered);
        }
        std::mem::swap(&mut self.degrees[local_node], degree);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompactAdjacency {
    shards: Vec<AdjacencyShard>,
    node_count: usize,
    max_degree: usize,
}

impl CompactAdjacency {
    fn empty(node_count: usize, max_degree: usize) -> Self {
        let mut shards =
            Vec::with_capacity(node_count.div_ceil(PARALLEL_ADJACENCY_NODES_PER_SHARD));
        for start_node in (0..node_count).step_by(PARALLEL_ADJACENCY_NODES_PER_SHARD) {
            let shard_node_count = PARALLEL_ADJACENCY_NODES_PER_SHARD.min(node_count - start_node);
            shards.push(AdjacencyShard {
                start_node,
                node_count: shard_node_count,
                slots: vec![0; shard_node_count * max_degree],
                degrees: vec![0; shard_node_count],
            });
        }
        Self {
            shards,
            node_count,
            max_degree,
        }
    }

    fn from_nested(adjacency: Vec<Vec<u32>>, max_degree: usize) -> Self {
        let node_count = adjacency.len();
        let mut shards =
            Vec::with_capacity(node_count.div_ceil(PARALLEL_ADJACENCY_NODES_PER_SHARD));
        for start_node in (0..node_count).step_by(PARALLEL_ADJACENCY_NODES_PER_SHARD) {
            let shard_node_count = PARALLEL_ADJACENCY_NODES_PER_SHARD.min(node_count - start_node);
            let mut slots = vec![0; shard_node_count * max_degree];
            let mut degrees = vec![0; shard_node_count];
            for local_node in 0..shard_node_count {
                let neighbors = &adjacency[start_node + local_node];
                assert!(neighbors.len() <= max_degree);
                let slot_start = local_node * max_degree;
                slots[slot_start..slot_start + neighbors.len()].copy_from_slice(neighbors);
                degrees[local_node] = u16::try_from(neighbors.len())
                    .expect("Vamana degree exceeds compact adjacency metadata");
            }
            shards.push(AdjacencyShard {
                start_node,
                node_count: shard_node_count,
                slots,
                degrees,
            });
        }
        Self {
            shards,
            node_count,
            max_degree,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.node_count
    }

    fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    pub(crate) fn neighbors(&self, node: usize) -> &[u32] {
        self.shards[ParallelAdjacency::shard_index(node)].neighbors(node, self.max_degree)
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &[u32]> {
        (0..self.node_count).map(|node| self.neighbors(node))
    }

    fn replace(&mut self, node: usize, neighbors: &[u32]) {
        self.shards[ParallelAdjacency::shard_index(node)].replace(node, self.max_degree, neighbors);
    }

    fn push(&mut self, node: usize, neighbor: u32) {
        self.shards[ParallelAdjacency::shard_index(node)].push(node, self.max_degree, neighbor);
    }

    fn swap_remove(&mut self, node: usize, slot: usize) {
        self.shards[ParallelAdjacency::shard_index(node)].swap_remove(node, self.max_degree, slot);
    }

    pub(crate) fn permute_and_map_neighbors(&mut self, old_to_new: &[u32]) {
        assert_eq!(old_to_new.len(), self.node_count);
        let mut visited = vec![false; self.node_count];
        let mut slots = vec![0; self.max_degree];
        for start in 0..self.node_count {
            if visited[start] {
                continue;
            }
            slots.copy_from_slice(self.node_slots(start));
            let mut degree = self.node_degree(start);
            let mut current = start;
            loop {
                visited[current] = true;
                let destination = old_to_new[current] as usize;
                assert!(destination < self.node_count);
                self.swap_node_with_buffer(destination, &mut slots, &mut degree);
                current = destination;
                if current == start {
                    break;
                }
                assert!(
                    !visited[current],
                    "DiskANN locality mapping is not a permutation"
                );
            }
        }
        for node in 0..self.node_count {
            let neighbors = self.neighbors_mut(node);
            for neighbor in neighbors.iter_mut() {
                *neighbor = old_to_new[*neighbor as usize];
            }
            neighbors.sort_unstable();
        }
    }

    fn node_slots(&self, node: usize) -> &[u32] {
        let shard = &self.shards[ParallelAdjacency::shard_index(node)];
        let local_node = node - shard.start_node;
        let start = local_node * self.max_degree;
        &shard.slots[start..start + self.max_degree]
    }

    fn node_degree(&self, node: usize) -> u16 {
        let shard = &self.shards[ParallelAdjacency::shard_index(node)];
        shard.degrees[node - shard.start_node]
    }

    fn neighbors_mut(&mut self, node: usize) -> &mut [u32] {
        self.shards[ParallelAdjacency::shard_index(node)].neighbors_mut(node, self.max_degree)
    }

    fn swap_node_with_buffer(&mut self, node: usize, slots: &mut [u32], degree: &mut u16) {
        self.shards[ParallelAdjacency::shard_index(node)].swap_node_with_buffer(
            node,
            self.max_degree,
            slots,
            degree,
        );
    }
}

impl Index<usize> for CompactAdjacency {
    type Output = [u32];

    fn index(&self, index: usize) -> &Self::Output {
        self.neighbors(index)
    }
}

struct ParallelAdjacency {
    shards: Vec<RwLock<AdjacencyShard>>,
    node_count: usize,
    max_degree: usize,
}

impl ParallelAdjacency {
    fn new_random(node_count: usize, max_degree: usize, seed: u64) -> Self {
        assert!(max_degree <= node_count.saturating_sub(1));
        let shard_count = node_count.div_ceil(PARALLEL_ADJACENCY_NODES_PER_SHARD);
        let shards = (0..shard_count)
            .into_par_iter()
            .map(|shard_index| {
                let start_node = shard_index * PARALLEL_ADJACENCY_NODES_PER_SHARD;
                let shard_node_count =
                    PARALLEL_ADJACENCY_NODES_PER_SHARD.min(node_count - start_node);
                let mut slots = vec![0; shard_node_count * max_degree];
                let degrees = vec![max_degree as u16; shard_node_count];
                let mut rng = StdRng::seed_from_u64(derived_seed(seed, shard_index as u64));
                let mut swaps = SparseTable::with_capacity(max_degree);
                for local_node in 0..shard_node_count {
                    let node = start_node + local_node;
                    let slot_start = local_node * max_degree;
                    sample_random_neighbors_into(
                        &mut rng,
                        node_count,
                        node,
                        &mut swaps,
                        &mut slots[slot_start..slot_start + max_degree],
                    );
                }
                RwLock::new(AdjacencyShard {
                    start_node,
                    node_count: shard_node_count,
                    slots,
                    degrees,
                })
            })
            .collect();
        Self {
            shards,
            node_count,
            max_degree,
        }
    }

    #[cfg(test)]
    fn new(
        node_count: usize,
        max_degree: usize,
        mut neighbors_for: impl FnMut(usize) -> Vec<u32>,
    ) -> Self {
        let mut shards =
            Vec::with_capacity(node_count.div_ceil(PARALLEL_ADJACENCY_NODES_PER_SHARD));
        for start_node in (0..node_count).step_by(PARALLEL_ADJACENCY_NODES_PER_SHARD) {
            let shard_node_count = PARALLEL_ADJACENCY_NODES_PER_SHARD.min(node_count - start_node);
            let mut slots = vec![0; shard_node_count * max_degree];
            let mut degrees = vec![0; shard_node_count];
            for local_node in 0..shard_node_count {
                let neighbors = neighbors_for(start_node + local_node);
                assert!(
                    neighbors.len() <= max_degree,
                    "parallel Vamana initializer exceeds maximum degree"
                );
                let slot_start = local_node * max_degree;
                slots[slot_start..slot_start + neighbors.len()].copy_from_slice(&neighbors);
                degrees[local_node] = neighbors.len() as u16;
            }
            shards.push(RwLock::new(AdjacencyShard {
                start_node,
                node_count: shard_node_count,
                slots,
                degrees,
            }));
        }
        Self {
            shards,
            node_count,
            max_degree,
        }
    }

    fn node_count(&self) -> usize {
        self.node_count
    }

    fn shard_index(node: usize) -> usize {
        node / PARALLEL_ADJACENCY_NODES_PER_SHARD
    }

    fn copy_neighbors(&self, node: usize, target: &mut Vec<u32>) {
        let shard = self.shards[Self::shard_index(node)]
            .read()
            .expect("parallel Vamana adjacency shard lock poisoned");
        target.extend_from_slice(shard.neighbors(node, self.max_degree));
    }

    fn replace(&self, node: usize, neighbors: &[u32]) {
        let mut shard = self.shards[Self::shard_index(node)]
            .write()
            .expect("parallel Vamana adjacency shard lock poisoned");
        shard.replace(node, self.max_degree, neighbors);
    }

    fn update_from_buffer(
        &self,
        node: usize,
        replacement: &mut Vec<u32>,
        update: impl FnOnce(&[u32], &mut Vec<u32>),
    ) {
        let mut shard = self.shards[Self::shard_index(node)]
            .write()
            .expect("parallel Vamana adjacency shard lock poisoned");
        update(shard.neighbors(node, self.max_degree), replacement);
        shard.replace(node, self.max_degree, replacement);
    }

    fn into_adjacency(self) -> io::Result<CompactAdjacency> {
        let mut shards = Vec::with_capacity(self.shards.len());
        for shard in self.shards {
            let shard = shard.into_inner().map_err(|_| {
                io::Error::other("parallel Vamana adjacency shard lock poisoned during finish")
            })?;
            shards.push(shard);
        }
        Ok(CompactAdjacency {
            shards,
            node_count: self.node_count,
            max_degree: self.max_degree,
        })
    }
}

enum BuildVisitStates {
    Dense {
        states: Vec<u8>,
        touched_nodes: Vec<u32>,
    },
    Sparse(SparseTable<u8>),
}

pub(crate) struct GreedySearchScratch {
    visit_states: BuildVisitStates,
    results: BinaryHeap<ScoredNode>,
    frontier: BinaryHeap<Reverse<ScoredNode>>,
    neighbor_buffer: Vec<u32>,
    candidate_ids: Vec<u32>,
    prune_unique: Vec<u32>,
    prune_pool: Vec<ScoredNode>,
    prune_selected: Vec<u32>,
    #[cfg(test)]
    peak_retained: usize,
    #[cfg(test)]
    peak_frontier: usize,
}

impl GreedySearchScratch {
    fn new(node_count: usize, max_degree: usize, search_list_size: usize) -> Self {
        let expected_visited = search_list_size
            .saturating_mul(max_degree)
            .saturating_add(1)
            .min(node_count);
        let dense_bytes = node_count
            .saturating_mul(size_of::<u8>())
            .saturating_add(expected_visited.saturating_mul(size_of::<u32>()));
        let sparse_bytes =
            sparse_table_memory_bytes(expected_visited, size_of::<u8>()).unwrap_or(usize::MAX);
        let visit_states = if sparse_bytes
            .checked_mul(SPARSE_BUILD_VISITED_MIN_MEMORY_SAVINGS)
            .is_some_and(|threshold| threshold < dense_bytes)
        {
            BuildVisitStates::Sparse(SparseTable::with_capacity(expected_visited))
        } else {
            BuildVisitStates::Dense {
                states: vec![0; node_count],
                touched_nodes: Vec::with_capacity(expected_visited),
            }
        };
        Self {
            visit_states,
            results: BinaryHeap::new(),
            frontier: BinaryHeap::new(),
            neighbor_buffer: Vec::with_capacity(max_degree),
            candidate_ids: Vec::with_capacity(search_list_size),
            prune_unique: Vec::with_capacity(search_list_size.saturating_add(max_degree)),
            prune_pool: Vec::with_capacity(search_list_size.saturating_add(max_degree)),
            prune_selected: Vec::with_capacity(max_degree),
            #[cfg(test)]
            peak_retained: 0,
            #[cfg(test)]
            peak_frontier: 0,
        }
    }

    fn begin_search(&mut self) {
        match &mut self.visit_states {
            BuildVisitStates::Dense {
                states,
                touched_nodes,
            } => {
                for node in touched_nodes.drain(..) {
                    states[node as usize] = 0;
                }
            }
            BuildVisitStates::Sparse(states) => states.clear(),
        }
        self.results.clear();
        self.frontier.clear();
        self.neighbor_buffer.clear();
        #[cfg(test)]
        {
            self.peak_retained = 0;
            self.peak_frontier = 0;
        }
    }

    fn is_visited(&self, node: usize) -> bool {
        match &self.visit_states {
            BuildVisitStates::Dense { states, .. } => states[node] != 0,
            BuildVisitStates::Sparse(states) => states.get(node as u32).is_some(),
        }
    }

    fn mark_visited(&mut self, node: usize) {
        match &mut self.visit_states {
            BuildVisitStates::Dense {
                states,
                touched_nodes,
            } => {
                if states[node] == 0 {
                    touched_nodes.push(node as u32);
                }
                states[node] = 1;
            }
            BuildVisitStates::Sparse(states) => {
                states.insert(node as u32, 1);
            }
        }
    }

    #[cfg(test)]
    fn is_expanded(&self, node: usize) -> bool {
        match &self.visit_states {
            BuildVisitStates::Dense { states, .. } => states[node] == 3,
            BuildVisitStates::Sparse(states) => states.get(node as u32) == Some(&3),
        }
    }

    fn is_retained_unexpanded(&self, node: usize) -> bool {
        match &self.visit_states {
            BuildVisitStates::Dense { states, .. } => states[node] == 2,
            BuildVisitStates::Sparse(states) => states.get(node as u32) == Some(&2),
        }
    }

    fn mark_expanded(&mut self, node: usize) {
        match &mut self.visit_states {
            BuildVisitStates::Dense { states, .. } => states[node] = 3,
            BuildVisitStates::Sparse(states) => {
                states.insert(node as u32, 3);
            }
        }
    }

    fn mark_retained(&mut self, node: usize) {
        match &mut self.visit_states {
            BuildVisitStates::Dense { states, .. } => states[node] = 2,
            BuildVisitStates::Sparse(states) => {
                states.insert(node as u32, 2);
            }
        }
    }

    #[cfg(test)]
    fn uses_sparse_states(&self) -> bool {
        matches!(self.visit_states, BuildVisitStates::Sparse(_))
    }

    fn insert_candidate(&mut self, candidate: ScoredNode, search_list_size: usize) {
        self.mark_visited(candidate.id as usize);
        if search_list_size == 0 {
            return;
        }
        if self.results.len() == search_list_size
            && self.results.peek().is_some_and(|worst| candidate >= *worst)
        {
            return;
        }
        if self.results.len() == search_list_size {
            let evicted = self
                .results
                .pop()
                .expect("full result heap has a worst node");
            self.mark_visited(evicted.id as usize);
        }
        self.results.push(candidate);
        self.frontier.push(Reverse(candidate));
        self.mark_retained(candidate.id as usize);
        if self.frontier.len() > search_list_size.saturating_mul(2) {
            let visit_states = &self.visit_states;
            self.frontier
                .retain(|Reverse(candidate)| match visit_states {
                    BuildVisitStates::Dense { states, .. } => states[candidate.id as usize] == 2,
                    BuildVisitStates::Sparse(states) => states.get(candidate.id) == Some(&2),
                });
        }
        #[cfg(test)]
        {
            self.peak_retained = self.peak_retained.max(self.results.len());
            self.peak_frontier = self.peak_frontier.max(self.frontier.len());
        }
    }

    fn pop_nearest_unexpanded(&mut self) -> Option<ScoredNode> {
        while let Some(Reverse(candidate)) = self.frontier.pop() {
            if self.is_retained_unexpanded(candidate.id as usize) {
                return Some(candidate);
            }
        }
        None
    }

    #[cfg(test)]
    fn peak_retained_len(&self) -> usize {
        self.peak_retained
    }

    #[cfg(test)]
    fn peak_frontier_len(&self) -> usize {
        self.peak_frontier
    }
}

impl ParallelVamanaBuilder<'_> {
    fn run_pass(&self, params: DiskAnnBuildParams, alpha: f32, rng: &mut StdRng) {
        let mut order = (0..self.adjacency.node_count()).collect::<Vec<_>>();
        order.shuffle(rng);
        let worker_count = rayon::current_num_threads().max(1);
        let scratches = (0..worker_count)
            .map(|_| {
                Mutex::new(GreedySearchScratch::new(
                    self.adjacency.node_count(),
                    params.max_degree,
                    params.build_search_list_size,
                ))
            })
            .collect::<Vec<_>>();
        let mut reverse_edges =
            Vec::with_capacity(PARALLEL_BUILD_BATCH_NODES.saturating_mul(params.max_degree));
        for batch in order.chunks(PARALLEL_BUILD_BATCH_NODES) {
            // Every node in a batch searches and prunes against the same graph
            // snapshot. Results are committed in shuffled order below, making
            // the persisted graph independent of Rayon scheduling and worker
            // count without serializing the expensive distance work.
            let mut selected_by_node = (0..batch.len())
                .map(|_| Vec::with_capacity(params.max_degree))
                .collect::<Vec<_>>();
            batch
                .par_iter()
                .zip(selected_by_node.par_iter_mut())
                .for_each(|(&node, selected)| {
                    let worker = rayon::current_thread_index().unwrap_or(0) % worker_count;
                    let mut scratch = scratches[worker]
                        .lock()
                        .expect("parallel Vamana scratch lock poisoned");
                    self.greedy_search(node, params.build_search_list_size, &mut scratch);
                    let mut candidate_ids = std::mem::take(&mut scratch.candidate_ids);
                    candidate_ids.clear();
                    candidate_ids.extend(scratch.results.iter().map(|candidate| candidate.id));
                    let mut unique = std::mem::take(&mut scratch.prune_unique);
                    let mut pool = std::mem::take(&mut scratch.prune_pool);
                    self.adjacency
                        .copy_neighbors(node, &mut scratch.neighbor_buffer);
                    robust_prune_candidates_into(
                        self.vectors,
                        self.dimension,
                        node,
                        &candidate_ids,
                        &scratch.neighbor_buffer,
                        self.adjacency.node_count(),
                        params.max_degree,
                        alpha,
                        self.metric,
                        &mut unique,
                        &mut pool,
                        selected,
                    );
                    scratch.neighbor_buffer.clear();
                    scratch.candidate_ids = candidate_ids;
                    scratch.prune_unique = unique;
                    scratch.prune_pool = pool;
                });

            reverse_edges.clear();
            for (&node, selected) in batch.iter().zip(&selected_by_node) {
                self.adjacency.replace(node, selected);
                reverse_edges.extend(selected.iter().map(|&neighbor| (neighbor, node as u32)));
            }
            let reverse_edge_groups = group_reverse_edges(&mut reverse_edges);
            reverse_edge_groups.par_iter().for_each(|group| {
                let worker = rayon::current_thread_index().unwrap_or(0) % worker_count;
                let mut scratch = scratches[worker]
                    .lock()
                    .expect("parallel Vamana scratch lock poisoned");
                self.insert_reverse_edges(
                    reverse_edges[group.start].0 as usize,
                    &reverse_edges[group.clone()],
                    params.max_degree,
                    alpha,
                    &mut scratch,
                );
            });
        }
    }

    fn greedy_search(
        &self,
        query_node: usize,
        search_list_size: usize,
        scratch: &mut GreedySearchScratch,
    ) {
        scratch.begin_search();
        let entry = self.entry_node as usize;
        scratch.insert_candidate(
            ScoredNode {
                id: self.entry_node,
                distance: self.search_distance.between(entry, query_node),
            },
            search_list_size,
        );

        while let Some(current) = scratch.pop_nearest_unexpanded() {
            scratch.mark_expanded(current.id as usize);
            self.adjacency
                .copy_neighbors(current.id as usize, &mut scratch.neighbor_buffer);
            for slot in 0..scratch.neighbor_buffer.len() {
                let neighbor = scratch.neighbor_buffer[slot];
                let neighbor = neighbor as usize;
                if neighbor >= self.adjacency.node_count() || scratch.is_visited(neighbor) {
                    continue;
                }
                scratch.insert_candidate(
                    ScoredNode {
                        id: neighbor as u32,
                        distance: self.search_distance.between(neighbor, query_node),
                    },
                    search_list_size,
                );
            }
            scratch.neighbor_buffer.clear();
        }
    }

    fn insert_reverse_edges(
        &self,
        node: usize,
        reverse_edges: &[(u32, u32)],
        max_degree: usize,
        alpha: f32,
        scratch: &mut GreedySearchScratch,
    ) {
        let mut incoming = std::mem::take(&mut scratch.candidate_ids);
        incoming.clear();
        incoming.extend(reverse_edges.iter().map(|&(_, source)| source));
        let mut selected = std::mem::take(&mut scratch.prune_selected);
        let mut unique = std::mem::take(&mut scratch.prune_unique);
        let mut pool = std::mem::take(&mut scratch.prune_pool);
        self.adjacency
            .update_from_buffer(node, &mut selected, |neighbors, selected| {
                let missing = incoming
                    .iter()
                    .filter(|neighbor| !neighbors.contains(neighbor))
                    .count();
                if missing == 0 {
                    selected.clear();
                    selected.extend_from_slice(neighbors);
                    return;
                }
                if neighbors.len().saturating_add(missing) <= max_degree {
                    selected.clear();
                    selected.extend_from_slice(neighbors);
                    selected.extend(
                        incoming
                            .iter()
                            .copied()
                            .filter(|neighbor| !neighbors.contains(neighbor)),
                    );
                    return;
                }
                robust_prune_candidates_into(
                    self.vectors,
                    self.dimension,
                    node,
                    &incoming,
                    neighbors,
                    self.adjacency.node_count(),
                    max_degree,
                    alpha,
                    self.metric,
                    &mut unique,
                    &mut pool,
                    selected,
                );
            });
        scratch.candidate_ids = incoming;
        scratch.prune_selected = selected;
        scratch.prune_unique = unique;
        scratch.prune_pool = pool;
    }

    fn finish(self) -> io::Result<VamanaGraph> {
        let adjacency = self.adjacency.into_adjacency()?;
        Ok(VamanaGraph {
            entry_node: self.entry_node,
            adjacency,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn robust_prune_candidates(
    vectors: &[f32],
    dimension: usize,
    node: usize,
    candidates: &[u32],
    current_neighbors: &[u32],
    node_count: usize,
    max_degree: usize,
    alpha: f32,
    metric: MetricType,
) -> Vec<u32> {
    let mut unique = Vec::new();
    let mut pool = Vec::new();
    let mut selected = Vec::new();
    robust_prune_candidates_into(
        vectors,
        dimension,
        node,
        candidates,
        current_neighbors,
        node_count,
        max_degree,
        alpha,
        metric,
        &mut unique,
        &mut pool,
        &mut selected,
    );
    selected
}

#[allow(clippy::too_many_arguments)]
fn robust_prune_candidates_into(
    vectors: &[f32],
    dimension: usize,
    node: usize,
    candidates: &[u32],
    current_neighbors: &[u32],
    node_count: usize,
    max_degree: usize,
    alpha: f32,
    metric: MetricType,
    unique: &mut Vec<u32>,
    pool: &mut Vec<ScoredNode>,
    selected: &mut Vec<u32>,
) {
    unique.clear();
    unique.extend(
        candidates
            .iter()
            .chain(current_neighbors.iter())
            .copied()
            .filter(|candidate| *candidate as usize != node && (*candidate as usize) < node_count),
    );
    unique.sort_unstable();
    unique.dedup();
    pool.clear();
    pool.extend(unique.iter().copied().map(|candidate| ScoredNode {
        id: candidate,
        distance: distance_between(vectors, dimension, node, candidate as usize, metric),
    }));
    pool.sort_unstable_by(scored_node_order);

    selected.clear();
    let mut next = 0usize;
    while next < pool.len() && selected.len() < max_degree {
        let pivot = pool[next];
        selected.push(pivot.id);
        next += 1;
        let mut retained = next;
        if metric == MetricType::L2 && dimension < 256 {
            let mut candidate_index = next;
            while candidate_index + 4 <= pool.len() {
                let candidates = [
                    pool[candidate_index],
                    pool[candidate_index + 1],
                    pool[candidate_index + 2],
                    pool[candidate_index + 3],
                ];
                let distances = distance_between_four(
                    vectors,
                    dimension,
                    pivot.id as usize,
                    [
                        candidates[0].id as usize,
                        candidates[1].id as usize,
                        candidates[2].id as usize,
                        candidates[3].id as usize,
                    ],
                );
                for (candidate, distance) in candidates.into_iter().zip(distances) {
                    if retain_after_prune(metric, distance, candidate.distance, alpha) {
                        pool[retained] = candidate;
                        retained += 1;
                    }
                }
                candidate_index += 4;
            }
            for candidate_index in candidate_index..pool.len() {
                let candidate = pool[candidate_index];
                let distance = distance_between(
                    vectors,
                    dimension,
                    pivot.id as usize,
                    candidate.id as usize,
                    metric,
                );
                if retain_after_prune(metric, distance, candidate.distance, alpha) {
                    pool[retained] = candidate;
                    retained += 1;
                }
            }
        } else if metric == MetricType::L2 {
            for candidate_index in next..pool.len() {
                let candidate = pool[candidate_index];
                if distance_between_exceeds(
                    vectors,
                    dimension,
                    pivot.id as usize,
                    candidate.id as usize,
                    alpha,
                    candidate.distance,
                ) {
                    pool[retained] = candidate;
                    retained += 1;
                }
            }
        } else {
            for candidate_index in next..pool.len() {
                let candidate = pool[candidate_index];
                let distance = distance_between(
                    vectors,
                    dimension,
                    pivot.id as usize,
                    candidate.id as usize,
                    metric,
                );
                if retain_after_prune(metric, distance, candidate.distance, alpha) {
                    pool[retained] = candidate;
                    retained += 1;
                }
            }
        }
        pool.truncate(retained);
    }
}

#[inline]
fn retain_after_prune(
    metric: MetricType,
    selected_to_candidate: f32,
    source_to_candidate: f32,
    alpha: f32,
) -> bool {
    match metric {
        MetricType::InnerProduct => selected_to_candidate >= alpha * source_to_candidate,
        MetricType::L2 | MetricType::Cosine => alpha * selected_to_candidate > source_to_candidate,
    }
}

fn group_reverse_edges(reverse_edges: &mut Vec<(u32, u32)>) -> Vec<std::ops::Range<usize>> {
    reverse_edges.sort_unstable();
    reverse_edges.dedup();
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < reverse_edges.len() {
        let target = reverse_edges[start].0;
        let mut end = start + 1;
        while end < reverse_edges.len() && reverse_edges[end].0 == target {
            end += 1;
        }
        groups.push(start..end);
        start = end;
    }
    groups
}

fn validate_build_inputs(
    vectors: &[f32],
    count: usize,
    dimension: usize,
    params: DiskAnnBuildParams,
) -> io::Result<()> {
    if count == 0 || dimension == 0 || params.max_degree == 0 {
        return Err(invalid_input(
            "Vamana count, dimension, and maximum degree must be greater than zero",
        ));
    }
    if count > u32::MAX as usize {
        return Err(invalid_input(
            "Vamana vector count exceeds u32 node ID limit",
        ));
    }
    if params.max_degree > u16::MAX as usize {
        return Err(invalid_input(
            "Vamana maximum degree exceeds u16 adjacency degree limit",
        ));
    }
    let expected = count
        .checked_mul(dimension)
        .ok_or_else(|| invalid_input("Vamana vector shape overflows usize"))?;
    if vectors.len() != expected {
        return Err(invalid_input(format!(
            "Vamana vector length {} does not match {}",
            vectors.len(),
            expected
        )));
    }
    Ok(())
}

fn centroid_entry(vectors: &[f32], count: usize, dimension: usize, metric: MetricType) -> usize {
    let mut centroid = vec![0.0f32; dimension];
    for vector in vectors.chunks_exact(dimension) {
        for (sum, &value) in centroid.iter_mut().zip(vector) {
            *sum += value;
        }
    }
    for value in &mut centroid {
        *value /= count as f32;
    }
    (0..count)
        .min_by(|&left, &right| {
            node_distance(vectors, dimension, left, &centroid, metric)
                .total_cmp(&node_distance(vectors, dimension, right, &centroid, metric))
                .then_with(|| left.cmp(&right))
        })
        .expect("validated non-empty Vamana data")
}

fn nearest_two_centroids(
    vector: &[f32],
    centroids: &[f32],
    centroid_count: usize,
    dimension: usize,
) -> [usize; 2] {
    debug_assert!(centroid_count > 0);
    let mut first = (0usize, f32::MAX);
    let mut second = (0usize, f32::MAX);
    for centroid in 0..centroid_count {
        let start = centroid * dimension;
        let distance = fvec_l2sqr(vector, &centroids[start..start + dimension]);
        let candidate = (centroid, distance);
        if distance
            .total_cmp(&first.1)
            .then_with(|| centroid.cmp(&first.0))
            .is_lt()
        {
            second = first;
            first = candidate;
        } else if distance
            .total_cmp(&second.1)
            .then_with(|| centroid.cmp(&second.0))
            .is_lt()
        {
            second = candidate;
        }
    }
    if centroid_count == 1 {
        second = first;
    }
    [first.0, second.0]
}

fn random_neighbors(rng: &mut StdRng, count: usize, node: usize, degree: usize) -> Vec<u32> {
    if degree == count.saturating_sub(1) {
        let mut neighbors = (0..count)
            .filter(|&candidate| candidate != node)
            .map(|candidate| candidate as u32)
            .collect::<Vec<_>>();
        neighbors.shuffle(rng);
        return neighbors;
    }

    let mut neighbors = Vec::with_capacity(degree);
    while neighbors.len() < degree {
        let candidate = rng.gen_range(0..count);
        if candidate != node && !neighbors.contains(&(candidate as u32)) {
            neighbors.push(candidate as u32);
        }
    }
    neighbors
}

fn sample_random_neighbors_into(
    rng: &mut StdRng,
    count: usize,
    node: usize,
    swaps: &mut SparseTable<u32>,
    neighbors: &mut [u32],
) {
    debug_assert!(node < count);
    debug_assert!(neighbors.len() <= count.saturating_sub(1));
    swaps.clear();
    let population = count - 1;
    for (index, neighbor) in neighbors.iter_mut().enumerate() {
        let selected_index = rng.gen_range(index..population);
        let selected = swaps
            .get(selected_index as u32)
            .copied()
            .unwrap_or(selected_index as u32);
        let replacement = swaps.get(index as u32).copied().unwrap_or(index as u32);
        if selected_index != index {
            swaps.insert(selected_index as u32, replacement);
        }
        *neighbor = if selected as usize >= node {
            selected + 1
        } else {
            selected
        };
    }
}

fn derived_seed(seed: u64, stream: u64) -> u64 {
    let mut value = seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn node_distance(
    vectors: &[f32],
    dimension: usize,
    node: usize,
    query: &[f32],
    metric: MetricType,
) -> f32 {
    let start = node * dimension;
    fvec_distance(&vectors[start..start + dimension], query, metric)
}

fn distance_between(
    vectors: &[f32],
    dimension: usize,
    left: usize,
    right: usize,
    metric: MetricType,
) -> f32 {
    let left_start = left * dimension;
    let right_start = right * dimension;
    fvec_distance(
        &vectors[left_start..left_start + dimension],
        &vectors[right_start..right_start + dimension],
        metric,
    )
}

#[inline]
fn distance_between_four(
    vectors: &[f32],
    dimension: usize,
    left: usize,
    rights: [usize; 4],
) -> [f32; 4] {
    let left_start = left * dimension;
    let right_starts = rights.map(|right| right * dimension);
    fvec_l2sqr_four(
        &vectors[left_start..left_start + dimension],
        &vectors[right_starts[0]..right_starts[0] + dimension],
        &vectors[right_starts[1]..right_starts[1] + dimension],
        &vectors[right_starts[2]..right_starts[2] + dimension],
        &vectors[right_starts[3]..right_starts[3] + dimension],
    )
}

#[inline]
fn distance_between_exceeds(
    vectors: &[f32],
    dimension: usize,
    left: usize,
    right: usize,
    scale: f32,
    threshold: f32,
) -> bool {
    let left_start = left * dimension;
    let right_start = right * dimension;
    let left = &vectors[left_start..left_start + dimension];
    let right = &vectors[right_start..right_start + dimension];
    fvec_l2sqr_scaled_exceeds(left, right, scale, threshold)
}

fn scored_node_order(left: &ScoredNode, right: &ScoredNode) -> std::cmp::Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.id.cmp(&right.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::DiskAnnBuildParams;
    use crate::pq::ProductQuantizer;

    #[test]
    fn vamana_pq_build_distance_matches_decoded_centroid_distance_for_4bit_codes() {
        let mut pq = ProductQuantizer::with_nbits(4, 2, 4);
        pq.centroids = (0..pq.d * pq.ksub)
            .map(|index| index as f32 * 0.125)
            .collect();
        let codes = [0x21, 0x43];
        let distance = PqBuildDistance::new(&pq, &codes, 2, MetricType::L2).unwrap();
        let mut left = vec![0.0; pq.d];
        let mut right = vec![0.0; pq.d];
        pq.decode(&codes[..1], &mut left);
        pq.decode(&codes[1..], &mut right);

        assert_eq!(distance.between(0, 1), fvec_l2sqr(&left, &right));
        assert_eq!(distance.between(1, 0), distance.between(0, 1));
        assert_eq!(distance.between(0, 0), 0.0);
    }

    #[test]
    fn vamana_pq_build_distance_and_pruning_follow_inner_product_semantics() {
        let mut pq = ProductQuantizer::with_nbits(2, 1, 4);
        pq.centroids = (0..pq.d * pq.ksub)
            .map(|index| index as f32 * 0.25 - 1.0)
            .collect();
        let codes = [0x01, 0x03];
        let distance = PqBuildDistance::new(&pq, &codes, 2, MetricType::InnerProduct).unwrap();
        let mut left = vec![0.0; pq.d];
        let mut right = vec![0.0; pq.d];
        pq.decode(&codes[..1], &mut left);
        pq.decode(&codes[1..], &mut right);

        assert_eq!(
            distance.between(0, 1),
            fvec_distance(&left, &right, MetricType::InnerProduct)
        );
        assert!(retain_after_prune(
            MetricType::InnerProduct,
            -1.0,
            -2.0,
            1.0
        ));
        assert!(!retain_after_prune(
            MetricType::InnerProduct,
            -4.0,
            -3.0,
            1.0
        ));
    }

    #[test]
    fn vamana_pq_guided_build_keeps_exact_pruned_graph_bounded_and_reachable() {
        let dimension = 8;
        let count = 128;
        let vectors = (0..count * dimension)
            .map(|offset| ((offset * 37) % 101) as f32)
            .collect::<Vec<_>>();
        let mut pq = ProductQuantizer::with_nbits(dimension, 2, 4);
        pq.train(&vectors, count);
        let mut codes = vec![0; count * pq.code_size()];
        pq.encode_batch(&vectors, count, &mut codes);
        let params = DiskAnnBuildParams {
            max_degree: 12,
            build_search_list_size: 32,
            ..DiskAnnBuildParams::default()
        };

        let (graph, _) = VamanaGraph::build_with_pq_stats(
            &vectors,
            &pq,
            &codes,
            count,
            dimension,
            MetricType::L2,
            params,
        )
        .unwrap();

        assert!(graph.is_fully_reachable());
        assert!(graph
            .adjacency
            .iter()
            .all(|neighbors| neighbors.len() <= params.max_degree));
    }

    #[test]
    fn vamana_greedy_search_returns_nearest_unique_nodes() {
        let graph =
            VamanaGraph::from_adjacency(0, vec![vec![1, 2, 2], vec![3], vec![4], vec![], vec![]]);
        let vectors = [0.0, 1.0, 2.0, 3.0, 4.0];

        let result = graph.greedy_search(&vectors, 1, &[3.1], 3);

        assert_eq!(
            result.iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![3, 4, 2]
        );
    }

    #[test]
    fn vamana_greedy_search_caps_search_list_size_to_node_count() {
        let graph = VamanaGraph::from_adjacency(0, vec![vec![]]);
        assert_eq!(graph.greedy_search(&[1.0], 1, &[1.0], usize::MAX).len(), 1);

        let empty = VamanaGraph::from_adjacency(0, vec![]);
        assert!(empty.greedy_search(&[], 1, &[1.0], usize::MAX).is_empty());
    }

    #[test]
    fn vamana_prune_removes_occluded_and_duplicate_neighbors() {
        let graph = VamanaGraph::from_adjacency(0, vec![vec![], vec![], vec![], vec![]]);
        let vectors = [0.0, 1.0, 1.1, -1.0];

        let selected = graph.robust_prune(&vectors, 1, 0, &[1, 2, 3, 3, 0], 2, 1.2);

        assert_eq!(selected, vec![1, 3]);
    }

    #[test]
    fn vamana_build_sequential_is_deterministic_bounded_and_reachable() {
        let dimension = 2;
        let count = 64;
        let vectors = (0..count)
            .flat_map(|node| [(node % 8) as f32, (node / 8) as f32])
            .collect::<Vec<_>>();
        let params = DiskAnnBuildParams {
            max_degree: 8,
            build_search_list_size: 16,
            ..DiskAnnBuildParams::default()
        };

        let first = VamanaGraph::build_sequential(&vectors, count, dimension, params).unwrap();
        let second = VamanaGraph::build_sequential(&vectors, count, dimension, params).unwrap();

        assert_eq!(first, second);
        assert!(first.adjacency.iter().all(|neighbors| neighbors.len() <= 8));
        assert!(first.is_fully_reachable());
    }

    #[test]
    fn vamana_build_parallel_preserves_graph_invariants() {
        let dimension = 4;
        let count = 128;
        let vectors = (0..count * dimension)
            .map(|offset| ((offset * 37) % 101) as f32)
            .collect::<Vec<_>>();
        let params = DiskAnnBuildParams {
            max_degree: 12,
            build_search_list_size: 32,
            ..DiskAnnBuildParams::default()
        };

        let graph = VamanaGraph::build(&vectors, count, dimension, params).unwrap();

        assert!(graph
            .adjacency
            .iter()
            .all(|neighbors| neighbors.len() <= 12));
        assert!(graph.is_fully_reachable());
    }

    #[test]
    fn vamana_parallel_build_is_deterministic_across_worker_counts() {
        let dimension = 4;
        let count = 160;
        let vectors = (0..count * dimension)
            .map(|offset| ((offset * 37) % 101) as f32)
            .collect::<Vec<_>>();
        let params = DiskAnnBuildParams {
            max_degree: 12,
            build_search_list_size: 32,
            seed: 77,
            ..DiskAnnBuildParams::default()
        };
        let build = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| VamanaGraph::build(&vectors, count, dimension, params).unwrap())
        };

        assert_eq!(build(1), build(4));
        assert_eq!(build(4), build(4));
    }

    #[test]
    fn vamana_overlapping_shard_build_is_bounded_and_reachable() {
        let count = 128;
        let dimension = 8;
        let vectors = (0..count * dimension)
            .map(|offset| {
                let cluster = (offset / dimension) % 8;
                cluster as f32 * 10.0 + (offset % dimension) as f32 * 0.01
            })
            .collect::<Vec<_>>();
        let params = DiskAnnBuildParams {
            max_degree: 8,
            build_search_list_size: 16,
            seed: 91,
            ..DiskAnnBuildParams::default()
        };

        let (graph, stats) = VamanaGraph::build_sharded_with_stats(
            &vectors,
            count,
            dimension,
            MetricType::L2,
            params,
            4,
        )
        .unwrap();
        let mut pq = ProductQuantizer::with_nbits_balanced(dimension, 2, 4);
        pq.train(&vectors, count);
        let mut pq_codes = vec![0; count * pq.code_size()];
        pq.encode_batch(&vectors, count, &mut pq_codes);
        let (pq_graph, _) = VamanaGraph::build_sharded_with_pq_stats(
            &vectors,
            &pq,
            &pq_codes,
            count,
            dimension,
            MetricType::L2,
            DiskAnnBuildParams {
                build_distance: crate::diskann::DiskAnnBuildDistance::ProductQuantized,
                ..params
            },
            4,
        )
        .unwrap();

        assert!(graph.is_fully_reachable());
        assert!(pq_graph.is_fully_reachable());
        assert_ne!(
            graph, pq_graph,
            "PQ-guided sharded construction must not silently use the full-precision path"
        );
        assert!(stats.initialization > Duration::ZERO);
        for (node, neighbors) in graph.adjacency.iter().enumerate() {
            assert!(neighbors.len() <= params.max_degree);
            assert!(!neighbors.contains(&(node as u32)));
            let mut unique = neighbors.to_vec();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), neighbors.len());
        }
    }

    #[test]
    fn overlapping_shards_capacity_balance_degenerate_assignments() {
        let count = 128;
        let dimension = 4;
        let shard_count = 8;
        let vectors = vec![0.0; count * dimension];
        let centroids = vec![0.0; shard_count * dimension];
        let mut assignments = vec![[0, 1]; count];
        let mut memberships = (0..shard_count).map(|_| Vec::new()).collect::<Vec<_>>();
        memberships[0].extend(0..count as u32);
        memberships[1].extend(0..count as u32);

        rebalance_overlapping_shards(
            &vectors,
            dimension,
            &centroids,
            &mut assignments,
            &mut memberships,
        )
        .unwrap();

        let capacity = overlapping_shard_capacity(count, shard_count).unwrap();
        assert!(memberships.iter().all(|members| members.len() <= capacity));
        assert!(assignments.iter().all(|pair| pair[0] != pair[1]));
        let mut seen = vec![0usize; count];
        for members in memberships {
            for node in members {
                seen[node as usize] += 1;
            }
        }
        assert!(seen.into_iter().all(|memberships| memberships == 2));
    }

    #[test]
    fn vamana_parallel_build_rejects_degree_above_contiguous_storage_limit() {
        let error = VamanaGraph::build(
            &[0.0, 1.0],
            2,
            1,
            DiskAnnBuildParams {
                max_degree: u16::MAX as usize + 1,
                build_search_list_size: u16::MAX as usize + 1,
                ..DiskAnnBuildParams::default()
            },
        )
        .expect_err("parallel adjacency degree is stored as u16");

        assert!(error.to_string().contains("u16"));
    }

    #[test]
    fn vamana_memory_estimate_includes_candidate_and_batched_reverse_edge_buffers() {
        let search_list_size = 100;
        let max_degree = 64;
        let estimate = estimate_vamana_memory_bytes(0, max_degree, search_list_size, 1).unwrap();
        let expected_worker_bytes = search_list_size * 3 * size_of::<ScoredNode>()
            + (search_list_size + max_degree) * (size_of::<ScoredNode>() + size_of::<u32>())
            + max_degree * 2 * size_of::<u32>()
            + search_list_size * size_of::<u32>()
            + 8 * max_degree * size_of::<(u32, u32)>();

        assert!(estimate.build_peak_bytes >= expected_worker_bytes);
    }

    #[test]
    fn vamana_build_parallel_repairs_disconnected_clustered_graph() {
        let count = 512;
        let dimension = 8;
        let mut vectors = vec![0.0f32; count * dimension];
        for node in 0..count {
            let cluster = node / 128;
            let local = node % 128;
            for dim in 0..dimension {
                vectors[node * dimension + dim] =
                    cluster as f32 * 20.0 + dim as f32 * 0.01 + (local % 16) as f32 * 0.001;
            }
        }
        let params = DiskAnnBuildParams {
            max_degree: 8,
            build_search_list_size: 16,
            ..DiskAnnBuildParams::default()
        };

        let graph = VamanaGraph::build(&vectors, count, dimension, params).unwrap();

        assert!(
            std::any::type_name_of_val(&graph.adjacency).contains("CompactAdjacency"),
            "finished Vamana graph must retain compact adjacency storage"
        );
        assert!(graph.is_fully_reachable());
        assert!(graph
            .adjacency
            .iter()
            .all(|neighbors| neighbors.len() <= params.max_degree));
    }

    #[test]
    fn vamana_connectivity_repair_reuses_reachability_and_bounds_source_search() {
        let count = 1_024;
        let vectors = (0..count).map(|node| node as f32).collect::<Vec<_>>();
        let adjacency = (0..count)
            .map(|node| vec![(node ^ 1) as u32])
            .collect::<Vec<_>>();
        let mut graph = VamanaGraph::from_adjacency(0, adjacency);

        let stats = graph
            .repair_connectivity_with_stats(&vectors, 1, 1, MetricType::L2)
            .unwrap();

        assert!(graph.is_fully_reachable());
        assert_eq!(stats.full_reachability_traversals, 1);
        assert_eq!(stats.edges_added, count / 2 - 1);
        assert!(
            stats.source_distance_evaluations <= stats.edges_added * 64,
            "bounded repair evaluated {} sources for {} edges",
            stats.source_distance_evaluations,
            stats.edges_added
        );
    }

    #[test]
    fn vamana_parallel_adjacency_stores_nodes_contiguously_and_isolates_updates() {
        let adjacency = ParallelAdjacency::new(4, 3, |node| {
            vec![((node + 1) % 4) as u32, ((node + 2) % 4) as u32]
        });
        let mut neighbors = Vec::new();
        adjacency.copy_neighbors(2, &mut neighbors);
        assert_eq!(neighbors, vec![3, 0]);

        adjacency.replace(2, &[1]);
        neighbors.clear();
        adjacency.copy_neighbors(2, &mut neighbors);
        assert_eq!(neighbors, vec![1]);

        neighbors.clear();
        adjacency.copy_neighbors(1, &mut neighbors);
        assert_eq!(neighbors, vec![2, 3]);
    }

    #[test]
    fn vamana_reverse_edges_are_deduplicated_and_grouped_by_target() {
        let mut reverse_edges = vec![(7, 3), (2, 9), (7, 1), (2, 9), (7, 3), (2, 4)];

        let groups = group_reverse_edges(&mut reverse_edges);

        assert_eq!(reverse_edges, vec![(2, 4), (2, 9), (7, 1), (7, 3)]);
        assert_eq!(groups, vec![0..2, 2..4]);
        assert!(groups.iter().all(|group| reverse_edges[group.clone()]
            .iter()
            .all(|edge| edge.0 == reverse_edges[group.start].0)));
    }

    #[test]
    fn vamana_parallel_random_initialization_is_bounded_unique_and_deterministic() {
        let first = ParallelAdjacency::new_random(1_024, 64, 91);
        let second = ParallelAdjacency::new_random(1_024, 64, 91);
        let mut first_neighbors = Vec::new();
        let mut second_neighbors = Vec::new();

        for node in 0..1_024 {
            first_neighbors.clear();
            second_neighbors.clear();
            first.copy_neighbors(node, &mut first_neighbors);
            second.copy_neighbors(node, &mut second_neighbors);
            assert_eq!(first_neighbors, second_neighbors);
            assert_eq!(first_neighbors.len(), 64);
            assert!(!first_neighbors.contains(&(node as u32)));
            first_neighbors.sort_unstable();
            first_neighbors.dedup();
            assert_eq!(first_neighbors.len(), 64);
        }
    }

    #[test]
    fn vamana_greedy_scratch_uses_reusable_dense_bytes_for_million_node_graphs() {
        let mut scratch = GreedySearchScratch::new(1_000_000, 64, 100);

        assert!(!scratch.uses_sparse_states());
        scratch.begin_search();
        scratch.mark_visited(999_999);
        scratch.mark_retained(999_999);
        assert!(scratch.is_visited(999_999));
        assert!(scratch.is_retained_unexpanded(999_999));

        scratch.begin_search();

        assert!(!scratch.is_visited(999_999));
        let BuildVisitStates::Dense {
            states,
            touched_nodes,
        } = &scratch.visit_states
        else {
            panic!("million-node graph should use byte-dense visit states");
        };
        assert_eq!(states.len(), 1_000_000);
        assert!(touched_nodes.is_empty());
    }

    #[test]
    fn vamana_greedy_scratch_uses_sparse_states_for_large_graphs() {
        let mut scratch = GreedySearchScratch::new(100_000_000, 64, 100);

        assert!(scratch.uses_sparse_states());
        let BuildVisitStates::Sparse(states) = &scratch.visit_states else {
            panic!("large graph should use sparse visit states");
        };
        assert!(
            std::any::type_name_of_val(states).contains("SparseTable"),
            "Vamana hot-path sparse states must use the internal open-addressed table"
        );
        scratch.begin_search();
        scratch.mark_visited(99_999_999);
        scratch.mark_expanded(99_999_999);
        assert!(scratch.is_visited(99_999_999));
        assert!(scratch.is_expanded(99_999_999));

        scratch.begin_search();

        assert!(!scratch.is_visited(99_999_999));
        assert!(!scratch.is_expanded(99_999_999));
        assert!(scratch.neighbor_buffer.capacity() >= 64);
    }

    #[test]
    fn vamana_parallel_greedy_search_keeps_worker_frontiers_bounded() {
        let count = 256;
        let vectors = (0..count).map(|node| node as f32).collect::<Vec<_>>();
        let adjacency = ParallelAdjacency::new(count, 32, |node| {
            (1..=32)
                .map(|step| ((node + step) % count) as u32)
                .collect::<Vec<_>>()
        });
        let builder = ParallelVamanaBuilder {
            vectors: &vectors,
            dimension: 1,
            metric: MetricType::L2,
            entry_node: 0,
            adjacency,
            search_distance: BuildSearchDistance::FullPrecision {
                vectors: &vectors,
                dimension: 1,
                metric: MetricType::L2,
            },
        };
        let mut scratch = GreedySearchScratch::new(count, 32, 16);

        builder.greedy_search(127, 16, &mut scratch);

        assert_eq!(scratch.results.len(), 16);
        assert!(scratch.peak_retained_len() <= 16);
        assert!(scratch.peak_frontier_len() <= 32);
    }

    #[test]
    fn vamana_parallel_greedy_search_uses_heap_frontiers() {
        let scratch = GreedySearchScratch::new(1024, 64, 100);

        assert!(
            std::any::type_name_of_val(&scratch.frontier).contains("BinaryHeap"),
            "parallel Vamana frontier must avoid sorted Vec insertion and removal"
        );
    }
}
