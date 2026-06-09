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
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io;

#[derive(Debug, Clone, Copy)]
pub struct HnswBuildParams {
    pub m: usize,
    pub ef_construction: usize,
    pub max_level: usize,
}

impl Default for HnswBuildParams {
    fn default() -> Self {
        Self {
            m: 20,
            ef_construction: 150,
            max_level: 7,
        }
    }
}

impl HnswBuildParams {
    pub fn sanitized(self) -> Self {
        Self {
            m: self.m.max(1),
            ef_construction: self.ef_construction.max(1),
            max_level: self.max_level.max(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HnswGraph {
    d: usize,
    metric: MetricType,
    vectors: Vec<f32>,
    levels: Vec<usize>,
    neighbors: Vec<Vec<Vec<usize>>>,
    entry_point: usize,
    max_observed_level: usize,
    params: HnswBuildParams,
}

impl HnswGraph {
    pub fn build(
        vectors: &[f32],
        n: usize,
        d: usize,
        metric: MetricType,
        params: HnswBuildParams,
    ) -> io::Result<Self> {
        let expected_len = n.checked_mul(d).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "n * dimension overflows usize")
        })?;
        if vectors.len() < expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "vector data length {} is shorter than n*d {}",
                    vectors.len(),
                    expected_len
                ),
            ));
        }

        let params = params.sanitized();
        let mut graph = HnswGraph {
            d,
            metric,
            vectors: vectors[..n * d].to_vec(),
            levels: Vec::with_capacity(n),
            neighbors: Vec::with_capacity(n),
            entry_point: 0,
            max_observed_level: 0,
            params,
        };

        for node in 0..n {
            graph.insert(node);
        }
        Ok(graph)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        vectors: Vec<f32>,
        n: usize,
        d: usize,
        metric: MetricType,
        levels: Vec<usize>,
        neighbors: Vec<Vec<Vec<usize>>>,
        entry_point: usize,
        max_observed_level: usize,
        params: HnswBuildParams,
    ) -> io::Result<Self> {
        let expected_len = n.checked_mul(d).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "n * dimension overflows usize")
        })?;
        if vectors.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "graph vector length {} does not match n*d {}",
                    vectors.len(),
                    expected_len
                ),
            ));
        }
        if levels.len() != n || neighbors.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "graph level metadata does not match vector count",
            ));
        }
        if n == 0 {
            return Ok(Self {
                d,
                metric,
                vectors,
                levels,
                neighbors,
                entry_point: 0,
                max_observed_level: 0,
                params: params.sanitized(),
            });
        }
        if entry_point >= n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("graph entry point {} out of range {}", entry_point, n),
            ));
        }
        let observed = levels.iter().copied().max().unwrap_or(0);
        if max_observed_level != observed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "graph max level {} does not match observed {}",
                    max_observed_level, observed
                ),
            ));
        }
        if levels[entry_point] < max_observed_level {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "graph entry point does not reach max observed level",
            ));
        }
        for node in 0..n {
            if neighbors[node].len() != levels[node] + 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("graph node {} has invalid level adjacency", node),
                ));
            }
            for (level, level_neighbors) in neighbors[node].iter().enumerate() {
                for &neighbor in level_neighbors {
                    if neighbor >= n || levels[neighbor] < level {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "graph edge {} -> {} at level {} is invalid",
                                node, neighbor, level
                            ),
                        ));
                    }
                }
            }
        }
        Ok(Self {
            d,
            metric,
            vectors,
            levels,
            neighbors,
            entry_point,
            max_observed_level,
            params: params.sanitized(),
        })
    }

    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.levels.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut ep = self.entry_point;
        let mut ep_dist = self.distance_to_query(query, ep);
        for level in (1..=self.max_observed_level).rev() {
            let (next, dist) = self.greedy_search_query(query, ep, ep_dist, level);
            ep = next;
            ep_dist = dist;
        }

        let mut visited = vec![0usize; self.levels.len()];
        let candidates = self.search_layer_query(query, ep, ef.max(k), 0, &mut visited, 1);
        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.id, n.dist))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn max_degree(&self) -> usize {
        self.neighbors
            .iter()
            .flat_map(|levels| levels.iter().map(Vec::len))
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    pub(crate) fn levels(&self) -> &[usize] {
        &self.levels
    }

    pub(crate) fn neighbors(&self) -> &[Vec<Vec<usize>>] {
        &self.neighbors
    }

    pub(crate) fn entry_point(&self) -> usize {
        self.entry_point
    }

    pub(crate) fn max_observed_level(&self) -> usize {
        self.max_observed_level
    }

    fn insert(&mut self, node: usize) {
        let level = random_level(node, self.params.m, self.params.max_level);
        self.levels.push(level);
        self.neighbors.push(vec![Vec::new(); level + 1]);

        if node == 0 {
            self.entry_point = 0;
            self.max_observed_level = level;
            return;
        }

        let mut ep = self.entry_point;
        let mut ep_dist = self.distance_between(node, ep);

        for layer in ((level + 1)..=self.max_observed_level).rev() {
            let (next, dist) = self.greedy_search_node(node, ep, ep_dist, layer);
            ep = next;
            ep_dist = dist;
        }

        let mut visited = vec![0usize; self.levels.len()];
        let mut visit_mark = 1usize;
        for layer in (0..=level.min(self.max_observed_level)).rev() {
            let candidates = self.search_layer_node(
                node,
                ep,
                self.params.ef_construction,
                layer,
                &mut visited,
                visit_mark,
            );
            visit_mark = advance_visit_mark(&mut visited, visit_mark);
            let selected = self.select_neighbors(candidates, self.max_neighbors(layer));
            for neighbor in selected {
                self.connect(node, neighbor.id, layer);
            }
            if let Some(best) = self.neighbors[node][layer]
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    self.distance_between(node, a)
                        .total_cmp(&self.distance_between(node, b))
                })
            {
                ep = best;
            }
        }

        if level > self.max_observed_level {
            self.entry_point = node;
            self.max_observed_level = level;
        }
    }

    fn connect(&mut self, a: usize, b: usize, level: usize) {
        if !self.neighbors[a][level].contains(&b) {
            self.neighbors[a][level].push(b);
        }
        if level < self.neighbors[b].len() && !self.neighbors[b][level].contains(&a) {
            self.neighbors[b][level].push(a);
            let pruned = self.pruned_neighbors(b, level, self.max_neighbors(level));
            self.neighbors[b][level] = pruned;
        }
        let pruned = self.pruned_neighbors(a, level, self.max_neighbors(level));
        self.neighbors[a][level] = pruned;
    }

    fn pruned_neighbors(&self, node: usize, level: usize, max_neighbors: usize) -> Vec<usize> {
        let mut ranked: Vec<ScoredNode> = self.neighbors[node][level]
            .iter()
            .map(|&id| ScoredNode {
                id,
                dist: self.distance_between(node, id),
            })
            .collect();
        ranked.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        ranked
            .into_iter()
            .take(max_neighbors)
            .map(|node| node.id)
            .collect()
    }

    fn select_neighbors(
        &self,
        mut candidates: Vec<ScoredNode>,
        max_neighbors: usize,
    ) -> Vec<ScoredNode> {
        candidates.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        let mut selected: Vec<ScoredNode> = Vec::with_capacity(max_neighbors);
        let mut backfill: Vec<ScoredNode> = Vec::new();
        for candidate in candidates {
            if selected.len() >= max_neighbors {
                break;
            }
            let closer_to_selected = selected
                .iter()
                .any(|neighbor| self.distance_between(candidate.id, neighbor.id) < candidate.dist);
            if !closer_to_selected {
                selected.push(candidate);
            } else {
                backfill.push(candidate);
            }
        }
        for candidate in backfill {
            if selected.len() >= max_neighbors {
                break;
            }
            if !selected.iter().any(|neighbor| neighbor.id == candidate.id) {
                selected.push(candidate);
            }
        }
        selected
    }

    fn greedy_search_query(
        &self,
        query: &[f32],
        mut current: usize,
        mut current_dist: f32,
        level: usize,
    ) -> (usize, f32) {
        loop {
            let mut best = current;
            let mut best_dist = current_dist;
            for &neighbor in self.neighbors_at(current, level) {
                let dist = self.distance_to_query(query, neighbor);
                if dist < best_dist {
                    best = neighbor;
                    best_dist = dist;
                }
            }
            if best == current {
                return (current, current_dist);
            }
            current = best;
            current_dist = best_dist;
        }
    }

    fn greedy_search_node(
        &self,
        node: usize,
        mut current: usize,
        mut current_dist: f32,
        level: usize,
    ) -> (usize, f32) {
        loop {
            let mut best = current;
            let mut best_dist = current_dist;
            for &neighbor in self.neighbors_at(current, level) {
                let dist = self.distance_between(node, neighbor);
                if dist < best_dist {
                    best = neighbor;
                    best_dist = dist;
                }
            }
            if best == current {
                return (current, current_dist);
            }
            current = best;
            current_dist = best_dist;
        }
    }

    fn search_layer_query(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
        visited: &mut [usize],
        visit_mark: usize,
    ) -> Vec<ScoredNode> {
        self.search_layer(entry, ef, level, visited, visit_mark, |id| {
            self.distance_to_query(query, id)
        })
    }

    fn search_layer_node(
        &self,
        node: usize,
        entry: usize,
        ef: usize,
        level: usize,
        visited: &mut [usize],
        visit_mark: usize,
    ) -> Vec<ScoredNode> {
        self.search_layer(entry, ef, level, visited, visit_mark, |id| {
            self.distance_between(node, id)
        })
    }

    fn search_layer(
        &self,
        entry: usize,
        ef: usize,
        level: usize,
        visited: &mut [usize],
        visit_mark: usize,
        mut distance: impl FnMut(usize) -> f32,
    ) -> Vec<ScoredNode> {
        let entry_dist = distance(entry);
        visited[entry] = visit_mark;

        let mut candidates = BinaryHeap::new();
        candidates.push(Reverse(HeapNode {
            id: entry,
            dist: entry_dist,
        }));

        let mut results = BinaryHeap::new();
        results.push(HeapNode {
            id: entry,
            dist: entry_dist,
        });

        while let Some(Reverse(current)) = candidates.pop() {
            let worst = results
                .peek()
                .map(|node| node.dist)
                .unwrap_or(f32::INFINITY);
            if current.dist > worst && results.len() >= ef {
                break;
            }

            for &neighbor in self.neighbors_at(current.id, level) {
                if visited[neighbor] == visit_mark {
                    continue;
                }
                visited[neighbor] = visit_mark;
                let dist = distance(neighbor);
                let worst = results
                    .peek()
                    .map(|node| node.dist)
                    .unwrap_or(f32::INFINITY);
                if results.len() < ef || dist < worst {
                    candidates.push(Reverse(HeapNode { id: neighbor, dist }));
                    results.push(HeapNode { id: neighbor, dist });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result: Vec<ScoredNode> = results
            .into_iter()
            .map(|node| ScoredNode {
                id: node.id,
                dist: node.dist,
            })
            .collect();
        result.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        result
    }

    fn max_neighbors(&self, level: usize) -> usize {
        if level == 0 {
            self.params.m * 2
        } else {
            self.params.m
        }
    }

    fn neighbors_at(&self, node: usize, level: usize) -> &[usize] {
        self.neighbors
            .get(node)
            .and_then(|levels| levels.get(level))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn distance_between(&self, a: usize, b: usize) -> f32 {
        let va = &self.vectors[a * self.d..(a + 1) * self.d];
        let vb = &self.vectors[b * self.d..(b + 1) * self.d];
        fvec_distance(va, vb, self.metric)
    }

    fn distance_to_query(&self, query: &[f32], id: usize) -> f32 {
        let vector = &self.vectors[id * self.d..(id + 1) * self.d];
        fvec_distance(query, vector, self.metric)
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoredNode {
    id: usize,
    dist: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapNode {
    id: usize,
    dist: f32,
}

impl Eq for HeapNode {}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

fn random_level(node: usize, m: usize, max_level: usize) -> usize {
    if node == 0 || max_level <= 1 {
        // Keep the first insertion deterministic. Later higher-level nodes replace
        // the entry point as they appear, while tiny lists naturally stay flat.
        return 0;
    }
    let mut x = splitmix64(node as u64 + 0x9E37_79B9_7F4A_7C15);
    let mut level = 0;
    let threshold = (u64::MAX / m.max(2) as u64).max(1);
    while level + 1 < max_level && x < threshold {
        level += 1;
        x = splitmix64(x);
    }
    level
}

fn advance_visit_mark(visited: &mut [usize], visit_mark: usize) -> usize {
    visit_mark.checked_add(1).unwrap_or_else(|| {
        visited.fill(0);
        1
    })
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::MetricType;

    #[test]
    fn test_hnsw_recalls_query_vector_on_single_partition() {
        let d = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| [i as f32 * 0.01, 1.0, 2.0, 3.0])
            .collect();
        let params = HnswBuildParams {
            m: 8,
            ef_construction: 32,
            max_level: 6,
        };

        let graph = HnswGraph::build(&data, n, d, MetricType::L2, params).unwrap();
        let query_id = 17;
        let results = graph.search(&data[query_id * d..(query_id + 1) * d], 5, 32);

        assert_eq!(results[0].0, query_id);
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn test_hnsw_empty_graph_returns_no_results() {
        let graph =
            HnswGraph::build(&[], 0, 4, MetricType::L2, HnswBuildParams::default()).unwrap();

        assert!(graph.search(&[0.0, 0.0, 0.0, 0.0], 10, 20).is_empty());
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert_eq!(graph.max_degree(), 0);
    }

    #[test]
    fn test_hnsw_build_rejects_short_vector_input() {
        let err = HnswGraph::build(
            &[0.0, 1.0, 2.0],
            2,
            2,
            MetricType::L2,
            HnswBuildParams::default(),
        )
        .unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("shorter than n*d"));
    }

    #[test]
    fn test_hnsw_respects_neighbor_degree_bound() {
        let d = 8;
        let n = 512;
        let data = generate_clustered_data(n, d, 16);
        let params = HnswBuildParams {
            m: 12,
            ef_construction: 100,
            max_level: 6,
        };

        let graph = HnswGraph::build(&data, n, d, MetricType::L2, params).unwrap();

        assert_eq!(graph.len(), n);
        assert!(graph.max_degree() <= params.m * 2);
    }

    #[test]
    fn test_hnsw_large_partition_recall_tracks_exact_search() {
        let d = 16;
        let n = 4096;
        let nq = 32;
        let k = 10;
        let data = generate_clustered_data(n, d, 32);
        let params = HnswBuildParams {
            m: 16,
            ef_construction: 200,
            max_level: 7,
        };

        let graph = HnswGraph::build(&data, n, d, MetricType::L2, params).unwrap();
        let mut hits = 0usize;
        for qi in 0..nq {
            let query = &data[qi * d..(qi + 1) * d];
            let expected = exact_topk(&data, n, d, query, k);
            let actual = graph.search(query, k, 200);
            hits += actual
                .iter()
                .filter(|(id, _)| expected.contains(id))
                .count();
        }

        let recall = hits as f32 / (nq * k) as f32;
        assert!(recall >= 0.95, "recall={}", recall);
    }

    #[test]
    fn test_hnsw_neighbor_selection_backfills_after_diversification() {
        let d = 1;
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let graph = HnswGraph::build(
            &data,
            4,
            d,
            MetricType::L2,
            HnswBuildParams {
                m: 2,
                ef_construction: 4,
                max_level: 1,
            },
        )
        .unwrap();
        let candidates = vec![
            ScoredNode { id: 1, dist: 1.0 },
            ScoredNode { id: 2, dist: 2.0 },
            ScoredNode { id: 3, dist: 3.0 },
        ];

        let selected = graph.select_neighbors(candidates, 3);

        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_hnsw_greedy_search_chooses_best_improving_neighbor() {
        let graph = HnswGraph::from_parts(
            vec![0.0, 5.0, 2.0],
            3,
            1,
            MetricType::L2,
            vec![0, 0, 0],
            vec![vec![vec![1, 2]], vec![vec![]], vec![vec![]]],
            0,
            0,
            HnswBuildParams::default(),
        )
        .unwrap();

        let (next, dist) = graph.greedy_search_query(&[2.0], 0, 4.0, 0);

        assert_eq!(next, 2);
        assert_eq!(dist, 0.0);
    }

    fn exact_topk(data: &[f32], n: usize, d: usize, query: &[f32], k: usize) -> Vec<usize> {
        let mut distances: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let vector = &data[i * d..(i + 1) * d];
                (fvec_distance(query, vector, MetricType::L2), i)
            })
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances[..k].iter().map(|&(_, id)| id).collect()
    }

    fn generate_clustered_data(n: usize, d: usize, num_clusters: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            let cluster = i % num_clusters;
            for j in 0..d {
                data[i * d + j] = cluster as f32 * 20.0 + j as f32 * 0.01 + i as f32 * 0.0001;
            }
        }
        data
    }
}
