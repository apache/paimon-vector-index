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

use crate::blas::sgemm_a_bt;
use crate::distance::{fvec_l2sqr, fvec_norm_l2sqr};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct KMeansConfig {
    pub niter: usize,
    pub nredo: usize,
    pub max_points_per_centroid: usize,
    pub seed: u64,
    /// Balance factor: penalizes large clusters to produce more uniform partitions.
    /// 0.0 = standard k-means. Higher values = more balanced.
    /// Typical value: 0.1 for IVF construction.
    pub balance_factor: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        KMeansConfig {
            niter: 25,
            nredo: 1,
            max_points_per_centroid: 256,
            seed: 1234,
            balance_factor: 0.0,
        }
    }
}

const EPS: f32 = 1.0 / 1024.0;

/// Threshold above which hierarchical k-means is used.
const HIERARCHICAL_THRESHOLD: usize = 256;

pub fn kmeans_train(config: &KMeansConfig, data: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    if k > HIERARCHICAL_THRESHOLD && n > k {
        kmeans_train_hierarchical(config, data, n, d, k)
    } else {
        kmeans_train_with_init(config, data, n, d, k, None)
    }
}

/// Hierarchical k-means for large k (> 256).
/// Starts with initial_k clusters and iteratively splits the largest until target k is reached.
fn kmeans_train_hierarchical(
    config: &KMeansConfig,
    data: &[f32],
    n: usize,
    d: usize,
    target_k: usize,
) -> Vec<f32> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Clone)]
    struct Cluster {
        centroid: Vec<f32>,
        indices: Vec<usize>,
    }

    impl Eq for Cluster {}
    impl PartialEq for Cluster {
        fn eq(&self, other: &Self) -> bool {
            self.indices.len() == other.indices.len()
        }
    }
    impl Ord for Cluster {
        fn cmp(&self, other: &Self) -> Ordering {
            self.indices.len().cmp(&other.indices.len())
        }
    }
    impl PartialOrd for Cluster {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut rng = StdRng::seed_from_u64(config.seed);

    // Subsample for training
    let max_n = target_k * config.max_points_per_centroid;
    let (train_data, train_n) = if n > max_n {
        let sub = subsample(data, n, d, max_n, &mut rng);
        (sub, max_n)
    } else {
        (data.to_vec(), n)
    };

    // Step 1: Train initial_k clusters
    let initial_k = 16.min(target_k);
    let initial_config = KMeansConfig {
        niter: config.niter,
        seed: config.seed,
        ..KMeansConfig::default()
    };
    let initial_centroids =
        kmeans_train_with_init(&initial_config, &train_data, train_n, d, initial_k, None);

    // Assign all points to initial clusters
    let mut assignments = vec![0usize; train_n];
    assign_clusters_fast(
        &train_data,
        train_n,
        d,
        &initial_centroids,
        initial_k,
        &mut assignments,
        0.0,
    );

    // Build initial clusters
    let mut heap: BinaryHeap<Cluster> = BinaryHeap::new();
    for c in 0..initial_k {
        let indices: Vec<usize> = (0..train_n).filter(|&i| assignments[i] == c).collect();
        let centroid = initial_centroids[c * d..(c + 1) * d].to_vec();
        heap.push(Cluster { centroid, indices });
    }

    // Step 2: Iteratively split the largest cluster
    let mut finalized: Vec<Vec<f32>> = Vec::new();
    let split_k = 2; // Split into 2 each time

    while finalized.len() + heap.len() < target_k {
        let largest = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        if largest.indices.len() < split_k * 2 {
            finalized.push(largest.centroid);
            continue;
        }

        // Extract sub-data for this cluster
        let sub_n = largest.indices.len();
        let mut sub_data = vec![0.0f32; sub_n * d];
        for (new_idx, &orig_idx) in largest.indices.iter().enumerate() {
            sub_data[new_idx * d..(new_idx + 1) * d]
                .copy_from_slice(&train_data[orig_idx * d..(orig_idx + 1) * d]);
        }

        // Run k-means to split
        let sub_config = KMeansConfig {
            niter: 10,
            seed: config.seed + finalized.len() as u64,
            ..KMeansConfig::default()
        };
        let sub_centroids = kmeans_train_with_init(&sub_config, &sub_data, sub_n, d, split_k, None);

        // Reassign points in this cluster
        let mut sub_assignments = vec![0usize; sub_n];
        assign_clusters_fast(
            &sub_data,
            sub_n,
            d,
            &sub_centroids,
            split_k,
            &mut sub_assignments,
            0.0,
        );

        for sc in 0..split_k {
            let sub_indices: Vec<usize> = (0..sub_n)
                .filter(|&i| sub_assignments[i] == sc)
                .map(|i| largest.indices[i])
                .collect();
            let centroid = sub_centroids[sc * d..(sc + 1) * d].to_vec();
            if !sub_indices.is_empty() {
                heap.push(Cluster {
                    centroid,
                    indices: sub_indices,
                });
            }
        }
    }

    // Collect all centroids
    let mut result = Vec::with_capacity(target_k * d);
    for c in finalized {
        result.extend_from_slice(&c);
    }
    while let Some(cluster) = heap.pop() {
        result.extend_from_slice(&cluster.centroid);
        if result.len() >= target_k * d {
            break;
        }
    }

    // Pad if needed
    result.resize(target_k * d, 0.0);
    result
}

pub fn kmeans_train_with_init(
    config: &KMeansConfig,
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    initial_centroids: Option<&[f32]>,
) -> Vec<f32> {
    if n == 0 || k == 0 {
        return vec![0.0; k * d];
    }

    let mut rng = StdRng::seed_from_u64(config.seed);

    let max_n = k * config.max_points_per_centroid;
    let (train_data, train_n) = if n > max_n {
        let sub = subsample(data, n, d, max_n, &mut rng);
        (sub, max_n)
    } else {
        (data.to_vec(), n)
    };

    if train_n <= k {
        let mut centroids = vec![0.0f32; k * d];
        for i in 0..k {
            let src = i % train_n;
            centroids[i * d..(i + 1) * d].copy_from_slice(&train_data[src * d..(src + 1) * d]);
        }
        return centroids;
    }

    let mut best_centroids = vec![0.0f32; k * d];
    let mut best_obj = f32::MAX;

    let nredo = if initial_centroids.is_some() {
        1
    } else {
        config.nredo
    };

    for redo in 0..nredo {
        let mut centroids = if redo == 0 {
            if let Some(init) = initial_centroids {
                init.to_vec()
            } else {
                kmeans_plusplus_init(&train_data, train_n, d, k, &mut rng)
            }
        } else {
            kmeans_plusplus_init(&train_data, train_n, d, k, &mut rng)
        };
        let mut assignments = vec![0usize; train_n];
        let mut prev_obj = f32::MAX;

        for _iter in 0..config.niter {
            let obj = assign_clusters_fast(
                &train_data,
                train_n,
                d,
                &centroids,
                k,
                &mut assignments,
                config.balance_factor,
            );
            update_centroids(
                &train_data,
                train_n,
                d,
                &mut centroids,
                k,
                &assignments,
                &mut rng,
            );

            if prev_obj < f32::MAX {
                let rel_change = (prev_obj - obj).abs() / prev_obj.max(1e-10);
                if rel_change < 1e-6 {
                    break;
                }
            }
            prev_obj = obj;
        }

        if prev_obj < best_obj {
            best_obj = prev_obj;
            best_centroids.copy_from_slice(&centroids);
        }
    }

    best_centroids
}

fn kmeans_plusplus_init(data: &[f32], n: usize, d: usize, k: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut centroids = vec![0.0f32; k * d];

    let first = rng.gen_range(0..n);
    centroids[..d].copy_from_slice(&data[first * d..(first + 1) * d]);

    let mut min_dists = vec![f32::MAX; n];

    for c in 1..k {
        let prev = &centroids[(c - 1) * d..c * d];
        let mut total = 0.0f32;
        for i in 0..n {
            let dist = fvec_l2sqr(&data[i * d..(i + 1) * d], prev);
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
            total += min_dists[i];
        }

        let target = rng.gen::<f32>() * total;
        let mut cumulative = 0.0f32;
        let mut selected = n - 1;
        for i in 0..n {
            cumulative += min_dists[i];
            if cumulative >= target {
                selected = i;
                break;
            }
        }

        centroids[c * d..(c + 1) * d].copy_from_slice(&data[selected * d..(selected + 1) * d]);
    }

    centroids
}

/// Fast assignment using sgemm: ||x-c||² = ||x||² + ||c||² - 2·x·cᵀ.
/// Supports balance_factor to penalize large clusters.
fn assign_clusters_fast(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    assignments: &mut [usize],
    balance_factor: f32,
) -> f32 {
    // Cap ip_matrix size to ~16MB. Chunk if n*k would be too large.
    const MAX_MATRIX_ELEMS: usize = 4 * 1024 * 1024; // 16MB / 4 bytes
    if n * k > MAX_MATRIX_ELEMS {
        let chunk_n = MAX_MATRIX_ELEMS / k;
        let mut total_obj = 0.0f32;
        let mut offset = 0;
        while offset < n {
            let cn = (n - offset).min(chunk_n);
            total_obj += assign_clusters_fast(
                &data[offset * d..(offset + cn) * d],
                cn,
                d,
                centroids,
                k,
                &mut assignments[offset..offset + cn],
                balance_factor,
            );
            offset += cn;
        }
        return total_obj;
    }

    let x_norms: Vec<f32> = (0..n)
        .map(|i| fvec_norm_l2sqr(&data[i * d..(i + 1) * d]))
        .collect();
    let c_norms: Vec<f32> = (0..k)
        .map(|c| fvec_norm_l2sqr(&centroids[c * d..(c + 1) * d]))
        .collect();

    let mut ip_matrix = vec![0.0f32; n * k];
    sgemm_a_bt(n, k, d, 1.0, data, centroids, 0.0, &mut ip_matrix);

    // Compute cluster sizes for balance penalty
    let mut cluster_sizes = vec![0u32; k];
    if balance_factor > 0.0 {
        for &a in assignments.iter() {
            if a < k {
                cluster_sizes[a] += 1;
            }
        }
    }

    let mut total_obj = 0.0f32;
    for i in 0..n {
        let mut best = 0;
        let mut best_dist = f32::MAX;
        let row = i * k;
        for c in 0..k {
            let mut dist = x_norms[i] + c_norms[c] - 2.0 * ip_matrix[row + c];
            // Balance penalty: prefer smaller clusters
            if balance_factor > 0.0 && cluster_sizes[c] > 0 {
                dist += balance_factor * (cluster_sizes[c] as f32).ln();
            }
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        assignments[i] = best;
        total_obj += best_dist;
    }

    total_obj
}

fn update_centroids(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &mut [f32],
    k: usize,
    assignments: &[usize],
    rng: &mut StdRng,
) {
    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f32; k * d];

    for i in 0..n {
        let c = assignments[i];
        counts[c] += 1;
        for j in 0..d {
            sums[c * d + j] += data[i * d + j];
        }
    }

    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                centroids[c * d + j] = sums[c * d + j] * inv;
            }
        }
    }

    for c in 0..k {
        if counts[c] > 0 {
            continue;
        }

        let donor = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &cnt)| cnt)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if counts[donor] <= 1 {
            let idx = rng.gen_range(0..n);
            centroids[c * d..(c + 1) * d].copy_from_slice(&data[idx * d..(idx + 1) * d]);
            continue;
        }

        let donor_copy: Vec<f32> = centroids[donor * d..(donor + 1) * d].to_vec();
        centroids[c * d..(c + 1) * d].copy_from_slice(&donor_copy);

        for j in 0..d {
            if j.is_multiple_of(2) {
                centroids[c * d + j] *= 1.0 + EPS;
                centroids[donor * d + j] *= 1.0 - EPS;
            } else {
                centroids[c * d + j] *= 1.0 - EPS;
                centroids[donor * d + j] *= 1.0 + EPS;
            }
        }

        counts[c] = counts[donor] / 2;
        counts[donor] -= counts[c];
    }
}

pub fn find_nearest(point: &[f32], centroids: &[f32], k: usize, d: usize) -> usize {
    let mut best = 0;
    let mut best_dist = f32::MAX;
    for c in 0..k {
        let dist = fvec_l2sqr(point, &centroids[c * d..(c + 1) * d]);
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }
    best
}

pub fn find_topk(
    point: &[f32],
    centroids: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
) -> (Vec<usize>, Vec<f32>) {
    let nprobe = nprobe.min(k);
    let mut dists: Vec<(f32, usize)> = (0..k)
        .map(|c| (fvec_l2sqr(point, &centroids[c * d..(c + 1) * d]), c))
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let indices: Vec<usize> = dists[..nprobe].iter().map(|&(_, i)| i).collect();
    let distances: Vec<f32> = dists[..nprobe].iter().map(|&(d, _)| d).collect();
    (indices, distances)
}

/// Batch find top-nprobe nearest centroids for multiple queries using sgemm.
/// Returns (all_indices, all_distances) each of length nq * nprobe.
pub fn find_topk_batch(
    queries: &[f32],
    nq: usize,
    centroids: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let nprobe = nprobe.min(k);

    if nq == 1 {
        let (indices, distances) = find_topk(&queries[..d], centroids, k, d, nprobe);
        return (vec![indices], vec![distances]);
    }

    // Precompute norms
    let q_norms: Vec<f32> = (0..nq)
        .map(|i| fvec_norm_l2sqr(&queries[i * d..(i + 1) * d]))
        .collect();
    let c_norms: Vec<f32> = (0..k)
        .map(|c| fvec_norm_l2sqr(&centroids[c * d..(c + 1) * d]))
        .collect();

    // Batch inner products: ip[nq × k] = queries[nq × d] · centroids[k × d]^T
    let mut ip_matrix = vec![0.0f32; nq * k];
    sgemm_a_bt(nq, k, d, 1.0, queries, centroids, 0.0, &mut ip_matrix);

    // Extract top-nprobe per query
    let mut all_indices = Vec::with_capacity(nq);
    let mut all_distances = Vec::with_capacity(nq);

    for qi in 0..nq {
        let row = qi * k;
        let mut dists: Vec<(f32, usize)> = (0..k)
            .map(|c| {
                let dist = q_norms[qi] + c_norms[c] - 2.0 * ip_matrix[row + c];
                (dist.max(0.0), c)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        all_indices.push(dists[..nprobe].iter().map(|&(_, i)| i).collect());
        all_distances.push(dists[..nprobe].iter().map(|&(d, _)| d).collect());
    }

    (all_indices, all_distances)
}

// --- Streaming Coreset K-means ---

/// Streaming k-means trainer for very large datasets.
/// Processes data in chunks, compresses each chunk into a weighted coreset,
/// then trains final centroids on the accumulated coreset.
pub struct StreamingKMeans {
    pub d: usize,
    pub k: usize,
    pub chunk_size: usize,
    config: KMeansConfig,
    /// Accumulated coreset: (centroids, weights)
    coreset_centroids: Vec<f32>,
    coreset_weights: Vec<f32>,
}

impl StreamingKMeans {
    /// Create a streaming k-means trainer.
    /// chunk_size: number of vectors per chunk (e.g., k * 256)
    pub fn new(d: usize, k: usize, chunk_size: usize, config: KMeansConfig) -> Self {
        StreamingKMeans {
            d,
            k,
            chunk_size,
            config,
            coreset_centroids: Vec::new(),
            coreset_weights: Vec::new(),
        }
    }

    /// Feed a chunk of training data. Can be called multiple times.
    /// Each chunk is compressed into k weighted centroids (coreset).
    pub fn add_chunk(&mut self, data: &[f32], n: usize) {
        let d = self.d;
        let chunk_k = self.k.min(n);

        if chunk_k == 0 || n == 0 {
            return;
        }

        // Train k-means on this chunk
        let chunk_config = KMeansConfig {
            niter: 15,
            seed: self.config.seed + self.coreset_weights.len() as u64,
            ..KMeansConfig::default()
        };
        let centroids = kmeans_train_with_init(&chunk_config, data, n, d, chunk_k, None);

        // Assign points to centroids to compute weights
        let mut assignments = vec![0usize; n];
        assign_clusters_fast(data, n, d, &centroids, chunk_k, &mut assignments, 0.0);

        let mut weights = vec![0.0f32; chunk_k];
        for &a in &assignments {
            weights[a] += 1.0;
        }

        // Append to coreset
        self.coreset_centroids.extend_from_slice(&centroids);
        self.coreset_weights.extend_from_slice(&weights);
    }

    /// Finalize: train final centroids on the accumulated weighted coreset.
    pub fn finalize(&self) -> Vec<f32> {
        let d = self.d;
        let coreset_n = self.coreset_weights.len();

        if coreset_n == 0 {
            return vec![0.0f32; self.k * d];
        }

        if coreset_n <= self.k {
            let mut result = self.coreset_centroids.clone();
            result.resize(self.k * d, 0.0);
            return result;
        }

        // Weighted k-means on coreset
        weighted_kmeans_train(
            &self.config,
            &self.coreset_centroids,
            &self.coreset_weights,
            coreset_n,
            d,
            self.k,
        )
    }

    /// Total vectors processed so far.
    pub fn total_weight(&self) -> f32 {
        self.coreset_weights.iter().sum()
    }
}

/// Weighted k-means: each point has a weight that affects centroid computation.
fn weighted_kmeans_train(
    config: &KMeansConfig,
    data: &[f32],
    weights: &[f32],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(config.seed);

    if n <= k {
        let mut centroids = vec![0.0f32; k * d];
        for i in 0..k {
            let src = i % n;
            centroids[i * d..(i + 1) * d].copy_from_slice(&data[src * d..(src + 1) * d]);
        }
        return centroids;
    }

    let mut centroids = kmeans_plusplus_init(data, n, d, k, &mut rng);
    let mut assignments = vec![0usize; n];

    for _iter in 0..config.niter {
        // Assign (unweighted distance)
        assign_clusters_fast(data, n, d, &centroids, k, &mut assignments, 0.0);

        // Update with weights
        let mut sums = vec![0.0f32; k * d];
        let mut total_weights = vec![0.0f32; k];

        for i in 0..n {
            let c = assignments[i];
            let w = weights[i];
            total_weights[c] += w;
            for j in 0..d {
                sums[c * d + j] += w * data[i * d + j];
            }
        }

        for c in 0..k {
            if total_weights[c] > 0.0 {
                let inv = 1.0 / total_weights[c];
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] * inv;
                }
            } else {
                // Reinit empty cluster
                let idx = rng.gen_range(0..n);
                centroids[c * d..(c + 1) * d].copy_from_slice(&data[idx * d..(idx + 1) * d]);
            }
        }
    }

    centroids
}

fn subsample(data: &[f32], n: usize, d: usize, target_n: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..target_n {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    let mut result = vec![0.0f32; target_n * d];
    for (out_i, &src_i) in indices[..target_n].iter().enumerate() {
        result[out_i * d..(out_i + 1) * d].copy_from_slice(&data[src_i * d..(src_i + 1) * d]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_clusters() {
        let mut data = Vec::new();
        for _ in 0..50 {
            data.push(0.1);
            data.push(0.1);
        }
        for _ in 0..50 {
            data.push(10.1);
            data.push(10.1);
        }

        let config = KMeansConfig::default();
        let centroids = kmeans_train(&config, &data, 100, 2, 2);

        let c0 = if centroids[0] < 5.0 {
            &centroids[0..2]
        } else {
            &centroids[2..4]
        };
        let c1 = if centroids[0] < 5.0 {
            &centroids[2..4]
        } else {
            &centroids[0..2]
        };

        assert!(c0[0] < 2.0 && c0[1] < 2.0);
        assert!(c1[0] > 8.0 && c1[1] > 8.0);
    }

    #[test]
    fn test_find_topk() {
        let centroids = [0.0, 0.0, 10.0, 0.0, 5.0, 5.0];
        let query = [1.0, 1.0];
        let (indices, _) = find_topk(&query, &centroids, 3, 2, 2);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_hot_start_converges_faster() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 500;
        let d = 4;
        let k = 4;

        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>() * 10.0).collect();

        let config = KMeansConfig {
            niter: 25,
            ..KMeansConfig::default()
        };
        let centroids = kmeans_train(&config, &data, n, d, k);

        // Hot-start with previous centroids should converge in fewer iterations
        let config2 = KMeansConfig {
            niter: 3,
            ..KMeansConfig::default()
        };
        let centroids2 = kmeans_train_with_init(&config2, &data, n, d, k, Some(&centroids));

        // Should be very close to the original since it started from converged state
        let mut total_diff = 0.0f32;
        for i in 0..k * d {
            total_diff += (centroids[i] - centroids2[i]).abs();
        }
        assert!(
            total_diff < 1.0,
            "Hot-start centroids drifted too much: {}",
            total_diff
        );
    }

    #[test]
    fn test_streaming_coreset_kmeans() {
        let n = 5000;
        let d = 4;
        let k = 10;
        let chunk_size = 1000;

        let mut rng = StdRng::seed_from_u64(42);
        // Generate clustered data
        let mut data = Vec::new();
        for cluster in 0..k {
            let cx = cluster as f32 * 20.0;
            let cy = cluster as f32 * 20.0;
            for _ in 0..n / k {
                data.push(cx + rng.gen::<f32>() * 2.0);
                data.push(cy + rng.gen::<f32>() * 2.0);
                data.push(rng.gen::<f32>());
                data.push(rng.gen::<f32>());
            }
        }

        let config = KMeansConfig::default();
        let mut streaming = StreamingKMeans::new(d, k, chunk_size, config);

        // Feed data in chunks
        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk_n = chunk_end - chunk_start;
            streaming.add_chunk(&data[chunk_start * d..chunk_end * d], chunk_n);
        }

        assert!((streaming.total_weight() - n as f32).abs() < 1.0);

        let centroids = streaming.finalize();
        assert_eq!(centroids.len(), k * d);

        // Centroids should be diverse
        let first = &centroids[0..d];
        let mut diverse = false;
        for i in 1..k {
            if fvec_l2sqr(&centroids[i * d..(i + 1) * d], first) > 1.0 {
                diverse = true;
                break;
            }
        }
        assert!(diverse, "Streaming centroids are not diverse");
    }

    #[test]
    fn test_hierarchical_kmeans() {
        let n = 2000;
        let d = 4;
        let k = 300; // > 256, triggers hierarchical

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>() * 100.0).collect();

        let config = KMeansConfig::default();
        let centroids = kmeans_train(&config, &data, n, d, k);

        assert_eq!(centroids.len(), k * d);

        // All centroids should be finite
        for &v in &centroids {
            assert!(v.is_finite(), "Non-finite centroid value: {}", v);
        }

        // Centroids should be diverse (not all the same)
        let first = &centroids[0..d];
        let mut all_same = true;
        for i in 1..k {
            if &centroids[i * d..(i + 1) * d] != first {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "All centroids are identical");
    }
}
