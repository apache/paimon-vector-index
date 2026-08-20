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

//! Coarse-assignment block-size shoot-out for the Phase 3 plan.
//!
//! The production `assign_clusters_fast` splits rows into per-thread blocks
//! sized by `MAX_MATRIX_ELEMS / threads / k` — 64 rows/thread at k=4096 with
//! 16 threads. Each `sgemm(block_rows, k, d)` call re-streams and re-packs
//! the 12.6 MiB centroid matrix, so tiny blocks amortize poorly.
//!
//! This bench replicates the assign path (sgemm + argmin, same math as
//! `assign_block`) with configurable block sizes and compares throughput.
//! Assignments must match the production path exactly (identical summation
//! order per row: same sgemm shapes differ only in row count, and argmin is
//! per-row, so results are bit-identical across block sizes).
//!
//! Env overrides: `CA_N` (default 1,000,000), `CA_D` (768), `CA_K` (4096),
//! `CA_TRAIN_N` (100,000). Thread count follows the Rayon pool.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::redundant_closure)]
#![allow(unused_variables)]

use paimon_vindex_core::blas::sgemm_a_bt;
use paimon_vindex_core::distance::fvec_norm_l2sqr;
use paimon_vindex_core::kmeans::{self, KMeansConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn generate_vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * d];
    for v in data.iter_mut() {
        *v = rng.gen_range(-1.0f32..1.0f32);
    }
    data
}

/// Clustered synthetic data: a Gaussian mixture so nearest-centroid identity
/// is stable, mimicking real embedding structure (uniform random vectors in
/// 768-d are nearly equidistant to every centroid, which makes exact-argmin
/// agreement meaningless as a metric).
fn generate_clustered_vectors(n: usize, d: usize, num_clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cluster_means = vec![0.0f32; num_clusters * d];
    for v in cluster_means.iter_mut() {
        *v = rng.gen_range(-1.0f32..1.0f32);
    }
    let mut data = vec![0.0f32; n * d];
    for row in 0..n {
        let c = rng.gen_range(0..num_clusters);
        let mean = &cluster_means[c * d..(c + 1) * d];
        for k in 0..d {
            // Box-Muller-ish cheap noise: sum of two uniforms, scaled small
            // relative to the mean spread so clusters stay separable.
            let noise = (rng.gen_range(-1.0f32..1.0) + rng.gen_range(-1.0f32..1.0)) * 0.08;
            data[row * d + k] = mean[k] + noise;
        }
    }
    data
}

/// Replicates `assign_block`: one sgemm + per-row argmin over k.
fn assign_block_rows(
    data: &[f32],
    rows: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    c_norms: &[f32],
    assignments: &mut [usize],
    ip_matrix: &mut Vec<f32>,
) {
    if ip_matrix.len() < rows * k {
        ip_matrix.resize(rows * k, 0.0);
    }
    sgemm_a_bt(
        rows,
        k,
        d,
        1.0,
        data,
        centroids,
        0.0,
        &mut ip_matrix[..rows * k],
    );
    for i in 0..rows {
        let x_norm = fvec_norm_l2sqr(&data[i * d..(i + 1) * d]);
        let row = i * k;
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for c in 0..k {
            let dist = x_norm + c_norms[c] - 2.0 * ip_matrix[row + c];
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        assignments[i] = best;
    }
}

/// Parallel assignment with a fixed block size (rows per sgemm call).
fn assign_with_block(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    c_norms: &[f32],
    block_rows: usize,
    assignments: &mut [usize],
) {
    assignments
        .par_chunks_mut(block_rows)
        .enumerate()
        .for_each_init(
            || Vec::new(),
            |ip, (block_idx, block_assign)| {
                let row0 = block_idx * block_rows;
                let rows = block_assign.len();
                assign_block_rows(
                    &data[row0 * d..(row0 + rows) * d],
                    rows,
                    d,
                    centroids,
                    k,
                    c_norms,
                    block_assign,
                    ip,
                );
            },
        );
}

fn main() {
    let n = env_usize("CA_N", 1_000_000);
    let d = env_usize("CA_D", 768);
    let k = env_usize("CA_K", 4096);
    let train_n = env_usize("CA_TRAIN_N", 100_000);
    let threads = rayon::current_num_threads();

    println!("=== coarse assign block-size shoot-out ===");
    println!("n={n} d={d} k={k} threads={threads}");
    let flops = 2.0 * n as f64 * k as f64 * d as f64;

    let train = generate_vectors(train_n, d, 20260830);
    // CA_CLUSTERED=1 switches to Gaussian-mixture data (default: 1 — uniform
    // random is only useful for raw-throughput comparison, its argmin identity
    // is unstable and makes agreement metrics meaningless).
    let clustered = env_usize("CA_CLUSTERED", 1) == 1;
    let data = if clustered {
        generate_clustered_vectors(n, d, 1024, 20260831)
    } else {
        generate_vectors(n, d, 20260831)
    };
    println!(
        "data={}",
        if clustered {
            "clustered(gmm-1024)"
        } else {
            "uniform"
        }
    );

    // centroid quality irrelevant for assign throughput
    let cfg = KMeansConfig {
        niter: 3,
        ..Default::default()
    };
    let t = Instant::now();
    let centroids = kmeans::kmeans_train(&cfg, &train, train_n, d, k);
    println!("train(niter=3): {:.1}s", t.elapsed().as_secs_f64());

    let c_norms: Vec<f32> = (0..k)
        .map(|c| fvec_norm_l2sqr(&centroids[c * d..(c + 1) * d]))
        .collect();

    // Current production block size: MAX_MATRIX_ELEMS / threads / k.
    let prod_block = ((4 * 1024 * 1024) / threads / k).max(1);
    let mut reference = vec![0usize; n];

    let mut results: Vec<(usize, f64)> = Vec::new();
    for &block in &[prod_block, 128, 256, 512, 1024, 2048] {
        let mut assignments = vec![0usize; n];
        let t = Instant::now();
        assign_with_block(
            &data,
            n,
            d,
            &centroids,
            k,
            &c_norms,
            block,
            &mut assignments,
        );
        let secs = t.elapsed().as_secs_f64();
        let gflops = flops / secs / 1e9;
        let label = if block == prod_block {
            " (production)"
        } else {
            ""
        };
        if block == prod_block {
            reference.copy_from_slice(&assignments);
        }
        let diff = assignments
            .iter()
            .zip(reference.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!("block={block:>5}{label:<13}: {secs:>7.2}s  {gflops:>7.0} GFLOP/s  diff={diff}");
        results.push((block, secs));
    }

    // V: approximate assign via Vamana graph over centroids (Phase 4).
    // Lance-shaped params: small degree, small search list, top-1.
    {
        use paimon_vindex_core::diskann::{
            DiskAnnBuildDistance, DiskAnnBuildParams, DiskAnnRawVectorEncoding,
            DiskAnnStorageLayout,
        };
        use paimon_vindex_core::vamana::VamanaGraph;

        let params = DiskAnnBuildParams {
            max_degree: 12,
            build_search_list_size: 32,
            alpha: 1.2,
            seed: 42,
            memory_budget_bytes: 1024 * 1024 * 1024,
            storage_layout: DiskAnnStorageLayout::Compact,
            raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
            build_distance: DiskAnnBuildDistance::FullPrecision,
        };
        let t_build = Instant::now();
        let graph = VamanaGraph::build(&centroids, k, d, params).expect("vamana build");
        let build_secs = t_build.elapsed().as_secs_f64();

        for &ef in &[15usize, 32] {
            let mut assignments = vec![0usize; n];
            let t = Instant::now();
            assignments
                .par_chunks_mut(4096)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let row0 = chunk_idx * 4096;
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let q = &data[(row0 + i) * d..(row0 + i + 1) * d];
                        let res = graph.greedy_search(&centroids, d, q, ef);
                        *slot = res.first().map(|s| s.id as usize).unwrap_or(0);
                    }
                });
            let secs = t.elapsed().as_secs_f64();
            let rows_per_s = n as f64 / secs;
            // Agreement + distance-ratio of disagreements, sampled on the
            // first 100K rows. Ratio = d(chosen)/d(true-nearest); ~1.0 means
            // the graph picked a bucket that is *tied* for nearest — harmless
            // for IVF recall. Large ratios are the real failure mode.
            let sample = n.min(100_000);
            let mut agree = 0usize;
            let mut ratios: Vec<f32> = Vec::new();
            for i in 0..sample {
                if assignments[i] == reference[i] {
                    agree += 1;
                    continue;
                }
                let q = &data[i * d..(i + 1) * d];
                let dist = |c: usize| -> f32 {
                    let cent = &centroids[c * d..(c + 1) * d];
                    q.iter()
                        .zip(cent.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum()
                };
                let chosen = dist(assignments[i]).max(1e-30).sqrt();
                let true_best = dist(reference[i]).max(1e-30).sqrt();
                ratios.push(chosen / true_best);
            }
            ratios.sort_by(|a, b| a.total_cmp(b));
            let pct = |p: f64| -> f32 {
                if ratios.is_empty() {
                    1.0
                } else {
                    ratios[((ratios.len() - 1) as f64 * p) as usize]
                }
            };
            println!(
                "V vamana ef={ef:>2} (graph build {build_secs:.2}s): {secs:>7.2}s  {rows_per_s:>9.0} rows/s  speedup {:.2}x  agree {:.2}%  dist-ratio p50={:.4} p99={:.4} max={:.4}",
                results[0].1 / secs,
                100.0 * agree as f64 / sample as f64,
                pct(0.5),
                pct(0.99),
                ratios.last().copied().unwrap_or(1.0)
            );
        }
    }

    let (best_block, best_secs) = results
        .iter()
        .copied()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let prod_secs = results[0].1;
    println!(
        "best: block={} speedup {:.2}x over production (gate: >=1.5x to proceed)",
        best_block,
        prod_secs / best_secs
    );
}
