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
    let data = generate_vectors(n, d, 20260831);

    let mut cfg = KMeansConfig::default();
    cfg.niter = 3; // centroid quality irrelevant for assign throughput
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
