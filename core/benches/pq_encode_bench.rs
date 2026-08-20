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

//! PQ encode implementation shoot-out for the Phase 2 batch-encode plan.
//!
//! Compares three nbits=8 encode implementations on synthetic data shaped
//! like the production workload (d=768, m=192, ksub=256, dsub=4):
//!
//! - A (status quo): `ProductQuantizer::encode_batch`, per-vector-per-sub
//!   `sgemm(1, ksub, dsub)` distance tables.
//! - B (blocked GEMM): row blocks; per (block, sub) gather the sub-slices
//!   into a contiguous buffer and run one `sgemm(block_rows, ksub, dsub)`,
//!   then per-row argmin.
//! - C (direct distance loop): per (block, sub) compute L2 distances with a
//!   plain loop over ksub centroids using the cached centroid norms; dsub=4
//!   keeps the whole sub-codebook (4 KiB) in L1.
//!
//! Codes from B and C must match A byte-for-byte (identical summation order
//! is NOT guaranteed for B, so ulp-tie diffs are counted and reported
//! separately per the plan's consistency standard).
//!
//! Env overrides: `PQ_ENC_N` (default 1,000,000), `PQ_ENC_D` (768),
//! `PQ_ENC_M` (192), `PQ_ENC_TRAIN_N` (100,000), `PQ_ENC_BLOCK` (1024).
//! Thread count follows the global Rayon pool (`RAYON_NUM_THREADS`).

use paimon_vindex_core::blas::sgemm_a_bt;
use paimon_vindex_core::distance::fvec_norm_l2sqr;
use paimon_vindex_core::pq::ProductQuantizer;
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

/// B: blocked GEMM distance tables. Per (block, sub): gather the sub-slices
/// of all rows in the block into a contiguous [block_rows x dsub] buffer,
/// one sgemm for inner products, then argmin with the norms identity.
fn encode_blocked_gemm(
    pq: &ProductQuantizer,
    data: &[f32],
    n: usize,
    codes: &mut [u8],
    block_rows: usize,
) {
    let d = pq.d;
    let m = pq.m;
    let ksub = pq.ksub;
    let cs = pq.code_size();
    let norms = &pq.centroid_norms_cache;
    assert!(!norms.is_empty(), "norms cache required");

    codes
        .par_chunks_mut(block_rows * cs)
        .enumerate()
        .for_each(|(block_idx, block_codes)| {
            let row0 = block_idx * block_rows;
            let rows = block_rows.min(n - row0);
            let block_data = &data[row0 * d..(row0 + rows) * d];

            // Reused per-thread scratch: gathered A-matrix, ip table, row norms.
            let mut gathered = vec![0.0f32; rows * 4]; // dsub <= 4 in target shape
            let mut ip = vec![0.0f32; rows * ksub];
            let mut q_norms = vec![0.0f32; rows];

            for sub in 0..m {
                let range = pq.chunk_range(sub);
                let dsub = range.len();
                let c_base = pq.centroid_chunk_base(sub);
                let centroids = &pq.centroids[c_base..c_base + ksub * dsub];
                let norms_base = sub * ksub;

                // Gather the non-contiguous sub-slices.
                if gathered.len() < rows * dsub {
                    gathered.resize(rows * dsub, 0.0);
                }
                for r in 0..rows {
                    let src = &block_data[r * d + range.start..r * d + range.end];
                    gathered[r * dsub..(r + 1) * dsub].copy_from_slice(src);
                    q_norms[r] = fvec_norm_l2sqr(src);
                }

                sgemm_a_bt(
                    rows,
                    ksub,
                    dsub,
                    1.0,
                    &gathered[..rows * dsub],
                    centroids,
                    0.0,
                    &mut ip[..rows * ksub],
                );

                for r in 0..rows {
                    let row_ip = &ip[r * ksub..(r + 1) * ksub];
                    let qn = q_norms[r];
                    let mut best = 0usize;
                    let mut best_dist = f32::MAX;
                    for j in 0..ksub {
                        let dist = (qn + norms[norms_base + j] - 2.0 * row_ip[j]).max(0.0);
                        if dist < best_dist {
                            best_dist = dist;
                            best = j;
                        }
                    }
                    block_codes[r * cs + sub] = best as u8;
                }
            }
        });
}

/// C: direct per-block distance loop, no GEMM. dsub=4 keeps each
/// sub-codebook (256 x 4 f32 = 4 KiB) in L1; the inner loop is a simple
/// dot-product + norms identity the compiler auto-vectorizes.
fn encode_direct(
    pq: &ProductQuantizer,
    data: &[f32],
    n: usize,
    codes: &mut [u8],
    block_rows: usize,
) {
    let d = pq.d;
    let m = pq.m;
    let ksub = pq.ksub;
    let cs = pq.code_size();
    let norms = &pq.centroid_norms_cache;
    assert!(!norms.is_empty(), "norms cache required");

    codes
        .par_chunks_mut(block_rows * cs)
        .enumerate()
        .for_each(|(block_idx, block_codes)| {
            let row0 = block_idx * block_rows;
            let rows = block_rows.min(n - row0);
            let block_data = &data[row0 * d..(row0 + rows) * d];

            for sub in 0..m {
                let range = pq.chunk_range(sub);
                let dsub = range.len();
                let c_base = pq.centroid_chunk_base(sub);
                let centroids = &pq.centroids[c_base..c_base + ksub * dsub];
                let norms_base = sub * ksub;

                for r in 0..rows {
                    let q = &block_data[r * d + range.start..r * d + range.end];
                    let qn = fvec_norm_l2sqr(q);
                    let mut best = 0usize;
                    let mut best_dist = f32::MAX;
                    for j in 0..ksub {
                        let c = &centroids[j * dsub..(j + 1) * dsub];
                        let mut ipv = 0.0f32;
                        for k in 0..dsub {
                            ipv += q[k] * c[k];
                        }
                        let dist = (qn + norms[norms_base + j] - 2.0 * ipv).max(0.0);
                        if dist < best_dist {
                            best_dist = dist;
                            best = j;
                        }
                    }
                    block_codes[r * cs + sub] = best as u8;
                }
            }
        });
}

/// C-v2: transposed-codebook distance loop. Per sub, centroids are
/// transposed once to [dsub][ksub] column-major so the inner j-loop is
/// stride-1 and auto-vectorizes: dist_j = norms_j - 2*(q0*c0j + .. + q3*c3j)
/// (qn is constant per row and dropped for argmin). No ip table traffic.
fn encode_transposed(
    pq: &ProductQuantizer,
    data: &[f32],
    n: usize,
    codes: &mut [u8],
    block_rows: usize,
) {
    let d = pq.d;
    let m = pq.m;
    let ksub = pq.ksub;
    let cs = pq.code_size();
    let norms = &pq.centroid_norms_cache;
    assert!(!norms.is_empty(), "norms cache required");

    // One-time transpose: per sub, [ksub][dsub] -> [dsub][ksub].
    let max_dsub = (0..m).map(|s| pq.chunk_dim(s)).max().unwrap_or(0);
    let mut tcode = vec![0.0f32; m * max_dsub * ksub];
    for sub in 0..m {
        let dsub = pq.chunk_dim(sub);
        let c_base = pq.centroid_chunk_base(sub);
        let dst = &mut tcode[sub * max_dsub * ksub..];
        for j in 0..ksub {
            for k in 0..dsub {
                dst[k * ksub + j] = pq.centroids[c_base + j * dsub + k];
            }
        }
    }

    codes
        .par_chunks_mut(block_rows * cs)
        .enumerate()
        .for_each(|(block_idx, block_codes)| {
            let row0 = block_idx * block_rows;
            let rows = block_rows.min(n - row0);
            let block_data = &data[row0 * d..(row0 + rows) * d];
            // Per-thread reusable score buffer (norms_j - 2*ip_j), one sub at a time.
            let mut scores = vec![0.0f32; ksub];

            for r in 0..rows {
                let row = &block_data[r * d..(r + 1) * d];
                for sub in 0..m {
                    let range = pq.chunk_range(sub);
                    let dsub = range.len();
                    let q = &row[range];
                    let t = &tcode[sub * max_dsub * ksub..sub * max_dsub * ksub + dsub * ksub];
                    let nb = &norms[sub * ksub..(sub + 1) * ksub];

                    // scores = norms - 2 * sum_k q[k] * t[k][*]  (stride-1 over j)
                    let q0 = -2.0 * q[0];
                    for j in 0..ksub {
                        scores[j] = nb[j] + q0 * t[j];
                    }
                    for k in 1..dsub {
                        let qk = -2.0 * q[k];
                        let tk = &t[k * ksub..(k + 1) * ksub];
                        for j in 0..ksub {
                            scores[j] += qk * tk[j];
                        }
                    }

                    let mut best = 0usize;
                    let mut best_score = f32::MAX;
                    for (j, &s) in scores.iter().enumerate() {
                        if s < best_score {
                            best_score = s;
                            best = j;
                        }
                    }
                    block_codes[r * cs + sub] = best as u8;
                }
            }
        });
}

/// C-v3 (aarch64): transposed codebook + explicit NEON with vectorized
/// argmin. Per (row, sub): scores_j = nb_j - 2*sum_k q_k * t_kj computed
/// 4-wide with FMA; min value + index tracked in SIMD lanes, horizontal
/// reduce with smallest-index tie-break (matches A's first-strictly-smaller
/// semantics up to ulp ties).
#[cfg(target_arch = "aarch64")]
fn encode_neon(pq: &ProductQuantizer, data: &[f32], n: usize, codes: &mut [u8], block_rows: usize) {
    use std::arch::aarch64::*;

    let d = pq.d;
    let m = pq.m;
    let ksub = pq.ksub;
    let cs = pq.code_size();
    let norms = &pq.centroid_norms_cache;
    assert!(!norms.is_empty(), "norms cache required");
    assert_eq!(ksub % 4, 0);

    let max_dsub = (0..m).map(|s| pq.chunk_dim(s)).max().unwrap_or(0);
    let mut tcode = vec![0.0f32; m * max_dsub * ksub];
    for sub in 0..m {
        let dsub = pq.chunk_dim(sub);
        let c_base = pq.centroid_chunk_base(sub);
        let dst = &mut tcode[sub * max_dsub * ksub..];
        for j in 0..ksub {
            for k in 0..dsub {
                dst[k * ksub + j] = pq.centroids[c_base + j * dsub + k];
            }
        }
    }

    codes
        .par_chunks_mut(block_rows * cs)
        .enumerate()
        .for_each(|(block_idx, block_codes)| {
            let row0 = block_idx * block_rows;
            let rows = block_rows.min(n - row0);
            let block_data = &data[row0 * d..(row0 + rows) * d];

            for r in 0..rows {
                let row = &block_data[r * d..(r + 1) * d];
                for sub in 0..m {
                    let range = pq.chunk_range(sub);
                    let dsub = range.len();
                    debug_assert_eq!(dsub, 4, "NEON kernel assumes dsub=4");
                    let q = &row[range];
                    let tbase = sub * max_dsub * ksub;
                    let t0 = &tcode[tbase..tbase + ksub];
                    let t1 = &tcode[tbase + ksub..tbase + 2 * ksub];
                    let t2 = &tcode[tbase + 2 * ksub..tbase + 3 * ksub];
                    let t3 = &tcode[tbase + 3 * ksub..tbase + 4 * ksub];
                    let nb = &norms[sub * ksub..(sub + 1) * ksub];

                    unsafe {
                        let q0 = vdupq_n_f32(-2.0 * q[0]);
                        let q1 = vdupq_n_f32(-2.0 * q[1]);
                        let q2 = vdupq_n_f32(-2.0 * q[2]);
                        let q3 = vdupq_n_f32(-2.0 * q[3]);

                        let mut min_val = vdupq_n_f32(f32::MAX);
                        let mut min_idx = vdupq_n_u32(0);
                        let lane0: [u32; 4] = [0, 1, 2, 3];
                        let mut cur_idx = vld1q_u32(lane0.as_ptr());
                        let step = vdupq_n_u32(4);

                        for j in (0..ksub).step_by(4) {
                            let mut s = vld1q_f32(nb.as_ptr().add(j));
                            s = vfmaq_f32(s, q0, vld1q_f32(t0.as_ptr().add(j)));
                            s = vfmaq_f32(s, q1, vld1q_f32(t1.as_ptr().add(j)));
                            s = vfmaq_f32(s, q2, vld1q_f32(t2.as_ptr().add(j)));
                            s = vfmaq_f32(s, q3, vld1q_f32(t3.as_ptr().add(j)));

                            let mask = vcltq_f32(s, min_val);
                            min_val = vbslq_f32(mask, s, min_val);
                            min_idx = vbslq_u32(mask, cur_idx, min_idx);
                            cur_idx = vaddq_u32(cur_idx, step);
                        }

                        // Horizontal reduce: min value, then smallest index on ties.
                        let mut vals = [0.0f32; 4];
                        let mut idxs = [0u32; 4];
                        vst1q_f32(vals.as_mut_ptr(), min_val);
                        vst1q_u32(idxs.as_mut_ptr(), min_idx);
                        let mut best = idxs[0];
                        let mut best_val = vals[0];
                        for l in 1..4 {
                            if vals[l] < best_val || (vals[l] == best_val && idxs[l] < best) {
                                best_val = vals[l];
                                best = idxs[l];
                            }
                        }
                        block_codes[r * cs + sub] = best as u8;
                    }
                }
            }
        });
}

fn diff_codes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Recompute both candidate distances in f64 for every differing code and
/// report the worst relative gap. A genuine ulp tie has a relative gap around
/// 1e-7 (f32 epsilon); a real bug shows up orders of magnitude larger.
fn verify_ties(
    pq: &ProductQuantizer,
    data: &[f32],
    codes_a: &[u8],
    codes_b: &[u8],
    n: usize,
    label: &str,
) {
    let d = pq.d;
    let cs = pq.code_size();
    let mut checked = 0usize;
    let mut worst_rel = 0.0f64;
    for r in 0..n {
        for sub in 0..pq.m {
            let ca = codes_a[r * cs + sub] as usize;
            let cb = codes_b[r * cs + sub] as usize;
            if ca == cb {
                continue;
            }
            let range = pq.chunk_range(sub);
            let dsub = range.len();
            let q = &data[r * d + range.start..r * d + range.end];
            let c_base = pq.centroid_chunk_base(sub);
            let dist = |code: usize| -> f64 {
                let c = &pq.centroids[c_base + code * dsub..c_base + (code + 1) * dsub];
                q.iter()
                    .zip(c.iter())
                    .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                    .sum()
            };
            let da = dist(ca);
            let db = dist(cb);
            let rel = (da - db).abs() / da.max(db).max(f64::MIN_POSITIVE);
            worst_rel = worst_rel.max(rel);
            checked += 1;
        }
    }
    println!(
        "tie-check {label}: {checked} diffs, worst relative gap {:.2e} (ulp tie ~1e-7)",
        worst_rel
    );
    assert!(
        worst_rel < 1e-5,
        "{label}: non-tie code difference detected (rel gap {worst_rel:.2e})"
    );
}

fn main() {
    let n = env_usize("PQ_ENC_N", 1_000_000);
    let d = env_usize("PQ_ENC_D", 768);
    let m = env_usize("PQ_ENC_M", 192);
    let train_n = env_usize("PQ_ENC_TRAIN_N", 100_000);
    let block = env_usize("PQ_ENC_BLOCK", 1024);

    println!("=== PQ encode shoot-out ===");
    println!(
        "n={} d={} m={} ksub=256 dsub={} block={} threads={}",
        n,
        d,
        m,
        d / m,
        block,
        rayon::current_num_threads()
    );

    let train = generate_vectors(train_n, d, 20260820);
    let data = generate_vectors(n, d, 20260821);

    let mut pq = ProductQuantizer::new(d, m);
    let t = Instant::now();
    pq.train(&train, train_n);
    println!(
        "train: {:.1}s (m={} ksub={})",
        t.elapsed().as_secs_f64(),
        pq.m,
        pq.ksub
    );
    assert_eq!(pq.nbits, 8);
    assert!(!pq.centroid_norms_cache.is_empty());

    let cs = pq.code_size();
    let mut codes_a = vec![0u8; n * cs];
    let mut codes_b = vec![0u8; n * cs];
    let mut codes_c = vec![0u8; n * cs];

    // A: status quo.
    let t = Instant::now();
    pq.encode_batch(&data, n, &mut codes_a);
    let a_secs = t.elapsed().as_secs_f64();
    println!(
        "A status-quo   : {:>8.2}s  {:>12.0} rows/s",
        a_secs,
        n as f64 / a_secs
    );

    // B: blocked GEMM.
    let t = Instant::now();
    encode_blocked_gemm(&pq, &data, n, &mut codes_b, block);
    let b_secs = t.elapsed().as_secs_f64();
    println!(
        "B blocked-gemm : {:>8.2}s  {:>12.0} rows/s  speedup {:>5.1}x  diff {}",
        b_secs,
        n as f64 / b_secs,
        a_secs / b_secs,
        diff_codes(&codes_a, &codes_b)
    );

    // C: direct loop.
    let t = Instant::now();
    encode_direct(&pq, &data, n, &mut codes_c, block);
    let c_secs = t.elapsed().as_secs_f64();
    println!(
        "C direct-loop  : {:>8.2}s  {:>12.0} rows/s  speedup {:>5.1}x  diff {}",
        c_secs,
        n as f64 / c_secs,
        a_secs / c_secs,
        diff_codes(&codes_a, &codes_c)
    );

    // C-v2: transposed codebook.
    let mut codes_t = vec![0u8; n * cs];
    let t = Instant::now();
    encode_transposed(&pq, &data, n, &mut codes_t, block);
    let t_secs = t.elapsed().as_secs_f64();
    println!(
        "T transposed   : {:>8.2}s  {:>12.0} rows/s  speedup {:>5.1}x  diff {}",
        t_secs,
        n as f64 / t_secs,
        a_secs / t_secs,
        diff_codes(&codes_a, &codes_t)
    );

    // C-v3: explicit NEON (aarch64 only).
    #[cfg(target_arch = "aarch64")]
    {
        let mut codes_n = vec![0u8; n * cs];
        let t = Instant::now();
        encode_neon(&pq, &data, n, &mut codes_n, block);
        let n_secs = t.elapsed().as_secs_f64();
        println!(
            "N neon         : {:>8.2}s  {:>12.0} rows/s  speedup {:>5.1}x  diff {}",
            n_secs,
            n as f64 / n_secs,
            a_secs / n_secs,
            diff_codes(&codes_a, &codes_n)
        );
        let diff_rate_n = diff_codes(&codes_a, &codes_n) as f64 / (n * cs) as f64;
        println!("diff-rate: N={:.2e}", diff_rate_n);
        verify_ties(&pq, &data, &codes_a, &codes_n, n, "N");
    }

    let diff_rate_b = diff_codes(&codes_a, &codes_b) as f64 / (n * cs) as f64;
    let diff_rate_c = diff_codes(&codes_a, &codes_c) as f64 / (n * cs) as f64;
    let diff_rate_t = diff_codes(&codes_a, &codes_t) as f64 / (n * cs) as f64;
    println!(
        "diff-rate: B={:.2e} C={:.2e} T={:.2e} (plan gate: <=1e-6, all ulp ties)",
        diff_rate_b, diff_rate_c, diff_rate_t
    );
    verify_ties(&pq, &data, &codes_a, &codes_b, n, "B");
    verify_ties(&pq, &data, &codes_a, &codes_c, n, "C");
    verify_ties(&pq, &data, &codes_a, &codes_t, n, "T");
}
