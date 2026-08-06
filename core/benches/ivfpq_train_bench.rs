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

//! Single-run IVF-PQ training benchmark.
//!
//! `train_total_s` is the authoritative `IVFPQIndex::train` wall time.
//! The mirrored phase timings (`prep_s`, `coarse_s`, `pq_s`) replay the
//! InnerProduct/no-OPQ training path with public APIs for attribution only;
//! never sum them as a total.
//!
//! The `target-768` scenario can be overridden with `TRAIN_N`, `TRAIN_D`,
//! `TRAIN_NLIST`, and `TRAIN_PQ_M`. Thread count follows the global Rayon
//! pool (`RAYON_NUM_THREADS`).

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use paimon_vindex_core::kmeans::{self, KMeansConfig};
use paimon_vindex_core::pq::ProductQuantizer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

struct Scenario {
    name: &'static str,
    n: usize,
    d: usize,
    nlist: usize,
    pq_m: usize,
}

fn main() {
    println!("=== Paimon IVF-PQ Training Benchmark ===");
    println!(
        "metric=InnerProduct, use_opq=false, threads={}",
        rayon::current_num_threads()
    );
    println!();
    println!(
        "{:<11} {:>8} {:>5} {:>6} {:>5} {:>8} {:>9} {:>8} {:>14}",
        "scenario", "n", "d", "nlist", "pq_m", "prep_s", "coarse_s", "pq_s", "train_total_s"
    );
    println!(
        "{:<11} {:>8} {:>5} {:>6} {:>5} {:>8} {:>9} {:>8} {:>14}",
        "---------", "------", "---", "-----", "----", "------", "-------", "----", "-------------"
    );

    run_scenario(&Scenario {
        name: "small",
        n: 50_000,
        d: 128,
        nlist: 1024,
        pq_m: 16,
    });

    run_scenario(&Scenario {
        name: "target-768",
        n: env_usize("TRAIN_N", 244_606),
        d: env_usize("TRAIN_D", 768),
        nlist: env_usize("TRAIN_NLIST", 1024),
        pq_m: env_usize("TRAIN_PQ_M", 96),
    });
}

fn run_scenario(s: &Scenario) {
    let data = generate_normalized_vectors(s.n, s.d, 20260806);

    // Authoritative total: real IVFPQIndex::train.
    let mut index = IVFPQIndex::new(s.d, s.nlist, s.pq_m, MetricType::InnerProduct, false);
    let t_train = Instant::now();
    index.train(&data, s.n);
    let train_secs = t_train.elapsed().as_secs_f64();

    // Mirrored phases for attribution only.
    let t_prep = Instant::now();
    let effective_data = data[..s.n * s.d].to_vec();
    let prep_secs = t_prep.elapsed().as_secs_f64();

    let km_config = KMeansConfig::default();
    let t_coarse = Instant::now();
    let centroids = kmeans::kmeans_train(&km_config, &effective_data, s.n, s.d, s.nlist);
    let coarse_secs = t_coarse.elapsed().as_secs_f64();

    let mut pq = ProductQuantizer::new(s.d, s.pq_m);
    let t_pq = Instant::now();
    pq.train(&effective_data, s.n);
    let pq_secs = t_pq.elapsed().as_secs_f64();

    // Keep results observable so nothing is optimized away.
    let checksum: f32 =
        centroids.iter().take(8).sum::<f32>() + pq.centroids.iter().take(8).sum::<f32>();

    println!(
        "{:<11} {:>8} {:>5} {:>6} {:>5} {:>8.3} {:>9.3} {:>8.3} {:>14.3}",
        s.name, s.n, s.d, s.nlist, s.pq_m, prep_secs, coarse_secs, pq_secs, train_secs
    );
    debug_assert!(checksum.is_finite());
}

fn generate_normalized_vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * d];
    for row in data.chunks_mut(d) {
        let mut norm_sq = 0.0f32;
        for v in row.iter_mut() {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
            norm_sq += *v * *v;
        }
        let inv = 1.0 / norm_sq.sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    data
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("invalid {}: {}", name, v))
        })
        .unwrap_or(default)
}
