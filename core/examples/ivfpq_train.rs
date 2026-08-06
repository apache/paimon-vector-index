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
//! The authoritative number for the speed gate is the `IVFPQIndex::train` wall
//! time. The mirrored phase timings (input preparation, coarse K-Means, PQ)
//! exist only for attribution and must never be summed for the gate.

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use paimon_vindex_core::kmeans::{self, KMeansConfig};
use paimon_vindex_core::pq::ProductQuantizer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("invalid {}: {}", name, v))
        })
        .unwrap_or(default)
}

fn main() {
    let n = env_usize("TRAIN_N", 244_606);
    let d = env_usize("TRAIN_D", 768);
    let nlist = env_usize("TRAIN_NLIST", 1024);
    let pq_m = env_usize("TRAIN_PQ_M", 96);

    let mut rng = StdRng::seed_from_u64(20260806);
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

    // Authoritative total: real IVFPQIndex::train (InnerProduct, no OPQ).
    let mut index = IVFPQIndex::new(d, nlist, pq_m, MetricType::InnerProduct, false);
    let t_train = Instant::now();
    index.train(&data, n);
    let train_secs = t_train.elapsed().as_secs_f64();

    // Mirrored phases for attribution only (matches the InnerProduct/no-OPQ
    // path in IVFPQIndex::train): clone input, coarse k-means, PQ train.
    let t_prep = Instant::now();
    let effective_data = data[..n * d].to_vec();
    let prep_secs = t_prep.elapsed().as_secs_f64();

    let km_config = KMeansConfig::default();
    let t_coarse = Instant::now();
    let centroids = kmeans::kmeans_train(&km_config, &effective_data, n, d, nlist);
    let coarse_secs = t_coarse.elapsed().as_secs_f64();

    let mut pq = ProductQuantizer::new(d, pq_m);
    let t_pq = Instant::now();
    pq.train(&effective_data, n);
    let pq_secs = t_pq.elapsed().as_secs_f64();

    // Keep results observable so nothing is optimized away.
    let checksum: f32 =
        centroids.iter().take(8).sum::<f32>() + pq.centroids.iter().take(8).sum::<f32>();

    println!(
        "ivfpq_train n={} d={} nlist={} pq_m={} threads={} train_total_s={:.3} \
         mirror_prep_s={:.3} mirror_coarse_s={:.3} mirror_pq_s={:.3} checksum={:.6}",
        n,
        d,
        nlist,
        pq_m,
        rayon::current_num_threads(),
        train_secs,
        prep_secs,
        coarse_secs,
        pq_secs,
        checksum
    );
}
