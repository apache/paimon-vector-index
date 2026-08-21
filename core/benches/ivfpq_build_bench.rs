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

//! End-to-end IVF-PQ build benchmark.
//!
//! Configure with `BENCH_N`, `BENCH_D`, `BENCH_NLIST`, `BENCH_M`, and
//! `BENCH_REPEATS`. Thread count follows `RAYON_NUM_THREADS`.

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let n = env_usize("BENCH_N", 50_000);
    let d = env_usize("BENCH_D", 128);
    let nlist = env_usize("BENCH_NLIST", 1024);
    let m = env_usize("BENCH_M", 16);
    let repeats = env_usize("BENCH_REPEATS", 3);
    let data = generate_normalized_vectors(n, d, 20260821);
    let ids = (0..n as i64).collect::<Vec<_>>();

    println!("n,d,nlist,m,threads,phase,run,seconds,rows_per_second");
    let mut index = IVFPQIndex::new(d, nlist, m, MetricType::InnerProduct, false);
    let started = Instant::now();
    index.train(&data, n);
    println!(
        "{n},{d},{nlist},{m},{},train,0,{:.6},0",
        rayon::current_num_threads(),
        started.elapsed().as_secs_f64()
    );

    for run in 1..=repeats {
        let started = Instant::now();
        index.add(&data, &ids, n);
        let seconds = started.elapsed().as_secs_f64();
        assert_eq!(index.ids.iter().map(Vec::len).sum::<usize>(), n * run);
        println!(
            "{n},{d},{nlist},{m},{},add,{run},{seconds:.6},{:.0}",
            rayon::current_num_threads(),
            n as f64 / seconds
        );
    }
}

fn generate_normalized_vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * d];
    for row in data.chunks_mut(d) {
        let mut norm_sq = 0.0f32;
        for value in row.iter_mut() {
            *value = rng.gen::<f32>() * 2.0 - 1.0;
            norm_sq += *value * *value;
        }
        let inv_norm = norm_sq.sqrt().recip();
        for value in row {
            *value *= inv_norm;
        }
    }
    data
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("invalid {name}: {value}"))
        })
        .unwrap_or(default)
}
