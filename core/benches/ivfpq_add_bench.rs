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

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    d: usize,
    m: usize,
    nlist: usize,
    rows: usize,
}

const CASES: [Case; 10] = [
    Case {
        name: "dsub4",
        d: 768,
        m: 192,
        nlist: 1,
        rows: 1,
    },
    Case {
        name: "dsub4",
        d: 768,
        m: 192,
        nlist: 1,
        rows: 7,
    },
    Case {
        name: "dsub4",
        d: 768,
        m: 192,
        nlist: 1,
        rows: 8,
    },
    Case {
        name: "dsub4",
        d: 768,
        m: 192,
        nlist: 1,
        rows: 31,
    },
    Case {
        name: "dsub4",
        d: 768,
        m: 192,
        nlist: 1,
        rows: 32,
    },
    Case {
        name: "dsub4",
        d: 768,
        m: 192,
        nlist: 1,
        rows: 512,
    },
    Case {
        name: "dsub4",
        d: 768,
        m: 192,
        nlist: 1,
        rows: 4096,
    },
    Case {
        name: "coverage_dsub4_batch32768",
        d: 32,
        m: 8,
        nlist: 1,
        rows: 32768,
    },
    Case {
        name: "coverage_dsub8_batch32768",
        d: 64,
        m: 8,
        nlist: 1,
        rows: 32768,
    },
    Case {
        name: "coverage_e2e_residual",
        d: 768,
        m: 192,
        nlist: 4096,
        rows: 32,
    },
];

fn new_index(
    case: Case,
    quantizer_centroids: &[f32],
    centroids: &[f32],
    norms: &[f32],
) -> IVFPQIndex {
    let mut index = IVFPQIndex::new(case.d, case.nlist, case.m, MetricType::L2, false);
    index.set_quantizer_centroids(quantizer_centroids.to_vec());
    index.pq.centroids = centroids.to_vec();
    index.pq.centroid_norms_cache = norms.to_vec();
    index
}

fn bench_ivfpq_add(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("ivfpq_add/threads{}", rayon::current_num_threads()));
    for case in CASES {
        let mut rng = StdRng::seed_from_u64(42);
        let data = (0..case.rows * case.d)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect::<Vec<_>>();
        let ids = (0..case.rows as i64).collect::<Vec<_>>();
        let quantizer_centroids = if case.nlist == 1 {
            vec![0.0; case.d]
        } else {
            (0..case.nlist * case.d)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect()
        };
        let dsub = case.d / case.m;
        let centroids = (0..case.m * 256 * dsub)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect::<Vec<_>>();
        let norms = centroids
            .chunks_exact(dsub)
            .map(|centroid| centroid.iter().map(|value| value * value).sum())
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements(case.rows as u64));
        group.bench_with_input(
            BenchmarkId::new(
                case.name,
                format!(
                    "d{}_m{}_dsub{dsub}_nlist{}_rows{}",
                    case.d, case.m, case.nlist, case.rows
                ),
            ),
            &case,
            |b, &case| {
                b.iter_batched(
                    || new_index(case, &quantizer_centroids, &centroids, &norms),
                    |mut index| index.add(black_box(&data), black_box(&ids), case.rows),
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_ivfpq_add
}
criterion_main!(benches);
