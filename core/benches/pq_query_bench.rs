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

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::io::{write_index, IVFPQIndexReader, PosWriter};
use paimon_vindex_core::ivfpq::{search_batch_reader_filter, IVFPQIndex, RowIdFilter};
use paimon_vindex_core::pq::ProductQuantizer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Cursor;
use std::time::Duration;

fn distance_tables(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_distance_table");
    for (d, m, bits) in [(128, 32, 4), (128, 16, 8), (768, 96, 8), (768, 16, 8)] {
        let mut rng = StdRng::seed_from_u64(42);
        let mut pq = ProductQuantizer::with_nbits(d, m, bits);
        pq.set_centroids(
            (0..d * pq.ksub())
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect(),
        );
        let query: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut table = vec![0.0; m * pq.ksub()];
        for metric in [MetricType::L2, MetricType::InnerProduct] {
            group.bench_function(
                BenchmarkId::new(format!("{metric:?}"), format!("d{d}_m{m}_b{bits}")),
                |b| {
                    b.iter(|| {
                        pq.compute_distance_table(black_box(&query), metric, black_box(&mut table));
                        black_box(&table);
                    })
                },
            );
        }
    }
    group.finish();
}

struct SampleFilter {
    percent: usize,
}

impl RowIdFilter for SampleFilter {
    fn contains(&self, id: i64) -> bool {
        // Stable shuffled membership throughout each list, rather than a prefix.
        (id as u64).wrapping_mul(0x9e3779b97f4a7c15).rotate_left(17) % 100 < self.percent as u64
    }
}

fn reader_scans(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq4_reader");
    let (d, m, nlist, nq, k, nprobe) = (128, 32, 8, 16, 10, 4);
    for rows in [256, 4096] {
        let mut rng = StdRng::seed_from_u64(42);
        let mut index = IVFPQIndex::with_nbits(d, nlist, m, 4, MetricType::InnerProduct, false);
        index.set_quantizer_centroids((0..nlist * d).map(|_| rng.gen_range(-1.0..1.0)).collect());
        index
            .pq
            .set_centroids((0..d * 16).map(|_| rng.gen_range(-1.0..1.0)).collect());
        for list in 0..nlist {
            index.ids[list] = (list * rows..(list + 1) * rows)
                .map(|id| id as i64)
                .collect();
            index.codes[list] = (0..rows * m / 2).map(|_| rng.gen()).collect();
        }
        let queries: Vec<f32> = (0..nq * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        for percent in [1, 6, 25, 100] {
            let filter = SampleFilter { percent };
            let filter_ref = (percent != 100).then_some(&filter as &dyn RowIdFilter);
            let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
            reader.optimize_for_search().unwrap();
            let expected =
                search_batch_reader_filter(&mut reader, &queries, nq, k, nprobe, filter_ref)
                    .unwrap();
            assert!(expected.0.iter().all(|&id| id < 0 || filter.contains(id)));
            let signature = expected
                .0
                .iter()
                .zip(&expected.1)
                .fold(0u64, |hash, (&id, &dist)| {
                    hash.wrapping_mul(31)
                        .wrapping_add(id as u64)
                        .wrapping_add(dist.to_bits() as u64)
                });
            eprintln!("pq4 signature rows={rows} percent={percent}: {signature:016x}");
            group.bench_function(
                BenchmarkId::new(format!("rows{rows}"), format!("filter{percent}")),
                |b| {
                    b.iter(|| {
                        black_box(
                            search_batch_reader_filter(
                                &mut reader,
                                black_box(&queries),
                                nq,
                                k,
                                nprobe,
                                filter_ref,
                            )
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = distance_tables, reader_scans
}
criterion_main!(benches);
