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

//! Calibrate sparse/dense 4-bit reader crossover without changing score semantics.
//! Each row is the median of three >=40ms runs, after an untimed result capture.
use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::io::{write_index, IVFPQIndexReader, PosWriter};
use paimon_vindex_core::ivfpq::{search_batch_reader_filter, IVFPQIndex, RowIdFilter};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::{
    hint::black_box,
    io::Cursor,
    time::{Duration, Instant},
};
struct Filter {
    ranks: Vec<usize>,
    matches: usize,
}
impl RowIdFilter for Filter {
    fn contains(&self, id: i64) -> bool {
        self.ranks
            .get(id as usize)
            .is_some_and(|&rank| rank < self.matches)
    }
}
fn main() {
    assert_eq!(
        rayon::current_num_threads(),
        1,
        "run with RAYON_NUM_THREADS=1"
    );
    let nq = 16;
    let k = 3;
    println!("m,d,rows,matches,density_percent,median_us,result_signature");
    let validation = std::env::var_os("PQ_SPARSE_BENCH_VALIDATION").is_some();
    let overflow = std::env::var_os("PQ_SPARSE_BENCH_OVERFLOW").is_some();
    let widths: &[usize] = if overflow {
        &[258]
    } else if validation {
        &[24, 48, 128]
    } else {
        &[16, 32, 64, 96, 196, 256]
    };
    let lengths: &[usize] = if validation {
        &[512, 2048, 8192, 16384]
    } else {
        &[128, 256, 1024, 4096, 6800]
    };
    let densities: &[usize] = if validation {
        &[100, 500, 1000, 1875, 2500, 5000, 10000]
    } else {
        &[100, 312, 625, 1250, 2500, 5000, 10000]
    };
    for &m in widths {
        let d = m * 4;
        for &rows in lengths {
            let mut rng = StdRng::seed_from_u64(42);
            let mut index = IVFPQIndex::with_nbits(d, 1, m, 4, MetricType::InnerProduct, false);
            index.set_quantizer_centroids(vec![0.0; d]);
            index
                .pq
                .set_centroids((0..d * 16).map(|_| rng.gen_range(-1.0..1.0)).collect());
            index.ids[0] = (0..rows as i64).collect();
            index.codes[0] = (0..rows * m / 2).map(|_| rng.gen()).collect();
            let queries: Vec<f32> = (0..nq * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let mut bytes = Vec::new();
            write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
            let mut reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();
            let mut ranks: Vec<usize> = (0..rows).collect();
            ranks.shuffle(&mut rng);
            let mut filter = Filter { ranks, matches: 0 };
            for &density in densities {
                filter.matches = rows * density / 10000;
                let expected =
                    search_batch_reader_filter(&mut reader, &queries, nq, k, 1, Some(&filter))
                        .unwrap();
                let signature =
                    expected
                        .0
                        .iter()
                        .zip(&expected.1)
                        .fold(0u64, |sum, (&id, &dist)| {
                            sum.wrapping_mul(31)
                                .wrapping_add(id as u64)
                                .wrapping_add(u64::from(dist.to_bits()))
                        });
                let mut times = [0.0f64; 3];
                for time in &mut times {
                    let start = Instant::now();
                    let mut iterations = 0;
                    while start.elapsed() < Duration::from_millis(40) {
                        black_box(
                            search_batch_reader_filter(
                                &mut reader,
                                &queries,
                                nq,
                                k,
                                1,
                                Some(&filter),
                            )
                            .unwrap(),
                        );
                        iterations += 1;
                    }
                    *time = start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "{m},{d},{rows},{},{:.2},{:.3},{signature:016x}",
                    filter.matches,
                    density as f64 / 100.0,
                    times[1]
                );
            }
        }
    }
}
