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

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::io::{write_index, IVFPQIndexReader, PosWriter};
use paimon_vindex_core::ivfpq::{search_batch_reader_filter, IVFPQIndex, RowIdFilter};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::io::Cursor;
use std::time::{Duration, Instant};

const D: usize = 768;
const M: usize = 96;
const NLIST: usize = 8;
const NPROBE: usize = 4;
const NQ: usize = 8;
const K: usize = 3;
const ROWS_PER_LIST: usize = 6_800;
const WARMUPS: usize = 3;
const ROUNDS: usize = 15;

struct DensityFilter {
    matching_remainders: i64,
}

impl RowIdFilter for DensityFilter {
    fn contains(&self, id: i64) -> bool {
        id.rem_euclid(16) < self.matching_remainders
    }
}

fn search(
    reader: &mut IVFPQIndexReader<Cursor<Vec<u8>>>,
    queries: &[f32],
    filter: &DensityFilter,
) -> Duration {
    let started = Instant::now();
    let result = search_batch_reader_filter(reader, queries, NQ, K, NPROBE, Some(filter)).unwrap();
    let elapsed = started.elapsed();
    black_box(result);
    elapsed
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    assert!(
        std::env::var_os("PAIMON_VINDEX_LOG_IVFPQ_TIMING").is_none(),
        "unset PAIMON_VINDEX_LOG_IVFPQ_TIMING for this benchmark"
    );
    assert_eq!(
        rayon::current_num_threads(),
        1,
        "run with RAYON_NUM_THREADS=1"
    );

    let mut rng = StdRng::seed_from_u64(42);
    let mut index = IVFPQIndex::new(D, NLIST, M, MetricType::InnerProduct, false);
    index.quantizer_centroids = (0..NLIST * D)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    index.pq.centroids = (0..M * index.pq.ksub * index.pq.dsub)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    for list_id in 0..NLIST {
        let first_id = list_id * ROWS_PER_LIST;
        index.ids[list_id] = (first_id..first_id + ROWS_PER_LIST)
            .map(|id| id as i64)
            .collect();
        index.codes[list_id] = (0..ROWS_PER_LIST * M).map(|_| rng.gen()).collect();
    }
    let queries = (0..NQ * D)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
    let densities = [
        (1, "6.25"),
        (2, "12.50"),
        (3, "18.75"),
        (4, "25.00"),
        (8, "50.00"),
        (16, "100.00"),
    ];

    println!(
        "shape: d={D} m={M} nlist={NLIST} nprobe={NPROBE} nq={NQ} k={K} rows_per_list={ROWS_PER_LIST} threads=1 rounds={ROUNDS}"
    );
    println!("density_percent,p50_ms");
    for (matching_remainders, density) in densities {
        let filter = DensityFilter {
            matching_remainders,
        };
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        for _ in 0..WARMUPS {
            search(&mut reader, &queries, &filter);
        }
        let mut samples = (0..ROUNDS)
            .map(|_| search(&mut reader, &queries, &filter))
            .collect::<Vec<_>>();
        println!(
            "{density},{:.3}",
            median(&mut samples).as_secs_f64() * 1_000.0
        );
    }
}
