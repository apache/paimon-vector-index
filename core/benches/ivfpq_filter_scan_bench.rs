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
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::io::Cursor;
use std::time::{Duration, Instant};

const D: usize = 768;
const M: usize = 96;
const NLIST: usize = 128;
const NPROBE: usize = 64;
const NQ: usize = 64;
const K: usize = 3;
const ROWS_PER_LIST: usize = 6_800;
const WARMUPS: usize = 1;
const ROUNDS: usize = 3;

struct DensityFilter<'a> {
    row_ranks: &'a [usize],
    max_rank: usize,
}

impl RowIdFilter for DensityFilter<'_> {
    fn contains(&self, id: i64) -> bool {
        self.row_ranks
            .get(id as usize)
            .is_some_and(|&rank| rank < self.max_rank)
    }
}

fn randomized_row_ranks() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(43);
    let mut row_ranks = vec![0; NLIST * ROWS_PER_LIST];
    let mut row_order = (0..ROWS_PER_LIST).collect::<Vec<_>>();
    for list_id in 0..NLIST {
        row_order.shuffle(&mut rng);
        let base = list_id * ROWS_PER_LIST;
        for (rank, &row) in row_order.iter().enumerate() {
            row_ranks[base + row] = rank;
        }
    }
    row_ranks
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
    assert_eq!((NQ, NPROBE), (64, 64), "benchmark production query shape");
    assert!(
        std::env::var_os("PAIMON_VINDEX_LOG_IVFPQ_TIMING").is_none(),
        "unset PAIMON_VINDEX_LOG_IVFPQ_TIMING for this benchmark"
    );
    assert_eq!(
        rayon::current_num_threads(),
        1,
        "run with RAYON_NUM_THREADS=1"
    );
    const { assert!(NQ >= 64, "keep production query-table reuse enabled") };

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
    let row_ranks = randomized_row_ranks();
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
        "shape: d={D} m={M} nlist={NLIST} nprobe={NPROBE} nq={NQ} k={K} rows_per_list={ROWS_PER_LIST} threads=1 warmups={WARMUPS} rounds={ROUNDS}"
    );
    println!("density_percent,p50_ms");
    for (matching_sixteenths, density) in densities {
        let filter = DensityFilter {
            row_ranks: &row_ranks,
            max_rank: ROWS_PER_LIST * matching_sixteenths / 16,
        };
        assert_eq!(
            (0..ROWS_PER_LIST)
                .filter(|&id| filter.contains(id as i64))
                .count(),
            filter.max_rank
        );
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
