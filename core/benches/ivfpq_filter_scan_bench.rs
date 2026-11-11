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
    let bits = std::env::var("IVFPQ_BENCH_BITS")
        .unwrap_or_else(|_| "8".to_string())
        .parse::<usize>()
        .expect("IVFPQ_BENCH_BITS must be 4 or 8");
    assert!(matches!(bits, 4 | 8), "IVFPQ_BENCH_BITS must be 4 or 8");
    assert_eq!((NQ, NPROBE), (64, 64), "benchmark production query shape");
    for name in [
        "PAIMON_VINDEX_LOG_IVFPQ_BATCH_TIMING",
        "PAIMON_VINDEX_LOG_IVFPQ_BATCH_REUSE",
    ] {
        assert!(
            std::env::var_os(name).is_none(),
            "unset {name} for this benchmark"
        );
    }
    assert_eq!(
        rayon::current_num_threads(),
        1,
        "run with RAYON_NUM_THREADS=1"
    );
    const { assert!(NQ >= 64, "keep production query-table reuse enabled") };

    let mut rng = StdRng::seed_from_u64(42);
    let mut index = IVFPQIndex::with_nbits(D, NLIST, M, bits, MetricType::InnerProduct, false);
    index.set_quantizer_centroids(
        (0..NLIST * D)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect(),
    );
    index.pq.set_centroids(
        (0..M * index.pq.ksub() * index.pq.dsub())
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect(),
    );
    for list_id in 0..NLIST {
        let first_id = list_id * ROWS_PER_LIST;
        index.ids[list_id] = (first_id..first_id + ROWS_PER_LIST)
            .map(|id| id as i64)
            .collect();
        index.codes[list_id] = (0..ROWS_PER_LIST * index.pq.code_size())
            .map(|_| rng.gen())
            .collect();
    }
    let queries = (0..NQ * D)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect::<Vec<_>>();
    let row_ranks = randomized_row_ranks();
    let mut bytes = Vec::new();
    write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
    let densities = [
        (10, "0.10"),
        (100, "1.00"),
        (625, "6.25"),
        (1250, "12.50"),
        (1875, "18.75"),
        (2500, "25.00"),
        (5000, "50.00"),
        (10000, "100.00"),
    ];

    println!(
        "shape: bits={bits} d={D} m={M} nlist={NLIST} nprobe={NPROBE} nq={NQ} k={K} rows_per_list={ROWS_PER_LIST} threads=1 warmups={WARMUPS} rounds={ROUNDS}"
    );
    println!("density_percent,matches_per_list,p50_ms,ids_signature,distances_signature");
    for (matching_basis_points, density) in densities {
        let filter = DensityFilter {
            row_ranks: &row_ranks,
            max_rank: ROWS_PER_LIST * matching_basis_points / 10000,
        };
        assert_eq!(
            (0..ROWS_PER_LIST)
                .filter(|&id| filter.contains(id as i64))
                .count(),
            filter.max_rank
        );
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let result =
            search_batch_reader_filter(&mut reader, &queries, NQ, K, NPROBE, Some(&filter))
                .unwrap();
        let ids_signature = result.0.iter().fold(0u64, |hash, &id| {
            hash.wrapping_mul(31).wrapping_add(id as u64)
        });
        let distances_signature = result.1.iter().fold(0u64, |hash, &distance| {
            hash.wrapping_mul(31)
                .wrapping_add(u64::from(distance.to_bits()))
        });
        for _ in 0..WARMUPS {
            search(&mut reader, &queries, &filter);
        }
        let mut samples = (0..ROUNDS)
            .map(|_| search(&mut reader, &queries, &filter))
            .collect::<Vec<_>>();
        println!(
            "{density},{},{:.3},{ids_signature:016x},{distances_signature:016x}",
            filter.max_rank,
            median(&mut samples).as_secs_f64() * 1_000.0
        );
    }
}
