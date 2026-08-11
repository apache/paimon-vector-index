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
use roaring::RoaringTreemap;
use std::hint::black_box;
use std::io::Cursor;
use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

const D: usize = 128;
const M: usize = 32;
const NLIST: usize = 64;
const NPROBE: usize = 8;
const NQ: usize = 64;
const K: usize = 10;
const ROWS_PER_LIST: [usize; 3] = [1_000, 10_000, 100_000];
const ROUNDS: usize = 10;
const FILTER_RATES: [(&str, u64, u64); 6] = [
    ("1%", 1, 100),
    ("5%", 1, 20),
    ("10%", 1, 10),
    ("12.5%", 1, 8),
    ("25%", 1, 4),
    ("100%", 1, 1),
];

fn percentile(samples: &[Duration], percentile: usize) -> f64 {
    let mut values = samples.to_vec();
    values.sort_unstable();
    values[(percentile * values.len()).div_ceil(100).saturating_sub(1)].as_secs_f64() * 1_000.0
}

fn process_cpu_time() -> Duration {
    let mut time = MaybeUninit::<libc::timespec>::uninit();
    let status = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, time.as_mut_ptr()) };
    assert_eq!(status, 0, "failed to read process CPU time");
    let time = unsafe { time.assume_init() };
    Duration::new(time.tv_sec as u64, time.tv_nsec as u32)
}

fn search(
    reader: &mut IVFPQIndexReader<Cursor<Vec<u8>>>,
    queries: &[f32],
    filter: Option<&RoaringTreemap>,
) -> (Duration, Duration) {
    let cpu_started = process_cpu_time();
    let started = Instant::now();
    let row_filter = filter.map(|value| value as &dyn RowIdFilter);
    let result = search_batch_reader_filter(reader, queries, NQ, K, NPROBE, row_filter).unwrap();
    let elapsed = started.elapsed();
    let cpu_elapsed = process_cpu_time() - cpu_started;
    if let Some(filter) = filter {
        assert!(result
            .0
            .iter()
            .filter(|&&id| id >= 0)
            .all(|&id| filter.contains(id as u64)));
    }
    black_box(result);
    (elapsed, cpu_elapsed)
}

fn run_shape(rows_per_list: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut index = IVFPQIndex::with_nbits(D, NLIST, M, 4, MetricType::InnerProduct, false);
    index.quantizer_centroids = (0..NLIST * D)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    index.pq.centroids = (0..M * index.pq.ksub * index.pq.dsub)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    for list_id in 0..NLIST {
        let first_id = list_id * rows_per_list;
        index.ids[list_id] = (first_id..first_id + rows_per_list)
            .map(|id| id as i64)
            .collect();
        index.codes[list_id] = (0..rows_per_list * index.pq.code_size())
            .map(|_| rng.gen())
            .collect();
    }
    let queries = (0..NQ * D)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
    let mut reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();

    let total_rows = (NLIST * rows_per_list) as u64;
    let mut cases = vec![("none", None)];
    cases.extend(FILTER_RATES.map(|(label, numerator, denominator)| {
        let filter = (0..total_rows)
            .filter(|id| id % denominator < numerator)
            .collect::<RoaringTreemap>();
        (label, Some(filter))
    }));
    for (_, filter) in &cases {
        black_box(search(&mut reader, &queries, filter.as_ref()));
    }

    let mut wall_samples = vec![Vec::new(); cases.len()];
    let mut cpu_samples = vec![Vec::new(); cases.len()];
    for round in 0..ROUNDS {
        for offset in 0..cases.len() {
            let case = (round + offset) % cases.len();
            let (wall, cpu) = search(&mut reader, &queries, cases[case].1.as_ref());
            wall_samples[case].push(wall);
            cpu_samples[case].push(cpu);
        }
    }

    let unfiltered_p50 = percentile(&wall_samples[0], 50);
    println!(
        "shape: 4-bit reader d={D} m={M} nlist={NLIST} rows_per_list={rows_per_list} nprobe={NPROBE} nq={NQ} vectors={total_rows} threads={} rounds={ROUNDS}",
        rayon::current_num_threads()
    );
    println!("filter_rate,matching_rows,p50_wall_ms,p95_wall_ms,p50_cpu_ms,p50_vs_unfiltered");
    for (case, (label, filter)) in cases.iter().enumerate() {
        let p50 = percentile(&wall_samples[case], 50);
        let p95 = percentile(&wall_samples[case], 95);
        let cpu_p50 = percentile(&cpu_samples[case], 50);
        println!(
            "{label},{},{p50:.3},{p95:.3},{cpu_p50:.3},{:.3}",
            filter.as_ref().map_or(total_rows, RoaringTreemap::len),
            p50 / unfiltered_p50
        );
    }
}

fn main() {
    for rows_per_list in ROWS_PER_LIST {
        run_shape(rows_per_list);
    }
}
