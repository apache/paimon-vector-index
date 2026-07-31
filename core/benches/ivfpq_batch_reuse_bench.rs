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
use paimon_vindex_core::ivfpq::{
    search_batch_reader_with_reuse_mode, IVFPQIndex, IvfPqBatchTableReuseMode,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::io::Cursor;
use std::time::Instant;

const D: usize = 768;
const M: usize = 16;
const NLIST: usize = 256;
const NPROBE: usize = 8;
const NQ: usize = 256;
const K: usize = 10;
const ROWS_PER_LIST: usize = 390;
const ROUNDS: usize = 100;

#[derive(Clone, Copy)]
struct CpuTimes {
    user: f64,
    system: f64,
}

#[derive(Clone, Copy)]
struct Sample {
    wall: f64,
    user: f64,
    system: f64,
}

fn cpu_times() -> CpuTimes {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    assert_eq!(result, 0);
    let usage = unsafe { usage.assume_init() };
    CpuTimes {
        user: usage.ru_utime.tv_sec as f64 + usage.ru_utime.tv_usec as f64 / 1_000_000.0,
        system: usage.ru_stime.tv_sec as f64 + usage.ru_stime.tv_usec as f64 / 1_000_000.0,
    }
}

fn search(
    reader: &mut IVFPQIndexReader<Cursor<Vec<u8>>>,
    queries: &[f32],
    mode: IvfPqBatchTableReuseMode,
) -> ((Vec<i64>, Vec<f32>), Sample) {
    let cpu_before = cpu_times();
    let started = Instant::now();
    let result = search_batch_reader_with_reuse_mode(reader, queries, NQ, K, NPROBE, mode).unwrap();
    let wall = started.elapsed().as_secs_f64();
    let cpu_after = cpu_times();
    black_box(result.0.iter().fold(0i64, |sum, id| sum.wrapping_add(*id)));
    (
        result,
        Sample {
            wall,
            user: cpu_after.user - cpu_before.user,
            system: cpu_after.system - cpu_before.system,
        },
    )
}

fn percentile(samples: &[Sample], percentile: usize, value: impl Fn(&Sample) -> f64) -> f64 {
    let mut values = samples.iter().map(value).collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[(percentile * values.len()).div_ceil(100).saturating_sub(1)]
}

fn main() {
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

    let modes = [
        IvfPqBatchTableReuseMode::Off,
        IvfPqBatchTableReuseMode::On,
        IvfPqBatchTableReuseMode::Auto,
    ];
    let mut readers = [
        IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap(),
        IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap(),
        IVFPQIndexReader::open(Cursor::new(bytes)).unwrap(),
    ];
    let expected = search(&mut readers[0], &queries, modes[0]).0;
    assert_eq!(search(&mut readers[1], &queries, modes[1]).0, expected);
    assert_eq!(search(&mut readers[2], &queries, modes[2]).0, expected);

    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    let orders = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
    for round in 0..ROUNDS {
        for &mode_index in &orders[round % orders.len()] {
            let (_, sample) = search(&mut readers[mode_index], &queries, modes[mode_index]);
            samples[mode_index].push(sample);
        }
    }

    println!(
        "shape: d={D} m={M} nlist={NLIST} nprobe={NPROBE} nq={NQ} vectors={} threads={} rounds={ROUNDS}",
        NLIST * ROWS_PER_LIST,
        rayon::current_num_threads()
    );
    println!("mode,percentile,wall_ms,user_cpu_ms,sys_cpu_ms,total_cpu_ms,cpu/wall");
    for (mode, samples) in ["off", "on", "auto"].into_iter().zip(&samples) {
        for p in [50, 90, 95, 99] {
            let wall = percentile(samples, p, |sample| sample.wall);
            let user = percentile(samples, p, |sample| sample.user);
            let system = percentile(samples, p, |sample| sample.system);
            let total = percentile(samples, p, |sample| sample.user + sample.system);
            println!(
                "{mode},p{p},{:.3},{:.3},{:.3},{:.3},{:.2}",
                wall * 1_000.0,
                user * 1_000.0,
                system * 1_000.0,
                total * 1_000.0,
                total / wall
            );
        }
    }
}
