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

//! Full trained/serialized PQ query matrix. Configure PQ_QUALITY_BITS (4/8),
//! PQ_QUALITY_METRIC (l2/ip/cosine), PQ_QUALITY_OPQ (0/1), and optional
//! PQ_QUALITY_DATA_DIR containing converted fashion_{base,query}.fvecs.

use paimon_vindex_core::distance::{fvec_distance, fvec_normalize, MetricType};
use paimon_vindex_core::io::{write_index, IVFPQIndexReader, PosWriter};
use paimon_vindex_core::ivfpq::{
    search_batch_reader_filter, search_with_reader_filter, IVFPQIndex, RowIdFilter,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::{
    collections::HashSet,
    hint::black_box,
    io::Cursor,
    path::Path,
    time::{Duration, Instant},
};

const NQ: usize = 64;
const K: usize = 10;
const NPROBE: usize = 8;

struct Filter(usize);
impl RowIdFilter for Filter {
    fn contains(&self, id: i64) -> bool {
        (id as u64).wrapping_mul(0x9e3779b97f4a7c15).rotate_left(17) % 100 < self.0 as u64
    }
}

fn read_fvecs(path: &Path) -> (usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    let d = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let width = (d + 1) * 4;
    assert!(d > 0 && bytes.len().is_multiple_of(width));
    let mut values = Vec::with_capacity(bytes.len() / width * d);
    for row in bytes.chunks_exact(width) {
        assert_eq!(u32::from_le_bytes(row[..4].try_into().unwrap()) as usize, d);
        values.extend(
            row[4..]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap())),
        );
    }
    (d, values)
}

fn dataset() -> (String, usize, Vec<f32>, Vec<f32>) {
    if let Ok(directory) = std::env::var("PQ_QUALITY_DATA_DIR") {
        let (d, base) = read_fvecs(&Path::new(&directory).join("fashion_base.fvecs"));
        let (qd, mut queries) = read_fvecs(&Path::new(&directory).join("fashion_query.fvecs"));
        assert_eq!(d, qd);
        assert!(queries.len() >= NQ * d);
        queries.truncate(NQ * d);
        return ("fashion".into(), d, base, queries);
    }
    let (d, n) = (128, 20_000);
    let mut rng = StdRng::seed_from_u64(42);
    let centers: Vec<f32> = (0..32 * d).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let mut generate = |rows: usize| {
        (0..rows * d)
            .map(|i| centers[(i / d % 32) * d + i % d] + rng.gen_range(-0.5..0.5))
            .collect::<Vec<_>>()
    };
    let base = generate(n);
    let queries = generate(NQ);
    ("synthetic".into(), d, base, queries)
}

fn recall(ids: &[i64], truth: &[HashSet<i64>]) -> f64 {
    ids.chunks_exact(K)
        .zip(truth)
        .map(|(got, expected)| got.iter().filter(|id| expected.contains(id)).count())
        .sum::<usize>() as f64
        / (NQ * K) as f64
}

fn main() {
    let bits = std::env::var("PQ_QUALITY_BITS")
        .unwrap_or_else(|_| "8".into())
        .parse::<usize>()
        .unwrap();
    assert!(matches!(bits, 4 | 8));
    let metric = match std::env::var("PQ_QUALITY_METRIC")
        .unwrap_or_else(|_| "l2".into())
        .as_str()
    {
        "l2" => MetricType::L2,
        "ip" => MetricType::InnerProduct,
        "cosine" => MetricType::Cosine,
        _ => panic!("invalid metric"),
    };
    let opq = std::env::var("PQ_QUALITY_OPQ").as_deref() == Ok("1");
    let (name, d, mut data, mut queries) = dataset();
    if metric != MetricType::L2 {
        for row in data.chunks_exact_mut(d).chain(queries.chunks_exact_mut(d)) {
            fvec_normalize(row);
        }
    }
    let n = data.len() / d;
    let train_n = if name == "fashion" { 16384 } else { 4096 };
    let m = if bits == 8 { d / 8 } else { d / 4 };
    let ids: Vec<i64> = (0..n as i64).collect();
    let mut index = IVFPQIndex::with_nbits(d, 64, m, bits, metric, opq);
    let started = Instant::now();
    index.train(&data[..train_n * d], train_n);
    let train_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    index.add(&data, &ids, n);
    let add_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut bytes = Vec::new();
    write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
    let write_ms = started.elapsed().as_secs_f64() * 1000.0;
    let wire_signature = bytes.iter().fold(0u64, |hash, &value| {
        hash.wrapping_mul(31).wrapping_add(u64::from(value))
    });
    println!("dataset,d,n,train_n,bits,m,metric,opq,threads,nq,k,nprobe,density,train_ms,add_ms,write_ms,file_bytes,wire_signature,first_us,sequential_recall,batch_recall,sequential_qps,batch_qps,sequential_ms,batch_ms,sequential_queries,batch_queries");
    for density in [100, 1, 10] {
        let filter = Filter(density);
        let filter_ref = (density != 100).then_some(&filter as &dyn RowIdFilter);
        let truth: Vec<HashSet<i64>> = queries
            .par_chunks_exact(d)
            .map(|q| {
                let mut distances: Vec<_> = data
                    .chunks_exact(d)
                    .enumerate()
                    .filter(|(id, _)| filter.contains(*id as i64))
                    .map(|(id, v)| (fvec_distance(q, v, metric), id as i64))
                    .collect();
                assert!(distances.len() >= K);
                distances
                    .select_nth_unstable_by(K - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                distances[..K].iter().map(|&(_, id)| id).collect()
            })
            .collect();
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        reader.optimize_for_search().unwrap();
        let started = Instant::now();
        black_box(
            search_with_reader_filter(&mut reader, &queries[..d], K, NPROBE, filter_ref).unwrap(),
        );
        let first_us = started.elapsed().as_secs_f64() * 1e6;
        let sequential_ids: Vec<i64> = queries
            .chunks_exact(d)
            .flat_map(|q| {
                search_with_reader_filter(&mut reader, q, K, NPROBE, filter_ref)
                    .unwrap()
                    .0
            })
            .collect();
        let (batch_ids, _) =
            search_batch_reader_filter(&mut reader, &queries, NQ, K, NPROBE, filter_ref).unwrap();
        assert!(sequential_ids
            .iter()
            .chain(&batch_ids)
            .all(|&id| id < 0 || filter.contains(id)));
        let started = Instant::now();
        let mut sequential_queries = 0usize;
        while started.elapsed() < Duration::from_millis(500) {
            for q in queries.chunks_exact(d) {
                black_box(
                    search_with_reader_filter(&mut reader, q, K, NPROBE, filter_ref).unwrap(),
                );
            }
            sequential_queries += NQ;
        }
        let sequential_s = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut batch_queries = 0usize;
        while started.elapsed() < Duration::from_millis(500) {
            black_box(
                search_batch_reader_filter(&mut reader, &queries, NQ, K, NPROBE, filter_ref)
                    .unwrap(),
            );
            batch_queries += NQ;
        }
        let batch_s = started.elapsed().as_secs_f64();
        println!("{name},{d},{n},{train_n},{bits},{m},{metric:?},{opq},{},{NQ},{K},{NPROBE},{density},{train_ms:.3},{add_ms:.3},{write_ms:.3},{},{wire_signature:016x},{first_us:.3},{:.6},{:.6},{:.3},{:.3},{:.3},{:.3},{sequential_queries},{batch_queries}",
            rayon::current_num_threads(), bytes.len(), recall(&sequential_ids, &truth), recall(&batch_ids, &truth), sequential_queries as f64/sequential_s, batch_queries as f64/batch_s, sequential_s*1000.0, batch_s*1000.0);
    }
}
