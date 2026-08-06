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

use paimon_vindex_core::distance::{fvec_distance, MetricType};
use paimon_vindex_core::io::{write_index, PosWriter};
use paimon_vindex_core::ivfflat::IVFFlatIndex;
use paimon_vindex_core::ivfflat_io::write_ivfflat_index;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use paimon_vindex_core::ivfsq::IVFSQIndex;
use paimon_vindex_core::ivfsq_io::write_ivfsq_index;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    run_scenario(Scenario {
        name: "small-lists",
        d: 64,
        n: 20_000,
        nq: 50,
        k: 10,
        nlist: 64,
        pq_m: 8,
        nprobes: &[1, 4, 8, 16, 32, 64],
        metric: MetricType::L2,
    });

    println!();

    run_scenario(Scenario {
        name: "large-lists",
        d: 64,
        n: 50_000,
        nq: 50,
        k: 10,
        nlist: 8,
        pq_m: 8,
        nprobes: &[1, 2, 4, 8],
        metric: MetricType::L2,
    });

    println!();

    // Exercises the hierarchical coarse k-means path (nlist > 256) with the
    // target workload's InnerProduct metric.
    run_scenario(Scenario {
        name: "inner-product-hierarchical",
        d: 64,
        n: 100_000,
        nq: 50,
        k: 10,
        nlist: 1024,
        pq_m: 8,
        nprobes: &[8, 16, 32, 64],
        metric: MetricType::InnerProduct,
    });
}

struct Scenario<'a> {
    name: &'a str,
    d: usize,
    n: usize,
    nq: usize,
    k: usize,
    nlist: usize,
    pq_m: usize,
    nprobes: &'a [usize],
    metric: MetricType,
}

fn run_scenario(s: Scenario<'_>) {
    println!("=== IVF Recall Attribution Benchmark ===");
    println!(
        "scenario: {}, n={}, nq={}, d={}, nlist={}, avg_list={}, k={}, metric={:?}",
        s.name,
        s.n,
        s.nq,
        s.d,
        s.nlist,
        s.n / s.nlist,
        s.k,
        s.metric
    );

    let mut data = generate_clustered_data(s.n, s.d, 32, 42);
    if s.metric == MetricType::InnerProduct {
        for row in data.chunks_mut(s.d) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
    let ids: Vec<i64> = (0..s.n as i64).collect();
    let queries = &data[..s.nq * s.d].to_vec();

    let start = Instant::now();
    let ground_truth = brute_force_ground_truth(&data, queries, s.n, s.nq, s.d, s.k, s.metric);
    println!("ground truth: {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut ivfpq = IVFPQIndex::new(s.d, s.nlist, s.pq_m, s.metric, false);
    ivfpq.train(&data, s.n);
    ivfpq.add(&data, &ids, s.n);
    ivfpq.build_precomputed_table();
    println!("build IVF-PQ: {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut ivfflat = IVFFlatIndex::new(s.d, s.nlist, s.metric);
    ivfflat.train(&data, s.n);
    ivfflat.add(&data, &ids, s.n);
    println!("build IVF-FLAT: {:.2}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut ivfsq = IVFSQIndex::new(s.d, s.nlist, s.metric);
    ivfsq.train(&data, s.n);
    ivfsq.add(&data, &ids, s.n);
    println!("build IVF-SQ scan: {:.2}s", start.elapsed().as_secs_f64());
    print_base_sizes(&ivfpq, &ivfflat, &ivfsq);

    println!();
    println!("baseline exact scans over stored representations");
    println!("index      nprobe  recall@{}  query_ms  us/query", s.k);
    println!("---------  ------  ---------  --------  --------");

    for &nprobe in s.nprobes {
        let mut distances = vec![0.0f32; s.nq * s.k];
        let mut labels = vec![0i64; s.nq * s.k];
        let start = Instant::now();
        ivfpq.search(queries, s.nq, s.k, nprobe, &mut distances, &mut labels);
        let elapsed = start.elapsed();
        print_row(
            "IVF-PQ",
            nprobe,
            recall_at_k(&labels, &ground_truth, s.nq, s.k),
            elapsed,
            s.nq,
        );

        let mut distances = vec![0.0f32; s.nq * s.k];
        let mut labels = vec![0i64; s.nq * s.k];
        let start = Instant::now();
        ivfflat.search(queries, s.nq, s.k, nprobe, &mut distances, &mut labels);
        let elapsed = start.elapsed();
        print_row(
            "IVF-FLAT",
            nprobe,
            recall_at_k(&labels, &ground_truth, s.nq, s.k),
            elapsed,
            s.nq,
        );

        let mut distances = vec![0.0f32; s.nq * s.k];
        let mut labels = vec![0i64; s.nq * s.k];
        let start = Instant::now();
        ivfsq.search(queries, s.nq, s.k, nprobe, &mut distances, &mut labels);
        let elapsed = start.elapsed();
        print_row(
            "IVF-SQ",
            nprobe,
            recall_at_k(&labels, &ground_truth, s.nq, s.k),
            elapsed,
            s.nq,
        );
    }
}

fn print_base_sizes(ivfpq: &IVFPQIndex, ivfflat: &IVFFlatIndex, ivfsq: &IVFSQIndex) {
    let mut pq = Vec::new();
    write_index(ivfpq, &mut PosWriter::new(&mut pq)).unwrap();
    let mut flat = Vec::new();
    write_ivfflat_index(ivfflat, &mut PosWriter::new(&mut flat)).unwrap();
    let mut sq = Vec::new();
    write_ivfsq_index(ivfsq, &mut PosWriter::new(&mut sq)).unwrap();

    println!(
        "serialized sizes: IVF-PQ={:.2} MiB, IVF-FLAT={:.2} MiB, IVF-SQ={:.2} MiB",
        bytes_to_mib(pq.len()),
        bytes_to_mib(flat.len()),
        bytes_to_mib(sq.len())
    );
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn print_row(index: &str, nprobe: usize, recall: f64, elapsed: std::time::Duration, nq: usize) {
    let ms = elapsed.as_secs_f64() * 1000.0;
    println!(
        "{:<9}  {:>6}  {:>8.2}%  {:>8.2}  {:>8.1}",
        index,
        nprobe,
        recall * 100.0,
        ms,
        ms * 1000.0 / nq as f64
    );
}

fn recall_at_k(labels: &[i64], ground_truth: &[Vec<i64>], nq: usize, k: usize) -> f64 {
    let mut hits = 0usize;
    for qi in 0..nq {
        let gt: HashSet<i64> = ground_truth[qi].iter().copied().collect();
        hits += labels[qi * k..(qi + 1) * k]
            .iter()
            .filter(|id| gt.contains(id))
            .count();
    }
    hits as f64 / (nq * k) as f64
}

fn brute_force_ground_truth(
    data: &[f32],
    queries: &[f32],
    n: usize,
    nq: usize,
    d: usize,
    k: usize,
    metric: MetricType,
) -> Vec<Vec<i64>> {
    (0..nq)
        .map(|qi| {
            let query = &queries[qi * d..(qi + 1) * d];
            let mut distances: Vec<(f32, i64)> = (0..n)
                .map(|i| {
                    let vector = &data[i * d..(i + 1) * d];
                    (fvec_distance(query, vector, metric), i as i64)
                })
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            distances[..k].iter().map(|&(_, id)| id).collect()
        })
        .collect()
}

fn generate_clustered_data(n: usize, d: usize, num_clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng_state = seed;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let mut centers = vec![0.0f32; num_clusters * d];
    for value in &mut centers {
        *value = next() * 30.0;
    }

    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let cluster = i % num_clusters;
        for j in 0..d {
            data[i * d + j] = centers[cluster * d + j] + next();
        }
    }
    data
}
