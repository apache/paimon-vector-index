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

//! Reproducible range recall and interleaved Top-K service-latency check.
//! Output belongs in target/; timings are measurements, not hard CI thresholds.
use paimon_vindex_core::diskann::{DiskAnnBuildParams, DiskAnnIndex, DiskAnnRawVectorEncoding};
use paimon_vindex_core::diskann_io::write_diskann_index;
use paimon_vindex_core::distance::{fvec_l2sqr, MetricType};
use paimon_vindex_core::index::{VectorIndexReader, VectorSearchParams};
use paimon_vindex_core::io::PosWriter;
use paimon_vindex_core::range::{Bound, DistanceBand, RangeSearchParams};
use std::collections::HashSet;
use std::io::Cursor;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (n, d, nq) = (4096, 32, 32);
    let mut state = 42u64;
    let data = (0..n * d)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as f32 / u32::MAX as f32
        })
        .collect::<Vec<_>>();
    let queries = &data[..nq * d];
    let mut index = DiskAnnIndex::with_pq_bits(
        d,
        MetricType::L2,
        4,
        4,
        DiskAnnBuildParams {
            max_degree: 24,
            build_search_list_size: 64,
            raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
            seed: 42,
            ..Default::default()
        },
    );
    index.train(&data, n)?;
    index.add(&data, &(0..n as i64).collect::<Vec<_>>());
    let mut bytes = Vec::new();
    write_diskann_index(&index, &mut PosWriter::new(&mut bytes))?;
    let mut reader = VectorIndexReader::open(Cursor::new(bytes))?;
    reader.optimize_for_search()?;
    let topk = VectorSearchParams::with_l_search(10, 100);
    let expected = queries
        .chunks_exact(d)
        .map(|q| reader.search(q, topk))
        .collect::<Result<Vec<_>, _>>()?;
    println!("kind,target_hits,queries,hits,recall,p95_us,pq_evaluations,vector_evaluations,read_calls,read_bytes");
    let measure_topk = |reader: &mut VectorIndexReader<Cursor<Vec<u8>>>,
                        interleave: bool|
     -> Result<(), Box<dyn std::error::Error>> {
        let mut latencies = Vec::new();
        let started = Instant::now();
        let broad = DistanceBand::from_raw(Bound::Unbounded, Bound::Finite(100.0), MetricType::L2)?;
        while started.elapsed() < Duration::from_secs(1) {
            for (i, query) in queries.chunks_exact(d).enumerate() {
                if interleave {
                    let previous = reader.diskann_search_stats();
                    let previous_unified = reader.search_stats();
                    let result =
                        reader.range_search(query, RangeSearchParams::diskann(broad, 64))?;
                    assert_eq!(result.labels().len(), n);
                    assert_eq!(reader.diskann_search_stats(), previous);
                    assert_eq!(reader.search_stats(), previous_unified);
                }
                let begin = Instant::now();
                let actual = reader.search(query, topk)?;
                latencies.push(begin.elapsed().as_nanos() as u64);
                assert_eq!(actual, expected[i]);
            }
        }
        latencies.sort_unstable();
        println!(
            "{},0,{},0,1.0,{:.3},0,0,0,0",
            if interleave {
                "topk_after_wide_range"
            } else {
                "topk_only"
            },
            latencies.len(),
            latencies[latencies.len() * 95 / 100] as f64 / 1000.0
        );
        Ok(())
    };
    measure_topk(&mut reader, false)?;
    for target in [1, 16, 128, n] {
        let (mut hits, mut expected_count, mut matched) = (0usize, 0usize, 0usize);
        let (mut pq, mut vectors, mut calls, mut read_bytes) = (0usize, 0usize, 0usize, 0usize);
        let mut latencies = Vec::new();
        for query in queries.chunks_exact(d) {
            let distances = data
                .chunks_exact(d)
                .map(|v| fvec_l2sqr(query, v))
                .collect::<Vec<_>>();
            let mut sorted = distances.clone();
            sorted.sort_unstable_by(f32::total_cmp);
            let upper = if target == n {
                sorted[n - 1] + 1.0
            } else {
                (sorted[target - 1] + sorted[target]) * 0.5
            };
            let truth = distances
                .iter()
                .enumerate()
                .filter(|(_, value)| **value < upper)
                .map(|(id, _)| id as i64)
                .collect::<HashSet<_>>();
            let band =
                DistanceBand::from_raw(Bound::Unbounded, Bound::Finite(upper), MetricType::L2)?;
            let params = RangeSearchParams::diskann(band, 64)
                .with_initial_slack(0.0)
                .with_range_slack(1.2);
            let started = Instant::now();
            let result = reader.range_search(query, params)?;
            latencies.push(started.elapsed().as_nanos() as u64);
            let unique = result.labels().iter().copied().collect::<HashSet<_>>();
            assert_eq!(unique.len(), result.labels().len());
            for (&id, &distance) in result.labels().iter().zip(result.raw_distances()) {
                assert!(
                    distance < upper && truth.contains(&id),
                    "range precision failure"
                );
            }
            hits += unique.len();
            expected_count += truth.len();
            matched += unique.intersection(&truth).count();
            let stats = result.diskann_stats().unwrap()[0];
            pq += stats.pq_evaluations;
            vectors += stats.vector_evaluations;
            calls += stats.read_calls;
            read_bytes += stats.read_bytes;
            if target == n {
                assert_eq!(unique.len(), n);
            }
        }
        latencies.sort_unstable();
        println!(
            "range,{target},{nq},{hits},{:.6},{:.3},{pq},{vectors},{calls},{read_bytes}",
            matched as f64 / expected_count as f64,
            latencies[nq * 95 / 100] as f64 / 1000.0
        );
    }
    measure_topk(&mut reader, true)?;
    Ok(())
}
