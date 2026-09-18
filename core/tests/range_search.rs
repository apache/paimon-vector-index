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

//! Cross-family semantics for distance range search.
//!
//! Band and cut unit tests live in `core/src/range.rs`; this file covers the
//! reader-level contract. Every expected value here is a **relation** — an
//! oracle, an equivalence between two runtime paths, or a set identity — never a
//! frozen f32 bit pattern. Development happens on darwin/aarch64 while upstream
//! CI runs ubuntu x86_64, and `fvec_l2sqr` dispatches to NEON versus AVX2 with
//! different lane grouping, so a hardcoded bit pattern captured locally would
//! fail upstream.

use paimon_vindex_core::distance::{fvec_l2sqr, MetricType};
use paimon_vindex_core::index::{VectorIndexReader, VectorSearchParams};
use paimon_vindex_core::io::{
    write_index, IVFPQIndexReader, PosWriter, ReadRequest, SeekRead, SeekReadCapabilities,
};
use paimon_vindex_core::ivfflat::IVFFlatIndex;
use paimon_vindex_core::ivfflat_io::write_ivfflat_index;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use paimon_vindex_core::ivfrq::IVFRQIndex;
use paimon_vindex_core::ivfrq_io::write_ivfrq_index;
use paimon_vindex_core::ivfsq::IVFSQIndex;
use paimon_vindex_core::ivfsq_io::{write_ivfsq_index, IVFSQIndexReader};
use paimon_vindex_core::range::{
    Bound, DistanceBand, QueryResult, RangeSearchResult, VectorRangeSearchParams,
};
use paimon_vindex_core::read_options::VectorIndexReaderOptions;
use paimon_vindex_core::rq::RQRotation;
use paimon_vindex_core::sq::ScalarQuantizer;
use std::collections::HashSet;
use std::io::{self, Cursor};
use std::sync::{Arc, Mutex};

use roaring::RoaringTreemap;

type Reader = VectorIndexReader<Cursor<Vec<u8>>>;

fn pq_fixture(bits: usize, residual: bool, opq: bool) -> IVFPQIndex {
    let mut index = IVFPQIndex::with_nbits(8, 4, 4, bits, MetricType::L2, opq);
    index.by_residual = residual;
    index.set_quantizer_centroids(
        (0..index.nlist)
            .flat_map(|list| (0..index.d).map(move |dim| (list * 4 + dim) as f32 / 4.0))
            .collect(),
    );
    index.pq.set_centroids(
        (0..index.pq.m())
            .flat_map(|sub| {
                (0..index.pq.ksub()).flat_map(move |code| {
                    (0..2).map(move |dim| ((code * 7 + sub * 3 + dim) % 31) as f32 / 16.0)
                })
            })
            .collect(),
    );
    if let Some(rotation) = &mut index.opq {
        rotation.rotation = vec![0.0; index.d * index.d];
        for dim in 0..index.d {
            rotation.rotation[dim * index.d + (dim + 1) % index.d] = 1.0;
        }
        rotation.is_trained = true;
    }
    for list in 0..3 {
        for row in 0..273 {
            index.ids[list].push(1000 + (list * 273 + row) as i64);
            let codes = (0..index.pq.m())
                .map(|sub| ((row * (sub + 1) + list + sub * 5) % index.pq.ksub()) as u8)
                .collect::<Vec<_>>();
            if bits == 4 {
                index.codes[list].extend(codes.chunks_exact(2).map(|pair| pair[0] | pair[1] << 4));
            } else {
                index.codes[list].extend(codes);
            }
        }
    }
    index
}

fn pq_dense_opq_fixture(bits: usize, residual: bool) -> IVFPQIndex {
    let dimension = 64;
    let rows_per_list = 1027;
    let mut index = IVFPQIndex::with_nbits(dimension, 3, 8, bits, MetricType::L2, true);
    index.by_residual = residual;
    index.set_quantizer_centroids(
        (0..index.nlist)
            .flat_map(|list| {
                (0..dimension).map(move |coordinate| {
                    (list as f32 - 1.0) * 1.5 + ((coordinate * 7) % 17) as f32 / 29.0
                })
            })
            .collect(),
    );
    index.pq.set_centroids(
        (0..index.pq.m())
            .flat_map(|sub| {
                (0..index.pq.ksub()).flat_map(move |code| {
                    (0..8).map(move |coordinate| {
                        ((code * 13 + sub * 7 + coordinate * 11) % 97) as f32 / 23.0 - 2.0
                    })
                })
            })
            .collect(),
    );
    let rotation = index.opq.as_mut().unwrap();
    rotation.rotation = (0..dimension)
        .flat_map(|row| {
            (0..dimension).map(move |column| {
                if (row & column).count_ones() % 2 == 0 {
                    0.125
                } else {
                    -0.125
                }
            })
        })
        .collect();
    rotation.is_trained = true;
    for left in rotation.rotation.chunks_exact(dimension) {
        assert!(left.iter().all(|value| value.abs() == 0.125));
    }
    for (row, left) in rotation.rotation.chunks_exact(dimension).enumerate() {
        for (column, right) in rotation.rotation.chunks_exact(dimension).enumerate() {
            let dot = left
                .iter()
                .zip(right)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            assert_eq!(dot, if row == column { 1.0 } else { 0.0 });
        }
    }
    for list in 0..index.nlist {
        for row in 0..rows_per_list {
            index.ids[list].push(10_000 + (list * rows_per_list + row) as i64);
            let codes = (0..index.pq.m())
                .map(|sub| {
                    ((row * (2 * sub + 1) + row / 16 + list * 17 + sub * 11) % index.pq.ksub())
                        as u8
                })
                .collect::<Vec<_>>();
            if bits == 4 {
                index.codes[list].extend(codes.chunks_exact(2).map(|pair| pair[0] | pair[1] << 4));
            } else {
                index.codes[list].extend(codes);
            }
        }
    }
    index
}

fn pq_coarse_batch_fixture() -> IVFPQIndex {
    let fixture = pq_dense_opq_fixture(8, true);
    let mut index = IVFPQIndex::with_nbits(64, 128, 8, 8, MetricType::L2, true);
    let mut centroids = fixture.quantizer_centroids().to_vec();
    centroids.resize(index.nlist * index.d, 1000.0);
    index.set_quantizer_centroids(centroids);
    index.pq = fixture.pq;
    index.opq = fixture.opq;
    for list in 0..fixture.nlist {
        index.ids[list] = fixture.ids[list][..7].to_vec();
        index.codes[list] = fixture.codes[list][..7 * index.pq.code_size()].to_vec();
    }
    index
}

fn pq_bytes(index: &IVFPQIndex) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_index(index, &mut PosWriter::new(&mut bytes)).unwrap();
    bytes
}

fn pq_reader(index: &IVFPQIndex) -> Reader {
    VectorIndexReader::open(Cursor::new(pq_bytes(index))).unwrap()
}

fn pq_oracle(index: &IVFPQIndex, query: &[f32], nprobe: usize) -> Vec<(i64, f32)> {
    let mut rotated = query.to_vec();
    if let Some(rotation) = &index.opq {
        for (dim, value) in rotated.iter_mut().enumerate() {
            *value = query
                .iter()
                .enumerate()
                .map(|(column, value)| value * rotation.rotation[dim * index.d + column])
                .sum();
        }
    }
    let mut lists = (0..index.nlist)
        .map(|list| {
            let distance = rotated
                .iter()
                .zip(&index.quantizer_centroids()[list * index.d..(list + 1) * index.d])
                .map(|(value, centroid)| (value - centroid).powi(2))
                .sum::<f32>();
            (distance, list)
        })
        .collect::<Vec<_>>();
    lists.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let mut rows = Vec::new();
    for &(_, list) in lists.iter().take(nprobe) {
        for (row, &id) in index.ids[list].iter().enumerate() {
            let code =
                &index.codes[list][row * index.pq.code_size()..(row + 1) * index.pq.code_size()];
            let mut distance = 0.0;
            for sub in 0..index.pq.m() {
                let label = if index.pq.nbits() == 4 {
                    (code[sub / 2] >> (4 * (sub % 2))) & 15
                } else {
                    code[sub]
                } as usize;
                let mut term = 0.0;
                for dim in 0..index.pq.dsub() {
                    let coordinate = sub * index.pq.dsub() + dim;
                    let coarse = if index.by_residual {
                        index.quantizer_centroids()[list * index.d + coordinate]
                    } else {
                        0.0
                    };
                    let centroid = index.pq.centroids()
                        [(sub * index.pq.ksub() + label) * index.pq.dsub() + dim];
                    term += (rotated[coordinate] - coarse - centroid).powi(2);
                }
                distance += term;
            }
            rows.push((id, distance));
        }
    }
    rows.sort_by_key(|&(id, _)| id);
    rows
}

fn pq_rows(query: QueryResult<'_>) -> Vec<(i64, f32)> {
    let mut rows = query
        .labels
        .iter()
        .copied()
        .zip(query.distances.iter().copied())
        .collect::<Vec<_>>();
    rows.sort_by_key(|&(id, _)| id);
    rows
}

fn pq_all_params(nprobe: usize) -> VectorRangeSearchParams {
    VectorRangeSearchParams::new(
        DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap(),
        nprobe,
    )
}

fn pq_entry_points(
    reader: &mut Reader,
    query: &[f32],
    params: VectorRangeSearchParams,
    filter: &[u8],
) -> [io::Result<RangeSearchResult>; 4] {
    [
        reader.range_search(query, params),
        reader.range_search_with_roaring_filter(query, params, filter),
        reader.range_search_batch(query, 1, params),
        reader.range_search_batch_with_roaring_filter(query, 1, params, filter),
    ]
}

#[test]
fn pq_range_matches_decoded_oracle_for_both_code_widths() {
    for bits in [4, 8] {
        for residual in [false, true] {
            for opq in [false, true] {
                let index = pq_fixture(bits, residual, opq);
                let query = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
                for nprobe in [1, 2, 4, 20] {
                    let expected = pq_oracle(&index, &query, nprobe);
                    let band = l2(0.5, 30.0);
                    let result = pq_reader(&index)
                        .range_search(&query, VectorRangeSearchParams::new(band, nprobe))
                        .unwrap();
                    assert_eq!(
                        pq_rows(result.query(0)),
                        expected
                            .into_iter()
                            .filter(|&(_, value)| band.admit(value))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn pq_range_dense_opq_parallel_batches_preserve_float_adc_results() {
    let pools = [1, 4].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    });
    let query_count = 16;
    for bits in [4, 8] {
        for residual in [false, true] {
            let index = pq_dense_opq_fixture(bits, residual);
            assert!(index.pq.dsub() >= 8);
            let dimension = index.d;
            let queries = (0..query_count)
                .flat_map(|query| {
                    (0..dimension).map(move |coordinate| {
                        if coordinate == 0 {
                            (query % 3) as f32 * 12.0 - 12.0
                        } else {
                            ((query * 19 + coordinate * 11 + query * coordinate) % 101) as f32
                                / 31.0
                                - 1.5
                        }
                    })
                })
                .collect::<Vec<_>>();
            let allowed = index
                .ids
                .iter()
                .flatten()
                .copied()
                .filter(|id| id % 2 == 0)
                .collect::<HashSet<_>>();
            assert!(index.ids.iter().all(|ids| {
                ids.iter().filter(|id| allowed.contains(id)).count() * query_count > 8192
            }));
            let filter = serialize_roaring(&allowed);
            let params = pq_all_params(index.nlist);
            let bytes = pq_bytes(&index);
            let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
            let reference = pools[0].install(|| {
                queries
                    .chunks_exact(dimension)
                    .map(|query| pq_rows(reader.range_search(query, params).unwrap().query(0)))
                    .collect::<Vec<_>>()
            });
            for (query_index, query) in queries.chunks_exact(dimension).enumerate() {
                let expected = pq_oracle(&index, query, index.nlist);
                assert_eq!(reference[query_index].len(), expected.len());
                for (&(id, distance), &(expected_id, expected_distance)) in
                    reference[query_index].iter().zip(&expected)
                {
                    assert_eq!(id, expected_id);
                    assert!(
                        (distance - expected_distance).abs()
                            <= 1e-5 * expected_distance.abs().max(1.0),
                        "PQ{bits}, residual={residual}, query={query_index}, id={id}: \
                         SIMD distance {distance}, scalar oracle {expected_distance}"
                    );
                }
            }
            let mut distances = reference[0]
                .iter()
                .map(|&(_, value)| value)
                .collect::<Vec<_>>();
            distances.sort_by(f32::total_cmp);
            let cut = distances[distances.len() / 2];
            assert!(distances[0] < cut && cut < *distances.last().unwrap());
            let bands = [
                params.band(),
                DistanceBand::new(Bound::Unbounded, Bound::Finite(cut), MetricType::L2).unwrap(),
                DistanceBand::new(Bound::Finite(cut), Bound::Unbounded, MetricType::L2).unwrap(),
                l2(cut, cut.next_up()),
                l2(cut.next_down(), cut),
            ];
            for pool in &pools {
                pool.install(|| {
                    let mut reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
                    for optimized in [false, true] {
                        if optimized {
                            reader.optimize_for_search().unwrap();
                        }
                        for band in bands {
                            let band_params = VectorRangeSearchParams::new(band, index.nlist);
                            let batch = reader
                                .range_search_batch(&queries, query_count, band_params)
                                .unwrap();
                            let filtered = reader
                                .range_search_batch_with_roaring_filter(
                                    &queries,
                                    query_count,
                                    band_params,
                                    &filter,
                                )
                                .unwrap();
                            assert_eq!(batch.query_count(), query_count);
                            assert_eq!(filtered.query_count(), query_count);
                            assert_eq!(batch.call_stats().list_reads(), index.nlist);
                            assert_eq!(filtered.call_stats().list_reads(), index.nlist);
                            for (query_index, query) in queries.chunks_exact(dimension).enumerate()
                            {
                                let expected = reference[query_index]
                                    .iter()
                                    .copied()
                                    .filter(|&(_, value)| band.admit(value))
                                    .collect::<Vec<_>>();
                                let expected_filtered = expected
                                    .iter()
                                    .copied()
                                    .filter(|(id, _)| allowed.contains(id))
                                    .collect::<Vec<_>>();
                                assert_eq!(pq_rows(batch.query(query_index)), expected);
                                assert_eq!(pq_rows(filtered.query(query_index)), expected_filtered);
                                for (result, scanned, committed) in [
                                    (&batch, reference[query_index].len(), expected.len()),
                                    (&filtered, allowed.len(), expected_filtered.len()),
                                ] {
                                    let stats = result.query(query_index).stats;
                                    assert_eq!(stats.rows_scanned(), scanned);
                                    assert_eq!(stats.rows_committed(), committed);
                                    assert_eq!(stats.lists_probed(), index.nlist);
                                    assert_eq!(stats.early_abandoned(), 0);
                                }
                                if band == params.band() || query_index == 0 {
                                    let single = reader.range_search(query, band_params).unwrap();
                                    let single_filtered = reader
                                        .range_search_with_filter(
                                            query,
                                            band_params,
                                            Some(&allowed),
                                        )
                                        .unwrap();
                                    assert_eq!(pq_rows(single.query(0)), expected);
                                    assert_eq!(
                                        pq_rows(single_filtered.query(0)),
                                        expected_filtered
                                    );
                                }
                            }
                        }
                    }
                });
            }
        }
    }
}

#[test]
fn pq_range_coarse_parallel_partial_probes_preserve_query_order() {
    let index = pq_coarse_batch_fixture();
    let query_count = 16;
    let queries = (0..query_count)
        .flat_map(|query_index| {
            (0..index.d).map(move |coordinate| {
                if coordinate == 0 {
                    (query_index % 3) as f32 * 12.0 - 12.0
                } else {
                    (query_index * 17 + coordinate) as f32 / 64.0 - 2.0
                }
            })
        })
        .collect::<Vec<_>>();
    let reversed = queries
        .chunks_exact(index.d)
        .rev()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let params = pq_all_params(2);
    let snapshots = [1, 4].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut reader = pq_reader(&index);
                let batch = reader
                    .range_search_batch(&queries, query_count, params)
                    .unwrap();
                let reordered = reader
                    .range_search_batch(&reversed, query_count, params)
                    .unwrap();
                assert_eq!(batch.query_count(), query_count);
                assert_eq!(reordered.query_count(), query_count);
                let rows = (0..query_count)
                    .map(|query_index| pq_rows(batch.query(query_index)))
                    .collect::<Vec<_>>();
                for (query_index, query) in queries.chunks_exact(index.d).enumerate() {
                    let single = reader.range_search(query, params).unwrap();
                    assert_eq!(rows[query_index], pq_rows(single.query(0)));
                    assert_eq!(
                        rows[query_index],
                        pq_rows(reordered.query(query_count - 1 - query_index))
                    );
                    assert_eq!(batch.query(query_index).stats.lists_probed(), 2);
                    let expected = pq_oracle(&index, query, 2);
                    assert_eq!(rows[query_index].len(), 14);
                    assert_eq!(rows[query_index].len(), expected.len());
                    for (&(id, distance), &(expected_id, expected_distance)) in
                        rows[query_index].iter().zip(&expected)
                    {
                        assert_eq!(id, expected_id);
                        assert!(
                            (distance - expected_distance).abs()
                                <= 1e-5 * expected_distance.abs().max(1.0)
                        );
                    }
                }
                assert!(rows.windows(2).all(|pair| pair[0] != pair[1]));
                rows
            })
    });
    assert_eq!(snapshots[0], snapshots[1]);
}

#[test]
fn pq_range_coarse_parallel_overflow_precedes_payload_io() {
    let index = pq_coarse_batch_fixture();
    let query_count = 16;
    let params = pq_all_params(1);
    let empty_filter = serialize_roaring(&HashSet::new());
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let valid_prefix = vec![0.0; (query_count - 1) * index.d];
                pq_reader(&index)
                    .range_search_batch(&valid_prefix, query_count - 1, params)
                    .unwrap();
                for (late_value, far_value, diagnostic) in [
                    (1e20, 1000.0, "query-centroid distance"),
                    (0.0, 1e20, "query-centroid distance for list 127"),
                    (f32::MAX, 1000.0, "rotated query"),
                ] {
                    let mut queries = vec![0.0; query_count * index.d];
                    queries[(query_count - 1) * index.d..].fill(late_value);
                    let trace = Arc::new(Mutex::new(SqReadTrace::default()));
                    let source = SqRecordingReader {
                        inner: Cursor::new(pq_bytes(&index)),
                        trace: Arc::clone(&trace),
                    };
                    let mut reader = IVFPQIndexReader::open(source).unwrap();
                    reader.ensure_loaded().unwrap();
                    *reader.quantizer_centroids.last_mut().unwrap() = far_value;
                    assert!(queries.iter().all(|value| value.is_finite()));
                    assert!(reader
                        .quantizer_centroids
                        .iter()
                        .all(|value| value.is_finite()));
                    *trace.lock().unwrap() = SqReadTrace::default();
                    for result in [
                        reader.range_search_batch(&queries, query_count, params),
                        reader.range_search_batch_with_roaring_filter(
                            &queries,
                            query_count,
                            params,
                            &empty_filter,
                        ),
                    ] {
                        let error = result.unwrap_err();
                        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
                        assert!(error.to_string().contains(diagnostic), "{error}");
                    }
                    assert_eq!(trace.lock().unwrap().calls, 0);
                }
            });
    }
}

#[test]
fn pq_range_boundaries_partition_the_full_estimated_result() {
    for bits in [4, 8] {
        let index = pq_fixture(bits, true, false);
        let query = [0.25; 8];
        let expected = pq_oracle(&index, &query, 4);
        let cut = expected[200].1;
        let mut reader = pq_reader(&index);
        let bands = [
            pq_all_params(4).band(),
            DistanceBand::new(Bound::Unbounded, Bound::Finite(cut), MetricType::L2).unwrap(),
            DistanceBand::new(Bound::Finite(cut), Bound::Unbounded, MetricType::L2).unwrap(),
            l2(cut, cut.next_up()),
            l2(cut.next_down(), cut),
            l2(cut, cut),
        ];
        for band in bands {
            let result = reader
                .range_search(&query, VectorRangeSearchParams::new(band, 4))
                .unwrap();
            assert_eq!(
                pq_rows(result.query(0)),
                expected
                    .iter()
                    .copied()
                    .filter(|&(_, value)| band.admit(value))
                    .collect::<Vec<_>>()
            );
            assert_eq!(result.query(0).stats.early_abandoned(), 0);
        }
        assert!(expected.iter().any(|&(_, value)| value == cut));
        assert!(expected.len() > 200);
    }
}

#[test]
fn pq_range_batch_filters_stats_and_permutations_match_single_queries() {
    for bits in [4, 8] {
        for opq in [false, true] {
            let index = pq_fixture(bits, true, opq);
            let queries = [vec![0.25; 8], vec![1.5; 8], vec![4.0; 8], vec![0.25; 8]].concat();
            let mut reader = pq_reader(&index);
            for step in [1, 2, 127, 10000] {
                let allowed = (1000..1819).step_by(step).collect::<HashSet<i64>>();
                let filter = serialize_roaring(&allowed);
                for nprobe in [1, 2, 4, 20] {
                    let params = VectorRangeSearchParams::new(l2(0.5, 20.0), nprobe);
                    let batch = reader.range_search_batch(&queries, 4, params).unwrap();
                    let filtered = reader
                        .range_search_batch_with_roaring_filter(&queries, 4, params, &filter)
                        .unwrap();
                    for (query_index, query) in queries.chunks_exact(8).enumerate() {
                        let single = reader.range_search(query, params).unwrap();
                        assert_eq!(pq_rows(batch.query(query_index)), pq_rows(single.query(0)));
                        let single_filtered = reader
                            .range_search_with_roaring_filter(query, params, &filter)
                            .unwrap();
                        let expected = pq_rows(batch.query(query_index))
                            .into_iter()
                            .filter(|&(id, _)| allowed.contains(&id))
                            .collect::<Vec<_>>();
                        assert_eq!(pq_rows(filtered.query(query_index)), expected);
                        assert_eq!(pq_rows(single_filtered.query(0)), expected);
                        let scanned = pq_oracle(&index, query, nprobe)
                            .iter()
                            .filter(|&&(id, _)| allowed.contains(&id))
                            .count();
                        let stats = filtered.query(query_index).stats;
                        assert_eq!(stats.rows_scanned(), scanned);
                        assert_eq!(stats.rows_committed(), expected.len());
                        assert_eq!(stats.lists_probed(), nprobe.min(index.nlist));
                        assert_eq!(stats.early_abandoned(), 0);
                    }
                    assert!(filtered.call_stats().list_reads() <= 3);
                    assert_eq!(
                        filtered.call_stats().list_reads(),
                        batch.call_stats().list_reads()
                    );
                    if nprobe >= 4 {
                        assert_eq!(batch.call_stats().list_reads(), 3);
                    }
                    let order = [2, 0, 3, 1];
                    let permuted_queries = order
                        .iter()
                        .flat_map(|&position| {
                            queries[position * 8..(position + 1) * 8].iter().copied()
                        })
                        .collect::<Vec<_>>();
                    let permuted = reader
                        .range_search_batch(&permuted_queries, 4, params)
                        .unwrap();
                    for (position, &original) in order.iter().enumerate() {
                        assert_eq!(
                            pq_rows(permuted.query(position)),
                            pq_rows(batch.query(original))
                        );
                    }
                }
            }
            for allowed in [HashSet::new(), HashSet::from([999999])] {
                let filter = serialize_roaring(&allowed);
                for result in pq_entry_points(&mut reader, &[0.25; 8], pq_all_params(4), &filter)
                    .into_iter()
                    .skip(1)
                    .step_by(2)
                {
                    let result = result.unwrap();
                    assert!(result.labels().is_empty());
                    assert_eq!(result.query(0).stats.rows_scanned(), 0);
                }
            }
        }
    }
}

#[test]
fn pq_range_native_entry_points_and_optimization_leave_top_k_unchanged() {
    for bits in [4, 8] {
        for residual in [false, true] {
            for opq in [false, true] {
                let index = pq_fixture(bits, residual, opq);
                let mut reader = IVFPQIndexReader::open(Cursor::new(pq_bytes(&index))).unwrap();
                let query = [0.25; 8];
                let params = pq_all_params(4);
                let expected = pq_oracle(&index, &query, 4);
                let allowed = index.ids.iter().flatten().copied().collect::<HashSet<_>>();
                let filter = serialize_roaring(&allowed);
                for optimized in [false, true] {
                    if optimized {
                        reader.optimize_for_search().unwrap();
                    }
                    let before = reader.search(&query, 25, 4).unwrap();
                    let precomputed = reader.precomputed_table.clone();
                    for result in [
                        reader.range_search(&query, params),
                        reader.range_search_with_roaring_filter(&query, params, &filter),
                        reader.range_search_batch(&query, 1, params),
                        reader.range_search_batch_with_roaring_filter(&query, 1, params, &filter),
                        reader.range_search_with_filter(&query, params, Some(&allowed)),
                        reader.range_search_batch_with_filter(&query, 1, params, Some(&allowed)),
                    ] {
                        let result = result.unwrap();
                        assert_eq!(pq_rows(result.query(0)), expected);
                        assert_eq!(result.query(0).stats.rows_scanned(), expected.len());
                        assert_eq!(result.query(0).stats.lists_probed(), 4);
                        assert_eq!(result.call_stats().list_reads(), 3);
                    }
                    assert_eq!(reader.search(&query, 25, 4).unwrap(), before);
                    assert_eq!(reader.precomputed_table, precomputed);
                }
            }
        }
    }
}

#[test]
fn pq_range_empty_bands_validate_before_skipping_metadata_io() {
    let bytes = pq_bytes(&pq_fixture(4, true, false));
    let mut reader = VectorIndexReader::open(Cursor::new(bytes[..64].to_vec())).unwrap();
    let mut native = IVFPQIndexReader::open(Cursor::new(bytes[..64].to_vec())).unwrap();
    let empty = VectorRangeSearchParams::new(l2(1.0, 1.0), 4);
    let filter = serialize_roaring(&HashSet::new());
    let query = [0.0; 8];
    let unified = pq_entry_points(&mut reader, &query, empty, &filter);
    let direct = [
        native.range_search(&query, empty),
        native.range_search_with_roaring_filter(&query, empty, &filter),
        native.range_search_batch(&query, 1, empty),
        native.range_search_batch_with_roaring_filter(&query, 1, empty, &filter),
    ];
    for result in unified.into_iter().chain(direct) {
        let result = result.unwrap();
        assert_eq!(result.lims(), &[0, 0]);
        assert_eq!(result.query(0).stats.lists_probed(), 0);
        assert_eq!(result.query(0).stats.rows_scanned(), 0);
        assert_eq!(result.call_stats().list_reads(), 0);
    }
    for bad_query in [
        vec![0.0; 7],
        vec![f32::NAN; 8],
        vec![f32::INFINITY; 8],
        vec![f32::NEG_INFINITY; 8],
    ] {
        for result in pq_entry_points(&mut reader, &bad_query, empty, &filter) {
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);
        }
        assert_eq!(
            native.range_search(&bad_query, empty).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
    for params in [empty, pq_all_params(4)] {
        assert_eq!(
            reader
                .range_search_with_roaring_filter(&query, params, b"invalid")
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            native
                .range_search_batch_with_roaring_filter(&query, 1, params, b"invalid")
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
    for count in [0, 2, usize::MAX] {
        assert_eq!(
            reader
                .range_search_batch(&query, count, empty)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            native
                .range_search_batch(&query, count, empty)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
    for result in pq_entry_points(&mut reader, &query, pq_all_params(0), &filter) {
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }
}

#[test]
fn pq_range_metric_mismatch_and_uncertified_metrics_fail_loud() {
    let filter = serialize_roaring(&HashSet::new());
    for metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
        let mut index = pq_fixture(8, false, false);
        index.metric = metric;
        let mut reader = pq_reader(&index);
        let mut native = IVFPQIndexReader::open(Cursor::new(pq_bytes(&index))).unwrap();
        for band_metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
            for nprobe in [0, 4] {
                let band =
                    DistanceBand::new(Bound::Finite(1.0), Bound::Finite(1.0), band_metric).unwrap();
                let params = VectorRangeSearchParams::new(band, nprobe);
                for result in pq_entry_points(&mut reader, &[0.0; 8], params, &filter)
                    .into_iter()
                    .chain([native.range_search(&[0.0; 8], params)])
                {
                    if nprobe == 0 || metric != band_metric {
                        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);
                    } else if metric != MetricType::L2 {
                        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Unsupported);
                    } else {
                        assert!(result.is_ok());
                    }
                }
            }
        }
    }
}

#[test]
fn pq_range_rejects_nonfinite_unselected_coarse_data_and_rotation() {
    let filter = serialize_roaring(&HashSet::new());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        for rotation in [false, true] {
            let index = pq_fixture(8, true, true);
            let mut reader = pq_reader(&index);
            let VectorIndexReader::IvfPq(native) = &mut reader else {
                unreachable!()
            };
            native.ensure_loaded().unwrap();
            if rotation {
                native.opq.as_mut().unwrap().rotation[0] = value;
            } else {
                let last = native.quantizer_centroids.len() - 1;
                native.quantizer_centroids[last] = value;
            }
            for result in pq_entry_points(&mut reader, &[2.0; 8], pq_all_params(1), &filter) {
                assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
            }
        }
    }
}

#[test]
fn pq_range_nonfinite_estimates_are_checked_only_for_eligible_rows() {
    for bits in [4, 8] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let mut index = pq_fixture(bits, false, false);
            for list in 0..index.nlist {
                index.ids[list].clear();
                index.codes[list].clear();
            }
            index.ids[0] = vec![1, 2];
            index.codes[0] = vec![0; index.pq.code_size() * 2];
            index.codes[0][index.pq.code_size()] = 1;
            let mut centroids = index.pq.centroids().to_vec();
            centroids[index.pq.dsub()] = value;
            index.pq.set_centroids(centroids);
            let mut reader = pq_reader(&index);
            for band in [pq_all_params(4).band(), l2(0.0, 0.001)] {
                let params = VectorRangeSearchParams::new(band, 4);
                assert_eq!(
                    reader.range_search(&[0.0; 8], params).unwrap_err().kind(),
                    io::ErrorKind::InvalidData
                );
                for allowed in [HashSet::new(), HashSet::from([1])] {
                    let filter = serialize_roaring(&allowed);
                    let filtered = reader
                        .range_search_with_roaring_filter(&[0.0; 8], params, &filter)
                        .unwrap();
                    assert_eq!(filtered.query(0).stats.rows_scanned(), allowed.len());
                    assert_eq!(filtered.query(0).stats.early_abandoned(), 0);
                }
                let filter = serialize_roaring(&HashSet::from([2]));
                assert_eq!(
                    reader
                        .range_search_batch_with_roaring_filter(&[0.0; 16], 2, params, &filter)
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::InvalidData
                );
            }
        }
    }
}

#[test]
fn pq_range_uses_estimated_not_original_distances() {
    for bits in [4, 8] {
        let mut index = IVFPQIndex::with_nbits(8, 1, 4, bits, MetricType::L2, false);
        index.set_quantizer_centroids(vec![0.0; 8]);
        index.pq.set_centroids(vec![0.0; index.d * index.pq.ksub()]);
        let original = [0.25; 8];
        index.add(&original, &[42], 1);
        let band = l2(0.0, 0.25);
        assert!(!band.admit(fvec_l2sqr(&original, &[0.0; 8])));
        let result = pq_reader(&index)
            .range_search(&[0.0; 8], VectorRangeSearchParams::new(band, 1))
            .unwrap();
        assert_eq!(pq_rows(result.query(0)), vec![(42, 0.0)]);
    }
}

#[test]
fn pq_range_direct_adc_avoids_norm_cancellation_and_rejects_sum_overflow() {
    for bits in [4, 8] {
        let mut index = IVFPQIndex::with_nbits(8, 1, 4, bits, MetricType::L2, false);
        index.by_residual = false;
        index.set_quantizer_centroids(vec![0.0; 8]);
        index
            .pq
            .set_centroids(vec![100_000_008.0; index.d * index.pq.ksub()]);
        index.ids[0] = vec![42];
        index.codes[0] = vec![0; index.pq.code_size()];
        let mut reader = pq_reader(&index);
        let result = reader
            .range_search(&[100_000_000.0; 8], pq_all_params(1))
            .unwrap();
        assert_eq!(pq_rows(result.query(0)), vec![(42, 512.0)]);
        let value = (f32::MAX / 4.0).sqrt();
        assert!((2.0 * value * value).is_finite());
        index
            .pq
            .set_centroids(vec![value; index.d * index.pq.ksub()]);
        assert_eq!(
            pq_reader(&index)
                .range_search(&[0.0; 8], pq_all_params(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}

struct PqStreamingSource {
    prefix: Vec<u8>,
    rows: usize,
    code_size: usize,
    tail_code: u8,
    reads: std::sync::Arc<std::sync::Mutex<Vec<(usize, usize)>>>,
}

impl SeekRead for PqStreamingSource {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        assert!(ranges.len() <= 3);
        assert!(
            ranges
                .iter()
                .map(|request| request.buf.len())
                .sum::<usize>()
                <= 64 * 1024 * 1024
        );
        for request in ranges {
            let start = request.pos as usize;
            let end = start + request.buf.len();
            if end > self.prefix.len() + self.rows * self.code_size {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            self.reads.lock().unwrap().push((start, request.buf.len()));
            request.buf.fill(0);
            if start < self.prefix.len() {
                let prefix_end = end.min(self.prefix.len());
                request.buf[..prefix_end - start].copy_from_slice(&self.prefix[start..prefix_end]);
            }
            for column in 0..self.code_size {
                let tail_position = self.prefix.len() + (column + 1) * self.rows - 1;
                if (start..end).contains(&tail_position) {
                    request.buf[tail_position - start] = self.tail_code;
                }
            }
        }
        Ok(())
    }

    fn read_capabilities(&self) -> SeekReadCapabilities {
        SeekReadCapabilities {
            max_ranges_per_pread: 3,
            ..Default::default()
        }
    }
}

fn pq_streaming_source(bits: usize, corrupt_first_code: bool) -> PqStreamingSource {
    let dimension = 256;
    let mut index = IVFPQIndex::with_nbits(dimension, 1, dimension, bits, MetricType::L2, false);
    index.by_residual = false;
    index.set_quantizer_centroids(vec![0.0; dimension]);
    index.pq.set_centroids(
        (0..dimension * index.pq.ksub())
            .map(|position| (position % index.pq.ksub()) as f32)
            .collect(),
    );
    index.ids[0] = vec![0];
    index.codes[0] = vec![0; index.pq.code_size()];
    let mut bytes = pq_bytes(&index);
    let rows = 64 * 1024 * 1024 / index.pq.code_size() + 1;
    let offset_table = 64 + dimension * 4 + dimension * index.pq.ksub() * 4;
    let list_offset = offset_table + 16;
    bytes[32..40].copy_from_slice(&(rows as i64).to_le_bytes());
    bytes[offset_table + 8..offset_table + 12].copy_from_slice(&(rows as i32).to_le_bytes());
    bytes[offset_table + 12..offset_table + 16].copy_from_slice(&(rows as i32).to_le_bytes());
    bytes.truncate(list_offset);
    bytes.extend_from_slice(&0i64.to_le_bytes());
    bytes.extend_from_slice(&(rows as i32).to_le_bytes());
    bytes.push(0);
    bytes.resize(bytes.len() + rows - 1, 1);
    if corrupt_first_code {
        let codebook = 64 + dimension * 4;
        bytes[codebook..codebook + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    }
    PqStreamingSource {
        prefix: bytes,
        rows,
        code_size: index.pq.code_size(),
        tail_code: if bits == 4 { 0x11 } else { 1 },
        reads: Default::default(),
    }
}

#[test]
fn pq_range_streams_oversized_lists_once_and_propagates_collector_errors() {
    let pools = [1, 4].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    });
    let query_count = 33;
    let selected_prefix_rows = 257;
    assert!(query_count * selected_prefix_rows > 8192);
    assert!(query_count * 256 * 256 * size_of::<f32>() > 8 * 1024 * 1024);
    let queries = (0..query_count)
        .flat_map(|query| vec![(query + 1) as f32 / 64.0; 256])
        .collect::<Vec<_>>();
    for bits in [4, 8] {
        for corrupt in [false, true] {
            let source = pq_streaming_source(bits, corrupt);
            let rows = source.rows;
            let code_start = source.prefix.len();
            let code_bytes = rows * source.code_size;
            let reads = source.reads.clone();
            let mut reader = VectorIndexReader::open(source).unwrap();
            let allowed = (0..selected_prefix_rows as i64)
                .chain(std::iter::once(rows as i64 - 1))
                .collect::<HashSet<_>>();
            let filter = serialize_roaring(&allowed);
            let mut reference_reads = None;
            for pool in &pools {
                pool.install(|| {
                    reads.lock().unwrap().clear();
                    let outcome = reader.range_search_batch_with_roaring_filter(
                        &queries,
                        query_count,
                        pq_all_params(1),
                        &filter,
                    );
                    let payload_reads = reads
                        .lock()
                        .unwrap()
                        .iter()
                        .copied()
                        .filter(|&(start, _)| start >= code_start)
                        .collect::<Vec<_>>();
                    let read_bytes = payload_reads
                        .iter()
                        .map(|&(_, length)| length)
                        .sum::<usize>();
                    if let Some(reference) = &reference_reads {
                        assert_eq!(
                            &payload_reads, reference,
                            "worker count must not change I/O"
                        );
                    } else {
                        reference_reads = Some(payload_reads);
                    }
                    if corrupt {
                        assert_eq!(outcome.unwrap_err().kind(), io::ErrorKind::InvalidData);
                        assert!(
                            read_bytes < code_bytes,
                            "the collector error must stop further chunks"
                        );
                    } else {
                        let result = outcome.unwrap();
                        assert_eq!(
                            read_bytes, code_bytes,
                            "shared list payload must be read exactly once"
                        );
                        assert_eq!(result.call_stats().list_reads(), 1);
                        assert_eq!(result.query_count(), query_count);
                        for (query_index, query) in queries.chunks_exact(256).enumerate() {
                            let mut expected = (0..selected_prefix_rows as i64)
                                .map(|id| (id, 256.0 * query[0].powi(2)))
                                .collect::<Vec<_>>();
                            expected.push((rows as i64 - 1, 256.0 * (query[0] - 1.0).powi(2)));
                            assert_eq!(pq_rows(result.query(query_index)), expected);
                            let stats = result.query(query_index).stats;
                            assert_eq!(stats.rows_scanned(), allowed.len());
                            assert_eq!(stats.rows_committed(), allowed.len());
                            assert_eq!(stats.lists_probed(), 1);
                            assert_eq!(stats.early_abandoned(), 0);
                        }
                    }
                });
            }
        }
    }
}

#[test]
fn pq_range_empty_lists_and_shared_probe_reads_have_exact_counts() {
    let index = pq_fixture(4, true, false);
    let queries = index.quantizer_centroids().to_vec();
    let result = pq_reader(&index)
        .range_search_batch(&queries, 4, pq_all_params(1))
        .unwrap();
    assert_eq!(result.call_stats().list_reads(), 3);
    for query in 0..4 {
        let expected = if query < 3 { 273 } else { 0 };
        assert_eq!(result.query(query).stats.rows_scanned(), expected);
        assert_eq!(result.query(query).stats.lists_probed(), 1);
        assert_eq!(result.query(query).labels.len(), expected);
    }
    let mut empty = pq_fixture(8, true, false);
    for list in 0..empty.nlist {
        empty.ids[list].clear();
        empty.codes[list].clear();
    }
    let result = pq_reader(&empty)
        .range_search_batch(&queries, 4, pq_all_params(4))
        .unwrap();
    assert_eq!(result.call_stats().list_reads(), 0);
    assert_eq!(result.lims(), &[0, 0, 0, 0, 0]);
    for query in 0..4 {
        assert_eq!(result.query(query).stats.rows_scanned(), 0);
        assert_eq!(result.query(query).stats.lists_probed(), 4);
    }
}

fn rq_fixture(dimension: usize, bits: usize, nlist: usize, per_list: usize) -> IVFRQIndex {
    let mut index = IVFRQIndex::with_bits(dimension, nlist, bits, MetricType::L2);
    index.set_quantizer_centroids(
        (0..nlist)
            .flat_map(|list| (0..dimension).map(move |dim| list as f32 * 4.0 + dim as f32 * 0.01))
            .collect(),
    );
    let mut state = 7181u64;
    let mut vectors = Vec::new();
    for list in 0..nlist {
        for _ in 0..per_list {
            for dim in 0..dimension {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0;
                vectors.push(index.quantizer_centroids()[list * dimension + dim] + noise);
            }
        }
    }
    let ids = (0..nlist * per_list)
        .map(|row| 1000 + (nlist * per_list - row) as i64)
        .collect::<Vec<_>>();
    index.add(&vectors, &ids, ids.len());
    assert!(index.ids.iter().all(|ids| ids.len() == per_list));
    index
}

fn rq_bytes(index: &IVFRQIndex) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_ivfrq_index(index, &mut PosWriter::new(&mut bytes)).unwrap();
    bytes
}

fn rq_reader(index: &IVFRQIndex) -> Reader {
    VectorIndexReader::open(Cursor::new(rq_bytes(index))).unwrap()
}

fn rq_estimated_oracle(index: &IVFRQIndex, query: &[f32], nprobe: usize) -> Vec<(i64, f32)> {
    let rotation = RQRotation::new(index.d, index.rotation_seed, index.rotation_rounds);
    let mut rotated = vec![0.0; index.padded_d];
    rotation.rotate(query, &mut rotated, &mut vec![0.0; index.padded_d]);
    let query_sum = rotated.iter().sum::<f32>();
    let mut lists = (0..index.nlist)
        .map(|list| {
            let distance = fvec_l2sqr(
                query,
                &index.quantizer_centroids()[list * index.d..(list + 1) * index.d],
            );
            (list, distance)
        })
        .collect::<Vec<_>>();
    lists.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let mut rows = Vec::new();
    for (list, coarse_distance) in lists.into_iter().take(nprobe) {
        for (position, &id) in index.ids[list].iter().enumerate() {
            let code = &index.codes[list]
                [position * index.code_size()..(position + 1) * index.code_size()];
            let byte_sum = |plane: usize, byte: usize| {
                let pattern = code[plane * index.plane_size() + byte];
                (0..8)
                    .rev()
                    .filter(|bit| pattern & (1 << bit) != 0)
                    .fold(0.0, |sum, bit| sum + rotated[byte * 8 + bit])
            };
            let mut unsigned = (0..index.plane_size())
                .map(|byte| byte_sum(0, byte))
                .sum::<f32>()
                * (1usize << (index.bits - 1)) as f32;
            for plane in 1..index.bits {
                let weight = (1usize << (index.bits - 1 - plane)) as f32;
                for byte in 0..index.plane_size() {
                    unsigned += weight * byte_sum(plane, byte);
                }
            }
            let center = ((1usize << index.bits) - 1) as f32 * 0.5;
            let factors = index.factors[list][position].full;
            let estimate = factors.f_add
                + coarse_distance
                + factors.f_rescale * (unsigned - center * query_sum);
            rows.push((id, estimate));
        }
    }
    rows
}

#[test]
fn rq_range_matches_estimated_distance_oracle() {
    for bits in 1..=8 {
        for dimension in [13, 256] {
            let index = rq_fixture(dimension, bits, 4, 37);
            let query = (0..dimension)
                .map(|dim| dim as f32 * 0.01 + 0.2)
                .collect::<Vec<_>>();
            for nprobe in [1, 3, 4] {
                let oracle = rq_estimated_oracle(&index, &query, nprobe);
                let mut values = oracle.iter().map(|row| row.1).collect::<Vec<_>>();
                values.sort_by(f32::total_cmp);
                let lower = values[values.len() / 4].max(0.0);
                let upper = values[values.len() * 3 / 4].max(lower);
                for band in [
                    l2(lower, upper),
                    DistanceBand::new(Bound::Unbounded, Bound::Finite(upper), MetricType::L2)
                        .unwrap(),
                    DistanceBand::new(Bound::Finite(lower), Bound::Unbounded, MetricType::L2)
                        .unwrap(),
                    DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap(),
                ] {
                    let result = rq_reader(&index)
                        .range_search(&query, VectorRangeSearchParams::new(band, nprobe))
                        .unwrap();
                    let expected = oracle
                        .iter()
                        .copied()
                        .filter(|row| band.admit(row.1))
                        .collect();
                    assert_eq!(
                        pairs_of(result.query(0)),
                        bits_of(expected),
                        "bits={bits}, dimension={dimension}"
                    );
                    assert_eq!(result.query(0).stats.rows_scanned(), nprobe * 37);
                    assert_eq!(result.query(0).stats.early_abandoned(), 0);
                    assert_eq!(
                        result.query(0).stats.rows_committed(),
                        result.labels().len()
                    );
                    assert_eq!(result.call_stats().list_reads(), nprobe);
                }
            }
        }
    }
}

fn rq_all_distances() -> DistanceBand {
    DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap()
}

#[test]
fn rq_range_bounded_probes_match_oracle_across_many_lists() {
    let index = rq_fixture(13, 4, 65, 3);
    let queries = [0.2, 126.0, 256.2]
        .into_iter()
        .flat_map(|base| (0..index.d).map(move |dimension| base + dimension as f32 * 0.01))
        .collect::<Vec<_>>();
    let allowed = index
        .ids
        .iter()
        .flatten()
        .copied()
        .filter(|row| row % 2 == 0)
        .collect::<HashSet<_>>();
    let filter = serialize_roaring(&allowed);
    for width in [1, 3, 16, 33, 65, usize::MAX] {
        let nprobe = width.min(index.nlist);
        for band in [rq_all_distances(), l2(0.0, 256.0)] {
            let params = VectorRangeSearchParams::new(band, width);
            let mut reader = rq_reader(&index);
            let batch = reader.range_search_batch(&queries, 3, params).unwrap();
            let filtered = reader
                .range_search_batch_with_roaring_filter(&queries, 3, params, &filter)
                .unwrap();
            for (query_index, query) in queries.chunks_exact(index.d).enumerate() {
                let expected = rq_estimated_oracle(&index, query, nprobe)
                    .into_iter()
                    .filter(|row| band.admit(row.1))
                    .collect::<Vec<_>>();
                assert_eq!(
                    pairs_of(batch.query(query_index)),
                    bits_of(expected.clone())
                );
                let single = reader.range_search(query, params).unwrap();
                assert_eq!(pairs_of(single.query(0)), bits_of(expected.clone()));
                assert_eq!(
                    pairs_of(filtered.query(query_index)),
                    bits_of(
                        expected
                            .into_iter()
                            .filter(|row| allowed.contains(&row.0))
                            .collect()
                    )
                );
                assert_eq!(batch.query(query_index).stats.rows_scanned(), nprobe * 3);
            }
        }
    }
}

#[test]
fn rq_range_batch_single_filters_and_statistics_agree() {
    for bits in [1, 4, 8] {
        let index = rq_fixture(256, bits, 4, 37);
        let queries = [0.2, 4.2, 2.0, 8.7, 0.2]
            .into_iter()
            .flat_map(|offset| (0..index.d).map(move |dim| dim as f32 * 0.01 + offset))
            .collect::<Vec<_>>();
        let all_ids = index.ids.iter().flatten().copied().collect::<HashSet<_>>();
        let sparse_ids = all_ids
            .iter()
            .copied()
            .filter(|id| id % 3 == 0)
            .collect::<HashSet<_>>();
        for nprobe in [1, 2, 9] {
            let params = VectorRangeSearchParams::new(l2(10.0, 2000.0), nprobe);
            let mut reader = rq_reader(&index);
            let unfiltered = reader.range_search_batch(&queries, 5, params).unwrap();
            for allowed in [&all_ids, &sparse_ids, &HashSet::new()] {
                let filter = serialize_roaring(allowed);
                let batch = reader
                    .range_search_batch_with_roaring_filter(&queries, 5, params, &filter)
                    .unwrap();
                let mut probed_lists = HashSet::new();
                for (query_index, query) in queries.chunks_exact(index.d).enumerate() {
                    let oracle = rq_estimated_oracle(&index, query, nprobe);
                    for (list, ids) in index.ids.iter().enumerate() {
                        if oracle.iter().any(|row| ids.contains(&row.0)) {
                            probed_lists.insert(list);
                        }
                    }
                    let single = reader
                        .range_search_with_roaring_filter(query, params, &filter)
                        .unwrap();
                    let unfiltered_single = reader.range_search(query, params).unwrap();
                    assert_eq!(
                        pairs_of(single.query(0)),
                        pairs_of(batch.query(query_index))
                    );
                    assert_eq!(
                        pairs_of(unfiltered_single.query(0)),
                        pairs_of(unfiltered.query(query_index))
                    );
                    let expected = oracle
                        .iter()
                        .copied()
                        .filter(|row| allowed.contains(&row.0) && params.band().admit(row.1))
                        .collect();
                    assert_eq!(pairs_of(batch.query(query_index)), bits_of(expected));
                    let stats = batch.query(query_index).stats;
                    assert_eq!(stats.lists_probed(), nprobe.min(index.nlist));
                    assert_eq!(
                        stats.rows_scanned(),
                        oracle.iter().filter(|row| allowed.contains(&row.0)).count()
                    );
                    assert_eq!(
                        stats.rows_committed(),
                        batch.query(query_index).labels.len()
                    );
                    assert_eq!(stats.early_abandoned(), 0);
                    assert_eq!(
                        batch.lims()[query_index + 1] - batch.lims()[query_index],
                        stats.rows_committed()
                    );
                    if allowed == &all_ids {
                        assert_eq!(
                            pairs_of(batch.query(query_index)),
                            pairs_of(unfiltered.query(query_index))
                        );
                    }
                }
                assert_eq!(batch.call_stats().list_reads(), probed_lists.len());
            }
        }
    }
}

#[test]
fn rq_range_membership_is_estimated_not_exact() {
    let dimension = 13;
    let mut index = IVFRQIndex::with_bits(dimension, 1, 1, MetricType::L2);
    index.set_quantizer_centroids(vec![0.0; dimension]);
    let vectors = (0..96 * dimension)
        .map(|value| ((value * 37 % 113) as f32 - 56.0) / 23.0)
        .collect::<Vec<_>>();
    let ids = (0..96).collect::<Vec<i64>>();
    index.add(&vectors, &ids, ids.len());
    let query = vec![0.3; dimension];
    let oracle = rq_estimated_oracle(&index, &query, 1);
    let (witness, estimated, exact) = oracle
        .iter()
        .find_map(|&(id, estimate)| {
            let exact = fvec_l2sqr(
                &query,
                &vectors[id as usize * dimension..(id as usize + 1) * dimension],
            );
            (estimate > 0.0 && (estimate - exact).abs() > 0.01).then_some((id, estimate, exact))
        })
        .expect("fixture must distinguish estimated and exact distances");
    let band = l2(0.0, (estimated + exact) * 0.5);
    let result = rq_reader(&index)
        .range_search(&query, VectorRangeSearchParams::new(band, 1))
        .unwrap();
    assert_eq!(result.labels().contains(&witness), band.admit(estimated));
    assert_ne!(result.labels().contains(&witness), band.admit(exact));
    assert_eq!(
        pairs_of(result.query(0)),
        bits_of(oracle.into_iter().filter(|row| band.admit(row.1)).collect())
    );
}

#[test]
fn rq_range_does_not_prune_on_coarse_bounds_or_clamp_estimates() {
    for bits in [1, 4, 8] {
        let mut index = rq_fixture(13, bits, 1, 37);
        let negative_id = index.ids[0][0];
        for (row, factors) in index.factors[0].iter_mut().enumerate() {
            factors.coarse.f_add = 1000.0;
            factors.coarse.f_rescale = 0.0;
            factors.coarse.f_error = 0.0;
            factors.full.f_add = if row == 0 { -1.0 } else { 0.5 };
            factors.full.f_rescale = 0.0;
            if bits == 1 {
                factors.coarse = factors.full;
            }
        }
        let query = index.quantizer_centroids().to_vec();
        let mut reader = rq_reader(&index);
        let all = reader
            .range_search(&query, VectorRangeSearchParams::new(rq_all_distances(), 1))
            .unwrap();
        assert_eq!(all.labels().len(), 37);
        assert!(pairs_of(all.query(0)).contains(&(negative_id, (-1.0f32).to_bits())));
        let finite = reader
            .range_search(&query, VectorRangeSearchParams::new(l2(0.0, 1.0), 1))
            .unwrap();
        assert_eq!(finite.labels().len(), 36);
        assert!(!finite.labels().contains(&negative_id));
        assert_eq!(finite.query(0).stats.early_abandoned(), 0);
    }
}

#[test]
fn rq_range_preserves_topk_results_and_last_search_stats() {
    for bits in [1, 4] {
        let index = rq_fixture(256, bits, 4, 65);
        let query = vec![0.3; index.d];
        let filter = serialize_roaring(
            &index
                .ids
                .iter()
                .flatten()
                .copied()
                .filter(|id| id % 2 == 0)
                .collect(),
        );
        let mut reader = rq_reader(&index);
        for filtered in [false, true] {
            let params = VectorSearchParams::new(7, 3);
            let before = if filtered {
                reader
                    .search_with_roaring_filter(&query, params, &filter)
                    .unwrap()
            } else {
                reader.search(&query, params).unwrap()
            };
            let stats = reader.ivfrq_search_stats();
            let range = reader
                .range_search(&query, VectorRangeSearchParams::new(rq_all_distances(), 3))
                .unwrap();
            assert!(range.labels().len() > 7);
            assert_eq!(reader.ivfrq_search_stats(), stats);
            let after = if filtered {
                reader
                    .search_with_roaring_filter(&query, params, &filter)
                    .unwrap()
            } else {
                reader.search(&query, params).unwrap()
            };
            assert_eq!(before, after);
            assert_eq!(reader.ivfrq_search_stats(), stats);
        }
    }
}

#[test]
fn rq_range_nonfinite_factors_fail_only_when_consumed() {
    use std::io::ErrorKind;
    for bits in [1, 4] {
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for field in [0, 1] {
                let mut index = rq_fixture(13, bits, 2, 37);
                let bad_id = index.ids[0][36];
                let factors = if bits == 1 {
                    &mut index.factors[0][36].coarse
                } else {
                    &mut index.factors[0][36].full
                };
                if field == 0 {
                    factors.f_add = nonfinite;
                } else {
                    factors.f_rescale = nonfinite;
                }
                let query = vec![0.2; index.d];
                let params = VectorRangeSearchParams::new(l2(0.0, 0.01), 2);
                let mut reader = rq_reader(&index);
                assert_eq!(
                    reader.range_search(&query, params).unwrap_err().kind(),
                    ErrorKind::InvalidData
                );
                assert_eq!(
                    reader
                        .range_search_batch(&query.repeat(2), 2, params)
                        .unwrap_err()
                        .kind(),
                    ErrorKind::InvalidData
                );
                let only_bad = serialize_roaring(&HashSet::from([bad_id]));
                assert_eq!(
                    reader
                        .range_search_with_roaring_filter(&query, params, &only_bad)
                        .unwrap_err()
                        .kind(),
                    ErrorKind::InvalidData
                );
                assert_eq!(
                    reader
                        .range_search_batch_with_roaring_filter(
                            &query.repeat(2),
                            2,
                            params,
                            &only_bad
                        )
                        .unwrap_err()
                        .kind(),
                    ErrorKind::InvalidData
                );
                let only_good = serialize_roaring(
                    &index
                        .ids
                        .iter()
                        .flatten()
                        .copied()
                        .filter(|&id| id != bad_id)
                        .collect(),
                );
                let result = reader
                    .range_search_with_roaring_filter(&query, params, &only_good)
                    .unwrap();
                assert_eq!(result.query(0).stats.rows_scanned(), 73);
            }
        }
    }
    let mut index = rq_fixture(13, 4, 1, 37);
    for factors in &mut index.factors[0] {
        factors.coarse.f_add = f32::NAN;
        factors.coarse.f_rescale = f32::NEG_INFINITY;
        factors.coarse.f_error = f32::INFINITY;
    }
    let query = vec![0.2; index.d];
    let result = rq_reader(&index)
        .range_search(&query, VectorRangeSearchParams::new(rq_all_distances(), 1))
        .unwrap();
    assert_eq!(
        pairs_of(result.query(0)),
        bits_of(rq_estimated_oracle(&index, &query, 1))
    );
}

#[test]
fn rq_range_rejects_nonfinite_centroids_and_arithmetic_overflow() {
    use paimon_vindex_core::ivfrq_io::IVF_RQ_HEADER_SIZE;
    use std::io::ErrorKind;
    let index = rq_fixture(13, 4, 2, 37);
    let params = VectorRangeSearchParams::new(rq_all_distances(), 1);
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bytes = rq_bytes(&index);
        let offset = IVF_RQ_HEADER_SIZE + index.d * 4;
        bytes[offset..offset + 4].copy_from_slice(&nonfinite.to_le_bytes());
        let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();
        assert_eq!(
            reader
                .range_search(&vec![0.0; index.d], params)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
    }
    assert_eq!(
        rq_reader(&index)
            .range_search(&vec![f32::MAX; index.d], params)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );
    assert_eq!(
        rq_reader(&index)
            .range_search(&vec![1e20; index.d], params)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );
    let mut index = rq_fixture(13, 4, 1, 37);
    for factors in &mut index.factors[0] {
        factors.full.f_add = f32::MAX;
        factors.full.f_rescale = f32::MAX;
    }
    let mut reader = rq_reader(&index);
    assert_eq!(
        reader
            .range_search(&vec![1.0; index.d], params)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );
}

#[test]
fn rq_range_rejects_overflow_in_an_unselected_centroid() {
    use paimon_vindex_core::ivfrq_io::IVF_RQ_HEADER_SIZE;
    use std::io::ErrorKind;

    for nlist in [2, 65] {
        let index = rq_fixture(13, 4, nlist, 37);
        let mut bytes = rq_bytes(&index);
        let offset = IVF_RQ_HEADER_SIZE + (nlist - 1) * index.d * 4;
        bytes[offset..offset + 4].copy_from_slice(&1e20f32.to_le_bytes());
        let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();
        let params = VectorRangeSearchParams::new(rq_all_distances(), 1);
        let query = vec![0.0; index.d];
        let queries = query.repeat(2);
        let filter = serialize_roaring(&index.ids[0].iter().copied().collect());
        for error in [
            reader.range_search(&query, params).unwrap_err(),
            reader.range_search_batch(&queries, 2, params).unwrap_err(),
            reader
                .range_search_with_roaring_filter(&query, params, &filter)
                .unwrap_err(),
            reader
                .range_search_batch_with_roaring_filter(&queries, 2, params, &filter)
                .unwrap_err(),
        ] {
            assert_eq!(error.kind(), ErrorKind::InvalidData);
        }
    }
}

#[test]
fn rq_range_filters_preserve_signed_ids_and_fixed_probe_width() {
    let mut index = rq_fixture(13, 4, 2, 37);
    index.ids[0][0] = i64::MIN;
    index.ids[0][1] = -1;
    index.ids[0][2] = 1i64 << 33;
    index.ids[0][3] = i64::MAX;
    let mut filter = RoaringTreemap::new();
    for id in [i64::MIN as u64, u64::MAX, 1u64 << 33, i64::MAX as u64] {
        filter.insert(id);
    }
    let mut bytes = Vec::new();
    filter.serialize_into(&mut bytes).unwrap();
    let query = vec![0.2; index.d];
    let params = VectorRangeSearchParams::new(rq_all_distances(), 1);
    let mut reader = rq_reader(&index);
    let all = reader.range_search(&query, params).unwrap();
    assert!(all.labels().contains(&i64::MIN));
    assert!(all.labels().contains(&-1));
    let filtered = reader
        .range_search_with_roaring_filter(&query, params, &bytes)
        .unwrap();
    assert_eq!(
        filtered.labels().iter().copied().collect::<HashSet<_>>(),
        HashSet::from([1i64 << 33, i64::MAX])
    );
    let far_list = serialize_roaring(&index.ids[1].iter().copied().collect());
    let far = reader
        .range_search_with_roaring_filter(&query, params, &far_list)
        .unwrap();
    assert!(far.labels().is_empty());
    assert_eq!(far.query(0).stats.lists_probed(), 1);
    assert_eq!(far.query(0).stats.rows_scanned(), 0);
    assert_eq!(far.call_stats().list_reads(), 1);
}

#[test]
fn rq_range_unified_metric_capability_and_validation_precedence() {
    use std::io::ErrorKind;
    for metric in [MetricType::Cosine, MetricType::InnerProduct] {
        let mut index = IVFRQIndex::with_bits(13, 1, 4, metric);
        index.set_quantizer_centroids(vec![0.0; 13]);
        index.add(&[1.0; 13], &[1], 1);
        let mut reader = rq_reader(&index);
        let filter = serialize_roaring(&HashSet::from([1]));
        for band in [
            DistanceBand::new(Bound::Unbounded, Bound::Unbounded, metric).unwrap(),
            DistanceBand::new(Bound::Finite(1.0), Bound::Finite(1.0), metric).unwrap(),
        ] {
            let params = VectorRangeSearchParams::new(band, 1);
            for error in [
                reader.range_search(&[1.0; 13], params).unwrap_err(),
                reader
                    .range_search_batch(&[1.0; 26], 2, params)
                    .unwrap_err(),
                reader
                    .range_search_with_roaring_filter(&[1.0; 13], params, &filter)
                    .unwrap_err(),
                reader
                    .range_search_batch_with_roaring_filter(&[1.0; 26], 2, params, &filter)
                    .unwrap_err(),
            ] {
                assert_eq!(error.kind(), ErrorKind::Unsupported);
            }
            assert_eq!(
                reader
                    .range_search(&[f32::NAN; 13], params)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidInput
            );
            assert_eq!(
                reader
                    .range_search_with_roaring_filter(&[1.0; 13], params, &[255])
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidInput
            );
            assert_eq!(
                reader
                    .range_search(&[1.0; 13], VectorRangeSearchParams::new(band, 0))
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidInput
            );
        }
    }
}

// --- fixtures --------------------------------------------------------------
//
// Indexes are built by assigning rows to lists **explicitly** rather than by
// calling `train`, for two reasons. First, k-means could leave a list empty,
// which would make the `list_reads` assertions fail intermittently. Second, the
// band constants below only assert something if the intra-cluster distance
// distribution actually straddles them, and explicit construction is what makes
// that distribution controllable.
//
// The noise amplitude is scaled by `1/sqrt(d)` so the intra-cluster squared
// distance is **independent of d**: a sum of `d` terms each of order `A^2/d`
// stays at order `A^2`. That is what lets one set of band constants stay
// meaningful across the d = 8, d = 16 and d = 256 fixtures.

/// Per-dimension centroid spacing. Inter-cluster squared distance is about
/// `d * SPACING^2`, which dwarfs every band used here, so rows in other lists
/// are always out of band and therefore exercise the early-abandon path.
const SPACING: f32 = 10.0;
/// Chosen so the mean intra-cluster squared distance is about 2.0 and the spread
/// covers roughly 0..6, which straddles the 2.0 and 3.0 cuts the tests use.
const NOISE: f32 = 1.73;
/// Generator seed. A hex constant used as a **construction input**, never as an
/// expected value.
const SEED: u64 = 0x5eed_1234;

pub const ASYMMETRIC_NLIST: usize = 4;
const ASYMMETRIC_D: usize = 16;
const ASYMMETRIC_BIG: usize = 200;
const ASYMMETRIC_SMALL: usize = 20;
/// Tighter than [`NOISE`] so that the narrow `[0, 0.25)` band used by the batch
/// statistics test captures a large fraction of each cluster; the hit counts then
/// track the cluster sizes and differ by about ten times.
const ASYMMETRIC_NOISE: f32 = 0.474;

/// A fixed-seed linear congruential generator, so the corpus does not depend on
/// the `rand` version.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform in `[-1, 1)`.
    fn next_symmetric(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn centroid_component(list_id: usize, dim: usize) -> f32 {
    list_id as f32 * SPACING + dim as f32 * 0.01
}

fn serialize(index: &IVFFlatIndex) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_ivfflat_index(index, &mut PosWriter::new(&mut bytes)).unwrap();
    bytes
}

/// Builds an IVF-Flat reader with `n` rows spread evenly over `nlist` lists.
///
/// Returns the reader plus the corpus in id order, so an oracle can rescore it.
pub fn build_flat_fixture(n: usize, d: usize, nlist: usize) -> (Reader, Vec<f32>, Vec<i64>, usize) {
    build_flat_fixture_with_metric(n, d, nlist, MetricType::L2)
}

pub fn build_flat_fixture_with_metric(
    n: usize,
    d: usize,
    nlist: usize,
    metric: MetricType,
) -> (Reader, Vec<f32>, Vec<i64>, usize) {
    assert_eq!(n % nlist, 0, "the fixture spreads rows evenly over lists");
    let rows_per_list = n / nlist;
    let mut rng = Lcg::new(SEED);
    let mut index = IVFFlatIndex::new(d, nlist, metric);
    index.set_quantizer_centroids(
        (0..nlist)
            .flat_map(|list_id| (0..d).map(move |dim| centroid_component(list_id, dim)))
            .collect(),
    );

    // Corpus positions interleave the lists, so consecutive positions land in
    // different clusters and a batch taken from the front spans several of them.
    //
    // Ids are assigned **consecutively within each list**, deliberately not
    // derived from the interleaved position. With `id = 1000 + row * nlist +
    // list_id` and an even `nlist`, every id in a list would share one parity, so
    // an `id % 2` filter would keep or drop whole lists at a time and a
    // per-query filter comparison could have nothing left to compare. Consecutive
    // ids inside a list give both parities and every residue mod 3.
    let mut vectors_by_id = vec![0.0f32; n * d];
    let mut ids_by_id = vec![0i64; n];
    let mut next_id = 1000i64;
    for list_id in 0..nlist {
        let mut list_ids = Vec::with_capacity(rows_per_list);
        let mut list_vectors = Vec::with_capacity(rows_per_list * d);
        for row in 0..rows_per_list {
            let global = row * nlist + list_id;
            let id = next_id;
            next_id += 1;
            list_ids.push(id);
            for dim in 0..d {
                let value = centroid_component(list_id, dim)
                    + rng.next_symmetric() * NOISE / (d as f32).sqrt();
                list_vectors.push(value);
                vectors_by_id[global * d + dim] = value;
            }
            ids_by_id[global] = id;
        }
        index.ids[list_id] = list_ids;
        index.vectors[list_id] = list_vectors;
    }

    let reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();
    (reader, vectors_by_id, ids_by_id, d)
}

/// Two clusters whose sizes differ by an order of magnitude, with one query
/// sitting at the centre of each.
///
/// The balanced fixture cannot serve the per-query statistics test: with equal,
/// translation-symmetric clusters, two queries probing every list produce
/// identical `rows_scanned` and `lists_probed`, so an inequality assertion would
/// fail on a *correct* implementation.
pub fn build_asymmetric_fixture() -> (Reader, Vec<f32>, usize) {
    let mut rng = Lcg::new(SEED);
    let d = ASYMMETRIC_D;
    let mut index = IVFFlatIndex::new(d, ASYMMETRIC_NLIST, MetricType::L2);
    index.set_quantizer_centroids(
        (0..ASYMMETRIC_NLIST)
            .flat_map(|list_id| (0..d).map(move |dim| centroid_component(list_id, dim)))
            .collect(),
    );

    let mut next_id = 1000i64;
    for list_id in 0..ASYMMETRIC_NLIST {
        let rows = if list_id == 0 {
            ASYMMETRIC_BIG
        } else {
            ASYMMETRIC_SMALL
        };
        let mut list_ids = Vec::with_capacity(rows);
        let mut list_vectors = Vec::with_capacity(rows * d);
        for _ in 0..rows {
            list_ids.push(next_id);
            next_id += 1;
            for dim in 0..d {
                list_vectors.push(
                    centroid_component(list_id, dim)
                        + rng.next_symmetric() * ASYMMETRIC_NOISE / (d as f32).sqrt(),
                );
            }
        }
        index.ids[list_id] = list_ids;
        index.vectors[list_id] = list_vectors;
    }

    // Query 0 sits at the centre of the large cluster, query 1 at the centre of
    // a small one, so a narrow band around each yields hit counts tracking the
    // cluster sizes.
    let mut queries = Vec::with_capacity(2 * d);
    for list_id in [0usize, 1] {
        for dim in 0..d {
            queries.push(centroid_component(list_id, dim));
        }
    }

    let reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();
    (reader, queries, d)
}

/// The asymmetric corpus in id order, for oracle rescoring.
pub fn asymmetric_corpus() -> (Vec<f32>, Vec<i64>) {
    let mut rng = Lcg::new(SEED);
    let d = ASYMMETRIC_D;
    let mut vectors = Vec::new();
    let mut ids = Vec::new();
    let mut next_id = 1000i64;
    for list_id in 0..ASYMMETRIC_NLIST {
        let rows = if list_id == 0 {
            ASYMMETRIC_BIG
        } else {
            ASYMMETRIC_SMALL
        };
        for _ in 0..rows {
            ids.push(next_id);
            next_id += 1;
            for dim in 0..d {
                vectors.push(
                    centroid_component(list_id, dim)
                        + rng.next_symmetric() * ASYMMETRIC_NOISE / (d as f32).sqrt(),
                );
            }
        }
    }
    (vectors, ids)
}

/// A minimal DiskANN index, built only so that it can be opened; range search
/// fails at family dispatch before touching any data.
pub fn build_diskann_fixture() -> Reader {
    use paimon_vindex_core::diskann::{DiskAnnBuildParams, DiskAnnIndex};
    use paimon_vindex_core::diskann_io::write_diskann_index;

    let d = 8;
    let n = 32;
    let mut rng = Lcg::new(SEED);
    let data: Vec<f32> = (0..n * d).map(|_| rng.next_symmetric()).collect();
    let ids: Vec<i64> = (0..n as i64).collect();
    let mut index = DiskAnnIndex::new(d, MetricType::L2, 4, DiskAnnBuildParams::default());
    index.train(&data, n).unwrap();
    index.add(&data, &ids);
    let mut bytes = Vec::new();
    write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
    VectorIndexReader::open(Cursor::new(bytes)).unwrap()
}

// --- helpers ---------------------------------------------------------------

/// The rows a band truly contains, scored with the same crate's distance kernel
/// so the oracle is safe across platforms, and filtered by `band.admit` so that
/// a locally rewritten comparison cannot disagree with the reader at a cut.
pub fn brute_force_band(
    query: &[f32],
    vectors: &[f32],
    ids: &[i64],
    d: usize,
    band: DistanceBand,
) -> Vec<(i64, f32)> {
    ids.iter()
        .enumerate()
        .filter_map(|(row, &id)| {
            let distance = fvec_l2sqr(query, &vectors[row * d..(row + 1) * d]);
            band.admit(distance).then_some((id, distance))
        })
        .collect()
}

/// One query's result as a sorted multiset of `(label, distance bits)`.
///
/// Comparing bits rather than f32 makes NaN and ±0 compare deterministically.
/// Ordering is not part of the contract, hence the sort. Comparing labels alone
/// would miss two real defects: a batch merge attaching a row's distance to
/// another label, and a distance that is simply computed wrongly.
pub fn pairs_of(result: QueryResult<'_>) -> Vec<(i64, u32)> {
    let mut pairs: Vec<(i64, u32)> = result
        .labels
        .iter()
        .zip(result.distances)
        .map(|(id, dist)| (*id, dist.to_bits()))
        .collect();
    pairs.sort_unstable();
    pairs
}

fn bits_of(rows: Vec<(i64, f32)>) -> Vec<(i64, u32)> {
    let mut pairs: Vec<(i64, u32)> = rows
        .into_iter()
        .map(|(id, dist)| (id, dist.to_bits()))
        .collect();
    pairs.sort_unstable();
    pairs
}

fn l2(lower: f32, upper: f32) -> DistanceBand {
    DistanceBand::new(Bound::Finite(lower), Bound::Finite(upper), MetricType::L2).unwrap()
}

/// Serializes an allow-list the way the reader's Roaring decoder expects it.
fn serialize_roaring(allowed: &HashSet<i64>) -> Vec<u8> {
    let mut map = RoaringTreemap::new();
    for &id in allowed {
        map.insert(u64::try_from(id).expect("fixture ids are non-negative"));
    }
    let mut bytes = Vec::new();
    map.serialize_into(&mut bytes).unwrap();
    bytes
}

fn build_sq_index(dimension: usize, rows_per_list: usize, nlist: usize) -> IVFSQIndex {
    let mut index = IVFSQIndex::new(dimension, nlist, MetricType::L2);
    index.set_quantizer_centroids(
        (0..nlist)
            .flat_map(|list_id| {
                (0..dimension).map(move |component| list_id as f32 + component as f32 * 0.01)
            })
            .collect(),
    );
    index.sq = ScalarQuantizer::with_bounds(dimension, -10.0, 10.0);
    for list_id in 0..nlist {
        index.list_sqs[list_id] = ScalarQuantizer::with_dimension_bounds(
            dimension,
            (0..dimension)
                .map(|component| -0.7 - component as f32 * 0.001)
                .collect(),
            vec![1.3 + list_id as f32 * 0.1; dimension],
        );
        for row in (0..rows_per_list).rev() {
            index.ids[list_id].push((1i64 << 33) + (list_id * rows_per_list + row) as i64);
            index.codes[list_id]
                .extend((0..dimension).map(|component| ((row * 17 + component * 13) % 256) as u8));
        }
    }
    index
}

fn serialize_sq(index: &IVFSQIndex) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_ivfsq_index(index, &mut PosWriter::new(&mut bytes)).unwrap();
    bytes
}

#[test]
fn ivf_sq_range_matches_sq_estimates_for_all_entry_points() {
    let dimension = 65;
    let nlist = 4;
    let rows_per_list = 67;
    let index = build_sq_index(dimension, rows_per_list, nlist);
    let queries: Vec<f32> = [0.0, 1.5, 3.0]
        .into_iter()
        .flat_map(|offset| (0..dimension).map(move |component| offset + component as f32 * 0.01))
        .collect();
    let mut reader = VectorIndexReader::open(Cursor::new(serialize_sq(&index))).unwrap();
    let allowed: HashSet<i64> = index
        .ids
        .iter()
        .flatten()
        .copied()
        .filter(|id| id % 3 == 0)
        .collect();
    let filter = serialize_roaring(&allowed);
    for nprobe in [1, 2, nlist, nlist + 10] {
        let band = l2(10.0, 180.0);
        let params = VectorRangeSearchParams::new(band, nprobe);
        let batch = reader.range_search_batch(&queries, 3, params).unwrap();
        let filtered_batch = reader
            .range_search_batch_with_roaring_filter(&queries, 3, params, &filter)
            .unwrap();
        for (query_index, query) in queries.chunks_exact(dimension).enumerate() {
            let (ids, distances) = top_k(&mut reader, query, rows_per_list * nlist, nprobe);
            let expected = bits_of(
                ids.into_iter()
                    .zip(distances)
                    .filter(|(id, distance)| *id != -1 && band.admit(*distance))
                    .collect(),
            );
            assert!(!expected.is_empty());
            assert_eq!(pairs_of(batch.query(query_index)), expected);
            assert_eq!(
                pairs_of(reader.range_search(query, params).unwrap().query(0)),
                expected
            );
            let filtered: Vec<_> = expected
                .into_iter()
                .filter(|(id, _)| allowed.contains(id))
                .collect();
            assert!(!filtered.is_empty());
            assert_eq!(pairs_of(filtered_batch.query(query_index)), filtered);
            assert_eq!(
                pairs_of(
                    reader
                        .range_search_with_roaring_filter(query, params, &filter)
                        .unwrap()
                        .query(0)
                ),
                filtered,
            );
        }
    }
}

#[test]
fn ivf_sq_range_uses_estimated_not_original_distance() {
    let mut index = IVFSQIndex::new(1, 1, MetricType::L2);
    index.set_quantizer_centroids(vec![0.0]);
    index.sq = ScalarQuantizer::with_bounds(1, 0.0, 255.0);
    index.list_sqs[0] = index.sq.clone();
    let vectors = [0.49, 0.51];
    let ids = [10, 20];
    index.add(&vectors, &ids, 2);
    let mut reader = VectorIndexReader::open(Cursor::new(serialize_sq(&index))).unwrap();
    for band in [l2(0.0, 0.1), l2(0.2, 0.3)] {
        let result = reader
            .range_search(&[0.0], VectorRangeSearchParams::new(band, 1))
            .unwrap();
        let (labels, distances) = top_k(&mut reader, &[0.0], 2, 1);
        let expected = bits_of(
            labels
                .into_iter()
                .zip(distances)
                .filter(|(_, distance)| band.admit(*distance))
                .collect(),
        );
        assert_eq!(pairs_of(result.query(0)), expected);
        assert_ne!(
            pairs_of(result.query(0)),
            bits_of(brute_force_band(&[0.0], &vectors, &ids, 1, band))
        );
    }
}

#[test]
fn ivf_sq_range_preserves_boundaries_and_has_no_top_k_cap() {
    let index = build_sq_index(65, 67, 3);
    let mut reader = VectorIndexReader::open(Cursor::new(serialize_sq(&index))).unwrap();
    let query = vec![0.3; 65];
    let (ids, distances) = top_k(&mut reader, &query, 201, 3);
    let lower = distances[20];
    let upper = distances[80];
    assert!(lower < upper);
    for band in [
        l2(lower, upper),
        l2(lower, lower),
        l2(0.0, 0.0),
        DistanceBand::new(Bound::Unbounded, Bound::Finite(upper), MetricType::L2).unwrap(),
        DistanceBand::new(Bound::Finite(lower), Bound::Unbounded, MetricType::L2).unwrap(),
        DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap(),
    ] {
        let result = reader
            .range_search(&query, VectorRangeSearchParams::new(band, 10))
            .unwrap();
        let expected = bits_of(
            ids.iter()
                .copied()
                .zip(distances.iter().copied())
                .filter(|(_, distance)| band.admit(*distance))
                .collect(),
        );
        assert_eq!(pairs_of(result.query(0)), expected);
        assert_eq!(result.query(0).stats.rows_committed(), expected.len());
        if band.is_empty() {
            assert_eq!(result.query(0).stats.lists_probed(), 0);
            assert_eq!(result.query(0).stats.rows_scanned(), 0);
            assert_eq!(result.call_stats().list_reads(), 0);
        } else {
            assert_eq!(result.query(0).stats.lists_probed(), 3);
            assert_eq!(result.query(0).stats.rows_scanned(), 201);
        }
    }
}

#[test]
fn ivf_sq_range_parallel_scans_preserve_query_order_and_membership() {
    let index = build_sq_index(65, 2_101, 4);
    let bytes = serialize_sq(&index);
    let queries: Vec<f32> = [0.0, 0.5, 2.0]
        .into_iter()
        .flat_map(|value| vec![value; 65])
        .collect();
    let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();
    let whole = DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    let all = reader
        .range_search_batch(&queries, 3, VectorRangeSearchParams::new(whole, 4))
        .unwrap();
    let band = l2(15.0, 180.0);
    let params = VectorRangeSearchParams::new(band, 4);
    let expected: Vec<_> = (0..3)
        .map(|query_index| {
            bits_of(
                all.query(query_index)
                    .labels
                    .iter()
                    .copied()
                    .zip(all.query(query_index).distances.iter().copied())
                    .filter(|(_, distance)| band.admit(*distance))
                    .collect(),
            )
        })
        .collect();
    let mut permuted = queries[130..].to_vec();
    permuted.extend_from_slice(&queries[..130]);
    let allowed: HashSet<_> = index
        .ids
        .iter()
        .flatten()
        .copied()
        .filter(|id| id % 5 == 0)
        .collect();
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let result = reader.range_search_batch(&permuted, 3, params).unwrap();
                let filtered = reader
                    .range_search_batch_with_roaring_filter(
                        &queries,
                        3,
                        params,
                        &serialize_roaring(&allowed),
                    )
                    .unwrap();
                for (query_index, &original_index) in [2, 0, 1].iter().enumerate() {
                    assert!(!expected[original_index].is_empty());
                    assert_eq!(
                        pairs_of(result.query(query_index)),
                        expected[original_index]
                    );
                    let single = reader
                        .range_search(
                            &queries[original_index * 65..(original_index + 1) * 65],
                            params,
                        )
                        .unwrap();
                    assert_eq!(pairs_of(single.query(0)), expected[original_index]);
                    assert_eq!(
                        pairs_of(filtered.query(original_index)),
                        expected[original_index]
                            .iter()
                            .copied()
                            .filter(|(id, _)| allowed.contains(id))
                            .collect::<Vec<_>>()
                    );
                }
            });
    }
}

#[derive(Default)]
struct SqReadTrace {
    calls: usize,
    ranges: usize,
    max_bytes: usize,
}

struct SqRecordingReader {
    inner: Cursor<Vec<u8>>,
    trace: Arc<Mutex<SqReadTrace>>,
}

impl SeekRead for SqRecordingReader {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> std::io::Result<()> {
        let mut trace = self.trace.lock().unwrap();
        trace.calls += 1;
        trace.ranges += ranges.len();
        for request in ranges.iter() {
            trace.max_bytes = trace.max_bytes.max(request.buf.len());
        }
        self.inner.pread(ranges)
    }
}

#[test]
fn ivf_sq_range_reuses_shared_lists_and_cache_with_query_local_filters() {
    let mut index = build_sq_index(33, 67, 4);
    index.ids[3].clear();
    index.codes[3].clear();
    let bytes = serialize_sq(&index);
    let whole = DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    let params = VectorRangeSearchParams::new(whole, 4);
    for budget in [0, 4 * 1024 * 1024] {
        let trace = Arc::new(Mutex::new(SqReadTrace::default()));
        let source = SqRecordingReader {
            inner: Cursor::new(bytes.clone()),
            trace: Arc::clone(&trace),
        };
        let mut reader =
            IVFSQIndexReader::open_with_options(source, VectorIndexReaderOptions::new(budget))
                .unwrap();
        *trace.lock().unwrap() = SqReadTrace::default();
        let queries = vec![0.0; 3 * 33];
        let batch = reader.range_search_batch(&queries, 3, params).unwrap();
        assert_eq!(batch.call_stats().list_reads(), 3);
        assert_eq!(trace.lock().unwrap().calls, 1);
        assert_eq!(trace.lock().unwrap().ranges, 3);
        assert_eq!(batch.lims(), &[0, 201, 402, 603]);
        let empty_filter = serialize_roaring(&HashSet::new());
        let filtered = reader
            .range_search_batch_with_roaring_filter(&queries, 3, params, &empty_filter)
            .unwrap();
        assert!(filtered.labels().is_empty());
        let repeated = reader.range_search(&queries[..33], params).unwrap();
        assert_eq!(pairs_of(repeated.query(0)), pairs_of(batch.query(0)));
        if budget > 0 {
            assert_eq!(trace.lock().unwrap().calls, 1);
            assert_eq!(filtered.call_stats().list_reads(), 0);
            assert_eq!(repeated.call_stats().list_reads(), 0);
        } else {
            assert_eq!(trace.lock().unwrap().calls, 3);
            assert_eq!(repeated.call_stats().list_reads(), 3);
        }
    }
}

#[test]
fn ivf_sq_range_respects_bounded_reads_and_deduplicates_partial_probes() {
    struct SingleRangeReader(SqRecordingReader);

    impl SeekRead for SingleRangeReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> std::io::Result<()> {
            assert!(ranges.len() <= 1);
            self.0.pread(ranges)
        }

        fn read_capabilities(&self) -> paimon_vindex_core::io::SeekReadCapabilities {
            paimon_vindex_core::io::SeekReadCapabilities {
                max_ranges_per_pread: 1,
                ..Default::default()
            }
        }
    }

    let index = build_sq_index(33, 67, 4);
    let mut queries = index.quantizer_centroids()[..66].to_vec();
    queries.extend_from_slice(&index.quantizer_centroids()[..33]);
    let trace = Arc::new(Mutex::new(SqReadTrace::default()));
    let source = SingleRangeReader(SqRecordingReader {
        inner: Cursor::new(serialize_sq(&index)),
        trace: Arc::clone(&trace),
    });
    let mut reader = IVFSQIndexReader::open(source).unwrap();
    let band = DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    for (nprobe, reads, rows) in [(1, 2, 67), (4, 4, 268)] {
        *trace.lock().unwrap() = SqReadTrace::default();
        let result = reader
            .range_search_batch(&queries, 3, VectorRangeSearchParams::new(band, nprobe))
            .unwrap();
        assert_eq!(result.call_stats().list_reads(), reads);
        assert_eq!(trace.lock().unwrap().calls, reads);
        for query_index in 0..3 {
            assert_eq!(result.query(query_index).stats.lists_probed(), nprobe);
            assert_eq!(result.query(query_index).stats.rows_scanned(), rows);
            assert_eq!(result.query(query_index).labels.len(), rows);
        }
        assert_eq!(pairs_of(result.query(0)), pairs_of(result.query(2)));
    }
}

#[test]
fn ivf_sq_range_streams_oversized_lists_for_single_batch_and_filter() {
    let dimension = 512;
    let count = 131_073;
    let mut index = IVFSQIndex::new(dimension, 1, MetricType::L2);
    index.set_quantizer_centroids(vec![0.0; dimension]);
    index.sq = ScalarQuantizer::with_bounds(dimension, 0.0, 1.0);
    index.list_sqs[0] = index.sq.clone();
    index.ids[0] = (0..count as i64).collect();
    index.codes[0] = vec![255; count * dimension];
    index.codes[0][..32 * dimension].fill(0);
    index.codes[0][32 * dimension..64 * dimension].fill(64);
    index.codes[0][(count - 1) * dimension..].fill(0);
    let trace = Arc::new(Mutex::new(SqReadTrace::default()));
    let source = SqRecordingReader {
        inner: Cursor::new(serialize_sq(&index)),
        trace: Arc::clone(&trace),
    };
    drop(index);
    let mut reader = IVFSQIndexReader::open(source).unwrap();
    let mut queries = vec![0.0; dimension];
    queries.extend(vec![0.25; dimension]);
    let params = VectorRangeSearchParams::new(l2(0.0, 1.0), 1);
    let mut check_streamed_queries = || {
        *trace.lock().unwrap() = SqReadTrace::default();
        let result = reader.range_search_batch(&queries, 2, params).unwrap();
        assert_eq!(result.call_stats().list_reads(), 1);
        assert!(trace.lock().unwrap().calls > 2);
        assert!(trace.lock().unwrap().max_bytes < count * dimension);
        assert!(trace.lock().unwrap().max_bytes <= 64 * 1024 * 1024);
        let mut first_ids: Vec<_> = (0..32).collect();
        first_ids.push((count - 1) as i64);
        assert_eq!(result.query(0).labels, first_ids);
        assert_eq!(result.query(1).labels, (32..64).collect::<Vec<i64>>());
        let allowed: HashSet<i64> = (0..count as i64).filter(|id| id % 2 == 0).collect();
        let filter = serialize_roaring(&allowed);
        let filtered = reader
            .range_search_batch_with_roaring_filter(&queries, 2, params, &filter)
            .unwrap();
        for (query_index, query) in queries.chunks_exact(dimension).enumerate() {
            assert_eq!(result.query(query_index).stats.rows_scanned(), count);
            assert!(result.query(query_index).stats.early_abandoned() > count - 100);
            let single = reader.range_search(query, params).unwrap();
            assert_eq!(
                pairs_of(single.query(0)),
                pairs_of(result.query(query_index))
            );
            let expected: Vec<_> = pairs_of(result.query(query_index))
                .into_iter()
                .filter(|(id, _)| allowed.contains(id))
                .collect();
            assert_eq!(pairs_of(filtered.query(query_index)), expected);
            assert_eq!(
                pairs_of(
                    reader
                        .range_search_with_roaring_filter(query, params, &filter)
                        .unwrap()
                        .query(0)
                ),
                expected
            );
        }
    };
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(&mut check_streamed_queries);
    }
}

#[test]
fn ivf_sq_range_batch_validates_inputs_before_empty_band_shortcuts() {
    use std::io::ErrorKind::{InvalidInput, Unsupported};

    for metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
        let mut index = build_sq_index(33, 35, 2);
        index.metric = metric;
        let mut reader = IVFSQIndexReader::open(Cursor::new(serialize_sq(&index))).unwrap();
        let empty = DistanceBand::new(Bound::Finite(1.0), Bound::Finite(1.0), metric).unwrap();
        let mismatched_metric = if metric == MetricType::L2 {
            MetricType::Cosine
        } else {
            MetricType::L2
        };
        let mismatched =
            DistanceBand::new(Bound::Finite(1.0), Bound::Finite(1.0), mismatched_metric).unwrap();
        let query = [0.0; 33];
        for (case, queries, query_count, band, nprobe) in [
            ("dimension", &query[..32], 1, empty, 2),
            ("NaN", &[f32::NAN; 33], 1, empty, 2),
            ("infinity", &[f32::INFINITY; 33], 1, empty, 2),
            ("nprobe", &query, 1, empty, 0),
            ("metric", &query, 1, mismatched, 2),
            ("zero queries", &[], 0, empty, 2),
            ("query count overflow", &query, usize::MAX, empty, 2),
        ] {
            let error = reader
                .range_search_batch(
                    queries,
                    query_count,
                    VectorRangeSearchParams::new(band, nprobe),
                )
                .unwrap_err();
            assert_eq!(error.kind(), InvalidInput, "{case}, metric={metric:?}");
        }
        let params = VectorRangeSearchParams::new(empty, 2);
        assert_eq!(
            reader
                .range_search_batch_with_roaring_filter(&query, 1, params, &[0xde, 0xad])
                .unwrap_err()
                .kind(),
            InvalidInput
        );
        let result = reader.range_search_batch(&query, 1, params);
        if metric == MetricType::L2 {
            assert!(result.unwrap().labels().is_empty());
        } else {
            assert_eq!(result.unwrap_err().kind(), Unsupported);
        }
    }
}

#[test]
fn ivf_sq_range_wrappers_delegate_input_validation() {
    let bytes = serialize_sq(&build_sq_index(33, 35, 2));
    let mut unified = VectorIndexReader::open(Cursor::new(bytes.clone())).unwrap();
    let mut direct = IVFSQIndexReader::open(Cursor::new(bytes)).unwrap();
    let query = [0.0; 32];
    let filter = serialize_roaring(&HashSet::new());
    let params = VectorRangeSearchParams::new(l2(1.0, 1.0), 2);
    for result in [
        unified.range_search(&query, params),
        unified.range_search_with_roaring_filter(&query, params, &filter),
        unified.range_search_batch(&query, 1, params),
        unified.range_search_batch_with_roaring_filter(&query, 1, params, &filter),
        direct.range_search(&query, params),
        direct.range_search_with_roaring_filter(&query, params, &filter),
        direct.range_search_batch_with_roaring_filter(&query, 1, params, &filter),
    ] {
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }
}

#[test]
fn ivf_sq_range_propagates_nonfinite_estimates_and_payload_errors() {
    let index = build_sq_index(33, 35, 2);
    let bytes = serialize_sq(&index);
    let whole = DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    let mut reader = VectorIndexReader::open(Cursor::new(bytes.clone())).unwrap();
    for band in [whole, l2(0.0, 1.0)] {
        let params = VectorRangeSearchParams::new(band, 2);
        assert_eq!(
            reader
                .range_search(&[f32::MAX; 33], params)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(
            reader
                .range_search_batch(&[f32::MAX; 66], 2, params)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }
    let mut truncated = bytes;
    truncated.truncate(truncated.len() - 10);
    let mut reader = VectorIndexReader::open(Cursor::new(truncated)).unwrap();
    assert_eq!(
        reader
            .range_search(&[0.0; 33], VectorRangeSearchParams::new(whole, 2))
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::UnexpectedEof
    );
}

// --- Task 8: the engine ----------------------------------------------------

#[test]
fn ivf_flat_range_matches_a_brute_force_oracle_at_full_probe() {
    let (mut reader, vectors, ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);
    // nprobe == nlist means full coverage, and IVF-Flat computes exact
    // distances, so the result must match the oracle row for row as a multiset.
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    let got = pairs_of(result.query(0));
    let want = bits_of(brute_force_band(&query, &vectors, &ids, d, band));
    assert!(
        !want.is_empty(),
        "the band matched nothing, so this test would be vacuous"
    );
    assert_eq!(got, want, "labels and distances must both equal the oracle");
}

#[test]
fn an_empty_band_returns_zero_rows() {
    let (mut reader, ..) = build_flat_fixture(64, 8, 4);
    let band = l2(1.0, 1.0);
    let result = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(band, 4))
        .unwrap();
    assert_eq!(result.query(0).labels.len(), 0);
    assert_eq!(
        result.call_stats().list_reads(),
        0,
        "an empty band probes no list at all"
    );
}

#[test]
fn a_band_whose_metric_disagrees_with_the_index_is_rejected() {
    // The mismatch matrix: index metric x band metric, where only equality is
    // permitted.
    for index_metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
        let (mut reader, ..) = build_flat_fixture_with_metric(128, 8, 4, index_metric);
        for band_metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
            let band =
                DistanceBand::new(Bound::Finite(0.0), Bound::Finite(1.0), band_metric).unwrap();
            let outcome = reader.range_search(&[0.1; 8], VectorRangeSearchParams::new(band, 4));
            if band_metric != index_metric {
                assert_eq!(
                    outcome.unwrap_err().kind(),
                    std::io::ErrorKind::InvalidInput,
                    "index {index_metric:?} / band {band_metric:?} must report InvalidInput"
                );
            } else if band_metric != MetricType::L2 {
                // Matching but not yet certified.
                assert_eq!(
                    outcome.unwrap_err().kind(),
                    std::io::ErrorKind::Unsupported,
                    "index {index_metric:?} / band {band_metric:?}"
                );
            } else {
                assert!(outcome.is_ok(), "L2/L2 must succeed");
            }
        }
    }
}

#[test]
fn an_empty_band_does_not_mask_a_bad_query() {
    let (mut reader, ..) = build_flat_fixture(64, 8, 4);
    let params = VectorRangeSearchParams::new(l2(1.0, 1.0), 4);
    // Wrong dimension.
    assert!(reader.range_search(&[0.0; 7], params).is_err());
    // Non-finite query.
    assert!(reader.range_search(&[f32::NAN; 8], params).is_err());
}

#[test]
fn an_unsupported_index_type_fails_loud_for_every_band() {
    // The family capability check must precede the empty-band short-circuit: a
    // family that cannot do range search rejects every band.
    let mut reader = build_diskann_fixture();
    let empty = l2(1.0, 1.0);
    let err = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(empty, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

#[test]
fn a_whole_space_band_returns_every_probed_row() {
    let (mut reader, vectors, ids, d) = build_flat_fixture(256, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    assert_eq!(
        result.query(0).labels.len(),
        ids.len(),
        "an unbounded band at full probe returns the whole corpus"
    );
    assert_eq!(
        pairs_of(result.query(0)),
        bits_of(brute_force_band(&query, &vectors, &ids, d, band))
    );
}

#[test]
fn a_narrow_band_probing_fewer_lists_returns_no_more_rows() {
    // "No cap" promises no truncation, not completeness: a smaller nprobe covers
    // less and therefore returns fewer in-band rows.
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);
    let wide = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    let narrow = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 1))
        .unwrap();
    let wide_set: HashSet<(i64, u32)> = pairs_of(wide.query(0)).into_iter().collect();
    let narrow_pairs = pairs_of(narrow.query(0));
    assert!(
        !narrow_pairs.is_empty(),
        "the narrow probe found nothing, so this test would be vacuous"
    );
    assert!(
        narrow_pairs.len() <= wide_set.len(),
        "a lower nprobe returned {} rows against the full probe's {}",
        narrow_pairs.len(),
        wide_set.len()
    );
    for pair in &narrow_pairs {
        assert!(
            wide_set.contains(pair),
            "a lower nprobe must return a subset of the full-probe result"
        );
    }
}

// --- Task 9: the filter variants -------------------------------------------

#[test]
fn a_filtered_range_equals_filtering_the_unfiltered_result() {
    let (mut reader, vectors, ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);
    let allowed: HashSet<i64> = ids.iter().copied().filter(|id| id % 3 == 0).collect();

    let unfiltered = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    let mut want: Vec<(i64, u32)> = pairs_of(unfiltered.query(0))
        .into_iter()
        .filter(|(id, _)| allowed.contains(id))
        .collect();

    let roaring = serialize_roaring(&allowed);
    let filtered = reader
        .range_search_with_roaring_filter(&query, VectorRangeSearchParams::new(band, 8), &roaring)
        .unwrap();
    let mut got: Vec<(i64, u32)> = pairs_of(filtered.query(0));
    got.sort_unstable();
    want.sort_unstable();
    assert!(
        !want.is_empty(),
        "the filter admitted nothing, so this test would be vacuous"
    );
    assert_eq!(
        got, want,
        "the filtered variant's (label, distance) must equal filtering the unfiltered result"
    );
}

#[test]
fn a_malformed_roaring_filter_is_rejected_even_for_an_empty_band() {
    let (mut reader, ..) = build_flat_fixture(64, 8, 4);
    let empty = l2(1.0, 1.0);
    let err = reader
        .range_search_with_roaring_filter(
            &[0.0; 8],
            VectorRangeSearchParams::new(empty, 4),
            b"not roaring",
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn a_non_flat_index_rejects_a_filtered_range() {
    let mut reader = build_diskann_fixture();
    let allowed: HashSet<i64> = (0..8).collect();
    let err = reader
        .range_search_with_roaring_filter(
            &[0.0; 8],
            VectorRangeSearchParams::new(l2(0.0, 1.0), 4),
            &serialize_roaring(&allowed),
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

// --- Task 10: the public batch API -----------------------------------------

#[test]
fn batch_range_equals_running_each_query_alone() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let nq = 4;
    let queries = vectors[0..nq * d].to_vec();
    let band = l2(0.0, 2.5);
    // Partial probe on purpose: at nprobe == nlist every list is scanned, so the
    // probe *selection* cannot differ and the test would pass even if batch and
    // single-query ranked centroids differently. That is precisely how a real
    // batch/single divergence went unnoticed here.
    let nprobe = 3;
    assert!(
        nprobe < 8,
        "nprobe must be below nlist for this test to bite"
    );
    let params = VectorRangeSearchParams::new(band, nprobe);

    let batch = reader.range_search_batch(&queries, nq, params).unwrap();
    for qi in 0..nq {
        let single = reader
            .range_search(&queries[qi * d..(qi + 1) * d], params)
            .unwrap();
        // Ordering is not part of the contract, so compare per-query multisets.
        assert!(
            !pairs_of(single.query(0)).is_empty(),
            "query {qi} matched nothing, so this comparison would be vacuous"
        );
        assert_eq!(
            pairs_of(batch.query(qi)),
            pairs_of(single.query(0)),
            "query {qi}'s batched (label, distance) must equal its single-query result"
        );
    }
}

#[test]
fn batch_range_is_invariant_under_query_permutation() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(256, 16, 8);
    let band = l2(0.0, 2.5);
    // Partial probe, for the same reason as above.
    let params = VectorRangeSearchParams::new(band, 3);
    let a = vectors[0..d].to_vec();
    let b = vectors[d..2 * d].to_vec();

    let ab: Vec<f32> = a.iter().chain(b.iter()).copied().collect();
    let ba: Vec<f32> = b.iter().chain(a.iter()).copied().collect();
    let r_ab = reader.range_search_batch(&ab, 2, params).unwrap();
    let r_ba = reader.range_search_batch(&ba, 2, params).unwrap();

    assert_eq!(pairs_of(r_ab.query(0)), pairs_of(r_ba.query(1)));
    assert_eq!(pairs_of(r_ab.query(1)), pairs_of(r_ba.query(0)));
}

#[test]
fn a_band_splits_into_adjacent_sub_bands_without_losing_rows() {
    // Bucket closure: range([a,c)) == range([a,b)) union range([b,c)) as multisets.
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let whole = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(0.0, 4.0), 8))
        .unwrap();
    let left = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(0.0, 2.0), 8))
        .unwrap();
    let right = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(2.0, 4.0), 8))
        .unwrap();

    // Both halves must be populated, or the identity holds trivially and this
    // test proves nothing about the cut.
    assert!(
        !left.query(0).labels.is_empty(),
        "the left sub-band is empty, so this test would be vacuous"
    );
    assert!(
        !right.query(0).labels.is_empty(),
        "the right sub-band is empty, so this test would be vacuous"
    );
    // A row sitting exactly on the split is the case the half-open boundary
    // exists for: it must appear in the right band and not the left. Rather than
    // hoping the corpus contains one, split at a distance that a row actually
    // has -- otherwise this assertion is vacuous, which is a trap an earlier
    // version of this test fell into.
    let seam = *whole
        .query(0)
        .distances
        .iter()
        .find(|d| **d > 0.0)
        .expect("the parent band must contain a row at a positive distance");
    let seam_left = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(0.0, seam), 8))
        .unwrap();
    let seam_right = reader
        .range_search(&query, VectorRangeSearchParams::new(l2(seam, 4.0), 8))
        .unwrap();
    let seam_id = whole
        .query(0)
        .labels
        .iter()
        .zip(whole.query(0).distances)
        .find(|(_, d)| **d == seam)
        .map(|(id, _)| *id)
        .expect("the seam distance came from this result");
    assert!(
        !seam_left.query(0).labels.contains(&seam_id),
        "row {seam_id} sits exactly on the split and must be excluded by the \
         right-open left band"
    );
    assert!(
        seam_right.query(0).labels.contains(&seam_id),
        "row {seam_id} sits exactly on the split and must be included by the \
         left-closed right band"
    );

    let mut want: Vec<i64> = whole.query(0).labels.to_vec();
    let mut got: Vec<i64> = left
        .query(0)
        .labels
        .iter()
        .chain(right.query(0).labels)
        .copied()
        .collect();
    want.sort_unstable();
    got.sort_unstable();
    assert_eq!(
        got, want,
        "adjacent sub-bands must union to the parent band"
    );
}

#[test]
fn a_batch_roaring_filter_equals_filtering_the_unfiltered_batch() {
    let (mut reader, vectors, ids, d) = build_flat_fixture(512, 16, 8);
    let nq = 3;
    let queries = vectors[0..nq * d].to_vec();
    let band = l2(0.0, 2.5);
    let params = VectorRangeSearchParams::new(band, 8);
    let allowed: HashSet<i64> = ids.iter().copied().filter(|id| id % 2 == 0).collect();

    let plain = reader.range_search_batch(&queries, nq, params).unwrap();
    let filtered = reader
        .range_search_batch_with_roaring_filter(&queries, nq, params, &serialize_roaring(&allowed))
        .unwrap();
    for qi in 0..nq {
        let mut want: Vec<(i64, u32)> = pairs_of(plain.query(qi))
            .into_iter()
            .filter(|(id, _)| allowed.contains(id))
            .collect();
        let mut got: Vec<(i64, u32)> = pairs_of(filtered.query(qi));
        got.sort_unstable();
        want.sort_unstable();
        assert!(!want.is_empty(), "query {qi} would be a vacuous comparison");
        assert_eq!(got, want, "query {qi}'s (label, distance)");
    }
}

#[test]
fn a_batch_with_a_mismatched_query_count_is_rejected() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(64, 8, 4);
    let band = l2(0.0, 1.0);
    // Data for three queries but four declared.
    let err = reader
        .range_search_batch(&vectors[0..3 * d], 4, VectorRangeSearchParams::new(band, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn a_malformed_batch_roaring_filter_is_rejected_even_for_an_empty_band() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(64, 8, 4);
    let empty = l2(1.0, 1.0);
    let err = reader
        .range_search_batch_with_roaring_filter(
            &vectors[0..2 * d],
            2,
            VectorRangeSearchParams::new(empty, 4),
            b"not roaring",
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn a_non_flat_index_rejects_batch_range() {
    let mut reader = build_diskann_fixture();
    let band = l2(0.0, 1.0);
    let err = reader
        .range_search_batch(&[0.0; 16], 2, VectorRangeSearchParams::new(band, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

// --- Task 11: differential assertions instead of frozen f32 bits -----------
//
// These replace two donor tests that asserted hardcoded f32 bit patterns
// captured on darwin/aarch64. Upstream CI is ubuntu x86_64, and `fvec_l2sqr`
// dispatches to AVX2 there versus NEON here with different lane grouping, so
// those assertions cannot hold on both. Each one below compares two runtime
// paths inside a single build, or a path against an oracle, which holds on any
// target.

/// Top-K through the public enum API, for the shared-rows comparison below.
fn top_k(reader: &mut Reader, query: &[f32], k: usize, nprobe: usize) -> (Vec<i64>, Vec<f32>) {
    reader
        .search(query, VectorSearchParams::new(k, nprobe))
        .unwrap()
}

#[test]
fn top_k_and_range_agree_on_the_rows_they_share() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let k = 32;
    let (labels, distances) = top_k(&mut reader, &query, k, 8);
    let upper = distances[k - 1] + 1.0;
    let band = l2(0.0, upper);
    let range = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    let in_range: HashSet<i64> = range.query(0).labels.iter().copied().collect();
    let mut compared = 0usize;
    for (label, distance) in labels.iter().zip(&distances) {
        if *distance < upper {
            assert!(
                in_range.contains(label),
                "top-K row {label} at distance {distance} is inside the band but absent \
                 from the range result"
            );
            compared += 1;
        }
    }
    assert!(
        compared > 0,
        "no top-K row fell inside the band, so this test would be vacuous"
    );
}

#[test]
fn the_parallel_arm_agrees_with_brute_force() {
    // The scale must genuinely cross the parallel threshold.
    // PARALLEL_FLAT_SCAN_MIN_COMPONENTS is 1024 * 1024, and
    // scan_components = sum(list_rows * queries_for_list) * d, so
    // 8192 rows * 256 dims * 1 query = 2_097_152, which clears it.
    const N: usize = 8192;
    const D: usize = 256;
    let (mut reader, vectors, ids, d) = build_flat_fixture(N, D, 16);
    assert_eq!(d, D);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 3.0);

    // nprobe = nlist makes scan_components cross the threshold.
    let parallel = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 16))
        .unwrap();

    // The reference is the ground truth over the **same** working set, not
    // "repeat with nprobe = 1 and union the results" -- that would be a
    // different working set and could only establish a subset relation, which a
    // miscomputing parallel arm would also satisfy. nprobe = nlist means the
    // probe set is every list, so the oracle is a full brute-force scan.
    let want = bits_of(brute_force_band(&query, &vectors, &ids, d, band));
    let got = pairs_of(parallel.query(0));
    assert!(
        !want.is_empty(),
        "the band matched nothing, so this test would be vacuous"
    );
    assert_eq!(
        got, want,
        "the parallel arm's (label, distance) must equal brute force over the same working set"
    );

    // And assert the parallel branch was actually taken.
    let components = N * D; // one query, every list
    assert!(
        components > 1024 * 1024,
        "fixture scale {components} must exceed PARALLEL_FLAT_SCAN_MIN_COMPONENTS, \
         otherwise this exercises the sequential arm"
    );
}

#[test]
fn the_l2_cutoff_does_not_change_which_rows_are_returned() {
    // Flat-L2 cutoff on versus off: widening the band to the whole space is
    // equivalent to disabling early abandon.
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let bounded = l2(0.0, 2.0);
    let unbounded = DistanceBand::new(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    let with_cutoff = reader
        .range_search(&query, VectorRangeSearchParams::new(bounded, 8))
        .unwrap();
    let without = reader
        .range_search(&query, VectorRangeSearchParams::new(unbounded, 8))
        .unwrap();
    let mut a = with_cutoff.query(0).labels.to_vec();
    let mut b: Vec<i64> = without
        .query(0)
        .labels
        .iter()
        .zip(without.query(0).distances)
        .filter(|(_, dist)| **dist < 2.0)
        .map(|(id, _)| *id)
        .collect();
    a.sort_unstable();
    b.sort_unstable();
    assert!(
        !a.is_empty(),
        "no rows matched, so this test would be vacuous"
    );
    assert_eq!(
        a, b,
        "early abandon changes only work, never which rows return"
    );
    // And prove early abandon actually happened, or this test verifies nothing.
    assert!(
        with_cutoff.query(0).stats.early_abandoned() > 0,
        "a narrow band must abandon some rows early; zero means the path was never taken"
    );
    assert_eq!(
        without.query(0).stats.early_abandoned(),
        0,
        "a whole-space band has an infinite cutoff and must not enter the early-abandon kernel"
    );
}

#[test]
fn early_abandoned_counts_exactly_the_rows_above_the_upper_cut() {
    // `early_abandoned` is a published statistic, and what it counts is decided
    // by the abandon cutoff. That cutoff is now the raw upper cut, so the set is
    // exactly stateable rather than merely bounded: the scan abandons a row when
    // its running sum passes `upper`, and a partial sum of non-negative terms
    // only grows, so the completed distance passes it too. Conversely a row
    // above `upper` fails the check on its completed sum at the latest. Hence
    // abandoned <=> distance > upper, and this asserts that equality instead of
    // the `> 0` the neighbouring tests settle for.
    //
    // The three-way split it implies is the real content: rows above the cut
    // never reach the band test, rows *on* the cut do reach it and are rejected
    // there (the interval is half-open), and rows below `lower` reach it too.
    // Only the first group is counted here.
    const NLIST: usize = 8;
    // d = 256 spans two probe strides, so a row can fail the cutoff part way
    // through its vector rather than only on the completed sum. Both exits go
    // through the same `note_abandoned` call, and nothing observable from here
    // distinguishes them, so this widens what the fixture covers rather than
    // letting the count be asserted only for the simpler exit.
    let (mut reader, vectors, ids, d) = build_flat_fixture(256, 256, NLIST);
    let query = vectors[0..d].to_vec();

    let distances: Vec<f32> = (0..ids.len())
        .map(|row| fvec_l2sqr(&query, &vectors[row * d..(row + 1) * d]))
        .collect();
    let mut sorted = distances.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("distances are finite"));

    // A lower cut above the nearest rows, so the "rejected but not abandoned"
    // group is non-empty; an upper cut mid-corpus, so the abandoned group is
    // non-empty and a row sits exactly on it.
    let lower = sorted[ids.len() / 8];
    let upper = sorted[ids.len() / 2];
    let band = DistanceBand::new(Bound::Finite(lower), Bound::Finite(upper), MetricType::L2)
        .expect("a well-ordered band");

    // nprobe = nlist, so every row is probed and the expected counts are over
    // the whole corpus rather than over an unknown probed subset.
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, NLIST))
        .unwrap();
    let stats = result.query(0).stats;

    let above = distances.iter().filter(|value| **value > upper).count();
    let on_the_cut = distances.iter().filter(|value| **value == upper).count();
    let below_lower = distances.iter().filter(|value| **value < lower).count();
    let in_band = distances
        .iter()
        .filter(|value| **value >= lower && **value < upper)
        .count();

    // Each group must be populated, or the equalities below hold vacuously.
    assert!(above > 0 && below_lower > 0 && in_band > 0);
    assert_eq!(
        on_the_cut, 1,
        "the cut is a row's own distance by construction"
    );

    assert_eq!(
        stats.early_abandoned(),
        above,
        "early_abandoned must be exactly the rows above the upper cut"
    );
    assert_eq!(stats.rows_scanned(), ids.len());
    assert_eq!(stats.rows_committed(), in_band);
    // The rows the cutoff let through: everything at or below `upper`, whether
    // the band then admits it or not. A row *on* the cut is in this group, which
    // is what distinguishes the raw cut from one nudged even a single ULP.
    assert_eq!(
        stats.rows_scanned() - stats.early_abandoned(),
        ids.len() - above,
        "a row sitting on the cut must reach the band test, not be abandoned"
    );
}

#[test]
fn batch_stats_are_per_query_and_shared_lists_are_counted_once() {
    let (mut reader, queries, d) = build_asymmetric_fixture();
    let band = l2(0.0, 0.25);
    // nprobe = nlist means both queries probe the same full set of lists, so the
    // number of unique lists is exactly nlist.
    let result = reader
        .range_search_batch(&queries, 2, VectorRangeSearchParams::new(band, 8))
        .unwrap();
    // The fixture guarantees every list is non-empty, so list_reads must be
    // *exactly* the list count: a `<=` would also accept 1, or even 0.
    assert_eq!(
        result.call_stats().list_reads(),
        ASYMMETRIC_NLIST,
        "each unique non-empty list counts exactly once, neither per (list, query) nor missed"
    );
    for qi in 0..2 {
        assert_eq!(result.query(qi).stats.lists_probed(), ASYMMETRIC_NLIST);
        assert_eq!(
            result.query(qi).stats.rows_committed(),
            result.query(qi).labels.len()
        );
    }

    // Per-query stats must match running each query on its own, field by field.
    let tuple = |st: &paimon_vindex_core::range::RangeSearchStats| {
        (
            st.rows_scanned(),
            st.rows_committed(),
            st.early_abandoned(),
            st.lists_probed(),
        )
    };
    let singles: Vec<_> = (0..2)
        .map(|qi| {
            reader
                .range_search(
                    &queries[qi * d..(qi + 1) * d],
                    VectorRangeSearchParams::new(band, 8),
                )
                .unwrap()
        })
        .collect();
    // Prove with an oracle that the two queries have different hit counts. This
    // step does not depend on the implementation, so it validates that the
    // fixture really is asymmetric rather than that the implementation happened
    // to compute two different numbers.
    let (corpus, corpus_ids) = asymmetric_corpus();
    let truth: Vec<usize> = (0..2)
        .map(|qi| {
            brute_force_band(
                &queries[qi * d..(qi + 1) * d],
                &corpus,
                &corpus_ids,
                d,
                band,
            )
            .len()
        })
        .collect();
    assert_ne!(
        truth[0], truth[1],
        "the fixture produced symmetric hit counts ({truth:?}), so this test cannot \
         demonstrate per-query isolation"
    );
    assert_ne!(
        tuple(singles[0].query(0).stats),
        tuple(singles[1].query(0).stats)
    );
    for (qi, single) in singles.iter().enumerate() {
        assert_eq!(
            tuple(result.query(qi).stats),
            tuple(single.query(0).stats),
            "query {qi}'s batch stats must equal its standalone stats field by field"
        );
    }
}

#[test]
fn an_empty_band_batch_reports_no_probing() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(64, 8, 4);
    let empty = l2(1.0, 1.0);
    let result = reader
        .range_search_batch(
            &vectors[0..2 * d],
            2,
            VectorRangeSearchParams::new(empty, 4),
        )
        .unwrap();
    assert_eq!(result.call_stats().list_reads(), 0);
    for qi in 0..2 {
        assert_eq!(result.query(qi).stats.lists_probed(), 0);
        assert_eq!(result.query(qi).labels.len(), 0);
    }
}

#[test]
fn stats_report_real_work() {
    let (mut reader, vectors, _ids, d) = build_flat_fixture(512, 16, 8);
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 4))
        .unwrap();
    let stats = result.query(0).stats;
    assert_eq!(
        stats.lists_probed(),
        4,
        "the number of logical ranks probed"
    );
    assert_eq!(stats.rows_committed(), result.query(0).labels.len());
    assert!(
        stats.rows_scanned() >= stats.rows_committed(),
        "rows_scanned includes rejected and early-abandoned rows, so it cannot be smaller"
    );
    assert!(stats.early_abandoned() <= stats.rows_scanned());
    assert_eq!(result.call_stats().list_reads(), 4);
}

#[test]
fn early_abandon_keeps_rows_just_inside_the_upper_cut() {
    // Regression guard for a real defect. Pruning and committing used to be two
    // kernels: one reduced 128-element blocks into a scalar running total across
    // four accumulators, the other reduced once (two accumulators on NEON, one on
    // AVX2). Their sums differ by a few ULP once d >= 128, so a row genuinely in
    // band could be abandoned -- measured 1.24% of boundary rows at d=128, 1.65%
    // at d=256 and 6.06% at d=768. `fvec_l2sqr_unless_exceeds` now prunes against
    // the accumulation it commits, so there is no divergence to cover and the cut
    // is used raw. Restoring the two-kernel scan while keeping the raw cut makes
    // this test fail, which is what makes it a guard rather than a formality.
    //
    // The trigger is narrow, so the test has to aim at it precisely: for each
    // probed row, put the upper cut exactly one ULP **above that row's own
    // distance**. The row is then in band by construction and must come back. A
    // cut placed anywhere else -- a quantile of the whole corpus, say -- lands
    // in a sparse gap where no row is within a few ULP of it, and the defect is
    // invisible. That is not hypothetical: an earlier version of this test used
    // a median cut and passed even with the fix reverted.
    const D: usize = 768;
    let (mut reader, vectors, ids, d) = build_flat_fixture(256, D, 4);
    assert_eq!(d, D, "the divergence only appears once d >= 128");
    let query = vectors[0..d].to_vec();

    // Only rows the probe actually reaches can be dropped by early abandon, so
    // restrict to the query's own list and check every one of them.
    let probed: Vec<(i64, f32)> = ids
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, fvec_l2sqr(&query, &vectors[row * d..(row + 1) * d])))
        .filter(|(_, dist)| *dist < 10.0)
        .collect();
    assert!(
        probed.len() > 20,
        "expected a populated near cluster, got {} rows",
        probed.len()
    );

    let mut checked = 0usize;
    for &(id, distance) in &probed {
        // One ULP above the row's distance: the row is strictly inside.
        let upper = f32::from_bits(distance.to_bits() + 1);
        let band = l2(0.0, upper);
        let result = reader
            .range_search(&query, VectorRangeSearchParams::new(band, 4))
            .unwrap();
        let returned: HashSet<i64> = result.query(0).labels.iter().copied().collect();
        assert!(
            returned.contains(&id),
            "row {id} at distance {distance:?} is inside [0, {upper:?}) but was dropped; \
             pruning and committing have diverged again"
        );
        // And the whole result must still agree with the oracle exactly.
        assert_eq!(
            pairs_of(result.query(0)),
            bits_of(brute_force_band(&query, &vectors, &ids, d, band)),
            "cut {upper:?} produced a result differing from brute force"
        );
        checked += 1;
    }
    assert!(checked > 20, "vacuous: only {checked} cuts exercised");
}

#[test]
fn a_row_exactly_on_the_upper_cut_is_excluded() {
    // The half-open interval's other edge. This is a membership property rather
    // than an early-abandon one: the cut is exclusive, and pruning at that same
    // value must not turn the exclusion into an inclusion.
    const D: usize = 768;
    let (mut reader, vectors, ids, d) = build_flat_fixture(256, D, 4);
    let query = vectors[0..d].to_vec();
    let probed: Vec<(i64, f32)> = ids
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, fvec_l2sqr(&query, &vectors[row * d..(row + 1) * d])))
        // A zero distance would make the band `[0, 0)`, which is empty and takes
        // the short-circuit instead of exercising the cut.
        .filter(|(_, dist)| *dist > 0.0 && *dist < 10.0)
        .collect();
    assert!(probed.len() > 8, "expected a populated near cluster");

    for &(id, distance) in probed.iter().take(8) {
        let result = reader
            .range_search(&query, VectorRangeSearchParams::new(l2(0.0, distance), 4))
            .unwrap();
        let returned: HashSet<i64> = result.query(0).labels.iter().copied().collect();
        assert!(
            !returned.contains(&id),
            "row {id} at distance {distance:?} sits exactly on the open upper cut \
             and must be excluded"
        );
    }
}

#[test]
fn range_probe_selection_is_invariant_to_batch_size() {
    // A query must select the same lists whether it runs alone or beside other
    // queries, otherwise batch size becomes observable query semantics.
    // `kmeans::find_topk_batch` scores `nq == 1` with the direct `fvec_l2sqr`
    // kernel and takes an SGEMM path for `nq > 1`, but recomputes every selected
    // centroid's distance with the direct kernel once its error bound says the
    // ranking is ambiguous, so the two agree. This test is the end-to-end guard
    // on that; `kmeans` carries the focused one.
    //
    // The centroids below are chosen so the two arithmetics genuinely disagree:
    // at a magnitude of 1e9 the true squared distances (16384 and 4096) are far
    // below the rounding granularity of ||q|| and ||c||, so norm reconstruction
    // collapses both to 0 and then breaks the tie by index -- picking the
    // *farther* centroid.
    const D: usize = 1;
    const BASE: f32 = 1.0e9;
    let mut index = IVFFlatIndex::new(D, 2, MetricType::L2);
    // List 0's centroid is the farther one, so an index-order tie-break picks it.
    index.set_quantizer_centroids(vec![BASE + 128.0, BASE + 64.0]);
    for (list_id, id) in [(0usize, 7000i64), (1usize, 7001i64)] {
        index.ids[list_id] = vec![id];
        let centroid = index.quantizer_centroids()[list_id];
        index.vectors[list_id] = vec![centroid];
    }
    let mut reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();

    let query = vec![BASE];
    // nprobe = 1 so only the single best-ranked list is scanned; the band is wide
    // enough that whichever list is chosen contributes its row.
    let band = DistanceBand::new(Bound::Finite(0.0), Bound::Unbounded, MetricType::L2).unwrap();
    let params = VectorRangeSearchParams::new(band, 1);

    let alone = reader.range_search(&query, params).unwrap();
    let alone_labels: Vec<i64> = alone.query(0).labels.to_vec();
    assert_eq!(
        alone_labels.len(),
        1,
        "nprobe = 1 over single-row lists must return exactly one row"
    );

    // The same query as row 0 of a two-query batch, which is the case that used
    // to switch arithmetic. The partner query is deliberately elsewhere.
    let mut batched = query.clone();
    batched.push(BASE + 64.0);
    let batch = reader.range_search_batch(&batched, 2, params).unwrap();

    assert_eq!(
        pairs_of(batch.query(0)),
        pairs_of(alone.query(0)),
        "query 0 selected a different list when batched: probe selection must not \
         depend on nq"
    );
    // And it must be the genuinely nearer centroid, which is list 1.
    assert_eq!(
        alone_labels,
        vec![7001],
        "direct scoring must pick the nearer centroid (list 1), not the \
         index-order tie-break that norm reconstruction produces"
    );
}

#[test]
fn a_caller_bug_outranks_an_unsupported_family() {
    // A family that cannot serve range search must not swallow invalid input: an
    // FFI caller reading `Unsupported` as "fall back to a scan" would silently
    // paper over its own bug. The top-K entry points already validate params
    // before dispatching, and these must match.
    let mut reader = build_diskann_fixture();
    let band = l2(0.0, 1.0);

    // nprobe == 0 on a family that does not support range search at all.
    let err = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(band, 0))
        .unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidInput,
        "nprobe == 0 is a caller bug even on an unsupported family"
    );

    // Malformed filter bytes, likewise.
    let err = reader
        .range_search_with_roaring_filter(
            &[0.0; 8],
            VectorRangeSearchParams::new(band, 4),
            b"not roaring",
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    // And on the batch entry points.
    let err = reader
        .range_search_batch(&[0.0; 16], 2, VectorRangeSearchParams::new(band, 0))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    let err = reader
        .range_search_batch_with_roaring_filter(
            &[0.0; 16],
            2,
            VectorRangeSearchParams::new(band, 4),
            b"not roaring",
        )
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    // A band whose metric disagrees with the index is also a caller bug, and was
    // the field the first version of this hoist forgot.
    let cosine_band =
        DistanceBand::new(Bound::Finite(0.0), Bound::Finite(1.0), MetricType::Cosine).unwrap();
    let err = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(cosine_band, 4))
        .unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidInput,
        "a cosine band against an L2 index is a caller bug even on an unsupported family"
    );

    // With valid input, the family gap is still reported.
    let err = reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(band, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

#[test]
fn a_zero_nprobe_outranks_the_metric_capability_gap() {
    // Two gates can reject the same call, and the order is observable: an FFI
    // caller reading `Unsupported` as "fall back to a scan" would silently paper
    // over a zero nprobe if the metric gap were reported first.
    let (mut cosine_reader, ..) = build_flat_fixture_with_metric(64, 8, 4, MetricType::Cosine);
    let cosine_band =
        DistanceBand::new(Bound::Finite(0.0), Bound::Finite(1.0), MetricType::Cosine).unwrap();
    let err = cosine_reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(cosine_band, 0))
        .unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::InvalidInput,
        "nprobe == 0 must be reported before the uncertified-metric gap"
    );

    // With a valid nprobe the metric gap is what is left to report.
    let err = cosine_reader
        .range_search(&[0.0; 8], VectorRangeSearchParams::new(cosine_band, 4))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

/// Builds a single-list index whose payload exceeds the 64 MiB threshold that
/// sends reads down the chunked streaming path.
///
/// The reader streams a list whose payload exceeds `MAX_IVF_BATCH_READ_BYTES`
/// (64 MiB in `index_io_util`, not re-exported, hence the literal below). A row
/// costs `d * 4` bytes, so `n * d * 4` alone clears it at d = 64 and 262_145
/// rows; the real payload is larger still, since it also carries a header and
/// the encoded ids.
///
/// Peak footprint is a few multiples of the 67 MiB vector data: this function
/// holds the corpus, the index's copy, and the serialized bytes at once.
fn build_oversized_fixture() -> (Reader, Vec<f32>, Vec<i64>, usize) {
    const D: usize = 64;
    const ROWS: usize = 262_145; // (64 MiB / (64 * 4)) + 1
    let mut rng = Lcg::new(SEED);
    let mut index = IVFFlatIndex::new(D, 1, MetricType::L2);
    index.set_quantizer_centroids((0..D).map(|dim| centroid_component(0, dim)).collect());

    let mut vectors = vec![0.0f32; ROWS * D];
    let mut ids = vec![0i64; ROWS];
    for row in 0..ROWS {
        ids[row] = 1000 + row as i64;
        for dim in 0..D {
            vectors[row * D + dim] =
                centroid_component(0, dim) + rng.next_symmetric() * NOISE / (D as f32).sqrt();
        }
    }
    index.ids[0] = ids.clone();
    index.vectors[0] = vectors.clone();
    let reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();
    (reader, vectors, ids, D)
}

#[test]
fn an_oversized_list_streams_and_still_matches_the_oracle() {
    // The streaming branch has its own chunk collection, filtering, tally
    // accumulation and output merging, none of which the other range tests
    // reach: every other fixture is far below the threshold. It costs about
    // 67 MiB and a couple of seconds, which is worth paying to stop this branch
    // from shipping with no coverage at all.
    let (mut reader, vectors, ids, d) = build_oversized_fixture();
    // Guard the premise: if the fixture ever drops below the threshold, this
    // test silently becomes a duplicate of the ordinary path. The literal
    // mirrors `index_io_util::MAX_IVF_BATCH_READ_BYTES`, which is not re-exported;
    // if that constant grows, this assertion is what will fail and say so.
    let payload_bytes = ids.len() * d * std::mem::size_of::<f32>();
    assert!(
        payload_bytes > 64 * 1024 * 1024,
        "fixture payload {payload_bytes} B must exceed the oversized threshold, \
         otherwise the streaming branch is never entered"
    );
    let query = vectors[0..d].to_vec();
    let band = l2(0.0, 2.0);

    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 1))
        .unwrap();
    let want = bits_of(brute_force_band(&query, &vectors, &ids, d, band));
    assert!(!want.is_empty(), "vacuous: the band matched nothing");
    assert_eq!(
        pairs_of(result.query(0)),
        want,
        "the streamed oversized list must return exactly the oracle's rows"
    );

    // Statistics must survive chunking: the list is read once however many
    // chunks it takes, and every row is accounted for.
    assert_eq!(
        result.call_stats().list_reads(),
        1,
        "one list, one logical read"
    );
    let stats = result.query(0).stats;
    assert_eq!(stats.lists_probed(), 1);
    assert_eq!(stats.rows_committed(), result.query(0).labels.len());
    assert_eq!(
        stats.rows_scanned(),
        ids.len(),
        "every row in the streamed list must be counted as scanned"
    );

    // And the filtered path through the same branch.
    let allowed: HashSet<i64> = ids.iter().copied().filter(|id| id % 2 == 0).collect();
    let filtered = reader
        .range_search_with_roaring_filter(
            &query,
            VectorRangeSearchParams::new(band, 1),
            &serialize_roaring(&allowed),
        )
        .unwrap();
    let mut expect: Vec<(i64, u32)> = pairs_of(result.query(0))
        .into_iter()
        .filter(|(id, _)| allowed.contains(id))
        .collect();
    expect.sort_unstable();
    assert!(!expect.is_empty(), "vacuous: the filter admitted nothing");
    assert_eq!(pairs_of(filtered.query(0)), expect);
}

#[test]
fn early_abandon_survives_intermediate_underflow() {
    // A normal cut does not stop the individual squared terms from landing in
    // the subnormal range, where a rounding carries absolute rather than
    // relative error. Every other test in this file works at ordinary
    // magnitudes and never enters that regime.
    //
    // Pruning and committing are one accumulation, so a rounding that happens
    // happens once and to both. This checks that the claim survives where it is
    // least comfortable: the monotonicity argument holds for gradual underflow
    // too -- adding a non-negative subnormal is still non-decreasing -- but
    // "still true in the subnormal range" is worth a fixture rather than a
    // sentence.
    //
    // Reaching the regime needs care, and two earlier attempts at this test
    // missed. The first perturbed 1.0 by 1e-24, far below half an ULP at 1.0, so
    // every coordinate rounded back to exactly 1.0 and every distance was zero.
    // The second offset a base by whole ULPs, which makes the difference an
    // exact power of two -- subnormal, but squaring a power of two is exact, so
    // it still produced no rounding at all. Both passed against any threshold,
    // including zero. So this version asserts the regime it needs rather than
    // assuming it: subnormal *and* inexact.
    const D: usize = 256;
    const ROWS: usize = 64;

    // Coordinates sit at or just above 2^-65, so a square lands in
    // [2^-130, 2^-128). That is deep enough into the subnormal range that the
    // result keeps about 19 bits rather than 24, so squaring a full 24-bit
    // mantissa has to round -- in the multiply on both AVX2 and NEON, neither of
    // which fuses it (`vmlaq_f32` is a separate multiply and add; `vfmaq_f32` is
    // the fused one). 256 such terms sum back to ~4e-37, which is normal, so the
    // band comparison itself is ordinary arithmetic.
    fn coordinate(row: usize, dim: usize) -> f32 {
        let scatter =
            (row as u32).wrapping_mul(2_654_435_761) ^ (dim as u32).wrapping_mul(2_246_822_519);
        f32::from_bits((62u32 << 23) | (scatter & 0x007f_ffff))
    }

    let mut vectors = vec![0.0f32; ROWS * D];
    let mut ids = vec![0i64; ROWS];
    for row in 0..ROWS {
        ids[row] = 2000 + row as i64;
        for dim in 0..D {
            vectors[row * D + dim] = coordinate(row, dim);
        }
    }
    // The origin, so each difference is the stored coordinate itself and the
    // subtraction contributes no rounding of its own.
    let query = vec![0.0f32; D];

    // Assert the regime per row, not just in aggregate: it is the row the
    // pruning kernel rules on that has to be in it. A mantissa of zero squares
    // exactly -- `coordinate(0, 0)` is exactly 2^-65 -- so allow one such
    // coordinate per row rather than demanding all D round.
    for row in 0..ROWS {
        let mut inexact = 0usize;
        for &value in &vectors[row * D..(row + 1) * D] {
            let squared = value * value;
            assert!(
                squared > 0.0 && squared < f32::MIN_POSITIVE,
                "row {row}: every squared term must be subnormal for this test \
                 to mean anything, got {squared:e}"
            );
            if f64::from(squared) != f64::from(value) * f64::from(value) {
                inexact += 1;
            }
        }
        assert!(
            inexact >= D - 1,
            "row {row}: the squarings must actually round, only {inexact} of {D} did"
        );
    }

    let distances: Vec<f32> = (0..ROWS)
        .map(|row| fvec_l2sqr(&query, &vectors[row * D..(row + 1) * D]))
        .collect();
    assert!(
        distances.iter().all(|d| *d >= f32::MIN_POSITIVE),
        "the summed distances must be normal, got {:e}",
        distances.iter().copied().fold(f32::INFINITY, f32::min)
    );

    let mut sorted = distances.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("distances are finite"));
    // Cut at the median, so the tightest excluded row sits exactly on the cut
    // and the tightest retained row is a hair below it. Pruning has to make its
    // decision right at the boundary, in the regime asserted above, which is
    // where a prune diverging from the commit would drop a row it must keep.
    let cut = sorted[ROWS / 2];
    let nearest_kept = sorted[ROWS / 2 - 1];
    assert!(
        nearest_kept < cut && nearest_kept > cut * 0.99,
        "the retained row nearest the cut must be close to it, got {nearest_kept:e} against {cut:e}"
    );

    let mut index = IVFFlatIndex::new(D, 1, MetricType::L2);
    index.set_quantizer_centroids(query.clone());
    index.ids[0] = ids.clone();
    index.vectors[0] = vectors.clone();
    let mut reader = VectorIndexReader::open(Cursor::new(serialize(&index))).unwrap();

    let band = l2(0.0, cut);
    let result = reader
        .range_search(&query, VectorRangeSearchParams::new(band, 1))
        .unwrap();
    let want = bits_of(brute_force_band(&query, &vectors, &ids, D, band));
    assert!(
        !want.is_empty() && want.len() < ROWS,
        "the band must split the corpus, got {} of {ROWS}",
        want.len()
    );
    assert!(
        result.query(0).stats.early_abandoned() > 0,
        "no row was abandoned early, so the pruning kernel never ruled on this regime"
    );
    assert_eq!(
        pairs_of(result.query(0)),
        want,
        "early abandon dropped an in-band row whose distance arithmetic underflows"
    );
}
