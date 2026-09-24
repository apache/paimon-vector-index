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
use paimon_vindex_core::ivfpq::{search_batch_reader, IVFPQIndex};
use std::io::Cursor;

const ROWS: usize = 225;

fn build_index(m: usize) -> IVFPQIndex {
    let mut index = IVFPQIndex::with_nbits(m, 1, m, 4, MetricType::L2, false);
    index.set_quantizer_centroids(vec![0.0; m]);
    index.pq.centroids = vec![0.0; m * 16];
    for sub in 0..m {
        index.pq.centroids[sub * 16 + 1] = 1.0;
    }
    index.pq.rebuild_norms_cache();

    // The first 200 distances equal 1, matching the full LUT maximum.
    // This isolates accumulation overflow from LUT-range calibration.
    let mut data = vec![0.0; ROWS * m];
    for row in 0..200 {
        data[row * m] = 1.0;
    }
    data[200 * m..].fill(1.0);
    let ids = (0..ROWS as i64).collect::<Vec<_>>();
    index.add(&data, &ids, ROWS);
    index
}

fn assert_distances(ids: &[i64], distances: &[f32], m: usize) {
    assert_eq!(ids.len(), ROWS);
    assert_eq!(distances.len(), ROWS);
    let mut seen = vec![false; ROWS];
    for (&id, &distance) in ids.iter().zip(distances) {
        assert!((0..ROWS as i64).contains(&id));
        assert!(!seen[id as usize], "duplicate row ID {id}");
        seen[id as usize] = true;
        let expected = if id < 200 { 1.0 } else { m as f32 };
        assert!(
            (distance - expected).abs() < 1e-4,
            "m={m}, row={id}: expected {expected}, got {distance}"
        );
    }
}

#[test]
fn four_bit_in_memory_scans_avoid_accumulator_overflow() {
    for m in [256, 258, 512] {
        let mut index = build_index(m);
        let query = vec![0.0; m];
        for fastscan in [false, true] {
            if fastscan {
                index.build_search_structures();
            }
            let mut ids = vec![-1; ROWS];
            let mut distances = vec![0.0; ROWS];
            index.search(&query, 1, ROWS, 1, &mut distances, &mut ids);
            assert_distances(&ids, &distances, m);
            index.search_with_max_codes(&query, 1, ROWS, 1, ROWS, &mut distances, &mut ids);
            assert_distances(&ids, &distances, m);
        }
    }
}

#[test]
fn four_bit_reader_scans_avoid_accumulator_overflow() {
    for m in [256, 258, 512] {
        let index = build_index(m);
        let query = vec![0.0; m];
        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();
        for precomputed in [false, true] {
            if precomputed {
                reader.optimize_for_search().unwrap();
            }
            let (ids, distances) = reader.search(&query, ROWS, 1).unwrap();
            assert_distances(&ids, &distances, m);
            let (ids, distances) =
                search_batch_reader(&mut reader, &query.repeat(2), 2, ROWS, 1).unwrap();
            for row in 0..2 {
                let start = row * ROWS;
                assert_distances(
                    &ids[start..start + ROWS],
                    &distances[start..start + ROWS],
                    m,
                );
            }
        }
    }
}
