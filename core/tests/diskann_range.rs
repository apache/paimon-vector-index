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

use paimon_vindex_core::diskann::{
    DiskAnnBuildParams, DiskAnnIndex, DiskAnnRawVectorEncoding, DiskAnnStorageLayout,
};
use paimon_vindex_core::diskann_io::{
    write_diskann_index, DiskAnnHeader, DiskAnnIndexReader, DISKANN_HEADER_SIZE,
};
use paimon_vindex_core::distance::{fvec_distance, preprocess_vectors, MetricType};
use paimon_vindex_core::index::{VectorIndexReader, VectorSearchParams};
use paimon_vindex_core::io::PosWriter;
use paimon_vindex_core::range::{Bound, DiskAnnRangeSearchParams, DistanceBand};
use roaring::RoaringTreemap;
use std::io::Cursor;

fn fixture(
    metric: MetricType,
    layout: DiskAnnStorageLayout,
    encoding: DiskAnnRawVectorEncoding,
    bits: usize,
) -> VectorIndexReader<Cursor<Vec<u8>>> {
    let mut index = DiskAnnIndex::with_pq_bits(
        4,
        metric,
        2,
        bits,
        DiskAnnBuildParams {
            storage_layout: layout,
            raw_vector_encoding: encoding,
            max_degree: 8,
            build_search_list_size: 16,
            ..Default::default()
        },
    );
    let data = fixture_data();
    index.train(&data, 64).unwrap();
    index.add(&data, &(0..64).map(i64::from).collect::<Vec<_>>());
    let mut bytes = Vec::new();
    write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
    VectorIndexReader::open(Cursor::new(bytes)).unwrap()
}

fn fixture_data() -> Vec<f32> {
    (0..64)
        .flat_map(|i| {
            if i == 0 {
                [0.0; 4]
            } else {
                [
                    i as f32 / 8.0,
                    (i % 5) as f32 / 4.0,
                    (i % 7) as f32 / 4.0,
                    1.0,
                ]
            }
        })
        .collect()
}

#[test]
fn unified_range_entry_points_accept_diskann_options_and_preserve_topk() {
    let mut reader = fixture(
        MetricType::L2,
        DiskAnnStorageLayout::Compact,
        DiskAnnRawVectorEncoding::F32,
        4,
    );
    let query = [0.0; 4];
    reader
        .search(&query, VectorSearchParams::with_l_search(5, 16))
        .unwrap();
    let topk_stats = reader.search_stats();
    let params = DiskAnnRangeSearchParams::new(
        DistanceBand::from_raw(Bound::Unbounded, Bound::Finite(100.0), MetricType::L2).unwrap(),
        2,
    );
    let expected = reader.diskann_range_search(&query, params).unwrap();
    let actual = reader.range_search(&query, params).unwrap();
    assert_eq!(actual.labels(), expected.labels());
    assert_eq!(actual.raw_distances(), expected.raw_distances());
    assert_eq!(actual.diskann_stats(), expected.diskann_stats());
    assert!(reader.supports_range_search());
    assert!(paimon_vindex_core::index::IndexType::DiskAnn.supports_range_search(MetricType::L2));
    let queries = [0.0; 8];
    let batch = reader.range_search_batch(&queries, 2, params).unwrap();
    assert_eq!(batch.labels().len(), expected.labels().len() * 2);
    let mut filter = Vec::new();
    RoaringTreemap::from_iter([0, 3, 9])
        .serialize_into(&mut filter)
        .unwrap();
    let filtered = reader
        .range_search_with_roaring_filter(&query, params, &filter)
        .unwrap();
    assert_eq!(filtered.labels().len(), 3);
    let filtered_batch = reader
        .range_search_batch_with_roaring_filter(&queries, 2, params, &filter)
        .unwrap();
    assert_eq!(filtered_batch.labels().len(), 6);
    assert_eq!(reader.search_stats(), topk_stats);
    assert!(reader
        .range_search_with_roaring_filter(&query, params, &[])
        .is_err());
}

#[test]
fn diskann_range_is_uncapped_and_preserves_topk_state() {
    let mut reader = fixture(
        MetricType::L2,
        DiskAnnStorageLayout::Compact,
        DiskAnnRawVectorEncoding::F32,
        4,
    );
    let query = [0.0; 4];
    let topk_params = VectorSearchParams::with_l_search(5, 16);
    let before = reader.search(&query, topk_params).unwrap();
    let stats = reader.diskann_search_stats().unwrap();
    let unified_stats = reader.search_stats();
    let band =
        DistanceBand::from_raw(Bound::Unbounded, Bound::Finite(100.0), MetricType::L2).unwrap();
    let result = reader
        .diskann_range_search(&query, DiskAnnRangeSearchParams::new(band, 2))
        .unwrap();
    assert_eq!(
        result.labels().len(),
        64,
        "range must not stop at starting_l"
    );
    assert!(result.diskann_stats().unwrap()[0].second_phase);
    assert_eq!(reader.diskann_search_stats().unwrap(), stats);
    assert_eq!(reader.search_stats(), unified_stats);
    assert_eq!(reader.search(&query, topk_params).unwrap(), before);
}

#[test]
fn diskann_range_layout_metric_filter_and_boundary_matrix() {
    for metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
        for layout in [
            DiskAnnStorageLayout::Compact,
            DiskAnnStorageLayout::Interleaved,
        ] {
            for encoding in [DiskAnnRawVectorEncoding::F32, DiskAnnRawVectorEncoding::F16] {
                for bits in [4, 8] {
                    let mut reader = fixture(metric, layout, encoding, bits);
                    let query = [1.0, 0.0, 0.0, 0.0];
                    let all =
                        DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, metric).unwrap();
                    let full = reader
                        .diskann_range_search(&query, DiskAnnRangeSearchParams::new(all, 2))
                        .unwrap();
                    assert_eq!(full.labels().len(), 64);
                    assert!(full.diskann_stats().unwrap()[0].exhaustive);
                    // Oracle starts from independently prepared persisted vectors,
                    // never from another range result. This catches broken scoring.
                    let source = fixture_data();
                    let mut stored = preprocess_vectors(&source, 64, 4, metric);
                    if encoding == DiskAnnRawVectorEncoding::F16 {
                        for value in &mut stored {
                            *value = half::f16::from_f32(*value).to_f32();
                        }
                    }
                    for (&id, &distance) in full.labels().iter().zip(full.raw_distances()) {
                        let expected = fvec_distance(
                            &query,
                            &stored[id as usize * 4..id as usize * 4 + 4],
                            metric,
                        );
                        assert!(
                            (distance - expected).abs() < 1e-5 * expected.abs().max(1.0),
                            "{metric:?} {layout:?} {encoding:?} id={id}: {distance} != {expected}"
                        );
                    }
                    let broad =
                        DistanceBand::from_raw(Bound::Unbounded, Bound::Finite(100.0), metric)
                            .unwrap();
                    let graph = reader
                        .diskann_range_search(&query, DiskAnnRangeSearchParams::new(broad, 2))
                        .unwrap();
                    assert_eq!(
                        graph.labels().len(),
                        64,
                        "graph expansion {metric:?} {layout:?}"
                    );
                    assert!(graph.diskann_stats().unwrap()[0].second_phase);
                    let cut = full.raw_distances()[31];
                    let band = DistanceBand::from_raw(
                        Bound::Finite(cut),
                        Bound::Finite(cut.next_up()),
                        metric,
                    )
                    .unwrap();
                    let mut params = DiskAnnRangeSearchParams::new(band, 4);
                    params.exhaustive = true;
                    let expected = full
                        .labels()
                        .iter()
                        .zip(full.raw_distances())
                        .filter(|(_, d)| band.admit_raw(**d))
                        .map(|(&id, &d)| (id, d))
                        .collect::<Vec<_>>();
                    let actual = reader.diskann_range_search(&query, params).unwrap();
                    assert_eq!(
                        actual
                            .labels()
                            .iter()
                            .copied()
                            .zip(actual.raw_distances().iter().copied())
                            .collect::<Vec<_>>(),
                        expected
                    );
                    let filter = (0..64).filter(|id| id % 3 == 0).collect::<RoaringTreemap>();
                    let mut bytes = Vec::new();
                    filter.serialize_into(&mut bytes).unwrap();
                    let queries = [query, query].concat();
                    let batch = reader
                        .diskann_range_search_batch_with_roaring_filter(&queries, 2, params, &bytes)
                        .unwrap();
                    let selected = expected
                        .iter()
                        .filter(|(id, _)| filter.contains(*id as u64))
                        .map(|(id, _)| *id)
                        .collect::<Vec<_>>();
                    assert_eq!(batch.query(0).labels, selected);
                    assert_eq!(batch.query(1).labels, selected);
                    assert_eq!(batch.query(0).raw_distances, batch.query(1).raw_distances);
                    assert!(reader
                        .diskann_range_search_with_roaring_filter(&query, params, &[])
                        .is_err());
                }
            }
        }
    }
}

#[test]
fn diskann_range_validates_before_empty_band_and_rejects_budget_exhaustion() {
    let mut reader = fixture(
        MetricType::L2,
        DiskAnnStorageLayout::Compact,
        DiskAnnRawVectorEncoding::F32,
        4,
    );
    let empty =
        DistanceBand::from_raw(Bound::Finite(0.0), Bound::Finite(0.0), MetricType::L2).unwrap();
    let mut params = DiskAnnRangeSearchParams::new(empty, 0);
    assert!(reader.diskann_range_search(&[0.0; 4], params).is_err());
    params.starting_l = 2;
    assert!(reader.diskann_range_search(&[f32::NAN; 4], params).is_err());
    assert!(reader.diskann_range_search_batch(&[], 0, params).is_err());
    assert!(reader
        .diskann_range_search_with_roaring_filter(&[0.0; 4], params, &[])
        .is_err());
    params.band =
        DistanceBand::from_raw(Bound::Unbounded, Bound::Finite(100.0), MetricType::L2).unwrap();
    params.max_working_bytes = 1;
    assert!(reader
        .diskann_range_search(&[0.0; 4], params)
        .unwrap_err()
        .to_string()
        .contains("budget"));
}

// Rewrite only test bytes to control topology after the writer's BFS permutation.
// Each edge fits one canonical delta-varint byte; the final node has degree zero.
fn bridge_index() -> Vec<u8> {
    let data = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0];
    let mut index = DiskAnnIndex::with_pq_bits(
        4,
        MetricType::L2,
        2,
        4,
        DiskAnnBuildParams {
            max_degree: 2,
            build_search_list_size: 4,
            raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
            ..Default::default()
        },
    );
    let mut centroids = vec![0.0; 4 * 16];
    centroids[2] = 2.0;
    centroids[4] = 0.5;
    index.pq.set_centroids(centroids);
    index.add(&data, &[0, 1, 2]);
    let mut bytes = Vec::new();
    write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
    let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes.clone())).unwrap();
    reader.ensure_resident().unwrap();
    let ids = (0..3)
        .map(|n| reader.row_id(n).unwrap() as usize)
        .collect::<Vec<_>>();
    let mut nodes = [0; 3];
    for (node, &id) in ids.iter().enumerate() {
        nodes[id] = node;
    }
    let mut header = DiskAnnHeader::decode(&bytes[..DISKANN_HEADER_SIZE]).unwrap();
    header.entry_node = nodes[0] as _;
    bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
    let locators = header.sections.adjacency_index.offset as usize;
    bytes[locators..locators + header.sections.adjacency_index.length as usize].fill(0);
    let adjacency = header.sections.adjacency.offset as usize;
    bytes[adjacency..adjacency + header.sections.adjacency.length as usize].fill(0);
    let mut offset = 0u16;
    for (node, &id) in ids.iter().enumerate() {
        let base = locators + 8 + node * 2;
        bytes[base..base + 2].copy_from_slice(&offset.to_le_bytes());
        if id < 2 {
            let metadata = locators + 8 + 3 * 2 + node * 2;
            bytes[metadata..metadata + 2].copy_from_slice(&1u16.to_le_bytes());
            bytes[adjacency + offset as usize] = nodes[id + 1] as u8;
            offset += 1;
        }
    }
    bytes
}

#[test]
fn diskann_range_crosses_filtered_and_out_of_band_bridges_and_resets_visitation() {
    let bytes = bridge_index();
    let band =
        DistanceBand::from_raw(Bound::Finite(0.1), Bound::Finite(1.0), MetricType::L2).unwrap();
    let mut params = DiskAnnRangeSearchParams::new(band, 1);
    let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();
    assert!(reader
        .diskann_range_search(&[0.0; 4], params)
        .unwrap()
        .labels()
        .is_empty());
    params.range_slack = 4.0;
    let result = reader.diskann_range_search(&[0.0; 4], params).unwrap();
    assert_eq!(result.labels(), [2]);
    assert_eq!(result.raw_distances(), [0.25]);
    let mut filter = Vec::new();
    [2u64]
        .into_iter()
        .collect::<RoaringTreemap>()
        .serialize_into(&mut filter)
        .unwrap();
    let result = reader
        .diskann_range_search_with_roaring_filter(&[0.0; 4], params, &filter)
        .unwrap();
    assert_eq!(
        result.labels(),
        [2],
        "excluded seed and bridge must still navigate"
    );
    assert_eq!(result.diskann_stats().unwrap()[0].vector_evaluations, 4);
}

#[test]
fn diskann_range_read_failure_does_not_poison_reader_or_topk_statistics() {
    use paimon_vindex_core::io::{ReadRequest, SeekRead};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    struct Source {
        bytes: Vec<u8>,
        fail: Arc<AtomicBool>,
    }
    impl SeekRead for Source {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> std::io::Result<()> {
            if self.fail.load(Ordering::SeqCst) {
                return Err(std::io::Error::other("injected range read failure"));
            }
            for request in ranges {
                request.buf.copy_from_slice(
                    &self.bytes[request.pos as usize..request.pos as usize + request.buf.len()],
                );
            }
            Ok(())
        }
    }
    let fail = Arc::new(AtomicBool::new(false));
    let mut reader = VectorIndexReader::open(Source {
        bytes: bridge_index(),
        fail: fail.clone(),
    })
    .unwrap();
    let topk = VectorSearchParams::with_l_search(1, 3);
    let before = reader.search(&[0.0; 4], topk).unwrap();
    let stats = reader.diskann_search_stats().unwrap();
    let all = DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
    fail.store(true, Ordering::SeqCst);
    assert!(reader
        .diskann_range_search(&[0.0; 4], DiskAnnRangeSearchParams::new(all, 1))
        .is_err());
    assert_eq!(reader.diskann_search_stats().unwrap(), stats);
    fail.store(false, Ordering::SeqCst);
    assert_eq!(reader.search(&[0.0; 4], topk).unwrap(), before);
    assert_eq!(
        reader
            .diskann_range_search(&[0.0; 4], DiskAnnRangeSearchParams::new(all, 1))
            .unwrap()
            .labels()
            .len(),
        3
    );
}

#[test]
fn diskann_range_rejects_overflowed_cosine_norms_even_when_final_score_is_finite() {
    let mut bytes = bridge_index();
    let mut header = DiskAnnHeader::decode(&bytes[..DISKANN_HEADER_SIZE]).unwrap();
    header.metric = MetricType::Cosine as u32;
    let vectors = header.sections.vectors.offset as usize;
    bytes[vectors..vectors + 4].copy_from_slice(&f32::MAX.to_le_bytes());
    bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
    let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();
    let band =
        DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, MetricType::Cosine).unwrap();
    let params = DiskAnnRangeSearchParams::new(band, 1);
    let error = reader
        .diskann_range_search(&[1.0, 0.0, 0.0, 0.0], params)
        .unwrap_err();
    assert!(error.to_string().contains("cosine arithmetic"));
}

#[test]
fn diskann_range_budgets_complete_high_dimensional_records() {
    use paimon_vindex_core::io::{ReadRequest, SeekRead, SeekReadCapabilities};
    struct SmallWindows(Cursor<Vec<u8>>);
    impl SeekRead for SmallWindows {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> std::io::Result<()> {
            self.0.pread(ranges)
        }
        fn read_capabilities(&self) -> SeekReadCapabilities {
            SeekReadCapabilities {
                estimated_random_read_latency_nanos: 1_000_000,
                preferred_window_bytes: 4096,
                max_ranges_per_pread: 1,
            }
        }
    }
    for encoding in [DiskAnnRawVectorEncoding::F32, DiskAnnRawVectorEncoding::F16] {
        let d = 32768;
        let mut index = DiskAnnIndex::with_pq_bits(
            d,
            MetricType::L2,
            1,
            4,
            DiskAnnBuildParams {
                max_degree: 2,
                build_search_list_size: 4,
                raw_vector_encoding: encoding,
                ..Default::default()
            },
        );
        index.pq.set_centroids(vec![0.0; d * 16]);
        index.add(&vec![0.0; d * 2], &[1, 2]);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = VectorIndexReader::open(SmallWindows(Cursor::new(bytes))).unwrap();
        let window = reader.read_plan().unwrap().window_bytes;
        let record = d * if encoding == DiskAnnRawVectorEncoding::F32 {
            4
        } else {
            2
        };
        assert!(record > window);
        let band =
            DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
        let mut params = DiskAnnRangeSearchParams::new(band, 1);
        params.max_working_bytes = window * 2 + d * 4;
        assert!(reader
            .diskann_range_search(&vec![0.0; d], params)
            .unwrap_err()
            .to_string()
            .contains("budget"));
        params.max_working_bytes = window + record + d * 4;
        assert_eq!(
            reader
                .diskann_range_search(&vec![0.0; d], params)
                .unwrap()
                .labels()
                .len(),
            2
        );
    }
}

#[test]
fn diskann_range_slack_widens_negative_inner_product_cuts() {
    let mut bytes = bridge_index();
    let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes.clone())).unwrap();
    reader.ensure_resident().unwrap();
    let mut header = DiskAnnHeader::decode(&bytes[..DISKANN_HEADER_SIZE]).unwrap();
    header.metric = MetricType::InnerProduct as u32;
    for node in 0..3 {
        let id = reader.row_id(node).unwrap() as usize;
        let value = [2.0f32, 0.0, 1.0][id];
        let offset = header.sections.vectors.offset as usize + node * 16;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        bytes[header.sections.pq_codes.offset as usize + node] = [1, 0, 2][id];
    }
    bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
    let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();
    let band = DistanceBand::from_raw(
        Bound::Finite(-1.5),
        Bound::Finite(-0.5),
        MetricType::InnerProduct,
    )
    .unwrap();
    let mut params = DiskAnnRangeSearchParams::new(band, 1);
    let query = [1.0, 0.0, 0.0, 0.0];
    assert!(reader
        .diskann_range_search(&query, params)
        .unwrap()
        .labels()
        .is_empty());
    params.range_slack = 2.0;
    let result = reader.diskann_range_search(&query, params).unwrap();
    assert_eq!(result.labels(), [2]);
    assert_eq!(result.raw_distances(), [-1.0]);
}

#[test]
fn diskann_range_zero_cosine_query_scans_all_stored_vectors() {
    let mut reader = fixture(
        MetricType::Cosine,
        DiskAnnStorageLayout::Compact,
        DiskAnnRawVectorEncoding::F16,
        4,
    );
    let band = DistanceBand::from_raw(
        Bound::Finite(1.0),
        Bound::Finite(1.0f32.next_up()),
        MetricType::Cosine,
    )
    .unwrap();
    let result = reader
        .diskann_range_search(&[0.0; 4], DiskAnnRangeSearchParams::new(band, 1))
        .unwrap();
    assert_eq!(result.labels().len(), 64);
    assert_eq!(result.raw_distances(), vec![1.0; 64]);
    assert!(result.diskann_stats().unwrap()[0].exhaustive);
}

#[test]
fn diskann_range_batches_remote_windows_within_adapter_limits() {
    use paimon_vindex_core::io::{ReadRequest, SeekRead, SeekReadCapabilities};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };
    struct RemoteSource {
        bytes: Vec<u8>,
        limit: usize,
        window: usize,
        calls: Arc<Mutex<Vec<Vec<usize>>>>,
        fail_batch: Arc<AtomicBool>,
    }
    impl SeekRead for RemoteSource {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> std::io::Result<()> {
            assert!(ranges.len() <= self.limit);
            self.calls
                .lock()
                .unwrap()
                .push(ranges.iter().map(|r| r.buf.len()).collect());
            if ranges.len() > 1 && self.fail_batch.load(Ordering::SeqCst) {
                let range = &mut ranges[0];
                range.buf.copy_from_slice(
                    &self.bytes[range.pos as usize..range.pos as usize + range.buf.len()],
                );
                return Err(std::io::Error::other(
                    "injected partial range batch failure",
                ));
            }
            for range in ranges {
                range.buf.copy_from_slice(
                    &self.bytes[range.pos as usize..range.pos as usize + range.buf.len()],
                );
            }
            Ok(())
        }
        fn read_capabilities(&self) -> SeekReadCapabilities {
            SeekReadCapabilities {
                estimated_random_read_latency_nanos: 2_000_000,
                preferred_window_bytes: self.window,
                max_ranges_per_pread: self.limit,
            }
        }
    }
    for layout in [
        DiskAnnStorageLayout::Compact,
        DiskAnnStorageLayout::Interleaved,
    ] {
        let (n, d) = (512, 128);
        let mut state = 73u64;
        let data = (0..n * d)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 32) as f32 / u32::MAX as f32
            })
            .collect::<Vec<_>>();
        let mut index = DiskAnnIndex::with_pq_bits(
            d,
            MetricType::L2,
            4,
            4,
            DiskAnnBuildParams {
                storage_layout: layout,
                raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
                max_degree: 16,
                build_search_list_size: 32,
                ..Default::default()
            },
        );
        index.train(&data, n).unwrap();
        index.add(&data, &(0..n as i64).collect::<Vec<_>>());
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        for limit in [1, 3, 32] {
            for window in [4096, 32768] {
                for (budget, exhaustive) in [
                    (64 * 1024 * 1024, false),
                    (64 * 1024, false),
                    (64 * 1024, true),
                ] {
                    let calls = Arc::new(Mutex::new(Vec::new()));
                    let fail_batch = Arc::new(AtomicBool::new(false));
                    let mut reader = DiskAnnIndexReader::open(RemoteSource {
                        bytes: bytes.clone(),
                        limit,
                        window,
                        calls: calls.clone(),
                        fail_batch: fail_batch.clone(),
                    })
                    .unwrap();
                    reader.ensure_resident().unwrap();
                    let topk = reader.search(&data[..d], 5, 64).unwrap();
                    let topk_stats = reader.last_search_stats();
                    calls.lock().unwrap().clear();
                    let band = DistanceBand::from_raw(
                        Bound::Unbounded,
                        Bound::Finite(1e20),
                        MetricType::L2,
                    )
                    .unwrap();
                    let mut params = DiskAnnRangeSearchParams::new(band, 4);
                    params.max_working_bytes = budget.max(window * 8);
                    params.exhaustive = exhaustive;
                    let result = reader.range_search(&data[..d], params).unwrap();
                    assert_eq!(result.labels().len(), n);
                    let trace = calls.lock().unwrap();
                    assert!(trace.iter().all(|batch| batch.len() <= limit));
                    let batched = trace.iter().any(|batch| batch.len() > 1);
                    if limit > 1 && (window == 4096 || exhaustive) {
                        assert!(batched, "independent {layout:?} windows must be batched");
                    }
                    let stats = result.diskann_stats().unwrap()[0];
                    assert_eq!(stats.read_calls, trace.len());
                    assert_eq!(stats.read_bytes, trace.iter().flatten().sum::<usize>());
                    drop(trace);
                    assert_eq!(reader.last_search_stats(), topk_stats);
                    if batched {
                        fail_batch.store(true, Ordering::SeqCst);
                        let error = reader.range_search(&data[..d], params).unwrap_err();
                        assert!(error.to_string().contains("partial range batch failure"));
                        assert_eq!(reader.last_search_stats(), topk_stats);
                        fail_batch.store(false, Ordering::SeqCst);
                        assert_eq!(
                            reader
                                .range_search(&data[..d], params)
                                .unwrap()
                                .labels()
                                .len(),
                            n
                        );
                    }
                    if exhaustive {
                        let mut filter = Vec::new();
                        [5u64, 127, 211, 509]
                            .into_iter()
                            .collect::<RoaringTreemap>()
                            .serialize_into(&mut filter)
                            .unwrap();
                        calls.lock().unwrap().clear();
                        let selected = reader
                            .range_search_with_roaring_filter(&data[..d], params, &filter)
                            .unwrap();
                        let mut labels = selected.labels().to_vec();
                        labels.sort_unstable();
                        assert_eq!(labels, [5, 127, 211, 509]);
                        let stats = selected.diskann_stats().unwrap()[0];
                        let trace = calls.lock().unwrap();
                        assert_eq!(stats.vector_evaluations, 4);
                        assert_eq!(stats.read_calls, trace.len());
                        assert_eq!(stats.read_bytes, trace.iter().flatten().sum::<usize>());
                        drop(trace);
                        filter.clear();
                        [u64::MAX]
                            .into_iter()
                            .collect::<RoaringTreemap>()
                            .serialize_into(&mut filter)
                            .unwrap();
                        calls.lock().unwrap().clear();
                        let absent = reader
                            .range_search_with_roaring_filter(&data[..d], params, &filter)
                            .unwrap();
                        assert!(absent.labels().is_empty());
                        assert!(
                            calls.lock().unwrap().is_empty(),
                            "no eligible windows should be read"
                        );
                    }
                    assert_eq!(reader.search(&data[..d], 5, 64).unwrap(), topk);
                }
            }
        }
    }
}
