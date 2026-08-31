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
    DiskAnnBuildDistance, DiskAnnBuildParams, DiskAnnIndex, DiskAnnRawVectorEncoding,
    DiskAnnStorageLayout,
};
use paimon_vindex_core::diskann_io::write_diskann_index;
use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::index::{
    IndexType, VectorIndexMetadata, VectorIndexReader, VectorSearchParams,
};
use paimon_vindex_core::io::{write_index, PosWriter};
use paimon_vindex_core::ivfflat::IVFFlatIndex;
use paimon_vindex_core::ivfflat_io::write_ivfflat_index;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use paimon_vindex_core::ivfrq::IVFRQIndex;
use paimon_vindex_core::ivfrq_io::write_ivfrq_index;
use paimon_vindex_core::ivfsq::IVFSQIndex;
use paimon_vindex_core::ivfsq_io::write_ivfsq_index;
use paimon_vindex_core::sq::ScalarQuantizer;
use std::fmt::Write as _;
use std::io::Cursor;

struct FixtureCase {
    name: &'static str,
    fixture_hex: &'static str,
    build: fn() -> Vec<u8>,
    index_type: IndexType,
    dimension: usize,
    nlist: usize,
    metric: MetricType,
    total_vectors: i64,
    pq_m: Option<usize>,
    pq_bits: Option<usize>,
    query: Vec<f32>,
    params: VectorSearchParams,
    expected_first_id: i64,
}

#[test]
fn storage_format_v1_golden_fixtures_match_current_writers_and_readers() {
    for case in fixture_cases() {
        let generated = (case.build)();
        let fixture = hex_to_bytes(case.fixture_hex);
        assert_eq!(
            bytes_to_hex(&generated),
            bytes_to_hex(&fixture),
            "{} writer output changed",
            case.name
        );
        assert_diskann_fixture_variant(&case, &fixture);

        let mut reader = VectorIndexReader::open(Cursor::new(fixture)).unwrap();
        assert_metadata(&reader.metadata(), &case);
        let (ids, distances) = reader.search(&case.query, case.params).unwrap();
        assert_eq!(
            ids.len(),
            case.params.top_k,
            "{} result id count",
            case.name
        );
        assert_eq!(
            distances.len(),
            case.params.top_k,
            "{} result distance count",
            case.name
        );
        assert_eq!(ids[0], case.expected_first_id, "{} nearest id", case.name);
        assert!(
            distances[0].is_finite(),
            "{} nearest distance should be finite",
            case.name
        );
    }
}

fn assert_diskann_fixture_variant(case: &FixtureCase, fixture: &[u8]) {
    if case.index_type != IndexType::DiskAnn {
        return;
    }
    assert_eq!(read_u32(fixture, 60) as usize, case.pq_bits.unwrap());
    let expected_encoding = if case.name == "diskann_raw_row_ids_v1" {
        DiskAnnRawVectorEncoding::F32
    } else {
        DiskAnnRawVectorEncoding::F16
    };
    let element_size = match expected_encoding {
        DiskAnnRawVectorEncoding::F32 => 4,
        DiskAnnRawVectorEncoding::F16 => 2,
    };
    assert_eq!(read_u32(fixture, 76), expected_encoding as u32);
    assert_eq!(
        read_u32(fixture, 80) as usize,
        case.dimension * element_size
    );
    match case.name {
        "diskann_compact_multipage_v1" => {
            assert!(
                read_u64(fixture, 184) > 4096,
                "fixture must span multiple adjacency pages"
            );
            assert!(
                read_u64(fixture, 200) > 4096,
                "fixture must contain more raw-vector data than one logical page"
            );
        }
        "diskann_interleaved_4bit_v1" => {
            assert_ne!(
                read_u32(fixture, 12) & (1 << 5),
                0,
                "fixture must select the interleaved layout"
            );
            assert_eq!(
                read_u64(fixture, 200),
                0,
                "interleaved layout must not have a separate vector section"
            );
        }
        compact_name => {
            assert_eq!(
                read_u64(fixture, 200),
                case.total_vectors as u64 * case.dimension as u64 * element_size as u64,
                "{compact_name} compact raw vectors must be densely packed"
            );
            if compact_name == "diskann_raw_row_ids_v1" {
                let row_ids_offset = read_u64(fixture, 112) as usize;
                assert_eq!(
                    read_u32(fixture, row_ids_offset),
                    0,
                    "extreme row IDs must select raw i64 encoding"
                );
            }
        }
    }
}

#[test]
fn storage_format_v1_golden_fixtures_support_search_warmup() {
    for case in fixture_cases() {
        let fixture = hex_to_bytes(case.fixture_hex);

        let mut baseline = VectorIndexReader::open(Cursor::new(fixture.clone())).unwrap();
        let expected = baseline.search(&case.query, case.params).unwrap();

        let mut optimized = VectorIndexReader::open(Cursor::new(fixture)).unwrap();
        optimized.optimize_for_search().unwrap();
        let actual = optimized.search(&case.query, case.params).unwrap();

        assert_eq!(actual.0, expected.0, "{} optimized ids", case.name);
        assert_eq!(
            actual.1.len(),
            expected.1.len(),
            "{} optimized distance count",
            case.name
        );
        for (actual, expected) in actual.1.iter().zip(expected.1.iter()) {
            assert!(
                (actual - expected).abs() < 1e-4,
                "{} optimized distance {} should match {}",
                case.name,
                actual,
                expected
            );
        }
    }
}

#[test]
#[ignore]
fn print_storage_format_v1_fixture_hex() {
    for case in fixture_cases() {
        println!("-- {} --", case.name);
        let fixture = bytes_to_hex(&(case.build)());
        println!("{}", fixture);
        if std::env::var_os("UPDATE_STORAGE_FORMAT_FIXTURES").is_some() {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(format!("{}.hex", case.name));
            std::fs::write(path, fixture).unwrap();
        }
    }
}

fn fixture_cases() -> Vec<FixtureCase> {
    vec![
        FixtureCase {
            name: "ivf_flat_v1",
            fixture_hex: include_str!("fixtures/ivf_flat_v1.hex"),
            build: build_ivf_flat_fixture,
            index_type: IndexType::IvfFlat,
            dimension: 2,
            nlist: 2,
            metric: MetricType::L2,
            total_vectors: 3,
            pq_m: None,
            pq_bits: None,
            query: vec![0.0, 0.0],
            params: VectorSearchParams::new(2, 2),
            expected_first_id: 7,
        },
        FixtureCase {
            name: "ivf_pq_v1",
            fixture_hex: include_str!("fixtures/ivf_pq_v1.hex"),
            build: build_ivf_pq_fixture,
            index_type: IndexType::IvfPq,
            dimension: 1,
            nlist: 2,
            metric: MetricType::L2,
            total_vectors: 3,
            pq_m: Some(1),
            pq_bits: Some(8),
            query: vec![0.0],
            params: VectorSearchParams::new(2, 2),
            expected_first_id: 10,
        },
        FixtureCase {
            name: "ivf_pq_4bit_v1",
            fixture_hex: include_str!("fixtures/ivf_pq_4bit_v1.hex"),
            build: build_ivf_pq_4bit_fixture,
            index_type: IndexType::IvfPq,
            dimension: 2,
            nlist: 2,
            metric: MetricType::L2,
            total_vectors: 3,
            pq_m: Some(2),
            pq_bits: Some(4),
            query: vec![0.0, 0.0],
            params: VectorSearchParams::new(2, 2),
            expected_first_id: 5,
        },
        FixtureCase {
            name: "ivf_rq_v1",
            fixture_hex: include_str!("fixtures/ivf_rq_v1.hex"),
            build: build_ivf_rq_fixture,
            index_type: IndexType::IvfRq,
            dimension: 8,
            nlist: 2,
            metric: MetricType::L2,
            total_vectors: 3,
            pq_m: None,
            pq_bits: None,
            query: vec![0.0; 8],
            params: VectorSearchParams::new(2, 2),
            expected_first_id: 42,
        },
        FixtureCase {
            name: "ivf_sq_v1",
            fixture_hex: include_str!("fixtures/ivf_sq_v1.hex"),
            build: build_ivf_sq_fixture,
            index_type: IndexType::IvfSq,
            dimension: 2,
            nlist: 2,
            metric: MetricType::L2,
            total_vectors: 2,
            pq_m: None,
            pq_bits: Some(8),
            query: vec![0.0, 0.0],
            params: VectorSearchParams::new(1, 2),
            expected_first_id: 7,
        },
        FixtureCase {
            name: "diskann_v1",
            fixture_hex: include_str!("fixtures/diskann_v1.hex"),
            build: build_diskann_fixture,
            index_type: IndexType::DiskAnn,
            dimension: 1,
            nlist: 1,
            metric: MetricType::L2,
            total_vectors: 1,
            pq_m: Some(1),
            pq_bits: Some(8),
            query: vec![0.0],
            params: VectorSearchParams::with_l_search(1, 4),
            expected_first_id: 7,
        },
        FixtureCase {
            name: "diskann_compact_multipage_v1",
            fixture_hex: include_str!("fixtures/diskann_compact_multipage_v1.hex"),
            build: build_diskann_compact_multipage_fixture,
            index_type: IndexType::DiskAnn,
            dimension: 16,
            nlist: 1,
            metric: MetricType::L2,
            total_vectors: 513,
            pq_m: Some(4),
            pq_bits: Some(8),
            query: vec![0.0; 16],
            params: VectorSearchParams::with_l_search(1, 513),
            expected_first_id: 10_000,
        },
        FixtureCase {
            name: "diskann_interleaved_4bit_v1",
            fixture_hex: include_str!("fixtures/diskann_interleaved_4bit_v1.hex"),
            build: build_diskann_interleaved_4bit_fixture,
            index_type: IndexType::DiskAnn,
            dimension: 4,
            nlist: 1,
            metric: MetricType::L2,
            total_vectors: 17,
            pq_m: Some(2),
            pq_bits: Some(4),
            query: vec![0.0; 4],
            params: VectorSearchParams::with_l_search(1, 16),
            expected_first_id: -500,
        },
        FixtureCase {
            name: "diskann_raw_row_ids_v1",
            fixture_hex: include_str!("fixtures/diskann_raw_row_ids_v1.hex"),
            build: build_diskann_raw_row_ids_fixture,
            index_type: IndexType::DiskAnn,
            dimension: 1,
            nlist: 1,
            metric: MetricType::L2,
            total_vectors: 3,
            pq_m: Some(1),
            pq_bits: Some(8),
            query: vec![0.0],
            params: VectorSearchParams::with_l_search(1, 4),
            expected_first_id: i64::MIN,
        },
    ]
}

fn build_diskann_fixture() -> Vec<u8> {
    let mut index = DiskAnnIndex::new(
        1,
        MetricType::L2,
        1,
        DiskAnnBuildParams {
            max_degree: 2,
            build_search_list_size: 4,
            alpha: 1.2,
            seed: 42,
            memory_budget_bytes: 1024 * 1024,
            ..DiskAnnBuildParams::default()
        },
    );
    index.pq.centroids = (0..256).map(|code| code as f32 * 0.25).collect();
    index.pq.rebuild_norms_cache();
    index.ids = vec![7];
    index.vectors = vec![0.0];
    write_diskann_fixture(index)
}

fn build_diskann_compact_multipage_fixture() -> Vec<u8> {
    let dimension = 16;
    let count = 513;
    let mut index = DiskAnnIndex::new(
        dimension,
        MetricType::L2,
        4,
        DiskAnnBuildParams {
            max_degree: 32,
            build_search_list_size: 64,
            alpha: 1.2,
            seed: 7,
            memory_budget_bytes: 16 * 1024 * 1024,
            storage_layout: DiskAnnStorageLayout::Compact,
            build_distance: DiskAnnBuildDistance::FullPrecision,
            ..DiskAnnBuildParams::default()
        },
    );
    index.pq.centroids = (0..dimension)
        .flat_map(|coordinate| (0..256).map(move |code| code as f32 + coordinate as f32 * 0.001))
        .collect();
    index.pq.rebuild_norms_cache();
    index.ids = (0..count).map(|node| 10_000 + node as i64 * 7).collect();
    index.vectors = (0..count)
        .flat_map(|node| {
            (0..dimension).map(move |coordinate| {
                if node == 0 {
                    0.0
                } else {
                    // Integer-valued coordinates keep every squared L2
                    // accumulation exactly representable in f32. The golden
                    // graph is therefore stable across SIMD implementations
                    // and coverage instrumentation.
                    let mut value = (node as u32)
                        .wrapping_mul(747_796_405)
                        .wrapping_add((coordinate as u32).wrapping_mul(2_891_336_453))
                        .wrapping_add(277_803_737);
                    value = (value ^ (value >> 16)).wrapping_mul(2_246_822_519);
                    value ^= value >> 13;
                    1.0 + (value % 512) as f32
                }
            })
        })
        .collect();
    write_diskann_fixture(index)
}

fn build_diskann_interleaved_4bit_fixture() -> Vec<u8> {
    let dimension = 4;
    let count = 17;
    let mut index = DiskAnnIndex::with_pq_bits(
        dimension,
        MetricType::L2,
        2,
        4,
        DiskAnnBuildParams {
            max_degree: 4,
            build_search_list_size: 8,
            alpha: 1.2,
            seed: 11,
            memory_budget_bytes: 1024 * 1024,
            storage_layout: DiskAnnStorageLayout::Interleaved,
            raw_vector_encoding: DiskAnnRawVectorEncoding::F16,
            ..DiskAnnBuildParams::default()
        },
    );
    index.pq.centroids = (0..dimension)
        .flat_map(|coordinate| {
            (0..16).map(move |code| code as f32 * 0.5 + coordinate as f32 * 0.001)
        })
        .collect();
    index.pq.rebuild_norms_cache();
    index.ids = (0..count).map(|node| -500 + node as i64 * 11).collect();
    index.vectors = (0..count)
        .flat_map(|node| {
            (0..dimension)
                .map(move |coordinate| node as f32 + coordinate as f32 * 0.001 * node as f32)
        })
        .collect();
    write_diskann_fixture(index)
}

fn build_diskann_raw_row_ids_fixture() -> Vec<u8> {
    let mut index = DiskAnnIndex::new(
        1,
        MetricType::L2,
        1,
        DiskAnnBuildParams {
            max_degree: 2,
            build_search_list_size: 4,
            alpha: 1.2,
            seed: 13,
            memory_budget_bytes: 1024 * 1024,
            raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
            ..DiskAnnBuildParams::default()
        },
    );
    index.pq.centroids = (0..256).map(|code| code as f32).collect();
    index.pq.rebuild_norms_cache();
    index.ids = vec![i64::MIN, 0, i64::MAX];
    index.vectors = vec![0.0, 1.0, 2.0];
    write_diskann_fixture(index)
}

fn write_diskann_fixture(index: DiskAnnIndex) -> Vec<u8> {
    // Pin the builder to one worker so the golden graph is independent of the
    // host's Rayon worker count and scheduling.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let mut buf = Vec::new();
            write_diskann_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
            buf
        })
}

fn build_ivf_flat_fixture() -> Vec<u8> {
    let mut index = IVFFlatIndex::new(2, 2, MetricType::L2);
    index.quantizer_centroids = vec![0.0, 0.0, 10.0, 10.0];
    index.ids = vec![vec![42, 7], vec![99]];
    index.vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![10.0, 10.0]];
    let mut buf = Vec::new();
    write_ivfflat_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
    buf
}

fn build_ivf_pq_fixture() -> Vec<u8> {
    let mut index = IVFPQIndex::new(1, 2, 1, MetricType::L2, false);
    index.quantizer_centroids = vec![0.0, 10.0];
    index.pq.centroids = (0..index.pq.ksub).map(|code| code as f32 * 0.25).collect();
    index.pq.rebuild_norms_cache();
    index.ids = vec![vec![20, 10], vec![30]];
    index.codes = vec![vec![1, 0], vec![0]];

    let mut buf = Vec::new();
    write_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
    buf
}

fn build_ivf_pq_4bit_fixture() -> Vec<u8> {
    let mut index = IVFPQIndex::with_nbits(2, 2, 2, 4, MetricType::L2, false);
    index.quantizer_centroids = vec![0.0, 0.0, 10.0, 10.0];
    index.pq.centroids = (0..index.pq.m)
        .flat_map(|_| (0..index.pq.ksub).map(|code| code as f32 * 0.5))
        .collect();
    index.pq.rebuild_norms_cache();
    index.ids = vec![vec![8, 5], vec![30]];
    index.codes = vec![vec![0x11, 0x00], vec![0x00]];

    let mut buf = Vec::new();
    write_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
    buf
}

fn build_ivf_rq_fixture() -> Vec<u8> {
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    ];
    let mut index = IVFRQIndex::with_bits(8, 2, 4, MetricType::L2);
    index.train(&data, 3);
    index.add(&data, &[42, 7, 99], 3);

    let mut buf = Vec::new();
    write_ivfrq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
    buf
}

fn build_ivf_sq_fixture() -> Vec<u8> {
    let sq = ScalarQuantizer::with_dimension_bounds(2, vec![0.0, 0.0], vec![1.0, 1.0]);
    let mut index = IVFSQIndex::new(2, 2, MetricType::L2);
    index.quantizer_centroids = vec![0.0, 0.0, 10.0, 10.0];
    index.sq = sq.clone();
    index.list_sqs = vec![sq; 2];
    index.ids = vec![vec![7], vec![99]];
    index.codes = vec![vec![0, 0], vec![0, 0]];

    let mut buf = Vec::new();
    write_ivfsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
    buf
}

fn assert_metadata(metadata: &VectorIndexMetadata, case: &FixtureCase) {
    assert_eq!(
        metadata.index_type, case.index_type,
        "{} index type",
        case.name
    );
    assert_eq!(
        metadata.dimension, case.dimension,
        "{} dimension",
        case.name
    );
    assert_eq!(metadata.nlist, case.nlist, "{} nlist", case.name);
    assert_eq!(metadata.metric, case.metric, "{} metric", case.name);
    assert_eq!(
        metadata.total_vectors, case.total_vectors,
        "{} total vectors",
        case.name
    );
    assert_eq!(metadata.pq_m, case.pq_m, "{} pq m", case.name);
    assert_eq!(metadata.pq_bits, case.pq_bits, "{} pq bits", case.name);
    assert_eq!(
        metadata.rq_bits,
        (case.index_type == IndexType::IvfRq).then_some(4),
        "{} rq bits",
        case.name
    );
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let digits: String = hex.chars().filter(|ch| !ch.is_whitespace()).collect();
    assert!(
        digits.len().is_multiple_of(2),
        "fixture hex must contain complete bytes"
    );
    digits
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let byte = std::str::from_utf8(pair).unwrap();
            u8::from_str_radix(byte, 16).unwrap()
        })
        .collect()
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut hex = String::new();
    for (idx, byte) in bytes.iter().enumerate() {
        if idx > 0 {
            if idx.is_multiple_of(32) {
                hex.push('\n');
            } else {
                hex.push(' ');
            }
        }
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    hex.push('\n');
    hex
}
