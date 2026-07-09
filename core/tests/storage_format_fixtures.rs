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
use paimon_vindex_core::hnsw::HnswBuildParams;
use paimon_vindex_core::index::{
    IndexType, VectorIndexMetadata, VectorIndexReader, VectorSearchParams,
};
use paimon_vindex_core::io::{write_index, PosWriter};
use paimon_vindex_core::ivfflat::IVFFlatIndex;
use paimon_vindex_core::ivfflat_io::write_ivfflat_index;
use paimon_vindex_core::ivfhnswflat::IVFHNSWFlatIndex;
use paimon_vindex_core::ivfhnswflat_io::write_ivfhnswflat_index;
use paimon_vindex_core::ivfhnswsq::IVFHNSWSQIndex;
use paimon_vindex_core::ivfhnswsq_io::write_ivfhnswsq_index;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use paimon_vindex_core::ivfrq::IVFRQIndex;
use paimon_vindex_core::ivfrq_io::write_ivfrq_index;
use paimon_vindex_core::rq::RQCodeFactors;
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
    hnsw: Option<HnswBuildParams>,
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
        println!("{}", bytes_to_hex(&(case.build)()));
    }
}

fn fixture_cases() -> Vec<FixtureCase> {
    let hnsw = HnswBuildParams {
        m: 2,
        ef_construction: 8,
        max_level: 3,
    };
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
            hnsw: None,
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
            hnsw: None,
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
            hnsw: None,
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
            hnsw: None,
            query: vec![0.0; 8],
            params: VectorSearchParams::new(2, 2),
            expected_first_id: 7,
        },
        FixtureCase {
            name: "ivf_hnsw_flat_v1",
            fixture_hex: include_str!("fixtures/ivf_hnsw_flat_v1.hex"),
            build: build_ivf_hnsw_flat_fixture,
            index_type: IndexType::IvfHnswFlat,
            dimension: 2,
            nlist: 2,
            metric: MetricType::L2,
            total_vectors: 2,
            pq_m: None,
            hnsw: Some(hnsw),
            query: vec![0.0, 0.0],
            params: VectorSearchParams::with_ef_search(1, 2, 8),
            expected_first_id: 7,
        },
        FixtureCase {
            name: "ivf_hnsw_sq_v1",
            fixture_hex: include_str!("fixtures/ivf_hnsw_sq_v1.hex"),
            build: build_ivf_hnsw_sq_fixture,
            index_type: IndexType::IvfHnswSq,
            dimension: 2,
            nlist: 2,
            metric: MetricType::L2,
            total_vectors: 2,
            pq_m: None,
            hnsw: Some(hnsw),
            query: vec![0.0, 0.0],
            params: VectorSearchParams::with_ef_search(1, 2, 8),
            expected_first_id: 7,
        },
    ]
}

fn build_ivf_flat_fixture() -> Vec<u8> {
    let index = IVFFlatIndex {
        d: 2,
        nlist: 2,
        metric: MetricType::L2,
        quantizer_centroids: vec![0.0, 0.0, 10.0, 10.0],
        ids: vec![vec![42, 7], vec![99]],
        vectors: vec![vec![1.0, 0.0, 0.0, 0.0], vec![10.0, 10.0]],
    };
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
    let mut index = IVFRQIndex::new(8, 2, MetricType::L2);
    index.quantizer_centroids = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    ];
    index.ids = vec![vec![42, 7], vec![99]];
    index.codes = vec![vec![0xFF, 0x00], vec![0x00]];
    index.factors = vec![
        vec![
            RQCodeFactors {
                residual_norm_sqr: 1.0,
                vector_norm_sqr: 1.0,
                dp_multiplier: 0.0,
            },
            RQCodeFactors::zero(),
        ],
        vec![RQCodeFactors::zero()],
    ];

    let mut buf = Vec::new();
    write_ivfrq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
    buf
}

fn build_ivf_hnsw_flat_fixture() -> Vec<u8> {
    let mut index = IVFHNSWFlatIndex::new(2, 2, MetricType::L2, fixture_hnsw_params());
    index.flat.quantizer_centroids = vec![0.0, 0.0, 10.0, 10.0];
    index.flat.ids = vec![vec![7], vec![99]];
    index.flat.vectors = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    index.build_graphs().unwrap();

    let mut buf = Vec::new();
    write_ivfhnswflat_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
    buf
}

fn build_ivf_hnsw_sq_fixture() -> Vec<u8> {
    let sq = ScalarQuantizer::with_dimension_bounds(2, vec![0.0, 0.0], vec![1.0, 1.0]);
    let mut index = IVFHNSWSQIndex::new(2, 2, MetricType::L2, fixture_hnsw_params());
    index.quantizer_centroids = vec![0.0, 0.0, 10.0, 10.0];
    index.sq = sq.clone();
    index.list_sqs = vec![sq; 2];
    index.ids = vec![vec![7], vec![99]];
    index.codes = vec![vec![0, 0], vec![0, 0]];
    index.build_graphs().unwrap();

    let mut buf = Vec::new();
    write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
    buf
}

fn fixture_hnsw_params() -> HnswBuildParams {
    HnswBuildParams {
        m: 2,
        ef_construction: 8,
        max_level: 3,
    }
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
    assert_eq!(
        metadata
            .hnsw
            .map(|params| (params.m, params.ef_construction, params.max_level)),
        case.hnsw
            .map(|params| (params.m, params.ef_construction, params.max_level)),
        "{} hnsw params",
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
