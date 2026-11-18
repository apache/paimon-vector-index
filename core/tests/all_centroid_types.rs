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

use paimon_vindex_core::index::{
    CentroidShardReader, VectorIndexConfig, VectorIndexReader, VectorIndexTrainer,
    VectorIndexTraining, VectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::PosWriter;
use std::collections::HashMap;
use std::io::Cursor;

#[test]
fn centroid_trainer_rejects_models_too_large_for_the_family_reader() {
    // Include shapes where the centroid matrix fits but SQ bounds, PQ rotation,
    // or RQ metadata exceed the reader's allocation limit.
    for (kind, dimension, nlist) in [
        ("ivf_flat", 3072, 65_536),
        ("diskann", 3072, 65_536),
        ("ivf_sq", 1024, 50_000),
        ("ivf_pq", 12_288, 1),
        ("ivf_rq", 1024, 130_561),
    ] {
        let mut options = HashMap::from([
            ("index.type".into(), kind.into()),
            ("dimension".into(), dimension.to_string()),
            ("metric".into(), "l2".into()),
            ("nlist".into(), nlist.to_string()),
        ]);
        if kind == "ivf_pq" {
            options.insert("use-opq".into(), "true".into());
        }
        let error = VectorIndexTrainer::centroid_from_options(&options)
            .err()
            .expect("oversized training model should be rejected before allocation");
        assert!(error.to_string().contains("metadata"), "{kind}: {error}");
    }
}

#[test]
fn frozen_centroid_models_restore_and_preserve_encoding() {
    // Produced by the pre-release Java/JNI writer from [1,2], [2,1], [12,10], [10,12].
    let fixtures: &[(&str, &[u8])] = &[
        (
            "ivf_flat",
            include_bytes!("fixtures/centroid/ivf_flat.model"),
        ),
        ("ivf_pq", include_bytes!("fixtures/centroid/ivf_pq.model")),
        ("ivf_sq", include_bytes!("fixtures/centroid/ivf_sq.model")),
        ("ivf_rq", include_bytes!("fixtures/centroid/ivf_rq.model")),
        ("diskann", include_bytes!("fixtures/centroid/diskann.model")),
    ];
    for &(kind, bytes) in fixtures {
        let model = VectorIndexTraining::deserialize(bytes).unwrap();
        assert_eq!(model.index_type().as_str(), kind);
        assert_eq!(model.nlist().unwrap(), 2);
        assert_eq!(model.assign_centroid(&[1., 1.]).unwrap(), 0);
        assert_eq!(
            model.find_nearest_centroids(&[11., 11.], 2).unwrap(),
            [1, 0]
        );
        assert_eq!(model.serialize().unwrap(), bytes);
        let payload_end = if kind == "ivf_pq" {
            bytes.len()
        } else {
            let len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
            32 + len
        };
        // Each empty list has an EOF offset, zero rows and zero encoded ID bytes.
        for field in [0, 8, 12] {
            let mut invalid_entry = bytes.to_vec();
            invalid_entry[payload_end - 16 + field] ^= 1;
            assert!(VectorIndexTraining::deserialize(&invalid_entry).is_err());
        }
    }
}

#[test]
fn centroid_size_objective_accounts_for_repeated_models() {
    for (kind, budget) in [("ivf_flat", "525"), ("diskann", "610")] {
        let mut options: HashMap<String, String> = [
            ("index.type", kind),
            ("dimension", "128"),
            ("metric", "l2"),
            ("expected-vector-count", "1000000"),
            ("max-bytes-per-vector", budget),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
        if kind == "diskann" {
            for (k, v) in [
                ("pq.m", "32"),
                ("pq.bits", "8"),
                ("diskann.max-degree", "8"),
                ("diskann.build-search-list-size", "16"),
                ("diskann.raw-vector-encoding", "f32"),
            ] {
                options.insert(k.into(), v.into());
            }
        } else {
            options.insert("nlist".into(), "256".into());
        }
        assert!(VectorIndexConfig::from_options(&options).is_ok());
        options.insert("nlist".into(), "256".into());
        let error = VectorIndexTrainer::centroid_from_options(&options)
            .err()
            .unwrap();
        assert!(error.to_string().contains("estimated persisted size"));
        options.insert("max-bytes-per-vector".into(), "700".into());
        assert!(VectorIndexTrainer::centroid_from_options(&options).is_ok());
    }
}

#[test]
fn diskann_centroid_size_objective_includes_per_graph_page_alignment() {
    for layout in ["compact", "interleaved"] {
        let options = [
            ("index.type", "diskann"),
            ("dimension", "8"),
            ("metric", "l2"),
            ("pq.bits", "4"),
            ("pq.m", "2"),
            ("diskann.storage-layout", layout),
            ("expected-vector-count", "16"),
            ("nlist", "16"),
            // Each one-row graph needs a header page, PQ metadata and an adjacency page.
            ("max-bytes-per-vector", "8192"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
        let error = VectorIndexTrainer::centroid_from_options(&options)
            .err()
            .expect("a two-page per-graph budget is too small");
        assert!(
            error.to_string().contains("estimated persisted size"),
            "{error}"
        );
    }
}

#[test]
fn diskann_cosine_accepts_appends_after_writing_a_centroid_shard() {
    let options = [
        ("index.type", "diskann"),
        ("dimension", "2"),
        ("metric", "cosine"),
        ("pq.m", "1"),
        ("diskann.raw-vector-encoding", "f16"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let model =
        VectorIndexTrainer::new_centroid(VectorIndexConfig::from_options(&options).unwrap(), 1)
            .unwrap()
            .add_training_vectors(&[1., 0., 0., 100_000.], 2)
            .unwrap()
            .finish()
            .unwrap();
    let mut writer = model.create_centroid_writer(0).unwrap();
    writer.add_centroid_vectors(&[11], &[1., 0.], 1).unwrap();
    writer.write(&mut PosWriter::new(Vec::new())).unwrap();
    writer
        .add_centroid_vectors(&[42], &[0., 100_000.], 1)
        .unwrap();
    let mut bytes = Vec::new();
    writer.write(&mut PosWriter::new(&mut bytes)).unwrap();
    let mut reader = CentroidShardReader::open(Cursor::new(bytes), 0).unwrap();
    let (ids, _) = reader
        .search(&[0., 1.], VectorSearchParams::with_l_search(2, 2), None)
        .unwrap();
    assert_eq!(ids, [42, 11]);
}

#[test]
fn diskann_router_rejects_noncanonical_options_of_the_same_length() {
    let mut bytes = include_bytes!("fixtures/centroid/diskann.model").to_vec();
    // Reorder two valid JSON entries without changing the byte count or semantics.
    let payload_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let start = 32 + payload_len;
    let fields = std::str::from_utf8(&bytes[start + 1..bytes.len() - 1]).unwrap();
    let mut fields: Vec<_> = fields.split(',').collect();
    fields.swap(0, 1);
    let changed = fields.join(",");
    let end = bytes.len() - 1;
    bytes[start + 1..end].copy_from_slice(changed.as_bytes());
    assert!(VectorIndexTraining::deserialize(&bytes).is_err());
}

#[test]
fn all_index_models_roundtrip_route_and_filter_absolute_ids() {
    let data: Vec<f32> = (0..512)
        .flat_map(|i| {
            [
                (i / 128) as f32 * 100.0 + (i % 128) as f32 * 0.17,
                (i % 19) as f32,
                (i % 7) as f32 + 1.0,
                3.0,
            ]
        })
        .collect();
    for kind in ["ivf_flat", "ivf_pq", "ivf_sq", "ivf_rq", "diskann"] {
        for metric in ["l2", "cosine", "inner_product"] {
            let mut options: HashMap<String, String> =
                [("index.type", kind), ("dimension", "4"), ("metric", metric)]
                    .into_iter()
                    .map(|(k, v)| (k.into(), v.into()))
                    .collect();
            if kind != "diskann" {
                options.insert("nlist".into(), "4".into());
            }
            if kind == "ivf_pq" || kind == "diskann" {
                options.insert("pq.m".into(), "2".into());
            }
            if kind == "ivf_rq" {
                options.insert("rq.bits".into(), "3".into());
            }
            if kind == "diskann" {
                options.insert("diskann.max-degree".into(), "8".into());
                options.insert("diskann.build-search-list-size".into(), "16".into());
                options.insert("diskann.raw-vector-encoding".into(), "f32".into());
            }
            let config = VectorIndexConfig::from_options(&options).unwrap();
            let mut trainer = VectorIndexTrainer::new_centroid(config, 4).unwrap();
            trainer.add_training_vectors_mut(&data, 512).unwrap();
            let model = trainer.finish().unwrap();
            let payload = model.serialize().unwrap();
            let restored = VectorIndexTraining::deserialize(&payload).unwrap();
            assert_eq!(restored.index_type(), model.index_type());
            assert_eq!(restored.nlist().unwrap(), 4);
            assert_eq!(restored.serialize().unwrap(), payload);
            let all_ids: Vec<i64> = (0..512).map(|i| (1i64 << 35) + i as i64 * 101).collect();
            let mut whole =
                VectorIndexWriter::new(VectorIndexTraining::deserialize(&payload).unwrap());
            whole.add_vectors(&all_ids, &data, 512).unwrap();
            let mut whole_bytes = Vec::new();
            whole.write(&mut PosWriter::new(&mut whole_bytes)).unwrap();
            let mut whole_reader = VectorIndexReader::open(Cursor::new(whole_bytes)).unwrap();
            let whole_params = if kind == "diskann" {
                VectorSearchParams::with_l_search(512, 512)
            } else {
                VectorSearchParams::new(512, 4)
            };
            let (whole_ids, whole_distances) =
                whole_reader.search(&data[..4], whole_params).unwrap();
            let whole_scores: HashMap<_, _> = whole_ids.into_iter().zip(whole_distances).collect();
            let mut shards: Vec<_> = (0..4)
                .map(|c| restored.create_centroid_writer(c).unwrap())
                .collect();
            let mut counts = [0; 4];
            for (i, vector) in data.chunks_exact(4).enumerate() {
                let c = restored.assign_centroid(vector).unwrap();
                assert_eq!(restored.find_nearest_centroids(vector, 1).unwrap(), [c]);
                let id = (1i64 << 35) + i as i64 * 101;
                shards[c].add_centroid_vectors(&[id], vector, 1).unwrap();
                counts[c] += 1;
            }
            for (c, mut writer) in shards.into_iter().enumerate() {
                let mut bytes = Vec::new();
                let written = writer.write(&mut PosWriter::new(&mut bytes));
                if kind == "diskann" && counts[c] == 0 {
                    assert!(written.is_err());
                    continue;
                }
                written.unwrap();
                if kind != "diskann" {
                    assert!(CentroidShardReader::open(Cursor::new(&bytes), 4).is_err());
                    if counts[c] > 0 {
                        assert!(
                            CentroidShardReader::open(Cursor::new(&bytes), (c + 1) % 4).is_err()
                        );
                    }
                }
                let mut reader = CentroidShardReader::open(Cursor::new(bytes), c).unwrap();
                assert_eq!(reader.metadata().total_vectors, counts[c]);
                let params = if kind == "diskann" {
                    VectorSearchParams::with_l_search(512, 512)
                } else {
                    VectorSearchParams::new(512, 1)
                };
                let actual = reader.search(&data[..4], params, None).unwrap();
                assert_eq!(
                    actual.0.iter().filter(|&&id| id >= 0).count(),
                    counts[c] as usize
                );
                for (&id, &score) in actual.0.iter().zip(&actual.1) {
                    if id >= 0 {
                        let reference = whole_scores[&id];
                        assert!(
                            (score - reference).abs() <= 1e-3 * reference.abs().max(1.0),
                            "kind={kind}, metric={metric}, row={id}: {score} vs {reference}"
                        );
                    }
                }
                let mut allowed = roaring::RoaringTreemap::new();
                for &id in &actual.0 {
                    if id >= 0 && id % 2 == 0 {
                        allowed.insert(id as u64);
                    }
                }
                let mut filter = Vec::new();
                allowed.serialize_into(&mut filter).unwrap();
                let filtered = reader.search(&data[..4], params, Some(&filter)).unwrap();
                assert_eq!(
                    filtered.0.iter().filter(|&&id| id >= 0).count(),
                    allowed.len() as usize
                );
                assert!(filtered
                    .0
                    .iter()
                    .all(|&id| id < 0 || allowed.contains(id as u64)));
                let queries = [&data[..4], &data[800..804]].concat();
                let batch = reader
                    .search_batch(&queries, 2, params, Some(&filter))
                    .unwrap();
                for row in 0..2 {
                    let values = &batch.0[row * 512..(row + 1) * 512];
                    assert_eq!(
                        values.iter().filter(|&&id| id >= 0).count(),
                        allowed.len() as usize
                    );
                    assert!(values
                        .iter()
                        .all(|&id| id < 0 || allowed.contains(id as u64)));
                }
            }
            assert!(restored.create_centroid_writer(4).is_err());
            assert!(restored.assign_centroid(&[f32::NAN; 4]).is_err());
            assert!(VectorIndexTraining::deserialize(&payload[..payload.len() - 1]).is_err());
        }
    }
}
