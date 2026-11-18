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
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexReader, VectorIndexTrainer, VectorIndexTraining,
    VectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::PosWriter;
use rand::{rngs::StdRng, Rng, SeedableRng};
use roaring::RoaringTreemap;
use std::io::Cursor;

fn data() -> Vec<f32> {
    (0..256)
        .flat_map(|i| [i as f32 * 0.13, (i % 17) as f32 + 1.0, 2.0, 3.0])
        .collect()
}

#[test]
fn centroid_model_roundtrip_and_routed_search() {
    let data = data();
    for metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
        for opq in [false, true] {
            let config = VectorIndexConfig::ivf_pq(4, 4, metric, opq).unwrap();
            let training = VectorIndexTrainer::train(config, &data, 256).unwrap();
            let payload = training.serialize().unwrap();
            let restored = VectorIndexTraining::deserialize(&payload).unwrap();
            assert_eq!(restored.nlist().unwrap(), 4);
            assert_eq!(
                restored
                    .resolve_nprobe(VectorSearchParams::new(10, 2), 256, None)
                    .unwrap(),
                2
            );
            assert!(restored
                .resolve_nprobe(VectorSearchParams::with_l_search(10, 20), 256, None)
                .is_err());
            assert_eq!(restored.serialize().unwrap(), payload);
            let mut shards: Vec<_> = (0..4)
                .map(|c| restored.create_centroid_writer(c).unwrap())
                .collect();
            let mut counts = [0; 4];
            for (i, vector) in data.chunks_exact(4).enumerate() {
                let centroid = restored.assign_centroid(vector).unwrap();
                assert_eq!(training.assign_centroid(vector).unwrap(), centroid);
                assert_eq!(
                    restored.find_nearest_centroids(vector, 4).unwrap(),
                    training.find_nearest_centroids(vector, 4).unwrap()
                );
                let id = (1i64 << 35) + i as i64 * 101;
                shards[centroid]
                    .add_routed_ivf_centroid_vectors(&[id], vector, 1)
                    .unwrap();
                counts[centroid] += 1;
            }
            // Creating writers must not consume the shared model.
            assert_eq!(restored.serialize().unwrap(), payload);
            for (centroid, mut shard) in shards.into_iter().enumerate() {
                assert!(shard.add_vectors(&[0], &data[..4], 1).is_err());
                let mut bytes = Vec::new();
                shard.write(&mut PosWriter::new(&mut bytes)).unwrap();
                let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();
                assert_eq!(reader.total_vectors(), counts[centroid]);
                let params = VectorSearchParams::new(256, 1);
                // The query may route elsewhere: a routed search must still scan this list.
                let query = &data[..4];
                let actual = reader
                    .search_routed_ivf_shard(query, params, centroid, None)
                    .unwrap();
                let expected = reader
                    .search(query, VectorSearchParams::new(256, 4))
                    .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(
                    actual.0.iter().filter(|&&id| id >= 0).count(),
                    counts[centroid] as usize
                );
                let allowed: RoaringTreemap = actual
                    .0
                    .iter()
                    .copied()
                    .filter(|id| *id >= 0 && id % 2 == 0)
                    .map(|id| id as u64)
                    .collect();
                let mut filter = Vec::new();
                allowed.serialize_into(&mut filter).unwrap();
                let single = reader
                    .search_routed_ivf_shard(query, params, centroid, Some(&filter))
                    .unwrap();
                assert_eq!(
                    single.0.iter().filter(|&&id| id >= 0).count(),
                    allowed.len() as usize
                );
                let queries = [query, query].concat();
                let batch = reader
                    .search_routed_ivf_shard_batch(&queries, 2, params, centroid, Some(&filter))
                    .unwrap();
                for row in 0..2 {
                    assert_eq!(&batch.0[row * 256..(row + 1) * 256], single.0.as_slice());
                    for (a, b) in batch.1[row * 256..(row + 1) * 256].iter().zip(&single.1) {
                        assert!(a == b || (a - b).abs() <= 1e-3 * b.abs().max(1.0));
                    }
                }
            }
        }
    }
}

#[test]
fn centroid_model_rejects_invalid_payload_and_requests() {
    let data = data();
    let training = VectorIndexTrainer::train(
        VectorIndexConfig::ivf_pq(4, 4, MetricType::L2, false).unwrap(),
        &data,
        256,
    )
    .unwrap();
    let payload = training.serialize().unwrap();
    for length in [0, 8, payload.len() - 1] {
        assert!(VectorIndexTraining::deserialize(&payload[..length]).is_err());
    }
    let mut corrupt = payload.clone();
    corrupt[0] ^= 1;
    assert!(VectorIndexTraining::deserialize(&corrupt).is_err());
    let mut trailing = payload;
    trailing.push(0);
    assert!(VectorIndexTraining::deserialize(&trailing).is_err());
    assert!(training.create_centroid_writer(4).is_err());
    assert!(training.assign_centroid(&[f32::NAN; 4]).is_err());
    assert!(training.find_nearest_centroids(&data[..4], 0).is_err());
    assert!(training.find_nearest_centroids(&data[..3], 1).is_err());
    let mut shard = training.create_centroid_writer(0).unwrap();
    assert!(shard
        .add_routed_ivf_centroid_vectors(&[-1], &data[..4], 1)
        .is_err());
    assert!(shard
        .add_routed_ivf_centroid_vectors(&[1], &data[..3], 1)
        .is_err());
    let mut normal = VectorIndexWriter::new(training);
    assert!(normal
        .add_routed_ivf_centroid_vectors(&[1], &data[..4], 1)
        .is_err());

    let options = [
        ("index.type", "diskann"),
        ("dimension", "4"),
        ("pq.m", "1"),
        ("metric", "l2"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let unsupported = VectorIndexTrainer::train(
        VectorIndexConfig::from_options(&options).unwrap(),
        &data,
        256,
    )
    .unwrap();
    assert!(unsupported.serialize().is_err());
    assert!(unsupported.create_centroid_writer(0).is_err());
}

#[test]
fn compressed_shards_match_one_file_with_the_same_model() {
    let (n, d, nlist) = (768, 128, 8);
    let mut rng = StdRng::seed_from_u64(20261117);
    let data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(0.0..255.0)).collect();
    let queries: Vec<f32> = (0..12 * d).map(|_| rng.gen_range(0.0..255.0)).collect();
    let ids: Vec<i64> = (0..n).map(|i| (1i64 << 35) + i as i64 * 101).collect();
    for m in [32, 64] {
        for opq in [false, true] {
            for encoding in ["auto", "canonical"] {
                let options = [
                    ("index.type", "ivf_pq".to_string()),
                    ("dimension", d.to_string()),
                    ("nlist", nlist.to_string()),
                    ("pq.m", m.to_string()),
                    ("metric", "l2".to_string()),
                    ("use-opq", opq.to_string()),
                    ("ivf.pq-encoding", encoding.to_string()),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
                let training = VectorIndexTrainer::train(
                    VectorIndexConfig::from_options(&options).unwrap(),
                    &data,
                    n,
                )
                .unwrap();
                let model =
                    VectorIndexTraining::deserialize(&training.serialize().unwrap()).unwrap();
                let mut whole = VectorIndexWriter::new(training);
                // Deliberately use different insertion batches and orders for the two layouts.
                for start in (0..n).step_by(257) {
                    let end = (start + 257).min(n);
                    whole
                        .add_vectors(&ids[start..end], &data[start * d..end * d], end - start)
                        .unwrap();
                }
                let mut shards = Vec::new();
                for c in 0..nlist {
                    let mut shard = model.create_centroid_writer(c).unwrap();
                    for i in (0..n).rev() {
                        let v = &data[i * d..(i + 1) * d];
                        if model.assign_centroid(v).unwrap() == c {
                            shard
                                .add_routed_ivf_centroid_vectors(&ids[i..i + 1], v, 1)
                                .unwrap();
                        }
                    }
                    shards.push(reader(shard));
                }
                let mut whole = reader(whole);
                for optimized in [false, true] {
                    if optimized {
                        whole.optimize_for_search().unwrap();
                        for shard in &mut shards {
                            shard.optimize_for_search().unwrap();
                        }
                    }
                    for query in queries.chunks_exact(d) {
                        for nprobe in [1, 4, nlist] {
                            let params = VectorSearchParams::new(10, nprobe);
                            let expected = whole.search(query, params).unwrap();
                            let mut candidates = Vec::new();
                            for c in model.find_nearest_centroids(query, nprobe).unwrap() {
                                let (ids, scores) = shards[c]
                                    .search_routed_ivf_shard(query, params, c, None)
                                    .unwrap();
                                candidates
                                    .extend(ids.into_iter().zip(scores).filter(|(id, _)| *id >= 0));
                            }
                            candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
                            candidates.truncate(10);
                            assert_eq!(candidates.iter().map(|p| p.0).collect::<Vec<_>>(), expected.0,
                            "m={m}, opq={opq}, encoding={encoding}, optimized={optimized}, nprobe={nprobe}");
                            for ((_, actual), expected) in candidates.iter().zip(&expected.1) {
                                assert!(
                                    (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn reader(mut writer: VectorIndexWriter) -> VectorIndexReader<Cursor<Vec<u8>>> {
    let mut bytes = Vec::new();
    writer.write(&mut PosWriter::new(&mut bytes)).unwrap();
    VectorIndexReader::open(Cursor::new(bytes)).unwrap()
}
