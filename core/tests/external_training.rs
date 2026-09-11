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
    CpuIvfTrainingAlgorithm, VectorIndexConfig, VectorIndexTrainer, VectorIndexTraining,
    VectorIndexWriter,
};
use paimon_vindex_core::io::PosWriter;
use std::collections::HashMap;

fn config(metric: &str, d: usize, nlist: usize, max_points: usize) -> VectorIndexConfig {
    VectorIndexConfig::from_options(&HashMap::from([
        ("index.type".into(), "ivf_sq".into()),
        ("dimension".into(), d.to_string()),
        ("nlist".into(), nlist.to_string()),
        ("metric".into(), metric.into()),
        (
            "ivf.train.max-points-per-centroid".into(),
            max_points.to_string(),
        ),
    ]))
    .unwrap()
}

fn serialize(training: VectorIndexTraining, data: &[f32], n: usize) -> Vec<u8> {
    let mut writer = VectorIndexWriter::new(training);
    writer
        .add_vectors(&(0..n as i64).collect::<Vec<_>>(), data, n)
        .unwrap();
    let mut bytes = Vec::new();
    writer.write(&mut PosWriter::new(&mut bytes)).unwrap();
    bytes
}

#[test]
fn prepared_cpu_auto_preserves_original_index_bytes() {
    for (metric, n, d, nlist, max_points) in [
        ("l2", 128, 8, 4, 256),
        ("l2", 128, 8, 4, 2),
        ("cosine", 128, 8, 4, 256),
        ("inner_product", 128, 8, 4, 256),
        ("l2", 600, 4, 300, 2),
    ] {
        let data = (0..n * d)
            .map(|i| ((i * 37 % 997) as f32).sin())
            .collect::<Vec<_>>();
        let original =
            VectorIndexTrainer::train(config(metric, d, nlist, max_points), &data, n).unwrap();
        let prepared = VectorIndexTrainer::new(config(metric, d, nlist, max_points))
            .unwrap()
            .add_training_vectors(&data, n)
            .unwrap()
            .prepare_training()
            .unwrap();
        let centers = prepared
            .fit_centroids_cpu(CpuIvfTrainingAlgorithm::Auto, None)
            .unwrap();
        let external = prepared.finish_with_ivf_centroids(centers).unwrap();
        assert_eq!(
            serialize(original, &data, n),
            serialize(external, &data, n),
            "{metric}, nlist={nlist}"
        );
    }
}

#[test]
fn supplied_centers_are_retained_and_sq_is_recalibrated() {
    let data = [-2.0, -2.0, 10.0, 10.0, 12.0, 12.0];
    let centers = vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0];
    let prepared = VectorIndexTrainer::new(config("l2", 2, 3, 256))
        .unwrap()
        .add_training_vectors(&data, 3)
        .unwrap()
        .prepare_training()
        .unwrap();
    let training = prepared.finish_with_ivf_centroids(centers.clone()).unwrap();
    let VectorIndexWriter::IvfSq(index) = VectorIndexWriter::new(training) else {
        unreachable!()
    };
    assert_eq!(index.quantizer_centroids(), centers);
    assert_eq!(index.sq.mins, [-2.0, -2.0]);
    assert_eq!(index.sq.maxs, [2.0, 2.0]);
    assert!(index.ids.iter().all(Vec::is_empty));
    assert!(index
        .list_sqs
        .iter()
        .all(|sq| sq.mins == index.sq.mins && sq.maxs == index.sq.maxs));
}

#[test]
fn prepared_sampling_is_bounded_and_independent_of_batch_boundaries() {
    let data = (0..70_000 * 2).map(|i| i as f32).collect::<Vec<_>>();
    let prepare = |batch_rows: usize| {
        let mut trainer = VectorIndexTrainer::new(config("l2", 2, 2, 3)).unwrap();
        for chunk in data.chunks(batch_rows * 2) {
            trainer
                .add_training_vectors_mut(chunk, chunk.len() / 2)
                .unwrap();
        }
        trainer.prepare_training().unwrap()
    };
    let a = prepare(70_000);
    let b = prepare(511);
    assert_eq!(a.sample(), b.sample());
    assert_eq!(a.sample().len(), 6 * 2);
    assert_eq!(a.calibration_vector_count(), 65_536);
    assert_eq!(a.vectors_seen(), 70_000);
}

#[test]
fn external_centers_reject_wrong_shapes_and_non_finite_values() {
    for centers in [vec![0.0], vec![f32::NAN; 4], vec![f32::INFINITY; 4]] {
        let prepared = VectorIndexTrainer::new(config("l2", 2, 2, 256))
            .unwrap()
            .add_training_vectors(&[0.0, 0.0, 1.0, 1.0], 2)
            .unwrap()
            .prepare_training()
            .unwrap();
        assert!(prepared.finish_with_ivf_centroids(centers).is_err());
    }
}
