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

use paimon_vindex_core::index::{VectorIndexConfig, VectorIndexTrainer, VectorIndexWriter};
use paimon_vindex_core::io::PosWriter;
use std::collections::HashMap;

fn writer(metric: &str, d: usize, data: &[f32]) -> VectorIndexWriter {
    let config = VectorIndexConfig::from_options(&HashMap::from([
        ("index.type".into(), "ivf_sq".into()),
        ("dimension".into(), d.to_string()),
        ("nlist".into(), "4".into()),
        ("metric".into(), metric.into()),
        ("ivf.coarse-assignment".into(), "exact".into()),
    ]))
    .unwrap();
    VectorIndexWriter::new(VectorIndexTrainer::train(config, data, data.len() / d).unwrap())
}

fn bytes(writer: &mut VectorIndexWriter) -> Vec<u8> {
    let mut out = Vec::new();
    writer.write(&mut PosWriter::new(&mut out)).unwrap();
    out
}

#[test]
fn externally_assigned_and_encoded_batches_preserve_index_bytes() {
    for metric in ["l2", "cosine", "inner_product"] {
        for d in [7, 8, 9] {
            let n = 1025;
            let data = (0..n * d)
                .map(|i| ((i * 37 % 997) as f32).sin())
                .collect::<Vec<_>>();
            let ids = (0..n).map(|i| 5000 - i as i64 * 17).collect::<Vec<_>>();
            let mut original = writer(metric, d, &data);
            let model = original.ivf_sq_encoding_model().unwrap();
            assert!(model.exact_assignment);
            assert_eq!(model.mins.len(), 4 * d);
            original.add_vectors(&ids, &data, n).unwrap();
            let VectorIndexWriter::IvfSq(index) = &original else {
                unreachable!()
            };
            let mut lists = vec![0u32; n];
            let mut codes = vec![0u8; n * d];
            for list in 0..4 {
                for (local, &id) in index.ids[list].iter().enumerate() {
                    let row = ((5000 - id) / 17) as usize;
                    lists[row] = list as u32;
                    codes[row * d..(row + 1) * d]
                        .copy_from_slice(&index.codes[list][local * d..(local + 1) * d]);
                }
            }
            let mut assigned = writer(metric, d, &data);
            let mut encoded = writer(metric, d, &data);
            for start in (0..n).step_by(127) {
                let end = (start + 127).min(n);
                assigned
                    .add_preassigned_vectors(
                        &ids[start..end],
                        &data[start * d..end * d],
                        &lists[start..end],
                        end - start,
                    )
                    .unwrap();
                encoded
                    .add_encoded_vectors(
                        &ids[start..end],
                        &codes[start * d..end * d],
                        &lists[start..end],
                        end - start,
                    )
                    .unwrap();
            }
            assert_eq!(bytes(&mut original), bytes(&mut assigned), "{metric}, {d}");
            assert_eq!(bytes(&mut original), bytes(&mut encoded), "{metric}, {d}");
        }
    }
}

#[test]
fn invalid_external_batches_do_not_mutate_writer() {
    let data = [0., 0., 1., 1., 2., 2., 3., 3.];
    let mut w = writer("l2", 2, &data);
    w.add_vectors(&[99], &data[..2], 1).unwrap();
    let before = bytes(&mut w);
    assert!(w
        .add_preassigned_vectors(&[1, 2], &data[..4], &[0, 4], 2)
        .is_err());
    assert!(w
        .add_preassigned_vectors(&[1, 2], &[0., 0., f32::NAN, 0.], &[0, 1], 2)
        .is_err());
    assert!(w
        .add_preassigned_vectors(&[1], &data[..4], &[0, 1], 2)
        .is_err());
    assert!(w.add_encoded_vectors(&[1, 2], &[0; 3], &[0, 1], 2).is_err());
    assert!(w
        .add_encoded_vectors(&[1, 2], &[0; 4], &[0, u32::MAX], 2)
        .is_err());
    assert!(w.add_encoded_vectors(&[], &[], &[], 0).is_err());
    assert_eq!(before, bytes(&mut w));
}
