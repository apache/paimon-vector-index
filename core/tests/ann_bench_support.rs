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

#[path = "../benches/support/ann_bench_support.rs"]
mod ann_bench_support;

use ann_bench_support::{
    add_fixed_round_latency, default_training_vector_count, inspect_public_dataset,
    parse_storage_case_names, resolve_shape_value, should_isolate_indexes, PublicDatasetShape,
};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let suffix = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "paimon-ann-bench-support-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_i32_records(path: &Path, rows: usize, width: usize) {
    let mut file = File::create(path).unwrap();
    for row in 0..rows {
        file.write_all(&(width as i32).to_le_bytes()).unwrap();
        for column in 0..width {
            file.write_all(&((row * width + column) as i32).to_le_bytes())
                .unwrap();
        }
    }
}

#[test]
fn public_dataset_shape_is_inferred_without_loading_vector_payloads() {
    let directory = TestDirectory::new();
    let base = directory.0.join("base.fvecs");
    let queries = directory.0.join("query.fvecs");
    let truth = directory.0.join("truth.ivecs");
    write_i32_records(&base, 10, 8);
    write_i32_records(&queries, 3, 8);
    write_i32_records(&truth, 3, 5);

    assert_eq!(
        inspect_public_dataset(&base, &queries, &truth).unwrap(),
        PublicDatasetShape {
            vector_count: 10,
            query_count: 3,
            dimension: 8,
            ground_truth_width: 5,
        }
    );
}

#[test]
fn public_dataset_shape_rejects_query_and_ground_truth_mismatches() {
    let directory = TestDirectory::new();
    let base = directory.0.join("base.fvecs");
    let queries = directory.0.join("query.fvecs");
    let truth = directory.0.join("truth.ivecs");
    write_i32_records(&base, 10, 8);
    write_i32_records(&queries, 3, 4);
    write_i32_records(&truth, 2, 5);

    let dimension_error = inspect_public_dataset(&base, &queries, &truth).unwrap_err();
    assert!(dimension_error
        .to_string()
        .contains("base/query fvec dimensions differ"));

    write_i32_records(&queries, 3, 8);
    let count_error = inspect_public_dataset(&base, &queries, &truth).unwrap_err();
    assert!(count_error
        .to_string()
        .contains("query/ground-truth counts differ"));
}

#[test]
fn explicit_public_shape_is_an_assertion_and_generated_shape_keeps_defaults() {
    assert_eq!(
        resolve_shape_value("ANN_D", None, Some(128), 64).unwrap(),
        128
    );
    assert_eq!(
        resolve_shape_value("ANN_D", Some(128), Some(128), 64).unwrap(),
        128
    );
    assert_eq!(resolve_shape_value("ANN_D", None, None, 64).unwrap(), 64);

    let error = resolve_shape_value("ANN_D", Some(960), Some(128), 64).unwrap_err();
    assert!(error
        .to_string()
        .contains("ANN_D=960 does not match public dataset shape 128"));
}

#[test]
fn training_sample_default_scales_with_nlist_and_is_capped_by_the_dataset() {
    assert_eq!(
        default_training_vector_count(1_000_000, 1_024).unwrap(),
        65_536
    );
    assert_eq!(
        default_training_vector_count(1_000_000, 4_096).unwrap(),
        262_144
    );
    assert_eq!(
        default_training_vector_count(10_000, 1_024).unwrap(),
        10_000
    );
    assert!(default_training_vector_count(usize::MAX, usize::MAX).is_err());
}

#[test]
fn only_multi_index_public_parent_runs_require_process_isolation() {
    assert!(should_isolate_indexes(true, 6, false, false));
    assert!(!should_isolate_indexes(false, 6, false, false));
    assert!(!should_isolate_indexes(true, 1, false, false));
    assert!(!should_isolate_indexes(true, 6, true, false));
    assert!(!should_isolate_indexes(true, 6, false, true));
}

#[test]
fn storage_case_selection_defaults_to_all_and_validates_subsets() {
    assert_eq!(
        parse_storage_case_names(None).unwrap(),
        [
            "local_ssd_warm_cache",
            "remote_cache_2ms",
            "object_store_20ms"
        ]
    );
    assert_eq!(
        parse_storage_case_names(Some("object-store-20ms,remote_cache_2ms,object_store_20ms"))
            .unwrap(),
        ["object_store_20ms", "remote_cache_2ms"]
    );
    assert!(parse_storage_case_names(Some("")).is_err());
    assert!(parse_storage_case_names(Some("unknown")).is_err());
}

#[test]
fn fixed_round_latency_adds_the_idealized_dependency_cost() {
    assert_eq!(
        add_fixed_round_latency(Duration::from_millis(7), 8, Duration::from_millis(20)),
        Duration::from_millis(167)
    );
    assert_eq!(
        add_fixed_round_latency(Duration::MAX, usize::MAX, Duration::MAX),
        Duration::MAX
    );
}
