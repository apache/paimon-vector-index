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

//! Shared support for the public ANN benchmark and its focused tests.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::time::Duration;

pub(crate) const DEFAULT_TRAINING_VECTORS: usize = 65_536;
pub(crate) const ALL_STORAGE_CASE_NAMES: [&str; 3] = [
    "local_ssd_warm_cache",
    "remote_cache_2ms",
    "object_store_20ms",
];
const TRAINING_VECTORS_PER_LIST: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PublicDatasetShape {
    pub(crate) vector_count: usize,
    pub(crate) query_count: usize,
    pub(crate) dimension: usize,
    pub(crate) ground_truth_width: usize,
}

pub(crate) fn inspect_public_dataset(
    base: &Path,
    queries: &Path,
    ground_truth: &Path,
) -> io::Result<PublicDatasetShape> {
    let (vector_count, dimension) = inspect_i32_records(base)?;
    let (query_count, query_dimension) = inspect_i32_records(queries)?;
    if query_dimension != dimension {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("base/query fvec dimensions differ: {dimension}/{query_dimension}"),
        ));
    }
    let (ground_truth_count, ground_truth_width) = inspect_i32_records(ground_truth)?;
    if ground_truth_count != query_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("query/ground-truth counts differ: {query_count}/{ground_truth_count}"),
        ));
    }
    Ok(PublicDatasetShape {
        vector_count,
        query_count,
        dimension,
        ground_truth_width,
    })
}

pub(crate) fn resolve_shape_value(
    name: &str,
    explicit: Option<usize>,
    inferred: Option<usize>,
    generated_default: usize,
) -> io::Result<usize> {
    match (explicit, inferred) {
        (Some(value), Some(inferred)) if value != inferred => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name}={value} does not match public dataset shape {inferred}"),
        )),
        (Some(value), _) => Ok(value),
        (None, Some(inferred)) => Ok(inferred),
        (None, None) => Ok(generated_default),
    }
}

pub(crate) fn default_training_vector_count(
    vector_count: usize,
    nlist: usize,
) -> io::Result<usize> {
    let list_training_vectors = nlist
        .checked_mul(TRAINING_VECTORS_PER_LIST)
        .ok_or_else(|| io::Error::other("ANN training sample count overflows usize"))?;
    Ok(vector_count.min(DEFAULT_TRAINING_VECTORS.max(list_training_vectors)))
}

pub(crate) fn should_isolate_indexes(
    has_public_dataset: bool,
    selected_index_count: usize,
    is_child: bool,
    reuses_index: bool,
) -> bool {
    has_public_dataset && selected_index_count > 1 && !is_child && !reuses_index
}

pub(crate) fn parse_storage_case_names(value: Option<&str>) -> io::Result<Vec<&'static str>> {
    let value = value.unwrap_or("all");
    if value.trim().eq_ignore_ascii_case("all") {
        return Ok(ALL_STORAGE_CASE_NAMES.to_vec());
    }
    let mut selected = Vec::new();
    for item in value.split(',') {
        let normalized = item.trim().to_ascii_lowercase().replace('-', "_");
        let Some(name) = ALL_STORAGE_CASE_NAMES
            .iter()
            .copied()
            .find(|name| *name == normalized)
        else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unknown ANN_STORAGE_CASES value '{item}'; expected all or a comma-separated subset of {}",
                    ALL_STORAGE_CASE_NAMES.join(",")
                ),
            ));
        };
        if !selected.contains(&name) {
            selected.push(name);
        }
    }
    if selected.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "ANN_STORAGE_CASES must select at least one storage case",
        ));
    }
    Ok(selected)
}

pub(crate) fn add_fixed_round_latency(
    elapsed: Duration,
    rounds: usize,
    latency_per_round: Duration,
) -> Duration {
    let rounds = u32::try_from(rounds).unwrap_or(u32::MAX);
    elapsed.saturating_add(latency_per_round.saturating_mul(rounds))
}

pub(crate) fn inspect_i32_records(path: &Path) -> io::Result<(usize, usize)> {
    let file = File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;
    let width = i32::from_le_bytes(header);
    if width <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} has invalid record width {width}", path.display()),
        ));
    }
    let record_bytes = 4u64
        .checked_add(
            (width as u64)
                .checked_mul(4)
                .ok_or_else(|| io::Error::other("record byte size overflow"))?,
        )
        .ok_or_else(|| io::Error::other("record byte size overflow"))?;
    if file_bytes == 0 || file_bytes % record_bytes != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} size {file_bytes} is not a multiple of record size {record_bytes}",
                path.display()
            ),
        ));
    }
    let rows = usize::try_from(file_bytes / record_bytes)
        .map_err(|_| io::Error::other("record count exceeds usize"))?;
    Ok((rows, width as usize))
}
