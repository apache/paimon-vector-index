// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::diskann::{
    DiskAnnBuildDistance, DiskAnnBuildParams, DiskAnnRawVectorEncoding, DiskAnnStorageLayout,
};
use crate::index::IndexType;
use crate::read_options::StorageProfile;
use crate::rq::padded_dimension;
use std::io;

pub const MIN_IVF_TRAINING_VECTORS: usize = 65_536;
pub const IVF_TRAINING_VECTORS_PER_LIST: usize = 64;
pub const DEFAULT_IVF_LIST_FRACTION: usize = 16;
pub const DEFAULT_IVF_MIN_NPROBE: usize = 8;
pub const DEFAULT_IVF_CANDIDATES_PER_RESULT: usize = 4;

/// A user setting which is either resolved by the planner or explicitly pinned.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AutoValue<T> {
    #[default]
    Auto,
    Explicit(T),
}

impl<T: Copy> AutoValue<T> {
    pub fn explicit(self) -> Option<T> {
        match self {
            Self::Auto => None,
            Self::Explicit(value) => Some(value),
        }
    }
}

/// Accuracy, storage, and build constraints used by optional offline tuning.
///
/// Recall is deliberately optional: callers without representative queries and
/// ground truth can still resolve deterministic storage constraints, but must
/// not claim a measured recall guarantee.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TuningObjective {
    pub target_recall: Option<f32>,
    pub max_bytes_per_vector: Option<usize>,
    pub max_build_seconds: Option<f64>,
    pub storage_profile: StorageProfile,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallCandidate<T> {
    pub value: T,
    pub measured_recall: f32,
    pub bytes_per_vector: usize,
    pub build_seconds: f64,
}

/// Selects the smallest candidate satisfying every supplied measured target.
///
/// Ties prefer lower build time. No fallback is hidden: callers get `None`
/// when the calibration sample does not contain a candidate meeting the goal.
pub fn select_calibrated_candidate<T: Copy>(
    candidates: &[RecallCandidate<T>],
    objective: TuningObjective,
) -> Option<RecallCandidate<T>> {
    candidates
        .iter()
        .copied()
        .filter(|candidate| {
            objective
                .target_recall
                .is_none_or(|target| candidate.measured_recall >= target)
                && objective
                    .max_bytes_per_vector
                    .is_none_or(|limit| candidate.bytes_per_vector <= limit)
                && objective
                    .max_build_seconds
                    .is_none_or(|limit| candidate.build_seconds <= limit)
        })
        .min_by(|left, right| {
            left.bytes_per_vector
                .cmp(&right.bytes_per_vector)
                .then_with(|| left.build_seconds.total_cmp(&right.build_seconds))
                .then_with(|| right.measured_recall.total_cmp(&left.measured_recall))
        })
}

/// Chooses the nearest power-of-two list count around `sqrt(vector_count)`.
///
/// At least 64 training vectors per list are retained when the corpus is large
/// enough. Tiny corpora degrade to one list instead of creating poorly trained
/// centroids.
pub fn infer_ivf_nlist(vector_count: usize) -> io::Result<usize> {
    if vector_count == 0 {
        return Err(invalid_input(
            "expected vector count must be greater than 0 for automatic nlist",
        ));
    }

    let root = (vector_count as f64).sqrt().max(1.0);
    let lower = floor_power_of_two(root.floor() as usize);
    let upper = lower.checked_mul(2).unwrap_or(lower);
    let mut candidate = if root - lower as f64 <= upper as f64 - root {
        lower
    } else {
        upper
    };
    let training_cap = (vector_count / IVF_TRAINING_VECTORS_PER_LIST).max(1);
    candidate = candidate.min(floor_power_of_two(training_cap));
    Ok(candidate.max(1))
}

pub fn default_training_vector_count(vector_count: usize, nlist: usize) -> io::Result<usize> {
    if vector_count == 0 {
        return Err(invalid_input("vector count must be greater than 0"));
    }
    if nlist == 0 {
        return Err(invalid_input("nlist must be greater than 0"));
    }
    let per_list = nlist
        .checked_mul(IVF_TRAINING_VECTORS_PER_LIST)
        .ok_or_else(|| invalid_input("automatic training vector count overflows usize"))?;
    Ok(vector_count.min(MIN_IVF_TRAINING_VECTORS.max(per_list)))
}

/// Resolves the initial number of IVF lists to probe.
///
/// The policy scans at least 1/16 of coarse lists and enough average list rows
/// for four candidates per requested result. Filtering scales this initial
/// width by inverse selectivity; search wrappers may still expand progressively
/// when invalid/padded results remain.
pub fn infer_ivf_nprobe(
    nlist: usize,
    vector_count: usize,
    top_k: usize,
    matching_count: Option<usize>,
) -> io::Result<usize> {
    if nlist == 0 {
        return Err(invalid_input("nlist must be greater than 0"));
    }
    if vector_count == 0 {
        return Err(invalid_input("vector count must be greater than 0"));
    }
    if top_k == 0 {
        return Err(invalid_input("top_k must be greater than 0"));
    }

    let average_list_rows = vector_count.div_ceil(nlist).max(1);
    let candidate_rows = top_k
        .checked_mul(DEFAULT_IVF_CANDIDATES_PER_RESULT)
        .ok_or_else(|| invalid_input("automatic nprobe candidate count overflows usize"))?;
    let candidate_lists = candidate_rows.div_ceil(average_list_rows);
    let mut nprobe = DEFAULT_IVF_MIN_NPROBE
        .max(nlist.div_ceil(DEFAULT_IVF_LIST_FRACTION))
        .max(candidate_lists)
        .min(nlist);

    if let Some(matching_count) = matching_count {
        if matching_count == 0 {
            return Ok(1);
        }
        nprobe = ((nprobe as u128)
            .saturating_mul(vector_count as u128)
            .div_ceil(matching_count as u128)
            .min(nlist as u128)) as usize;
    }
    Ok(nprobe.clamp(1, nlist))
}

pub fn infer_diskann_l_search(top_k: usize) -> io::Result<usize> {
    if top_k == 0 {
        return Err(invalid_input("top_k must be greater than 0"));
    }
    Ok(top_k.saturating_mul(2).max(100))
}

/// Returns the largest supported RQ bit width fitting an approximate per-row
/// budget. Multi-bit RQ stores five f32 factors (20 bytes) in addition to
/// padded bitplanes; one-bit RQ needs only two factors (8 bytes).
pub fn infer_rq_bits(dimension: usize, max_bytes_per_vector: usize) -> io::Result<usize> {
    if dimension == 0 {
        return Err(invalid_input("dimension must be greater than 0"));
    }
    let padded = padded_dimension(dimension);
    for bits in (1..=8).rev() {
        let factor_bytes = if bits == 1 { 8 } else { 20 };
        let code_bytes = padded
            .checked_mul(bits)
            .and_then(|value| value.checked_div(8))
            .and_then(|value| value.checked_add(factor_bytes))
            .ok_or_else(|| invalid_input("RQ row byte estimate overflows usize"))?;
        if code_bytes <= max_bytes_per_vector {
            return Ok(bits);
        }
    }
    Err(invalid_input(format!(
        "max bytes per vector {max_bytes_per_vector} cannot fit one-bit RQ codes and factors"
    )))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskAnnBuildPreset {
    FastBuild,
    Balanced,
    HighRecall,
}

pub fn diskann_build_preset(
    preset: DiskAnnBuildPreset,
    dimension: usize,
    storage_profile: StorageProfile,
    memory_budget_bytes: usize,
    seed: u64,
) -> io::Result<DiskAnnBuildParams> {
    if dimension == 0 {
        return Err(invalid_input("dimension must be greater than 0"));
    }
    if memory_budget_bytes == 0 {
        return Err(invalid_input(
            "DiskANN memory budget must be greater than 0",
        ));
    }
    let (max_degree, build_search_list_size, alpha, raw_vector_encoding, build_distance) =
        match preset {
            DiskAnnBuildPreset::FastBuild => (
                48,
                64,
                1.15,
                DiskAnnRawVectorEncoding::F16,
                DiskAnnBuildDistance::ProductQuantized,
            ),
            DiskAnnBuildPreset::Balanced => (
                64,
                100,
                1.2,
                DiskAnnRawVectorEncoding::F16,
                DiskAnnBuildDistance::ProductQuantized,
            ),
            DiskAnnBuildPreset::HighRecall => (
                96,
                200,
                1.2,
                DiskAnnRawVectorEncoding::F32,
                DiskAnnBuildDistance::FullPrecision,
            ),
        };
    let record_bytes = dimension
        .checked_mul(raw_vector_encoding.element_size())
        .and_then(|bytes| bytes.checked_add(max_degree * size_of::<u32>()))
        .ok_or_else(|| invalid_input("DiskANN interleaved row size overflows usize"))?;
    let storage_layout = match storage_profile {
        StorageProfile::Memory | StorageProfile::LocalStorage if record_bytes <= 4096 => {
            DiskAnnStorageLayout::Interleaved
        }
        StorageProfile::Auto
        | StorageProfile::Memory
        | StorageProfile::LocalStorage
        | StorageProfile::RemoteStorage
        | StorageProfile::ObjectStore => DiskAnnStorageLayout::Compact,
    };
    Ok(DiskAnnBuildParams {
        max_degree,
        build_search_list_size,
        alpha,
        seed,
        memory_budget_bytes,
        storage_layout,
        raw_vector_encoding,
        build_distance,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexRecommendation {
    pub index_type: IndexType,
    pub reason: &'static str,
}

/// Advisory only: index type changes persistence, accuracy, and latency
/// semantics, so callers must explicitly accept this recommendation.
pub fn recommend_index(
    vector_count: usize,
    dimension: usize,
    objective: TuningObjective,
) -> io::Result<IndexRecommendation> {
    if vector_count == 0 || dimension == 0 {
        return Err(invalid_input(
            "vector count and dimension must be greater than 0",
        ));
    }
    if vector_count <= 100_000 {
        return Ok(IndexRecommendation {
            index_type: IndexType::IvfFlat,
            reason: "small corpus favors exact values and simple construction",
        });
    }
    if objective
        .max_bytes_per_vector
        .is_some_and(|bytes| bytes <= 32)
    {
        return Ok(IndexRecommendation {
            index_type: IndexType::IvfPq,
            reason: "strict row-size budget favors product quantization",
        });
    }
    if matches!(
        objective.storage_profile,
        StorageProfile::RemoteStorage | StorageProfile::ObjectStore
    ) {
        return Ok(IndexRecommendation {
            index_type: IndexType::IvfRq,
            reason: "remote storage favors a compact one-round IVF scan",
        });
    }
    if objective.target_recall.is_some_and(|recall| recall >= 0.95) {
        return Ok(IndexRecommendation {
            index_type: IndexType::DiskAnn,
            reason: "high recall on memory or local storage favors graph traversal",
        });
    }
    Ok(IndexRecommendation {
        index_type: IndexType::IvfSq,
        reason: "balanced default favors one-byte scalar codes and one-round reads",
    })
}

fn floor_power_of_two(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        1usize << (usize::BITS - 1 - value.leading_zeros())
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_nlist_is_power_of_two_and_keeps_training_density() {
        assert_eq!(infer_ivf_nlist(1_000_000).unwrap(), 1024);
        assert_eq!(infer_ivf_nlist(1_183_514).unwrap(), 1024);
        assert_eq!(infer_ivf_nlist(10).unwrap(), 1);
        assert!(infer_ivf_nlist(0).is_err());
    }

    #[test]
    fn balanced_diskann_uses_compact_f16_rerank_vectors() {
        let build = diskann_build_preset(
            DiskAnnBuildPreset::Balanced,
            960,
            StorageProfile::RemoteStorage,
            8 * 1024 * 1024 * 1024,
            42,
        )
        .unwrap();

        assert_eq!(build.raw_vector_encoding, DiskAnnRawVectorEncoding::F16);
        assert_eq!(
            DiskAnnBuildParams::default().raw_vector_encoding,
            DiskAnnRawVectorEncoding::F16
        );
    }

    #[test]
    fn automatic_training_count_is_bounded_and_overflow_checked() {
        assert_eq!(
            default_training_vector_count(1_000_000, 1024).unwrap(),
            65_536
        );
        assert_eq!(
            default_training_vector_count(1_000_000, 4096).unwrap(),
            262_144
        );
        assert_eq!(default_training_vector_count(10_000, 1024).unwrap(), 10_000);
        assert!(default_training_vector_count(usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn automatic_nprobe_scales_for_lists_topk_and_filters() {
        assert_eq!(infer_ivf_nprobe(64, 1_000_000, 10, None).unwrap(), 8);
        assert_eq!(infer_ivf_nprobe(1024, 1_000_000, 10, None).unwrap(), 64);
        assert_eq!(
            infer_ivf_nprobe(1024, 1_000_000, 10, Some(10_000)).unwrap(),
            1024
        );
        assert_eq!(infer_ivf_nprobe(1024, 1_000_000, 10, Some(0)).unwrap(), 1);
    }

    #[test]
    fn calibrated_candidate_never_hides_an_unsatisfied_target() {
        let candidates = [
            RecallCandidate {
                value: 4,
                measured_recall: 0.90,
                bytes_per_vector: 48,
                build_seconds: 2.0,
            },
            RecallCandidate {
                value: 6,
                measured_recall: 0.96,
                bytes_per_vector: 72,
                build_seconds: 3.0,
            },
        ];
        let selected = select_calibrated_candidate(
            &candidates,
            TuningObjective {
                target_recall: Some(0.95),
                max_bytes_per_vector: Some(80),
                ..TuningObjective::default()
            },
        )
        .unwrap();
        assert_eq!(selected.value, 6);
        assert!(select_calibrated_candidate(
            &candidates,
            TuningObjective {
                target_recall: Some(0.99),
                ..TuningObjective::default()
            }
        )
        .is_none());
    }

    #[test]
    fn rq_budget_accounts_for_padding_and_fixed_factors() {
        assert_eq!(infer_rq_bits(100, 88).unwrap(), 4);
        assert_eq!(infer_rq_bits(100, 72).unwrap(), 3);
        assert!(infer_rq_bits(100, 23).is_err());
    }

    #[test]
    fn diskann_preset_chooses_layout_from_deployment_profile() {
        let local = diskann_build_preset(
            DiskAnnBuildPreset::Balanced,
            128,
            StorageProfile::LocalStorage,
            1 << 30,
            42,
        )
        .unwrap();
        assert_eq!(local.storage_layout, DiskAnnStorageLayout::Interleaved);
        let remote = diskann_build_preset(
            DiskAnnBuildPreset::Balanced,
            128,
            StorageProfile::RemoteStorage,
            1 << 30,
            42,
        )
        .unwrap();
        assert_eq!(remote.storage_layout, DiskAnnStorageLayout::Compact);
    }
}
