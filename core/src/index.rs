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

use crate::autotune::{
    default_training_vector_count, diskann_build_preset, infer_diskann_l_search, infer_ivf_nlist,
    infer_ivf_nprobe_with_filter_expansion_cap, infer_rq_bits, DiskAnnBuildPreset, TuningObjective,
};
use crate::diskann::{
    diskann_training_sample_limit, validate_diskann_format_configuration,
    validate_diskann_training_budget, DiskAnnBuildDistance, DiskAnnBuildParams, DiskAnnIndex,
    DiskAnnRawVectorEncoding, DiskAnnStorageLayout,
};
use crate::diskann_io::{write_diskann_index, DiskAnnIndexReader, DISKANN_MAGIC};
pub use crate::diskann_search::DiskAnnSearchStats;
use crate::distance::MetricType;
use crate::io::{write_index, IVFPQIndexReader, ReadRequest, SeekRead, SeekWrite, MAGIC};
use crate::ivfflat::IVFFlatIndex;
use crate::ivfflat_io::{
    search_batch_ivfflat_reader, search_batch_ivfflat_reader_roaring_filter, write_ivfflat_index,
    IVFFlatIndexReader, IVFFLAT_MAGIC,
};
use crate::ivfpq::{
    search_batch_reader_roaring_filter_with_reuse_mode_and_budget,
    search_batch_reader_with_reuse_mode_and_budget, search_with_reader,
    search_with_reader_roaring_filter, IVFPQIndex,
};
pub use crate::ivfpq::{IvfPqBatchTableReuseMode, DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES};
use crate::ivfrq::IVFRQIndex;
pub use crate::ivfrq_io::IVFRQSearchStats;
use crate::ivfrq_io::{
    search_batch_ivfrq_reader, search_batch_ivfrq_reader_roaring_filter, write_ivfrq_index,
    IVFRQIndexReader, IVF_RQ_MAGIC,
};
use crate::ivfsq::IVFSQIndex;
use crate::ivfsq_io::{
    search_batch_ivfsq_reader, search_batch_ivfsq_reader_roaring_filter, write_ivfsq_index,
    IVFSQIndexReader, IVF_SQ_MAGIC,
};
pub use crate::read_options::{DeploymentProfile, VectorIndexReadPlan, VectorIndexReaderOptions};
use crate::rq::{is_supported_rq_bits, padded_dimension, DEFAULT_RQ_BITS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roaring::RoaringTreemap;
use std::collections::{HashMap, HashSet};
use std::io::{self, Cursor};

/// Default ratio between the serialized PQ code and the raw `f32` vector.
///
/// At 8 bits per subquantizer this resolves to one PQ subquantizer per four
/// dimensions, for example `m=32` at 128 dimensions and `m=240` at 960.
pub const DEFAULT_PQ_CODE_RATIO: f64 = 0.0625;
const PERSISTED_ROW_ID_ESTIMATE_BYTES: usize = 10;
const MAX_IVF_BATCH_RETRY_BUFFER_BYTES: usize = 64 * 1024 * 1024;

/// Resolve a concrete PQ subquantizer count from a target code/raw byte ratio.
///
/// DiskANN uses balanced chunks, so the nearest value in `1..=dimension` is
/// valid. Ties prefer the larger `m` to avoid silently choosing the
/// lower-recall configuration.
pub fn infer_pq_m(dimension: usize, pq_bits: usize, code_ratio: f64) -> io::Result<usize> {
    validate_positive(dimension, "dimension")?;
    validate_pq_code_ratio(pq_bits, code_ratio)?;

    let target_m = dimension as f64 * 32.0 * code_ratio / pq_bits as f64;
    Ok(target_m.round().clamp(1.0, dimension as f64) as usize)
}

fn infer_uniform_pq_m(dimension: usize, pq_bits: usize, code_ratio: f64) -> io::Result<usize> {
    validate_positive(dimension, "dimension")?;
    validate_pq_code_ratio(pq_bits, code_ratio)?;
    let target_m = dimension as f64 * 32.0 * code_ratio / pq_bits as f64;
    let mut best_m = 1;
    let mut best_distance = f64::INFINITY;
    let mut consider = |candidate: usize| {
        let distance = (candidate as f64 - target_m).abs();
        if distance < best_distance || (distance == best_distance && candidate > best_m) {
            best_m = candidate;
            best_distance = distance;
        }
    };

    let mut divisor = 1;
    while divisor <= dimension / divisor {
        if dimension.is_multiple_of(divisor) {
            consider(divisor);
            let paired = dimension / divisor;
            if paired != divisor {
                consider(paired);
            }
        }
        divisor += 1;
    }
    Ok(best_m)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IndexType {
    IvfFlat = 0,
    IvfPq = 1,
    IvfRq = 4,
    DiskAnn = 5,
    IvfSq = 6,
}

impl IndexType {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::IvfFlat),
            1 => Some(Self::IvfPq),
            4 => Some(Self::IvfRq),
            5 => Some(Self::DiskAnn),
            6 => Some(Self::IvfSq),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::IvfFlat => "ivf_flat",
            Self::IvfPq => "ivf_pq",
            Self::IvfRq => "ivf_rq",
            Self::DiskAnn => "diskann",
            Self::IvfSq => "ivf_sq",
        }
    }
}

#[derive(Debug, Clone)]
pub enum VectorIndexConfig {
    IvfFlat {
        dimension: usize,
        nlist: usize,
        metric: MetricType,
    },
    IvfPq {
        dimension: usize,
        nlist: usize,
        m: usize,
        metric: MetricType,
        use_opq: bool,
    },
    IvfRq {
        dimension: usize,
        nlist: usize,
        bits: usize,
        metric: MetricType,
    },
    IvfSq {
        dimension: usize,
        nlist: usize,
        metric: MetricType,
    },
    DiskAnn {
        dimension: usize,
        metric: MetricType,
        pq_m: usize,
        pq_bits: usize,
        build: DiskAnnBuildParams,
    },
}

impl VectorIndexConfig {
    pub fn from_options(options: &HashMap<String, String>) -> io::Result<Self> {
        let plan = VectorIndexBuildPlan::from_options(options)?;
        if plan.objective.max_build_seconds.is_some() {
            return Err(invalid_input(
                "max-build-seconds requires measured offline calibration through \
                 VectorIndexBuildPlan and select_calibrated_candidate",
            ));
        }
        Ok(plan.config)
    }

    pub fn resolved(&self) -> ResolvedVectorIndexConfig {
        ResolvedVectorIndexConfig::from(self)
    }

    pub fn ivf_pq(
        dimension: usize,
        nlist: usize,
        metric: MetricType,
        use_opq: bool,
    ) -> io::Result<Self> {
        let config = Self::IvfPq {
            dimension,
            nlist,
            m: infer_uniform_pq_m(dimension, 8, DEFAULT_PQ_CODE_RATIO)?,
            metric,
            use_opq,
        };
        validate_config(&config)?;
        Ok(config)
    }

    pub fn disk_ann(
        dimension: usize,
        metric: MetricType,
        pq_bits: usize,
        build: DiskAnnBuildParams,
    ) -> io::Result<Self> {
        let config = Self::DiskAnn {
            dimension,
            metric,
            pq_m: infer_pq_m(dimension, pq_bits, DEFAULT_PQ_CODE_RATIO)?,
            pq_bits,
            build,
        };
        validate_config(&config)?;
        Ok(config)
    }

    pub fn ivf_rq(dimension: usize, nlist: usize, metric: MetricType) -> io::Result<Self> {
        let config = Self::IvfRq {
            dimension,
            nlist,
            bits: DEFAULT_RQ_BITS,
            metric,
        };
        validate_config(&config)?;
        Ok(config)
    }

    pub fn index_type(&self) -> IndexType {
        match self {
            Self::IvfFlat { .. } => IndexType::IvfFlat,
            Self::IvfPq { .. } => IndexType::IvfPq,
            Self::IvfRq { .. } => IndexType::IvfRq,
            Self::IvfSq { .. } => IndexType::IvfSq,
            Self::DiskAnn { .. } => IndexType::DiskAnn,
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::IvfFlat { dimension, .. }
            | Self::IvfPq { dimension, .. }
            | Self::IvfRq { dimension, .. }
            | Self::IvfSq { dimension, .. }
            | Self::DiskAnn { dimension, .. } => *dimension,
        }
    }

    pub fn nlist(&self) -> usize {
        match self {
            Self::IvfFlat { nlist, .. }
            | Self::IvfPq { nlist, .. }
            | Self::IvfRq { nlist, .. }
            | Self::IvfSq { nlist, .. } => *nlist,
            Self::DiskAnn { .. } => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedVectorIndexConfig {
    pub index_type: IndexType,
    pub dimension: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub pq_m: Option<usize>,
    pub pq_bits: Option<usize>,
    pub rq_bits: Option<usize>,
    pub use_opq: bool,
    pub diskann_build: Option<DiskAnnBuildParams>,
}

impl From<&VectorIndexConfig> for ResolvedVectorIndexConfig {
    fn from(config: &VectorIndexConfig) -> Self {
        match config {
            VectorIndexConfig::IvfFlat {
                dimension,
                nlist,
                metric,
            }
            | VectorIndexConfig::IvfSq {
                dimension,
                nlist,
                metric,
            } => Self {
                index_type: config.index_type(),
                dimension: *dimension,
                nlist: *nlist,
                metric: *metric,
                pq_m: None,
                pq_bits: None,
                rq_bits: None,
                use_opq: false,
                diskann_build: None,
            },
            VectorIndexConfig::IvfPq {
                dimension,
                nlist,
                m,
                metric,
                use_opq,
            } => Self {
                index_type: IndexType::IvfPq,
                dimension: *dimension,
                nlist: *nlist,
                metric: *metric,
                pq_m: Some(*m),
                pq_bits: Some(8),
                rq_bits: None,
                use_opq: *use_opq,
                diskann_build: None,
            },
            VectorIndexConfig::IvfRq {
                dimension,
                nlist,
                bits,
                metric,
            } => Self {
                index_type: IndexType::IvfRq,
                dimension: *dimension,
                nlist: *nlist,
                metric: *metric,
                pq_m: None,
                pq_bits: None,
                rq_bits: Some(*bits),
                use_opq: false,
                diskann_build: None,
            },
            VectorIndexConfig::DiskAnn {
                dimension,
                metric,
                pq_m,
                pq_bits,
                build,
            } => Self {
                index_type: IndexType::DiskAnn,
                dimension: *dimension,
                nlist: 1,
                metric: *metric,
                pq_m: Some(*pq_m),
                pq_bits: Some(*pq_bits),
                rq_bits: None,
                use_opq: false,
                diskann_build: Some(*build),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorIndexBuildPlan {
    pub config: VectorIndexConfig,
    pub expected_vector_count: Option<usize>,
    pub objective: TuningObjective,
}

impl VectorIndexBuildPlan {
    pub fn from_options(options: &HashMap<String, String>) -> io::Result<Self> {
        let mut options = ConfigOptions::new(options)?;
        let index_type = parse_index_type_option(&options.required("index.type")?)?;
        let dimension = parse_usize_option("dimension", &options.required("dimension")?)?;
        let expected_vector_count = options
            .optional("expected-vector-count")
            .map(|value| parse_usize_option("expected-vector-count", &value))
            .transpose()?;
        if expected_vector_count == Some(0) {
            return Err(invalid_input(
                "expected-vector-count must be greater than 0",
            ));
        }
        let metric = parse_metric_option(&options.required("metric")?)?;
        let target_recall = options
            .optional("target-recall")
            .map(|value| parse_f32_option("target-recall", &value))
            .transpose()?;
        if target_recall.is_some_and(|recall| !recall.is_finite() || !(0.0..=1.0).contains(&recall))
        {
            return Err(invalid_input("target-recall must be finite and in [0, 1]"));
        }
        let max_bytes_per_vector = options
            .optional("max-bytes-per-vector")
            .map(|value| parse_usize_option("max-bytes-per-vector", &value))
            .transpose()?;
        let max_build_seconds = options
            .optional("max-build-seconds")
            .map(|value| parse_f64_option("max-build-seconds", &value))
            .transpose()?;
        if max_build_seconds.is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0) {
            return Err(invalid_input(
                "max-build-seconds must be finite and greater than 0",
            ));
        }
        let deployment_profile = options
            .optional("deployment-profile")
            .map(|value| parse_deployment_profile_option("deployment-profile", &value))
            .transpose()?
            .unwrap_or(DeploymentProfile::Auto);
        let objective = TuningObjective {
            target_recall,
            max_bytes_per_vector,
            max_build_seconds,
            deployment_profile,
        };

        let config = match index_type {
            IndexType::IvfFlat => VectorIndexConfig::IvfFlat {
                dimension,
                nlist: parse_nlist_options(&mut options, expected_vector_count)?,
                metric,
            },
            IndexType::IvfPq => VectorIndexConfig::IvfPq {
                dimension,
                nlist: parse_nlist_options(&mut options, expected_vector_count)?,
                m: parse_pq_m_options(
                    &mut options,
                    dimension,
                    8,
                    true,
                    max_bytes_per_vector
                        .map(|bytes| persisted_code_budget(bytes, "IVF-PQ"))
                        .transpose()?,
                )?,
                metric,
                use_opq: match options.optional("use-opq") {
                    Some(use_opq) if use_opq.trim() == "auto" => {
                        target_recall.is_some_and(|recall| recall >= 0.9)
                    }
                    Some(use_opq) => parse_bool_option("use-opq", &use_opq)?,
                    None => target_recall.is_some_and(|recall| recall >= 0.9),
                },
            },
            IndexType::IvfRq => {
                let explicit_bits = options
                    .optional("rq.bits")
                    .map(|value| parse_usize_option("rq.bits", &value))
                    .transpose()?;
                let bits = match explicit_bits {
                    Some(bits) => bits,
                    None => max_bytes_per_vector
                        .map(|bytes| {
                            infer_rq_bits(dimension, persisted_code_budget(bytes, "IVF-RQ")?)
                        })
                        .transpose()?
                        .unwrap_or(DEFAULT_RQ_BITS),
                };
                VectorIndexConfig::IvfRq {
                    dimension,
                    nlist: parse_nlist_options(&mut options, expected_vector_count)?,
                    bits,
                    metric,
                }
            }
            IndexType::IvfSq => VectorIndexConfig::IvfSq {
                dimension,
                nlist: parse_nlist_options(&mut options, expected_vector_count)?,
                metric,
            },
            IndexType::DiskAnn => {
                let pq_bits = match options.optional("pq.bits") {
                    Some(value) => parse_usize_option("pq.bits", &value)?,
                    None if max_bytes_per_vector.is_some_and(|bytes| {
                        bytes
                            < dimension
                                .saturating_mul(4)
                                .saturating_add(64 * size_of::<u32>())
                    }) =>
                    {
                        4
                    }
                    None => 8,
                };
                let build = parse_diskann_options(
                    &mut options,
                    dimension,
                    deployment_profile,
                    target_recall,
                    max_bytes_per_vector,
                )?;
                let max_pq_code_bytes = max_bytes_per_vector
                    .map(|bytes| {
                        let non_pq_bytes =
                            estimate_diskann_row_bytes(dimension, 0, pq_bits, build)?;
                        bytes.checked_sub(non_pq_bytes).ok_or_else(|| {
                            invalid_input(format!(
                                "max-bytes-per-vector {bytes} cannot fit DiskANN raw vectors, \
                                 graph edges, and row-ID metadata ({non_pq_bytes} bytes before PQ)"
                            ))
                        })
                    })
                    .transpose()?;
                VectorIndexConfig::DiskAnn {
                    dimension,
                    metric,
                    pq_m: parse_pq_m_options(
                        &mut options,
                        dimension,
                        pq_bits,
                        false,
                        max_pq_code_bytes,
                    )?,
                    pq_bits,
                    build,
                }
            }
        };

        options.reject_unknown()?;
        validate_config(&config)?;
        if let Some(max_bytes) = max_bytes_per_vector {
            validate_persisted_size_objective(&config, expected_vector_count, max_bytes)?;
        }
        Ok(Self {
            config,
            expected_vector_count,
            objective,
        })
    }
}

struct ConfigOptions {
    values: HashMap<String, String>,
    used: HashSet<String>,
}

impl ConfigOptions {
    fn new(options: &HashMap<String, String>) -> io::Result<Self> {
        let mut values = HashMap::new();
        for (key, value) in options {
            let key = key.trim().to_string();
            if key.is_empty() {
                return Err(invalid_input("option key must not be empty"));
            }
            if values.insert(key.clone(), value.clone()).is_some() {
                return Err(invalid_input(format!("duplicate option key '{}'", key)));
            }
        }
        Ok(Self {
            values,
            used: HashSet::new(),
        })
    }

    fn required(&mut self, key: &str) -> io::Result<String> {
        self.optional(key)
            .ok_or_else(|| invalid_input(format!("missing required option '{}'", key)))
    }

    fn optional(&mut self, key: &str) -> Option<String> {
        if let Some(value) = self.values.get(key) {
            self.used.insert(key.to_string());
            Some(value.clone())
        } else {
            None
        }
    }

    fn reject_unknown(&self) -> io::Result<()> {
        let mut unknown = self
            .values
            .keys()
            .filter(|key| !self.used.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        if unknown.is_empty() {
            Ok(())
        } else {
            unknown.sort();
            Err(invalid_input(format!(
                "unknown vector index option(s): {}",
                unknown.join(", ")
            )))
        }
    }
}

fn parse_nlist_options(
    options: &mut ConfigOptions,
    expected_vector_count: Option<usize>,
) -> io::Result<usize> {
    match options.optional("nlist").as_deref().map(str::trim) {
        Some("auto") | None => {
            let vector_count = expected_vector_count.ok_or_else(|| {
                invalid_input(
                    "automatic nlist requires option 'expected-vector-count'; \
                     set nlist explicitly when the final row count is unknown",
                )
            })?;
            infer_ivf_nlist(vector_count)
        }
        Some(value) => parse_usize_option("nlist", value),
    }
}

fn parse_deployment_profile_option(name: &str, value: &str) -> io::Result<DeploymentProfile> {
    match value.trim() {
        "auto" => Ok(DeploymentProfile::Auto),
        "memory" => Ok(DeploymentProfile::Memory),
        "local" | "local_storage" => Ok(DeploymentProfile::LocalStorage),
        "remote" | "remote_storage" => Ok(DeploymentProfile::RemoteStorage),
        "object_store" => Ok(DeploymentProfile::ObjectStore),
        _ => Err(invalid_input(format!(
            "option '{name}' must be auto, memory, local_storage, remote_storage, or object_store"
        ))),
    }
}

fn parse_pq_m_options(
    options: &mut ConfigOptions,
    dimension: usize,
    pq_bits: usize,
    uniform_chunks: bool,
    max_code_bytes: Option<usize>,
) -> io::Result<usize> {
    let explicit_m = options
        .optional("pq.m")
        .map(|value| parse_usize_option("pq.m", &value))
        .transpose()?;
    let explicit_code_ratio = options
        .optional("pq.code-ratio")
        .map(|value| parse_f64_option("pq.code-ratio", &value))
        .transpose()?;
    let code_ratio = explicit_code_ratio
        .or_else(|| {
            max_code_bytes.map(|bytes| {
                let raw_bytes = dimension.saturating_mul(size_of::<f32>()).max(1);
                (bytes as f64 / raw_bytes as f64).min(pq_bits as f64 / 32.0)
            })
        })
        .unwrap_or(DEFAULT_PQ_CODE_RATIO);
    validate_pq_code_ratio(pq_bits, code_ratio)?;

    match explicit_m {
        Some(m) => Ok(m),
        None if uniform_chunks => infer_uniform_pq_m(dimension, pq_bits, code_ratio),
        None => infer_pq_m(dimension, pq_bits, code_ratio),
    }
}

fn persisted_code_budget(max_bytes_per_vector: usize, index_name: &str) -> io::Result<usize> {
    max_bytes_per_vector
        .checked_sub(PERSISTED_ROW_ID_ESTIMATE_BYTES)
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| {
            invalid_input(format!(
                "max-bytes-per-vector {max_bytes_per_vector} cannot fit the persisted \
                 row-ID encoding for {index_name}"
            ))
        })
}

fn estimate_diskann_row_bytes(
    dimension: usize,
    pq_m: usize,
    pq_bits: usize,
    build: DiskAnnBuildParams,
) -> io::Result<usize> {
    let raw_bytes = dimension
        .checked_mul(build.raw_vector_encoding.element_size())
        .ok_or_else(|| invalid_input("DiskANN raw row byte estimate overflows usize"))?;
    let pq_bytes = pq_m
        .checked_mul(pq_bits)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or_else(|| invalid_input("DiskANN PQ row byte estimate overflows usize"))?;
    let adjacency_bytes = build
        .max_degree
        .checked_mul(size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(8))
        .ok_or_else(|| invalid_input("DiskANN adjacency row byte estimate overflows usize"))?;
    raw_bytes
        .checked_add(pq_bytes)
        .and_then(|bytes| bytes.checked_add(adjacency_bytes))
        .and_then(|bytes| bytes.checked_add(PERSISTED_ROW_ID_ESTIMATE_BYTES))
        // Row-ID order and block locators are persisted for filtered lookup.
        .and_then(|bytes| bytes.checked_add(12))
        .ok_or_else(|| invalid_input("DiskANN persisted row byte estimate overflows usize"))
}

fn validate_persisted_size_objective(
    config: &VectorIndexConfig,
    expected_vector_count: Option<usize>,
    max_bytes_per_vector: usize,
) -> io::Result<()> {
    let dimension = config.dimension();
    let nlist = config.nlist();
    let centroid_bytes = nlist
        .checked_mul(dimension)
        .and_then(|values| values.checked_mul(size_of::<f32>()))
        .ok_or_else(|| invalid_input("persisted centroid size estimate overflows usize"))?;
    let list_metadata_bytes = nlist
        .checked_mul(24)
        .ok_or_else(|| invalid_input("persisted list metadata estimate overflows usize"))?;

    let (row_bytes, fixed_bytes) = match config {
        VectorIndexConfig::IvfFlat { .. } => (
            dimension
                .checked_mul(size_of::<f32>())
                .and_then(|bytes| bytes.checked_add(PERSISTED_ROW_ID_ESTIMATE_BYTES))
                .ok_or_else(|| invalid_input("IVF-FLAT row byte estimate overflows usize"))?,
            centroid_bytes.checked_add(list_metadata_bytes),
        ),
        VectorIndexConfig::IvfSq { .. } => (
            dimension
                .checked_add(PERSISTED_ROW_ID_ESTIMATE_BYTES)
                .ok_or_else(|| invalid_input("IVF-SQ row byte estimate overflows usize"))?,
            centroid_bytes
                .checked_add(list_metadata_bytes)
                .and_then(|bytes| {
                    nlist
                        .checked_mul(dimension)
                        .and_then(|values| values.checked_mul(2 * size_of::<f32>()))
                        .and_then(|quantizer_bytes| bytes.checked_add(quantizer_bytes))
                }),
        ),
        VectorIndexConfig::IvfPq { m, use_opq, .. } => {
            let codebook_bytes = dimension
                .checked_mul(256)
                .and_then(|values| values.checked_mul(size_of::<f32>()))
                .ok_or_else(|| invalid_input("IVF-PQ codebook estimate overflows usize"))?;
            let opq_bytes = if *use_opq {
                dimension
                    .checked_mul(dimension)
                    .and_then(|values| values.checked_mul(size_of::<f32>()))
                    .ok_or_else(|| invalid_input("OPQ matrix estimate overflows usize"))?
            } else {
                0
            };
            (
                m.checked_add(PERSISTED_ROW_ID_ESTIMATE_BYTES)
                    .ok_or_else(|| invalid_input("IVF-PQ row byte estimate overflows usize"))?,
                centroid_bytes
                    .checked_add(list_metadata_bytes)
                    .and_then(|bytes| bytes.checked_add(codebook_bytes))
                    .and_then(|bytes| bytes.checked_add(opq_bytes)),
            )
        }
        VectorIndexConfig::IvfRq { bits, .. } => {
            let code_bytes = padded_dimension(dimension)
                .checked_mul(*bits)
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8)
                .ok_or_else(|| invalid_input("IVF-RQ code estimate overflows usize"))?;
            let factor_bytes = if *bits == 1 { 8 } else { 20 };
            (
                code_bytes
                    .checked_add(factor_bytes)
                    .and_then(|bytes| bytes.checked_add(PERSISTED_ROW_ID_ESTIMATE_BYTES))
                    .ok_or_else(|| invalid_input("IVF-RQ row byte estimate overflows usize"))?,
                centroid_bytes.checked_add(list_metadata_bytes),
            )
        }
        VectorIndexConfig::DiskAnn {
            pq_m,
            pq_bits,
            build,
            ..
        } => {
            let ksub = 1usize << pq_bits;
            let codebook_bytes = dimension
                .checked_mul(ksub)
                .and_then(|values| values.checked_mul(size_of::<f32>()))
                .ok_or_else(|| invalid_input("DiskANN codebook estimate overflows usize"))?;
            (
                estimate_diskann_row_bytes(dimension, *pq_m, *pq_bits, *build)?,
                Some(codebook_bytes),
            )
        }
    };
    let fixed_bytes = fixed_bytes
        .ok_or_else(|| invalid_input("persisted fixed-size estimate overflows usize"))?;
    let amortized_fixed_bytes = expected_vector_count
        .map(|count| fixed_bytes.saturating_add(count - 1) / count)
        .unwrap_or(0);
    let estimated_bytes = row_bytes
        .checked_add(amortized_fixed_bytes)
        .ok_or_else(|| invalid_input("persisted per-vector estimate overflows usize"))?;
    if estimated_bytes > max_bytes_per_vector {
        return Err(invalid_input(format!(
            "max-bytes-per-vector {max_bytes_per_vector} cannot be satisfied by {}: \
             estimated persisted size is {estimated_bytes} bytes per vector \
             ({row_bytes} row bytes + {amortized_fixed_bytes} amortized fixed bytes)",
            config.index_type().as_str()
        )));
    }
    Ok(())
}

fn validate_pq_code_ratio(pq_bits: usize, code_ratio: f64) -> io::Result<()> {
    if !matches!(pq_bits, 4 | 8) {
        return Err(invalid_input(format!(
            "pq.bits must be 4 or 8, got {pq_bits}"
        )));
    }
    let max_ratio = pq_bits as f64 / 32.0;
    if !code_ratio.is_finite() || code_ratio <= 0.0 || code_ratio > max_ratio {
        return Err(invalid_input(format!(
            "pq.code-ratio must be finite and in (0, {max_ratio}] for {pq_bits}-bit PQ"
        )));
    }
    Ok(())
}

fn parse_diskann_options(
    options: &mut ConfigOptions,
    dimension: usize,
    deployment_profile: DeploymentProfile,
    target_recall: Option<f32>,
    max_bytes_per_vector: Option<usize>,
) -> io::Result<DiskAnnBuildParams> {
    let defaults = DiskAnnBuildParams::default();
    let preset = match options.optional("diskann.build-preset").as_deref() {
        Some("fast_build") => DiskAnnBuildPreset::FastBuild,
        Some("balanced") => DiskAnnBuildPreset::Balanced,
        Some("high_recall") => DiskAnnBuildPreset::HighRecall,
        Some(value) => {
            return Err(invalid_input(format!(
                "diskann.build-preset must be fast_build, balanced, or high_recall, got '{value}'"
            )))
        }
        None if target_recall.is_some_and(|recall| recall >= 0.97) => {
            DiskAnnBuildPreset::HighRecall
        }
        None if target_recall.is_some_and(|recall| recall <= 0.85) => DiskAnnBuildPreset::FastBuild,
        None => DiskAnnBuildPreset::Balanced,
    };
    let seed = match options.optional("diskann.seed") {
        Some(value) => parse_u64_option("diskann.seed", &value)?,
        None => defaults.seed,
    };
    let memory_budget_bytes = match options.optional("diskann.memory-budget-bytes") {
        Some(value) => parse_usize_option("diskann.memory-budget-bytes", &value)?,
        None => defaults.memory_budget_bytes,
    };
    let preset_values = diskann_build_preset(
        preset,
        dimension,
        deployment_profile,
        memory_budget_bytes,
        seed,
    )?;
    let max_degree = match options.optional("diskann.max-degree") {
        Some(value) => parse_usize_option("diskann.max-degree", &value)?,
        None => preset_values.max_degree,
    };
    Ok(DiskAnnBuildParams {
        max_degree,
        build_search_list_size: match options.optional("diskann.build-search-list-size") {
            Some(value) => parse_usize_option("diskann.build-search-list-size", &value)?,
            None => preset_values.build_search_list_size.max(max_degree),
        },
        alpha: match options.optional("diskann.alpha") {
            Some(value) => parse_f32_option("diskann.alpha", &value)?,
            None => preset_values.alpha,
        },
        seed,
        memory_budget_bytes,
        storage_layout: match options.optional("diskann.storage-layout") {
            Some(value) => match value.trim() {
                "compact" => DiskAnnStorageLayout::Compact,
                "interleaved" => DiskAnnStorageLayout::Interleaved,
                "auto" => preset_values.storage_layout,
                _ => {
                    return Err(invalid_input(
                        "diskann.storage-layout must be auto, compact, or interleaved",
                    ))
                }
            },
            None => preset_values.storage_layout,
        },
        raw_vector_encoding: match options.optional("diskann.raw-vector-encoding") {
            Some(value) => match value.trim() {
                "f32" => DiskAnnRawVectorEncoding::F32,
                "f16" => DiskAnnRawVectorEncoding::F16,
                "auto" => preset_values.raw_vector_encoding,
                _ => {
                    return Err(invalid_input(
                        "diskann.raw-vector-encoding must be auto, f32, or f16",
                    ))
                }
            },
            None if max_bytes_per_vector.is_some_and(|bytes| {
                bytes
                    < dimension
                        .saturating_mul(size_of::<f32>())
                        .saturating_add(max_degree * size_of::<u32>())
            }) =>
            {
                DiskAnnRawVectorEncoding::F16
            }
            None => preset_values.raw_vector_encoding,
        },
        build_distance: match options.optional("diskann.build-distance") {
            Some(value) => {
                match value.trim() {
                    "full_precision" => DiskAnnBuildDistance::FullPrecision,
                    "product_quantized" => DiskAnnBuildDistance::ProductQuantized,
                    "auto" => preset_values.build_distance,
                    _ => return Err(invalid_input(
                        "diskann.build-distance must be auto, full_precision, or product_quantized",
                    )),
                }
            }
            None => preset_values.build_distance,
        },
    })
}

fn parse_index_type_option(value: &str) -> io::Result<IndexType> {
    match value.trim() {
        "ivf_flat" => Ok(IndexType::IvfFlat),
        "ivf_pq" => Ok(IndexType::IvfPq),
        "ivf_rq" => Ok(IndexType::IvfRq),
        "ivf_sq" => Ok(IndexType::IvfSq),
        "diskann" => Ok(IndexType::DiskAnn),
        _ => Err(invalid_input(format!(
            "unknown index.type '{}'; expected ivf_flat, ivf_sq, ivf_pq, ivf_rq, or diskann",
            value
        ))),
    }
}

fn parse_metric_option(value: &str) -> io::Result<MetricType> {
    match value.trim() {
        "l2" => Ok(MetricType::L2),
        "inner_product" => Ok(MetricType::InnerProduct),
        "cosine" => Ok(MetricType::Cosine),
        _ => Err(invalid_input(format!(
            "unknown metric '{}'; expected l2, inner_product, or cosine",
            value
        ))),
    }
}

fn parse_usize_option(name: &str, value: &str) -> io::Result<usize> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| invalid_input(format!("option '{}' must be a positive integer", name)))
}

fn parse_u64_option(name: &str, value: &str) -> io::Result<u64> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| invalid_input(format!("option '{}' must be a non-negative integer", name)))
}

fn parse_f32_option(name: &str, value: &str) -> io::Result<f32> {
    value
        .trim()
        .parse::<f32>()
        .map_err(|_| invalid_input(format!("option '{}' must be a number", name)))
}

fn parse_f64_option(name: &str, value: &str) -> io::Result<f64> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| invalid_input(format!("option '{}' must be a number", name)))
}

fn parse_bool_option(name: &str, value: &str) -> io::Result<bool> {
    match value.trim() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(invalid_input(format!(
            "option '{}' must be true or false",
            name
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SearchWidth {
    Auto = 0,
    IvfNProbe = 1,
    DiskAnnLSearch = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorSearchParams {
    pub top_k: usize,
    pub search_width: SearchWidth,
    pub width: usize,
    /// Caps inverse-selectivity expansion of the initial automatic IVF nprobe.
    ///
    /// `None` preserves unlimited expansion. Lower factors reduce initial search
    /// work but may reduce recall compared with uncapped automatic search.
    /// Progressive search may exceed this initial cap only when filtered results
    /// do not fill `top_k`.
    pub max_initial_filter_expansion_factor: Option<usize>,
    pub ivfpq_batch_table_reuse: IvfPqBatchTableReuseMode,
    pub ivfpq_batch_table_reuse_max_bytes: usize,
}

impl VectorSearchParams {
    pub fn new(top_k: usize, nprobe: usize) -> Self {
        Self {
            top_k,
            search_width: SearchWidth::IvfNProbe,
            width: nprobe,
            max_initial_filter_expansion_factor: None,
            ivfpq_batch_table_reuse: IvfPqBatchTableReuseMode::Auto,
            ivfpq_batch_table_reuse_max_bytes: DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
        }
    }

    pub fn with_l_search(top_k: usize, l_search: usize) -> Self {
        Self {
            top_k,
            search_width: SearchWidth::DiskAnnLSearch,
            width: l_search,
            max_initial_filter_expansion_factor: None,
            ivfpq_batch_table_reuse: IvfPqBatchTableReuseMode::Auto,
            ivfpq_batch_table_reuse_max_bytes: DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
        }
    }

    pub fn automatic(top_k: usize) -> Self {
        Self {
            top_k,
            search_width: SearchWidth::Auto,
            width: 0,
            max_initial_filter_expansion_factor: None,
            ivfpq_batch_table_reuse: IvfPqBatchTableReuseMode::Auto,
            ivfpq_batch_table_reuse_max_bytes: DEFAULT_IVFPQ_BATCH_TABLE_REUSE_MAX_BYTES,
        }
    }

    /// Limits filter-driven expansion of the initial automatic IVF nprobe.
    ///
    /// A factor of 1 keeps the unfiltered automatic width. Lower factors reduce
    /// initial search work but may reduce recall compared with uncapped automatic
    /// search. This setting applies only to automatic IVF search; progressive
    /// expansion occurs only when fewer than `top_k` filtered results are found.
    pub fn with_max_initial_filter_expansion_factor(mut self, factor: usize) -> Self {
        self.max_initial_filter_expansion_factor = Some(factor);
        self
    }

    pub fn with_ivfpq_batch_table_reuse(mut self, mode: IvfPqBatchTableReuseMode) -> Self {
        self.ivfpq_batch_table_reuse = mode;
        self
    }

    pub fn with_ivfpq_batch_table_reuse_max_bytes(mut self, max_bytes: usize) -> Self {
        self.ivfpq_batch_table_reuse_max_bytes = max_bytes;
        self
    }

    pub fn configured_ivf_nprobe(self) -> Option<usize> {
        (self.search_width == SearchWidth::IvfNProbe).then_some(self.width)
    }

    pub fn configured_diskann_l_search(self) -> Option<usize> {
        (self.search_width == SearchWidth::DiskAnnLSearch).then_some(self.width)
    }

    fn validate(self) -> io::Result<()> {
        validate_positive(self.top_k, "top_k")?;
        if let Some(factor) = self.max_initial_filter_expansion_factor {
            validate_positive(factor, "maximum initial filter expansion factor")?;
            if self.search_width != SearchWidth::Auto {
                return Err(invalid_input(
                    "maximum initial filter expansion factor requires automatic IVF search",
                ));
            }
        }
        validate_positive(
            self.ivfpq_batch_table_reuse_max_bytes,
            "IVF-PQ batch table reuse max bytes",
        )
    }

    fn resolve_ivf_nprobe(
        self,
        nlist: usize,
        vector_count: usize,
        matching_count: Option<usize>,
    ) -> io::Result<usize> {
        match self.search_width {
            SearchWidth::Auto => infer_ivf_nprobe_with_filter_expansion_cap(
                nlist,
                vector_count,
                self.top_k,
                matching_count,
                self.max_initial_filter_expansion_factor,
            ),
            SearchWidth::IvfNProbe if self.width > 0 => Ok(self.width.min(nlist)),
            SearchWidth::IvfNProbe => Err(invalid_input("nprobe must be greater than 0")),
            SearchWidth::DiskAnnLSearch => Err(invalid_input(
                "DiskANN l_search cannot be used with an IVF index",
            )),
        }
    }

    #[cfg(test)]
    fn resolve_diskann_l_search(self) -> io::Result<usize> {
        self.resolve_diskann_l_search_with(None)
    }

    fn resolve_diskann_l_search_with(self, calibrated: Option<usize>) -> io::Result<usize> {
        if self.max_initial_filter_expansion_factor.is_some() {
            return Err(invalid_input(
                "maximum initial filter expansion factor is only valid for IVF indexes",
            ));
        }
        match self.search_width {
            SearchWidth::Auto => Ok(calibrated
                .unwrap_or(infer_diskann_l_search(self.top_k)?)
                .max(self.top_k)),
            SearchWidth::DiskAnnLSearch if self.width > 0 => Ok(self.width.max(self.top_k)),
            SearchWidth::DiskAnnLSearch => Err(invalid_input("l_search must be greater than 0")),
            SearchWidth::IvfNProbe => Err(invalid_input(
                "IVF nprobe cannot be used with a DiskANN index",
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorIndexMetadata {
    pub index_type: IndexType,
    pub dimension: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub pq_m: Option<usize>,
    pub pq_bits: Option<usize>,
    pub rq_bits: Option<usize>,
    pub diskann: Option<DiskAnnMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskAnnMetadata {
    pub max_degree: usize,
    pub build_search_list_size: usize,
    pub alpha: f32,
}

pub struct VectorIndexTrainer {
    writer: VectorIndexWriter,
    training_data: Vec<f32>,
    training_vector_count: usize,
    training_vectors_seen: usize,
    training_sample_limit: usize,
    training_rng: StdRng,
}

impl VectorIndexTrainer {
    pub fn new(config: VectorIndexConfig) -> io::Result<Self> {
        let training_sample_limit = match &config {
            VectorIndexConfig::DiskAnn {
                dimension,
                metric,
                pq_m,
                pq_bits,
                build,
            } => diskann_training_sample_limit(
                *dimension,
                *metric,
                *pq_m,
                *pq_bits,
                build.memory_budget_bytes,
            )?,
            _ => default_training_vector_count(usize::MAX, config.nlist())?,
        };
        let training_seed = match &config {
            VectorIndexConfig::DiskAnn { build, .. } => build.seed,
            _ => 1234,
        };
        let writer = VectorIndexWriter::from_config(config)?;
        Ok(Self {
            writer,
            training_data: Vec::new(),
            training_vector_count: 0,
            training_vectors_seen: 0,
            training_sample_limit,
            training_rng: StdRng::seed_from_u64(training_seed),
        })
    }

    pub fn train(
        config: VectorIndexConfig,
        data: &[f32],
        n: usize,
    ) -> io::Result<VectorIndexTraining> {
        Self::new(config)?.add_training_vectors(data, n)?.finish()
    }

    pub fn dimension(&self) -> usize {
        self.writer.dimension()
    }

    pub fn add_training_vectors(mut self, data: &[f32], n: usize) -> io::Result<Self> {
        self.add_training_vectors_mut(data, n)?;
        Ok(self)
    }

    pub fn add_training_vectors_mut(&mut self, data: &[f32], n: usize) -> io::Result<&mut Self> {
        validate_vectors(data, n, self.dimension(), "training data")?;
        let training_vectors_seen = self.training_vectors_seen.checked_add(n).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "training vector count overflows usize",
            )
        })?;
        let dimension = self.writer.dimension();
        for (batch_index, vector) in data.chunks_exact(dimension).enumerate() {
            let stream_index = self.training_vectors_seen + batch_index;
            if stream_index < self.training_sample_limit {
                self.training_data.extend_from_slice(vector);
            } else {
                let replacement = self.training_rng.gen_range(0..=stream_index);
                if replacement < self.training_sample_limit {
                    let start = replacement * dimension;
                    self.training_data[start..start + dimension].copy_from_slice(vector);
                }
            }
        }
        self.training_vector_count = training_vectors_seen.min(self.training_sample_limit);
        self.training_vectors_seen = training_vectors_seen;
        Ok(self)
    }

    pub fn finish(mut self) -> io::Result<VectorIndexTraining> {
        if self.training_vector_count == 0 || self.training_data.is_empty() {
            return Err(invalid_input("no training vectors added"));
        }
        self.writer
            .train_internal(&self.training_data, self.training_vector_count)?;
        Ok(VectorIndexTraining { inner: self.writer })
    }
}

pub struct VectorIndexTraining {
    inner: VectorIndexWriter,
}

impl VectorIndexTraining {
    pub fn index_type(&self) -> IndexType {
        self.inner.index_type()
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

pub enum VectorIndexWriter {
    IvfFlat(IVFFlatIndex),
    IvfSq(IVFSQIndex),
    IvfPq(IVFPQIndex),
    IvfRq(IVFRQIndex),
    DiskAnn(DiskAnnIndex),
}

impl VectorIndexWriter {
    pub fn new(training: VectorIndexTraining) -> Self {
        training.inner
    }

    fn from_config(config: VectorIndexConfig) -> io::Result<Self> {
        validate_config(&config)?;
        Ok(match config {
            VectorIndexConfig::IvfFlat {
                dimension,
                nlist,
                metric,
            } => Self::IvfFlat(IVFFlatIndex::new(dimension, nlist, metric)),
            VectorIndexConfig::IvfSq {
                dimension,
                nlist,
                metric,
            } => Self::IvfSq(IVFSQIndex::new(dimension, nlist, metric)),
            VectorIndexConfig::IvfPq {
                dimension,
                nlist,
                m,
                metric,
                use_opq,
            } => Self::IvfPq(IVFPQIndex::new(dimension, nlist, m, metric, use_opq)),
            VectorIndexConfig::IvfRq {
                dimension,
                nlist,
                bits,
                metric,
            } => Self::IvfRq(IVFRQIndex::with_bits(dimension, nlist, bits, metric)),
            VectorIndexConfig::DiskAnn {
                dimension,
                metric,
                pq_m,
                pq_bits,
                build,
            } => Self::DiskAnn(DiskAnnIndex::with_pq_bits(
                dimension, metric, pq_m, pq_bits, build,
            )),
        })
    }

    pub fn index_type(&self) -> IndexType {
        match self {
            Self::IvfFlat(_) => IndexType::IvfFlat,
            Self::IvfSq(_) => IndexType::IvfSq,
            Self::IvfPq(_) => IndexType::IvfPq,
            Self::IvfRq(_) => IndexType::IvfRq,
            Self::DiskAnn(_) => IndexType::DiskAnn,
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::IvfFlat(index) => index.d,
            Self::IvfSq(index) => index.d,
            Self::IvfPq(index) => index.d,
            Self::IvfRq(index) => index.d,
            Self::DiskAnn(index) => index.d,
        }
    }

    fn train_internal(&mut self, data: &[f32], n: usize) -> io::Result<()> {
        debug_assert_eq!(Some(data.len()), n.checked_mul(self.dimension()));
        match self {
            Self::IvfFlat(index) => index.train(data, n),
            Self::IvfSq(index) => index.train(data, n),
            Self::IvfPq(index) => index.train(data, n),
            Self::IvfRq(index) => index.train(data, n),
            Self::DiskAnn(index) => return index.train(data, n),
        }
        Ok(())
    }

    pub fn add_vectors(&mut self, ids: &[i64], data: &[f32], n: usize) -> io::Result<()> {
        validate_vectors(data, n, self.dimension(), "vector data")?;
        if ids.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("ids length {} does not match vector count {}", ids.len(), n),
            ));
        }
        match self {
            Self::IvfFlat(index) => index.add(data, ids, n),
            Self::IvfSq(index) => index.add(data, ids, n),
            Self::IvfPq(index) => index.add(data, ids, n),
            Self::IvfRq(index) => index.add(data, ids, n),
            Self::DiskAnn(index) => index.add(data, ids),
        }
        Ok(())
    }

    pub fn write(&mut self, out: &mut dyn SeekWrite) -> io::Result<()> {
        match self {
            Self::IvfFlat(index) => write_ivfflat_index(index, out),
            Self::IvfSq(index) => write_ivfsq_index(index, out),
            Self::IvfPq(index) => write_index(index, out),
            Self::IvfRq(index) => write_ivfrq_index(index, out),
            Self::DiskAnn(index) => write_diskann_index(index, out),
        }
    }
}

pub enum VectorIndexReader<R: SeekRead> {
    IvfFlat(IVFFlatIndexReader<R>),
    IvfSq(IVFSQIndexReader<R>),
    IvfPq(IVFPQIndexReader<R>),
    IvfRq(IVFRQIndexReader<R>),
    DiskAnn(DiskAnnIndexReader<R>),
}

impl<R: SeekRead> VectorIndexReader<R> {
    pub fn open(reader: R) -> io::Result<Self> {
        Self::open_with_options(reader, VectorIndexReaderOptions::default())
    }

    pub fn open_with_options(mut reader: R, options: VectorIndexReaderOptions) -> io::Result<Self> {
        let mut header = [0u8; 64];
        if let Err(header_error) = reader.pread(&mut [ReadRequest::new(0, &mut header)]) {
            let mut magic_buf = [0u8; 4];
            reader.pread(&mut [ReadRequest::new(0, &mut magic_buf)])?;
            let magic = u32::from_le_bytes(magic_buf);
            if !matches!(
                magic,
                IVFFLAT_MAGIC | IVF_SQ_MAGIC | MAGIC | IVF_RQ_MAGIC | DISKANN_MAGIC
            ) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown vector index magic: 0x{:08X}", magic),
                ));
            }
            return Err(header_error);
        }
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());

        match magic {
            IVFFLAT_MAGIC => Ok(Self::IvfFlat(IVFFlatIndexReader::open_with_header(
                reader, header,
            )?)),
            IVF_SQ_MAGIC => Ok(Self::IvfSq(IVFSQIndexReader::open_with_header(
                reader, header,
            )?)),
            MAGIC => Ok(Self::IvfPq(IVFPQIndexReader::open_with_header(
                reader, header,
            )?)),
            IVF_RQ_MAGIC => Ok(Self::IvfRq(IVFRQIndexReader::open_with_header(
                reader, header,
            )?)),
            DISKANN_MAGIC => Ok(Self::DiskAnn(DiskAnnIndexReader::open_with_options(
                reader, options,
            )?)),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown vector index magic: 0x{:08X}", magic),
            )),
        }
    }

    pub fn index_type(&self) -> IndexType {
        match self {
            Self::IvfFlat(_) => IndexType::IvfFlat,
            Self::IvfSq(_) => IndexType::IvfSq,
            Self::IvfPq(_) => IndexType::IvfPq,
            Self::IvfRq(_) => IndexType::IvfRq,
            Self::DiskAnn(_) => IndexType::DiskAnn,
        }
    }

    pub fn metadata(&self) -> VectorIndexMetadata {
        match self {
            Self::IvfFlat(reader) => VectorIndexMetadata {
                index_type: IndexType::IvfFlat,
                dimension: reader.d,
                nlist: reader.nlist,
                metric: reader.metric,
                total_vectors: reader.total_vectors,
                pq_m: None,
                pq_bits: None,
                rq_bits: None,
                diskann: None,
            },
            Self::IvfSq(reader) => VectorIndexMetadata {
                index_type: IndexType::IvfSq,
                dimension: reader.d,
                nlist: reader.nlist,
                metric: reader.metric,
                total_vectors: reader.total_vectors,
                pq_m: None,
                pq_bits: Some(8),
                rq_bits: None,
                diskann: None,
            },
            Self::IvfPq(reader) => VectorIndexMetadata {
                index_type: IndexType::IvfPq,
                dimension: reader.d,
                nlist: reader.nlist,
                metric: reader.metric,
                total_vectors: reader.total_vectors,
                pq_m: Some(reader.m),
                pq_bits: Some(reader.pq.nbits),
                rq_bits: None,
                diskann: None,
            },
            Self::IvfRq(reader) => VectorIndexMetadata {
                index_type: IndexType::IvfRq,
                dimension: reader.d,
                nlist: reader.nlist,
                metric: reader.metric,
                total_vectors: reader.total_vectors,
                pq_m: None,
                pq_bits: None,
                rq_bits: Some(reader.num_bits),
                diskann: None,
            },
            Self::DiskAnn(reader) => VectorIndexMetadata {
                index_type: IndexType::DiskAnn,
                dimension: reader.header.dimension as usize,
                nlist: 1,
                metric: MetricType::from_code(reader.header.metric)
                    .expect("validated DiskANN metric code"),
                total_vectors: reader.header.vector_count as i64,
                pq_m: Some(reader.header.pq_m as usize),
                pq_bits: Some(reader.header.pq_bits as usize),
                rq_bits: None,
                diskann: Some(DiskAnnMetadata {
                    max_degree: reader.header.max_degree as usize,
                    build_search_list_size: reader.header.build_search_list_size as usize,
                    alpha: reader.header.alpha,
                }),
            },
        }
    }

    pub fn dimension(&self) -> usize {
        self.metadata().dimension
    }

    pub fn total_vectors(&self) -> i64 {
        self.metadata().total_vectors
    }

    pub fn diskann_search_stats(&self) -> Option<DiskAnnSearchStats> {
        match self {
            Self::DiskAnn(reader) => Some(reader.last_search_stats()),
            _ => None,
        }
    }

    pub fn ivfrq_search_stats(&self) -> Option<IVFRQSearchStats> {
        match self {
            Self::IvfRq(reader) => Some(reader.last_search_stats()),
            _ => None,
        }
    }

    pub fn read_plan(&self) -> Option<VectorIndexReadPlan> {
        match self {
            Self::DiskAnn(reader) => Some(reader.vector_read_plan()),
            _ => None,
        }
    }

    pub fn optimize_for_search(&mut self) -> io::Result<()> {
        match self {
            Self::IvfFlat(reader) => reader.ensure_loaded(),
            Self::IvfSq(reader) => reader.optimize_for_search(),
            Self::IvfPq(reader) => reader.optimize_for_search(),
            Self::IvfRq(reader) => reader.ensure_loaded(),
            Self::DiskAnn(reader) => reader.optimize_for_search(),
        }
    }

    /// Warm query-dependent caches with representative queries. DiskANN runs
    /// the graph and rerank path; other index types perform their normal
    /// resident optimization because they do not expose a paged query cache.
    pub fn warmup_queries(
        &mut self,
        queries: &[f32],
        query_count: usize,
        l_search: usize,
    ) -> io::Result<()> {
        let expected_len = query_count
            .checked_mul(self.dimension())
            .ok_or_else(|| invalid_input("warmup query count * dimension overflows usize"))?;
        if queries.len() != expected_len {
            return Err(invalid_input(format!(
                "warmup queries length {} does not match query count * dimension {}",
                queries.len(),
                expected_len
            )));
        }
        validate_finite_values(queries, expected_len, "warmup queries")?;
        match self {
            Self::DiskAnn(reader) => reader.warmup_queries(queries, l_search),
            _ => self.optimize_for_search(),
        }
    }

    pub fn calibrate_search_width(
        &mut self,
        queries: &[f32],
        query_count: usize,
        top_k: usize,
    ) -> io::Result<usize> {
        validate_queries(queries, query_count, self.dimension())?;
        validate_positive(top_k, "top_k")?;
        match self {
            Self::DiskAnn(reader) => reader.calibrate_l_search(queries, top_k),
            _ => Err(invalid_input(
                "search-width calibration is currently only available for DiskANN",
            )),
        }
    }

    pub fn search(
        &mut self,
        query: &[f32],
        params: VectorSearchParams,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        validate_query(query, self.dimension())?;
        params.validate()?;
        match self {
            Self::IvfFlat(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe = params.resolve_ivf_nprobe(reader.nlist, total_vectors, None)?;
                progressive_ivf_search(
                    params,
                    reader.nlist,
                    nprobe,
                    1,
                    params.top_k,
                    total_vectors,
                    |nprobe| reader.search(query, params.top_k, nprobe),
                )
            }
            Self::IvfSq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe = params.resolve_ivf_nprobe(reader.nlist, total_vectors, None)?;
                progressive_ivf_search(
                    params,
                    reader.nlist,
                    nprobe,
                    1,
                    params.top_k,
                    total_vectors,
                    |nprobe| reader.search(query, params.top_k, nprobe),
                )
            }
            Self::IvfPq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe = params.resolve_ivf_nprobe(reader.nlist, total_vectors, None)?;
                progressive_ivf_search(
                    params,
                    reader.nlist,
                    nprobe,
                    1,
                    params.top_k,
                    total_vectors,
                    |nprobe| search_with_reader(reader, query, params.top_k, nprobe),
                )
            }
            Self::IvfRq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe = params.resolve_ivf_nprobe(reader.nlist, total_vectors, None)?;
                progressive_ivf_search(
                    params,
                    reader.nlist,
                    nprobe,
                    1,
                    params.top_k,
                    total_vectors,
                    |nprobe| reader.search(query, params.top_k, nprobe),
                )
            }
            Self::DiskAnn(reader) => {
                let l_search = params.resolve_diskann_l_search_with(reader.calibrated_l_search)?;
                reader.search(query, params.top_k, l_search)
            }
        }
    }

    pub fn search_with_roaring_filter(
        &mut self,
        query: &[f32],
        params: VectorSearchParams,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        validate_query(query, self.dimension())?;
        params.validate()?;
        let matching_count = if params.search_width == SearchWidth::Auto {
            Some(decode_roaring_filter_cardinality(roaring_filter_bytes)?)
        } else {
            None
        };
        match self {
            Self::IvfFlat(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe =
                    params.resolve_ivf_nprobe(reader.nlist, total_vectors, matching_count)?;
                progressive_ivf_search(
                    params,
                    reader.nlist,
                    nprobe,
                    1,
                    params.top_k,
                    matching_count.unwrap_or(total_vectors),
                    |nprobe| {
                        reader.search_with_roaring_filter(
                            query,
                            params.top_k,
                            nprobe,
                            roaring_filter_bytes,
                        )
                    },
                )
            }
            Self::IvfSq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe =
                    params.resolve_ivf_nprobe(reader.nlist, total_vectors, matching_count)?;
                progressive_ivf_search(
                    params,
                    reader.nlist,
                    nprobe,
                    1,
                    params.top_k,
                    matching_count.unwrap_or(total_vectors),
                    |nprobe| {
                        reader.search_with_roaring_filter(
                            query,
                            params.top_k,
                            nprobe,
                            roaring_filter_bytes,
                        )
                    },
                )
            }
            Self::IvfPq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe =
                    params.resolve_ivf_nprobe(reader.nlist, total_vectors, matching_count)?;
                progressive_ivf_search(
                    params,
                    reader.nlist,
                    nprobe,
                    1,
                    params.top_k,
                    matching_count.unwrap_or(total_vectors),
                    |nprobe| {
                        search_with_reader_roaring_filter(
                            reader,
                            query,
                            params.top_k,
                            nprobe,
                            roaring_filter_bytes,
                        )
                    },
                )
            }
            Self::IvfRq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe =
                    params.resolve_ivf_nprobe(reader.nlist, total_vectors, matching_count)?;
                progressive_ivf_search(
                    params,
                    reader.nlist,
                    nprobe,
                    1,
                    params.top_k,
                    matching_count.unwrap_or(total_vectors),
                    |nprobe| {
                        reader.search_with_roaring_filter(
                            query,
                            params.top_k,
                            nprobe,
                            roaring_filter_bytes,
                        )
                    },
                )
            }
            Self::DiskAnn(reader) => reader.search_with_roaring_filter(
                query,
                params.top_k,
                params.resolve_diskann_l_search_with(reader.calibrated_l_search)?,
                roaring_filter_bytes,
            ),
        }
    }

    pub fn search_batch(
        &mut self,
        queries: &[f32],
        query_count: usize,
        params: VectorSearchParams,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        validate_queries(queries, query_count, self.dimension())?;
        params.validate()?;
        match self {
            Self::IvfFlat(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe = params.resolve_ivf_nprobe(reader.nlist, total_vectors, None)?;
                progressive_ivf_batch_search(
                    params,
                    reader.nlist,
                    nprobe,
                    queries,
                    query_count,
                    params.top_k,
                    total_vectors,
                    |active_queries, active_query_count, nprobe| {
                        search_batch_ivfflat_reader(
                            reader,
                            active_queries,
                            active_query_count,
                            params.top_k,
                            nprobe,
                        )
                    },
                )
            }
            Self::IvfSq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe = params.resolve_ivf_nprobe(reader.nlist, total_vectors, None)?;
                progressive_ivf_batch_search(
                    params,
                    reader.nlist,
                    nprobe,
                    queries,
                    query_count,
                    params.top_k,
                    total_vectors,
                    |active_queries, active_query_count, nprobe| {
                        search_batch_ivfsq_reader(
                            reader,
                            active_queries,
                            active_query_count,
                            params.top_k,
                            nprobe,
                        )
                    },
                )
            }
            Self::IvfPq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe = params.resolve_ivf_nprobe(reader.nlist, total_vectors, None)?;
                progressive_ivf_batch_search(
                    params,
                    reader.nlist,
                    nprobe,
                    queries,
                    query_count,
                    params.top_k,
                    total_vectors,
                    |active_queries, active_query_count, nprobe| {
                        search_batch_reader_with_reuse_mode_and_budget(
                            reader,
                            active_queries,
                            active_query_count,
                            params.top_k,
                            nprobe,
                            params.ivfpq_batch_table_reuse,
                            params.ivfpq_batch_table_reuse_max_bytes,
                        )
                    },
                )
            }
            Self::IvfRq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe = params.resolve_ivf_nprobe(reader.nlist, total_vectors, None)?;
                let mut aggregate_stats = IVFRQSearchStats::default();
                let result = progressive_ivf_batch_search(
                    params,
                    reader.nlist,
                    nprobe,
                    queries,
                    query_count,
                    params.top_k,
                    total_vectors,
                    |active_queries, active_query_count, nprobe| {
                        let result = search_batch_ivfrq_reader(
                            reader,
                            active_queries,
                            active_query_count,
                            params.top_k,
                            nprobe,
                        );
                        if result.is_ok() {
                            aggregate_stats.merge(reader.last_search_stats());
                        }
                        result
                    },
                );
                if result.is_ok() {
                    aggregate_stats.query_count = query_count;
                    reader.set_last_search_stats(aggregate_stats);
                }
                result
            }
            Self::DiskAnn(reader) => reader.search_batch(
                queries,
                params.top_k,
                params.resolve_diskann_l_search_with(reader.calibrated_l_search)?,
            ),
        }
    }

    pub fn search_batch_with_roaring_filter(
        &mut self,
        queries: &[f32],
        query_count: usize,
        params: VectorSearchParams,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        validate_queries(queries, query_count, self.dimension())?;
        params.validate()?;
        let matching_count = if params.search_width == SearchWidth::Auto {
            Some(decode_roaring_filter_cardinality(roaring_filter_bytes)?)
        } else {
            None
        };
        match self {
            Self::IvfFlat(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe =
                    params.resolve_ivf_nprobe(reader.nlist, total_vectors, matching_count)?;
                progressive_ivf_batch_search(
                    params,
                    reader.nlist,
                    nprobe,
                    queries,
                    query_count,
                    params.top_k,
                    matching_count.unwrap_or(total_vectors),
                    |active_queries, active_query_count, nprobe| {
                        search_batch_ivfflat_reader_roaring_filter(
                            reader,
                            active_queries,
                            active_query_count,
                            params.top_k,
                            nprobe,
                            roaring_filter_bytes,
                        )
                    },
                )
            }
            Self::IvfSq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe =
                    params.resolve_ivf_nprobe(reader.nlist, total_vectors, matching_count)?;
                progressive_ivf_batch_search(
                    params,
                    reader.nlist,
                    nprobe,
                    queries,
                    query_count,
                    params.top_k,
                    matching_count.unwrap_or(total_vectors),
                    |active_queries, active_query_count, nprobe| {
                        search_batch_ivfsq_reader_roaring_filter(
                            reader,
                            active_queries,
                            active_query_count,
                            params.top_k,
                            nprobe,
                            roaring_filter_bytes,
                        )
                    },
                )
            }
            Self::IvfPq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe =
                    params.resolve_ivf_nprobe(reader.nlist, total_vectors, matching_count)?;
                progressive_ivf_batch_search(
                    params,
                    reader.nlist,
                    nprobe,
                    queries,
                    query_count,
                    params.top_k,
                    matching_count.unwrap_or(total_vectors),
                    |active_queries, active_query_count, nprobe| {
                        search_batch_reader_roaring_filter_with_reuse_mode_and_budget(
                            reader,
                            active_queries,
                            active_query_count,
                            params.top_k,
                            nprobe,
                            roaring_filter_bytes,
                            params.ivfpq_batch_table_reuse,
                            params.ivfpq_batch_table_reuse_max_bytes,
                        )
                    },
                )
            }
            Self::IvfRq(reader) => {
                let total_vectors = usize::try_from(reader.total_vectors)
                    .map_err(|_| invalid_input("negative IVF vector count"))?;
                let nprobe =
                    params.resolve_ivf_nprobe(reader.nlist, total_vectors, matching_count)?;
                let mut aggregate_stats = IVFRQSearchStats::default();
                let result = progressive_ivf_batch_search(
                    params,
                    reader.nlist,
                    nprobe,
                    queries,
                    query_count,
                    params.top_k,
                    matching_count.unwrap_or(total_vectors),
                    |active_queries, active_query_count, nprobe| {
                        let result = search_batch_ivfrq_reader_roaring_filter(
                            reader,
                            active_queries,
                            active_query_count,
                            params.top_k,
                            nprobe,
                            roaring_filter_bytes,
                        );
                        if result.is_ok() {
                            aggregate_stats.merge(reader.last_search_stats());
                        }
                        result
                    },
                );
                if result.is_ok() {
                    aggregate_stats.query_count = query_count;
                    reader.set_last_search_stats(aggregate_stats);
                }
                result
            }
            Self::DiskAnn(reader) => reader.search_batch_with_roaring_filter(
                queries,
                params.top_k,
                params.resolve_diskann_l_search_with(reader.calibrated_l_search)?,
                roaring_filter_bytes,
            ),
        }
    }
}

fn decode_roaring_filter_cardinality(bytes: &[u8]) -> io::Result<usize> {
    let filter = RoaringTreemap::deserialize_from(&mut Cursor::new(bytes)).map_err(|error| {
        invalid_input(format!("invalid serialized RoaringTreemap filter: {error}"))
    })?;
    usize::try_from(filter.len())
        .map_err(|_| invalid_input("RoaringTreemap filter cardinality exceeds usize"))
}

fn progressive_ivf_search(
    params: VectorSearchParams,
    nlist: usize,
    initial_nprobe: usize,
    query_count: usize,
    top_k: usize,
    available_matches: usize,
    mut search: impl FnMut(usize) -> io::Result<(Vec<i64>, Vec<f32>)>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let required_per_query = top_k.min(available_matches);
    let mut nprobe = initial_nprobe;
    loop {
        let result = search(nprobe)?;
        if params.search_width != SearchWidth::Auto
            || nprobe >= nlist
            || required_per_query == 0
            || result
                .1
                .chunks_exact(top_k)
                .take(query_count)
                .all(|distances| ivf_search_result_is_complete(distances, required_per_query))
        {
            return Ok(result);
        }
        nprobe = nprobe.saturating_mul(2).min(nlist);
    }
}

/// Runs automatic IVF batch expansion independently for each query.
///
/// Queries which already produced the required number of results are removed from later rounds.
/// Retried queries are still searched from the first probed list; reusing work across rounds is
/// intentionally left to the incremental-search implementation.
fn progressive_ivf_batch_search(
    params: VectorSearchParams,
    nlist: usize,
    initial_nprobe: usize,
    queries: &[f32],
    query_count: usize,
    top_k: usize,
    available_matches: usize,
    search: impl FnMut(&[f32], usize, usize) -> io::Result<(Vec<i64>, Vec<f32>)>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    progressive_ivf_batch_search_with_retry_buffer_limit(
        params,
        nlist,
        initial_nprobe,
        queries,
        query_count,
        top_k,
        available_matches,
        MAX_IVF_BATCH_RETRY_BUFFER_BYTES,
        search,
    )
}

#[allow(clippy::too_many_arguments)]
fn progressive_ivf_batch_search_with_retry_buffer_limit(
    params: VectorSearchParams,
    nlist: usize,
    initial_nprobe: usize,
    queries: &[f32],
    query_count: usize,
    top_k: usize,
    available_matches: usize,
    retry_buffer_limit_bytes: usize,
    mut search: impl FnMut(&[f32], usize, usize) -> io::Result<(Vec<i64>, Vec<f32>)>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    if params.search_width != SearchWidth::Auto {
        return search(queries, query_count, initial_nprobe);
    }

    let dimension = queries.len() / query_count;
    let required_per_query = top_k.min(available_matches);
    let mut nprobe = initial_nprobe;
    let (mut result_ids, mut result_distances) = search(queries, query_count, nprobe)?;
    let mut active_queries = result_distances
        .chunks_exact(top_k)
        .take(query_count)
        .enumerate()
        .filter_map(|(query_index, distances)| {
            (!ivf_search_result_is_complete(distances, required_per_query)).then_some(query_index)
        })
        .collect::<Vec<_>>();

    loop {
        if nprobe >= nlist || required_per_query == 0 || active_queries.is_empty() {
            return Ok((result_ids, result_distances));
        }

        nprobe = nprobe.saturating_mul(2).min(nlist);
        if active_queries.len() == query_count {
            drop(result_ids);
            drop(result_distances);
            (result_ids, result_distances) = search(queries, query_count, nprobe)?;
            active_queries = result_distances
                .chunks_exact(top_k)
                .take(query_count)
                .enumerate()
                .filter_map(|(query_index, distances)| {
                    (!ivf_search_result_is_complete(distances, required_per_query))
                        .then_some(query_index)
                })
                .collect();
            continue;
        }

        let bytes_per_retry_query = dimension
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_add(
                top_k.saturating_mul(std::mem::size_of::<i64>() + std::mem::size_of::<f32>()),
            );
        let retry_chunk_size = retry_buffer_limit_bytes
            .checked_div(bytes_per_retry_query)
            .unwrap_or(active_queries.len())
            .max(1);
        let mut next_active_queries = Vec::new();
        for active_chunk in active_queries.chunks(retry_chunk_size) {
            let mut packed_queries = Vec::with_capacity(active_chunk.len() * dimension);
            for &query_index in active_chunk {
                let start = query_index * dimension;
                packed_queries.extend_from_slice(&queries[start..start + dimension]);
            }

            let (round_ids, round_distances) = search(&packed_queries, active_chunk.len(), nprobe)?;
            for (round_index, &query_index) in active_chunk.iter().enumerate() {
                let round_start = round_index * top_k;
                let result_start = query_index * top_k;
                result_ids[result_start..result_start + top_k]
                    .copy_from_slice(&round_ids[round_start..round_start + top_k]);
                result_distances[result_start..result_start + top_k]
                    .copy_from_slice(&round_distances[round_start..round_start + top_k]);

                if !ivf_search_result_is_complete(
                    &round_distances[round_start..round_start + top_k],
                    required_per_query,
                ) {
                    next_active_queries.push(query_index);
                }
            }
        }
        active_queries = next_active_queries;
    }
}

fn ivf_search_result_is_complete(distances: &[f32], required: usize) -> bool {
    distances
        .iter()
        .filter(|&&distance| distance != f32::MAX)
        .take(required)
        .count()
        >= required
}

fn validate_config(config: &VectorIndexConfig) -> io::Result<()> {
    validate_positive(config.dimension(), "dimension")?;
    validate_positive(config.nlist(), "nlist")?;
    match config {
        VectorIndexConfig::IvfPq { dimension, m, .. } => {
            validate_positive(*m, "m")?;
            if dimension % m != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("dimension {} must be divisible by m {}", dimension, m),
                ));
            }
        }
        VectorIndexConfig::IvfRq { bits, .. } if !is_supported_rq_bits(*bits) => {
            return Err(invalid_input(format!(
                "rq.bits must be in 1..=8, got {bits}"
            )));
        }
        VectorIndexConfig::DiskAnn {
            dimension,
            metric,
            pq_m,
            pq_bits,
            build,
        } => {
            validate_diskann_config(*dimension, *metric, *pq_m, *pq_bits, *build)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_diskann_config(
    dimension: usize,
    metric: MetricType,
    pq_m: usize,
    pq_bits: usize,
    build: DiskAnnBuildParams,
) -> io::Result<()> {
    validate_diskann_format_configuration(dimension, pq_m, pq_bits, build)?;
    validate_positive(build.memory_budget_bytes, "DiskANN memory budget")?;
    validate_diskann_training_budget(dimension, metric, pq_m, pq_bits, build.memory_budget_bytes)
}

fn validate_positive(value: usize, name: &str) -> io::Result<()> {
    if value == 0 {
        Err(invalid_input(format!("{} must be greater than 0", name)))
    } else {
        Ok(())
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn validate_vectors(data: &[f32], n: usize, dimension: usize, value_name: &str) -> io::Result<()> {
    validate_positive(n, "vector count")?;
    let expected_len = n.checked_mul(dimension).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "vector count * dimension overflows usize",
        )
    })?;
    if data.len() != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{} length {} does not match vector count * dimension {}",
                value_name,
                data.len(),
                expected_len
            ),
        ));
    }
    validate_finite_values(data, expected_len, value_name)?;
    Ok(())
}

fn validate_query(query: &[f32], dimension: usize) -> io::Result<()> {
    if query.len() != dimension {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "query length {} does not match index dimension {}",
                query.len(),
                dimension
            ),
        ));
    }
    validate_finite_values(query, dimension, "query")
}

fn validate_queries(queries: &[f32], query_count: usize, dimension: usize) -> io::Result<()> {
    validate_positive(query_count, "query count")?;
    let expected_len = query_count.checked_mul(dimension).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "nq * dimension overflows usize",
        )
    })?;
    if queries.len() != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "queries length {} does not match nq * dimension {}",
                queries.len(),
                expected_len
            ),
        ));
    }
    validate_finite_values(queries, expected_len, "queries")
}

fn validate_finite_values(values: &[f32], len: usize, value_name: &str) -> io::Result<()> {
    for (offset, &value) in values[..len].iter().enumerate() {
        if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} contains non-finite value at offset {}: {}",
                    value_name, offset, value
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::DISKANN_MAX_PQ_TRAINING_VECTORS;
    use crate::io::{PosWriter, ReadRequest, SeekRead};
    use roaring::RoaringTreemap;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct ReadRendezvousState {
        armed: bool,
        arrivals: usize,
    }

    struct CountingReader {
        inner: Cursor<Vec<u8>>,
        calls: Arc<AtomicUsize>,
    }

    impl SeekRead for CountingReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.pread(ranges)
        }
    }

    #[derive(Default)]
    struct ReadRendezvous {
        state: Mutex<ReadRendezvousState>,
        ready: Condvar,
    }

    impl ReadRendezvous {
        fn arm(&self) {
            *self.state.lock().unwrap() = ReadRendezvousState {
                armed: true,
                arrivals: 0,
            };
        }

        fn wait_for_peer(&self) -> io::Result<()> {
            let mut state = self.state.lock().unwrap();
            if !state.armed {
                return Ok(());
            }
            state.arrivals += 1;
            if state.arrivals >= 2 {
                state.armed = false;
                self.ready.notify_all();
                return Ok(());
            }

            let deadline = Instant::now() + Duration::from_secs(5);
            while state.armed {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    state.armed = false;
                    self.ready.notify_all();
                    return Err(io::Error::other(
                        "timed out waiting for a concurrent cloned-reader pread",
                    ));
                }
                let (next, timeout) = self.ready.wait_timeout(state, remaining).unwrap();
                state = next;
                if timeout.timed_out() && state.armed {
                    state.armed = false;
                    self.ready.notify_all();
                    return Err(io::Error::other(
                        "timed out waiting for a concurrent cloned-reader pread",
                    ));
                }
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    struct ConcurrentCloneReader {
        bytes: Arc<[u8]>,
        active_reads: Arc<AtomicUsize>,
        max_active_reads: Arc<AtomicUsize>,
        rendezvous: Arc<ReadRendezvous>,
    }

    impl SeekRead for ConcurrentCloneReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            let active = self.active_reads.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active_reads.fetch_max(active, Ordering::SeqCst);
            let result = self.rendezvous.wait_for_peer().and_then(|()| {
                ranges.iter_mut().try_for_each(|range| {
                    let start = usize::try_from(range.pos)
                        .map_err(|_| io::Error::other("test read offset exceeds usize"))?;
                    let end = start
                        .checked_add(range.buf.len())
                        .ok_or_else(|| io::Error::other("test read range overflows"))?;
                    let source = self
                        .bytes
                        .get(start..end)
                        .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))?;
                    range.buf.copy_from_slice(source);
                    Ok(())
                })
            });
            self.active_reads.fetch_sub(1, Ordering::SeqCst);
            result
        }

        fn try_clone_reader(&self) -> io::Result<Option<Self>> {
            Ok(Some(self.clone()))
        }
    }

    fn generate_clustered_data(n: usize, d: usize, clusters: usize) -> Vec<f32> {
        let mut data = vec![0.0; n * d];
        for i in 0..n {
            let cluster = i % clusters;
            for j in 0..d {
                data[i * d + j] = cluster as f32 * 20.0 + j as f32 * 0.01 + i as f32 * 0.0001;
            }
        }
        data
    }

    fn roundtrip(config: VectorIndexConfig) {
        let d = config.dimension();
        let nlist = config.nlist();
        let n = 512;
        let data = generate_clustered_data(n, d, nlist);
        let ids = (0..n as i64).collect::<Vec<_>>();

        let mut writer = build_writer(config.clone(), &data, n);
        assert_eq!(writer.index_type(), config.index_type());
        writer.add_vectors(&ids, &data, n).unwrap();

        let mut buf = Vec::new();
        writer.write(&mut PosWriter::new(&mut buf)).unwrap();

        let mut reader = VectorIndexReader::open(Cursor::new(buf)).unwrap();
        let metadata = reader.metadata();
        assert_eq!(metadata.index_type, config.index_type());
        assert_eq!(metadata.dimension, d);
        assert_eq!(metadata.nlist, nlist);
        assert_eq!(metadata.total_vectors, n as i64);
        match &config {
            VectorIndexConfig::IvfPq { m, .. } => {
                assert_eq!(metadata.pq_m, Some(*m));
                assert_eq!(metadata.pq_bits, Some(8));
            }
            VectorIndexConfig::IvfSq { .. } => {
                assert_eq!(metadata.pq_m, None);
                assert_eq!(metadata.pq_bits, Some(8));
            }
            VectorIndexConfig::DiskAnn {
                pq_m,
                pq_bits,
                build,
                ..
            } => {
                assert_eq!(metadata.pq_m, Some(*pq_m));
                assert_eq!(metadata.pq_bits, Some(*pq_bits));
                let diskann = metadata.diskann.expect("DiskANN metadata");
                assert_eq!(diskann.max_degree, build.max_degree);
                assert_eq!(diskann.build_search_list_size, build.build_search_list_size);
                assert_eq!(diskann.alpha, build.alpha);
            }
            _ => {
                assert_eq!(metadata.pq_m, None);
                assert_eq!(metadata.pq_bits, None);
            }
        }

        let params = if config.index_type() == IndexType::DiskAnn {
            VectorSearchParams::with_l_search(5, 100)
        } else {
            VectorSearchParams::new(5, nlist)
        };
        let (result_ids, result_dists) = reader.search(&data[0..d], params).unwrap();
        assert_eq!(result_ids.len(), 5);
        assert_eq!(result_dists.len(), 5);
        if config.index_type() == IndexType::DiskAnn {
            assert!(result_ids[0] >= 0);
        } else {
            assert_eq!(result_ids[0], 0);
        }
    }

    fn build_reader(config: VectorIndexConfig) -> (VectorIndexReader<Cursor<Vec<u8>>>, Vec<f32>) {
        let d = config.dimension();
        let nlist = config.nlist();
        let n = 512;
        let data = generate_clustered_data(n, d, nlist);
        let ids = (0..n as i64).collect::<Vec<_>>();

        let mut writer = build_writer(config, &data, n);
        writer.add_vectors(&ids, &data, n).unwrap();

        let mut buf = Vec::new();
        writer.write(&mut PosWriter::new(&mut buf)).unwrap();
        (VectorIndexReader::open(Cursor::new(buf)).unwrap(), data)
    }

    fn build_ivfflat_reader() -> VectorIndexReader<Cursor<Vec<u8>>> {
        let mut writer = build_writer(
            VectorIndexConfig::IvfFlat {
                dimension: 1,
                nlist: 1,
                metric: MetricType::L2,
            },
            &[0.0, 1.0],
            2,
        );
        writer.add_vectors(&[1, 2], &[0.0, 1.0], 2).unwrap();

        let mut bytes = Vec::new();
        writer.write(&mut PosWriter::new(&mut bytes)).unwrap();
        VectorIndexReader::open(Cursor::new(bytes)).unwrap()
    }

    fn build_writer(config: VectorIndexConfig, data: &[f32], n: usize) -> VectorIndexWriter {
        let training = VectorIndexTrainer::train(config, data, n).unwrap();
        VectorIndexWriter::new(training)
    }

    #[test]
    fn diskann_search_stats_are_available_through_unified_reader() {
        let dimension = 8;
        let (mut reader, data) = build_reader(VectorIndexConfig::DiskAnn {
            dimension,
            metric: MetricType::L2,
            pq_m: 2,
            pq_bits: 8,
            build: DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        });

        reader
            .search(
                &data[..dimension],
                VectorSearchParams::with_l_search(3, 100),
            )
            .unwrap();

        let stats = reader.diskann_search_stats().expect("DiskANN diagnostics");
        assert_eq!(stats.query_count, 1);
        assert!(stats.rerank_candidate_references >= 3);
        assert!(stats.rerank_unique_windows >= 1);
    }

    #[test]
    fn ivfrq_search_stats_are_available_through_unified_reader() {
        let dimension = 64;
        let (mut reader, data) = build_reader(VectorIndexConfig::IvfRq {
            dimension,
            nlist: 8,
            metric: MetricType::L2,
            bits: 4,
        });

        reader
            .search(&data[..dimension], VectorSearchParams::new(3, 8))
            .unwrap();

        let stats = reader.ivfrq_search_stats().expect("IVF-RQ diagnostics");
        assert_eq!(stats.query_count, 1);
        assert_eq!(stats.scanned_vectors, 512);
        assert_eq!(stats.eligible_vectors, 512);
        assert!(stats.refined_vectors > 0);
        assert!(reader.diskann_search_stats().is_none());
    }

    #[test]
    fn automatic_filtered_ivfrq_batch_stats_cover_all_progressive_rounds() {
        let dimension = 64;
        let nlist = 64;
        let (mut reader, data) = build_reader(VectorIndexConfig::IvfRq {
            dimension,
            nlist,
            metric: MetricType::L2,
            bits: 4,
        });
        let queries = [0, nlist - 1]
            .into_iter()
            .flat_map(|row| data[row * dimension..(row + 1) * dimension].iter().copied())
            .collect::<Vec<_>>();
        let mut filter = RoaringTreemap::new();
        for row_id in (0..512).step_by(nlist) {
            filter.insert(row_id as u64);
        }
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        reader
            .search_batch_with_roaring_filter(
                &queries,
                2,
                VectorSearchParams::automatic(8).with_max_initial_filter_expansion_factor(1),
                &filter_bytes,
            )
            .unwrap();

        let stats = reader.ivfrq_search_stats().expect("IVF-RQ diagnostics");
        assert_eq!(stats.query_count, 2);
        assert!(
            stats.scanned_vectors > 2 * 8,
            "statistics should include work from progressive retries"
        );
    }

    #[test]
    fn diskann_batch_search_overlaps_graph_reads_and_centralizes_filtered_rerank() {
        let dimension = 8;
        let count = 256;
        let data = generate_clustered_data(count, dimension, 16);
        let mut writer = build_writer(
            VectorIndexConfig::DiskAnn {
                dimension,
                metric: MetricType::L2,
                pq_m: 2,
                pq_bits: 8,
                build: DiskAnnBuildParams {
                    max_degree: 8,
                    build_search_list_size: 16,
                    ..DiskAnnBuildParams::default()
                },
            },
            &data,
            count,
        );
        writer
            .add_vectors(&(0..count as i64).collect::<Vec<_>>(), &data, count)
            .unwrap();
        let mut bytes = Vec::new();
        writer.write(&mut PosWriter::new(&mut bytes)).unwrap();

        let active_reads = Arc::new(AtomicUsize::new(0));
        let max_active_reads = Arc::new(AtomicUsize::new(0));
        let rendezvous = Arc::new(ReadRendezvous::default());
        let source = ConcurrentCloneReader {
            bytes: Arc::from(bytes),
            active_reads: Arc::clone(&active_reads),
            max_active_reads: Arc::clone(&max_active_reads),
            rendezvous: Arc::clone(&rendezvous),
        };
        let mut reader = VectorIndexReader::open_with_options(
            source,
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::Auto,
                0,
                0,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();
        reader.optimize_for_search().unwrap();
        let queries = [0, 17, 34, 51]
            .into_iter()
            .flat_map(|row| data[row * dimension..(row + 1) * dimension].iter().copied())
            .collect::<Vec<_>>();
        let params = VectorSearchParams::with_l_search(3, 32);
        let expected = queries
            .chunks_exact(dimension)
            .flat_map(|query| reader.search(query, params).unwrap().0)
            .collect::<Vec<_>>();
        max_active_reads.store(0, Ordering::SeqCst);
        rendezvous.arm();

        let (actual, _) = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| reader.search_batch(&queries, 4, params))
            .unwrap();

        assert_eq!(actual, expected);
        assert!(
            max_active_reads.load(Ordering::SeqCst) > 1,
            "DiskANN batch search should overlap reads from independent queries"
        );

        let mut filter = RoaringTreemap::new();
        for row_id in 0..count as u64 {
            filter.insert(row_id);
        }
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();
        let expected_filtered = queries
            .chunks_exact(dimension)
            .flat_map(|query| {
                reader
                    .search_with_roaring_filter(query, params, &filter_bytes)
                    .unwrap()
                    .0
            })
            .collect::<Vec<_>>();
        max_active_reads.store(0, Ordering::SeqCst);

        let (actual_filtered, _) = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| reader.search_batch_with_roaring_filter(&queries, 4, params, &filter_bytes))
            .unwrap();

        assert_eq!(actual_filtered, expected_filtered);
        assert!(
            max_active_reads.load(Ordering::SeqCst) <= 1,
            "filtered candidate workers must leave raw-vector I/O to the parent reranker"
        );
        assert!(
            reader.diskann_search_stats().unwrap().raw_vector_cache_hits > 0,
            "the parent reranker should reuse prewarmed raw-vector windows"
        );
    }

    fn assert_invalid_input_contains(result: io::Result<()>, expected: &str) {
        let err = result.expect_err("invalid input should be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains(expected),
            "error '{}' should contain '{}'",
            err,
            expected
        );
    }

    #[test]
    fn unified_reader_writer_roundtrips_all_index_types() {
        roundtrip(VectorIndexConfig::IvfFlat {
            dimension: 8,
            nlist: 4,
            metric: MetricType::L2,
        });
        roundtrip(VectorIndexConfig::ivf_pq(16, 4, MetricType::L2, false).unwrap());
        roundtrip(VectorIndexConfig::IvfRq {
            dimension: 8,
            nlist: 4,
            bits: DEFAULT_RQ_BITS,
            metric: MetricType::L2,
        });
        roundtrip(VectorIndexConfig::IvfSq {
            dimension: 8,
            nlist: 4,
            metric: MetricType::L2,
        });
        roundtrip(
            VectorIndexConfig::disk_ann(
                8,
                MetricType::L2,
                4,
                DiskAnnBuildParams {
                    max_degree: 8,
                    build_search_list_size: 16,
                    ..DiskAnnBuildParams::default()
                },
            )
            .unwrap(),
        );
        for metric in [MetricType::InnerProduct, MetricType::Cosine] {
            roundtrip(
                VectorIndexConfig::disk_ann(
                    8,
                    metric,
                    4,
                    DiskAnnBuildParams {
                        max_degree: 8,
                        build_search_list_size: 16,
                        ..DiskAnnBuildParams::default()
                    },
                )
                .unwrap(),
            );
        }
    }

    #[test]
    fn optimize_for_search_preserves_results() {
        for config in [
            VectorIndexConfig::IvfFlat {
                dimension: 8,
                nlist: 4,
                metric: MetricType::L2,
            },
            VectorIndexConfig::IvfPq {
                dimension: 16,
                nlist: 4,
                m: 4,
                metric: MetricType::L2,
                use_opq: false,
            },
            VectorIndexConfig::IvfRq {
                dimension: 8,
                nlist: 4,
                bits: DEFAULT_RQ_BITS,
                metric: MetricType::L2,
            },
            VectorIndexConfig::IvfSq {
                dimension: 8,
                nlist: 4,
                metric: MetricType::L2,
            },
        ] {
            let d = config.dimension();
            let nlist = config.nlist();
            let params = VectorSearchParams::new(5, nlist);
            let (mut baseline, data) = build_reader(config.clone());
            let query = data[0..d].to_vec();
            let expected = baseline.search(&query, params).unwrap();

            let (mut optimized, _) = build_reader(config);
            optimized.optimize_for_search().unwrap();
            let actual = optimized.search(&query, params).unwrap();

            assert_eq!(actual.0, expected.0);
            assert_eq!(actual.1.len(), expected.1.len());
            for (actual, expected) in actual.1.iter().zip(expected.1.iter()) {
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "optimized distance {} should match baseline {}",
                    actual,
                    expected
                );
            }
        }
    }

    #[test]
    fn representative_warmup_accepts_an_empty_query_set_for_resident_only_initialization() {
        let mut reader = build_ivfflat_reader();

        reader.warmup_queries(&[], 0, 0).unwrap();

        let error = reader
            .warmup_queries(&[0.0], 0, 0)
            .expect_err("empty warmup count must still validate the vector buffer");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("warmup queries length"));
    }

    #[test]
    fn unified_reader_rejects_unknown_magic() {
        let err = match VectorIndexReader::open(Cursor::new(vec![0xFF; 8])) {
            Ok(_) => panic!("unknown magic should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("unknown vector index magic"));
    }

    #[test]
    fn unified_ivf_reader_reuses_dispatch_header() {
        let d = 2;
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let ids = vec![10, 11];
        let mut index = IVFFlatIndex::new(d, 1, MetricType::L2);
        index.train(&data, 2);
        index.add(&data, &ids, 2);
        let mut bytes = Vec::new();
        write_ivfflat_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingReader {
            inner: Cursor::new(bytes),
            calls: Arc::clone(&calls),
        };
        let mut reader = VectorIndexReader::open(source).unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "type dispatch and format parsing share one header read"
        );
        reader.optimize_for_search().unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "resident IVF metadata adds only one read round"
        );
    }

    #[test]
    fn unified_config_rejects_invalid_pq_m() {
        let err = match VectorIndexTrainer::new(VectorIndexConfig::IvfPq {
            dimension: 10,
            nlist: 4,
            m: 3,
            metric: MetricType::L2,
            use_opq: false,
        }) {
            Ok(_) => panic!("invalid PQ config should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("must be divisible"));
    }

    #[test]
    fn unified_config_accepts_unaligned_rq_dimension_and_rejects_invalid_bits() {
        VectorIndexTrainer::new(VectorIndexConfig::IvfRq {
            dimension: 10,
            nlist: 4,
            bits: DEFAULT_RQ_BITS,
            metric: MetricType::L2,
        })
        .unwrap();
        let err = match VectorIndexTrainer::new(VectorIndexConfig::IvfRq {
            dimension: 10,
            nlist: 4,
            bits: 9,
            metric: MetricType::L2,
        }) {
            Ok(_) => panic!("invalid RQ config should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("rq.bits"));
    }

    #[test]
    fn pq_m_inference_preserves_relative_code_budget_across_dimensions_and_bits() {
        assert_eq!(infer_pq_m(128, 8, DEFAULT_PQ_CODE_RATIO).unwrap(), 32);
        assert_eq!(infer_pq_m(960, 8, DEFAULT_PQ_CODE_RATIO).unwrap(), 240);
        assert_eq!(infer_pq_m(128, 4, DEFAULT_PQ_CODE_RATIO).unwrap(), 64);
        assert_eq!(infer_pq_m(960, 4, DEFAULT_PQ_CODE_RATIO).unwrap(), 480);
    }

    #[test]
    fn pq_m_inference_selects_the_nearest_balanced_subvector_shape() {
        assert_eq!(infer_pq_m(10, 8, DEFAULT_PQ_CODE_RATIO).unwrap(), 3);
        assert_eq!(infer_pq_m(384, 8, 0.125).unwrap(), 192);
        assert_eq!(infer_pq_m(7, 4, DEFAULT_PQ_CODE_RATIO).unwrap(), 4);
    }

    #[test]
    fn pq_m_inference_rejects_invalid_code_ratios() {
        for ratio in [0.0, f64::NAN, f64::INFINITY, 0.2501] {
            let error = infer_pq_m(128, 8, ratio).expect_err("invalid ratio should fail");
            assert!(error.to_string().contains("pq.code-ratio"));
        }
        let error = infer_pq_m(128, 4, 0.1251)
            .expect_err("4-bit code ratio cannot exceed its full-width encoding");
        assert!(error.to_string().contains("(0, 0.125]"));
    }

    #[test]
    fn diskann_l_search_is_independent_of_ivf_nprobe() {
        let diskann = VectorSearchParams::with_l_search(10, 200);
        assert_eq!(diskann.search_width, SearchWidth::DiskAnnLSearch);
        assert_eq!(diskann.width, 200);
    }

    fn options(values: &[(&str, &str)]) -> HashMap<String, String> {
        values
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    #[test]
    fn diskann_config_index_type_code_and_name() {
        let index_type = IndexType::from_code(5).expect("DiskANN index type code should exist");
        assert_eq!(index_type.as_str(), "diskann");
    }

    #[test]
    fn ivfsq_uses_a_new_type_code_and_retired_type_codes_stay_reserved() {
        assert_eq!(IndexType::from_code(6), Some(IndexType::IvfSq));
        assert_eq!(IndexType::IvfSq.as_str(), "ivf_sq");
        assert_eq!(IndexType::from_code(2), None);
        assert_eq!(IndexType::from_code(3), None);
    }

    #[test]
    fn diskann_config_parses_without_nlist() {
        let config = VectorIndexConfig::from_options(&options(&[
            ("index.type", "diskann"),
            ("dimension", "128"),
            ("metric", "l2"),
        ]))
        .expect("valid DiskANN options should infer pq.m without nlist");

        assert_eq!(config.index_type(), IndexType::DiskAnn);
        assert_eq!(config.dimension(), 128);
        assert_eq!(config.nlist(), 1);
        let VectorIndexConfig::DiskAnn {
            pq_m,
            pq_bits,
            build,
            ..
        } = config
        else {
            panic!("expected DiskANN config");
        };
        assert_eq!(pq_m, 32);
        assert_eq!(pq_bits, 8);
        assert_eq!(build.memory_budget_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn pq_config_parses_code_ratio_and_explicit_m_takes_precedence() {
        let auto = VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_pq"),
            ("dimension", "128"),
            ("nlist", "4"),
            ("metric", "l2"),
            ("pq.code-ratio", "0.125"),
        ]))
        .expect("IVF-PQ should infer pq.m from a relative code budget");
        let VectorIndexConfig::IvfPq { m, .. } = auto else {
            panic!("expected IVF-PQ config");
        };
        assert_eq!(m, 64);

        let explicit = VectorIndexConfig::from_options(&options(&[
            ("index.type", "diskann"),
            ("dimension", "128"),
            ("metric", "l2"),
            ("pq.m", "16"),
            ("pq.code-ratio", "0.125"),
        ]))
        .expect("explicit pq.m should override the code ratio");
        let VectorIndexConfig::DiskAnn { pq_m, .. } = explicit else {
            panic!("expected DiskANN config");
        };
        assert_eq!(pq_m, 16);
    }

    #[test]
    fn ivf_config_resolves_automatic_nlist_from_expected_count() {
        let plan = VectorIndexBuildPlan::from_options(&options(&[
            ("index.type", "ivf_sq"),
            ("dimension", "100"),
            ("expected-vector-count", "1183514"),
            ("nlist", "auto"),
            ("metric", "cosine"),
        ]))
        .unwrap();
        assert_eq!(plan.expected_vector_count, Some(1_183_514));
        assert_eq!(plan.config.nlist(), 1024);
        assert_eq!(plan.config.resolved().nlist, 1024);

        let missing_count = VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_flat"),
            ("dimension", "8"),
            ("metric", "l2"),
        ]))
        .unwrap_err();
        assert!(missing_count.to_string().contains("expected-vector-count"));
    }

    #[test]
    fn capacity_goal_resolves_rq_bits_and_diskann_deployment_layout() {
        let rq = VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_rq"),
            ("dimension", "100"),
            ("nlist", "16"),
            ("metric", "l2"),
            ("max-bytes-per-vector", "88"),
        ]))
        .unwrap();
        assert_eq!(rq.resolved().rq_bits, Some(3));

        let diskann = VectorIndexConfig::from_options(&options(&[
            ("index.type", "diskann"),
            ("dimension", "128"),
            ("metric", "l2"),
            ("deployment-profile", "local_storage"),
            ("diskann.storage-layout", "auto"),
        ]))
        .unwrap();
        assert_eq!(
            diskann.resolved().diskann_build.unwrap().storage_layout,
            DiskAnnStorageLayout::Interleaved
        );
    }

    #[test]
    fn capacity_goal_rejects_indexes_that_cannot_meet_persisted_size_budget() {
        let flat = VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_flat"),
            ("dimension", "128"),
            ("nlist", "4"),
            ("metric", "l2"),
            ("max-bytes-per-vector", "64"),
        ]))
        .unwrap_err();
        assert!(flat.to_string().contains("cannot be satisfied"));

        let diskann = VectorIndexConfig::from_options(&options(&[
            ("index.type", "diskann"),
            ("dimension", "128"),
            ("metric", "l2"),
            ("max-bytes-per-vector", "64"),
        ]))
        .unwrap_err();
        assert!(diskann.to_string().contains("cannot fit DiskANN"));

        let pq = VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_pq"),
            ("dimension", "128"),
            ("nlist", "4"),
            ("metric", "l2"),
            ("max-bytes-per-vector", "80"),
        ]))
        .unwrap();
        assert!(pq.resolved().pq_m.unwrap() + PERSISTED_ROW_ID_ESTIMATE_BYTES <= 80);
    }

    #[test]
    fn build_time_goal_is_never_silently_guessed_from_hardware() {
        let values = options(&[
            ("index.type", "ivf_sq"),
            ("dimension", "16"),
            ("nlist", "4"),
            ("metric", "l2"),
            ("max-build-seconds", "10"),
        ]);
        let plan = VectorIndexBuildPlan::from_options(&values).unwrap();
        assert_eq!(plan.objective.max_build_seconds, Some(10.0));

        let error = VectorIndexConfig::from_options(&values).unwrap_err();
        assert!(error
            .to_string()
            .contains("requires measured offline calibration"));
    }

    #[test]
    fn tagged_search_width_rejects_cross_index_parameters() {
        assert_eq!(
            VectorSearchParams::automatic(10)
                .resolve_ivf_nprobe(1024, 1_000_000, None)
                .unwrap(),
            64
        );
        assert!(VectorSearchParams::with_l_search(10, 100)
            .resolve_ivf_nprobe(1024, 1_000_000, None)
            .unwrap_err()
            .to_string()
            .contains("cannot be used with an IVF"));
        assert!(VectorSearchParams::new(10, 64)
            .resolve_diskann_l_search()
            .unwrap_err()
            .to_string()
            .contains("cannot be used with a DiskANN"));
        assert!(VectorSearchParams::automatic(10)
            .with_max_initial_filter_expansion_factor(4)
            .resolve_diskann_l_search()
            .unwrap_err()
            .to_string()
            .contains("only valid for IVF"));
    }

    #[test]
    fn automatic_search_params_configure_initial_filter_expansion_cap() {
        let params = VectorSearchParams::automatic(3)
            .with_ivfpq_batch_table_reuse(IvfPqBatchTableReuseMode::Off)
            .with_ivfpq_batch_table_reuse_max_bytes(128 * 1024 * 1024)
            .with_max_initial_filter_expansion_factor(4);
        assert_eq!(params.max_initial_filter_expansion_factor, Some(4));
        assert_eq!(
            params.ivfpq_batch_table_reuse,
            IvfPqBatchTableReuseMode::Off
        );
        assert_eq!(params.ivfpq_batch_table_reuse_max_bytes, 128 * 1024 * 1024);
        assert_eq!(
            params
                .resolve_ivf_nprobe(256, 2_560_000, Some(256_000))
                .unwrap(),
            64
        );
    }

    #[test]
    fn initial_filter_expansion_cap_requires_positive_automatic_search() {
        assert!(VectorSearchParams::automatic(3)
            .with_max_initial_filter_expansion_factor(0)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("greater than 0"));
        assert!(VectorSearchParams::new(3, 16)
            .with_max_initial_filter_expansion_factor(4)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("automatic IVF search"));
    }

    #[test]
    fn ivfpq_batch_table_reuse_is_auto_by_default_and_can_be_disabled() {
        let params = VectorSearchParams::new(10, 4);
        assert_eq!(
            params.ivfpq_batch_table_reuse,
            IvfPqBatchTableReuseMode::Auto
        );
        assert_eq!(params.ivfpq_batch_table_reuse_max_bytes, 512 * 1024 * 1024);
        assert_eq!(
            params
                .with_ivfpq_batch_table_reuse(IvfPqBatchTableReuseMode::Off)
                .ivfpq_batch_table_reuse,
            IvfPqBatchTableReuseMode::Off
        );
        assert_eq!(
            params
                .with_ivfpq_batch_table_reuse_max_bytes(128 * 1024 * 1024)
                .ivfpq_batch_table_reuse_max_bytes,
            128 * 1024 * 1024
        );
        assert!(params
            .with_ivfpq_batch_table_reuse_max_bytes(0)
            .validate()
            .is_err());
    }

    #[test]
    fn automatic_filtered_search_expands_until_results_are_filled() {
        let mut observed = Vec::new();
        let result = progressive_ivf_search(
            VectorSearchParams::automatic(2),
            16,
            2,
            1,
            2,
            10,
            |nprobe| {
                observed.push(nprobe);
                if nprobe < 8 {
                    Ok((vec![7, -1], vec![1.0, f32::MAX]))
                } else {
                    Ok((vec![7, 8], vec![1.0, 2.0]))
                }
            },
        )
        .unwrap();
        assert_eq!(observed, vec![2, 4, 8]);
        assert_eq!(result.0, vec![7, 8]);
    }

    #[test]
    fn automatic_batch_search_retries_only_incomplete_queries() {
        let queries = vec![10.0, 20.0, 30.0];
        let mut observed = Vec::new();
        let result = progressive_ivf_batch_search(
            VectorSearchParams::automatic(2),
            8,
            2,
            &queries,
            3,
            2,
            10,
            |active_queries, active_query_count, nprobe| {
                observed.push((nprobe, active_query_count, active_queries.to_vec()));
                match nprobe {
                    2 => Ok((
                        vec![100, 101, 200, -1, 300, 301],
                        vec![1.0, 2.0, 1.0, f32::MAX, 1.0, 2.0],
                    )),
                    4 => Ok((vec![200, -1], vec![1.0, f32::MAX])),
                    8 => Ok((vec![200, 201], vec![1.0, 2.0])),
                    _ => unreachable!("unexpected nprobe {nprobe}"),
                }
            },
        )
        .unwrap();

        assert_eq!(
            observed,
            vec![
                (2, 3, vec![10.0, 20.0, 30.0]),
                (4, 1, vec![20.0]),
                (8, 1, vec![20.0]),
            ]
        );
        assert_eq!(result.0, vec![100, 101, 200, 201, 300, 301]);
        assert_eq!(result.1, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn automatic_batch_search_chunks_partial_retries_to_bound_memory() {
        let queries = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0];
        let mut observed = Vec::new();
        let result = progressive_ivf_batch_search_with_retry_buffer_limit(
            VectorSearchParams::automatic(2),
            4,
            2,
            &queries,
            4,
            2,
            10,
            32,
            |active_queries, active_query_count, nprobe| {
                observed.push((nprobe, active_query_count, active_queries.to_vec()));
                if nprobe == 2 {
                    return Ok((
                        vec![100, 101, 200, -1, 300, -1, 400, -1],
                        vec![1.0, 2.0, 1.0, f32::MAX, 1.0, f32::MAX, 1.0, f32::MAX],
                    ));
                }

                let query = active_queries[0] as i64;
                Ok((vec![query, query + 1], vec![1.0, 2.0]))
            },
        )
        .unwrap();

        assert_eq!(
            observed,
            vec![
                (2, 4, queries),
                (4, 1, vec![20.0, 21.0]),
                (4, 1, vec![30.0, 31.0]),
                (4, 1, vec![40.0, 41.0]),
            ]
        );
        assert_eq!(result.0, vec![100, 101, 20, 21, 30, 31, 40, 41]);
    }

    #[test]
    fn fixed_batch_search_runs_once_with_the_full_batch() {
        let queries = vec![10.0, 20.0, 30.0];
        let mut observed = Vec::new();
        let result = progressive_ivf_batch_search(
            VectorSearchParams::new(2, 4),
            8,
            4,
            &queries,
            3,
            2,
            10,
            |active_queries, active_query_count, nprobe| {
                observed.push((nprobe, active_query_count, active_queries.to_vec()));
                Ok((
                    vec![100, 101, 200, 201, 300, 301],
                    vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                ))
            },
        )
        .unwrap();

        assert_eq!(observed, vec![(4, 3, queries)]);
        assert_eq!(result.0, vec![100, 101, 200, 201, 300, 301]);
        assert_eq!(result.1, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn capped_automatic_filtered_search_can_expand_past_the_initial_cap() {
        let params = VectorSearchParams::automatic(2).with_max_initial_filter_expansion_factor(4);
        let initial_nprobe = params
            .resolve_ivf_nprobe(256, 2_560_000, Some(256_000))
            .unwrap();
        let mut observed = Vec::new();
        let result = progressive_ivf_search(params, 256, initial_nprobe, 1, 2, 256_000, |nprobe| {
            observed.push(nprobe);
            if nprobe < 128 {
                Ok((vec![7, -1], vec![1.0, f32::MAX]))
            } else {
                Ok((vec![7, 8], vec![1.0, 2.0]))
            }
        })
        .unwrap();
        assert_eq!(observed, vec![64, 128]);
        assert_eq!(result.0, vec![7, 8]);
    }

    #[test]
    fn automatic_search_treats_negative_row_ids_as_valid_results() {
        let mut observed = Vec::new();
        let result = progressive_ivf_search(
            VectorSearchParams::automatic(2),
            16,
            2,
            1,
            2,
            10,
            |nprobe| {
                observed.push(nprobe);
                Ok((vec![-1, -7], vec![1.0, 2.0]))
            },
        )
        .unwrap();
        assert_eq!(observed, vec![2]);
        assert_eq!(result.0, vec![-1, -7]);
    }

    #[test]
    fn search_params_reject_zero_top_k_before_index_dispatch() {
        assert!(VectorSearchParams::automatic(0)
            .validate()
            .unwrap_err()
            .to_string()
            .contains("top_k"));
        assert!(VectorSearchParams::new(0, 1).validate().is_err());
        assert!(VectorSearchParams::with_l_search(0, 100)
            .validate()
            .is_err());
    }

    #[test]
    fn pq_config_constructors_use_the_default_relative_budget() {
        let ivf = VectorIndexConfig::ivf_pq(128, 4, MetricType::L2, false).unwrap();
        let VectorIndexConfig::IvfPq { m, .. } = ivf else {
            panic!("expected IVF-PQ config");
        };
        assert_eq!(m, 32);

        let diskann =
            VectorIndexConfig::disk_ann(960, MetricType::L2, 8, DiskAnnBuildParams::default())
                .unwrap();
        let VectorIndexConfig::DiskAnn { pq_m, .. } = diskann else {
            panic!("expected DiskANN config");
        };
        assert_eq!(pq_m, 240);
    }

    #[test]
    fn diskann_config_parses_explicit_build_parameters() {
        let config = VectorIndexConfig::from_options(&options(&[
            ("index.type", "diskann"),
            ("dimension", "128"),
            ("metric", "l2"),
            ("pq.m", "16"),
            ("pq.bits", "4"),
            ("diskann.max-degree", "32"),
            ("diskann.build-search-list-size", "64"),
            ("diskann.alpha", "1.4"),
            ("diskann.seed", "7"),
            ("diskann.memory-budget-bytes", "123456"),
            ("diskann.storage-layout", "interleaved"),
            ("diskann.raw-vector-encoding", "f16"),
            ("diskann.build-distance", "full_precision"),
        ]))
        .expect("explicit DiskANN build parameters should parse");

        let VectorIndexConfig::DiskAnn {
            pq_m,
            pq_bits,
            build,
            ..
        } = config
        else {
            panic!("expected DiskANN config");
        };
        assert_eq!(pq_m, 16);
        assert_eq!(pq_bits, 4);
        assert_eq!(build.max_degree, 32);
        assert_eq!(build.build_search_list_size, 64);
        assert_eq!(build.alpha, 1.4);
        assert_eq!(build.seed, 7);
        assert_eq!(build.memory_budget_bytes, 123456);
        assert_eq!(build.storage_layout, DiskAnnStorageLayout::Interleaved);
        assert_eq!(build.raw_vector_encoding, DiskAnnRawVectorEncoding::F16);
        assert_eq!(build.build_distance, DiskAnnBuildDistance::FullPrecision);
    }

    #[test]
    fn diskann_build_width_default_follows_explicit_degree() {
        let diskann = VectorIndexConfig::from_options(&options(&[
            ("index.type", "diskann"),
            ("dimension", "128"),
            ("metric", "l2"),
            ("diskann.max-degree", "128"),
        ]))
        .expect("omitted Lbuild should follow an explicit DiskANN degree");
        let VectorIndexConfig::DiskAnn { build, .. } = diskann else {
            panic!("expected DiskANN config");
        };
        assert_eq!(build.max_degree, 128);
        assert_eq!(build.build_search_list_size, 128);
    }

    #[test]
    fn explicit_diskann_build_search_width_is_preserved() {
        let diskann = VectorIndexConfig::from_options(&options(&[
            ("index.type", "diskann"),
            ("dimension", "128"),
            ("metric", "l2"),
            ("diskann.max-degree", "128"),
            ("diskann.build-search-list-size", "256"),
        ]))
        .expect("explicit Lbuild should take precedence over its automatic default");
        let VectorIndexConfig::DiskAnn { build, .. } = diskann else {
            panic!("expected DiskANN config");
        };
        assert_eq!(build.build_search_list_size, 256);
    }

    #[test]
    fn diskann_config_accepts_all_supported_metrics() {
        for (name, expected) in [
            ("l2", MetricType::L2),
            ("inner_product", MetricType::InnerProduct),
            ("cosine", MetricType::Cosine),
        ] {
            let config = VectorIndexConfig::from_options(&options(&[
                ("index.type", "diskann"),
                ("dimension", "128"),
                ("metric", name),
                ("pq.m", "16"),
            ]))
            .expect("DiskANN should accept every public metric");
            let VectorIndexConfig::DiskAnn { metric, .. } = config else {
                panic!("expected DiskANN config");
            };
            assert_eq!(metric, expected);
        }
    }

    fn assert_diskann_metric_roundtrip(
        metric: MetricType,
        data: &[f32],
        ids: &[i64],
        queries: &[f32],
        expected_ids: &[i64],
        expected_distances: &[f32],
    ) {
        let dimension = 2;
        let count = ids.len();
        let config = VectorIndexConfig::DiskAnn {
            dimension,
            metric,
            pq_m: 1,
            pq_bits: 4,
            build: DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
                build_distance: DiskAnnBuildDistance::ProductQuantized,
                ..DiskAnnBuildParams::default()
            },
        };
        let mut writer = build_writer(config, data, count);
        writer.add_vectors(ids, data, count).unwrap();
        let mut bytes = Vec::new();
        writer.write(&mut PosWriter::new(&mut bytes)).unwrap();

        let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();
        assert_eq!(reader.metadata().metric, metric);
        let params = VectorSearchParams::with_l_search(1, count);
        let (single_ids, single_distances) = reader.search(&queries[..dimension], params).unwrap();
        assert_eq!(single_ids[0], expected_ids[0]);
        assert!((single_distances[0] - expected_distances[0]).abs() < 1e-5);

        let query_count = queries.len() / dimension;
        let (batch_ids, batch_distances) =
            reader.search_batch(queries, query_count, params).unwrap();
        assert_eq!(batch_ids, expected_ids);
        for (&actual, &expected) in batch_distances.iter().zip(expected_distances) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let mut filter = RoaringTreemap::new();
        filter.insert(expected_ids[1] as u64);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();
        let (filtered_ids, filtered_distances) = reader
            .search_with_roaring_filter(&queries[dimension..2 * dimension], params, &filter_bytes)
            .unwrap();
        assert_eq!(filtered_ids[0], expected_ids[1]);
        assert!((filtered_distances[0] - expected_distances[1]).abs() < 1e-5);
    }

    #[test]
    fn diskann_inner_product_roundtrips_single_batch_and_filtered_search() {
        let ids = (100..116).collect::<Vec<_>>();
        let data = vec![
            10.0, 0.0, // highest IP for [1, 0], deliberately far in L2
            1.0, 0.0, 0.0, 8.0, // highest IP for [0, 1]
            0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -2.0, 1.0, 1.0, -2.0, -3.0, -1.0, -1.0, -3.0, -4.0,
            0.5, 0.5, -4.0, -5.0, -2.0, -2.0, -5.0, -6.0, -1.0, -2.0, 0.0,
        ];

        assert_diskann_metric_roundtrip(
            MetricType::InnerProduct,
            &data,
            &ids,
            &[1.0, 0.0, 0.0, 1.0],
            &[100, 102],
            &[-10.0, -8.0],
        );
    }

    #[test]
    fn diskann_cosine_roundtrips_single_batch_and_filtered_search() {
        let ids = (200..216).collect::<Vec<_>>();
        let data = vec![
            10.0, 0.0, // cosine winner for [1, 0], deliberately far in L2
            1.0, 1.0, 0.0, 5.0, // cosine winner for [0, 1]
            1.0, 2.0, -1.0, 0.0, 0.0, -1.0, -2.0, 1.0, 1.0, -2.0, -3.0, -1.0, -1.0, -3.0, -4.0,
            0.5, 0.5, -4.0, -5.0, -2.0, -2.0, -5.0, -6.0, -1.0, -2.0, 0.0,
        ];

        assert_diskann_metric_roundtrip(
            MetricType::Cosine,
            &data,
            &ids,
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[200, 202, 200],
            &[0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn diskann_public_metrics_have_approximate_recall_below_full_scan_width() {
        let dimension = 16;
        let count = 512;
        let top_k = 10;
        let query_rows = (0..20).map(|index| index * 23 % count).collect::<Vec<_>>();
        let data = (0..count * dimension)
            .map(|offset| {
                let row = offset / dimension;
                let column = offset % dimension;
                ((row * 31 + column * 17) as f32 * 0.037).sin()
                    + ((row * 13 + column * 29) as f32 * 0.019).cos()
                    + (row % 7) as f32 * 0.03
            })
            .collect::<Vec<_>>();
        let queries = query_rows
            .iter()
            .flat_map(|&row| data[row * dimension..(row + 1) * dimension].iter().copied())
            .collect::<Vec<_>>();

        for metric in [MetricType::InnerProduct, MetricType::Cosine] {
            let mut writer = build_writer(
                VectorIndexConfig::DiskAnn {
                    dimension,
                    metric,
                    pq_m: 4,
                    pq_bits: 8,
                    build: DiskAnnBuildParams {
                        max_degree: 32,
                        build_search_list_size: 64,
                        raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
                        ..DiskAnnBuildParams::default()
                    },
                },
                &data,
                count,
            );
            writer
                .add_vectors(&(0..count as i64).collect::<Vec<_>>(), &data, count)
                .unwrap();
            let mut bytes = Vec::new();
            writer.write(&mut PosWriter::new(&mut bytes)).unwrap();
            let mut reader = VectorIndexReader::open(Cursor::new(bytes)).unwrap();

            let (actual, _) = reader
                .search_batch(
                    &queries,
                    query_rows.len(),
                    VectorSearchParams::with_l_search(top_k, 96),
                )
                .unwrap();
            let mut overlap = 0usize;
            for (query_index, query) in queries.chunks_exact(dimension).enumerate() {
                let mut exact = (0..count)
                    .map(|row| {
                        (
                            crate::distance::fvec_distance(
                                query,
                                &data[row * dimension..(row + 1) * dimension],
                                metric,
                            ),
                            row as i64,
                        )
                    })
                    .collect::<Vec<_>>();
                exact.sort_by(|left, right| {
                    left.0
                        .total_cmp(&right.0)
                        .then_with(|| left.1.cmp(&right.1))
                });
                let expected = exact
                    .iter()
                    .take(top_k)
                    .map(|entry| entry.1)
                    .collect::<HashSet<_>>();
                overlap += actual[query_index * top_k..(query_index + 1) * top_k]
                    .iter()
                    .filter(|row_id| expected.contains(row_id))
                    .count();
            }
            let recall = overlap as f32 / (query_rows.len() * top_k) as f32;
            assert!(
                recall >= 0.75,
                "{metric:?} recall@{top_k} was {recall} with l_search=96 << {count}"
            );
        }
    }

    #[test]
    fn diskann_config_rejects_invalid_shape_and_build_parameters() {
        for (extra, expected) in [
            (vec![("dimension", "1025")], "at most 1024"),
            (vec![("pq.m", "0")], "pq.m must be greater than 0"),
            (vec![("pq.m", "129")], "must not exceed dimension"),
            (vec![("pq.bits", "6")], "pq.bits must be 4 or 8"),
            (
                vec![("diskann.raw-vector-encoding", "bf16")],
                "must be auto, f32, or f16",
            ),
            (
                vec![
                    ("diskann.max-degree", "65"),
                    ("diskann.build-search-list-size", "64"),
                ],
                "build search-list size",
            ),
            (vec![("diskann.alpha", "0.9")], "alpha must be at least 1"),
            (
                vec![
                    ("diskann.max-degree", "1024"),
                    ("diskann.build-search-list-size", "1024"),
                ],
                "adjacency list",
            ),
            (
                vec![
                    ("dimension", "1024"),
                    ("pq.m", "16"),
                    ("diskann.raw-vector-encoding", "f32"),
                    ("diskann.storage-layout", "interleaved"),
                ],
                "raw vector and maximum adjacency",
            ),
        ] {
            let mut values = vec![
                ("index.type", "diskann"),
                ("dimension", "128"),
                ("metric", "l2"),
                ("pq.m", "16"),
            ];
            for replacement in extra {
                if let Some(existing) = values.iter_mut().find(|item| item.0 == replacement.0) {
                    *existing = replacement;
                } else {
                    values.push(replacement);
                }
            }

            let error = VectorIndexConfig::from_options(&options(&values))
                .expect_err("invalid DiskANN config should be rejected");
            assert!(
                error.to_string().contains(expected),
                "error '{}' should contain '{}'",
                error,
                expected
            );
        }
    }

    #[test]
    fn diskann_trainer_uses_configured_dimension() {
        let trainer = VectorIndexTrainer::new(VectorIndexConfig::DiskAnn {
            dimension: 8,
            metric: MetricType::L2,
            pq_m: 2,
            pq_bits: 8,
            build: DiskAnnBuildParams::default(),
        })
        .expect("DiskANN trainer should open");

        assert_eq!(trainer.dimension(), 8);
    }

    #[test]
    fn diskann_trainer_bounds_and_deterministically_reservoir_samples_training_data() {
        let dimension = 2;
        let count = DISKANN_MAX_PQ_TRAINING_VECTORS + 257;
        let data = (0..count * dimension)
            .map(|offset| offset as f32)
            .collect::<Vec<_>>();
        let config = || VectorIndexConfig::DiskAnn {
            dimension,
            metric: MetricType::L2,
            pq_m: 1,
            pq_bits: 8,
            build: DiskAnnBuildParams {
                seed: 73,
                ..DiskAnnBuildParams::default()
            },
        };

        let mut whole = VectorIndexTrainer::new(config()).unwrap();
        whole.add_training_vectors_mut(&data, count).unwrap();
        let mut batched = VectorIndexTrainer::new(config()).unwrap();
        for vectors in data.chunks(137 * dimension) {
            batched
                .add_training_vectors_mut(vectors, vectors.len() / dimension)
                .unwrap();
        }

        assert_eq!(whole.training_vector_count, DISKANN_MAX_PQ_TRAINING_VECTORS);
        assert_eq!(whole.training_vectors_seen, count);
        assert_eq!(
            whole.training_data.len(),
            DISKANN_MAX_PQ_TRAINING_VECTORS * dimension
        );
        assert_eq!(whole.training_data, batched.training_data);
        assert_ne!(
            &whole.training_data,
            &data[..DISKANN_MAX_PQ_TRAINING_VECTORS * dimension],
            "reservoir sampling must give tail vectors a chance to enter the sample"
        );
    }

    #[test]
    fn diskann_trainer_derives_reservoir_limit_from_memory_budget() {
        let dimension = 1024;
        let memory_budget_bytes = 128 * 1024 * 1024;
        let config = VectorIndexConfig::DiskAnn {
            dimension,
            metric: MetricType::L2,
            pq_m: 256,
            pq_bits: 8,
            build: DiskAnnBuildParams {
                memory_budget_bytes,
                ..DiskAnnBuildParams::default()
            },
        };
        let trainer = VectorIndexTrainer::new(config).unwrap();
        assert!(trainer.training_sample_limit < DISKANN_MAX_PQ_TRAINING_VECTORS);
        assert_eq!(
            trainer.training_sample_limit,
            diskann_training_sample_limit(dimension, MetricType::L2, 256, 8, memory_budget_bytes,)
                .unwrap()
        );
    }

    #[test]
    fn diskann_trainer_trains_existing_product_quantizer() {
        let dimension = 8;
        let count = 256;
        let data = generate_clustered_data(count, dimension, 16);
        let training = VectorIndexTrainer::train(
            VectorIndexConfig::DiskAnn {
                dimension,
                metric: MetricType::L2,
                pq_m: 2,
                pq_bits: 8,
                build: DiskAnnBuildParams::default(),
            },
            &data,
            count,
        )
        .expect("DiskANN PQ training should succeed");

        assert_eq!(training.index_type(), IndexType::DiskAnn);
        assert_eq!(training.dimension(), dimension);
    }

    #[test]
    fn diskann_writer_accumulates_multiple_vector_batches() {
        let dimension = 8;
        let training_count = 256;
        let training_data = generate_clustered_data(training_count, dimension, 16);
        let training = VectorIndexTrainer::train(
            VectorIndexConfig::DiskAnn {
                dimension,
                metric: MetricType::L2,
                pq_m: 2,
                pq_bits: 8,
                build: DiskAnnBuildParams::default(),
            },
            &training_data,
            training_count,
        )
        .unwrap();
        let mut writer = VectorIndexWriter::new(training);

        writer
            .add_vectors(&[10, 11], &training_data[..2 * dimension], 2)
            .unwrap();
        writer
            .add_vectors(&[12], &training_data[2 * dimension..3 * dimension], 1)
            .unwrap();

        let VectorIndexWriter::DiskAnn(index) = writer else {
            panic!("expected DiskANN writer");
        };
        assert_eq!(index.ids, vec![10, 11, 12]);
        assert_eq!(index.vectors, training_data[..3 * dimension]);
    }

    #[test]
    fn config_from_options_parses_all_index_types() {
        assert_eq!(
            VectorIndexConfig::from_options(&options(&[
                ("index.type", "ivf_flat"),
                ("dimension", "8"),
                ("nlist", "4"),
                ("metric", "l2"),
            ]))
            .unwrap()
            .index_type(),
            IndexType::IvfFlat
        );

        match VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_pq"),
            ("dimension", "16"),
            ("nlist", "4"),
            ("metric", "l2"),
            ("use-opq", "true"),
        ]))
        .unwrap()
        {
            VectorIndexConfig::IvfPq { m, use_opq, .. } => {
                assert_eq!(m, 4);
                assert!(use_opq);
            }
            _ => panic!("expected IVF PQ config"),
        }

        match VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_rq"),
            ("dimension", "8"),
            ("nlist", "4"),
            ("metric", "cosine"),
        ]))
        .unwrap()
        {
            VectorIndexConfig::IvfRq {
                dimension,
                nlist,
                bits,
                metric,
            } => {
                assert_eq!(dimension, 8);
                assert_eq!(nlist, 4);
                assert_eq!(bits, DEFAULT_RQ_BITS);
                assert_eq!(metric, MetricType::Cosine);
            }
            _ => panic!("expected IVF RQ config"),
        }

        match VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_sq"),
            ("dimension", "8"),
            ("nlist", "4"),
            ("metric", "inner_product"),
        ]))
        .unwrap()
        {
            VectorIndexConfig::IvfSq {
                dimension,
                nlist,
                metric,
            } => {
                assert_eq!(dimension, 8);
                assert_eq!(nlist, 4);
                assert_eq!(metric, MetricType::InnerProduct);
            }
            _ => panic!("expected IVF-SQ config"),
        }
    }

    #[test]
    fn unified_trainer_rejects_non_finite_training_data() {
        for (value, expected) in [
            (
                f32::NAN,
                "training data contains non-finite value at offset 0: NaN",
            ),
            (
                f32::INFINITY,
                "training data contains non-finite value at offset 0: inf",
            ),
            (
                f32::NEG_INFINITY,
                "training data contains non-finite value at offset 0: -inf",
            ),
        ] {
            assert_invalid_input_contains(
                VectorIndexTrainer::train(
                    VectorIndexConfig::IvfFlat {
                        dimension: 1,
                        nlist: 1,
                        metric: MetricType::L2,
                    },
                    &[value, 1.0],
                    2,
                )
                .map(|_| ()),
                expected,
            );
        }
    }

    #[test]
    fn config_from_options_rejects_unknown_options() {
        let err = VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_flat"),
            ("dimension", "8"),
            ("nlist", "4"),
            ("metric", "l2"),
            ("unused", "value"),
        ]))
        .unwrap_err();

        assert!(err.to_string().contains("unknown vector index option"));
    }

    #[test]
    fn config_from_options_rejects_alias_keys_and_values() {
        let err = VectorIndexConfig::from_options(&options(&[
            ("type", "ivf_flat"),
            ("dimension", "8"),
            ("nlist", "4"),
        ]))
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("missing required option 'index.type'"));

        let err = VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf-flat"),
            ("dimension", "8"),
            ("nlist", "4"),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("unknown index.type"));

        let err = VectorIndexConfig::from_options(&options(&[
            ("index.type", "IVF_FLAT"),
            ("dimension", "8"),
            ("nlist", "4"),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("unknown index.type"));

        let err = VectorIndexConfig::from_options(&options(&[
            ("index.type", "ivf_flat"),
            ("dimension", "8"),
            ("nlist", "4"),
            ("metric", "ip"),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("unknown metric"));
    }

    #[test]
    fn unified_writer_rejects_non_finite_vector_data() {
        for (value, expected) in [
            (
                f32::NAN,
                "vector data contains non-finite value at offset 0: NaN",
            ),
            (
                f32::INFINITY,
                "vector data contains non-finite value at offset 0: inf",
            ),
            (
                f32::NEG_INFINITY,
                "vector data contains non-finite value at offset 0: -inf",
            ),
        ] {
            let mut writer = build_writer(
                VectorIndexConfig::IvfFlat {
                    dimension: 1,
                    nlist: 1,
                    metric: MetricType::L2,
                },
                &[0.0, 1.0],
                2,
            );
            assert_invalid_input_contains(writer.add_vectors(&[1, 2], &[value, 1.0], 2), expected);
        }
    }

    #[test]
    fn unified_reader_rejects_non_finite_query() {
        let mut reader = build_ivfflat_reader();
        let err = reader
            .search(&[f32::NAN], VectorSearchParams::new(1, 1))
            .expect_err("non-finite query should be rejected");

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err
            .to_string()
            .contains("query contains non-finite value at offset 0: NaN"));
    }

    #[test]
    fn unified_reader_rejects_non_finite_batch_query() {
        let mut reader = build_ivfflat_reader();
        let err = reader
            .search_batch(&[f32::NEG_INFINITY], 1, VectorSearchParams::new(1, 1))
            .expect_err("non-finite batch query should be rejected");

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err
            .to_string()
            .contains("queries contains non-finite value at offset 0: -inf"));
    }

    #[test]
    fn unified_reader_rejects_non_finite_query_before_decoding_filter() {
        let mut reader = build_ivfflat_reader();
        let err = reader
            .search_with_roaring_filter(&[f32::NAN], VectorSearchParams::new(1, 1), &[0xFF])
            .expect_err("non-finite filtered query should be rejected");

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err
            .to_string()
            .contains("query contains non-finite value at offset 0: NaN"));
        assert!(!err.to_string().contains("invalid RoaringTreemap"));
    }
}
