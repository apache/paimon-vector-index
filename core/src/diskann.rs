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

use crate::distance::{preprocess_vectors, MetricType};
use crate::kmeans::KMeansConfig;
use crate::pq::ProductQuantizer;
use crate::vamana::{
    estimate_sharded_vamana_memory_bytes, estimate_vamana_memory_bytes, VamanaGraph,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::borrow::Cow;
use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

pub(crate) const DISKANN_ADJACENCY_LOCATOR_NODE_BYTES: usize = 4;
pub(crate) const DISKANN_ADJACENCY_LOCATOR_BLOCK_NODES: usize = 16;
/// Match the proven DiskANN training bound: more samples materially increase
/// memory and training time without consistently improving the codebook.
pub const DISKANN_MAX_PQ_TRAINING_VECTORS: usize = 50_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskAnnStorageLayout {
    /// Keep compressed adjacency pages and dense raw-vector records in separate sections.
    Compact,
    /// Store each raw vector immediately before its compressed adjacency list.
    Interleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DiskAnnRawVectorEncoding {
    /// Preserve indexed vectors and final distances as little-endian `f32`.
    F32 = 1,
    /// Store little-endian IEEE 754 binary16 values for approximate final reranking.
    F16 = 2,
}

impl DiskAnnRawVectorEncoding {
    pub(crate) const fn element_size(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F16 => size_of::<u16>(),
        }
    }

    pub(crate) const fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::F32),
            2 => Some(Self::F16),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskAnnBuildDistance {
    /// Use full-precision distances for graph traversal and robust pruning.
    FullPrecision,
    /// Use PQ distances for graph traversal and full precision for robust pruning.
    ProductQuantized,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskAnnBuildParams {
    pub max_degree: usize,
    pub build_search_list_size: usize,
    pub alpha: f32,
    pub seed: u64,
    pub memory_budget_bytes: usize,
    pub storage_layout: DiskAnnStorageLayout,
    pub raw_vector_encoding: DiskAnnRawVectorEncoding,
    pub build_distance: DiskAnnBuildDistance,
}

impl Default for DiskAnnBuildParams {
    fn default() -> Self {
        Self {
            max_degree: 64,
            build_search_list_size: 100,
            alpha: 1.2,
            seed: 42,
            memory_budget_bytes: 8 * 1024 * 1024 * 1024,
            storage_layout: DiskAnnStorageLayout::Compact,
            raw_vector_encoding: DiskAnnRawVectorEncoding::F16,
            build_distance: DiskAnnBuildDistance::ProductQuantized,
        }
    }
}

pub(crate) fn validate_diskann_format_configuration(
    dimension: usize,
    pq_m: usize,
    pq_bits: usize,
    build: DiskAnnBuildParams,
) -> io::Result<()> {
    if dimension == 0 {
        return Err(invalid_input("DiskANN dimension must be greater than 0"));
    }
    if dimension > 1024 {
        return Err(invalid_input("DiskANN v1 dimension must be at most 1024"));
    }
    if pq_m == 0 {
        return Err(invalid_input("DiskANN pq.m must be greater than 0"));
    }
    if pq_m > dimension {
        return Err(invalid_input(format!(
            "DiskANN pq.m {} must not exceed dimension {}",
            pq_m, dimension
        )));
    }
    if !matches!(pq_bits, 4 | 8) {
        return Err(invalid_input("DiskANN pq.bits must be 4 or 8"));
    }
    if build.max_degree == 0 {
        return Err(invalid_input(
            "DiskANN maximum degree must be greater than 0",
        ));
    }
    if build.max_degree > 1023 {
        return Err(invalid_input(format!(
            "DiskANN adjacency list size {} exceeds the v1 1023-neighbor page limit",
            build.max_degree.saturating_mul(size_of::<u32>())
        )));
    }
    if build.build_search_list_size < build.max_degree {
        return Err(invalid_input(format!(
            "DiskANN build search-list size {} must be at least maximum degree {}",
            build.build_search_list_size, build.max_degree
        )));
    }
    if u32::try_from(build.build_search_list_size).is_err() {
        return Err(invalid_input("DiskANN build search-list size exceeds u32"));
    }
    if !build.alpha.is_finite() || build.alpha < 1.0 {
        return Err(invalid_input("DiskANN alpha must be at least 1 and finite"));
    }
    let interleaved_record_bytes = dimension
        .checked_mul(build.raw_vector_encoding.element_size())
        .and_then(|vector_bytes| {
            build
                .max_degree
                .checked_mul(size_of::<u32>())
                .and_then(|adjacency_bytes| vector_bytes.checked_add(adjacency_bytes))
        });
    if build.storage_layout == DiskAnnStorageLayout::Interleaved
        && interleaved_record_bytes.is_none_or(|record_bytes| record_bytes > 4096)
    {
        return Err(invalid_input(
            "DiskANN interleaved raw vector and maximum adjacency list must fit in one page",
        ));
    }
    Ok(())
}

pub(crate) fn validate_diskann_training_budget(
    dimension: usize,
    metric: MetricType,
    pq_m: usize,
    pq_bits: usize,
    memory_budget_bytes: usize,
) -> io::Result<()> {
    let minimum_training_vectors = 1usize << pq_bits;
    pq_training_plan_with_sample_buffers(
        dimension,
        pq_m,
        minimum_training_vectors,
        minimum_training_vectors,
        memory_budget_bytes,
        usize::from(metric == MetricType::Cosine) + 1,
    )
    .map(|_| ())
    .map_err(|error| {
        invalid_input(format!(
            "DiskANN memory budget cannot fit minimum PQ training: {error}"
        ))
    })
}

pub(crate) fn diskann_training_sample_limit(
    dimension: usize,
    metric: MetricType,
    pq_m: usize,
    pq_bits: usize,
    memory_budget_bytes: usize,
) -> io::Result<usize> {
    pq_training_plan_with_sample_buffers(
        dimension,
        pq_m,
        1usize << pq_bits,
        DISKANN_MAX_PQ_TRAINING_VECTORS,
        memory_budget_bytes,
        usize::from(metric == MetricType::Cosine) + 1,
    )
    .map(|plan| plan.sample_count)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DiskAnnBuildStats {
    /// One for the normal parallel build; greater than one when the memory
    /// budget selected overlapping shard construction.
    pub graph_shards: usize,
    pub total: Duration,
    pub pq_encoding: Duration,
    pub vamana_initialization: Duration,
    pub vamana_pass_one: Duration,
    pub vamana_pass_two: Duration,
    pub connectivity_repair: Duration,
    pub locality_remap: Duration,
    pub resident_serialization: Duration,
    pub adjacency_serialization: Duration,
    pub vector_serialization: Duration,
}

impl DiskAnnBuildStats {
    pub fn accounted_duration(self) -> Duration {
        [
            self.pq_encoding,
            self.vamana_initialization,
            self.vamana_pass_one,
            self.vamana_pass_two,
            self.connectivity_repair,
            self.locality_remap,
            self.resident_serialization,
            self.adjacency_serialization,
            self.vector_serialization,
        ]
        .into_iter()
        .sum()
    }
}

pub struct DiskAnnIndex {
    pub d: usize,
    pub metric: MetricType,
    pub pq: ProductQuantizer,
    pub build_params: DiskAnnBuildParams,
    pub ids: Vec<i64>,
    pub vectors: Vec<f32>,
}

impl DiskAnnIndex {
    pub fn new(
        d: usize,
        metric: MetricType,
        pq_m: usize,
        build_params: DiskAnnBuildParams,
    ) -> Self {
        Self::with_pq_bits(d, metric, pq_m, 8, build_params)
    }

    pub fn with_pq_bits(
        d: usize,
        metric: MetricType,
        pq_m: usize,
        pq_bits: usize,
        build_params: DiskAnnBuildParams,
    ) -> Self {
        Self {
            d,
            metric,
            pq: ProductQuantizer::with_nbits_balanced(d, pq_m, pq_bits),
            build_params,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Creates an empty index that reuses the trained product quantizer.
    pub(crate) fn from_trained(trained: &DiskAnnIndex) -> Self {
        Self {
            d: trained.d,
            metric: trained.metric,
            pq: ProductQuantizer {
                d: trained.pq.d,
                m: trained.pq.m,
                nbits: trained.pq.nbits,
                dsub: trained.pq.dsub,
                ksub: trained.pq.ksub,
                chunk_offsets: trained.pq.chunk_offsets.clone(),
                centroids: trained.pq.centroids.clone(),
                centroid_norms_cache: trained.pq.centroid_norms_cache.clone(),
            },
            build_params: trained.build_params,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    pub fn train(&mut self, data: &[f32], n: usize) -> io::Result<()> {
        if n == 0 {
            return Err(invalid_input(
                "DiskANN training vector count must be greater than zero",
            ));
        }
        let expected_len = n
            .checked_mul(self.d)
            .ok_or_else(|| invalid_input("DiskANN training data length overflows usize"))?;
        if data.len() != expected_len {
            return Err(invalid_input(format!(
                "DiskANN training data length {} does not match n * dimension {}",
                data.len(),
                expected_len
            )));
        }
        let plan = self.training_plan(n)?;
        let sample =
            bounded_pq_training_sample(data, n, self.d, plan.sample_count, self.build_params.seed);
        let sample = sample
            .as_deref()
            .unwrap_or(&data[..plan.sample_count.saturating_mul(self.d)]);
        let processed = self.preprocess_vectors(sample, plan.sample_count);
        self.pq.train_hot_start_with_parallelism(
            &processed,
            plan.sample_count,
            &KMeansConfig::default(),
            false,
            plan.parallelism,
        );
        Ok(())
    }

    fn training_plan(&self, n: usize) -> io::Result<PqTrainingPlan> {
        pq_training_plan_with_sample_buffers(
            self.d,
            self.pq.m,
            self.pq.ksub,
            n,
            self.build_params.memory_budget_bytes,
            usize::from(self.metric == MetricType::Cosine) + 1,
        )
    }

    pub fn add(&mut self, data: &[f32], ids: &[i64]) {
        self.ids.extend_from_slice(ids);
        self.vectors
            .extend_from_slice(self.preprocess_vectors(data, ids.len()).as_ref());
    }

    pub fn estimate_build_memory_bytes(&self) -> io::Result<usize> {
        let n = self.ids.len();
        let workers = rayon::current_num_threads().max(1);
        let raw_vectors = checked_bytes(self.vectors.len(), size_of::<f32>(), "raw vectors")?;
        let row_ids = checked_bytes(n, size_of::<i64>(), "row IDs")?;
        let row_id_encoding_scratch = row_id_encoding_scratch_bytes(n)?;
        let pq_codes = checked_bytes(n, self.pq.code_size(), "PQ codes")?;
        let pq_codebook = checked_bytes(self.pq.centroids.len(), size_of::<f32>(), "PQ codebook")?;
        let pq_build_distances = if self.build_params.build_distance
            == DiskAnnBuildDistance::ProductQuantized
        {
            self.pq
                .m
                .checked_mul(self.pq.ksub)
                .and_then(|value| value.checked_mul(self.pq.ksub))
                .and_then(|value| value.checked_mul(size_of::<f32>()))
                .ok_or_else(|| invalid_input("DiskANN PQ build-distance table size overflows"))?
        } else {
            0
        };
        let row_id_order = checked_bytes(n, size_of::<u32>(), "row-ID order")?;
        let adjacency_index = checked_bytes(
            n,
            DISKANN_ADJACENCY_LOCATOR_NODE_BYTES,
            "adjacency locators",
        )?
        .checked_add(checked_bytes(
            n.div_ceil(DISKANN_ADJACENCY_LOCATOR_BLOCK_NODES),
            size_of::<u64>(),
            "adjacency locator block offsets",
        )?)
        .ok_or_else(|| invalid_input("DiskANN adjacency index size overflows usize"))?;
        let vamana = estimate_vamana_memory_bytes(
            n,
            self.build_params.max_degree,
            self.build_params.build_search_list_size,
            workers,
        )
        .ok_or_else(|| invalid_input("DiskANN Vamana memory estimate overflows usize"))?;
        let graph_stage_peak = vamana.build_peak_bytes.max(vamana.remap_peak_bytes);

        [
            raw_vectors,
            row_ids,
            row_id_encoding_scratch,
            pq_codes,
            pq_codebook,
            pq_build_distances,
            row_id_order,
            adjacency_index,
            graph_stage_peak,
        ]
        .into_iter()
        .try_fold(0usize, |total, value| {
            total
                .checked_add(value)
                .ok_or_else(|| invalid_input("DiskANN memory estimate overflows usize"))
        })
    }

    fn graph_build_shard_count(&self) -> io::Result<usize> {
        let estimated = self.estimate_build_memory_bytes()?;
        if estimated <= self.build_params.memory_budget_bytes {
            return Ok(1);
        }
        let n = self.ids.len();
        let workers = rayon::current_num_threads().max(1);
        let vamana = estimate_vamana_memory_bytes(
            n,
            self.build_params.max_degree,
            self.build_params.build_search_list_size,
            workers,
        )
        .ok_or_else(|| invalid_input("DiskANN Vamana memory estimate overflows usize"))?;
        let graph_peak = vamana.build_peak_bytes.max(vamana.remap_peak_bytes);
        let fixed_bytes = estimated
            .checked_sub(graph_peak)
            .ok_or_else(|| invalid_input("DiskANN fixed memory estimate underflows"))?;
        let max_shards = 64.min(n / 2);
        for shard_count in 2..=max_shards {
            let Some(sharded_graph) = estimate_sharded_vamana_memory_bytes(
                n,
                self.d,
                self.build_params.max_degree.min(n.saturating_sub(1)),
                shard_count,
                if self.build_params.build_distance == DiskAnnBuildDistance::ProductQuantized {
                    self.pq.code_size()
                } else {
                    0
                },
            ) else {
                continue;
            };
            if fixed_bytes
                .checked_add(sharded_graph)
                .is_some_and(|peak| peak <= self.build_params.memory_budget_bytes)
            {
                return Ok(shard_count);
            }
        }
        Err(invalid_input(format!(
            "DiskANN estimated build memory {} exceeds memory budget {}; overlapping sharded build also cannot fit",
            estimated, self.build_params.memory_budget_bytes
        )))
    }

    pub(crate) fn validate_for_write(&self) -> io::Result<()> {
        validate_diskann_format_configuration(self.d, self.pq.m, self.pq.nbits, self.build_params)?;
        validate_diskann_training_budget(
            self.d,
            self.metric,
            self.pq.m,
            self.pq.nbits,
            self.build_params.memory_budget_bytes,
        )?;
        if self.build_params.memory_budget_bytes == 0 {
            return Err(invalid_input(
                "DiskANN memory budget must be greater than zero",
            ));
        }
        if self.ids.is_empty() || u32::try_from(self.ids.len()).is_err() {
            return Err(invalid_input(
                "DiskANN vector count must be between 1 and u32::MAX",
            ));
        }
        let expected_vectors = self
            .ids
            .len()
            .checked_mul(self.d)
            .ok_or_else(|| invalid_input("DiskANN vector shape overflows usize"))?;
        if self.vectors.len() != expected_vectors {
            return Err(invalid_input(format!(
                "DiskANN vector length {} does not match {} row IDs * dimension {}",
                self.vectors.len(),
                self.ids.len(),
                self.d
            )));
        }
        if let Some(offset) = self.vectors.iter().position(|value| !value.is_finite()) {
            return Err(invalid_input(format!(
                "DiskANN vector data contains a non-finite value at offset {}",
                offset
            )));
        }
        if self.build_params.raw_vector_encoding == DiskAnnRawVectorEncoding::F16 {
            if let Some(offset) = self
                .vectors
                .iter()
                .position(|&value| !half::f16::from_f32(value).is_finite())
            {
                return Err(invalid_input(format!(
                    "DiskANN vector data at offset {} is outside the finite f16 range",
                    offset
                )));
            }
        }

        let expected_ksub = 1usize
            .checked_shl(self.pq.nbits as u32)
            .ok_or_else(|| invalid_input("DiskANN PQ centroid count overflows usize"))?;
        let expected_centroids = self
            .d
            .checked_mul(expected_ksub)
            .ok_or_else(|| invalid_input("DiskANN PQ codebook shape overflows usize"))?;
        if self.pq.d != self.d
            || self.pq.ksub != expected_ksub
            || self.pq.centroids.len() != expected_centroids
            || !self.pq.has_valid_layout()
        {
            return Err(invalid_input("DiskANN PQ codebook shape is invalid"));
        }
        if let Some(offset) = self
            .pq
            .centroids
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(invalid_input(format!(
                "DiskANN PQ codebook contains a non-finite value at offset {}",
                offset
            )));
        }
        Ok(())
    }

    pub(crate) fn prepare_build(&self) -> io::Result<PreparedDiskAnn> {
        self.validate_for_write()?;
        let graph_shards = self.graph_build_shard_count()?;

        let pq_started = Instant::now();
        let mut pq_codes = vec![0u8; self.ids.len() * self.pq.code_size()];
        self.pq
            .encode_batch(&self.vectors, self.ids.len(), &mut pq_codes);
        let pq_encoding = pq_started.elapsed();
        let (mut graph, vamana_stats) = if graph_shards > 1 {
            match self.build_params.build_distance {
                DiskAnnBuildDistance::FullPrecision => VamanaGraph::build_sharded_with_stats(
                    &self.vectors,
                    self.ids.len(),
                    self.d,
                    self.graph_metric(),
                    self.build_params,
                    graph_shards,
                )?,
                DiskAnnBuildDistance::ProductQuantized => VamanaGraph::build_sharded_with_pq_stats(
                    &self.vectors,
                    &self.pq,
                    &pq_codes,
                    self.ids.len(),
                    self.d,
                    self.graph_metric(),
                    self.build_params,
                    graph_shards,
                )?,
            }
        } else {
            match self.build_params.build_distance {
                DiskAnnBuildDistance::FullPrecision => VamanaGraph::build_with_stats(
                    &self.vectors,
                    self.ids.len(),
                    self.d,
                    self.graph_metric(),
                    self.build_params,
                )?,
                DiskAnnBuildDistance::ProductQuantized => VamanaGraph::build_with_pq_stats(
                    &self.vectors,
                    &self.pq,
                    &pq_codes,
                    self.ids.len(),
                    self.d,
                    self.graph_metric(),
                    self.build_params,
                )?,
            }
        };
        let locality_started = Instant::now();
        let permutation = bfs_locality_permutation(&graph);
        remap_graph_in_place(&mut graph, &permutation);
        let locality_remap = locality_started.elapsed();
        Ok(PreparedDiskAnn {
            graph,
            permutation,
            pq_codes,
            stats: DiskAnnBuildStats {
                graph_shards,
                pq_encoding,
                vamana_initialization: vamana_stats.initialization,
                vamana_pass_one: vamana_stats.pass_one,
                vamana_pass_two: vamana_stats.pass_two,
                connectivity_repair: vamana_stats.connectivity_repair,
                locality_remap,
                ..DiskAnnBuildStats::default()
            },
        })
    }

    fn graph_metric(&self) -> MetricType {
        match self.metric {
            // Cosine vectors are normalized during training and ingestion, so
            // squared L2 has identical ordering and a well-behaved prune ratio.
            MetricType::Cosine => MetricType::L2,
            metric => metric,
        }
    }

    fn preprocess_vectors<'a>(&self, data: &'a [f32], count: usize) -> Cow<'a, [f32]> {
        if self.metric == MetricType::Cosine {
            Cow::Owned(preprocess_vectors(data, count, self.d, self.metric))
        } else {
            Cow::Borrowed(&data[..count * self.d])
        }
    }
}

fn bounded_pq_training_sample(
    data: &[f32],
    count: usize,
    dimension: usize,
    sample_limit: usize,
    seed: u64,
) -> Option<Vec<f32>> {
    if count <= sample_limit {
        return None;
    }
    let mut sample = Vec::with_capacity(sample_limit.saturating_mul(dimension));
    sample.extend_from_slice(&data[..sample_limit * dimension]);
    let mut rng = StdRng::seed_from_u64(seed);
    for (stream_index, vector) in data
        .chunks_exact(dimension)
        .enumerate()
        .skip(sample_limit)
        .take(count - sample_limit)
    {
        let replacement = rng.gen_range(0..=stream_index);
        if replacement < sample_limit {
            let start = replacement * dimension;
            sample[start..start + dimension].copy_from_slice(vector);
        }
    }
    Some(sample)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PqTrainingPlan {
    sample_count: usize,
    parallelism: usize,
}

#[cfg(test)]
fn pq_training_plan(
    dimension: usize,
    pq_m: usize,
    ksub: usize,
    vector_count: usize,
    memory_budget_bytes: usize,
) -> io::Result<PqTrainingPlan> {
    pq_training_plan_with_sample_buffers(
        dimension,
        pq_m,
        ksub,
        vector_count,
        memory_budget_bytes,
        1,
    )
}

fn pq_training_plan_with_sample_buffers(
    dimension: usize,
    pq_m: usize,
    ksub: usize,
    vector_count: usize,
    memory_budget_bytes: usize,
    sample_buffer_count: usize,
) -> io::Result<PqTrainingPlan> {
    if vector_count == 0 {
        return Ok(PqTrainingPlan {
            sample_count: 0,
            parallelism: 1,
        });
    }
    let max_sample = vector_count.min(DISKANN_MAX_PQ_TRAINING_VECTORS);
    let min_sample = vector_count.min(ksub.max(1));
    let max_chunk_dimension = dimension.div_ceil(pq_m.max(1));
    let current_parallelism = rayon::current_num_threads().max(1).min(pq_m.max(1));
    let peak_for = |sample_count: usize, parallelism: usize| -> Option<usize> {
        let sample_bytes = sample_count
            .checked_mul(dimension)?
            .checked_mul(size_of::<f32>())?
            .checked_mul(sample_buffer_count)?;
        let codebook_bytes = dimension.checked_mul(ksub)?.checked_mul(size_of::<f32>())?;
        let subvector_copy_bytes = sample_count
            .checked_mul(max_chunk_dimension)?
            .checked_mul(2 * size_of::<f32>())?;
        let assignment_bytes = sample_count.checked_mul(size_of::<usize>() + size_of::<f32>())?;
        let score_matrix_bytes = sample_count
            .checked_mul(ksub)?
            .min(4 * 1024 * 1024)
            .checked_mul(size_of::<f32>())?;
        let centroid_scratch_bytes = max_chunk_dimension
            .checked_mul(ksub)?
            .checked_mul(3 * size_of::<f32>())?;
        let per_task = [
            subvector_copy_bytes,
            assignment_bytes,
            score_matrix_bytes,
            centroid_scratch_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))?;
        sample_bytes
            .checked_add(codebook_bytes)?
            .checked_add(per_task.checked_mul(parallelism)?)
    };
    if peak_for(min_sample, 1).is_none_or(|peak| peak > memory_budget_bytes) {
        return Err(invalid_input(format!(
            "{} bytes is below the estimated one-worker PQ training peak",
            memory_budget_bytes
        )));
    }

    let mut low = min_sample;
    let mut high = max_sample;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if peak_for(middle, 1).is_some_and(|peak| peak <= memory_budget_bytes) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let sample_count = low;
    let parallelism = (1..=current_parallelism)
        .rev()
        .find(|&parallelism| {
            peak_for(sample_count, parallelism).is_some_and(|peak| peak <= memory_budget_bytes)
        })
        .unwrap_or(1);
    Ok(PqTrainingPlan {
        sample_count,
        parallelism,
    })
}

fn checked_bytes(count: usize, item_size: usize, name: &str) -> io::Result<usize> {
    count
        .checked_mul(item_size)
        .ok_or_else(|| invalid_input(format!("DiskANN {} byte size overflows usize", name)))
}

fn row_id_encoding_scratch_bytes(count: usize) -> io::Result<usize> {
    checked_bytes(count, size_of::<i64>(), "row-ID encoding scratch")
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LocalityPermutation {
    pub(crate) new_to_old: Vec<u32>,
    pub(crate) old_to_new: Vec<u32>,
}

pub(crate) fn bfs_locality_permutation(graph: &VamanaGraph) -> LocalityPermutation {
    let count = graph.adjacency.len();
    let mut visited = vec![false; count];
    let mut new_to_old = Vec::with_capacity(count);
    let entry = graph.entry_node as usize;
    if entry < count {
        visited[entry] = true;
        let mut queue = VecDeque::from([entry]);
        while let Some(node) = queue.pop_front() {
            new_to_old.push(node as u32);
            for &neighbor in &graph.adjacency[node] {
                let neighbor = neighbor as usize;
                if neighbor < count && !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    for (node, was_visited) in visited.into_iter().enumerate() {
        if !was_visited {
            new_to_old.push(node as u32);
        }
    }

    let mut old_to_new = vec![0u32; count];
    for (new_id, &old_id) in new_to_old.iter().enumerate() {
        old_to_new[old_id as usize] = new_id as u32;
    }
    LocalityPermutation {
        new_to_old,
        old_to_new,
    }
}

pub(crate) fn remap_graph_in_place(graph: &mut VamanaGraph, permutation: &LocalityPermutation) {
    graph
        .adjacency
        .permute_and_map_neighbors(&permutation.old_to_new);
    graph.entry_node = permutation.old_to_new[graph.entry_node as usize];
}

pub(crate) struct PreparedDiskAnn {
    pub(crate) graph: VamanaGraph,
    pub(crate) permutation: LocalityPermutation,
    pub(crate) pq_codes: Vec<u8>,
    pub(crate) stats: DiskAnnBuildStats,
}

impl PreparedDiskAnn {
    pub(crate) fn row_id(&self, index: &DiskAnnIndex, new_id: usize) -> i64 {
        index.ids[self.permutation.new_to_old[new_id] as usize]
    }

    pub(crate) fn vector<'a>(&self, index: &'a DiskAnnIndex, new_id: usize) -> &'a [f32] {
        let old_id = self.permutation.new_to_old[new_id] as usize;
        &index.vectors[old_id * index.d..(old_id + 1) * index.d]
    }

    pub(crate) fn pq_code<'a>(&'a self, index: &DiskAnnIndex, new_id: usize) -> &'a [u8] {
        let old_id = self.permutation.new_to_old[new_id] as usize;
        let code_size = index.pq.code_size();
        &self.pq_codes[old_id * code_size..(old_id + 1) * code_size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_level_diskann_training_uses_the_same_bounded_deterministic_reservoir() {
        let dimension = 2;
        let count = DISKANN_MAX_PQ_TRAINING_VECTORS + 257;
        let data = (0..count * dimension)
            .map(|offset| offset as f32)
            .collect::<Vec<_>>();

        let first = bounded_pq_training_sample(
            &data,
            count,
            dimension,
            DISKANN_MAX_PQ_TRAINING_VECTORS,
            73,
        )
        .unwrap();
        let second = bounded_pq_training_sample(
            &data,
            count,
            dimension,
            DISKANN_MAX_PQ_TRAINING_VECTORS,
            73,
        )
        .unwrap();
        let different_seed = bounded_pq_training_sample(
            &data,
            count,
            dimension,
            DISKANN_MAX_PQ_TRAINING_VECTORS,
            74,
        )
        .unwrap();

        assert_eq!(first.len(), DISKANN_MAX_PQ_TRAINING_VECTORS * dimension);
        assert_eq!(first, second);
        assert_ne!(first, different_seed);
        assert_ne!(first, data[..DISKANN_MAX_PQ_TRAINING_VECTORS * dimension]);
        assert!(bounded_pq_training_sample(
            &data[..DISKANN_MAX_PQ_TRAINING_VECTORS * dimension],
            DISKANN_MAX_PQ_TRAINING_VECTORS,
            dimension,
            DISKANN_MAX_PQ_TRAINING_VECTORS,
            73
        )
        .is_none());
    }

    #[test]
    fn diskann_pq_training_plan_throttles_parallelism_and_samples_to_budget() {
        let unrestricted =
            pq_training_plan(1024, 256, 256, 50_000, 8 * 1024 * 1024 * 1024).unwrap();
        assert_eq!(unrestricted.sample_count, 50_000);
        assert_eq!(
            unrestricted.parallelism,
            rayon::current_num_threads().min(256)
        );

        let bounded = pq_training_plan(1024, 256, 256, 50_000, 64 * 1024 * 1024).unwrap();
        assert!(bounded.sample_count < unrestricted.sample_count);
        assert!(bounded.parallelism <= unrestricted.parallelism);
        assert!(pq_training_plan(1024, 1, 256, 256, 1).is_err());
    }

    #[test]
    fn diskann_pq_training_plan_accounts_for_retained_and_normalized_samples() {
        let budget = 128 * 1024 * 1024;
        let retained =
            pq_training_plan_with_sample_buffers(1024, 256, 256, 50_000, budget, 1).unwrap();
        let cosine =
            pq_training_plan_with_sample_buffers(1024, 256, 256, 50_000, budget, 2).unwrap();
        assert!(retained.sample_count < 50_000);
        assert!(cosine.sample_count < retained.sample_count);
    }

    #[test]
    fn low_level_diskann_training_uses_metric_aware_budget_and_propagates_failure() {
        let budget = 128 * 1024 * 1024;
        let build_params = DiskAnnBuildParams {
            memory_budget_bytes: budget,
            ..DiskAnnBuildParams::default()
        };
        let l2 = DiskAnnIndex::new(1024, MetricType::L2, 256, build_params);
        let cosine = DiskAnnIndex::new(1024, MetricType::Cosine, 256, build_params);
        let l2_plan = l2.training_plan(50_000).unwrap();
        let cosine_plan = cosine.training_plan(50_000).unwrap();

        assert_eq!(
            l2_plan,
            pq_training_plan_with_sample_buffers(1024, 256, 256, 50_000, budget, 1).unwrap()
        );
        assert_eq!(
            cosine_plan,
            pq_training_plan_with_sample_buffers(1024, 256, 256, 50_000, budget, 2).unwrap()
        );
        assert!(cosine_plan.sample_count < l2_plan.sample_count);

        let mut infeasible = DiskAnnIndex::new(
            8,
            MetricType::Cosine,
            2,
            DiskAnnBuildParams {
                memory_budget_bytes: 1,
                ..DiskAnnBuildParams::default()
            },
        );
        let error = infeasible
            .train(&[0.0; 8], 1)
            .expect_err("an infeasible low-level training budget must be returned");
        assert!(error.to_string().contains("PQ training peak"));
    }

    #[test]
    fn diskann_index_rejects_build_exceeding_memory_budget() {
        let build_params = DiskAnnBuildParams {
            memory_budget_bytes: 1,
            ..DiskAnnBuildParams::default()
        };
        let mut index = DiskAnnIndex::new(8, MetricType::L2, 2, build_params);
        index.add(&[0.0; 8], &[7]);

        let error = index
            .graph_build_shard_count()
            .expect_err("build should fail before graph allocation");
        assert!(error.to_string().contains("memory budget"));
    }

    #[test]
    fn diskann_memory_estimate_accounts_for_one_compact_graph_during_remap() {
        let count = 1024;
        let max_degree = 1024;
        let estimate = estimate_vamana_memory_bytes(count, max_degree, 1024, 1)
            .unwrap()
            .remap_peak_bytes;
        let one_graph = count * (max_degree * size_of::<u32>() + size_of::<u16>());

        assert!(
            estimate >= one_graph && estimate < 2 * one_graph,
            "in-place remapping should retain one compact graph; estimate={estimate}, one_graph={one_graph}"
        );
    }

    #[test]
    fn diskann_memory_estimate_reserves_row_id_encoding_scratch() {
        assert_eq!(row_id_encoding_scratch_bytes(1024).unwrap(), 8 * 1024);
        assert!(row_id_encoding_scratch_bytes(usize::MAX).is_err());
    }

    #[test]
    fn diskann_memory_budget_automatically_selects_overlapping_shards() {
        // Pin the worker count so this test exercises the same memory-plan
        // relationship on small CI runners and developer machines.
        rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build()
            .unwrap()
            .install(|| {
                let count = 2_048;
                let dimension = 8;
                let params = DiskAnnBuildParams {
                    max_degree: 256,
                    build_search_list_size: 256,
                    ..DiskAnnBuildParams::default()
                };
                let mut index = DiskAnnIndex::new(dimension, MetricType::L2, 2, params);
                index.ids = (0..count as i64).collect();
                index.vectors = vec![0.0; count * dimension];
                let full_peak = index.estimate_build_memory_bytes().unwrap();
                index.build_params.memory_budget_bytes = full_peak - 1;

                let shard_count = index.graph_build_shard_count().unwrap();

                assert!(shard_count > 1);
                assert!(shard_count <= 64);
            });
    }

    #[test]
    fn diskann_locality_permutation_uses_bfs_then_unreachable_ids() {
        let graph =
            VamanaGraph::from_adjacency(2, vec![vec![], vec![], vec![3, 1], vec![4], vec![]]);

        let permutation = bfs_locality_permutation(&graph);

        assert_eq!(permutation.new_to_old, vec![2, 3, 1, 4, 0]);
        assert_eq!(permutation.old_to_new, vec![4, 2, 0, 1, 3]);
    }

    #[test]
    fn diskann_locality_remaps_entry_and_neighbor_ids() {
        let mut graph =
            VamanaGraph::from_adjacency(2, vec![vec![], vec![], vec![3, 1], vec![4], vec![]]);
        let permutation = bfs_locality_permutation(&graph);

        remap_graph_in_place(&mut graph, &permutation);

        assert_eq!(graph.entry_node, 0);
        assert_eq!(
            graph
                .adjacency
                .iter()
                .map(<[u32]>::to_vec)
                .collect::<Vec<_>>(),
            vec![vec![1, 2], vec![3], vec![], vec![], vec![]]
        );
    }

    #[test]
    fn diskann_prepare_build_keeps_rows_codes_and_vectors_aligned() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 29) % 113) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count).unwrap();
        index.add(&data[..indexed_count * dimension], &ids);

        let prepared = index.prepare_build().unwrap();

        assert_eq!(prepared.graph.entry_node, 0);
        assert_eq!(prepared.pq_codes.len(), indexed_count * 2);
        for new_id in 0..indexed_count {
            let old_id = prepared.permutation.new_to_old[new_id] as usize;
            assert_eq!(prepared.row_id(&index, new_id), ids[old_id]);
            assert_eq!(
                prepared.vector(&index, new_id),
                &data[old_id * dimension..(old_id + 1) * dimension]
            );
            assert_eq!(prepared.pq_code(&index, new_id).len(), 2);
        }
    }

    #[test]
    fn diskann_cosine_normalizes_vectors_before_pq_encoding_and_persistence() {
        let mut index =
            DiskAnnIndex::with_pq_bits(2, MetricType::Cosine, 1, 4, DiskAnnBuildParams::default());

        index.add(&[3.0, 4.0, 0.0, 0.0], &[10, 11]);

        assert_eq!(index.vectors, vec![0.6, 0.8, 0.0, 0.0]);
    }

    #[test]
    fn diskann_only_allocates_metric_preprocessing_for_cosine() {
        let data = [3.0, 4.0];
        for metric in [MetricType::L2, MetricType::InnerProduct] {
            let index = DiskAnnIndex::with_pq_bits(2, metric, 1, 4, DiskAnnBuildParams::default());
            assert!(matches!(
                index.preprocess_vectors(&data, 1),
                Cow::Borrowed(_)
            ));
        }
        let index =
            DiskAnnIndex::with_pq_bits(2, MetricType::Cosine, 1, 4, DiskAnnBuildParams::default());
        assert!(matches!(index.preprocess_vectors(&data, 1), Cow::Owned(_)));
    }
}
