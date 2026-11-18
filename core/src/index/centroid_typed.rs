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

use super::*;
use std::collections::BTreeMap;

const TYPED_MAGIC: &[u8; 8] = b"PIVFTM02";
const TYPED_HEADER: usize = 32;

impl VectorIndexTrainer {
    /// Parse the same build options as an ordinary trainer. DiskANN additionally accepts
    /// an outer `nlist`, defaulting to 64 when omitted or set to `auto`.
    pub fn centroid_from_options(options: &HashMap<String, String>) -> io::Result<Self> {
        let mut options = ConfigOptions::new(options)?;
        let kind = parse_index_type_option(&options.required("index.type")?)?;
        let diskann_count = if kind == IndexType::DiskAnn {
            match options.values.remove("nlist").as_deref().map(str::trim) {
                None | Some("auto") => Some(64),
                Some(value) => Some(parse_usize_option("nlist", value)?),
            }
        } else {
            None
        };
        let plan = VectorIndexBuildPlan::from_options(&options.values)?;
        let nlist = diskann_count.unwrap_or_else(|| plan.config.nlist());
        if let Some(max_bytes) = plan.objective.max_bytes_per_vector {
            validate_persisted_size_objective(
                &plan.config,
                plan.expected_vector_count,
                max_bytes,
                Some(nlist),
            )?;
        }
        Self::new_centroid(plan.into_uncalibrated_config()?, nlist)
    }

    /// A centroid build uses shared IVF training state, or a routing-only model for DiskANN.
    pub fn new_centroid(mut config: VectorIndexConfig, nlist: usize) -> io::Result<Self> {
        validate_positive(nlist, "centroid count")?;
        if config.index_type() != IndexType::DiskAnn && config.nlist() != nlist {
            return Err(invalid_input("centroid count differs from IVF nlist"));
        }
        // Each shard must be able to restore the model with its native reader. Check the
        // reader's allocation limit before constructing writers or allocating training buffers.
        let d = config.dimension();
        match &config {
            VectorIndexConfig::IvfFlat { .. } | VectorIndexConfig::DiskAnn { .. } => {
                crate::ivfflat_io::flat_metadata_bytes(d, nlist)?
            }
            VectorIndexConfig::IvfPq { m, use_opq, .. } => {
                crate::io::pq_metadata_bytes(d, nlist, *m, 256, *use_opq)?
            }
            VectorIndexConfig::IvfSq { .. } => crate::ivfsq_io::sq_metadata_bytes(d, nlist)?,
            VectorIndexConfig::IvfRq { .. } => {
                crate::ivfrq_io::rq_metadata_bytes(d, nlist, crate::rq::DEFAULT_RQ_ROTATION_ROUNDS)?
            }
        };
        // Training residuals and externally routed writes must use the same exact assignments.
        match &mut config {
            VectorIndexConfig::IvfFlat {
                use_approximate_coarse_assignment,
                ..
            }
            | VectorIndexConfig::IvfPq {
                use_approximate_coarse_assignment,
                ..
            }
            | VectorIndexConfig::IvfSq {
                use_approximate_coarse_assignment,
                ..
            }
            | VectorIndexConfig::IvfRq {
                use_approximate_coarse_assignment,
                ..
            } => *use_approximate_coarse_assignment = false,
            VectorIndexConfig::DiskAnn { .. } => {}
        }
        let mut trainer = Self::new(config)?;
        if let VectorIndexWriter::DiskAnn(index) = &trainer.writer {
            let mut router = IVFFlatIndex::new(index.d, nlist, index.metric);
            router.set_approximate_coarse_assignment(false);
            trainer.training_sample_limit = trainer
                .training_sample_limit
                .min(default_training_vector_count(usize::MAX, nlist)?);
            trainer.centroid_router = Some(router);
        }
        Ok(trainer)
    }
}

impl VectorIndexWriter {
    pub(crate) fn routed_centroid(&self) -> Option<usize> {
        match self {
            Self::IvfFlat(i) => i.routed_centroid,
            Self::IvfSq(i) => i.routed_centroid,
            Self::IvfPq(i) => i.routed_centroid,
            Self::IvfRq(i) => i.routed_centroid,
            Self::DiskAnn(i) => i.routed_centroid,
        }
    }
    pub(super) fn nlist_for_centroid(&self) -> usize {
        match self {
            Self::IvfFlat(i) => i.nlist,
            Self::IvfSq(i) => i.nlist,
            Self::IvfPq(i) => i.nlist,
            Self::IvfRq(i) => i.nlist,
            Self::DiskAnn(_) => 1,
        }
    }
    pub(super) fn model_metric(&self) -> MetricType {
        match self {
            Self::IvfFlat(i) => i.metric,
            Self::IvfSq(i) => i.metric,
            Self::IvfPq(i) => i.metric,
            Self::IvfRq(i) => i.metric,
            Self::DiskAnn(i) => i.metric,
        }
    }
}

impl VectorIndexTraining {
    pub(super) fn coarse_centroids(&self) -> io::Result<&[f32]> {
        Ok(match &self.inner {
            VectorIndexWriter::IvfFlat(i) => i.quantizer_centroids(),
            VectorIndexWriter::IvfSq(i) => i.quantizer_centroids(),
            VectorIndexWriter::IvfPq(i) => i.quantizer_centroids(),
            VectorIndexWriter::IvfRq(i) => i.quantizer_centroids(),
            VectorIndexWriter::DiskAnn(_) => self
                .centroid_router
                .as_ref()
                .ok_or_else(|| invalid_input("DiskANN requires a centroid trainer for routing"))?
                .quantizer_centroids(),
        })
    }

    fn typed_flags(&self) -> u32 {
        match &self.inner {
            VectorIndexWriter::IvfFlat(i) => i.training_flags(),
            VectorIndexWriter::IvfSq(i) => i.training_flags(),
            VectorIndexWriter::IvfPq(i) => i.training_flags(),
            VectorIndexWriter::IvfRq(i) => i.training_flags(),
            VectorIndexWriter::DiskAnn(_) => 0,
        }
    }

    pub(super) fn empty_writer(&self) -> io::Result<VectorIndexWriter> {
        Ok(match &self.inner {
            VectorIndexWriter::IvfFlat(i) => {
                let mut v = IVFFlatIndex::new(i.d, i.nlist, i.metric);
                v.set_quantizer_centroids(i.quantizer_centroids().to_vec());
                v.set_approximate_coarse_assignment(i.training_flags() & 1 != 0);
                VectorIndexWriter::IvfFlat(v)
            }
            VectorIndexWriter::IvfSq(i) => {
                let mut v = IVFSQIndex::new(i.d, i.nlist, i.metric);
                v.set_quantizer_centroids(i.quantizer_centroids().to_vec());
                v.sq = i.sq.clone();
                v.list_sqs = i.list_sqs.clone();
                v.set_approximate_coarse_assignment(i.training_flags() & 1 != 0);
                VectorIndexWriter::IvfSq(v)
            }
            VectorIndexWriter::IvfRq(i) => {
                let mut v = IVFRQIndex::with_options(
                    i.d,
                    i.nlist,
                    i.bits,
                    i.metric,
                    i.rotation_seed,
                    i.rotation_rounds,
                );
                v.set_quantizer_centroids(i.quantizer_centroids().to_vec());
                v.set_approximate_coarse_assignment(i.training_flags() & 1 != 0);
                VectorIndexWriter::IvfRq(v)
            }
            VectorIndexWriter::IvfPq(i) => VectorIndexWriter::IvfPq(IVFPQIndex::from_trained(i)),
            VectorIndexWriter::DiskAnn(i) => {
                self.coarse_centroids()?;
                let mut v = DiskAnnIndex::with_pq_bits(
                    i.d,
                    i.metric,
                    i.pq.m(),
                    i.pq.nbits(),
                    i.build_params,
                );
                v.local_pq_training = i.local_pq_training;
                VectorIndexWriter::DiskAnn(v)
            }
        })
    }

    pub(super) fn serialize_typed(&self) -> io::Result<Vec<u8>> {
        let mut payload = Vec::new();
        let mut auxiliary = Vec::new();
        let mut output = PosWriter::new(&mut payload);
        match &self.inner {
            VectorIndexWriter::IvfFlat(i) => write_ivfflat_index(i, &mut output)?,
            VectorIndexWriter::IvfSq(i) => write_ivfsq_index(i, &mut output)?,
            VectorIndexWriter::IvfRq(i) => write_ivfrq_index(i, &mut output)?,
            VectorIndexWriter::IvfPq(_) => {
                return Err(invalid_model("IVF-PQ uses the v1 envelope"))
            }
            VectorIndexWriter::DiskAnn(i) => {
                let router = self.centroid_router.as_ref().ok_or_else(|| {
                    invalid_input("DiskANN requires a centroid trainer for serialization")
                })?;
                write_ivfflat_index(router, &mut output)?;
                auxiliary = serde_json::to_vec(&diskann_model_options(i))
                    .map_err(|e| invalid_model(&e.to_string()))?;
            }
        }
        let mut out = Vec::with_capacity(TYPED_HEADER + payload.len() + auxiliary.len());
        out.extend_from_slice(TYPED_MAGIC);
        out.extend_from_slice(&(self.index_type() as u32).to_le_bytes());
        out.extend_from_slice(&self.typed_flags().to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&(auxiliary.len() as u64).to_le_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&auxiliary);
        Ok(out)
    }

    pub(super) fn deserialize_typed(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < TYPED_HEADER || &bytes[..8] != TYPED_MAGIC {
            return Err(invalid_model("invalid typed centroid model"));
        }
        let kind = IndexType::from_code(u32::from_le_bytes(bytes[8..12].try_into().unwrap()))
            .ok_or_else(|| invalid_model("unknown centroid model type"))?;
        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let size = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
            .map_err(|_| invalid_model("model size overflow"))?;
        let aux = usize::try_from(u64::from_le_bytes(bytes[24..32].try_into().unwrap()))
            .map_err(|_| invalid_model("model auxiliary size overflow"))?;
        if flags & !1 != 0
            || TYPED_HEADER
                .checked_add(size)
                .and_then(|n| n.checked_add(aux))
                != Some(bytes.len())
            || (kind != IndexType::DiskAnn && aux != 0)
        {
            return Err(invalid_model("invalid typed model flags or lengths"));
        }
        let payload = &bytes[TYPED_HEADER..TYPED_HEADER + size];
        validate_empty_payload(payload, kind)?;
        let reader = VectorIndexReader::open(Cursor::new(payload))?;
        if reader.total_vectors() != 0
            || (kind != IndexType::DiskAnn && reader.index_type() != kind)
            || (kind == IndexType::DiskAnn && reader.index_type() != IndexType::IvfFlat)
        {
            return Err(invalid_model("model type or vector count mismatch"));
        }
        let mut routing = None;
        let mut writer = match reader {
            VectorIndexReader::IvfFlat(mut r) => {
                r.ensure_loaded()?;
                if r.list_counts.iter().any(|&n| n != 0) {
                    return Err(invalid_model("model contains vector lists"));
                }
                let mut i = IVFFlatIndex::new(r.d, r.nlist, r.metric);
                i.set_quantizer_centroids(r.quantizer_centroids);
                i.set_approximate_coarse_assignment(flags & 1 != 0);
                VectorIndexWriter::IvfFlat(i)
            }
            VectorIndexReader::IvfSq(mut r) => {
                r.ensure_loaded()?;
                if r.list_counts.iter().any(|&n| n != 0) {
                    return Err(invalid_model("model contains vector lists"));
                }
                let mut i = IVFSQIndex::new(r.d, r.nlist, r.metric);
                i.set_quantizer_centroids(r.quantizer_centroids);
                i.sq = r.sq;
                i.list_sqs = r.list_sqs;
                i.set_approximate_coarse_assignment(flags & 1 != 0);
                VectorIndexWriter::IvfSq(i)
            }
            VectorIndexReader::IvfRq(mut r) => {
                r.ensure_loaded()?;
                if r.list_counts.iter().any(|&n| n != 0) {
                    return Err(invalid_model("model contains vector lists"));
                }
                let mut i = IVFRQIndex::with_options(
                    r.d,
                    r.nlist,
                    r.num_bits,
                    r.metric,
                    r.rotation_seed,
                    r.rotation_rounds,
                );
                i.set_quantizer_centroids(r.quantizer_centroids);
                i.set_approximate_coarse_assignment(flags & 1 != 0);
                VectorIndexWriter::IvfRq(i)
            }
            _ => return Err(invalid_model("unsupported typed model payload")),
        };
        if kind == IndexType::DiskAnn {
            let options: BTreeMap<String, String> =
                serde_json::from_slice(&bytes[TYPED_HEADER + size..])
                    .map_err(|e| invalid_model(&e.to_string()))?;
            if options.get("dimension") != Some(&writer.dimension().to_string())
                || options.get("metric") != Some(&writer.model_metric().as_str().to_string())
                || options.get("index.type").map(String::as_str) != Some("diskann")
                || flags != 0
            {
                return Err(invalid_model("DiskANN config does not match routing model"));
            }
            let config = VectorIndexConfig::from_options(&options.into_iter().collect())?;
            let max = config.resolved().pq_train_max_points_per_centroid.unwrap();
            if let VectorIndexWriter::IvfFlat(router) = writer {
                routing = Some(router);
            }
            writer = VectorIndexWriter::from_config(config)?;
            if let VectorIndexWriter::DiskAnn(i) = &mut writer {
                i.local_pq_training = Some(max);
                let canonical = serde_json::to_vec(&diskann_model_options(i))
                    .map_err(|e| invalid_model(&e.to_string()))?;
                if canonical != bytes[TYPED_HEADER + size..] {
                    return Err(invalid_model("non-canonical DiskANN build options"));
                }
            }
        }
        let restored = Self {
            inner: writer,
            centroid_router: routing,
        };
        if restored.coarse_centroids()?.iter().any(|v| !v.is_finite()) {
            return Err(invalid_model("non-finite centroid model"));
        }
        Ok(restored)
    }
}

fn validate_empty_payload(payload: &[u8], kind: IndexType) -> io::Result<()> {
    if payload.len() < 64 {
        return Err(invalid_model("truncated model header"));
    }
    let d = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
    let offset = if kind == IndexType::IvfRq { 16 } else { 12 };
    let n = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    let expected = (|| {
        let centers = d.checked_mul(n)?.checked_mul(4)?;
        let sq = if kind == IndexType::IvfSq {
            n.checked_add(1)?.checked_mul(d)?.checked_mul(8)?
        } else {
            0
        };
        64usize
            .checked_add(centers)?
            .checked_add(sq)?
            .checked_add(n.checked_mul(16)?)
    })();
    if d == 0 || n == 0 || expected != Some(payload.len()) {
        return Err(invalid_model(
            "model must contain only complete training metadata",
        ));
    }
    validate_empty_list_table(payload, n)
}

impl<R: SeekRead> VectorIndexReader<R> {
    pub(super) fn search_typed_centroid_batch(
        &mut self,
        queries: &[f32],
        nq: usize,
        params: VectorSearchParams,
        c: usize,
        filter_bytes: Option<&[u8]>,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        if matches!(self, Self::DiskAnn(_)) {
            return if let Some(filter) = filter_bytes {
                self.search_batch_with_roaring_filter_impl(queries, nq, params, filter)
            } else {
                self.search_batch_impl(queries, nq, params)
            };
        }
        params.validate()?;
        validate_queries(queries, nq, self.dimension())?;
        if matches!(params.search_width, SearchWidth::DiskAnnLSearch) {
            return Err(invalid_input(
                "IVF centroid search does not accept DiskANN l_search",
            ));
        }
        let filter = filter_bytes.map(decode_roaring_filter).transpose()?;
        let filter = filter.as_ref().map(|f| f as &dyn crate::ivfpq::RowIdFilter);
        match self {
            Self::IvfFlat(r) => {
                r.ensure_loaded()?;
                check_bound_list(c, &r.list_counts)?;
                crate::ivfflat_io::search_batch_ivfflat_reader_filter_range_fixed(
                    r,
                    queries,
                    nq,
                    params.top_k,
                    0,
                    1,
                    &[],
                    &[],
                    filter,
                    Some(c),
                )
            }
            Self::IvfSq(r) => {
                r.ensure_loaded()?;
                check_bound_list(c, &r.list_counts)?;
                crate::ivfsq_io::search_batch_ivfsq_reader_filter_range_fixed(
                    r,
                    queries,
                    nq,
                    params.top_k,
                    0,
                    1,
                    &[],
                    &[],
                    filter,
                    Some(c),
                )
            }
            Self::IvfRq(r) => {
                r.ensure_loaded()?;
                check_bound_list(c, &r.list_counts)?;
                crate::ivfrq_io::search_batch_ivfrq_reader_filter_range_fixed(
                    r,
                    queries,
                    nq,
                    params.top_k,
                    0,
                    1,
                    &[],
                    &[],
                    filter,
                    Some(c),
                )
            }
            _ => Err(invalid_input("unexpected centroid reader type")),
        }
    }
}

fn check_bound_list(c: usize, counts: &[i32]) -> io::Result<()> {
    if c >= counts.len() || counts.iter().enumerate().any(|(i, &n)| i != c && n != 0) {
        return Err(invalid_input(
            "index contains vectors outside the requested centroid",
        ));
    }
    Ok(())
}

fn diskann_model_options(i: &DiskAnnIndex) -> BTreeMap<String, String> {
    let p = i.build_params;
    [
        ("index.type", "diskann".to_string()),
        ("dimension", i.d.to_string()),
        ("metric", i.metric.as_str().to_string()),
        ("pq.m", i.pq.m().to_string()),
        ("pq.bits", i.pq.nbits().to_string()),
        (
            "pq.train.max-points-per-centroid",
            i.local_pq_training.unwrap_or(256).to_string(),
        ),
        ("diskann.max-degree", p.max_degree.to_string()),
        (
            "diskann.build-search-list-size",
            p.build_search_list_size.to_string(),
        ),
        ("diskann.alpha", p.alpha.to_string()),
        ("diskann.seed", p.seed.to_string()),
        (
            "diskann.memory-budget-bytes",
            p.memory_budget_bytes.to_string(),
        ),
        (
            "diskann.storage-layout",
            match p.storage_layout {
                crate::diskann::DiskAnnStorageLayout::Compact => "compact",
                crate::diskann::DiskAnnStorageLayout::Interleaved => "interleaved",
            }
            .to_string(),
        ),
        (
            "diskann.raw-vector-encoding",
            match p.raw_vector_encoding {
                crate::diskann::DiskAnnRawVectorEncoding::F16 => "f16",
                crate::diskann::DiskAnnRawVectorEncoding::F32 => "f32",
            }
            .to_string(),
        ),
        (
            "diskann.build-distance",
            match p.build_distance {
                DiskAnnBuildDistance::FullPrecision => "full_precision",
                DiskAnnBuildDistance::ProductQuantized => "product_quantized",
            }
            .to_string(),
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}
