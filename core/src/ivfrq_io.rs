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

use crate::distance::{fvec_norm_l2sqr, preprocess_vectors, MetricType};
use crate::index_io_util::{
    decode_delta_varint_ids, encode_delta_varint_ids, pread_batched_payloads,
    validate_search_inputs,
};
use crate::io::{PreadCursor, ReadRequest, SeekRead, SeekWrite};
use crate::ivfpq::RowIdFilter;
use crate::ivfrq::IVFRQIndex;
use crate::kmeans;
use crate::rq::{
    is_supported_rq_bits, padded_dimension, RQCodeFactors, RQQueryContext, RQQueryTerms,
    RQRotation, RQVectorFactors, RaBitQuantizer, DEFAULT_RQ_ROTATION_ROUNDS, RQ_SCAN_BLOCK_SIZE,
};
use crate::topk::TopKHeap;
use rayon::prelude::*;
use roaring::RoaringTreemap;
use std::io;

pub const IVF_RQ_MAGIC: u32 = 0x49565251; // "IVRQ"
pub const IVF_RQ_VERSION: u32 = 1;
pub const IVF_RQ_HEADER_SIZE: usize = 64;

pub const IVF_RQ_ROTATION_TYPE_BLOCK_FHT: u32 = 2;
pub const IVF_RQ_FACTOR_LAYOUT_COMPACT_V1: u32 = 3;

const FLAG_DELTA_IDS: u32 = 1 << 0;
const FLAG_BLOCK_TRANSPOSED_CODES: u32 = 1 << 1;
const FLAG_BLOCK_SOA_FACTORS: u32 = 1 << 2;
const REQUIRED_FLAGS: u32 = FLAG_DELTA_IDS | FLAG_BLOCK_TRANSPOSED_CODES | FLAG_BLOCK_SOA_FACTORS;
const SUPPORTED_FLAGS: u32 = REQUIRED_FLAGS;
const FACTOR_BYTES: usize = 4;
const MAX_RQ_BATCH_READ_BYTES: usize = 64 * 1024 * 1024;
const PARALLEL_RQ_SCAN_MIN_CANDIDATES: usize = 8 * 1024;
const PARALLEL_RQ_SEED_LISTS: usize = 1;
const PARALLEL_RQ_SEED_VECTORS: usize = RQ_SCAN_BLOCK_SIZE;
const FASTSCAN_MIN_PADDED_DIMENSION: usize = 256;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IVFRQSearchStats {
    pub query_count: usize,
    pub scanned_vectors: usize,
    pub eligible_vectors: usize,
    pub coarse_distance_evaluations: usize,
    pub refined_vectors: usize,
    pub refined_coarse_byte_lookups: usize,
    pub extra_plane_byte_lookups: usize,
    pub final_distance_evaluations: usize,
    pub heap_admissions: usize,
    pub fastscan_blocks: usize,
    pub scalar_blocks: usize,
    pub seeded_lists: usize,
    pub parallel_list_tasks: usize,
}

impl IVFRQSearchStats {
    pub fn merge(&mut self, other: Self) {
        self.query_count = self.query_count.saturating_add(other.query_count);
        self.scanned_vectors = self.scanned_vectors.saturating_add(other.scanned_vectors);
        self.eligible_vectors = self.eligible_vectors.saturating_add(other.eligible_vectors);
        self.coarse_distance_evaluations = self
            .coarse_distance_evaluations
            .saturating_add(other.coarse_distance_evaluations);
        self.refined_vectors = self.refined_vectors.saturating_add(other.refined_vectors);
        self.refined_coarse_byte_lookups = self
            .refined_coarse_byte_lookups
            .saturating_add(other.refined_coarse_byte_lookups);
        self.extra_plane_byte_lookups = self
            .extra_plane_byte_lookups
            .saturating_add(other.extra_plane_byte_lookups);
        self.final_distance_evaluations = self
            .final_distance_evaluations
            .saturating_add(other.final_distance_evaluations);
        self.heap_admissions = self.heap_admissions.saturating_add(other.heap_admissions);
        self.fastscan_blocks = self.fastscan_blocks.saturating_add(other.fastscan_blocks);
        self.scalar_blocks = self.scalar_blocks.saturating_add(other.scalar_blocks);
        self.seeded_lists = self.seeded_lists.saturating_add(other.seeded_lists);
        self.parallel_list_tasks = self
            .parallel_list_tasks
            .saturating_add(other.parallel_list_tasks);
    }
}

struct RQListWritePlan {
    order: Vec<usize>,
    base_id: i64,
    id_bytes: Vec<u8>,
}

pub fn write_ivfrq_index(index: &IVFRQIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    validate_index_shape(index)?;
    let total_vectors = index.ids.iter().try_fold(0i64, |sum, ids| {
        sum.checked_add(usize_to_i64(ids.len(), "total vector count")?)
            .ok_or_else(|| invalid_input("total vector count exceeds i64"))
    })?;
    let write_plans = plan_sorted_lists(index);

    write_u32_le(out, IVF_RQ_MAGIC)?;
    write_u32_le(out, IVF_RQ_VERSION)?;
    write_i32_le(out, usize_to_i32(index.d, "dimension")?)?;
    write_i32_le(out, usize_to_i32(index.padded_d, "padded dimension")?)?;
    write_i32_le(out, usize_to_i32(index.nlist, "nlist")?)?;
    write_u32_le(out, index.metric as u32)?;
    write_u32_le(out, REQUIRED_FLAGS)?;
    write_u32_le(out, usize_to_u32(index.bits, "RQ bits")?)?;
    write_i64_le(out, total_vectors)?;
    write_u64_le(out, index.rotation_seed)?;
    write_u32_le(out, index.rotation_rounds)?;
    write_i32_le(out, usize_to_i32(index.plane_size(), "plane size")?)?;
    write_u32_le(out, IVF_RQ_ROTATION_TYPE_BLOCK_FHT)?;
    write_u32_le(out, IVF_RQ_FACTOR_LAYOUT_COMPACT_V1)?;

    write_f32_slice(out, &index.quantizer_centroids)?;

    let offset_table_bytes = index
        .nlist
        .checked_mul(16)
        .ok_or_else(|| invalid_input("IVF-RQ offset table size overflow"))?;
    let mut current_offset = out
        .pos()
        .checked_add(offset_table_bytes as u64)
        .ok_or_else(|| invalid_input("IVF-RQ data offset overflow"))?;
    let mut offsets = Vec::with_capacity(index.nlist);

    for plan in &write_plans {
        offsets.push((
            u64_to_i64(current_offset, "list offset")?,
            usize_to_i32(plan.order.len(), "list count")?,
            usize_to_i32(plan.id_bytes.len(), "delta ID bytes")?,
        ));
        if !plan.order.is_empty() {
            let code_bytes = checked_list_bytes(plan.order.len(), index.code_size())?;
            let factor_values =
                checked_list_bytes(plan.order.len(), index_factors_fields(index.bits))?;
            let factor_bytes = checked_list_bytes(factor_values, FACTOR_BYTES)?;
            let list_bytes = 16usize
                .checked_add(plan.id_bytes.len())
                .and_then(|size| size.checked_add(code_bytes))
                .and_then(|size| size.checked_add(factor_bytes))
                .ok_or_else(|| invalid_input("IVF-RQ list size overflow"))?;
            current_offset = current_offset
                .checked_add(list_bytes as u64)
                .ok_or_else(|| invalid_input("IVF-RQ list offset overflow"))?;
        }
    }

    for (offset, count, id_bytes) in offsets {
        write_i64_le(out, offset)?;
        write_i32_le(out, count)?;
        write_i32_le(out, id_bytes)?;
    }

    for (list_id, plan) in write_plans.into_iter().enumerate() {
        if plan.order.is_empty() {
            continue;
        }
        let (blocked_codes, blocked_factors) = block_list(index, list_id, &plan.order);
        write_i64_le(out, plan.base_id)?;
        write_i32_le(out, usize_to_i32(plan.id_bytes.len(), "delta ID bytes")?)?;
        write_i32_le(
            out,
            usize_to_i32(blocked_codes.len(), "blocked code bytes")?,
        )?;
        out.write_all(&plan.id_bytes)?;
        out.write_all(&blocked_codes)?;
        write_f32_slice(out, &blocked_factors)?;
    }
    Ok(())
}

fn plan_sorted_lists(index: &IVFRQIndex) -> Vec<RQListWritePlan> {
    (0..index.nlist)
        .map(|list_id| {
            let count = index.ids[list_id].len();
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by_key(|&position| index.ids[list_id][position]);
            let sorted_ids = order
                .iter()
                .map(|&position| index.ids[list_id][position])
                .collect::<Vec<_>>();
            let (base_id, id_bytes) = encode_delta_varint_ids(&sorted_ids);
            RQListWritePlan {
                order,
                base_id,
                id_bytes,
            }
        })
        .collect()
}

fn block_list(index: &IVFRQIndex, list_id: usize, order: &[usize]) -> (Vec<u8>, Vec<f32>) {
    let code_size = index.code_size();
    let plane_size = index.plane_size();
    let factor_fields = index_factors_fields(index.bits);
    let mut blocked_codes = Vec::with_capacity(order.len() * code_size);
    let mut blocked_factors = Vec::with_capacity(order.len() * factor_fields);

    for block in order.chunks(RQ_SCAN_BLOCK_SIZE) {
        for plane in 0..index.bits {
            for byte_idx in 0..plane_size {
                for &position in block {
                    blocked_codes.push(
                        index.codes[list_id][position * code_size + plane * plane_size + byte_idx],
                    );
                }
            }
        }
        for field in 0..factor_fields {
            for &position in block {
                blocked_factors.push(factor_value(
                    index.factors[list_id][position],
                    field,
                    index.bits,
                ));
            }
        }
    }
    (blocked_codes, blocked_factors)
}

pub struct IVFRQIndexReader<R: SeekRead> {
    reader: R,
    pub d: usize,
    pub padded_d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub rotation_seed: u64,
    pub rotation_rounds: u32,
    pub plane_size: usize,
    pub num_bits: usize,
    pub rotation_type: u32,
    pub factor_layout: u32,
    pub quantizer_centroids: Vec<f32>,
    pub quantizer_centroid_norms: Vec<f32>,
    pub list_offsets: Vec<i64>,
    pub list_counts: Vec<i32>,
    pub list_id_bytes_lens: Vec<i32>,
    quantizer: RaBitQuantizer,
    rotation: RQRotation,
    last_search_stats: IVFRQSearchStats,
    loaded: bool,
}

#[derive(Clone, Copy)]
struct RQListPayloadMeta {
    list_id: usize,
    count: usize,
    offset: u64,
    id_bytes_len: usize,
    code_bytes_len: usize,
    factor_values_len: usize,
    payload_len: usize,
}

impl<R: SeekRead> IVFRQIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        let mut header = [0u8; IVF_RQ_HEADER_SIZE];
        reader.pread(&mut [ReadRequest::new(0, &mut header)])?;
        Self::open_with_header(reader, header)
    }

    pub(crate) fn open_with_header(
        reader: R,
        header: [u8; IVF_RQ_HEADER_SIZE],
    ) -> io::Result<Self> {
        let mut header_reader = std::io::Cursor::new(header);
        let mut cursor = PreadCursor::new(&mut header_reader, 0);
        let magic = read_u32_le(&mut cursor)?;
        if magic != IVF_RQ_MAGIC {
            return Err(invalid_data(format!("invalid IVF-RQ magic: 0x{magic:08X}")));
        }
        let version = read_u32_le(&mut cursor)?;
        if version != IVF_RQ_VERSION {
            return Err(invalid_data(format!(
                "unsupported IVF-RQ version: {version}"
            )));
        }
        let d = validate_positive_i32(read_i32_le(&mut cursor)?, "dimension")? as usize;
        let padded_d =
            validate_positive_i32(read_i32_le(&mut cursor)?, "padded_dimension")? as usize;
        if padded_d != padded_dimension(d) {
            return Err(invalid_data(format!(
                "IVF-RQ padded_dimension {padded_d} does not match dimension-derived {}",
                padded_dimension(d)
            )));
        }
        let nlist = validate_positive_i32(read_i32_le(&mut cursor)?, "nlist")? as usize;
        let metric_code = read_u32_le(&mut cursor)?;
        let metric = MetricType::from_code(metric_code)
            .ok_or_else(|| invalid_data(format!("unknown metric type: {metric_code}")))?;
        let flags = read_u32_le(&mut cursor)?;
        let num_bits = read_u32_le(&mut cursor)? as usize;
        let total_vectors = read_i64_le(&mut cursor)?;
        let rotation_seed = read_u64_le(&mut cursor)?;
        let rotation_rounds = read_u32_le(&mut cursor)?;
        let plane_size = validate_positive_i32(read_i32_le(&mut cursor)?, "plane_size")? as usize;
        let rotation_type = read_u32_le(&mut cursor)?;
        let factor_layout = read_u32_le(&mut cursor)?;

        if total_vectors < 0 {
            return Err(invalid_data("negative IVF-RQ vector count"));
        }
        if plane_size != padded_d / 8 {
            return Err(invalid_data(format!(
                "IVF-RQ plane size {plane_size} does not match padded dimension {padded_d}"
            )));
        }
        if !is_supported_rq_bits(num_bits) {
            return Err(invalid_data(format!(
                "unsupported IVF-RQ bits {num_bits}; expected 1..=8"
            )));
        }
        if rotation_rounds != DEFAULT_RQ_ROTATION_ROUNDS {
            return Err(invalid_data(format!(
                "unsupported IVF-RQ rotation rounds {rotation_rounds}; expected {DEFAULT_RQ_ROTATION_ROUNDS}"
            )));
        }
        if rotation_type != IVF_RQ_ROTATION_TYPE_BLOCK_FHT {
            return Err(invalid_data(format!(
                "unsupported IVF-RQ rotation type {rotation_type}"
            )));
        }
        if factor_layout != IVF_RQ_FACTOR_LAYOUT_COMPACT_V1 {
            return Err(invalid_data(format!(
                "unsupported IVF-RQ factor layout {factor_layout}"
            )));
        }
        if flags & !SUPPORTED_FLAGS != 0 || flags & REQUIRED_FLAGS != REQUIRED_FLAGS {
            return Err(invalid_data(format!(
                "unsupported IVF-RQ flags 0x{flags:08X}"
            )));
        }

        Ok(Self {
            reader,
            d,
            padded_d,
            nlist,
            metric,
            total_vectors,
            rotation_seed,
            rotation_rounds,
            plane_size,
            num_bits,
            rotation_type,
            factor_layout,
            quantizer_centroids: Vec::new(),
            quantizer_centroid_norms: Vec::new(),
            list_offsets: Vec::new(),
            list_counts: Vec::new(),
            list_id_bytes_lens: Vec::new(),
            quantizer: RaBitQuantizer::new(d, num_bits),
            rotation: RQRotation::new(d, rotation_seed, rotation_rounds),
            last_search_stats: IVFRQSearchStats::default(),
            loaded: false,
        })
    }

    pub fn last_search_stats(&self) -> IVFRQSearchStats {
        self.last_search_stats
    }

    pub(crate) fn set_last_search_stats(&mut self, stats: IVFRQSearchStats) {
        self.last_search_stats = stats;
    }

    pub fn ensure_loaded(&mut self) -> io::Result<()> {
        if self.loaded {
            return Ok(());
        }
        let centroid_values = checked_section_size(self.nlist, self.d)?;
        let centroid_bytes = checked_list_bytes(centroid_values, FACTOR_BYTES)?;
        let offset_table_bytes = checked_list_bytes(self.nlist, 16)?;
        let metadata_bytes = centroid_bytes
            .checked_add(offset_table_bytes)
            .ok_or_else(|| invalid_data("IVF-RQ resident metadata size overflow"))?;
        let mut metadata = vec![0u8; metadata_bytes];
        self.reader
            .pread(&mut [ReadRequest::new(IVF_RQ_HEADER_SIZE as u64, &mut metadata)])?;

        self.quantizer_centroids = bytes_to_f32_vec(&metadata[..centroid_bytes])?;
        self.list_offsets = vec![0; self.nlist];
        self.list_counts = vec![0; self.nlist];
        self.list_id_bytes_lens = vec![0; self.nlist];
        for (list_id, entry) in metadata[centroid_bytes..].chunks_exact(16).enumerate() {
            self.list_offsets[list_id] = i64::from_le_bytes(entry[0..8].try_into().unwrap());
            self.list_counts[list_id] = validate_non_negative_i32(
                i32::from_le_bytes(entry[8..12].try_into().unwrap()),
                "list count",
            )?;
            self.list_id_bytes_lens[list_id] = validate_non_negative_i32(
                i32::from_le_bytes(entry[12..16].try_into().unwrap()),
                "delta ID bytes",
            )?;
        }
        let actual_total: i64 = self.list_counts.iter().map(|&count| count as i64).sum();
        if actual_total != self.total_vectors {
            return Err(invalid_data(format!(
                "IVF-RQ list counts total {actual_total} does not match header {}",
                self.total_vectors
            )));
        }

        self.quantizer_centroid_norms = self
            .quantizer_centroids
            .chunks_exact(self.d)
            .map(fvec_norm_l2sqr)
            .collect();
        self.loaded = true;
        Ok(())
    }

    pub fn read_inverted_list(&mut self, list_id: usize) -> io::Result<RQReadList> {
        self.ensure_loaded()?;
        let Some(meta) = self.list_payload_meta(list_id)? else {
            return Ok(RQReadList::empty(list_id));
        };
        let mut payload = vec![0u8; meta.payload_len];
        self.reader
            .pread(&mut [ReadRequest::new(meta.offset, &mut payload)])?;
        self.decode_inverted_list_payload(meta, payload)
    }

    fn read_inverted_lists(&mut self, list_ids: &[usize]) -> io::Result<Vec<RQReadList>> {
        self.ensure_loaded()?;
        let mut results: Vec<Option<RQReadList>> = (0..list_ids.len()).map(|_| None).collect();
        let mut metas = Vec::new();
        let mut payloads = Vec::new();
        for (input_index, &list_id) in list_ids.iter().enumerate() {
            if let Some(meta) = self.list_payload_meta(list_id)? {
                metas.push((input_index, meta));
                payloads.push(vec![0; meta.payload_len]);
            } else {
                results[input_index] = Some(RQReadList::empty(list_id));
            }
        }
        if !metas.is_empty() {
            let offsets = metas
                .iter()
                .map(|(_, meta)| meta.offset)
                .collect::<Vec<_>>();
            pread_batched_payloads(&mut self.reader, &offsets, &mut payloads)?;
            for ((input_index, meta), payload) in metas.into_iter().zip(payloads) {
                results[input_index] = Some(self.decode_inverted_list_payload(meta, payload)?);
            }
        }
        results
            .into_iter()
            .map(|result| result.ok_or_else(|| invalid_data("missing IVF-RQ list result")))
            .collect()
    }

    fn batch_read_end(&self, list_ids: &[usize]) -> io::Result<usize> {
        let mut payload_bytes = 0usize;
        let mut request_count = 0usize;
        let max_ranges = match self.reader.read_capabilities().max_ranges_per_pread {
            0 => usize::MAX,
            value => value,
        };
        for (index, &list_id) in list_ids.iter().enumerate() {
            let Some(meta) = self.list_payload_meta(list_id)? else {
                continue;
            };
            if request_count >= max_ranges {
                return Ok(index);
            }
            let next = payload_bytes
                .checked_add(meta.payload_len)
                .ok_or_else(|| invalid_data("IVF-RQ batch payload overflow"))?;
            if index > 0 && next > MAX_RQ_BATCH_READ_BYTES {
                return Ok(index);
            }
            payload_bytes = next;
            request_count += 1;
        }
        Ok(list_ids.len())
    }

    fn list_payload_meta(&self, list_id: usize) -> io::Result<Option<RQListPayloadMeta>> {
        if list_id >= self.nlist {
            return Err(invalid_input(format!(
                "list ID {list_id} out of range for nlist={}",
                self.nlist
            )));
        }
        let count = self.list_counts[list_id] as usize;
        if count == 0 {
            return Ok(None);
        }
        let id_bytes_len = self.list_id_bytes_lens[list_id] as usize;
        let code_bytes_len = checked_list_bytes(count, self.quantizer.code_size())?;
        let factor_values_len = checked_list_bytes(count, self.quantizer.factor_fields())?;
        let payload_len = 16usize
            .checked_add(id_bytes_len)
            .and_then(|size| size.checked_add(code_bytes_len))
            .and_then(|size| size.checked_add(factor_values_len * FACTOR_BYTES))
            .ok_or_else(|| invalid_data("IVF-RQ list payload overflow"))?;
        Ok(Some(RQListPayloadMeta {
            list_id,
            count,
            offset: checked_list_offset(self.list_offsets[list_id], list_id)?,
            id_bytes_len,
            code_bytes_len,
            factor_values_len,
            payload_len,
        }))
    }

    fn decode_inverted_list_payload(
        &self,
        meta: RQListPayloadMeta,
        mut payload: Vec<u8>,
    ) -> io::Result<RQReadList> {
        if payload.len() != meta.payload_len {
            return Err(invalid_data(format!(
                "IVF-RQ list {} payload length mismatch",
                meta.list_id
            )));
        }
        let base_id = i64::from_le_bytes(payload[0..8].try_into().unwrap());
        let id_bytes_len = i32::from_le_bytes(payload[8..12].try_into().unwrap());
        let code_bytes_len = i32::from_le_bytes(payload[12..16].try_into().unwrap());
        if id_bytes_len < 0 || id_bytes_len as usize != meta.id_bytes_len {
            return Err(invalid_data("IVF-RQ delta ID length mismatch"));
        }
        if code_bytes_len < 0 || code_bytes_len as usize != meta.code_bytes_len {
            return Err(invalid_data("IVF-RQ code length mismatch"));
        }
        let ids =
            decode_delta_varint_ids(base_id, &payload[16..16 + meta.id_bytes_len], meta.count)?;
        let code_start = 16 + meta.id_bytes_len;
        let factor_start = code_start + meta.code_bytes_len;
        let factors = bytes_to_f32_vec(&payload[factor_start..])?;
        if factors.len() != meta.factor_values_len {
            return Err(invalid_data("IVF-RQ factor count mismatch"));
        }
        payload.truncate(factor_start);
        Ok(RQReadList {
            list_id: meta.list_id,
            ids,
            blocked_code_start: code_start,
            blocked_code_end: factor_start,
            payload,
            blocked_factors: factors,
        })
    }

    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.search_with_filter(query, k, nprobe, None)
    }

    pub fn search_with_filter(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        filter: Option<&dyn RowIdFilter>,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        search_batch_ivfrq_reader_filter(self, query, 1, k, nprobe, filter)
    }

    pub fn search_with_roaring_filter(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.search_with_filter(query, k, nprobe, Some(&filter))
    }
}

pub struct RQReadList {
    pub list_id: usize,
    pub ids: Vec<i64>,
    payload: Vec<u8>,
    blocked_code_start: usize,
    blocked_code_end: usize,
    pub blocked_factors: Vec<f32>,
}

impl RQReadList {
    fn blocked_codes(&self) -> &[u8] {
        &self.payload[self.blocked_code_start..self.blocked_code_end]
    }

    fn empty(list_id: usize) -> Self {
        Self {
            list_id,
            ids: Vec::new(),
            payload: Vec::new(),
            blocked_code_start: 0,
            blocked_code_end: 0,
            blocked_factors: Vec::new(),
        }
    }
}

pub fn search_batch_ivfrq_reader<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfrq_reader_filter(reader, queries, nq, k, nprobe, None)
}

pub fn search_batch_ivfrq_reader_filter<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    reader.ensure_loaded()?;
    validate_search_inputs(queries, nq, reader.d, k, nprobe)?;
    let processed = preprocess_vectors(queries, nq, reader.d, reader.metric);
    let (all_probe_indices, all_probe_distances) = kmeans::find_topk_batch_with_centroid_norms(
        &processed,
        nq,
        &reader.quantizer_centroids,
        &reader.quantizer_centroid_norms,
        reader.nlist,
        reader.d,
        nprobe,
    );
    let query_norms = processed
        .chunks_exact(reader.d)
        .map(fvec_norm_l2sqr)
        .collect::<Vec<_>>();
    let all_query_terms = all_probe_indices
        .iter()
        .zip(&all_probe_distances)
        .enumerate()
        .map(|(query_index, (indices, distances))| {
            indices
                .iter()
                .zip(distances)
                .map(|(&list_id, &distance)| {
                    reader.quantizer.query_terms_from_coarse_distance(
                        distance,
                        query_norms[query_index],
                        reader.quantizer_centroid_norms[list_id],
                        reader.metric,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut query_contexts = Vec::with_capacity(nq);
    let mut rotated = vec![0.0; reader.padded_d];
    let mut scratch = vec![0.0; reader.padded_d];
    for query in processed.chunks_exact(reader.d) {
        reader.rotation.rotate(query, &mut rotated, &mut scratch);
        query_contexts.push(reader.quantizer.prepare_query(rotated.clone()));
    }

    let mut seen_lists = vec![false; reader.nlist];
    let mut unique_lists = Vec::new();
    for probes in &all_probe_indices {
        for &list_id in probes {
            if !seen_lists[list_id] {
                seen_lists[list_id] = true;
                unique_lists.push(list_id);
            }
        }
    }

    let mut heaps: Vec<TopKHeap> = (0..nq).map(|_| TopKHeap::new(k)).collect();
    let mut query_stats = vec![IVFRQSearchStats::default(); nq];
    let mut aggregate_stats = IVFRQSearchStats {
        query_count: nq,
        ..IVFRQSearchStats::default()
    };
    let mut batch_start = 0;
    while batch_start < unique_lists.len() {
        let count = reader.batch_read_end(&unique_lists[batch_start..])?;
        let batch_end = batch_start + count;
        let loaded_lists = reader.read_inverted_lists(&unique_lists[batch_start..batch_end])?;
        let mut list_positions = vec![usize::MAX; reader.nlist];
        for (position, list) in loaded_lists.iter().enumerate() {
            list_positions[list.list_id] = position;
        }
        let quantizer = &reader.quantizer;
        let candidate_count = loaded_lists
            .iter()
            .map(|list| list.ids.len())
            .sum::<usize>();
        if nq == 1 && candidate_count >= PARALLEL_RQ_SCAN_MIN_CANDIDATES {
            let mut seeded_lists = 0usize;
            if PARALLEL_RQ_SEED_LISTS > 0 {
                for list in loaded_lists
                    .iter()
                    .filter(|list| !list.ids.is_empty())
                    .take(PARALLEL_RQ_SEED_LISTS)
                {
                    let probe_position = all_probe_indices[0]
                        .iter()
                        .position(|&probe| probe == list.list_id)
                        .expect("loaded list must belong to the query probe set");
                    scan_blocked_list(
                        list,
                        quantizer,
                        &query_contexts[0],
                        all_query_terms[0][probe_position],
                        filter,
                        &mut heaps[0],
                        &mut aggregate_stats,
                        PARALLEL_RQ_SEED_VECTORS,
                    );
                    seeded_lists += 1;
                }
            }
            aggregate_stats.seeded_lists =
                aggregate_stats.seeded_lists.saturating_add(seeded_lists);
            let seeded_threshold = heaps[0].worst_distance().unwrap_or(f32::INFINITY);
            let per_list_results = loaded_lists
                .par_iter()
                .map(|list| {
                    let mut heap = TopKHeap::with_max_distance(k, seeded_threshold);
                    let mut stats = IVFRQSearchStats::default();
                    let list_id = list.list_id;
                    let probe_position = all_probe_indices[0]
                        .iter()
                        .position(|&probe| probe == list_id)
                        .expect("loaded list must belong to the query probe set");
                    scan_blocked_list(
                        list,
                        quantizer,
                        &query_contexts[0],
                        all_query_terms[0][probe_position],
                        filter,
                        &mut heap,
                        &mut stats,
                        usize::MAX,
                    );
                    (heap.into_sorted(), stats)
                })
                .collect::<Vec<_>>();
            aggregate_stats.parallel_list_tasks = aggregate_stats
                .parallel_list_tasks
                .saturating_add(per_list_results.len());
            for (results, stats) in per_list_results {
                aggregate_stats.merge(stats);
                for (distance, row_id) in results {
                    heaps[0].push(distance, row_id);
                }
            }
        } else {
            heaps
                .par_iter_mut()
                .zip(query_stats.par_iter_mut())
                .enumerate()
                .for_each(|(query_index, (heap, stats))| {
                    for (probe_position, &list_id) in
                        all_probe_indices[query_index].iter().enumerate()
                    {
                        let position = list_positions[list_id];
                        if position == usize::MAX {
                            continue;
                        }
                        scan_blocked_list(
                            &loaded_lists[position],
                            quantizer,
                            &query_contexts[query_index],
                            all_query_terms[query_index][probe_position],
                            filter,
                            heap,
                            stats,
                            usize::MAX,
                        );
                    }
                });
        }
        batch_start = batch_end;
    }
    for stats in query_stats {
        aggregate_stats.merge(stats);
    }
    reader.last_search_stats = aggregate_stats;

    let mut result_ids = vec![-1; nq * k];
    let mut result_distances = vec![f32::MAX; nq * k];
    for query_index in 0..nq {
        let sorted = std::mem::replace(&mut heaps[query_index], TopKHeap::new(0)).into_sorted();
        for (rank, &(distance, id)) in sorted.iter().enumerate() {
            result_ids[query_index * k + rank] = id;
            result_distances[query_index * k + rank] = distance;
        }
    }
    Ok((result_ids, result_distances))
}

pub fn search_batch_ivfrq_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_batch_ivfrq_reader_filter(reader, queries, nq, k, nprobe, Some(&filter))
}

fn scan_blocked_list(
    list: &RQReadList,
    quantizer: &RaBitQuantizer,
    query: &RQQueryContext,
    query_terms: RQQueryTerms,
    filter: Option<&dyn RowIdFilter>,
    heap: &mut TopKHeap,
    stats: &mut IVFRQSearchStats,
    vector_limit: usize,
) {
    let blocked_codes = list.blocked_codes();
    let plane_size = quantizer.plane_size();
    let bits = quantizer.bits();
    let code_size = quantizer.code_size();
    let factor_fields = quantizer.factor_fields();
    let query_sum = quantizer.query_sum(query);
    let center = ((1usize << bits) - 1) as f32 * 0.5;

    let scan_end = list.ids.len().min(vector_limit);
    let mut block_start = 0usize;
    while block_start < scan_end {
        let lanes = (scan_end - block_start).min(RQ_SCAN_BLOCK_SIZE);
        stats.scanned_vectors = stats.scanned_vectors.saturating_add(lanes);
        let code_block_start = block_start * code_size;
        let factor_block_start = block_start * factor_fields;
        let mut allowed = [true; RQ_SCAN_BLOCK_SIZE];
        let mut coarse_unsigned = [0.0f32; RQ_SCAN_BLOCK_SIZE];
        let used_fastscan = bits > 1
            && quantizer.padded_dimension() >= FASTSCAN_MIN_PADDED_DIMENSION
            && filter.is_none()
            && lanes == RQ_SCAN_BLOCK_SIZE;
        if used_fastscan {
            quantizer.fastscan_coarse_block(
                query,
                &blocked_codes
                    [code_block_start..code_block_start + plane_size * RQ_SCAN_BLOCK_SIZE],
                &mut coarse_unsigned,
            );
            stats.fastscan_blocks = stats.fastscan_blocks.saturating_add(1);
        } else if let Some(filter) = filter {
            stats.scalar_blocks = stats.scalar_blocks.saturating_add(1);
            for lane in 0..lanes {
                allowed[lane] = filter.contains(list.ids[block_start + lane]);
            }
            for byte_idx in 0..plane_size {
                let byte_start = code_block_start + byte_idx * lanes;
                for lane in 0..lanes {
                    if allowed[lane] {
                        coarse_unsigned[lane] += quantizer.byte_subset_sum(
                            query,
                            byte_idx,
                            blocked_codes[byte_start + lane],
                        );
                    }
                }
            }
        } else {
            stats.scalar_blocks = stats.scalar_blocks.saturating_add(1);
            for byte_idx in 0..plane_size {
                let byte_start = code_block_start + byte_idx * lanes;
                for lane in 0..lanes {
                    coarse_unsigned[lane] += quantizer.byte_subset_sum(
                        query,
                        byte_idx,
                        blocked_codes[byte_start + lane],
                    );
                }
            }
        }
        let eligible = allowed[..lanes].iter().filter(|&&value| value).count();
        stats.eligible_vectors = stats.eligible_vectors.saturating_add(eligible);
        stats.coarse_distance_evaluations =
            stats.coarse_distance_evaluations.saturating_add(eligible);

        let mut refine = [false; RQ_SCAN_BLOCK_SIZE];
        if bits == 1 {
            for lane in 0..lanes {
                if !allowed[lane] {
                    continue;
                }
                let factors = read_block_factor(list, factor_block_start, lanes, 0, lane, false);
                let distance = quantizer.estimate(
                    coarse_unsigned[lane] - 0.5 * query_sum,
                    factors,
                    query_terms,
                );
                if heap.should_consider(distance) {
                    stats.heap_admissions = stats.heap_admissions.saturating_add(1);
                    heap.push(distance, list.ids[block_start + lane]);
                }
            }
            block_start += lanes;
            continue;
        }

        for lane in 0..lanes {
            if !allowed[lane] {
                continue;
            }
            let factors = read_block_factor(list, factor_block_start, lanes, 0, lane, true);
            let estimate = quantizer.estimate(
                coarse_unsigned[lane] - 0.5 * query_sum,
                factors,
                query_terms,
            );
            let mut lower = quantizer.lower_bound(estimate, factors, query_terms);
            if used_fastscan {
                lower -= factors.f_rescale.abs() * quantizer.fastscan_ip_error(query);
            }
            refine[lane] = heap.should_consider(lower);
        }
        let refined = refine[..lanes].iter().filter(|&&value| value).count();
        stats.refined_vectors = stats.refined_vectors.saturating_add(refined);
        stats.extra_plane_byte_lookups = stats
            .extra_plane_byte_lookups
            .saturating_add(refined.saturating_mul(bits - 1).saturating_mul(plane_size));

        let mut full_unsigned = [0.0f32; RQ_SCAN_BLOCK_SIZE];
        let sign_weight = (1usize << (bits - 1)) as f32;
        for lane in 0..lanes {
            full_unsigned[lane] = coarse_unsigned[lane] * sign_weight;
        }
        for plane in 1..bits {
            let weight = (1usize << (bits - 1 - plane)) as f32;
            let plane_start = code_block_start + plane * plane_size * lanes;
            for byte_idx in 0..plane_size {
                let byte_start = plane_start + byte_idx * lanes;
                for lane in 0..lanes {
                    if refine[lane] {
                        full_unsigned[lane] += weight
                            * quantizer.byte_subset_sum(
                                query,
                                byte_idx,
                                blocked_codes[byte_start + lane],
                            );
                    }
                }
            }
        }
        for lane in 0..lanes {
            if !refine[lane] {
                continue;
            }
            let factors = read_block_factor(list, factor_block_start, lanes, 3, lane, false);
            let mut distance = quantizer.estimate(
                full_unsigned[lane] - center * query_sum,
                factors,
                query_terms,
            );
            if used_fastscan {
                let distance_error =
                    factors.f_rescale.abs() * sign_weight * quantizer.fastscan_ip_error(query);
                if !heap.should_consider(distance - distance_error) {
                    continue;
                }
                let mut exact_coarse = 0.0f32;
                stats.refined_coarse_byte_lookups =
                    stats.refined_coarse_byte_lookups.saturating_add(plane_size);
                for byte_idx in 0..plane_size {
                    exact_coarse += quantizer.byte_subset_sum(
                        query,
                        byte_idx,
                        blocked_codes[code_block_start + byte_idx * lanes + lane],
                    );
                }
                let mut exact_unsigned = sign_weight * exact_coarse;
                stats.extra_plane_byte_lookups = stats
                    .extra_plane_byte_lookups
                    .saturating_add((bits - 1).saturating_mul(plane_size));
                for plane in 1..bits {
                    let weight = (1usize << (bits - 1 - plane)) as f32;
                    let plane_start = code_block_start + plane * plane_size * lanes;
                    for byte_idx in 0..plane_size {
                        exact_unsigned += weight
                            * quantizer.byte_subset_sum(
                                query,
                                byte_idx,
                                blocked_codes[plane_start + byte_idx * lanes + lane],
                            );
                    }
                }
                distance =
                    quantizer.estimate(exact_unsigned - center * query_sum, factors, query_terms);
            }
            stats.final_distance_evaluations = stats.final_distance_evaluations.saturating_add(1);
            if heap.should_consider(distance) {
                stats.heap_admissions = stats.heap_admissions.saturating_add(1);
                heap.push(distance, list.ids[block_start + lane]);
            }
        }
        block_start += lanes;
    }
}

fn read_block_factor(
    list: &RQReadList,
    block_start: usize,
    lanes: usize,
    first_field: usize,
    lane: usize,
    has_error: bool,
) -> RQCodeFactors {
    RQCodeFactors {
        f_add: list.blocked_factors[block_start + first_field * lanes + lane],
        f_rescale: list.blocked_factors[block_start + (first_field + 1) * lanes + lane],
        f_error: if has_error {
            list.blocked_factors[block_start + (first_field + 2) * lanes + lane]
        } else {
            0.0
        },
    }
}

fn factor_value(factors: RQVectorFactors, field: usize, bits: usize) -> f32 {
    match (bits, field) {
        (_, 0) => factors.coarse.f_add,
        (_, 1) => factors.coarse.f_rescale,
        (2.., 2) => factors.coarse.f_error,
        (2.., 3) => factors.full.f_add,
        (2.., 4) => factors.full.f_rescale,
        _ => unreachable!("factor field is validated by the caller"),
    }
}

fn index_factors_fields(bits: usize) -> usize {
    if bits == 1 {
        2
    } else {
        5
    }
}

fn validate_index_shape(index: &IVFRQIndex) -> io::Result<()> {
    if index.d == 0 || index.nlist == 0 {
        return Err(invalid_input("IVF-RQ dimension and nlist must be positive"));
    }
    if !is_supported_rq_bits(index.bits) {
        return Err(invalid_input("IVF-RQ bits must be in 1..=8"));
    }
    if index.padded_d != padded_dimension(index.d) {
        return Err(invalid_input("IVF-RQ padded dimension mismatch"));
    }
    if index.rotation_rounds != DEFAULT_RQ_ROTATION_ROUNDS {
        return Err(invalid_input(format!(
            "IVF-RQ rotation rounds must be {DEFAULT_RQ_ROTATION_ROUNDS}"
        )));
    }
    if index.quantizer_centroids.len() != checked_section_size(index.nlist, index.d)?
        || index.rotated_centroids.len() != checked_section_size(index.nlist, index.padded_d)?
    {
        return Err(invalid_input("IVF-RQ centroid storage shape mismatch"));
    }
    if index.ids.len() != index.nlist
        || index.codes.len() != index.nlist
        || index.factors.len() != index.nlist
    {
        return Err(invalid_input("IVF-RQ list storage does not match nlist"));
    }
    for list_id in 0..index.nlist {
        let count = index.ids[list_id].len();
        if index.codes[list_id].len() != checked_list_bytes(count, index.code_size())? {
            return Err(invalid_input(format!(
                "IVF-RQ list {list_id} code length mismatch"
            )));
        }
        if index.factors[list_id].len() != count {
            return Err(invalid_input(format!(
                "IVF-RQ list {list_id} factor count mismatch"
            )));
        }
    }
    Ok(())
}

fn write_u32_le(out: &mut dyn SeekWrite, value: u32) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

fn write_i32_le(out: &mut dyn SeekWrite, value: i32) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

fn write_i64_le(out: &mut dyn SeekWrite, value: i64) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

fn write_u64_le(out: &mut dyn SeekWrite, value: u64) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

fn write_f32_slice(out: &mut dyn SeekWrite, values: &[f32]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    out.write_all(&bytes)
}

fn read_u32_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<i32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_i64_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<i64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_u64_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn bytes_to_f32_vec(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(invalid_data("IVF-RQ f32 section is not aligned"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn validate_positive_i32(value: i32, field: &str) -> io::Result<i32> {
    if value <= 0 {
        Err(invalid_data(format!(
            "invalid IVF-RQ {field}: {value}; expected positive"
        )))
    } else {
        Ok(value)
    }
}

fn validate_non_negative_i32(value: i32, field: &str) -> io::Result<i32> {
    if value < 0 {
        Err(invalid_data(format!(
            "invalid IVF-RQ {field}: {value}; expected non-negative"
        )))
    } else {
        Ok(value)
    }
}

fn usize_to_i32(value: usize, field: &str) -> io::Result<i32> {
    i32::try_from(value).map_err(|_| invalid_input(format!("{field} exceeds i32")))
}

fn usize_to_u32(value: usize, field: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_input(format!("{field} exceeds u32")))
}

fn usize_to_i64(value: usize, field: &str) -> io::Result<i64> {
    i64::try_from(value).map_err(|_| invalid_input(format!("{field} exceeds i64")))
}

fn u64_to_i64(value: u64, field: &str) -> io::Result<i64> {
    i64::try_from(value).map_err(|_| invalid_input(format!("{field} exceeds i64")))
}

fn checked_section_size(left: usize, right: usize) -> io::Result<usize> {
    left.checked_mul(right)
        .filter(|&size| size <= 1 << 30)
        .ok_or_else(|| invalid_data("IVF-RQ section size overflow"))
}

fn checked_list_bytes(count: usize, bytes_per_entry: usize) -> io::Result<usize> {
    count
        .checked_mul(bytes_per_entry)
        .ok_or_else(|| invalid_data("IVF-RQ list byte size overflow"))
}

fn checked_list_offset(offset: i64, list_id: usize) -> io::Result<u64> {
    u64::try_from(offset).map_err(|_| {
        invalid_data(format!(
            "negative IVF-RQ list offset {offset} for list {list_id}"
        ))
    })
}

fn decode_roaring_filter(bytes: &[u8]) -> io::Result<RoaringTreemap> {
    RoaringTreemap::deserialize_from(bytes)
        .map_err(|error| invalid_input(format!("invalid RoaringTreemap filter: {error}")))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{PosWriter, ReadRequest, SeekRead};
    use crate::ivfpq::RowIdFilter;
    use std::collections::HashSet;
    use std::io::{Cursor, Read, Seek, SeekFrom};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct ReaderStats {
        calls: usize,
        max_ranges_per_batch: usize,
    }

    struct CountingReader {
        inner: Cursor<Vec<u8>>,
        stats: Arc<Mutex<ReaderStats>>,
        max_ranges_per_pread: usize,
    }

    impl SeekRead for CountingReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            let mut stats = self.stats.lock().unwrap();
            stats.calls += 1;
            stats.max_ranges_per_batch = stats.max_ranges_per_batch.max(ranges.len());
            drop(stats);
            for range in ranges {
                self.inner.seek(SeekFrom::Start(range.pos))?;
                self.inner.read_exact(range.buf)?;
            }
            Ok(())
        }

        fn read_capabilities(&self) -> crate::io::SeekReadCapabilities {
            crate::io::SeekReadCapabilities {
                max_ranges_per_pread: self.max_ranges_per_pread,
                ..crate::io::SeekReadCapabilities::default()
            }
        }
    }

    #[test]
    fn ivfrq_four_bit_roundtrip_uses_blocked_layout() {
        let d = 13;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 50.0;
                (0..d).map(move |dim| cluster + i as f32 * 0.01 + dim as f32)
            })
            .collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = IVFRQIndexReader::open(Cursor::new(bytes)).unwrap();
        let (labels, distances) = reader.search(&data[37 * d..38 * d], 5, nlist).unwrap();

        assert_eq!(reader.num_bits, 4);
        assert_eq!(reader.padded_d, 64);
        assert_eq!(labels[0], ids[37]);
        assert!(distances[0].abs() <= 1e-3);
    }

    #[test]
    fn ivfrq_batch_reader_submits_multiple_lists_together() {
        let d = 64;
        let nlist = 3;
        let data: Vec<f32> = (0..96)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                (0..d).map(move |dim| cluster + dim as f32 * 0.01)
            })
            .collect();
        let ids: Vec<i64> = (0..96).collect();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, 96);
        index.add(&data, &ids, 96);
        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let stats = Arc::new(Mutex::new(ReaderStats::default()));
        let stream = CountingReader {
            inner: Cursor::new(bytes),
            stats: Arc::clone(&stats),
            max_ranges_per_pread: 0,
        };
        let mut reader = IVFRQIndexReader::open(stream).unwrap();
        search_batch_ivfrq_reader(&mut reader, &data[..d], 1, 1, nlist).unwrap();

        assert_eq!(stats.lock().unwrap().max_ranges_per_batch, nlist);
    }

    #[test]
    fn ivfrq_unfiltered_scan_matches_all_rows_filter() {
        let d = FASTSCAN_MIN_PADDED_DIMENSION;
        let nlist = 4;
        let n = 256;
        let nq = 8;
        let data = (0..n)
            .flat_map(|row| {
                let cluster = (row % nlist) as f32 * 40.0;
                (0..d).map(move |dimension| cluster + row as f32 * 0.003 + dimension as f32 * 0.01)
            })
            .collect::<Vec<_>>();
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);
        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let mut unfiltered_reader = IVFRQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let unfiltered =
            search_batch_ivfrq_reader(&mut unfiltered_reader, &data[..nq * d], nq, 10, nlist)
                .unwrap();
        let all_rows = ids.iter().copied().collect::<HashSet<_>>();
        let mut filtered_reader = IVFRQIndexReader::open(Cursor::new(bytes)).unwrap();
        let filtered = search_batch_ivfrq_reader_filter(
            &mut filtered_reader,
            &data[..nq * d],
            nq,
            10,
            nlist,
            Some(&all_rows),
        )
        .unwrap();

        assert_eq!(unfiltered, filtered);
        assert!(unfiltered_reader.last_search_stats().fastscan_blocks > 0);
        assert_eq!(filtered_reader.last_search_stats().fastscan_blocks, 0);
    }

    #[test]
    fn ivfrq_search_stats_report_two_stage_filtering_work() {
        let d = FASTSCAN_MIN_PADDED_DIMENSION;
        let nlist = 4;
        let n = 256;
        let nq = 8;
        let data = (0..n)
            .flat_map(|row| {
                let cluster = (row % nlist) as f32 * 40.0;
                (0..d).map(move |dimension| cluster + row as f32 * 0.003 + dimension as f32 * 0.01)
            })
            .collect::<Vec<_>>();
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);
        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let mut reader = IVFRQIndexReader::open(Cursor::new(bytes)).unwrap();
        search_batch_ivfrq_reader(&mut reader, &data[..nq * d], nq, 10, nlist).unwrap();

        let stats = reader.last_search_stats();
        assert_eq!(stats.query_count, nq);
        assert_eq!(stats.scanned_vectors, nq * n);
        assert_eq!(stats.eligible_vectors, nq * n);
        assert_eq!(stats.coarse_distance_evaluations, nq * n);
        assert!(stats.refined_vectors > 0);
        assert!(stats.refined_vectors <= stats.eligible_vectors);
        assert!(stats.final_distance_evaluations > 0);
        assert!(stats.final_distance_evaluations <= stats.refined_vectors);
        assert_eq!(
            stats.extra_plane_byte_lookups,
            (stats.refined_vectors + stats.refined_coarse_byte_lookups / index.plane_size())
                * (index.bits - 1)
                * index.plane_size()
        );
        assert!(stats.heap_admissions > 0);
        assert!(stats.fastscan_blocks > 0);
        assert!(stats.refined_coarse_byte_lookups > 0);
        assert_eq!(
            stats.refined_coarse_byte_lookups,
            stats.final_distance_evaluations * index.plane_size()
        );
    }

    #[test]
    fn ivfrq_resident_metadata_load_uses_one_read_round() {
        let d = 64;
        let nlist = 8;
        let data = (0..256)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                (0..d).map(move |dim| cluster + dim as f32 * 0.01)
            })
            .collect::<Vec<_>>();
        let ids = (0..256).collect::<Vec<i64>>();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, 256);
        index.add(&data, &ids, 256);
        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let stats = Arc::new(Mutex::new(ReaderStats::default()));
        let stream = CountingReader {
            inner: Cursor::new(bytes),
            stats: Arc::clone(&stats),
            max_ranges_per_pread: 0,
        };
        let mut reader = IVFRQIndexReader::open(stream).unwrap();
        *stats.lock().unwrap() = ReaderStats::default();
        reader.ensure_loaded().unwrap();

        assert_eq!(stats.lock().unwrap().calls, 1);
    }

    struct ThreadTrackingFilter {
        workers: AtomicU64,
    }

    impl RowIdFilter for ThreadTrackingFilter {
        fn contains(&self, _id: i64) -> bool {
            if let Some(worker) = rayon::current_thread_index() {
                self.workers.fetch_or(1u64 << worker, Ordering::Relaxed);
            }
            true
        }
    }

    #[test]
    fn ivfrq_batch_scans_queries_in_parallel_without_duplicate_reads() {
        let d = 64;
        let nlist = 8;
        let nq = 64;
        let n = 4096;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 50.0;
                (0..d).map(move |dim| cluster + i as f32 * 0.001 + dim as f32 * 0.01)
            })
            .collect();
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);
        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let stats = Arc::new(Mutex::new(ReaderStats::default()));
        let stream = CountingReader {
            inner: Cursor::new(bytes),
            stats: Arc::clone(&stats),
            max_ranges_per_pread: 0,
        };
        let mut reader = IVFRQIndexReader::open(stream).unwrap();
        let queries = data[..nq * d].to_vec();
        let filter = ThreadTrackingFilter {
            workers: AtomicU64::new(0),
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                search_batch_ivfrq_reader_filter(
                    &mut reader,
                    &queries,
                    nq,
                    10,
                    nlist,
                    Some(&filter),
                )
                .unwrap();
            });

        assert!(
            filter.workers.load(Ordering::Relaxed).count_ones() > 1,
            "batch scan should use more than one Rayon worker"
        );
        assert_eq!(stats.lock().unwrap().max_ranges_per_batch, nlist);
    }

    #[test]
    fn ivfrq_seeded_parallel_single_query_matches_sequential_index() {
        let d = 16;
        let nlist = 8;
        let n = 9_216;
        let k = 10;
        let data = (0..n)
            .flat_map(|row| {
                let cluster = (row % nlist) as f32 * 50.0;
                (0..d).map(move |dimension| cluster + row as f32 * 0.0007 + dimension as f32 * 0.01)
            })
            .collect::<Vec<_>>();
        let ids = (0..n as i64).collect::<Vec<_>>();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let query = &data[137 * d..138 * d];
        let mut expected_ids = vec![-1; k];
        let mut expected_distances = vec![f32::MAX; k];
        index.search(
            query,
            1,
            k,
            nlist,
            &mut expected_distances,
            &mut expected_ids,
        );

        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = IVFRQIndexReader::open(Cursor::new(bytes)).unwrap();
        let (actual_ids, actual_distances) = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| reader.search(query, k, nlist).unwrap());

        assert_eq!(actual_ids, expected_ids);
        for (actual, expected) in actual_distances.iter().zip(expected_distances) {
            assert!((actual - expected).abs() <= 1e-3);
        }
        let stats = reader.last_search_stats();
        assert_eq!(stats.seeded_lists, 1);
        assert_eq!(stats.parallel_list_tasks, nlist);
        assert_eq!(
            stats.scanned_vectors,
            n + PARALLEL_RQ_SEED_VECTORS,
            "the seed prefix is intentionally rescanned by its parallel list task"
        );
    }

    #[test]
    fn ivfrq_batch_respects_reader_range_capability() {
        let d = 64;
        let nlist = 5;
        let data = (0..160)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                (0..d).map(move |dim| cluster + dim as f32 * 0.01)
            })
            .collect::<Vec<_>>();
        let ids = (0..160).collect::<Vec<i64>>();
        let mut index = IVFRQIndex::with_bits(d, nlist, 4, MetricType::L2);
        index.train(&data, 160);
        index.add(&data, &ids, 160);
        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let stats = Arc::new(Mutex::new(ReaderStats::default()));
        let stream = CountingReader {
            inner: Cursor::new(bytes),
            stats: Arc::clone(&stats),
            max_ranges_per_pread: 2,
        };
        let mut reader = IVFRQIndexReader::open(stream).unwrap();
        reader.search(&data[..d], 1, nlist).unwrap();

        assert_eq!(stats.lock().unwrap().max_ranges_per_batch, 2);
    }

    #[test]
    fn ivfrq_header_rejects_the_unreleased_one_bit_layout() {
        let mut old_header = vec![0u8; IVF_RQ_HEADER_SIZE];
        old_header[0..4].copy_from_slice(&IVF_RQ_MAGIC.to_le_bytes());
        old_header[4..8].copy_from_slice(&IVF_RQ_VERSION.to_le_bytes());
        old_header[8..12].copy_from_slice(&8i32.to_le_bytes());
        old_header[12..16].copy_from_slice(&1i32.to_le_bytes());

        let error = match IVFRQIndexReader::open(Cursor::new(old_header)) {
            Ok(_) => panic!("old IVF-RQ layout must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("padded_dimension"));
    }

    #[test]
    fn ivfrq_header_records_data_bits_and_new_layout() {
        let d = 64;
        let data = vec![0.0; d * 2];
        let mut index = IVFRQIndex::with_bits(d, 1, 5, MetricType::L2);
        index.train(&data, 2);
        index.add(&data, &[7, 9], 2);
        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 5);
        assert_eq!(
            u32::from_le_bytes(bytes[56..60].try_into().unwrap()),
            IVF_RQ_ROTATION_TYPE_BLOCK_FHT
        );
        assert_eq!(
            u32::from_le_bytes(bytes[60..64].try_into().unwrap()),
            IVF_RQ_FACTOR_LAYOUT_COMPACT_V1
        );
    }

    #[test]
    fn ivfrq_four_bit_storage_omits_the_unused_full_error_factor() {
        let d = 64;
        let count = 32;
        let data = (0..count * d)
            .map(|offset| ((offset * 17) % 101) as f32)
            .collect::<Vec<_>>();
        let ids = (0..count as i64).collect::<Vec<_>>();
        let mut index = IVFRQIndex::with_bits(d, 1, 4, MetricType::L2);
        index.train(&data, count);
        index.add(&data, &ids, count);

        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let offset_entry = IVF_RQ_HEADER_SIZE + d * size_of::<f32>();
        let list_offset =
            i64::from_le_bytes(bytes[offset_entry..offset_entry + 8].try_into().unwrap()) as usize;
        let id_bytes = i32::from_le_bytes(
            bytes[offset_entry + 12..offset_entry + 16]
                .try_into()
                .unwrap(),
        ) as usize;
        let expected_factor_bytes = count * 5 * size_of::<f32>();
        let expected_list_bytes = 16 + id_bytes + count * index.code_size() + expected_factor_bytes;

        assert_eq!(bytes.len() - list_offset, expected_list_bytes);
    }

    #[test]
    fn ivfrq_one_bit_storage_omits_the_unused_coarse_error_factor() {
        let d = 64;
        let count = 32;
        let data = (0..count * d)
            .map(|offset| ((offset * 19) % 103) as f32)
            .collect::<Vec<_>>();
        let ids = (0..count as i64).collect::<Vec<_>>();
        let mut index = IVFRQIndex::with_bits(d, 1, 1, MetricType::L2);
        index.train(&data, count);
        index.add(&data, &ids, count);

        let mut bytes = Vec::new();
        write_ivfrq_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let offset_entry = IVF_RQ_HEADER_SIZE + d * size_of::<f32>();
        let list_offset =
            i64::from_le_bytes(bytes[offset_entry..offset_entry + 8].try_into().unwrap()) as usize;
        let id_bytes = i32::from_le_bytes(
            bytes[offset_entry + 12..offset_entry + 16]
                .try_into()
                .unwrap(),
        ) as usize;
        let expected_factor_bytes = count * 2 * size_of::<f32>();
        let expected_list_bytes = 16 + id_bytes + count * index.code_size() + expected_factor_bytes;

        assert_eq!(bytes.len() - list_offset, expected_list_bytes);
    }
}
