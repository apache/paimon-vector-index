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
use crate::hnsw::{HnswBuildParams, HnswGraph, HnswSearchWorkspace};
use crate::hnsw_search::{search_hnsw_lists, HnswSearchList};
use crate::index_io_util::{
    checked_list_bytes, checked_list_offset, checked_section_size, decode_delta_varint_ids,
    decode_graph, decode_roaring_filter, encode_delta_varint_ids, encode_graph, read_f32_vec,
    read_i32_le, read_i64_le, read_u32_le, u64_to_i64, usize_to_i32, usize_to_i64,
    validate_positive_i32, validate_reserved_zero, validate_search_inputs, write_f32_slice,
    write_i32_le, write_i64_le, write_u32_le,
};
use crate::io::{PreadCursor, ReadRequest, SeekRead, SeekWrite};
use crate::ivfhnswsq::IVFHNSWSQIndex;
use crate::kmeans;
use crate::row_id_filter::RowIdFilter;
use crate::sq::{ScalarQuantizer, ScalarQuantizerDecodeLut};
use crate::topk::TopKHeap;
use std::io;
use std::sync::Arc;

pub const IVF_HNSW_SQ_MAGIC: u32 = 0x49485351; // "IHSQ"
pub const IVF_HNSW_SQ_VERSION: u32 = 1;
pub const IVF_HNSW_SQ_HEADER_SIZE: usize = 64;
const FLAG_DELTA_IDS: u32 = 1 << 0;
const FLAG_GRAPH_V1: u32 = 1 << 1;
const REQUIRED_FLAGS: u32 = FLAG_DELTA_IDS | FLAG_GRAPH_V1;
const SUPPORTED_FLAGS: u32 = REQUIRED_FLAGS;
const MAX_COALESCED_READ_GAP_BYTES: u64 = 1 << 20;

pub fn write_ivfhnswsq_index(index: &IVFHNSWSQIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    validate_index_shape(index)?;
    let total_vectors = index.ids.iter().try_fold(0i64, |sum, ids| {
        let count = usize_to_i64(ids.len(), "total vector count")?;
        sum.checked_add(count).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "total vector count exceeds i64 length limit",
            )
        })
    })?;
    let sorted_lists: Vec<SortedSqGraphList> = (0..index.nlist)
        .map(|list_id| build_sorted_sq_graph_list(index, list_id))
        .collect::<io::Result<_>>()?;

    write_u32_le(out, IVF_HNSW_SQ_MAGIC)?;
    write_u32_le(out, IVF_HNSW_SQ_VERSION)?;
    write_i32_le(out, usize_to_i32(index.d, "dimension")?)?;
    write_i32_le(out, usize_to_i32(index.nlist, "nlist")?)?;
    write_u32_le(out, index.metric as u32)?;
    write_i64_le(out, total_vectors)?;
    let params = index.hnsw_params.sanitized();
    write_i32_le(out, usize_to_i32(params.m, "hnsw m")?)?;
    write_i32_le(
        out,
        usize_to_i32(params.ef_construction, "hnsw ef_construction")?,
    )?;
    write_i32_le(out, usize_to_i32(params.max_level, "hnsw max_level")?)?;
    let (sq_min, sq_max) = sq_global_bounds(&index.sq.mins, &index.sq.maxs);
    out.write_all(&sq_min.to_le_bytes())?;
    out.write_all(&sq_max.to_le_bytes())?;
    write_u32_le(out, REQUIRED_FLAGS)?;
    out.write_all(&[0u8; 12])?;

    write_f32_slice(out, &index.sq.mins)?;
    write_f32_slice(out, &index.sq.maxs)?;
    for sq in &index.list_sqs {
        write_f32_slice(out, &sq.mins)?;
        write_f32_slice(out, &sq.maxs)?;
    }
    write_f32_slice(out, &index.quantizer_centroids)?;

    let offset_table_size = index.nlist.checked_mul(24).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-HNSW-SQ offset table size overflow",
        )
    })?;
    let data_start = out
        .pos()
        .checked_add(offset_table_size as u64)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-HNSW-SQ data start offset overflow",
            )
        })?;
    let mut list_offsets = vec![0i64; index.nlist];
    let mut list_counts = vec![0i32; index.nlist];
    let mut list_graph_bytes_lens = vec![0i32; index.nlist];
    let mut list_payload_bytes_lens = vec![0i64; index.nlist];
    let mut current_offset = data_start;

    for list_id in 0..index.nlist {
        list_offsets[list_id] = u64_to_i64(current_offset, "list offset")?;
        let count = sorted_lists[list_id].ids.len();
        list_counts[list_id] = usize_to_i32(count, "list count")?;
        list_graph_bytes_lens[list_id] =
            usize_to_i32(sorted_lists[list_id].graph_bytes.len(), "graph bytes")?;
        if count > 0 {
            let payload_len = list_payload_len(
                count,
                index.sq.code_size(),
                sorted_lists[list_id].id_bytes.len(),
                sorted_lists[list_id].graph_bytes.len(),
            )?;
            list_payload_bytes_lens[list_id] = usize_to_i64(payload_len, "list payload bytes")?;
            current_offset = current_offset
                .checked_add(payload_len as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVF-HNSW-SQ offset overflow")
                })?;
        }
    }

    for list_id in 0..index.nlist {
        write_i64_le(out, list_offsets[list_id])?;
        write_i32_le(out, list_counts[list_id])?;
        write_i32_le(out, list_graph_bytes_lens[list_id])?;
        write_i64_le(out, list_payload_bytes_lens[list_id])?;
    }

    for list_id in 0..index.nlist {
        let list = &sorted_lists[list_id];
        if list.ids.is_empty() {
            continue;
        }
        write_i64_le(out, list.ids[0])?;
        write_i32_le(out, usize_to_i32(list.id_bytes.len(), "delta ID section")?)?;
        out.write_all(&list.id_bytes)?;
        out.write_all(&list.codes)?;
        out.write_all(&list.graph_bytes)?;
    }

    Ok(())
}

pub struct IVFHNSWSQIndexReader<R: SeekRead> {
    reader: R,
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub hnsw_params: HnswBuildParams,
    pub sq: ScalarQuantizer,
    pub list_sqs: Vec<ScalarQuantizer>,
    pub quantizer_centroids: Vec<f32>,
    pub list_offsets: Vec<i64>,
    pub list_counts: Vec<i32>,
    pub list_graph_bytes_lens: Vec<i32>,
    pub list_payload_bytes_lens: Vec<i64>,
    sq_decode_luts: Vec<Arc<ScalarQuantizerDecodeLut>>,
    loaded: bool,
}

impl<R: SeekRead> IVFHNSWSQIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        let mut cursor = PreadCursor::new(&mut reader, 0);

        let magic = read_u32_le(&mut cursor)?;
        if magic != IVF_HNSW_SQ_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVF_HNSW_SQ magic: 0x{:08X}", magic),
            ));
        }
        let version = read_u32_le(&mut cursor)?;
        if version != IVF_HNSW_SQ_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF_HNSW_SQ version: {}", version),
            ));
        }

        let d = validate_positive_i32(read_i32_le(&mut cursor)?, "d")? as usize;
        let nlist = validate_positive_i32(read_i32_le(&mut cursor)?, "nlist")? as usize;
        let metric_code = read_u32_le(&mut cursor)?;
        let metric = MetricType::from_code(metric_code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown metric type: {}", metric_code),
            )
        })?;
        let total_vectors = read_i64_le(&mut cursor)?;
        let hnsw_params = HnswBuildParams {
            m: validate_positive_i32(read_i32_le(&mut cursor)?, "hnsw m")? as usize,
            ef_construction: validate_positive_i32(
                read_i32_le(&mut cursor)?,
                "hnsw ef_construction",
            )? as usize,
            max_level: validate_positive_i32(read_i32_le(&mut cursor)?, "hnsw max_level")? as usize,
        }
        .sanitized();
        let sq_min_summary = read_f32_le(&mut cursor)?;
        let sq_max_summary = read_f32_le(&mut cursor)?;
        let flags = read_u32_le(&mut cursor)?;
        let mut reserved = [0u8; 12];
        cursor.read_exact(&mut reserved)?;
        validate_reserved_zero(&reserved, "IVF_HNSW_SQ")?;
        let unknown_flags = flags & !SUPPORTED_FLAGS;
        if unknown_flags != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF_HNSW_SQ flags: 0x{:08X}", unknown_flags),
            ));
        }
        if flags & REQUIRED_FLAGS != REQUIRED_FLAGS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF_HNSW_SQ v1 requires delta-varint IDs and graph v1",
            ));
        }

        let mins = read_f32_vec(&mut cursor, d)?;
        let maxs = read_f32_vec(&mut cursor, d)?;
        validate_sq_bounds(d, &mins, &maxs)?;
        let (sq_min, sq_max) = sq_global_bounds(&mins, &maxs);
        if sq_min.to_bits() != sq_min_summary.to_bits()
            || sq_max.to_bits() != sq_max_summary.to_bits()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SQ bounds summary does not match global SQ bounds",
            ));
        }
        let sq = ScalarQuantizer::with_dimension_bounds(d, mins, maxs);
        let mut list_sqs = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            let mins = read_f32_vec(&mut cursor, d)?;
            let maxs = read_f32_vec(&mut cursor, d)?;
            validate_sq_bounds(d, &mins, &maxs)?;
            list_sqs.push(ScalarQuantizer::with_dimension_bounds(d, mins, maxs));
        }

        Ok(Self {
            reader,
            d,
            nlist,
            metric,
            total_vectors,
            hnsw_params,
            sq,
            list_sqs,
            quantizer_centroids: Vec::new(),
            list_offsets: Vec::new(),
            list_counts: Vec::new(),
            list_graph_bytes_lens: Vec::new(),
            list_payload_bytes_lens: Vec::new(),
            sq_decode_luts: Vec::new(),
            loaded: false,
        })
    }

    pub fn ensure_loaded(&mut self) -> io::Result<()> {
        if self.loaded {
            return Ok(());
        }

        let quantizer_centroids_offset =
            IVF_HNSW_SQ_HEADER_SIZE as u64 + (self.d as u64) * 8 * (self.nlist as u64 + 1);
        let mut cursor = PreadCursor::new(&mut self.reader, quantizer_centroids_offset);
        self.quantizer_centroids =
            read_f32_vec(&mut cursor, checked_section_size(self.nlist, self.d)?)?;
        self.list_offsets = vec![0; self.nlist];
        self.list_counts = vec![0; self.nlist];
        self.list_graph_bytes_lens = vec![0; self.nlist];
        self.list_payload_bytes_lens = vec![0; self.nlist];
        for list_id in 0..self.nlist {
            self.list_offsets[list_id] = read_i64_le(&mut cursor)?;
            let count = read_i32_le(&mut cursor)?;
            if count < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative list count {} at list {}", count, list_id),
                ));
            }
            self.list_counts[list_id] = count;
            let graph_bytes_len = read_i32_le(&mut cursor)?;
            if graph_bytes_len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "negative graph_bytes_len {} at list {}",
                        graph_bytes_len, list_id
                    ),
                ));
            }
            self.list_graph_bytes_lens[list_id] = graph_bytes_len;
            let payload_bytes_len = read_i64_le(&mut cursor)?;
            if payload_bytes_len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "negative payload_bytes_len {} at list {}",
                        payload_bytes_len, list_id
                    ),
                ));
            }
            self.list_payload_bytes_lens[list_id] = payload_bytes_len;
        }

        self.loaded = true;
        Ok(())
    }

    pub fn optimize_for_search(&mut self) -> io::Result<()> {
        self.ensure_loaded()?;
        if self.sq_decode_luts.len() != self.nlist {
            // These LUTs only help when search falls back to scanning SQ codes,
            // for example filtered searches with a small candidate set. The
            // normal unfiltered path searches decoded vectors in the HNSW graph.
            self.sq_decode_luts = self
                .list_sqs
                .iter()
                .map(|sq| Arc::new(sq.build_decode_lut()))
                .collect();
        }
        Ok(())
    }

    pub fn read_inverted_list(
        &mut self,
        list_id: usize,
    ) -> io::Result<(Vec<i64>, Vec<u8>, Option<HnswGraph>)> {
        let Some(list) = self.read_graph_list(list_id)? else {
            return Ok((Vec::new(), Vec::new(), None));
        };
        Ok((list.ids, list.codes, Some(list.graph)))
    }

    fn read_graph_list(&mut self, list_id: usize) -> io::Result<Option<GraphList>> {
        self.ensure_loaded()?;
        let Some(meta) = self.list_payload_meta(list_id)? else {
            return Ok(None);
        };
        let mut payload = vec![0u8; meta.payload_len];
        self.reader
            .pread(&mut [ReadRequest::new(meta.offset, &mut payload)])?;

        self.decode_graph_list_payload(meta, &payload).map(Some)
    }

    fn read_graph_lists_coalesced(
        &mut self,
        list_ids: &[usize],
    ) -> io::Result<Vec<(usize, GraphList)>> {
        self.ensure_loaded()?;
        let mut metas = Vec::new();
        for &list_id in list_ids {
            if let Some(meta) = self.list_payload_meta(list_id)? {
                metas.push(meta);
            }
        }
        if metas.is_empty() {
            return Ok(Vec::new());
        }

        metas.sort_by_key(|meta| meta.offset);
        let mut loaded = Vec::with_capacity(metas.len());
        let mut range_start = metas[0].offset;
        let mut range_end = metas[0].end_offset()?;
        let mut range_payload_bytes = metas[0].payload_len;
        let mut range_metas = vec![metas[0]];
        for &meta in metas.iter().skip(1) {
            let meta_end = meta.end_offset()?;
            let gap = meta.offset.checked_sub(range_end).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF-HNSW-SQ list payload offsets overlap",
                )
            })?;
            if should_coalesce_gap(
                gap,
                range_start,
                meta_end,
                range_payload_bytes,
                meta.payload_len,
            ) {
                range_end = meta_end;
                range_payload_bytes = range_payload_bytes
                    .checked_add(meta.payload_len)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "coalesced IVF-HNSW-SQ requested payload bytes overflow",
                        )
                    })?;
                range_metas.push(meta);
            } else {
                self.read_coalesced_graph_list_range(
                    range_start,
                    range_end,
                    &range_metas,
                    &mut loaded,
                )?;
                range_start = meta.offset;
                range_end = meta_end;
                range_payload_bytes = meta.payload_len;
                range_metas.clear();
                range_metas.push(meta);
            }
        }
        self.read_coalesced_graph_list_range(range_start, range_end, &range_metas, &mut loaded)?;

        loaded.sort_by_key(|(list_id, _)| {
            list_ids
                .iter()
                .position(|&requested_id| requested_id == *list_id)
                .unwrap_or(usize::MAX)
        });
        Ok(loaded)
    }

    fn read_coalesced_graph_list_range(
        &mut self,
        range_start: u64,
        range_end: u64,
        metas: &[ListPayloadMeta],
        loaded: &mut Vec<(usize, GraphList)>,
    ) -> io::Result<()> {
        let byte_len = usize::try_from(range_end.checked_sub(range_start).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "coalesced IVF-HNSW-SQ read range is invalid",
            )
        })?)
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "coalesced IVF-HNSW-SQ read range exceeds usize",
            )
        })?;
        let mut payload = vec![0u8; byte_len];
        self.reader
            .pread(&mut [ReadRequest::new(range_start, &mut payload)])?;

        for &meta in metas {
            let start = usize::try_from(meta.offset - range_start).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "coalesced IVF-HNSW-SQ payload offset exceeds usize",
                )
            })?;
            let end = start.checked_add(meta.payload_len).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "coalesced IVF-HNSW-SQ payload slice overflows",
                )
            })?;
            loaded.push((
                meta.list_id,
                self.decode_graph_list_payload(meta, &payload[start..end])?,
            ));
        }
        Ok(())
    }

    fn list_payload_meta(&self, list_id: usize) -> io::Result<Option<ListPayloadMeta>> {
        if list_id >= self.nlist {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("list_id {} out of range (nlist={})", list_id, self.nlist),
            ));
        }
        let count = self.list_counts[list_id] as usize;
        if count == 0 {
            return Ok(None);
        }

        let offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
        let graph_bytes_len = self.list_graph_bytes_lens[list_id] as usize;
        if graph_bytes_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("list {} is missing HNSW graph", list_id),
            ));
        }
        let payload_len = self.list_payload_bytes_lens[list_id] as usize;
        if payload_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("list {} is missing payload length", list_id),
            ));
        }
        let minimum_payload_len = 12usize.checked_add(graph_bytes_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-HNSW-SQ minimum payload length overflow",
            )
        })?;
        if payload_len < minimum_payload_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "list {} payload length {} is shorter than expected {}",
                    list_id, payload_len, minimum_payload_len
                ),
            ));
        }
        Ok(Some(ListPayloadMeta {
            list_id,
            offset,
            count,
            payload_len,
        }))
    }

    fn decode_graph_list_payload(
        &self,
        meta: ListPayloadMeta,
        payload: &[u8],
    ) -> io::Result<GraphList> {
        if payload.len() != meta.payload_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "list {} payload length {} does not match expected {}",
                    meta.list_id,
                    payload.len(),
                    meta.payload_len
                ),
            ));
        }
        let base_header_len = 12usize;
        if payload.len() < base_header_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("list {} has truncated ID header", meta.list_id),
            ));
        }
        let base_id = i64::from_le_bytes(payload[0..8].try_into().unwrap());
        let id_bytes_len = i32::from_le_bytes(payload[8..12].try_into().unwrap());
        if id_bytes_len < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "negative id_bytes_len {} at list {}",
                    id_bytes_len, meta.list_id
                ),
            ));
        }
        let id_bytes_len = id_bytes_len as usize;
        let ids_end = base_header_len.checked_add(id_bytes_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-HNSW-SQ ID payload length overflow",
            )
        })?;
        let code_size = self.sq.code_size();
        let codes_bytes_len = checked_list_bytes(meta.count, code_size)?;
        let codes_end = ids_end.checked_add(codes_bytes_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-HNSW-SQ codes payload length overflow",
            )
        })?;
        if codes_end > payload.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("list {} has truncated SQ codes payload", meta.list_id),
            ));
        }
        let ids = decode_delta_varint_ids(base_id, &payload[base_header_len..ids_end], meta.count)?;
        let codes = payload[ids_end..codes_end].to_vec();
        let mut vectors = vec![0.0f32; meta.count * self.d];
        let centroid = self.list_centroid(meta.list_id).to_vec();
        self.list_sq(meta.list_id).decode_batch_with_offset(
            &codes,
            meta.count,
            &centroid,
            &mut vectors,
        );
        let graph = decode_graph(
            &payload[codes_end..],
            vectors,
            meta.count,
            self.d,
            self.metric,
            self.hnsw_params,
        )?;
        let graph = graph.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("list {} is missing HNSW graph", meta.list_id),
            )
        })?;
        Ok(GraphList {
            ids,
            codes,
            graph,
            centroid: Some(centroid),
            sq: self.list_sq(meta.list_id).clone(),
            sq_decode_lut: self.list_sq_decode_lut(meta.list_id).map(Arc::clone),
        })
    }

    fn list_centroid(&self, list_id: usize) -> &[f32] {
        &self.quantizer_centroids[list_id * self.d..(list_id + 1) * self.d]
    }

    fn list_sq(&self, list_id: usize) -> &ScalarQuantizer {
        self.list_sqs.get(list_id).unwrap_or(&self.sq)
    }

    fn list_sq_decode_lut(&self, list_id: usize) -> Option<&Arc<ScalarQuantizerDecodeLut>> {
        self.sq_decode_luts.get(list_id)
    }

    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        ef_search: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.search_with_filter(query, k, nprobe, ef_search, None)
    }

    pub fn search_with_filter(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        ef_search: usize,
        filter: Option<&dyn RowIdFilter>,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.ensure_loaded()?;
        validate_search_inputs(query, 1, self.d, k, nprobe)?;

        let q = preprocess_vectors(query, 1, self.d, self.metric);
        let (probe_indices, _) =
            kmeans::find_topk(&q, &self.quantizer_centroids, self.nlist, self.d, nprobe);
        let mut loaded_lists = Vec::with_capacity(probe_indices.len());
        for (_, list) in self.read_graph_lists_coalesced(&probe_indices)? {
            loaded_lists.push(list);
        }
        let search_lists: Vec<_> = loaded_lists
            .iter()
            .map(|list| HnswSearchList {
                ids: list.ids.as_slice(),
                graph: Some(&list.graph),
                payload: list,
            })
            .collect();
        let sorted = search_hnsw_lists(&q, &search_lists, k, ef_search, filter, |list, heap| {
            let list = list.payload;
            scan_sq_list(
                &q,
                &list.ids,
                &list.codes,
                list.centroid.as_deref(),
                &list.sq,
                list.sq_decode_lut.as_deref(),
                self.metric,
                filter,
                heap,
            );
        });
        let mut labels: Vec<i64> = sorted.iter().map(|&(_, id)| id).collect();
        let mut distances: Vec<f32> = sorted.iter().map(|&(dist, _)| dist).collect();
        labels.resize(k, -1);
        distances.resize(k, f32::MAX);
        Ok((labels, distances))
    }

    pub fn search_with_roaring_filter(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        ef_search: usize,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.search_with_filter(query, k, nprobe, ef_search, Some(&filter))
    }
}

pub fn search_batch_ivfhnswsq_reader<R: SeekRead>(
    reader: &mut IVFHNSWSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    ef_search: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfhnswsq_reader_filter(reader, queries, nq, k, nprobe, ef_search, None)
}

pub fn search_batch_ivfhnswsq_reader_filter<R: SeekRead>(
    reader: &mut IVFHNSWSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    ef_search: usize,
    filter: Option<&dyn RowIdFilter>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    reader.ensure_loaded()?;
    validate_search_inputs(queries, nq, reader.d, k, nprobe)?;

    let processed = preprocess_vectors(queries, nq, reader.d, reader.metric);
    let (all_probe_indices, _) = kmeans::find_topk_batch(
        &processed,
        nq,
        &reader.quantizer_centroids,
        reader.nlist,
        reader.d,
        nprobe,
    );

    let mut list_to_queries = vec![Vec::new(); reader.nlist];
    let mut unique_lists = Vec::new();
    for (qi, probe_indices) in all_probe_indices.iter().enumerate() {
        for &list_id in probe_indices {
            if list_to_queries[list_id].is_empty() {
                unique_lists.push(list_id);
            }
            list_to_queries[list_id].push(qi);
        }
    }

    let mut heaps: Vec<TopKHeap> = (0..nq).map(|_| TopKHeap::new(k)).collect();
    let mut search_workspace = HnswSearchWorkspace::new(ef_search.max(k));
    let mut query_filtered_counts = vec![0usize; nq];
    let mut loaded_lists = Vec::with_capacity(unique_lists.len());
    for (list_id, list) in reader.read_graph_lists_coalesced(&unique_lists)? {
        if let Some(f) = filter {
            let filtered = list.ids.iter().filter(|&&id| f.contains(id)).count();
            for &qi in &list_to_queries[list_id] {
                query_filtered_counts[qi] = query_filtered_counts[qi]
                    .checked_add(filtered)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "filtered vector count overflows usize",
                        )
                    })?;
            }
        }
        loaded_lists.push(LoadedBatchList {
            query_ids: std::mem::take(&mut list_to_queries[list_id]),
            ids: list.ids,
            codes: list.codes,
            graph: list.graph,
            centroid: list.centroid,
            sq: list.sq,
            sq_decode_lut: list.sq_decode_lut,
        });
    }

    for list in &loaded_lists {
        for &qi in &list.query_ids {
            let query = &processed[qi * reader.d..(qi + 1) * reader.d];
            let force_sq_scan = filter
                .map(|_| query_filtered_counts[qi] <= ef_search.max(k))
                .unwrap_or(false);
            if force_sq_scan {
                scan_sq_list(
                    query,
                    &list.ids,
                    &list.codes,
                    list.centroid.as_deref(),
                    &list.sq,
                    list.sq_decode_lut.as_deref(),
                    reader.metric,
                    filter,
                    &mut heaps[qi],
                );
            } else {
                let local_results = list.graph.search_with_reusable_workspace(
                    query,
                    ef_search.max(k),
                    ef_search.max(k),
                    &mut search_workspace,
                );
                for &(local_id, dist) in local_results {
                    let row_id = list.ids[local_id];
                    if filter.map(|f| f.contains(row_id)).unwrap_or(true) {
                        heaps[qi].push(dist, row_id);
                    }
                }
            }
        }
    }
    if filter.is_some() {
        for list in &loaded_lists {
            for &qi in &list.query_ids {
                if heaps[qi].len() >= k {
                    continue;
                }
                if query_filtered_counts[qi] <= ef_search.max(k) {
                    continue;
                }
                let query = &processed[qi * reader.d..(qi + 1) * reader.d];
                scan_sq_list(
                    query,
                    &list.ids,
                    &list.codes,
                    list.centroid.as_deref(),
                    &list.sq,
                    list.sq_decode_lut.as_deref(),
                    reader.metric,
                    filter,
                    &mut heaps[qi],
                );
            }
        }
    }

    let mut result_ids = vec![-1i64; nq * k];
    let mut result_dists = vec![f32::MAX; nq * k];
    for qi in 0..nq {
        let sorted = std::mem::replace(&mut heaps[qi], TopKHeap::new(0)).into_sorted();
        let base = qi * k;
        for (i, &(dist, id)) in sorted.iter().enumerate() {
            result_ids[base + i] = id;
            result_dists[base + i] = dist;
        }
    }

    Ok((result_ids, result_dists))
}

pub fn search_batch_ivfhnswsq_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFHNSWSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    ef_search: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_batch_ivfhnswsq_reader_filter(reader, queries, nq, k, nprobe, ef_search, Some(&filter))
}

struct GraphList {
    ids: Vec<i64>,
    codes: Vec<u8>,
    graph: HnswGraph,
    centroid: Option<Vec<f32>>,
    sq: ScalarQuantizer,
    sq_decode_lut: Option<Arc<ScalarQuantizerDecodeLut>>,
}

#[derive(Clone, Copy)]
struct ListPayloadMeta {
    list_id: usize,
    offset: u64,
    count: usize,
    payload_len: usize,
}

impl ListPayloadMeta {
    fn end_offset(self) -> io::Result<u64> {
        self.offset
            .checked_add(self.payload_len as u64)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF-HNSW-SQ list payload offset overflows u64",
                )
            })
    }
}

fn should_coalesce_gap(
    gap: u64,
    range_start: u64,
    next_range_end: u64,
    current_payload_bytes: usize,
    next_payload_bytes: usize,
) -> bool {
    if gap > MAX_COALESCED_READ_GAP_BYTES {
        return false;
    }
    let Some(requested_bytes) = current_payload_bytes.checked_add(next_payload_bytes) else {
        return false;
    };
    let Some(range_bytes) = next_range_end.checked_sub(range_start) else {
        return false;
    };
    range_bytes <= requested_bytes.saturating_mul(2) as u64
}

struct LoadedBatchList {
    query_ids: Vec<usize>,
    ids: Vec<i64>,
    codes: Vec<u8>,
    graph: HnswGraph,
    centroid: Option<Vec<f32>>,
    sq: ScalarQuantizer,
    sq_decode_lut: Option<Arc<ScalarQuantizerDecodeLut>>,
}

fn scan_sq_list(
    query: &[f32],
    ids: &[i64],
    codes: &[u8],
    centroid: Option<&[f32]>,
    sq: &ScalarQuantizer,
    sq_decode_lut: Option<&ScalarQuantizerDecodeLut>,
    metric: MetricType,
    filter: Option<&dyn RowIdFilter>,
    heap: &mut TopKHeap,
) {
    let context = sq.distance_context(query, metric);
    let code_size = sq.code_size();
    for (local_id, &row_id) in ids.iter().enumerate() {
        if filter.map(|f| !f.contains(row_id)).unwrap_or(false) {
            continue;
        }
        let code = &codes[local_id * code_size..(local_id + 1) * code_size];
        let dist = match (centroid, sq_decode_lut) {
            (Some(centroid), Some(lut)) => sq
                .distance_to_code_with_lut_offset_with_context(query, code, centroid, lut, context),
            (Some(centroid), None) => {
                sq.distance_to_code_with_offset_with_context(query, code, centroid, context)
            }
            (None, Some(lut)) => {
                sq.distance_to_code_with_lut_with_context(query, code, lut, context)
            }
            (None, None) => sq.distance_to_code_with_context(query, code, context),
        };
        heap.push(dist, row_id);
    }
}

fn validate_index_shape(index: &IVFHNSWSQIndex) -> io::Result<()> {
    if index.d == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dimension must be greater than 0",
        ));
    }
    if index.nlist == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "nlist must be greater than 0",
        ));
    }
    if index.sq.d != index.d {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SQ dimension does not match index dimension",
        ));
    }
    validate_sq_bounds(index.d, &index.sq.mins, &index.sq.maxs)?;
    if index.list_sqs.len() != index.nlist {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SQ list bounds count does not match nlist",
        ));
    }
    for (list_id, sq) in index.list_sqs.iter().enumerate() {
        if sq.d != index.d {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "SQ dimension for list {} does not match index dimension",
                    list_id
                ),
            ));
        }
        validate_sq_bounds(index.d, &sq.mins, &sq.maxs)?;
    }
    let centroid_len = checked_section_size(index.nlist, index.d)?;
    if index.quantizer_centroids.len() != centroid_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "quantizer centroid length {} does not match nlist*d {}",
                index.quantizer_centroids.len(),
                centroid_len
            ),
        ));
    }
    if index.ids.len() != index.nlist
        || index.codes.len() != index.nlist
        || index.graphs.len() != index.nlist
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inverted list count does not match nlist",
        ));
    }
    for list_id in 0..index.nlist {
        let count = index.ids[list_id].len();
        let expected_codes_len = checked_list_bytes(count, index.sq.code_size())?;
        if index.codes[list_id].len() != expected_codes_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "list {} SQ code length {} does not match count*d {}",
                    list_id,
                    index.codes[list_id].len(),
                    expected_codes_len
                ),
            ));
        }
        match &index.graphs[list_id] {
            Some(graph) if count > 0 => {
                let decoded = index.decode_list_vectors(list_id, count);
                if graph.len() != count || graph.vectors() != decoded.as_slice() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("list {} graph does not match SQ code storage", list_id),
                    ));
                }
            }
            Some(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("list {} has graph for an empty list", list_id),
                ));
            }
            None if count == 0 => {}
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "list {} is missing HNSW graph; call build_graphs first",
                        list_id
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_sq_bounds(d: usize, mins: &[f32], maxs: &[f32]) -> io::Result<()> {
    if mins.len() != d || maxs.len() != d {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "SQ bounds length mismatch: d={}, mins={}, maxs={}",
                d,
                mins.len(),
                maxs.len()
            ),
        ));
    }
    for (dim, (&min, &max)) in mins.iter().zip(maxs.iter()).enumerate() {
        if !min.is_finite() || !max.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("SQ bounds at dimension {} must be finite", dim),
            ));
        }
    }
    Ok(())
}

fn read_f32_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn sq_global_bounds(mins: &[f32], maxs: &[f32]) -> (f32, f32) {
    let min = mins.iter().copied().fold(f32::INFINITY, f32::min);
    let max = maxs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else {
        (0.0, 0.0)
    }
}

struct SortedSqGraphList {
    ids: Vec<i64>,
    id_bytes: Vec<u8>,
    codes: Vec<u8>,
    graph_bytes: Vec<u8>,
}

fn build_sorted_sq_graph_list(
    index: &IVFHNSWSQIndex,
    list_id: usize,
) -> io::Result<SortedSqGraphList> {
    let count = index.ids[list_id].len();
    if count == 0 {
        return Ok(SortedSqGraphList {
            ids: Vec::new(),
            id_bytes: Vec::new(),
            codes: Vec::new(),
            graph_bytes: Vec::new(),
        });
    }

    let code_size = index.sq.code_size();
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|&idx| index.ids[list_id][idx]);

    let ids: Vec<i64> = order.iter().map(|&idx| index.ids[list_id][idx]).collect();
    let (_, id_bytes) = encode_delta_varint_ids(&ids);

    let mut codes = vec![0u8; checked_list_bytes(count, code_size)?];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        codes[new_idx * code_size..(new_idx + 1) * code_size]
            .copy_from_slice(&index.codes[list_id][old_idx * code_size..(old_idx + 1) * code_size]);
    }

    let mut vectors = vec![0.0f32; count * index.d];
    let centroid = &index.quantizer_centroids[list_id * index.d..(list_id + 1) * index.d];
    index
        .list_sq(list_id)
        .decode_batch_with_offset(&codes, count, centroid, &mut vectors);
    let old_to_new = old_to_new_order(&order);
    let source_graph = index.graphs[list_id].as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("list {} is missing HNSW graph", list_id),
        )
    })?;
    let graph = reorder_graph(
        source_graph,
        &order,
        &old_to_new,
        vectors,
        index.d,
        index.metric,
        index.hnsw_params,
    )?;
    let graph_bytes = encode_graph(Some(&graph))?;

    Ok(SortedSqGraphList {
        ids,
        id_bytes,
        codes,
        graph_bytes,
    })
}

fn old_to_new_order(order: &[usize]) -> Vec<usize> {
    let mut old_to_new = vec![0; order.len()];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        old_to_new[old_idx] = new_idx;
    }
    old_to_new
}

fn reorder_graph(
    graph: &HnswGraph,
    order: &[usize],
    old_to_new: &[usize],
    vectors: Vec<f32>,
    d: usize,
    metric: MetricType,
    hnsw_params: HnswBuildParams,
) -> io::Result<HnswGraph> {
    let levels: Vec<usize> = order
        .iter()
        .map(|&old_idx| graph.levels()[old_idx])
        .collect();
    let neighbors: Vec<Vec<Vec<usize>>> = order
        .iter()
        .map(|&old_idx| {
            graph.neighbors()[old_idx]
                .iter()
                .map(|level_neighbors| {
                    level_neighbors
                        .iter()
                        .map(|&neighbor| old_to_new[neighbor])
                        .collect()
                })
                .collect()
        })
        .collect();
    HnswGraph::from_parts(
        vectors,
        order.len(),
        d,
        metric,
        levels,
        neighbors,
        old_to_new[graph.entry_point()],
        graph.max_observed_level(),
        hnsw_params,
    )
}

fn list_payload_len(
    count: usize,
    code_size: usize,
    id_bytes_len: usize,
    graph_bytes_len: usize,
) -> io::Result<usize> {
    let id_bytes = 12usize.checked_add(id_bytes_len).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-HNSW-SQ ID payload length overflow",
        )
    })?;
    let code_bytes = checked_list_bytes(count, code_size)?;
    id_bytes
        .checked_add(code_bytes)
        .and_then(|len| len.checked_add(graph_bytes_len))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-HNSW-SQ list payload length overflow",
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswBuildParams;
    use crate::io::PosWriter;
    use crate::io::{ReadRequest, SeekRead};
    use roaring::RoaringTreemap;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_ivfhnswsq_reader_rejects_missing_required_flags() {
        let mut buf = vec![0u8; IVF_HNSW_SQ_HEADER_SIZE + 16];
        buf[0..4].copy_from_slice(&IVF_HNSW_SQ_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&IVF_HNSW_SQ_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&2i32.to_le_bytes());
        buf[12..16].copy_from_slice(&1i32.to_le_bytes());
        buf[16..20].copy_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf[20..28].copy_from_slice(&0i64.to_le_bytes());
        buf[28..32].copy_from_slice(&2i32.to_le_bytes());
        buf[32..36].copy_from_slice(&8i32.to_le_bytes());
        buf[36..40].copy_from_slice(&3i32.to_le_bytes());
        buf[40..44].copy_from_slice(&0.0f32.to_le_bytes());
        buf[44..48].copy_from_slice(&0.0f32.to_le_bytes());
        buf[48..52].copy_from_slice(&0u32.to_le_bytes());

        let err = match IVFHNSWSQIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("missing required flags should be rejected"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("requires delta-varint IDs and graph v1"));
    }

    #[test]
    fn test_ivfhnswsq_reader_rejects_unknown_flags() {
        let mut buf = vec![0u8; IVF_HNSW_SQ_HEADER_SIZE + 16];
        buf[0..4].copy_from_slice(&IVF_HNSW_SQ_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&IVF_HNSW_SQ_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&2i32.to_le_bytes());
        buf[12..16].copy_from_slice(&1i32.to_le_bytes());
        buf[16..20].copy_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf[20..28].copy_from_slice(&0i64.to_le_bytes());
        buf[28..32].copy_from_slice(&2i32.to_le_bytes());
        buf[32..36].copy_from_slice(&8i32.to_le_bytes());
        buf[36..40].copy_from_slice(&3i32.to_le_bytes());
        buf[40..44].copy_from_slice(&0.0f32.to_le_bytes());
        buf[44..48].copy_from_slice(&0.0f32.to_le_bytes());
        buf[48..52].copy_from_slice(&(REQUIRED_FLAGS | (1 << 31)).to_le_bytes());

        let err = match IVFHNSWSQIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("unknown flags should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Unsupported IVF_HNSW_SQ flags"));
    }

    #[test]
    fn test_ivfhnswsq_reader_rejects_nonzero_reserved_bytes() {
        let mut buf = vec![0u8; IVF_HNSW_SQ_HEADER_SIZE + 16];
        buf[0..4].copy_from_slice(&IVF_HNSW_SQ_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&IVF_HNSW_SQ_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&2i32.to_le_bytes());
        buf[12..16].copy_from_slice(&1i32.to_le_bytes());
        buf[16..20].copy_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf[20..28].copy_from_slice(&0i64.to_le_bytes());
        buf[28..32].copy_from_slice(&2i32.to_le_bytes());
        buf[32..36].copy_from_slice(&8i32.to_le_bytes());
        buf[36..40].copy_from_slice(&3i32.to_le_bytes());
        buf[40..44].copy_from_slice(&0.0f32.to_le_bytes());
        buf[44..48].copy_from_slice(&0.0f32.to_le_bytes());
        buf[48..52].copy_from_slice(&REQUIRED_FLAGS.to_le_bytes());
        buf[52] = 1;

        let err = match IVFHNSWSQIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("non-zero reserved bytes should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("reserved bytes must be zero"));
    }

    #[test]
    fn test_ivfhnswsq_write_read_search_roundtrip() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [
                    cluster + i as f32 * 2.0,
                    10.0 + i as f32,
                    20.0 + i as f32,
                    30.0 + i as f32,
                ]
            })
            .collect();
        let ids: Vec<i64> = (10_000..10_000 + n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswsq_index(&index, &mut writer).unwrap();

        let mut reader = IVFHNSWSQIndexReader::open(Cursor::new(buf)).unwrap();
        let query_id = 23;
        let (labels, distances) = reader
            .search(&data[query_id * d..(query_id + 1) * d], 5, nlist, 32)
            .unwrap();

        assert_eq!(labels[0], ids[query_id]);
        assert!(distances[0].is_finite());
    }

    #[test]
    fn test_ivfhnswsq_write_read_preserves_sq_dimension_bounds() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, -100.0, 1.0, 100.0];
        let ids = vec![10, 11];
        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 2);
        index.add(&data, &ids, 2);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
        assert_eq!(
            u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            IVF_HNSW_SQ_VERSION
        );

        let reader = IVFHNSWSQIndexReader::open(Cursor::new(buf)).unwrap();

        assert_eq!(reader.sq.mins, index.sq.mins);
        assert_eq!(reader.sq.maxs, index.sq.maxs);
        assert_eq!(reader.sq.min, index.sq.min);
        assert_eq!(reader.sq.max, index.sq.max);
        assert_eq!(reader.list_sqs.len(), nlist);
        assert_eq!(reader.list_sqs[0].mins, index.list_sqs[0].mins);
        assert_eq!(reader.list_sqs[0].maxs, index.list_sqs[0].maxs);
    }

    #[test]
    fn test_ivfhnswsq_reader_rejects_mismatched_sq_bounds_summary() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, -100.0, 1.0, 100.0];
        let ids = vec![10, 11];
        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 2);
        index.add(&data, &ids, 2);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
        buf[40..44].copy_from_slice(&123.0f32.to_le_bytes());

        let err = match IVFHNSWSQIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("mismatched SQ bounds summary should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("SQ bounds summary"));
    }

    #[test]
    fn test_ivfhnswsq_reader_search_with_roaring_filter() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];
        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 3);
        index.add(&data, &ids, 3);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
        let mut reader = IVFHNSWSQIndexReader::open(Cursor::new(buf)).unwrap();

        let mut filter = RoaringTreemap::new();
        filter.insert(12);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        let (labels, _) = reader
            .search_with_roaring_filter(&[0.0, 0.0], 2, nlist, 8, &filter_bytes)
            .unwrap();

        assert_eq!(labels, vec![12, -1]);
    }

    #[test]
    fn test_ivfhnswsq_reader_optimized_filter_search_matches_unoptimized() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();

        let mut filter = RoaringTreemap::new();
        filter.insert(0);
        filter.insert(64);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        let mut baseline = IVFHNSWSQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let expected = baseline
            .search_with_roaring_filter(&data[0..d], 3, nlist, 64, &filter_bytes)
            .unwrap();

        let mut optimized = IVFHNSWSQIndexReader::open(Cursor::new(buf)).unwrap();
        optimized.optimize_for_search().unwrap();
        let actual = optimized
            .search_with_roaring_filter(&data[0..d], 3, nlist, 64, &filter_bytes)
            .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_ivfhnswsq_reader_search_coalesces_contiguous_list_reads() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();

        let pread_count = Arc::new(AtomicUsize::new(0));
        let cursor = CountingPreadCursor::new(buf, Arc::clone(&pread_count));
        let mut reader = IVFHNSWSQIndexReader::open(cursor).unwrap();
        reader.ensure_loaded().unwrap();
        pread_count.store(0, Ordering::SeqCst);

        reader.search(&data[0..d], 5, nlist, 32).unwrap();

        assert_eq!(pread_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_ivfhnswsq_write_read_search_roundtrip_cosine() {
        let d = 3;
        let nlist = 2;
        let data = vec![1.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.1];
        let ids = vec![10, 11, 12, 13];
        let mut index =
            IVFHNSWSQIndex::new(d, nlist, MetricType::Cosine, HnswBuildParams::default());
        index.train(&data, 4);
        index.add(&data, &ids, 4);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();

        let mut reader = IVFHNSWSQIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, distances) = reader.search(&[1.0, 0.0, 0.0], 2, nlist, 16).unwrap();

        assert_eq!(labels[0], 10);
        assert!(distances[0].is_finite());
    }

    #[test]
    fn test_ivfhnswsq_batch_matches_single_search() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
        let queries = [&data[7 * d..8 * d], &data[23 * d..24 * d]].concat();

        let mut batch_reader = IVFHNSWSQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let (batch_labels, batch_distances) =
            search_batch_ivfhnswsq_reader(&mut batch_reader, &queries, 2, 3, nlist, 32).unwrap();

        for qi in 0..2 {
            let mut single_reader = IVFHNSWSQIndexReader::open(Cursor::new(buf.clone())).unwrap();
            let (single_labels, single_distances) = single_reader
                .search(&queries[qi * d..(qi + 1) * d], 3, nlist, 32)
                .unwrap();
            assert_eq!(
                &batch_labels[qi * 3..(qi + 1) * 3],
                single_labels.as_slice()
            );
            assert_eq!(
                &batch_distances[qi * 3..(qi + 1) * 3],
                single_distances.as_slice()
            );
        }
    }

    #[test]
    fn test_ivfhnswsq_batch_reader_coalesces_contiguous_list_reads() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();

        let pread_count = Arc::new(AtomicUsize::new(0));
        let cursor = CountingPreadCursor::new(buf, Arc::clone(&pread_count));
        let mut reader = IVFHNSWSQIndexReader::open(cursor).unwrap();
        reader.ensure_loaded().unwrap();
        pread_count.store(0, Ordering::SeqCst);
        let queries = data[0..d].to_vec();

        search_batch_ivfhnswsq_reader(&mut reader, &queries, 1, 5, nlist, 32).unwrap();

        assert_eq!(pread_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_ivfhnswsq_reader_coalesces_small_gap_between_requested_lists() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();

        let pread_count = Arc::new(AtomicUsize::new(0));
        let cursor = CountingPreadCursor::new(buf, Arc::clone(&pread_count));
        let mut reader = IVFHNSWSQIndexReader::open(cursor).unwrap();
        reader.ensure_loaded().unwrap();
        assert!(reader.list_counts[..3].iter().all(|&count| count > 0));
        pread_count.store(0, Ordering::SeqCst);

        let lists = reader.read_graph_lists_coalesced(&[0, 2]).unwrap();

        assert_eq!(lists.len(), 2);
        assert_eq!(pread_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_ivfhnswsq_write_requires_graphs() {
        let mut index = IVFHNSWSQIndex::new(2, 1, MetricType::L2, HnswBuildParams::default());
        let data = vec![0.0, 0.0];
        index.train(&data, 1);
        index.add(&data, &[1], 1);

        let mut buf = Vec::new();
        let err = write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("build_graphs"));
    }

    #[test]
    fn test_ivfhnswsq_writer_rejects_stale_graph() {
        let mut index = IVFHNSWSQIndex::new(2, 1, MetricType::L2, HnswBuildParams::default());
        let data = vec![0.0, 0.0, 1.0, 0.0];
        index.train(&data, 2);
        index.add(&data, &[10, 11], 2);
        index.build_graphs().unwrap();
        index.graphs[0] = Some(
            HnswGraph::build(
                &[10.0, 10.0, 11.0, 11.0],
                2,
                2,
                MetricType::L2,
                HnswBuildParams::default(),
            )
            .unwrap(),
        );

        let mut buf = Vec::new();
        let err = write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("graph does not match"));
    }

    struct CountingPreadCursor {
        data: Vec<u8>,
        pread_count: Arc<AtomicUsize>,
    }

    impl CountingPreadCursor {
        fn new(data: Vec<u8>, pread_count: Arc<AtomicUsize>) -> Self {
            Self { data, pread_count }
        }
    }

    impl SeekRead for CountingPreadCursor {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            for range in ranges {
                self.pread_count.fetch_add(1, Ordering::SeqCst);
                let pos = range.pos as usize;
                let end = pos.checked_add(range.buf.len()).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "cursor position overflow")
                })?;
                if end > self.data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ));
                }
                range.buf.copy_from_slice(&self.data[pos..end]);
            }
            Ok(())
        }
    }
}
