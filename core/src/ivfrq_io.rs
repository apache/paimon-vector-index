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

use crate::distance::{fvec_madd, preprocess_vectors, MetricType};
use crate::index_io_util::{
    decode_delta_varint_ids, encode_delta_varint_ids, validate_search_inputs,
};
use crate::io::{PreadCursor, ReadRequest, SeekRead, SeekWrite};
use crate::ivfpq::RowIdFilter;
use crate::ivfrq::IVFRQIndex;
use crate::kmeans;
use crate::rq::{
    is_supported_query_bits, RQCodeFactors, RQDistanceContext, RQRotation, RaBitQuantizer,
    DEFAULT_RQ_QUERY_BITS, RQ_BYTE_LUT_MIN_LIST_SIZE,
};
use crate::topk::TopKHeap;
use roaring::RoaringTreemap;
use std::io;

pub const IVF_RQ_MAGIC: u32 = 0x49565251; // "IVRQ"
pub const IVF_RQ_VERSION: u32 = 1;
pub const IVF_RQ_HEADER_SIZE: usize = 64;

const FLAG_DELTA_IDS: u32 = 1 << 0;
const REQUIRED_FLAGS: u32 = FLAG_DELTA_IDS;
const SUPPORTED_FLAGS: u32 = REQUIRED_FLAGS;
const FACTOR_BYTES: usize = 12;

pub const IVF_RQ_NUM_BITS_ONE: u32 = 1;
pub const IVF_RQ_ROTATION_TYPE_KAC: u32 = 1;
pub const IVF_RQ_FACTOR_LAYOUT_RABITQ_V1: u32 = 1;

const FORMAT_FLAG_EX_CODES_PRESENT: u32 = 1 << 0;
const FORMAT_FLAG_ERROR_FACTOR_PRESENT: u32 = 1 << 1;
const SUPPORTED_FORMAT_FLAGS: u32 = FORMAT_FLAG_EX_CODES_PRESENT | FORMAT_FLAG_ERROR_FACTOR_PRESENT;
const CURRENT_FORMAT_FLAGS: u32 = 0;

struct SortedRQList {
    ids: Vec<i64>,
    id_bytes: Vec<u8>,
    codes: Vec<u8>,
    factors: Vec<RQCodeFactors>,
}

pub fn write_ivfrq_index(index: &IVFRQIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    validate_index_shape(index)?;
    let d = index.d;
    let nlist = index.nlist;
    let code_size = index.code_size();
    let total_vectors = index.ids.iter().try_fold(0i64, |sum, ids| {
        let count = usize_to_i64(ids.len(), "total vector count")?;
        sum.checked_add(count).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "total vector count exceeds i64 length limit",
            )
        })
    })?;

    let mut sorted_lists = Vec::with_capacity(nlist);
    for list_id in 0..nlist {
        let count = index.ids[list_id].len();
        if count == 0 {
            sorted_lists.push(SortedRQList {
                ids: Vec::new(),
                id_bytes: Vec::new(),
                codes: Vec::new(),
                factors: Vec::new(),
            });
            continue;
        }

        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by_key(|&idx| index.ids[list_id][idx]);

        let sorted_ids: Vec<i64> = order.iter().map(|&idx| index.ids[list_id][idx]).collect();
        let mut sorted_codes = Vec::with_capacity(count * code_size);
        let mut sorted_factors = Vec::with_capacity(count);
        for idx in order {
            sorted_codes
                .extend_from_slice(&index.codes[list_id][idx * code_size..(idx + 1) * code_size]);
            sorted_factors.push(index.factors[list_id][idx]);
        }
        let (_, id_bytes) = encode_delta_varint_ids(&sorted_ids);
        sorted_lists.push(SortedRQList {
            ids: sorted_ids,
            id_bytes,
            codes: sorted_codes,
            factors: sorted_factors,
        });
    }

    write_u32_le(out, IVF_RQ_MAGIC)?;
    write_u32_le(out, IVF_RQ_VERSION)?;
    write_i32_le(out, usize_to_i32(d, "dimension")?)?;
    write_i32_le(out, usize_to_i32(nlist, "nlist")?)?;
    write_u32_le(out, index.metric as u32)?;
    write_u32_le(out, REQUIRED_FLAGS)?;
    write_i64_le(out, total_vectors)?;
    write_u64_le(out, index.rotation_seed)?;
    write_u32_le(out, index.rotation_rounds)?;
    write_i32_le(out, usize_to_i32(code_size, "code size")?)?;
    write_u32_le(out, IVF_RQ_NUM_BITS_ONE)?;
    write_u32_le(out, IVF_RQ_ROTATION_TYPE_KAC)?;
    write_u32_le(out, IVF_RQ_FACTOR_LAYOUT_RABITQ_V1)?;
    write_u32_le(out, CURRENT_FORMAT_FLAGS)?;

    write_f32_slice(out, &index.quantizer_centroids)?;

    let offset_table_size = nlist.checked_mul(16).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-RQ offset table size overflow",
        )
    })?;
    let data_start = out
        .pos()
        .checked_add(offset_table_size as u64)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-RQ data start offset overflow",
            )
        })?;
    let mut list_offsets = vec![0i64; nlist];
    let mut list_counts = vec![0i32; nlist];
    let mut list_id_bytes_lens = vec![0i32; nlist];
    let mut current_offset = data_start;

    for list_id in 0..nlist {
        list_offsets[list_id] = u64_to_i64(current_offset, "list offset")?;
        let count = sorted_lists[list_id].ids.len();
        list_counts[list_id] = usize_to_i32(count, "list count")?;
        if count > 0 {
            let id_bytes_len = sorted_lists[list_id].id_bytes.len();
            list_id_bytes_lens[list_id] = usize_to_i32(id_bytes_len, "delta ID section")?;
            let code_bytes = checked_list_bytes(count, code_size)?;
            let factor_bytes = checked_list_bytes(count, FACTOR_BYTES)?;
            let list_bytes = 12usize
                .checked_add(id_bytes_len)
                .and_then(|len| len.checked_add(code_bytes))
                .and_then(|len| len.checked_add(factor_bytes))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVF-RQ list size overflow")
                })?;
            current_offset = current_offset
                .checked_add(list_bytes as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVF-RQ offset overflow")
                })?;
        }
    }

    for list_id in 0..nlist {
        write_i64_le(out, list_offsets[list_id])?;
        write_i32_le(out, list_counts[list_id])?;
        write_i32_le(out, list_id_bytes_lens[list_id])?;
    }

    for sorted_list in sorted_lists {
        if sorted_list.ids.is_empty() {
            continue;
        }
        write_i64_le(out, sorted_list.ids[0])?;
        write_i32_le(out, sorted_list.id_bytes.len() as i32)?;
        out.write_all(&sorted_list.id_bytes)?;
        out.write_all(&sorted_list.codes)?;
        write_factors(out, &sorted_list.factors)?;
    }

    Ok(())
}

pub struct IVFRQIndexReader<R: SeekRead> {
    reader: R,
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub rotation_seed: u64,
    pub rotation_rounds: u32,
    pub code_size: usize,
    pub num_bits: u32,
    pub rotation_type: u32,
    pub factor_layout: u32,
    pub format_flags: u32,
    pub quantizer_centroids: Vec<f32>,
    pub list_offsets: Vec<i64>,
    pub list_counts: Vec<i32>,
    pub list_id_bytes_lens: Vec<i32>,
    quantizer: RaBitQuantizer,
    rotation: RQRotation,
    delta_ids: bool,
    loaded: bool,
}

impl<R: SeekRead> IVFRQIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        let mut cursor = PreadCursor::new(&mut reader, 0);
        let magic = read_u32_le(&mut cursor)?;
        if magic != IVF_RQ_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVF-RQ magic: 0x{:08X}", magic),
            ));
        }
        let version = read_u32_le(&mut cursor)?;
        if version != IVF_RQ_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF-RQ version: {}", version),
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
        let flags = read_u32_le(&mut cursor)?;
        let total_vectors = read_i64_le(&mut cursor)?;
        if total_vectors < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("negative total vector count: {}", total_vectors),
            ));
        }
        let rotation_seed = read_u64_le(&mut cursor)?;
        let rotation_rounds = read_u32_le(&mut cursor)?;
        let code_size = validate_positive_i32(read_i32_le(&mut cursor)?, "code_size")? as usize;
        let num_bits = read_u32_le(&mut cursor)?;
        let rotation_type = read_u32_le(&mut cursor)?;
        let factor_layout = read_u32_le(&mut cursor)?;
        let format_flags = read_u32_le(&mut cursor)?;
        let expected_code_size = d.div_ceil(8);
        if code_size != expected_code_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IVF-RQ code_size {} does not match dimension-derived size {}",
                    code_size, expected_code_size
                ),
            ));
        }
        validate_rq_format(num_bits, rotation_type, factor_layout, format_flags)?;

        let unknown_flags = flags & !SUPPORTED_FLAGS;
        if unknown_flags != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF-RQ flags: 0x{:08X}", unknown_flags),
            ));
        }
        if flags & REQUIRED_FLAGS != REQUIRED_FLAGS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-RQ v1 requires delta IDs",
            ));
        }

        Ok(Self {
            reader,
            d,
            nlist,
            metric,
            total_vectors,
            rotation_seed,
            rotation_rounds,
            code_size,
            num_bits,
            rotation_type,
            factor_layout,
            format_flags,
            quantizer_centroids: Vec::new(),
            list_offsets: Vec::new(),
            list_counts: Vec::new(),
            list_id_bytes_lens: Vec::new(),
            quantizer: RaBitQuantizer::new(d),
            rotation: RQRotation::new(d, rotation_seed, rotation_rounds),
            delta_ids: true,
            loaded: false,
        })
    }

    pub fn ensure_loaded(&mut self) -> io::Result<()> {
        if self.loaded {
            return Ok(());
        }
        let mut cursor = PreadCursor::new(&mut self.reader, IVF_RQ_HEADER_SIZE as u64);
        self.quantizer_centroids =
            read_f32_vec(&mut cursor, checked_section_size(self.nlist, self.d)?)?;
        self.list_offsets = vec![0; self.nlist];
        self.list_counts = vec![0; self.nlist];
        self.list_id_bytes_lens = vec![0; self.nlist];
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
            let id_bytes_len = read_i32_le(&mut cursor)?;
            if id_bytes_len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative id_bytes_len {} at list {}", id_bytes_len, list_id),
                ));
            }
            self.list_id_bytes_lens[list_id] = id_bytes_len;
        }
        self.loaded = true;
        Ok(())
    }

    pub fn read_inverted_list(&mut self, list_id: usize) -> io::Result<RQReadList> {
        self.ensure_loaded()?;
        if list_id >= self.nlist {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("list_id {} out of range (nlist={})", list_id, self.nlist),
            ));
        }
        let count = self.list_counts[list_id] as usize;
        if count == 0 {
            return Ok(RQReadList {
                list_id,
                ids: Vec::new(),
                codes: Vec::new(),
                factors: Vec::new(),
            });
        }
        if !self.delta_ids {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-RQ reader only supports delta IDs",
            ));
        }
        let offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
        let id_bytes_len = self.list_id_bytes_lens[list_id] as usize;
        let code_bytes = checked_list_bytes(count, self.code_size)?;
        let factor_bytes = checked_list_bytes(count, FACTOR_BYTES)?;
        let payload_len = 12usize
            .checked_add(id_bytes_len)
            .and_then(|len| len.checked_add(code_bytes))
            .and_then(|len| len.checked_add(factor_bytes))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVF-RQ list payload overflow")
            })?;
        let mut payload = vec![0u8; payload_len];
        self.reader
            .pread(&mut [ReadRequest::new(offset, &mut payload)])?;
        let base_id = i64::from_le_bytes(payload[0..8].try_into().unwrap());
        let encoded_len = i32::from_le_bytes(payload[8..12].try_into().unwrap());
        if encoded_len < 0 || encoded_len as usize != id_bytes_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-RQ id_bytes_len mismatch",
            ));
        }
        let ids = decode_delta_varint_ids(base_id, &payload[12..12 + id_bytes_len], count)?;
        let code_start = 12 + id_bytes_len;
        let factor_start = code_start + code_bytes;
        Ok(RQReadList {
            list_id,
            ids,
            codes: payload[code_start..factor_start].to_vec(),
            factors: bytes_to_factors(&payload[factor_start..])?,
        })
    }

    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.search_with_query_bits(query, k, nprobe, DEFAULT_RQ_QUERY_BITS)
    }

    pub fn search_with_query_bits(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        query_bits: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.search_with_filter(query, k, nprobe, None, query_bits)
    }

    pub fn search_with_filter(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        filter: Option<&dyn RowIdFilter>,
        query_bits: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        search_batch_ivfrq_reader_filter_with_query_bits(
            self, query, 1, k, nprobe, filter, query_bits,
        )
    }

    pub fn search_with_roaring_filter(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.search_with_filter(query, k, nprobe, Some(&filter), DEFAULT_RQ_QUERY_BITS)
    }

    pub fn search_with_roaring_filter_and_query_bits(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        roaring_filter_bytes: &[u8],
        query_bits: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.search_with_filter(query, k, nprobe, Some(&filter), query_bits)
    }
}

pub struct RQReadList {
    pub list_id: usize,
    pub ids: Vec<i64>,
    pub codes: Vec<u8>,
    pub factors: Vec<RQCodeFactors>,
}

pub fn search_batch_ivfrq_reader<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfrq_reader_with_query_bits(reader, queries, nq, k, nprobe, DEFAULT_RQ_QUERY_BITS)
}

pub fn search_batch_ivfrq_reader_with_query_bits<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    query_bits: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfrq_reader_filter_with_query_bits(
        reader, queries, nq, k, nprobe, None, query_bits,
    )
}

pub fn search_batch_ivfrq_reader_filter<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfrq_reader_filter_with_query_bits(
        reader,
        queries,
        nq,
        k,
        nprobe,
        filter,
        DEFAULT_RQ_QUERY_BITS,
    )
}

pub fn search_batch_ivfrq_reader_filter_with_query_bits<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
    query_bits: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    reader.ensure_loaded()?;
    validate_search_inputs(queries, nq, reader.d, k, nprobe)?;
    validate_query_bits(query_bits)?;

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
    for list_id in unique_lists {
        let count = reader.list_counts[list_id] as usize;
        if count == 0 {
            continue;
        }
        let read_list = reader.read_inverted_list(list_id)?;
        if read_list.list_id != list_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-RQ inverted list read returned wrong list id",
            ));
        }
        let centroid = &reader.quantizer_centroids[list_id * reader.d..(list_id + 1) * reader.d];
        for &qi in &list_to_queries[list_id] {
            let query = &processed[qi * reader.d..(qi + 1) * reader.d];
            let rotated_query_residual =
                rotated_residual(query, centroid, reader.d, &reader.rotation);
            let distance_context = reader.quantizer.prepare_distance_context_with_query_bits(
                rotated_query_residual,
                query,
                count >= RQ_BYTE_LUT_MIN_LIST_SIZE,
                query_bits,
            );
            scan_read_list(
                &read_list,
                &reader.quantizer,
                reader.code_size,
                reader.metric,
                &distance_context,
                filter,
                &mut heaps[qi],
            );
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

pub fn search_batch_ivfrq_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfrq_reader_roaring_filter_with_query_bits(
        reader,
        queries,
        nq,
        k,
        nprobe,
        roaring_filter_bytes,
        DEFAULT_RQ_QUERY_BITS,
    )
}

pub fn search_batch_ivfrq_reader_roaring_filter_with_query_bits<R: SeekRead>(
    reader: &mut IVFRQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
    query_bits: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_batch_ivfrq_reader_filter_with_query_bits(
        reader,
        queries,
        nq,
        k,
        nprobe,
        Some(&filter),
        query_bits,
    )
}

fn scan_read_list(
    read_list: &RQReadList,
    quantizer: &RaBitQuantizer,
    code_size: usize,
    metric: MetricType,
    distance_context: &RQDistanceContext,
    filter: Option<&dyn RowIdFilter>,
    heap: &mut TopKHeap,
) {
    for (local_idx, &id) in read_list.ids.iter().enumerate() {
        if filter.map(|f| !f.contains(id)).unwrap_or(false) {
            continue;
        }
        let code = &read_list.codes[local_idx * code_size..(local_idx + 1) * code_size];
        let dist = quantizer.distance_to_code_prepared(
            distance_context,
            code,
            read_list.factors[local_idx],
            metric,
        );
        heap.push(dist, id);
    }
}

fn rotated_residual(query: &[f32], centroid: &[f32], d: usize, rotation: &RQRotation) -> Vec<f32> {
    let mut residual = vec![0.0f32; d];
    fvec_madd(query, centroid, -1.0, &mut residual);
    rotation.apply(&mut residual);
    residual
}

fn validate_query_bits(query_bits: usize) -> io::Result<()> {
    if is_supported_query_bits(query_bits) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "invalid IVF-RQ query_bits {}; expected 0, 4, or 8",
                query_bits
            ),
        ))
    }
}

fn validate_rq_format(
    num_bits: u32,
    rotation_type: u32,
    factor_layout: u32,
    format_flags: u32,
) -> io::Result<()> {
    if !matches!(num_bits, 1 | 2 | 4 | 8) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported IVF-RQ num_bits {}; expected 1, 2, 4, or 8",
                num_bits
            ),
        ));
    }
    if rotation_type != IVF_RQ_ROTATION_TYPE_KAC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported IVF-RQ rotation_type {}; expected {}",
                rotation_type, IVF_RQ_ROTATION_TYPE_KAC
            ),
        ));
    }
    if factor_layout != IVF_RQ_FACTOR_LAYOUT_RABITQ_V1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported IVF-RQ factor_layout {}; expected {}",
                factor_layout, IVF_RQ_FACTOR_LAYOUT_RABITQ_V1
            ),
        ));
    }
    let unknown_format_flags = format_flags & !SUPPORTED_FORMAT_FLAGS;
    if unknown_format_flags != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported IVF-RQ format_flags: 0x{:08X}",
                unknown_format_flags
            ),
        ));
    }
    if num_bits != IVF_RQ_NUM_BITS_ONE || format_flags != CURRENT_FORMAT_FLAGS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "IVF-RQ reader currently supports only num_bits=1 without optional sections; got num_bits={}, format_flags=0x{:08X}",
                num_bits, format_flags
            ),
        ));
    }
    Ok(())
}

fn validate_index_shape(index: &IVFRQIndex) -> io::Result<()> {
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
    if index.ids.len() != index.nlist
        || index.codes.len() != index.nlist
        || index.factors.len() != index.nlist
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-RQ list storage does not match nlist",
        ));
    }
    let centroid_len = checked_section_size(index.nlist, index.d)?;
    if index.quantizer_centroids.len() != centroid_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "centroid length {} does not match nlist*d {}",
                index.quantizer_centroids.len(),
                centroid_len
            ),
        ));
    }
    let code_size = index.code_size();
    for list_id in 0..index.nlist {
        let count = index.ids[list_id].len();
        let expected_code_len = checked_list_bytes(count, code_size)?;
        if index.codes[list_id].len() != expected_code_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "list {} code length {} does not match ids*code_size {}",
                    list_id,
                    index.codes[list_id].len(),
                    expected_code_len
                ),
            ));
        }
        if index.factors[list_id].len() != count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "list {} factor count {} does not match ids {}",
                    list_id,
                    index.factors[list_id].len(),
                    count
                ),
            ));
        }
    }
    Ok(())
}

fn write_factors(out: &mut dyn SeekWrite, factors: &[RQCodeFactors]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(factors.len() * FACTOR_BYTES);
    for factor in factors {
        bytes.extend_from_slice(&factor.residual_norm_sqr.to_le_bytes());
        bytes.extend_from_slice(&factor.vector_norm_sqr.to_le_bytes());
        bytes.extend_from_slice(&factor.dp_multiplier.to_le_bytes());
    }
    out.write_all(&bytes)
}

fn bytes_to_factors(bytes: &[u8]) -> io::Result<Vec<RQCodeFactors>> {
    if !bytes.len().is_multiple_of(FACTOR_BYTES) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IVF-RQ factor section is not 12-byte aligned",
        ));
    }
    Ok(bytes
        .chunks_exact(FACTOR_BYTES)
        .map(|chunk| RQCodeFactors {
            residual_norm_sqr: f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
            vector_norm_sqr: f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            dp_multiplier: f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
        })
        .collect())
}

fn write_u32_le(out: &mut dyn SeekWrite, v: u32) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

fn write_i32_le(out: &mut dyn SeekWrite, v: i32) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

fn write_i64_le(out: &mut dyn SeekWrite, v: i64) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

fn write_u64_le(out: &mut dyn SeekWrite, v: u64) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

fn write_f32_slice(out: &mut dyn SeekWrite, data: &[f32]) -> io::Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    out.write_all(&bytes)
}

fn read_u32_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_u64_le<R: SeekRead + ?Sized>(reader: &mut PreadCursor<'_, R>) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn validate_positive_i32(val: i32, field: &str) -> io::Result<i32> {
    if val <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid header field {}: {} (must be positive)", field, val),
        ));
    }
    Ok(val)
}

fn usize_to_i32(value: usize, field: &str) -> io::Result<i32> {
    if value > i32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} exceeds i32 length limit: {}", field, value),
        ));
    }
    Ok(value as i32)
}

fn usize_to_i64(value: usize, field: &str) -> io::Result<i64> {
    if value > i64::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} exceeds i64 length limit: {}", field, value),
        ));
    }
    Ok(value as i64)
}

fn u64_to_i64(value: u64, field: &str) -> io::Result<i64> {
    if value > i64::MAX as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} exceeds i64 offset limit: {}", field, value),
        ));
    }
    Ok(value as i64)
}

const MAX_SECTION_ELEMENTS: usize = 1 << 30;

fn checked_section_size(a: usize, b: usize) -> io::Result<usize> {
    let result = a.checked_mul(b).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "section size overflow in IVF-RQ header",
        )
    })?;
    if result > MAX_SECTION_ELEMENTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "section size {} exceeds maximum {}",
                result, MAX_SECTION_ELEMENTS
            ),
        ));
    }
    Ok(result)
}

fn checked_list_offset(offset: i64, list_id: usize) -> io::Result<u64> {
    if offset < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("negative list offset {} at list {}", offset, list_id),
        ));
    }
    Ok(offset as u64)
}

fn checked_list_bytes(count: usize, bytes_per_entry: usize) -> io::Result<usize> {
    count
        .checked_mul(bytes_per_entry)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "IVF-RQ list byte size overflow"))
}

fn read_f32_vec<R: SeekRead + ?Sized>(
    reader: &mut PreadCursor<'_, R>,
    count: usize,
) -> io::Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    reader.read_exact(&mut buf)?;
    bytes_to_f32_vec(&buf)
}

fn bytes_to_f32_vec(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "f32 byte section is not 4-byte aligned",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_roaring_filter(bytes: &[u8]) -> io::Result<RoaringTreemap> {
    RoaringTreemap::deserialize_from(bytes).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid RoaringTreemap filter: {}", e),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::PosWriter;
    use std::io::Cursor;

    #[test]
    fn ivfrq_write_read_search_roundtrip() {
        let d = 8;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            })
            .collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();
        let mut index = IVFRQIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfrq_index(&index, &mut writer).unwrap();

        let mut reader = IVFRQIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, distances) = reader.search(&data[7 * d..8 * d], 5, nlist).unwrap();

        assert_eq!(labels[0], ids[7]);
        assert!(distances[0] <= 1e-4);
    }

    #[test]
    fn ivfrq_reader_search_with_query_bits() {
        let d = 16;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 25.0;
                (0..d).map(move |j| cluster + i as f32 * 0.02 + j as f32 * 0.125)
            })
            .collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();
        let mut index = IVFRQIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfrq_index(&index, &mut writer).unwrap();

        let query = &data[37 * d..38 * d];
        for query_bits in [0, 4, 8] {
            let mut reader = IVFRQIndexReader::open(Cursor::new(buf.clone())).unwrap();
            let (labels, distances) = reader
                .search_with_query_bits(query, 5, nlist, query_bits)
                .unwrap();
            assert_eq!(labels[0], ids[37], "query_bits={}", query_bits);
            if query_bits == 0 {
                assert!(distances[0] <= 1e-3);
            } else {
                assert!(distances[0].is_finite());
            }
        }
    }

    #[test]
    fn ivfrq_reader_rejects_invalid_query_bits() {
        let d = 8;
        let nlist = 1;
        let n = 8;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| (0..d).map(move |j| i as f32 + j as f32 * 0.25))
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();
        let mut index = IVFRQIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfrq_index(&index, &mut writer).unwrap();

        let mut reader = IVFRQIndexReader::open(Cursor::new(buf)).unwrap();
        let err = reader
            .search_with_query_bits(&data[0..d], 1, nlist, 7)
            .unwrap_err();
        assert!(err.to_string().contains("query_bits"));
    }

    #[test]
    fn ivfrq_header_records_current_rq_format_fields() {
        let index = tiny_ivfrq_index();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfrq_index(&index, &mut writer).unwrap();

        assert_eq!(
            u32::from_le_bytes(buf[48..52].try_into().unwrap()),
            IVF_RQ_NUM_BITS_ONE
        );
        assert_eq!(
            u32::from_le_bytes(buf[52..56].try_into().unwrap()),
            IVF_RQ_ROTATION_TYPE_KAC
        );
        assert_eq!(
            u32::from_le_bytes(buf[56..60].try_into().unwrap()),
            IVF_RQ_FACTOR_LAYOUT_RABITQ_V1
        );
        assert_eq!(u32::from_le_bytes(buf[60..64].try_into().unwrap()), 0);

        let reader = IVFRQIndexReader::open(Cursor::new(buf)).unwrap();
        assert_eq!(reader.num_bits, IVF_RQ_NUM_BITS_ONE);
        assert_eq!(reader.rotation_type, IVF_RQ_ROTATION_TYPE_KAC);
        assert_eq!(reader.factor_layout, IVF_RQ_FACTOR_LAYOUT_RABITQ_V1);
        assert_eq!(reader.format_flags, 0);
    }

    #[test]
    fn ivfrq_reader_rejects_reserved_future_format_fields() {
        let index = tiny_ivfrq_index();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfrq_index(&index, &mut writer).unwrap();

        let mut future_num_bits = buf.clone();
        future_num_bits[48..52].copy_from_slice(&2u32.to_le_bytes());
        let err = open_err(future_num_bits);
        assert!(err.to_string().contains("num_bits=1"));

        let mut unknown_rotation = buf.clone();
        unknown_rotation[52..56].copy_from_slice(&99u32.to_le_bytes());
        let err = open_err(unknown_rotation);
        assert!(err.to_string().contains("rotation_type"));

        let mut unknown_layout = buf.clone();
        unknown_layout[56..60].copy_from_slice(&99u32.to_le_bytes());
        let err = open_err(unknown_layout);
        assert!(err.to_string().contains("factor_layout"));

        let mut optional_sections = buf;
        optional_sections[60..64].copy_from_slice(&FORMAT_FLAG_EX_CODES_PRESENT.to_le_bytes());
        let err = open_err(optional_sections);
        assert!(err.to_string().contains("optional sections"));
    }

    fn open_err(buf: Vec<u8>) -> io::Error {
        match IVFRQIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("expected IVF-RQ open to fail"),
            Err(err) => err,
        }
    }

    fn tiny_ivfrq_index() -> IVFRQIndex {
        let d = 8;
        let nlist = 2;
        let data: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        ];
        let ids = vec![7, 42];
        let mut index = IVFRQIndex::new(d, nlist, MetricType::L2);
        index.train(&data, 2);
        index.add(&data, &ids, 2);
        index
    }
}
