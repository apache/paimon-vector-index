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

//! Stable v1 storage and positional-I/O search for IVF-SQ8.

use crate::collect::{Collector, RangeCollector};
use crate::distance::{preprocess_vectors, MetricType};
use crate::index::validate_queries;
use crate::index_io_util::{
    bounded_ivf_payload_batch_end, bounded_ivf_stream_chunk_rows, bytes_to_f32_vec,
    checked_list_bytes, checked_list_offset, checked_section_size, decode_delta_varint_ids,
    decode_roaring_filter, encode_delta_varint_ids, ivf_payload_is_oversized,
    pread_batched_payloads, read_delta_varint_ids_at, u64_to_i64, usize_to_i32, usize_to_i64,
    validate_positive_i32, validate_reserved_zero, validate_search_inputs, write_f32_slice,
    write_i32_le, write_i64_le, write_u32_le,
};
use crate::io::{ReadRequest, SeekRead, SeekWrite};
use crate::ivfpq::RowIdFilter;
use crate::ivfsq::IVFSQIndex;
use crate::kmeans;
use crate::range::{
    prepare_range_queries, range_probe_lists, RangeResultBuilder, RangeSearchResult,
    VectorRangeSearchParams,
};
use crate::read_options::VectorIndexReaderOptions;
use crate::sq::ScalarQuantizer;
use crate::topk::TopKHeap;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::io;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

pub const IVF_SQ_MAGIC: u32 = 0x49565351; // "IVSQ"
pub const IVF_SQ_VERSION: u32 = 1;
pub const IVF_SQ_HEADER_SIZE: usize = 64;
pub const IVF_SQ_BITS: u32 = 8;
const FLAG_DELTA_IDS: u32 = 1 << 0;
const FLAG_BLOCKED_CODES: u32 = 1 << 1;
const REQUIRED_FLAGS: u32 = FLAG_DELTA_IDS | FLAG_BLOCKED_CODES;
const SUPPORTED_FLAGS: u32 = REQUIRED_FLAGS;
pub(crate) const IVF_SQ_SCAN_BLOCK_SIZE: usize = 32;

pub fn write_ivfsq_index(index: &IVFSQIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
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
    let sorted_lists = (0..index.nlist)
        .into_par_iter()
        .map(|list_id| build_sorted_sq_list_metadata(index, list_id))
        .collect::<io::Result<Vec<_>>>()?;

    write_u32_le(out, IVF_SQ_MAGIC)?;
    write_u32_le(out, IVF_SQ_VERSION)?;
    write_i32_le(out, usize_to_i32(index.d, "dimension")?)?;
    write_i32_le(out, usize_to_i32(index.nlist, "nlist")?)?;
    write_u32_le(out, index.metric as u32)?;
    write_i64_le(out, total_vectors)?;
    write_u32_le(out, IVF_SQ_BITS)?;
    write_u32_le(out, REQUIRED_FLAGS)?;
    let (sq_min, sq_max) = sq_global_bounds(&index.sq.mins, &index.sq.maxs);
    out.write_all(&sq_min.to_le_bytes())?;
    out.write_all(&sq_max.to_le_bytes())?;
    out.write_all(&[0u8; 20])?;

    write_f32_slice(out, &index.sq.mins)?;
    write_f32_slice(out, &index.sq.maxs)?;
    for sq in &index.list_sqs {
        write_f32_slice(out, &sq.mins)?;
        write_f32_slice(out, &sq.maxs)?;
    }
    write_f32_slice(out, index.quantizer_centroids())?;

    let offset_table_size = index.nlist.checked_mul(16).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-SQ offset table size overflow",
        )
    })?;
    let data_start = out
        .pos()
        .checked_add(offset_table_size as u64)
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "IVF-SQ data offset overflow")
        })?;
    let mut list_offsets = vec![0i64; index.nlist];
    let mut list_counts = vec![0i32; index.nlist];
    let mut list_id_bytes_lens = vec![0i32; index.nlist];
    let mut current_offset = data_start;

    for (list_id, list) in sorted_lists.iter().enumerate() {
        list_offsets[list_id] = u64_to_i64(current_offset, "list offset")?;
        list_counts[list_id] = usize_to_i32(list.order.len(), "list count")?;
        if !list.order.is_empty() {
            list_id_bytes_lens[list_id] = usize_to_i32(list.id_bytes.len(), "delta ID section")?;
            current_offset = current_offset
                .checked_add(list_payload_len(
                    list.order.len(),
                    index.code_size(),
                    list.id_bytes.len(),
                )? as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVF-SQ list offset overflow")
                })?;
        }
    }

    for list_id in 0..index.nlist {
        write_i64_le(out, list_offsets[list_id])?;
        write_i32_le(out, list_counts[list_id])?;
        write_i32_le(out, list_id_bytes_lens[list_id])?;
    }
    // Bound transposition scratch independently of the total index size. An
    // oversized individual list still uses at most one list's code buffer.
    const TRANSPOSE_BATCH_BYTES: usize = 16 * 1024 * 1024;
    let mut start = 0;
    while start < index.nlist {
        let mut end = start;
        let mut bytes = 0;
        while end < index.nlist {
            let next = index.codes[end].len();
            if end > start && next > TRANSPOSE_BATCH_BYTES.saturating_sub(bytes) {
                break;
            }
            bytes += next;
            end += 1;
        }
        let blocked = (start..end)
            .into_par_iter()
            .map(|list_id| {
                block_sorted_sq_codes(
                    &index.codes[list_id],
                    &sorted_lists[list_id].order,
                    index.d,
                    IVF_SQ_SCAN_BLOCK_SIZE,
                )
            })
            .collect::<Vec<_>>();
        for (list, codes) in sorted_lists[start..end].iter().zip(blocked) {
            if list.order.is_empty() {
                continue;
            }
            out.write_all(&codes)?;
            write_i64_le(out, list.base_id)?;
            write_i32_le(out, usize_to_i32(list.id_bytes.len(), "delta ID section")?)?;
            out.write_all(&list.id_bytes)?;
        }
        start = end;
    }
    Ok(())
}

pub struct IVFSQIndexReader<R: SeekRead> {
    reader: R,
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub sq: ScalarQuantizer,
    pub list_sqs: Vec<ScalarQuantizer>,
    pub quantizer_centroids: Vec<f32>,
    pub list_offsets: Vec<i64>,
    pub list_counts: Vec<i32>,
    pub list_id_bytes_lens: Vec<i32>,
    loaded: bool,
    list_cache: Option<SqListCache>,
}

impl<R: SeekRead> IVFSQIndexReader<R> {
    pub fn open(reader: R) -> io::Result<Self> {
        Self::open_with_options(reader, VectorIndexReaderOptions::new(0))
    }

    /// Open with a bounded cache of decoded partitions. `open` retains the
    /// uncached positional-I/O behavior for callers that manage their own cache.
    pub fn open_with_options(mut reader: R, options: VectorIndexReaderOptions) -> io::Result<Self> {
        let mut header = [0u8; IVF_SQ_HEADER_SIZE];
        reader.pread(&mut [ReadRequest::new(0, &mut header)])?;
        Self::open_with_header_and_options(reader, header, options)
    }

    pub(crate) fn open_with_header_and_options(
        mut reader: R,
        header: [u8; IVF_SQ_HEADER_SIZE],
        options: VectorIndexReaderOptions,
    ) -> io::Result<Self> {
        let read_u32 =
            |offset: usize| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
        let read_i32 =
            |offset: usize| i32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
        let read_i64 =
            |offset: usize| i64::from_le_bytes(header[offset..offset + 8].try_into().unwrap());
        let read_f32 =
            |offset: usize| f32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());

        let magic = read_u32(0);
        if magic != IVF_SQ_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVF-SQ magic: 0x{magic:08X}"),
            ));
        }
        let version = read_u32(4);
        if version != IVF_SQ_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF-SQ version: {version}"),
            ));
        }
        let d = validate_positive_i32(read_i32(8), "d")? as usize;
        let nlist = validate_positive_i32(read_i32(12), "nlist")? as usize;
        let metric_code = read_u32(16);
        let metric = MetricType::from_code(metric_code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown metric type: {metric_code}"),
            )
        })?;
        let total_vectors = read_i64(20);
        if total_vectors < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-SQ total vector count must be non-negative",
            ));
        }
        let bits = read_u32(28);
        if bits != IVF_SQ_BITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF-SQ bit width: {bits}"),
            ));
        }
        let flags = read_u32(32);
        let sq_min_summary = read_f32(36);
        let sq_max_summary = read_f32(40);
        validate_reserved_zero(&header[44..64], "IVF-SQ")?;
        let unknown_flags = flags & !SUPPORTED_FLAGS;
        if unknown_flags != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF-SQ flags: 0x{unknown_flags:08X}"),
            ));
        }
        if flags & REQUIRED_FLAGS != REQUIRED_FLAGS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-SQ v1 requires delta-varint IDs and 32-row blocked codes",
            ));
        }

        let bounds_values = checked_section_size(nlist + 1, d)?
            .checked_mul(2)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ bounds size overflow")
            })?;
        let bounds_bytes = bounds_values.checked_mul(4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-SQ bounds byte length overflow",
            )
        })?;
        let centroid_values = checked_section_size(nlist, d)?;
        let centroid_bytes = centroid_values.checked_mul(4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-SQ centroid byte length overflow",
            )
        })?;
        let offset_table_bytes = nlist.checked_mul(16).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-SQ offset table byte length overflow",
            )
        })?;
        let metadata_bytes = bounds_bytes
            .checked_add(centroid_bytes)
            .and_then(|size| size.checked_add(offset_table_bytes))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ metadata size overflow")
            })?;
        let mut metadata = vec![0u8; metadata_bytes];
        reader.pread(&mut [ReadRequest::new(IVF_SQ_HEADER_SIZE as u64, &mut metadata)])?;
        let (sq, list_sqs, mut position) = {
            let mut position = 0usize;
            let mut next_f32_section = |count: usize| -> io::Result<Vec<f32>> {
                let byte_len = count.checked_mul(4).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ f32 size overflow")
                })?;
                let end = position.checked_add(byte_len).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "IVF-SQ metadata offset overflow",
                    )
                })?;
                let values = bytes_to_f32_vec(&metadata[position..end])?;
                position = end;
                Ok(values)
            };

            let mins = next_f32_section(d)?;
            let maxs = next_f32_section(d)?;
            validate_sq_bounds(d, &mins, &maxs)?;
            let (sq_min, sq_max) = sq_global_bounds(&mins, &maxs);
            if sq_min.to_bits() != sq_min_summary.to_bits()
                || sq_max.to_bits() != sq_max_summary.to_bits()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF-SQ bounds summary does not match global SQ bounds",
                ));
            }
            let sq = ScalarQuantizer::with_dimension_bounds(d, mins, maxs);
            let mut list_sqs = Vec::with_capacity(nlist);
            for _ in 0..nlist {
                let mins = next_f32_section(d)?;
                let maxs = next_f32_section(d)?;
                validate_sq_bounds(d, &mins, &maxs)?;
                list_sqs.push(ScalarQuantizer::with_dimension_bounds(d, mins, maxs));
            }
            (sq, list_sqs, position)
        };

        let quantizer_centroids = bytes_to_f32_vec(&metadata[position..position + centroid_bytes])?;
        position += centroid_bytes;
        let offset_table = &metadata[position..];
        let mut list_offsets = vec![0; nlist];
        let mut list_counts = vec![0; nlist];
        let mut list_id_bytes_lens = vec![0; nlist];
        let mut actual_total = 0i64;
        for (list_id, entry) in offset_table.chunks_exact(16).enumerate() {
            list_offsets[list_id] = i64::from_le_bytes(entry[0..8].try_into().unwrap());
            let count = i32::from_le_bytes(entry[8..12].try_into().unwrap());
            let id_bytes_len = i32::from_le_bytes(entry[12..16].try_into().unwrap());
            if count < 0 || id_bytes_len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative IVF-SQ list metadata at list {list_id}"),
                ));
            }
            if count > 0 && id_bytes_len == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing delta ID bytes for non-empty IVF-SQ list {list_id}"),
                ));
            }
            actual_total = actual_total.checked_add(count as i64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ vector count overflow")
            })?;
            list_counts[list_id] = count;
            list_id_bytes_lens[list_id] = id_bytes_len;
        }
        if actual_total != total_vectors {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IVF-SQ header vector count {total_vectors} does not match list total {actual_total}"
                ),
            ));
        }

        let resident = size_of::<Self>()
            + quantizer_centroids.capacity() * size_of::<f32>()
            + list_offsets.capacity() * size_of::<i64>()
            + list_counts.capacity() * size_of::<i32>()
            + list_id_bytes_lens.capacity() * size_of::<i32>()
            + list_sqs.capacity() * size_of::<ScalarQuantizer>()
            + std::iter::once(&sq)
                .chain(&list_sqs)
                .map(|sq| (sq.mins.capacity() + sq.maxs.capacity()) * size_of::<f32>())
                .sum::<usize>();
        let list_cache =
            SqListCache::new(nlist, options.memory_budget_bytes.saturating_sub(resident));

        Ok(Self {
            reader,
            d,
            nlist,
            metric,
            total_vectors,
            sq,
            list_sqs,
            quantizer_centroids,
            list_offsets,
            list_counts,
            list_id_bytes_lens,
            loaded: true,
            list_cache,
        })
    }

    pub fn ensure_loaded(&mut self) -> io::Result<()> {
        debug_assert!(self.loaded);
        Ok(())
    }

    pub fn optimize_for_search(&mut self) -> io::Result<()> {
        self.ensure_loaded()
    }

    pub fn read_inverted_list(&mut self, list_id: usize) -> io::Result<(Vec<i64>, Vec<u8>)> {
        let mut lists = self.read_inverted_lists(&[list_id])?;
        let list = lists.pop().expect("one requested list has one result");
        Ok((list.ids, list.codes))
    }

    pub fn read_inverted_lists(&mut self, list_ids: &[usize]) -> io::Result<Vec<SqListData>> {
        self.ensure_loaded()?;
        let mut results = (0..list_ids.len()).map(|_| None).collect::<Vec<_>>();
        let mut metas = Vec::new();
        let mut payloads = Vec::new();
        for (input_index, &list_id) in list_ids.iter().enumerate() {
            if list_id >= self.nlist {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("list_id {list_id} out of range (nlist={})", self.nlist),
                ));
            }
            let count = self.list_counts[list_id] as usize;
            if count == 0 {
                results[input_index] = Some(SqListData {
                    list_id,
                    ids: Vec::new(),
                    codes: Vec::new(),
                });
                continue;
            }
            let id_bytes_len = self.list_id_bytes_lens[list_id] as usize;
            let payload_len = list_payload_len(count, self.d, id_bytes_len)?;
            metas.push(BatchedListRead {
                input_index,
                list_id,
                count,
                id_bytes_len,
                offset: checked_list_offset(self.list_offsets[list_id], list_id)?,
            });
            payloads.push(vec![0u8; payload_len]);
        }

        if !metas.is_empty() {
            let offsets = metas.iter().map(|meta| meta.offset).collect::<Vec<_>>();
            pread_batched_payloads(&mut self.reader, &offsets, &mut payloads)?;
            for (meta, payload) in metas.into_iter().zip(payloads) {
                let (ids, codes) =
                    decode_list_payload(payload, meta.count, meta.id_bytes_len, self.d)?;
                results[meta.input_index] = Some(SqListData {
                    list_id: meta.list_id,
                    ids,
                    codes,
                });
            }
        }
        results
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "missing batched IVF-SQ list read result",
                    )
                })
            })
            .collect()
    }

    fn read_scan_lists(&mut self, list_ids: &[usize]) -> io::Result<Vec<Arc<SqListData>>> {
        self.read_scan_lists_with_count(list_ids)
            .map(|(lists, _)| lists)
    }

    fn read_scan_lists_with_count(
        &mut self,
        list_ids: &[usize],
    ) -> io::Result<(Vec<Arc<SqListData>>, usize)> {
        if self.list_cache.is_none() {
            let lists = self.read_inverted_lists(list_ids)?;
            let reads = lists.iter().filter(|list| !list.ids.is_empty()).count();
            return Ok((lists.into_iter().map(Arc::new).collect(), reads));
        }
        let mut results = vec![None; list_ids.len()];
        let mut misses = Vec::new();
        for (position, &list_id) in list_ids.iter().enumerate() {
            if list_id >= self.nlist {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "IVF-SQ list ID out of range",
                ));
            }
            let cache = self.list_cache.as_mut().unwrap();
            if let Some(entry) = &cache.entries[list_id] {
                if entry.offset == self.list_offsets[list_id]
                    && entry.count == self.list_counts[list_id]
                    && entry.id_bytes_len == self.list_id_bytes_lens[list_id]
                    && entry.d == self.d
                {
                    results[position] = Some(Arc::clone(&entry.list));
                    continue;
                }
                // Public reader metadata may have been edited by a low-level
                // caller. Never serve a payload for a different range or shape.
                cache.remove(list_id);
                cache.order.retain(|&id| id != list_id);
            }
            misses.push((position, list_id));
        }
        let mut reads = 0;
        if !misses.is_empty() {
            let missing_ids = misses.iter().map(|&(_, id)| id).collect::<Vec<_>>();
            let loaded = self.read_inverted_lists(&missing_ids)?;
            reads = loaded.iter().filter(|list| !list.ids.is_empty()).count();
            for ((position, list_id), list) in misses.into_iter().zip(loaded) {
                let list = Arc::new(list);
                self.list_cache.as_mut().unwrap().insert(CachedSqList {
                    offset: self.list_offsets[list_id],
                    count: self.list_counts[list_id],
                    id_bytes_len: self.list_id_bytes_lens[list_id],
                    d: self.d,
                    list: Arc::clone(&list),
                });
                results[position] = Some(list);
            }
        }
        Ok((results.into_iter().map(Option::unwrap).collect(), reads))
    }

    fn batch_read_end(&self, list_ids: &[usize]) -> io::Result<usize> {
        let payload_lengths = list_ids
            .iter()
            .map(|&list_id| self.list_payload_len(list_id))
            .collect::<io::Result<Vec<_>>>()?;
        bounded_ivf_payload_batch_end(
            &payload_lengths,
            self.reader.read_capabilities().max_ranges_per_pread,
        )
    }

    fn list_payload_len(&self, list_id: usize) -> io::Result<usize> {
        if list_id >= self.nlist {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("list_id {list_id} out of range (nlist={})", self.nlist),
            ));
        }
        let count = self.list_counts[list_id] as usize;
        if count == 0 {
            Ok(0)
        } else {
            list_payload_len(count, self.d, self.list_id_bytes_lens[list_id] as usize)
        }
    }

    fn for_each_streamed_list_chunk(
        &mut self,
        list_id: usize,
        mut consume: impl FnMut(&[i64], &[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        self.ensure_loaded()?;
        let count = self.list_counts[list_id] as usize;
        let list_offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
        let code_bytes = checked_list_bytes(count, self.d)?;
        let id_offset = list_offset.checked_add(code_bytes as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ ID offset overflow")
        })?;
        let ids = read_delta_varint_ids_at(
            &mut self.reader,
            id_offset,
            count,
            self.list_id_bytes_lens[list_id] as usize,
            "IVF-SQ",
        )?;
        let retained_id_bytes = ids.len().checked_mul(size_of::<i64>()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-SQ decoded ID size overflow",
            )
        })?;
        let mut row_start = 0usize;
        while row_start < count {
            let chunk_rows = bounded_ivf_stream_chunk_rows(
                count - row_start,
                self.d,
                retained_id_bytes,
                IVF_SQ_SCAN_BLOCK_SIZE,
            )?;
            let chunk_bytes = chunk_rows.checked_mul(self.d).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ chunk size overflow")
            })?;
            let chunk_offset = list_offset
                .checked_add(row_start.checked_mul(self.d).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ chunk offset overflow")
                })? as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ chunk offset overflow")
                })?;
            let mut codes = vec![0u8; chunk_bytes];
            self.reader
                .pread(&mut [ReadRequest::new(chunk_offset, &mut codes)])?;
            let row_end = row_start + chunk_rows;
            consume(&ids[row_start..row_end], &codes)?;
            row_start = row_end;
        }
        Ok(())
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
        self.ensure_loaded()?;
        validate_search_inputs(query, 1, self.d, k, nprobe)?;
        let query = preprocess_vectors(query, 1, self.d, self.metric);
        let (probe_indices, _) = kmeans::find_topk(
            &query,
            &self.quantizer_centroids,
            self.nlist,
            self.d,
            nprobe,
        );
        let mut heap = TopKHeap::new(k);
        let d = self.d;
        let metric = self.metric;
        let mut batch_start = 0usize;
        while batch_start < probe_indices.len() {
            let first_list = probe_indices[batch_start];
            if ivf_payload_is_oversized(self.list_payload_len(first_list)?) {
                let centroid =
                    self.quantizer_centroids[first_list * d..(first_list + 1) * d].to_vec();
                let sq = self.list_sqs.get(first_list).unwrap_or(&self.sq).clone();
                let mut scratch = SqScanScratch::default();
                self.for_each_streamed_list_chunk(first_list, |ids, codes| {
                    scan_sq_rows(
                        &query,
                        ids,
                        codes,
                        &centroid,
                        &sq,
                        metric,
                        SqRowSelection::Filter(filter),
                        &mut scratch,
                        &mut heap,
                    )
                })?;
                batch_start += 1;
                continue;
            }
            let count = self.batch_read_end(&probe_indices[batch_start..])?.max(1);
            let batch_end = (batch_start + count).min(probe_indices.len());
            let lists = self.read_scan_lists(&probe_indices[batch_start..batch_end])?;
            let centroids = &self.quantizer_centroids;
            let list_sqs = &self.list_sqs;
            let global_sq = &self.sq;
            let candidate_count = lists.iter().map(|list| list.ids.len()).sum::<usize>();
            if candidate_count >= PARALLEL_SQ_SCAN_MIN_CANDIDATES {
                let first = &lists[0];
                scan_sq_list(
                    &query,
                    first,
                    &centroids[first.list_id * d..(first.list_id + 1) * d],
                    list_sqs.get(first.list_id).unwrap_or(global_sq),
                    metric,
                    filter,
                    &mut SqScanScratch::default(),
                    &mut heap,
                )?;
                let cutoff = heap.distance_limit();
                let per_list_results = lists[1..]
                    .par_iter()
                    .map_init(SqScanScratch::default, |scratch, list| {
                        let mut local_heap = TopKHeap::with_max_distance(k, cutoff);
                        let list_id = list.list_id;
                        scan_sq_list(
                            &query,
                            list,
                            &centroids[list_id * d..(list_id + 1) * d],
                            list_sqs.get(list_id).unwrap_or(global_sq),
                            metric,
                            filter,
                            scratch,
                            &mut local_heap,
                        )?;
                        Ok(local_heap.into_sorted())
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                for results in per_list_results {
                    for (distance, row_id) in results {
                        heap.push(distance, row_id);
                    }
                }
            } else {
                let mut scratch = SqScanScratch::default();
                for list in &lists {
                    let list_id = list.list_id;
                    scan_sq_list(
                        &query,
                        list,
                        &centroids[list_id * d..(list_id + 1) * d],
                        list_sqs.get(list_id).unwrap_or(global_sq),
                        metric,
                        filter,
                        &mut scratch,
                        &mut heap,
                    )?;
                }
            }
            batch_start = batch_end;
        }
        Ok(padded_results(heap, k))
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

    /// Returns every probed row whose SQ-estimated internal distance is in
    /// the half-open band. Results are unsorted, unpadded, and never truncated.
    /// Even probing every list does not guarantee membership under the original
    /// vectors' distances: scalar quantization can move a row across either cut.
    pub fn range_search(
        &mut self,
        query: &[f32],
        params: VectorRangeSearchParams,
    ) -> io::Result<RangeSearchResult> {
        self.range_search_with_filter(query, params, None)
    }

    pub fn range_search_with_filter(
        &mut self,
        query: &[f32],
        params: VectorRangeSearchParams,
        filter: Option<&dyn RowIdFilter>,
    ) -> io::Result<RangeSearchResult> {
        self.range_search_batch_with_filter(query, 1, params, filter)
    }

    /// Restricts membership to the serialized Roaring allow-list. Malformed
    /// filters are rejected even for an empty band.
    pub fn range_search_with_roaring_filter(
        &mut self,
        query: &[f32],
        params: VectorRangeSearchParams,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<RangeSearchResult> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.range_search_with_filter(query, params, Some(&filter))
    }

    /// Batched SQ-estimate range search; shared probed lists are read once.
    pub fn range_search_batch(
        &mut self,
        queries: &[f32],
        query_count: usize,
        params: VectorRangeSearchParams,
    ) -> io::Result<RangeSearchResult> {
        self.range_search_batch_with_filter(queries, query_count, params, None)
    }

    pub fn range_search_batch_with_roaring_filter(
        &mut self,
        queries: &[f32],
        query_count: usize,
        params: VectorRangeSearchParams,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<RangeSearchResult> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.range_search_batch_with_filter(queries, query_count, params, Some(&filter))
    }

    pub fn range_search_batch_with_filter(
        &mut self,
        queries: &[f32],
        query_count: usize,
        params: VectorRangeSearchParams,
        filter: Option<&dyn RowIdFilter>,
    ) -> io::Result<RangeSearchResult> {
        validate_queries(queries, query_count, self.d)?;
        if params.band().metric() != self.metric {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "band metric {:?} does not match index metric {:?}",
                    params.band().metric(),
                    self.metric
                ),
            ));
        }
        let nprobe = params.validate(self.nlist)?;
        let mut builder = RangeResultBuilder::new(query_count);
        let band = params.band();
        if band.is_empty() {
            return Ok(builder.build());
        }
        self.ensure_loaded()?;
        let dimension = self.d;
        let processed = prepare_range_queries(queries, dimension, self.metric)?;
        let queries = processed.as_ref();
        let metric = self.metric;
        let probe_lists = range_probe_lists(
            queries,
            &self.quantizer_centroids,
            dimension,
            self.nlist,
            nprobe,
            metric,
        )?;
        let mut list_to_queries = vec![Vec::new(); self.nlist];
        let mut unique_lists = Vec::new();
        for (query_index, lists) in probe_lists.iter().enumerate() {
            builder.record_lists_probed(query_index, lists.len());
            for &list_id in lists {
                if list_to_queries[list_id].is_empty() {
                    unique_lists.push(list_id);
                }
                list_to_queries[list_id].push(query_index);
            }
        }
        let mut collectors = (0..query_count)
            .map(|_| RangeCollector::new(band))
            .collect::<Vec<_>>();
        let mut scratch = SqScanScratch::default();
        let mut batch_start = 0;
        while batch_start < unique_lists.len() {
            let first_list = unique_lists[batch_start];
            if ivf_payload_is_oversized(self.list_payload_len(first_list)?) {
                let centroid = self.quantizer_centroids
                    [first_list * dimension..(first_list + 1) * dimension]
                    .to_vec();
                let sq = self.list_sqs.get(first_list).unwrap_or(&self.sq).clone();
                builder.record_list_read();
                self.for_each_streamed_list_chunk(first_list, |ids, codes| {
                    let masks = filter.map(|filter| sq_filter_masks(ids, filter));
                    let selection = SqRowSelection::from_masks(masks.as_deref());
                    let query_indices = &list_to_queries[first_list];
                    if query_indices.len() == 1 {
                        let query_index = query_indices[0];
                        scan_sq_rows(
                            &queries[query_index * dimension..(query_index + 1) * dimension],
                            ids,
                            codes,
                            &centroid,
                            &sq,
                            metric,
                            selection,
                            &mut scratch,
                            &mut collectors[query_index],
                        )?;
                    } else {
                        let mut chunk_collectors = query_indices
                            .iter()
                            .map(|&query_index| (query_index, RangeCollector::new(band)))
                            .collect::<Vec<_>>();
                        scan_sq_range_chunk(
                            queries,
                            ids,
                            codes,
                            &centroid,
                            &sq,
                            metric,
                            selection,
                            &mut scratch,
                            &mut chunk_collectors,
                        )?;
                        for (query_index, collector) in chunk_collectors {
                            collectors[query_index].merge(collector);
                        }
                    }
                    Ok(())
                })?;
                batch_start += 1;
                continue;
            }
            let count = self.batch_read_end(&unique_lists[batch_start..])?.max(1);
            let batch_end = (batch_start + count).min(unique_lists.len());
            let (lists, reads) =
                self.read_scan_lists_with_count(&unique_lists[batch_start..batch_end])?;
            for _ in 0..reads {
                builder.record_list_read();
            }
            let masks = filter.map(|filter| {
                lists
                    .iter()
                    .map(|list| sq_filter_masks(&list.ids, filter))
                    .collect::<Vec<_>>()
            });
            let candidates = lists.iter().fold(0usize, |total, list| {
                total.saturating_add(
                    list.ids
                        .len()
                        .saturating_mul(list_to_queries[list.list_id].len()),
                )
            });
            let scan_one = |query_index: usize,
                            position: usize,
                            scratch: &mut SqScanScratch,
                            collector: &mut RangeCollector| {
                let list = &lists[position];
                let list_id = list.list_id;
                let selection = SqRowSelection::from_masks(
                    masks.as_ref().map(|masks| masks[position].as_slice()),
                );
                scan_sq_rows(
                    &queries[query_index * dimension..(query_index + 1) * dimension],
                    &list.ids,
                    &list.codes,
                    &self.quantizer_centroids[list_id * dimension..(list_id + 1) * dimension],
                    self.list_sqs.get(list_id).unwrap_or(&self.sq),
                    metric,
                    selection,
                    scratch,
                    collector,
                )
            };
            if query_count == 1 && lists.len() > 1 && candidates >= PARALLEL_SQ_SCAN_MIN_CANDIDATES
            {
                let output = Mutex::new(&mut collectors[0]);
                lists.par_iter().enumerate().try_for_each_init(
                    SqScanScratch::default,
                    |scratch, (position, _)| {
                        let mut collector = RangeCollector::new(band);
                        scan_one(0, position, scratch, &mut collector)?;
                        output.lock().expect("range output lock").merge(collector);
                        Ok::<(), io::Error>(())
                    },
                )?;
            } else {
                let mut positions = vec![None; self.nlist];
                for (position, list) in lists.iter().enumerate() {
                    positions[list.list_id] = Some(position);
                }
                let scan_query =
                    |scratch: &mut SqScanScratch,
                     (query_index, collector): (usize, &mut RangeCollector)| {
                        for &list_id in &probe_lists[query_index] {
                            if let Some(position) = positions[list_id] {
                                scan_one(query_index, position, scratch, collector)?;
                            }
                        }
                        Ok::<(), io::Error>(())
                    };
                if query_count > 1 && candidates >= PARALLEL_SQ_SCAN_MIN_CANDIDATES {
                    collectors
                        .par_iter_mut()
                        .enumerate()
                        .try_for_each_init(SqScanScratch::default, scan_query)?;
                } else {
                    for query in collectors.iter_mut().enumerate() {
                        scan_query(&mut scratch, query)?;
                    }
                }
            }
            batch_start = batch_end;
        }
        for (query_index, collector) in collectors.into_iter().enumerate() {
            builder.record_scanned(query_index, collector.scanned());
            builder.record_early_abandoned(query_index, collector.early_abandoned());
            builder.take_rows(query_index, collector.into_rows());
        }
        Ok(builder.build())
    }
}

pub fn search_batch_ivfsq_reader<R: SeekRead>(
    reader: &mut IVFSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfsq_reader_filter(reader, queries, nq, k, nprobe, None)
}

pub fn search_batch_ivfsq_reader_filter<R: SeekRead>(
    reader: &mut IVFSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfsq_reader_filter_range(reader, queries, nq, k, 0, nprobe, &[], &[], filter)
}

pub(crate) fn search_batch_ivfsq_reader_filter_range<R: SeekRead>(
    reader: &mut IVFSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    probe_start: usize,
    probe_end: usize,
    seed_ids: &[i64],
    seed_distances: &[f32],
    filter: Option<&dyn RowIdFilter>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    reader.ensure_loaded()?;
    validate_search_inputs(queries, nq, reader.d, k, probe_end)?;
    if probe_start >= probe_end {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "probe range must be non-empty",
        ));
    }
    validate_batch_seed(seed_ids, seed_distances, nq, k)?;
    if nq == 1 && probe_start == 0 && seed_ids.is_empty() {
        return reader.search_with_filter(queries, k, probe_end, filter);
    }
    let processed = preprocess_vectors(queries, nq, reader.d, reader.metric);
    let (all_probe_indices, _) = kmeans::find_topk_batch(
        &processed,
        nq,
        &reader.quantizer_centroids,
        reader.nlist,
        reader.d,
        probe_end,
    );
    let mut seen = vec![false; reader.nlist];
    let mut unique_lists = Vec::new();
    for list_ids in &all_probe_indices {
        for &list_id in list_ids.iter().skip(probe_start) {
            if !seen[list_id] {
                seen[list_id] = true;
                unique_lists.push(list_id);
            }
        }
    }
    let mut list_to_queries = vec![Vec::new(); reader.nlist];
    for (query_index, list_ids) in all_probe_indices.iter().enumerate() {
        for &list_id in list_ids.iter().skip(probe_start) {
            list_to_queries[list_id].push(query_index);
        }
    }
    let d = reader.d;
    let metric = reader.metric;
    let mut heaps = (0..nq).map(|_| TopKHeap::new(k)).collect::<Vec<_>>();
    seed_heaps(&mut heaps, seed_ids, seed_distances, k);
    // Oversized-list chunks are scanned query-by-query, so one reusable
    // distance buffer is sufficient regardless of the batch width.
    let mut stream_scratch = SqScanScratch::default();
    let mut batch_start = 0usize;
    while batch_start < unique_lists.len() {
        let first_list = unique_lists[batch_start];
        if ivf_payload_is_oversized(reader.list_payload_len(first_list)?) {
            let query_indices = &list_to_queries[first_list];
            let centroid =
                reader.quantizer_centroids[first_list * d..(first_list + 1) * d].to_vec();
            let sq = reader
                .list_sqs
                .get(first_list)
                .unwrap_or(&reader.sq)
                .clone();
            reader.for_each_streamed_list_chunk(first_list, |ids, codes| {
                for &query_index in query_indices {
                    let query = &processed[query_index * d..(query_index + 1) * d];
                    scan_sq_rows(
                        query,
                        ids,
                        codes,
                        &centroid,
                        &sq,
                        metric,
                        SqRowSelection::Filter(filter),
                        &mut stream_scratch,
                        &mut heaps[query_index],
                    )?;
                }
                Ok(())
            })?;
            batch_start += 1;
            continue;
        }
        let count = reader.batch_read_end(&unique_lists[batch_start..])?.max(1);
        let batch_end = (batch_start + count).min(unique_lists.len());
        let loaded_lists = reader.read_scan_lists(&unique_lists[batch_start..batch_end])?;
        let centroids = &reader.quantizer_centroids;
        let list_sqs = &reader.list_sqs;
        let global_sq = &reader.sq;
        let mut list_positions = vec![None; reader.nlist];
        for (position, list) in loaded_lists.iter().enumerate() {
            list_positions[list.list_id] = Some(position);
        }
        // Keep a query's heap across partitions. Besides avoiding nprobe
        // allocations and merges, this carries the current cutoff into later scans.
        heaps.par_iter_mut().enumerate().try_for_each_init(
            SqScanScratch::default,
            |scratch, (query_index, heap)| {
                let query = &processed[query_index * d..(query_index + 1) * d];
                for &list_id in all_probe_indices[query_index].iter().skip(probe_start) {
                    if let Some(position) = list_positions[list_id] {
                        scan_sq_list(
                            query,
                            &loaded_lists[position],
                            &centroids[list_id * d..(list_id + 1) * d],
                            list_sqs.get(list_id).unwrap_or(global_sq),
                            metric,
                            filter,
                            scratch,
                            heap,
                        )?;
                    }
                }
                Ok::<(), io::Error>(())
            },
        )?;
        batch_start = batch_end;
    }

    let mut result_ids = Vec::with_capacity(nq * k);
    let mut result_distances = Vec::with_capacity(nq * k);
    for heap in heaps {
        let (ids, distances) = padded_results(heap, k);
        result_ids.extend(ids);
        result_distances.extend(distances);
    }
    Ok((result_ids, result_distances))
}

pub fn search_batch_ivfsq_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfsq_reader_roaring_filter_range(
        reader,
        queries,
        nq,
        k,
        0,
        nprobe,
        &[],
        &[],
        roaring_filter_bytes,
    )
}

pub(crate) fn search_batch_ivfsq_reader_roaring_filter_range<R: SeekRead>(
    reader: &mut IVFSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    probe_start: usize,
    probe_end: usize,
    seed_ids: &[i64],
    seed_distances: &[f32],
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_batch_ivfsq_reader_filter_range(
        reader,
        queries,
        nq,
        k,
        probe_start,
        probe_end,
        seed_ids,
        seed_distances,
        Some(&filter),
    )
}

fn validate_batch_seed(
    seed_ids: &[i64],
    seed_distances: &[f32],
    nq: usize,
    k: usize,
) -> io::Result<()> {
    let expected = nq
        .checked_mul(k)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "nq * k overflows usize"))?;
    if (seed_ids.is_empty() && seed_distances.is_empty())
        || (seed_ids.len() == expected && seed_distances.len() == expected)
    {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "seed result lengths must both equal nq * k",
        ))
    }
}

fn seed_heaps(heaps: &mut [TopKHeap], seed_ids: &[i64], seed_distances: &[f32], k: usize) {
    for (query_index, heap) in heaps.iter_mut().enumerate() {
        let start = query_index * k;
        for (&id, &distance) in seed_ids
            .get(start..start + k)
            .unwrap_or_default()
            .iter()
            .zip(seed_distances.get(start..start + k).unwrap_or_default())
        {
            if distance != f32::MAX {
                heap.push(distance, id);
            }
        }
    }
}

pub struct SqListData {
    pub list_id: usize,
    pub ids: Vec<i64>,
    pub codes: Vec<u8>,
}

struct CachedSqList {
    offset: i64,
    count: i32,
    id_bytes_len: i32,
    d: usize,
    list: Arc<SqListData>,
}

impl CachedSqList {
    fn retained_bytes(&self) -> usize {
        size_of::<SqListData>()
            + 2 * size_of::<usize>()
            + self.list.ids.capacity() * size_of::<i64>()
            + self.list.codes.capacity()
    }
}

/// FIFO eviction keeps cache bookkeeping O(1) without per-hit allocations.
/// Both the slot table and queue are reserved and charged to the budget up front.
struct SqListCache {
    entries: Vec<Option<CachedSqList>>,
    order: VecDeque<usize>,
    capacity_bytes: usize,
    retained_bytes: usize,
}

impl SqListCache {
    fn new(nlist: usize, budget: usize) -> Option<Self> {
        let fixed = nlist.checked_mul(size_of::<Option<CachedSqList>>() + size_of::<usize>())?;
        if budget <= fixed {
            return None;
        }
        let entries = (0..nlist).map(|_| None).collect::<Vec<_>>();
        let order = VecDeque::<usize>::with_capacity(nlist);
        let fixed = entries.capacity() * size_of::<Option<CachedSqList>>()
            + order.capacity() * size_of::<usize>();
        if budget <= fixed {
            return None;
        }
        Some(Self {
            entries,
            order,
            capacity_bytes: budget.saturating_sub(fixed),
            retained_bytes: 0,
        })
    }

    fn remove(&mut self, list_id: usize) {
        if let Some(entry) = self.entries[list_id].take() {
            self.retained_bytes -= entry.retained_bytes();
        }
    }

    fn insert(&mut self, entry: CachedSqList) {
        let bytes = entry.retained_bytes();
        let list_id = entry.list.list_id;
        if bytes > self.capacity_bytes || self.entries[list_id].is_some() {
            return;
        }
        while bytes > self.capacity_bytes - self.retained_bytes {
            let oldest = self.order.pop_front().expect("nonempty cache over budget");
            self.remove(oldest);
        }
        self.retained_bytes += bytes;
        self.order.push_back(list_id);
        self.entries[list_id] = Some(entry);
    }
}

#[derive(Clone, Copy)]
struct BatchedListRead {
    input_index: usize,
    list_id: usize,
    count: usize,
    id_bytes_len: usize,
    offset: u64,
}

#[derive(Default)]
struct SqScanScratch {
    parameters: Vec<f32>,
    distances: Vec<f32>,
}

#[derive(Clone, Copy)]
enum SqRowSelection<'a> {
    Filter(Option<&'a dyn RowIdFilter>),
    BlockMasks(&'a [u32]),
}

impl<'a> SqRowSelection<'a> {
    fn from_masks(masks: Option<&'a [u32]>) -> Self {
        masks.map(Self::BlockMasks).unwrap_or(Self::Filter(None))
    }
}

fn sq_filter_masks(ids: &[i64], filter: &dyn RowIdFilter) -> Vec<u32> {
    ids.chunks(IVF_SQ_SCAN_BLOCK_SIZE)
        .map(|block| {
            block.iter().enumerate().fold(0, |mask, (lane, &id)| {
                mask | (u32::from(filter.contains(id)) << lane)
            })
        })
        .collect()
}

// Below this point Rayon task setup and per-list heap merging dominate the
// blocked SQ arithmetic. Production-sized lists usually cross the threshold;
// small indexes stay on the lower-overhead sequential path.
const PARALLEL_SQ_SCAN_MIN_CANDIDATES: usize = 8 * 1024;

fn scan_sq_range_chunk<C: Collector + Send>(
    queries: &[f32],
    ids: &[i64],
    codes: &[u8],
    centroid: &[f32],
    sq: &ScalarQuantizer,
    metric: MetricType,
    selection: SqRowSelection<'_>,
    scratch: &mut SqScanScratch,
    collectors: &mut [(usize, C)],
) -> io::Result<()> {
    let dimension = centroid.len();
    let scan_query = |scratch: &mut SqScanScratch, (query_index, collector): &mut (usize, C)| {
        scan_sq_rows(
            &queries[*query_index * dimension..(*query_index + 1) * dimension],
            ids,
            codes,
            centroid,
            sq,
            metric,
            selection,
            scratch,
            collector,
        )
    };
    if collectors.len() > 1
        && ids.len().saturating_mul(collectors.len()) >= PARALLEL_SQ_SCAN_MIN_CANDIDATES
    {
        collectors
            .par_iter_mut()
            .try_for_each_init(SqScanScratch::default, scan_query)?;
    } else {
        for collector in collectors {
            scan_query(scratch, collector)?;
        }
    }
    Ok(())
}

fn scan_sq_list<C: Collector>(
    query: &[f32],
    list: &SqListData,
    centroid: &[f32],
    sq: &ScalarQuantizer,
    metric: MetricType,
    filter: Option<&dyn RowIdFilter>,
    scratch: &mut SqScanScratch,
    collector: &mut C,
) -> io::Result<()> {
    scan_sq_rows(
        query,
        &list.ids,
        &list.codes,
        centroid,
        sq,
        metric,
        SqRowSelection::Filter(filter),
        scratch,
        collector,
    )
}

fn scan_sq_rows<C: Collector>(
    query: &[f32],
    ids: &[i64],
    codes: &[u8],
    centroid: &[f32],
    sq: &ScalarQuantizer,
    metric: MetricType,
    selection: SqRowSelection<'_>,
    scratch: &mut SqScanScratch,
    collector: &mut C,
) -> io::Result<()> {
    if matches!(selection, SqRowSelection::BlockMasks(masks) if masks.iter().all(|&mask| mask == 0))
    {
        return Ok(());
    }
    let cutoff = collector.cutoff();
    if C::VALIDATE_COSINE_INPUTS {
        sq.distances_to_blocked_codes_with_offset_checked(
            query,
            codes,
            ids.len(),
            centroid,
            metric,
            IVF_SQ_SCAN_BLOCK_SIZE,
            cutoff,
            true,
            &mut scratch.parameters,
            &mut scratch.distances,
        );
    } else {
        sq.distances_to_blocked_codes_with_offset(
            query,
            codes,
            ids.len(),
            centroid,
            metric,
            IVF_SQ_SCAN_BLOCK_SIZE,
            cutoff,
            &mut scratch.parameters,
            &mut scratch.distances,
        );
    }
    let mut collect_row = |row_id, distance: f32| {
        if distance.is_finite() && distance >= cutoff {
            collector.note_abandoned();
        } else {
            collector.push(row_id, distance)?;
        }
        Ok::<(), io::Error>(())
    };
    match selection {
        SqRowSelection::Filter(filter) => {
            for (&row_id, &distance) in ids.iter().zip(&scratch.distances) {
                if filter
                    .map(|filter| !filter.contains(row_id))
                    .unwrap_or(false)
                {
                    continue;
                }
                collect_row(row_id, distance)?;
            }
        }
        SqRowSelection::BlockMasks(masks) => {
            for (block, &mask) in masks.iter().enumerate() {
                let mut remaining = mask;
                while remaining != 0 {
                    let position =
                        block * IVF_SQ_SCAN_BLOCK_SIZE + remaining.trailing_zeros() as usize;
                    collect_row(ids[position], scratch.distances[position])?;
                    remaining &= remaining - 1;
                }
            }
        }
    }
    Ok(())
}

fn padded_results(heap: TopKHeap, k: usize) -> (Vec<i64>, Vec<f32>) {
    let sorted = heap.into_sorted();
    let mut ids = sorted.iter().map(|&(_, id)| id).collect::<Vec<_>>();
    let mut distances = sorted
        .iter()
        .map(|&(distance, _)| distance)
        .collect::<Vec<_>>();
    ids.resize(k, -1);
    distances.resize(k, f32::MAX);
    (ids, distances)
}

fn validate_index_shape(index: &IVFSQIndex) -> io::Result<()> {
    if index.d == 0 || index.nlist == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-SQ dimension and nlist must be greater than zero",
        ));
    }
    validate_sq_bounds(index.d, &index.sq.mins, &index.sq.maxs)?;
    if index.list_sqs.len() != index.nlist
        || index.ids.len() != index.nlist
        || index.codes.len() != index.nlist
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-SQ inverted-list state does not match nlist",
        ));
    }
    if index.quantizer_centroids().len() != checked_section_size(index.nlist, index.d)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-SQ centroid storage does not match nlist * dimension",
        ));
    }
    for list_id in 0..index.nlist {
        validate_sq_bounds(
            index.d,
            &index.list_sqs[list_id].mins,
            &index.list_sqs[list_id].maxs,
        )?;
        let expected = checked_list_bytes(index.ids[list_id].len(), index.code_size())?;
        if index.codes[list_id].len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("IVF-SQ code length mismatch at list {list_id}"),
            ));
        }
    }
    Ok(())
}

fn validate_sq_bounds(d: usize, mins: &[f32], maxs: &[f32]) -> io::Result<()> {
    if mins.len() != d || maxs.len() != d {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IVF-SQ bounds length does not match dimension",
        ));
    }
    for (dimension, (&min, &max)) in mins.iter().zip(maxs).enumerate() {
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid IVF-SQ bounds at dimension {dimension}"),
            ));
        }
    }
    Ok(())
}

fn list_payload_len(count: usize, code_size: usize, id_bytes_len: usize) -> io::Result<usize> {
    12usize
        .checked_add(id_bytes_len)
        .and_then(|value| value.checked_add(count.checked_mul(code_size)?))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "IVF-SQ list size overflow"))
}

fn decode_list_payload(
    mut payload: Vec<u8>,
    count: usize,
    id_bytes_len: usize,
    code_size: usize,
) -> io::Result<(Vec<i64>, Vec<u8>)> {
    let expected = list_payload_len(count, code_size, id_bytes_len)?;
    if payload.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "IVF-SQ list payload length mismatch",
        ));
    }
    let code_bytes = checked_list_bytes(count, code_size)?;
    let id_header_end = code_bytes.checked_add(12).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "IVF-SQ list ID header offset overflow",
        )
    })?;
    let base_id = i64::from_le_bytes(payload[code_bytes..code_bytes + 8].try_into().unwrap());
    let stored_id_bytes_len =
        i32::from_le_bytes(payload[code_bytes + 8..id_header_end].try_into().unwrap());
    if stored_id_bytes_len < 0 || stored_id_bytes_len as usize != id_bytes_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IVF-SQ list ID length does not match offset table",
        ));
    }
    let ids = decode_delta_varint_ids(base_id, &payload[id_header_end..], count)?;
    payload.truncate(code_bytes);
    Ok((ids, payload))
}

fn build_sorted_sq_list_metadata(
    index: &IVFSQIndex,
    list_id: usize,
) -> io::Result<SortedSqListMetadata> {
    let count = index.ids[list_id].len();
    if count == 0 {
        return Ok(SortedSqListMetadata::default());
    }
    let mut order = (0..count).collect::<Vec<_>>();
    order.sort_by_key(|&position| index.ids[list_id][position]);
    let ids = order
        .iter()
        .map(|&position| index.ids[list_id][position])
        .collect::<Vec<_>>();
    let base_id = ids[0];
    let (_, id_bytes) = encode_delta_varint_ids(&ids);
    Ok(SortedSqListMetadata {
        base_id,
        order,
        id_bytes,
    })
}

fn block_sorted_sq_codes(
    row_major: &[u8],
    order: &[usize],
    d: usize,
    block_size: usize,
) -> Vec<u8> {
    debug_assert_eq!(row_major.len(), order.len() * d);
    let mut blocked = vec![0; row_major.len()];
    for block_start in (0..order.len()).step_by(block_size) {
        let block_len = (order.len() - block_start).min(block_size);
        let block = &mut blocked[block_start * d..(block_start + block_len) * d];
        for (dimension, column) in block.chunks_exact_mut(block_len).enumerate() {
            for (dst, &source_row) in column
                .iter_mut()
                .zip(&order[block_start..block_start + block_len])
            {
                *dst = row_major[source_row * d + dimension];
            }
        }
    }
    blocked
}

#[derive(Default)]
struct SortedSqListMetadata {
    base_id: i64,
    order: Vec<usize>,
    id_bytes: Vec<u8>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::PosWriter;
    use crate::io::ReadRequest;
    use crate::range::{Bound, DistanceBand};
    use roaring::RoaringTreemap;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn ivfsq_range_streamed_chunk_scans_queries_on_multiple_workers() {
        struct TrackingCollector<'a> {
            inner: RangeCollector,
            workers: &'a AtomicU64,
        }

        impl Collector for TrackingCollector<'_> {
            const VALIDATE_COSINE_INPUTS: bool = true;

            fn cutoff(&self) -> f32 {
                let worker = rayon::current_thread_index().unwrap();
                if self.workers.fetch_or(1 << worker, Ordering::Relaxed) == 0 {
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
                    while self.workers.load(Ordering::Relaxed).count_ones() < 2
                        && std::time::Instant::now() < deadline
                    {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
                self.inner.cutoff()
            }

            fn push(&mut self, id: i64, value: f32) -> io::Result<()> {
                self.inner.push(id, value)
            }

            fn note_abandoned(&mut self) {
                self.inner.note_abandoned();
            }
        }

        let dimension = 65;
        let count = 8_193;
        let ids = (0..count as i64).collect::<Vec<_>>();
        let codes = vec![0; count * dimension];
        let centroid = vec![0.0; dimension];
        let sq = ScalarQuantizer::with_bounds(dimension, 0.0, 1.0);
        let queries = (0..16)
            .flat_map(|query_index| vec![query_index as f32 * 0.25; dimension])
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let allowed: RoaringTreemap = (0..count as u64).filter(|id| id % 3 == 0).collect();
        let masks = sq_filter_masks(&ids, &allowed);
        for (selection, metric) in [
            SqRowSelection::Filter(None),
            SqRowSelection::BlockMasks(&masks),
        ]
        .into_iter()
        .flat_map(|selection| {
            [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct]
                .map(|metric| (selection, metric))
        }) {
            let band =
                DistanceBand::from_raw(Bound::Finite(1.0), Bound::Finite(200.0), metric).unwrap();
            let workers = AtomicU64::new(0);
            let query_indices = [14, 2, 12, 4, 10, 6, 8, 0];
            let mut collectors = query_indices
                .iter()
                .map(|&query_index| {
                    (
                        query_index,
                        TrackingCollector {
                            inner: RangeCollector::new(band),
                            workers: &workers,
                        },
                    )
                })
                .collect::<Vec<_>>();
            pool.install(|| {
                scan_sq_range_chunk(
                    &queries,
                    &ids,
                    &codes,
                    &centroid,
                    &sq,
                    metric,
                    selection,
                    &mut SqScanScratch::default(),
                    &mut collectors,
                )
                .unwrap();
            });
            assert!(
                workers.load(Ordering::Relaxed).count_ones() > 1,
                "streamed chunks must scan active queries on multiple Rayon workers"
            );
            for (query_index, collector) in collectors {
                let mut expected = RangeCollector::new(band);
                scan_sq_rows(
                    &queries[query_index * dimension..(query_index + 1) * dimension],
                    &ids,
                    &codes,
                    &centroid,
                    &sq,
                    metric,
                    selection,
                    &mut SqScanScratch::default(),
                    &mut expected,
                )
                .unwrap();
                assert_eq!(collector.inner.scanned(), expected.scanned());
                assert_eq!(
                    collector.inner.early_abandoned(),
                    expected.early_abandoned()
                );
                assert_eq!(collector.inner.into_rows(), expected.into_rows());
            }
        }
    }

    #[test]
    fn ivfsq_range_streamed_chunk_propagates_parallel_collector_failure() {
        struct FailingCollector;

        impl Collector for FailingCollector {
            fn cutoff(&self) -> f32 {
                f32::INFINITY
            }

            fn push(&mut self, _id: i64, _value: f32) -> io::Result<()> {
                Err(io::Error::other("parallel collector failed"))
            }
        }

        let ids = (0..4_097).collect::<Vec<_>>();
        let codes = vec![0; ids.len()];
        let sq = ScalarQuantizer::with_bounds(1, 0.0, 1.0);
        let error = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                scan_sq_range_chunk(
                    &[0.0, 1.0],
                    &ids,
                    &codes,
                    &[0.0],
                    &sq,
                    MetricType::L2,
                    SqRowSelection::Filter(None),
                    &mut SqScanScratch::default(),
                    &mut [(0, FailingCollector), (1, FailingCollector)],
                )
                .unwrap_err()
            });
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(error.to_string(), "parallel collector failed");
    }

    #[test]
    fn ivfsq_range_batch_evaluates_filter_once_per_list_row() {
        struct CountingFilter(AtomicUsize);

        impl RowIdFilter for CountingFilter {
            fn contains(&self, id: i64) -> bool {
                self.0.fetch_add(1, Ordering::Relaxed);
                id % 3 == 0
            }
        }

        let (index, data, ids) = build_index(37, 4, 1_024);
        let mut reader = IVFSQIndexReader::open(Cursor::new(serialized_index(&index))).unwrap();
        let filter = CountingFilter(AtomicUsize::new(0));
        let band =
            DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
        let result = reader
            .range_search_batch_with_filter(
                &data[..37 * 3],
                3,
                VectorRangeSearchParams::new(band, 4),
                Some(&filter),
            )
            .unwrap();
        assert_eq!(filter.0.load(Ordering::Relaxed), ids.len());
        let expected = ids.iter().filter(|&&id| id % 3 == 0).count();
        for query_index in 0..3 {
            assert_eq!(result.query(query_index).labels.len(), expected);
        }
    }

    #[test]
    fn ivfsq_range_cutoff_preserves_block_and_tail_membership() {
        for dimension in [1, 31, 32, 33, 64, 65, 128] {
            for count in [1, 31, 32, 33, 64, 67] {
                let sq = ScalarQuantizer::with_bounds(dimension, -1.3, 2.7);
                let query = vec![0.2; dimension];
                let centroid = vec![0.7; dimension];
                let ids = (0..count as i64).collect::<Vec<_>>();
                let codes = (0..count)
                    .flat_map(|row| {
                        (0..dimension)
                            .map(move |component| ((row * 17 + component * 7) % 256) as u8)
                    })
                    .collect::<Vec<_>>();
                let blocked = block_sorted_sq_codes(
                    &codes,
                    &(0..count).collect::<Vec<_>>(),
                    dimension,
                    IVF_SQ_SCAN_BLOCK_SIZE,
                );
                let mut full = SqScanScratch::default();
                sq.distances_to_blocked_codes_with_offset(
                    &query,
                    &blocked,
                    count,
                    &centroid,
                    MetricType::L2,
                    IVF_SQ_SCAN_BLOCK_SIZE,
                    f32::INFINITY,
                    &mut full.parameters,
                    &mut full.distances,
                );
                let allowed: RoaringTreemap = (0..count as u64).filter(|id| id % 3 == 0).collect();
                for filter in [None, Some(&allowed as &dyn RowIdFilter)] {
                    let masks = filter.map(|filter| sq_filter_masks(&ids, filter));
                    for selection in [
                        SqRowSelection::Filter(filter),
                        SqRowSelection::from_masks(masks.as_deref()),
                    ] {
                        for upper in [
                            Bound::Unbounded,
                            Bound::Finite(0.0),
                            Bound::Finite(full.distances[count / 2]),
                        ] {
                            let band =
                                DistanceBand::from_raw(Bound::Finite(0.0), upper, MetricType::L2)
                                    .unwrap();
                            let mut collector = RangeCollector::new(band);
                            scan_sq_rows(
                                &query,
                                &ids,
                                &blocked,
                                &centroid,
                                &sq,
                                MetricType::L2,
                                selection,
                                &mut SqScanScratch::default(),
                                &mut collector,
                            )
                            .unwrap();
                            let expected = ids
                                .iter()
                                .copied()
                                .zip(full.distances.iter().copied())
                                .filter(|(id, distance)| {
                                    filter.map(|filter| filter.contains(*id)).unwrap_or(true)
                                        && band.admit_raw(*distance)
                                })
                                .map(|(id, distance)| (id, distance.to_bits()))
                                .collect::<Vec<_>>();
                            let scanned = ids
                                .iter()
                                .filter(|&&id| {
                                    filter.map(|filter| filter.contains(id)).unwrap_or(true)
                                })
                                .count();
                            assert_eq!(collector.scanned(), scanned);
                            assert_eq!(collector.early_abandoned(), scanned - expected.len());
                            assert_eq!(
                                collector
                                    .into_rows()
                                    .into_iter()
                                    .map(|(id, distance)| (id, distance.to_bits()))
                                    .collect::<Vec<_>>(),
                                expected,
                                "dimension={dimension}, count={count}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ivfsq_streamed_list_propagates_consumer_failure() {
        let (index, _, _) = build_index(8, 1, 257);
        let mut reader = IVFSQIndexReader::open(Cursor::new(serialized_index(&index))).unwrap();
        let error = reader
            .for_each_streamed_list_chunk(0, |_, _| Err(io::Error::other("collector failed")))
            .unwrap_err();
        assert_eq!(error.to_string(), "collector failed");
    }

    #[test]
    fn ivfsq_partition_cache_reuses_payloads_and_keeps_filters_query_local() {
        let (index, data, _) = build_index(37, 8, 4_097);
        let bytes = serialized_index(&index);
        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingReader {
            inner: Cursor::new(bytes.clone()),
            calls: Arc::clone(&calls),
        };
        let mut cached = IVFSQIndexReader::open_with_options(
            source,
            VectorIndexReaderOptions::new(4 * 1024 * 1024),
        )
        .unwrap();
        let mut uncached = IVFSQIndexReader::open(Cursor::new(bytes)).unwrap();
        let query = &data[127 * 37..128 * 37];
        let expected = uncached.search(query, 10, 8).unwrap();
        assert_eq!(cached.search(query, 10, 8).unwrap(), expected);
        calls.store(0, Ordering::Relaxed);
        let first = cached.read_scan_lists(&[0, 1]).unwrap();
        let again = cached.read_scan_lists(&[1, 0]).unwrap();
        assert!(Arc::ptr_eq(&first[0], &again[1]));
        assert!(Arc::ptr_eq(&first[1], &again[0]));
        let filter = std::collections::HashSet::from([expected.0[3]]);
        assert_eq!(
            cached
                .search_with_filter(query, 10, 8, Some(&filter))
                .unwrap(),
            uncached
                .search_with_filter(query, 10, 8, Some(&filter))
                .unwrap()
        );
        assert_eq!(cached.search(query, 10, 8).unwrap(), expected);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "warm scans must not reenter positional I/O"
        );
    }

    #[test]
    fn ivfsq_partition_cache_is_bounded_and_evicts_without_retaining_duplicate_keys() {
        fn entry(list_id: usize, bytes: usize) -> CachedSqList {
            CachedSqList {
                offset: list_id as i64,
                count: 1,
                id_bytes_len: 1,
                d: bytes,
                list: Arc::new(SqListData {
                    list_id,
                    ids: vec![list_id as i64],
                    codes: vec![0; bytes],
                }),
            }
        }
        let fixed = 4 * (size_of::<Option<CachedSqList>>() + size_of::<usize>());
        let size = entry(0, 64).retained_bytes();
        let mut cache = SqListCache::new(4, fixed + size * 2).unwrap();
        cache.insert(entry(0, 64));
        cache.insert(entry(1, 64));
        cache.insert(entry(1, 64));
        assert_eq!(cache.order.iter().copied().collect::<Vec<_>>(), [0, 1]);
        cache.insert(entry(2, 64));
        assert!(cache.entries[0].is_none());
        assert!(cache.entries[1].is_some());
        assert!(cache.entries[2].is_some());
        assert_eq!(cache.retained_bytes, size * 2);
        cache.insert(entry(3, size * 3));
        assert!(
            cache.entries[3].is_none(),
            "an oversized partition must bypass the cache"
        );
        assert_eq!(cache.retained_bytes, size * 2);
        assert!(SqListCache::new(4, 0).is_none());
        assert!(SqListCache::new(4, fixed).is_none());
    }

    #[test]
    fn ivfsq_cache_misses_retry_after_io_failure_and_zero_budget_stays_uncached() {
        let (index, _, _) = build_index(5, 2, 65);
        let bytes = serialized_index(&index);
        let mut cached = IVFSQIndexReader::open_with_options(
            Cursor::new(bytes.clone()),
            VectorIndexReaderOptions::new(1024 * 1024),
        )
        .unwrap();
        cached.read_scan_lists(&[0]).unwrap();
        let offset = cached.list_offsets[1];
        cached.list_offsets[1] = bytes.len() as i64 + 1;
        assert!(cached.read_scan_lists(&[1]).is_err());
        assert!(cached.list_cache.as_ref().unwrap().entries[1].is_none());
        cached.list_offsets[1] = offset;
        let expected = cached.read_scan_lists(&[1]).unwrap();
        // Even a warmed entry must not hide an edited offset.
        cached.list_offsets[1] = bytes.len() as i64 + 1;
        assert!(cached.read_scan_lists(&[1]).is_err());
        cached.list_offsets[1] = offset;
        assert_eq!(
            cached.read_scan_lists(&[1]).unwrap()[0].ids,
            expected[0].ids
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingReader {
            inner: Cursor::new(bytes),
            calls: Arc::clone(&calls),
        };
        let mut uncached =
            IVFSQIndexReader::open_with_options(source, VectorIndexReaderOptions::new(0)).unwrap();
        assert!(uncached.list_cache.is_none());
        calls.store(0, Ordering::Relaxed);
        uncached.read_scan_lists(&[0, 1]).unwrap();
        uncached.read_scan_lists(&[0, 1]).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    fn build_index(d: usize, nlist: usize, n: usize) -> (IVFSQIndex, Vec<f32>, Vec<i64>) {
        let data = (0..n)
            .flat_map(|i| {
                (0..d).map(move |dimension| {
                    (i % nlist) as f32 * 100.0 + i as f32 * 0.01 + dimension as f32 * 0.1
                })
            })
            .collect::<Vec<_>>();
        let ids = (10_000..10_000 + n as i64).collect::<Vec<_>>();
        let mut index = IVFSQIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);
        (index, data, ids)
    }

    fn serialized_index(index: &IVFSQIndex) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_ivfsq_index(index, &mut PosWriter::new(&mut bytes)).unwrap();
        bytes
    }

    #[test]
    fn ivfsq_streamed_list_reader_matches_full_payload() {
        let (index, _, _) = build_index(8, 1, 257);
        let bytes = serialized_index(&index);
        let mut full_reader = IVFSQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let expected = full_reader.read_inverted_list(0).unwrap();
        let mut streamed_reader = IVFSQIndexReader::open(Cursor::new(bytes)).unwrap();
        let mut actual_ids = Vec::new();
        let mut actual_codes = Vec::new();
        streamed_reader
            .for_each_streamed_list_chunk(0, |ids, codes| {
                actual_ids.extend_from_slice(ids);
                actual_codes.extend_from_slice(codes);
                Ok(())
            })
            .unwrap();
        assert_eq!(actual_ids, expected.0);
        assert_eq!(actual_codes, expected.1);
    }

    #[test]
    fn ivfsq_write_read_search_roundtrip() {
        let (index, data, ids) = build_index(8, 4, 256);
        let mut reader = IVFSQIndexReader::open(Cursor::new(serialized_index(&index))).unwrap();
        let query_index = 23;
        let (labels, distances) = reader
            .search(&data[query_index * 8..(query_index + 1) * 8], 5, 4)
            .unwrap();
        assert_eq!(labels[0], ids[query_index]);
        assert!(distances[0].is_finite());
    }

    #[test]
    fn ivfsq_batch_matches_individual_search() {
        let (index, data, _) = build_index(8, 4, 256);
        let bytes = serialized_index(&index);
        let queries = [&data[0..8], &data[80..88]].concat();
        let mut batch_reader = IVFSQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let batch = search_batch_ivfsq_reader(&mut batch_reader, &queries, 2, 5, 4).unwrap();
        let mut single_reader = IVFSQIndexReader::open(Cursor::new(bytes)).unwrap();
        let first = single_reader.search(&queries[0..8], 5, 4).unwrap();
        let second = single_reader.search(&queries[8..16], 5, 4).unwrap();
        assert_eq!(batch.0, [first.0, second.0].concat());
        assert_eq!(batch.1, [first.1, second.1].concat());
    }

    #[test]
    fn ivfsq_single_query_batch_and_seeded_probe_ranges_match_full_search() {
        let d = 37;
        let nlist = 8;
        let (index, data, _) = build_index(d, nlist, 8_193);
        let bytes = serialized_index(&index);
        let mut reader = IVFSQIndexReader::open(Cursor::new(bytes)).unwrap();
        let query = &data[127 * d..128 * d];
        let expected = reader.search(query, 10, nlist).unwrap();
        let one = search_batch_ivfsq_reader(&mut reader, query, 1, 10, nlist).unwrap();
        assert_eq!(one, expected);
        let first =
            search_batch_ivfsq_reader_filter_range(&mut reader, query, 1, 10, 0, 3, &[], &[], None)
                .unwrap();
        let refined = search_batch_ivfsq_reader_filter_range(
            &mut reader,
            query,
            1,
            10,
            3,
            nlist,
            &first.0,
            &first.1,
            None,
        )
        .unwrap();
        // Repeated vectors can tie; verify distances and the returned IDs' scores.
        assert_eq!(refined.1, expected.1);
        for (&id, &distance) in refined.0.iter().zip(&refined.1) {
            let (ids, distances) = reader
                .search_with_filter(
                    query,
                    1,
                    nlist,
                    Some(&std::collections::HashSet::from([id])),
                )
                .unwrap();
            assert_eq!(ids, [id]);
            assert_eq!(distances, [distance]);
        }
    }

    #[test]
    fn ivfsq_large_batch_scans_queries_in_parallel_without_duplicate_reads() {
        let d = 16;
        let nlist = 8;
        let nq = 8;
        let k = 10;
        let (index, data, _) = build_index(d, nlist, 8_192);
        let bytes = serialized_index(&index);
        let queries = (0..nq)
            .flat_map(|query_index| {
                let row = query_index * 127;
                data[row * d..(row + 1) * d].iter()
            })
            .copied()
            .collect::<Vec<_>>();
        let mut expected = Vec::with_capacity(nq);
        for query in queries.chunks_exact(d) {
            let mut reader = IVFSQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
            expected.push(reader.search(query, k, nlist).unwrap());
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingReader {
            inner: Cursor::new(bytes),
            calls: Arc::clone(&calls),
        };
        let mut reader = IVFSQIndexReader::open(source).unwrap();
        calls.store(0, Ordering::Relaxed);
        let filter = ThreadTrackingFilter {
            workers: AtomicU64::new(0),
        };
        let actual = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                search_batch_ivfsq_reader_filter(&mut reader, &queries, nq, k, nlist, Some(&filter))
                    .unwrap()
            });

        for query_index in 0..nq {
            let actual_ids = &actual.0[query_index * k..(query_index + 1) * k];
            let actual_distances = &actual.1[query_index * k..(query_index + 1) * k];
            let (expected_ids, expected_distances) = &expected[query_index];
            let mut actual_pairs = actual_ids
                .iter()
                .zip(actual_distances)
                .map(|(&row_id, &distance)| (row_id, distance.to_bits()))
                .collect::<Vec<_>>();
            let mut expected_pairs = expected_ids
                .iter()
                .zip(expected_distances)
                .map(|(&row_id, &distance)| (row_id, distance.to_bits()))
                .collect::<Vec<_>>();
            actual_pairs.sort_unstable();
            expected_pairs.sort_unstable();
            assert_eq!(actual_pairs, expected_pairs);
            assert!(
                actual_distances.windows(2).all(|pair| pair[0] <= pair[1]),
                "parallel batch results must remain distance-sorted"
            );
        }
        assert!(
            filter.workers.load(Ordering::Relaxed).count_ones() > 1,
            "a large batch should scan queries on multiple Rayon workers"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "parallel scanning must not duplicate the multi-range list read"
        );
    }

    #[test]
    fn ivfsq_search_supports_roaring_filter() {
        let (index, data, ids) = build_index(4, 1, 64);
        let mut reader = IVFSQIndexReader::open(Cursor::new(serialized_index(&index))).unwrap();
        let mut filter = RoaringTreemap::new();
        filter.insert(ids[10] as u64);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();
        let (labels, _) = reader
            .search_with_roaring_filter(&data[0..4], 2, 1, &filter_bytes)
            .unwrap();
        assert_eq!(labels, vec![ids[10], -1]);
    }

    #[test]
    fn ivfsq_selected_lists_share_one_multi_range_pread() {
        let (index, data, _) = build_index(8, 8, 512);
        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingReader {
            inner: Cursor::new(serialized_index(&index)),
            calls: Arc::clone(&calls),
        };
        let mut reader = IVFSQIndexReader::open(source).unwrap();
        reader.ensure_loaded().unwrap();
        calls.store(0, Ordering::SeqCst);
        reader.search(&data[0..8], 5, 8).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn ivfsq_reader_entry_points_preserve_cache_policy_and_header_reads() {
        use crate::index::VectorIndexReader;

        let (index, data, _) = build_index(8, 8, 512);
        let bytes = serialized_index(&index);
        let query = &data[..8];
        let expected = IVFSQIndexReader::open(Cursor::new(bytes.clone()))
            .unwrap()
            .search(query, 5, 8)
            .unwrap();
        for (unified, budget, cached) in [
            (false, None, false),
            (false, Some(0), false),
            (false, Some(1), false),
            (false, Some(1024 * 1024), true),
            (true, None, true),
            (true, Some(0), false),
            (true, Some(1), false),
            (true, Some(1024 * 1024), true),
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let source = CountingReader {
                inner: Cursor::new(bytes.clone()),
                calls: Arc::clone(&calls),
            };
            let options = budget.map(VectorIndexReaderOptions::new);
            let mut reader = if unified {
                let reader = match options {
                    Some(options) => VectorIndexReader::open_with_options(source, options),
                    None => VectorIndexReader::open(source),
                }
                .unwrap();
                let VectorIndexReader::IvfSq(reader) = reader else {
                    panic!("SQ file must dispatch to the SQ reader");
                };
                reader
            } else {
                match options {
                    Some(options) => IVFSQIndexReader::open_with_options(source, options),
                    None => IVFSQIndexReader::open(source),
                }
                .unwrap()
            };
            reader.optimize_for_search().unwrap();
            assert_eq!(
                calls.swap(0, Ordering::Relaxed),
                2,
                "open should read the header and resident metadata exactly once"
            );
            assert_eq!(reader.search(query, 5, 8).unwrap(), expected);
            assert_eq!(calls.swap(0, Ordering::Relaxed), 1);
            assert_eq!(reader.search(query, 5, 8).unwrap(), expected);
            assert_eq!(
                calls.load(Ordering::Relaxed),
                usize::from(!cached),
                "cache policy for unified={unified}, budget={budget:?}"
            );
        }
    }

    #[test]
    fn ivfsq_blocked_reader_supports_all_metrics() {
        for metric in [MetricType::L2, MetricType::InnerProduct, MetricType::Cosine] {
            let d = 8;
            let nlist = 4;
            let n = 256;
            let data = (0..n * d)
                .map(|index| 1.0 + (index % 37) as f32 * 0.01)
                .collect::<Vec<_>>();
            let ids = (0..n as i64).collect::<Vec<_>>();
            let mut index = IVFSQIndex::new(d, nlist, metric);
            index.train(&data, n);
            index.add(&data, &ids, n);

            let mut expected_distances = vec![0.0; 10];
            let mut expected_ids = vec![0; 10];
            index.search(
                &data[0..d],
                1,
                10,
                nlist,
                &mut expected_distances,
                &mut expected_ids,
            );
            let mut reader = IVFSQIndexReader::open(Cursor::new(serialized_index(&index))).unwrap();
            let (actual_ids, actual_distances) = reader.search(&data[0..d], 10, nlist).unwrap();
            // The fixture repeats vectors every 37 rows. SIMD accumulation may
            // order those exactly tied row IDs differently, but must preserve
            // the same top-k membership and distance ordering.
            let mut actual_members = actual_ids.clone();
            let mut expected_members = expected_ids.clone();
            actual_members.sort_unstable();
            expected_members.sort_unstable();
            assert_eq!(actual_members, expected_members, "metric={metric:?}");
            for (actual, expected) in actual_distances.iter().zip(expected_distances) {
                assert!((actual - expected).abs() < 1e-3, "metric={metric:?}");
            }
        }
    }

    #[test]
    fn ivfsq_reader_validates_bits_flags_and_reserved_bytes() {
        let (index, _, _) = build_index(2, 1, 16);
        let bytes = serialized_index(&index);
        for (offset, value, expected) in [
            (28, 4u32, "bit width"),
            (32, 0u32, "requires delta-varint"),
            (32, REQUIRED_FLAGS | (1 << 31), "Unsupported IVF-SQ flags"),
        ] {
            let mut corrupted = bytes.clone();
            corrupted[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            let error = match IVFSQIndexReader::open(Cursor::new(corrupted)) {
                Ok(_) => panic!("corrupt header should fail"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(expected));
        }
        let mut corrupted = bytes;
        corrupted[44] = 1;
        let error = match IVFSQIndexReader::open(Cursor::new(corrupted)) {
            Ok(_) => panic!("reserved header bytes should be validated"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("reserved bytes"));
    }

    struct CountingReader {
        inner: Cursor<Vec<u8>>,
        calls: Arc<AtomicUsize>,
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

    impl SeekRead for CountingReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.pread(ranges)
        }

        fn try_clone_reader(&self) -> io::Result<Option<Self>> {
            Ok(None)
        }
    }
}
