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

use crate::distance::{preprocess_vectors, MetricType};
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
use crate::sq::ScalarQuantizer;
use crate::topk::TopKHeap;
use rayon::prelude::*;
use std::io;
use std::mem::size_of;

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
    write_f32_slice(out, &index.quantizer_centroids)?;

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
    for (list_id, list) in sorted_lists.iter().enumerate() {
        if list.order.is_empty() {
            continue;
        }
        let codes = block_sorted_sq_codes(
            &index.codes[list_id],
            &list.order,
            index.d,
            IVF_SQ_SCAN_BLOCK_SIZE,
        );
        out.write_all(&codes)?;
        write_i64_le(out, list.base_id)?;
        write_i32_le(out, usize_to_i32(list.id_bytes.len(), "delta ID section")?)?;
        out.write_all(&list.id_bytes)?;
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
}

impl<R: SeekRead> IVFSQIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        let mut header = [0u8; IVF_SQ_HEADER_SIZE];
        reader.pread(&mut [ReadRequest::new(0, &mut header)])?;
        Self::open_with_header(reader, header)
    }

    pub(crate) fn open_with_header(
        mut reader: R,
        header: [u8; IVF_SQ_HEADER_SIZE],
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
        mut consume: impl FnMut(&[i64], &[u8]),
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
            consume(&ids[row_start..row_end], &codes);
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
                        filter,
                        &mut scratch,
                        &mut heap,
                    );
                })?;
                batch_start += 1;
                continue;
            }
            let count = self.batch_read_end(&probe_indices[batch_start..])?.max(1);
            let batch_end = (batch_start + count).min(probe_indices.len());
            let lists = self.read_inverted_lists(&probe_indices[batch_start..batch_end])?;
            let centroids = &self.quantizer_centroids;
            let list_sqs = &self.list_sqs;
            let global_sq = &self.sq;
            let candidate_count = lists.iter().map(|list| list.ids.len()).sum::<usize>();
            if candidate_count >= PARALLEL_SQ_SCAN_MIN_CANDIDATES {
                let per_list_results = lists
                    .par_iter()
                    .map_init(SqScanScratch::default, |scratch, list| {
                        let mut local_heap = TopKHeap::new(k);
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
                        );
                        local_heap.into_sorted()
                    })
                    .collect::<Vec<_>>();
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
                    );
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
                        filter,
                        &mut stream_scratch,
                        &mut heaps[query_index],
                    );
                }
            })?;
            batch_start += 1;
            continue;
        }
        let count = reader.batch_read_end(&unique_lists[batch_start..])?.max(1);
        let batch_end = (batch_start + count).min(unique_lists.len());
        let loaded_lists = reader.read_inverted_lists(&unique_lists[batch_start..batch_end])?;
        let centroids = &reader.quantizer_centroids;
        let list_sqs = &reader.list_sqs;
        let global_sq = &reader.sq;
        let per_list_results = loaded_lists
            .par_iter()
            .map_init(SqScanScratch::default, |scratch, list| {
                let list_id = list.list_id;
                list_to_queries[list_id]
                    .iter()
                    .map(|&query_index| {
                        let query = &processed[query_index * d..(query_index + 1) * d];
                        let mut heap = TopKHeap::new(k);
                        scan_sq_list(
                            query,
                            list,
                            &centroids[list_id * d..(list_id + 1) * d],
                            list_sqs.get(list_id).unwrap_or(global_sq),
                            metric,
                            filter,
                            scratch,
                            &mut heap,
                        );
                        (query_index, heap.into_sorted())
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for list_results in per_list_results {
            for (query_index, results) in list_results {
                for (distance, row_id) in results {
                    heaps[query_index].push(distance, row_id);
                }
            }
        }
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

// Below this point Rayon task setup and per-list heap merging dominate the
// blocked SQ arithmetic. Production-sized lists usually cross the threshold;
// small indexes stay on the lower-overhead sequential path.
const PARALLEL_SQ_SCAN_MIN_CANDIDATES: usize = 8 * 1024;

fn scan_sq_list(
    query: &[f32],
    list: &SqListData,
    centroid: &[f32],
    sq: &ScalarQuantizer,
    metric: MetricType,
    filter: Option<&dyn RowIdFilter>,
    scratch: &mut SqScanScratch,
    heap: &mut TopKHeap,
) {
    scan_sq_rows(
        query,
        &list.ids,
        &list.codes,
        centroid,
        sq,
        metric,
        filter,
        scratch,
        heap,
    );
}

fn scan_sq_rows(
    query: &[f32],
    ids: &[i64],
    codes: &[u8],
    centroid: &[f32],
    sq: &ScalarQuantizer,
    metric: MetricType,
    filter: Option<&dyn RowIdFilter>,
    scratch: &mut SqScanScratch,
    heap: &mut TopKHeap,
) {
    sq.distances_to_blocked_codes_with_offset(
        query,
        codes,
        ids.len(),
        centroid,
        metric,
        IVF_SQ_SCAN_BLOCK_SIZE,
        &mut scratch.parameters,
        &mut scratch.distances,
    );
    for (&row_id, &distance) in ids.iter().zip(&scratch.distances) {
        if filter.map(|f| !f.contains(row_id)).unwrap_or(false) {
            continue;
        }
        if heap.should_consider(distance) {
            heap.push(distance, row_id);
        }
    }
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
    if index.quantizer_centroids.len() != checked_section_size(index.nlist, index.d)? {
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
    let mut blocked = Vec::with_capacity(row_major.len());
    for block_start in (0..order.len()).step_by(block_size) {
        let block_len = (order.len() - block_start).min(block_size);
        for dimension in 0..d {
            for lane in 0..block_len {
                let source_row = order[block_start + lane];
                blocked.push(row_major[source_row * d + dimension]);
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
    use roaring::RoaringTreemap;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;

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
    fn ivfsq_open_coalesces_resident_metadata() {
        let (index, _, _) = build_index(8, 32, 512);
        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingReader {
            inner: Cursor::new(serialized_index(&index)),
            calls: Arc::clone(&calls),
        };
        let mut reader = IVFSQIndexReader::open(source).unwrap();
        reader.optimize_for_search().unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "direct IVF-SQ open should use one header read and one resident-metadata read"
        );
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
