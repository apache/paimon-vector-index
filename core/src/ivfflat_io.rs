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

use crate::distance::{
    fvec_l2sqr, fvec_l2sqr_scaled_exceeds, fvec_normalize, MetricType, QueryDistance,
};
use crate::index_io_util::{
    bounded_ivf_payload_batch_end, bounded_ivf_stream_chunk_rows, ivf_payload_is_oversized,
    pread_batched_slices, read_delta_varint_ids_at, validate_reserved_zero,
};
use crate::io::{ReadRequest, SeekRead, SeekWrite};
use crate::ivfflat::IVFFlatIndex;
use crate::ivfpq::RowIdFilter;
use crate::kmeans;
use rayon::prelude::*;
use roaring::RoaringTreemap;
use std::io;
use std::mem::{align_of, size_of};

pub const IVFFLAT_MAGIC: u32 = 0x4956464C; // "IVFL"
pub const IVFFLAT_VERSION: u32 = 1;
pub const IVFFLAT_HEADER_SIZE: usize = 64;

const FLAG_DELTA_IDS: u32 = 1 << 0;
const REQUIRED_FLAGS: u32 = FLAG_DELTA_IDS;
const SUPPORTED_FLAGS: u32 = REQUIRED_FLAGS;
const IVFFLAT_WRITE_BUFFER_SIZE: usize = 64 * 1024 * 1024;
// Raw-vector scan cost scales with both rows and dimension. Below this amount,
// Rayon scheduling and list-local heap merging outweigh the saved CPU time.
const PARALLEL_FLAT_SCAN_MIN_COMPONENTS: usize = 1024 * 1024;

pub fn write_ivfflat_index(index: &IVFFlatIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    write_ivfflat_index_with_buffer_limit(index, out, IVFFLAT_WRITE_BUFFER_SIZE)
}

fn write_ivfflat_index_with_buffer_limit(
    index: &IVFFlatIndex,
    out: &mut dyn SeekWrite,
    buffer_limit: usize,
) -> io::Result<()> {
    let d = index.d;
    let nlist = index.nlist;
    validate_index_shape(index)?;
    let bytes_per_vector = d.checked_mul(size_of::<f32>()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-FLAT bytes per vector overflow",
        )
    })?;
    let mut write_buffer = Vec::new();
    let d_i32 = usize_to_i32(d, "dimension")?;
    let nlist_i32 = usize_to_i32(nlist, "nlist")?;
    let total_vectors = index.ids.iter().try_fold(0i64, |sum, ids| {
        let count = usize_to_i64(ids.len(), "total vector count")?;
        sum.checked_add(count).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "total vector count exceeds i64 length limit",
            )
        })
    })?;

    // Keep only sort permutations and encoded IDs resident. Materializing every
    // sorted raw-vector list at once duplicates the largest part of IVF-FLAT.
    let mut sorted_lists: Vec<(Vec<usize>, Vec<u8>)> = Vec::with_capacity(nlist);
    for list_id in 0..nlist {
        let count = index.ids[list_id].len();
        if count == 0 {
            sorted_lists.push((Vec::new(), Vec::new()));
            continue;
        }

        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by_key(|&idx| index.ids[list_id][idx]);

        let sorted_ids: Vec<i64> = order.iter().map(|&idx| index.ids[list_id][idx]).collect();
        let (_, id_bytes) = encode_delta_varint_ids(&sorted_ids);
        sorted_lists.push((order, id_bytes));
    }

    write_u32_le(out, IVFFLAT_MAGIC)?;
    write_u32_le(out, IVFFLAT_VERSION)?;
    write_i32_le(out, d_i32)?;
    write_i32_le(out, nlist_i32)?;
    write_u32_le(out, index.metric as u32)?;
    write_i64_le(out, total_vectors)?;
    write_u32_le(out, FLAG_DELTA_IDS)?;
    out.write_all(&[0u8; 32])?;

    write_f32_slice(
        out,
        &index.quantizer_centroids,
        &mut write_buffer,
        buffer_limit,
    )?;

    let offset_table_size = nlist.checked_mul(16).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-FLAT offset table size overflow",
        )
    })?;
    let data_start = out
        .pos()
        .checked_add(offset_table_size as u64)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-FLAT data start offset overflow",
            )
        })?;
    let mut list_offsets = vec![0i64; nlist];
    let mut list_counts = vec![0i32; nlist];
    let mut list_id_bytes_lens = vec![0i32; nlist];
    let mut current_offset = data_start;

    for list_id in 0..nlist {
        list_offsets[list_id] = u64_to_i64(current_offset, "list offset")?;
        let count = sorted_lists[list_id].0.len();
        list_counts[list_id] = usize_to_i32(count, "list count")?;
        if count > 0 {
            let id_bytes_len = sorted_lists[list_id].1.len();
            list_id_bytes_lens[list_id] = usize_to_i32(id_bytes_len, "delta ID section")?;
            let vector_bytes = checked_list_bytes(count, bytes_per_vector)?;
            let list_bytes = 12usize
                .checked_add(id_bytes_len)
                .and_then(|len| len.checked_add(vector_bytes))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVF-FLAT list size overflow")
                })?;
            current_offset = current_offset
                .checked_add(list_bytes as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVF-FLAT offset overflow")
                })?;
        }
    }

    for list_id in 0..nlist {
        write_i64_le(out, list_offsets[list_id])?;
        write_i32_le(out, list_counts[list_id])?;
        write_i32_le(out, list_id_bytes_lens[list_id])?;
    }

    for (list_id, (order, id_bytes)) in sorted_lists.into_iter().enumerate() {
        if order.is_empty() {
            continue;
        }
        write_i64_le(out, index.ids[list_id][order[0]])?;
        write_i32_le(out, id_bytes.len() as i32)?;
        out.write_all(&id_bytes)?;
        let vectors = &index.vectors[list_id];
        write_f32_iter(
            out,
            order
                .into_iter()
                .flat_map(|idx| vectors[idx * d..(idx + 1) * d].iter()),
            &mut write_buffer,
            buffer_limit,
        )?;
    }

    Ok(())
}

pub struct IVFFlatIndexReader<R: SeekRead> {
    reader: R,
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub quantizer_centroids: Vec<f32>,
    pub list_offsets: Vec<i64>,
    pub list_counts: Vec<i32>,
    pub list_id_bytes_lens: Vec<i32>,
    delta_ids: bool,
    loaded: bool,
}

struct FlatListData {
    list_id: usize,
    ids: Vec<i64>,
    payload: AlignedFlatPayload,
}

impl FlatListData {
    fn vectors(&self) -> &[f32] {
        self.payload.vectors()
    }
}

/// Owns one v1 list payload while exposing its raw-vector suffix as aligned
/// native `f32` values on little-endian hosts.
///
/// Delta-varint IDs make the vector suffix arbitrarily aligned in the file.
/// Prefixing the read target by at most three bytes lets search scan the
/// original I/O allocation directly instead of parsing and copying every raw
/// vector into a second allocation.
struct AlignedFlatPayload {
    storage: Vec<f32>,
    read_start: usize,
    payload_len: usize,
    vector_start: usize,
    vector_len: usize,
    decoded_vectors: Option<Vec<f32>>,
}

impl AlignedFlatPayload {
    fn new(payload_len: usize, vector_start: usize, vector_len: usize) -> io::Result<Self> {
        let vector_end = vector_start.checked_add(vector_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-FLAT vector suffix overflow",
            )
        })?;
        if vector_end != payload_len || !vector_len.is_multiple_of(size_of::<f32>()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-FLAT vector suffix has an invalid shape",
            ));
        }
        let alignment = align_of::<f32>();
        let read_start = (alignment - vector_start % alignment) % alignment;
        let storage_bytes = read_start.checked_add(payload_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-FLAT aligned payload overflow",
            )
        })?;
        let storage_len = storage_bytes.div_ceil(size_of::<f32>());
        Ok(Self {
            storage: vec![0.0; storage_len],
            read_start,
            payload_len,
            vector_start,
            vector_len,
            decoded_vectors: None,
        })
    }

    fn empty() -> Self {
        Self {
            storage: Vec::new(),
            read_start: 0,
            payload_len: 0,
            vector_start: 0,
            vector_len: 0,
            decoded_vectors: None,
        }
    }

    fn read_buf_mut(&mut self) -> &mut [u8] {
        let storage_bytes = self.storage.len() * size_of::<f32>();
        // SAFETY: `storage` is initialized and exclusively borrowed. Viewing
        // its allocation as bytes is valid for the exact initialized extent.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast::<u8>(), storage_bytes)
        };
        &mut bytes[self.read_start..self.read_start + self.payload_len]
    }

    fn read_bytes(&self) -> &[u8] {
        let storage_bytes = self.storage.len() * size_of::<f32>();
        // SAFETY: `storage` remains alive and immutably borrowed for the
        // returned byte slice's lifetime.
        let bytes = unsafe {
            std::slice::from_raw_parts(self.storage.as_ptr().cast::<u8>(), storage_bytes)
        };
        &bytes[self.read_start..self.read_start + self.payload_len]
    }

    fn prepare_vectors(&mut self) -> io::Result<()> {
        #[cfg(target_endian = "big")]
        {
            self.decoded_vectors = Some(bytes_to_f32_vec(&self.read_bytes()[self.vector_start..])?);
        }
        Ok(())
    }

    fn vectors(&self) -> &[f32] {
        if let Some(decoded) = &self.decoded_vectors {
            return decoded;
        }
        let vector_bytes = &self.read_bytes()[self.vector_start..];
        debug_assert_eq!(
            vector_bytes.as_ptr().align_offset(align_of::<f32>()),
            0,
            "aligned IVF-FLAT payload must expose an aligned vector suffix"
        );
        // SAFETY: `new` verifies a whole number of f32 values and chooses
        // `read_start` so this suffix is f32-aligned. On little-endian hosts
        // the file representation is the native f32 representation. Big-endian
        // hosts populate `decoded_vectors` before this method is used.
        unsafe {
            std::slice::from_raw_parts(
                vector_bytes.as_ptr().cast::<f32>(),
                self.vector_len / size_of::<f32>(),
            )
        }
    }
}

struct FlatListRead {
    input_index: usize,
    list_id: usize,
    count: usize,
    id_bytes_len: usize,
    offset: u64,
}

impl<R: SeekRead> IVFFlatIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        let mut header = [0u8; IVFFLAT_HEADER_SIZE];
        reader.pread(&mut [ReadRequest::new(0, &mut header)])?;
        Self::open_with_header(reader, header)
    }

    pub(crate) fn open_with_header(
        reader: R,
        header: [u8; IVFFLAT_HEADER_SIZE],
    ) -> io::Result<Self> {
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != IVFFLAT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVFFLAT magic: 0x{:08X}", magic),
            ));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != IVFFLAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVFFLAT version: {}", version),
            ));
        }

        let d = validate_positive_i32(i32::from_le_bytes(header[8..12].try_into().unwrap()), "d")?
            as usize;
        let nlist = validate_positive_i32(
            i32::from_le_bytes(header[12..16].try_into().unwrap()),
            "nlist",
        )? as usize;
        let metric_code = u32::from_le_bytes(header[16..20].try_into().unwrap());
        let metric = MetricType::from_code(metric_code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown metric type: {}", metric_code),
            )
        })?;
        let total_vectors = i64::from_le_bytes(header[20..28].try_into().unwrap());
        let flags = u32::from_le_bytes(header[28..32].try_into().unwrap());
        validate_reserved_zero(&header[32..64], "IVFFLAT")?;
        let unknown_flags = flags & !SUPPORTED_FLAGS;
        if unknown_flags != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVFFLAT flags: 0x{:08X}", unknown_flags),
            ));
        }
        if flags & REQUIRED_FLAGS != REQUIRED_FLAGS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVFFLAT v1 requires delta IDs",
            ));
        }

        Ok(Self {
            reader,
            d,
            nlist,
            metric,
            total_vectors,
            quantizer_centroids: Vec::new(),
            list_offsets: Vec::new(),
            list_counts: Vec::new(),
            list_id_bytes_lens: Vec::new(),
            delta_ids: true,
            loaded: false,
        })
    }

    pub fn ensure_loaded(&mut self) -> io::Result<()> {
        if self.loaded {
            return Ok(());
        }

        let centroid_count = checked_section_size(self.nlist, self.d)?;
        let centroid_bytes = centroid_count.checked_mul(4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-FLAT centroid bytes overflow",
            )
        })?;
        let table_bytes = self.nlist.checked_mul(16).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT offset table overflow")
        })?;
        let mut metadata = vec![
            0u8;
            centroid_bytes.checked_add(table_bytes).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF-FLAT metadata size overflow",
                )
            })?
        ];
        self.reader
            .pread(&mut [ReadRequest::new(IVFFLAT_HEADER_SIZE as u64, &mut metadata)])?;
        self.quantizer_centroids = bytes_to_f32_vec(&metadata[..centroid_bytes])?;
        self.list_offsets = vec![0; self.nlist];
        self.list_counts = vec![0; self.nlist];
        self.list_id_bytes_lens = vec![0; self.nlist];
        let mut actual_total = 0i64;
        for list_id in 0..self.nlist {
            let base = centroid_bytes + list_id * 16;
            self.list_offsets[list_id] =
                i64::from_le_bytes(metadata[base..base + 8].try_into().unwrap());
            let count = i32::from_le_bytes(metadata[base + 8..base + 12].try_into().unwrap());
            if count < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative list count {} at list {}", count, list_id),
                ));
            }
            self.list_counts[list_id] = count;
            actual_total = actual_total.checked_add(count as i64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT vector count overflow")
            })?;
            let id_bytes_len =
                i32::from_le_bytes(metadata[base + 12..base + 16].try_into().unwrap());
            if id_bytes_len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative id_bytes_len {} at list {}", id_bytes_len, list_id),
                ));
            }
            if count > 0 && id_bytes_len == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing delta ID bytes for non-empty IVF-FLAT list {list_id}"),
                ));
            }
            self.list_id_bytes_lens[list_id] = id_bytes_len;
        }
        if actual_total != self.total_vectors {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IVF-FLAT header vector count {} does not match list total {actual_total}",
                    self.total_vectors
                ),
            ));
        }

        self.loaded = true;
        Ok(())
    }

    pub fn read_inverted_list(&mut self, list_id: usize) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let mut lists = self.read_inverted_lists(&[list_id])?;
        let list = lists.pop().expect("one requested list has one result");
        let vectors = list.vectors().to_vec();
        Ok((list.ids, vectors))
    }

    fn read_inverted_lists(&mut self, list_ids: &[usize]) -> io::Result<Vec<FlatListData>> {
        self.ensure_loaded()?;
        if !self.delta_ids {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-FLAT reader only supports delta IDs",
            ));
        }
        let mut results = (0..list_ids.len()).map(|_| None).collect::<Vec<_>>();
        let mut metas = Vec::new();
        let mut payloads = Vec::new();
        for (input_index, &list_id) in list_ids.iter().enumerate() {
            if list_id >= self.nlist {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("list_id {} out of range (nlist={})", list_id, self.nlist),
                ));
            }
            let count = self.list_counts[list_id] as usize;
            if count == 0 {
                results[input_index] = Some(FlatListData {
                    list_id,
                    ids: Vec::new(),
                    payload: AlignedFlatPayload::empty(),
                });
                continue;
            }
            let offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
            let vector_bytes = checked_list_bytes(count, self.d * 4)?;
            let id_bytes_len = self.list_id_bytes_lens[list_id] as usize;
            let payload_len = 12usize
                .checked_add(id_bytes_len)
                .and_then(|len| len.checked_add(vector_bytes))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT list payload overflow")
                })?;
            metas.push(FlatListRead {
                input_index,
                list_id,
                count,
                id_bytes_len,
                offset,
            });
            payloads.push(AlignedFlatPayload::new(
                payload_len,
                12 + id_bytes_len,
                vector_bytes,
            )?);
        }
        if !metas.is_empty() {
            let offsets = metas.iter().map(|meta| meta.offset).collect::<Vec<_>>();
            let mut buffers = payloads
                .iter_mut()
                .map(AlignedFlatPayload::read_buf_mut)
                .collect::<Vec<_>>();
            pread_batched_slices(&mut self.reader, &offsets, &mut buffers)?;
            drop(buffers);
            for (meta, mut payload) in metas.into_iter().zip(payloads) {
                let bytes = payload.read_bytes();
                let base_id = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
                let encoded_len = i32::from_le_bytes(bytes[8..12].try_into().unwrap());
                if encoded_len < 0 || encoded_len as usize != meta.id_bytes_len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "IVF-FLAT id_bytes_len mismatch",
                    ));
                }
                let ids = decode_delta_varint_ids(
                    base_id,
                    &bytes[12..12 + meta.id_bytes_len],
                    meta.count,
                )?;
                payload.prepare_vectors()?;
                results[meta.input_index] = Some(FlatListData {
                    list_id: meta.list_id,
                    ids,
                    payload,
                });
            }
        }
        results
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "missing batched IVF-FLAT list read result",
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
                format!("list_id {} out of range (nlist={})", list_id, self.nlist),
            ));
        }
        let count = self.list_counts[list_id] as usize;
        if count == 0 {
            return Ok(0);
        }
        12usize
            .checked_add(self.list_id_bytes_lens[list_id] as usize)
            .and_then(|len| {
                self.d
                    .checked_mul(size_of::<f32>())
                    .and_then(|row_bytes| count.checked_mul(row_bytes))
                    .and_then(|vector_bytes| len.checked_add(vector_bytes))
            })
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT list payload overflow")
            })
    }

    fn for_each_streamed_list_chunk(
        &mut self,
        list_id: usize,
        mut consume: impl FnMut(&[i64], &[f32]),
    ) -> io::Result<()> {
        self.ensure_loaded()?;
        let count = self.list_counts[list_id] as usize;
        let list_offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
        let id_bytes_len = self.list_id_bytes_lens[list_id] as usize;
        let ids = read_delta_varint_ids_at(
            &mut self.reader,
            list_offset,
            count,
            id_bytes_len,
            "IVF-FLAT",
        )?;
        let row_bytes = self.d.checked_mul(size_of::<f32>()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT row size overflow")
        })?;
        let vector_offset = list_offset
            .checked_add((12usize + id_bytes_len) as u64)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF-FLAT vector offset overflow",
                )
            })?;
        let retained_id_bytes = ids.len().checked_mul(size_of::<i64>()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-FLAT decoded ID size overflow",
            )
        })?;
        let mut row_start = 0usize;
        while row_start < count {
            let chunk_rows =
                bounded_ivf_stream_chunk_rows(count - row_start, row_bytes, retained_id_bytes, 1)?;
            let vector_bytes = chunk_rows.checked_mul(row_bytes).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT chunk size overflow")
            })?;
            let mut payload = AlignedFlatPayload::new(vector_bytes, 0, vector_bytes)?;
            let chunk_offset = vector_offset
                .checked_add(row_start.checked_mul(row_bytes).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT chunk offset overflow")
                })? as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT chunk offset overflow")
                })?;
            self.reader
                .pread(&mut [ReadRequest::new(chunk_offset, payload.read_buf_mut())])?;
            payload.prepare_vectors()?;
            let row_end = row_start + chunk_rows;
            consume(&ids[row_start..row_end], payload.vectors());
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
        if query.len() != self.d {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "query length {} does not match index dimension {}",
                    query.len(),
                    self.d
                ),
            ));
        }
        if k == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "k must be greater than 0",
            ));
        }
        if nprobe == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "nprobe must be greater than 0",
            ));
        }

        let mut q = query.to_vec();
        if self.metric == MetricType::Cosine {
            fvec_normalize(&mut q);
        }

        let (probe_indices, _) =
            kmeans::find_topk(&q, &self.quantizer_centroids, self.nlist, self.d, nprobe);
        let mut heap = ReaderTopKHeap::new(k);
        let mut batch_start = 0usize;
        while batch_start < probe_indices.len() {
            let first_list = probe_indices[batch_start];
            if ivf_payload_is_oversized(self.list_payload_len(first_list)?) {
                let metric = self.metric;
                let d = self.d;
                self.for_each_streamed_list_chunk(first_list, |ids, vectors| {
                    scan_flat_rows(&q, ids, vectors, d, metric, filter, &mut heap);
                })?;
                batch_start += 1;
                continue;
            }
            let count = self.batch_read_end(&probe_indices[batch_start..])?.max(1);
            let batch_end = (batch_start + count).min(probe_indices.len());
            let lists = self.read_inverted_lists(&probe_indices[batch_start..batch_end])?;
            let scan_components = lists
                .iter()
                .map(|list| list.ids.len())
                .sum::<usize>()
                .saturating_mul(self.d);
            if lists.len() > 1 && scan_components >= PARALLEL_FLAT_SCAN_MIN_COMPONENTS {
                let per_list_results = lists
                    .par_iter()
                    .map(|list| {
                        let mut local_heap = ReaderTopKHeap::new(k);
                        scan_flat_list(&q, list, self.d, self.metric, filter, &mut local_heap);
                        local_heap.into_sorted()
                    })
                    .collect::<Vec<_>>();
                for results in per_list_results {
                    merge_flat_results(&mut heap, results);
                }
            } else {
                for list in &lists {
                    scan_flat_list(&q, list, self.d, self.metric, filter, &mut heap);
                }
            }
            batch_start = batch_end;
        }

        Ok(padded_flat_results(heap, k))
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

/// Batch search for IVF-FLAT readers. Each unique probed list is read once and
/// scanned for all queries that selected it.
pub fn search_batch_ivfflat_reader<R: SeekRead>(
    reader: &mut IVFFlatIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfflat_reader_filter(reader, queries, nq, k, nprobe, None)
}

pub fn search_batch_ivfflat_reader_filter<R: SeekRead>(
    reader: &mut IVFFlatIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    filter: Option<&dyn RowIdFilter>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    reader.ensure_loaded()?;
    let d = reader.d;
    let metric = reader.metric;
    if nq == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "nq must be greater than 0",
        ));
    }
    let expected_query_len = nq.checked_mul(d).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "nq * dimension overflows usize",
        )
    })?;
    if queries.len() != expected_query_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "queries length {} does not match nq * dimension {}",
                queries.len(),
                expected_query_len
            ),
        ));
    }
    if k == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "k must be greater than 0",
        ));
    }
    if nprobe == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "nprobe must be greater than 0",
        ));
    }

    let mut processed = queries[..expected_query_len].to_vec();
    if reader.metric == MetricType::Cosine {
        for qi in 0..nq {
            fvec_normalize(&mut processed[qi * d..(qi + 1) * d]);
        }
    }

    let (all_probe_indices, _) = kmeans::find_topk_batch(
        &processed,
        nq,
        &reader.quantizer_centroids,
        reader.nlist,
        d,
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

    let mut heaps: Vec<ReaderTopKHeap> = (0..nq).map(|_| ReaderTopKHeap::new(k)).collect();
    let mut batch_start = 0usize;
    while batch_start < unique_lists.len() {
        let first_list = unique_lists[batch_start];
        if ivf_payload_is_oversized(reader.list_payload_len(first_list)?) {
            let query_indices = &list_to_queries[first_list];
            reader.for_each_streamed_list_chunk(first_list, |ids, vectors| {
                for &qi in query_indices {
                    let query = &processed[qi * d..(qi + 1) * d];
                    scan_flat_rows(query, ids, vectors, d, metric, filter, &mut heaps[qi]);
                }
            })?;
            batch_start += 1;
            continue;
        }
        let count = reader.batch_read_end(&unique_lists[batch_start..])?.max(1);
        let batch_end = (batch_start + count).min(unique_lists.len());
        let loaded_lists = reader.read_inverted_lists(&unique_lists[batch_start..batch_end])?;
        let scan_components = loaded_lists
            .iter()
            .map(|list| {
                list.ids
                    .len()
                    .saturating_mul(list_to_queries[list.list_id].len())
            })
            .sum::<usize>()
            .saturating_mul(d);
        if loaded_lists.len() > 1 && scan_components >= PARALLEL_FLAT_SCAN_MIN_COMPONENTS {
            let per_list_results = loaded_lists
                .par_iter()
                .map(|list| {
                    let list_id = list.list_id;
                    list_to_queries[list_id]
                        .iter()
                        .map(|&qi| {
                            let query = &processed[qi * d..(qi + 1) * d];
                            let mut local_heap = ReaderTopKHeap::new(k);
                            scan_flat_list(query, list, d, reader.metric, filter, &mut local_heap);
                            (qi, local_heap.into_sorted())
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for list_results in per_list_results {
                for (qi, results) in list_results {
                    merge_flat_results(&mut heaps[qi], results);
                }
            }
        } else {
            for list in &loaded_lists {
                let list_id = list.list_id;
                for &qi in &list_to_queries[list_id] {
                    let query = &processed[qi * d..(qi + 1) * d];
                    scan_flat_list(query, list, d, reader.metric, filter, &mut heaps[qi]);
                }
            }
        }
        batch_start = batch_end;
    }

    let mut result_ids = vec![-1i64; nq * k];
    let mut result_dists = vec![f32::MAX; nq * k];
    for (qi, heap) in heaps.into_iter().enumerate() {
        let sorted = heap.into_sorted();
        let base = qi * k;
        for (i, &(dist, id)) in sorted.iter().enumerate() {
            result_ids[base + i] = id;
            result_dists[base + i] = dist;
        }
    }

    Ok((result_ids, result_dists))
}

pub fn search_batch_ivfflat_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFFlatIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_batch_ivfflat_reader_filter(reader, queries, nq, k, nprobe, Some(&filter))
}

fn scan_flat_list(
    query: &[f32],
    list: &FlatListData,
    d: usize,
    metric: MetricType,
    filter: Option<&dyn RowIdFilter>,
    heap: &mut ReaderTopKHeap,
) {
    scan_flat_rows(query, &list.ids, list.vectors(), d, metric, filter, heap);
}

fn scan_flat_rows(
    query: &[f32],
    ids: &[i64],
    vectors: &[f32],
    d: usize,
    metric: MetricType,
    filter: Option<&dyn RowIdFilter>,
    heap: &mut ReaderTopKHeap,
) {
    // In cosine mode this caches the query norm once per list instead of
    // recomputing it for every candidate vector.
    let distance_context = QueryDistance::new(query, metric);
    for (local_idx, &id) in ids.iter().enumerate() {
        if filter.is_some_and(|value| !value.contains(id)) {
            continue;
        }
        let vector = &vectors[local_idx * d..(local_idx + 1) * d];
        let distance = if metric == MetricType::L2 {
            if let Some(threshold) = heap.worst_distance() {
                if fvec_l2sqr_scaled_exceeds(query, vector, 1.0, threshold) {
                    continue;
                }
            }
            fvec_l2sqr(query, vector)
        } else {
            distance_context.distance_to(vector, None)
        };
        heap.push(distance, id);
    }
}

fn merge_flat_results(heap: &mut ReaderTopKHeap, results: Vec<(f32, i64)>) {
    for (distance, id) in results {
        heap.push(distance, id);
    }
}

fn padded_flat_results(heap: ReaderTopKHeap, k: usize) -> (Vec<i64>, Vec<f32>) {
    let sorted = heap.into_sorted();
    let mut labels = sorted.iter().map(|&(_, id)| id).collect::<Vec<_>>();
    let mut distances = sorted
        .iter()
        .map(|&(distance, _)| distance)
        .collect::<Vec<_>>();
    labels.resize(k, -1);
    distances.resize(k, f32::MAX);
    (labels, distances)
}

struct ReaderTopKHeap {
    k: usize,
    data: Vec<(f32, i64)>,
    worst_index: Option<usize>,
}

impl ReaderTopKHeap {
    fn new(k: usize) -> Self {
        Self {
            k,
            data: Vec::with_capacity(k),
            worst_index: None,
        }
    }

    #[inline]
    fn worst_distance(&self) -> Option<f32> {
        self.worst_index.map(|index| self.data[index].0)
    }

    #[inline]
    fn push(&mut self, dist: f32, id: i64) {
        if self.k == 0 {
            return;
        }
        if self.data.len() < self.k {
            self.data.push((dist, id));
            if self.data.len() == self.k {
                self.refresh_worst();
            }
            return;
        }
        let worst_index = self
            .worst_index
            .expect("a full IVF-FLAT top-k heap has a worst entry");
        if dist < self.data[worst_index].0 {
            self.data[worst_index] = (dist, id);
            self.refresh_worst();
        }
    }

    fn refresh_worst(&mut self) {
        self.worst_index = self
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
            .map(|(index, _)| index);
    }

    fn into_sorted(mut self) -> Vec<(f32, i64)> {
        self.data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.data
    }
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

fn write_f32_slice(
    out: &mut dyn SeekWrite,
    data: &[f32],
    buffer: &mut Vec<u8>,
    buffer_limit: usize,
) -> io::Result<()> {
    write_f32_iter(out, data.iter(), buffer, buffer_limit)
}

fn write_f32_iter<'a>(
    out: &mut dyn SeekWrite,
    data: impl Iterator<Item = &'a f32>,
    buffer: &mut Vec<u8>,
    buffer_limit: usize,
) -> io::Result<()> {
    let buffer_limit = buffer_limit.max(1);
    buffer.clear();
    for value in data {
        let bytes = value.to_le_bytes();
        let mut offset = 0;
        while offset < bytes.len() {
            let len = (buffer_limit - buffer.len()).min(bytes.len() - offset);
            buffer.extend_from_slice(&bytes[offset..offset + len]);
            offset += len;
            if buffer.len() == buffer_limit {
                out.write_all(buffer)?;
                buffer.clear();
            }
        }
    }
    if !buffer.is_empty() {
        out.write_all(buffer)?;
        buffer.clear();
    }
    Ok(())
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

fn validate_index_shape(index: &IVFFlatIndex) -> io::Result<()> {
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
    if index.ids.len() != index.nlist || index.vectors.len() != index.nlist {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-FLAT list storage does not match nlist",
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
    for list_id in 0..index.nlist {
        let expected_vector_len =
            index.ids[list_id]
                .len()
                .checked_mul(index.d)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "IVF-FLAT vector length overflow",
                    )
                })?;
        if index.vectors[list_id].len() != expected_vector_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "list {} vector length {} does not match ids*d {}",
                    list_id,
                    index.vectors[list_id].len(),
                    expected_vector_len
                ),
            ));
        }
    }
    Ok(())
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
            "section size overflow in IVF-FLAT header",
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
    count.checked_mul(bytes_per_entry).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "IVF-FLAT list byte size overflow",
        )
    })
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

fn encode_varint(mut val: u64, buf: &mut Vec<u8>) {
    while val >= 0x80 {
        buf.push((val as u8) | 0x80);
        val >>= 7;
    }
    buf.push(val as u8);
}

fn decode_varint(buf: &[u8], pos: &mut usize) -> io::Result<u64> {
    let mut val = 0u64;
    let mut shift = 0u32;
    loop {
        if *pos >= buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated varint",
            ));
        }
        let b = buf[*pos] as u64;
        *pos += 1;
        let payload = b & 0x7F;
        if shift == 63 && payload > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint exceeds u64 range",
            ));
        }
        val |= payload << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint exceeds 64 bits",
            ));
        }
    }
    Ok(val)
}

fn encode_delta_varint_ids(ids: &[i64]) -> (i64, Vec<u8>) {
    if ids.is_empty() {
        return (0, Vec::new());
    }
    let base = ids[0];
    let mut buf = Vec::with_capacity(ids.len() * 2);
    let mut prev = base;
    for &id in ids {
        let delta = (id as u64).wrapping_sub(prev as u64);
        encode_varint(delta, &mut buf);
        prev = id;
    }
    (base, buf)
}

fn decode_delta_varint_ids(base: i64, buf: &[u8], count: usize) -> io::Result<Vec<i64>> {
    let mut ids = Vec::with_capacity(count);
    let mut pos = 0;
    let mut current = base as u64;
    let mut prev_signed = base;
    for _ in 0..count {
        let delta = decode_varint(buf, &mut pos)?;
        current = current.wrapping_add(delta);
        let id = current as i64;
        if id < prev_signed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "decoded ID sequence is not monotonically non-decreasing",
            ));
        }
        prev_signed = id;
        ids.push(id);
    }
    Ok(ids)
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
    use crate::distance::MetricType;
    use crate::io::{PosWriter, ReadRequest};
    use crate::ivfflat::IVFFlatIndex;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;

    fn balanced_flat_index(d: usize, nlist: usize, rows_per_list: usize) -> IVFFlatIndex {
        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.quantizer_centroids = (0..nlist)
            .flat_map(|list_id| {
                (0..d).map(move |dimension| list_id as f32 * 10.0 + dimension as f32 * 0.01)
            })
            .collect();
        for list_id in 0..nlist {
            index.ids[list_id] = (0..rows_per_list)
                .map(|row| (list_id * rows_per_list + row) as i64)
                .collect();
            index.vectors[list_id] = (0..rows_per_list)
                .flat_map(|row| {
                    (0..d).map(move |dimension| {
                        list_id as f32 * 10.0 + row as f32 * 0.001 + dimension as f32 * 0.01
                    })
                })
                .collect();
        }
        index
    }

    fn serialized_flat_index(index: &IVFFlatIndex) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_ivfflat_index(index, &mut PosWriter::new(&mut bytes)).unwrap();
        bytes
    }

    struct MaxWriteWriter {
        bytes: Vec<u8>,
        max_write: usize,
    }

    impl SeekWrite for MaxWriteWriter {
        fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
            self.max_write = self.max_write.max(buf.len());
            self.bytes.extend_from_slice(buf);
            Ok(())
        }

        fn pos(&self) -> u64 {
            self.bytes.len() as u64
        }
    }

    #[test]
    fn ivfflat_chunked_writer_preserves_format_and_bounds() {
        const TEST_BUDGET: usize = 64;

        let mut index = IVFFlatIndex::new(3, 1, MetricType::L2);
        index.quantizer_centroids = vec![1.0, 2.0, 3.0];
        index.ids[0] = vec![50, 10, 40, 20, 30, 60, 0];
        index.vectors[0] = index.ids[0]
            .iter()
            .flat_map(|&id| [id as f32, id as f32 + 0.25, id as f32 + 0.5])
            .collect();

        let expected_bytes = serialized_flat_index(&index);
        let mut chunked = MaxWriteWriter {
            bytes: Vec::new(),
            max_write: 0,
        };
        write_ivfflat_index_with_buffer_limit(&index, &mut chunked, TEST_BUDGET).unwrap();

        assert_eq!(chunked.bytes, expected_bytes);
        assert!(chunked.max_write <= TEST_BUDGET);

        let mut reader = IVFFlatIndexReader::open(Cursor::new(chunked.bytes)).unwrap();
        let (ids, vectors) = reader.read_inverted_list(0).unwrap();
        assert_eq!(ids, vec![0, 10, 20, 30, 40, 50, 60]);
        assert_eq!(
            vectors,
            ids.iter()
                .flat_map(|&id| [id as f32, id as f32 + 0.25, id as f32 + 0.5])
                .collect::<Vec<_>>()
        );

        let wide_dimension = TEST_BUDGET / size_of::<f32>() + 1;
        let mut wide_index = IVFFlatIndex::new(wide_dimension, 1, MetricType::L2);
        wide_index.quantizer_centroids = vec![0.0; wide_dimension];
        wide_index.ids[0] = vec![1];
        wide_index.vectors[0] = vec![1.0; wide_dimension];
        let mut wide = MaxWriteWriter {
            bytes: Vec::new(),
            max_write: 0,
        };
        write_ivfflat_index_with_buffer_limit(&wide_index, &mut wide, TEST_BUDGET).unwrap();
        assert!(wide.max_write <= TEST_BUDGET);
    }

    #[test]
    fn ivfflat_streamed_list_reader_matches_full_payload() {
        let index = balanced_flat_index(8, 1, 257);
        let bytes = serialized_flat_index(&index);
        let mut full_reader = IVFFlatIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let expected = full_reader.read_inverted_list(0).unwrap();
        let mut streamed_reader = IVFFlatIndexReader::open(Cursor::new(bytes)).unwrap();
        let mut actual_ids = Vec::new();
        let mut actual_vectors = Vec::new();
        streamed_reader
            .for_each_streamed_list_chunk(0, |ids, vectors| {
                actual_ids.extend_from_slice(ids);
                actual_vectors.extend_from_slice(vectors);
            })
            .unwrap();
        assert_eq!(actual_ids, expected.0);
        assert_eq!(actual_vectors, expected.1);
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

    #[test]
    fn test_ivfflat_write_read_search_roundtrip() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut expected_distances = vec![0.0; 5];
        let mut expected_labels = vec![0; 5];
        index.search(
            &data[7 * d..8 * d],
            1,
            5,
            nlist,
            &mut expected_distances,
            &mut expected_labels,
        );

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfflat_index(&index, &mut writer).unwrap();

        let mut reader = IVFFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, distances) = reader.search(&data[7 * d..8 * d], 5, nlist).unwrap();

        assert_eq!(labels, expected_labels);
        assert_eq!(distances, expected_distances);
    }

    #[test]
    fn test_ivfflat_reader_handles_unaligned_vector_suffix_without_copy_contract_change() {
        let mut index = IVFFlatIndex::new(3, 1, MetricType::L2);
        index.quantizer_centroids = vec![0.0; 3];
        // These deltas occupy 1, 2, and 3 bytes, so the raw-vector suffix
        // starts at byte 18 inside the list payload instead of a f32 boundary.
        index.ids[0] = vec![1, 130, 16_515];
        index.vectors[0] = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_, encoded_ids) = encode_delta_varint_ids(&index.ids[0]);
        assert_eq!((12 + encoded_ids.len()) % align_of::<f32>(), 2);

        let mut reader =
            IVFFlatIndexReader::open(Cursor::new(serialized_flat_index(&index))).unwrap();
        let mut direct_lists = reader.read_inverted_lists(&[0]).unwrap();
        let direct = direct_lists.pop().unwrap();
        assert_eq!(
            direct.vectors().as_ptr().align_offset(align_of::<f32>()),
            0,
            "the direct scan buffer must repair the unaligned on-disk suffix"
        );
        assert_eq!(direct.vectors(), index.vectors[0]);

        let (ids, vectors) = reader.read_inverted_list(0).unwrap();
        assert_eq!(ids, index.ids[0]);
        assert_eq!(vectors, index.vectors[0]);

        let (result_ids, distances) = reader.search(&[1.0, 2.0, 3.0], 1, 1).unwrap();
        assert_eq!(result_ids, vec![130]);
        assert_eq!(distances, vec![0.0]);
    }

    #[test]
    fn test_ivfflat_reader_cosine_matches_in_memory_with_zero_vector() {
        let d = 2;
        let data = vec![3.0, 4.0, 0.0, 0.0, -3.0, -4.0, 4.0, 3.0];
        let ids = vec![10, 11, 12, 13];
        let query = [3.0, 4.0];
        let mut index = IVFFlatIndex::new(d, 1, MetricType::Cosine);
        index.train(&data, ids.len());
        index.add(&data, &ids, ids.len());

        let mut expected_distances = vec![0.0; ids.len()];
        let mut expected_labels = vec![0; ids.len()];
        index.search(
            &query,
            1,
            ids.len(),
            1,
            &mut expected_distances,
            &mut expected_labels,
        );

        let mut reader =
            IVFFlatIndexReader::open(Cursor::new(serialized_flat_index(&index))).unwrap();
        let actual = reader.search(&query, ids.len(), 1).unwrap();
        assert_eq!(actual.0, expected_labels);
        assert_eq!(actual.1, expected_distances);
    }

    #[test]
    fn test_ivfflat_reader_search_with_filter() {
        use std::collections::HashSet;

        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.train(&data, 3);
        index.add(&data, &ids, 3);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfflat_index(&index, &mut writer).unwrap();

        let filter: HashSet<i64> = [12].into_iter().collect();
        let mut reader = IVFFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, distances) = reader
            .search_with_filter(&[0.0, 0.0], 2, 1, Some(&filter))
            .unwrap();

        assert_eq!(labels, vec![12, -1]);
        assert_eq!(distances[0], 200.0);
        assert_eq!(distances[1], f32::MAX);
    }

    #[test]
    fn test_ivfflat_reader_search_with_roaring_filter_bytes() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.train(&data, 3);
        index.add(&data, &ids, 3);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfflat_index(&index, &mut writer).unwrap();

        let mut allowed = RoaringTreemap::new();
        allowed.insert(12);
        let mut filter_bytes = Vec::new();
        allowed.serialize_into(&mut filter_bytes).unwrap();

        let mut reader = IVFFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, distances) = reader
            .search_with_roaring_filter(&[0.0, 0.0], 2, 1, &filter_bytes)
            .unwrap();

        assert_eq!(labels, vec![12, -1]);
        assert_eq!(distances[0], 200.0);
        assert_eq!(distances[1], f32::MAX);
    }

    #[test]
    fn test_ivfflat_batch_reader_matches_single_reader_search() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfflat_index(&index, &mut writer).unwrap();

        let queries = [&data[7 * d..8 * d], &data[63 * d..64 * d]].concat();
        let k = 5;
        let nprobe = 3;
        let mut batch_reader = IVFFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let (batch_labels, batch_distances) =
            search_batch_ivfflat_reader(&mut batch_reader, &queries, 2, k, nprobe).unwrap();

        for qi in 0..2 {
            let mut single_reader = IVFFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
            let query = &queries[qi * d..(qi + 1) * d];
            let (single_labels, single_distances) = single_reader.search(query, k, nprobe).unwrap();
            assert_eq!(&batch_labels[qi * k..(qi + 1) * k], single_labels);
            assert_eq!(&batch_distances[qi * k..(qi + 1) * k], single_distances);
        }
    }

    #[test]
    fn test_ivfflat_large_single_query_scans_lists_in_parallel() {
        let d = 16;
        let nlist = 8;
        let index = balanced_flat_index(d, nlist, 8192);
        let query = index.vectors[3][17 * d..18 * d].to_vec();
        let k = 10;
        let mut expected_distances = vec![0.0; k];
        let mut expected_labels = vec![0; k];
        index.search(
            &query,
            1,
            k,
            nlist,
            &mut expected_distances,
            &mut expected_labels,
        );

        let filter = ThreadTrackingFilter {
            workers: AtomicU64::new(0),
        };
        let mut reader =
            IVFFlatIndexReader::open(Cursor::new(serialized_flat_index(&index))).unwrap();
        let actual = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                reader
                    .search_with_filter(&query, k, nlist, Some(&filter))
                    .unwrap()
            });

        let mut actual_pairs = actual.0.into_iter().zip(actual.1).collect::<Vec<_>>();
        let mut expected_pairs = expected_labels
            .into_iter()
            .zip(expected_distances)
            .collect::<Vec<_>>();
        actual_pairs.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        expected_pairs.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        assert_eq!(actual_pairs, expected_pairs);
        assert!(
            filter.workers.load(Ordering::Relaxed).count_ones() > 1,
            "a large single query should scan probed lists on multiple Rayon workers"
        );
    }

    #[test]
    fn test_ivfflat_batch_scans_lists_in_parallel_without_duplicate_reads() {
        let d = 16;
        let nlist = 8;
        let nq = 8;
        let k = 10;
        let index = balanced_flat_index(d, nlist, 1024);
        let queries = (0..nq)
            .flat_map(|query_index| {
                let row = query_index * 7;
                index.vectors[query_index][row * d..(row + 1) * d].iter()
            })
            .copied()
            .collect::<Vec<_>>();
        let bytes = serialized_flat_index(&index);
        let mut expected_ids = Vec::with_capacity(nq * k);
        let mut expected_distances = Vec::with_capacity(nq * k);
        for query in queries.chunks_exact(d) {
            let mut reader = IVFFlatIndexReader::open(Cursor::new(bytes.clone())).unwrap();
            let (ids, distances) = reader.search(query, k, nlist).unwrap();
            expected_ids.extend(ids);
            expected_distances.extend(distances);
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingReader {
            inner: Cursor::new(bytes),
            calls: Arc::clone(&calls),
        };
        let mut reader = IVFFlatIndexReader::open(source).unwrap();
        reader.ensure_loaded().unwrap();
        calls.store(0, Ordering::Relaxed);
        let filter = ThreadTrackingFilter {
            workers: AtomicU64::new(0),
        };
        let actual = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                search_batch_ivfflat_reader_filter(
                    &mut reader,
                    &queries,
                    nq,
                    k,
                    nlist,
                    Some(&filter),
                )
                .unwrap()
            });

        for query_index in 0..nq {
            let range = query_index * k..(query_index + 1) * k;
            assert!(
                actual.1[range.clone()]
                    .windows(2)
                    .all(|pair| pair[0] <= pair[1]),
                "parallel batch results must remain distance-sorted"
            );
            let mut actual_pairs = actual.0[range.clone()]
                .iter()
                .copied()
                .zip(actual.1[range.clone()].iter().copied())
                .collect::<Vec<_>>();
            let mut expected_pairs = expected_ids[range.clone()]
                .iter()
                .copied()
                .zip(expected_distances[range].iter().copied())
                .collect::<Vec<_>>();
            actual_pairs.sort_by_key(|&(id, _)| id);
            expected_pairs.sort_by_key(|&(id, _)| id);
            assert_eq!(actual_pairs, expected_pairs);
        }
        assert!(
            filter.workers.load(Ordering::Relaxed).count_ones() > 1,
            "a large batch should scan lists on multiple Rayon workers"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "parallel scan must not duplicate the multi-range list read"
        );
    }

    #[test]
    fn test_ivfflat_open_and_metadata_load_use_two_reads() {
        let index = balanced_flat_index(16, 32, 4);
        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingReader {
            inner: Cursor::new(serialized_flat_index(&index)),
            calls: Arc::clone(&calls),
        };

        let mut reader = IVFFlatIndexReader::open(source).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1, "header is one range read");
        reader.ensure_loaded().unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "centroids and the offset table are one contiguous range read"
        );
        reader.ensure_loaded().unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "resident metadata is not read twice"
        );
    }

    #[test]
    fn test_ivfflat_metadata_rejects_header_vector_count_mismatch() {
        let index = balanced_flat_index(8, 4, 16);
        let mut bytes = serialized_flat_index(&index);
        bytes[20..28].copy_from_slice(&(index.total_vectors() as i64 + 1).to_le_bytes());

        let mut reader = IVFFlatIndexReader::open(Cursor::new(bytes)).unwrap();
        let error = reader.ensure_loaded().unwrap_err();
        assert!(error
            .to_string()
            .contains("header vector count 65 does not match list total 64"));
    }

    #[test]
    fn test_ivfflat_batch_reader_search_with_roaring_filter_bytes() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.train(&data, 3);
        index.add(&data, &ids, 3);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfflat_index(&index, &mut writer).unwrap();

        let mut allowed = RoaringTreemap::new();
        allowed.insert(12);
        let mut filter_bytes = Vec::new();
        allowed.serialize_into(&mut filter_bytes).unwrap();

        let mut reader = IVFFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let queries = vec![0.0, 0.0, 10.0, 10.0];
        let (labels, distances) = search_batch_ivfflat_reader_roaring_filter(
            &mut reader,
            &queries,
            2,
            2,
            1,
            &filter_bytes,
        )
        .unwrap();

        assert_eq!(labels, vec![12, -1, 12, -1]);
        assert_eq!(distances, vec![200.0, f32::MAX, 0.0, f32::MAX]);
    }

    #[test]
    fn test_ivfflat_reader_validates_inputs() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let ids = vec![1, 2];

        let mut index = IVFFlatIndex::new(d, nlist, MetricType::L2);
        index.train(&data, 2);
        index.add(&data, &ids, 2);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfflat_index(&index, &mut writer).unwrap();

        let mut reader = IVFFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
        assert!(reader.search(&[0.0], 1, 1).is_err());

        let mut reader = IVFFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
        assert!(reader.search(&[0.0, 0.0], 0, 1).is_err());

        let mut reader = IVFFlatIndexReader::open(Cursor::new(buf)).unwrap();
        assert!(reader.search(&[0.0, 0.0], 1, 0).is_err());
    }

    #[test]
    fn test_ivfflat_writer_validates_shape_before_writing() {
        let mut index = IVFFlatIndex::new(2, 1, MetricType::L2);
        index.quantizer_centroids = vec![0.0, 0.0];
        index.ids[0] = vec![1, 2];
        index.vectors[0] = vec![0.0, 0.0];

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        let err = write_ivfflat_index(&index, &mut writer).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("vector length"));
    }

    #[test]
    fn test_ivfflat_reader_rejects_bad_magic() {
        let mut buf = vec![0u8; IVFFLAT_HEADER_SIZE];
        buf[0..4].copy_from_slice(&0x12345678u32.to_le_bytes());

        let err = match IVFFlatIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("bad magic should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_ivfflat_reader_rejects_missing_required_flags() {
        let mut buf = vec![0u8; IVFFLAT_HEADER_SIZE];
        buf[0..4].copy_from_slice(&IVFFLAT_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&IVFFLAT_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&2i32.to_le_bytes());
        buf[12..16].copy_from_slice(&1i32.to_le_bytes());
        buf[16..20].copy_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf[20..28].copy_from_slice(&0i64.to_le_bytes());
        buf[28..32].copy_from_slice(&0u32.to_le_bytes());

        let err = match IVFFlatIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("missing required flags should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("requires delta IDs"));
    }

    #[test]
    fn test_ivfflat_reader_rejects_unknown_flags() {
        let mut buf = vec![0u8; IVFFLAT_HEADER_SIZE];
        buf[0..4].copy_from_slice(&IVFFLAT_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&IVFFLAT_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&2i32.to_le_bytes());
        buf[12..16].copy_from_slice(&1i32.to_le_bytes());
        buf[16..20].copy_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf[20..28].copy_from_slice(&0i64.to_le_bytes());
        buf[28..32].copy_from_slice(&(REQUIRED_FLAGS | (1 << 31)).to_le_bytes());

        let err = match IVFFlatIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("unknown flags should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Unsupported IVFFLAT flags"));
    }

    #[test]
    fn test_ivfflat_reader_rejects_nonzero_reserved_bytes() {
        let mut buf = vec![0u8; IVFFLAT_HEADER_SIZE];
        buf[0..4].copy_from_slice(&IVFFLAT_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&IVFFLAT_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&2i32.to_le_bytes());
        buf[12..16].copy_from_slice(&1i32.to_le_bytes());
        buf[16..20].copy_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf[20..28].copy_from_slice(&0i64.to_le_bytes());
        buf[28..32].copy_from_slice(&REQUIRED_FLAGS.to_le_bytes());
        buf[32] = 1;

        let err = match IVFFlatIndexReader::open(Cursor::new(buf)) {
            Ok(_) => panic!("non-zero reserved bytes should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("reserved bytes must be zero"));
    }
}
