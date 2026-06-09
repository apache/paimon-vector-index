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

use crate::distance::{fvec_distance, fvec_normalize, MetricType};
use crate::io::{SeekRead, SeekWrite};
use crate::ivfflat::IVFFlatIndex;
use crate::ivfpq::RowIdFilter;
use crate::kmeans;
use roaring::RoaringTreemap;
use std::io;

pub const IVFFLAT_MAGIC: u32 = 0x4956464C; // "IVFL"
pub const IVFFLAT_VERSION: u32 = 1;
pub const IVFFLAT_HEADER_SIZE: usize = 64;

const FLAG_DELTA_IDS: u32 = 1 << 0;

pub fn write_ivfflat_index(index: &IVFFlatIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    let d = index.d;
    let nlist = index.nlist;
    validate_index_shape(index)?;
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

    let mut sorted_lists: Vec<(Vec<i64>, Vec<u8>, Vec<f32>)> = Vec::with_capacity(nlist);
    for list_id in 0..nlist {
        let count = index.ids[list_id].len();
        if count == 0 {
            sorted_lists.push((Vec::new(), Vec::new(), Vec::new()));
            continue;
        }

        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by_key(|&idx| index.ids[list_id][idx]);

        let sorted_ids: Vec<i64> = order.iter().map(|&idx| index.ids[list_id][idx]).collect();
        let mut sorted_vectors = Vec::with_capacity(count * d);
        for idx in order {
            sorted_vectors.extend_from_slice(&index.vectors[list_id][idx * d..(idx + 1) * d]);
        }
        let (_, id_bytes) = encode_delta_varint_ids(&sorted_ids);
        sorted_lists.push((sorted_ids, id_bytes, sorted_vectors));
    }

    write_u32_le(out, IVFFLAT_MAGIC)?;
    write_u32_le(out, IVFFLAT_VERSION)?;
    write_i32_le(out, d_i32)?;
    write_i32_le(out, nlist_i32)?;
    write_u32_le(out, index.metric as u32)?;
    write_i64_le(out, total_vectors)?;
    write_u32_le(out, FLAG_DELTA_IDS)?;
    out.write_all(&[0u8; 32])?;

    write_f32_slice(out, &index.quantizer_centroids)?;

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
            let vector_bytes = checked_list_bytes(
                count,
                d.checked_mul(4).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "IVF-FLAT bytes per vector overflow",
                    )
                })?,
            )?;
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

    for (sorted_ids, id_bytes, sorted_vectors) in sorted_lists {
        if sorted_ids.is_empty() {
            continue;
        }
        write_i64_le(out, sorted_ids[0])?;
        write_i32_le(out, id_bytes.len() as i32)?;
        out.write_all(&id_bytes)?;
        write_f32_slice(out, &sorted_vectors)?;
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

impl<R: SeekRead> IVFFlatIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        reader.seek(0)?;

        let magic = read_u32_le(&mut reader)?;
        if magic != IVFFLAT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVFFLAT magic: 0x{:08X}", magic),
            ));
        }
        let version = read_u32_le(&mut reader)?;
        if version != IVFFLAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVFFLAT version: {}", version),
            ));
        }

        let d = validate_positive_i32(read_i32_le(&mut reader)?, "d")? as usize;
        let nlist = validate_positive_i32(read_i32_le(&mut reader)?, "nlist")? as usize;
        let metric_code = read_u32_le(&mut reader)?;
        let metric = MetricType::from_code(metric_code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown metric type: {}", metric_code),
            )
        })?;
        let total_vectors = read_i64_le(&mut reader)?;
        let flags = read_u32_le(&mut reader)?;
        let mut reserved = [0u8; 32];
        reader.read_exact(&mut reserved)?;

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
            delta_ids: flags & FLAG_DELTA_IDS != 0,
            loaded: false,
        })
    }

    pub fn ensure_loaded(&mut self) -> io::Result<()> {
        if self.loaded {
            return Ok(());
        }

        self.reader.seek(IVFFLAT_HEADER_SIZE as u64)?;
        self.quantizer_centroids =
            read_f32_vec(&mut self.reader, checked_section_size(self.nlist, self.d)?)?;
        self.list_offsets = vec![0; self.nlist];
        self.list_counts = vec![0; self.nlist];
        self.list_id_bytes_lens = vec![0; self.nlist];
        for list_id in 0..self.nlist {
            self.list_offsets[list_id] = read_i64_le(&mut self.reader)?;
            let count = read_i32_le(&mut self.reader)?;
            if count < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative list count {} at list {}", count, list_id),
                ));
            }
            self.list_counts[list_id] = count;
            let id_bytes_len = read_i32_le(&mut self.reader)?;
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

    pub fn read_inverted_list(&mut self, list_id: usize) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.ensure_loaded()?;
        if list_id >= self.nlist {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("list_id {} out of range (nlist={})", list_id, self.nlist),
            ));
        }
        let count = self.list_counts[list_id] as usize;
        if count == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
        let vector_bytes = checked_list_bytes(count, self.d * 4)?;
        if self.delta_ids {
            let id_bytes_len = self.list_id_bytes_lens[list_id] as usize;
            let payload_len = 12usize
                .checked_add(id_bytes_len)
                .and_then(|len| len.checked_add(vector_bytes))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "IVF-FLAT list payload overflow")
                })?;
            let mut payload = vec![0u8; payload_len];
            self.reader.pread(offset, &mut payload)?;
            let base_id = i64::from_le_bytes(payload[0..8].try_into().unwrap());
            let encoded_len = i32::from_le_bytes(payload[8..12].try_into().unwrap());
            if encoded_len < 0 || encoded_len as usize != id_bytes_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF-FLAT id_bytes_len mismatch",
                ));
            }
            let ids = decode_delta_varint_ids(base_id, &payload[12..12 + id_bytes_len], count)?;
            let vectors = bytes_to_f32_vec(&payload[12 + id_bytes_len..])?;
            Ok((ids, vectors))
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-FLAT reader only supports delta IDs",
            ))
        }
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

        for list_id in probe_indices {
            let (ids, vectors) = self.read_inverted_list(list_id)?;
            for (local_idx, &id) in ids.iter().enumerate() {
                if let Some(f) = filter {
                    if !f.contains(id) {
                        continue;
                    }
                }
                let vector = &vectors[local_idx * self.d..(local_idx + 1) * self.d];
                heap.push(fvec_distance(&q, vector, self.metric), id);
            }
        }

        let sorted = heap.into_sorted();
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
    for list_id in unique_lists {
        let count = reader.list_counts[list_id] as usize;
        if count == 0 {
            continue;
        }
        let (ids, vectors) = reader.read_inverted_list(list_id)?;
        for &qi in &list_to_queries[list_id] {
            let query = &processed[qi * d..(qi + 1) * d];
            for (local_idx, &id) in ids.iter().enumerate() {
                if let Some(f) = filter {
                    if !f.contains(id) {
                        continue;
                    }
                }
                let vector = &vectors[local_idx * d..(local_idx + 1) * d];
                heaps[qi].push(fvec_distance(query, vector, reader.metric), id);
            }
        }
    }

    let mut result_ids = vec![-1i64; nq * k];
    let mut result_dists = vec![f32::MAX; nq * k];
    for qi in 0..nq {
        let sorted = std::mem::replace(&mut heaps[qi], ReaderTopKHeap::new(0)).into_sorted();
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

struct ReaderTopKHeap {
    k: usize,
    data: Vec<(f32, i64)>,
}

impl ReaderTopKHeap {
    fn new(k: usize) -> Self {
        Self {
            k,
            data: Vec::with_capacity(k),
        }
    }

    fn push(&mut self, dist: f32, id: i64) {
        if self.k == 0 {
            return;
        }
        if self.data.len() < self.k {
            self.data.push((dist, id));
            return;
        }
        if let Some((worst_idx, _)) = self
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
        {
            if dist < self.data[worst_idx].0 {
                self.data[worst_idx] = (dist, id);
            }
        }
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

fn write_f32_slice(out: &mut dyn SeekWrite, data: &[f32]) -> io::Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    out.write_all(&bytes)
}

fn read_u32_le(reader: &mut dyn SeekRead) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_le(reader: &mut dyn SeekRead) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64_le(reader: &mut dyn SeekRead) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
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

fn read_f32_vec(reader: &mut dyn SeekRead, count: usize) -> io::Result<Vec<f32>> {
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
    use crate::io::PosWriter;
    use crate::ivfflat::IVFFlatIndex;
    use std::io::Cursor;

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
}
