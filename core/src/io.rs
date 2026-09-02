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

use crate::distance::MetricType;
use crate::index_io_util::{
    bounded_ivf_payload_batch_end, bounded_ivf_stream_chunk_rows, bytes_to_f32_vec,
    decode_delta_varint_ids, encode_delta_varint_ids, pread_batched_slices,
    read_delta_varint_ids_at, validate_reserved_zero,
};
use crate::ivfpq::IVFPQIndex;
use crate::opq::OPQMatrix;
use crate::pq::ProductQuantizer;
use rayon::prelude::*;
use std::io;
use std::mem::size_of;
use std::time::{Duration, Instant};

pub const MAGIC: u32 = 0x49565051; // "IVPQ"
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 64;

pub const FLAG_HAS_OPQ: u32 = 1 << 0;
pub const FLAG_BY_RESIDUAL: u32 = 1 << 1;
pub const FLAG_DELTA_IDS: u32 = 1 << 2;
pub const FLAG_TRANSPOSED_CODES: u32 = 1 << 3;
const REQUIRED_FLAGS: u32 = FLAG_DELTA_IDS | FLAG_TRANSPOSED_CODES;
const SUPPORTED_FLAGS: u32 = FLAG_HAS_OPQ | FLAG_BY_RESIDUAL | REQUIRED_FLAGS;

pub struct ReadRequest<'a> {
    pub pos: u64,
    pub buf: &'a mut [u8],
}

impl<'a> ReadRequest<'a> {
    pub fn new(pos: u64, buf: &'a mut [u8]) -> Self {
        Self { pos, buf }
    }
}

/// Optional immutable-source hints used to refine DiskANN's automatic read
/// plan. Zero means unspecified for every field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SeekReadCapabilities {
    /// Estimated end-to-end latency of one representative random read.
    ///
    /// Zero means unknown and lets DiskANN reuse the mandatory header read's
    /// elapsed time while opening.
    pub estimated_random_read_latency_nanos: u64,
    /// Efficient coalesced window size for random reads.
    pub preferred_window_bytes: usize,
    /// Maximum ranges accepted by one `pread` invocation.
    pub max_ranges_per_pread: usize,
}

/// Positional access to one immutable byte sequence.
///
/// Implementations must return the same bytes for the lifetime of an opened
/// index Reader. Replacing an index requires opening a new Reader.
pub trait SeekRead: Send {
    /// Positional reads for one or more ranges.
    ///
    /// Implementations may execute requests sequentially, coalesce them, or issue
    /// them concurrently when the underlying source supports independent
    /// positional reads.
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()>;

    /// Creates an independent handle to the same immutable byte sequence when supported.
    fn try_clone_reader(&self) -> io::Result<Option<Self>>
    where
        Self: Sized,
    {
        Ok(None)
    }

    /// Optionally refines the latency-derived range plan.
    ///
    /// This describes an immutable reader implementation, not a local-file
    /// type. Object stores, remote caches, memory readers, and custom storage
    /// adapters can all provide the same hints.
    fn read_capabilities(&self) -> SeekReadCapabilities {
        SeekReadCapabilities::default()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ReadMetrics {
    pub elapsed: Duration,
    pub calls: usize,
    pub requested_bytes: usize,
}

struct MeasuredSeekRead<R> {
    inner: R,
    metrics: Option<ReadMetrics>,
}

impl<R> MeasuredSeekRead<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            metrics: None,
        }
    }
}

impl<R: SeekRead> SeekRead for MeasuredSeekRead<R> {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        let measurement = self.metrics.as_ref().map(|_| {
            let requested_bytes = ranges.iter().fold(0usize, |total, request| {
                total.saturating_add(request.buf.len())
            });
            (Instant::now(), requested_bytes)
        });
        let result = self.inner.pread(ranges);
        if let (Some(metrics), Some((started, requested_bytes))) =
            (self.metrics.as_mut(), measurement)
        {
            metrics.elapsed += started.elapsed();
            metrics.calls = metrics.calls.saturating_add(1);
            metrics.requested_bytes = metrics.requested_bytes.saturating_add(requested_bytes);
        }
        result
    }

    fn try_clone_reader(&self) -> io::Result<Option<Self>> {
        // Clones start with metrics disabled, so their I/O is not included here.
        Ok(self.inner.try_clone_reader()?.map(Self::new))
    }

    fn read_capabilities(&self) -> SeekReadCapabilities {
        self.inner.read_capabilities()
    }
}

pub(crate) struct PreadCursor<'a, R: SeekRead + ?Sized> {
    reader: &'a mut R,
    pos: u64,
}

impl<'a, R: SeekRead + ?Sized> PreadCursor<'a, R> {
    pub(crate) fn new(reader: &'a mut R, pos: u64) -> Self {
        Self { reader, pos }
    }

    pub(crate) fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.reader.pread(&mut [ReadRequest::new(self.pos, buf)])?;
        self.pos = self
            .pos
            .checked_add(buf.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "read cursor overflow"))?;
        Ok(())
    }
}

pub trait SeekWrite: Send {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()>;
    fn pos(&self) -> u64;
}

impl<T: io::Read + io::Seek + Send> SeekRead for T {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        let old_pos = io::Seek::stream_position(self)?;
        for range in ranges {
            io::Seek::seek(self, io::SeekFrom::Start(range.pos))?;
            io::Read::read_exact(self, range.buf)?;
        }
        io::Seek::seek(self, io::SeekFrom::Start(old_pos))?;
        Ok(())
    }
}

pub struct PosWriter<W: io::Write> {
    inner: W,
    pos: u64,
}

impl<W: io::Write> PosWriter<W> {
    pub fn new(inner: W) -> Self {
        PosWriter { inner, pos: 0 }
    }
}

impl<W: io::Write + Send> SeekWrite for PosWriter<W> {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.inner.write_all(buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn pos(&self) -> u64 {
        self.pos
    }
}

// --- Read/write helpers ---

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

fn validate_positive_i32(val: i32, field: &str) -> io::Result<i32> {
    if val <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid header field {}: {} (must be positive)", field, val),
        ));
    }
    Ok(val)
}

/// Max element count for any single section (~4GB of f32).
const MAX_SECTION_ELEMENTS: usize = 1 << 30;

fn checked_section_size(a: usize, b: usize) -> io::Result<usize> {
    let result = a.checked_mul(b).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "section size overflow in index header",
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
            "inverted list byte size overflow",
        )
    })
}

/// Write a complete IVF-PQ index with delta-varint ID encoding.
pub fn write_index(index: &IVFPQIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    let d = index.d;
    let nlist = index.nlist;
    let m = index.pq.m;
    let ksub = index.pq.ksub;
    let dsub = index.pq.dsub;
    let code_size = index.pq.code_size();
    if ksub == 16 && !m.is_multiple_of(2) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("4-bit IVF-PQ requires even m, got {}", m),
        ));
    }
    let d_i32 = usize_to_i32(d, "dimension")?;
    let nlist_i32 = usize_to_i32(nlist, "nlist")?;
    let m_i32 = usize_to_i32(m, "pq m")?;
    let ksub_i32 = usize_to_i32(ksub, "pq ksub")?;
    let dsub_i32 = usize_to_i32(dsub, "pq dsub")?;

    let mut flags: u32 = FLAG_DELTA_IDS | FLAG_TRANSPOSED_CODES;
    if index.opq.is_some() {
        flags |= FLAG_HAS_OPQ;
    }
    if index.by_residual {
        flags |= FLAG_BY_RESIDUAL;
    }

    let total_vectors = index.ids.iter().try_fold(0i64, |sum, ids| {
        let count = usize_to_i64(ids.len(), "total vector count")?;
        sum.checked_add(count).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "total vector count exceeds i64 length limit",
            )
        })
    })?;

    // Sort IDs within each list and prepare delta-varint encoded data
    let mut sorted_lists = Vec::with_capacity(nlist);
    for i in 0..nlist {
        let count = index.ids[i].len();
        let expected_code_bytes = checked_list_bytes(count, code_size)?;
        if index.codes[i].len() != expected_code_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("IVFPQ code length mismatch at list {i}"),
            ));
        }
        if count == 0 {
            sorted_lists.push(SortedPqListMetadata::default());
            continue;
        }

        // The compact codes are reordered lazily, one list at a time, while
        // writing. Keeping only the permutation avoids a second index-sized
        // code allocation in the writer.
        let mut indices: Vec<usize> = (0..count).collect();
        indices.sort_by_key(|&idx| index.ids[i][idx]);

        let sorted_ids: Vec<i64> = indices.iter().map(|&idx| index.ids[i][idx]).collect();
        let base_id = sorted_ids[0];
        let (_, id_bytes) = encode_delta_varint_ids(&sorted_ids);
        sorted_lists.push(SortedPqListMetadata {
            base_id,
            order: indices,
            id_bytes,
        });
    }

    // Header
    write_u32_le(out, MAGIC)?;
    write_u32_le(out, VERSION)?;
    write_i32_le(out, d_i32)?;
    write_i32_le(out, nlist_i32)?;
    write_i32_le(out, m_i32)?;
    write_i32_le(out, ksub_i32)?;
    write_i32_le(out, dsub_i32)?;
    write_u32_le(out, index.metric as u32)?;
    write_i64_le(out, total_vectors)?;
    write_u32_le(out, flags)?;
    out.write_all(&[0u8; 20])?;

    if let Some(ref opq) = index.opq {
        write_f32_slice(out, &opq.rotation)?;
    }

    write_f32_slice(out, index.quantizer_centroids())?;
    write_f32_slice(out, &index.pq.centroids)?;

    // Compute offsets for inverted lists
    // Delta-varint format per list: [base_id: i64][id_bytes_len: u32][id_bytes][codes]
    let offset_table_size = nlist.checked_mul(16).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVFPQ offset table size overflow",
        )
    })?;
    let data_start = out
        .pos()
        .checked_add(offset_table_size as u64)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVFPQ data start offset overflow",
            )
        })?;

    let mut list_offsets = vec![0i64; nlist];
    let mut list_counts = vec![0i32; nlist];
    let mut list_id_bytes_lens = vec![0i32; nlist];
    let mut current_offset = data_start;

    for i in 0..nlist {
        list_offsets[i] = u64_to_i64(current_offset, "list offset")?;
        let count = sorted_lists[i].order.len();
        list_counts[i] = usize_to_i32(count, "list count")?;
        if count > 0 {
            // base_id(8) + id_bytes_len(4) + id_bytes + codes
            let id_bytes_len = sorted_lists[i].id_bytes.len();
            list_id_bytes_lens[i] = usize_to_i32(id_bytes_len, "delta ID section")?;
            let code_bytes = checked_list_bytes(count, code_size)?;
            let list_bytes = 12usize
                .checked_add(id_bytes_len)
                .and_then(|len| len.checked_add(code_bytes))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVFPQ list size overflow")
                })?;
            current_offset = current_offset
                .checked_add(list_bytes as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVFPQ offset overflow")
                })?;
        }
    }

    // Write offset table
    for i in 0..nlist {
        write_i64_le(out, list_offsets[i])?;
        write_i32_le(out, list_counts[i])?;
        write_i32_le(out, list_id_bytes_lens[i])?;
    }

    // Write inverted list data
    let mut transposed = Vec::new();
    for i in 0..nlist {
        let list = &sorted_lists[i];
        if list.order.is_empty() {
            continue;
        }
        // base_id
        write_i64_le(out, list.base_id)?;
        // id_bytes_len + id_bytes
        write_i32_le(out, usize_to_i32(list.id_bytes.len(), "delta ID section")?)?;
        out.write_all(&list.id_bytes)?;
        // PQ codes — transpose for cache-friendly SIMD scan. Blocking keeps
        // both the row-major source and column-major destination hot while the
        // writer applies the row-ID sort permutation.
        transpose_sorted_pq_codes(&index.codes[i], &list.order, code_size, &mut transposed);
        out.write_all(&transposed)?;
    }

    Ok(())
}

fn transpose_sorted_pq_codes(
    codes: &[u8],
    order: &[usize],
    code_size: usize,
    transposed: &mut Vec<u8>,
) {
    const TILE: usize = 32;

    let count = order.len();
    transposed.resize(count * code_size, 0);
    for row_start in (0..count).step_by(TILE) {
        let row_end = (row_start + TILE).min(count);
        for column_start in (0..code_size).step_by(TILE) {
            let column_end = (column_start + TILE).min(code_size);
            for (target_row, &source_row) in order[row_start..row_end].iter().enumerate() {
                let target_row = row_start + target_row;
                let source = &codes
                    [source_row * code_size + column_start..source_row * code_size + column_end];
                for (column_offset, &code) in source.iter().enumerate() {
                    transposed[(column_start + column_offset) * count + target_row] = code;
                }
            }
        }
    }
}

#[derive(Default)]
struct SortedPqListMetadata {
    base_id: i64,
    order: Vec<usize>,
    id_bytes: Vec<u8>,
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

// --- Reader ---

pub struct IVFPQIndexReader<R: SeekRead> {
    reader: MeasuredSeekRead<R>,
    pub d: usize,
    pub nlist: usize,
    pub m: usize,
    pub ksub: usize,
    pub dsub: usize,
    pub metric: MetricType,
    pub by_residual: bool,
    pub total_vectors: i64,
    pub opq: Option<OPQMatrix>,
    pub quantizer_centroids: Vec<f32>,
    pub pq: ProductQuantizer,
    pub list_offsets: Vec<i64>,
    pub list_counts: Vec<i32>,
    pub list_id_bytes_lens: Vec<i32>,
    pub precomputed_table: Vec<f32>,
    pub transposed_codes: bool,
    /// Whether heavy data (centroids, codebooks, offset table) has been loaded
    loaded: bool,
    /// Whether file has OPQ rotation matrix
    has_opq: bool,
}

impl<R: SeekRead> IVFPQIndexReader<R> {
    /// Open an index file. Only reads the 64-byte header.
    /// Centroids, codebooks, and offset table are loaded lazily on first search.
    pub fn open(mut reader: R) -> io::Result<Self> {
        let mut header = [0u8; HEADER_SIZE];
        reader.pread(&mut [ReadRequest::new(0, &mut header)])?;
        Self::open_with_header(reader, header)
    }

    pub(crate) fn open_with_header(reader: R, header: [u8; HEADER_SIZE]) -> io::Result<Self> {
        let read_u32 =
            |offset: usize| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
        let read_i32 =
            |offset: usize| i32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
        let read_i64 =
            |offset: usize| i64::from_le_bytes(header[offset..offset + 8].try_into().unwrap());

        let magic = read_u32(0);
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVFPQ magic: 0x{:08X}", magic),
            ));
        }

        let version = read_u32(4);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVFPQ version: {}", version),
            ));
        }

        let d = validate_positive_i32(read_i32(8), "d")? as usize;
        let nlist = validate_positive_i32(read_i32(12), "nlist")? as usize;
        let m = validate_positive_i32(read_i32(16), "m")? as usize;
        let ksub = validate_positive_i32(read_i32(20), "ksub")? as usize;
        let dsub = validate_positive_i32(read_i32(24), "dsub")? as usize;

        if ksub != 16 && ksub != 256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported ksub {} (must be 16 or 256)", ksub),
            ));
        }
        if d != m * dsub {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "PQ invariant violated: d={} != m*dsub={}*{}={}",
                    d,
                    m,
                    dsub,
                    m * dsub
                ),
            ));
        }
        if ksub == 16 && !m.is_multiple_of(2) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("4-bit PQ requires even m, got {}", m),
            ));
        }

        let metric_code = read_u32(28);
        let metric = MetricType::from_code(metric_code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown metric type: {}", metric_code),
            )
        })?;
        let total_vectors = read_i64(32);
        if total_vectors < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVFPQ total vector count must be non-negative",
            ));
        }

        let flags = read_u32(40);
        validate_reserved_zero(&header[44..64], "IVFPQ")?;
        let unknown_flags = flags & !SUPPORTED_FLAGS;
        if unknown_flags != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVFPQ flags: 0x{:08X}", unknown_flags),
            ));
        }
        if flags & REQUIRED_FLAGS != REQUIRED_FLAGS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVFPQ v1 requires delta IDs and transposed codes",
            ));
        }
        let by_residual = flags & FLAG_BY_RESIDUAL != 0;
        let transposed_codes = flags & FLAG_TRANSPOSED_CODES != 0;
        let has_opq = flags & FLAG_HAS_OPQ != 0;

        Ok(IVFPQIndexReader {
            reader: MeasuredSeekRead::new(reader),
            d,
            nlist,
            m,
            ksub,
            dsub,
            metric,
            by_residual,
            total_vectors,
            opq: None,
            quantizer_centroids: Vec::new(),
            pq: ProductQuantizer {
                d,
                m,
                nbits: ksub.trailing_zeros() as usize,
                dsub,
                ksub,
                chunk_offsets: (0..=m).map(|chunk| chunk * dsub).collect(),
                centroids: Vec::new(),
                centroid_norms_cache: Vec::new(),
            },
            list_offsets: Vec::new(),
            list_counts: Vec::new(),
            list_id_bytes_lens: Vec::new(),
            precomputed_table: Vec::new(),
            transposed_codes,
            loaded: false,
            has_opq,
        })
    }

    pub(crate) fn begin_read_metrics(&mut self) {
        self.reader.metrics = Some(ReadMetrics::default());
    }

    pub(crate) fn end_read_metrics(&mut self) -> ReadMetrics {
        self.reader.metrics.take().unwrap_or_default()
    }

    /// Load centroids, codebooks, and offset table. Called automatically on first search.
    pub fn ensure_loaded(&mut self) -> io::Result<()> {
        if self.loaded {
            return Ok(());
        }

        let d = self.d;
        let nlist = self.nlist;
        let m = self.m;
        let ksub = self.ksub;
        let dsub = self.dsub;

        // Validate section sizes before allocating
        let rotation_count = checked_section_size(d, d)?;
        let centroids_count = checked_section_size(nlist, d)?;
        let mk = m
            .checked_mul(ksub)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "m*ksub overflow"))?;
        let pq_centroids_count = checked_section_size(mk, dsub)?;

        let rotation_bytes = if self.has_opq {
            rotation_count.checked_mul(4).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "OPQ byte length overflow")
            })?
        } else {
            0
        };
        let centroid_bytes = centroids_count.checked_mul(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "centroid byte length overflow")
        })?;
        let pq_centroid_bytes = pq_centroids_count.checked_mul(4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "PQ centroid byte length overflow",
            )
        })?;
        let offset_table_bytes = nlist.checked_mul(16).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVFPQ offset table byte length overflow",
            )
        })?;
        let metadata_bytes = rotation_bytes
            .checked_add(centroid_bytes)
            .and_then(|size| size.checked_add(pq_centroid_bytes))
            .and_then(|size| size.checked_add(offset_table_bytes))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVFPQ metadata size overflow")
            })?;
        let mut metadata = vec![0u8; metadata_bytes];
        self.reader
            .pread(&mut [ReadRequest::new(HEADER_SIZE as u64, &mut metadata)])?;
        let mut position = 0usize;

        if self.has_opq {
            let rotation = bytes_to_f32_vec(&metadata[position..position + rotation_bytes])?;
            position += rotation_bytes;
            self.opq = Some(OPQMatrix {
                d,
                m,
                rotation,
                is_trained: true,
                niter: 0,
                niter_pq: 0,
                niter_pq_0: 0,
                max_train_points: 0,
            });
        }

        self.quantizer_centroids =
            bytes_to_f32_vec(&metadata[position..position + centroid_bytes])?;
        position += centroid_bytes;

        let pq_centroids = bytes_to_f32_vec(&metadata[position..position + pq_centroid_bytes])?;
        position += pq_centroid_bytes;
        self.pq = ProductQuantizer {
            d,
            m,
            nbits: ksub.trailing_zeros() as usize,
            dsub,
            ksub,
            chunk_offsets: (0..=m).map(|chunk| chunk * dsub).collect(),
            centroids: pq_centroids,
            centroid_norms_cache: Vec::new(),
        };
        self.pq.rebuild_norms_cache();

        self.list_offsets = vec![0i64; nlist];
        self.list_counts = vec![0i32; nlist];
        self.list_id_bytes_lens = vec![0i32; nlist];
        let offset_table = &metadata[position..];
        let mut actual_total = 0i64;
        for (i, entry) in offset_table.chunks_exact(16).enumerate() {
            self.list_offsets[i] = i64::from_le_bytes(entry[0..8].try_into().unwrap());
            let count = i32::from_le_bytes(entry[8..12].try_into().unwrap());
            if count < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative list count {} at list {}", count, i),
                ));
            }
            self.list_counts[i] = count;
            actual_total = actual_total.checked_add(count as i64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVFPQ vector count overflow")
            })?;
            let id_bytes_len = i32::from_le_bytes(entry[12..16].try_into().unwrap());
            if id_bytes_len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative id_bytes_len {} at list {}", id_bytes_len, i),
                ));
            }
            self.list_id_bytes_lens[i] = id_bytes_len;
        }
        if actual_total != self.total_vectors {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IVFPQ header vector count {} does not match list total {actual_total}",
                    self.total_vectors
                ),
            ));
        }

        self.loaded = true;
        Ok(())
    }

    pub fn optimize_for_search(&mut self) -> io::Result<()> {
        self.ensure_loaded()?;
        if self.metric == MetricType::L2 && self.by_residual && self.precomputed_table.is_empty() {
            self.precomputed_table =
                compute_precomputed_table(&self.quantizer_centroids, &self.pq, self.nlist, self.d);
        }
        Ok(())
    }

    /// Read an inverted list's IDs and PQ codes.
    /// Calls ensure_loaded() if not yet loaded.
    pub fn read_inverted_list(&mut self, list_id: usize) -> io::Result<(Vec<i64>, Vec<u8>)> {
        let mut lists = self.read_inverted_list_payloads(&[list_id])?;
        let list = lists.pop().expect("one requested list has one result");
        let codes = list.codes().to_vec();
        Ok((list.ids, codes))
    }

    /// Read multiple inverted lists. Lists whose payload length is known from
    /// metadata are issued through a single batched pread call.
    pub fn read_inverted_lists(&mut self, list_ids: &[usize]) -> io::Result<Vec<InvertedListData>> {
        Ok(self
            .read_inverted_list_payloads(list_ids)?
            .into_iter()
            .map(InvertedListPayload::into_public)
            .collect())
    }

    /// Internal zero-copy form used by search. It retains the compact ID
    /// prefix and exposes the PQ-code suffix from the original read buffer.
    pub(crate) fn read_inverted_list_payloads(
        &mut self,
        list_ids: &[usize],
    ) -> io::Result<Vec<InvertedListPayload>> {
        self.ensure_loaded()?;

        let code_size = self.pq.code_size();
        let mut results: Vec<Option<InvertedListPayload>> =
            (0..list_ids.len()).map(|_| None).collect();
        let mut metas = Vec::new();
        let mut payloads = Vec::new();

        for (input_idx, &list_id) in list_ids.iter().enumerate() {
            if list_id >= self.nlist {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("list_id {} out of range (nlist={})", list_id, self.nlist),
                ));
            }
            let count = self.list_counts[list_id] as usize;
            if count == 0 {
                results[input_idx] = Some(InvertedListPayload::empty(list_id));
                continue;
            }

            let offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
            let code_bytes = checked_list_bytes(count, code_size)?;

            let id_bytes_len = self.list_id_bytes_lens[list_id];
            if id_bytes_len == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing id_bytes_len for non-empty IVFPQ list {}", list_id),
                ));
            }
            let payload_len = 12usize
                .checked_add(id_bytes_len as usize)
                .and_then(|len| len.checked_add(code_bytes))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "inverted list payload size overflow",
                    )
                })?;
            metas.push(BatchedListRead {
                input_idx,
                list_id,
                count,
                offset,
                id_bytes_len,
            });
            payloads.push(AlignedCodePayload::new(
                payload_len,
                12 + id_bytes_len as usize,
            )?);
        }

        if !metas.is_empty() {
            let offsets = metas.iter().map(|meta| meta.offset).collect::<Vec<_>>();
            let mut buffers = payloads
                .iter_mut()
                .map(AlignedCodePayload::read_buf_mut)
                .collect::<Vec<_>>();
            pread_batched_slices(&mut self.reader, &offsets, &mut buffers)?;
            drop(buffers);

            for (meta, payload) in metas.into_iter().zip(payloads) {
                results[meta.input_idx] = Some(decode_delta_list_payload(
                    meta.list_id,
                    payload,
                    meta.count,
                    meta.id_bytes_len,
                )?);
            }
        }

        results
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "missing batched inverted list read result",
                    )
                })
            })
            .collect()
    }

    pub(crate) fn batch_read_end(&self, list_ids: &[usize]) -> io::Result<usize> {
        let payload_lengths = list_ids
            .iter()
            .map(|&list_id| self.list_payload_len(list_id))
            .collect::<io::Result<Vec<_>>>()?;
        bounded_ivf_payload_batch_end(
            &payload_lengths,
            self.reader.read_capabilities().max_ranges_per_pread,
        )
    }

    pub(crate) fn list_payload_len(&self, list_id: usize) -> io::Result<usize> {
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
        let code_bytes = checked_list_bytes(count, self.pq.code_size())?;
        12usize
            .checked_add(self.list_id_bytes_lens[list_id] as usize)
            .and_then(|len| len.checked_add(code_bytes))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "inverted list payload size overflow",
                )
            })
    }

    pub(crate) fn for_each_streamed_list_chunk(
        &mut self,
        list_id: usize,
        mut consume: impl FnMut(&ProductQuantizer, &[i64], &[u8]),
    ) -> io::Result<()> {
        self.ensure_loaded()?;
        let count = self.list_counts[list_id] as usize;
        let list_offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
        let id_bytes_len = self.list_id_bytes_lens[list_id] as usize;
        let ids =
            read_delta_varint_ids_at(&mut self.reader, list_offset, count, id_bytes_len, "IVFPQ")?;
        let code_size = self.pq.code_size();
        let code_offset = list_offset
            .checked_add((12usize + id_bytes_len) as u64)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVFPQ code offset overflow")
            })?;
        let retained_id_bytes = ids.len().checked_mul(size_of::<i64>()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "IVFPQ decoded ID size overflow")
        })?;
        let mut row_start = 0usize;
        while row_start < count {
            let chunk_rows =
                bounded_ivf_stream_chunk_rows(count - row_start, code_size, retained_id_bytes, 1)?;
            let code_bytes = chunk_rows.checked_mul(code_size).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "IVFPQ chunk size overflow")
            })?;
            let mut payload = AlignedCodePayload::new(code_bytes, 0)?;
            if self.transposed_codes {
                let offsets = (0..code_size)
                    .map(|column| {
                        code_offset
                            .checked_add(
                                column
                                    .checked_mul(count)
                                    .and_then(|value| value.checked_add(row_start))
                                    .ok_or_else(|| {
                                        io::Error::new(
                                            io::ErrorKind::InvalidData,
                                            "IVFPQ transposed chunk offset overflow",
                                        )
                                    })? as u64,
                            )
                            .ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "IVFPQ transposed chunk offset overflow",
                                )
                            })
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let mut buffers = payload
                    .codes_mut()
                    .chunks_exact_mut(chunk_rows)
                    .collect::<Vec<_>>();
                pread_batched_slices(&mut self.reader, &offsets, &mut buffers)?;
            } else {
                let chunk_offset = code_offset
                    .checked_add(row_start.checked_mul(code_size).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "IVFPQ row-major chunk offset overflow",
                        )
                    })? as u64)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "IVFPQ row-major chunk offset overflow",
                        )
                    })?;
                self.reader
                    .pread(&mut [ReadRequest::new(chunk_offset, payload.codes_mut())])?;
            }
            let row_end = row_start + chunk_rows;
            consume(&self.pq, &ids[row_start..row_end], payload.codes());
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
        self.ensure_loaded()?;
        crate::ivfpq::search_with_reader(self, query, k, nprobe)
    }

    pub fn search_with_roaring_filter(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.ensure_loaded()?;
        crate::ivfpq::search_with_reader_roaring_filter(
            self,
            query,
            k,
            nprobe,
            roaring_filter_bytes,
        )
    }
}

pub struct InvertedListData {
    pub list_id: usize,
    pub ids: Vec<i64>,
    pub codes: Vec<u8>,
}

struct AlignedCodePayload {
    storage: Vec<u128>,
    read_start: usize,
    payload_len: usize,
    code_start: usize,
}

impl AlignedCodePayload {
    const ALIGNMENT: usize = std::mem::align_of::<u128>();

    fn new(payload_len: usize, code_start: usize) -> io::Result<Self> {
        if code_start > payload_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IVFPQ code suffix exceeds list payload",
            ));
        }
        let read_start = (Self::ALIGNMENT - code_start % Self::ALIGNMENT) % Self::ALIGNMENT;
        let storage_bytes = read_start.checked_add(payload_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "IVFPQ aligned payload overflow")
        })?;
        Ok(Self {
            storage: vec![0; storage_bytes.div_ceil(Self::ALIGNMENT)],
            read_start,
            payload_len,
            code_start,
        })
    }

    fn empty() -> Self {
        Self {
            storage: Vec::new(),
            read_start: 0,
            payload_len: 0,
            code_start: 0,
        }
    }

    fn storage_bytes_mut(&mut self) -> &mut [u8] {
        let byte_len = self.storage.len() * size_of::<u128>();
        // SAFETY: `storage` is initialized and exclusively borrowed for the
        // returned byte view.
        unsafe { std::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast::<u8>(), byte_len) }
    }

    fn storage_bytes(&self) -> &[u8] {
        let byte_len = self.storage.len() * size_of::<u128>();
        // SAFETY: `storage` remains alive and immutably borrowed.
        unsafe { std::slice::from_raw_parts(self.storage.as_ptr().cast::<u8>(), byte_len) }
    }

    fn read_buf_mut(&mut self) -> &mut [u8] {
        let start = self.read_start;
        let end = start + self.payload_len;
        &mut self.storage_bytes_mut()[start..end]
    }

    fn read_bytes(&self) -> &[u8] {
        &self.storage_bytes()[self.read_start..self.read_start + self.payload_len]
    }

    fn codes(&self) -> &[u8] {
        let codes = &self.read_bytes()[self.code_start..];
        debug_assert_eq!(
            codes.as_ptr().align_offset(Self::ALIGNMENT),
            0,
            "IVFPQ search codes must retain SIMD-friendly alignment"
        );
        codes
    }

    fn codes_mut(&mut self) -> &mut [u8] {
        let code_start = self.read_start + self.code_start;
        let code_end = self.read_start + self.payload_len;
        &mut self.storage_bytes_mut()[code_start..code_end]
    }
}

pub(crate) struct InvertedListPayload {
    pub(crate) list_id: usize,
    pub(crate) ids: Vec<i64>,
    payload: AlignedCodePayload,
}

impl InvertedListPayload {
    fn empty(list_id: usize) -> Self {
        Self {
            list_id,
            ids: Vec::new(),
            payload: AlignedCodePayload::empty(),
        }
    }

    pub(crate) fn codes(&self) -> &[u8] {
        self.payload.codes()
    }

    fn into_public(self) -> InvertedListData {
        let codes = self.codes().to_vec();
        InvertedListData {
            list_id: self.list_id,
            ids: self.ids,
            codes,
        }
    }
}

#[derive(Clone, Copy)]
struct BatchedListRead {
    input_idx: usize,
    list_id: usize,
    count: usize,
    offset: u64,
    id_bytes_len: i32,
}

fn decode_delta_list_payload(
    list_id: usize,
    payload: AlignedCodePayload,
    count: usize,
    id_bytes_len_from_table: i32,
) -> io::Result<InvertedListPayload> {
    let id_bytes_len = id_bytes_len_from_table as usize;
    let header_len = 12usize.checked_add(id_bytes_len).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "inverted list payload size overflow",
        )
    })?;
    let bytes = payload.read_bytes();
    if bytes.len() < header_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated delta inverted list payload",
        ));
    }
    let base_id = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let encoded_id_bytes_len = i32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if encoded_id_bytes_len != id_bytes_len_from_table {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "offset table id_bytes_len {} does not match list header {}",
                id_bytes_len_from_table, encoded_id_bytes_len
            ),
        ));
    }
    let id_bytes = &bytes[12..header_len];
    let ids = decode_delta_varint_ids(base_id, id_bytes, count)?;
    Ok(InvertedListPayload {
        list_id,
        ids,
        payload,
    })
}

#[allow(dead_code)]
fn compute_precomputed_table(
    centroids: &[f32],
    pq: &ProductQuantizer,
    nlist: usize,
    d: usize,
) -> Vec<f32> {
    let m = pq.m;
    let ksub = pq.ksub;
    let dsub = pq.dsub;
    let table_size = nlist * m * ksub;
    let mut table = vec![0.0f32; table_size];

    let pq_norms = pq.compute_centroid_norms();
    table
        .par_chunks_mut(m * ksub)
        .enumerate()
        .for_each(|(i, list_table)| {
            let centroid = &centroids[i * d..(i + 1) * d];

            for sub in 0..m {
                let sub_centroid = &centroid[sub * dsub..(sub + 1) * dsub];
                let pq_base = sub * ksub * dsub;

                for j in 0..ksub {
                    let pq_off = pq_base + j * dsub;
                    let mut ip = 0.0f32;
                    for dd in 0..dsub {
                        ip += sub_centroid[dd] * pq.centroids[pq_off + dd];
                    }
                    list_table[sub * ksub + j] = pq_norms[sub * ksub + j] + 2.0 * ip;
                }
            }
        });

    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct ReadStats {
        pread_calls: usize,
    }

    struct CountingPreadCursor {
        inner: Cursor<Vec<u8>>,
        stats: Arc<Mutex<ReadStats>>,
    }

    impl CountingPreadCursor {
        fn new(data: Vec<u8>, stats: Arc<Mutex<ReadStats>>) -> Self {
            CountingPreadCursor {
                inner: Cursor::new(data),
                stats,
            }
        }
    }

    impl SeekRead for CountingPreadCursor {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            for range in ranges {
                self.stats.lock().unwrap().pread_calls += 1;
                let old_pos = io::Seek::stream_position(&mut self.inner)?;
                io::Seek::seek(&mut self.inner, io::SeekFrom::Start(range.pos))?;
                let result = io::Read::read_exact(&mut self.inner, range.buf);
                io::Seek::seek(&mut self.inner, io::SeekFrom::Start(old_pos))?;
                result?;
            }
            Ok(())
        }
    }

    #[test]
    fn blocked_pq_transpose_matches_reference_across_tile_edges() {
        for count in [0usize, 1, 31, 32, 33, 67] {
            for code_size in [1usize, 25, 32, 240] {
                let codes = (0..count * code_size)
                    .map(|offset| ((offset * 131 + 17) % 251) as u8)
                    .collect::<Vec<_>>();
                let mut order = (0..count).collect::<Vec<_>>();
                if count > 1 {
                    order.rotate_left(count / 3);
                    order.reverse();
                }

                let mut reference = Vec::with_capacity(count * code_size);
                for column in 0..code_size {
                    for &source_row in &order {
                        reference.push(codes[source_row * code_size + column]);
                    }
                }

                let mut transposed = Vec::new();
                transpose_sorted_pq_codes(&codes, &order, code_size, &mut transposed);
                assert_eq!(
                    transposed, reference,
                    "count={count}, code_size={code_size}"
                );
            }
        }
    }

    #[test]
    fn test_varint_roundtrip() {
        let ids = [0, 127, 128, 16_383, 1_000_000];
        let (base, encoded) = encode_delta_varint_ids(&ids);
        assert_eq!(
            decode_delta_varint_ids(base, &encoded, ids.len()).unwrap(),
            ids
        );
    }

    #[test]
    fn test_varint_above_u64_max_returns_error() {
        let mut bytes = vec![0xFFu8; 9];
        bytes.push(0x02); // 10th byte with payload > 1 at shift=63
        assert!(decode_delta_varint_ids(0, &bytes, 1).is_err());
    }

    #[test]
    fn test_delta_varint_ids_roundtrip() {
        let ids = vec![3i64, 7, 12, 15, 23, 100, 200];
        let (base, encoded) = encode_delta_varint_ids(&ids);
        let decoded = decode_delta_varint_ids(base, &encoded, ids.len()).unwrap();
        assert_eq!(decoded, ids);
        // Delta-varint should be much smaller than raw int64
        assert!(encoded.len() < ids.len() * 8);
    }

    #[test]
    fn test_write_read_roundtrip_delta_ids() {
        let d = 8;
        let nlist = 2;
        let m = 2;

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        let n = 300;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        index.train(&data, n);
        index.add(&data, &ids, n);

        // Write with delta-varint IDs
        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let mut cursor = Cursor::new(&buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();
        assert_eq!(reader.total_vectors, n as i64);

        // Read each list and verify IDs are sorted
        for list_id in 0..nlist {
            let (ids, _) = reader.read_inverted_list(list_id).unwrap();
            for i in 1..ids.len() {
                assert!(ids[i] >= ids[i - 1], "IDs not sorted in list {}", list_id);
            }
        }
    }

    #[test]
    fn ivfpq_streamed_list_reader_matches_full_transposed_payload() {
        let d = 8;
        let m = 2;
        let n = 300;
        let mut index = IVFPQIndex::new(d, 1, m, MetricType::L2, false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(43);
        let data = (0..n * d).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
        let ids = (0..n as i64).collect::<Vec<_>>();
        index.train(&data, n);
        index.add(&data, &ids, n);
        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let mut full_reader = IVFPQIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let expected = full_reader.read_inverted_list(0).unwrap();
        let mut streamed_reader = IVFPQIndexReader::open(Cursor::new(bytes)).unwrap();
        streamed_reader.ensure_loaded().unwrap();
        let mut actual_ids = Vec::new();
        let mut actual_codes = Vec::new();
        streamed_reader
            .for_each_streamed_list_chunk(0, |_, ids, codes| {
                actual_ids.extend_from_slice(ids);
                actual_codes.extend_from_slice(codes);
            })
            .unwrap();
        assert_eq!(actual_ids, expected.0);
        assert_eq!(actual_codes, expected.1);
    }

    #[test]
    fn test_ivfpq_search_payload_uses_one_pread_and_aligned_codes() {
        let d = 8;
        let nlist = 2;
        let m = 2;

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        let n = 300;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let stats = Arc::new(Mutex::new(ReadStats::default()));
        let stream = CountingPreadCursor::new(buf, Arc::clone(&stats));
        let mut reader = IVFPQIndexReader::open(stream).unwrap();
        reader.ensure_loaded().unwrap();

        {
            let mut stats = stats.lock().unwrap();
            stats.pread_calls = 0;
        }

        let non_empty_list = reader
            .list_counts
            .iter()
            .position(|&count| count > 0)
            .unwrap();
        assert!(
            reader.list_id_bytes_lens[non_empty_list] > 0,
            "v1 files must store id_bytes_len in the offset table"
        );
        let expected_requested_bytes = reader.list_payload_len(non_empty_list).unwrap();
        reader.begin_read_metrics();
        let mut lists = reader
            .read_inverted_list_payloads(&[non_empty_list])
            .unwrap();
        let read_metrics = reader.end_read_metrics();
        let list = lists.pop().unwrap();
        let read_ids = &list.ids;
        let codes = list.codes();

        assert!(!read_ids.is_empty());
        assert!(!codes.is_empty());
        assert_eq!(
            codes.as_ptr().align_offset(AlignedCodePayload::ALIGNMENT),
            0,
            "the transposed scan should not lose code alignment after the ID prefix"
        );

        let stats = stats.lock().unwrap();
        assert_eq!(
            stats.pread_calls, 1,
            "delta-varint lists with offset-table id length should use one pread"
        );
        assert_eq!(read_metrics.calls, 1);
        assert_eq!(read_metrics.requested_bytes, expected_requested_bytes);
    }

    #[test]
    fn test_ivfpq_open_coalesces_resident_metadata() {
        let d = 8;
        let nlist = 16;
        let m = 2;
        let n = 512;
        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(91);
        let data = (0..n * d).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
        let ids = (0..n as i64).collect::<Vec<_>>();
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut bytes = Vec::new();
        write_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let stats = Arc::new(Mutex::new(ReadStats::default()));
        let source = CountingPreadCursor::new(bytes, Arc::clone(&stats));
        let mut reader = IVFPQIndexReader::open(source).unwrap();
        reader.ensure_loaded().unwrap();
        assert_eq!(
            stats.lock().unwrap().pread_calls,
            2,
            "direct IVF-PQ open should use one header read and one resident-metadata read"
        );
    }

    #[test]
    fn test_default_pread_handles_multiple_ranges() {
        let mut cursor = Cursor::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let mut first = [0u8; 2];
        let mut second = [0u8; 3];

        cursor
            .pread(&mut [
                ReadRequest::new(2, &mut first),
                ReadRequest::new(5, &mut second),
            ])
            .unwrap();

        assert_eq!(first, [2, 3]);
        assert_eq!(second, [5, 6, 7]);
    }

    #[test]
    fn test_write_read_4bit() {
        let d = 16;
        let nlist = 4;
        let m = 8;

        let mut index = IVFPQIndex::with_nbits(d, nlist, m, 4, MetricType::L2, false);
        let n = 500;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        index.train(&data, n);
        index.add(&data, &ids, n);
        assert_eq!(index.pq.code_size(), m / 2);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_index(&index, &mut writer).unwrap();

        let mut cursor = Cursor::new(&buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();
        assert_eq!(reader.pq.nbits, 4);
        assert_eq!(reader.pq.code_size(), m / 2);

        let (result_ids, result_dists) = reader.search(&data[0..d], 5, 4).unwrap();
        assert!(!result_ids.is_empty());
        assert!(result_ids.contains(&0));
        for i in 1..result_dists.len() {
            assert!(result_dists[i] >= result_dists[i - 1]);
        }
    }

    #[test]
    #[should_panic(expected = "4-bit IVF-PQ requires even m")]
    fn construction_rejects_odd_4bit_subquantizer_count() {
        let _ = IVFPQIndex::with_nbits(15, 4, 5, 4, MetricType::L2, false);
    }

    #[test]
    #[ignore]
    fn test_space_savings() {
        let d = 128;
        let nlist = 64;
        let m = 16;
        let n = 100_000;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // Clustered data for realistic IVF distribution
        let num_clusters = 64;
        let mut centers = vec![0.0f32; num_clusters * d];
        for v in centers.iter_mut() {
            *v = rng.gen::<f32>() * 100.0;
        }
        let data: Vec<f32> = (0..n * d)
            .map(|i| {
                let cluster = (i / d) % num_clusters;
                centers[cluster * d + i % d] + rng.gen::<f32>() * 2.0 - 1.0
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFPQIndex::new(d, nlist, m, MetricType::L2, false);
        index.train(&data, n);
        index.add(&data, &ids, n);

        let mut delta_buf = Vec::new();
        let mut delta_writer = PosWriter::new(&mut delta_buf);
        write_index(&index, &mut delta_writer).unwrap();

        let delta_size = delta_buf.len();

        // Compute ID-only sizes for clearer comparison
        let total_id_bytes_raw = n * 8;
        let total_id_bytes_delta: usize = (0..nlist)
            .map(|i| {
                let count = index.ids[i].len();
                if count == 0 {
                    0
                } else {
                    let mut sorted: Vec<i64> = index.ids[i].clone();
                    sorted.sort();
                    let (_, encoded) = encode_delta_varint_ids(&sorted);
                    8 + 4 + encoded.len() // base_id + len + data
                }
            })
            .sum();
        let total_id_savings_pct =
            (1.0 - total_id_bytes_delta as f64 / total_id_bytes_raw as f64) * 100.0;

        eprintln!("=== Space Benchmark: 100K vectors, d=128, M=16, nlist=64 ===");
        eprintln!(
            "Raw int64 IDs:     {} bytes ({:.1} KB)",
            total_id_bytes_raw,
            total_id_bytes_raw as f64 / 1024.0
        );
        eprintln!(
            "Delta-varint IDs:  {} bytes ({:.1} KB)",
            total_id_bytes_delta,
            total_id_bytes_delta as f64 / 1024.0
        );
        eprintln!(
            "ID compression:    {:.1}x ({:.1}% saved)",
            total_id_bytes_raw as f64 / total_id_bytes_delta as f64,
            (1.0 - total_id_bytes_delta as f64 / total_id_bytes_raw as f64) * 100.0
        );
        eprintln!();
        eprintln!(
            "Total file (delta):{} bytes ({:.1} KB)",
            delta_size,
            delta_size as f64 / 1024.0
        );
        eprintln!("ID savings:        {:.1}%", total_id_savings_pct);

        assert!(
            total_id_savings_pct > 70.0,
            "Expected >70% ID savings, got {:.1}%",
            total_id_savings_pct
        );

        // Verify search still works with delta-varint format
        let mut cursor = Cursor::new(&delta_buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();
        let (result_ids, result_dists) = reader.search(&data[0..d], 10, 8).unwrap();
        assert!(!result_ids.is_empty());
        assert!(result_ids.contains(&0));
        for i in 1..result_dists.len() {
            assert!(result_dists[i] >= result_dists[i - 1]);
        }
    }

    #[test]
    fn test_corrupt_delta_ids_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes()); // d
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&1i32.to_le_bytes()); // m
        buf.extend_from_slice(&256i32.to_le_bytes()); // ksub
        buf.extend_from_slice(&4i32.to_le_bytes()); // dsub
        buf.extend_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf.extend_from_slice(&1i64.to_le_bytes()); // total_vectors
        let flags = FLAG_DELTA_IDS | FLAG_TRANSPOSED_CODES | FLAG_BY_RESIDUAL;
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]); // padding

        buf.extend_from_slice(&[0u8; 16]); // quantizer centroids (nlist=1, d=4)
        buf.extend_from_slice(&vec![0u8; 256 * 4 * 4]); // pq centroids (m=1, ksub=256, dsub=4)

        // Offset table: one list
        let list_data_offset = buf.len() as i64 + 16; // after 16 bytes of offset entry
        buf.extend_from_slice(&list_data_offset.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes()); // count=1
        buf.extend_from_slice(&0i32.to_le_bytes()); // padding

        // List data: base_id + id_bytes_len=0 (truncated — not enough varints for count=1)
        buf.extend_from_slice(&123i64.to_le_bytes()); // base_id
        buf.extend_from_slice(&0i32.to_le_bytes()); // id_bytes_len = 0, but count=1

        let mut cursor = Cursor::new(&buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();
        let result = reader.read_inverted_list(0);
        assert!(
            result.is_err(),
            "should return error on truncated delta IDs"
        );
    }

    #[test]
    fn test_negative_id_bytes_len_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes()); // d
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&1i32.to_le_bytes()); // m
        buf.extend_from_slice(&256i32.to_le_bytes()); // ksub
        buf.extend_from_slice(&4i32.to_le_bytes()); // dsub
        buf.extend_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf.extend_from_slice(&1i64.to_le_bytes()); // total_vectors
        let flags = FLAG_DELTA_IDS | FLAG_TRANSPOSED_CODES | FLAG_BY_RESIDUAL;
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]); // padding

        buf.extend_from_slice(&[0u8; 16]); // quantizer centroids
        buf.extend_from_slice(&vec![0u8; 256 * 4 * 4]); // pq centroids

        let list_data_offset = buf.len() as i64 + 16;
        buf.extend_from_slice(&list_data_offset.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes()); // count=1
        buf.extend_from_slice(&0i32.to_le_bytes()); // padding

        buf.extend_from_slice(&0i64.to_le_bytes()); // base_id
        buf.extend_from_slice(&(-1i32).to_le_bytes()); // negative id_bytes_len

        let mut cursor = Cursor::new(&buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();
        let result = reader.read_inverted_list(0);
        assert!(
            result.is_err(),
            "negative id_bytes_len should return error, not panic"
        );
    }

    #[test]
    fn test_large_gap_ids_roundtrip() {
        let ids = vec![i64::MIN, 0, i64::MAX];
        let (base, encoded) = encode_delta_varint_ids(&ids);
        let decoded = decode_delta_varint_ids(base, &encoded, ids.len()).unwrap();
        assert_eq!(decoded, ids);
    }

    #[test]
    fn test_delta_ids_wraparound_returns_error() {
        // base_id = i64::MAX, delta = 1 would wrap to i64::MIN (non-monotonic)
        let (_, id_bytes) = encode_delta_varint_ids(&[i64::MAX, i64::MIN]);
        let id_bytes = id_bytes[1..].to_vec();
        let result = decode_delta_varint_ids(i64::MAX, &id_bytes, 1);
        assert!(
            result.is_err(),
            "wrapped delta IDs should be rejected as non-monotonic"
        );
    }

    #[test]
    fn test_negative_list_count_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes()); // d
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&1i32.to_le_bytes()); // m
        buf.extend_from_slice(&256i32.to_le_bytes()); // ksub
        buf.extend_from_slice(&4i32.to_le_bytes()); // dsub
        buf.extend_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf.extend_from_slice(&1i64.to_le_bytes()); // total_vectors
        let flags = FLAG_DELTA_IDS | FLAG_TRANSPOSED_CODES | FLAG_BY_RESIDUAL;
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]); // padding
        buf.extend_from_slice(&[0u8; 16]); // quantizer centroids
        buf.extend_from_slice(&vec![0u8; 256 * 4 * 4]); // pq centroids

        // Offset table with negative count
        buf.extend_from_slice(&0i64.to_le_bytes()); // offset
        buf.extend_from_slice(&(-1i32).to_le_bytes()); // negative count
        buf.extend_from_slice(&0i32.to_le_bytes()); // padding

        let mut cursor = Cursor::new(&buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();
        let result = reader.ensure_loaded();
        assert!(
            result.is_err(),
            "negative list count should return error, not panic"
        );
    }

    #[test]
    fn test_negative_header_d_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(-1i32).to_le_bytes()); // invalid d
                                                       // remaining header fields don't matter — open should fail
        buf.extend_from_slice(&[0u8; 64 - 12]);

        let mut cursor = Cursor::new(&buf);
        let result = IVFPQIndexReader::open(&mut cursor);
        assert!(result.is_err(), "negative d should return error");
    }

    #[test]
    fn test_negative_header_nlist_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes()); // d
        buf.extend_from_slice(&(-1i32).to_le_bytes()); // invalid nlist
        buf.extend_from_slice(&[0u8; 64 - 16]);

        let mut cursor = Cursor::new(&buf);
        let result = IVFPQIndexReader::open(&mut cursor);
        assert!(result.is_err(), "negative nlist should return error");
    }

    #[test]
    fn test_huge_pq_section_size_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        // m=10000, ksub=256, dsub=10000 → m*ksub*dsub = 2.56 billion > MAX_SECTION_ELEMENTS
        // d = m*dsub = 100_000_000
        buf.extend_from_slice(&100_000_000i32.to_le_bytes()); // d
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&10_000i32.to_le_bytes()); // m
        buf.extend_from_slice(&256i32.to_le_bytes()); // ksub (valid)
        buf.extend_from_slice(&10_000i32.to_le_bytes()); // dsub
        buf.extend_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        let flags = FLAG_DELTA_IDS | FLAG_TRANSPOSED_CODES | FLAG_BY_RESIDUAL;
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        let mut cursor = Cursor::new(&buf);
        let mut reader = IVFPQIndexReader::open(&mut cursor).unwrap();
        let result = reader.ensure_loaded();
        assert!(
            result.is_err(),
            "huge m*ksub*dsub should return error, not panic"
        );
    }

    #[test]
    fn test_huge_opq_offset_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&i32::MAX.to_le_bytes()); // huge d
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&1i32.to_le_bytes()); // m
        buf.extend_from_slice(&256i32.to_le_bytes()); // ksub
        buf.extend_from_slice(&1i32.to_le_bytes()); // dsub
        buf.extend_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        let flags = FLAG_HAS_OPQ | FLAG_DELTA_IDS | FLAG_TRANSPOSED_CODES | FLAG_BY_RESIDUAL;
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        let mut cursor = Cursor::new(&buf);
        let result = IVFPQIndexReader::open(&mut cursor);
        assert!(
            result.is_err(),
            "huge d*d OPQ offset should return error, not panic"
        );
    }

    #[test]
    fn test_unsupported_ksub_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes()); // d
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&1i32.to_le_bytes()); // m
        buf.extend_from_slice(&3i32.to_le_bytes()); // ksub=3, unsupported
        buf.extend_from_slice(&4i32.to_le_bytes()); // dsub
        buf.extend_from_slice(&[0u8; 64 - 7 * 4]);

        let mut cursor = Cursor::new(&buf);
        let result = IVFPQIndexReader::open(&mut cursor);
        assert!(result.is_err(), "unsupported ksub should return error");
    }

    #[test]
    fn test_missing_required_flags_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes()); // d
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&1i32.to_le_bytes()); // m
        buf.extend_from_slice(&256i32.to_le_bytes()); // ksub
        buf.extend_from_slice(&4i32.to_le_bytes()); // dsub
        buf.extend_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        buf.extend_from_slice(&FLAG_BY_RESIDUAL.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        let mut cursor = Cursor::new(&buf);
        let err = match IVFPQIndexReader::open(&mut cursor) {
            Ok(_) => panic!("missing required flags should be rejected"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("requires delta IDs and transposed codes"));
    }

    #[test]
    fn test_unknown_flags_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes()); // d
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&1i32.to_le_bytes()); // m
        buf.extend_from_slice(&256i32.to_le_bytes()); // ksub
        buf.extend_from_slice(&4i32.to_le_bytes()); // dsub
        buf.extend_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        let flags = REQUIRED_FLAGS | (1 << 31);
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        let mut cursor = Cursor::new(&buf);
        let err = match IVFPQIndexReader::open(&mut cursor) {
            Ok(_) => panic!("unknown flags should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Unsupported IVFPQ flags"));
    }

    #[test]
    fn test_nonzero_reserved_bytes_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes());
        buf.extend_from_slice(&256i32.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes());
        buf.extend_from_slice(&(MetricType::L2 as u32).to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());
        buf.extend_from_slice(&REQUIRED_FLAGS.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        buf[44] = 1;

        let mut cursor = Cursor::new(&buf);
        let err = match IVFPQIndexReader::open(&mut cursor) {
            Ok(_) => panic!("non-zero reserved bytes should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("reserved bytes must be zero"));
    }

    #[test]
    fn test_d_not_equal_m_times_dsub_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&4i32.to_le_bytes()); // d=4
        buf.extend_from_slice(&1i32.to_le_bytes()); // nlist
        buf.extend_from_slice(&3i32.to_le_bytes()); // m=3, d != m*dsub
        buf.extend_from_slice(&256i32.to_le_bytes()); // ksub
        buf.extend_from_slice(&1i32.to_le_bytes()); // dsub=1, m*dsub=3 != d=4
        buf.extend_from_slice(&[0u8; 64 - 7 * 4]);

        let mut cursor = Cursor::new(&buf);
        let result = IVFPQIndexReader::open(&mut cursor);
        assert!(result.is_err(), "d != m*dsub should return error");
    }
}
