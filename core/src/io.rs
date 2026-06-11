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
use crate::ivfpq::IVFPQIndex;
use crate::opq::OPQMatrix;
use crate::pq::ProductQuantizer;
use std::io;

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

pub trait SeekRead: Send {
    /// Positional reads for one or more ranges.
    ///
    /// Implementations may execute requests sequentially, coalesce them, or issue
    /// them concurrently when the underlying source supports independent
    /// positional reads.
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()>;
}

pub(crate) struct PreadCursor<'a, R: SeekRead + ?Sized> {
    reader: &'a mut R,
    pos: u64,
}

impl<'a, R: SeekRead + ?Sized> PreadCursor<'a, R> {
    pub(crate) fn new(reader: &'a mut R, pos: u64) -> Self {
        Self { reader, pos }
    }

    pub(crate) fn seek(&mut self, pos: u64) {
        self.pos = pos;
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

// --- Varint encoding ---

fn encode_varint(mut val: u64, buf: &mut Vec<u8>) {
    while val >= 0x80 {
        buf.push((val as u8) | 0x80);
        val >>= 7;
    }
    buf.push(val as u8);
}

fn decode_varint(buf: &[u8], pos: &mut usize) -> io::Result<u64> {
    let mut val: u64 = 0;
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

/// Encode sorted i64 IDs as delta-varint. Returns (base_id, encoded_bytes).
/// Uses unsigned subtraction to handle the full i64 range without overflow.
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

/// Decode delta-varint encoded IDs using wrapping unsigned arithmetic
/// (inverse of encode_delta_varint_ids). Validates monotonically non-decreasing
/// signed order — rejects corrupt data that would wrap around.
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

fn read_f32_vec<R: SeekRead + ?Sized>(
    reader: &mut PreadCursor<'_, R>,
    count: usize,
) -> io::Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    reader.read_exact(&mut buf)?;
    let floats: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(floats)
}

/// Write a complete IVF-PQ index with delta-varint ID encoding.
pub fn write_index(index: &IVFPQIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    let d = index.d;
    let nlist = index.nlist;
    let m = index.pq.m;
    let ksub = index.pq.ksub;
    let dsub = index.pq.dsub;
    let code_size = index.pq.code_size();
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
    let mut sorted_lists: Vec<(Vec<i64>, Vec<u8>, Vec<u8>)> = Vec::with_capacity(nlist);
    for i in 0..nlist {
        let count = index.ids[i].len();
        if count == 0 {
            sorted_lists.push((Vec::new(), Vec::new(), Vec::new()));
            continue;
        }

        // Sort by ID, reorder codes accordingly
        let mut indices: Vec<usize> = (0..count).collect();
        indices.sort_by_key(|&idx| index.ids[i][idx]);

        let sorted_ids: Vec<i64> = indices.iter().map(|&idx| index.ids[i][idx]).collect();
        let code_bytes = checked_list_bytes(count, code_size)?;
        let mut sorted_codes = vec![0u8; code_bytes];
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_codes[new_idx * code_size..(new_idx + 1) * code_size]
                .copy_from_slice(&index.codes[i][old_idx * code_size..(old_idx + 1) * code_size]);
        }

        let (_, id_bytes) = encode_delta_varint_ids(&sorted_ids);
        sorted_lists.push((sorted_ids, id_bytes, sorted_codes));
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

    write_f32_slice(out, &index.quantizer_centroids)?;
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
        let count = sorted_lists[i].0.len();
        list_counts[i] = usize_to_i32(count, "list count")?;
        if count > 0 {
            // base_id(8) + id_bytes_len(4) + id_bytes + codes
            let id_bytes_len = sorted_lists[i].1.len();
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
    for i in 0..nlist {
        let (ref sorted_ids, ref id_bytes, ref sorted_codes) = sorted_lists[i];
        if sorted_ids.is_empty() {
            continue;
        }
        // base_id
        write_i64_le(out, sorted_ids[0])?;
        // id_bytes_len + id_bytes
        write_i32_le(out, usize_to_i32(id_bytes.len(), "delta ID section")?)?;
        out.write_all(id_bytes)?;
        // PQ codes — transpose for cache-friendly SIMD scan
        let count = sorted_ids.len();
        if code_size == m {
            // 8-bit: transpose from [n][M] to [M][n]
            let transposed_len = checked_list_bytes(count, m)?;
            let mut transposed = vec![0u8; transposed_len];
            for vec_idx in 0..count {
                for sub in 0..m {
                    transposed[sub * count + vec_idx] = sorted_codes[vec_idx * m + sub];
                }
            }
            out.write_all(&transposed)?;
        } else {
            // 4-bit: transpose from [n][M/2] to [M/2][n]
            // Each byte at position `pair` in a vector goes to column `pair`
            let cs = code_size;
            let transposed_len = checked_list_bytes(count, cs)?;
            let mut transposed = vec![0u8; transposed_len];
            for vec_idx in 0..count {
                for pair in 0..cs {
                    transposed[pair * count + vec_idx] = sorted_codes[vec_idx * cs + pair];
                }
            }
            out.write_all(&transposed)?;
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

// --- Reader ---

pub struct IVFPQIndexReader<R: SeekRead> {
    reader: R,
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
    /// File offset where centroids section starts (for lazy loading)
    centroids_offset: u64,
    /// Whether file has OPQ rotation matrix
    has_opq: bool,
}

impl<R: SeekRead> IVFPQIndexReader<R> {
    /// Open an index file. Only reads the 64-byte header.
    /// Centroids, codebooks, and offset table are loaded lazily on first search.
    pub fn open(mut reader: R) -> io::Result<Self> {
        let mut cursor = PreadCursor::new(&mut reader, 0);

        let magic = read_u32_le(&mut cursor)?;
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVFPQ magic: 0x{:08X}", magic),
            ));
        }

        let version = read_u32_le(&mut cursor)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVFPQ version: {}", version),
            ));
        }

        let d = validate_positive_i32(read_i32_le(&mut cursor)?, "d")? as usize;
        let nlist = validate_positive_i32(read_i32_le(&mut cursor)?, "nlist")? as usize;
        let m = validate_positive_i32(read_i32_le(&mut cursor)?, "m")? as usize;
        let ksub = validate_positive_i32(read_i32_le(&mut cursor)?, "ksub")? as usize;
        let dsub = validate_positive_i32(read_i32_le(&mut cursor)?, "dsub")? as usize;

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

        let metric_code = read_u32_le(&mut cursor)?;
        let metric = MetricType::from_code(metric_code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown metric type: {}", metric_code),
            )
        })?;
        let total_vectors = read_i64_le(&mut cursor)?;

        let flags = read_u32_le(&mut cursor)?;
        let mut skip = [0u8; 20];
        cursor.read_exact(&mut skip)?;
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
        let centroids_offset = if has_opq {
            let opq_elements = checked_section_size(d, d)?;
            HEADER_SIZE as u64 + (opq_elements * 4) as u64
        } else {
            HEADER_SIZE as u64
        };

        Ok(IVFPQIndexReader {
            reader,
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
                centroids: Vec::new(),
                centroid_norms_cache: Vec::new(),
            },
            list_offsets: Vec::new(),
            list_counts: Vec::new(),
            list_id_bytes_lens: Vec::new(),
            precomputed_table: Vec::new(),
            transposed_codes,
            loaded: false,
            centroids_offset,
            has_opq,
        })
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

        // Seek to start of data sections
        let mut cursor = PreadCursor::new(&mut self.reader, self.centroids_offset);
        if self.has_opq {
            cursor.seek(HEADER_SIZE as u64);
            let rotation = read_f32_vec(&mut cursor, rotation_count)?;
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

        self.quantizer_centroids = read_f32_vec(&mut cursor, centroids_count)?;

        let pq_centroids = read_f32_vec(&mut cursor, pq_centroids_count)?;
        self.pq = ProductQuantizer {
            d,
            m,
            nbits: ksub.trailing_zeros() as usize,
            dsub,
            ksub,
            centroids: pq_centroids,
            centroid_norms_cache: Vec::new(),
        };
        self.pq.rebuild_norms_cache();

        self.list_offsets = vec![0i64; nlist];
        self.list_counts = vec![0i32; nlist];
        self.list_id_bytes_lens = vec![0i32; nlist];
        for i in 0..nlist {
            self.list_offsets[i] = read_i64_le(&mut cursor)?;
            let count = read_i32_le(&mut cursor)?;
            if count < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative list count {} at list {}", count, i),
                ));
            }
            self.list_counts[i] = count;
            let id_bytes_len = read_i32_le(&mut cursor)?;
            if id_bytes_len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative id_bytes_len {} at list {}", id_bytes_len, i),
                ));
            }
            self.list_id_bytes_lens[i] = id_bytes_len;
        }

        self.loaded = true;
        Ok(())
    }

    /// Read an inverted list's IDs and PQ codes.
    /// Calls ensure_loaded() if not yet loaded.
    pub fn read_inverted_list(&mut self, list_id: usize) -> io::Result<(Vec<i64>, Vec<u8>)> {
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
        let code_size = self.pq.code_size();
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
        let mut payload = vec![0u8; payload_len];
        self.reader
            .pread(&mut [ReadRequest::new(offset, &mut payload)])?;
        decode_delta_list_payload(&payload, count, id_bytes_len)
    }

    /// Read multiple inverted lists. Lists whose payload length is known from
    /// metadata are issued through a single batched pread call.
    pub fn read_inverted_lists(&mut self, list_ids: &[usize]) -> io::Result<Vec<InvertedListData>> {
        self.ensure_loaded()?;

        let code_size = self.pq.code_size();
        let mut results: Vec<Option<InvertedListData>> =
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
                results[input_idx] = Some(InvertedListData {
                    list_id,
                    ids: Vec::new(),
                    codes: Vec::new(),
                });
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
            payloads.push(vec![0u8; payload_len]);
        }

        if !metas.is_empty() {
            {
                let mut requests: Vec<_> = payloads
                    .iter_mut()
                    .zip(metas.iter())
                    .map(|(payload, meta)| ReadRequest::new(meta.offset, payload.as_mut_slice()))
                    .collect();
                self.reader.pread(&mut requests)?;
            }

            for (meta, payload) in metas.into_iter().zip(payloads) {
                let (ids, codes) =
                    decode_delta_list_payload(&payload, meta.count, meta.id_bytes_len)?;
                results[meta.input_idx] = Some(InvertedListData {
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
                        "missing batched inverted list read result",
                    )
                })
            })
            .collect()
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

#[derive(Clone, Copy)]
struct BatchedListRead {
    input_idx: usize,
    list_id: usize,
    count: usize,
    offset: u64,
    id_bytes_len: i32,
}

fn decode_delta_list_payload(
    payload: &[u8],
    count: usize,
    id_bytes_len_from_table: i32,
) -> io::Result<(Vec<i64>, Vec<u8>)> {
    let id_bytes_len = id_bytes_len_from_table as usize;
    let header_len = 12usize.checked_add(id_bytes_len).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "inverted list payload size overflow",
        )
    })?;
    if payload.len() < header_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated delta inverted list payload",
        ));
    }
    let base_id = i64::from_le_bytes(payload[0..8].try_into().unwrap());
    let encoded_id_bytes_len = i32::from_le_bytes(payload[8..12].try_into().unwrap());
    if encoded_id_bytes_len != id_bytes_len_from_table {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "offset table id_bytes_len {} does not match list header {}",
                id_bytes_len_from_table, encoded_id_bytes_len
            ),
        ));
    }
    let id_bytes = &payload[12..header_len];
    let ids = decode_delta_varint_ids(base_id, id_bytes, count)?;
    let codes = payload[header_len..].to_vec();
    Ok((ids, codes))
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

    for i in 0..nlist {
        let centroid = &centroids[i * d..(i + 1) * d];
        let tab_base = i * m * ksub;

        for sub in 0..m {
            let sub_centroid = &centroid[sub * dsub..(sub + 1) * dsub];
            let pq_base = sub * ksub * dsub;

            for j in 0..ksub {
                let pq_off = pq_base + j * dsub;
                let mut ip = 0.0f32;
                for dd in 0..dsub {
                    ip += sub_centroid[dd] * pq.centroids[pq_off + dd];
                }
                table[tab_base + sub * ksub + j] = pq_norms[sub * ksub + j] + 2.0 * ip;
            }
        }
    }

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
    fn test_varint_roundtrip() {
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        encode_varint(127, &mut buf);
        encode_varint(128, &mut buf);
        encode_varint(16383, &mut buf);
        encode_varint(1_000_000, &mut buf);

        let mut pos = 0;
        assert_eq!(decode_varint(&buf, &mut pos).unwrap(), 0);
        assert_eq!(decode_varint(&buf, &mut pos).unwrap(), 127);
        assert_eq!(decode_varint(&buf, &mut pos).unwrap(), 128);
        assert_eq!(decode_varint(&buf, &mut pos).unwrap(), 16383);
        assert_eq!(decode_varint(&buf, &mut pos).unwrap(), 1_000_000);
    }

    #[test]
    fn test_varint_above_u64_max_returns_error() {
        let mut bytes = vec![0xFFu8; 9];
        bytes.push(0x02); // 10th byte with payload > 1 at shift=63
        let mut pos = 0;
        assert!(decode_varint(&bytes, &mut pos).is_err());
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
    fn test_read_inverted_list_uses_pread_after_metadata_loaded() {
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
        let (read_ids, codes) = reader.read_inverted_list(non_empty_list).unwrap();

        assert!(!read_ids.is_empty());
        assert!(!codes.is_empty());

        let stats = stats.lock().unwrap();
        assert_eq!(
            stats.pread_calls, 1,
            "delta-varint lists with offset-table id length should use one pread"
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
        let mut id_bytes = Vec::new();
        encode_varint(1, &mut id_bytes);
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
