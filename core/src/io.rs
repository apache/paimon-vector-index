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

pub trait SeekRead: Send {
    fn seek(&mut self, pos: u64) -> io::Result<()>;
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()>;

    /// Positional read: read `buf.len()` bytes at `pos` without changing the cursor.
    /// Thread-safe if the underlying implementation supports it (e.g., pread(2)).
    /// Default implementation falls back to seek + read_exact.
    fn pread(&mut self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        self.seek(pos)?;
        self.read_exact(buf)
    }

    /// Whether this implementation supports true concurrent pread (no shared cursor).
    fn supports_concurrent_pread(&self) -> bool {
        false
    }
}

pub trait SeekWrite: Send {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()>;
    fn pos(&self) -> u64;
}

impl<T: io::Read + io::Seek + Send> SeekRead for T {
    fn seek(&mut self, pos: u64) -> io::Result<()> {
        io::Seek::seek(self, io::SeekFrom::Start(pos))?;
        Ok(())
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        io::Read::read_exact(self, buf)
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

fn decode_varint(buf: &[u8], pos: &mut usize) -> u64 {
    let mut val: u64 = 0;
    let mut shift = 0;
    loop {
        let b = buf[*pos] as u64;
        *pos += 1;
        val |= (b & 0x7F) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    val
}

/// Encode sorted i64 IDs as delta-varint. Returns (base_id, encoded_bytes).
fn encode_delta_varint_ids(ids: &[i64]) -> (i64, Vec<u8>) {
    if ids.is_empty() {
        return (0, Vec::new());
    }
    let base = ids[0];
    let mut buf = Vec::with_capacity(ids.len() * 2);
    let mut prev = base;
    for &id in ids {
        let delta = (id - prev) as u64;
        encode_varint(delta, &mut buf);
        prev = id;
    }
    (base, buf)
}

/// Decode delta-varint encoded IDs.
fn decode_delta_varint_ids(base: i64, buf: &[u8], count: usize) -> Vec<i64> {
    let mut ids = Vec::with_capacity(count);
    let mut pos = 0;
    let mut current = base;
    for _ in 0..count {
        let delta = decode_varint(buf, &mut pos) as i64;
        current += delta;
        ids.push(current);
    }
    ids
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

fn read_f32_vec(reader: &mut dyn SeekRead, count: usize) -> io::Result<Vec<f32>> {
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

    let mut flags: u32 = FLAG_DELTA_IDS | FLAG_TRANSPOSED_CODES;
    if index.opq.is_some() {
        flags |= FLAG_HAS_OPQ;
    }
    if index.by_residual {
        flags |= FLAG_BY_RESIDUAL;
    }

    let total_vectors: i64 = index.ids.iter().map(|l| l.len() as i64).sum();

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
        let mut sorted_codes = vec![0u8; count * code_size];
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
    write_i32_le(out, d as i32)?;
    write_i32_le(out, nlist as i32)?;
    write_i32_le(out, m as i32)?;
    write_i32_le(out, ksub as i32)?;
    write_i32_le(out, dsub as i32)?;
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
    let offset_table_size = nlist * 16;
    let data_start = out.pos() + offset_table_size as u64;

    let mut list_offsets = vec![0i64; nlist];
    let mut list_counts = vec![0i32; nlist];
    let mut current_offset = data_start;

    for i in 0..nlist {
        list_offsets[i] = current_offset as i64;
        let count = sorted_lists[i].0.len();
        list_counts[i] = count as i32;
        if count > 0 {
            // base_id(8) + id_bytes_len(4) + id_bytes + codes
            let id_bytes_len = sorted_lists[i].1.len();
            current_offset += 8 + 4 + id_bytes_len as u64 + (count * code_size) as u64;
        }
    }

    // Write offset table
    for i in 0..nlist {
        write_i64_le(out, list_offsets[i])?;
        write_i32_le(out, list_counts[i])?;
        write_i32_le(out, 0)?;
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
        write_i32_le(out, id_bytes.len() as i32)?;
        out.write_all(id_bytes)?;
        // PQ codes — transpose for cache-friendly SIMD scan
        let count = sorted_ids.len();
        if code_size == m {
            // 8-bit: transpose from [n][M] to [M][n]
            let mut transposed = vec![0u8; count * m];
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
            let mut transposed = vec![0u8; count * cs];
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

/// Write index with raw int64 IDs (v1/v2 without FLAG_DELTA_IDS). For benchmarking.
pub fn write_index_raw_ids(index: &IVFPQIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    let d = index.d;
    let nlist = index.nlist;
    let m = index.pq.m;
    let ksub = index.pq.ksub;
    let dsub = index.pq.dsub;

    let mut flags: u32 = 0;
    if index.opq.is_some() {
        flags |= FLAG_HAS_OPQ;
    }
    if index.by_residual {
        flags |= FLAG_BY_RESIDUAL;
    }

    let total_vectors: i64 = index.ids.iter().map(|l| l.len() as i64).sum();

    write_u32_le(out, MAGIC)?;
    write_u32_le(out, VERSION)?;
    write_i32_le(out, d as i32)?;
    write_i32_le(out, nlist as i32)?;
    write_i32_le(out, m as i32)?;
    write_i32_le(out, ksub as i32)?;
    write_i32_le(out, dsub as i32)?;
    write_u32_le(out, index.metric as u32)?;
    write_i64_le(out, total_vectors)?;
    write_u32_le(out, flags)?;
    out.write_all(&[0u8; 20])?;

    if let Some(ref opq) = index.opq {
        write_f32_slice(out, &opq.rotation)?;
    }
    write_f32_slice(out, &index.quantizer_centroids)?;
    write_f32_slice(out, &index.pq.centroids)?;

    let offset_table_size = nlist * 16;
    let data_start = out.pos() + offset_table_size as u64;
    let mut list_offsets = vec![0i64; nlist];
    let mut list_counts = vec![0i32; nlist];
    let mut current_offset = data_start;
    for i in 0..nlist {
        list_offsets[i] = current_offset as i64;
        let count = index.ids[i].len();
        list_counts[i] = count as i32;
        let cs = index.pq.code_size();
        current_offset += (count * 8 + count * cs) as u64;
    }
    for i in 0..nlist {
        write_i64_le(out, list_offsets[i])?;
        write_i32_le(out, list_counts[i])?;
        write_i32_le(out, 0)?;
    }
    for i in 0..nlist {
        for &id in &index.ids[i] {
            write_i64_le(out, id)?;
        }
        out.write_all(&index.codes[i])?;
    }

    Ok(())
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
    pub precomputed_table: Vec<f32>,
    delta_ids: bool,
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
        reader.seek(0)?;

        let magic = read_u32_le(&mut reader)?;
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVFPQ magic: 0x{:08X}", magic),
            ));
        }

        let version = read_u32_le(&mut reader)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVFPQ version: {}", version),
            ));
        }

        let d = read_i32_le(&mut reader)? as usize;
        let nlist = read_i32_le(&mut reader)? as usize;
        let m = read_i32_le(&mut reader)? as usize;
        let ksub = read_i32_le(&mut reader)? as usize;
        let dsub = read_i32_le(&mut reader)? as usize;
        let metric_code = read_u32_le(&mut reader)?;
        let metric = MetricType::from_code(metric_code).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown metric type: {}", metric_code),
            )
        })?;
        let total_vectors = read_i64_le(&mut reader)?;

        let flags = read_u32_le(&mut reader)?;
        let mut skip = [0u8; 20];
        reader.read_exact(&mut skip)?;
        let by_residual = flags & FLAG_BY_RESIDUAL != 0;
        let delta_ids = flags & FLAG_DELTA_IDS != 0;
        let transposed_codes = flags & FLAG_TRANSPOSED_CODES != 0;
        let has_opq = flags & FLAG_HAS_OPQ != 0;
        let centroids_offset = HEADER_SIZE as u64 + if has_opq { (d * d * 4) as u64 } else { 0 };

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
                nbits: 8,
                dsub,
                ksub,
                centroids: Vec::new(),
                centroid_norms_cache: Vec::new(),
            },
            list_offsets: Vec::new(),
            list_counts: Vec::new(),
            precomputed_table: Vec::new(),
            delta_ids,
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

        // Seek to start of data sections
        if self.has_opq {
            self.reader.seek(HEADER_SIZE as u64)?;
            let rotation = read_f32_vec(&mut self.reader, d * d)?;
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
        } else {
            self.reader.seek(self.centroids_offset)?;
        }

        self.quantizer_centroids = read_f32_vec(&mut self.reader, nlist * d)?;

        let pq_centroids = read_f32_vec(&mut self.reader, m * ksub * dsub)?;
        self.pq = ProductQuantizer {
            d,
            m,
            nbits: 8,
            dsub,
            ksub,
            centroids: pq_centroids,
            centroid_norms_cache: Vec::new(),
        };
        self.pq.rebuild_norms_cache();

        self.list_offsets = vec![0i64; nlist];
        self.list_counts = vec![0i32; nlist];
        for i in 0..nlist {
            self.list_offsets[i] = read_i64_le(&mut self.reader)?;
            self.list_counts[i] = read_i32_le(&mut self.reader)?;
            let _pad = read_i32_le(&mut self.reader)?;
        }

        self.loaded = true;
        Ok(())
    }

    /// Read an inverted list's IDs and PQ codes.
    /// Calls ensure_loaded() if not yet loaded.
    pub fn read_inverted_list(&mut self, list_id: usize) -> io::Result<(Vec<i64>, Vec<u8>)> {
        self.ensure_loaded()?;
        let count = self.list_counts[list_id] as usize;
        if count == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let offset = self.list_offsets[list_id] as u64;
        self.reader.seek(offset)?;

        let ids = if self.delta_ids {
            // Delta-varint format: [base_id: i64][id_bytes_len: u32][id_bytes...]
            let base_id = read_i64_le(&mut self.reader)?;
            let id_bytes_len = read_i32_le(&mut self.reader)? as usize;
            let mut id_bytes = vec![0u8; id_bytes_len];
            self.reader.read_exact(&mut id_bytes)?;
            decode_delta_varint_ids(base_id, &id_bytes, count)
        } else {
            // Raw int64 format
            let mut id_buf = vec![0u8; count * 8];
            self.reader.read_exact(&mut id_buf)?;
            id_buf
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        };

        let code_size = self.pq.code_size();
        let mut codes = vec![0u8; count * code_size];
        self.reader.read_exact(&mut codes)?;

        Ok((ids, codes))
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

    #[test]
    fn test_varint_roundtrip() {
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        encode_varint(127, &mut buf);
        encode_varint(128, &mut buf);
        encode_varint(16383, &mut buf);
        encode_varint(1_000_000, &mut buf);

        let mut pos = 0;
        assert_eq!(decode_varint(&buf, &mut pos), 0);
        assert_eq!(decode_varint(&buf, &mut pos), 127);
        assert_eq!(decode_varint(&buf, &mut pos), 128);
        assert_eq!(decode_varint(&buf, &mut pos), 16383);
        assert_eq!(decode_varint(&buf, &mut pos), 1_000_000);
    }

    #[test]
    fn test_delta_varint_ids_roundtrip() {
        let ids = vec![3i64, 7, 12, 15, 23, 100, 200];
        let (base, encoded) = encode_delta_varint_ids(&ids);
        let decoded = decode_delta_varint_ids(base, &encoded, ids.len());
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
        assert!(reader.delta_ids);
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

        // Write with raw int64 IDs
        let mut raw_buf = Vec::new();
        let mut raw_writer = PosWriter::new(&mut raw_buf);
        write_index_raw_ids(&index, &mut raw_writer).unwrap();

        // Write with delta-varint IDs
        let mut delta_buf = Vec::new();
        let mut delta_writer = PosWriter::new(&mut delta_buf);
        write_index(&index, &mut delta_writer).unwrap();

        let raw_size = raw_buf.len();
        let delta_size = delta_buf.len();
        let savings_pct = (1.0 - delta_size as f64 / raw_size as f64) * 100.0;

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
            "Total file (raw):  {} bytes ({:.1} KB)",
            raw_size,
            raw_size as f64 / 1024.0
        );
        eprintln!(
            "Total file (delta):{} bytes ({:.1} KB)",
            delta_size,
            delta_size as f64 / 1024.0
        );
        eprintln!("Total savings:     {:.1}%", savings_pct);

        // Delta-varint should save at least 20% on total file size
        assert!(
            savings_pct > 10.0,
            "Expected >10% savings, got {:.1}%",
            savings_pct
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
}
