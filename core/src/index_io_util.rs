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

use crate::io::{ReadRequest, SeekRead, SeekWrite};
use roaring::RoaringTreemap;
use std::io;
use std::mem::size_of;

pub(crate) const MAX_IVF_BATCH_READ_BYTES: usize = 64 * 1024 * 1024;
const IVF_STREAM_ALLOCATION_SLACK: usize = 64;

pub(crate) fn ivf_payload_is_oversized(payload_len: usize) -> bool {
    payload_len > MAX_IVF_BATCH_READ_BYTES
}

/// Resolves a row count whose list payload plus the retained decoded IDs stays
/// within the IVF search allocation bound. Blocked layouts can require chunks
/// before the final one to contain a whole number of rows per block.
pub(crate) fn bounded_ivf_stream_chunk_rows(
    remaining_rows: usize,
    row_bytes: usize,
    retained_id_bytes: usize,
    row_alignment: usize,
) -> io::Result<usize> {
    if remaining_rows == 0 {
        return Ok(0);
    }
    if row_bytes == 0 || row_alignment == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IVF streaming row shape must be greater than zero",
        ));
    }
    let available = MAX_IVF_BATCH_READ_BYTES
        .checked_sub(retained_id_bytes)
        .and_then(|bytes| bytes.checked_sub(IVF_STREAM_ALLOCATION_SLACK))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF decoded row IDs exceed the bounded streaming allocation",
            )
        })?;
    let mut rows = (available / row_bytes).min(remaining_rows);
    if rows < remaining_rows && row_alignment > 1 {
        rows -= rows % row_alignment;
    }
    if rows == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "one IVF streaming block exceeds the bounded allocation",
        ));
    }
    Ok(rows)
}

/// Reads and validates one delta-varint ID section without retaining the list's
/// potentially much larger vector/code payload. The encoded prefix and decoded
/// IDs are checked together before allocation.
pub(crate) fn read_delta_varint_ids_at<R: SeekRead>(
    reader: &mut R,
    offset: u64,
    count: usize,
    id_bytes_len: usize,
    format_name: &str,
) -> io::Result<Vec<i64>> {
    let prefix_len = 12usize.checked_add(id_bytes_len).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{format_name} ID prefix size overflows usize"),
        )
    })?;
    let decoded_bytes = count.checked_mul(size_of::<i64>()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{format_name} decoded ID size overflows usize"),
        )
    })?;
    if prefix_len
        .checked_add(decoded_bytes)
        .is_none_or(|peak| peak > MAX_IVF_BATCH_READ_BYTES)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{format_name} ID section and decoded IDs exceed the bounded streaming allocation"
            ),
        ));
    }
    let mut prefix = vec![0u8; prefix_len];
    reader.pread(&mut [ReadRequest::new(offset, &mut prefix)])?;
    let base_id = i64::from_le_bytes(prefix[0..8].try_into().unwrap());
    let stored_id_bytes_len = i32::from_le_bytes(prefix[8..12].try_into().unwrap());
    if stored_id_bytes_len < 0 || stored_id_bytes_len as usize != id_bytes_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{format_name} ID length does not match the offset table"),
        ));
    }
    decode_delta_varint_ids(base_id, &prefix[12..], count)
}

/// Returns the largest non-empty prefix whose aggregate payload and range
/// count fit one IVF read/scan batch. Zero-length entries do not consume a
/// range. A single oversized payload is returned alone.
pub(crate) fn bounded_ivf_payload_batch_end(
    payload_lengths: &[usize],
    max_ranges_per_pread: usize,
) -> io::Result<usize> {
    if payload_lengths.is_empty() {
        return Ok(0);
    }
    let max_ranges = match max_ranges_per_pread {
        0 => usize::MAX,
        value => value,
    };
    let mut bytes = 0usize;
    let mut ranges = 0usize;
    for (index, &payload_len) in payload_lengths.iter().enumerate() {
        let next_ranges = ranges + usize::from(payload_len != 0);
        let next_bytes = bytes.checked_add(payload_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF batch payload size overflow",
            )
        })?;
        if index > 0 && (next_ranges > max_ranges || next_bytes > MAX_IVF_BATCH_READ_BYTES) {
            return Ok(index);
        }
        ranges = next_ranges;
        bytes = next_bytes;
    }
    Ok(payload_lengths.len())
}

/// Reads payloads in capability-bounded multi-range batches.
///
/// A single payload larger than `max_batch_bytes` is still issued alone so
/// callers never stall at the same batch boundary.
pub(crate) fn pread_batched_payloads<R: SeekRead>(
    reader: &mut R,
    offsets: &[u64],
    payloads: &mut [Vec<u8>],
) -> io::Result<()> {
    let mut buffers = payloads
        .iter_mut()
        .map(Vec::as_mut_slice)
        .collect::<Vec<_>>();
    pread_batched_slices(reader, offsets, &mut buffers)
}

/// Reads caller-owned byte slices with the same bounded IVF multi-range plan.
///
/// This variant lets typed or aligned payload owners receive bytes directly
/// without first allocating a second `Vec<u8>` for every inverted list.
pub(crate) fn pread_batched_slices<R: SeekRead>(
    reader: &mut R,
    offsets: &[u64],
    payloads: &mut [&mut [u8]],
) -> io::Result<()> {
    pread_batched_slices_with_limit(reader, offsets, payloads, MAX_IVF_BATCH_READ_BYTES)
}

#[cfg(test)]
fn pread_batched_payloads_with_limit<R: SeekRead>(
    reader: &mut R,
    offsets: &[u64],
    payloads: &mut [Vec<u8>],
    max_batch_bytes: usize,
) -> io::Result<()> {
    let mut buffers = payloads
        .iter_mut()
        .map(Vec::as_mut_slice)
        .collect::<Vec<_>>();
    pread_batched_slices_with_limit(reader, offsets, &mut buffers, max_batch_bytes)
}

fn pread_batched_slices_with_limit<R: SeekRead>(
    reader: &mut R,
    offsets: &[u64],
    payloads: &mut [&mut [u8]],
    max_batch_bytes: usize,
) -> io::Result<()> {
    if offsets.len() != payloads.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF batch offsets and payloads must have the same length",
        ));
    }
    let max_ranges = match reader.read_capabilities().max_ranges_per_pread {
        0 => usize::MAX,
        value => value,
    };
    let mut start = 0usize;
    while start < payloads.len() {
        let mut end = start;
        let mut bytes = 0usize;
        while end < payloads.len() && end - start < max_ranges {
            let next = bytes.checked_add(payloads[end].len()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF batch payload size overflow",
                )
            })?;
            if end > start && next > max_batch_bytes {
                break;
            }
            bytes = next;
            end += 1;
        }
        if end == start {
            end += 1;
        }
        let mut requests = payloads[start..end]
            .iter_mut()
            .zip(&offsets[start..end])
            .map(|(payload, &offset)| ReadRequest::new(offset, payload))
            .collect::<Vec<_>>();
        reader.pread(&mut requests)?;
        start = end;
    }
    Ok(())
}

pub(crate) fn validate_search_inputs(
    queries: &[f32],
    nq: usize,
    d: usize,
    k: usize,
    nprobe: usize,
) -> io::Result<()> {
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
    Ok(())
}

pub(crate) fn validate_reserved_zero(bytes: &[u8], format_name: &str) -> io::Result<()> {
    if bytes.iter().any(|&byte| byte != 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} reserved bytes must be zero", format_name),
        ));
    }
    Ok(())
}

pub(crate) fn encode_delta_varint_ids(ids: &[i64]) -> (i64, Vec<u8>) {
    if ids.is_empty() {
        return (0, Vec::new());
    }
    let base = ids[0];
    let mut buf = Vec::with_capacity(ids.len() * 2);
    let mut prev = base;
    for &id in ids {
        let delta = (id as u64).wrapping_sub(prev as u64);
        write_u64_varint(&mut buf, delta);
        prev = id;
    }
    (base, buf)
}

pub(crate) fn decode_delta_varint_ids(base: i64, buf: &[u8], count: usize) -> io::Result<Vec<i64>> {
    let mut ids = Vec::with_capacity(count);
    let mut pos = 0;
    let mut current = base as u64;
    let mut prev_signed = base;
    for _ in 0..count {
        let delta = read_u64_varint(buf, &mut pos)?;
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
    if pos != buf.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing bytes in delta-varint ID section",
        ));
    }
    Ok(ids)
}

fn write_u64_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

#[inline]
fn read_u64_varint(bytes: &[u8], pos: &mut usize) -> io::Result<u64> {
    let first = bytes.get(*pos).copied().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "truncated delta-varint value")
    })?;
    *pos += 1;
    if first < 0x80 {
        return Ok(first as u64);
    }

    let mut value = (first & 0x7f) as u64;
    let mut shift = 7u32;
    for _ in 1..10 {
        let byte = bytes.get(*pos).copied().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "truncated delta-varint value")
        })?;
        *pos += 1;
        if shift == 63 && (byte & 0x7e) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "delta-varint value exceeds u64 limit",
            ));
        }
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "delta-varint value exceeds u64 limit",
    ))
}

pub(crate) fn write_u32_le(out: &mut dyn SeekWrite, v: u32) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

pub(crate) fn write_i32_le(out: &mut dyn SeekWrite, v: i32) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

pub(crate) fn write_i64_le(out: &mut dyn SeekWrite, v: i64) -> io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

pub(crate) fn write_f32_slice(out: &mut dyn SeekWrite, data: &[f32]) -> io::Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    out.write_all(&bytes)
}

pub(crate) fn bytes_to_f32_vec(bytes: &[u8]) -> io::Result<Vec<f32>> {
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

pub(crate) fn validate_positive_i32(val: i32, field: &str) -> io::Result<i32> {
    if val <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid header field {}: {} (must be positive)", field, val),
        ));
    }
    Ok(val)
}

pub(crate) fn usize_to_i32(value: usize, field: &str) -> io::Result<i32> {
    if value > i32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} exceeds i32 length limit: {}", field, value),
        ));
    }
    Ok(value as i32)
}

pub(crate) fn usize_to_i64(value: usize, field: &str) -> io::Result<i64> {
    if value > i64::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} exceeds i64 length limit: {}", field, value),
        ));
    }
    Ok(value as i64)
}

pub(crate) fn u64_to_i64(value: u64, field: &str) -> io::Result<i64> {
    if value > i64::MAX as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} exceeds i64 offset limit: {}", field, value),
        ));
    }
    Ok(value as i64)
}

const MAX_SECTION_ELEMENTS: usize = 1 << 30;

pub(crate) fn checked_section_size(a: usize, b: usize) -> io::Result<usize> {
    let result = a
        .checked_mul(b)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "section size overflow"))?;
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

pub(crate) fn checked_list_offset(offset: i64, list_id: usize) -> io::Result<u64> {
    if offset < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("negative list offset {} at list {}", offset, list_id),
        ));
    }
    Ok(offset as u64)
}

pub(crate) fn checked_list_bytes(count: usize, bytes_per_entry: usize) -> io::Result<usize> {
    count
        .checked_mul(bytes_per_entry)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "list byte size overflow"))
}

pub(crate) fn decode_roaring_filter(bytes: &[u8]) -> io::Result<RoaringTreemap> {
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
    use crate::io::SeekReadCapabilities;

    struct RecordingReader {
        bytes: Vec<u8>,
        max_ranges: usize,
        calls: Vec<usize>,
    }

    impl SeekRead for RecordingReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            self.calls.push(ranges.len());
            for range in ranges {
                let start = range.pos as usize;
                let end = start + range.buf.len();
                range.buf.copy_from_slice(&self.bytes[start..end]);
            }
            Ok(())
        }

        fn read_capabilities(&self) -> SeekReadCapabilities {
            SeekReadCapabilities {
                max_ranges_per_pread: self.max_ranges,
                ..SeekReadCapabilities::default()
            }
        }
    }

    #[test]
    fn batched_pread_honors_range_and_byte_limits_without_dropping_payloads() {
        let mut reader = RecordingReader {
            bytes: (0..32).collect(),
            max_ranges: 2,
            calls: Vec::new(),
        };
        let offsets = [0, 3, 6, 9, 12];
        let mut payloads = vec![vec![0; 3]; offsets.len()];
        pread_batched_payloads_with_limit(&mut reader, &offsets, &mut payloads, 6).unwrap();
        assert_eq!(reader.calls, vec![2, 2, 1]);
        assert_eq!(payloads[0], vec![0, 1, 2]);
        assert_eq!(payloads[4], vec![12, 13, 14]);

        let mut byte_limited = RecordingReader {
            bytes: (0..32).collect(),
            max_ranges: 8,
            calls: Vec::new(),
        };
        let mut payloads = vec![vec![0; 4], vec![0; 4], vec![0; 4]];
        pread_batched_payloads_with_limit(&mut byte_limited, &[0, 4, 8], &mut payloads, 6).unwrap();
        assert_eq!(byte_limited.calls, vec![1, 1, 1]);
    }

    #[test]
    fn aggregate_ivf_batch_plan_bounds_allocated_payloads() {
        assert_eq!(bounded_ivf_payload_batch_end(&[32, 32, 32], 2).unwrap(), 2);
        assert_eq!(
            bounded_ivf_payload_batch_end(
                &[
                    MAX_IVF_BATCH_READ_BYTES / 2,
                    MAX_IVF_BATCH_READ_BYTES / 2,
                    1
                ],
                8,
            )
            .unwrap(),
            2
        );
        assert_eq!(
            bounded_ivf_payload_batch_end(&[MAX_IVF_BATCH_READ_BYTES + 1, 1], 8).unwrap(),
            1
        );
        assert_eq!(bounded_ivf_payload_batch_end(&[0, 0, 1], 1).unwrap(), 3);
    }

    #[test]
    fn ivf_stream_chunks_reserve_decoded_ids_and_block_alignment() {
        let retained_ids = 8 * 1024 * 1024;
        let rows =
            bounded_ivf_stream_chunk_rows(usize::MAX, 1024 * size_of::<f32>(), retained_ids, 1)
                .unwrap();
        assert!(rows * 1024 * size_of::<f32>() + retained_ids <= MAX_IVF_BATCH_READ_BYTES);

        let blocked = bounded_ivf_stream_chunk_rows(100_000, 1024, retained_ids, 32).unwrap();
        assert!(blocked.is_multiple_of(32));
        assert!(blocked * 1024 + retained_ids <= MAX_IVF_BATCH_READ_BYTES);
    }

    #[test]
    fn ivf_stream_rejects_id_state_that_leaves_no_payload_block() {
        let error =
            bounded_ivf_stream_chunk_rows(1, 4096, MAX_IVF_BATCH_READ_BYTES, 1).unwrap_err();
        assert!(error.to_string().contains("bounded streaming allocation"));
    }
}
