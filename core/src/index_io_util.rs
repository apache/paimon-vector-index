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
use crate::hnsw::{HnswBuildParams, HnswGraph};
use crate::io::{PreadCursor, SeekRead, SeekWrite};
use roaring::RoaringTreemap;
use std::io;

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

const HNSW_GRAPH_MAGIC: u32 = 0x48574752; // "HWGR"
const HNSW_GRAPH_VERSION: u32 = 1;
const HNSW_GRAPH_FLAG_DELTA_VARINT: u32 = 1 << 0;
const HNSW_GRAPH_REQUIRED_FLAGS: u32 = HNSW_GRAPH_FLAG_DELTA_VARINT;
const HNSW_GRAPH_SUPPORTED_FLAGS: u32 = HNSW_GRAPH_REQUIRED_FLAGS;

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

pub(crate) fn encode_graph(graph: Option<&HnswGraph>) -> io::Result<Vec<u8>> {
    let Some(graph) = graph else {
        return Ok(Vec::new());
    };
    let mut buf = Vec::new();
    write_u32_fixed(&mut buf, HNSW_GRAPH_MAGIC)?;
    write_u32_fixed(&mut buf, HNSW_GRAPH_VERSION)?;
    write_u32_fixed(&mut buf, HNSW_GRAPH_REQUIRED_FLAGS)?;
    write_u32_varint(&mut buf, graph.len())?;
    write_u32_varint(&mut buf, graph.entry_point())?;
    write_u32_varint(&mut buf, graph.max_observed_level())?;
    for &level in graph.levels() {
        write_u32_varint(&mut buf, level)?;
    }
    for node_levels in graph.neighbors() {
        for level_neighbors in node_levels {
            write_u32_varint(&mut buf, level_neighbors.len())?;
            let mut sorted_neighbors = level_neighbors.clone();
            sorted_neighbors.sort_unstable();
            let mut previous = 0usize;
            for neighbor in sorted_neighbors {
                let delta = neighbor.checked_sub(previous).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "graph neighbor delta underflow",
                    )
                })?;
                write_u32_varint(&mut buf, delta)?;
                previous = neighbor;
            }
        }
    }
    Ok(buf)
}

#[cfg(test)]
pub(crate) fn encode_graph_u32_for_size_estimate(graph: &HnswGraph) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_u32_fixed(&mut buf, checked_u32_value(graph.len())?)?;
    write_u32_fixed(&mut buf, checked_u32_value(graph.entry_point())?)?;
    write_u32_fixed(&mut buf, checked_u32_value(graph.max_observed_level())?)?;
    for &level in graph.levels() {
        write_u32_fixed(&mut buf, checked_u32_value(level)?)?;
    }
    for node_levels in graph.neighbors() {
        for level_neighbors in node_levels {
            write_u32_fixed(&mut buf, checked_u32_value(level_neighbors.len())?)?;
            for &neighbor in level_neighbors {
                write_u32_fixed(&mut buf, checked_u32_value(neighbor)?)?;
            }
        }
    }
    Ok(buf)
}

pub(crate) fn decode_graph(
    bytes: &[u8],
    vectors: Vec<f32>,
    count: usize,
    d: usize,
    metric: MetricType,
    hnsw_params: HnswBuildParams,
) -> io::Result<Option<HnswGraph>> {
    if bytes.is_empty() {
        return Ok(None);
    }
    let mut pos = 0usize;
    let graph_magic = read_u32_fixed(bytes, &mut pos)?;
    if graph_magic != HNSW_GRAPH_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid HNSW graph magic: 0x{:08X}", graph_magic),
        ));
    }
    let graph_version = read_u32_fixed(bytes, &mut pos)?;
    if graph_version != HNSW_GRAPH_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported HNSW graph version: {}", graph_version),
        ));
    }
    let graph_flags = read_u32_fixed(bytes, &mut pos)?;
    let unknown_graph_flags = graph_flags & !HNSW_GRAPH_SUPPORTED_FLAGS;
    if unknown_graph_flags != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Unsupported HNSW graph flags: 0x{:08X}",
                unknown_graph_flags
            ),
        ));
    }
    if graph_flags & HNSW_GRAPH_REQUIRED_FLAGS != HNSW_GRAPH_REQUIRED_FLAGS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "HNSW graph v1 requires delta-varint adjacency encoding",
        ));
    }
    let graph_count = read_u32_varint(bytes, &mut pos)? as usize;
    if graph_count != count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "graph count {} does not match list count {}",
                graph_count, count
            ),
        ));
    }
    let entry_point = read_u32_varint(bytes, &mut pos)? as usize;
    let max_observed_level = read_u32_varint(bytes, &mut pos)? as usize;
    let mut levels = Vec::with_capacity(count);
    for node in 0..count {
        let level = read_u32_varint(bytes, &mut pos)? as usize;
        if level >= hnsw_params.max_level {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "graph node {} level {} exceeds max_level {}",
                    node,
                    level,
                    hnsw_params.max_level - 1
                ),
            ));
        }
        levels.push(level);
    }
    let mut neighbors = Vec::with_capacity(count);
    for (node, &level) in levels.iter().enumerate() {
        let mut node_levels = Vec::with_capacity(level + 1);
        for graph_level in 0..=level {
            let degree = read_u32_varint(bytes, &mut pos)? as usize;
            let max_degree = if graph_level == 0 {
                hnsw_params.m.saturating_mul(2)
            } else {
                hnsw_params.m
            };
            if degree > max_degree {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "graph node {} degree {} at level {} exceeds max degree {}",
                        node, degree, graph_level, max_degree
                    ),
                ));
            }
            let mut level_neighbors = Vec::with_capacity(degree);
            let mut previous = 0usize;
            for _ in 0..degree {
                let delta = read_u32_varint(bytes, &mut pos)? as usize;
                let neighbor = previous.checked_add(delta).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "graph neighbor id overflow")
                })?;
                if neighbor > u32::MAX as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("graph neighbor id {} is out of range", neighbor),
                    ));
                }
                level_neighbors.push(neighbor);
                previous = neighbor;
            }
            node_levels.push(level_neighbors);
        }
        neighbors.push(node_levels);
    }
    if pos != bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing bytes in HNSW graph section",
        ));
    }
    Ok(Some(HnswGraph::from_parts(
        vectors,
        count,
        d,
        metric,
        levels,
        neighbors,
        entry_point,
        max_observed_level,
        hnsw_params,
    )?))
}

fn checked_u32_value(value: usize) -> io::Result<u32> {
    if value > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("value {} exceeds u32 limit", value),
        ));
    }
    Ok(value as u32)
}

fn write_u32_fixed(buf: &mut Vec<u8>, value: u32) -> io::Result<()> {
    buf.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u32_fixed(bytes: &[u8], pos: &mut usize) -> io::Result<u32> {
    let end = pos.checked_add(4).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "graph section position overflow",
        )
    })?;
    if end > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated HNSW graph section",
        ));
    }
    let value = u32::from_le_bytes(bytes[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(value)
}

fn write_u32_varint(buf: &mut Vec<u8>, value: usize) -> io::Result<()> {
    write_u64_varint(buf, checked_u32_value(value)? as u64);
    Ok(())
}

fn write_u64_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn read_u32_varint(bytes: &[u8], pos: &mut usize) -> io::Result<u32> {
    let value = read_u64_varint(bytes, pos)?;
    u32::try_from(value).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("HNSW graph varint value {} exceeds u32 limit", value),
        )
    })
}

fn read_u64_varint(bytes: &[u8], pos: &mut usize) -> io::Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for _ in 0..10 {
        if *pos >= bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated HNSW graph varint",
            ));
        }
        let byte = bytes[*pos];
        *pos += 1;
        if shift == 63 && (byte & 0x7e) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HNSW graph varint exceeds u64 limit",
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
        "HNSW graph varint exceeds u64 limit",
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

pub(crate) fn read_u32_le<R: SeekRead + ?Sized>(
    reader: &mut PreadCursor<'_, R>,
) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn read_i32_le<R: SeekRead + ?Sized>(
    reader: &mut PreadCursor<'_, R>,
) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

pub(crate) fn read_i64_le<R: SeekRead + ?Sized>(
    reader: &mut PreadCursor<'_, R>,
) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

pub(crate) fn read_f32_vec<R: SeekRead + ?Sized>(
    reader: &mut PreadCursor<'_, R>,
    count: usize,
) -> io::Result<Vec<f32>> {
    let byte_len = count.checked_mul(4).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "f32 section byte length overflow",
        )
    })?;
    let mut buf = vec![0u8; byte_len];
    reader.read_exact(&mut buf)?;
    bytes_to_f32_vec(&buf)
}

pub(crate) fn read_f32_vec_checked<R: SeekRead + ?Sized>(
    reader: &mut PreadCursor<'_, R>,
    count: usize,
    section: &str,
) -> io::Result<Vec<f32>> {
    let values = read_f32_vec(reader, count)?;
    validate_finite_f32_values(&values, section)?;
    Ok(values)
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

pub(crate) fn bytes_to_f32_vec_checked(bytes: &[u8], section: &str) -> io::Result<Vec<f32>> {
    let values = bytes_to_f32_vec(bytes)?;
    validate_finite_f32_values(&values, section)?;
    Ok(values)
}

pub(crate) fn validate_finite_f32_value(value: f32, section: &str) -> io::Result<()> {
    if !value.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} contains non-finite value: {}", section, value),
        ));
    }
    Ok(())
}

pub(crate) fn validate_finite_f32_values(values: &[f32], section: &str) -> io::Result<()> {
    for (offset, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} contains non-finite value at offset {}: {}",
                    section, offset, value
                ),
            ));
        }
    }
    Ok(())
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
