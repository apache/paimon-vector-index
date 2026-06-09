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

use crate::distance::{preprocess_vectors, MetricType};
use crate::hnsw::{HnswBuildParams, HnswGraph};
use crate::hnsw_search::{search_hnsw_lists, HnswSearchList};
use crate::index_io_util::{
    checked_list_bytes, checked_list_offset, checked_section_size, decode_graph,
    decode_roaring_filter, encode_graph, read_f32_vec, read_i32_le, read_i64_le, read_u32_le,
    u64_to_i64, usize_to_i32, usize_to_i64, validate_positive_i32, validate_search_inputs,
    write_f32_slice, write_i32_le, write_i64_le, write_u32_le,
};
use crate::io::{SeekRead, SeekWrite};
use crate::ivfhnswsq::IVFHNSWSQIndex;
use crate::ivfpq::RowIdFilter;
use crate::kmeans;
use crate::sq::ScalarQuantizer;
use crate::topk::TopKHeap;
use std::io;

pub const IVF_HNSW_SQ_MAGIC: u32 = 0x49485351; // "IHSQ"
pub const IVF_HNSW_SQ_VERSION: u32 = 1;
pub const IVF_HNSW_SQ_HEADER_SIZE: usize = 64;

pub fn write_ivfhnswsq_index(index: &IVFHNSWSQIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
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
    let graph_bytes: Vec<Vec<u8>> = (0..index.nlist)
        .map(|list_id| {
            if index.ids[list_id].is_empty() {
                Ok(Vec::new())
            } else {
                encode_graph(index.graphs[list_id].as_ref())
            }
        })
        .collect::<io::Result<_>>()?;

    write_u32_le(out, IVF_HNSW_SQ_MAGIC)?;
    write_u32_le(out, IVF_HNSW_SQ_VERSION)?;
    write_i32_le(out, usize_to_i32(index.d, "dimension")?)?;
    write_i32_le(out, usize_to_i32(index.nlist, "nlist")?)?;
    write_u32_le(out, index.metric as u32)?;
    write_i64_le(out, total_vectors)?;
    let params = index.hnsw_params.sanitized();
    write_i32_le(out, usize_to_i32(params.m, "hnsw m")?)?;
    write_i32_le(
        out,
        usize_to_i32(params.ef_construction, "hnsw ef_construction")?,
    )?;
    write_i32_le(out, usize_to_i32(params.max_level, "hnsw max_level")?)?;
    out.write_all(&index.sq.min.to_le_bytes())?;
    out.write_all(&index.sq.max.to_le_bytes())?;
    out.write_all(&[0u8; 16])?;

    write_f32_slice(out, &index.quantizer_centroids)?;

    let offset_table_size = index.nlist.checked_mul(24).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-HNSW-SQ offset table size overflow",
        )
    })?;
    let data_start = out
        .pos()
        .checked_add(offset_table_size as u64)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-HNSW-SQ data start offset overflow",
            )
        })?;
    let mut list_offsets = vec![0i64; index.nlist];
    let mut list_counts = vec![0i32; index.nlist];
    let mut list_graph_bytes_lens = vec![0i32; index.nlist];
    let mut current_offset = data_start;

    for list_id in 0..index.nlist {
        list_offsets[list_id] = u64_to_i64(current_offset, "list offset")?;
        let count = index.ids[list_id].len();
        list_counts[list_id] = usize_to_i32(count, "list count")?;
        list_graph_bytes_lens[list_id] = usize_to_i32(graph_bytes[list_id].len(), "graph bytes")?;
        if count > 0 {
            let payload_len =
                list_payload_len(count, index.sq.code_size(), graph_bytes[list_id].len())?;
            current_offset = current_offset
                .checked_add(payload_len as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVF-HNSW-SQ offset overflow")
                })?;
        }
    }

    for list_id in 0..index.nlist {
        write_i64_le(out, list_offsets[list_id])?;
        write_i32_le(out, list_counts[list_id])?;
        write_i32_le(out, list_graph_bytes_lens[list_id])?;
        write_i64_le(out, 0)?;
    }

    for list_id in 0..index.nlist {
        if index.ids[list_id].is_empty() {
            continue;
        }
        for &id in &index.ids[list_id] {
            write_i64_le(out, id)?;
        }
        out.write_all(&index.codes[list_id])?;
        out.write_all(&graph_bytes[list_id])?;
    }

    Ok(())
}

pub struct IVFHNSWSQIndexReader<R: SeekRead> {
    reader: R,
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub hnsw_params: HnswBuildParams,
    pub sq: ScalarQuantizer,
    pub quantizer_centroids: Vec<f32>,
    pub list_offsets: Vec<i64>,
    pub list_counts: Vec<i32>,
    pub list_graph_bytes_lens: Vec<i32>,
    loaded: bool,
}

impl<R: SeekRead> IVFHNSWSQIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        reader.seek(0)?;

        let magic = read_u32_le(&mut reader)?;
        if magic != IVF_HNSW_SQ_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVF_HNSW_SQ magic: 0x{:08X}", magic),
            ));
        }
        let version = read_u32_le(&mut reader)?;
        if version != IVF_HNSW_SQ_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF_HNSW_SQ version: {}", version),
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
        let hnsw_params = HnswBuildParams {
            m: validate_positive_i32(read_i32_le(&mut reader)?, "hnsw m")? as usize,
            ef_construction: validate_positive_i32(
                read_i32_le(&mut reader)?,
                "hnsw ef_construction",
            )? as usize,
            max_level: validate_positive_i32(read_i32_le(&mut reader)?, "hnsw max_level")? as usize,
        }
        .sanitized();
        let mut min_bytes = [0u8; 4];
        let mut max_bytes = [0u8; 4];
        reader.read_exact(&mut min_bytes)?;
        reader.read_exact(&mut max_bytes)?;
        let sq_min = f32::from_le_bytes(min_bytes);
        let sq_max = f32::from_le_bytes(max_bytes);
        if !sq_min.is_finite() || !sq_max.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SQ bounds must be finite",
            ));
        }
        let mut reserved = [0u8; 16];
        reader.read_exact(&mut reserved)?;

        Ok(Self {
            reader,
            d,
            nlist,
            metric,
            total_vectors,
            hnsw_params,
            sq: ScalarQuantizer::with_bounds(d, sq_min, sq_max),
            quantizer_centroids: Vec::new(),
            list_offsets: Vec::new(),
            list_counts: Vec::new(),
            list_graph_bytes_lens: Vec::new(),
            loaded: false,
        })
    }

    pub fn ensure_loaded(&mut self) -> io::Result<()> {
        if self.loaded {
            return Ok(());
        }

        self.reader.seek(IVF_HNSW_SQ_HEADER_SIZE as u64)?;
        self.quantizer_centroids =
            read_f32_vec(&mut self.reader, checked_section_size(self.nlist, self.d)?)?;
        self.list_offsets = vec![0; self.nlist];
        self.list_counts = vec![0; self.nlist];
        self.list_graph_bytes_lens = vec![0; self.nlist];
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
            let graph_bytes_len = read_i32_le(&mut self.reader)?;
            if graph_bytes_len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "negative graph_bytes_len {} at list {}",
                        graph_bytes_len, list_id
                    ),
                ));
            }
            self.list_graph_bytes_lens[list_id] = graph_bytes_len;
            let _reserved = read_i64_le(&mut self.reader)?;
        }

        self.loaded = true;
        Ok(())
    }

    pub fn read_inverted_list(
        &mut self,
        list_id: usize,
    ) -> io::Result<(Vec<i64>, Vec<u8>, Option<HnswGraph>)> {
        let Some(list) = self.read_graph_list(list_id)? else {
            return Ok((Vec::new(), Vec::new(), None));
        };
        Ok((list.ids, list.codes, Some(list.graph)))
    }

    fn read_graph_list(&mut self, list_id: usize) -> io::Result<Option<GraphList>> {
        self.ensure_loaded()?;
        if list_id >= self.nlist {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("list_id {} out of range (nlist={})", list_id, self.nlist),
            ));
        }
        let count = self.list_counts[list_id] as usize;
        if count == 0 {
            return Ok(None);
        }

        let offset = checked_list_offset(self.list_offsets[list_id], list_id)?;
        let graph_bytes_len = self.list_graph_bytes_lens[list_id] as usize;
        if graph_bytes_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("list {} is missing HNSW graph", list_id),
            ));
        }
        let payload_len = list_payload_len(count, self.sq.code_size(), graph_bytes_len)?;
        let mut payload = vec![0u8; payload_len];
        self.reader.pread(offset, &mut payload)?;

        let ids_bytes_len = checked_list_bytes(count, 8)?;
        let code_size = self.sq.code_size();
        let codes_bytes_len = checked_list_bytes(count, code_size)?;
        let ids = payload[..ids_bytes_len]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let codes = payload[ids_bytes_len..ids_bytes_len + codes_bytes_len].to_vec();
        let mut vectors = vec![0.0f32; count * self.d];
        self.sq.decode_batch(&codes, count, &mut vectors);
        let graph = decode_graph(
            &payload[ids_bytes_len + codes_bytes_len..],
            vectors,
            count,
            self.d,
            self.metric,
            self.hnsw_params,
        )?;
        let graph = graph.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("list {} is missing HNSW graph", list_id),
            )
        })?;
        Ok(Some(GraphList { ids, codes, graph }))
    }

    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        ef_search: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.search_with_filter(query, k, nprobe, ef_search, None)
    }

    pub fn search_with_filter(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        ef_search: usize,
        filter: Option<&dyn RowIdFilter>,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.ensure_loaded()?;
        validate_search_inputs(query, 1, self.d, k, nprobe)?;

        let q = preprocess_vectors(query, 1, self.d, self.metric);
        let (probe_indices, _) =
            kmeans::find_topk(&q, &self.quantizer_centroids, self.nlist, self.d, nprobe);
        let mut loaded_lists = Vec::with_capacity(probe_indices.len());
        for list_id in probe_indices {
            if let Some(list) = self.read_graph_list(list_id)? {
                loaded_lists.push(list);
            }
        }
        let search_lists: Vec<_> = loaded_lists
            .iter()
            .map(|list| HnswSearchList {
                ids: list.ids.as_slice(),
                graph: Some(&list.graph),
                payload: list,
            })
            .collect();
        let sorted = search_hnsw_lists(&q, &search_lists, k, ef_search, filter, |list, heap| {
            let list = list.payload;
            scan_sq_list(
                &q,
                &list.ids,
                &list.codes,
                &self.sq,
                self.metric,
                filter,
                heap,
            );
        });
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
        ef_search: usize,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.search_with_filter(query, k, nprobe, ef_search, Some(&filter))
    }
}

pub fn search_batch_ivfhnswsq_reader<R: SeekRead>(
    reader: &mut IVFHNSWSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    ef_search: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfhnswsq_reader_filter(reader, queries, nq, k, nprobe, ef_search, None)
}

pub fn search_batch_ivfhnswsq_reader_filter<R: SeekRead>(
    reader: &mut IVFHNSWSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    ef_search: usize,
    filter: Option<&dyn RowIdFilter>,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    reader.ensure_loaded()?;
    validate_search_inputs(queries, nq, reader.d, k, nprobe)?;

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
    let mut query_filtered_counts = vec![0usize; nq];
    let mut loaded_lists = Vec::with_capacity(unique_lists.len());
    for list_id in unique_lists {
        let Some(list) = reader.read_graph_list(list_id)? else {
            continue;
        };
        if let Some(f) = filter {
            let filtered = list.ids.iter().filter(|&&id| f.contains(id)).count();
            for &qi in &list_to_queries[list_id] {
                query_filtered_counts[qi] = query_filtered_counts[qi]
                    .checked_add(filtered)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "filtered vector count overflows usize",
                        )
                    })?;
            }
        }
        loaded_lists.push(LoadedBatchList {
            query_ids: std::mem::take(&mut list_to_queries[list_id]),
            ids: list.ids,
            codes: list.codes,
            graph: list.graph,
        });
    }

    for list in &loaded_lists {
        for &qi in &list.query_ids {
            let query = &processed[qi * reader.d..(qi + 1) * reader.d];
            let force_sq_scan = filter
                .map(|_| query_filtered_counts[qi] <= ef_search.max(k))
                .unwrap_or(false);
            if force_sq_scan {
                scan_sq_list(
                    query,
                    &list.ids,
                    &list.codes,
                    &reader.sq,
                    reader.metric,
                    filter,
                    &mut heaps[qi],
                );
            } else {
                let local_results = list.graph.search(query, ef_search.max(k), ef_search.max(k));
                for (local_id, dist) in local_results {
                    let row_id = list.ids[local_id];
                    if filter.map(|f| f.contains(row_id)).unwrap_or(true) {
                        heaps[qi].push(dist, row_id);
                    }
                }
            }
        }
    }
    if filter.is_some() {
        for list in &loaded_lists {
            for &qi in &list.query_ids {
                if heaps[qi].len() >= k {
                    continue;
                }
                if query_filtered_counts[qi] <= ef_search.max(k) {
                    continue;
                }
                let query = &processed[qi * reader.d..(qi + 1) * reader.d];
                scan_sq_list(
                    query,
                    &list.ids,
                    &list.codes,
                    &reader.sq,
                    reader.metric,
                    filter,
                    &mut heaps[qi],
                );
            }
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

pub fn search_batch_ivfhnswsq_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFHNSWSQIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    ef_search: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_batch_ivfhnswsq_reader_filter(reader, queries, nq, k, nprobe, ef_search, Some(&filter))
}

struct GraphList {
    ids: Vec<i64>,
    codes: Vec<u8>,
    graph: HnswGraph,
}

struct LoadedBatchList {
    query_ids: Vec<usize>,
    ids: Vec<i64>,
    codes: Vec<u8>,
    graph: HnswGraph,
}

fn scan_sq_list(
    query: &[f32],
    ids: &[i64],
    codes: &[u8],
    sq: &ScalarQuantizer,
    metric: MetricType,
    filter: Option<&dyn RowIdFilter>,
    heap: &mut TopKHeap,
) {
    let context = sq.distance_context(query, metric);
    let code_size = sq.code_size();
    for (local_id, &row_id) in ids.iter().enumerate() {
        if filter.map(|f| !f.contains(row_id)).unwrap_or(false) {
            continue;
        }
        let code = &codes[local_id * code_size..(local_id + 1) * code_size];
        heap.push(
            sq.distance_to_code_with_context(query, code, context),
            row_id,
        );
    }
}

fn validate_index_shape(index: &IVFHNSWSQIndex) -> io::Result<()> {
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
    if index.sq.d != index.d {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SQ dimension does not match index dimension",
        ));
    }
    let centroid_len = checked_section_size(index.nlist, index.d)?;
    if index.quantizer_centroids.len() != centroid_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "quantizer centroid length {} does not match nlist*d {}",
                index.quantizer_centroids.len(),
                centroid_len
            ),
        ));
    }
    if index.ids.len() != index.nlist
        || index.codes.len() != index.nlist
        || index.graphs.len() != index.nlist
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inverted list count does not match nlist",
        ));
    }
    for list_id in 0..index.nlist {
        let count = index.ids[list_id].len();
        let expected_codes_len = checked_list_bytes(count, index.sq.code_size())?;
        if index.codes[list_id].len() != expected_codes_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "list {} SQ code length {} does not match count*d {}",
                    list_id,
                    index.codes[list_id].len(),
                    expected_codes_len
                ),
            ));
        }
        match &index.graphs[list_id] {
            Some(graph) if count > 0 => {
                let mut decoded = vec![0.0f32; count * index.d];
                index
                    .sq
                    .decode_batch(&index.codes[list_id], count, &mut decoded);
                if graph.len() != count || graph.vectors() != decoded.as_slice() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("list {} graph does not match SQ code storage", list_id),
                    ));
                }
            }
            Some(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("list {} has graph for an empty list", list_id),
                ));
            }
            None if count == 0 => {}
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "list {} is missing HNSW graph; call build_graphs first",
                        list_id
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn list_payload_len(count: usize, code_size: usize, graph_bytes_len: usize) -> io::Result<usize> {
    let id_bytes = checked_list_bytes(count, 8)?;
    let code_bytes = checked_list_bytes(count, code_size)?;
    id_bytes
        .checked_add(code_bytes)
        .and_then(|len| len.checked_add(graph_bytes_len))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-HNSW-SQ list payload length overflow",
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswBuildParams;
    use crate::io::PosWriter;
    use roaring::RoaringTreemap;
    use std::io::Cursor;

    #[test]
    fn test_ivfhnswsq_write_read_search_roundtrip() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [
                    cluster + i as f32 * 2.0,
                    10.0 + i as f32,
                    20.0 + i as f32,
                    30.0 + i as f32,
                ]
            })
            .collect();
        let ids: Vec<i64> = (10_000..10_000 + n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswsq_index(&index, &mut writer).unwrap();

        let mut reader = IVFHNSWSQIndexReader::open(Cursor::new(buf)).unwrap();
        let query_id = 23;
        let (labels, distances) = reader
            .search(&data[query_id * d..(query_id + 1) * d], 5, nlist, 32)
            .unwrap();

        assert_eq!(labels[0], ids[query_id]);
        assert!(distances[0].is_finite());
    }

    #[test]
    fn test_ivfhnswsq_reader_search_with_roaring_filter() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];
        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 3);
        index.add(&data, &ids, 3);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
        let mut reader = IVFHNSWSQIndexReader::open(Cursor::new(buf)).unwrap();

        let mut filter = RoaringTreemap::new();
        filter.insert(12);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        let (labels, _) = reader
            .search_with_roaring_filter(&[0.0, 0.0], 2, nlist, 8, &filter_bytes)
            .unwrap();

        assert_eq!(labels, vec![12, -1]);
    }

    #[test]
    fn test_ivfhnswsq_write_read_search_roundtrip_cosine() {
        let d = 3;
        let nlist = 2;
        let data = vec![1.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.1];
        let ids = vec![10, 11, 12, 13];
        let mut index =
            IVFHNSWSQIndex::new(d, nlist, MetricType::Cosine, HnswBuildParams::default());
        index.train(&data, 4);
        index.add(&data, &ids, 4);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();

        let mut reader = IVFHNSWSQIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, distances) = reader.search(&[1.0, 0.0, 0.0], 2, nlist, 16).unwrap();

        assert_eq!(labels[0], 10);
        assert!(distances[0].is_finite());
    }

    #[test]
    fn test_ivfhnswsq_batch_matches_single_search() {
        let d = 4;
        let nlist = 4;
        let n = 128;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i % nlist) as f32 * 100.0;
                [cluster + i as f32 * 0.01, 1.0, 2.0, 3.0]
            })
            .collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWSQIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap();
        let queries = [&data[7 * d..8 * d], &data[23 * d..24 * d]].concat();

        let mut batch_reader = IVFHNSWSQIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let (batch_labels, batch_distances) =
            search_batch_ivfhnswsq_reader(&mut batch_reader, &queries, 2, 3, nlist, 32).unwrap();

        for qi in 0..2 {
            let mut single_reader = IVFHNSWSQIndexReader::open(Cursor::new(buf.clone())).unwrap();
            let (single_labels, single_distances) = single_reader
                .search(&queries[qi * d..(qi + 1) * d], 3, nlist, 32)
                .unwrap();
            assert_eq!(
                &batch_labels[qi * 3..(qi + 1) * 3],
                single_labels.as_slice()
            );
            assert_eq!(
                &batch_distances[qi * 3..(qi + 1) * 3],
                single_distances.as_slice()
            );
        }
    }

    #[test]
    fn test_ivfhnswsq_write_requires_graphs() {
        let mut index = IVFHNSWSQIndex::new(2, 1, MetricType::L2, HnswBuildParams::default());
        let data = vec![0.0, 0.0];
        index.train(&data, 1);
        index.add(&data, &[1], 1);

        let mut buf = Vec::new();
        let err = write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("build_graphs"));
    }

    #[test]
    fn test_ivfhnswsq_writer_rejects_stale_graph() {
        let mut index = IVFHNSWSQIndex::new(2, 1, MetricType::L2, HnswBuildParams::default());
        let data = vec![0.0, 0.0, 1.0, 0.0];
        index.train(&data, 2);
        index.add(&data, &[10, 11], 2);
        index.build_graphs().unwrap();
        index.graphs[0] = Some(
            HnswGraph::build(
                &[10.0, 10.0, 11.0, 11.0],
                2,
                2,
                MetricType::L2,
                HnswBuildParams::default(),
            )
            .unwrap(),
        );

        let mut buf = Vec::new();
        let err = write_ivfhnswsq_index(&index, &mut PosWriter::new(&mut buf)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("graph does not match"));
    }
}
