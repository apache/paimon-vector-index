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

use crate::distance::{fvec_distance, preprocess_vectors, MetricType};
use crate::hnsw::{HnswBuildParams, HnswGraph};
use crate::index_io_util::{
    bytes_to_f32_vec, checked_list_bytes, checked_list_offset, checked_section_size, decode_graph,
    decode_roaring_filter, encode_graph, read_f32_vec, read_i32_le, read_i64_le, read_u32_le,
    u64_to_i64, usize_to_i32, usize_to_i64, validate_positive_i32, validate_search_inputs,
    write_f32_slice, write_i32_le, write_i64_le, write_u32_le,
};
use crate::io::{SeekRead, SeekWrite};
use crate::ivfhnswflat::IVFHNSWFlatIndex;
use crate::ivfpq::RowIdFilter;
use crate::kmeans;
use crate::topk::TopKHeap;
use std::io;

pub const IVF_HNSW_FLAT_MAGIC: u32 = 0x4948464C; // "IHFL"
pub const IVF_HNSW_FLAT_VERSION: u32 = 1;
pub const IVF_HNSW_FLAT_HEADER_SIZE: usize = 64;

pub fn write_ivfhnswflat_index(
    index: &IVFHNSWFlatIndex,
    out: &mut dyn SeekWrite,
) -> io::Result<()> {
    validate_index_shape(index)?;
    let d = index.flat.d;
    let nlist = index.flat.nlist;
    let total_vectors = index.flat.ids.iter().try_fold(0i64, |sum, ids| {
        let count = usize_to_i64(ids.len(), "total vector count")?;
        sum.checked_add(count).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "total vector count exceeds i64 length limit",
            )
        })
    })?;
    let graph_bytes: Vec<Vec<u8>> = (0..nlist)
        .map(|list_id| {
            if index.flat.ids[list_id].is_empty() {
                Ok(Vec::new())
            } else {
                encode_graph(index.graphs[list_id].as_ref())
            }
        })
        .collect::<io::Result<_>>()?;

    write_u32_le(out, IVF_HNSW_FLAT_MAGIC)?;
    write_u32_le(out, IVF_HNSW_FLAT_VERSION)?;
    write_i32_le(out, usize_to_i32(d, "dimension")?)?;
    write_i32_le(out, usize_to_i32(nlist, "nlist")?)?;
    write_u32_le(out, index.flat.metric as u32)?;
    write_i64_le(out, total_vectors)?;
    let params = index.hnsw_params.sanitized();
    write_i32_le(out, usize_to_i32(params.m, "hnsw m")?)?;
    write_i32_le(
        out,
        usize_to_i32(params.ef_construction, "hnsw ef_construction")?,
    )?;
    write_i32_le(out, usize_to_i32(params.max_level, "hnsw max_level")?)?;
    out.write_all(&[0u8; 24])?;

    write_f32_slice(out, &index.flat.quantizer_centroids)?;

    let offset_table_size = nlist.checked_mul(24).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-HNSW-FLAT offset table size overflow",
        )
    })?;
    let data_start = out
        .pos()
        .checked_add(offset_table_size as u64)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-HNSW-FLAT data start offset overflow",
            )
        })?;
    let mut list_offsets = vec![0i64; nlist];
    let mut list_counts = vec![0i32; nlist];
    let mut list_graph_bytes_lens = vec![0i32; nlist];
    let mut current_offset = data_start;

    for list_id in 0..nlist {
        list_offsets[list_id] = u64_to_i64(current_offset, "list offset")?;
        let count = index.flat.ids[list_id].len();
        list_counts[list_id] = usize_to_i32(count, "list count")?;
        list_graph_bytes_lens[list_id] = usize_to_i32(graph_bytes[list_id].len(), "graph bytes")?;
        if count > 0 {
            let payload_len = list_payload_len(count, d, graph_bytes[list_id].len())?;
            current_offset = current_offset
                .checked_add(payload_len as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "IVF-HNSW-FLAT offset overflow")
                })?;
        }
    }

    for list_id in 0..nlist {
        write_i64_le(out, list_offsets[list_id])?;
        write_i32_le(out, list_counts[list_id])?;
        write_i32_le(out, list_graph_bytes_lens[list_id])?;
        write_i64_le(out, 0)?;
    }

    for list_id in 0..nlist {
        if index.flat.ids[list_id].is_empty() {
            continue;
        }
        for &id in &index.flat.ids[list_id] {
            write_i64_le(out, id)?;
        }
        write_f32_slice(out, &index.flat.vectors[list_id])?;
        out.write_all(&graph_bytes[list_id])?;
    }

    Ok(())
}

pub struct IVFHNSWFlatIndexReader<R: SeekRead> {
    reader: R,
    pub d: usize,
    pub nlist: usize,
    pub metric: MetricType,
    pub total_vectors: i64,
    pub hnsw_params: HnswBuildParams,
    pub quantizer_centroids: Vec<f32>,
    pub list_offsets: Vec<i64>,
    pub list_counts: Vec<i32>,
    pub list_graph_bytes_lens: Vec<i32>,
    loaded: bool,
}

impl<R: SeekRead> IVFHNSWFlatIndexReader<R> {
    pub fn open(mut reader: R) -> io::Result<Self> {
        reader.seek(0)?;

        let magic = read_u32_le(&mut reader)?;
        if magic != IVF_HNSW_FLAT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid IVF_HNSW_FLAT magic: 0x{:08X}", magic),
            ));
        }
        let version = read_u32_le(&mut reader)?;
        if version != IVF_HNSW_FLAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported IVF_HNSW_FLAT version: {}", version),
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
        let mut reserved = [0u8; 24];
        reader.read_exact(&mut reserved)?;

        Ok(Self {
            reader,
            d,
            nlist,
            metric,
            total_vectors,
            hnsw_params,
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

        self.reader.seek(IVF_HNSW_FLAT_HEADER_SIZE as u64)?;
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
    ) -> io::Result<(Vec<i64>, Vec<f32>, Option<HnswGraph>)> {
        let Some(list) = self.read_graph_list(list_id)? else {
            return Ok((Vec::new(), Vec::new(), None));
        };
        let vectors = list.graph.vectors().to_vec();
        Ok((list.ids, vectors, Some(list.graph)))
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
        let payload_len = list_payload_len(count, self.d, graph_bytes_len)?;
        let mut payload = vec![0u8; payload_len];
        self.reader.pread(offset, &mut payload)?;

        let ids_bytes_len = checked_list_bytes(count, 8)?;
        let vector_bytes_len = checked_list_bytes(
            count,
            self.d.checked_mul(4).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "IVF-HNSW-FLAT bytes per vector overflow",
                )
            })?,
        )?;
        let ids = payload[..ids_bytes_len]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let vectors = bytes_to_f32_vec(&payload[ids_bytes_len..ids_bytes_len + vector_bytes_len])?;
        let graph = decode_graph(
            &payload[ids_bytes_len + vector_bytes_len..],
            vectors.clone(),
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
        Ok(Some(GraphList { ids, graph }))
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

        let q = preprocess_vectors(query, 1, self.d, self.metric);
        let (probe_indices, _) =
            kmeans::find_topk(&q, &self.quantizer_centroids, self.nlist, self.d, nprobe);
        let mut loaded_lists = Vec::with_capacity(probe_indices.len());
        for list_id in probe_indices {
            if let Some(list) = self.read_graph_list(list_id)? {
                loaded_lists.push(list);
            }
        }
        let mut heap = TopKHeap::new(k);
        let force_flat_scan = filter
            .map(|f| count_filtered(&loaded_lists, f) <= ef_search.max(k))
            .unwrap_or(false);

        for list in &loaded_lists {
            if force_flat_scan {
                scan_flat_list(
                    &q,
                    &list.ids,
                    list.graph.vectors(),
                    self.d,
                    self.metric,
                    filter,
                    &mut heap,
                );
            } else {
                let local_results = list.graph.search(&q, ef_search.max(k), ef_search.max(k));
                for (local_id, dist) in local_results {
                    let row_id = list.ids[local_id];
                    if filter.map(|f| f.contains(row_id)).unwrap_or(true) {
                        heap.push(dist, row_id);
                    }
                }
            }
        }
        if filter.is_some() && heap.len() < k {
            for list in &loaded_lists {
                scan_flat_list(
                    &q,
                    &list.ids,
                    list.graph.vectors(),
                    self.d,
                    self.metric,
                    filter,
                    &mut heap,
                );
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
        ef_search: usize,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.search_with_filter(query, k, nprobe, ef_search, Some(&filter))
    }
}

pub fn search_batch_ivfhnswflat_reader<R: SeekRead>(
    reader: &mut IVFHNSWFlatIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    ef_search: usize,
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    search_batch_ivfhnswflat_reader_filter(reader, queries, nq, k, nprobe, ef_search, None)
}

pub fn search_batch_ivfhnswflat_reader_filter<R: SeekRead>(
    reader: &mut IVFHNSWFlatIndexReader<R>,
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
            graph: list.graph,
        });
    }

    for list in &loaded_lists {
        for &qi in &list.query_ids {
            let query = &processed[qi * reader.d..(qi + 1) * reader.d];
            let force_flat_scan = filter
                .map(|_| query_filtered_counts[qi] <= ef_search.max(k))
                .unwrap_or(false);
            if force_flat_scan {
                scan_flat_list(
                    query,
                    &list.ids,
                    list.graph.vectors(),
                    reader.d,
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
                let query = &processed[qi * reader.d..(qi + 1) * reader.d];
                scan_flat_list(
                    query,
                    &list.ids,
                    list.graph.vectors(),
                    reader.d,
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

pub fn search_batch_ivfhnswflat_reader_roaring_filter<R: SeekRead>(
    reader: &mut IVFHNSWFlatIndexReader<R>,
    queries: &[f32],
    nq: usize,
    k: usize,
    nprobe: usize,
    ef_search: usize,
    roaring_filter_bytes: &[u8],
) -> io::Result<(Vec<i64>, Vec<f32>)> {
    let filter = decode_roaring_filter(roaring_filter_bytes)?;
    search_batch_ivfhnswflat_reader_filter(reader, queries, nq, k, nprobe, ef_search, Some(&filter))
}

struct GraphList {
    ids: Vec<i64>,
    graph: HnswGraph,
}

struct LoadedBatchList {
    query_ids: Vec<usize>,
    ids: Vec<i64>,
    graph: HnswGraph,
}

fn count_filtered(lists: &[GraphList], filter: &dyn RowIdFilter) -> usize {
    lists
        .iter()
        .map(|list| list.ids.iter().filter(|&&id| filter.contains(id)).count())
        .sum()
}

fn scan_flat_list(
    query: &[f32],
    ids: &[i64],
    vectors: &[f32],
    d: usize,
    metric: MetricType,
    filter: Option<&dyn RowIdFilter>,
    heap: &mut TopKHeap,
) {
    for (local_id, &row_id) in ids.iter().enumerate() {
        if filter.map(|f| !f.contains(row_id)).unwrap_or(false) {
            continue;
        }
        let vector = &vectors[local_id * d..(local_id + 1) * d];
        heap.push(fvec_distance(query, vector, metric), row_id);
    }
}

fn validate_index_shape(index: &IVFHNSWFlatIndex) -> io::Result<()> {
    if index.flat.d == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dimension must be greater than 0",
        ));
    }
    if index.flat.nlist == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "nlist must be greater than 0",
        ));
    }
    if index.flat.ids.len() != index.flat.nlist
        || index.flat.vectors.len() != index.flat.nlist
        || index.graphs.len() != index.flat.nlist
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IVF-HNSW-FLAT list storage does not match nlist",
        ));
    }
    let centroid_len = checked_section_size(index.flat.nlist, index.flat.d)?;
    if index.flat.quantizer_centroids.len() != centroid_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "centroid length {} does not match nlist*d {}",
                index.flat.quantizer_centroids.len(),
                centroid_len
            ),
        ));
    }
    for list_id in 0..index.flat.nlist {
        let count = index.flat.ids[list_id].len();
        let expected_vector_len = count.checked_mul(index.flat.d).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-HNSW-FLAT vector length overflow",
            )
        })?;
        if index.flat.vectors[list_id].len() != expected_vector_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "list {} vector length {} does not match ids*d {}",
                    list_id,
                    index.flat.vectors[list_id].len(),
                    expected_vector_len
                ),
            ));
        }
        match &index.graphs[list_id] {
            Some(graph)
                if graph.len() == count
                    && graph.vectors() == index.flat.vectors[list_id].as_slice() => {}
            Some(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("list {} graph does not match vector storage", list_id),
                ));
            }
            None if count == 0 => {}
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("list {} is missing HNSW graph", list_id),
                ));
            }
        }
    }
    Ok(())
}

fn list_payload_len(count: usize, d: usize, graph_bytes_len: usize) -> io::Result<usize> {
    let id_bytes = checked_list_bytes(count, 8)?;
    let vector_bytes = checked_list_bytes(
        count,
        d.checked_mul(4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-HNSW-FLAT bytes per vector overflow",
            )
        })?,
    )?;
    id_bytes
        .checked_add(vector_bytes)
        .and_then(|len| len.checked_add(graph_bytes_len))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "IVF-HNSW-FLAT list payload overflow",
            )
        })
}

#[cfg(test)]
mod tests {
    use crate::distance::MetricType;
    use crate::hnsw::HnswBuildParams;
    use crate::index_io_util::decode_graph;
    use crate::io::{PosWriter, SeekRead};
    use crate::ivfhnswflat::IVFHNSWFlatIndex;
    use crate::ivfhnswflat_io::{
        search_batch_ivfhnswflat_reader, search_batch_ivfhnswflat_reader_roaring_filter,
        write_ivfhnswflat_index, IVFHNSWFlatIndexReader, IVF_HNSW_FLAT_HEADER_SIZE,
    };
    use roaring::RoaringTreemap;
    use std::io;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_ivfhnswflat_write_read_search_roundtrip() {
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

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let query = &data[7 * d..8 * d];
        let mut expected_distances = vec![0.0; 5];
        let mut expected_labels = vec![0; 5];
        index.search(
            query,
            1,
            5,
            nlist,
            32,
            &mut expected_distances,
            &mut expected_labels,
        );

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, distances) = reader.search(query, 5, nlist, 32).unwrap();

        assert_eq!(labels, expected_labels);
        assert_eq!(distances, expected_distances);
    }

    #[test]
    fn test_ivfhnswflat_write_read_search_roundtrip_cosine() {
        let d = 3;
        let nlist = 2;
        let data = vec![1.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.1];
        let ids = vec![10, 11, 12, 13];

        let mut index =
            IVFHNSWFlatIndex::new(d, nlist, MetricType::Cosine, HnswBuildParams::default());
        index.train(&data, 4);
        index.add(&data, &ids, 4);
        index.build_graphs().unwrap();

        let query = [9.0, 1.0, 0.0];
        let mut expected_distances = vec![0.0; 2];
        let mut expected_labels = vec![0; 2];
        index.search(
            &query,
            1,
            2,
            nlist,
            8,
            &mut expected_distances,
            &mut expected_labels,
        );

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, distances) = reader.search(&query, 2, nlist, 8).unwrap();

        assert_eq!(labels, expected_labels);
        assert_eq!(distances, expected_distances);
    }

    #[test]
    fn test_ivfhnswflat_reader_filter_backfills_exact_results() {
        use std::collections::HashSet;

        let d = 2;
        let nlist = 1;
        let n = 128;
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            data.push(i as f32);
            data.push(0.0);
        }
        let ids: Vec<i64> = (0..n as i64).collect();

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();

        let filter: HashSet<i64> = (0..n as i64).filter(|id| id % 2 == 0).collect();
        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let (labels, _) = reader
            .search_with_filter(&[127.0, 0.0], 10, 1, 1, Some(&filter))
            .unwrap();

        assert_eq!(
            labels,
            vec![126, 124, 122, 120, 118, 116, 114, 112, 110, 108]
        );
    }

    #[test]
    fn test_ivfhnswflat_batch_reader_matches_single_reader_search() {
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

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, n);
        index.add(&data, &ids, n);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();

        let queries = [&data[7 * d..8 * d], &data[63 * d..64 * d]].concat();
        let k = 5;
        let nprobe = 3;
        let ef_search = 32;
        let mut batch_reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
        let (batch_labels, batch_distances) =
            search_batch_ivfhnswflat_reader(&mut batch_reader, &queries, 2, k, nprobe, ef_search)
                .unwrap();

        for qi in 0..2 {
            let mut single_reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
            let query = &queries[qi * d..(qi + 1) * d];
            let (single_labels, single_distances) =
                single_reader.search(query, k, nprobe, ef_search).unwrap();
            assert_eq!(&batch_labels[qi * k..(qi + 1) * k], single_labels);
            assert_eq!(&batch_distances[qi * k..(qi + 1) * k], single_distances);
        }
    }

    #[test]
    fn test_ivfhnswflat_batch_reader_search_with_roaring_filter_bytes() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0];
        let ids = vec![10, 11, 12];

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 3);
        index.add(&data, &ids, 3);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();

        let mut allowed = RoaringTreemap::new();
        allowed.insert(12);
        let mut filter_bytes = Vec::new();
        allowed.serialize_into(&mut filter_bytes).unwrap();

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let queries = vec![0.0, 0.0, 10.0, 10.0];
        let (labels, distances) = search_batch_ivfhnswflat_reader_roaring_filter(
            &mut reader,
            &queries,
            2,
            2,
            1,
            4,
            &filter_bytes,
        )
        .unwrap();

        assert_eq!(labels, vec![12, -1, 12, -1]);
        assert_eq!(distances, vec![200.0, f32::MAX, 0.0, f32::MAX]);
    }

    #[test]
    fn test_ivfhnswflat_batch_reader_validates_inputs() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 1.0, 0.0];
        let ids = vec![10, 11];

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 2);
        index.add(&data, &ids, 2);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
        assert!(search_batch_ivfhnswflat_reader(&mut reader, &[], 0, 1, 1, 4).is_err());

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
        assert!(search_batch_ivfhnswflat_reader(&mut reader, &[0.0], 1, 1, 1, 4).is_err());

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf.clone())).unwrap();
        assert!(search_batch_ivfhnswflat_reader(&mut reader, &[0.0, 0.0], 1, 0, 1, 4).is_err());

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf)).unwrap();
        assert!(search_batch_ivfhnswflat_reader(&mut reader, &[0.0, 0.0], 1, 1, 0, 4).is_err());
    }

    #[test]
    fn test_ivfhnswflat_reader_filter_reads_probed_list_once() {
        use std::collections::HashSet;

        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let ids = vec![10, 11, 12, 13];

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 4);
        index.add(&data, &ids, 4);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();

        let pread_count = Arc::new(AtomicUsize::new(0));
        let cursor = CountingPreadCursor::new(buf, Arc::clone(&pread_count));
        let filter: HashSet<i64> = [10, 12].into_iter().collect();
        let mut reader = IVFHNSWFlatIndexReader::open(cursor).unwrap();

        reader
            .search_with_filter(&[0.0, 0.0], 2, 1, 1, Some(&filter))
            .unwrap();

        assert_eq!(pread_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_ivfhnswflat_reader_rejects_truncated_graph_section() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let ids = vec![10, 11, 12, 13];

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 4);
        index.add(&data, &ids, 4);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();
        buf.pop();

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let err = reader.search(&[0.0, 0.0], 2, 1, 4).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn test_ivfhnswflat_reader_rejects_missing_graph_section() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let ids = vec![10, 11, 12, 13];

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 4);
        index.add(&data, &ids, 4);
        index.build_graphs().unwrap();

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        write_ivfhnswflat_index(&index, &mut writer).unwrap();
        let graph_len_offset = IVF_HNSW_FLAT_HEADER_SIZE + nlist * d * 4 + 8 + 4;
        buf[graph_len_offset..graph_len_offset + 4].copy_from_slice(&0i32.to_le_bytes());

        let mut reader = IVFHNSWFlatIndexReader::open(Cursor::new(buf)).unwrap();
        let err = reader.search(&[0.0, 0.0], 2, 1, 4).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("missing HNSW graph"));
    }

    #[test]
    fn test_ivfhnswflat_decoder_rejects_level_above_hnsw_max_before_allocation() {
        let params = HnswBuildParams {
            m: 2,
            ef_construction: 8,
            max_level: 3,
        };
        let mut graph_bytes = Vec::new();
        append_u32(&mut graph_bytes, 1);
        append_u32(&mut graph_bytes, 0);
        append_u32(&mut graph_bytes, 0);
        append_u32(&mut graph_bytes, params.max_level as u32 + 1);

        let err =
            decode_graph(&graph_bytes, vec![0.0, 0.0], 1, 2, MetricType::L2, params).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("level"));
    }

    #[test]
    fn test_ivfhnswflat_decoder_rejects_degree_above_hnsw_bound_before_allocation() {
        let params = HnswBuildParams {
            m: 2,
            ef_construction: 8,
            max_level: 3,
        };
        let mut graph_bytes = Vec::new();
        append_u32(&mut graph_bytes, 1);
        append_u32(&mut graph_bytes, 0);
        append_u32(&mut graph_bytes, 0);
        append_u32(&mut graph_bytes, 0);
        append_u32(&mut graph_bytes, (params.m * 2) as u32 + 1);

        let err =
            decode_graph(&graph_bytes, vec![0.0, 0.0], 1, 2, MetricType::L2, params).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("degree"));
    }

    #[test]
    fn test_ivfhnswflat_writer_requires_built_graphs() {
        let d = 2;
        let nlist = 1;
        let data = vec![0.0, 0.0, 1.0, 0.0];
        let ids = vec![10, 11];

        let mut index = IVFHNSWFlatIndex::new(d, nlist, MetricType::L2, HnswBuildParams::default());
        index.train(&data, 2);
        index.add(&data, &ids, 2);

        let mut buf = Vec::new();
        let mut writer = PosWriter::new(&mut buf);
        let err = write_ivfhnswflat_index(&index, &mut writer).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("missing HNSW graph"));
    }

    struct CountingPreadCursor {
        data: Vec<u8>,
        pos: usize,
        pread_count: Arc<AtomicUsize>,
    }

    impl CountingPreadCursor {
        fn new(data: Vec<u8>, pread_count: Arc<AtomicUsize>) -> Self {
            Self {
                data,
                pos: 0,
                pread_count,
            }
        }
    }

    impl SeekRead for CountingPreadCursor {
        fn seek(&mut self, pos: u64) -> io::Result<()> {
            self.pos = pos as usize;
            Ok(())
        }

        fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
            let end = self.pos.checked_add(buf.len()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::UnexpectedEof, "cursor position overflow")
            })?;
            if end > self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            buf.copy_from_slice(&self.data[self.pos..end]);
            self.pos = end;
            Ok(())
        }

        fn pread(&mut self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
            self.pread_count.fetch_add(1, Ordering::SeqCst);
            let pos = pos as usize;
            let end = pos.checked_add(buf.len()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::UnexpectedEof, "cursor position overflow")
            })?;
            if end > self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            buf.copy_from_slice(&self.data[pos..end]);
            Ok(())
        }
    }

    fn append_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }
}
