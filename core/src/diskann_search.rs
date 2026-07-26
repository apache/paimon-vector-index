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

use crate::diskann::DiskAnnRawVectorEncoding;
use crate::diskann_io::{
    decode_adjacency_list, CacheLockMetrics, OffsetLru, SectionRange, SharedWindowCacheLookup,
    DISKANN_PAGE_SIZE,
};
use crate::distance::{
    fvec_distance, fvec_l2sqr, pq_distance_four_codes, pq_distance_from_table, preprocess_vectors,
    MetricType,
};
use crate::index_io_util::decode_roaring_filter;
use crate::io::{ReadRequest, SeekRead};
use crate::read_options::ReadPlan;
use crate::sparse_table::{estimated_memory_bytes as sparse_table_memory_bytes, SparseTable};
use half::prelude::{HalfBitsSliceExt, HalfFloatSliceExt};
use rayon::prelude::*;
use roaring::{RoaringBitmap, RoaringTreemap};
use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::io;
use std::sync::Arc;

const QUERY_WINDOW_BUFFER_LIMIT_BYTES: usize = 8 * 1024 * 1024;
const QUERY_ADJACENCY_WINDOW_LIMIT_BYTES: usize = 8 * 1024 * 1024;
const BATCH_WINDOW_BUFFER_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const FILTERED_BATCH_RERANK_MAX_BYTES: usize = 64 * 1024 * 1024;
const FILTERED_BATCH_RERANK_MAX_RANGES: usize = 1024;
const BATCH_QUERY_CHUNK_SIZE: usize = 1024;
const FILTERED_PQ_MAX_QUERY_TILE_SIZE: usize = 4;
const FILTERED_PQ_TILE_TABLE_LIMIT_BYTES: usize = 2 * 1024 * 1024;
const FILTERED_SINGLE_PQ_NODE_CHUNK_SIZE: usize = 1024;
const PARALLEL_EXACT_RERANK_MIN_COMPONENTS: usize = 16 * 1024;
const PARALLEL_SESSION_MAX_QUERIES_PER_WORKER: usize = 4;
const SPARSE_VISITED_MIN_MEMORY_SAVINGS: usize = 2;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DiskAnnSearchStats {
    pub query_count: usize,
    pub query_chunks: usize,
    pub max_queries_per_chunk: usize,
    pub filtered_exhaustive_queries: usize,
    pub filtered_graph_queries: usize,
    pub filtered_graph_fallbacks: usize,
    pub pq_distance_evaluations: usize,
    pub pq_code_loads: usize,
    pub adjacency_cache_hits: usize,
    pub adjacency_cache_misses: usize,
    pub adjacency_cache_waits: usize,
    pub adjacency_cache_evictions: usize,
    pub adjacency_cache_lock_acquisitions: usize,
    pub adjacency_cache_lock_wait_nanos: u64,
    pub query_adjacency_cache_peak_bytes: usize,
    pub query_adjacency_cache_evictions: usize,
    pub rerank_candidate_references: usize,
    pub rerank_unique_windows: usize,
    pub rerank_chunks: usize,
    pub raw_vector_cache_hits: usize,
    pub raw_vector_cache_misses: usize,
    pub raw_vector_cache_evictions: usize,
    pub parallel_exact_rerank_chunks: usize,
    pub parallel_exact_rerank_references: usize,
    pub parallel_session_queries: usize,
}

impl DiskAnnSearchStats {
    fn record_adjacency_cache_lock(&mut self, metrics: CacheLockMetrics) {
        self.adjacency_cache_lock_acquisitions = self
            .adjacency_cache_lock_acquisitions
            .saturating_add(metrics.acquisitions);
        self.adjacency_cache_lock_wait_nanos = self
            .adjacency_cache_lock_wait_nanos
            .saturating_add(metrics.wait_nanos);
    }

    fn merge_candidate_generation(&mut self, worker: Self) {
        self.filtered_exhaustive_queries = self
            .filtered_exhaustive_queries
            .saturating_add(worker.filtered_exhaustive_queries);
        self.filtered_graph_queries = self
            .filtered_graph_queries
            .saturating_add(worker.filtered_graph_queries);
        self.filtered_graph_fallbacks = self
            .filtered_graph_fallbacks
            .saturating_add(worker.filtered_graph_fallbacks);
        self.pq_distance_evaluations = self
            .pq_distance_evaluations
            .saturating_add(worker.pq_distance_evaluations);
        self.pq_code_loads = self.pq_code_loads.saturating_add(worker.pq_code_loads);
        self.adjacency_cache_hits = self
            .adjacency_cache_hits
            .saturating_add(worker.adjacency_cache_hits);
        self.adjacency_cache_misses = self
            .adjacency_cache_misses
            .saturating_add(worker.adjacency_cache_misses);
        self.adjacency_cache_waits = self
            .adjacency_cache_waits
            .saturating_add(worker.adjacency_cache_waits);
        self.adjacency_cache_evictions = self
            .adjacency_cache_evictions
            .saturating_add(worker.adjacency_cache_evictions);
        self.adjacency_cache_lock_acquisitions = self
            .adjacency_cache_lock_acquisitions
            .saturating_add(worker.adjacency_cache_lock_acquisitions);
        self.adjacency_cache_lock_wait_nanos = self
            .adjacency_cache_lock_wait_nanos
            .saturating_add(worker.adjacency_cache_lock_wait_nanos);
        self.query_adjacency_cache_peak_bytes = self
            .query_adjacency_cache_peak_bytes
            .max(worker.query_adjacency_cache_peak_bytes);
        self.query_adjacency_cache_evictions = self
            .query_adjacency_cache_evictions
            .saturating_add(worker.query_adjacency_cache_evictions);
    }

    fn merge_complete_query(&mut self, worker: Self) {
        let worker_query_count = worker.query_count;
        self.merge_candidate_generation(worker);
        self.rerank_candidate_references = self
            .rerank_candidate_references
            .saturating_add(worker.rerank_candidate_references);
        self.rerank_unique_windows = self
            .rerank_unique_windows
            .saturating_add(worker.rerank_unique_windows);
        self.rerank_chunks = self.rerank_chunks.saturating_add(worker.rerank_chunks);
        self.raw_vector_cache_hits = self
            .raw_vector_cache_hits
            .saturating_add(worker.raw_vector_cache_hits);
        self.raw_vector_cache_misses = self
            .raw_vector_cache_misses
            .saturating_add(worker.raw_vector_cache_misses);
        self.raw_vector_cache_evictions = self
            .raw_vector_cache_evictions
            .saturating_add(worker.raw_vector_cache_evictions);
        self.parallel_exact_rerank_chunks = self
            .parallel_exact_rerank_chunks
            .saturating_add(worker.parallel_exact_rerank_chunks);
        self.parallel_exact_rerank_references = self
            .parallel_exact_rerank_references
            .saturating_add(worker.parallel_exact_rerank_references);
        self.parallel_session_queries = self
            .parallel_session_queries
            .saturating_add(worker_query_count.max(1));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadWindow {
    pub offset: u64,
    pub length: usize,
}

impl ReadWindow {
    pub const fn new(offset: u64, length: usize) -> Self {
        Self { offset, length }
    }
}

pub(crate) struct ReadWindowPlanner {
    plan: ReadPlan,
    section: SectionRange,
}

impl ReadWindowPlanner {
    pub const fn new(plan: ReadPlan, section: SectionRange) -> Self {
        Self { plan, section }
    }

    #[cfg(test)]
    pub const fn beam_width(&self) -> usize {
        self.plan.graph_beam_width
    }

    pub fn plan_logical_pages(
        &self,
        logical_pages: impl IntoIterator<Item = usize>,
    ) -> Vec<ReadWindow> {
        let mut windows = BTreeMap::new();
        for logical_page in logical_pages {
            if let Some(window) = self.window_for_logical_page(logical_page) {
                windows.insert(window.offset, window.length);
            }
        }
        windows
            .into_iter()
            .map(|(offset, length)| ReadWindow::new(offset, length))
            .collect()
    }

    fn window_for_logical_page(&self, logical_page: usize) -> Option<ReadWindow> {
        let window_size = self.plan.window_bytes as u64;
        let relative_page = (logical_page as u64).checked_mul(DISKANN_PAGE_SIZE as u64)?;
        if relative_page >= self.section.length {
            return None;
        }
        let relative_window = relative_page / window_size * window_size;
        let length = window_size.min(self.section.length - relative_window) as usize;
        Some(ReadWindow::new(
            self.section.offset + relative_window,
            length,
        ))
    }
}

pub(crate) struct VectorWindowPlanner {
    section: SectionRange,
    record_size: usize,
    records_per_window: usize,
}

impl VectorWindowPlanner {
    fn new(plan: ReadPlan, section: SectionRange, record_size: usize) -> io::Result<Self> {
        if record_size == 0 {
            return Err(invalid_data(
                "DiskANN raw-vector record size must be greater than zero",
            ));
        }
        Ok(Self {
            section,
            record_size,
            records_per_window: (plan.window_bytes / record_size).max(1),
        })
    }

    fn window_for_node(&self, node: usize) -> Option<ReadWindow> {
        let record_offset = node.checked_mul(self.record_size)?;
        if u64::try_from(record_offset).ok()? >= self.section.length {
            return None;
        }
        let first_node = node / self.records_per_window * self.records_per_window;
        let relative_offset = first_node.checked_mul(self.record_size)?;
        let maximum_length = self.records_per_window.checked_mul(self.record_size)?;
        let remaining = usize::try_from(self.section.length - relative_offset as u64).ok()?;
        Some(ReadWindow::new(
            self.section.offset.checked_add(relative_offset as u64)?,
            maximum_length.min(remaining),
        ))
    }

    fn plan_nodes(&self, nodes: impl IntoIterator<Item = usize>) -> Vec<ReadWindow> {
        let mut windows = BTreeMap::new();
        for node in nodes {
            if let Some(window) = self.window_for_node(node) {
                windows.insert(window.offset, window.length);
            }
        }
        windows
            .into_iter()
            .map(|(offset, length)| ReadWindow::new(offset, length))
            .collect()
    }

    fn record<'a>(
        &self,
        window: ReadWindow,
        payload: &'a [u8],
        node: usize,
    ) -> io::Result<&'a [u8]> {
        let absolute_offset = self
            .section
            .offset
            .checked_add(
                u64::try_from(
                    node.checked_mul(self.record_size)
                        .ok_or_else(|| invalid_data("DiskANN raw-vector offset overflows"))?,
                )
                .map_err(|_| invalid_data("DiskANN raw-vector offset exceeds u64"))?,
            )
            .ok_or_else(|| invalid_data("DiskANN raw-vector offset overflows"))?;
        let relative_offset = absolute_offset
            .checked_sub(window.offset)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| invalid_data("DiskANN raw-vector window starts after its record"))?;
        let record_end = relative_offset
            .checked_add(self.record_size)
            .ok_or_else(|| invalid_data("DiskANN raw-vector record range overflows"))?;
        payload
            .get(relative_offset..record_end)
            .ok_or_else(|| invalid_data("DiskANN raw-vector record is truncated"))
    }
}

#[derive(Debug, Clone, Copy)]
struct SearchCandidate {
    node: usize,
    distance: f32,
}

#[derive(Debug, Clone, Copy)]
struct ExactSearchResult {
    row_id: i64,
    distance: f32,
}

#[derive(Clone, Copy)]
struct ExactRerankReference<'a> {
    query_index: usize,
    row_id: i64,
    record: &'a [u8],
}

enum WindowPayload {
    Owned(Vec<u8>),
    Shared(Arc<Vec<u8>>),
}

impl WindowPayload {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(payload) => payload,
            Self::Shared(payload) => payload,
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Owned(payload) => payload.capacity(),
            Self::Shared(payload) => payload.capacity(),
        }
    }
}

impl From<Vec<u8>> for WindowPayload {
    fn from(payload: Vec<u8>) -> Self {
        Self::Owned(payload)
    }
}

#[derive(Default)]
struct AdjacencyWindowCache {
    entries: HashMap<u64, WindowPayload>,
    recency: OffsetLru,
    retained_capacity: usize,
}

impl AdjacencyWindowCache {
    fn contains_key(&self, offset: &u64) -> bool {
        self.entries.contains_key(offset)
    }

    fn get(&self, offset: &u64) -> Option<&WindowPayload> {
        self.entries.get(offset)
    }

    fn insert(&mut self, offset: u64, payload: WindowPayload) {
        let payload_capacity = payload.capacity();
        if let Some(previous) = self.entries.insert(offset, payload) {
            self.retained_capacity = self.retained_capacity.saturating_sub(previous.capacity());
            self.recency.remove(offset);
        }
        self.retained_capacity = self.retained_capacity.saturating_add(payload_capacity);
        self.recency.touch(offset);
    }

    fn touch_windows(&mut self, windows: &[ReadWindow]) {
        for window in windows {
            if self.entries.contains_key(&window.offset) {
                self.recency.touch(window.offset);
            }
        }
    }

    fn trim(&mut self, window_buffers: &mut WindowBufferPool, capacity_limit: usize) -> usize {
        let mut evictions = 0usize;
        while self.retained_capacity > capacity_limit {
            let Some(offset) = self.recency.pop_oldest() else {
                break;
            };
            if let Some(payload) = self.entries.remove(&offset) {
                self.retained_capacity = self.retained_capacity.saturating_sub(payload.capacity());
                if let WindowPayload::Owned(payload) = payload {
                    window_buffers.recycle(payload);
                }
                evictions = evictions.saturating_add(1);
            }
        }
        evictions
    }

    fn recycle(&mut self, window_buffers: &mut WindowBufferPool) {
        for (_, payload) in self.entries.drain() {
            if let WindowPayload::Owned(payload) = payload {
                window_buffers.recycle(payload);
            }
        }
        self.recency.clear();
        self.retained_capacity = 0;
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn retained_capacity(&self) -> usize {
        self.retained_capacity
    }

    #[cfg(test)]
    fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional);
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.entries.capacity()
    }
}

fn prepare_adjacency_window_cache(
    required_windows: &[ReadWindow],
    incoming_bytes: usize,
    cache: &mut AdjacencyWindowCache,
    window_buffers: &mut WindowBufferPool,
) -> usize {
    // A window that was available when the read plan was assembled may still be
    // needed by this round even when its individual page does not need loading.
    // Mark all round inputs as most-recent before making room for missing
    // windows, otherwise trimming for a different page can evict one that
    // decode is about to consume.
    cache.touch_windows(required_windows);
    cache.trim(
        window_buffers,
        QUERY_ADJACENCY_WINDOW_LIMIT_BYTES.saturating_sub(incoming_bytes),
    )
}

fn share_window_payload(payload: Vec<u8>) -> Arc<Vec<u8>> {
    Arc::new(payload)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilteredCandidateStrategy {
    Exhaustive {
        target_candidates: usize,
    },
    Graph {
        target_candidates: usize,
        search_list_size: usize,
    },
}

#[derive(Default)]
pub(crate) struct DiskAnnQueryScratch {
    visited: Vec<bool>,
    sparse_visited: SparseTable<()>,
    uses_sparse_visited: bool,
    touched_nodes: Vec<usize>,
    distance_table: Vec<f32>,
    candidates: Vec<SearchCandidate>,
    rerank_candidates: Vec<SearchCandidate>,
    rerank_windows: HashSet<u64>,
    retained_candidates: BinaryHeap<SearchCandidate>,
    frontier: BinaryHeap<Reverse<SearchCandidate>>,
    selected_nodes: Vec<usize>,
    loaded_adjacency_pages: HashSet<usize>,
    adjacency_windows: AdjacencyWindowCache,
    vector_windows: VectorWindowCache,
    window_buffers: WindowBufferPool,
    neighbor_buffer: Vec<u32>,
    scored_neighbors: Vec<SearchCandidate>,
}

#[derive(Default)]
struct VectorWindowCache {
    entries: HashMap<u64, WindowPayload>,
    recency: OffsetLru,
    retained_capacity: usize,
}

#[derive(Debug, Default)]
struct VectorWindowLoadStats {
    hits: usize,
    misses: usize,
    evictions: usize,
}

impl VectorWindowCache {
    fn contains_key(&self, offset: &u64) -> bool {
        self.entries.contains_key(offset)
    }

    fn get(&self, offset: &u64) -> Option<&[u8]> {
        self.entries.get(offset).map(WindowPayload::as_slice)
    }

    fn insert(&mut self, offset: u64, payload: impl Into<WindowPayload>) {
        let payload = payload.into();
        let payload_capacity = payload.capacity();
        if let Some(previous) = self.entries.insert(offset, payload) {
            self.retained_capacity = self.retained_capacity.saturating_sub(previous.capacity());
            self.recency.remove(offset);
        }
        self.retained_capacity = self.retained_capacity.saturating_add(payload_capacity);
        self.recency.touch(offset);
    }

    #[cfg(test)]
    fn remove(&mut self, offset: u64) -> Option<WindowPayload> {
        let payload = self.entries.remove(&offset)?;
        self.retained_capacity = self.retained_capacity.saturating_sub(payload.capacity());
        self.recency.remove(offset);
        Some(payload)
    }

    #[cfg(test)]
    fn touch(&mut self, offset: u64) {
        if self.entries.contains_key(&offset) {
            self.recency.touch(offset);
        }
    }

    fn touch_windows(&mut self, windows: &[ReadWindow]) {
        for window in windows {
            debug_assert!(self.entries.contains_key(&window.offset));
            self.recency.touch(window.offset);
        }
    }

    fn trim(&mut self, window_buffers: &mut WindowBufferPool, capacity_limit: usize) -> usize {
        let mut evictions = 0usize;
        while self.retained_capacity > capacity_limit {
            let Some(offset) = self.recency.pop_oldest() else {
                break;
            };
            if let Some(payload) = self.entries.remove(&offset) {
                self.retained_capacity = self.retained_capacity.saturating_sub(payload.capacity());
                if let WindowPayload::Owned(payload) = payload {
                    window_buffers.recycle(payload);
                }
                evictions = evictions.saturating_add(1);
            }
        }
        evictions
    }

    fn recycle(&mut self, window_buffers: &mut WindowBufferPool) {
        for (_, payload) in self.entries.drain() {
            if let WindowPayload::Owned(payload) = payload {
                window_buffers.recycle(payload);
            }
        }
        self.recency.clear();
        self.retained_capacity = 0;
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[cfg(test)]
    fn retained_capacity(&self) -> usize {
        self.retained_capacity
    }
}

struct WindowBufferPool {
    buffers: Vec<Vec<u8>>,
    retained_capacity: usize,
    retained_capacity_limit: usize,
}

impl Default for WindowBufferPool {
    fn default() -> Self {
        Self {
            buffers: Vec::new(),
            retained_capacity: 0,
            retained_capacity_limit: QUERY_WINDOW_BUFFER_LIMIT_BYTES,
        }
    }
}

impl WindowBufferPool {
    #[cfg(test)]
    fn with_retained_capacity_limit(retained_capacity_limit: usize) -> Self {
        Self {
            retained_capacity_limit,
            ..Self::default()
        }
    }

    fn recycle(&mut self, mut buffer: Vec<u8>) {
        buffer.clear();
        let capacity = buffer.capacity();
        let Some(retained_capacity) = self.retained_capacity.checked_add(capacity) else {
            return;
        };
        if retained_capacity > self.retained_capacity_limit {
            return;
        }
        self.retained_capacity = retained_capacity;
        self.buffers.push(buffer);
    }

    fn take(&mut self, len: usize) -> io::Result<Vec<u8>> {
        let best_fit = self
            .buffers
            .last()
            .is_some_and(|buffer| buffer.capacity() == len)
            .then(|| self.buffers.len() - 1)
            .or_else(|| {
                self.buffers
                    .iter()
                    .enumerate()
                    .filter(|(_, buffer)| buffer.capacity() >= len)
                    .min_by_key(|(_, buffer)| buffer.capacity())
                    .map(|(index, _)| index)
            })
            .or_else(|| {
                self.buffers
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, buffer)| buffer.capacity())
                    .map(|(index, _)| index)
            });
        let mut buffer = if let Some(index) = best_fit {
            let buffer = self.buffers.swap_remove(index);
            self.retained_capacity -= buffer.capacity();
            buffer
        } else {
            Vec::new()
        };
        let additional_capacity = len.saturating_sub(buffer.capacity());
        if additional_capacity != 0 && buffer.try_reserve_exact(additional_capacity).is_err() {
            self.recycle(buffer);
            return Err(invalid_input(
                "DiskANN query window buffer allocation failed",
            ));
        }
        buffer.resize(len, 0);
        Ok(buffer)
    }

    fn set_retained_capacity_limit(&mut self, retained_capacity_limit: usize) {
        self.retained_capacity_limit = retained_capacity_limit;
        while self.retained_capacity > self.retained_capacity_limit {
            let Some(buffer) = self.buffers.pop() else {
                self.retained_capacity = 0;
                break;
            };
            self.retained_capacity -= buffer.capacity();
        }
    }
}

impl DiskAnnQueryScratch {
    #[cfg(test)]
    fn with_window_buffer_limit(retained_capacity_limit: usize) -> Self {
        Self {
            window_buffers: WindowBufferPool::with_retained_capacity_limit(retained_capacity_limit),
            ..Self::default()
        }
    }

    fn set_window_buffer_limit(&mut self, retained_capacity_limit: usize) {
        self.window_buffers
            .set_retained_capacity_limit(retained_capacity_limit);
    }

    #[cfg(test)]
    fn begin_search(&mut self, vector_count: usize) {
        self.begin_graph_search(vector_count, vector_count, 1)
            .expect("test-sized DiskANN visited allocation");
    }

    fn begin_graph_search(
        &mut self,
        vector_count: usize,
        search_list_size: usize,
        max_degree: usize,
    ) -> io::Result<()> {
        self.begin_rerank();
        let expected_visited = search_list_size
            .saturating_mul(max_degree)
            .saturating_add(1)
            .min(vector_count);
        let dense_bytes = vector_count.div_ceil(8);
        let sparse_bytes =
            sparse_table_memory_bytes(expected_visited, size_of::<()>()).unwrap_or(usize::MAX);
        // Dense bitmap probes are substantially cheaper than open-addressed
        // hashing. Prefer them unless sparse storage saves at least 2x memory.
        self.uses_sparse_visited = sparse_bytes
            .checked_mul(SPARSE_VISITED_MIN_MEMORY_SAVINGS)
            .is_some_and(|threshold| threshold < dense_bytes);
        if self.uses_sparse_visited {
            if expected_visited > self.sparse_visited.entry_capacity() {
                self.sparse_visited = SparseTable::try_with_capacity(expected_visited)
                    .map_err(|_| invalid_input("DiskANN sparse visited allocation failed"))?;
            }
        } else {
            self.visited.resize(vector_count, false);
        }
        Ok(())
    }

    fn begin_rerank(&mut self) {
        if self.uses_sparse_visited {
            self.sparse_visited.clear();
            self.touched_nodes.clear();
        } else {
            for node in self.touched_nodes.drain(..) {
                self.visited[node] = false;
            }
        }
        self.candidates.clear();
        self.rerank_candidates.clear();
        self.rerank_windows.clear();
        self.retained_candidates.clear();
        self.frontier.clear();
        self.selected_nodes.clear();
        self.loaded_adjacency_pages.clear();
        self.recycle_adjacency_windows();
        self.neighbor_buffer.clear();
        self.scored_neighbors.clear();
    }

    fn recycle_adjacency_windows(&mut self) {
        self.adjacency_windows.recycle(&mut self.window_buffers);
    }

    fn recycle_vector_windows(&mut self) {
        self.vector_windows.recycle(&mut self.window_buffers);
    }

    fn recycle_window_caches(&mut self) {
        self.recycle_adjacency_windows();
        self.recycle_vector_windows();
    }

    fn prepare_distance_table(&mut self, len: usize) -> &mut [f32] {
        self.distance_table.resize(len, 0.0);
        &mut self.distance_table
    }

    fn select_round(&mut self, limit: usize) {
        self.selected_nodes.clear();
        while self.selected_nodes.len() < limit {
            let Some(Reverse(candidate)) = self.frontier.pop() else {
                break;
            };
            if self
                .retained_candidates
                .peek()
                .is_some_and(|worst| candidate > *worst)
            {
                self.frontier.clear();
                break;
            }
            self.selected_nodes.push(candidate.node);
        }
    }

    fn insert_graph_candidate(
        &mut self,
        candidate: SearchCandidate,
        limit: usize,
    ) -> io::Result<()> {
        if limit == 0 {
            return Ok(());
        }
        let replacing_worst = self.retained_candidates.len() == limit;
        if replacing_worst {
            let Some(worst) = self.retained_candidates.peek().copied() else {
                return Ok(());
            };
            if candidate >= worst {
                return Ok(());
            }
        } else {
            self.retained_candidates
                .try_reserve(1)
                .map_err(|_| invalid_input("DiskANN graph candidate allocation failed"))?;
        }
        self.frontier
            .try_reserve(1)
            .map_err(|_| invalid_input("DiskANN graph frontier allocation failed"))?;
        if replacing_worst {
            self.retained_candidates.pop();
        }
        self.retained_candidates.push(candidate);
        self.frontier.push(Reverse(candidate));
        if self.frontier.len() > limit.saturating_mul(2) {
            let worst = *self
                .retained_candidates
                .peek()
                .expect("non-empty retained DiskANN candidates");
            self.frontier
                .retain(|Reverse(candidate)| *candidate <= worst);
        }
        Ok(())
    }

    fn finish_graph_candidates(&mut self) {
        self.candidates.extend(self.retained_candidates.drain());
        sort_candidates(&mut self.candidates);
    }

    #[cfg(test)]
    fn is_visited(&self, node: usize) -> bool {
        if self.uses_sparse_visited {
            self.sparse_visited.get(node as u32).is_some()
        } else {
            self.visited[node]
        }
    }

    fn mark_visited(&mut self, node: usize) -> bool {
        if self.uses_sparse_visited {
            return self.sparse_visited.insert(node as u32, ()).is_none();
        }
        if self.visited[node] {
            return false;
        }
        self.visited[node] = true;
        self.touched_nodes.push(node);
        true
    }

    #[cfg(test)]
    fn visited_capacity(&self) -> usize {
        self.visited.capacity()
    }

    #[cfg(test)]
    fn uses_sparse_visited(&self) -> bool {
        self.uses_sparse_visited
    }

    #[cfg(test)]
    fn retained_window_capacity(&self) -> usize {
        self.window_buffers.retained_capacity
    }
}

fn window_buffer_limit_per_worker(worker_count: usize) -> usize {
    QUERY_WINDOW_BUFFER_LIMIT_BYTES.min(BATCH_WINDOW_BUFFER_LIMIT_BYTES / worker_count.max(1))
}

fn prepare_vector_window_cache(
    windows: &[ReadWindow],
    cache: &mut VectorWindowCache,
    window_buffers: &mut WindowBufferPool,
    capacity_limit: usize,
) -> (bool, usize) {
    let retain = windows
        .iter()
        .try_fold(0usize, |total, window| total.checked_add(window.length))
        .is_some_and(|total| total <= capacity_limit);
    if !retain {
        let evictions = cache.len();
        cache.recycle(window_buffers);
        return (false, evictions);
    }
    let evictions = cache.trim(window_buffers, capacity_limit);
    (true, evictions)
}

type CandidatePartition = Vec<(usize, Vec<usize>)>;
type SessionQueryOutput = (usize, Vec<i64>, Vec<f32>, DiskAnnSearchStats);

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialEq for ExactSearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.row_id == other.row_id && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for ExactSearchResult {}

impl PartialOrd for ExactSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExactSearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.row_id.cmp(&other.row_id))
    }
}

impl<R: SeekRead> crate::diskann_io::DiskAnnIndexReader<R> {
    fn preprocess_queries<'a>(&self, queries: &'a [f32], query_count: usize) -> Cow<'a, [f32]> {
        if self.header.metric_type() == MetricType::Cosine {
            Cow::Owned(preprocess_vectors(
                queries,
                query_count,
                self.header.dimension as usize,
                MetricType::Cosine,
            ))
        } else {
            Cow::Borrowed(queries)
        }
    }

    fn take_batch_workers(
        &mut self,
        worker_count: usize,
        filtered: bool,
    ) -> io::Result<Option<Vec<Self>>> {
        if filtered {
            self.ensure_resident()?;
        } else {
            self.optimize_for_search()?;
        }
        let mut workers = std::mem::take(&mut self.batch_workers);
        workers.truncate(worker_count);
        while workers.len() < worker_count {
            let worker = if filtered {
                self.try_clone_for_filtered_search()
            } else {
                self.try_clone_for_search()
            };
            match worker {
                Ok(Some(worker)) => workers.push(worker),
                Ok(None) => {
                    self.batch_workers = workers;
                    return Ok(None);
                }
                Err(error) => {
                    self.batch_workers = workers;
                    return Err(error);
                }
            }
        }
        let window_buffer_limit = window_buffer_limit_per_worker(worker_count);
        for worker in &mut workers {
            worker.refresh_shared_state_from(self);
            worker.limit_raw_vector_cache_bytes(window_buffer_limit);
            worker
                .query_scratch
                .set_window_buffer_limit(window_buffer_limit);
        }
        Ok(Some(workers))
    }

    pub(crate) fn search_batch(
        &mut self,
        queries: &[f32],
        top_k: usize,
        l_search: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        self.last_search_stats = DiskAnnSearchStats {
            query_count,
            ..DiskAnnSearchStats::default()
        };
        if top_k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let processed_queries = self.preprocess_queries(queries, query_count);
        let queries = processed_queries.as_ref();
        let worker_count = query_count.min(rayon::current_num_threads());
        if worker_count <= 1 {
            self.batch_workers.clear();
            if self.header.is_interleaved() {
                return self.search_batch_direct_serial(queries, top_k, l_search);
            }
            return self.search_batch_serial(queries, top_k, l_search);
        }

        let Some(mut workers) = self.take_batch_workers(worker_count, false)? else {
            if self.header.is_interleaved() {
                return self.search_batch_direct_serial(queries, top_k, l_search);
            }
            return self.search_batch_serial(queries, top_k, l_search);
        };
        if self.header.is_interleaved()
            || query_count <= worker_count.saturating_mul(PARALLEL_SESSION_MAX_QUERIES_PER_WORKER)
        {
            let result =
                self.search_batch_in_parallel_sessions(queries, top_k, l_search, &mut workers);
            self.batch_workers = workers;
            return result;
        }
        let result = (|| {
            let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
            let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
            for query_chunk in queries.chunks(BATCH_QUERY_CHUNK_SIZE * dimension) {
                let chunk_query_count = query_chunk.len() / dimension;
                self.record_query_chunk(chunk_query_count);
                let worker_outputs = workers
                    .par_iter_mut()
                    .enumerate()
                    .map(|(worker_index, worker)| {
                        worker.last_search_stats = DiskAnnSearchStats::default();
                        let mut partition = Vec::new();
                        for query_index in (worker_index..chunk_query_count).step_by(worker_count) {
                            let query = &query_chunk
                                [query_index * dimension..(query_index + 1) * dimension];
                            let candidates = worker
                                .generate_unfiltered_candidate_nodes(query, top_k, l_search)?;
                            partition.push((query_index, candidates));
                        }
                        Ok::<_, io::Error>((partition, worker.last_search_stats))
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let mut partitions = Vec::with_capacity(worker_outputs.len());
                for (partition, worker_stats) in worker_outputs {
                    self.last_search_stats
                        .merge_candidate_generation(worker_stats);
                    partitions.push(partition);
                }
                let (chunk_ids, chunk_distances) =
                    self.rerank_candidate_batch_streaming(query_chunk, top_k, partitions)?;
                ids.extend(chunk_ids);
                distances.extend(chunk_distances);
            }
            Ok((ids, distances))
        })();
        self.batch_workers = workers;
        result
    }

    /// Warms resident metadata and the query-dependent adjacency/raw-vector
    /// caches with representative queries without changing reported search
    /// statistics.
    pub fn warmup_queries(&mut self, queries: &[f32], l_search: usize) -> io::Result<()> {
        let dimension = self.header.dimension as usize;
        if !queries.len().is_multiple_of(dimension) {
            return Err(invalid_input(format!(
                "warmup query length {} is not divisible by dimension {}",
                queries.len(),
                dimension
            )));
        }
        if queries.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input("warmup query values must be finite"));
        }
        self.optimize_for_search()?;
        if queries.is_empty() {
            return Ok(());
        }
        let saved_stats = self.last_search_stats;
        // Replay on the parent Reader so both adjacency and raw-vector windows
        // are useful to the subsequent single-query path. A batch warm-up may
        // otherwise populate only retained worker-local raw-vector caches.
        let result = queries
            .chunks_exact(dimension)
            .try_for_each(|query| self.search(query, 1, l_search).map(|_| ()));
        self.last_search_stats = saved_stats;
        result
    }

    /// Calibrates the automatic search width from representative queries.
    ///
    /// This is a stability proxy, not a ground-truth recall guarantee: it
    /// chooses the first width whose Top-K overlap with the next wider search
    /// reaches 98% across the sample.
    pub fn calibrate_l_search(&mut self, queries: &[f32], top_k: usize) -> io::Result<usize> {
        let dimension = self.header.dimension as usize;
        if queries.is_empty() || !queries.len().is_multiple_of(dimension) {
            return Err(invalid_input(
                "calibration queries must contain one or more complete vectors",
            ));
        }
        if top_k == 0 {
            return Err(invalid_input("calibration top_k must be greater than 0"));
        }
        if queries.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input("calibration query values must be finite"));
        }
        self.optimize_for_search()?;
        let widths = [
            100usize.max(top_k),
            200usize.max(top_k),
            400usize.max(top_k),
        ];
        let mut results = Vec::with_capacity(widths.len());
        let saved_stats = self.last_search_stats;
        for width in widths {
            results.push(self.search_batch(queries, top_k, width)?);
        }
        self.last_search_stats = saved_stats;
        let chosen = if topk_result_stability(
            &results[0].0,
            &results[0].1,
            &results[1].0,
            &results[1].1,
            top_k,
        ) >= 0.98
        {
            widths[0]
        } else if topk_result_stability(
            &results[1].0,
            &results[1].1,
            &results[2].0,
            &results[2].1,
            top_k,
        ) >= 0.98
        {
            widths[1]
        } else {
            widths[2]
        };
        self.calibrated_l_search = Some(chosen);
        for worker in &mut self.batch_workers {
            worker.calibrated_l_search = Some(chosen);
        }
        Ok(chosen)
    }

    pub(crate) fn search_batch_with_roaring_filter(
        &mut self,
        queries: &[f32],
        top_k: usize,
        l_search: usize,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        self.last_search_stats = DiskAnnSearchStats {
            query_count,
            ..DiskAnnSearchStats::default()
        };
        if top_k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let processed_queries = self.preprocess_queries(queries, query_count);
        let queries = processed_queries.as_ref();
        if filter.is_empty() {
            return Ok((
                vec![-1; query_count * top_k],
                vec![f32::MAX; query_count * top_k],
            ));
        }
        let matching_nodes = self.matching_nodes_for_filter(&filter)?;
        if matching_nodes.is_empty() {
            return Ok((
                vec![-1; query_count * top_k],
                vec![f32::MAX; query_count * top_k],
            ));
        }
        if self.header.is_interleaved() {
            let worker_count = query_count.min(rayon::current_num_threads());
            if worker_count <= 1 {
                self.batch_workers.clear();
                return self.search_batch_with_matching_nodes_direct_serial(
                    queries,
                    top_k,
                    l_search,
                    &matching_nodes,
                );
            }
            let Some(mut workers) = self.take_batch_workers(worker_count, true)? else {
                return self.search_batch_with_matching_nodes_direct_serial(
                    queries,
                    top_k,
                    l_search,
                    &matching_nodes,
                );
            };
            let result = self.search_batch_with_matching_nodes_in_parallel_sessions(
                queries,
                top_k,
                l_search,
                &matching_nodes,
                &mut workers,
            );
            self.batch_workers = workers;
            return result;
        }
        let matching_count = usize::try_from(matching_nodes.len()).unwrap_or(usize::MAX);
        if let FilteredCandidateStrategy::Exhaustive { target_candidates } =
            select_filtered_candidate_strategy(
                self.header.vector_count as usize,
                matching_count,
                top_k,
                l_search,
                self.header.max_degree as usize,
                self.read_plan(),
                self.adjacency_fully_preloaded(),
            )
        {
            return self.search_batch_filtered_exhaustive(
                queries,
                top_k,
                &matching_nodes,
                target_candidates,
            );
        }
        let worker_count = query_count.min(rayon::current_num_threads());
        if worker_count <= 1 {
            self.batch_workers.clear();
            return self.search_batch_with_matching_nodes_serial(
                queries,
                top_k,
                l_search,
                &matching_nodes,
            );
        }

        let Some(mut workers) = self.take_batch_workers(worker_count, true)? else {
            return self.search_batch_with_matching_nodes_serial(
                queries,
                top_k,
                l_search,
                &matching_nodes,
            );
        };
        if query_count <= worker_count.saturating_mul(PARALLEL_SESSION_MAX_QUERIES_PER_WORKER) {
            let result = self.search_batch_with_matching_nodes_in_parallel_sessions(
                queries,
                top_k,
                l_search,
                &matching_nodes,
                &mut workers,
            );
            self.batch_workers = workers;
            return result;
        }
        let result = (|| {
            let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
            let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
            for query_chunk in queries.chunks(BATCH_QUERY_CHUNK_SIZE * dimension) {
                let chunk_query_count = query_chunk.len() / dimension;
                self.record_query_chunk(chunk_query_count);
                let worker_outputs = workers
                    .par_iter_mut()
                    .enumerate()
                    .map(|(worker_index, worker)| {
                        worker.last_search_stats = DiskAnnSearchStats::default();
                        let mut partition = Vec::new();
                        for query_index in (worker_index..chunk_query_count).step_by(worker_count) {
                            let query = &query_chunk
                                [query_index * dimension..(query_index + 1) * dimension];
                            let candidates = worker.generate_filtered_candidates(
                                query,
                                top_k,
                                l_search,
                                &matching_nodes,
                            )?;
                            partition.push((
                                query_index,
                                candidates
                                    .into_iter()
                                    .map(|candidate| candidate.node)
                                    .collect(),
                            ));
                        }
                        Ok::<_, io::Error>((partition, worker.last_search_stats))
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let mut partitions = Vec::with_capacity(worker_outputs.len());
                for (partition, worker_stats) in worker_outputs {
                    self.last_search_stats
                        .merge_candidate_generation(worker_stats);
                    partitions.push(partition);
                }
                let (chunk_ids, chunk_distances) =
                    self.rerank_candidate_batch_streaming(query_chunk, top_k, partitions)?;
                ids.extend(chunk_ids);
                distances.extend(chunk_distances);
            }
            Ok((ids, distances))
        })();
        self.batch_workers = workers;
        result
    }

    fn search_batch_in_parallel_sessions(
        &mut self,
        queries: &[f32],
        top_k: usize,
        l_search: usize,
        workers: &mut [Self],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        self.record_query_chunk(query_count);
        let worker_count = workers.len();
        let worker_outputs = workers
            .par_iter_mut()
            .enumerate()
            .map(|(worker_index, worker)| {
                let mut outputs = Vec::new();
                for query_index in (worker_index..query_count).step_by(worker_count) {
                    let query = &queries[query_index * dimension..(query_index + 1) * dimension];
                    let (ids, distances) = worker.search_preprocessed(query, top_k, l_search)?;
                    outputs.push((query_index, ids, distances, worker.last_search_stats));
                }
                Ok::<_, io::Error>(outputs)
            })
            .collect::<io::Result<Vec<_>>>()?;
        self.collect_parallel_session_outputs(worker_outputs, query_count, top_k)
    }

    fn search_batch_with_matching_nodes_in_parallel_sessions(
        &mut self,
        queries: &[f32],
        top_k: usize,
        l_search: usize,
        matching_nodes: &RoaringBitmap,
        workers: &mut [Self],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        self.record_query_chunk(query_count);
        let worker_count = workers.len();
        let worker_outputs = workers
            .par_iter_mut()
            .enumerate()
            .map(|(worker_index, worker)| {
                let mut outputs = Vec::new();
                for query_index in (worker_index..query_count).step_by(worker_count) {
                    let query = &queries[query_index * dimension..(query_index + 1) * dimension];
                    worker.last_search_stats = DiskAnnSearchStats {
                        query_count: 1,
                        ..DiskAnnSearchStats::default()
                    };
                    let (ids, distances) = worker.search_with_matching_nodes(
                        query,
                        top_k,
                        l_search,
                        matching_nodes,
                    )?;
                    outputs.push((query_index, ids, distances, worker.last_search_stats));
                }
                Ok::<_, io::Error>(outputs)
            })
            .collect::<io::Result<Vec<_>>>()?;
        self.collect_parallel_session_outputs(worker_outputs, query_count, top_k)
    }

    fn collect_parallel_session_outputs(
        &mut self,
        worker_outputs: Vec<Vec<SessionQueryOutput>>,
        query_count: usize,
        top_k: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let mut ordered = (0..query_count)
            .map(|_| None)
            .collect::<Vec<Option<(Vec<i64>, Vec<f32>)>>>();
        for (query_index, ids, distances, stats) in worker_outputs.into_iter().flatten() {
            let slot = ordered
                .get_mut(query_index)
                .ok_or_else(|| invalid_data("DiskANN parallel session query index is invalid"))?;
            if slot.replace((ids, distances)).is_some() {
                return Err(invalid_data(
                    "DiskANN parallel sessions returned a duplicate query",
                ));
            }
            self.last_search_stats.merge_complete_query(stats);
        }
        if ordered.iter().any(Option::is_none) {
            return Err(invalid_data(
                "DiskANN parallel sessions did not return every query",
            ));
        }
        let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
        let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
        for result in ordered {
            let (query_ids, query_distances) =
                result.expect("validated DiskANN parallel session output");
            ids.extend(query_ids);
            distances.extend(query_distances);
        }
        Ok((ids, distances))
    }

    fn search_batch_filtered_exhaustive(
        &mut self,
        queries: &[f32],
        top_k: usize,
        matching_nodes: &RoaringBitmap,
        candidate_limit: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        let matching_count = usize::try_from(matching_nodes.len()).unwrap_or(usize::MAX);
        let pq_m = self.header.pq_m as usize;
        let pq_ksub = 1usize << self.header.pq_bits;
        let tile_size = filtered_pq_query_tile_size(pq_m, pq_ksub);
        let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
        let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
        for query_chunk in queries.chunks(BATCH_QUERY_CHUNK_SIZE * dimension) {
            let chunk_query_count = query_chunk.len() / dimension;
            self.record_query_chunk(chunk_query_count);
            let candidate_sets = self.exhaustive_filtered_candidate_nodes_batch(
                query_chunk,
                matching_nodes,
                candidate_limit,
            )?;
            self.last_search_stats.filtered_exhaustive_queries = self
                .last_search_stats
                .filtered_exhaustive_queries
                .saturating_add(chunk_query_count);
            self.last_search_stats.pq_distance_evaluations = self
                .last_search_stats
                .pq_distance_evaluations
                .saturating_add(chunk_query_count.saturating_mul(matching_count));
            self.last_search_stats.pq_code_loads =
                self.last_search_stats.pq_code_loads.saturating_add(
                    chunk_query_count
                        .div_ceil(tile_size)
                        .saturating_mul(matching_count),
                );
            let partition = candidate_sets.into_iter().enumerate().collect();
            let (chunk_ids, chunk_distances) =
                self.rerank_candidate_batch_streaming(query_chunk, top_k, vec![partition])?;
            ids.extend(chunk_ids);
            distances.extend(chunk_distances);
        }
        Ok((ids, distances))
    }

    fn exhaustive_filtered_candidate_nodes_batch(
        &self,
        queries: &[f32],
        matching_nodes: &RoaringBitmap,
        candidate_limit: usize,
    ) -> io::Result<Vec<Vec<usize>>> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        let pq_m = self.header.pq_m as usize;
        let pq = self.pq()?;
        let pq_ksub = pq.ksub;
        let pq_code_size = pq.code_size();
        let pq_codes = self.pq_codes()?;
        let metric = self.header.metric_type();
        let tile_size = filtered_pq_query_tile_size(pq_m, pq_ksub);
        let tile_count = query_count.div_ceil(tile_size);
        let mut tiles = (0..tile_count)
            .into_par_iter()
            .map(|tile_index| {
                let query_start = tile_index * tile_size;
                let query_end = (query_start + tile_size).min(query_count);
                let tile_query_count = query_end - query_start;
                let mut distance_tables = vec![0.0f32; tile_query_count * pq_m * pq_ksub];
                for tile_query_index in 0..tile_query_count {
                    let query_index = query_start + tile_query_index;
                    let query = &queries[query_index * dimension..(query_index + 1) * dimension];
                    let table_start = tile_query_index * pq_m * pq_ksub;
                    pq.compute_distance_table(
                        query,
                        metric,
                        &mut distance_tables[table_start..table_start + pq_m * pq_ksub],
                    );
                }
                let mut heaps = (0..tile_query_count)
                    .map(|_| BinaryHeap::new())
                    .collect::<Vec<BinaryHeap<SearchCandidate>>>();
                for heap in &mut heaps {
                    heap.try_reserve(candidate_limit.min(1024)).map_err(|_| {
                        invalid_input("DiskANN filtered candidate allocation failed")
                    })?;
                }
                for node in matching_nodes.iter() {
                    let node = node as usize;
                    let code_start = node
                        .checked_mul(pq_code_size)
                        .ok_or_else(|| invalid_data("DiskANN PQ code offset overflows"))?;
                    let codes = pq_codes
                        .get(code_start..code_start + pq_code_size)
                        .ok_or_else(|| invalid_data("DiskANN PQ codes are truncated"))?;
                    for tile_query_index in 0..tile_query_count {
                        let table_start = tile_query_index * pq_m * pq_ksub;
                        push_bounded_candidate(
                            &mut heaps[tile_query_index],
                            SearchCandidate {
                                node,
                                distance: pq.distance_from_table(
                                    &distance_tables[table_start..table_start + pq_m * pq_ksub],
                                    codes,
                                ),
                            },
                            candidate_limit,
                        )?;
                    }
                }
                let candidates = heaps
                    .into_iter()
                    .map(|heap| {
                        let mut candidates = heap.into_vec();
                        sort_candidates(&mut candidates);
                        candidates
                            .into_iter()
                            .map(|candidate| candidate.node)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Ok::<_, io::Error>((query_start, candidates))
            })
            .collect::<io::Result<Vec<_>>>()?;
        tiles.sort_unstable_by_key(|(query_start, _)| *query_start);
        Ok(tiles
            .into_iter()
            .flat_map(|(_, candidates)| candidates)
            .collect())
    }

    fn search_batch_direct_serial(
        &mut self,
        queries: &[f32],
        top_k: usize,
        l_search: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        let mut aggregate = DiskAnnSearchStats {
            query_count,
            query_chunks: usize::from(query_count != 0),
            max_queries_per_chunk: query_count,
            ..DiskAnnSearchStats::default()
        };
        let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
        let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
        for query in queries.chunks_exact(dimension) {
            let (query_ids, query_distances) = self.search_preprocessed(query, top_k, l_search)?;
            aggregate.merge_complete_query(self.last_search_stats);
            ids.extend(query_ids);
            distances.extend(query_distances);
        }
        aggregate.parallel_session_queries = 0;
        self.last_search_stats = aggregate;
        Ok((ids, distances))
    }

    fn search_batch_with_matching_nodes_direct_serial(
        &mut self,
        queries: &[f32],
        top_k: usize,
        l_search: usize,
        matching_nodes: &RoaringBitmap,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        let mut aggregate = DiskAnnSearchStats {
            query_count,
            query_chunks: usize::from(query_count != 0),
            max_queries_per_chunk: query_count,
            ..DiskAnnSearchStats::default()
        };
        let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
        let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
        for query in queries.chunks_exact(dimension) {
            self.last_search_stats = DiskAnnSearchStats {
                query_count: 1,
                ..DiskAnnSearchStats::default()
            };
            let (query_ids, query_distances) =
                self.search_with_matching_nodes(query, top_k, l_search, matching_nodes)?;
            aggregate.merge_complete_query(self.last_search_stats);
            ids.extend(query_ids);
            distances.extend(query_distances);
        }
        aggregate.parallel_session_queries = 0;
        self.last_search_stats = aggregate;
        Ok((ids, distances))
    }

    fn search_batch_serial(
        &mut self,
        queries: &[f32],
        top_k: usize,
        l_search: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
        let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
        for query_chunk in queries.chunks(BATCH_QUERY_CHUNK_SIZE * dimension) {
            let chunk_query_count = query_chunk.len() / dimension;
            self.record_query_chunk(chunk_query_count);
            let mut partition = Vec::with_capacity(chunk_query_count);
            for (query_index, query) in query_chunk.chunks_exact(dimension).enumerate() {
                let candidates =
                    self.generate_unfiltered_candidate_nodes(query, top_k, l_search)?;
                partition.push((query_index, candidates));
            }
            let (chunk_ids, chunk_distances) =
                self.rerank_candidate_batch_streaming(query_chunk, top_k, vec![partition])?;
            ids.extend(chunk_ids);
            distances.extend(chunk_distances);
        }
        Ok((ids, distances))
    }

    fn search_batch_with_matching_nodes_serial(
        &mut self,
        queries: &[f32],
        top_k: usize,
        l_search: usize,
        matching_nodes: &RoaringBitmap,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
        let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
        for query_chunk in queries.chunks(BATCH_QUERY_CHUNK_SIZE * dimension) {
            let chunk_query_count = query_chunk.len() / dimension;
            self.record_query_chunk(chunk_query_count);
            let mut partition = Vec::with_capacity(chunk_query_count);
            for (query_index, query) in query_chunk.chunks_exact(dimension).enumerate() {
                let candidates =
                    self.generate_filtered_candidates(query, top_k, l_search, matching_nodes)?;
                partition.push((
                    query_index,
                    candidates
                        .into_iter()
                        .map(|candidate| candidate.node)
                        .collect(),
                ));
            }
            let (chunk_ids, chunk_distances) =
                self.rerank_candidate_batch_streaming(query_chunk, top_k, vec![partition])?;
            ids.extend(chunk_ids);
            distances.extend(chunk_distances);
        }
        Ok((ids, distances))
    }

    fn record_query_chunk(&mut self, query_count: usize) {
        self.last_search_stats.query_chunks = self.last_search_stats.query_chunks.saturating_add(1);
        self.last_search_stats.max_queries_per_chunk = self
            .last_search_stats
            .max_queries_per_chunk
            .max(query_count);
    }

    fn generate_unfiltered_candidate_nodes(
        &mut self,
        query: &[f32],
        top_k: usize,
        l_search: usize,
    ) -> io::Result<Vec<usize>> {
        self.ensure_resident()?;
        let search_list_size = resolve_diskann_l_search(top_k, l_search);
        let mut scratch = std::mem::take(&mut self.query_scratch);
        let result = (|| {
            self.generate_graph_candidates(
                query,
                search_list_size,
                self.read_plan().graph_beam_width,
                &mut scratch,
            )?;
            let rerank_count = search_list_size
                .min(top_k.saturating_mul(4).max(64))
                .min(scratch.candidates.len());
            if rerank_count == scratch.candidates.len() {
                return Ok(scratch
                    .candidates
                    .iter()
                    .map(|candidate| candidate.node)
                    .collect());
            }
            if self.header.is_interleaved() {
                let planner =
                    ReadWindowPlanner::new(self.read_plan(), self.header.sections.adjacency);
                expand_rerank_candidates_within_seed_windows(
                    &scratch.candidates,
                    rerank_count,
                    |node| {
                        let page = self.adjacency_locator(node)?.page_index as usize;
                        planner
                            .window_for_logical_page(page)
                            .map(|window| window.offset)
                            .ok_or_else(|| invalid_data("DiskANN adjacency page is out of range"))
                    },
                    &mut scratch.rerank_windows,
                    &mut scratch.rerank_candidates,
                )?;
                return Ok(scratch
                    .rerank_candidates
                    .iter()
                    .map(|candidate| candidate.node)
                    .collect());
            }
            let planner = VectorWindowPlanner::new(
                self.read_plan(),
                self.header.sections.vectors,
                self.header.vector_record_size as usize,
            )?;
            expand_rerank_candidates_within_seed_windows(
                &scratch.candidates,
                rerank_count,
                |node| {
                    planner
                        .window_for_node(node)
                        .map(|window| window.offset)
                        .ok_or_else(|| invalid_data("DiskANN raw-vector record is out of range"))
                },
                &mut scratch.rerank_windows,
                &mut scratch.rerank_candidates,
            )?;
            Ok(scratch
                .rerank_candidates
                .iter()
                .map(|candidate| candidate.node)
                .collect())
        })();
        scratch.recycle_adjacency_windows();
        self.query_scratch = scratch;
        result
    }

    pub fn search(
        &mut self,
        query: &[f32],
        top_k: usize,
        l_search: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.last_search_stats = DiskAnnSearchStats {
            query_count: 1,
            ..DiskAnnSearchStats::default()
        };
        let dimension = self.header.dimension as usize;
        if query.len() != dimension {
            return Err(invalid_input(format!(
                "query dimension mismatch: expected {}, got {}",
                dimension,
                query.len()
            )));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input("query values must be finite"));
        }
        let processed_query = self.preprocess_queries(query, 1);
        self.search_preprocessed(processed_query.as_ref(), top_k, l_search)
    }

    fn search_preprocessed(
        &mut self,
        query: &[f32],
        top_k: usize,
        l_search: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.last_search_stats = DiskAnnSearchStats {
            query_count: 1,
            ..DiskAnnSearchStats::default()
        };
        if top_k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        self.ensure_resident()?;
        let search_list_size = resolve_diskann_l_search(top_k, l_search);
        let mut scratch = std::mem::take(&mut self.query_scratch);
        let result = (|| {
            self.generate_graph_candidates(
                query,
                search_list_size,
                self.read_plan().graph_beam_width,
                &mut scratch,
            )?;

            let rerank_count = search_list_size
                .min(top_k.saturating_mul(4).max(64))
                .min(scratch.candidates.len());
            if rerank_count == scratch.candidates.len() {
                if self.header.is_interleaved() {
                    return self.rerank_interleaved(
                        query,
                        &scratch.candidates,
                        top_k,
                        &mut scratch.adjacency_windows,
                        &mut scratch.loaded_adjacency_pages,
                        &mut scratch.window_buffers,
                    );
                }
                return self.rerank(
                    query,
                    &scratch.candidates,
                    top_k,
                    &mut scratch.vector_windows,
                    &mut scratch.window_buffers,
                );
            }
            if self.header.is_interleaved() {
                let planner =
                    ReadWindowPlanner::new(self.read_plan(), self.header.sections.adjacency);
                expand_rerank_candidates_within_seed_windows(
                    &scratch.candidates,
                    rerank_count,
                    |node| {
                        let page = self.adjacency_locator(node)?.page_index as usize;
                        planner
                            .window_for_logical_page(page)
                            .map(|window| window.offset)
                            .ok_or_else(|| invalid_data("DiskANN adjacency page is out of range"))
                    },
                    &mut scratch.rerank_windows,
                    &mut scratch.rerank_candidates,
                )?;
                return self.rerank_interleaved(
                    query,
                    &scratch.rerank_candidates,
                    top_k,
                    &mut scratch.adjacency_windows,
                    &mut scratch.loaded_adjacency_pages,
                    &mut scratch.window_buffers,
                );
            }
            let planner = VectorWindowPlanner::new(
                self.read_plan(),
                self.header.sections.vectors,
                self.header.vector_record_size as usize,
            )?;
            expand_rerank_candidates_within_seed_windows(
                &scratch.candidates,
                rerank_count,
                |node| {
                    planner
                        .window_for_node(node)
                        .map(|window| window.offset)
                        .ok_or_else(|| invalid_data("DiskANN raw-vector record is out of range"))
                },
                &mut scratch.rerank_windows,
                &mut scratch.rerank_candidates,
            )?;
            self.rerank(
                query,
                &scratch.rerank_candidates,
                top_k,
                &mut scratch.vector_windows,
                &mut scratch.window_buffers,
            )
        })();
        scratch.recycle_adjacency_windows();
        self.query_scratch = scratch;
        result
    }

    fn generate_graph_candidates(
        &mut self,
        query: &[f32],
        search_list_size: usize,
        beam_width: usize,
        scratch: &mut DiskAnnQueryScratch,
    ) -> io::Result<()> {
        let vector_count = self.header.vector_count as usize;
        let search_list_size = search_list_size.min(vector_count);
        scratch.begin_graph_search(
            vector_count,
            search_list_size,
            self.header.max_degree as usize,
        )?;
        let pq = self.pq()?;
        let distance_table_len = pq.m * pq.ksub;
        pq.compute_distance_table(
            query,
            self.header.metric_type(),
            scratch.prepare_distance_table(distance_table_len),
        );

        let entry_node = self.header.entry_node as usize;
        scratch.mark_visited(entry_node);
        scratch.insert_graph_candidate(
            SearchCandidate {
                node: entry_node,
                distance: self.pq_distance(entry_node, &scratch.distance_table)?,
            },
            search_list_size,
        )?;
        let mut expanded_count = 0usize;
        while expanded_count < search_list_size {
            scratch.select_round(beam_width.min(search_list_size - expanded_count));
            if scratch.selected_nodes.is_empty() {
                break;
            }
            self.load_adjacency_pages(
                &scratch.selected_nodes,
                &mut scratch.adjacency_windows,
                &mut scratch.loaded_adjacency_pages,
                &mut scratch.window_buffers,
            )?;
            expanded_count += scratch.selected_nodes.len();
            for selected_node_index in 0..scratch.selected_nodes.len() {
                let node = scratch.selected_nodes[selected_node_index];
                self.decode_adjacency_neighbors(
                    node,
                    &scratch.adjacency_windows,
                    &mut scratch.neighbor_buffer,
                )?;
                let mut retained_neighbors = 0;
                for neighbor_index in 0..scratch.neighbor_buffer.len() {
                    let neighbor = scratch.neighbor_buffer[neighbor_index] as usize;
                    if !scratch.mark_visited(neighbor) {
                        continue;
                    }
                    scratch.neighbor_buffer[retained_neighbors] = neighbor as u32;
                    retained_neighbors += 1;
                }
                scratch.neighbor_buffer.truncate(retained_neighbors);
                score_pq_neighbors(
                    &scratch.distance_table,
                    self.pq_codes()?,
                    self.header.pq_m as usize,
                    self.header.pq_bits as usize,
                    &scratch.neighbor_buffer,
                    &mut scratch.scored_neighbors,
                )?;
                for scored_index in 0..scratch.scored_neighbors.len() {
                    let candidate = scratch.scored_neighbors[scored_index];
                    scratch.insert_graph_candidate(candidate, search_list_size)?;
                }
            }
            let evictions = scratch.adjacency_windows.trim(
                &mut scratch.window_buffers,
                QUERY_ADJACENCY_WINDOW_LIMIT_BYTES,
            );
            self.last_search_stats.query_adjacency_cache_evictions = self
                .last_search_stats
                .query_adjacency_cache_evictions
                .saturating_add(evictions);
        }
        scratch.finish_graph_candidates();
        Ok(())
    }

    pub fn search_with_roaring_filter(
        &mut self,
        query: &[f32],
        top_k: usize,
        l_search: usize,
        roaring_filter_bytes: &[u8],
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let dimension = self.header.dimension as usize;
        if query.len() != dimension {
            return Err(invalid_input(format!(
                "query dimension mismatch: expected {}, got {}",
                dimension,
                query.len()
            )));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input("query values must be finite"));
        }
        let processed_query = self.preprocess_queries(query, 1);
        let query = processed_query.as_ref();
        let filter = decode_roaring_filter(roaring_filter_bytes)?;
        self.search_with_decoded_roaring_filter(query, top_k, l_search, &filter)
    }

    fn search_with_decoded_roaring_filter(
        &mut self,
        query: &[f32],
        top_k: usize,
        l_search: usize,
        filter: &RoaringTreemap,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        self.last_search_stats = DiskAnnSearchStats {
            query_count: 1,
            ..DiskAnnSearchStats::default()
        };
        if top_k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if filter.is_empty() {
            return Ok((vec![-1; top_k], vec![f32::MAX; top_k]));
        }
        let matching_nodes = self.matching_nodes_for_filter(filter)?;
        if matching_nodes.is_empty() {
            return Ok((vec![-1; top_k], vec![f32::MAX; top_k]));
        }
        self.search_with_matching_nodes(query, top_k, l_search, &matching_nodes)
    }

    fn exhaustive_filtered_candidates(
        &mut self,
        query: &[f32],
        matching_nodes: &RoaringBitmap,
        candidate_limit: usize,
    ) -> io::Result<Vec<SearchCandidate>> {
        let matching_count = usize::try_from(matching_nodes.len()).unwrap_or(usize::MAX);
        self.last_search_stats.filtered_exhaustive_queries = self
            .last_search_stats
            .filtered_exhaustive_queries
            .saturating_add(1);
        self.last_search_stats.pq_distance_evaluations = self
            .last_search_stats
            .pq_distance_evaluations
            .saturating_add(matching_count);
        self.last_search_stats.pq_code_loads = self
            .last_search_stats
            .pq_code_loads
            .saturating_add(matching_count);
        let mut scratch = std::mem::take(&mut self.query_scratch);
        let result = (|| {
            scratch.begin_rerank();
            let pq = self.pq()?;
            let distance_table_len = pq.m * pq.ksub;
            pq.compute_distance_table(
                query,
                self.header.metric_type(),
                scratch.prepare_distance_table(distance_table_len),
            );
            let pq_codes = self.pq_codes()?;
            let pq_m = self.header.pq_m as usize;
            let pq_bits = self.header.pq_bits as usize;
            let mut candidates = BinaryHeap::new();
            candidates
                .try_reserve(candidate_limit.min(1024))
                .map_err(|_| invalid_input("DiskANN filtered candidate allocation failed"))?;
            for node in matching_nodes.iter() {
                scratch.neighbor_buffer.push(node);
                if scratch.neighbor_buffer.len() == FILTERED_SINGLE_PQ_NODE_CHUNK_SIZE {
                    score_filtered_candidate_chunk(
                        &scratch.distance_table,
                        pq_codes,
                        pq_m,
                        pq_bits,
                        &scratch.neighbor_buffer,
                        &mut scratch.scored_neighbors,
                        &mut candidates,
                        candidate_limit,
                    )?;
                    scratch.neighbor_buffer.clear();
                }
            }
            if !scratch.neighbor_buffer.is_empty() {
                score_filtered_candidate_chunk(
                    &scratch.distance_table,
                    pq_codes,
                    pq_m,
                    pq_bits,
                    &scratch.neighbor_buffer,
                    &mut scratch.scored_neighbors,
                    &mut candidates,
                    candidate_limit,
                )?;
                scratch.neighbor_buffer.clear();
            }
            let mut candidates = candidates.into_vec();
            sort_candidates(&mut candidates);
            Ok(candidates)
        })();
        self.query_scratch = scratch;
        result
    }

    fn matching_nodes_for_filter(&mut self, filter: &RoaringTreemap) -> io::Result<RoaringBitmap> {
        self.ensure_resident()?;
        if use_row_id_order(filter.len(), self.header.vector_count as usize) {
            if let Some(row_id_order) = self.ensure_row_id_order()? {
                let matching_ranges =
                    matching_ranges_from_row_id_order(&row_id_order, filter, |node| {
                        self.row_id(node)
                    })?;
                let mut matching = RoaringBitmap::new();
                for range in matching_ranges {
                    for &node in &row_id_order[range] {
                        matching.insert(node);
                    }
                }
                return Ok(matching);
            }
        }
        matching_nodes_from_sequential_row_ids(filter, |visitor| self.try_for_each_row_id(visitor))
    }

    fn search_with_matching_nodes(
        &mut self,
        query: &[f32],
        top_k: usize,
        l_search: usize,
        matching_nodes: &RoaringBitmap,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        if top_k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let candidates =
            self.generate_filtered_candidates(query, top_k, l_search, matching_nodes)?;
        self.rerank_with_query_scratch(query, &candidates, top_k)
    }

    fn generate_filtered_candidates(
        &mut self,
        query: &[f32],
        top_k: usize,
        l_search: usize,
        matching_nodes: &RoaringBitmap,
    ) -> io::Result<Vec<SearchCandidate>> {
        let matching_count = usize::try_from(matching_nodes.len()).unwrap_or(usize::MAX);
        let strategy = select_filtered_candidate_strategy(
            self.header.vector_count as usize,
            matching_count,
            top_k,
            l_search,
            self.header.max_degree as usize,
            self.read_plan(),
            self.adjacency_fully_preloaded(),
        );
        match strategy {
            FilteredCandidateStrategy::Exhaustive { target_candidates } => {
                self.exhaustive_filtered_candidates(query, matching_nodes, target_candidates)
            }
            FilteredCandidateStrategy::Graph {
                target_candidates,
                search_list_size,
            } => {
                self.last_search_stats.filtered_graph_queries = self
                    .last_search_stats
                    .filtered_graph_queries
                    .saturating_add(1);
                let mut scratch = std::mem::take(&mut self.query_scratch);
                let graph_candidates = (|| {
                    self.generate_graph_candidates(
                        query,
                        search_list_size,
                        self.options()
                            .read_tier
                            .read_plan()
                            .filtered_graph_beam_width,
                        &mut scratch,
                    )?;
                    Ok::<_, io::Error>(post_filter_graph_candidates(
                        &scratch.candidates,
                        matching_nodes,
                        target_candidates,
                    ))
                })();
                scratch.recycle_adjacency_windows();
                self.query_scratch = scratch;
                if let Some(candidates) = graph_candidates? {
                    Ok(candidates)
                } else {
                    self.last_search_stats.filtered_graph_fallbacks = self
                        .last_search_stats
                        .filtered_graph_fallbacks
                        .saturating_add(1);
                    self.exhaustive_filtered_candidates(query, matching_nodes, target_candidates)
                }
            }
        }
    }

    fn pq_distance(&self, node: usize, distance_table: &[f32]) -> io::Result<f32> {
        let pq = self.pq()?;
        let code_size = pq.code_size();
        let start = node
            .checked_mul(code_size)
            .ok_or_else(|| invalid_data("DiskANN PQ code offset overflows"))?;
        let end = start + code_size;
        let codes = self
            .pq_codes()?
            .get(start..end)
            .ok_or_else(|| invalid_data("DiskANN PQ codes are truncated"))?;
        Ok(pq.distance_from_table(distance_table, codes))
    }

    fn rerank(
        &mut self,
        query: &[f32],
        candidates: &[SearchCandidate],
        top_k: usize,
        window_cache: &mut VectorWindowCache,
        window_buffers: &mut WindowBufferPool,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let record_size = self.header.vector_record_size as usize;
        let encoding = self.header.raw_vector_encoding();
        let planner =
            VectorWindowPlanner::new(self.read_plan(), self.header.sections.vectors, record_size)?;
        let windows = planner.plan_nodes(candidates.iter().map(|candidate| candidate.node));
        self.last_search_stats.rerank_candidate_references = self
            .last_search_stats
            .rerank_candidate_references
            .saturating_add(candidates.len());
        self.last_search_stats.rerank_unique_windows = self
            .last_search_stats
            .rerank_unique_windows
            .saturating_add(windows.len());
        self.last_search_stats.rerank_chunks = self
            .last_search_stats
            .rerank_chunks
            .saturating_add(usize::from(!windows.is_empty()));
        let raw_vector_cache_bytes = self.options().raw_vector_cache_bytes;
        let (retain_vector_windows, preparation_evictions) = prepare_vector_window_cache(
            &windows,
            window_cache,
            window_buffers,
            raw_vector_cache_bytes,
        );
        let distance_kernel = selected_raw_vector_distance_kernel(query.len());
        let metric = self.header.metric_type();
        self.last_search_stats.raw_vector_cache_evictions = self
            .last_search_stats
            .raw_vector_cache_evictions
            .saturating_add(preparation_evictions);
        let cache_load = self.load_vector_windows(&windows, window_cache, window_buffers)?;
        self.last_search_stats.raw_vector_cache_hits = self
            .last_search_stats
            .raw_vector_cache_hits
            .saturating_add(cache_load.hits);
        self.last_search_stats.raw_vector_cache_misses = self
            .last_search_stats
            .raw_vector_cache_misses
            .saturating_add(cache_load.misses);
        self.last_search_stats.raw_vector_cache_evictions = self
            .last_search_stats
            .raw_vector_cache_evictions
            .saturating_add(cache_load.evictions);
        let result = (|| {
            let mut exact = BinaryHeap::new();
            exact
                .try_reserve(top_k.min(candidates.len()))
                .map_err(|_| invalid_input("DiskANN exact result allocation failed"))?;
            for candidate in candidates {
                let window = planner
                    .window_for_node(candidate.node)
                    .ok_or_else(|| invalid_data("DiskANN raw-vector record is out of range"))?;
                let payload = window_cache
                    .get(&window.offset)
                    .ok_or_else(|| invalid_data("DiskANN vector window is not loaded"))?;
                let record = planner.record(window, payload, candidate.node)?;
                let distance =
                    raw_vector_distance(query, record, encoding, metric, distance_kernel)?;
                let row_id = self.row_id(candidate.node)?;
                push_bounded_exact_result(
                    &mut exact,
                    ExactSearchResult { row_id, distance },
                    top_k,
                )?;
            }
            let exact = exact.into_sorted_vec();
            let mut ids = exact.iter().map(|result| result.row_id).collect::<Vec<_>>();
            let mut distances = exact
                .iter()
                .map(|result| result.distance)
                .collect::<Vec<_>>();
            ids.resize(top_k, -1);
            distances.resize(top_k, f32::MAX);
            Ok((ids, distances))
        })();
        if retain_vector_windows && result.is_ok() {
            window_cache.touch_windows(&windows);
            let evictions = window_cache.trim(window_buffers, raw_vector_cache_bytes);
            self.last_search_stats.raw_vector_cache_evictions = self
                .last_search_stats
                .raw_vector_cache_evictions
                .saturating_add(evictions);
        } else {
            window_cache.recycle(window_buffers);
        }
        result
    }

    fn rerank_interleaved(
        &mut self,
        query: &[f32],
        candidates: &[SearchCandidate],
        top_k: usize,
        adjacency_windows: &mut AdjacencyWindowCache,
        loaded_pages: &mut HashSet<usize>,
        window_buffers: &mut WindowBufferPool,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let nodes = candidates
            .iter()
            .map(|candidate| candidate.node)
            .collect::<Vec<_>>();
        self.load_adjacency_pages(&nodes, adjacency_windows, loaded_pages, window_buffers)?;
        let planner = ReadWindowPlanner::new(self.read_plan(), self.header.sections.adjacency);
        let windows = nodes
            .iter()
            .map(|&node| {
                let page = self.adjacency_locator(node)?.page_index as usize;
                planner
                    .window_for_logical_page(page)
                    .map(|window| window.offset)
                    .ok_or_else(|| invalid_data("DiskANN adjacency page is out of range"))
            })
            .collect::<io::Result<BTreeSet<_>>>()?;
        self.last_search_stats.rerank_candidate_references = self
            .last_search_stats
            .rerank_candidate_references
            .saturating_add(candidates.len());
        self.last_search_stats.rerank_unique_windows = self
            .last_search_stats
            .rerank_unique_windows
            .saturating_add(windows.len());
        self.last_search_stats.rerank_chunks = self
            .last_search_stats
            .rerank_chunks
            .saturating_add(usize::from(!windows.is_empty()));
        let record_size = self.header.vector_record_size as usize;
        let encoding = self.header.raw_vector_encoding();
        let distance_kernel = selected_raw_vector_distance_kernel(query.len());
        let metric = self.header.metric_type();
        let mut exact = BinaryHeap::new();
        exact
            .try_reserve(top_k.min(candidates.len()))
            .map_err(|_| invalid_input("DiskANN exact result allocation failed"))?;
        for candidate in candidates {
            let locator = self.adjacency_locator(candidate.node)?;
            let page_index = locator.page_index as usize;
            let window = planner
                .window_for_logical_page(page_index)
                .ok_or_else(|| invalid_data("DiskANN adjacency page is out of range"))?;
            let payload = if let Some(hot) = self.hot_adjacency_window(window.offset, window.length)
            {
                hot
            } else {
                adjacency_windows
                    .get(&window.offset)
                    .map(WindowPayload::as_slice)
                    .ok_or_else(|| invalid_data("DiskANN adjacency rerank window is not loaded"))?
            };
            let page_offset = self.header.sections.adjacency.offset
                + page_index as u64 * DISKANN_PAGE_SIZE as u64
                - window.offset;
            let record_offset = (page_offset as usize)
                .checked_add(locator.byte_offset as usize)
                .and_then(|offset| offset.checked_sub(record_size))
                .ok_or_else(|| invalid_data("DiskANN interleaved vector offset underflows"))?;
            let record = payload
                .get(record_offset..record_offset + record_size)
                .ok_or_else(|| invalid_data("DiskANN interleaved raw vector is truncated"))?;
            let distance = raw_vector_distance(query, record, encoding, metric, distance_kernel)?;
            push_bounded_exact_result(
                &mut exact,
                ExactSearchResult {
                    row_id: self.row_id(candidate.node)?,
                    distance,
                },
                top_k,
            )?;
        }
        let exact = exact.into_sorted_vec();
        let mut ids = exact.iter().map(|result| result.row_id).collect::<Vec<_>>();
        let mut distances = exact
            .iter()
            .map(|result| result.distance)
            .collect::<Vec<_>>();
        ids.resize(top_k, -1);
        distances.resize(top_k, f32::MAX);
        Ok((ids, distances))
    }

    fn rerank_with_query_scratch(
        &mut self,
        query: &[f32],
        candidates: &[SearchCandidate],
        top_k: usize,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        let mut scratch = std::mem::take(&mut self.query_scratch);
        let result = {
            scratch.begin_rerank();
            if self.header.is_interleaved() {
                self.rerank_interleaved(
                    query,
                    candidates,
                    top_k,
                    &mut scratch.adjacency_windows,
                    &mut scratch.loaded_adjacency_pages,
                    &mut scratch.window_buffers,
                )
            } else {
                self.rerank(
                    query,
                    candidates,
                    top_k,
                    &mut scratch.vector_windows,
                    &mut scratch.window_buffers,
                )
            }
        };
        scratch.recycle_adjacency_windows();
        self.query_scratch = scratch;
        result
    }

    fn rerank_candidate_batch_streaming(
        &mut self,
        queries: &[f32],
        top_k: usize,
        partitions: Vec<CandidatePartition>,
    ) -> io::Result<(Vec<i64>, Vec<f32>)> {
        if self.header.is_interleaved() {
            return Err(invalid_data(
                "DiskANN interleaved rerank must run inside a search session",
            ));
        }
        let dimension = self.header.dimension as usize;
        let query_count = queries.len() / dimension;
        let distance_kernel = selected_raw_vector_distance_kernel(dimension);
        let metric = self.header.metric_type();
        let mut candidate_sets = (0..query_count).map(|_| None).collect::<Vec<_>>();
        for (query_index, candidates) in partitions.into_iter().flatten() {
            let slot = candidate_sets
                .get_mut(query_index)
                .ok_or_else(|| invalid_data("DiskANN filtered batch query index is invalid"))?;
            if slot.replace(candidates).is_some() {
                return Err(invalid_data(
                    "DiskANN filtered batch contains duplicate query results",
                ));
            }
        }
        if candidate_sets.iter().any(Option::is_none) {
            return Err(invalid_data(
                "DiskANN filtered batch is missing query candidates",
            ));
        }

        let record_size = self.header.vector_record_size as usize;
        let encoding = self.header.raw_vector_encoding();
        let planner =
            VectorWindowPlanner::new(self.read_plan(), self.header.sections.vectors, record_size)?;
        let mut grouped = HashMap::<u64, (ReadWindow, Vec<(usize, usize)>)>::new();
        for (query_index, candidates) in candidate_sets.into_iter().enumerate() {
            for node in candidates.expect("validated DiskANN batch candidates") {
                let window = planner
                    .window_for_node(node)
                    .ok_or_else(|| invalid_data("DiskANN raw-vector record is out of range"))?;
                grouped
                    .entry(window.offset)
                    .or_insert_with(|| (window, Vec::new()))
                    .1
                    .push((query_index, node));
            }
        }
        let mut window_groups = grouped.into_values().collect::<Vec<_>>();
        window_groups.sort_unstable_by_key(|(window, _)| window.offset);
        let windows = window_groups
            .iter()
            .map(|(window, _)| *window)
            .collect::<Vec<_>>();
        let chunks = plan_streaming_window_chunks(&windows);
        self.last_search_stats.rerank_candidate_references = self
            .last_search_stats
            .rerank_candidate_references
            .saturating_add(
                window_groups
                    .iter()
                    .map(|(_, references)| references.len())
                    .sum::<usize>(),
            );
        self.last_search_stats.rerank_unique_windows = self
            .last_search_stats
            .rerank_unique_windows
            .saturating_add(windows.len());
        self.last_search_stats.rerank_chunks = self
            .last_search_stats
            .rerank_chunks
            .saturating_add(chunks.len());
        let mut exact_heaps = (0..query_count)
            .map(|_| BinaryHeap::new())
            .collect::<Vec<BinaryHeap<ExactSearchResult>>>();
        let raw_vector_cache_bytes = self.options().raw_vector_cache_bytes;
        let mut scratch = std::mem::take(&mut self.query_scratch);
        scratch.begin_rerank();
        let result = (|| {
            for chunk in chunks {
                let chunk_windows = &windows[chunk.clone()];
                let cache_load = self.load_vector_windows(
                    chunk_windows,
                    &mut scratch.vector_windows,
                    &mut scratch.window_buffers,
                )?;
                self.last_search_stats.raw_vector_cache_hits = self
                    .last_search_stats
                    .raw_vector_cache_hits
                    .saturating_add(cache_load.hits);
                self.last_search_stats.raw_vector_cache_misses = self
                    .last_search_stats
                    .raw_vector_cache_misses
                    .saturating_add(cache_load.misses);
                self.last_search_stats.raw_vector_cache_evictions = self
                    .last_search_stats
                    .raw_vector_cache_evictions
                    .saturating_add(cache_load.evictions);
                let chunk_reference_count = window_groups[chunk.clone()]
                    .iter()
                    .map(|(_, references)| references.len())
                    .sum::<usize>();
                let parallel = rayon::current_num_threads() > 1
                    && chunk_reference_count.saturating_mul(dimension)
                        >= PARALLEL_EXACT_RERANK_MIN_COMPONENTS;
                if parallel {
                    self.last_search_stats.parallel_exact_rerank_chunks = self
                        .last_search_stats
                        .parallel_exact_rerank_chunks
                        .saturating_add(1);
                    self.last_search_stats.parallel_exact_rerank_references = self
                        .last_search_stats
                        .parallel_exact_rerank_references
                        .saturating_add(chunk_reference_count);
                    let mut rerank_references = Vec::with_capacity(chunk_reference_count);
                    for (window, references) in &window_groups[chunk.clone()] {
                        let payload = scratch
                            .vector_windows
                            .get(&window.offset)
                            .ok_or_else(|| invalid_data("DiskANN vector window is not loaded"))?;
                        for &(query_index, node) in references {
                            rerank_references.push(ExactRerankReference {
                                query_index,
                                row_id: self.row_id(node)?,
                                record: planner.record(*window, payload, node)?,
                            });
                        }
                    }
                    let exact_results = rerank_references
                        .par_iter()
                        .map(|reference| {
                            let query = &queries[reference.query_index * dimension
                                ..(reference.query_index + 1) * dimension];
                            Ok::<_, io::Error>((
                                reference.query_index,
                                ExactSearchResult {
                                    row_id: reference.row_id,
                                    distance: raw_vector_distance(
                                        query,
                                        reference.record,
                                        encoding,
                                        metric,
                                        distance_kernel,
                                    )?,
                                },
                            ))
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    for (query_index, exact) in exact_results {
                        push_bounded_exact_result(&mut exact_heaps[query_index], exact, top_k)?;
                    }
                } else {
                    for (window, references) in &window_groups[chunk.clone()] {
                        let payload = scratch
                            .vector_windows
                            .get(&window.offset)
                            .ok_or_else(|| invalid_data("DiskANN vector window is not loaded"))?;
                        for &(query_index, node) in references {
                            let query =
                                &queries[query_index * dimension..(query_index + 1) * dimension];
                            push_bounded_exact_result(
                                &mut exact_heaps[query_index],
                                ExactSearchResult {
                                    row_id: self.row_id(node)?,
                                    distance: raw_vector_distance(
                                        query,
                                        planner.record(*window, payload, node)?,
                                        encoding,
                                        metric,
                                        distance_kernel,
                                    )?,
                                },
                                top_k,
                            )?;
                        }
                    }
                }
                scratch.vector_windows.touch_windows(chunk_windows);
                let evictions = scratch
                    .vector_windows
                    .trim(&mut scratch.window_buffers, raw_vector_cache_bytes);
                self.last_search_stats.raw_vector_cache_evictions = self
                    .last_search_stats
                    .raw_vector_cache_evictions
                    .saturating_add(evictions);
            }

            let mut ids = Vec::with_capacity(query_count.saturating_mul(top_k));
            let mut distances = Vec::with_capacity(query_count.saturating_mul(top_k));
            for heap in exact_heaps {
                let exact = heap.into_sorted_vec();
                ids.extend(exact.iter().map(|result| result.row_id));
                distances.extend(exact.iter().map(|result| result.distance));
                ids.resize(ids.len() + top_k - exact.len(), -1);
                distances.resize(distances.len() + top_k - exact.len(), f32::MAX);
            }
            Ok((ids, distances))
        })();
        if result.is_err() {
            scratch.recycle_window_caches();
        } else {
            scratch.recycle_adjacency_windows();
        }
        self.query_scratch = scratch;
        result
    }

    fn load_adjacency_pages(
        &mut self,
        nodes: &[usize],
        window_cache: &mut AdjacencyWindowCache,
        loaded_pages: &mut HashSet<usize>,
        window_buffers: &mut WindowBufferPool,
    ) -> io::Result<()> {
        let planner = ReadWindowPlanner::new(self.read_plan(), self.header.sections.adjacency);
        let mut pages = BTreeSet::new();
        let mut required_windows = Vec::with_capacity(nodes.len());
        for &node in nodes {
            let page = self.adjacency_locator(node)?.page_index as usize;
            let window = planner
                .window_for_logical_page(page)
                .ok_or_else(|| invalid_data("DiskANN adjacency page is out of range"))?;
            let window_is_hot = self
                .hot_adjacency_window(window.offset, window.length)
                .is_some();
            if !window_is_hot {
                required_windows.push(window);
            }
            let window_is_available = window_is_hot || window_cache.contains_key(&window.offset);
            if !loaded_pages.contains(&page) || !window_is_available {
                pages.insert(page);
            }
        }
        if pages.is_empty() {
            return Ok(());
        }
        let windows = planner.plan_logical_pages(pages.iter().copied());
        let cold_windows = windows
            .iter()
            .copied()
            .filter(|window| {
                self.hot_adjacency_window(window.offset, window.length)
                    .is_none()
            })
            .collect::<Vec<_>>();
        let incoming_bytes = cold_windows
            .iter()
            .filter(|window| !window_cache.contains_key(&window.offset))
            .fold(0usize, |total, window| total.saturating_add(window.length));
        let preparation_evictions = prepare_adjacency_window_cache(
            &required_windows,
            incoming_bytes,
            window_cache,
            window_buffers,
        );
        self.last_search_stats.query_adjacency_cache_evictions = self
            .last_search_stats
            .query_adjacency_cache_evictions
            .saturating_add(preparation_evictions);
        self.load_adjacency_windows(&cold_windows, window_cache, window_buffers)?;
        self.last_search_stats.query_adjacency_cache_peak_bytes = self
            .last_search_stats
            .query_adjacency_cache_peak_bytes
            .max(window_cache.retained_capacity());
        for page_index in pages {
            let window = planner
                .window_for_logical_page(page_index)
                .ok_or_else(|| invalid_data("DiskANN adjacency page is out of range"))?;
            let payload = if let Some(hot) = self.hot_adjacency_window(window.offset, window.length)
            {
                hot
            } else {
                window_cache
                    .get(&window.offset)
                    .map(WindowPayload::as_slice)
                    .ok_or_else(|| {
                        invalid_data("DiskANN adjacency validation window is not loaded")
                    })?
            };
            let page_offset = self.header.sections.adjacency.offset
                + page_index as u64 * DISKANN_PAGE_SIZE as u64
                - window.offset;
            let page_start = page_offset as usize;
            let page_end = page_start + DISKANN_PAGE_SIZE as usize;
            self.validate_adjacency_page(
                page_index,
                payload
                    .get(page_start..page_end)
                    .ok_or_else(|| invalid_data("DiskANN adjacency page is truncated"))?,
            )?;
            loaded_pages.insert(page_index);
        }
        Ok(())
    }

    fn decode_adjacency_neighbors(
        &self,
        node: usize,
        window_cache: &AdjacencyWindowCache,
        neighbors: &mut Vec<u32>,
    ) -> io::Result<()> {
        let locator = self.adjacency_locator(node)?;
        let page_index = locator.page_index as usize;
        let planner = ReadWindowPlanner::new(self.read_plan(), self.header.sections.adjacency);
        let window = planner
            .window_for_logical_page(page_index)
            .ok_or_else(|| invalid_data("DiskANN adjacency page is out of range"))?;
        let payload = if let Some(hot) = self.hot_adjacency_window(window.offset, window.length) {
            hot
        } else {
            window_cache
                .get(&window.offset)
                .map(WindowPayload::as_slice)
                .ok_or_else(|| invalid_data("DiskANN adjacency decode window is not loaded"))?
        };
        let page_offset = self.header.sections.adjacency.offset
            + page_index as u64 * DISKANN_PAGE_SIZE as u64
            - window.offset;
        let start = page_offset as usize + locator.byte_offset as usize;
        let bytes = payload
            .get(start..)
            .ok_or_else(|| invalid_data("DiskANN adjacency list is truncated"))?;
        decode_adjacency_list(bytes, locator.degree(), locator.encoding(), neighbors)?;
        Ok(())
    }

    fn load_adjacency_windows(
        &mut self,
        windows: &[ReadWindow],
        local_cache: &mut AdjacencyWindowCache,
        window_buffers: &mut WindowBufferPool,
    ) -> io::Result<()> {
        let mut pending = windows
            .iter()
            .copied()
            .filter(|window| !local_cache.contains_key(&window.offset))
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return Ok(());
        }
        if self.options().adjacency_cache_bytes == 0 {
            self.last_search_stats.adjacency_cache_misses = self
                .last_search_stats
                .adjacency_cache_misses
                .saturating_add(pending.len());
            let mut payloads = Vec::with_capacity(pending.len());
            for window in &pending {
                payloads.push(window_buffers.take(window.length)?);
            }
            let read_result = {
                let mut requests = pending
                    .iter()
                    .zip(payloads.iter_mut())
                    .map(|(window, payload)| ReadRequest::new(window.offset, payload))
                    .collect::<Vec<_>>();
                self.pread_ranges(&mut requests)
            };
            if let Err(error) = read_result {
                for payload in payloads {
                    window_buffers.recycle(payload);
                }
                return Err(error);
            }
            for (window, payload) in pending.into_iter().zip(payloads) {
                local_cache.insert(window.offset, WindowPayload::Owned(payload));
            }
            return Ok(());
        }

        while !pending.is_empty() {
            let mut reserved = Vec::new();
            let mut waiting = Vec::new();
            for window in &pending {
                let (lookup, lock_metrics) = self
                    .adjacency_cache()?
                    .lookup_or_reserve(window.offset, window.length)?;
                self.last_search_stats
                    .record_adjacency_cache_lock(lock_metrics);
                match lookup {
                    SharedWindowCacheLookup::Hit(payload) => {
                        local_cache.insert(window.offset, WindowPayload::Shared(payload));
                        self.last_search_stats.adjacency_cache_hits = self
                            .last_search_stats
                            .adjacency_cache_hits
                            .saturating_add(1);
                    }
                    SharedWindowCacheLookup::Reserved => {
                        reserved.push(*window);
                        self.last_search_stats.adjacency_cache_misses = self
                            .last_search_stats
                            .adjacency_cache_misses
                            .saturating_add(1);
                    }
                    SharedWindowCacheLookup::Loading => {
                        waiting.push(*window);
                        self.last_search_stats.adjacency_cache_waits = self
                            .last_search_stats
                            .adjacency_cache_waits
                            .saturating_add(1);
                    }
                }
            }

            if !reserved.is_empty() {
                let mut payloads = Vec::with_capacity(reserved.len());
                for window in &reserved {
                    match window_buffers.take(window.length) {
                        Ok(payload) => payloads.push(payload),
                        Err(error) => {
                            let lock_metrics = self.adjacency_cache()?.cancel(
                                &reserved
                                    .iter()
                                    .map(|window| window.offset)
                                    .collect::<Vec<_>>(),
                            )?;
                            self.last_search_stats
                                .record_adjacency_cache_lock(lock_metrics);
                            for payload in payloads {
                                window_buffers.recycle(payload);
                            }
                            return Err(error);
                        }
                    }
                }
                let read_result = {
                    let mut requests = reserved
                        .iter()
                        .zip(payloads.iter_mut())
                        .map(|(window, payload)| ReadRequest::new(window.offset, payload))
                        .collect::<Vec<_>>();
                    self.pread_ranges(&mut requests)
                };
                if let Err(error) = read_result {
                    let lock_metrics = self.adjacency_cache()?.cancel(
                        &reserved
                            .iter()
                            .map(|window| window.offset)
                            .collect::<Vec<_>>(),
                    )?;
                    self.last_search_stats
                        .record_adjacency_cache_lock(lock_metrics);
                    for payload in payloads {
                        window_buffers.recycle(payload);
                    }
                    return Err(error);
                }
                for (window, payload) in reserved.into_iter().zip(payloads) {
                    let payload = share_window_payload(payload);
                    local_cache.insert(window.offset, WindowPayload::Shared(Arc::clone(&payload)));
                    let (evictions, lock_metrics) =
                        self.adjacency_cache()?.publish(window.offset, payload)?;
                    self.last_search_stats
                        .record_adjacency_cache_lock(lock_metrics);
                    self.last_search_stats.adjacency_cache_evictions = self
                        .last_search_stats
                        .adjacency_cache_evictions
                        .saturating_add(evictions);
                }
            }

            for window in waiting {
                let (payload, lock_metrics) = self
                    .adjacency_cache()?
                    .wait_for(window.offset, window.length)?;
                self.last_search_stats
                    .record_adjacency_cache_lock(lock_metrics);
                if let Some(payload) = payload {
                    local_cache.insert(window.offset, WindowPayload::Shared(payload));
                    self.last_search_stats.adjacency_cache_hits = self
                        .last_search_stats
                        .adjacency_cache_hits
                        .saturating_add(1);
                }
            }
            pending.retain(|window| !local_cache.contains_key(&window.offset));
        }
        Ok(())
    }

    fn load_vector_windows(
        &mut self,
        windows: &[ReadWindow],
        cache: &mut VectorWindowCache,
        window_buffers: &mut WindowBufferPool,
    ) -> io::Result<VectorWindowLoadStats> {
        let mut pending = windows
            .iter()
            .copied()
            .filter(|window| !cache.contains_key(&window.offset))
            .collect::<Vec<_>>();
        let mut stats = VectorWindowLoadStats {
            hits: windows.len().saturating_sub(pending.len()),
            ..VectorWindowLoadStats::default()
        };
        if pending.is_empty() {
            return Ok(stats);
        }

        if self.options().raw_vector_cache_bytes == 0 {
            stats.misses = pending.len();
            let mut payloads = Vec::with_capacity(pending.len());
            for window in &pending {
                payloads.push(window_buffers.take(window.length)?);
            }
            let read_result = {
                let mut requests = pending
                    .iter()
                    .zip(payloads.iter_mut())
                    .map(|(window, payload)| ReadRequest::new(window.offset, payload))
                    .collect::<Vec<_>>();
                self.pread_ranges(&mut requests)
            };
            if let Err(error) = read_result {
                for payload in payloads {
                    window_buffers.recycle(payload);
                }
                return Err(error);
            }
            for (window, payload) in pending.into_iter().zip(payloads) {
                cache.insert(window.offset, payload);
            }
            return Ok(stats);
        }

        while !pending.is_empty() {
            let mut reserved = Vec::new();
            let mut waiting = Vec::new();
            for window in &pending {
                match self
                    .raw_vector_cache()?
                    .lookup_or_reserve(window.offset, window.length)?
                    .0
                {
                    SharedWindowCacheLookup::Hit(payload) => {
                        cache.insert(window.offset, WindowPayload::Shared(payload));
                        stats.hits = stats.hits.saturating_add(1);
                    }
                    SharedWindowCacheLookup::Reserved => {
                        reserved.push(*window);
                        stats.misses = stats.misses.saturating_add(1);
                    }
                    SharedWindowCacheLookup::Loading => waiting.push(*window),
                }
            }

            if !reserved.is_empty() {
                let mut payloads = Vec::with_capacity(reserved.len());
                for window in &reserved {
                    match window_buffers.take(window.length) {
                        Ok(payload) => payloads.push(payload),
                        Err(error) => {
                            self.raw_vector_cache()?.cancel(
                                &reserved
                                    .iter()
                                    .map(|window| window.offset)
                                    .collect::<Vec<_>>(),
                            )?;
                            for payload in payloads {
                                window_buffers.recycle(payload);
                            }
                            return Err(error);
                        }
                    }
                }
                let read_result = {
                    let mut requests = reserved
                        .iter()
                        .zip(payloads.iter_mut())
                        .map(|(window, payload)| ReadRequest::new(window.offset, payload))
                        .collect::<Vec<_>>();
                    self.pread_ranges(&mut requests)
                };
                if let Err(error) = read_result {
                    self.raw_vector_cache()?.cancel(
                        &reserved
                            .iter()
                            .map(|window| window.offset)
                            .collect::<Vec<_>>(),
                    )?;
                    for payload in payloads {
                        window_buffers.recycle(payload);
                    }
                    return Err(error);
                }
                for (window, payload) in reserved.into_iter().zip(payloads) {
                    let payload = share_window_payload(payload);
                    cache.insert(window.offset, WindowPayload::Shared(Arc::clone(&payload)));
                    stats.evictions = stats.evictions.saturating_add(
                        self.raw_vector_cache()?.publish(window.offset, payload)?.0,
                    );
                }
            }

            for window in waiting {
                if let Some(payload) = self
                    .raw_vector_cache()?
                    .wait_for(window.offset, window.length)?
                    .0
                {
                    cache.insert(window.offset, WindowPayload::Shared(payload));
                    stats.hits = stats.hits.saturating_add(1);
                }
            }
            pending.retain(|window| !cache.contains_key(&window.offset));
        }
        Ok(stats)
    }
}

fn sort_candidates(candidates: &mut [SearchCandidate]) {
    candidates.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then_with(|| left.node.cmp(&right.node))
    });
}

fn desired_filtered_candidate_count(matching_count: usize, top_k: usize) -> usize {
    matching_count.min(top_k.saturating_mul(4).max(64))
}

fn resolve_diskann_l_search(top_k: usize, l_search: usize) -> usize {
    let configured = if l_search == 0 {
        top_k.saturating_mul(2).max(100)
    } else {
        l_search
    };
    top_k.max(configured)
}

fn topk_result_stability(
    left_ids: &[i64],
    left_distances: &[f32],
    right_ids: &[i64],
    right_distances: &[f32],
    top_k: usize,
) -> f32 {
    if top_k == 0
        || left_ids.len() != right_ids.len()
        || left_ids.len() != left_distances.len()
        || right_ids.len() != right_distances.len()
        || !left_ids.len().is_multiple_of(top_k)
    {
        return 0.0;
    }
    let mut overlap = 0usize;
    let mut denominator = 0usize;
    for (((left_query_ids, left_query_distances), right_query_ids), right_query_distances) in
        left_ids
            .chunks_exact(top_k)
            .zip(left_distances.chunks_exact(top_k))
            .zip(right_ids.chunks_exact(top_k))
            .zip(right_distances.chunks_exact(top_k))
    {
        let mut right_counts = HashMap::<i64, usize>::with_capacity(top_k);
        for (&row_id, &distance) in right_query_ids.iter().zip(right_query_distances) {
            if distance != f32::MAX {
                *right_counts.entry(row_id).or_default() += 1;
            }
        }
        for (&row_id, &distance) in left_query_ids.iter().zip(left_query_distances) {
            if distance != f32::MAX {
                denominator += 1;
                if right_counts.get_mut(&row_id).is_some_and(|count| {
                    if *count == 0 {
                        false
                    } else {
                        *count -= 1;
                        true
                    }
                }) {
                    overlap += 1;
                }
            }
        }
    }
    if denominator == 0 {
        1.0
    } else {
        overlap as f32 / denominator as f32
    }
}

fn filtered_pq_query_tile_size(pq_m: usize, pq_ksub: usize) -> usize {
    let table_bytes_per_query = pq_m
        .saturating_mul(pq_ksub)
        .saturating_mul(size_of::<f32>())
        .max(1);
    (FILTERED_PQ_TILE_TABLE_LIMIT_BYTES / table_bytes_per_query)
        .clamp(1, FILTERED_PQ_MAX_QUERY_TILE_SIZE)
}

fn expand_rerank_candidates_within_seed_windows(
    candidates: &[SearchCandidate],
    seed_count: usize,
    mut window_for_node: impl FnMut(usize) -> io::Result<u64>,
    selected_windows: &mut HashSet<u64>,
    selected: &mut Vec<SearchCandidate>,
) -> io::Result<()> {
    selected.clear();
    selected_windows.clear();
    if candidates.is_empty() || seed_count == 0 {
        return Ok(());
    }
    if seed_count >= candidates.len() {
        selected
            .try_reserve(candidates.len())
            .map_err(|_| invalid_input("DiskANN rerank candidate allocation failed"))?;
        selected.extend_from_slice(candidates);
        return Ok(());
    }
    selected_windows
        .try_reserve(seed_count)
        .map_err(|_| invalid_input("DiskANN rerank window allocation failed"))?;
    for candidate in candidates.iter().take(seed_count) {
        selected_windows.insert(window_for_node(candidate.node)?);
    }
    selected
        .try_reserve(candidates.len().min(seed_count.saturating_mul(2)))
        .map_err(|_| invalid_input("DiskANN rerank candidate allocation failed"))?;
    for candidate in candidates {
        if selected_windows.contains(&window_for_node(candidate.node)?) {
            selected.push(*candidate);
        }
    }
    Ok(())
}

fn adaptive_filtered_search_list_size(
    vector_count: usize,
    matching_count: usize,
    target_candidates: usize,
    configured_l_search: usize,
) -> usize {
    if vector_count == 0 || matching_count == 0 || target_candidates == 0 {
        return 0;
    }
    let scaled_target = (target_candidates as u128)
        .saturating_mul(vector_count as u128)
        .div_ceil(matching_count as u128)
        .saturating_mul(2)
        .min(vector_count as u128) as usize;
    configured_l_search.max(scaled_target).min(vector_count)
}

#[allow(clippy::too_many_arguments)]
fn select_filtered_candidate_strategy(
    vector_count: usize,
    matching_count: usize,
    top_k: usize,
    l_search: usize,
    max_degree: usize,
    read_plan: ReadPlan,
    adjacency_fully_preloaded: bool,
) -> FilteredCandidateStrategy {
    let target_candidates = desired_filtered_candidate_count(matching_count, top_k);
    let exhaustive = FilteredCandidateStrategy::Exhaustive { target_candidates };
    let configured_l_search = resolve_diskann_l_search(top_k, l_search);
    if vector_count == 0
        || matching_count != vector_count
        || configured_l_search < 200
        || (read_plan.window_bytes > 16 * 1024 && !adjacency_fully_preloaded)
    {
        return exhaustive;
    }
    let search_list_size = adaptive_filtered_search_list_size(
        vector_count,
        matching_count,
        target_candidates,
        configured_l_search,
    );
    let graph_work = search_list_size
        .saturating_mul(max_degree.saturating_add(1))
        .min(vector_count);
    if graph_work > matching_count / 2 {
        return exhaustive;
    }
    FilteredCandidateStrategy::Graph {
        target_candidates,
        search_list_size,
    }
}

fn post_filter_graph_candidates(
    graph_candidates: &[SearchCandidate],
    matching_nodes: &RoaringBitmap,
    target_candidates: usize,
) -> Option<Vec<SearchCandidate>> {
    let mut filtered = Vec::with_capacity(target_candidates);
    for candidate in graph_candidates {
        let Ok(node) = u32::try_from(candidate.node) else {
            continue;
        };
        if matching_nodes.contains(node) {
            filtered.push(*candidate);
            if filtered.len() == target_candidates {
                break;
            }
        }
    }
    (filtered.len() == target_candidates).then_some(filtered)
}

fn plan_streaming_window_chunks(windows: &[ReadWindow]) -> Vec<std::ops::Range<usize>> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < windows.len() {
        let mut end = start;
        let mut payload_bytes = 0usize;
        while end < windows.len() && end - start < FILTERED_BATCH_RERANK_MAX_RANGES {
            let next_bytes = payload_bytes.saturating_add(windows[end].length);
            if end != start && next_bytes > FILTERED_BATCH_RERANK_MAX_BYTES {
                break;
            }
            payload_bytes = next_bytes;
            end += 1;
        }
        chunks.push(start..end);
        start = end;
    }
    chunks
}

fn use_row_id_order(filter_cardinality: u64, vector_count: usize) -> bool {
    filter_cardinality != 0 && filter_cardinality <= (vector_count / 16) as u64
}

fn matching_nodes_from_sequential_row_ids(
    filter: &RoaringTreemap,
    visit_row_ids: impl FnOnce(&mut dyn FnMut(usize, i64) -> io::Result<()>) -> io::Result<()>,
) -> io::Result<RoaringBitmap> {
    let mut matching = RoaringBitmap::new();
    let mut visitor = |node: usize, row_id: i64| {
        if row_id >= 0 && filter.contains(row_id as u64) {
            let node = u32::try_from(node)
                .map_err(|_| invalid_data("DiskANN internal node ID exceeds u32"))?;
            matching.insert(node);
        }
        Ok(())
    };
    visit_row_ids(&mut visitor)?;
    Ok(matching)
}

fn matching_ranges_from_row_id_order(
    order: &[u32],
    filter: &RoaringTreemap,
    mut row_id_at: impl FnMut(usize) -> io::Result<i64>,
) -> io::Result<Vec<std::ops::Range<usize>>> {
    let mut matches = Vec::new();
    for row_id in filter.iter() {
        let Ok(row_id) = i64::try_from(row_id) else {
            continue;
        };
        let start =
            row_id_order_partition_point(order, |node| Ok(row_id_at(node as usize)? < row_id))?;
        let end =
            row_id_order_partition_point(order, |node| Ok(row_id_at(node as usize)? <= row_id))?;
        if start != end {
            matches.push(start..end);
        }
    }
    Ok(matches)
}

fn row_id_order_partition_point(
    order: &[u32],
    mut predicate: impl FnMut(u32) -> io::Result<bool>,
) -> io::Result<usize> {
    let mut left = 0usize;
    let mut right = order.len();
    while left < right {
        let middle = left + (right - left) / 2;
        if predicate(order[middle])? {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    Ok(left)
}

fn push_bounded_candidate(
    candidates: &mut BinaryHeap<SearchCandidate>,
    candidate: SearchCandidate,
    limit: usize,
) -> io::Result<()> {
    if limit == 0 {
        return Ok(());
    }
    if candidates.len() < limit {
        candidates
            .try_reserve(1)
            .map_err(|_| invalid_input("DiskANN filtered candidate allocation failed"))?;
        candidates.push(candidate);
        return Ok(());
    }
    if candidates
        .peek()
        .is_some_and(|worst| candidate.cmp(worst) == Ordering::Less)
    {
        candidates.pop();
        candidates.push(candidate);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score_filtered_candidate_chunk(
    distance_table: &[f32],
    pq_codes: &[u8],
    pq_m: usize,
    pq_bits: usize,
    nodes: &[u32],
    scored: &mut Vec<SearchCandidate>,
    candidates: &mut BinaryHeap<SearchCandidate>,
    candidate_limit: usize,
) -> io::Result<()> {
    score_pq_neighbors(distance_table, pq_codes, pq_m, pq_bits, nodes, scored)?;
    for candidate in scored.iter().copied() {
        push_bounded_candidate(candidates, candidate, candidate_limit)?;
    }
    Ok(())
}

fn push_bounded_exact_result(
    results: &mut BinaryHeap<ExactSearchResult>,
    result: ExactSearchResult,
    limit: usize,
) -> io::Result<()> {
    if limit == 0 {
        return Ok(());
    }
    if results.len() < limit {
        results
            .try_reserve(1)
            .map_err(|_| invalid_input("DiskANN exact result allocation failed"))?;
        results.push(result);
    } else if results.peek().is_some_and(|worst| result < *worst) {
        results.pop();
        results.push(result);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawVectorDistanceKernel {
    Scalar,
    #[cfg(all(target_endian = "little", target_arch = "x86_64"))]
    Avx2,
    #[cfg(all(target_endian = "little", target_arch = "aarch64"))]
    Neon,
}

fn selected_raw_vector_distance_kernel(dimension: usize) -> RawVectorDistanceKernel {
    #[cfg(all(target_endian = "little", target_arch = "x86_64"))]
    if dimension >= 8 && is_x86_feature_detected!("avx2") {
        return RawVectorDistanceKernel::Avx2;
    }
    #[cfg(all(target_endian = "little", target_arch = "aarch64"))]
    if dimension >= 4 {
        return RawVectorDistanceKernel::Neon;
    }
    RawVectorDistanceKernel::Scalar
}

#[cfg(test)]
fn l2_distance_from_le_bytes(query: &[f32], bytes: &[u8]) -> io::Result<f32> {
    l2_distance_from_le_bytes_with_kernel(
        query,
        bytes,
        selected_raw_vector_distance_kernel(query.len()),
    )
}

fn raw_vector_distance(
    query: &[f32],
    bytes: &[u8],
    encoding: DiskAnnRawVectorEncoding,
    metric: MetricType,
    f32_kernel: RawVectorDistanceKernel,
) -> io::Result<f32> {
    match encoding {
        DiskAnnRawVectorEncoding::F32 if metric == MetricType::L2 => {
            l2_distance_from_le_bytes_with_kernel(query, bytes, f32_kernel)
        }
        DiskAnnRawVectorEncoding::F32 => metric_distance_from_f32_le_bytes(query, bytes, metric),
        DiskAnnRawVectorEncoding::F16 if metric == MetricType::L2 => {
            l2_distance_from_f16_le_bytes(query, bytes)
        }
        DiskAnnRawVectorEncoding::F16 => metric_distance_from_f16_le_bytes(query, bytes, metric),
    }
}

fn metric_distance_from_f32_le_bytes(
    query: &[f32],
    bytes: &[u8],
    metric: MetricType,
) -> io::Result<f32> {
    let expected_len = query
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid_data("DiskANN raw vector size overflows"))?;
    if bytes.len() != expected_len {
        return Err(invalid_data("DiskANN raw vector record has invalid length"));
    }
    let mut dot = 0.0f32;
    let mut query_norm = 0.0f32;
    let mut vector_norm = 0.0f32;
    for (&query_value, component) in query.iter().zip(bytes.chunks_exact(size_of::<f32>())) {
        let value = f32::from_le_bytes(
            component
                .try_into()
                .expect("validated four-byte raw vector component"),
        );
        if !value.is_finite() {
            return Err(invalid_data("DiskANN raw vectors must be finite"));
        }
        dot += query_value * value;
        if metric == MetricType::Cosine {
            query_norm += query_value * query_value;
            vector_norm += value * value;
        }
    }
    Ok(match metric {
        MetricType::InnerProduct => -dot,
        MetricType::Cosine if query_norm > 0.0 && vector_norm > 0.0 => {
            1.0 - dot / (query_norm * vector_norm).sqrt()
        }
        MetricType::Cosine => 1.0,
        MetricType::L2 => unreachable!("L2 uses the selected raw-vector kernel"),
    })
}

fn metric_distance_from_f16_le_bytes(
    query: &[f32],
    bytes: &[u8],
    metric: MetricType,
) -> io::Result<f32> {
    let expected_len = query
        .len()
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| invalid_data("DiskANN f16 raw-vector size overflows"))?;
    if bytes.len() != expected_len {
        return Err(invalid_data(
            "DiskANN f16 raw-vector record has invalid length",
        ));
    }
    if query.len() > 1024 {
        return Err(invalid_data(
            "DiskANN f16 raw-vector dimension exceeds the v1 limit",
        ));
    }
    let mut decoded = [0.0f32; 1024];
    for (slot, component) in decoded[..query.len()]
        .iter_mut()
        .zip(bytes.chunks_exact(size_of::<u16>()))
    {
        let value = half::f16::from_bits(u16::from_le_bytes(
            component
                .try_into()
                .expect("validated two-byte raw-vector component"),
        ));
        if !value.is_finite() {
            return Err(invalid_data("DiskANN raw vectors must be finite"));
        }
        *slot = value.to_f32();
    }
    Ok(fvec_distance(query, &decoded[..query.len()], metric))
}

fn l2_distance_from_f16_le_bytes(query: &[f32], bytes: &[u8]) -> io::Result<f32> {
    let expected_len = query
        .len()
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| invalid_data("DiskANN f16 raw-vector size overflows"))?;
    if bytes.len() != expected_len {
        return Err(invalid_data(
            "DiskANN f16 raw-vector record has invalid length",
        ));
    }
    if query.len() > 1024 {
        return Err(invalid_data(
            "DiskANN f16 raw-vector dimension exceeds the v1 limit",
        ));
    }
    #[cfg(all(target_endian = "little", target_arch = "aarch64"))]
    if query.len() >= 4 {
        // SAFETY: AArch64 guarantees NEON. The kernel uses unaligned loads,
        // validates every binary16 exponent, and stays inside the checked
        // query/record length.
        return unsafe { l2_distance_from_f16_le_bytes_neon(query, bytes) };
    }
    let mut bits = [0u16; 1024];
    for (slot, component) in bits[..query.len()]
        .iter_mut()
        .zip(bytes.chunks_exact(size_of::<u16>()))
    {
        *slot = u16::from_le_bytes(
            component
                .try_into()
                .expect("validated two-byte raw-vector component"),
        );
    }
    let values = bits[..query.len()].reinterpret_cast::<half::f16>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_data("DiskANN raw vectors must be finite"));
    }
    let mut decoded = [0.0f32; 1024];
    values.convert_to_f32_slice(&mut decoded[..query.len()]);
    Ok(fvec_l2sqr(query, &decoded[..query.len()]))
}

#[cfg(all(target_endian = "little", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn l2_distance_from_f16_le_bytes_neon(query: &[f32], bytes: &[u8]) -> io::Result<f32> {
    use std::arch::aarch64::*;

    let exponent_mask = vdup_n_u16(0x7c00);
    let mut invalid = vdup_n_u16(0);
    let mut sum = vdupq_n_f32(0.0);
    let mut index = 0usize;
    while index + 4 <= query.len() {
        let bits = unsafe { vld1_u16(bytes.as_ptr().add(index * size_of::<u16>()).cast::<u16>()) };
        invalid = vorr_u16(
            invalid,
            vceq_u16(vand_u16(bits, exponent_mask), exponent_mask),
        );
        let values = vcvt_f32_f16(vreinterpret_f16_u16(bits));
        let query_values = unsafe { vld1q_f32(query.as_ptr().add(index)) };
        let delta = vsubq_f32(query_values, values);
        sum = vmlaq_f32(sum, delta, delta);
        index += 4;
    }
    if vmaxv_u16(invalid) != 0 {
        return Err(invalid_data("DiskANN raw vectors must be finite"));
    }

    let mut distance = vaddvq_f32(sum);
    while index < query.len() {
        let start = index * size_of::<u16>();
        let value = half::f16::from_bits(u16::from_le_bytes(
            bytes[start..start + size_of::<u16>()]
                .try_into()
                .expect("validated two-byte raw-vector component"),
        ));
        if !value.is_finite() {
            return Err(invalid_data("DiskANN raw vectors must be finite"));
        }
        let delta = query[index] - value.to_f32();
        distance += delta * delta;
        index += 1;
    }
    Ok(distance)
}

fn l2_distance_from_le_bytes_with_kernel(
    query: &[f32],
    bytes: &[u8],
    kernel: RawVectorDistanceKernel,
) -> io::Result<f32> {
    let expected_len = query
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid_data("DiskANN raw vector size overflows"))?;
    if bytes.len() != expected_len {
        return Err(invalid_data("DiskANN raw vector record has invalid length"));
    }
    match kernel {
        RawVectorDistanceKernel::Scalar => l2_distance_from_le_bytes_scalar(query, bytes),
        #[cfg(all(target_endian = "little", target_arch = "x86_64"))]
        RawVectorDistanceKernel::Avx2 => unsafe { l2_distance_from_le_bytes_avx2(query, bytes) },
        #[cfg(all(target_endian = "little", target_arch = "aarch64"))]
        RawVectorDistanceKernel::Neon => unsafe { l2_distance_from_le_bytes_neon(query, bytes) },
    }
}

fn l2_distance_from_le_bytes_scalar(query: &[f32], bytes: &[u8]) -> io::Result<f32> {
    let mut distance = 0.0f32;
    for (&query_value, component) in query.iter().zip(bytes.chunks_exact(4)) {
        let value = f32::from_le_bytes(
            component
                .try_into()
                .expect("validated four-byte raw vector component"),
        );
        if !value.is_finite() {
            return Err(invalid_data("DiskANN raw vectors must be finite"));
        }
        let delta = query_value - value;
        distance += delta * delta;
    }
    Ok(distance)
}

#[cfg(all(target_endian = "little", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_from_le_bytes_avx2(query: &[f32], bytes: &[u8]) -> io::Result<f32> {
    use std::arch::x86_64::*;

    let exponent_mask = _mm256_set1_epi32(0x7f80_0000u32 as i32);
    let mut invalid = _mm256_setzero_si256();
    let mut sum = _mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= query.len() {
        let bits = unsafe {
            _mm256_loadu_si256(
                bytes
                    .as_ptr()
                    .add(index * size_of::<f32>())
                    .cast::<__m256i>(),
            )
        };
        let values = _mm256_castsi256_ps(bits);
        let query_values = unsafe { _mm256_loadu_ps(query.as_ptr().add(index)) };
        invalid = _mm256_or_si256(
            invalid,
            _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent_mask), exponent_mask),
        );
        let delta = _mm256_sub_ps(query_values, values);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(delta, delta));
        index += 8;
    }
    if _mm256_movemask_ps(_mm256_castsi256_ps(invalid)) != 0 {
        return Err(invalid_data("DiskANN raw vectors must be finite"));
    }

    let hi = _mm256_extractf128_ps::<1>(sum);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
    let mut distance = _mm_cvtss_f32(sum32);
    while index < query.len() {
        let start = index * size_of::<f32>();
        let value = f32::from_le_bytes(
            bytes[start..start + size_of::<f32>()]
                .try_into()
                .expect("validated four-byte raw vector component"),
        );
        if !value.is_finite() {
            return Err(invalid_data("DiskANN raw vectors must be finite"));
        }
        let delta = query[index] - value;
        distance += delta * delta;
        index += 1;
    }
    Ok(distance)
}

#[cfg(all(target_endian = "little", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn l2_distance_from_le_bytes_neon(query: &[f32], bytes: &[u8]) -> io::Result<f32> {
    use std::arch::aarch64::*;

    let exponent_mask = vdupq_n_u32(0x7f80_0000);
    let mut invalid = vdupq_n_u32(0);
    let mut sum = vdupq_n_f32(0.0);
    let mut index = 0usize;
    while index + 4 <= query.len() {
        let bits = unsafe { vld1q_u32(bytes.as_ptr().add(index * size_of::<f32>()).cast::<u32>()) };
        let values = vreinterpretq_f32_u32(bits);
        let query_values = unsafe { vld1q_f32(query.as_ptr().add(index)) };
        invalid = vorrq_u32(
            invalid,
            vceqq_u32(vandq_u32(bits, exponent_mask), exponent_mask),
        );
        let delta = vsubq_f32(query_values, values);
        sum = vmlaq_f32(sum, delta, delta);
        index += 4;
    }
    if vmaxvq_u32(invalid) != 0 {
        return Err(invalid_data("DiskANN raw vectors must be finite"));
    }

    let mut distance = vaddvq_f32(sum);
    while index < query.len() {
        let start = index * size_of::<f32>();
        let value = f32::from_le_bytes(
            bytes[start..start + size_of::<f32>()]
                .try_into()
                .expect("validated four-byte raw vector component"),
        );
        if !value.is_finite() {
            return Err(invalid_data("DiskANN raw vectors must be finite"));
        }
        let delta = query[index] - value;
        distance += delta * delta;
        index += 1;
    }
    Ok(distance)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn score_pq_neighbors(
    distance_table: &[f32],
    pq_codes: &[u8],
    pq_m: usize,
    pq_bits: usize,
    nodes: &[u32],
    scored: &mut Vec<SearchCandidate>,
) -> io::Result<usize> {
    if pq_m == 0 {
        return Err(invalid_data("DiskANN PQ subspace count must be positive"));
    }
    let (code_size, ksub) = match pq_bits {
        4 => (pq_m.div_ceil(2), 16),
        8 => (pq_m, 256),
        _ => return Err(invalid_data("DiskANN PQ bits must be 4 or 8")),
    };
    let table_len = pq_m
        .checked_mul(ksub)
        .ok_or_else(|| invalid_data("DiskANN PQ distance-table length overflows"))?;
    if distance_table.len() < table_len {
        return Err(invalid_data("DiskANN PQ distance table is truncated"));
    }

    scored.clear();
    if scored.capacity() < nodes.len() {
        scored.reserve(nodes.len());
    }
    let mut node_index = 0;
    let mut four_code_batches = 0;
    let prefetch_lookahead = pq_prefetch_lookahead(code_size);
    while node_index + 4 <= nodes.len() {
        for &future_node in nodes
            .iter()
            .skip(node_index.saturating_add(prefetch_lookahead))
            .take(4)
        {
            prefetch_pq_code(pq_codes, future_node as usize, code_size);
        }
        let batch_nodes = [
            nodes[node_index] as usize,
            nodes[node_index + 1] as usize,
            nodes[node_index + 2] as usize,
            nodes[node_index + 3] as usize,
        ];
        let mut offsets = [0; 4];
        for index in 0..4 {
            offsets[index] = batch_nodes[index]
                .checked_mul(code_size)
                .ok_or_else(|| invalid_data("DiskANN PQ code offset overflows"))?;
            let end = offsets[index]
                .checked_add(code_size)
                .ok_or_else(|| invalid_data("DiskANN PQ code range overflows"))?;
            if end > pq_codes.len() {
                return Err(invalid_data("DiskANN PQ codes are truncated"));
            }
        }
        let distances = if pq_bits == 4 {
            pq_distance_four_packed_4bit(distance_table, pq_codes, pq_m, offsets)
        } else {
            pq_distance_four_codes(distance_table, pq_codes, pq_m, ksub, offsets)
        };
        for index in 0..4 {
            scored.push(SearchCandidate {
                node: batch_nodes[index],
                distance: distances[index],
            });
        }
        node_index += 4;
        four_code_batches += 1;
    }

    for node in &nodes[node_index..] {
        let node = *node as usize;
        let start = node
            .checked_mul(code_size)
            .ok_or_else(|| invalid_data("DiskANN PQ code offset overflows"))?;
        let end = start
            .checked_add(code_size)
            .ok_or_else(|| invalid_data("DiskANN PQ code range overflows"))?;
        let codes = pq_codes
            .get(start..end)
            .ok_or_else(|| invalid_data("DiskANN PQ codes are truncated"))?;
        scored.push(SearchCandidate {
            node,
            distance: if pq_bits == 4 {
                pq_distance_packed_4bit(distance_table, codes, pq_m)
            } else {
                pq_distance_from_table(distance_table, codes, pq_m, ksub)
            },
        });
    }
    Ok(four_code_batches)
}

#[inline]
fn pq_prefetch_lookahead(code_size: usize) -> usize {
    // Aim for roughly 256 bytes of independent PQ-code work between the hint
    // and its use, while bounding both tiny-code overhead and large-code delay.
    (256 / code_size.max(1)).clamp(4, 16)
}

#[inline]
fn prefetch_pq_code(pq_codes: &[u8], node: usize, code_size: usize) {
    let Some(offset) = node.checked_mul(code_size) else {
        return;
    };
    let Some(byte) = pq_codes.get(offset) else {
        return;
    };
    let pointer = byte as *const u8;
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `pointer` comes from an in-bounds slice element and the intrinsic
    // only issues a non-faulting read hint.
    unsafe {
        std::arch::x86_64::_mm_prefetch(pointer.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `pointer` comes from an in-bounds slice element. `prfm` is a
    // non-faulting cache hint and neither dereferences nor mutates Rust memory.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{address}]",
            address = in(reg) pointer,
            options(readonly, nostack, preserves_flags)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = pointer;
}

#[inline]
fn pq_distance_packed_4bit(distance_table: &[f32], codes: &[u8], pq_m: usize) -> f32 {
    let mut distance = 0.0f32;
    for sub in 0..pq_m {
        let byte = codes[sub / 2];
        let code = if sub.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        distance += distance_table[sub * 16 + code as usize];
    }
    distance
}

#[inline]
fn pq_distance_four_packed_4bit(
    distance_table: &[f32],
    codes: &[u8],
    pq_m: usize,
    offsets: [usize; 4],
) -> [f32; 4] {
    let mut distances = [0.0f32; 4];
    for sub in 0..pq_m {
        let table_start = sub * 16;
        for vector in 0..4 {
            let byte = codes[offsets[vector] + sub / 2];
            let code = if sub.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            distances[vector] += distance_table[table_start + code as usize];
        }
    }
    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::{
        DiskAnnBuildParams, DiskAnnIndex, DiskAnnRawVectorEncoding, DiskAnnStorageLayout,
    };
    use crate::diskann_io::{write_diskann_index, DiskAnnIndexReader};
    use crate::distance::MetricType;
    use crate::index::VectorIndexReaderOptions;
    use crate::io::{PosWriter, ReadRequest, SeekRead, SeekReadCapabilities};
    use crate::read_options::DeploymentProfile;
    use roaring::RoaringTreemap;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Mutex};

    type ReadRounds = Arc<Mutex<Vec<Vec<(u64, usize)>>>>;

    #[test]
    fn diskann_compact_vector_windows_pack_complete_records_without_page_padding() {
        let section = SectionRange::new(8192, 7 * 3840);
        let local =
            VectorWindowPlanner::new(DeploymentProfile::LocalStorage.read_plan(), section, 3840)
                .unwrap();
        assert_eq!(
            local.plan_nodes([6, 1, 0, 6]),
            vec![
                ReadWindow::new(8192, 4 * 3840),
                ReadWindow::new(8192 + 4 * 3840, 3 * 3840),
            ]
        );

        let remote =
            VectorWindowPlanner::new(DeploymentProfile::RemoteStorage.read_plan(), section, 3840)
                .unwrap();
        assert_eq!(
            remote.plan_nodes([6, 0, 6]),
            vec![ReadWindow::new(8192, 26880)]
        );
        assert_eq!(
            remote.window_for_node(6),
            Some(ReadWindow::new(8192, 26880))
        );
        assert_eq!(remote.window_for_node(7), None);
    }

    #[test]
    fn diskann_f16_raw_vector_distance_decodes_little_endian_components() {
        let vector = [1.25f32, -2.5, 4.0];
        let bytes = vector
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let distance = raw_vector_distance(
            &[1.0, -2.0, 5.0],
            &bytes,
            DiskAnnRawVectorEncoding::F16,
            MetricType::L2,
            RawVectorDistanceKernel::Scalar,
        )
        .unwrap();
        assert!((distance - 1.3125).abs() < 1e-6);
    }

    #[test]
    fn diskann_raw_vector_distance_preserves_ip_and_cosine_score_semantics() {
        let vector = [3.0f32, 4.0];
        let f32_bytes = vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let f16_bytes = vector
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();

        for (bytes, encoding) in [
            (f32_bytes.as_slice(), DiskAnnRawVectorEncoding::F32),
            (f16_bytes.as_slice(), DiskAnnRawVectorEncoding::F16),
        ] {
            assert_eq!(
                raw_vector_distance(
                    &[1.0, -2.0],
                    bytes,
                    encoding,
                    MetricType::InnerProduct,
                    RawVectorDistanceKernel::Scalar,
                )
                .unwrap(),
                5.0
            );
            assert!(
                (raw_vector_distance(
                    &[1.0, 0.0],
                    bytes,
                    encoding,
                    MetricType::Cosine,
                    RawVectorDistanceKernel::Scalar,
                )
                .unwrap()
                    - 0.4)
                    .abs()
                    < 1e-6
            );
        }
    }

    #[test]
    fn diskann_f16_raw_vector_distance_handles_unaligned_wide_records() {
        let query = (0..65).map(|index| index as f32 * 0.25).collect::<Vec<_>>();
        let vector = (0..65)
            .map(|index| index as f32 * -0.5 + 3.0)
            .collect::<Vec<_>>();
        let mut storage = vec![0x7f];
        storage.extend(
            vector
                .iter()
                .flat_map(|&value| half::f16::from_f32(value).to_bits().to_le_bytes()),
        );
        let bytes = &storage[1..];
        let expected = query
            .iter()
            .zip(&vector)
            .map(|(left, right)| {
                let decoded = half::f16::from_f32(*right).to_f32();
                let delta = left - decoded;
                delta * delta
            })
            .sum::<f32>();

        let distance = l2_distance_from_f16_le_bytes(&query, bytes).unwrap();

        assert!((distance - expected).abs() <= expected.abs() * 1.0e-5);
        let mut non_finite = bytes.to_vec();
        let non_finite_offset = size_of::<[u16; 8]>();
        non_finite[non_finite_offset..non_finite_offset + size_of::<u16>()]
            .copy_from_slice(&half::f16::INFINITY.to_bits().to_le_bytes());
        assert_eq!(
            l2_distance_from_f16_le_bytes(&query, &non_finite)
                .expect_err("SIMD F16 distance must reject infinity")
                .kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn diskann_compact_f16_roundtrips_dense_vector_records() {
        let dimension = 8;
        let count = 64;
        let data = (0..count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                raw_vector_encoding: DiskAnnRawVectorEncoding::F16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        assert_eq!(header.raw_vector_encoding(), DiskAnnRawVectorEncoding::F16);
        assert_eq!(header.vector_record_size, (dimension * 2) as u32);
        assert_eq!(
            header.sections.vectors.length,
            (count * dimension * 2) as u64
        );

        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        let (result_ids, distances) = reader.search(&data[..dimension], 1, 100).unwrap();
        assert_eq!(result_ids, vec![ids[0]]);
        assert_eq!(distances, vec![0.0]);
    }

    #[test]
    fn diskann_interleaved_layout_searches_and_batches_without_a_vector_section() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                storage_layout: DiskAnnStorageLayout::Interleaved,
                raw_vector_encoding: DiskAnnRawVectorEncoding::F16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        assert!(header.is_interleaved());
        assert_eq!(header.raw_vector_encoding(), DiskAnnRawVectorEncoding::F16);
        assert_eq!(header.sections.vectors.length, 0);
        assert_eq!(header.file_len, header.sections.vectors.offset);

        let queries = &data[..dimension * 4];
        let mut batch_reader = DiskAnnIndexReader::open(CloneCountingReader {
            bytes: Arc::from(bytes.clone()),
            clone_count: Arc::new(AtomicUsize::new(0)),
            reads: Arc::new(Mutex::new(Vec::new())),
        })
        .unwrap();
        let batch = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| batch_reader.search_batch(queries, 5, 100))
            .unwrap();
        assert_eq!(batch_reader.last_search_stats().parallel_session_queries, 4);

        let mut single_reader = DiskAnnIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let mut expected_ids = Vec::new();
        let mut expected_distances = Vec::new();
        for query in queries.chunks_exact(dimension) {
            let (query_ids, query_distances) = single_reader.search(query, 5, 100).unwrap();
            expected_ids.extend(query_ids);
            expected_distances.extend(query_distances);
        }
        assert_eq!(batch.0, expected_ids);
        assert_eq!(batch.1, expected_distances);
        assert_eq!(batch.0[0], ids[0]);

        let mut filter = RoaringTreemap::new();
        filter.extend([ids[0] as u64, ids[1] as u64]);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();
        let filtered_batch = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                batch_reader.search_batch_with_roaring_filter(queries, 3, 100, &filter_bytes)
            })
            .unwrap();
        let mut filtered_single = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        let mut filtered_ids = Vec::new();
        let mut filtered_distances = Vec::new();
        for query in queries.chunks_exact(dimension) {
            let (query_ids, query_distances) = filtered_single
                .search_with_roaring_filter(query, 3, 100, &filter_bytes)
                .unwrap();
            filtered_ids.extend(query_ids);
            filtered_distances.extend(query_distances);
        }
        assert_eq!(filtered_batch, (filtered_ids, filtered_distances));
    }

    #[test]
    fn diskann_pq_neighbor_scoring_batches_four_codes_and_matches_scalar_distance() {
        let pq_m = 4;
        let distance_table = (0..pq_m * 256)
            .map(|index| index as f32 * 0.25)
            .collect::<Vec<_>>();
        let pq_codes = vec![
            1, 2, 3, 4, //
            5, 6, 7, 8, //
            9, 10, 11, 12, //
            13, 14, 15, 16, //
            17, 18, 19, 20,
        ];
        let nodes = [4_u32, 1, 3, 0, 2];
        let mut scored = Vec::new();

        let four_code_batches =
            score_pq_neighbors(&distance_table, &pq_codes, pq_m, 8, &nodes, &mut scored).unwrap();

        assert_eq!(four_code_batches, 1);
        assert_eq!(
            scored
                .iter()
                .map(|candidate| candidate.node)
                .collect::<Vec<_>>(),
            nodes.iter().map(|node| *node as usize).collect::<Vec<_>>()
        );
        for (candidate, node) in scored.iter().zip(nodes) {
            let start = node as usize * pq_m;
            let expected =
                pq_distance_from_table(&distance_table, &pq_codes[start..start + pq_m], pq_m, 256);
            assert_eq!(candidate.distance, expected);
        }

        assert!(score_pq_neighbors(
            &distance_table,
            &pq_codes[..pq_codes.len() - 1],
            pq_m,
            8,
            &nodes,
            &mut scored,
        )
        .is_err());
    }

    #[test]
    fn diskann_packed_4bit_neighbor_scoring_uses_vector_code_size() {
        let pq_m = 4;
        let pq_bits = 4;
        let distance_table = (0..pq_m * 16)
            .map(|index| index as f32 * 0.25)
            .collect::<Vec<_>>();
        let pq_codes = vec![
            0x10, 0x32, // node 0: [0, 1, 2, 3]
            0x54, 0x76, // node 1: [4, 5, 6, 7]
            0x98, 0xBA, // node 2: [8, 9, 10, 11]
            0xDC, 0xFE, // node 3: [12, 13, 14, 15]
            0x21, 0x43, // node 4: [1, 2, 3, 4]
        ];
        let nodes = [4, 1, 3, 2, 0];
        let mut scored = Vec::new();

        let batches = score_pq_neighbors(
            &distance_table,
            &pq_codes,
            pq_m,
            pq_bits,
            &nodes,
            &mut scored,
        )
        .unwrap();

        assert_eq!(batches, 1);
        for (actual, &node) in scored.iter().zip(&nodes) {
            let start = node as usize * 2;
            let expected = [0, 1, 2, 3]
                .into_iter()
                .map(|sub| {
                    let byte = pq_codes[start + sub / 2];
                    let code = if sub.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    distance_table[sub * 16 + code as usize]
                })
                .sum::<f32>();
            assert_eq!(actual.node, node as usize);
            assert_eq!(actual.distance, expected);
        }

        let error = score_pq_neighbors(
            &distance_table,
            &pq_codes[..pq_codes.len() - 1],
            pq_m,
            pq_bits,
            &nodes,
            &mut scored,
        )
        .expect_err("a packed code truncated after the last low nibble must fail closed");
        assert!(error.to_string().contains("truncated"));
    }

    #[test]
    fn diskann_filtered_candidate_selection_stays_bounded_and_deterministic() {
        let mut candidates = std::collections::BinaryHeap::new();
        for node in (0..1000).rev() {
            push_bounded_candidate(
                &mut candidates,
                SearchCandidate {
                    node,
                    distance: (node / 2) as f32,
                },
                10,
            )
            .unwrap();
            assert!(candidates.len() <= 10);
        }
        let mut selected = candidates.into_vec();
        sort_candidates(&mut selected);

        assert_eq!(
            selected
                .iter()
                .map(|candidate| candidate.node)
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn diskann_filtered_candidate_target_is_bounded_by_matches() {
        assert_eq!(desired_filtered_candidate_count(1_000, 10), 64);
        assert_eq!(desired_filtered_candidate_count(20, 10), 20);
        assert_eq!(desired_filtered_candidate_count(1_000, usize::MAX), 1_000);
    }

    #[test]
    fn diskann_filtered_pq_tile_reuses_four_queries_without_exceeding_table_budget() {
        assert_eq!(filtered_pq_query_tile_size(16, 256), 4);
        assert_eq!(filtered_pq_query_tile_size(1024, 256), 2);
        assert_eq!(filtered_pq_query_tile_size(1024, 16), 4);
        for (pq_m, pq_ksub) in [(16, 256), (1024, 256), (1024, 16)] {
            let tile_size = filtered_pq_query_tile_size(pq_m, pq_ksub);
            let table_bytes = tile_size * pq_m * pq_ksub * size_of::<f32>();
            assert!(table_bytes <= FILTERED_PQ_TILE_TABLE_LIMIT_BYTES);
        }
    }

    #[test]
    fn diskann_rerank_expands_candidates_only_within_seed_windows() {
        let planner = ReadWindowPlanner::new(
            DeploymentProfile::LocalStorage.read_plan(),
            SectionRange::new(4096, 4 * DISKANN_PAGE_SIZE as u64),
        );
        let candidates = [
            SearchCandidate {
                node: 1,
                distance: 1.0,
            },
            SearchCandidate {
                node: 16,
                distance: 2.0,
            },
            SearchCandidate {
                node: 2,
                distance: 3.0,
            },
            SearchCandidate {
                node: 31,
                distance: 4.0,
            },
            SearchCandidate {
                node: 32,
                distance: 5.0,
            },
        ];
        let mut selected = Vec::new();
        let mut selected_windows = HashSet::new();

        expand_rerank_candidates_within_seed_windows(
            &candidates,
            2,
            |node| {
                planner
                    .window_for_logical_page(node / 16)
                    .map(|window| window.offset)
                    .ok_or_else(|| invalid_data("test vector page is out of range"))
            },
            &mut selected_windows,
            &mut selected,
        )
        .unwrap();

        assert_eq!(
            selected
                .iter()
                .map(|candidate| candidate.node)
                .collect::<Vec<_>>(),
            vec![1, 16, 2, 31, 32]
        );
    }

    #[test]
    fn diskann_bounded_exact_heap_matches_full_sort_with_row_id_ties() {
        let mut results = (0..100)
            .map(|row_id| ExactSearchResult {
                row_id,
                distance: ((row_id * 37) % 11) as f32,
            })
            .collect::<Vec<_>>();
        let mut expected = results.clone();
        expected.sort();
        expected.truncate(7);
        let mut heap = BinaryHeap::new();

        for result in results.drain(..).rev() {
            push_bounded_exact_result(&mut heap, result, 7).unwrap();
            assert!(heap.len() <= 7);
        }

        assert_eq!(heap.into_sorted_vec(), expected);
    }

    #[test]
    fn diskann_adaptive_filter_strategy_gates_selectivity_cost_and_access_pattern() {
        let scan = FilteredCandidateStrategy::Exhaustive {
            target_candidates: 64,
        };
        assert_eq!(
            select_filtered_candidate_strategy(
                1_000_000,
                499_999,
                10,
                100,
                8,
                DeploymentProfile::LocalStorage.read_plan(),
                false,
            ),
            scan
        );
        assert_eq!(
            select_filtered_candidate_strategy(
                10_000,
                10_000,
                10,
                100,
                64,
                DeploymentProfile::LocalStorage.read_plan(),
                false,
            ),
            scan
        );
        assert_eq!(
            select_filtered_candidate_strategy(
                1_000_000,
                1_000_000,
                10,
                100,
                64,
                DeploymentProfile::LocalStorage.read_plan(),
                false,
            ),
            scan
        );

        let graph = FilteredCandidateStrategy::Graph {
            target_candidates: 64,
            search_list_size: 200,
        };
        assert_eq!(
            select_filtered_candidate_strategy(
                1_000_000,
                1_000_000,
                10,
                200,
                64,
                DeploymentProfile::LocalStorage.read_plan(),
                false,
            ),
            graph
        );
        assert_eq!(
            select_filtered_candidate_strategy(
                1_000_000,
                1_000_000,
                10,
                200,
                64,
                DeploymentProfile::ObjectStore.read_plan(),
                false,
            ),
            scan
        );
        assert_eq!(
            select_filtered_candidate_strategy(
                1_000_000,
                1_000_000,
                10,
                200,
                64,
                DeploymentProfile::ObjectStore.read_plan(),
                true,
            ),
            graph
        );
    }

    #[test]
    fn diskann_adaptive_filter_strategy_scales_and_caps_search_list_safely() {
        assert_eq!(
            adaptive_filtered_search_list_size(1_000_000, 500_000, 64, 100),
            256
        );
        assert_eq!(adaptive_filtered_search_list_size(10, 5, 5, usize::MAX), 10);
        assert_eq!(
            adaptive_filtered_search_list_size(
                usize::MAX,
                usize::MAX,
                usize::MAX,
                resolve_diskann_l_search(usize::MAX, 0),
            ),
            usize::MAX
        );
    }

    #[test]
    fn diskann_automatic_l_search_scales_with_top_k_and_preserves_explicit_values() {
        assert_eq!(resolve_diskann_l_search(10, 0), 100);
        assert_eq!(resolve_diskann_l_search(100, 0), 200);
        assert_eq!(resolve_diskann_l_search(usize::MAX, 0), usize::MAX);
        assert_eq!(resolve_diskann_l_search(100, 64), 100);
        assert_eq!(resolve_diskann_l_search(10, 64), 64);
    }

    #[test]
    fn diskann_graph_candidates_are_post_filtered_and_require_target_count() {
        let graph = vec![
            SearchCandidate {
                node: 1,
                distance: 1.0,
            },
            SearchCandidate {
                node: 2,
                distance: 2.0,
            },
            SearchCandidate {
                node: 3,
                distance: 3.0,
            },
        ];
        let matching = RoaringBitmap::from_iter([2, 3]);

        assert!(post_filter_graph_candidates(&graph, &matching, 3).is_none());
        assert_eq!(
            post_filter_graph_candidates(&graph, &matching, 2)
                .unwrap()
                .iter()
                .map(|candidate| candidate.node)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn diskann_filtered_batch_rerank_chunks_bound_bytes_and_ranges() {
        let range_limited = (0..1025)
            .map(|index| ReadWindow::new(index * 4096, 4096))
            .collect::<Vec<_>>();
        assert_eq!(
            plan_streaming_window_chunks(&range_limited),
            vec![0..1024, 1024..1025]
        );

        let byte_limited = vec![
            ReadWindow::new(0, 40 * 1024 * 1024),
            ReadWindow::new(40 * 1024 * 1024, 30 * 1024 * 1024),
        ];
        assert_eq!(
            plan_streaming_window_chunks(&byte_limited),
            vec![0..1, 1..2]
        );
    }

    #[test]
    fn diskann_filtered_lookup_returns_duplicate_rows_and_ignores_oversized_ids() {
        let row_ids = [7, -1, 7, 3, 9];
        let order = vec![1, 3, 0, 2, 4];
        let mut filter = RoaringTreemap::new();
        filter.insert(7);
        filter.insert(i64::MAX as u64 + 1);

        let ranges =
            matching_ranges_from_row_id_order(&order, &filter, |node| Ok(row_ids[node])).unwrap();
        let nodes = ranges
            .into_iter()
            .flat_map(|range| order[range].iter().copied())
            .collect::<Vec<_>>();

        assert_eq!(nodes, vec![0, 2]);
        assert!(!use_row_id_order(0, 64));
        assert!(use_row_id_order(4, 64));
        assert!(!use_row_id_order(5, 64));
    }

    #[test]
    fn diskann_filtered_batch_dense_translation_visits_each_row_id_once() {
        let row_ids = [-5, 7, 7, 3, 11];
        let mut filter = RoaringTreemap::new();
        filter.insert(7);
        filter.insert(i64::MAX as u64 + 1);
        let mut visits = 0usize;

        let matching = matching_nodes_from_sequential_row_ids(&filter, |visitor| {
            for (node, &row_id) in row_ids.iter().enumerate() {
                visits += 1;
                visitor(node, row_id)?;
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(visits, row_ids.len());
        assert_eq!(matching.iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn diskann_bounded_heap_frontier_reuses_allocation_and_retains_total_order() {
        let mut scratch = DiskAnnQueryScratch::default();
        scratch.begin_search(1_000);
        for node in (0..1_000).rev() {
            scratch
                .insert_graph_candidate(
                    SearchCandidate {
                        node,
                        distance: (node / 2) as f32,
                    },
                    10,
                )
                .unwrap();
            assert!(scratch.retained_candidates.len() <= 10);
            assert!(scratch.frontier.len() <= 20);
        }
        let frontier_capacity = scratch.frontier.capacity();
        assert!(frontier_capacity <= 40);

        scratch.finish_graph_candidates();

        assert_eq!(
            scratch
                .candidates
                .iter()
                .map(|candidate| candidate.node)
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );

        scratch.begin_search(1_000);
        assert_eq!(scratch.frontier.capacity(), frontier_capacity);
    }

    #[test]
    fn diskann_bounded_frontier_matches_round_by_round_vector_oracle() {
        let discovered_rounds = (0..12)
            .map(|round| {
                (0..9)
                    .map(|slot| {
                        let node = 1 + round * 9 + ((slot * 5 + round) % 9);
                        SearchCandidate {
                            node,
                            distance: ((node * 17) % 23) as f32,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (limit, beam_width) in [(8, 1), (16, 4), (32, 16)] {
            let entry = SearchCandidate {
                node: 0,
                distance: 11.0,
            };
            let mut scratch = DiskAnnQueryScratch::default();
            scratch.begin_search(256);
            scratch.insert_graph_candidate(entry, limit).unwrap();
            let mut oracle = vec![(entry, false)];

            for discovered in &discovered_rounds {
                scratch.select_round(beam_width);
                let oracle_selected = oracle
                    .iter_mut()
                    .filter(|(_, expanded)| !*expanded)
                    .take(beam_width)
                    .map(|(candidate, expanded)| {
                        *expanded = true;
                        candidate.node
                    })
                    .collect::<Vec<_>>();
                assert_eq!(scratch.selected_nodes, oracle_selected);
                if scratch.selected_nodes.is_empty() {
                    break;
                }
                for &candidate in discovered {
                    scratch.insert_graph_candidate(candidate, limit).unwrap();
                    oracle.push((candidate, false));
                }
                oracle.sort_by_key(|entry| entry.0);
                oracle.truncate(limit);
            }

            scratch.finish_graph_candidates();
            let oracle_candidates = oracle
                .into_iter()
                .map(|(candidate, _)| candidate)
                .collect::<Vec<_>>();
            assert_eq!(scratch.candidates, oracle_candidates);
        }
    }

    #[test]
    fn diskann_query_scratch_reuses_visited_bitmap_and_clears_touched_nodes() {
        let mut scratch = DiskAnnQueryScratch::default();
        scratch.begin_search(1_000_000);
        let capacity = scratch.visited_capacity();

        assert!(scratch.mark_visited(7));
        assert!(!scratch.mark_visited(7));
        assert!(scratch.mark_visited(999_999));

        scratch.begin_search(1_000_000);

        assert_eq!(scratch.visited_capacity(), capacity);
        assert!(!scratch.is_visited(7));
        assert!(!scratch.is_visited(999_999));
    }

    #[test]
    fn diskann_adaptive_visited_switches_between_dense_and_sparse_storage() {
        let mut scratch = DiskAnnQueryScratch::default();
        scratch.begin_graph_search(1024, 100, 8).unwrap();
        assert!(!scratch.uses_sparse_visited());
        assert!(scratch.mark_visited(7));
        assert!(!scratch.mark_visited(7));

        scratch.begin_graph_search(1_000_000, 100, 64).unwrap();
        assert!(!scratch.uses_sparse_visited());
        assert!(scratch.mark_visited(999_999));
        assert!(!scratch.mark_visited(999_999));

        scratch.begin_graph_search(100_000_000, 100, 64).unwrap();
        assert!(scratch.uses_sparse_visited());
        assert!(
            std::any::type_name_of_val(&scratch.sparse_visited).contains("SparseTable"),
            "DiskANN query hot-path sparse visited state must use the internal table"
        );
        assert!(scratch.mark_visited(99_999_999));
        assert!(!scratch.mark_visited(99_999_999));

        scratch.begin_graph_search(100_000_000, 100, 64).unwrap();
        assert!(scratch.mark_visited(99_999_999));
        assert_eq!(scratch.visited_capacity(), 1_000_000);
    }

    #[test]
    fn diskann_query_scratch_clears_graph_buffers_and_keeps_vector_working_set() {
        let mut scratch = DiskAnnQueryScratch::default();
        scratch.begin_search(1024);
        scratch.prepare_distance_table(512).fill(1.0);
        scratch.candidates.reserve(128);
        scratch.candidates.push(SearchCandidate {
            node: 7,
            distance: 1.0,
        });
        scratch.loaded_adjacency_pages.reserve(16);
        scratch.loaded_adjacency_pages.insert(0);
        scratch.adjacency_windows.reserve(16);
        scratch.adjacency_windows.insert(0, vec![0; 4096].into());
        scratch.vector_windows.insert(4096, vec![0; 64 * 1024]);
        let candidate_capacity = scratch.candidates.capacity();
        let page_cache_capacity = scratch.loaded_adjacency_pages.capacity();
        let window_cache_capacity = scratch.adjacency_windows.capacity();

        scratch.begin_search(1024);

        assert!(scratch.candidates.is_empty());
        assert!(scratch.loaded_adjacency_pages.is_empty());
        assert!(scratch.adjacency_windows.is_empty());
        assert_eq!(scratch.vector_windows.get(&4096).unwrap().len(), 64 * 1024);
        assert_eq!(scratch.retained_window_capacity(), 4 * 1024);
        assert_eq!(scratch.candidates.capacity(), candidate_capacity);
        assert_eq!(
            scratch.loaded_adjacency_pages.capacity(),
            page_cache_capacity
        );
        assert_eq!(scratch.adjacency_windows.capacity(), window_cache_capacity);
        assert_eq!(scratch.distance_table.len(), 512);
    }

    #[test]
    fn diskann_query_scratch_drops_windows_over_its_retained_capacity_limit() {
        let mut scratch = DiskAnnQueryScratch::with_window_buffer_limit(4096);
        scratch.adjacency_windows.insert(0, vec![0; 8192].into());

        scratch.begin_search(1);

        assert_eq!(scratch.retained_window_capacity(), 0);
    }

    #[test]
    fn diskann_query_adjacency_cache_evicts_oldest_window_to_fit_budget() {
        let mut cache = AdjacencyWindowCache::default();
        let mut pool = WindowBufferPool::with_retained_capacity_limit(4096);
        cache.insert(0, vec![0; 4096].into());
        cache.insert(4096, vec![0; 4096].into());

        let evictions = cache.trim(&mut pool, 4096);

        assert_eq!(evictions, 1);
        assert!(!cache.contains_key(&0));
        assert!(cache.contains_key(&4096));
        assert_eq!(cache.retained_capacity(), 4096);
    }

    #[test]
    fn diskann_raw_vector_working_set_is_not_retained_over_query_limit() {
        let windows = [
            ReadWindow::new(0, QUERY_WINDOW_BUFFER_LIMIT_BYTES),
            ReadWindow::new(QUERY_WINDOW_BUFFER_LIMIT_BYTES as u64, 1),
        ];
        let mut cache = VectorWindowCache::default();
        cache.insert(0, vec![0; DISKANN_PAGE_SIZE as usize]);
        cache.touch(0);
        let mut pool = WindowBufferPool::default();

        let (retain, _) = prepare_vector_window_cache(
            &windows,
            &mut cache,
            &mut pool,
            QUERY_WINDOW_BUFFER_LIMIT_BYTES,
        );

        assert!(!retain);
        assert!(cache.is_empty());
        assert!(cache.recency.is_empty());
    }

    #[test]
    fn diskann_raw_vector_working_set_drops_overallocated_buffers() {
        let windows = [ReadWindow::new(0, 1)];
        let mut oversized = Vec::with_capacity(QUERY_WINDOW_BUFFER_LIMIT_BYTES + 1);
        oversized.push(0);
        let mut cache = VectorWindowCache::default();
        cache.insert(0, oversized);
        cache.touch(0);
        let mut pool = WindowBufferPool::default();

        let (retain, _) = prepare_vector_window_cache(
            &windows,
            &mut cache,
            &mut pool,
            QUERY_WINDOW_BUFFER_LIMIT_BYTES,
        );

        assert!(retain, "the requested one-byte window is cacheable");
        assert!(
            cache.is_empty(),
            "an oversized allocation must be replaced before the read"
        );
    }

    #[test]
    fn diskann_raw_vector_cache_evicts_only_least_recently_used_windows() {
        let window_bytes = 4 * 1024 * 1024;
        let offsets = [0, window_bytes as u64, (2 * window_bytes) as u64];
        let mut cache = VectorWindowCache::default();
        for offset in offsets {
            cache.insert(offset, vec![0; window_bytes]);
            cache.touch(offset);
        }
        let mut pool = WindowBufferPool::default();

        let evictions = cache.trim(&mut pool, QUERY_WINDOW_BUFFER_LIMIT_BYTES);

        assert_eq!(evictions, 1);
        assert!(!cache.contains_key(&offsets[0]));
        assert!(cache.contains_key(&offsets[1]));
        assert!(cache.contains_key(&offsets[2]));
        assert_eq!(cache.retained_capacity(), QUERY_WINDOW_BUFFER_LIMIT_BYTES);
        assert_eq!(cache.recency.oldest_offsets(), offsets[1..]);
    }

    #[test]
    fn diskann_adjacency_read_keeps_cached_windows_required_by_current_round() {
        let window_bytes = 1024 * 1024;
        let mut cache = AdjacencyWindowCache::default();
        for window in 0..9 {
            cache.insert((window * window_bytes) as u64, vec![0; window_bytes].into());
        }
        let required = [ReadWindow::new(0, window_bytes)];
        let mut pool = WindowBufferPool::default();

        let evictions =
            prepare_adjacency_window_cache(&required, window_bytes, &mut cache, &mut pool);

        assert_eq!(evictions, 2);
        assert!(
            cache.contains_key(&0),
            "a cached window selected for this graph round must survive preparation trimming"
        );
        assert!(!cache.contains_key(&(window_bytes as u64)));
        assert!(!cache.contains_key(&((2 * window_bytes) as u64)));
    }

    #[test]
    fn diskann_vector_window_cache_tracks_capacity_incrementally() {
        let mut cache = VectorWindowCache::default();
        let mut first = Vec::with_capacity(64);
        first.resize(4, 1);
        let mut second = Vec::with_capacity(128);
        second.resize(4, 2);
        cache.insert(10, first);
        cache.insert(20, second);
        cache.touch(10);
        assert_eq!(cache.retained_capacity(), 192);

        let mut buffers = WindowBufferPool::with_retained_capacity_limit(0);
        assert_eq!(cache.trim(&mut buffers, 64), 1);
        assert!(cache.contains_key(&10));
        assert!(!cache.contains_key(&20));
        assert_eq!(cache.retained_capacity(), 64);

        cache.remove(10);
        assert_eq!(cache.retained_capacity(), 0);
    }

    #[test]
    fn diskann_raw_vector_cache_uses_constant_time_recency_updates() {
        let cache = VectorWindowCache::default();

        assert!(
            std::any::type_name_of_val(&cache.recency).ends_with("OffsetLru"),
            "raw-vector cache recency must use linked hash updates instead of scans"
        );
    }

    #[test]
    fn diskann_window_buffer_pool_reuses_capacity_without_reusing_content() {
        let mut pool = WindowBufferPool::with_retained_capacity_limit(8192);
        let mut original = vec![7u8; 4096];
        let allocation = original.as_mut_ptr();
        pool.recycle(original);

        let reused = pool.take(4096).unwrap();

        assert_eq!(reused.as_ptr(), allocation);
        assert_eq!(reused.len(), 4096);
        assert!(reused.iter().all(|value| *value == 0));
        assert_eq!(pool.retained_capacity, 0);
    }

    #[test]
    fn diskann_batch_workers_split_the_aggregate_window_buffer_budget() {
        assert_eq!(window_buffer_limit_per_worker(1), 8 * 1024 * 1024);
        assert_eq!(window_buffer_limit_per_worker(8), 8 * 1024 * 1024);
        assert_eq!(window_buffer_limit_per_worker(16), 4 * 1024 * 1024);
    }

    #[test]
    fn diskann_query_scratch_reuses_round_selection_buffers() {
        let mut scratch = DiskAnnQueryScratch::default();
        for (node, distance) in [(7, 2.0), (11, 3.0), (13, 4.0)] {
            scratch
                .insert_graph_candidate(SearchCandidate { node, distance }, 4)
                .unwrap();
        }

        scratch.select_round(2);
        assert_eq!(scratch.selected_nodes, vec![7, 11]);
        let nodes_capacity = scratch.selected_nodes.capacity();

        scratch.select_round(1);

        assert_eq!(scratch.selected_nodes, vec![13]);
        assert_eq!(scratch.selected_nodes.capacity(), nodes_capacity);
    }

    #[test]
    fn diskann_raw_vector_distance_decodes_page_bytes_directly() {
        let bytes = [1.0f32, 3.0f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let distance = l2_distance_from_le_bytes(&[2.0, 5.0], &bytes).unwrap();

        assert_eq!(distance, 5.0);
    }

    #[test]
    fn diskann_raw_vector_distance_uses_available_simd_kernel() {
        let query = (0..33).map(|index| index as f32 * 0.25).collect::<Vec<_>>();
        let vector = (0..33)
            .map(|index| index as f32 * -0.5 + 3.0)
            .collect::<Vec<_>>();
        let mut storage = vec![0x7f];
        storage.extend(vector.iter().flat_map(|value| value.to_le_bytes()));
        let unaligned = &storage[1..];
        let expected = query
            .iter()
            .zip(&vector)
            .map(|(left, right)| {
                let delta = left - right;
                delta * delta
            })
            .sum::<f32>();

        let kernel = selected_raw_vector_distance_kernel(query.len());
        let distance = l2_distance_from_le_bytes(&query, unaligned).unwrap();
        let explicit_distance =
            l2_distance_from_le_bytes_with_kernel(&query, unaligned, kernel).unwrap();

        assert!((distance - expected).abs() <= expected.abs() * 1.0e-5);
        assert_eq!(explicit_distance, distance);
        #[cfg(all(target_endian = "little", target_arch = "x86_64"))]
        assert_eq!(
            selected_raw_vector_distance_kernel(query.len()),
            if is_x86_feature_detected!("avx2") {
                RawVectorDistanceKernel::Avx2
            } else {
                RawVectorDistanceKernel::Scalar
            }
        );
        #[cfg(all(target_endian = "little", target_arch = "aarch64"))]
        assert_eq!(
            selected_raw_vector_distance_kernel(query.len()),
            RawVectorDistanceKernel::Neon
        );
        #[cfg(not(all(
            target_endian = "little",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )))]
        assert_eq!(
            selected_raw_vector_distance_kernel(query.len()),
            RawVectorDistanceKernel::Scalar
        );

        let mut non_finite = unaligned.to_vec();
        non_finite[8 * size_of::<f32>()..9 * size_of::<f32>()]
            .copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert_eq!(
            l2_distance_from_le_bytes(&query, &non_finite)
                .expect_err("SIMD exact distance must reject infinity")
                .kind(),
            io::ErrorKind::InvalidData
        );

        non_finite[8 * size_of::<f32>()..9 * size_of::<f32>()]
            .copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            l2_distance_from_le_bytes(&query, &non_finite)
                .expect_err("SIMD exact distance must reject NaN")
                .kind(),
            io::ErrorKind::InvalidData
        );
    }

    struct RoundRecordingReader {
        inner: Cursor<Vec<u8>>,
        rounds: ReadRounds,
    }

    impl SeekRead for RoundRecordingReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            self.rounds.lock().unwrap().push(
                ranges
                    .iter()
                    .map(|range| (range.pos, range.buf.len()))
                    .collect(),
            );
            for range in ranges {
                self.inner.set_position(range.pos);
                io::Read::read_exact(&mut self.inner, range.buf)?;
            }
            Ok(())
        }
    }

    struct CapabilityRecordingReader {
        inner: Cursor<Vec<u8>>,
        rounds: ReadRounds,
        capabilities: SeekReadCapabilities,
    }

    impl SeekRead for CapabilityRecordingReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            self.rounds.lock().unwrap().push(
                ranges
                    .iter()
                    .map(|range| (range.pos, range.buf.len()))
                    .collect(),
            );
            for range in ranges {
                self.inner.set_position(range.pos);
                io::Read::read_exact(&mut self.inner, range.buf)?;
            }
            Ok(())
        }

        fn read_capabilities(&self) -> SeekReadCapabilities {
            self.capabilities
        }
    }

    #[derive(Clone)]
    struct CloneCountingReader {
        bytes: Arc<[u8]>,
        clone_count: Arc<AtomicUsize>,
        reads: Arc<Mutex<Vec<(u64, usize)>>>,
    }

    impl SeekRead for CloneCountingReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            for range in ranges {
                self.reads
                    .lock()
                    .unwrap()
                    .push((range.pos, range.buf.len()));
                let start = usize::try_from(range.pos)
                    .map_err(|_| io::Error::other("test read offset exceeds usize"))?;
                let end = start
                    .checked_add(range.buf.len())
                    .ok_or_else(|| io::Error::other("test read range overflows"))?;
                range.buf.copy_from_slice(
                    self.bytes
                        .get(start..end)
                        .ok_or(io::ErrorKind::UnexpectedEof)?,
                );
            }
            Ok(())
        }

        fn try_clone_reader(&self) -> io::Result<Option<Self>> {
            self.clone_count.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(Some(self.clone()))
        }
    }

    #[derive(Clone)]
    struct ToggleFailReader {
        inner: Cursor<Vec<u8>>,
        fail_reads: Arc<AtomicBool>,
    }

    impl SeekRead for ToggleFailReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            if self.fail_reads.load(AtomicOrdering::SeqCst) {
                return Err(io::Error::other("injected query read failure"));
            }
            for range in ranges {
                self.inner.set_position(range.pos);
                io::Read::read_exact(&mut self.inner, range.buf)?;
            }
            Ok(())
        }

        fn try_clone_reader(&self) -> io::Result<Option<Self>> {
            Ok(Some(self.clone()))
        }
    }

    #[test]
    fn diskann_read_tiers_plan_aligned_deduplicated_and_clipped_windows() {
        let section = SectionRange::new(139_264, 20 * 4096);

        let local = ReadWindowPlanner::new(DeploymentProfile::LocalStorage.read_plan(), section);
        assert_eq!(local.beam_width(), 4);
        assert_eq!(
            local.plan_logical_pages([2, 0, 2, 0]),
            vec![ReadWindow::new(139_264, 16 * 1024)]
        );

        let object_store =
            ReadWindowPlanner::new(DeploymentProfile::ObjectStore.read_plan(), section);
        assert_eq!(object_store.beam_width(), 16);
        assert_eq!(
            object_store.plan_logical_pages([19, 1, 16, 0, 15, 19]),
            vec![
                ReadWindow::new(139_264, 64 * 1024),
                ReadWindow::new(139_264 + 64 * 1024, 16 * 1024),
            ]
        );
    }

    #[test]
    fn diskann_reader_capabilities_refine_windows_and_beam_width() {
        let plan = DeploymentProfile::RemoteStorage
            .read_plan()
            .with_capabilities(SeekReadCapabilities {
                estimated_random_read_latency_nanos: 0,
                preferred_alignment_bytes: 8 * 1024,
                preferred_window_bytes: 16 * 1024,
                max_ranges_per_pread: 2,
            });
        let planner = ReadWindowPlanner::new(plan, SectionRange::new(128 * 1024, 64 * 1024));

        assert_eq!(planner.beam_width(), 2);
        assert_eq!(
            planner.plan_logical_pages([0, 1, 3, 4]),
            vec![
                ReadWindow::new(128 * 1024, 16 * 1024),
                ReadWindow::new(144 * 1024, 16 * 1024),
            ]
        );
    }

    #[test]
    fn diskann_reader_capabilities_bound_ranges_per_pread_call() {
        let dimension = 8;
        let count = 32;
        let data = (0..count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (0..count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let rounds = Arc::new(Mutex::new(Vec::new()));
        let source = CapabilityRecordingReader {
            inner: Cursor::new(bytes),
            rounds: Arc::clone(&rounds),
            capabilities: SeekReadCapabilities {
                estimated_random_read_latency_nanos: 0,
                preferred_alignment_bytes: 0,
                preferred_window_bytes: 0,
                max_ranges_per_pread: 2,
            },
        };
        let mut reader = DiskAnnIndexReader::open(source).unwrap();
        rounds.lock().unwrap().clear();
        let mut buffers = [[0u8; 1]; 5];
        let mut requests = buffers
            .iter_mut()
            .enumerate()
            .map(|(offset, buffer)| ReadRequest::new(offset as u64, buffer))
            .collect::<Vec<_>>();

        reader.pread_ranges(&mut requests).unwrap();

        assert_eq!(
            rounds
                .lock()
                .unwrap()
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
    }

    #[test]
    fn diskann_pq_prefetch_lookahead_is_bounded_by_code_work() {
        assert_eq!(pq_prefetch_lookahead(1), 16);
        assert_eq!(pq_prefetch_lookahead(16), 16);
        assert_eq!(pq_prefetch_lookahead(64), 4);
        assert_eq!(pq_prefetch_lookahead(1024), 4);
    }

    #[test]
    fn diskann_representative_query_warmup_preserves_observed_search_stats() {
        let dimension = 8;
        let count = 64;
        let data = (0..count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (0..count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        reader.search(&data[..dimension], 3, 16).unwrap();
        let stats = reader.last_search_stats();

        reader
            .warmup_queries(&data[dimension..5 * dimension], 16)
            .unwrap();

        assert_eq!(reader.last_search_stats(), stats);
        let result = reader
            .search(&data[dimension..2 * dimension], 1, 16)
            .unwrap();
        assert_eq!(result.0.len(), 1);
        assert!(result.1[0].is_finite());
    }

    #[test]
    fn diskann_window_buffers_are_recycled_after_read_failure() {
        let dimension = 8;
        let count = 256;
        let data = (0..count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (0..count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let fail_reads = Arc::new(AtomicBool::new(false));
        let mut reader = DiskAnnIndexReader::open(ToggleFailReader {
            inner: Cursor::new(bytes),
            fail_reads: Arc::clone(&fail_reads),
        })
        .unwrap();
        reader.ensure_resident().unwrap();
        fail_reads.store(true, AtomicOrdering::SeqCst);
        let planner = ReadWindowPlanner::new(
            DeploymentProfile::LocalStorage.read_plan(),
            reader.header.sections.vectors,
        );
        let window = planner.window_for_logical_page(0).unwrap();
        let mut cache = VectorWindowCache::default();
        let mut pool = WindowBufferPool::with_retained_capacity_limit(window.length);

        let error = reader
            .load_vector_windows(&[window], &mut cache, &mut pool)
            .expect_err("injected read failure must propagate");

        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(cache.is_empty());
        assert_eq!(pool.retained_capacity, window.length);
    }

    #[test]
    fn diskann_batch_read_failure_clears_query_local_raw_vector_cache() {
        let dimension = 128;
        let count = 512;
        let data = (0..count * dimension)
            .map(|offset| ((offset * 31) % 997) as f32 * 0.01)
            .collect::<Vec<_>>();
        let ids = (0..count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            8,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let fail_reads = Arc::new(AtomicBool::new(false));
        let mut reader = DiskAnnIndexReader::open(ToggleFailReader {
            inner: Cursor::new(bytes),
            fail_reads: Arc::clone(&fail_reads),
        })
        .unwrap();
        reader.ensure_resident().unwrap();
        reader
            .rerank_with_query_scratch(
                &data[..dimension],
                &[SearchCandidate {
                    node: 0,
                    distance: 0.0,
                }],
                1,
            )
            .unwrap();
        assert!(!reader.query_scratch.vector_windows.is_empty());

        let second_page_node =
            DISKANN_PAGE_SIZE as usize / reader.header.vector_record_size as usize;
        fail_reads.store(true, AtomicOrdering::SeqCst);
        let error = reader
            .rerank_candidate_batch_streaming(
                &data[second_page_node * dimension..(second_page_node + 1) * dimension],
                1,
                vec![vec![(0, vec![second_page_node])]],
            )
            .expect_err("injected batch rerank read failure must propagate");

        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(reader.query_scratch.vector_windows.is_empty());
        assert!(reader.query_scratch.vector_windows.recency.is_empty());
    }

    #[test]
    fn diskann_graph_search_finds_and_exactly_reranks_query_vector() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let query_index = 37;
        let query = &data[query_index * dimension..(query_index + 1) * dimension];
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();

        let (result_ids, distances) = reader.search(query, 5, 100).unwrap();

        assert_eq!(result_ids.len(), 5);
        assert_eq!(result_ids[0], ids[query_index]);
        assert_eq!(distances[0], 0.0);
        assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));
        let cached_vector_capacity = reader
            .query_scratch
            .vector_windows
            .entries
            .values()
            .map(WindowPayload::capacity)
            .sum::<usize>();
        assert!(cached_vector_capacity > 0);
        assert!(cached_vector_capacity <= QUERY_WINDOW_BUFFER_LIMIT_BYTES);
        assert!(reader.query_scratch.adjacency_windows.is_empty());
        assert!(reader.query_scratch.retained_window_capacity() <= QUERY_WINDOW_BUFFER_LIMIT_BYTES);
    }

    #[test]
    fn diskann_4bit_roundtrip_supports_graph_and_filtered_search() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::with_pq_bits(
            dimension,
            MetricType::L2,
            2,
            4,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let query_index = 37;
        let query = &data[query_index * dimension..(query_index + 1) * dimension];
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();

        reader.ensure_resident().unwrap();
        assert_eq!(reader.header.pq_bits, 4);
        assert_eq!(reader.pq().unwrap().ksub, 16);
        assert_eq!(reader.pq_codes().unwrap().len(), indexed_count);

        let (result_ids, distances) = reader.search(query, 5, 100).unwrap();
        assert_eq!(result_ids[0], ids[query_index]);
        assert_eq!(distances[0], 0.0);

        let mut filter = RoaringTreemap::new();
        filter.extend(ids[32..48].iter().map(|row_id| *row_id as u64));
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();
        let (filtered_ids, filtered_distances) = reader
            .search_with_roaring_filter(query, 5, 100, &filter_bytes)
            .unwrap();
        assert_eq!(filtered_ids[0], ids[query_index]);
        assert_eq!(filtered_distances[0], 0.0);

        let mut queries = Vec::new();
        queries.extend_from_slice(query);
        queries.extend_from_slice(query);
        let (batch_ids, batch_distances) = reader
            .search_batch_with_roaring_filter(&queries, 5, 100, &filter_bytes)
            .unwrap();
        assert_eq!(batch_ids[0], ids[query_index]);
        assert_eq!(batch_ids[5], ids[query_index]);
        assert_eq!(batch_distances[0], 0.0);
        assert_eq!(batch_distances[5], 0.0);
    }

    #[test]
    fn diskann_repeated_single_query_reuses_raw_vector_windows() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut reader = DiskAnnIndexReader::open(CloneCountingReader {
            bytes: Arc::from(bytes),
            clone_count: Arc::new(AtomicUsize::new(0)),
            reads: Arc::clone(&reads),
        })
        .unwrap();
        let query = &data[..dimension];

        reads.lock().unwrap().clear();
        let first = reader.search(query, 5, 100).unwrap();
        let first_vector_reads = reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(offset, _)| {
                *offset >= header.sections.vectors.offset
                    && *offset < header.sections.vectors.offset + header.sections.vectors.length
            })
            .count();
        assert!(first_vector_reads > 0);
        let first_stats = reader.last_search_stats();
        assert_eq!(first_stats.raw_vector_cache_hits, 0);
        assert_eq!(
            first_stats.raw_vector_cache_misses,
            first_stats.rerank_unique_windows
        );

        reads.lock().unwrap().clear();
        let second = reader.search(query, 5, 100).unwrap();
        let second_vector_reads = reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(offset, _)| {
                *offset >= header.sections.vectors.offset
                    && *offset < header.sections.vectors.offset + header.sections.vectors.length
            })
            .count();

        assert_eq!(second, first);
        let second_stats = reader.last_search_stats();
        assert_eq!(
            second_stats.raw_vector_cache_hits,
            second_stats.rerank_unique_windows
        );
        assert_eq!(second_stats.raw_vector_cache_misses, 0);
        assert_eq!(
            second_vector_reads, 0,
            "the previous single-query raw-vector working set should be reused"
        );
    }

    #[test]
    fn diskann_repeated_query_reuses_shared_adjacency_windows() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut reader = DiskAnnIndexReader::open_with_options(
            CloneCountingReader {
                bytes: Arc::from(bytes),
                clone_count: Arc::new(AtomicUsize::new(0)),
                reads: Arc::clone(&reads),
            },
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::Auto,
                0,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();
        let query = &data[..dimension];

        reader.search(query, 5, 100).unwrap();
        reads.lock().unwrap().clear();
        reader.search(query, 5, 100).unwrap();
        let adjacency_reads = reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(offset, _)| {
                *offset >= header.sections.adjacency.offset
                    && *offset < header.sections.adjacency.offset + header.sections.adjacency.length
            })
            .count();

        assert_eq!(
            adjacency_reads, 0,
            "the shared cold-adjacency cache should serve a repeated graph traversal"
        );
        let stats = reader.last_search_stats();
        assert!(stats.adjacency_cache_hits > 0);
        assert_eq!(stats.adjacency_cache_misses, 0);
    }

    #[test]
    fn diskann_shared_adjacency_payload_is_zero_copy() {
        let mut payload = vec![7u8; DISKANN_PAGE_SIZE as usize];
        let allocation = payload.as_mut_ptr();

        let shared = share_window_payload(payload);

        assert_eq!(shared.as_ptr(), allocation);
        assert_eq!(shared.len(), DISKANN_PAGE_SIZE as usize);
        assert!(shared.iter().all(|value| *value == 7));
    }

    #[test]
    fn diskann_zero_adjacency_cache_budget_disables_window_reuse() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut reader = DiskAnnIndexReader::open_with_options(
            CloneCountingReader {
                bytes: Arc::from(bytes),
                clone_count: Arc::new(AtomicUsize::new(0)),
                reads: Arc::clone(&reads),
            },
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::Auto,
                0,
                0,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();
        let query = &data[..dimension];

        reader.search(query, 5, 100).unwrap();
        reads.lock().unwrap().clear();
        reader.search(query, 5, 100).unwrap();
        let adjacency_reads = reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(offset, _)| {
                *offset >= header.sections.adjacency.offset
                    && *offset < header.sections.adjacency.offset + header.sections.adjacency.length
            })
            .count();

        assert!(adjacency_reads > 0);
        let stats = reader.last_search_stats();
        assert_eq!(stats.adjacency_cache_hits, 0);
        assert!(stats.adjacency_cache_misses > 0);
    }

    #[test]
    fn diskann_zero_raw_vector_cache_budget_disables_window_reuse() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut reader = DiskAnnIndexReader::open_with_options(
            CloneCountingReader {
                bytes: Arc::from(bytes),
                clone_count: Arc::new(AtomicUsize::new(0)),
                reads: Arc::clone(&reads),
            },
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::Auto,
                16 * 1024 * 1024,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                0,
            ),
        )
        .unwrap();
        let query = &data[..dimension];

        reader.search(query, 5, 100).unwrap();
        reads.lock().unwrap().clear();
        reader.search(query, 5, 100).unwrap();
        let second_vector_reads = reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(offset, _)| {
                *offset >= header.sections.vectors.offset
                    && *offset < header.sections.vectors.offset + header.sections.vectors.length
            })
            .count();

        assert!(second_vector_reads > 0);
        assert!(reader.query_scratch.vector_windows.is_empty());
    }

    #[test]
    fn diskann_oversized_rerank_counts_evicted_cached_windows() {
        let dimension = 8;
        let count = 256;
        let data = (0..count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = DiskAnnIndexReader::open_with_options(
            Cursor::new(bytes),
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::Auto,
                16 * 1024 * 1024,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                DISKANN_PAGE_SIZE as usize,
            ),
        )
        .unwrap();
        reader.optimize_for_search().unwrap();
        let second_page_node =
            DISKANN_PAGE_SIZE as usize / reader.header.vector_record_size as usize;
        let first_page = SearchCandidate {
            node: 0,
            distance: 0.0,
        };

        reader
            .rerank_with_query_scratch(&data[..dimension], &[first_page], 1)
            .unwrap();
        assert_eq!(reader.query_scratch.vector_windows.len(), 1);

        reader.last_search_stats = DiskAnnSearchStats::default();
        reader
            .rerank_with_query_scratch(
                &data[..dimension],
                &[
                    first_page,
                    SearchCandidate {
                        node: second_page_node,
                        distance: 0.0,
                    },
                ],
                1,
            )
            .unwrap();

        assert_eq!(reader.last_search_stats().raw_vector_cache_evictions, 2);
        assert_eq!(reader.last_search_stats().raw_vector_cache_hits, 1);
        assert_eq!(reader.last_search_stats().raw_vector_cache_misses, 1);
    }

    #[test]
    fn diskann_single_query_cache_reuses_nonconsecutive_hot_vector_window() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 256;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let second_page_node = DISKANN_PAGE_SIZE as usize / header.vector_record_size as usize;
        assert!(second_page_node < indexed_count);
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut reader = DiskAnnIndexReader::open(CloneCountingReader {
            bytes: Arc::from(bytes),
            clone_count: Arc::new(AtomicUsize::new(0)),
            reads: Arc::clone(&reads),
        })
        .unwrap();
        reader.optimize_for_search().unwrap();
        let first_page_candidate = [SearchCandidate {
            node: 0,
            distance: 0.0,
        }];
        let second_page_candidate = [SearchCandidate {
            node: second_page_node,
            distance: 0.0,
        }];
        let vector_read_count = || {
            reads
                .lock()
                .unwrap()
                .iter()
                .filter(|(offset, _)| {
                    *offset >= header.sections.vectors.offset
                        && *offset < header.sections.vectors.offset + header.sections.vectors.length
                })
                .count()
        };

        reads.lock().unwrap().clear();
        reader
            .rerank_with_query_scratch(&data[..dimension], &first_page_candidate, 1)
            .unwrap();
        assert_eq!(vector_read_count(), 1);

        reads.lock().unwrap().clear();
        reader
            .rerank_with_query_scratch(
                &data[second_page_node * dimension..(second_page_node + 1) * dimension],
                &second_page_candidate,
                1,
            )
            .unwrap();
        assert_eq!(vector_read_count(), 1);

        reads.lock().unwrap().clear();
        reader
            .rerank_with_query_scratch(&data[..dimension], &first_page_candidate, 1)
            .unwrap();

        assert_eq!(
            vector_read_count(),
            0,
            "a bounded immutable-window cache should retain nonconsecutive hot pages"
        );
    }

    #[test]
    fn diskann_coalesced_search_batches_graph_beam_and_exact_rerank_reads() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let rounds = Arc::new(Mutex::new(Vec::new()));
        let recording = RoundRecordingReader {
            inner: Cursor::new(bytes),
            rounds: Arc::clone(&rounds),
        };
        let mut reader = DiskAnnIndexReader::open_with_options(
            recording,
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::ObjectStore,
                0,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();

        reader.search(&data[..dimension], 5, 100).unwrap();

        let rounds = rounds.lock().unwrap();
        let adjacency_rounds = rounds
            .iter()
            .filter(|round| {
                round.iter().any(|(offset, _)| {
                    *offset >= header.sections.adjacency.offset
                        && *offset
                            < header.sections.adjacency.offset + header.sections.adjacency.length
                })
            })
            .count();
        let vector_rounds = rounds
            .iter()
            .filter(|round| {
                round.iter().any(|(offset, length)| {
                    *offset >= header.sections.vectors.offset
                        && *offset < header.sections.vectors.offset + header.sections.vectors.length
                        && *length > 1
                })
            })
            .count();
        assert!(adjacency_rounds <= 7, "got {adjacency_rounds} graph rounds");
        assert_eq!(vector_rounds, 1, "rerank must use one batched pread");
    }

    #[test]
    fn diskann_unfiltered_batch_shared_rerank_reads_overlapping_window_once() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let clone_count = Arc::new(AtomicUsize::new(0));
        let mut reader = DiskAnnIndexReader::open(CloneCountingReader {
            bytes: Arc::from(bytes),
            clone_count,
            reads: Arc::clone(&reads),
        })
        .unwrap();
        let query = &data[..dimension];
        let queries = [query, query].concat();

        let (batch_ids, batch_distances) = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| reader.search_batch(&queries, 5, 100))
            .unwrap();

        assert_eq!(&batch_ids[..5], &batch_ids[5..]);
        assert_eq!(&batch_distances[..5], &batch_distances[5..]);
        let vector_reads = reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(offset, _)| {
                *offset >= header.sections.vectors.offset
                    && *offset < header.sections.vectors.offset + header.sections.vectors.length
            })
            .copied()
            .collect::<Vec<_>>();
        assert!(
            !vector_reads.is_empty(),
            "small batches must run complete exact reranks in parallel sessions"
        );
        assert_eq!(
            vector_reads.len(),
            vector_reads.iter().copied().collect::<HashSet<_>>().len(),
            "parallel sessions must share immutable raw-vector windows"
        );
        assert_eq!(reader.last_search_stats().parallel_session_queries, 2);
        let retained_vector_capacity = reader
            .batch_workers
            .iter()
            .flat_map(|worker| worker.query_scratch.vector_windows.entries.values())
            .map(WindowPayload::capacity)
            .sum::<usize>();
        assert!(retained_vector_capacity > 0);
        assert!(
            retained_vector_capacity
                <= reader.options().raw_vector_cache_bytes * reader.batch_workers.len()
        );
        assert_eq!(
            reader
                .batch_workers
                .iter()
                .map(|worker| worker.query_scratch.vector_windows.recency.len())
                .sum::<usize>(),
            reader
                .batch_workers
                .iter()
                .map(|worker| worker.query_scratch.vector_windows.len())
                .sum::<usize>()
        );
        assert_eq!(reader.last_search_stats().parallel_exact_rerank_chunks, 0);
    }

    #[test]
    fn diskann_batch_reuses_worker_readers() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let clone_count = Arc::new(AtomicUsize::new(0));
        let mut reader = DiskAnnIndexReader::open(CloneCountingReader {
            bytes: Arc::from(bytes),
            clone_count: Arc::clone(&clone_count),
            reads: Arc::new(Mutex::new(Vec::new())),
        })
        .unwrap();
        let queries = data[..dimension].repeat(4);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        pool.install(|| reader.search_batch(&queries, 5, 100))
            .unwrap();
        let first_batch_clones = clone_count.load(AtomicOrdering::SeqCst);
        pool.install(|| reader.search_batch(&queries, 5, 100))
            .unwrap();

        assert_eq!(first_batch_clones, 4);
        assert_eq!(
            clone_count.load(AtomicOrdering::SeqCst),
            first_batch_clones,
            "the second batch should reuse retained storage handles"
        );
    }

    #[test]
    fn diskann_batch_restores_worker_pool_after_read_failure() {
        let dimension = 8;
        let count = 256;
        let data = (0..count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (0..count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let fail_reads = Arc::new(AtomicBool::new(false));
        let mut reader = DiskAnnIndexReader::open_with_options(
            ToggleFailReader {
                inner: Cursor::new(bytes),
                fail_reads: Arc::clone(&fail_reads),
            },
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::Auto,
                0,
                0,
                4 * 1024 * 1024 * 1024,
                0,
            ),
        )
        .unwrap();
        let queries = data[..dimension].repeat(4);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        pool.install(|| reader.search_batch(&queries, 5, 100))
            .unwrap();
        assert_eq!(reader.batch_workers.len(), 4);
        fail_reads.store(true, AtomicOrdering::SeqCst);
        pool.install(|| reader.search_batch(&queries, 5, 100))
            .expect_err("injected worker read failure must propagate");
        assert_eq!(reader.batch_workers.len(), 4);
        fail_reads.store(false, AtomicOrdering::SeqCst);
        pool.install(|| reader.search_batch(&queries, 5, 100))
            .unwrap();
        assert_eq!(reader.batch_workers.len(), 4);
    }

    #[test]
    fn diskann_repeated_batch_reuses_reader_raw_vector_cache() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut reader = DiskAnnIndexReader::open(CloneCountingReader {
            bytes: Arc::from(bytes),
            clone_count: Arc::new(AtomicUsize::new(0)),
            reads: Arc::clone(&reads),
        })
        .unwrap();
        let queries = [&data[..dimension], &data[..dimension]].concat();

        reader.search_batch(&queries, 5, 100).unwrap();
        reads.lock().unwrap().clear();
        reader.search_batch(&queries, 5, 100).unwrap();
        let vector_reads = reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(offset, _)| {
                *offset >= header.sections.vectors.offset
                    && *offset < header.sections.vectors.offset + header.sections.vectors.length
            })
            .count();

        assert_eq!(vector_reads, 0);
        let stats = reader.last_search_stats();
        assert!(stats.raw_vector_cache_hits > 0);
        assert_eq!(stats.raw_vector_cache_misses, 0);
    }

    #[test]
    fn diskann_parallel_exact_rerank_matches_single_queries() {
        let dimension = 128;
        let training_count = 512;
        let indexed_count = 256;
        let data = (0..training_count * dimension)
            .map(|offset| {
                ((offset * 31) % 997) as f32 * 0.01 + (offset / dimension) as f32 * 0.0001
            })
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            16,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let query_indices = [0, 31, 127, 255];
        let queries = query_indices
            .into_iter()
            .flat_map(|query_index| {
                data[query_index * dimension..(query_index + 1) * dimension]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        let mut batch_reader = DiskAnnIndexReader::open(Cursor::new(bytes.clone())).unwrap();

        let batch = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| batch_reader.search_batch(&queries, 10, 100))
            .unwrap();
        let mut single_reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        let mut expected_ids = Vec::new();
        let mut expected_distances = Vec::new();
        for query in queries.chunks_exact(dimension) {
            let (query_ids, query_distances) = single_reader.search(query, 10, 100).unwrap();
            expected_ids.extend(query_ids);
            expected_distances.extend(query_distances);
        }

        assert_eq!(batch.0, expected_ids);
        assert_eq!(batch.1, expected_distances);
        let stats = batch_reader.last_search_stats();
        assert!(stats.rerank_candidate_references >= 4 * 64);
        assert_eq!(stats.parallel_exact_rerank_chunks, 1);
        assert_eq!(
            stats.parallel_exact_rerank_references,
            stats.rerank_candidate_references
        );
    }

    #[test]
    fn diskann_batch_query_chunks_bound_live_candidates() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        let query_count = BATCH_QUERY_CHUNK_SIZE + 1;
        let queries = data[..dimension].repeat(query_count);

        let (batch_ids, batch_distances) = reader.search_batch(&queries, 1, 100).unwrap();

        assert_eq!(batch_ids, vec![ids[0]; query_count]);
        assert!(batch_distances.iter().all(|distance| *distance == 0.0));
        let stats = reader.last_search_stats();
        assert_eq!(stats.query_count, query_count);
        assert_eq!(stats.query_chunks, 2);
        assert_eq!(stats.max_queries_per_chunk, BATCH_QUERY_CHUNK_SIZE);
    }

    #[test]
    fn diskann_hot_adjacency_does_not_copy_into_query_window_cache() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = DiskAnnIndexReader::open_with_options(
            Cursor::new(bytes),
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::ObjectStore,
                16 * 1024 * 1024,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();
        reader.optimize_for_search().unwrap();
        let mut window_cache = AdjacencyWindowCache::default();
        let mut page_cache = HashSet::new();
        let mut window_buffers = WindowBufferPool::default();

        reader
            .load_adjacency_pages(
                &[reader.header.entry_node as usize],
                &mut window_cache,
                &mut page_cache,
                &mut window_buffers,
            )
            .unwrap();

        assert!(window_cache.is_empty());
        assert!(!page_cache.is_empty());
    }

    #[test]
    fn diskann_filtered_search_skips_graph_and_pads_sparse_results() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let rounds = Arc::new(Mutex::new(Vec::new()));
        let recording = RoundRecordingReader {
            inner: Cursor::new(bytes),
            rounds: Arc::clone(&rounds),
        };
        let mut reader = DiskAnnIndexReader::open_with_options(
            recording,
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::ObjectStore,
                0,
                0,
                4 * 1024 * 1024 * 1024,
                0,
            ),
        )
        .unwrap();
        let query_index = 37;
        let query = &data[query_index * dimension..(query_index + 1) * dimension];
        let mut filter = RoaringTreemap::new();
        filter.insert(ids[query_index] as u64);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        let (result_ids, distances) = reader
            .search_with_roaring_filter(query, 3, 100, &filter_bytes)
            .unwrap();

        assert_eq!(result_ids, vec![ids[query_index], -1, -1]);
        assert_eq!(distances, vec![0.0, f32::MAX, f32::MAX]);
        let rounds = rounds.lock().unwrap();
        assert!(!rounds.iter().flatten().any(|(offset, _)| {
            *offset >= header.sections.adjacency.offset
                && *offset < header.sections.adjacency.offset + header.sections.adjacency.length
        }));
        assert_eq!(
            rounds
                .iter()
                .filter(|round| round.iter().any(|(offset, length)| {
                    *offset >= header.sections.vectors.offset
                        && *offset < header.sections.vectors.offset + header.sections.vectors.length
                        && *length > 1
                }))
                .count(),
            1
        );
    }

    #[test]
    fn diskann_broad_random_access_filter_uses_graph_candidate_io() {
        let dimension = 8;
        let indexed_count = 1024;
        let data = (0..indexed_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32 + (offset / dimension) as f32 * 0.01)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 8,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, indexed_count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let rounds = Arc::new(Mutex::new(Vec::new()));
        let recording = RoundRecordingReader {
            inner: Cursor::new(bytes),
            rounds: Arc::clone(&rounds),
        };
        let mut reader = DiskAnnIndexReader::open(recording).unwrap();
        let mut filter = RoaringTreemap::new();
        filter.extend(ids.iter().map(|row_id| *row_id as u64));
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        let (result_ids, distances) = reader
            .search_with_roaring_filter(&data[..dimension], 10, 200, &filter_bytes)
            .unwrap();

        assert_eq!(result_ids[0], ids[0]);
        assert_eq!(distances[0], 0.0);
        assert!(rounds.lock().unwrap().iter().flatten().any(|(offset, _)| {
            *offset >= header.sections.adjacency.offset
                && *offset < header.sections.adjacency.offset + header.sections.adjacency.length
        }));
    }

    #[test]
    fn diskann_adaptive_filtered_recall_matrix_stays_within_one_percentage_point() {
        let dimension = 8;
        let indexed_count = 10_000;
        let data = (0..indexed_count)
            .flat_map(|node| {
                (0..dimension).map(move |component| {
                    let mut hash = (node as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add((component as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
                    hash ^= hash >> 30;
                    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    hash ^= hash >> 27;
                    let noise = (hash as u32) as f32 / u32::MAX as f32;
                    noise + (node / 256) as f32 * 0.25
                })
            })
            .collect::<Vec<_>>();
        let ids = (0..indexed_count)
            .map(|node| (node / 2) as i64)
            .collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 16,
                build_search_list_size: 100,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, indexed_count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = DiskAnnIndexReader::open_with_options(
            Cursor::new(bytes),
            VectorIndexReaderOptions::new(4 * 1024 * 1024 * 1024),
        )
        .unwrap();
        reader.ensure_resident().unwrap();

        for distribution in ["random", "clustered"] {
            for selectivity_basis_points in [1usize, 10, 100, 1000, 5000, 10_000] {
                let matching_count = indexed_count
                    .saturating_mul(selectivity_basis_points)
                    .div_ceil(10_000)
                    .max(1);
                let ordered_nodes = if distribution == "random" {
                    (0..indexed_count)
                        .map(|index| (index * 4051) % indexed_count)
                        .collect::<Vec<_>>()
                } else {
                    (0..indexed_count).collect::<Vec<_>>()
                };
                let matching = RoaringBitmap::from_iter(
                    ordered_nodes[..matching_count]
                        .iter()
                        .map(|node| *node as u32),
                );
                let strategy = select_filtered_candidate_strategy(
                    indexed_count,
                    matching_count,
                    10,
                    200,
                    16,
                    DeploymentProfile::Memory.read_plan(),
                    false,
                );
                assert_eq!(
                    matches!(strategy, FilteredCandidateStrategy::Graph { .. }),
                    selectivity_basis_points == 10_000,
                    "unexpected strategy for {distribution} at {selectivity_basis_points} bps"
                );

                let mut hits = 0usize;
                let mut total = 0usize;
                for &query_node in ordered_nodes[..10.min(matching_count)].iter() {
                    let query = &data[query_node * dimension..(query_node + 1) * dimension];
                    let target = desired_filtered_candidate_count(matching_count, 10);
                    let baseline_candidates = reader
                        .exhaustive_filtered_candidates(query, &matching, target)
                        .unwrap();
                    let baseline = reader
                        .rerank_with_query_scratch(query, &baseline_candidates, 10)
                        .unwrap()
                        .0;
                    let adaptive_candidates = reader
                        .generate_filtered_candidates(query, 10, 200, &matching)
                        .unwrap();
                    let adaptive = reader
                        .rerank_with_query_scratch(query, &adaptive_candidates, 10)
                        .unwrap()
                        .0;

                    let mut expected_counts = HashMap::<i64, usize>::new();
                    for row_id in baseline.into_iter().filter(|row_id| *row_id >= 0) {
                        *expected_counts.entry(row_id).or_default() += 1;
                        total += 1;
                    }
                    for row_id in adaptive.into_iter().filter(|row_id| *row_id >= 0) {
                        if expected_counts.get_mut(&row_id).is_some_and(|count| {
                            if *count == 0 {
                                false
                            } else {
                                *count -= 1;
                                true
                            }
                        }) {
                            hits += 1;
                        }
                    }
                }
                let recall = hits as f64 / total as f64;
                assert!(
                    recall + 0.01 + f64::EPSILON >= 1.0,
                    "{distribution} filter at {selectivity_basis_points} bps has Recall@10 {recall:.4} against exhaustive scan"
                );
            }
        }
    }

    #[test]
    fn diskann_zero_k_filtered_search_still_rejects_malformed_filter() {
        let header = crate::diskann_io::DiskAnnHeader::for_layout(
            8,
            1,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 1,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..crate::diskann_io::DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();

        let error = reader
            .search_with_roaring_filter(&[0.0; 8], 0, 100, &[0xff])
            .expect_err("malformed filters must be rejected even for zero k");

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn diskann_filtered_batch_validates_filter_before_cloning_workers() {
        let header = crate::diskann_io::DiskAnnHeader::for_layout(
            8,
            1,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 1,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..crate::diskann_io::DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        let clone_count = Arc::new(AtomicUsize::new(0));
        let mut reader = DiskAnnIndexReader::open(CloneCountingReader {
            bytes: Arc::from(bytes),
            clone_count: Arc::clone(&clone_count),
            reads: Arc::new(Mutex::new(Vec::new())),
        })
        .unwrap();

        let error = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| reader.search_batch_with_roaring_filter(&[0.0; 16], 1, 100, &[0xff]))
            .expect_err("malformed filters must fail before batch fan-out");

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(clone_count.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn diskann_filtered_batch_shares_lookup_without_preloading_adjacency() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = crate::diskann_io::DiskAnnHeader::decode(&bytes[..256]).unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let clone_count = Arc::new(AtomicUsize::new(0));
        let mut reader = DiskAnnIndexReader::open_with_options(
            CloneCountingReader {
                bytes: Arc::from(bytes),
                clone_count: Arc::clone(&clone_count),
                reads: Arc::clone(&reads),
            },
            VectorIndexReaderOptions::with_cache_budgets(
                DeploymentProfile::ObjectStore,
                4096,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();
        let mut filter = RoaringTreemap::new();
        filter.insert(ids[0] as u64);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                reader
                    .search_batch_with_roaring_filter(&data[..dimension * 2], 1, 100, &filter_bytes)
                    .unwrap();
            });

        assert_eq!(
            clone_count.load(AtomicOrdering::SeqCst),
            0,
            "resident-PQ batch scan must not clone the storage reader"
        );
        assert!(reads
            .lock()
            .unwrap()
            .iter()
            .any(|(offset, _)| { *offset == header.sections.row_id_order.offset }));
        assert!(!reads.lock().unwrap().iter().any(|(offset, _)| {
            *offset >= header.sections.adjacency.offset
                && *offset < header.sections.adjacency.offset + header.sections.adjacency.length
        }));
        assert_eq!(
            reads
                .lock()
                .unwrap()
                .iter()
                .filter(|(offset, length)| {
                    *offset >= header.sections.vectors.offset
                        && *offset < header.sections.vectors.offset + header.sections.vectors.length
                        && *length > 1
                })
                .count(),
            1,
            "the parent reranker must read a shared vector window once"
        );
    }

    #[test]
    fn diskann_filtered_streaming_batch_matches_single_queries_with_duplicate_row_ids() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let mut ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        ids[0] = 7;
        ids[1] = 7;
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut filter = RoaringTreemap::new();
        filter.insert(7);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        let queries = &data[..dimension * 2];
        let mut batch_reader = DiskAnnIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        let batch = batch_reader
            .search_batch_with_roaring_filter(queries, 3, 100, &filter_bytes)
            .unwrap();
        let mut single_reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        let mut expected_ids = Vec::new();
        let mut expected_distances = Vec::new();
        for query in queries.chunks_exact(dimension) {
            let (ids, distances) = single_reader
                .search_with_roaring_filter(query, 3, 100, &filter_bytes)
                .unwrap();
            expected_ids.extend(ids);
            expected_distances.extend(distances);
        }

        assert_eq!(batch.0, expected_ids);
        assert_eq!(batch.1, expected_distances);
        assert_eq!(&batch.0[..3], &[7, 7, -1]);
        assert_eq!(&batch.0[3..], &[7, 7, -1]);
    }

    #[test]
    fn diskann_search_stats_report_tiled_filtered_batch() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = DiskAnnIndexReader::open(CloneCountingReader {
            bytes: Arc::from(bytes),
            clone_count: Arc::new(AtomicUsize::new(0)),
            reads: Arc::new(Mutex::new(Vec::new())),
        })
        .unwrap();
        let mut filter = RoaringTreemap::new();
        filter.insert(ids[0] as u64);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                reader
                    .search_batch_with_roaring_filter(&data[..dimension * 2], 1, 100, &filter_bytes)
                    .unwrap();
            });

        let stats = reader.last_search_stats();
        assert_eq!(stats.query_count, 2);
        assert_eq!(stats.filtered_exhaustive_queries, 2);
        assert_eq!(stats.filtered_graph_queries, 0);
        assert_eq!(stats.pq_distance_evaluations, 2);
        assert_eq!(stats.pq_code_loads, 1);
        assert_eq!(stats.rerank_candidate_references, 2);
        assert_eq!(stats.rerank_unique_windows, 1);
    }

    #[test]
    fn diskann_filtered_pq_batch_kernel_reuses_code_loads() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut filter = RoaringTreemap::new();
        filter.extend(ids[..32].iter().map(|row_id| *row_id as u64));
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();
        let queries = &data[..dimension * 4];
        let mut batch_reader = DiskAnnIndexReader::open(Cursor::new(bytes.clone())).unwrap();

        let batch = batch_reader
            .search_batch_with_roaring_filter(queries, 3, 100, &filter_bytes)
            .unwrap();
        let mut single_reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        let mut expected_ids = Vec::new();
        let mut expected_distances = Vec::new();
        for query in queries.chunks_exact(dimension) {
            let (query_ids, query_distances) = single_reader
                .search_with_roaring_filter(query, 3, 100, &filter_bytes)
                .unwrap();
            expected_ids.extend(query_ids);
            expected_distances.extend(query_distances);
        }

        assert_eq!(batch.0, expected_ids);
        assert_eq!(batch.1, expected_distances);
        let stats = batch_reader.last_search_stats();
        assert_eq!(stats.filtered_exhaustive_queries, 4);
        assert_eq!(stats.pq_distance_evaluations, 4 * 32);
        assert_eq!(
            stats.pq_code_loads, 32,
            "one four-query tile should load each matching PQ code once"
        );
    }

    #[test]
    fn diskann_filtered_search_clamps_extreme_l_search_without_panicking() {
        let header = crate::diskann_io::DiskAnnHeader::for_layout(
            8,
            1,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 1,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..crate::diskann_io::DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        let filter = RoaringTreemap::new();
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            reader.search_with_roaring_filter(&[0.0; 8], 1, usize::MAX, &filter_bytes)
        }))
        .expect("public io::Result search API must not panic for extreme l_search")
        .unwrap();

        assert_eq!(result.0, vec![-1]);
        assert_eq!(result.1, vec![f32::MAX]);
    }

    #[test]
    fn topk_stability_ignores_padding_and_aggregates_queries() {
        let left = [10, 11, -1, 20, 21, 22];
        let right = [11, 10, -1, 20, 99, 22];
        let left_distances = [1.0, 2.0, f32::MAX, 1.0, 2.0, 3.0];
        let right_distances = [2.0, 1.0, f32::MAX, 1.0, 9.0, 3.0];

        assert_eq!(
            topk_result_stability(&left, &left_distances, &right, &right_distances, 3),
            4.0 / 5.0
        );
        assert_eq!(topk_result_stability(&[], &[], &[], &[], 0), 0.0);
        assert_eq!(
            topk_result_stability(&[1, 2], &[1.0, 2.0], &[1], &[1.0], 2),
            0.0
        );
    }

    #[test]
    fn topk_stability_supports_negative_ids_and_duplicate_multiplicity() {
        assert_eq!(
            topk_result_stability(&[-1, -7], &[1.0, 2.0], &[-1, -8], &[1.0, 2.0], 2),
            0.5
        );
        assert_eq!(
            topk_result_stability(&[7, 7], &[1.0, 2.0], &[7, 8], &[1.0, 2.0], 2),
            0.5
        );
        assert_eq!(
            topk_result_stability(&[-1, 7], &[f32::MAX, 2.0], &[-1, 7], &[f32::MAX, 2.0], 2,),
            1.0
        );
    }

    #[test]
    fn diskann_calibration_selects_and_remembers_smallest_stable_width() {
        let dimension = 8;
        let indexed_count = 64;
        let data = (0..indexed_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32 + (offset / dimension) as f32 * 0.01)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, indexed_count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        let queries = &data[..4 * dimension];

        assert_eq!(reader.calibrate_l_search(queries, 5).unwrap(), 100);
        assert_eq!(reader.calibrated_l_search, Some(100));
        assert_eq!(reader.last_search_stats(), DiskAnnSearchStats::default());

        let error = reader.calibrate_l_search(&[], 5).unwrap_err();
        assert!(error.to_string().contains("one or more complete vectors"));
    }

    #[test]
    fn diskann_search_stats_report_actual_filtered_scan() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        let mut filter = RoaringTreemap::new();
        filter.insert(ids[0] as u64);
        let mut filter_bytes = Vec::new();
        filter.serialize_into(&mut filter_bytes).unwrap();

        reader
            .search_with_roaring_filter(&data[..dimension], 1, 100, &filter_bytes)
            .unwrap();

        let stats = reader.last_search_stats();
        assert_eq!(stats.query_count, 1);
        assert_eq!(stats.filtered_exhaustive_queries, 1);
        assert_eq!(stats.filtered_graph_queries, 0);
        assert_eq!(stats.filtered_graph_fallbacks, 0);
        assert_eq!(stats.pq_distance_evaluations, 1);
        assert_eq!(stats.rerank_candidate_references, 1);
        assert_eq!(stats.rerank_unique_windows, 1);
        assert_eq!(stats.rerank_chunks, 1);
        assert_eq!(
            reader.query_scratch.visited_capacity(),
            0,
            "an exhaustive filtered rerank must not allocate graph visited storage"
        );
    }
}
