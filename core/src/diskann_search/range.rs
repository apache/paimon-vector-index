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

//! Two-stage graph range search, independently implemented from the published
//! DiskANN algorithm. No Top-K scratch, mutable window cache or counters are used.

use super::*;
use crate::diskann_io::DiskAnnIndexReader;
use crate::range::{
    prepare_range_queries, Bound, DiskAnnRangeSearchParams, DiskAnnRangeSearchStats,
    RangeResultBuilder, RangeSearchParams, RangeSearchResult,
};
use std::collections::VecDeque;

const MAX_IO_BATCH: usize = 32;
const PREFETCH_SCRATCH_BYTES: usize = 4096;

#[derive(Default)]
struct LocalWindow {
    offset: Option<u64>,
    buffer_offset: Option<u64>,
    bytes: Vec<u8>,
    cached: HashMap<u64, Vec<u8>>,
    cache_order: VecDeque<u64>,
    cache_bytes: usize,
    cache_limit: usize,
}

impl LocalWindow {
    fn bytes(&self) -> &[u8] {
        self.offset
            .and_then(|offset| self.cached.get(&offset))
            .map(Vec::as_slice)
            .unwrap_or(&self.bytes)
    }

    fn load<R: SeekRead>(
        &mut self,
        reader: &mut DiskAnnIndexReader<R>,
        window: ReadWindow,
        stats: &mut DiskAnnRangeSearchStats,
    ) -> io::Result<()> {
        if self
            .cached
            .get(&window.offset)
            .is_some_and(|v| v.len() == window.length)
            || (self.buffer_offset == Some(window.offset) && self.bytes.len() == window.length)
        {
            self.offset = Some(window.offset);
            return Ok(());
        }
        self.offset = None;
        self.buffer_offset = None;
        if self.bytes.capacity() < window.length {
            self.bytes
                .try_reserve_exact(window.length - self.bytes.len())
                .map_err(|_| invalid_input("range I/O allocation failed"))?;
        }
        self.bytes.resize(window.length, 0);
        reader.pread_ranges(&mut [ReadRequest::new(window.offset, &mut self.bytes)])?;
        stats.read_calls += 1;
        stats.read_bytes += window.length;
        self.offset = Some(window.offset);
        self.buffer_offset = Some(window.offset);
        // Cache admission is query-local and charged for payload capacity plus
        // conservative hash-table/queue overhead. No shared Top-K LRU is used.
        let charge = self.bytes.capacity().saturating_add(256);
        if self.cache_bytes.saturating_add(charge) <= self.cache_limit {
            self.cached
                .try_reserve(1)
                .map_err(|_| invalid_input("range window cache allocation failed"))?;
            self.cache_order
                .try_reserve(1)
                .map_err(|_| invalid_input("range window order allocation failed"))?;
            self.cache_bytes += charge;
            self.cached
                .insert(window.offset, std::mem::take(&mut self.bytes));
            self.cache_order.push_back(window.offset);
            self.buffer_offset = None;
        }
        Ok(())
    }

    fn prefetch<R: SeekRead>(
        &mut self,
        reader: &mut DiskAnnIndexReader<R>,
        windows: &[ReadWindow],
        stats: &mut DiskAnnRangeSearchStats,
    ) -> io::Result<()> {
        let mut pending = Vec::new();
        let mut incoming = 0usize;
        for &window in windows {
            if self.cached.contains_key(&window.offset)
                || (self.buffer_offset == Some(window.offset) && self.bytes.len() == window.length)
            {
                continue;
            }
            incoming = incoming.saturating_add(window.length).saturating_add(256);
            pending.push((window, Vec::new()));
        }
        if pending.is_empty() || incoming > self.cache_limit {
            return Ok(());
        }
        // Retain all windows needed by this batch; evict only older query-local
        // windows. The whole batch fits before allocating any new payloads.
        let existing = self.cache_order.len();
        for _ in 0..existing {
            if self.cache_bytes.saturating_add(incoming) <= self.cache_limit {
                break;
            }
            let offset = self
                .cache_order
                .pop_front()
                .expect("nonempty range window order");
            if windows.iter().any(|w| w.offset == offset) {
                self.cache_order.push_back(offset);
                continue;
            }
            if let Some(bytes) = self.cached.remove(&offset) {
                self.cache_bytes -= bytes.capacity() + 256;
            }
        }
        if self.cache_bytes.saturating_add(incoming) > self.cache_limit {
            return Ok(());
        }
        self.cached
            .try_reserve(pending.len())
            .map_err(|_| invalid_input("range window cache allocation failed"))?;
        self.cache_order
            .try_reserve(pending.len())
            .map_err(|_| invalid_input("range window order allocation failed"))?;
        let mut actual = 0usize;
        for (window, bytes) in &mut pending {
            bytes
                .try_reserve_exact(window.length)
                .map_err(|_| invalid_input("range prefetch allocation failed"))?;
            bytes.resize(window.length, 0);
            actual = actual.saturating_add(bytes.capacity()).saturating_add(256);
        }
        if self.cache_bytes.saturating_add(actual) > self.cache_limit {
            return Err(invalid_input(
                "DiskANN range working memory budget exceeded",
            ));
        }
        let mut requests = pending
            .iter_mut()
            .map(|(window, bytes)| ReadRequest::new(window.offset, bytes))
            .collect::<Vec<_>>();
        reader.pread_ranges(&mut requests)?;
        let max_ranges = reader.read_capabilities().max_ranges_per_pread;
        stats.read_calls += if max_ranges == 0 {
            1
        } else {
            requests.len().div_ceil(max_ranges)
        };
        stats.read_bytes += requests.iter().map(|r| r.buf.len()).sum::<usize>();
        drop(requests);
        for (window, bytes) in pending {
            self.cache_bytes += bytes.capacity() + 256;
            self.cached.insert(window.offset, bytes);
            self.cache_order.push_back(window.offset);
        }
        Ok(())
    }
}

struct RangeSession<'a, R: SeekRead> {
    reader: &'a mut DiskAnnIndexReader<R>,
    adjacency: LocalWindow,
    vectors: LocalWindow,
    stats: DiskAnnRangeSearchStats,
}

impl<R: SeekRead> RangeSession<'_, R> {
    fn node_window(&self, node: usize, adjacency: bool) -> io::Result<ReadWindow> {
        let window = if adjacency || self.reader.header.is_interleaved() {
            ReadWindowPlanner::new(
                self.reader.read_plan(),
                self.reader.header.sections.adjacency,
            )
            .window_for_logical_page(self.reader.adjacency_locator(node)?.page_index as usize)
        } else {
            VectorWindowPlanner::new(
                self.reader.read_plan(),
                self.reader.header.sections.vectors,
                self.reader.header.vector_record_size as usize,
            )?
            .window_for_node(node)
        };
        window.ok_or_else(|| invalid_data("range prefetch window is out of range"))
    }

    fn io_batch_size(&self, adjacency: bool) -> usize {
        let adjacency = adjacency || self.reader.header.is_interleaved();
        let window = self.reader.read_plan().window_bytes;
        let (limit, window) = if adjacency {
            (self.adjacency.cache_limit, window)
        } else {
            (
                self.vectors.cache_limit,
                window.max(self.reader.header.vector_record_size as usize),
            )
        };
        let cap = self.reader.read_capabilities().max_ranges_per_pread;
        (limit / window.saturating_add(256))
            .min(MAX_IO_BATCH)
            .min(if cap == 0 { MAX_IO_BATCH } else { cap })
            .max(1)
    }

    fn prefetch_nodes(
        &mut self,
        nodes: impl Iterator<Item = usize>,
        adjacency: bool,
    ) -> io::Result<()> {
        let adjacency = adjacency || self.reader.header.is_interleaved();
        let cache_limit = if adjacency {
            self.adjacency.cache_limit
        } else {
            self.vectors.cache_limit
        };
        if cache_limit == 0 {
            return Ok(());
        }
        let mut windows = Vec::with_capacity(MAX_IO_BATCH);
        for node in nodes {
            let window = self.node_window(node, adjacency)?;
            if adjacency
                && self
                    .reader
                    .hot_adjacency_window(window.offset, window.length)
                    .is_some()
            {
                continue;
            }
            if !windows
                .iter()
                .any(|w: &ReadWindow| w.offset == window.offset)
            {
                windows.push(window);
            }
        }
        if adjacency {
            self.adjacency
                .prefetch(self.reader, &windows, &mut self.stats)
        } else {
            self.vectors
                .prefetch(self.reader, &windows, &mut self.stats)
        }
    }

    fn adjacency_window(&mut self, node: usize) -> io::Result<(ReadWindow, usize)> {
        let locator = self.reader.adjacency_locator(node)?;
        let page = locator.page_index as usize;
        let section = self.reader.header.sections.adjacency;
        let planner = ReadWindowPlanner::new(self.reader.read_plan(), section);
        let window = planner
            .window_for_logical_page(page)
            .ok_or_else(|| invalid_data("range adjacency page is out of range"))?;
        let start =
            (section.offset + page as u64 * u64::from(DISKANN_PAGE_SIZE) - window.offset) as usize;
        if self
            .reader
            .hot_adjacency_window(window.offset, window.length)
            .is_none()
        {
            self.adjacency.load(self.reader, window, &mut self.stats)?;
            let bytes = self
                .adjacency
                .bytes()
                .get(start..start + DISKANN_PAGE_SIZE as usize)
                .ok_or_else(|| invalid_data("range adjacency page is truncated"))?;
            self.reader.validate_adjacency_page(page, bytes)?;
        }
        Ok((window, start + locator.byte_offset as usize))
    }

    fn neighbors(&mut self, node: usize, out: &mut Vec<u32>) -> io::Result<()> {
        let locator = self.reader.adjacency_locator(node)?;
        let (window, start) = self.adjacency_window(node)?;
        let payload = self
            .reader
            .hot_adjacency_window(window.offset, window.length)
            .unwrap_or_else(|| self.adjacency.bytes());
        let bytes = payload
            .get(start..)
            .ok_or_else(|| invalid_data("range adjacency is truncated"))?;
        decode_adjacency_list(bytes, locator.degree(), locator.encoding(), out)?;
        self.stats.nodes_expanded += 1;
        Ok(())
    }

    fn distance(&mut self, node: usize, query: &[f32]) -> io::Result<f32> {
        let record_size = self.reader.header.vector_record_size as usize;
        let encoding = self.reader.header.raw_vector_encoding();
        let metric = self.reader.header.metric_type();
        let kernel = selected_raw_vector_distance_kernel(query.len());
        let distance = if self.reader.header.is_interleaved() {
            let (window, end) = self.adjacency_window(node)?;
            let start = end
                .checked_sub(record_size)
                .ok_or_else(|| invalid_data("range vector offset underflows"))?;
            let payload = self
                .reader
                .hot_adjacency_window(window.offset, window.length)
                .unwrap_or_else(|| self.adjacency.bytes());
            let record = payload
                .get(start..end)
                .ok_or_else(|| invalid_data("range vector is truncated"))?;
            checked_distance(query, record, encoding, metric, kernel)?
        } else {
            let planner = VectorWindowPlanner::new(
                self.reader.read_plan(),
                self.reader.header.sections.vectors,
                record_size,
            )?;
            let window = planner
                .window_for_node(node)
                .ok_or_else(|| invalid_data("range vector is out of range"))?;
            self.vectors.load(self.reader, window, &mut self.stats)?;
            let record = planner.record(window, self.vectors.bytes(), node)?;
            checked_distance(query, record, encoding, metric, kernel)?
        };
        if !distance.is_finite() {
            return Err(invalid_data("non-finite DiskANN range distance"));
        }
        self.stats.vector_evaluations += 1;
        Ok(distance)
    }

    fn collect(
        &self,
        node: usize,
        distance: f32,
        params: DiskAnnRangeSearchParams,
        filter: Option<&RoaringTreemap>,
        rows: &mut Vec<(i64, f32)>,
    ) -> io::Result<()> {
        let id = self.reader.row_id(node)?;
        if params.band.admit_raw(distance)
            && filter.is_none_or(|f| id >= 0 && f.contains(id as u64))
        {
            if rows.len() == rows.capacity() {
                rows.try_reserve(1)
                    .map_err(|_| invalid_input("range result allocation failed"))?;
            }
            rows.push((id, distance));
        }
        Ok(())
    }

    fn run(
        &mut self,
        query: &[f32],
        params: DiskAnnRangeSearchParams,
        filter: Option<&RoaringTreemap>,
    ) -> io::Result<Vec<(i64, f32)>> {
        let n = self.reader.header.vector_count as usize;
        let l = params.starting_l.min(n);
        let mut rows = Vec::new();
        if n == 0 || params.band.is_empty() || filter.is_some_and(RoaringTreemap::is_empty) {
            return Ok(rows);
        }
        let window_bytes = self.reader.read_plan().window_bytes;
        let vector_bytes = window_bytes.max(self.reader.header.vector_record_size as usize);
        let mut fixed = window_bytes
            .saturating_add(vector_bytes)
            .saturating_add(query.len().saturating_mul(4));
        let upper = match params.band.raw_upper() {
            Bound::Finite(v) => Some(v),
            Bound::Unbounded => None,
        };
        // Zero cosine queries have no useful PQ navigation ordering. Scanning also
        // handles lower-only bands without inventing a finite sentinel radius.
        let exhaustive = params.exhaustive
            || upper.is_none()
            || (params.band.metric() == MetricType::Cosine && query.iter().all(|v| *v == 0.0));
        if !exhaustive {
            let pq = self.reader.pq()?;
            fixed = fixed
                .saturating_add(n.div_ceil(64).saturating_mul(8))
                .saturating_add(l.saturating_mul(6 * size_of::<SearchCandidate>()))
                .saturating_add((self.reader.header.max_degree as usize).saturating_mul(32))
                .saturating_add(pq.m().saturating_mul(pq.ksub()).saturating_mul(4));
        }
        if fixed > params.max_working_bytes {
            return Err(invalid_input(
                "DiskANN range working memory budget exceeded",
            ));
        }
        {
            let adjacency_bytes = self.reader.header.sections.adjacency.length as usize;
            let vector_bytes = self.reader.header.sections.vectors.length as usize;
            let overhead = adjacency_bytes
                .div_ceil(window_bytes)
                .saturating_add(vector_bytes.div_ceil(window_bytes))
                .saturating_mul(256);
            let allowance = adjacency_bytes
                .saturating_add(vector_bytes)
                .saturating_add(overhead)
                .min(8 * 1024 * 1024)
                .min(
                    params
                        .max_working_bytes
                        .saturating_sub(fixed)
                        .saturating_sub(PREFETCH_SCRATCH_BYTES)
                        / 2,
                );
            if self.reader.header.is_interleaved() {
                self.adjacency.cache_limit = allowance;
            } else {
                self.adjacency.cache_limit = adjacency_bytes
                    .saturating_add(adjacency_bytes.div_ceil(window_bytes).saturating_mul(256))
                    .min(allowance / 4);
                self.vectors.cache_limit = allowance - self.adjacency.cache_limit;
            }
            fixed += allowance;
            if allowance != 0 {
                fixed += PREFETCH_SCRATCH_BYTES;
            }
        }
        if exhaustive {
            self.stats.exhaustive = true;
            let batch = self.io_batch_size(false);
            let mut start = 0;
            while start < n {
                let mut nodes = [0; MAX_IO_BATCH];
                let mut count = 0;
                let mut end = start;
                let mut last_window = None;
                // A remote window can hold hundreds of short vectors. Group
                // distinct windows, keeping one eligible representative each.
                while end < n {
                    let id = self.reader.row_id(end)?;
                    if filter.is_some_and(|f| id < 0 || !f.contains(id as u64)) {
                        end += 1;
                        continue;
                    }
                    let window = self.node_window(end, false)?.offset;
                    if last_window != Some(window) {
                        if count == batch {
                            break;
                        }
                        nodes[count] = end;
                        count += 1;
                        last_window = Some(window);
                    }
                    end += 1;
                }
                self.prefetch_nodes(nodes[..count].iter().copied(), false)?;
                for node in start..end {
                    let id = self.reader.row_id(node)?;
                    if filter.is_some_and(|f| id < 0 || !f.contains(id as u64)) {
                        continue;
                    }
                    let distance = self.distance(node, query)?;
                    self.collect(node, distance, params, filter, &mut rows)?;
                }
                start = end;
            }
            return Ok(rows);
        }
        let mut seen = Vec::new();
        seen.try_reserve_exact(n.div_ceil(64))
            .map_err(|_| invalid_input("range visited allocation failed"))?;
        seen.resize(n.div_ceil(64), 0u64);
        let pq = self.reader.pq()?;
        let mut table = vec![0.0; pq.m() * pq.ksub()];
        pq.compute_distance_table(query, self.reader.header.metric_type(), &mut table);
        if table.iter().any(|v| !v.is_finite()) {
            return Err(invalid_data("non-finite range PQ table"));
        }
        let mut best = BinaryHeap::<SearchCandidate>::with_capacity(l);
        let mut frontier = BinaryHeap::<Reverse<SearchCandidate>>::with_capacity(
            l.saturating_mul(2).saturating_add(1),
        );
        let entry = self.reader.header.entry_node as usize;
        mark(&mut seen, entry);
        let estimate = self.reader.pq_distance(entry, &table)?;
        if !estimate.is_finite() {
            return Err(invalid_data("non-finite range PQ score"));
        }
        let candidate = SearchCandidate {
            node: entry,
            distance: self.distance(entry, query)?,
        };
        self.stats.pq_evaluations += 1;
        best.push(candidate);
        frontier.push(Reverse(candidate));
        let mut neighbors = Vec::with_capacity(self.reader.header.max_degree as usize);
        let mut scored = Vec::with_capacity(self.reader.header.max_degree as usize);
        while let Some(Reverse(candidate)) = frontier.pop() {
            if best.len() == l && best.peek().is_some_and(|worst| candidate > *worst) {
                break;
            }
            self.neighbors(candidate.node, &mut neighbors)?;
            neighbors.retain(|node| mark(&mut seen, *node as usize));
            score_pq_neighbors(
                &table,
                self.reader.pq_codes()?,
                self.reader.header.pq_m as usize,
                self.reader.header.pq_bits as usize,
                &neighbors,
                &mut scored,
            )?;
            sort_candidates(&mut scored);
            self.stats.pq_evaluations += scored.len();
            if scored.iter().any(|estimate| !estimate.distance.is_finite()) {
                return Err(invalid_data("non-finite range PQ score"));
            }
            for batch in scored.chunks(self.io_batch_size(false)) {
                self.prefetch_nodes(batch.iter().map(|c| c.node), false)?;
                for estimate in batch {
                    // PQ orders work but never discards a node before stored-vector
                    // scoring. Otherwise heavy compression can defeat the initial
                    // outer-hit fraction and silently suppress range expansion.
                    let candidate = SearchCandidate {
                        node: estimate.node,
                        distance: self.distance(estimate.node, query)?,
                    };
                    if best.len() < l || best.peek().is_some_and(|worst| candidate < *worst) {
                        if best.len() == l {
                            best.pop();
                        }
                        best.push(candidate);
                        frontier.push(Reverse(candidate));
                        if frontier.len() > l.saturating_mul(2) {
                            let worst = *best.peek().expect("nonempty range candidate heap");
                            frontier.retain(|Reverse(c)| *c <= worst);
                        }
                    }
                }
            }
        }
        // Rebuild visitation: the first stage marks discovered nodes, including
        // discarded/unexpanded nodes. They must be eligible in the range stage.
        seen.fill(0);
        let upper = upper.expect("finite upper in graph mode");
        let navigation_cut =
            f64::from(upper) + f64::from(params.range_slack - 1.0) * f64::from(upper).abs();
        let mut seeds = best.into_vec();
        seeds.sort_unstable_by_key(|c| c.node); // Coalesce raw-vector window accesses.
        let mut in_outer = 0;
        for candidate in &seeds {
            mark(&mut seen, candidate.node);
            let distance = candidate.distance;
            self.collect(candidate.node, distance, params, filter, &mut rows)?;
            if distance < upper {
                in_outer += 1;
            }
        }
        if in_outer < ((l as f32 * params.initial_slack) as usize) {
            return Ok(rows);
        }
        let mut queue = VecDeque::new();
        let max_queue = params.max_working_bytes.saturating_sub(fixed) / size_of::<usize>();
        for candidate in &seeds {
            if f64::from(candidate.distance) <= navigation_cut {
                enqueue(&mut queue, candidate.node, max_queue)?;
            }
        }
        self.stats.second_phase = !queue.is_empty();
        let beam = self
            .io_batch_size(true)
            .min(self.reader.read_plan().graph_beam_width);
        while !queue.is_empty() {
            let mut nodes = [0; MAX_IO_BATCH];
            let mut count = 0;
            while count < beam {
                let Some(node) = queue.pop_front() else {
                    break;
                };
                nodes[count] = node;
                count += 1;
            }
            self.prefetch_nodes(nodes[..count].iter().copied(), true)?;
            for &node in &nodes[..count] {
                self.neighbors(node, &mut neighbors)?;
                // Physical node order improves locality without imposing output order.
                neighbors.sort_unstable();
                neighbors.retain(|neighbor| mark(&mut seen, *neighbor as usize));
                for batch in neighbors.chunks(self.io_batch_size(false)) {
                    self.prefetch_nodes(batch.iter().map(|node| *node as usize), false)?;
                    for &neighbor in batch {
                        let neighbor = neighbor as usize;
                        let distance = self.distance(neighbor, query)?;
                        self.collect(neighbor, distance, params, filter, &mut rows)?;
                        if f64::from(distance) <= navigation_cut {
                            enqueue(&mut queue, neighbor, max_queue)?;
                        }
                    }
                }
            }
        }
        Ok(rows)
    }
}

fn checked_distance(
    query: &[f32],
    bytes: &[u8],
    encoding: DiskAnnRawVectorEncoding,
    metric: MetricType,
    kernel: RawVectorDistanceKernel,
) -> io::Result<f32> {
    if metric == MetricType::Cosine {
        let width = if encoding == DiskAnnRawVectorEncoding::F32 {
            4
        } else {
            2
        };
        if bytes.len()
            != query
                .len()
                .checked_mul(width)
                .ok_or_else(|| invalid_data("range vector size overflow"))?
        {
            return Err(invalid_data("invalid range vector record size"));
        }
        let (mut dot, mut qnorm, mut vnorm) = (0.0f32, 0.0f32, 0.0f32);
        for (&q, component) in query.iter().zip(bytes.chunks_exact(width)) {
            let value = if width == 4 {
                f32::from_le_bytes(component.try_into().expect("four-byte component"))
            } else {
                half::f16::from_bits(u16::from_le_bytes(
                    component.try_into().expect("two-byte component"),
                ))
                .to_f32()
            };
            dot += q * value;
            qnorm += q * q;
            vnorm += value * value;
        }
        if !dot.is_finite()
            || !qnorm.is_finite()
            || !vnorm.is_finite()
            || !(qnorm * vnorm).is_finite()
        {
            return Err(invalid_data("non-finite DiskANN range cosine arithmetic"));
        }
    }
    raw_vector_distance(query, bytes, encoding, metric, kernel)
}

fn mark(seen: &mut [u64], node: usize) -> bool {
    let mask = 1u64 << (node % 64);
    let word = &mut seen[node / 64];
    let fresh = *word & mask == 0;
    *word |= mask;
    fresh
}

fn enqueue(queue: &mut VecDeque<usize>, node: usize, limit: usize) -> io::Result<()> {
    if queue.len() == queue.capacity() {
        let desired = queue.capacity().saturating_mul(2).max(8).min(limit);
        if desired <= queue.capacity() {
            return Err(invalid_input(
                "DiskANN range working memory budget exceeded",
            ));
        }
        queue
            .try_reserve_exact(desired - queue.len())
            .map_err(|_| invalid_input("range frontier allocation failed"))?;
        if queue.capacity() > limit {
            return Err(invalid_input(
                "DiskANN range working memory budget exceeded",
            ));
        }
    }
    queue.push_back(node);
    Ok(())
}

impl<R: SeekRead> DiskAnnIndexReader<R> {
    pub fn range_search(
        &mut self,
        query: &[f32],
        params: impl Into<RangeSearchParams>,
    ) -> io::Result<RangeSearchResult> {
        self.range_search_batch_impl(query, 1, params, None)
    }

    pub fn range_search_with_roaring_filter(
        &mut self,
        query: &[f32],
        params: impl Into<RangeSearchParams>,
        filter: &[u8],
    ) -> io::Result<RangeSearchResult> {
        let filter = decode_roaring_filter(filter)?;
        self.range_search_batch_impl(query, 1, params, Some(&filter))
    }

    pub fn range_search_batch(
        &mut self,
        queries: &[f32],
        count: usize,
        params: impl Into<RangeSearchParams>,
    ) -> io::Result<RangeSearchResult> {
        self.range_search_batch_impl(queries, count, params, None)
    }

    pub fn range_search_batch_with_roaring_filter(
        &mut self,
        queries: &[f32],
        count: usize,
        params: impl Into<RangeSearchParams>,
        filter: &[u8],
    ) -> io::Result<RangeSearchResult> {
        let filter = decode_roaring_filter(filter)?;
        self.range_search_batch_impl(queries, count, params, Some(&filter))
    }

    pub(crate) fn range_search_batch_impl(
        &mut self,
        queries: &[f32],
        count: usize,
        params: impl Into<RangeSearchParams>,
        filter: Option<&RoaringTreemap>,
    ) -> io::Result<RangeSearchResult> {
        let d = self.header.dimension as usize;
        let params = params.into();
        params.validate_shape()?;
        if params.band().metric() != self.header.metric_type() {
            return Err(invalid_input("range band metric does not match index"));
        }
        if count == 0
            || count.checked_mul(d) != Some(queries.len())
            || queries.iter().any(|v| !v.is_finite())
        {
            return Err(invalid_input("invalid DiskANN range query shape or values"));
        }
        if !params.is_diskann() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "IVF range options cannot be used with a DiskANN reader",
            ));
        }
        let params = params.diskann_params();
        if self.header.metric_type() == MetricType::Cosine
            && d.saturating_mul(size_of::<f32>()) > params.max_working_bytes
        {
            return Err(invalid_input(
                "DiskANN range working memory budget exceeded",
            ));
        }
        let mut result = RangeResultBuilder::new(count);
        let mut stats = Vec::with_capacity(count);
        if !params.band.is_empty() && !filter.is_some_and(RoaringTreemap::is_empty) {
            self.ensure_resident()?;
        }
        for (index, query) in queries.chunks_exact(d).enumerate() {
            let query = prepare_range_queries(query, d, self.header.metric_type())?;
            let mut session = RangeSession {
                reader: self,
                adjacency: LocalWindow::default(),
                vectors: LocalWindow::default(),
                stats: DiskAnnRangeSearchStats::default(),
            };
            let rows = session.run(query.as_ref(), params, filter)?;
            result.record_scanned(index, session.stats.vector_evaluations);
            result.take_rows(index, rows);
            stats.push(session.stats);
        }
        Ok(result.build().with_diskann_stats(stats))
    }
}
