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

use crate::io::{ReadRequest, SeekRead, SeekReadCapabilities};
use std::io;
use std::sync::{Arc, Mutex};

/// The algorithm actually executed by a top-K call.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum SearchStrategy {
    #[default]
    NotExecuted = 0,
    Ivf = 1,
    DiskAnnGraph = 2,
    FilteredExhaustive = 3,
    GraphThenExhaustive = 4,
    Mixed = 5,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum SearchFallbackReason {
    #[default]
    None = 0,
    /// Automatic IVF expanded, or DiskANN fell back after graph filtering.
    InsufficientResults = 1,
}

/// Statistics for the last top-K call through `VectorIndexReader`.
///
/// A batch is one call: counters sum all queries, retries and worker sessions;
/// widths are maxima across queries. IVF candidates count selected list rows
/// before filtering (including repeated work in single-query retries). DiskANN
/// candidates count PQ distance evaluations, including exhaustive fallback work.
/// Reads count `SeekRead::pread` invocations, requested ranges and bytes, including
/// lazy initialization. They describe storage adapter requests, not physical I/O.
/// Cache counters count IVF payload entries or DiskANN adjacency/raw-vector
/// windows. Families without a payload cache report zero. Range searches,
/// warmup and optimization preserve this snapshot; failed top-K calls clear it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SearchStats {
    pub query_count: u64,
    /// 1 = IVF nprobe, 2 = DiskANN search-list size; 0 before a successful call.
    pub search_width: u64,
    pub initial_width: u64,
    pub effective_width: u64,
    pub candidate_count: u64,
    pub strategy: SearchStrategy,
    pub fallback_reason: SearchFallbackReason,
    pub read_calls: u64,
    pub read_ranges: u64,
    pub read_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

#[derive(Default)]
struct ObservationState {
    active: bool,
    stats: SearchStats,
}

/// Shared by DiskANN worker readers so their storage requests belong to the
/// parent call. No thread-local state: separate readers remain independent.
#[derive(Clone, Default)]
pub(crate) struct SearchObservation(Arc<Mutex<ObservationState>>);

impl SearchObservation {
    pub(crate) fn begin(&self, query_count: usize, diskann: bool) -> SearchCall {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.stats = SearchStats {
            query_count: query_count as u64,
            search_width: if diskann { 2 } else { 1 },
            strategy: if diskann {
                SearchStrategy::DiskAnnGraph
            } else {
                SearchStrategy::Ivf
            },
            ..SearchStats::default()
        };
        state.active = true;
        SearchCall(self.clone(), false)
    }

    pub(crate) fn finish(&self, succeeded: bool) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.active = false;
        if !succeeded {
            state.stats = SearchStats::default();
        }
    }

    pub(crate) fn snapshot(&self) -> SearchStats {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .stats
    }

    pub(crate) fn update(&self, update: impl FnOnce(&mut SearchStats)) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if state.active {
            update(&mut state.stats);
        }
    }

    pub(crate) fn record_ivf(&self, probes: &[usize], start: usize, counts: &[i32]) {
        self.update(|stats| {
            let width = probes.len() as u64;
            if stats.initial_width == 0 {
                stats.initial_width = width;
            }
            if start > 0 || width > stats.effective_width && stats.effective_width > 0 {
                stats.fallback_reason = SearchFallbackReason::InsufficientResults;
            }
            stats.effective_width = stats.effective_width.max(width);
            for &list in probes.iter().skip(start) {
                stats.candidate_count = stats.candidate_count.saturating_add(counts[list] as u64);
            }
        });
    }

    pub(crate) fn record_graph(&self, width: usize) {
        self.update(|stats| {
            stats.initial_width = stats.initial_width.max(width as u64);
            stats.effective_width = stats.effective_width.max(width as u64);
        });
    }
}

/// Clears a partial snapshot when a search unwinds through an FFI/JNI boundary.
pub(crate) struct SearchCall(SearchObservation, bool);

impl SearchCall {
    pub(crate) fn finish(mut self, succeeded: bool) {
        self.0.finish(succeeded);
        self.1 = true;
    }
}

impl Drop for SearchCall {
    fn drop(&mut self) {
        if !self.1 {
            self.0.finish(false);
        }
    }
}

pub(crate) struct ObservedReader<R> {
    inner: R,
    pub(crate) observation: SearchObservation,
}

impl<R> ObservedReader<R> {
    pub(crate) fn new(inner: R) -> Self {
        Self {
            inner,
            observation: SearchObservation::default(),
        }
    }
}

impl<R: SeekRead> SeekRead for ObservedReader<R> {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        if !ranges.is_empty() {
            self.observation.update(|stats| {
                stats.read_calls = stats.read_calls.saturating_add(1);
                stats.read_ranges = stats.read_ranges.saturating_add(ranges.len() as u64);
                for range in ranges.iter() {
                    stats.read_bytes = stats.read_bytes.saturating_add(range.buf.len() as u64);
                }
            });
        }
        self.inner.pread(ranges)
    }

    fn try_clone_reader(&self) -> io::Result<Option<Self>> {
        Ok(self.inner.try_clone_reader()?.map(|inner| Self {
            inner,
            observation: self.observation.clone(),
        }))
    }

    fn read_capabilities(&self) -> SeekReadCapabilities {
        self.inner.read_capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwinding_clears_partial_statistics_and_allows_the_next_call() {
        let observation = SearchObservation::default();
        let result = std::panic::catch_unwind(|| {
            let _call = observation.begin(3, false);
            observation.update(|stats| {
                stats.read_bytes = 42;
                panic!("failed while collecting statistics");
            });
        });
        assert!(result.is_err());
        assert_eq!(observation.snapshot(), SearchStats::default());
        observation.update(|stats| stats.read_bytes = 99);
        assert_eq!(observation.snapshot(), SearchStats::default());
        let call = observation.begin(1, false);
        observation.record_ivf(&[0], 0, &[7]);
        call.finish(true);
        assert_eq!(observation.snapshot().candidate_count, 7);
    }
}
