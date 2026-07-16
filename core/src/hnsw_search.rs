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

use crate::hnsw::{HnswGraph, HnswSearchWorkspace};
use crate::ivfpq::RowIdFilter;
use crate::topk::TopKHeap;

pub(crate) struct HnswSearchList<'a, P> {
    pub(crate) ids: &'a [i64],
    pub(crate) graph: Option<&'a HnswGraph>,
    pub(crate) payload: P,
}

pub(crate) fn search_hnsw_lists<'a, P, F>(
    query: &[f32],
    lists: &[HnswSearchList<'a, P>],
    k: usize,
    ef_search: usize,
    filter: Option<&dyn RowIdFilter>,
    mut scan_list: F,
) -> Vec<(f32, i64)>
where
    F: FnMut(&HnswSearchList<'a, P>, &mut TopKHeap),
{
    let mut heap = TopKHeap::new(k);
    let scan_threshold = ef_search.max(k);
    let mut workspace = HnswSearchWorkspace::new(scan_threshold);
    let force_scan = filter
        .map(|f| has_at_most_matching_ids(lists.iter().map(|list| list.ids), f, scan_threshold))
        .unwrap_or(false);

    for list in lists {
        if force_scan {
            scan_list(list, &mut heap);
            continue;
        }
        if let Some(graph) = list.graph {
            let local_results = graph.search_with_reusable_workspace(
                query,
                scan_threshold,
                scan_threshold,
                &mut workspace,
            );
            for &(local_id, dist) in local_results {
                let row_id = list.ids[local_id];
                if filter.map(|f| f.contains(row_id)).unwrap_or(true) {
                    heap.push(dist, row_id);
                }
            }
        } else {
            scan_list(list, &mut heap);
        }
    }

    if filter.is_some() && heap.len() < k && !force_scan {
        for list in lists {
            scan_list(list, &mut heap);
        }
    }

    heap.into_sorted()
}

pub(crate) fn has_at_most_matching_ids<'a>(
    list_ids: impl IntoIterator<Item = &'a [i64]>,
    filter: &dyn RowIdFilter,
    limit: usize,
) -> bool {
    let mut remaining = limit;
    for ids in list_ids {
        for &id in ids {
            if filter.contains(id) {
                if remaining == 0 {
                    return false;
                }
                remaining -= 1;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingFilter {
        calls: AtomicUsize,
        matches: fn(i64) -> bool,
    }

    impl CountingFilter {
        fn new(matches: fn(i64) -> bool) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                matches,
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl RowIdFilter for CountingFilter {
        fn contains(&self, id: i64) -> bool {
            self.calls.fetch_add(1, Ordering::Relaxed);
            (self.matches)(id)
        }
    }

    #[test]
    fn test_matching_id_count_at_limit_checks_all_ids() {
        let first_ids = [0, 1, 2, 3];
        let second_ids = [4, 5, 6, 7, 8, 9];
        let lists = [first_ids.as_slice(), second_ids.as_slice()];
        let filter = CountingFilter::new(|_| true);

        assert!(has_at_most_matching_ids(lists, &filter, 10));
        assert_eq!(filter.calls(), 10);
    }

    #[test]
    fn test_matching_id_count_stops_after_limit_is_exceeded() {
        let first_ids: Vec<i64> = (0..100).collect();
        let second_ids: Vec<i64> = (100..200).collect();
        let lists = [first_ids.as_slice(), second_ids.as_slice()];
        let filter = CountingFilter::new(|id| id % 2 == 0);

        assert!(!has_at_most_matching_ids(lists, &filter, 10));
        assert_eq!(filter.calls(), 21);
    }

    #[test]
    fn test_matching_id_count_with_zero_limit() {
        let ids = [1, 2, 3];
        let lists = [ids.as_slice()];
        let filter = CountingFilter::new(|id| id == 2);

        assert!(!has_at_most_matching_ids(lists, &filter, 0));
        assert_eq!(filter.calls(), 2);
    }
}
