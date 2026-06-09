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

use crate::hnsw::HnswGraph;
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
    let mut visited = Vec::new();
    let mut visit_mark = 1usize;
    let force_scan = filter
        .map(|f| count_filtered(lists, f) <= ef_search.max(k))
        .unwrap_or(false);

    for list in lists {
        if force_scan {
            scan_list(list, &mut heap);
            continue;
        }
        if let Some(graph) = list.graph {
            let local_results = graph.search_with_workspace(
                query,
                ef_search.max(k),
                ef_search.max(k),
                &mut visited,
                &mut visit_mark,
            );
            for (local_id, dist) in local_results {
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

fn count_filtered<P>(lists: &[HnswSearchList<'_, P>], filter: &dyn RowIdFilter) -> usize {
    lists
        .iter()
        .map(|list| list.ids.iter().filter(|&&id| filter.contains(id)).count())
        .sum()
}
