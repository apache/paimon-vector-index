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

use std::collections::HashMap;

pub(crate) struct TopKHeap {
    k: usize,
    data: Vec<(f32, i64)>,
    positions: HashMap<i64, usize>,
}

impl TopKHeap {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            data: Vec::with_capacity(k),
            positions: HashMap::with_capacity(k),
        }
    }

    pub(crate) fn push(&mut self, dist: f32, id: i64) {
        if self.k == 0 {
            return;
        }
        if let Some(&idx) = self.positions.get(&id) {
            if dist < self.data[idx].0 {
                self.data[idx].0 = dist;
            }
            return;
        }
        if self.data.len() < self.k {
            self.positions.insert(id, self.data.len());
            self.data.push((dist, id));
            return;
        }
        if let Some((worst_idx, _)) = self
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.0.total_cmp(&b.0))
        {
            if dist < self.data[worst_idx].0 {
                let old_id = self.data[worst_idx].1;
                self.positions.remove(&old_id);
                self.data[worst_idx] = (dist, id);
                self.positions.insert(id, worst_idx);
            }
        }
    }

    pub(crate) fn into_sorted(mut self) -> Vec<(f32, i64)> {
        self.data.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.data
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_heap_updates_duplicate_and_replaces_worst() {
        let mut heap = TopKHeap::new(2);

        heap.push(10.0, 7);
        heap.push(5.0, 8);
        heap.push(1.0, 7);
        heap.push(3.0, 9);

        assert_eq!(heap.into_sorted(), vec![(1.0, 7), (3.0, 9)]);
    }
}
