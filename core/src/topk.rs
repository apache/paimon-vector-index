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
                self.sift_down(idx);
            }
            return;
        }
        if self.data.len() < self.k {
            self.positions.insert(id, self.data.len());
            self.data.push((dist, id));
            self.sift_up(self.data.len() - 1);
            return;
        }
        if dist < self.data[0].0 {
            let old_id = self.data[0].1;
            self.positions.remove(&old_id);
            self.data[0] = (dist, id);
            self.positions.insert(id, 0);
            self.sift_down(0);
        }
    }

    pub(crate) fn into_sorted(mut self) -> Vec<(f32, i64)> {
        self.data.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.data
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.data[parent].0.total_cmp(&self.data[index].0).is_ge() {
                break;
            }
            self.swap_entries(parent, index);
            index = parent;
        }
    }

    fn sift_down(&mut self, mut index: usize) {
        loop {
            let left = index * 2 + 1;
            let right = left + 1;
            let mut worst = index;
            if left < self.data.len() && self.data[left].0.total_cmp(&self.data[worst].0).is_gt() {
                worst = left;
            }
            if right < self.data.len() && self.data[right].0.total_cmp(&self.data[worst].0).is_gt()
            {
                worst = right;
            }
            if worst == index {
                break;
            }
            self.swap_entries(index, worst);
            index = worst;
        }
    }

    fn swap_entries(&mut self, left: usize, right: usize) {
        self.data.swap(left, right);
        self.positions.insert(self.data[left].1, left);
        self.positions.insert(self.data[right].1, right);
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

    #[test]
    fn test_topk_heap_keeps_max_at_root_after_duplicate_update() {
        let mut heap = TopKHeap::new(3);
        heap.push(8.0, 1);
        heap.push(6.0, 2);
        heap.push(4.0, 3);
        heap.push(1.0, 1);
        heap.push(5.0, 4);

        assert_eq!(heap.into_sorted(), vec![(1.0, 1), (4.0, 3), (5.0, 4)]);
    }
}
