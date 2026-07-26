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

const EMPTY_KEY: u32 = u32::MAX;
const MIN_CAPACITY: usize = 8;

pub(crate) fn estimated_memory_bytes(expected_entries: usize, value_size: usize) -> Option<usize> {
    let capacity = expected_entries
        .checked_mul(2)?
        .max(MIN_CAPACITY)
        .checked_next_power_of_two()?;
    capacity
        .checked_mul(std::mem::size_of::<u32>().checked_add(value_size)?)?
        .checked_add(expected_entries.checked_mul(std::mem::size_of::<u32>())?)
}

pub(crate) struct SparseTable<V> {
    keys: Vec<u32>,
    values: Vec<V>,
    touched_slots: Vec<u32>,
}

impl<V: Copy + Default> SparseTable<V> {
    pub(crate) fn with_capacity(expected_entries: usize) -> Self {
        Self::try_with_capacity(expected_entries).expect("SparseTable allocation failed")
    }

    pub(crate) fn try_with_capacity(expected_entries: usize) -> Result<Self, ()> {
        let capacity = expected_entries
            .checked_mul(2)
            .ok_or(())?
            .max(MIN_CAPACITY)
            .checked_next_power_of_two()
            .ok_or(())?;
        let mut keys = Vec::new();
        keys.try_reserve_exact(capacity).map_err(|_| ())?;
        keys.resize(capacity, EMPTY_KEY);
        let mut values = Vec::new();
        values.try_reserve_exact(capacity).map_err(|_| ())?;
        values.resize(capacity, V::default());
        let mut touched_slots = Vec::new();
        touched_slots
            .try_reserve_exact(expected_entries)
            .map_err(|_| ())?;
        Ok(Self {
            keys,
            values,
            touched_slots,
        })
    }

    pub(crate) fn get(&self, key: u32) -> Option<&V> {
        assert_ne!(key, EMPTY_KEY, "u32::MAX is reserved by SparseTable");
        let slot = self.find_slot(key)?;
        (self.keys[slot] == key).then(|| &self.values[slot])
    }

    pub(crate) fn insert(&mut self, key: u32, value: V) -> Option<V> {
        assert_ne!(key, EMPTY_KEY, "u32::MAX is reserved by SparseTable");
        let slot = self.find_slot(key).expect("SparseTable must have a slot");
        if self.keys[slot] == key {
            return Some(std::mem::replace(&mut self.values[slot], value));
        }
        if self.touched_slots.len() + 1 > self.keys.len() * 7 / 10 {
            self.grow();
        }
        let slot = self
            .find_slot(key)
            .expect("grown SparseTable must have a slot");
        self.keys[slot] = key;
        self.values[slot] = value;
        self.touched_slots.push(slot as u32);
        None
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.touched_slots.len()
    }

    pub(crate) fn clear(&mut self) {
        for slot in self.touched_slots.drain(..) {
            self.keys[slot as usize] = EMPTY_KEY;
        }
    }

    pub(crate) fn entry_capacity(&self) -> usize {
        self.keys.len() * 7 / 10
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.keys.len()
    }

    fn find_slot(&self, key: u32) -> Option<usize> {
        let mask = self.keys.len() - 1;
        let mut slot = mix_u32(key) as usize & mask;
        for _ in 0..self.keys.len() {
            let stored = self.keys[slot];
            if stored == EMPTY_KEY || stored == key {
                return Some(slot);
            }
            slot = (slot + 1) & mask;
        }
        None
    }

    fn grow(&mut self) {
        let capacity = self
            .keys
            .len()
            .checked_mul(2)
            .expect("SparseTable capacity overflow");
        let old_keys = std::mem::replace(&mut self.keys, vec![EMPTY_KEY; capacity]);
        let old_values = std::mem::replace(&mut self.values, vec![V::default(); capacity]);
        let old_touched = std::mem::replace(
            &mut self.touched_slots,
            Vec::with_capacity(capacity.saturating_mul(7) / 10),
        );
        for old_slot in old_touched {
            let old_slot = old_slot as usize;
            let key = old_keys[old_slot];
            let slot = self
                .find_slot(key)
                .expect("empty grown SparseTable must have a slot");
            self.keys[slot] = key;
            self.values[slot] = old_values[old_slot];
            self.touched_slots.push(slot as u32);
        }
    }
}

impl<V: Copy + Default> Default for SparseTable<V> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

fn mix_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}

#[cfg(test)]
mod tests {
    use super::SparseTable;

    #[test]
    fn sparse_table_inserts_and_updates_values() {
        let mut table = SparseTable::with_capacity(2);

        assert_eq!(table.insert(7, 1u8), None);
        assert_eq!(table.get(7), Some(&1));
        assert_eq!(table.insert(7, 3), Some(1));
        assert_eq!(table.get(7), Some(&3));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn sparse_table_clear_reuses_allocated_storage() {
        let mut table = SparseTable::with_capacity(4);
        for key in 0..4 {
            table.insert(key, key);
        }
        let capacity = table.capacity();

        table.clear();

        assert_eq!(table.len(), 0);
        assert_eq!(table.get(2), None);
        assert_eq!(table.capacity(), capacity);
        assert_eq!(table.insert(11, 9), None);
        assert_eq!(table.get(11), Some(&9));
    }

    #[test]
    fn sparse_table_grows_past_the_initial_estimate() {
        let mut table = SparseTable::with_capacity(1);

        for key in 0..1_000 {
            assert_eq!(table.insert(key, key + 1), None);
        }

        assert_eq!(table.len(), 1_000);
        for key in 0..1_000 {
            assert_eq!(table.get(key), Some(&(key + 1)));
        }
    }
}
