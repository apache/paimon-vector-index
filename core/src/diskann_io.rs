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

//! DiskANN v1 persistence and readers.
//!
//! The normative byte layout is documented in the repository's
//! [storage-format specification](https://github.com/apache/paimon-vector-index/blob/main/core/STORAGE_FORMAT.md#diskann-v1).

use crate::diskann::{
    validate_diskann_format_configuration, DiskAnnBuildParams, DiskAnnBuildStats, DiskAnnIndex,
    DiskAnnRawVectorEncoding, DiskAnnStorageLayout, PreparedDiskAnn,
    DISKANN_ADJACENCY_LOCATOR_BLOCK_NODES as ADJACENCY_LOCATOR_BLOCK_NODES,
    DISKANN_ADJACENCY_LOCATOR_NODE_BYTES,
};
use crate::diskann_search::{DiskAnnQueryScratch, DiskAnnSearchStats};
use crate::distance::MetricType;
use crate::io::{ReadRequest, SeekRead, SeekReadCapabilities, SeekWrite};
use crate::pq::ProductQuantizer;
use crate::read_options::{
    ReadPlan, ResolvedVectorIndexReaderOptions, StorageProfile, VectorIndexReaderOptions,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io;
use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub const DISKANN_MAGIC: u32 = 0x4E4E4144; // "DANN"
pub const DISKANN_VERSION: u32 = 1;
pub const DISKANN_HEADER_SIZE: usize = 256;
pub const DISKANN_PAGE_SIZE: u32 = 4096;
const FLAG_BFS_LAYOUT: u32 = 1 << 0;
const FLAG_SEPARATE_ADJACENCY_AND_VECTORS: u32 = 1 << 1;
const FLAG_ADAPTIVE_ADJACENCY: u32 = 1 << 2;
const FLAG_PQ_CODES: u32 = 1 << 3;
const FLAG_ROW_ID_ORDER: u32 = 1 << 4;
const FLAG_INTERLEAVED_ADJACENCY_AND_VECTORS: u32 = 1 << 5;
pub const DISKANN_REQUIRED_FLAGS: u32 =
    FLAG_BFS_LAYOUT | FLAG_ADAPTIVE_ADJACENCY | FLAG_PQ_CODES | FLAG_ROW_ID_ORDER;
const DISKANN_SUPPORTED_FLAGS: u32 = DISKANN_REQUIRED_FLAGS
    | FLAG_SEPARATE_ADJACENCY_AND_VECTORS
    | FLAG_INTERLEAVED_ADJACENCY_AND_VECTORS;
const SECTION_COUNT: usize = 7;
const ADJACENCY_LOCATOR_SIZE: u32 = DISKANN_ADJACENCY_LOCATOR_NODE_BYTES as u32;
const ADJACENCY_LOCATOR_ENCODING: u32 = 3;
const ADJACENCY_LOCATOR_BLOCK_BASE_SIZE: usize = size_of::<u64>();
const ADJACENCY_RAW_U32_FLAG: u16 = 1 << 15;
const ADJACENCY_DEGREE_MASK: u16 = ADJACENCY_RAW_U32_FLAG - 1;
const ROW_ID_SECTION_HEADER_SIZE: usize = 32;
const ROW_ID_ENCODING_RAW_I64: u32 = 0;
const ROW_ID_ENCODING_FOR_BITPACK: u32 = 1;
const PQ_CODEBOOK_MAGIC: u32 = 0x3151_5044; // "DPQ1"
const PQ_CODEBOOK_VERSION: u32 = 1;
const PQ_CODEBOOK_HEADER_SIZE: usize = 32;
const DISKANN_WRITE_BUFFER_SIZE: usize = 1024 * 1024;
const DISKANN_RESIDENT_DECODE_BUFFER_SIZE: usize = 1024 * 1024;
const DISKANN_ADJACENCY_PRELOAD_ALIGNMENT: usize = 64 * 1024;
const AUTO_PROFILE_MEMORY_LATENCY_THRESHOLD: Duration = Duration::from_micros(50);
const AUTO_PROFILE_LOCAL_LATENCY_THRESHOLD: Duration = Duration::from_micros(750);
const AUTO_PROFILE_REMOTE_LATENCY_THRESHOLD: Duration = Duration::from_millis(3);
const AUTO_PROFILE_PROBE_COUNT: usize = 3;
const AUTO_PROFILE_PROBE_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionRange {
    pub offset: u64,
    pub length: u64,
}

impl SectionRange {
    pub const fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskAnnSections {
    pub codebook: SectionRange,
    pub row_ids: SectionRange,
    pub pq_codes: SectionRange,
    pub row_id_order: SectionRange,
    pub adjacency_index: SectionRange,
    pub adjacency: SectionRange,
    pub vectors: SectionRange,
}

impl DiskAnnSections {
    fn from_array(sections: [SectionRange; SECTION_COUNT]) -> Self {
        Self {
            codebook: sections[0],
            row_ids: sections[1],
            pq_codes: sections[2],
            row_id_order: sections[3],
            adjacency_index: sections[4],
            adjacency: sections[5],
            vectors: sections[6],
        }
    }

    fn as_array(self) -> [SectionRange; SECTION_COUNT] {
        [
            self.codebook,
            self.row_ids,
            self.pq_codes,
            self.row_id_order,
            self.adjacency_index,
            self.adjacency,
            self.vectors,
        ]
    }
}

impl Index<usize> for DiskAnnSections {
    type Output = SectionRange;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.codebook,
            1 => &self.row_ids,
            2 => &self.pq_codes,
            3 => &self.row_id_order,
            4 => &self.adjacency_index,
            5 => &self.adjacency,
            6 => &self.vectors,
            _ => panic!("DiskANN section index {index} is out of range"),
        }
    }
}

impl IndexMut<usize> for DiskAnnSections {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.codebook,
            1 => &mut self.row_ids,
            2 => &mut self.pq_codes,
            3 => &mut self.row_id_order,
            4 => &mut self.adjacency_index,
            5 => &mut self.adjacency,
            6 => &mut self.vectors,
            _ => panic!("DiskANN section index {index} is out of range"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdjacencyLocator {
    pub page_index: u32,
    pub byte_offset: u16,
    degree_and_flags: u16,
}

impl AdjacencyLocator {
    fn new(
        page_index: u32,
        byte_offset: u16,
        degree: usize,
        encoding: AdjacencyListEncoding,
    ) -> io::Result<Self> {
        let degree = u16::try_from(degree)
            .map_err(|_| invalid_input("DiskANN adjacency degree exceeds u16"))?;
        if degree > ADJACENCY_DEGREE_MASK {
            return Err(invalid_input(
                "DiskANN adjacency degree exceeds locator capacity",
            ));
        }
        let encoding_flag = match encoding {
            AdjacencyListEncoding::DeltaVarint => 0,
            AdjacencyListEncoding::RawU32 => ADJACENCY_RAW_U32_FLAG,
        };
        Ok(Self {
            page_index,
            byte_offset,
            degree_and_flags: degree | encoding_flag,
        })
    }

    pub(crate) fn degree(self) -> usize {
        usize::from(self.degree_and_flags & ADJACENCY_DEGREE_MASK)
    }

    pub(crate) fn encoding(self) -> AdjacencyListEncoding {
        if self.degree_and_flags & ADJACENCY_RAW_U32_FLAG == 0 {
            AdjacencyListEncoding::DeltaVarint
        } else {
            AdjacencyListEncoding::RawU32
        }
    }
}

#[derive(Debug)]
struct AdjacencyIndex {
    block_offsets: Box<[u64]>,
    relative_offsets: Box<[u16]>,
    degree_and_flags: Box<[u16]>,
}

impl AdjacencyIndex {
    #[cfg(test)]
    fn from_locators(locators: &[AdjacencyLocator]) -> io::Result<Self> {
        let block_count = locators.len().div_ceil(ADJACENCY_LOCATOR_BLOCK_NODES);
        let mut block_offsets = Vec::new();
        block_offsets
            .try_reserve_exact(block_count)
            .map_err(|_| invalid_data("DiskANN adjacency block-offset allocation failed"))?;
        let mut relative_offsets = Vec::new();
        relative_offsets
            .try_reserve_exact(locators.len())
            .map_err(|_| invalid_data("DiskANN adjacency relative-offset allocation failed"))?;
        let mut degree_and_flags = Vec::new();
        degree_and_flags
            .try_reserve_exact(locators.len())
            .map_err(|_| invalid_data("DiskANN adjacency metadata allocation failed"))?;

        for (node, locator) in locators.iter().copied().enumerate() {
            let absolute_offset = adjacency_locator_absolute_offset(locator)?;
            if node.is_multiple_of(ADJACENCY_LOCATOR_BLOCK_NODES) {
                block_offsets.push(absolute_offset);
            }
            let block_offset = *block_offsets
                .last()
                .expect("each adjacency locator belongs to a block");
            let relative_offset = absolute_offset
                .checked_sub(block_offset)
                .and_then(|offset| u16::try_from(offset).ok())
                .ok_or_else(|| {
                    invalid_data("DiskANN adjacency locator exceeds its block offset range")
                })?;
            relative_offsets.push(relative_offset);
            degree_and_flags.push(locator.degree_and_flags);
        }
        Ok(Self {
            block_offsets: block_offsets.into_boxed_slice(),
            relative_offsets: relative_offsets.into_boxed_slice(),
            degree_and_flags: degree_and_flags.into_boxed_slice(),
        })
    }

    fn len(&self) -> usize {
        self.relative_offsets.len()
    }

    fn locator(&self, node: usize) -> Option<AdjacencyLocator> {
        let relative_offset = u64::from(*self.relative_offsets.get(node)?);
        let degree_and_flags = *self.degree_and_flags.get(node)?;
        let block_offset = *self
            .block_offsets
            .get(node / ADJACENCY_LOCATOR_BLOCK_NODES)?;
        let absolute_offset = block_offset.checked_add(relative_offset)?;
        let page_index = u32::try_from(absolute_offset / u64::from(DISKANN_PAGE_SIZE)).ok()?;
        let byte_offset = u16::try_from(absolute_offset % u64::from(DISKANN_PAGE_SIZE)).ok()?;
        Some(AdjacencyLocator {
            page_index,
            byte_offset,
            degree_and_flags,
        })
    }

    fn partition_point(&self, mut predicate: impl FnMut(AdjacencyLocator) -> bool) -> usize {
        let mut left = 0;
        let mut right = self.len();
        while left < right {
            let middle = left + (right - left) / 2;
            let locator = self
                .locator(middle)
                .expect("validated DiskANN adjacency index");
            if predicate(locator) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        left
    }
}

fn adjacency_locator_absolute_offset(locator: AdjacencyLocator) -> io::Result<u64> {
    u64::from(locator.page_index)
        .checked_mul(u64::from(DISKANN_PAGE_SIZE))
        .and_then(|offset| offset.checked_add(u64::from(locator.byte_offset)))
        .ok_or_else(|| invalid_data("DiskANN adjacency locator offset overflows"))
}

fn adjacency_index_serialized_len(vector_count: usize) -> io::Result<u64> {
    let block_count = vector_count.div_ceil(ADJACENCY_LOCATOR_BLOCK_NODES);
    let block_bytes = block_count
        .checked_mul(ADJACENCY_LOCATOR_BLOCK_BASE_SIZE)
        .ok_or_else(|| invalid_input("DiskANN adjacency block-offset size overflows usize"))?;
    let locator_bytes = vector_count
        .checked_mul(ADJACENCY_LOCATOR_SIZE as usize)
        .ok_or_else(|| invalid_input("DiskANN adjacency locator size overflows usize"))?;
    u64::try_from(
        block_bytes
            .checked_add(locator_bytes)
            .ok_or_else(|| invalid_input("DiskANN adjacency index size overflows usize"))?,
    )
    .map_err(|_| invalid_input("DiskANN adjacency index size exceeds u64"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdjacencyListEncoding {
    DeltaVarint,
    RawU32,
}

fn encode_adjacency_list(
    neighbors: &[u32],
    encoded: &mut Vec<u8>,
) -> io::Result<AdjacencyListEncoding> {
    encoded.clear();
    let (encoding, encoded_len) = plan_adjacency_list(neighbors)?;
    encoded
        .try_reserve_exact(encoded_len)
        .map_err(|_| invalid_input("DiskANN adjacency allocation failed"))?;
    match encoding {
        AdjacencyListEncoding::DeltaVarint => {
            let mut previous = 0u32;
            for &neighbor in neighbors {
                append_u32_varint(encoded, neighbor - previous);
                previous = neighbor;
            }
        }
        AdjacencyListEncoding::RawU32 => {
            for &neighbor in neighbors {
                encoded.extend_from_slice(&neighbor.to_le_bytes());
            }
        }
    }
    Ok(encoding)
}

fn plan_adjacency_list(neighbors: &[u32]) -> io::Result<(AdjacencyListEncoding, usize)> {
    let raw_len = neighbors
        .len()
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| invalid_input("DiskANN adjacency list size overflows usize"))?;
    let delta_len = adjacency_delta_varint_len(neighbors)
        .ok_or_else(|| invalid_input("DiskANN adjacency neighbors must be strictly increasing"))?;
    if neighbors.is_empty() || delta_len < raw_len {
        return Ok((AdjacencyListEncoding::DeltaVarint, delta_len));
    }
    Ok((AdjacencyListEncoding::RawU32, raw_len))
}

fn adjacency_delta_varint_len(neighbors: &[u32]) -> Option<usize> {
    let mut previous = 0u32;
    let mut encoded_len = 0usize;
    for (index, &neighbor) in neighbors.iter().enumerate() {
        if index != 0 && neighbor <= previous {
            return None;
        }
        encoded_len = encoded_len.checked_add(u32_varint_len(neighbor - previous))?;
        previous = neighbor;
    }
    Some(encoded_len)
}

fn u32_varint_len(value: u32) -> usize {
    let significant_bits = (u32::BITS - value.leading_zeros()).max(1);
    significant_bits.div_ceil(7) as usize
}

pub(crate) fn decode_adjacency_list(
    bytes: &[u8],
    degree: usize,
    encoding: AdjacencyListEncoding,
    neighbors: &mut Vec<u32>,
) -> io::Result<usize> {
    neighbors.clear();
    neighbors
        .try_reserve(degree)
        .map_err(|_| invalid_data("DiskANN adjacency decode allocation failed"))?;
    match encoding {
        AdjacencyListEncoding::DeltaVarint => {
            let mut position = 0usize;
            let mut previous = 0u32;
            for _ in 0..degree {
                let delta = read_u32_varint(bytes, &mut position)?;
                let neighbor = previous
                    .checked_add(delta)
                    .ok_or_else(|| invalid_data("DiskANN adjacency delta overflows u32"))?;
                neighbors.push(neighbor);
                previous = neighbor;
            }
            Ok(position)
        }
        AdjacencyListEncoding::RawU32 => {
            let encoded_len = degree
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| invalid_data("DiskANN raw adjacency size overflows usize"))?;
            let encoded = bytes
                .get(..encoded_len)
                .ok_or_else(|| invalid_data("DiskANN raw adjacency list is truncated"))?;
            neighbors.extend(encoded.chunks_exact(4).map(|value| {
                u32::from_le_bytes(value.try_into().expect("fixed adjacency neighbor"))
            }));
            Ok(encoded_len)
        }
    }
}

fn append_u32_varint(encoded: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        encoded.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    encoded.push(value as u8);
}

fn read_u32_varint(bytes: &[u8], position: &mut usize) -> io::Result<u32> {
    let mut value = 0u32;
    let start = *position;
    for shift in (0..=28).step_by(7) {
        let byte = *bytes
            .get(*position)
            .ok_or_else(|| invalid_data("DiskANN adjacency varint is truncated"))?;
        *position += 1;
        if shift == 28 && byte > 0x0f {
            return Err(invalid_data("DiskANN adjacency varint exceeds u32"));
        }
        value |= u32::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            if *position - start > 1 && byte == 0 {
                return Err(invalid_data("DiskANN adjacency varint is not canonical"));
            }
            return Ok(value);
        }
    }
    Err(invalid_data("DiskANN adjacency varint exceeds five bytes"))
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiskAnnHeader {
    pub flags: u32,
    pub dimension: u32,
    pub metric: u32,
    pub vector_count: u64,
    pub entry_node: u32,
    pub max_degree: u32,
    pub build_search_list_size: u32,
    pub alpha: f32,
    pub seed: u64,
    pub pq_m: u32,
    pub pq_bits: u32,
    pub page_size: u32,
    pub adjacency_locator_size: u32,
    pub adjacency_locator_encoding: u32,
    pub raw_vector_encoding: u32,
    pub vector_record_size: u32,
    pub file_len: u64,
    pub sections: DiskAnnSections,
}

pub struct DiskAnnIndexReader<R: SeekRead> {
    reader: R,
    pub header: DiskAnnHeader,
    resident: Option<Arc<DiskAnnResidentData>>,
    options: ResolvedVectorIndexReaderOptions,
    automatic_cache_budgets: bool,
    read_capabilities: SeekReadCapabilities,
    effective_storage_profile: StorageProfile,
    auto_storage_profile_probe_time: Option<Duration>,
    hot_adjacency: Arc<[u8]>,
    row_id_order: Arc<Mutex<RowIdOrderState>>,
    pub(crate) query_scratch: Box<DiskAnnQueryScratch>,
    pub(crate) last_search_stats: DiskAnnSearchStats,
    pub(crate) batch_workers: Vec<DiskAnnIndexReader<R>>,
    pub(crate) calibrated_l_search: Option<usize>,
}

struct DiskAnnResidentData {
    pq: ProductQuantizer,
    row_ids: RowIdStorage,
    pq_codes: Vec<u8>,
    adjacency_index: AdjacencyIndex,
    adjacency_validation: AdjacencyValidationCache,
    adjacency_cache: SharedWindowCache,
    raw_vector_cache: SharedWindowCache,
}

#[derive(Clone, Copy, Default)]
struct OffsetLruLink {
    older: Option<u64>,
    newer: Option<u64>,
}

#[derive(Default)]
pub(crate) struct OffsetLru {
    links: HashMap<u64, OffsetLruLink>,
    oldest: Option<u64>,
    newest: Option<u64>,
}

impl OffsetLru {
    pub(crate) fn touch(&mut self, offset: u64) {
        if self.newest == Some(offset) {
            return;
        }
        let already_present = match self.links.entry(offset) {
            std::collections::hash_map::Entry::Occupied(_) => true,
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(OffsetLruLink::default());
                false
            }
        };
        if already_present {
            self.detach(offset);
        }
        let older = self.newest;
        if let Some(older) = older {
            self.links
                .get_mut(&older)
                .expect("DiskANN LRU newest offset must exist")
                .newer = Some(offset);
        } else {
            self.oldest = Some(offset);
        }
        let link = self
            .links
            .get_mut(&offset)
            .expect("DiskANN LRU touched offset must exist");
        link.older = older;
        link.newer = None;
        self.newest = Some(offset);
    }

    pub(crate) fn remove(&mut self, offset: u64) {
        if self.links.contains_key(&offset) {
            self.detach(offset);
            self.links.remove(&offset);
        }
    }

    pub(crate) fn pop_oldest(&mut self) -> Option<u64> {
        let offset = self.oldest?;
        self.remove(offset);
        Some(offset)
    }

    pub(crate) fn clear(&mut self) {
        self.links.clear();
        self.oldest = None;
        self.newest = None;
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.links.len()
    }

    #[cfg(test)]
    pub(crate) fn oldest_offsets(&self) -> Vec<u64> {
        let mut offsets = Vec::with_capacity(self.links.len());
        let mut current = self.oldest;
        while let Some(offset) = current {
            offsets.push(offset);
            current = self.links.get(&offset).and_then(|link| link.newer);
        }
        debug_assert_eq!(offsets.len(), self.links.len());
        offsets
    }

    fn detach(&mut self, offset: u64) {
        let link = *self
            .links
            .get(&offset)
            .expect("DiskANN LRU detached offset must exist");
        if let Some(older) = link.older {
            self.links
                .get_mut(&older)
                .expect("DiskANN LRU older offset must exist")
                .newer = link.newer;
        } else {
            self.oldest = link.newer;
        }
        if let Some(newer) = link.newer {
            self.links
                .get_mut(&newer)
                .expect("DiskANN LRU newer offset must exist")
                .older = link.older;
        } else {
            self.newest = link.older;
        }
    }
}

pub(crate) enum SharedWindowCacheLookup {
    Hit(Arc<Vec<u8>>),
    Reserved,
    Loading,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CacheLockMetrics {
    pub(crate) acquisitions: usize,
    pub(crate) wait_nanos: u64,
}

struct SharedWindowCacheState {
    entries: HashMap<u64, Arc<Vec<u8>>>,
    loading: HashSet<u64>,
    recency: OffsetLru,
    retained_bytes: usize,
}

struct SharedWindowCacheShard {
    capacity_bytes: AtomicUsize,
    state: Mutex<SharedWindowCacheState>,
    waiters: Condvar,
}

const SHARED_WINDOW_CACHE_SHARDS: usize = 16;
const MAX_SHARED_READ_WINDOW_BYTES: usize = 64 * 1024;

pub(crate) struct SharedWindowCache {
    shards: Box<[SharedWindowCacheShard]>,
}

impl SharedWindowCache {
    fn new(capacity_bytes: usize) -> Self {
        Self::new_with_max_shards(capacity_bytes, SHARED_WINDOW_CACHE_SHARDS)
    }

    fn new_with_max_shards(capacity_bytes: usize, max_shards: usize) -> Self {
        let max_shards = max_shards.max(1);
        let shard_count = if capacity_bytes >= max_shards * MAX_SHARED_READ_WINDOW_BYTES {
            max_shards
        } else {
            1
        };
        let base_capacity = capacity_bytes / shard_count;
        let remainder = capacity_bytes % shard_count;
        let shards = (0..shard_count)
            .map(|shard| SharedWindowCacheShard {
                capacity_bytes: AtomicUsize::new(base_capacity + usize::from(shard < remainder)),
                state: Mutex::new(SharedWindowCacheState {
                    entries: HashMap::new(),
                    loading: HashSet::new(),
                    recency: OffsetLru::default(),
                    retained_bytes: 0,
                }),
                waiters: Condvar::new(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { shards }
    }

    fn shard_index(&self, offset: u64) -> usize {
        let page = offset / u64::from(DISKANN_PAGE_SIZE);
        let mut mixed = page.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((mixed ^ (mixed >> 31)) as usize) % self.shards.len()
    }

    fn shard(&self, offset: u64) -> &SharedWindowCacheShard {
        &self.shards[self.shard_index(offset)]
    }

    #[cfg(test)]
    fn shard_count(&self) -> usize {
        self.shards.len()
    }

    #[cfg(test)]
    fn total_capacity(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.capacity_bytes.load(AtomicOrdering::Relaxed))
            .sum()
    }

    fn add_lock_metrics(total: &mut CacheLockMetrics, metrics: CacheLockMetrics) {
        total.acquisitions = total.acquisitions.saturating_add(metrics.acquisitions);
        total.wait_nanos = total.wait_nanos.saturating_add(metrics.wait_nanos);
    }

    fn set_total_capacity(&self, capacity_bytes: usize) -> io::Result<()> {
        let base_capacity = capacity_bytes / self.shards.len();
        let remainder = capacity_bytes % self.shards.len();
        for (shard_index, shard) in self.shards.iter().enumerate() {
            let shard_capacity = base_capacity + usize::from(shard_index < remainder);
            shard
                .capacity_bytes
                .store(shard_capacity, AtomicOrdering::Relaxed);
            let (mut state, _) = Self::lock_state(shard)?;
            while state.retained_bytes > shard_capacity {
                let Some(oldest) = state.recency.pop_oldest() else {
                    break;
                };
                if let Some(evicted) = state.entries.remove(&oldest) {
                    state.retained_bytes = state.retained_bytes.saturating_sub(evicted.capacity());
                }
            }
        }
        Ok(())
    }

    fn lock_state(
        shard: &SharedWindowCacheShard,
    ) -> io::Result<(MutexGuard<'_, SharedWindowCacheState>, CacheLockMetrics)> {
        let started = Instant::now();
        let state = shard
            .state
            .lock()
            .map_err(|_| invalid_data("DiskANN shared window cache state is poisoned"))?;
        Ok((
            state,
            CacheLockMetrics {
                acquisitions: 1,
                wait_nanos: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            },
        ))
    }

    fn remove_loading(&self, offsets: &[u64], shard_index: usize) -> io::Result<CacheLockMetrics> {
        let shard = &self.shards[shard_index];
        let (mut state, metrics) = Self::lock_state(shard)?;
        for offset in offsets {
            if self.shard_index(*offset) == shard_index {
                state.loading.remove(offset);
            }
        }
        shard.waiters.notify_all();
        Ok(metrics)
    }

    pub(crate) fn lookup_or_reserve(
        &self,
        offset: u64,
        length: usize,
    ) -> io::Result<(SharedWindowCacheLookup, CacheLockMetrics)> {
        let shard = self.shard(offset);
        let (mut state, metrics) = Self::lock_state(shard)?;
        if let Some(payload) = state.entries.get(&offset).cloned() {
            if payload.len() == length {
                state.recency.touch(offset);
                return Ok((SharedWindowCacheLookup::Hit(payload), metrics));
            }
            state.entries.remove(&offset);
            state.recency.remove(offset);
            state.retained_bytes = state.retained_bytes.saturating_sub(payload.capacity());
        }
        if state.loading.contains(&offset) {
            return Ok((SharedWindowCacheLookup::Loading, metrics));
        }
        state.loading.insert(offset);
        Ok((SharedWindowCacheLookup::Reserved, metrics))
    }

    pub(crate) fn publish(
        &self,
        offset: u64,
        payload: Arc<Vec<u8>>,
    ) -> io::Result<(usize, CacheLockMetrics)> {
        let shard = self.shard(offset);
        let (mut state, metrics) = Self::lock_state(shard)?;
        state.loading.remove(&offset);
        if let Some(previous) = state.entries.insert(offset, Arc::clone(&payload)) {
            state.retained_bytes = state.retained_bytes.saturating_sub(previous.capacity());
        }
        state.retained_bytes = state.retained_bytes.saturating_add(payload.capacity());
        state.recency.touch(offset);
        let mut evictions = 0usize;
        let capacity_bytes = shard.capacity_bytes.load(AtomicOrdering::Relaxed);
        while state.retained_bytes > capacity_bytes {
            let Some(oldest) = state.recency.pop_oldest() else {
                break;
            };
            if let Some(evicted) = state.entries.remove(&oldest) {
                state.retained_bytes = state.retained_bytes.saturating_sub(evicted.capacity());
                evictions = evictions.saturating_add(1);
            }
        }
        shard.waiters.notify_all();
        Ok((evictions, metrics))
    }

    pub(crate) fn cancel(&self, offsets: &[u64]) -> io::Result<CacheLockMetrics> {
        let mut metrics = CacheLockMetrics::default();
        for shard_index in 0..self.shards.len() {
            if offsets
                .iter()
                .any(|offset| self.shard_index(*offset) == shard_index)
            {
                Self::add_lock_metrics(&mut metrics, self.remove_loading(offsets, shard_index)?);
            }
        }
        Ok(metrics)
    }

    pub(crate) fn wait_for(
        &self,
        offset: u64,
        length: usize,
    ) -> io::Result<(Option<Arc<Vec<u8>>>, CacheLockMetrics)> {
        let shard = self.shard(offset);
        let (mut state, metrics) = Self::lock_state(shard)?;
        while state.loading.contains(&offset) {
            state = shard
                .waiters
                .wait(state)
                .map_err(|_| invalid_data("DiskANN shared window cache state is poisoned"))?;
        }
        let payload = state
            .entries
            .get(&offset)
            .filter(|payload| payload.len() == length)
            .cloned();
        if payload.is_some() {
            state.recency.touch(offset);
        }
        Ok((payload, metrics))
    }
}

const ADJACENCY_PAGE_UNVALIDATED: u8 = 0;
const ADJACENCY_PAGE_VALIDATING: u8 = 1;
const ADJACENCY_PAGE_VALID: u8 = 2;
const ADJACENCY_PAGE_INVALID: u8 = 3;

#[derive(Clone)]
struct CachedValidationError {
    kind: io::ErrorKind,
    message: String,
}

impl CachedValidationError {
    fn from_error(error: &io::Error) -> Self {
        Self {
            kind: error.kind(),
            message: error.to_string(),
        }
    }

    fn to_error(&self) -> io::Error {
        io::Error::new(self.kind, self.message.clone())
    }
}

struct AdjacencyValidationCache {
    states: Box<[AtomicU8]>,
    errors: Mutex<HashMap<usize, CachedValidationError>>,
    wait_lock: Mutex<()>,
    waiters: Condvar,
}

impl AdjacencyValidationCache {
    fn new(page_count: usize) -> io::Result<Self> {
        let mut states = Vec::new();
        states
            .try_reserve_exact(page_count)
            .map_err(|_| invalid_data("DiskANN adjacency validation cache allocation failed"))?;
        states.extend((0..page_count).map(|_| AtomicU8::new(ADJACENCY_PAGE_UNVALIDATED)));
        Ok(Self {
            states: states.into_boxed_slice(),
            errors: Mutex::new(HashMap::new()),
            wait_lock: Mutex::new(()),
            waiters: Condvar::new(),
        })
    }

    fn get_or_validate(
        &self,
        page_index: usize,
        validate: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        let state = self
            .states
            .get(page_index)
            .ok_or_else(|| invalid_data("DiskANN adjacency page is out of range"))?;
        loop {
            match state.load(AtomicOrdering::Acquire) {
                ADJACENCY_PAGE_UNVALIDATED => {
                    if state
                        .compare_exchange(
                            ADJACENCY_PAGE_UNVALIDATED,
                            ADJACENCY_PAGE_VALIDATING,
                            AtomicOrdering::AcqRel,
                            AtomicOrdering::Acquire,
                        )
                        .is_err()
                    {
                        continue;
                    }
                    let mut claim = AdjacencyValidationClaim {
                        state,
                        wait_lock: &self.wait_lock,
                        waiters: &self.waiters,
                        published: false,
                    };
                    return match validate() {
                        Ok(()) => {
                            claim.publish(ADJACENCY_PAGE_VALID);
                            Ok(())
                        }
                        Err(error) => {
                            self.errors
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .insert(page_index, CachedValidationError::from_error(&error));
                            claim.publish(ADJACENCY_PAGE_INVALID);
                            Err(error)
                        }
                    };
                }
                ADJACENCY_PAGE_VALIDATING => {
                    let guard = self
                        .wait_lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let _guard = self
                        .waiters
                        .wait_while(guard, |_| {
                            state.load(AtomicOrdering::Acquire) == ADJACENCY_PAGE_VALIDATING
                        })
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                ADJACENCY_PAGE_VALID => return Ok(()),
                ADJACENCY_PAGE_INVALID => {
                    let errors = self
                        .errors
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    return Err(errors.get(&page_index).map_or_else(
                        || invalid_data("DiskANN adjacency page failed validation"),
                        CachedValidationError::to_error,
                    ));
                }
                _ => return Err(invalid_data("invalid DiskANN adjacency validation state")),
            }
        }
    }
}

struct AdjacencyValidationClaim<'a> {
    state: &'a AtomicU8,
    wait_lock: &'a Mutex<()>,
    waiters: &'a Condvar,
    published: bool,
}

impl AdjacencyValidationClaim<'_> {
    fn publish(&mut self, state: u8) {
        let _guard = self
            .wait_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.state.store(state, AtomicOrdering::Release);
        self.published = true;
        self.waiters.notify_all();
    }
}

impl Drop for AdjacencyValidationClaim<'_> {
    fn drop(&mut self) {
        if !self.published {
            let _guard = self
                .wait_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.state
                .store(ADJACENCY_PAGE_UNVALIDATED, AtomicOrdering::Release);
            self.waiters.notify_all();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RowIdStorage {
    Raw(Vec<i64>),
    ForBitPacked {
        base: i64,
        bit_width: u8,
        count: usize,
        payload: Vec<u8>,
    },
}

impl RowIdStorage {
    #[cfg(test)]
    fn encode(row_ids: Vec<i64>) -> io::Result<Self> {
        Self::encode_from_fn(row_ids.len(), |node| row_ids[node])
    }

    fn encode_from_fn(count: usize, row_id_at: impl Fn(usize) -> i64) -> io::Result<Self> {
        if count == 0 {
            return Err(invalid_input("DiskANN row IDs must not be empty"));
        }
        let mut base = row_id_at(0);
        let mut maximum = base;
        for node in 1..count {
            let row_id = row_id_at(node);
            base = base.min(row_id);
            maximum = maximum.max(row_id);
        }
        let span = u64::try_from(maximum as i128 - base as i128)
            .expect("the difference between two i64 values fits in u64");
        let bit_width = if span == 0 {
            0
        } else {
            (u64::BITS - span.leading_zeros()) as u8
        };
        if bit_width == u64::BITS as u8 {
            let mut row_ids = Vec::new();
            row_ids
                .try_reserve_exact(count)
                .map_err(|_| invalid_input("DiskANN raw row-ID allocation failed"))?;
            row_ids.extend((0..count).map(row_id_at));
            return Ok(Self::Raw(row_ids));
        }

        let payload_len = packed_row_id_payload_len(count, bit_width)?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(payload_len)
            .map_err(|_| invalid_input("DiskANN packed row-ID allocation failed"))?;
        payload.resize(payload_len, 0);
        for node in 0..count {
            let row_id = row_id_at(node);
            let delta = u64::try_from(row_id as i128 - base as i128)
                .expect("the row ID is not below the selected base");
            pack_row_id_delta(&mut payload, node, bit_width, delta)?;
        }
        Ok(Self::ForBitPacked {
            base,
            bit_width,
            count,
            payload,
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::Raw(row_ids) => row_ids.len(),
            Self::ForBitPacked { count, .. } => *count,
        }
    }

    #[cfg(test)]
    fn bit_width(&self) -> u8 {
        match self {
            Self::Raw(_) => u64::BITS as u8,
            Self::ForBitPacked { bit_width, .. } => *bit_width,
        }
    }

    fn get(&self, node: usize) -> Option<i64> {
        match self {
            Self::Raw(row_ids) => row_ids.get(node).copied(),
            Self::ForBitPacked {
                base,
                bit_width,
                count,
                payload,
            } => {
                if node >= *count {
                    return None;
                }
                let delta = unpack_row_id_delta(payload, node, *bit_width)?;
                i64::try_from(*base as i128 + delta as i128).ok()
            }
        }
    }

    fn try_for_each(
        &self,
        mut visitor: impl FnMut(usize, i64) -> io::Result<()>,
    ) -> io::Result<()> {
        match self {
            Self::Raw(row_ids) => {
                for (node, &row_id) in row_ids.iter().enumerate() {
                    visitor(node, row_id)?;
                }
            }
            Self::ForBitPacked {
                base,
                bit_width,
                count,
                payload,
            } => {
                let mut bit_offset = 0usize;
                for node in 0..*count {
                    let delta = unpack_row_id_delta_at_bit_offset(payload, bit_offset, *bit_width)
                        .ok_or_else(|| {
                            invalid_data("DiskANN packed row-ID payload is truncated")
                        })?;
                    let row_id = i64::try_from(*base as i128 + delta as i128)
                        .map_err(|_| invalid_data("DiskANN packed row ID overflows i64"))?;
                    visitor(node, row_id)?;
                    bit_offset = bit_offset
                        .checked_add(*bit_width as usize)
                        .ok_or_else(|| invalid_data("DiskANN packed row-ID offset overflows"))?;
                }
            }
        }
        Ok(())
    }

    fn payload_len(&self) -> io::Result<usize> {
        match self {
            Self::Raw(row_ids) => row_ids
                .len()
                .checked_mul(size_of::<i64>())
                .ok_or_else(|| invalid_input("DiskANN raw row-ID length overflows usize")),
            Self::ForBitPacked { payload, .. } => Ok(payload.len()),
        }
    }

    fn serialized_len(&self) -> io::Result<usize> {
        ROW_ID_SECTION_HEADER_SIZE
            .checked_add(self.payload_len()?)
            .ok_or_else(|| invalid_input("DiskANN row-ID section length overflows usize"))
    }

    fn section_header(&self) -> [u8; ROW_ID_SECTION_HEADER_SIZE] {
        let mut bytes = [0u8; ROW_ID_SECTION_HEADER_SIZE];
        match self {
            Self::Raw(row_ids) => {
                put_u32(&mut bytes, 0, ROW_ID_ENCODING_RAW_I64);
                put_u32(&mut bytes, 4, u64::BITS);
                put_u64(&mut bytes, 8, row_ids.len() as u64);
            }
            Self::ForBitPacked {
                base,
                bit_width,
                count,
                ..
            } => {
                put_u32(&mut bytes, 0, ROW_ID_ENCODING_FOR_BITPACK);
                put_u32(&mut bytes, 4, *bit_width as u32);
                put_u64(&mut bytes, 8, *count as u64);
                put_u64(&mut bytes, 16, *base as u64);
            }
        }
        bytes
    }
}

fn packed_row_id_payload_len(count: usize, bit_width: u8) -> io::Result<usize> {
    count
        .checked_mul(bit_width as usize)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or_else(|| invalid_input("DiskANN packed row-ID length overflows usize"))
}

fn raw_row_id_section_len(count: usize) -> io::Result<usize> {
    count
        .checked_mul(size_of::<i64>())
        .and_then(|payload| payload.checked_add(ROW_ID_SECTION_HEADER_SIZE))
        .ok_or_else(|| invalid_input("DiskANN raw row-ID section length overflows usize"))
}

fn pack_row_id_delta(payload: &mut [u8], node: usize, bit_width: u8, delta: u64) -> io::Result<()> {
    if bit_width == 0 {
        return Ok(());
    }
    let bit_offset = node
        .checked_mul(bit_width as usize)
        .ok_or_else(|| invalid_input("DiskANN packed row-ID offset overflows usize"))?;
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    let byte_count = (shift + bit_width as usize).div_ceil(8);
    let encoded = (delta as u128) << shift;
    let destination = payload
        .get_mut(byte_offset..byte_offset + byte_count)
        .ok_or_else(|| invalid_input("DiskANN packed row-ID payload is truncated"))?;
    for (index, byte) in destination.iter_mut().enumerate() {
        *byte |= (encoded >> (index * 8)) as u8;
    }
    Ok(())
}

fn unpack_row_id_delta(payload: &[u8], node: usize, bit_width: u8) -> Option<u64> {
    let bit_offset = node.checked_mul(bit_width as usize)?;
    unpack_row_id_delta_at_bit_offset(payload, bit_offset, bit_width)
}

fn unpack_row_id_delta_at_bit_offset(
    payload: &[u8],
    bit_offset: usize,
    bit_width: u8,
) -> Option<u64> {
    if bit_width == 0 {
        return Some(0);
    }
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    let byte_count = (shift + bit_width as usize).div_ceil(8);
    let source = payload.get(byte_offset..byte_offset + byte_count)?;
    let encoded = source
        .iter()
        .enumerate()
        .fold(0u128, |value, (index, &byte)| {
            value | ((byte as u128) << (index * 8))
        });
    let mask = (1u128 << bit_width) - 1;
    Some(((encoded >> shift) & mask) as u64)
}

#[derive(Debug, Clone, Copy)]
struct RowIdSectionHeader {
    encoding: u32,
    bit_width: u8,
    count: usize,
    base: i64,
}

fn decode_row_id_section_header(
    bytes: &[u8],
    section_len: usize,
    expected_count: usize,
) -> io::Result<RowIdSectionHeader> {
    if bytes.len() < ROW_ID_SECTION_HEADER_SIZE {
        return Err(invalid_data("DiskANN row-ID section header is truncated"));
    }
    if bytes[24..ROW_ID_SECTION_HEADER_SIZE]
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(invalid_data(
            "DiskANN row-ID section reserved bytes must be zero",
        ));
    }
    let count = usize::try_from(get_u64(bytes, 8))
        .map_err(|_| invalid_data("DiskANN row-ID count exceeds usize"))?;
    if count != expected_count {
        return Err(invalid_data("DiskANN row-ID count does not match header"));
    }
    let encoding = get_u32(bytes, 0);
    let width = get_u32(bytes, 4);
    let base = get_u64(bytes, 16) as i64;
    let (bit_width, payload_len) = match encoding {
        ROW_ID_ENCODING_RAW_I64 => {
            if width != u64::BITS || base != 0 {
                return Err(invalid_data("invalid DiskANN raw row-ID metadata"));
            }
            (
                u64::BITS as u8,
                count
                    .checked_mul(size_of::<i64>())
                    .ok_or_else(|| invalid_data("DiskANN raw row-ID length overflows usize"))?,
            )
        }
        ROW_ID_ENCODING_FOR_BITPACK => {
            let bit_width = u8::try_from(width)
                .map_err(|_| invalid_data("invalid DiskANN FOR row-ID bit width"))?;
            if bit_width >= u64::BITS as u8 {
                return Err(invalid_data("invalid DiskANN FOR row-ID bit width"));
            }
            let payload_len = packed_row_id_payload_len(count, bit_width)
                .map_err(|_| invalid_data("DiskANN packed row-ID length overflows usize"))?;
            (bit_width, payload_len)
        }
        _ => return Err(invalid_data("unsupported DiskANN row-ID encoding")),
    };
    if ROW_ID_SECTION_HEADER_SIZE.checked_add(payload_len) != Some(section_len) {
        return Err(invalid_data("invalid DiskANN row-ID payload length"));
    }
    Ok(RowIdSectionHeader {
        encoding,
        bit_width,
        count,
        base,
    })
}

fn validate_row_id_storage(storage: &RowIdStorage) -> io::Result<()> {
    if let RowIdStorage::ForBitPacked {
        bit_width,
        count,
        payload,
        ..
    } = storage
    {
        let used_bits = count
            .checked_mul(*bit_width as usize)
            .ok_or_else(|| invalid_data("DiskANN packed row-ID length overflows usize"))?;
        let tail_bits = used_bits % 8;
        if tail_bits != 0
            && payload
                .last()
                .is_some_and(|&byte| byte & !((1u8 << tail_bits) - 1) != 0)
        {
            return Err(invalid_data("DiskANN packed row-ID tail bits must be zero"));
        }
    }
    storage.try_for_each(|_, _| Ok(()))
}

#[cfg(test)]
fn encode_row_id_section(storage: &RowIdStorage) -> io::Result<Vec<u8>> {
    let section_len = storage.serialized_len()?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(section_len)
        .map_err(|_| invalid_input("DiskANN row-ID section allocation failed"))?;
    bytes.extend_from_slice(&storage.section_header());
    match storage {
        RowIdStorage::Raw(row_ids) => row_ids
            .iter()
            .for_each(|row_id| bytes.extend_from_slice(&row_id.to_le_bytes())),
        RowIdStorage::ForBitPacked { payload, .. } => bytes.extend_from_slice(payload),
    }
    Ok(bytes)
}

#[cfg(test)]
fn decode_row_id_section(bytes: &[u8], expected_count: usize) -> io::Result<RowIdStorage> {
    let header = decode_row_id_section_header(bytes, bytes.len(), expected_count)?;
    let payload = &bytes[ROW_ID_SECTION_HEADER_SIZE..];
    let storage = match header.encoding {
        ROW_ID_ENCODING_RAW_I64 => {
            let mut row_ids = Vec::new();
            row_ids
                .try_reserve_exact(header.count)
                .map_err(|_| invalid_data("DiskANN raw row-ID allocation failed"))?;
            row_ids.extend(payload.chunks_exact(8).map(|value| {
                i64::from_le_bytes(value.try_into().expect("validated eight-byte row ID"))
            }));
            RowIdStorage::Raw(row_ids)
        }
        ROW_ID_ENCODING_FOR_BITPACK => {
            let mut packed = Vec::new();
            packed
                .try_reserve_exact(payload.len())
                .map_err(|_| invalid_data("DiskANN packed row-ID allocation failed"))?;
            packed.extend_from_slice(payload);
            RowIdStorage::ForBitPacked {
                base: header.base,
                bit_width: header.bit_width,
                count: header.count,
                payload: packed,
            }
        }
        _ => unreachable!("row-ID encoding was validated"),
    };
    validate_row_id_storage(&storage)?;
    Ok(storage)
}

#[derive(Default)]
enum RowIdOrderState {
    #[default]
    NotLoaded,
    Loaded(Arc<[u32]>),
    UnavailableByBudget,
}

fn classify_auto_storage_profile(random_read_latency: Duration) -> StorageProfile {
    if random_read_latency < AUTO_PROFILE_MEMORY_LATENCY_THRESHOLD {
        StorageProfile::Memory
    } else if random_read_latency < AUTO_PROFILE_LOCAL_LATENCY_THRESHOLD {
        StorageProfile::LocalStorage
    } else if random_read_latency < AUTO_PROFILE_REMOTE_LATENCY_THRESHOLD {
        StorageProfile::RemoteStorage
    } else {
        StorageProfile::ObjectStore
    }
}

impl<R: SeekRead> DiskAnnIndexReader<R> {
    pub fn open(reader: R) -> io::Result<Self> {
        Self::open_with_options(reader, VectorIndexReaderOptions::default())
    }

    pub fn open_with_options(mut reader: R, options: VectorIndexReaderOptions) -> io::Result<Self> {
        let read_capabilities = reader.read_capabilities();
        let mut bytes = [0u8; DISKANN_HEADER_SIZE];
        reader
            .pread(&mut [ReadRequest::new(0, &mut bytes)])
            .map_err(|error| map_read_error(error, "header"))?;
        let header = DiskAnnHeader::decode(&bytes)?;
        let effective_storage_profile = options.storage_profile;
        let automatic_cache_budgets = options.uses_automatic_cache_budgets();
        let options = options.resolve_cache_budgets(
            resident_steady_bytes(&header)?,
            usize::try_from(header.sections.adjacency.length).unwrap_or(usize::MAX),
            usize::try_from(header.sections.vectors.length).unwrap_or(usize::MAX),
        );
        Ok(Self {
            reader,
            header,
            resident: None,
            options,
            automatic_cache_budgets,
            read_capabilities,
            effective_storage_profile,
            auto_storage_profile_probe_time: None,
            hot_adjacency: Arc::from([]),
            row_id_order: Arc::new(Mutex::new(RowIdOrderState::NotLoaded)),
            query_scratch: Box::default(),
            last_search_stats: DiskAnnSearchStats::default(),
            batch_workers: Vec::new(),
            calibrated_l_search: None,
        })
    }

    pub fn ensure_resident(&mut self) -> io::Result<()> {
        if self.resident.is_some() {
            return Ok(());
        }
        let peak_bytes = resident_peak_bytes(&self.header)?;
        if peak_bytes > self.options.max_resident_bytes {
            return Err(invalid_data(format!(
                "DiskANN resident warmup requires {} bytes, exceeding reader budget {}",
                peak_bytes, self.options.max_resident_bytes
            )));
        }

        if self.effective_storage_profile == StorageProfile::Auto {
            if let Some(
                profile @ (StorageProfile::Memory
                | StorageProfile::LocalStorage
                | StorageProfile::RemoteStorage
                | StorageProfile::ObjectStore),
            ) = self.reader.preferred_storage_profile()
            {
                self.effective_storage_profile = profile;
            }
        }
        if self.effective_storage_profile == StorageProfile::Auto {
            let latency = self.measure_random_read_latency()?;
            self.effective_storage_profile = classify_auto_storage_profile(latency);
            self.auto_storage_profile_probe_time = Some(latency);
        }
        if self.automatic_cache_budgets {
            self.options = VectorIndexReaderOptions::new(
                self.effective_storage_profile,
                self.options.max_resident_bytes,
            )
            .resolve_cache_budgets(
                resident_steady_bytes(&self.header)?,
                usize::try_from(self.header.sections.adjacency.length).unwrap_or(usize::MAX),
                usize::try_from(self.header.sections.vectors.length).unwrap_or(usize::MAX),
            );
        }

        let (mut pq, row_ids, pq_codes, adjacency_index) =
            read_resident_sections(&mut self.reader, &self.header)?;
        pq.try_rebuild_norms_cache()
            .map_err(|_| invalid_data("DiskANN PQ norms allocation failed"))?;
        validate_pq_code_padding(&self.header, &pq_codes)?;
        let adjacency_validation =
            AdjacencyValidationCache::new(adjacency_page_count(&self.header)?)?;
        self.resident = Some(Arc::new(DiskAnnResidentData {
            pq,
            row_ids,
            pq_codes,
            adjacency_index,
            adjacency_validation,
            adjacency_cache: SharedWindowCache::new(self.options.adjacency_cache_bytes),
            raw_vector_cache: SharedWindowCache::new(self.options.raw_vector_cache_bytes),
        }));
        Ok(())
    }

    pub fn pq(&self) -> io::Result<&ProductQuantizer> {
        Ok(&self.resident()?.pq)
    }

    pub fn row_id(&self, node: usize) -> io::Result<i64> {
        self.resident()?
            .row_ids
            .get(node)
            .ok_or_else(|| invalid_data("DiskANN row-ID node is out of range"))
    }

    pub fn row_id_count(&self) -> io::Result<usize> {
        Ok(self.resident()?.row_ids.len())
    }

    pub(crate) fn try_for_each_row_id(
        &self,
        visitor: impl FnMut(usize, i64) -> io::Result<()>,
    ) -> io::Result<()> {
        self.resident()?.row_ids.try_for_each(visitor)
    }

    pub fn pq_codes(&self) -> io::Result<&[u8]> {
        Ok(&self.resident()?.pq_codes)
    }

    pub(crate) fn adjacency_locator(&self, node: usize) -> io::Result<AdjacencyLocator> {
        self.resident()?
            .adjacency_index
            .locator(node)
            .ok_or_else(|| invalid_data("DiskANN adjacency index is truncated"))
    }

    pub(crate) fn adjacency_cache(&self) -> io::Result<&SharedWindowCache> {
        Ok(&self.resident()?.adjacency_cache)
    }

    pub(crate) fn raw_vector_cache(&self) -> io::Result<&SharedWindowCache> {
        Ok(&self.resident()?.raw_vector_cache)
    }

    fn resize_shared_cache_budgets(&self, total_bytes: usize) -> io::Result<()> {
        let desired_adjacency = self.options.adjacency_cache_bytes;
        let desired_raw = self.options.raw_vector_cache_bytes;
        let desired_total = desired_adjacency.saturating_add(desired_raw);
        let total_bytes = total_bytes.min(desired_total);
        let adjacency_bytes = if desired_total == 0 {
            0
        } else {
            usize::try_from(
                (total_bytes as u128 * desired_adjacency as u128) / desired_total as u128,
            )
            .unwrap_or(total_bytes)
        };
        let raw_bytes = total_bytes.saturating_sub(adjacency_bytes);
        let resident = self.resident()?;
        resident
            .adjacency_cache
            .set_total_capacity(adjacency_bytes)?;
        resident.raw_vector_cache.set_total_capacity(raw_bytes)
    }

    fn loaded_row_id_order_bytes(&self) -> io::Result<usize> {
        let state = self
            .row_id_order
            .lock()
            .map_err(|_| invalid_data("DiskANN row-ID lookup state is poisoned"))?;
        Ok(match &*state {
            RowIdOrderState::Loaded(order) => order
                .len()
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| invalid_data("DiskANN row-ID order size overflows usize"))?,
            RowIdOrderState::NotLoaded | RowIdOrderState::UnavailableByBudget => 0,
        })
    }

    pub(crate) fn ensure_row_id_order(&mut self) -> io::Result<Option<Arc<[u32]>>> {
        self.ensure_resident()?;
        let mut state = self
            .row_id_order
            .lock()
            .map_err(|_| invalid_data("DiskANN row-ID lookup state is poisoned"))?;
        match &*state {
            RowIdOrderState::Loaded(order) => return Ok(Some(order.clone())),
            RowIdOrderState::UnavailableByBudget => return Ok(None),
            RowIdOrderState::NotLoaded => {}
        }
        let peak_bytes = row_id_order_peak_bytes(&self.header, self.hot_adjacency.len())?;
        if peak_bytes > self.options.max_resident_bytes {
            *state = RowIdOrderState::UnavailableByBudget;
            return Ok(None);
        }
        // Reserve the decode peak before allocating the immutable lookup.
        // Cache hits remain lock-free with respect to this budget operation;
        // only cache publication reads the adjusted capacity.
        self.resize_shared_cache_budgets(self.options.max_resident_bytes - peak_bytes)?;
        let order = read_u32_section(
            &mut self.reader,
            self.header.sections.row_id_order,
            "row-ID order",
        )?;
        validate_row_id_order(&self.resident()?.row_ids, &order)?;
        let order: Arc<[u32]> = Arc::from(order);
        *state = RowIdOrderState::Loaded(order.clone());
        let steady_with_order = resident_steady_bytes(&self.header)?
            .checked_add(self.hot_adjacency.len())
            .and_then(|bytes| {
                order
                    .len()
                    .checked_mul(size_of::<u32>())
                    .and_then(|order_bytes| bytes.checked_add(order_bytes))
            })
            .ok_or_else(|| invalid_data("DiskANN filtered resident size overflows usize"))?;
        self.resize_shared_cache_budgets(
            self.options
                .max_resident_bytes
                .saturating_sub(steady_with_order),
        )?;
        Ok(Some(order))
    }

    pub fn optimize_for_search(&mut self) -> io::Result<()> {
        self.ensure_resident()?;
        if self.options.adjacency_preload_bytes == 0 || !self.hot_adjacency.is_empty() {
            return Ok(());
        }
        let adjacency = self.header.sections.adjacency;
        let requested_len = self
            .options
            .adjacency_preload_bytes
            .min(adjacency.length as usize);
        let requested_len = requested_len
            .div_ceil(DISKANN_ADJACENCY_PRELOAD_ALIGNMENT)
            .saturating_mul(DISKANN_ADJACENCY_PRELOAD_ALIGNMENT)
            .min(adjacency.length as usize);
        let row_id_order_bytes = self.loaded_row_id_order_bytes()?;
        let available_bytes = self
            .options
            .max_resident_bytes
            .saturating_sub(resident_steady_bytes(&self.header)?)
            .saturating_sub(row_id_order_bytes);
        let mut preload_len = requested_len.min(available_bytes);
        if preload_len < adjacency.length as usize {
            preload_len = preload_len / DISKANN_ADJACENCY_PRELOAD_ALIGNMENT
                * DISKANN_ADJACENCY_PRELOAD_ALIGNMENT;
        }
        if preload_len == 0 {
            self.resize_shared_cache_budgets(available_bytes)?;
            return Ok(());
        }
        self.resize_shared_cache_budgets(available_bytes.saturating_sub(preload_len))?;
        let mut payload = vec![0u8; preload_len];
        self.reader
            .pread(&mut [ReadRequest::new(adjacency.offset, &mut payload)])
            .map_err(|error| map_read_error(error, "adjacency preload"))?;
        let payload: Arc<[u8]> = Arc::from(payload);
        let resident = self
            .resident
            .as_ref()
            .expect("resident sections were loaded before adjacency preload");
        payload
            .par_chunks_exact(DISKANN_PAGE_SIZE as usize)
            .enumerate()
            .try_for_each(|(page_index, page)| {
                resident
                    .adjacency_validation
                    .get_or_validate(page_index, || {
                        validate_adjacency_page_payload(
                            &self.header,
                            &resident.adjacency_index,
                            page_index,
                            page,
                        )
                    })
            })?;
        self.hot_adjacency = payload;
        Ok(())
    }

    pub fn effective_storage_profile(&self) -> StorageProfile {
        self.effective_storage_profile
    }

    pub fn auto_storage_profile_probe_time(&self) -> Option<Duration> {
        self.auto_storage_profile_probe_time
    }

    fn measure_random_read_latency(&mut self) -> io::Result<Duration> {
        let probe_len = usize::try_from(self.header.file_len)
            .unwrap_or(usize::MAX)
            .clamp(1, AUTO_PROFILE_PROBE_BYTES);
        let max_start = self.header.file_len.saturating_sub(probe_len as u64);
        let alignment = probe_len as u64;
        let aligned_max = max_start / alignment * alignment;
        let align_down = |offset: u64| offset / alignment * alignment;
        let offsets = [
            align_down(aligned_max / 4),
            align_down(aligned_max / 2),
            aligned_max,
        ];
        let mut latencies = [Duration::ZERO; AUTO_PROFILE_PROBE_COUNT];
        let mut buffer = vec![0u8; probe_len];
        for (index, &offset) in offsets.iter().enumerate() {
            let started = Instant::now();
            self.reader
                .pread(&mut [ReadRequest::new(offset, &mut buffer)])
                .map_err(|error| map_read_error(error, "storage profile probe"))?;
            latencies[index] = started.elapsed();
        }
        latencies.sort_unstable();
        Ok(latencies[AUTO_PROFILE_PROBE_COUNT / 2])
    }

    pub(crate) const fn options(&self) -> ResolvedVectorIndexReaderOptions {
        let mut options = self.options;
        options.storage_profile = self.effective_storage_profile;
        options
    }

    pub fn read_capabilities(&self) -> SeekReadCapabilities {
        self.read_capabilities
    }

    pub(crate) fn read_plan(&self) -> ReadPlan {
        self.options()
            .storage_profile
            .read_plan()
            .with_capabilities(self.read_capabilities)
    }

    pub(crate) fn limit_raw_vector_cache_bytes(&mut self, limit: usize) {
        self.options.raw_vector_cache_bytes = self.options.raw_vector_cache_bytes.min(limit);
    }

    pub fn last_search_stats(&self) -> DiskAnnSearchStats {
        self.last_search_stats
    }

    pub(crate) fn pread_ranges(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        let max_ranges = self.read_capabilities.max_ranges_per_pread;
        if max_ranges == 0 || ranges.len() <= max_ranges {
            return self
                .reader
                .pread(ranges)
                .map_err(|error| map_read_error(error, "paged section"));
        }
        for chunk in ranges.chunks_mut(max_ranges) {
            self.reader
                .pread(chunk)
                .map_err(|error| map_read_error(error, "paged section"))?;
        }
        Ok(())
    }

    pub(crate) fn hot_adjacency_window(&self, offset: u64, length: usize) -> Option<&[u8]> {
        let relative =
            usize::try_from(offset.checked_sub(self.header.sections.adjacency.offset)?).ok()?;
        let end = relative.checked_add(length)?;
        self.hot_adjacency.get(relative..end)
    }

    pub(crate) fn adjacency_fully_preloaded(&self) -> bool {
        u64::try_from(self.hot_adjacency.len()).ok() == Some(self.header.sections.adjacency.length)
    }

    pub(crate) fn try_clone_for_search(&mut self) -> io::Result<Option<Self>> {
        self.optimize_for_search()?;
        self.try_clone_with_shared_state()
    }

    pub(crate) fn try_clone_for_filtered_search(&mut self) -> io::Result<Option<Self>> {
        self.ensure_resident()?;
        self.try_clone_with_shared_state()
    }

    fn try_clone_with_shared_state(&self) -> io::Result<Option<Self>> {
        let Some(reader) = self.reader.try_clone_reader()? else {
            return Ok(None);
        };
        Ok(Some(Self {
            reader,
            header: self.header.clone(),
            resident: self.resident.clone(),
            options: self.options,
            automatic_cache_budgets: self.automatic_cache_budgets,
            read_capabilities: self.read_capabilities,
            effective_storage_profile: self.effective_storage_profile,
            auto_storage_profile_probe_time: self.auto_storage_profile_probe_time,
            hot_adjacency: self.hot_adjacency.clone(),
            row_id_order: self.row_id_order.clone(),
            query_scratch: Box::default(),
            last_search_stats: DiskAnnSearchStats::default(),
            batch_workers: Vec::new(),
            calibrated_l_search: self.calibrated_l_search,
        }))
    }

    pub(crate) fn refresh_shared_state_from(&mut self, source: &Self) {
        self.header = source.header.clone();
        self.resident = source.resident.clone();
        self.options = source.options;
        self.automatic_cache_budgets = source.automatic_cache_budgets;
        self.read_capabilities = source.read_capabilities;
        self.effective_storage_profile = source.effective_storage_profile;
        self.auto_storage_profile_probe_time = source.auto_storage_profile_probe_time;
        self.calibrated_l_search = source.calibrated_l_search;
        self.hot_adjacency = source.hot_adjacency.clone();
        self.row_id_order = source.row_id_order.clone();
    }

    pub(crate) fn validate_adjacency_page(&self, page_index: usize, page: &[u8]) -> io::Result<()> {
        let resident = self.resident()?;
        resident
            .adjacency_validation
            .get_or_validate(page_index, || {
                validate_adjacency_page_payload(
                    &self.header,
                    &resident.adjacency_index,
                    page_index,
                    page,
                )
            })
    }

    fn resident(&self) -> io::Result<&DiskAnnResidentData> {
        self.resident
            .as_deref()
            .ok_or_else(|| invalid_data("DiskANN resident sections are not loaded"))
    }
}

fn validate_adjacency_page_payload(
    header: &DiskAnnHeader,
    locators: &AdjacencyIndex,
    page_index: usize,
    page: &[u8],
) -> io::Result<()> {
    if page.len() != DISKANN_PAGE_SIZE as usize {
        return Err(invalid_data("DiskANN adjacency page is truncated"));
    }
    let first = locators.partition_point(|locator| (locator.page_index as usize) < page_index);
    let end = locators.partition_point(|locator| (locator.page_index as usize) <= page_index);
    if first == end {
        return Err(invalid_data("DiskANN adjacency page has no indexed nodes"));
    }
    let vector_count = header.vector_count as usize;
    let record_prefix = if header.is_interleaved() {
        header.vector_record_size as usize
    } else {
        0
    };
    let mut used_end = 0usize;
    let mut neighbors = Vec::new();
    for node in 0..end - first {
        let source = first + node;
        let locator = locators
            .locator(source)
            .ok_or_else(|| invalid_data("DiskANN adjacency index is truncated"))?;
        let start = locator.byte_offset as usize;
        let record_start = start
            .checked_sub(record_prefix)
            .ok_or_else(|| invalid_data("DiskANN interleaved vector offset underflows"))?;
        if record_start != used_end {
            return Err(invalid_data(
                "DiskANN adjacency page records must be contiguous",
            ));
        }
        if header.is_interleaved() {
            let vector = page
                .get(record_start..start)
                .ok_or_else(|| invalid_data("DiskANN interleaved raw vector is truncated"))?;
            validate_raw_vector_bytes(vector, header.raw_vector_encoding())?;
        }
        let bytes = page
            .get(start..)
            .ok_or_else(|| invalid_data("DiskANN adjacency list is truncated"))?;
        let consumed =
            decode_adjacency_list(bytes, locator.degree(), locator.encoding(), &mut neighbors)?;
        let list_end = start
            .checked_add(consumed)
            .ok_or_else(|| invalid_data("DiskANN adjacency locator range overflows"))?;
        if let Some(next) = locators
            .locator(source + 1)
            .filter(|next| next.page_index == locator.page_index)
        {
            if list_end.checked_add(record_prefix) != Some(next.byte_offset as usize) {
                return Err(invalid_data(
                    "DiskANN adjacency locator ranges must be contiguous",
                ));
            }
        }
        let mut previous = None;
        for &neighbor in &neighbors {
            if neighbor as usize >= vector_count {
                return Err(invalid_data("DiskANN adjacency neighbor is out of range"));
            }
            if neighbor as usize == source {
                return Err(invalid_data("DiskANN adjacency contains a self edge"));
            }
            if previous.is_some_and(|previous| neighbor <= previous) {
                return Err(invalid_data(
                    "DiskANN adjacency neighbors must be strictly increasing",
                ));
            }
            previous = Some(neighbor);
        }
        let raw_len = neighbors
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| invalid_data("DiskANN adjacency list size overflows usize"))?;
        let delta_len = adjacency_delta_varint_len(&neighbors)
            .expect("validated DiskANN adjacency neighbors are strictly increasing");
        let minimal_encoding = if neighbors.is_empty() || delta_len < raw_len {
            AdjacencyListEncoding::DeltaVarint
        } else {
            AdjacencyListEncoding::RawU32
        };
        if locator.encoding() != minimal_encoding {
            return Err(invalid_data(
                "DiskANN adjacency list does not use its minimal encoding",
            ));
        }
        used_end = list_end;
    }
    if page[used_end..].iter().any(|&byte| byte != 0) {
        return Err(invalid_data("DiskANN adjacency page tail must be zero"));
    }
    Ok(())
}

fn validate_raw_vector_bytes(vector: &[u8], encoding: DiskAnnRawVectorEncoding) -> io::Result<()> {
    let has_non_finite = match encoding {
        DiskAnnRawVectorEncoding::F32 => vector.chunks_exact(size_of::<f32>()).any(|value| {
            !f32::from_le_bytes(value.try_into().expect("four-byte vector component")).is_finite()
        }),
        DiskAnnRawVectorEncoding::F16 => vector.chunks_exact(size_of::<u16>()).any(|value| {
            !half::f16::from_bits(u16::from_le_bytes(
                value.try_into().expect("two-byte vector component"),
            ))
            .is_finite()
        }),
    };
    if !vector.len().is_multiple_of(encoding.element_size()) || has_non_finite {
        return Err(invalid_data(
            "DiskANN interleaved raw vector contains a non-finite or malformed component",
        ));
    }
    Ok(())
}

impl DiskAnnHeader {
    pub fn for_layout(
        dimension: usize,
        vector_count: usize,
        entry_node: u32,
        pq_m: usize,
        build: DiskAnnBuildParams,
    ) -> io::Result<Self> {
        Self::for_layout_with_pq_bits(dimension, vector_count, entry_node, pq_m, 8, build)
    }

    pub fn for_layout_with_pq_bits(
        dimension: usize,
        vector_count: usize,
        entry_node: u32,
        pq_m: usize,
        pq_bits: usize,
        build: DiskAnnBuildParams,
    ) -> io::Result<Self> {
        let row_ids_len = raw_row_id_section_len(vector_count)?;
        Self::for_layout_with_adjacency_pages_and_pq_bits(
            dimension,
            vector_count,
            entry_node,
            pq_m,
            pq_bits,
            MetricType::L2,
            build,
            row_ids_len,
            1,
        )
    }

    #[cfg(test)]
    fn for_layout_with_adjacency_pages(
        dimension: usize,
        vector_count: usize,
        entry_node: u32,
        pq_m: usize,
        build: DiskAnnBuildParams,
        row_ids_len: usize,
        adjacency_pages: usize,
    ) -> io::Result<Self> {
        Self::for_layout_with_adjacency_pages_and_pq_bits(
            dimension,
            vector_count,
            entry_node,
            pq_m,
            8,
            MetricType::L2,
            build,
            row_ids_len,
            adjacency_pages,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn for_layout_with_adjacency_pages_and_pq_bits(
        dimension: usize,
        vector_count: usize,
        entry_node: u32,
        pq_m: usize,
        pq_bits: usize,
        metric: MetricType,
        build: DiskAnnBuildParams,
        row_ids_len: usize,
        adjacency_pages: usize,
    ) -> io::Result<Self> {
        validate_diskann_format_configuration(dimension, pq_m, pq_bits, build)?;
        if vector_count == 0 || u32::try_from(vector_count).is_err() {
            return Err(invalid_input(
                "DiskANN vector count must be between 1 and u32::MAX",
            ));
        }
        if entry_node as usize >= vector_count {
            return Err(invalid_input("DiskANN entry node is out of range"));
        }
        if adjacency_pages == 0 || u32::try_from(adjacency_pages).is_err() {
            return Err(invalid_input(
                "DiskANN adjacency page count must be between 1 and u32::MAX",
            ));
        }
        let maximum_row_ids_len = raw_row_id_section_len(vector_count)?;
        if !(ROW_ID_SECTION_HEADER_SIZE..=maximum_row_ids_len).contains(&row_ids_len) {
            return Err(invalid_input("invalid DiskANN row-ID section length"));
        }
        let dimension_u32 =
            u32::try_from(dimension).map_err(|_| invalid_input("DiskANN dimension exceeds u32"))?;
        let pq_m_u32 =
            u32::try_from(pq_m).map_err(|_| invalid_input("DiskANN pq.m exceeds u32"))?;
        let pq_bits_u32 =
            u32::try_from(pq_bits).map_err(|_| invalid_input("DiskANN pq.bits exceeds u32"))?;
        let pq_ksub = diskann_pq_ksub(pq_bits_u32)?;
        let pq_code_size = diskann_pq_code_size(pq_m, pq_bits_u32)?;
        let max_degree = u32::try_from(build.max_degree)
            .map_err(|_| invalid_input("DiskANN maximum degree exceeds u32"))?;
        if max_degree == 0 || max_degree > 1023 {
            return Err(invalid_input(
                "DiskANN maximum degree must be between 1 and 1023",
            ));
        }
        let build_search_list_size = u32::try_from(build.build_search_list_size)
            .map_err(|_| invalid_input("DiskANN build search-list size exceeds u32"))?;
        let raw_vector_encoding = build.raw_vector_encoding as u32;
        let vector_record_size = dimension_u32
            .checked_mul(build.raw_vector_encoding.element_size() as u32)
            .ok_or_else(|| invalid_input("DiskANN vector record size overflows u32"))?;
        if vector_record_size == 0 || vector_record_size > DISKANN_PAGE_SIZE {
            return Err(invalid_input(
                "DiskANN raw vector does not fit in a logical page",
            ));
        }
        if build.storage_layout == DiskAnnStorageLayout::Interleaved {
            let maximum_adjacency_bytes = max_degree
                .checked_mul(size_of::<u32>() as u32)
                .ok_or_else(|| invalid_input("DiskANN adjacency record size overflows u32"))?;
            if vector_record_size
                .checked_add(maximum_adjacency_bytes)
                .is_none_or(|record_size| record_size > DISKANN_PAGE_SIZE)
            {
                return Err(invalid_input(
                    "DiskANN interleaved raw vector and maximum adjacency list do not fit in one page",
                ));
            }
        }

        let codebook_len = pq_codebook_serialized_len(dimension, pq_m, pq_ksub)?;
        let row_ids_len = row_ids_len as u64;
        let pq_codes_len = checked_mul_u64(vector_count as u64, pq_code_size as u64, "PQ codes")?;
        let row_id_order_len = checked_mul_u64(vector_count as u64, 4, "row-ID order")?;
        let adjacency_index_len = adjacency_index_serialized_len(vector_count)?;
        let adjacency_len = checked_mul_u64(
            adjacency_pages as u64,
            DISKANN_PAGE_SIZE as u64,
            "adjacency pages",
        )?;
        let vectors_len = match build.storage_layout {
            DiskAnnStorageLayout::Compact => checked_mul_u64(
                vector_count as u64,
                vector_record_size as u64,
                "raw vector records",
            )?,
            DiskAnnStorageLayout::Interleaved => 0,
        };

        let codebook = SectionRange::new(DISKANN_PAGE_SIZE as u64, codebook_len);
        let row_ids = SectionRange::new(section_end(codebook)?, row_ids_len);
        let pq_codes = SectionRange::new(section_end(row_ids)?, pq_codes_len);
        let row_id_order = SectionRange::new(section_end(pq_codes)?, row_id_order_len);
        let adjacency_index = SectionRange::new(section_end(row_id_order)?, adjacency_index_len);
        let adjacency = SectionRange::new(
            align_up(section_end(adjacency_index)?, DISKANN_PAGE_SIZE as u64)?,
            adjacency_len,
        );
        let vectors = SectionRange::new(section_end(adjacency)?, vectors_len);
        let file_len = section_end(vectors)?;

        Ok(Self {
            flags: DISKANN_REQUIRED_FLAGS
                | match build.storage_layout {
                    DiskAnnStorageLayout::Compact => FLAG_SEPARATE_ADJACENCY_AND_VECTORS,
                    DiskAnnStorageLayout::Interleaved => FLAG_INTERLEAVED_ADJACENCY_AND_VECTORS,
                },
            dimension: dimension_u32,
            metric: metric as u32,
            vector_count: vector_count as u64,
            entry_node,
            max_degree,
            build_search_list_size,
            alpha: build.alpha,
            seed: build.seed,
            pq_m: pq_m_u32,
            pq_bits: pq_bits_u32,
            page_size: DISKANN_PAGE_SIZE,
            adjacency_locator_size: ADJACENCY_LOCATOR_SIZE,
            adjacency_locator_encoding: ADJACENCY_LOCATOR_ENCODING,
            raw_vector_encoding,
            vector_record_size,
            file_len,
            sections: DiskAnnSections {
                codebook,
                row_ids,
                pq_codes,
                row_id_order,
                adjacency_index,
                adjacency,
                vectors,
            },
        })
    }

    pub fn encode(&self) -> [u8; DISKANN_HEADER_SIZE] {
        let mut bytes = [0u8; DISKANN_HEADER_SIZE];
        put_u32(&mut bytes, 0, DISKANN_MAGIC);
        put_u32(&mut bytes, 4, DISKANN_VERSION);
        put_u32(&mut bytes, 8, DISKANN_HEADER_SIZE as u32);
        put_u32(&mut bytes, 12, self.flags);
        put_u32(&mut bytes, 16, self.dimension);
        put_u32(&mut bytes, 20, self.metric);
        put_u64(&mut bytes, 24, self.vector_count);
        put_u32(&mut bytes, 32, self.entry_node);
        put_u32(&mut bytes, 36, self.max_degree);
        put_u32(&mut bytes, 40, self.build_search_list_size);
        put_u32(&mut bytes, 44, self.alpha.to_bits());
        put_u64(&mut bytes, 48, self.seed);
        put_u32(&mut bytes, 56, self.pq_m);
        put_u32(&mut bytes, 60, self.pq_bits);
        put_u32(&mut bytes, 64, self.page_size);
        put_u32(&mut bytes, 68, self.adjacency_locator_size);
        put_u32(&mut bytes, 72, self.adjacency_locator_encoding);
        put_u32(&mut bytes, 76, self.raw_vector_encoding);
        put_u32(&mut bytes, 80, self.vector_record_size);
        put_u32(&mut bytes, 84, SECTION_COUNT as u32);
        put_u64(&mut bytes, 88, self.file_len);
        for (index, section) in self.sections.as_array().into_iter().enumerate() {
            let offset = 96 + index * 16;
            put_u64(&mut bytes, offset, section.offset);
            put_u64(&mut bytes, offset + 8, section.length);
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < DISKANN_HEADER_SIZE {
            return Err(invalid_data(format!(
                "DiskANN header is truncated: {} bytes",
                bytes.len()
            )));
        }
        if get_u32(bytes, 0) != DISKANN_MAGIC {
            return Err(invalid_data("invalid DiskANN magic"));
        }
        if get_u32(bytes, 4) != DISKANN_VERSION {
            return Err(invalid_data("unsupported DiskANN version"));
        }
        if get_u32(bytes, 8) != DISKANN_HEADER_SIZE as u32 {
            return Err(invalid_data("invalid DiskANN header size"));
        }
        let flags = get_u32(bytes, 12);
        if flags & DISKANN_REQUIRED_FLAGS != DISKANN_REQUIRED_FLAGS
            || flags & !DISKANN_SUPPORTED_FLAGS != 0
            || (flags & FLAG_SEPARATE_ADJACENCY_AND_VECTORS != 0)
                == (flags & FLAG_INTERLEAVED_ADJACENCY_AND_VECTORS != 0)
        {
            return Err(invalid_data("invalid DiskANN required flags"));
        }
        if get_u32(bytes, 84) != SECTION_COUNT as u32 {
            return Err(invalid_data("invalid DiskANN section count"));
        }
        if bytes[208..DISKANN_HEADER_SIZE]
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(invalid_data("DiskANN reserved header bytes must be zero"));
        }
        let mut sections = [SectionRange::new(0, 0); SECTION_COUNT];
        for (index, section) in sections.iter_mut().enumerate() {
            let offset = 96 + index * 16;
            *section = SectionRange::new(get_u64(bytes, offset), get_u64(bytes, offset + 8));
        }
        let header = Self {
            flags,
            dimension: get_u32(bytes, 16),
            metric: get_u32(bytes, 20),
            vector_count: get_u64(bytes, 24),
            entry_node: get_u32(bytes, 32),
            max_degree: get_u32(bytes, 36),
            build_search_list_size: get_u32(bytes, 40),
            alpha: f32::from_bits(get_u32(bytes, 44)),
            seed: get_u64(bytes, 48),
            pq_m: get_u32(bytes, 56),
            pq_bits: get_u32(bytes, 60),
            page_size: get_u32(bytes, 64),
            adjacency_locator_size: get_u32(bytes, 68),
            adjacency_locator_encoding: get_u32(bytes, 72),
            raw_vector_encoding: get_u32(bytes, 76),
            vector_record_size: get_u32(bytes, 80),
            file_len: get_u64(bytes, 88),
            sections: DiskAnnSections::from_array(sections),
        };
        if MetricType::from_code(header.metric).is_none()
            || !matches!(header.pq_bits, 4 | 8)
            || header.page_size != DISKANN_PAGE_SIZE
            || DiskAnnRawVectorEncoding::from_code(header.raw_vector_encoding).is_none()
        {
            return Err(invalid_data("unsupported DiskANN v1 layout"));
        }
        if !header.alpha.is_finite() {
            return Err(invalid_data("DiskANN alpha must be finite"));
        }
        header.validate_layout()?;
        Ok(header)
    }

    fn validate_layout(&self) -> io::Result<()> {
        if self.dimension == 0 || self.dimension > 1024 {
            return Err(invalid_data("invalid DiskANN dimension"));
        }
        if self.vector_count == 0 || self.vector_count > u32::MAX as u64 {
            return Err(invalid_data("invalid DiskANN vector count"));
        }
        if self.entry_node as u64 >= self.vector_count {
            return Err(invalid_data("invalid DiskANN entry node"));
        }
        if self.pq_m == 0 || self.pq_m > self.dimension {
            return Err(invalid_data("invalid DiskANN PQ shape"));
        }
        if self.max_degree == 0
            || self.max_degree > 1023
            || self.build_search_list_size < self.max_degree
        {
            return Err(invalid_data("invalid DiskANN build parameters"));
        }
        if self.alpha < 1.0 {
            return Err(invalid_data("invalid DiskANN alpha"));
        }
        let vector_count = usize::try_from(self.vector_count)
            .map_err(|_| invalid_data("DiskANN vector count exceeds usize"))?;
        let adjacency_pages =
            usize::try_from(self.sections.adjacency.length / DISKANN_PAGE_SIZE as u64)
                .map_err(|_| invalid_data("DiskANN adjacency page count exceeds usize"))?;
        if adjacency_pages == 0
            || !self
                .sections
                .adjacency
                .length
                .is_multiple_of(DISKANN_PAGE_SIZE as u64)
        {
            return Err(invalid_data("invalid DiskANN adjacency section length"));
        }
        let expected = Self::for_layout_with_adjacency_pages_and_pq_bits(
            self.dimension as usize,
            vector_count,
            self.entry_node,
            self.pq_m as usize,
            self.pq_bits as usize,
            self.metric_type(),
            DiskAnnBuildParams {
                max_degree: self.max_degree as usize,
                build_search_list_size: self.build_search_list_size as usize,
                alpha: self.alpha,
                seed: self.seed,
                memory_budget_bytes: 1,
                storage_layout: self.storage_layout(),
                raw_vector_encoding: self.raw_vector_encoding(),
                ..DiskAnnBuildParams::default()
            },
            usize::try_from(self.sections.row_ids.length)
                .map_err(|_| invalid_data("DiskANN row-ID section exceeds usize"))?,
            adjacency_pages,
        )
        .map_err(|error| invalid_data(format!("invalid DiskANN layout: {}", error)))?;
        if self.adjacency_locator_size != expected.adjacency_locator_size
            || self.adjacency_locator_encoding != expected.adjacency_locator_encoding
            || self.raw_vector_encoding != expected.raw_vector_encoding
            || self.vector_record_size != expected.vector_record_size
            || self.sections != expected.sections
            || self.file_len != expected.file_len
        {
            return Err(invalid_data("invalid DiskANN section layout"));
        }
        Ok(())
    }

    pub(crate) fn storage_layout(&self) -> DiskAnnStorageLayout {
        if self.flags & FLAG_INTERLEAVED_ADJACENCY_AND_VECTORS != 0 {
            DiskAnnStorageLayout::Interleaved
        } else {
            DiskAnnStorageLayout::Compact
        }
    }

    pub(crate) fn is_interleaved(&self) -> bool {
        self.storage_layout() == DiskAnnStorageLayout::Interleaved
    }

    pub(crate) fn raw_vector_encoding(&self) -> DiskAnnRawVectorEncoding {
        DiskAnnRawVectorEncoding::from_code(self.raw_vector_encoding)
            .expect("validated DiskANN raw-vector encoding")
    }

    pub(crate) fn metric_type(&self) -> MetricType {
        MetricType::from_code(self.metric).expect("validated DiskANN metric")
    }
}

pub fn write_diskann_index(index: &DiskAnnIndex, out: &mut dyn SeekWrite) -> io::Result<()> {
    write_diskann_index_with_stats(index, out).map(|_| ())
}

pub fn write_diskann_index_with_stats(
    index: &DiskAnnIndex,
    out: &mut dyn SeekWrite,
) -> io::Result<DiskAnnBuildStats> {
    let total_started = Instant::now();
    if out.pos() != 0 {
        return Err(invalid_input("DiskANN output must start at offset zero"));
    }
    let prepared = index.prepare_build()?;
    let mut stats = prepared.stats;
    let interleaved_vector_bytes = match index.build_params.storage_layout {
        DiskAnnStorageLayout::Compact => 0,
        DiskAnnStorageLayout::Interleaved => index
            .d
            .checked_mul(index.build_params.raw_vector_encoding.element_size())
            .ok_or_else(|| invalid_input("DiskANN raw vector size overflows usize"))?,
    };
    let adjacency_layout = plan_adjacency_layout(&prepared, interleaved_vector_bytes)?;
    let row_ids =
        RowIdStorage::encode_from_fn(index.ids.len(), |new_id| prepared.row_id(index, new_id))?;
    let row_ids_len = row_ids.serialized_len()?;
    let header = DiskAnnHeader::for_layout_with_adjacency_pages_and_pq_bits(
        index.d,
        index.ids.len(),
        prepared.graph.entry_node,
        index.pq.m,
        index.pq.nbits,
        index.metric,
        index.build_params,
        row_ids_len,
        adjacency_layout.page_count,
    )?;
    let resident_started = Instant::now();
    out.write_all(&header.encode())?;
    pad_to(out, header.sections.codebook.offset)?;
    {
        let mut resident_writer = ChunkedSectionWriter::new(out);
        write_pq_codebook(&index.pq, &mut resident_writer)?;
        write_row_id_section(&row_ids, &mut resident_writer)?;
        for new_id in 0..index.ids.len() {
            resident_writer.write_bytes(prepared.pq_code(index, new_id))?;
        }
        let mut row_id_order = (0..index.ids.len() as u32).collect::<Vec<_>>();
        row_id_order.sort_unstable_by(|&left, &right| {
            prepared
                .row_id(index, left as usize)
                .cmp(&prepared.row_id(index, right as usize))
                .then_with(|| left.cmp(&right))
        });
        for node in row_id_order {
            resident_writer.write_bytes(&node.to_le_bytes())?;
        }
        write_adjacency_index(&adjacency_layout, &mut resident_writer)?;
        resident_writer.finish()?;
    }
    stats.resident_serialization = resident_started.elapsed();
    let adjacency_started = Instant::now();
    pad_to(out, header.sections.adjacency.offset)?;
    {
        let mut adjacency_writer = ChunkedSectionWriter::new(out);
        write_adjacency_pages(index, &prepared, &adjacency_layout, &mut adjacency_writer)?;
        adjacency_writer.finish()?;
    }
    stats.adjacency_serialization = adjacency_started.elapsed();
    let vector_started = Instant::now();
    if !header.is_interleaved() {
        write_vector_records(index, &prepared, &header, out)?;
    }
    stats.vector_serialization = vector_started.elapsed();
    if out.pos() != header.file_len {
        return Err(invalid_input(format!(
            "DiskANN writer ended at {}, expected {}",
            out.pos(),
            header.file_len
        )));
    }
    stats.total = total_started.elapsed();
    Ok(stats)
}

struct AdjacencyLayout {
    locators: Vec<AdjacencyLocator>,
    page_count: usize,
}

fn write_adjacency_index(
    layout: &AdjacencyLayout,
    writer: &mut ChunkedSectionWriter<'_>,
) -> io::Result<()> {
    for block_start in (0..layout.locators.len()).step_by(ADJACENCY_LOCATOR_BLOCK_NODES) {
        let block_offset = adjacency_locator_absolute_offset(layout.locators[block_start])?;
        writer.write_bytes(&block_offset.to_le_bytes())?;
    }
    for (node, locator) in layout.locators.iter().copied().enumerate() {
        let block_start = node / ADJACENCY_LOCATOR_BLOCK_NODES * ADJACENCY_LOCATOR_BLOCK_NODES;
        let block_offset = adjacency_locator_absolute_offset(layout.locators[block_start])?;
        let relative_offset = adjacency_locator_absolute_offset(locator)?
            .checked_sub(block_offset)
            .and_then(|offset| u16::try_from(offset).ok())
            .ok_or_else(|| {
                invalid_input("DiskANN adjacency locator exceeds its block offset range")
            })?;
        writer.write_bytes(&relative_offset.to_le_bytes())?;
    }
    for locator in &layout.locators {
        writer.write_bytes(&locator.degree_and_flags.to_le_bytes())?;
    }
    Ok(())
}

fn plan_adjacency_layout(
    prepared: &PreparedDiskAnn,
    interleaved_vector_bytes: usize,
) -> io::Result<AdjacencyLayout> {
    let mut locators = Vec::new();
    locators
        .try_reserve_exact(prepared.graph.adjacency.len())
        .map_err(|_| invalid_input("DiskANN adjacency locator allocation failed"))?;
    let mut page_index = 0u32;
    let mut byte_offset = 0usize;
    for neighbors in prepared.graph.adjacency.iter() {
        let (encoding, list_bytes) = plan_adjacency_list(neighbors)?;
        let record_bytes = interleaved_vector_bytes
            .checked_add(list_bytes)
            .ok_or_else(|| invalid_input("DiskANN interleaved record size overflows usize"))?;
        if record_bytes > DISKANN_PAGE_SIZE as usize {
            return Err(invalid_input(
                "DiskANN vector and adjacency list do not fit in a logical page",
            ));
        }
        if byte_offset + record_bytes > DISKANN_PAGE_SIZE as usize {
            page_index = page_index
                .checked_add(1)
                .ok_or_else(|| invalid_input("DiskANN adjacency page count exceeds u32"))?;
            byte_offset = 0;
        }
        locators.push(AdjacencyLocator::new(
            page_index,
            u16::try_from(byte_offset + interleaved_vector_bytes)
                .map_err(|_| invalid_input("DiskANN adjacency byte offset exceeds u16"))?,
            neighbors.len(),
            encoding,
        )?);
        byte_offset += record_bytes;
    }
    Ok(AdjacencyLayout {
        locators,
        page_count: page_index as usize + 1,
    })
}

fn write_adjacency_pages(
    index: &DiskAnnIndex,
    prepared: &PreparedDiskAnn,
    layout: &AdjacencyLayout,
    writer: &mut ChunkedSectionWriter<'_>,
) -> io::Result<()> {
    let mut page = vec![0u8; DISKANN_PAGE_SIZE as usize];
    let mut current_page = 0u32;
    let mut encoded = Vec::new();
    let mut encoded_vector = Vec::new();
    for (node, locator) in layout.locators.iter().enumerate() {
        while locator.page_index > current_page {
            writer.write_bytes(&page)?;
            page.fill(0);
            current_page += 1;
        }
        let encoding = encode_adjacency_list(&prepared.graph.adjacency[node], &mut encoded)?;
        if encoding != locator.encoding() {
            return Err(invalid_input(
                "DiskANN adjacency encoding changed after layout planning",
            ));
        }
        let start = locator.byte_offset as usize;
        if index.build_params.storage_layout == DiskAnnStorageLayout::Interleaved {
            encode_raw_vector(
                prepared.vector(index, node),
                index.build_params.raw_vector_encoding,
                &mut encoded_vector,
            )?;
            let vector_start = start
                .checked_sub(encoded_vector.len())
                .ok_or_else(|| invalid_input("DiskANN interleaved vector offset underflows"))?;
            page[vector_start..start].copy_from_slice(&encoded_vector);
        }
        let end = start
            .checked_add(encoded.len())
            .ok_or_else(|| invalid_input("DiskANN adjacency page range overflows"))?;
        page.get_mut(start..end)
            .ok_or_else(|| invalid_input("DiskANN adjacency list crosses a logical page"))?
            .copy_from_slice(&encoded);
    }
    writer.write_bytes(&page)?;
    Ok(())
}

fn write_vector_records(
    index: &DiskAnnIndex,
    prepared: &PreparedDiskAnn,
    header: &DiskAnnHeader,
    out: &mut dyn SeekWrite,
) -> io::Result<()> {
    let mut writer = ChunkedSectionWriter::new(out);
    let mut encoded = Vec::new();
    for new_id in 0..index.ids.len() {
        encode_raw_vector(
            prepared.vector(index, new_id),
            header.raw_vector_encoding(),
            &mut encoded,
        )?;
        writer.write_bytes(&encoded)?;
    }
    writer.finish()
}

fn encode_raw_vector(
    vector: &[f32],
    encoding: DiskAnnRawVectorEncoding,
    encoded: &mut Vec<u8>,
) -> io::Result<()> {
    encoded.clear();
    let encoded_len = vector
        .len()
        .checked_mul(encoding.element_size())
        .ok_or_else(|| invalid_input("DiskANN raw-vector encoding size overflows usize"))?;
    encoded
        .try_reserve_exact(encoded_len)
        .map_err(|_| invalid_input("DiskANN raw-vector encoding allocation failed"))?;
    match encoding {
        DiskAnnRawVectorEncoding::F32 => {
            for &value in vector {
                encoded.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        DiskAnnRawVectorEncoding::F16 => {
            for &value in vector {
                let value = half::f16::from_f32(value);
                if !value.is_finite() {
                    return Err(invalid_input(
                        "DiskANN vector value is outside the finite f16 range",
                    ));
                }
                encoded.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    Ok(())
}

struct ChunkedSectionWriter<'a> {
    out: &'a mut dyn SeekWrite,
    buffer: Vec<u8>,
}

impl<'a> ChunkedSectionWriter<'a> {
    fn new(out: &'a mut dyn SeekWrite) -> Self {
        Self {
            out,
            buffer: Vec::with_capacity(DISKANN_WRITE_BUFFER_SIZE),
        }
    }

    fn write_bytes(&mut self, mut bytes: &[u8]) -> io::Result<()> {
        while !bytes.is_empty() {
            if self.buffer.len() == DISKANN_WRITE_BUFFER_SIZE {
                self.flush()?;
            }
            let available = DISKANN_WRITE_BUFFER_SIZE - self.buffer.len();
            let chunk = available.min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..chunk]);
            bytes = &bytes[chunk..];
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.buffer.is_empty() {
            self.out.write_all(&self.buffer)?;
            self.buffer.clear();
        }
        Ok(())
    }

    fn finish(mut self) -> io::Result<()> {
        self.flush()
    }
}

fn write_pq_codebook(
    pq: &ProductQuantizer,
    writer: &mut ChunkedSectionWriter<'_>,
) -> io::Result<()> {
    if !pq.has_valid_layout() {
        return Err(invalid_input("DiskANN PQ codebook layout is invalid"));
    }
    let mut header = [0u8; PQ_CODEBOOK_HEADER_SIZE];
    put_u32(&mut header, 0, PQ_CODEBOOK_MAGIC);
    put_u32(&mut header, 4, PQ_CODEBOOK_VERSION);
    put_u32(
        &mut header,
        8,
        u32::try_from(pq.d).map_err(|_| invalid_input("DiskANN PQ dimension exceeds u32"))?,
    );
    put_u32(
        &mut header,
        12,
        u32::try_from(pq.m).map_err(|_| invalid_input("DiskANN PQ m exceeds u32"))?,
    );
    put_u32(
        &mut header,
        16,
        u32::try_from(pq.nbits).map_err(|_| invalid_input("DiskANN PQ bits exceeds u32"))?,
    );
    put_u32(
        &mut header,
        20,
        u32::try_from(pq.ksub).map_err(|_| invalid_input("DiskANN PQ ksub exceeds u32"))?,
    );
    put_u32(
        &mut header,
        24,
        u32::try_from(pq.chunk_offsets.len())
            .map_err(|_| invalid_input("DiskANN PQ chunk-offset count exceeds u32"))?,
    );
    writer.write_bytes(&header)?;
    for &offset in &pq.chunk_offsets {
        writer.write_bytes(
            &u32::try_from(offset)
                .map_err(|_| invalid_input("DiskANN PQ chunk offset exceeds u32"))?
                .to_le_bytes(),
        )?;
    }
    for &value in &pq.centroids {
        writer.write_bytes(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_row_id_section(
    storage: &RowIdStorage,
    writer: &mut ChunkedSectionWriter<'_>,
) -> io::Result<()> {
    writer.write_bytes(&storage.section_header())?;
    match storage {
        RowIdStorage::Raw(row_ids) => {
            for row_id in row_ids {
                writer.write_bytes(&row_id.to_le_bytes())?;
            }
        }
        RowIdStorage::ForBitPacked { payload, .. } => writer.write_bytes(payload)?,
    }
    Ok(())
}

fn pad_to(out: &mut dyn SeekWrite, target: u64) -> io::Result<()> {
    let padding = target
        .checked_sub(out.pos())
        .ok_or_else(|| invalid_input("DiskANN writer passed a section offset"))?;
    let mut remaining =
        usize::try_from(padding).map_err(|_| invalid_input("DiskANN padding exceeds usize"))?;
    const ZEROES: [u8; 4096] = [0; 4096];
    while remaining != 0 {
        let chunk = remaining.min(ZEROES.len());
        out.write_all(&ZEROES[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

fn resident_peak_bytes(header: &DiskAnnHeader) -> io::Result<usize> {
    let serialized_codebook = section_len_usize(header.sections.codebook, "PQ codebook")?;
    let decoded_codebook = decoded_pq_codebook_bytes(header)?;
    let serialized_row_ids = section_len_usize(header.sections.row_ids, "row IDs")?;
    let decoded_row_ids = resident_row_id_bytes(header)?;
    let pq_codes = section_len_usize(header.sections.pq_codes, "PQ codes")?;
    let serialized_adjacency_index =
        section_len_usize(header.sections.adjacency_index, "adjacency index")?;
    let adjacency_validation = adjacency_validation_bytes(header)?;
    let pq_norms = (header.pq_m as usize)
        .checked_mul(diskann_pq_ksub(header.pq_bits)?)
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or_else(|| invalid_data("DiskANN PQ norms size overflows usize"))?;
    let serialized_payloads = checked_resident_sum([
        serialized_codebook,
        serialized_row_ids,
        pq_codes,
        serialized_adjacency_index,
        1,
    ])?;
    let codebook_decode = serialized_payloads
        .checked_add(decoded_codebook)
        .ok_or_else(|| invalid_data("DiskANN resident peak size overflows usize"))?;
    // `Vec::split_off` temporarily retains both the serialized row-ID
    // allocation and its payload while the raw/packed representation is
    // materialized.
    let row_id_decode = checked_resident_sum([
        serialized_row_ids,
        pq_codes,
        serialized_adjacency_index,
        decoded_codebook,
        decoded_row_ids,
        1,
    ])?;
    let adjacency_decode = checked_resident_sum([
        pq_codes,
        serialized_adjacency_index,
        decoded_codebook,
        decoded_row_ids,
        serialized_adjacency_index,
        1,
    ])?;
    let steady = checked_resident_sum([
        decoded_codebook,
        decoded_row_ids,
        pq_codes,
        pq_norms,
        serialized_adjacency_index,
        adjacency_validation,
    ])?;
    Ok(codebook_decode
        .max(row_id_decode)
        .max(adjacency_decode)
        .max(steady))
}

fn resident_steady_bytes(header: &DiskAnnHeader) -> io::Result<usize> {
    let pq_norms = (header.pq_m as usize)
        .checked_mul(diskann_pq_ksub(header.pq_bits)?)
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or_else(|| invalid_data("DiskANN PQ norms size overflows usize"))?;
    [
        decoded_pq_codebook_bytes(header)?,
        resident_row_id_bytes(header)?,
        section_len_usize(header.sections.pq_codes, "PQ codes")?,
        section_len_usize(header.sections.adjacency_index, "adjacency index")?,
        adjacency_validation_bytes(header)?,
        pq_norms,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| invalid_data("DiskANN resident size overflows usize"))
    })
}

fn checked_resident_sum<const N: usize>(parts: [usize; N]) -> io::Result<usize> {
    parts.into_iter().try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| invalid_data("DiskANN resident size overflows usize"))
    })
}

fn decoded_pq_codebook_bytes(header: &DiskAnnHeader) -> io::Result<usize> {
    let offsets = (header.pq_m as usize)
        .checked_add(1)
        .and_then(|count| count.checked_mul(size_of::<usize>()))
        .ok_or_else(|| invalid_data("DiskANN decoded PQ offset size overflows usize"))?;
    let centroids = (header.dimension as usize)
        .checked_mul(diskann_pq_ksub(header.pq_bits)?)
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or_else(|| invalid_data("DiskANN decoded PQ centroid size overflows usize"))?;
    offsets
        .checked_add(centroids)
        .ok_or_else(|| invalid_data("DiskANN decoded PQ codebook size overflows usize"))
}

fn adjacency_page_count(header: &DiskAnnHeader) -> io::Result<usize> {
    let adjacency_bytes = section_len_usize(header.sections.adjacency, "adjacency")?;
    if !adjacency_bytes.is_multiple_of(DISKANN_PAGE_SIZE as usize) {
        return Err(invalid_data(
            "DiskANN adjacency section is not page-aligned",
        ));
    }
    Ok(adjacency_bytes / DISKANN_PAGE_SIZE as usize)
}

fn adjacency_validation_bytes(header: &DiskAnnHeader) -> io::Result<usize> {
    adjacency_page_count(header)?
        .checked_mul(size_of::<AtomicU8>())
        .ok_or_else(|| invalid_data("DiskANN adjacency validation size overflows usize"))
}

fn resident_row_id_bytes(header: &DiskAnnHeader) -> io::Result<usize> {
    section_len_usize(header.sections.row_ids, "row IDs")?
        .checked_sub(ROW_ID_SECTION_HEADER_SIZE)
        .ok_or_else(|| invalid_data("DiskANN row-ID section header is truncated"))
}

fn row_id_order_peak_bytes(header: &DiskAnnHeader, hot_adjacency: usize) -> io::Result<usize> {
    let order = section_len_usize(header.sections.row_id_order, "row-ID order")?;
    let scratch = order.min(DISKANN_RESIDENT_DECODE_BUFFER_SIZE);
    [
        resident_steady_bytes(header)?,
        hot_adjacency,
        order,
        scratch,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| invalid_data("DiskANN row-ID lookup peak size overflows usize"))
    })
}

fn validate_row_id_order(row_ids: &RowIdStorage, order: &[u32]) -> io::Result<()> {
    if order.len() != row_ids.len() {
        return Err(invalid_data(
            "DiskANN row-ID order has an invalid node count",
        ));
    }
    let mut previous = None;
    for &node in order {
        let node_index = node as usize;
        let row_id = row_ids
            .get(node_index)
            .ok_or_else(|| invalid_data("DiskANN row-ID order node is out of range"))?;
        let key = (row_id, node);
        if previous.is_some_and(|previous| key <= previous) {
            return Err(invalid_data(
                "DiskANN row-ID order must be globally strictly increasing",
            ));
        }
        previous = Some(key);
    }
    Ok(())
}

fn decode_adjacency_index(bytes: &[u8], header: &DiskAnnHeader) -> io::Result<AdjacencyIndex> {
    let section = header.sections.adjacency_index;
    let section_len = section_len_usize(section, "adjacency index")?;
    let locator_count = header.vector_count as usize;
    let expected_len = usize::try_from(adjacency_index_serialized_len(locator_count)?)
        .map_err(|_| invalid_data("DiskANN adjacency index exceeds usize"))?;
    if section_len != expected_len {
        return Err(invalid_data(
            "DiskANN adjacency index has an invalid byte length",
        ));
    }
    let block_count = locator_count.div_ceil(ADJACENCY_LOCATOR_BLOCK_NODES);
    let block_bytes = block_count
        .checked_mul(size_of::<u64>())
        .ok_or_else(|| invalid_data("DiskANN adjacency block-offset size overflows usize"))?;
    let relative_bytes = locator_count
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| invalid_data("DiskANN adjacency relative-offset size overflows usize"))?;
    if bytes.len() != section_len {
        return Err(invalid_data("DiskANN adjacency index payload is truncated"));
    }
    let relative_offset = block_bytes;
    let metadata_offset = relative_offset
        .checked_add(relative_bytes)
        .ok_or_else(|| invalid_data("DiskANN adjacency metadata position overflows"))?;
    let mut block_offsets = Vec::new();
    block_offsets
        .try_reserve_exact(block_count)
        .map_err(|_| invalid_data("DiskANN adjacency block-offset allocation failed"))?;
    block_offsets.extend(bytes[..relative_offset].chunks_exact(8).map(|value| {
        u64::from_le_bytes(value.try_into().expect("validated eight-byte block offset"))
    }));
    let mut relative_offsets = Vec::new();
    relative_offsets
        .try_reserve_exact(locator_count)
        .map_err(|_| invalid_data("DiskANN adjacency relative-offset allocation failed"))?;
    relative_offsets.extend(
        bytes[relative_offset..metadata_offset]
            .chunks_exact(2)
            .map(|value| {
                u16::from_le_bytes(
                    value
                        .try_into()
                        .expect("validated two-byte relative offset"),
                )
            }),
    );
    let mut degree_and_flags = Vec::new();
    degree_and_flags
        .try_reserve_exact(locator_count)
        .map_err(|_| invalid_data("DiskANN adjacency metadata allocation failed"))?;
    degree_and_flags.extend(bytes[metadata_offset..].chunks_exact(2).map(|value| {
        u16::from_le_bytes(
            value
                .try_into()
                .expect("validated two-byte locator metadata"),
        )
    }));
    let index = AdjacencyIndex {
        block_offsets: block_offsets.into_boxed_slice(),
        relative_offsets: relative_offsets.into_boxed_slice(),
        degree_and_flags: degree_and_flags.into_boxed_slice(),
    };
    validate_adjacency_index(header, &index)?;
    Ok(index)
}

fn validate_adjacency_index(header: &DiskAnnHeader, locators: &AdjacencyIndex) -> io::Result<()> {
    if locators.len() != header.vector_count as usize
        || locators.degree_and_flags.len() != locators.len()
        || locators.block_offsets.len() != locators.len().div_ceil(ADJACENCY_LOCATOR_BLOCK_NODES)
    {
        return Err(invalid_data(
            "DiskANN adjacency index has an invalid locator count",
        ));
    }
    let page_count = u32::try_from(header.sections.adjacency.length / DISKANN_PAGE_SIZE as u64)
        .map_err(|_| invalid_data("DiskANN adjacency page count exceeds u32"))?;
    let mut previous_page = 0u32;
    let mut previous_offset = 0usize;
    let mut previous_degree = 0usize;
    let mut previous_encoding = AdjacencyListEncoding::DeltaVarint;
    let record_prefix = if header.is_interleaved() {
        header.vector_record_size as usize
    } else {
        0
    };
    for node in 0..locators.len() {
        if node.is_multiple_of(ADJACENCY_LOCATOR_BLOCK_NODES)
            && locators.relative_offsets[node] != 0
        {
            return Err(invalid_data(
                "DiskANN adjacency locator block must start at relative offset zero",
            ));
        }
        let locator = locators
            .locator(node)
            .ok_or_else(|| invalid_data("DiskANN adjacency locator offset is invalid"))?;
        if locator.page_index >= page_count {
            return Err(invalid_data(
                "DiskANN adjacency locator page is out of range",
            ));
        }
        if node == 0 && locator.page_index != 0 {
            return Err(invalid_data(
                "DiskANN adjacency locator must start on page zero",
            ));
        }
        if locator.page_index < previous_page
            || locator.page_index > previous_page + u32::from(node != 0)
        {
            return Err(invalid_data(
                "DiskANN adjacency locator pages are not monotonic",
            ));
        }
        if locator.degree() as u32 > header.max_degree {
            return Err(invalid_data("DiskANN adjacency degree exceeds maximum"));
        }
        let offset = locator.byte_offset as usize;
        if offset < record_prefix || offset > DISKANN_PAGE_SIZE as usize {
            return Err(invalid_data(
                "DiskANN adjacency locator offset exceeds its page",
            ));
        }
        if locator.encoding() == AdjacencyListEncoding::RawU32 {
            let end = offset
                .checked_add(locator.degree() * size_of::<u32>())
                .ok_or_else(|| invalid_data("DiskANN adjacency locator range overflows"))?;
            if end > DISKANN_PAGE_SIZE as usize {
                return Err(invalid_data("DiskANN adjacency locator crosses a page"));
            }
        }
        if node == 0 && offset != record_prefix {
            return Err(invalid_data(
                "DiskANN adjacency index has an invalid first record offset",
            ));
        }
        if locator.page_index == previous_page {
            if node != 0 {
                match previous_encoding {
                    AdjacencyListEncoding::DeltaVarint => {
                        let minimum_offset =
                            previous_offset + usize::from(previous_degree != 0) + record_prefix;
                        if offset < minimum_offset {
                            return Err(invalid_data("DiskANN adjacency locator ranges overlap"));
                        }
                    }
                    AdjacencyListEncoding::RawU32 => {
                        let expected_offset = previous_offset
                            .checked_add(previous_degree * size_of::<u32>())
                            .and_then(|offset| offset.checked_add(record_prefix))
                            .ok_or_else(|| {
                                invalid_data("DiskANN adjacency locator range overflows")
                            })?;
                        if offset != expected_offset {
                            return Err(invalid_data(
                                "DiskANN adjacency locator ranges must be contiguous",
                            ));
                        }
                    }
                }
            }
        } else {
            if offset != record_prefix {
                return Err(invalid_data(
                    "DiskANN adjacency page has an invalid first record offset",
                ));
            }
        }
        previous_page = locator.page_index;
        previous_offset = offset;
        previous_degree = locator.degree();
        previous_encoding = locator.encoding();
    }
    if previous_page.checked_add(1) != Some(page_count) {
        return Err(invalid_data(
            "DiskANN adjacency pages are not fully indexed",
        ));
    }
    Ok(())
}

fn section_len_usize(section: SectionRange, name: &str) -> io::Result<usize> {
    usize::try_from(section.length)
        .map_err(|_| invalid_data(format!("DiskANN {} section exceeds usize", name)))
}

fn decode_pq_codebook(bytes: &[u8], header: &DiskAnnHeader) -> io::Result<ProductQuantizer> {
    if bytes.len() < PQ_CODEBOOK_HEADER_SIZE {
        return Err(invalid_data("DiskANN PQ codebook header is truncated"));
    }
    if get_u32(bytes, 0) != PQ_CODEBOOK_MAGIC
        || get_u32(bytes, 4) != PQ_CODEBOOK_VERSION
        || get_u32(bytes, 8) != header.dimension
        || get_u32(bytes, 12) != header.pq_m
        || get_u32(bytes, 16) != header.pq_bits
        || get_u32(bytes, 20) != diskann_pq_ksub(header.pq_bits)? as u32
        || get_u32(bytes, 24) != header.pq_m + 1
        || get_u32(bytes, 28) != 0
    {
        return Err(invalid_data("invalid DiskANN PQ codebook metadata"));
    }
    let offset_count = header.pq_m as usize + 1;
    let offsets_bytes = offset_count
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| invalid_data("DiskANN PQ chunk-offset size overflows usize"))?;
    let centroids_offset = PQ_CODEBOOK_HEADER_SIZE
        .checked_add(offsets_bytes)
        .ok_or_else(|| invalid_data("DiskANN PQ codebook offset overflows usize"))?;
    let expected_len = usize::try_from(pq_codebook_serialized_len(
        header.dimension as usize,
        header.pq_m as usize,
        diskann_pq_ksub(header.pq_bits)?,
    )?)
    .map_err(|_| invalid_data("DiskANN PQ codebook size exceeds usize"))?;
    if bytes.len() != expected_len {
        return Err(invalid_data("invalid DiskANN PQ codebook length"));
    }
    let mut chunk_offsets = Vec::new();
    chunk_offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| invalid_data("DiskANN PQ chunk-offset allocation failed"))?;
    for encoded in bytes[PQ_CODEBOOK_HEADER_SIZE..centroids_offset].chunks_exact(4) {
        chunk_offsets.push(u32::from_le_bytes(
            encoded.try_into().expect("four-byte PQ chunk offset"),
        ) as usize);
    }
    let mut pq = ProductQuantizer::try_with_chunk_offsets(
        header.dimension as usize,
        header.pq_bits as usize,
        chunk_offsets,
    )
    .map_err(invalid_data)?;
    let mut centroids = Vec::new();
    centroids
        .try_reserve_exact(header.dimension as usize * pq.ksub)
        .map_err(|_| invalid_data("DiskANN PQ centroid allocation failed"))?;
    for encoded in bytes[centroids_offset..].chunks_exact(4) {
        let value =
            f32::from_le_bytes(encoded.try_into().expect("validated four-byte PQ centroid"));
        if !value.is_finite() {
            return Err(invalid_data("DiskANN PQ centroids must be finite"));
        }
        centroids.push(value);
    }
    pq.centroids = centroids;
    if !pq.has_valid_layout() {
        return Err(invalid_data("invalid DiskANN PQ codebook layout"));
    }
    Ok(pq)
}

fn decode_row_id_section_owned(
    mut bytes: Vec<u8>,
    expected_count: usize,
) -> io::Result<RowIdStorage> {
    let header = decode_row_id_section_header(&bytes, bytes.len(), expected_count)?;
    let payload = bytes.split_off(ROW_ID_SECTION_HEADER_SIZE);
    drop(bytes);
    let storage = match header.encoding {
        ROW_ID_ENCODING_RAW_I64 => {
            let mut row_ids = Vec::new();
            row_ids
                .try_reserve_exact(header.count)
                .map_err(|_| invalid_data("DiskANN raw row-ID allocation failed"))?;
            row_ids.extend(payload.chunks_exact(8).map(|value| {
                i64::from_le_bytes(value.try_into().expect("validated eight-byte row ID"))
            }));
            RowIdStorage::Raw(row_ids)
        }
        ROW_ID_ENCODING_FOR_BITPACK => RowIdStorage::ForBitPacked {
            base: header.base,
            bit_width: header.bit_width,
            count: header.count,
            payload,
        },
        _ => unreachable!("row-ID encoding was validated"),
    };
    validate_row_id_storage(&storage)?;
    Ok(storage)
}

fn read_u32_section<R: SeekRead>(
    reader: &mut R,
    section: SectionRange,
    name: &str,
) -> io::Result<Vec<u32>> {
    let section_len = section_len_usize(section, name)?;
    if !section_len.is_multiple_of(size_of::<u32>()) {
        return Err(invalid_data(format!(
            "DiskANN {} section is not u32-aligned",
            name
        )));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(section_len / size_of::<u32>())
        .map_err(|_| invalid_data(format!("DiskANN {} allocation failed", name)))?;
    let scratch_len = section_len
        .min(DISKANN_RESIDENT_DECODE_BUFFER_SIZE)
        .max(size_of::<u32>());
    let mut scratch = vec![0u8; scratch_len];
    let mut loaded = 0usize;
    while loaded < section_len {
        let chunk_len = (section_len - loaded).min(scratch.len());
        let chunk = &mut scratch[..chunk_len];
        reader
            .pread(&mut [ReadRequest::new(section.offset + loaded as u64, chunk)])
            .map_err(|error| map_read_error(error, name))?;
        values.extend(
            chunk.chunks_exact(4).map(|bytes| {
                u32::from_le_bytes(bytes.try_into().expect("validated four-byte value"))
            }),
        );
        loaded += chunk_len;
    }
    Ok(values)
}

fn read_resident_sections<R: SeekRead>(
    reader: &mut R,
    header: &DiskAnnHeader,
) -> io::Result<(ProductQuantizer, RowIdStorage, Vec<u8>, AdjacencyIndex)> {
    let sections = [
        (header.sections.codebook, "PQ codebook"),
        (header.sections.row_ids, "row IDs"),
        (header.sections.pq_codes, "PQ codes"),
        (header.sections.adjacency_index, "adjacency index"),
        (
            SectionRange::new(header.file_len - 1, 1),
            "file length probe",
        ),
    ];
    let mut payloads = Vec::with_capacity(sections.len());
    for (section, name) in sections {
        let len = section_len_usize(section, name)?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(len)
            .map_err(|_| invalid_data(format!("DiskANN {name} allocation failed")))?;
        payload.resize(len, 0);
        payloads.push(payload);
    }
    let max_ranges = match reader.read_capabilities().max_ranges_per_pread {
        0 => usize::MAX,
        value => value,
    };
    let mut start = 0usize;
    while start < payloads.len() {
        let end = payloads.len().min(start.saturating_add(max_ranges));
        let mut requests = sections[start..end]
            .iter()
            .zip(&mut payloads[start..end])
            .map(|((section, _), payload)| ReadRequest::new(section.offset, payload))
            .collect::<Vec<_>>();
        reader
            .pread(&mut requests)
            .map_err(|error| map_read_error(error, "resident sections"))?;
        start = end;
    }
    let mut payloads = payloads.into_iter();
    let pq = decode_pq_codebook(
        &payloads.next().expect("five resident section payloads"),
        header,
    )?;
    let row_ids = decode_row_id_section_owned(
        payloads.next().expect("five resident section payloads"),
        header.vector_count as usize,
    )?;
    let pq_codes = payloads.next().expect("five resident section payloads");
    let adjacency_index = decode_adjacency_index(
        &payloads.next().expect("five resident section payloads"),
        header,
    )?;
    let _file_tail = payloads.next().expect("file length probe payload");
    Ok((pq, row_ids, pq_codes, adjacency_index))
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
fn get_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("fixed u16 field"),
    )
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed u32 field"),
    )
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed u64 field"),
    )
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn map_read_error(error: io::Error, section: &str) -> io::Error {
    if error.kind() == io::ErrorKind::UnexpectedEof {
        invalid_data(format!("DiskANN {} is truncated", section))
    } else {
        error
    }
}

fn checked_mul_u64(left: u64, right: u64, name: &str) -> io::Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_input(format!("DiskANN {} length overflows u64", name)))
}

fn diskann_pq_ksub(pq_bits: u32) -> io::Result<usize> {
    match pq_bits {
        4 => Ok(16),
        8 => Ok(256),
        _ => Err(invalid_input("DiskANN pq.bits must be 4 or 8")),
    }
}

fn diskann_pq_code_size(pq_m: usize, pq_bits: u32) -> io::Result<usize> {
    match pq_bits {
        4 => Ok(pq_m.div_ceil(2)),
        8 => Ok(pq_m),
        _ => Err(invalid_input("DiskANN pq.bits must be 4 or 8")),
    }
}

fn pq_codebook_serialized_len(dimension: usize, pq_m: usize, ksub: usize) -> io::Result<u64> {
    let offsets = pq_m
        .checked_add(1)
        .and_then(|count| count.checked_mul(size_of::<u32>()))
        .ok_or_else(|| invalid_input("DiskANN PQ chunk-offset size overflows usize"))?;
    let centroids = dimension
        .checked_mul(ksub)
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or_else(|| invalid_input("DiskANN PQ centroid size overflows usize"))?;
    u64::try_from(
        PQ_CODEBOOK_HEADER_SIZE
            .checked_add(offsets)
            .and_then(|bytes| bytes.checked_add(centroids))
            .ok_or_else(|| invalid_input("DiskANN PQ codebook size overflows usize"))?,
    )
    .map_err(|_| invalid_input("DiskANN PQ codebook size exceeds u64"))
}

fn validate_pq_code_padding(header: &DiskAnnHeader, codes: &[u8]) -> io::Result<()> {
    if header.pq_bits != 4 || header.pq_m.is_multiple_of(2) {
        return Ok(());
    }
    let code_size = diskann_pq_code_size(header.pq_m as usize, header.pq_bits)?;
    if codes
        .chunks_exact(code_size)
        .any(|code| code.last().is_some_and(|byte| byte & 0xf0 != 0))
    {
        return Err(invalid_data(
            "DiskANN odd 4-bit PQ codes must zero the unused high nibble",
        ));
    }
    Ok(())
}

fn section_end(section: SectionRange) -> io::Result<u64> {
    section
        .offset
        .checked_add(section.length)
        .ok_or_else(|| invalid_input("DiskANN section end overflows u64"))
}

fn align_up(value: u64, alignment: u64) -> io::Result<u64> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| invalid_input("DiskANN section alignment overflows u64"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskann::{DiskAnnBuildParams, DiskAnnIndex, DiskAnnStorageLayout};
    use crate::distance::MetricType;
    use crate::io::{PosWriter, ReadRequest, SeekRead};
    use crate::read_options::StorageProfile;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn diskann_shared_adjacency_cache_reports_lock_acquisitions() {
        let cache = SharedWindowCache::new(4096);

        let (lookup, metrics) = cache.lookup_or_reserve(7, 4).unwrap();

        assert!(matches!(lookup, SharedWindowCacheLookup::Reserved));
        assert_eq!(metrics.acquisitions, 1);
        cache.cancel(&[7]).unwrap();
    }

    #[test]
    fn diskann_shared_adjacency_cache_shards_production_sized_budgets() {
        let cache = SharedWindowCache::new(16 * 1024 * 1024);

        assert_eq!(cache.shard_count(), 16);
    }

    #[test]
    fn diskann_shared_window_cache_hashes_coalesced_offsets_across_all_shards() {
        let cache = SharedWindowCache::new(16 * 1024 * 1024);
        let shard_indexes = (0..80)
            .map(|window| cache.shard_index(window * 64 * 1024))
            .collect::<HashSet<_>>();

        assert_eq!(shard_indexes.len(), SHARED_WINDOW_CACHE_SHARDS);
    }

    #[test]
    fn diskann_shared_adjacency_cache_keeps_one_full_coalesced_window_per_shard() {
        let cache = SharedWindowCache::new(64 * 1024);

        assert_eq!(cache.shard_count(), 1);
    }

    #[test]
    fn diskann_shared_adjacency_cache_is_bounded_and_lru() {
        let cache = SharedWindowCache::new(8);
        assert!(matches!(
            cache.lookup_or_reserve(0, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));
        assert_eq!(cache.publish(0, Arc::new(vec![0u8; 4])).unwrap().0, 0);
        assert!(matches!(
            cache.lookup_or_reserve(4, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));
        assert_eq!(cache.publish(4, Arc::new(vec![1u8; 4])).unwrap().0, 0);
        assert!(matches!(
            cache.lookup_or_reserve(0, 4).unwrap().0,
            SharedWindowCacheLookup::Hit(_)
        ));
        assert!(matches!(
            cache.lookup_or_reserve(8, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));

        assert_eq!(cache.publish(8, Arc::new(vec![2u8; 4])).unwrap().0, 1);
        assert!(matches!(
            cache.lookup_or_reserve(0, 4).unwrap().0,
            SharedWindowCacheLookup::Hit(_)
        ));
        assert!(matches!(
            cache.lookup_or_reserve(4, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));
        cache.cancel(&[4]).unwrap();
    }

    #[test]
    fn diskann_shared_adjacency_cache_charges_allocation_capacity() {
        let cache = SharedWindowCache::new(8);
        assert!(matches!(
            cache.lookup_or_reserve(0, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));
        let mut payload = Vec::with_capacity(64);
        payload.resize(4, 0);

        assert_eq!(cache.publish(0, Arc::new(payload)).unwrap().0, 1);
        assert!(matches!(
            cache.lookup_or_reserve(0, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));
        cache.cancel(&[0]).unwrap();
    }

    #[test]
    fn diskann_offset_lru_supports_constant_time_oldest_eviction() {
        let mut lru = OffsetLru::default();
        lru.touch(10);
        lru.touch(20);
        lru.touch(30);
        lru.touch(10);
        assert_eq!(lru.oldest_offsets(), vec![20, 30, 10]);

        lru.remove(30);
        assert_eq!(lru.pop_oldest(), Some(20));
        assert_eq!(lru.pop_oldest(), Some(10));
        assert_eq!(lru.pop_oldest(), None);
    }

    #[test]
    fn diskann_shared_adjacency_cache_waits_for_one_loader() {
        let cache = Arc::new(SharedWindowCache::new(4096));
        assert!(matches!(
            cache.lookup_or_reserve(7, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));
        assert!(matches!(
            cache.lookup_or_reserve(7, 4).unwrap().0,
            SharedWindowCacheLookup::Loading
        ));
        let (started_tx, started_rx) = mpsc::channel();
        let waiter_cache = Arc::clone(&cache);
        let waiter = thread::spawn(move || {
            started_tx.send(()).unwrap();
            waiter_cache.wait_for(7, 4).unwrap().0.unwrap()
        });
        started_rx.recv().unwrap();

        cache.publish(7, Arc::new(vec![3u8; 4])).unwrap();

        assert_eq!(&*waiter.join().unwrap(), &[3u8; 4]);
    }

    #[test]
    fn diskann_shared_adjacency_cache_cancel_releases_waiters() {
        let cache = SharedWindowCache::new(4096);
        assert!(matches!(
            cache.lookup_or_reserve(7, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));

        cache.cancel(&[7]).unwrap();

        assert!(cache.wait_for(7, 4).unwrap().0.is_none());
        assert!(matches!(
            cache.lookup_or_reserve(7, 4).unwrap().0,
            SharedWindowCacheLookup::Reserved
        ));
        cache.cancel(&[7]).unwrap();
    }

    #[test]
    fn diskann_shared_adjacency_validation_is_single_flight() {
        let cache = Arc::new(AdjacencyValidationCache::new(1).unwrap());
        let validation_count = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let first_cache = Arc::clone(&cache);
        let first_count = Arc::clone(&validation_count);
        let first = thread::spawn(move || {
            first_cache
                .get_or_validate(0, || {
                    first_count.fetch_add(1, AtomicOrdering::SeqCst);
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .unwrap();
        });
        started_rx.recv().unwrap();

        let second_cache = Arc::clone(&cache);
        let second_count = Arc::clone(&validation_count);
        let second = thread::spawn(move || {
            second_cache
                .get_or_validate(0, || {
                    second_count.fetch_add(1, AtomicOrdering::SeqCst);
                    Ok(())
                })
                .unwrap();
        });

        release_tx.send(()).unwrap();
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(validation_count.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn diskann_shared_adjacency_validation_caches_failures() {
        let cache = AdjacencyValidationCache::new(1).unwrap();
        let validation_count = AtomicUsize::new(0);

        let first = cache
            .get_or_validate(0, || {
                validation_count.fetch_add(1, AtomicOrdering::SeqCst);
                Err(invalid_data("corrupt immutable adjacency page"))
            })
            .expect_err("corrupt page must fail validation");
        let second = cache
            .get_or_validate(0, || {
                validation_count.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(())
            })
            .expect_err("cached corrupt page must keep failing");

        assert_eq!(validation_count.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(second.kind(), first.kind());
        assert_eq!(second.to_string(), first.to_string());
    }

    #[derive(Clone)]
    struct RecordingReader {
        inner: Cursor<Vec<u8>>,
        reads: Arc<Mutex<Vec<(u64, usize)>>>,
    }

    #[derive(Clone)]
    struct RoundRecordingReader {
        inner: Cursor<Vec<u8>>,
        rounds: ReadRounds,
        max_ranges_per_pread: usize,
    }

    type ReadRounds = Arc<Mutex<Vec<Vec<(u64, usize)>>>>;

    #[derive(Clone)]
    struct ProfileHintReader {
        inner: Cursor<Vec<u8>>,
        hint: StorageProfile,
        hint_calls: Arc<AtomicUsize>,
    }

    #[derive(Default)]
    struct CountingWriter {
        bytes: Vec<u8>,
        write_calls: usize,
        write_lengths: Vec<usize>,
    }

    impl SeekWrite for CountingWriter {
        fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
            self.write_calls += 1;
            self.write_lengths.push(buf.len());
            self.bytes.extend_from_slice(buf);
            Ok(())
        }

        fn pos(&self) -> u64 {
            self.bytes.len() as u64
        }
    }

    impl SeekRead for RecordingReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            for range in ranges {
                self.reads
                    .lock()
                    .unwrap()
                    .push((range.pos, range.buf.len()));
                self.inner.set_position(range.pos);
                io::Read::read_exact(&mut self.inner, range.buf)?;
            }
            Ok(())
        }

        fn try_clone_reader(&self) -> io::Result<Option<Self>> {
            Ok(Some(self.clone()))
        }
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

        fn try_clone_reader(&self) -> io::Result<Option<Self>> {
            Ok(Some(self.clone()))
        }

        fn read_capabilities(&self) -> SeekReadCapabilities {
            SeekReadCapabilities {
                max_ranges_per_pread: self.max_ranges_per_pread,
                ..SeekReadCapabilities::default()
            }
        }
    }

    impl SeekRead for ProfileHintReader {
        fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
            for range in ranges {
                self.inner.set_position(range.pos);
                io::Read::read_exact(&mut self.inner, range.buf)?;
            }
            Ok(())
        }

        fn preferred_storage_profile(&self) -> Option<StorageProfile> {
            self.hint_calls.fetch_add(1, AtomicOrdering::SeqCst);
            Some(self.hint)
        }

        fn try_clone_reader(&self) -> io::Result<Option<Self>> {
            Ok(Some(self.clone()))
        }
    }

    fn initialize_empty_adjacency_index(bytes: &mut [u8], header: &DiskAnnHeader) {
        let codebook = header.sections.codebook.offset as usize;
        put_u32(bytes, codebook, PQ_CODEBOOK_MAGIC);
        put_u32(bytes, codebook + 4, PQ_CODEBOOK_VERSION);
        put_u32(bytes, codebook + 8, header.dimension);
        put_u32(bytes, codebook + 12, header.pq_m);
        put_u32(bytes, codebook + 16, header.pq_bits);
        put_u32(bytes, codebook + 20, 1 << header.pq_bits);
        put_u32(bytes, codebook + 24, header.pq_m + 1);
        let dimension = header.dimension as usize;
        let pq_m = header.pq_m as usize;
        let wide_chunks = dimension % pq_m;
        let narrow_width = dimension / pq_m;
        let mut chunk_offset = 0usize;
        for chunk in 0..=pq_m {
            put_u32(
                bytes,
                codebook + PQ_CODEBOOK_HEADER_SIZE + chunk * size_of::<u32>(),
                chunk_offset as u32,
            );
            if chunk < pq_m {
                chunk_offset += narrow_width + usize::from(chunk < wide_chunks);
            }
        }

        let row_ids = header.sections.row_ids.offset as usize;
        put_u32(bytes, row_ids, ROW_ID_ENCODING_RAW_I64);
        put_u32(bytes, row_ids + 4, u64::BITS);
        put_u64(bytes, row_ids + 8, header.vector_count);
        let page_count = (header.sections.adjacency.length / DISKANN_PAGE_SIZE as u64) as usize;
        let base = header.sections.adjacency_index.offset as usize;
        let vector_count = header.vector_count as usize;
        let block_count = vector_count.div_ceil(ADJACENCY_LOCATOR_BLOCK_NODES);
        let relative_base = base + block_count * size_of::<u64>();
        let metadata_base = relative_base + vector_count * size_of::<u16>();
        for block in 0..block_count {
            let node = block * ADJACENCY_LOCATOR_BLOCK_NODES;
            let block_offset = node.min(page_count - 1) as u64 * u64::from(DISKANN_PAGE_SIZE);
            put_u64(bytes, base + block * size_of::<u64>(), block_offset);
        }
        for node in 0..header.vector_count as usize {
            let block_node = node / ADJACENCY_LOCATOR_BLOCK_NODES * ADJACENCY_LOCATOR_BLOCK_NODES;
            let block_page = block_node.min(page_count - 1);
            let page = node.min(page_count - 1);
            let relative_offset = (page - block_page) * DISKANN_PAGE_SIZE as usize;
            put_u16(
                bytes,
                relative_base + node * size_of::<u16>(),
                relative_offset as u16,
            );
            put_u16(bytes, metadata_base + node * size_of::<u16>(), 0);
        }
    }

    fn serialized_adjacency_locator(
        bytes: &[u8],
        header: &DiskAnnHeader,
        node: usize,
    ) -> AdjacencyLocator {
        let vector_count = header.vector_count as usize;
        let block_count = vector_count.div_ceil(ADJACENCY_LOCATOR_BLOCK_NODES);
        let base = header.sections.adjacency_index.offset as usize;
        let relative_base = base + block_count * size_of::<u64>();
        let metadata_base = relative_base + vector_count * size_of::<u16>();
        let block_offset = get_u64(
            bytes,
            base + node / ADJACENCY_LOCATOR_BLOCK_NODES * size_of::<u64>(),
        );
        let relative_offset = u64::from(get_u16(bytes, relative_base + node * size_of::<u16>()));
        let absolute_offset = block_offset + relative_offset;
        AdjacencyLocator {
            page_index: (absolute_offset / u64::from(DISKANN_PAGE_SIZE)) as u32,
            byte_offset: (absolute_offset % u64::from(DISKANN_PAGE_SIZE)) as u16,
            degree_and_flags: get_u16(bytes, metadata_base + node * size_of::<u16>()),
        }
    }

    #[test]
    fn diskann_block_locator_index_roundtrips_block_and_u16_boundaries() {
        let mut locators = (0..17)
            .map(|node| {
                AdjacencyLocator::new(
                    node.min(16) as u32,
                    0,
                    node % 5,
                    if node % 2 == 0 {
                        AdjacencyListEncoding::DeltaVarint
                    } else {
                        AdjacencyListEncoding::RawU32
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        locators[15].byte_offset = DISKANN_PAGE_SIZE as u16 - 1;

        let index = AdjacencyIndex::from_locators(&locators).unwrap();

        assert_eq!(adjacency_index_serialized_len(locators.len()).unwrap(), 84);
        assert_eq!(
            index.block_offsets.as_ref(),
            &[0, 16 * DISKANN_PAGE_SIZE as u64]
        );
        assert_eq!(index.relative_offsets[15], u16::MAX);
        assert_eq!(index.len(), locators.len());
        for (node, expected) in locators.iter().copied().enumerate() {
            assert_eq!(index.locator(node), Some(expected));
        }
        assert_eq!(index.locator(locators.len()), None);
    }

    #[test]
    fn diskann_adjacency_codec_selects_the_smaller_encoding_and_roundtrips() {
        let compact = [100, 101, 105, 110];
        assert_eq!(
            plan_adjacency_list(&compact).unwrap(),
            (AdjacencyListEncoding::DeltaVarint, 4)
        );
        let mut encoded = Vec::new();
        let encoding = encode_adjacency_list(&compact, &mut encoded).unwrap();
        assert_eq!(encoding, AdjacencyListEncoding::DeltaVarint);
        assert_eq!(encoded, [100, 1, 4, 5]);
        let mut decoded = Vec::new();
        let consumed =
            decode_adjacency_list(&encoded, compact.len(), encoding, &mut decoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, compact);

        let sparse = [1 << 31, u32::MAX];
        assert_eq!(
            plan_adjacency_list(&sparse).unwrap(),
            (AdjacencyListEncoding::RawU32, 8)
        );
        let encoding = encode_adjacency_list(&sparse, &mut encoded).unwrap();
        assert_eq!(encoding, AdjacencyListEncoding::RawU32);
        assert_eq!(
            encoded,
            [
                0x00, 0x00, 0x00, 0x80, // 1 << 31
                0xff, 0xff, 0xff, 0xff, // u32::MAX
            ]
        );
        let consumed =
            decode_adjacency_list(&encoded, sparse.len(), encoding, &mut decoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, sparse);
    }

    #[test]
    fn diskann_adjacency_delta_varint_rejects_corrupt_encodings() {
        let mut decoded = Vec::new();
        let truncated =
            decode_adjacency_list(&[0x80], 1, AdjacencyListEncoding::DeltaVarint, &mut decoded)
                .unwrap_err();
        assert!(truncated.to_string().contains("truncated"));

        let non_canonical = decode_adjacency_list(
            &[0x80, 0x00],
            1,
            AdjacencyListEncoding::DeltaVarint,
            &mut decoded,
        )
        .unwrap_err();
        assert!(non_canonical.to_string().contains("canonical"));

        let exceeds_u32 = decode_adjacency_list(
            &[0xff, 0xff, 0xff, 0xff, 0x10],
            1,
            AdjacencyListEncoding::DeltaVarint,
            &mut decoded,
        )
        .unwrap_err();
        assert!(exceeds_u32.to_string().contains("exceeds u32"));

        let delta_overflow = decode_adjacency_list(
            &[0xff, 0xff, 0xff, 0xff, 0x0f, 0x01],
            2,
            AdjacencyListEncoding::DeltaVarint,
            &mut decoded,
        )
        .unwrap_err();
        assert!(delta_overflow.to_string().contains("overflows u32"));
    }

    #[test]
    fn diskann_writer_uses_adaptive_adjacency_format_and_fewer_pages() {
        let dimension = 8;
        let count = 512;
        let data = (0..count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (0..count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 16,
                build_search_list_size: 32,
                seed: 73,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);

        let prepared = index.prepare_build().unwrap();
        let mut fixed_pages = 1usize;
        let mut fixed_offset = 0usize;
        for neighbors in prepared.graph.adjacency.iter() {
            let bytes = size_of_val(neighbors);
            if fixed_offset + bytes > DISKANN_PAGE_SIZE as usize {
                fixed_pages += 1;
                fixed_offset = 0;
            }
            fixed_offset += bytes;
        }

        let mut bytes = Vec::new();
        let stats =
            write_diskann_index_with_stats(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = DiskAnnHeader::decode(&bytes[..DISKANN_HEADER_SIZE]).unwrap();
        let adaptive_pages = header.sections.adjacency.length as usize / DISKANN_PAGE_SIZE as usize;

        assert!(stats.accounted_duration() <= stats.total);
        assert_eq!(header.adjacency_locator_encoding, 3);
        assert!(
            adaptive_pages < fixed_pages,
            "adaptive={adaptive_pages}, fixed={fixed_pages}"
        );
    }

    #[test]
    fn diskann_row_id_codec_roundtrips_random_access() {
        let cases = [
            (0, vec![5, 5, 5]),
            (1, vec![-5, -4, -5]),
            (7, vec![-64, 63, 0]),
            (8, vec![-64, 64, 0]),
            (9, vec![-64, 192, 7]),
            (20, vec![-7, (1_i64 << 19) - 7, 42]),
            (63, vec![i64::MIN, -1, i64::MIN]),
        ];

        for (expected_width, row_ids) in cases {
            let storage = RowIdStorage::encode(row_ids.clone()).unwrap();
            assert_eq!(storage.bit_width(), expected_width);
            assert_eq!(storage.len(), row_ids.len());
            assert_eq!(
                (0..row_ids.len())
                    .map(|node| storage.get(node).unwrap())
                    .collect::<Vec<_>>(),
                row_ids
            );
            assert_eq!(storage.get(row_ids.len()), None);
        }
    }

    #[test]
    fn diskann_row_id_sequential_decoder_handles_raw_and_packed_widths() {
        let cases = [
            vec![5, 5, 5],
            vec![-5, -4, -5],
            vec![-64, 63, 0],
            vec![-64, 64, 0],
            vec![-64, 192, 7],
            vec![-7, (1_i64 << 19) - 7, 42],
            vec![i64::MIN, -1, i64::MIN],
            vec![i64::MIN, 0, i64::MAX],
        ];

        for row_ids in cases {
            let storage = RowIdStorage::encode(row_ids.clone()).unwrap();
            let mut decoded = Vec::new();

            storage
                .try_for_each(|node, row_id| {
                    decoded.push((node, row_id));
                    Ok(())
                })
                .unwrap();

            assert_eq!(decoded, row_ids.into_iter().enumerate().collect::<Vec<_>>());
        }
    }

    #[test]
    fn diskann_row_id_codec_uses_raw_fallback_for_full_width_span() {
        let row_ids = [i64::MIN, 0, i64::MAX];

        let storage = RowIdStorage::encode(row_ids.to_vec()).unwrap();

        assert!(matches!(storage, RowIdStorage::Raw(_)));
        assert_eq!(storage.bit_width(), 64);
        assert_eq!(storage.get(0), Some(i64::MIN));
        assert_eq!(storage.get(2), Some(i64::MAX));
    }

    #[test]
    fn diskann_row_id_section_roundtrips_self_describing_encoding() {
        let storage = RowIdStorage::encode(vec![-7, 0, 17, -7]).unwrap();

        let bytes = encode_row_id_section(&storage).unwrap();
        let decoded = decode_row_id_section(&bytes, 4).unwrap();

        assert_eq!(get_u32(&bytes, 0), ROW_ID_ENCODING_FOR_BITPACK);
        assert_eq!(get_u32(&bytes, 4), 5);
        assert_eq!(get_u64(&bytes, 8), 4);
        assert_eq!(get_u64(&bytes, 16) as i64, -7);
        assert!(bytes[24..32].iter().all(|&byte| byte == 0));
        assert_eq!(decoded, storage);
    }

    #[test]
    fn diskann_row_id_section_rejects_corrupt_metadata_and_tail_bits() {
        let storage = RowIdStorage::encode(vec![10, 11, 10]).unwrap();
        let valid = encode_row_id_section(&storage).unwrap();
        let mut corruptions = Vec::new();

        let mut unknown_encoding = valid.clone();
        put_u32(&mut unknown_encoding, 0, 99);
        corruptions.push(unknown_encoding);
        let mut invalid_width = valid.clone();
        put_u32(&mut invalid_width, 4, 64);
        corruptions.push(invalid_width);
        let mut invalid_count = valid.clone();
        put_u64(&mut invalid_count, 8, 4);
        corruptions.push(invalid_count);
        let mut nonzero_reserved = valid.clone();
        nonzero_reserved[24] = 1;
        corruptions.push(nonzero_reserved);
        let mut nonzero_tail = valid.clone();
        *nonzero_tail.last_mut().unwrap() |= 0x80;
        corruptions.push(nonzero_tail);

        for bytes in corruptions {
            let error = decode_row_id_section(&bytes, 3)
                .expect_err("corrupt row-ID metadata must fail closed");
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn diskann_raw_row_id_section_roundtrips_and_validates_exact_layout() {
        let storage = RowIdStorage::encode(vec![i64::MIN, 0, i64::MAX]).unwrap();
        let valid = encode_row_id_section(&storage).unwrap();

        assert_eq!(get_u32(&valid, 0), ROW_ID_ENCODING_RAW_I64);
        assert_eq!(get_u32(&valid, 4), 64);
        assert_eq!(decode_row_id_section(&valid, 3).unwrap(), storage);

        let mut nonzero_base = valid.clone();
        put_u64(&mut nonzero_base, 16, 1);
        assert!(decode_row_id_section(&nonzero_base, 3).is_err());
        assert!(decode_row_id_section(&valid[..valid.len() - 1], 3).is_err());
        let mut extended = valid;
        extended.push(0);
        assert!(decode_row_id_section(&extended, 3).is_err());
    }

    #[test]
    fn diskann_for_row_id_section_rejects_decoded_i64_overflow() {
        let storage = RowIdStorage::encode(vec![i64::MAX - 1, i64::MAX]).unwrap();
        let mut bytes = encode_row_id_section(&storage).unwrap();
        put_u64(&mut bytes, 16, i64::MAX as u64);

        let error = decode_row_id_section(&bytes, 2)
            .expect_err("base plus packed delta above i64::MAX must fail closed");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("overflows"));
    }

    #[test]
    fn diskann_header_roundtrips_fixed_256_byte_layout() {
        let header = DiskAnnHeader {
            flags: DISKANN_REQUIRED_FLAGS | FLAG_SEPARATE_ADJACENCY_AND_VECTORS,
            dimension: 128,
            metric: 0,
            vector_count: 1,
            entry_node: 0,
            max_degree: 64,
            build_search_list_size: 100,
            alpha: 1.2,
            seed: 42,
            pq_m: 16,
            pq_bits: 8,
            page_size: 4096,
            adjacency_locator_size: 4,
            adjacency_locator_encoding: 3,
            raw_vector_encoding: DiskAnnRawVectorEncoding::F32 as u32,
            vector_record_size: 512,
            file_len: 143872,
            sections: DiskAnnSections {
                codebook: SectionRange::new(4096, 131172),
                row_ids: SectionRange::new(135268, 40),
                pq_codes: SectionRange::new(135308, 16),
                row_id_order: SectionRange::new(135324, 4),
                adjacency_index: SectionRange::new(135328, 12),
                adjacency: SectionRange::new(139264, 4096),
                vectors: SectionRange::new(143360, 512),
            },
        };

        let encoded = header.encode();
        assert_eq!(encoded.len(), 256);
        assert_eq!(&encoded[..4], b"DANN");
        assert!(encoded[208..].iter().all(|&byte| byte == 0));
        assert_eq!(DiskAnnHeader::decode(&encoded).unwrap(), header);
    }

    #[test]
    fn diskann_header_accepts_public_metrics_and_rejects_unknown_codes() {
        let header = DiskAnnHeader::for_layout(8, 2, 0, 2, DiskAnnBuildParams::default()).unwrap();
        for metric in [MetricType::L2, MetricType::InnerProduct, MetricType::Cosine] {
            let mut bytes = header.encode();
            put_u32(&mut bytes, 20, metric as u32);
            assert_eq!(DiskAnnHeader::decode(&bytes).unwrap().metric_type(), metric);
        }
        let mut bytes = header.encode();
        put_u32(&mut bytes, 20, 3);
        assert!(DiskAnnHeader::decode(&bytes).is_err());
    }

    #[test]
    fn diskann_header_constructor_rejects_layouts_the_reader_would_reject() {
        let invalid_builds = [
            DiskAnnBuildParams {
                max_degree: 2,
                build_search_list_size: 1,
                ..DiskAnnBuildParams::default()
            },
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 1,
                alpha: 0.5,
                ..DiskAnnBuildParams::default()
            },
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 1,
                alpha: f32::NAN,
                ..DiskAnnBuildParams::default()
            },
        ];
        for build in invalid_builds {
            assert!(
                DiskAnnHeader::for_layout(8, 2, 0, 2, build).is_err(),
                "header constructor must enforce reader build-parameter invariants"
            );
        }
        assert!(
            DiskAnnHeader::for_layout(8, 2, 2, 2, DiskAnnBuildParams::default()).is_err(),
            "entry node must be in range"
        );
        assert!(
            DiskAnnHeader::for_layout(8, 2, 0, 9, DiskAnnBuildParams::default()).is_err(),
            "PQ shape must be valid"
        );
    }

    #[test]
    fn diskann_writer_rejects_non_finite_persisted_values_before_output() {
        fn one_vector_index() -> DiskAnnIndex {
            let mut index = DiskAnnIndex::new(
                1,
                MetricType::L2,
                1,
                DiskAnnBuildParams {
                    max_degree: 1,
                    build_search_list_size: 1,
                    ..DiskAnnBuildParams::default()
                },
            );
            index.pq.centroids = (0..256).map(|code| code as f32).collect();
            index.pq.rebuild_norms_cache();
            index.ids = vec![7];
            index.vectors = vec![0.0];
            index
        }

        let mut invalid_codebook = one_vector_index();
        invalid_codebook.pq.centroids[0] = f32::NAN;
        let mut codebook_output = Vec::new();
        assert!(
            write_diskann_index(&invalid_codebook, &mut PosWriter::new(&mut codebook_output))
                .is_err()
        );
        assert!(codebook_output.is_empty());

        let mut invalid_vector = one_vector_index();
        invalid_vector.vectors[0] = f32::INFINITY;
        let mut vector_output = Vec::new();
        assert!(
            write_diskann_index(&invalid_vector, &mut PosWriter::new(&mut vector_output)).is_err()
        );
        assert!(vector_output.is_empty());

        let mut f16_overflow = one_vector_index();
        f16_overflow.build_params.raw_vector_encoding = DiskAnnRawVectorEncoding::F16;
        f16_overflow.vectors[0] = 70_000.0;
        let mut f16_output = Vec::new();
        let error = write_diskann_index(&f16_overflow, &mut PosWriter::new(&mut f16_output))
            .expect_err("finite f32 outside binary16 range must be rejected");
        assert!(error.to_string().contains("finite f16 range"));
        assert!(f16_output.is_empty());

        let mut invalid_pq_shape = one_vector_index();
        invalid_pq_shape.pq.chunk_offsets[1] = 0;
        let mut pq_shape_output = Vec::new();
        assert!(
            write_diskann_index(&invalid_pq_shape, &mut PosWriter::new(&mut pq_shape_output))
                .is_err()
        );
        assert!(pq_shape_output.is_empty());

        let mut invalid_build = one_vector_index();
        invalid_build.build_params.build_search_list_size = 0;
        let mut build_output = Vec::new();
        assert!(
            write_diskann_index(&invalid_build, &mut PosWriter::new(&mut build_output)).is_err()
        );
        assert!(build_output.is_empty());
    }

    #[test]
    fn diskann_header_requires_exactly_one_storage_layout() {
        let mut bytes = DiskAnnHeader::for_layout(8, 2, 0, 2, DiskAnnBuildParams::default())
            .unwrap()
            .encode();
        put_u32(&mut bytes, 12, DISKANN_REQUIRED_FLAGS);
        assert!(DiskAnnHeader::decode(&bytes).is_err());

        put_u32(
            &mut bytes,
            12,
            DISKANN_REQUIRED_FLAGS
                | FLAG_SEPARATE_ADJACENCY_AND_VECTORS
                | FLAG_INTERLEAVED_ADJACENCY_AND_VECTORS,
        );
        assert!(DiskAnnHeader::decode(&bytes).is_err());
    }

    #[test]
    fn diskann_header_rejects_version_flags_reserved_and_shape_corruption() {
        let valid = DiskAnnHeader::for_layout(8, 2, 0, 2, DiskAnnBuildParams::default())
            .unwrap()
            .encode();
        let corruptions: &[(usize, u32, &str)] = &[
            (4, DISKANN_VERSION + 1, "version"),
            (8, DISKANN_HEADER_SIZE as u32 - 1, "header size"),
            (12, get_u32(&valid, 12) | (1 << 31), "unknown feature flag"),
            (76, 0, "raw-vector encoding"),
            (80, 31, "raw-vector record size"),
            (84, SECTION_COUNT as u32 - 1, "section count"),
        ];
        for &(offset, value, name) in corruptions {
            let mut bytes = valid;
            put_u32(&mut bytes, offset, value);
            assert!(
                DiskAnnHeader::decode(&bytes).is_err(),
                "{name} corruption must fail closed"
            );
        }

        let mut reserved = valid;
        reserved[208] = 1;
        assert!(
            DiskAnnHeader::decode(&reserved).is_err(),
            "non-zero reserved bytes must fail closed"
        );
    }

    #[test]
    fn diskann_auto_storage_profile_latency_classifier_has_stable_boundaries() {
        assert_eq!(
            classify_auto_storage_profile(
                AUTO_PROFILE_MEMORY_LATENCY_THRESHOLD - Duration::from_nanos(1),
            ),
            StorageProfile::Memory
        );
        assert_eq!(
            classify_auto_storage_profile(AUTO_PROFILE_MEMORY_LATENCY_THRESHOLD),
            StorageProfile::LocalStorage
        );
        assert_eq!(
            classify_auto_storage_profile(
                AUTO_PROFILE_LOCAL_LATENCY_THRESHOLD - Duration::from_nanos(1),
            ),
            StorageProfile::LocalStorage
        );
        assert_eq!(
            classify_auto_storage_profile(AUTO_PROFILE_LOCAL_LATENCY_THRESHOLD),
            StorageProfile::RemoteStorage
        );
        assert_eq!(
            classify_auto_storage_profile(
                AUTO_PROFILE_REMOTE_LATENCY_THRESHOLD - Duration::from_nanos(1),
            ),
            StorageProfile::RemoteStorage
        );
        assert_eq!(
            classify_auto_storage_profile(AUTO_PROFILE_REMOTE_LATENCY_THRESHOLD),
            StorageProfile::ObjectStore
        );
    }

    #[test]
    fn diskann_auto_storage_profile_resolves_once_and_explicit_profile_skips_detection() {
        let header = DiskAnnHeader::for_layout(8, 2, 0, 2, DiskAnnBuildParams::default()).unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);

        let auto_hint_calls = Arc::new(AtomicUsize::new(0));
        let mut automatic = DiskAnnIndexReader::open_with_options(
            ProfileHintReader {
                inner: Cursor::new(bytes.clone()),
                hint: StorageProfile::ObjectStore,
                hint_calls: Arc::clone(&auto_hint_calls),
            },
            VectorIndexReaderOptions::default(),
        )
        .unwrap();
        assert_eq!(automatic.effective_storage_profile(), StorageProfile::Auto);

        automatic.search(&[0.0; 8], 1, 10).unwrap();
        automatic.search(&[0.0; 8], 1, 10).unwrap();
        let clone = automatic.try_clone_for_search().unwrap().unwrap();

        assert_eq!(
            automatic.effective_storage_profile(),
            StorageProfile::ObjectStore
        );
        assert_eq!(
            clone.effective_storage_profile(),
            StorageProfile::ObjectStore
        );
        assert_eq!(automatic.auto_storage_profile_probe_time(), None);
        assert_eq!(auto_hint_calls.load(AtomicOrdering::SeqCst), 1);

        let mut measured = DiskAnnIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        measured.search(&[0.0; 8], 1, 10).unwrap();
        assert_ne!(measured.effective_storage_profile(), StorageProfile::Auto);
        assert!(measured.auto_storage_profile_probe_time().is_some());

        let explicit_hint_calls = Arc::new(AtomicUsize::new(0));
        let mut explicit = DiskAnnIndexReader::open_with_options(
            ProfileHintReader {
                inner: Cursor::new(bytes),
                hint: StorageProfile::ObjectStore,
                hint_calls: Arc::clone(&explicit_hint_calls),
            },
            VectorIndexReaderOptions::new(StorageProfile::LocalStorage, 4 * 1024 * 1024 * 1024),
        )
        .unwrap();
        explicit.search(&[0.0; 8], 1, 10).unwrap();

        assert_eq!(
            explicit.effective_storage_profile(),
            StorageProfile::LocalStorage
        );
        assert_eq!(explicit.auto_storage_profile_probe_time(), None);
        assert_eq!(explicit_hint_calls.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn diskann_header_calculates_aligned_section_layout() {
        let header = DiskAnnHeader::for_layout(
            128,
            31,
            0,
            16,
            DiskAnnBuildParams {
                raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();

        assert_eq!(header.adjacency_locator_size, 4);
        assert_eq!(header.adjacency_locator_encoding, 3);
        assert_eq!(
            header.raw_vector_encoding,
            DiskAnnRawVectorEncoding::F32 as u32
        );
        assert_eq!(header.vector_record_size, 512);
        assert_eq!(
            header.sections,
            DiskAnnSections {
                codebook: SectionRange::new(4096, 131172),
                row_ids: SectionRange::new(135268, 280),
                pq_codes: SectionRange::new(135548, 496),
                row_id_order: SectionRange::new(136044, 124),
                adjacency_index: SectionRange::new(136168, 140),
                adjacency: SectionRange::new(139264, 4096),
                vectors: SectionRange::new(143360, 15872),
            }
        );
        assert_eq!(header.file_len, 159232);
    }

    #[test]
    fn diskann_interleaved_layout_rejects_non_finite_raw_vectors_during_warmup() {
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
                storage_layout: DiskAnnStorageLayout::Interleaved,
                raw_vector_encoding: DiskAnnRawVectorEncoding::F16,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, count);
        index.add(&data, &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let mut locator_reader = DiskAnnIndexReader::open(Cursor::new(bytes.clone())).unwrap();
        locator_reader.ensure_resident().unwrap();
        let locator = locator_reader.adjacency_locator(0).unwrap();
        let header = locator_reader.header.clone();
        let vector_offset = header.sections.adjacency.offset as usize
            + locator.page_index as usize * DISKANN_PAGE_SIZE as usize
            + locator.byte_offset as usize
            - header.vector_record_size as usize;
        bytes[vector_offset..vector_offset + 2]
            .copy_from_slice(&half::f16::NAN.to_bits().to_le_bytes());

        let mut reader = DiskAnnIndexReader::open_with_options(
            Cursor::new(bytes),
            VectorIndexReaderOptions::with_cache_budgets(
                StorageProfile::Auto,
                header.sections.adjacency.length as usize,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();
        let error = reader
            .optimize_for_search()
            .expect_err("non-finite interleaved raw vectors must fail closed");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn diskann_header_uses_packed_4bit_pq_layout() {
        let header = DiskAnnHeader::for_layout_with_pq_bits(
            128,
            31,
            0,
            16,
            4,
            DiskAnnBuildParams::default(),
        )
        .unwrap();

        assert_eq!(header.pq_bits, 4);
        assert_eq!(
            header.sections.codebook.length,
            (PQ_CODEBOOK_HEADER_SIZE + 17 * size_of::<u32>() + 128 * 16 * 4) as u64
        );
        assert_eq!(header.sections.pq_codes.length, 31 * 8);
        assert_eq!(DiskAnnHeader::decode(&header.encode()).unwrap(), header);
    }

    #[test]
    fn diskann_header_rejects_previous_adjacency_locator_encoding() {
        let header = DiskAnnHeader::for_layout(8, 2, 0, 2, DiskAnnBuildParams::default()).unwrap();
        let mut encoded = header.encode();
        put_u32(&mut encoded, 72, 2);

        let error = DiskAnnHeader::decode(&encoded)
            .expect_err("the unpublished fixed-u32 locator encoding must be rejected");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("layout"));
    }

    #[test]
    fn diskann_header_uses_encoded_row_id_section_length() {
        let build = DiskAnnBuildParams::default();
        let row_ids_len = ROW_ID_SECTION_HEADER_SIZE + 78;

        let header =
            DiskAnnHeader::for_layout_with_adjacency_pages(128, 31, 0, 16, build, row_ids_len, 1)
                .unwrap();

        assert_eq!(header.sections.row_ids.length, row_ids_len as u64);
        assert_eq!(
            header.sections.pq_codes.offset,
            header.sections.row_ids.offset + row_ids_len as u64
        );
        assert!(DiskAnnHeader::for_layout_with_adjacency_pages(
            128,
            31,
            0,
            16,
            build,
            ROW_ID_SECTION_HEADER_SIZE - 1,
            1,
        )
        .is_err());
    }

    #[test]
    fn diskann_writer_serializes_aligned_and_consistently_remapped_sections() {
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
                raw_vector_encoding: DiskAnnRawVectorEncoding::F32,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);

        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();

        let header = DiskAnnHeader::decode(&bytes[..DISKANN_HEADER_SIZE]).unwrap();
        assert_eq!(bytes.len() as u64, header.file_len);
        assert_eq!(header.sections.codebook.offset, 4096);
        assert_eq!(header.sections.adjacency.offset % 4096, 0);
        assert_eq!(header.sections.vectors.offset % 4096, 0);

        assert!(
            header.sections.row_ids.length < raw_row_id_section_len(indexed_count).unwrap() as u64
        );
        let row_offset = header.sections.row_ids.offset as usize;
        let row_end = section_end(header.sections.row_ids).unwrap() as usize;
        let row_ids = decode_row_id_section(&bytes[row_offset..row_end], indexed_count).unwrap();
        let first_row_id = row_ids.get(0).unwrap();
        let old_id = (first_row_id - 1000) as usize;
        let vector_offset = header.sections.vectors.offset as usize;
        let first_value =
            f32::from_le_bytes(bytes[vector_offset..vector_offset + 4].try_into().unwrap());
        assert_eq!(first_value, data[old_id * dimension]);

        assert_eq!(
            header.sections.row_id_order.length,
            indexed_count as u64 * 4
        );
        assert_eq!(
            header.sections.adjacency_index.length,
            adjacency_index_serialized_len(indexed_count).unwrap()
        );

        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        reader.ensure_resident().unwrap();
        assert_eq!(reader.row_id(0).unwrap(), first_row_id);
        assert_eq!(reader.row_id_count().unwrap(), indexed_count);
    }

    #[test]
    fn diskann_writer_chunks_all_sections_into_bounded_write_calls() {
        let dimension = 8;
        let training_count = 1024;
        let indexed_count = 1024;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (1000..1000 + indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 64,
                build_search_list_size: 100,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut writer = CountingWriter::default();

        write_diskann_index(&index, &mut writer).unwrap();

        let header = DiskAnnHeader::decode(&writer.bytes[..DISKANN_HEADER_SIZE]).unwrap();
        let adjacency_pages =
            header.sections.adjacency.length as usize / DISKANN_PAGE_SIZE as usize;
        assert!(adjacency_pages > 1);
        let page_sized_writes = writer
            .write_lengths
            .iter()
            .filter(|&&length| length == DISKANN_PAGE_SIZE as usize)
            .count();
        assert!(
            page_sized_writes < adjacency_pages,
            "adjacency serialization issued one output call per logical page"
        );
        let full_chunks = writer.bytes.len().div_ceil(DISKANN_WRITE_BUFFER_SIZE);
        assert!(
            writer.write_calls <= full_chunks + 8,
            "DiskANN serialization crossed the output boundary {} times for {} chunks",
            writer.write_calls,
            full_chunks
        );
    }

    #[test]
    fn diskann_row_id_order_is_persisted_by_row_id_then_node() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 8;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = vec![9, -1, 9, 3, 7, 3, 11, 7];
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 4,
                build_search_list_size: 8,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = DiskAnnHeader::decode(&bytes[..DISKANN_HEADER_SIZE]).unwrap();
        let row_ids = decode_row_id_section(
            &bytes[header.sections.row_ids.offset as usize
                ..section_end(header.sections.row_ids).unwrap() as usize],
            indexed_count,
        )
        .unwrap();
        let order = bytes[header.sections.row_id_order.offset as usize
            ..section_end(header.sections.row_id_order).unwrap() as usize]
            .chunks_exact(4)
            .map(|value| u32::from_le_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>();

        validate_row_id_order(&row_ids, &order).unwrap();
        let ordered_keys = order
            .iter()
            .map(|&node| (row_ids.get(node as usize).unwrap(), node))
            .collect::<Vec<_>>();
        assert_eq!(
            ordered_keys.iter().map(|key| key.0).collect::<Vec<_>>(),
            vec![-1, 3, 3, 7, 7, 9, 9, 11]
        );
        assert!(ordered_keys.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn diskann_row_id_order_corruption_fails_closed() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        let order = header.sections.row_id_order.offset as usize;
        bytes[order..order + 4].copy_from_slice(&0u32.to_le_bytes());
        bytes[order + 4..order + 8].copy_from_slice(&0u32.to_le_bytes());
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();

        let error = reader
            .ensure_row_id_order()
            .expect_err("duplicate row-order nodes must fail closed");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("strictly increasing"));
    }

    #[test]
    fn diskann_row_id_order_budget_fallback_is_cached_without_reading_lookup() {
        let header = DiskAnnHeader::for_layout(
            8,
            64,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let budget = resident_peak_bytes(&header).unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        let reads = Arc::new(Mutex::new(Vec::new()));
        let mut reader = DiskAnnIndexReader::open_with_options(
            RecordingReader {
                inner: Cursor::new(bytes),
                reads: Arc::clone(&reads),
            },
            VectorIndexReaderOptions::with_cache_budgets(
                StorageProfile::Auto,
                16 * 1024 * 1024,
                16 * 1024 * 1024,
                budget,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();
        reader.hot_adjacency = Arc::from(vec![0u8; 16 * 1024]);

        assert!(reader.ensure_row_id_order().unwrap().is_none());
        let reads_after_first_attempt = reads.lock().unwrap().len();
        assert!(reader.ensure_row_id_order().unwrap().is_none());

        assert_eq!(reads.lock().unwrap().len(), reads_after_first_attempt);
        assert!(!reads
            .lock()
            .unwrap()
            .iter()
            .any(|(offset, _)| *offset == header.sections.row_id_order.offset));
    }

    #[test]
    fn diskann_row_id_order_reserves_budget_from_shared_caches() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 64;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (0..indexed_count as i64).collect::<Vec<_>>();
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
        let header = DiskAnnHeader::decode(&bytes[..DISKANN_HEADER_SIZE]).unwrap();
        let order_bytes = header.sections.row_id_order.length as usize;
        let minimum_peak = resident_peak_bytes(&header)
            .unwrap()
            .max(row_id_order_peak_bytes(&header, 0).unwrap());
        let budget = minimum_peak + 4 * 1024;
        let mut reader = DiskAnnIndexReader::open_with_options(
            Cursor::new(bytes),
            VectorIndexReaderOptions::with_cache_budgets(
                StorageProfile::Auto,
                0,
                16 * 1024,
                budget,
                16 * 1024,
            ),
        )
        .unwrap();
        reader.ensure_resident().unwrap();
        let initial_capacity = reader
            .adjacency_cache()
            .unwrap()
            .total_capacity()
            .saturating_add(reader.raw_vector_cache().unwrap().total_capacity());
        assert_eq!(initial_capacity, 32 * 1024);

        assert!(reader.ensure_row_id_order().unwrap().is_some());
        let final_capacity = reader
            .adjacency_cache()
            .unwrap()
            .total_capacity()
            .saturating_add(reader.raw_vector_cache().unwrap().total_capacity());
        assert_eq!(
            final_capacity,
            (budget - resident_steady_bytes(&header).unwrap() - order_bytes).min(32 * 1024)
        );
        assert!(resident_steady_bytes(&header).unwrap() + order_bytes + final_capacity <= budget);
    }

    #[test]
    fn diskann_header_rejects_overlapping_sections() {
        let mut header =
            DiskAnnHeader::for_layout(128, 31, 0, 16, DiskAnnBuildParams::default()).unwrap();
        header.sections.row_ids.offset -= 1;

        let error = DiskAnnHeader::decode(&header.encode())
            .expect_err("overlapping sections should fail closed");

        assert!(error.to_string().contains("section layout"));
    }

    #[test]
    fn diskann_reader_loads_resident_sections_in_one_multi_range_round() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 24;
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

        let rounds = Arc::new(Mutex::new(Vec::new()));
        let recording = RoundRecordingReader {
            inner: Cursor::new(bytes.clone()),
            rounds: Arc::clone(&rounds),
            max_ranges_per_pread: 0,
        };
        let mut reader = DiskAnnIndexReader::open_with_options(
            recording,
            VectorIndexReaderOptions::with_cache_budgets(
                StorageProfile::LocalStorage,
                0,
                0,
                4 * 1024 * 1024 * 1024,
                0,
            ),
        )
        .unwrap();
        assert_eq!(
            rounds.lock().unwrap().as_slice(),
            &[vec![(0, DISKANN_HEADER_SIZE)]]
        );

        reader.ensure_resident().unwrap();

        assert_eq!(
            rounds.lock().unwrap().as_slice(),
            &[
                vec![(0, DISKANN_HEADER_SIZE)],
                vec![
                    (
                        reader.header.sections.codebook.offset,
                        reader.header.sections.codebook.length as usize,
                    ),
                    (
                        reader.header.sections.row_ids.offset,
                        reader.header.sections.row_ids.length as usize,
                    ),
                    (
                        reader.header.sections.pq_codes.offset,
                        reader.header.sections.pq_codes.length as usize,
                    ),
                    (
                        reader.header.sections.adjacency_index.offset,
                        reader.header.sections.adjacency_index.length as usize,
                    ),
                    (reader.header.file_len - 1, 1),
                ],
            ]
        );
        assert_eq!(reader.row_id_count().unwrap(), indexed_count);
        assert_eq!(reader.pq_codes().unwrap().len(), indexed_count * 2);
        assert_eq!(reader.pq().unwrap().centroids, index.pq.centroids);

        let limited_rounds = Arc::new(Mutex::new(Vec::new()));
        let limited_recording = RoundRecordingReader {
            inner: Cursor::new(bytes),
            rounds: Arc::clone(&limited_rounds),
            max_ranges_per_pread: 2,
        };
        let mut limited_reader = DiskAnnIndexReader::open_with_options(
            limited_recording,
            VectorIndexReaderOptions::with_cache_budgets(
                StorageProfile::LocalStorage,
                0,
                0,
                4 * 1024 * 1024 * 1024,
                0,
            ),
        )
        .unwrap();
        limited_reader.ensure_resident().unwrap();
        assert_eq!(
            limited_rounds
                .lock()
                .unwrap()
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![1, 2, 2, 1]
        );
        assert_eq!(limited_reader.row_id_count().unwrap(), indexed_count);
        assert_eq!(limited_reader.pq_codes().unwrap().len(), indexed_count * 2);
    }

    #[test]
    fn diskann_reader_rejects_malformed_adjacency_degree() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        let locator_offset = header.sections.adjacency_index.offset as usize;
        bytes[locator_offset + 12..locator_offset + 14]
            .copy_from_slice(&((header.max_degree + 1) as u16).to_le_bytes());
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();

        let error = reader
            .ensure_resident()
            .expect_err("degree above R must fail closed");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("degree"));
    }

    #[test]
    fn diskann_adjacency_index_rejects_gaps_and_page_crossing() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 2,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let gap = AdjacencyIndex::from_locators(&[
            AdjacencyLocator::new(0, 0, 1, AdjacencyListEncoding::RawU32).unwrap(),
            AdjacencyLocator::new(0, 8, 0, AdjacencyListEncoding::DeltaVarint).unwrap(),
        ])
        .unwrap();
        let crossing = AdjacencyIndex::from_locators(&[
            AdjacencyLocator::new(0, 4092, 2, AdjacencyListEncoding::RawU32).unwrap(),
            AdjacencyLocator::new(0, 4092, 0, AdjacencyListEncoding::DeltaVarint).unwrap(),
        ])
        .unwrap();

        assert!(validate_adjacency_index(&header, &gap)
            .unwrap_err()
            .to_string()
            .contains("contiguous"));
        assert!(validate_adjacency_index(&header, &crossing)
            .unwrap_err()
            .to_string()
            .contains("crosses"));
    }

    #[test]
    fn diskann_adjacency_page_rejects_noncanonical_varints_and_encoded_gaps() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 2,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let locators = AdjacencyIndex::from_locators(&[
            AdjacencyLocator::new(0, 0, 1, AdjacencyListEncoding::DeltaVarint).unwrap(),
            AdjacencyLocator::new(0, 2, 0, AdjacencyListEncoding::DeltaVarint).unwrap(),
        ])
        .unwrap();
        let mut page = vec![0u8; DISKANN_PAGE_SIZE as usize];
        page[..2].copy_from_slice(&[0x81, 0x00]);

        let noncanonical =
            validate_adjacency_page_payload(&header, &locators, 0, &page).unwrap_err();
        assert!(noncanonical.to_string().contains("canonical"));

        page[..2].copy_from_slice(&[0x01, 0x00]);
        let gap = validate_adjacency_page_payload(&header, &locators, 0, &page).unwrap_err();
        assert!(gap.to_string().contains("contiguous"));
    }

    #[test]
    fn diskann_adjacency_page_rejects_nonminimal_adaptive_encoding() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 2,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let locators = AdjacencyIndex::from_locators(&[
            AdjacencyLocator::new(0, 0, 1, AdjacencyListEncoding::RawU32).unwrap(),
            AdjacencyLocator::new(0, 4, 0, AdjacencyListEncoding::DeltaVarint).unwrap(),
        ])
        .unwrap();
        let mut page = vec![0u8; DISKANN_PAGE_SIZE as usize];
        put_u32(&mut page, 0, 1);

        let error = validate_adjacency_page_payload(&header, &locators, 0, &page)
            .expect_err("raw u32 must be rejected when delta-varint is smaller");

        assert!(error.to_string().contains("minimal"));
    }

    #[test]
    fn diskann_adjacency_page_rejects_unsorted_or_self_neighbors() {
        let dimension = 8;
        let training_count = 256;
        let indexed_count = 16;
        let data = (0..training_count * dimension)
            .map(|offset| ((offset * 31) % 127) as f32)
            .collect::<Vec<_>>();
        let ids = (0..indexed_count as i64).collect::<Vec<_>>();
        let mut index = DiskAnnIndex::new(
            dimension,
            MetricType::L2,
            2,
            DiskAnnBuildParams {
                max_degree: 4,
                build_search_list_size: 8,
                ..DiskAnnBuildParams::default()
            },
        );
        index.train(&data, training_count);
        index.add(&data[..indexed_count * dimension], &ids);
        let mut bytes = Vec::new();
        write_diskann_index(&index, &mut PosWriter::new(&mut bytes)).unwrap();
        let header = DiskAnnHeader::decode(&bytes[..DISKANN_HEADER_SIZE]).unwrap();
        let (node, locator) = (0..indexed_count)
            .find_map(|node| {
                let locator = serialized_adjacency_locator(&bytes, &header, node);
                (locator.degree() > 0).then_some((node, locator))
            })
            .unwrap();
        let neighbor_offset = header.sections.adjacency.offset as usize
            + locator.page_index as usize * DISKANN_PAGE_SIZE as usize
            + locator.byte_offset as usize;
        match locator.encoding() {
            AdjacencyListEncoding::DeltaVarint => {
                bytes[neighbor_offset] = node as u8;
            }
            AdjacencyListEncoding::RawU32 => {
                put_u32(&mut bytes, neighbor_offset, node as u32);
            }
        }
        let page_start = header.sections.adjacency.offset as usize
            + locator.page_index as usize * DISKANN_PAGE_SIZE as usize;
        let page = bytes[page_start..page_start + DISKANN_PAGE_SIZE as usize].to_vec();
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        reader.ensure_resident().unwrap();

        let error = reader
            .validate_adjacency_page(locator.page_index as usize, &page)
            .expect_err("self edges must fail closed");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("self edge"));
    }

    #[test]
    fn diskann_reader_records_shared_adjacency_validation_in_resident_data() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();
        reader.ensure_resident().unwrap();
        let page = vec![0u8; DISKANN_PAGE_SIZE as usize];

        reader.validate_adjacency_page(0, &page).unwrap();

        assert_eq!(
            reader
                .resident
                .as_ref()
                .unwrap()
                .adjacency_validation
                .states[0]
                .load(AtomicOrdering::Acquire),
            ADJACENCY_PAGE_VALID
        );
    }

    #[test]
    fn diskann_reader_rejects_non_finite_codebook_during_warmup() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        let first_centroid = header.sections.codebook.offset as usize
            + PQ_CODEBOOK_HEADER_SIZE
            + (header.pq_m as usize + 1) * size_of::<u32>();
        bytes[first_centroid..first_centroid + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut reader = DiskAnnIndexReader::open(Cursor::new(bytes)).unwrap();

        let error = reader
            .ensure_resident()
            .expect_err("non-finite PQ centroid must fail closed");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("finite"));
    }

    #[test]
    fn diskann_reader_maps_truncated_header_to_invalid_data() {
        let error = match DiskAnnIndexReader::open(Cursor::new(b"DANN".to_vec())) {
            Ok(_) => panic!("truncated header must fail closed"),
            Err(error) => error,
        };

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("truncated"));
    }

    #[test]
    fn diskann_reader_rejects_resident_sections_above_reader_budget_before_reading() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let recording = RecordingReader {
            inner: Cursor::new(header.encode().to_vec()),
            reads: Arc::clone(&reads),
        };
        let mut reader = DiskAnnIndexReader::open_with_options(
            recording,
            VectorIndexReaderOptions::new(StorageProfile::Auto, 1),
        )
        .unwrap();

        let error = reader
            .ensure_resident()
            .expect_err("resident data above the configured budget must fail closed");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("resident"));
        assert_eq!(
            reads.lock().unwrap().as_slice(),
            &[(0, DISKANN_HEADER_SIZE)]
        );
    }

    #[test]
    fn diskann_adjacency_preload_rounds_up_for_every_storage_profile() {
        let header = DiskAnnHeader::for_layout_with_adjacency_pages(
            8,
            5_000,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
            raw_row_id_section_len(5_000).unwrap(),
            32,
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        for storage_profile in [
            StorageProfile::Memory,
            StorageProfile::LocalStorage,
            StorageProfile::RemoteStorage,
            StorageProfile::ObjectStore,
        ] {
            let reads = Arc::new(Mutex::new(Vec::new()));
            let recording = RecordingReader {
                inner: Cursor::new(bytes.clone()),
                reads: Arc::clone(&reads),
            };
            let mut reader = DiskAnnIndexReader::open_with_options(
                recording,
                VectorIndexReaderOptions::with_cache_budgets(
                    storage_profile,
                    4096,
                    16 * 1024 * 1024,
                    4 * 1024 * 1024 * 1024,
                    8 * 1024 * 1024,
                ),
            )
            .unwrap();

            reader.optimize_for_search().unwrap();

            assert_eq!(
                reads.lock().unwrap().last().copied(),
                Some((header.sections.adjacency.offset, 64 * 1024)),
                "{storage_profile:?} must honor adjacency_preload_bytes"
            );
        }
    }

    #[test]
    fn diskann_adjacency_preload_validates_pages_during_warmup() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        bytes[header.sections.adjacency.offset as usize] = 1;
        let mut reader = DiskAnnIndexReader::open_with_options(
            Cursor::new(bytes),
            VectorIndexReaderOptions::with_cache_budgets(
                StorageProfile::Auto,
                DISKANN_PAGE_SIZE as usize,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();

        let error = reader
            .optimize_for_search()
            .expect_err("preloaded adjacency pages must be validated during warmup");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("tail"));
        assert!(reader.hot_adjacency.is_empty());
    }

    #[test]
    fn diskann_adjacency_preload_stays_within_resident_budget() {
        let header = DiskAnnHeader::for_layout_with_adjacency_pages(
            8,
            5_000,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
            raw_row_id_section_len(5_000).unwrap(),
            64,
        )
        .unwrap();
        assert!(header.sections.adjacency.length > 128 * 1024);
        let max_resident_bytes = resident_peak_bytes(&header).unwrap() + 64 * 1024;
        let available_preload_bytes = max_resident_bytes - resident_steady_bytes(&header).unwrap();
        let expected_preload_bytes = available_preload_bytes / (64 * 1024) * (64 * 1024);
        assert_eq!(expected_preload_bytes, 64 * 1024);
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        let mut reader = DiskAnnIndexReader::open_with_options(
            Cursor::new(bytes),
            VectorIndexReaderOptions::with_cache_budgets(
                StorageProfile::Auto,
                128 * 1024,
                16 * 1024 * 1024,
                max_resident_bytes,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();

        reader.optimize_for_search().unwrap();

        assert_eq!(reader.hot_adjacency.len(), expected_preload_bytes);
        assert!(
            resident_steady_bytes(&header).unwrap() + reader.hot_adjacency.len()
                <= max_resident_bytes
        );
    }

    #[test]
    fn diskann_resident_peak_accounts_for_batched_decode_buffers() {
        let vector_count = 1_000_000;
        let header = DiskAnnHeader::for_layout_with_adjacency_pages(
            128,
            vector_count,
            0,
            32,
            DiskAnnBuildParams::default(),
            raw_row_id_section_len(vector_count).unwrap(),
            1,
        )
        .unwrap();
        let decoded_codebook_bytes = 128 * diskann_pq_ksub(header.pq_bits).unwrap() * 4
            + (header.pq_m as usize + 1) * size_of::<usize>();
        let decoded_row_id_bytes = vector_count * size_of::<i64>();
        let row_decode_phase_bytes = header.sections.row_ids.length as usize
            + header.sections.pq_codes.length as usize
            + header.sections.adjacency_index.length as usize
            + decoded_codebook_bytes
            + decoded_row_id_bytes;

        assert!(
            resident_peak_bytes(&header).unwrap() >= row_decode_phase_bytes,
            "the budget must cover serialized row IDs and their decoded representation at once"
        );
    }

    #[test]
    fn diskann_search_clones_share_resident_and_hot_adjacency() {
        let header = DiskAnnHeader::for_layout_with_adjacency_pages(
            8,
            5_000,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 8,
                build_search_list_size: 16,
                ..DiskAnnBuildParams::default()
            },
            raw_row_id_section_len(5_000).unwrap(),
            32,
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        initialize_empty_adjacency_index(&mut bytes, &header);
        let mut reader = DiskAnnIndexReader::open_with_options(
            RecordingReader {
                inner: Cursor::new(bytes),
                reads: Arc::new(Mutex::new(Vec::new())),
            },
            VectorIndexReaderOptions::with_cache_budgets(
                StorageProfile::ObjectStore,
                4096,
                16 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
                8 * 1024 * 1024,
            ),
        )
        .unwrap();
        reader.optimize_for_search().unwrap();

        let clone = reader.try_clone_for_search().unwrap().unwrap();

        assert!(Arc::ptr_eq(
            reader.resident.as_ref().unwrap(),
            clone.resident.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(&reader.hot_adjacency, &clone.hot_adjacency));
    }

    #[test]
    fn diskann_reader_budget_bounds_decoded_peak_not_only_serialized_bytes() {
        let header = DiskAnnHeader::for_layout(
            8,
            2,
            0,
            2,
            DiskAnnBuildParams {
                max_degree: 1,
                build_search_list_size: 2,
                ..DiskAnnBuildParams::default()
            },
        )
        .unwrap();
        let serialized_resident_bytes = (section_end(header.sections.adjacency_index).unwrap()
            - header.sections.codebook.offset) as usize;
        let mut bytes = vec![0u8; header.file_len as usize];
        bytes[..DISKANN_HEADER_SIZE].copy_from_slice(&header.encode());
        let mut reader = DiskAnnIndexReader::open_with_options(
            Cursor::new(bytes),
            VectorIndexReaderOptions::new(StorageProfile::Auto, serialized_resident_bytes),
        )
        .unwrap();

        let error = reader
            .ensure_resident()
            .expect_err("budget must include decoded vectors, PQ norms, and load scratch");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("reader budget"));
    }

    #[test]
    fn diskann_memory_accounts_for_row_id_payload_without_section_header() {
        let row_ids_len = ROW_ID_SECTION_HEADER_SIZE + 78;
        let header = DiskAnnHeader::for_layout_with_adjacency_pages(
            128,
            31,
            0,
            16,
            DiskAnnBuildParams::default(),
            row_ids_len,
            1,
        )
        .unwrap();
        let non_row_id_bytes = decoded_pq_codebook_bytes(&header).unwrap()
            + header.sections.pq_codes.length as usize
            + header.sections.adjacency_index.length as usize
            + adjacency_validation_bytes(&header).unwrap()
            + header.pq_m as usize * 256 * size_of::<f32>();

        assert_eq!(
            resident_steady_bytes(&header).unwrap() - non_row_id_bytes,
            row_ids_len - ROW_ID_SECTION_HEADER_SIZE
        );
    }
}
