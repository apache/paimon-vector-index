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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum StorageProfile {
    #[default]
    Auto = 0,
    Memory = 1,
    LocalStorage = 2,
    RemoteStorage = 3,
    ObjectStore = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadPlan {
    pub window_bytes: usize,
    pub graph_beam_width: usize,
    pub filtered_graph_beam_width: usize,
}

impl StorageProfile {
    pub(crate) const fn read_plan(self) -> ReadPlan {
        match self {
            Self::Auto | Self::LocalStorage => ReadPlan {
                window_bytes: 16 * 1024,
                graph_beam_width: 4,
                filtered_graph_beam_width: 4,
            },
            Self::Memory => ReadPlan {
                window_bytes: 4 * 1024,
                graph_beam_width: 16,
                filtered_graph_beam_width: 4,
            },
            Self::RemoteStorage => ReadPlan {
                window_bytes: 32 * 1024,
                graph_beam_width: 16,
                filtered_graph_beam_width: 4,
            },
            Self::ObjectStore => ReadPlan {
                window_bytes: 64 * 1024,
                graph_beam_width: 16,
                filtered_graph_beam_width: 4,
            },
        }
    }
}

impl ReadPlan {
    pub(crate) fn with_capabilities(
        mut self,
        capabilities: crate::io::SeekReadCapabilities,
    ) -> Self {
        let alignment = capabilities.preferred_alignment_bytes;
        if alignment > 0 {
            self.window_bytes = self.window_bytes.max(alignment);
        }
        if capabilities.preferred_window_bytes > 0 {
            self.window_bytes = capabilities.preferred_window_bytes.max(alignment.max(1));
        }
        // Bound pathological adapter hints while keeping enough room for one
        // complete DiskANN page. The planner clips the final window to the
        // section, so a non-page-sized capability hint must not split a page
        // across two windows.
        self.window_bytes = self.window_bytes.clamp(4 * 1024, 1024 * 1024);
        self.window_bytes = self.window_bytes.div_ceil(4 * 1024) * (4 * 1024);
        self.window_bytes = self.window_bytes.min(1024 * 1024);
        if capabilities.max_ranges_per_pread > 0 {
            self.graph_beam_width = self
                .graph_beam_width
                .min(capabilities.max_ranges_per_pread)
                .max(1);
            self.filtered_graph_beam_width = self
                .filtered_graph_beam_width
                .min(capabilities.max_ranges_per_pread)
                .max(1);
        }
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorIndexReaderOptions {
    pub storage_profile: StorageProfile,
    pub memory_budget_bytes: usize,
    cache_overrides: Option<ResolvedVectorIndexReaderOptions>,
}

impl Default for VectorIndexReaderOptions {
    fn default() -> Self {
        Self {
            storage_profile: StorageProfile::Auto,
            memory_budget_bytes: 4 * 1024 * 1024 * 1024,
            cache_overrides: None,
        }
    }
}

impl VectorIndexReaderOptions {
    pub fn new(storage_profile: StorageProfile, memory_budget_bytes: usize) -> Self {
        Self {
            storage_profile,
            memory_budget_bytes,
            cache_overrides: None,
        }
    }

    pub(crate) fn resolve_cache_budgets(
        self,
        resident_steady_bytes: usize,
        adjacency_section_bytes: usize,
        raw_vector_section_bytes: usize,
    ) -> ResolvedVectorIndexReaderOptions {
        if let Some(mut overrides) = self.cache_overrides {
            overrides.storage_profile = self.storage_profile;
            return overrides;
        }
        let available = self
            .memory_budget_bytes
            .saturating_sub(resident_steady_bytes);
        let preload_cap = match self.storage_profile {
            StorageProfile::Auto | StorageProfile::Memory | StorageProfile::LocalStorage => {
                16 * 1024 * 1024
            }
            StorageProfile::RemoteStorage => 32 * 1024 * 1024,
            StorageProfile::ObjectStore => 64 * 1024 * 1024,
        };
        let adjacency_preload_bytes = (available / 2)
            .min(preload_cap)
            .min(adjacency_section_bytes);
        let after_preload = available.saturating_sub(adjacency_preload_bytes);
        let per_cache_cap = match self.storage_profile {
            StorageProfile::RemoteStorage | StorageProfile::ObjectStore => usize::MAX,
            StorageProfile::Auto | StorageProfile::Memory | StorageProfile::LocalStorage => {
                64 * 1024 * 1024
            }
        };
        let cold_adjacency_bytes = adjacency_section_bytes.saturating_sub(adjacency_preload_bytes);
        let mut adjacency_cache_bytes = (after_preload / 2)
            .min(per_cache_cap)
            .min(cold_adjacency_bytes);
        let mut raw_vector_cache_bytes = after_preload
            .saturating_sub(adjacency_cache_bytes)
            .min(per_cache_cap)
            .min(raw_vector_section_bytes);
        let mut unassigned = after_preload
            .saturating_sub(adjacency_cache_bytes)
            .saturating_sub(raw_vector_cache_bytes);
        let adjacency_extra = unassigned
            .min(per_cache_cap.saturating_sub(adjacency_cache_bytes))
            .min(cold_adjacency_bytes.saturating_sub(adjacency_cache_bytes));
        adjacency_cache_bytes = adjacency_cache_bytes.saturating_add(adjacency_extra);
        unassigned = unassigned.saturating_sub(adjacency_extra);
        let vector_extra = unassigned
            .min(per_cache_cap.saturating_sub(raw_vector_cache_bytes))
            .min(raw_vector_section_bytes.saturating_sub(raw_vector_cache_bytes));
        raw_vector_cache_bytes = raw_vector_cache_bytes.saturating_add(vector_extra);
        ResolvedVectorIndexReaderOptions {
            storage_profile: self.storage_profile,
            adjacency_preload_bytes,
            adjacency_cache_bytes,
            max_resident_bytes: self.memory_budget_bytes,
            raw_vector_cache_bytes,
        }
    }

    pub(crate) const fn uses_automatic_cache_budgets(self) -> bool {
        self.cache_overrides.is_none()
    }

    #[cfg(test)]
    pub(crate) fn with_cache_budgets(
        storage_profile: StorageProfile,
        adjacency_preload_bytes: usize,
        adjacency_cache_bytes: usize,
        max_resident_bytes: usize,
        raw_vector_cache_bytes: usize,
    ) -> Self {
        Self {
            storage_profile,
            memory_budget_bytes: max_resident_bytes,
            cache_overrides: Some(ResolvedVectorIndexReaderOptions {
                storage_profile,
                adjacency_preload_bytes,
                adjacency_cache_bytes,
                max_resident_bytes,
                raw_vector_cache_bytes,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedVectorIndexReaderOptions {
    pub storage_profile: StorageProfile,
    pub adjacency_preload_bytes: usize,
    pub adjacency_cache_bytes: usize,
    pub max_resident_bytes: usize,
    pub raw_vector_cache_bytes: usize,
}

#[cfg(test)]
mod tests {
    use crate::io::SeekReadCapabilities;

    use super::{StorageProfile, VectorIndexReaderOptions};

    #[test]
    fn storage_profiles_have_stable_codes_and_distinct_read_plans() {
        assert_eq!(StorageProfile::Auto as u32, 0);
        assert_eq!(StorageProfile::Memory as u32, 1);
        assert_eq!(StorageProfile::LocalStorage as u32, 2);
        assert_eq!(StorageProfile::RemoteStorage as u32, 3);
        assert_eq!(StorageProfile::ObjectStore as u32, 4);

        let options = VectorIndexReaderOptions::default();
        assert_eq!(options.storage_profile, StorageProfile::Auto);
        assert_eq!(options.memory_budget_bytes, 4 * 1024 * 1024 * 1024);

        let resolved = options.resolve_cache_budgets(1024, 16 * 1024 * 1024, 8 * 1024 * 1024);
        assert!(resolved.adjacency_preload_bytes > 0);
        assert_eq!(
            resolved.adjacency_preload_bytes + resolved.adjacency_cache_bytes,
            16 * 1024 * 1024
        );
        assert!(resolved.raw_vector_cache_bytes > 0);
        assert!(
            1024 + resolved.adjacency_preload_bytes
                + resolved.adjacency_cache_bytes
                + resolved.raw_vector_cache_bytes
                <= options.memory_budget_bytes
        );

        let memory = StorageProfile::Memory.read_plan();
        let local = StorageProfile::LocalStorage.read_plan();
        let remote = StorageProfile::RemoteStorage.read_plan();
        let object_store = StorageProfile::ObjectStore.read_plan();
        assert_eq!(
            (
                memory.window_bytes,
                memory.graph_beam_width,
                memory.filtered_graph_beam_width
            ),
            (4096, 16, 4)
        );
        assert_eq!(
            (
                local.window_bytes,
                local.graph_beam_width,
                local.filtered_graph_beam_width
            ),
            (16 * 1024, 4, 4)
        );
        assert_eq!(
            (
                remote.window_bytes,
                remote.graph_beam_width,
                remote.filtered_graph_beam_width
            ),
            (32 * 1024, 16, 4)
        );
        assert_eq!(
            (
                object_store.window_bytes,
                object_store.graph_beam_width,
                object_store.filtered_graph_beam_width
            ),
            (64 * 1024, 16, 4)
        );
    }

    #[test]
    fn remote_cache_budgets_use_available_memory_beyond_the_local_cache_caps() {
        const MIB: usize = 1024 * 1024;
        let resolved = VectorIndexReaderOptions::new(StorageProfile::RemoteStorage, 4 * 1024 * MIB)
            .resolve_cache_budgets(256 * MIB, 512 * MIB, 2 * 1024 * MIB);

        assert_eq!(resolved.adjacency_preload_bytes, 32 * MIB);
        assert_eq!(
            resolved.adjacency_cache_bytes,
            512 * MIB - resolved.adjacency_preload_bytes
        );
        assert_eq!(resolved.raw_vector_cache_bytes, 2 * 1024 * MIB);
        assert!(
            256 * MIB
                + resolved.adjacency_preload_bytes
                + resolved.adjacency_cache_bytes
                + resolved.raw_vector_cache_bytes
                <= resolved.max_resident_bytes
        );
    }

    #[test]
    fn capability_windows_are_bounded_and_keep_complete_diskann_pages() {
        let plan =
            StorageProfile::LocalStorage
                .read_plan()
                .with_capabilities(SeekReadCapabilities {
                    preferred_alignment_bytes: 6_000,
                    preferred_window_bytes: 10_000,
                    max_ranges_per_pread: 2,
                });
        assert_eq!(plan.window_bytes, 12 * 1024);
        assert_eq!(plan.graph_beam_width, 2);
        assert_eq!(plan.filtered_graph_beam_width, 2);

        let bounded =
            StorageProfile::RemoteStorage
                .read_plan()
                .with_capabilities(SeekReadCapabilities {
                    preferred_alignment_bytes: usize::MAX,
                    preferred_window_bytes: usize::MAX,
                    max_ranges_per_pread: 0,
                });
        assert_eq!(bounded.window_bytes, 1024 * 1024);
    }
}
