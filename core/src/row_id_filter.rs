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

use roaring::RoaringTreemap;
use std::collections::HashSet;
use std::io;

pub trait RowIdFilter: Sync {
    fn contains(&self, id: i64) -> bool;
}

impl RowIdFilter for HashSet<i64> {
    fn contains(&self, id: i64) -> bool {
        HashSet::contains(self, &id)
    }
}

impl RowIdFilter for RoaringTreemap {
    fn contains(&self, id: i64) -> bool {
        id >= 0 && RoaringTreemap::contains(self, id as u64)
    }
}

/// A row-id filter with optional inclusion and exclusion bitmaps.
///
/// Exclusion takes precedence. If the row ID is not excluded, an inclusion
/// bitmap restricts the result when present; otherwise the row ID is accepted.
#[derive(Clone, Debug)]
pub struct RoaringRowIdFilter {
    included_row_ids: Option<RoaringTreemap>,
    excluded_row_ids: Option<RoaringTreemap>,
}

impl RoaringRowIdFilter {
    pub fn new(
        included_row_ids: Option<RoaringTreemap>,
        excluded_row_ids: Option<RoaringTreemap>,
    ) -> Self {
        Self {
            included_row_ids,
            excluded_row_ids,
        }
    }

    pub(crate) fn from_serialized(
        included_row_ids: Option<&[u8]>,
        excluded_row_ids: Option<&[u8]>,
    ) -> io::Result<Self> {
        Ok(Self::new(
            decode_optional_roaring_filter(included_row_ids, "include")?,
            decode_optional_roaring_filter(excluded_row_ids, "exclude")?,
        ))
    }
}

impl RowIdFilter for RoaringRowIdFilter {
    fn contains(&self, id: i64) -> bool {
        // Negative row IDs are not possible in current Paimon framework.
        if id < 0 {
            return false;
        }
        let id = id as u64;

        if self
            .excluded_row_ids
            .as_ref()
            .is_some_and(|excluded| excluded.contains(id))
        {
            return false;
        }

        self.included_row_ids
            .as_ref()
            .is_none_or(|included| included.contains(id))
    }
}

fn decode_optional_roaring_filter(
    bytes: Option<&[u8]>,
    filter_name: &str,
) -> io::Result<Option<RoaringTreemap>> {
    bytes
        .map(|bytes| {
            RoaringTreemap::deserialize_from(bytes).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid {} RoaringTreemap filter: {}", filter_name, e),
                )
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclude_takes_precedence_over_include() {
        let included = RoaringTreemap::from_iter([1, 2]);
        let excluded = RoaringTreemap::from_iter([1, 3]);
        let filter = RoaringRowIdFilter::new(Some(included), Some(excluded));

        assert!(!filter.contains(1));
        assert!(filter.contains(2));
        assert!(!filter.contains(3));
        assert!(!filter.contains(4));
    }

    #[test]
    fn missing_include_allows_rows_not_excluded() {
        let excluded = RoaringTreemap::from_iter([1]);
        let filter = RoaringRowIdFilter::new(None, Some(excluded));

        assert!(!filter.contains(1));
        assert!(filter.contains(2));
    }

    #[test]
    fn missing_filters_allow_all_rows() {
        let filter = RoaringRowIdFilter::new(None, None);

        assert!(filter.contains(1));
    }

    #[test]
    fn negative_row_ids_are_rejected() {
        let filters = [
            RoaringRowIdFilter::new(None, None),
            RoaringRowIdFilter::new(Some(RoaringTreemap::new()), None),
            RoaringRowIdFilter::new(None, Some(RoaringTreemap::new())),
            RoaringRowIdFilter::new(Some(RoaringTreemap::new()), Some(RoaringTreemap::new())),
        ];

        for filter in filters {
            assert!(!filter.contains(-1));
        }
    }
}
