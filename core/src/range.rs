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

//! Shared types for distance range search (distance bands).
//!
//! The internal interval is always half-open, `[lower, upper)`, expressed in the
//! index's own distance space: squared distance for L2, `1 - cos` for cosine and
//! `-inner_product` for inner product.
//!
//! # Unboundedness is a structural state, not a sentinel value
//!
//! Representing "unbounded" with a finite sentinel drops rows on two of the
//! three metrics. `fvec_cosine_distance_with_norms` does not clamp, so two
//! identical normalized vectors can produce about `-1.19e-7`, and a
//! `lower = 0.0` sentinel would exclude the most similar rows. An inner-product
//! distance of `-ip` can be exactly `f32::MAX`, and a half-open interval with
//! `upper = f32::MAX` excludes precisely that value. L2's `lower = 0.0` happens
//! to be safe because squared distances are non-negative, but that is a
//! coincidence of one metric and should not become the representation for three.

use rayon::prelude::*;
use std::{borrow::Cow, io};

use crate::distance::{fvec_norm_l2sqr, fvec_normalize, MetricType};
use crate::kmeans;

/// One side's bound on an interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    /// This side is unbounded. Comparisons read it as ∓∞ by direction; no
    /// non-finite cut value is ever constructed.
    Unbounded,
    Finite(f32),
}

/// A validated internal interval `[lower, upper)` together with the metric it
/// belongs to.
///
/// Use [`Self::from_endpoints`] for public-distance predicates, or explicitly
/// opt into internal cuts with [`Self::from_raw`]. Ambiguous raw construction
/// is intentionally unavailable:
///
/// ```compile_fail
/// use paimon_vindex_core::distance::MetricType;
/// use paimon_vindex_core::range::{Bound, DistanceBand};
/// DistanceBand::new(Bound::Unbounded, Bound::Finite(4.0), MetricType::L2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceBand {
    lower: Bound,
    upper: Bound,
    metric: MetricType,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

impl DistanceBand {
    /// Constructs a band from raw f32 cuts: squared L2, cosine distance, or
    /// negative inner product. Rejects non-finite cuts, negative squared-L2
    /// cuts, and inverted intervals. Prefer [`Self::from_endpoints`] for public
    /// endpoint literals; this method does not convert units or operators.
    ///
    /// This has to fail loud rather than "quietly return no rows". Any caller can
    /// pass an illegal value, and an empty result would be read upstream as
    /// "this bucket genuinely has no matches" -- indistinguishable from a
    /// correct answer, and so undetectable.
    pub fn from_raw(lower: Bound, upper: Bound, metric: MetricType) -> io::Result<Self> {
        for (side, bound) in [("lower", lower), ("upper", upper)] {
            if let Bound::Finite(value) = bound {
                if !value.is_finite() {
                    return Err(invalid(format!("{side} cut is not finite: {value}")));
                }
                if metric == MetricType::L2 && value < 0.0 {
                    return Err(invalid(format!(
                        "{side} cut must be non-negative for squared-L2: {value}"
                    )));
                }
            }
        }
        if let (Bound::Finite(low), Bound::Finite(high)) = (lower, upper) {
            if low > high {
                return Err(invalid(format!("inverted band: [{low}, {high})")));
            }
        }
        Ok(Self {
            lower,
            upper,
            metric,
        })
    }

    pub fn metric(&self) -> MetricType {
        self.metric
    }

    /// Inclusive raw lower cut, not the original public lower endpoint.
    pub fn raw_lower(&self) -> Bound {
        self.lower
    }

    /// Exclusive raw upper cut, not the original public upper endpoint.
    pub fn raw_upper(&self) -> Bound {
        self.upper
    }

    /// An empty interval (`lower == upper`) is legal and returns zero rows.
    pub fn is_empty(&self) -> bool {
        matches!((self.lower, self.upper), (Bound::Finite(low), Bound::Finite(high)) if low >= high)
    }

    /// Membership test for a raw distance. Left-closed, right-open.
    ///
    /// A non-finite value is never a member, whichever side is unbounded.
    /// Without this an unbounded side would admit `NaN` and the matching
    /// infinity, because each side's comparison short-circuits to `true` --
    /// which would sit oddly beside the collector, where a non-finite computed
    /// distance fails loud rather than being committed. The two are not the
    /// same response -- this is a total predicate, so it answers "not a member"
    /// rather than raising -- but neither should ever call such a value a match,
    /// and this is the public membership authority.
    #[inline]
    pub fn admit_raw(&self, value: f32) -> bool {
        if !value.is_finite() {
            return false;
        }
        let above_lower = match self.lower {
            Bound::Unbounded => true,
            Bound::Finite(low) => value >= low,
        };
        let below_upper = match self.upper {
            Bound::Unbounded => true,
            Bound::Finite(high) => value < high,
        };
        above_lower && below_upper
    }
}

pub(crate) fn prepare_range_queries(
    queries: &[f32],
    dimension: usize,
    metric: MetricType,
) -> io::Result<Cow<'_, [f32]>> {
    if metric != MetricType::Cosine {
        return Ok(Cow::Borrowed(queries));
    }
    let mut normalized = queries.to_vec();
    for query in normalized.chunks_exact_mut(dimension) {
        let norm = fvec_normalize(query);
        if !norm.is_finite() || query.iter().any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "non-finite normalized query",
            ));
        }
    }
    Ok(Cow::Owned(normalized))
}

const PARALLEL_RANGE_MIN_COARSE_COMPONENTS: usize = 128 * 1024;

pub(crate) fn range_probe_lists(
    queries: &[f32],
    centroids: &[f32],
    dimension: usize,
    nlist: usize,
    nprobe: usize,
    metric: MetricType,
) -> io::Result<Vec<Vec<usize>>> {
    if metric == MetricType::L2 {
        return Ok(kmeans::find_topk_batch(
            queries,
            queries.len() / dimension,
            centroids,
            nlist,
            dimension,
            nprobe,
        )
        .0);
    }
    if centroids.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "non-finite IVF centroid",
        ));
    }
    let probe_query = |query: &[f32]| {
        kmeans::find_topk_checked(query, centroids, nlist, dimension, nprobe)
            .map(|lists| lists.into_iter().map(|(_, list)| list).collect())
            .map_err(|list| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("non-finite query-centroid distance for list {list}"),
                )
            })
    };
    let query_count = queries.len() / dimension;
    let coarse_work = query_count.saturating_mul(nlist).saturating_mul(dimension);
    if query_count > 1 && coarse_work >= PARALLEL_RANGE_MIN_COARSE_COMPONENTS {
        queries
            .par_chunks_exact(dimension)
            .map(probe_query)
            .collect()
    } else {
        queries.chunks_exact(dimension).map(probe_query).collect()
    }
}

pub(crate) fn checked_cosine_norm(vector: &[f32]) -> io::Result<f32> {
    let norm_squared = fvec_norm_l2sqr(vector);
    if vector.iter().any(|value| !value.is_finite()) || !norm_squared.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "non-finite cosine vector or norm",
        ));
    }
    Ok(norm_squared.sqrt())
}

/// The comparison operator for one endpoint of a distance predicate.
///
/// This is a Rust API type only. It has no stable numeric, wire, or C ABI
/// representation; a binding layer must define and validate its own external
/// values separately. Nothing outside this crate currently consumes it, so
/// fixing a numbering here would freeze an ABI before it has been designed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutOperator {
    Ge,
    Gt,
    Le,
    Lt,
}

/// One endpoint of a predicate. `value` is the already-folded literal from the
/// right-hand side, passed through as-is: no squaring, no square root, no
/// binary search on the caller's part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceEndpoint {
    pub value: f64,
    pub op: CutOperator,
}

/// What a stored squared-L2 f32 distance displays as in SQL: `sqrtf`, then
/// widened to double.
#[inline]
fn l2_public_value(stored_bits: u32) -> f64 {
    MetricType::L2.public_distance(f32::from_bits(stored_bits))
}

fn ordered_value(key: u32) -> f32 {
    f32::from_bits(if key & 0x8000_0000 != 0 {
        key ^ 0x8000_0000
    } else {
        !key
    })
}

fn first_linear_cut(endpoint: DistanceEndpoint) -> Option<f32> {
    let admits = |key| {
        let value = f64::from(ordered_value(key));
        match endpoint.op {
            CutOperator::Ge | CutOperator::Lt => value >= endpoint.value,
            CutOperator::Gt | CutOperator::Le => value > endpoint.value,
        }
    };
    let mut low = 0x0080_0000;
    let mut high = 0xff7f_ffff;
    if !admits(high) {
        return None;
    }
    while low < high {
        let middle = low + (high - low) / 2;
        if admits(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Some(ordered_value(low))
}

/// The first bit pattern satisfying `l2_public_value(bits) >= endpoint`.
///
/// The search interval is `0..=f32::MAX.to_bits()`, i.e. the **non-negative**
/// f32 values. That is complete for L2, whose squared distances are
/// non-negative, but not for cosine, which can go slightly negative — which is
/// why cosine needs a conversion of its own.
fn first_ge(endpoint: f64) -> Option<u32> {
    let mut low = 0u32;
    let mut high = f32::MAX.to_bits();
    if l2_public_value(high) < endpoint {
        return None;
    }
    while low < high {
        let mid = low + (high - low) / 2;
        if l2_public_value(mid) >= endpoint {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Some(low)
}

/// The first bit pattern satisfying `l2_public_value(bits) > endpoint`.
fn first_gt(endpoint: f64) -> Option<u32> {
    let mut low = 0u32;
    let mut high = f32::MAX.to_bits();
    if l2_public_value(high) <= endpoint {
        return None;
    }
    while low < high {
        let mid = low + (high - low) / 2;
        if l2_public_value(mid) > endpoint {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Some(low)
}

fn unsupported_endpoint(value: f64) -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        format!("endpoint {value} has no representable cut; do not push this predicate down"),
    )
}

impl DistanceBand {
    /// Derives a band from a predicate's two endpoints. `None` on either side
    /// means that side is unbounded.
    ///
    /// Side and operator must agree: `lower` accepts only `Ge`/`Gt` and `upper`
    /// only `Le`/`Lt`. A mismatch is an error rather than a reinterpretation —
    /// "the lower bound is `<`" is meaningless, and guessing the intent would
    /// only mask a dispatch bug in the caller.
    ///
    /// Endpoints use [`MetricType::public_distance`]. Cosine searches the whole
    /// finite f32 axis, including negative roundoff. Inner product negates the
    /// endpoint and reverses both its side and comparison. Neither path rounds
    /// the f64 endpoint to f32 before deciding membership. Out-of-domain linear
    /// cuts become empty or structurally unbounded bands. L2 retains its
    /// `Unsupported` result when no representable square-root cut exists.
    pub fn from_endpoints(
        lower: Option<DistanceEndpoint>,
        upper: Option<DistanceEndpoint>,
        metric: MetricType,
    ) -> io::Result<Self> {
        for ep in [lower, upper].into_iter().flatten() {
            if !ep.value.is_finite() {
                return Err(invalid(format!("endpoint is not finite: {}", ep.value)));
            }
        }
        // Reduce each side's operator to the primitive it needs, which also
        // rejects a mismatched side before the metric is consulted. A `lower`
        // end is closed, so its cut is the first admitted value; an `upper` end
        // is open, so its cut is the first excluded one.
        type CutFinder = fn(f64) -> Option<u32>;
        let lower_finder: Option<(DistanceEndpoint, CutFinder)> = match lower {
            None => None,
            Some(ep) => Some((
                ep,
                match ep.op {
                    CutOperator::Ge => first_ge,
                    CutOperator::Gt => first_gt,
                    CutOperator::Le | CutOperator::Lt => {
                        return Err(invalid(format!(
                            "lower endpoint accepts Ge or Gt, got {:?}",
                            ep.op
                        )))
                    }
                },
            )),
        };
        let upper_finder: Option<(DistanceEndpoint, CutFinder)> = match upper {
            None => None,
            Some(ep) => Some((
                ep,
                match ep.op {
                    CutOperator::Lt => first_ge,
                    CutOperator::Le => first_gt,
                    CutOperator::Ge | CutOperator::Gt => {
                        return Err(invalid(format!(
                            "upper endpoint accepts Le or Lt, got {:?}",
                            ep.op
                        )))
                    }
                },
            )),
        };
        if metric != MetricType::L2 {
            if matches!((lower, upper), (Some(low), Some(high)) if low.value > high.value) {
                return Err(invalid("inverted public band"));
            }
            let reverse = |endpoint: DistanceEndpoint| DistanceEndpoint {
                value: -endpoint.value,
                op: match endpoint.op {
                    CutOperator::Ge => CutOperator::Le,
                    CutOperator::Gt => CutOperator::Lt,
                    CutOperator::Le => CutOperator::Ge,
                    CutOperator::Lt => CutOperator::Gt,
                },
            };
            let (lower, upper) = if metric == MetricType::InnerProduct {
                (upper.map(reverse), lower.map(reverse))
            } else {
                (lower, upper)
            };
            let lower_cut = lower.map(first_linear_cut);
            let upper_cut = upper.and_then(first_linear_cut);
            if lower_cut == Some(None)
                || upper_cut == Some(-f32::MAX)
                || matches!((lower_cut, upper_cut), (Some(Some(low)), Some(high)) if low > high)
            {
                return Self::from_raw(Bound::Finite(0.0), Bound::Finite(0.0), metric);
            }
            return Self::from_raw(
                lower_cut.flatten().map_or(Bound::Unbounded, Bound::Finite),
                upper_cut.map_or(Bound::Unbounded, Bound::Finite),
                metric,
            );
        }
        let resolve = |side: Option<(DistanceEndpoint, CutFinder)>| -> io::Result<Bound> {
            match side {
                None => Ok(Bound::Unbounded),
                Some((ep, find)) => Ok(Bound::Finite(f32::from_bits(
                    find(ep.value).ok_or_else(|| unsupported_endpoint(ep.value))?,
                ))),
            }
        };
        let lower_bound = resolve(lower_finder)?;
        let upper_bound = resolve(upper_finder)?;
        Self::from_raw(lower_bound, upper_bound, metric)
    }
}

/// Per-query statistics. Fields stay private so they can be extended.
#[derive(Debug, Clone, Copy, Default)]
pub struct RangeSearchStats {
    lists_probed: usize,
    rows_scanned: usize,
    rows_committed: usize,
    early_abandoned: usize,
}

impl RangeSearchStats {
    /// The number of **logical ranks** probed. A rank dropped under budget
    /// pressure and re-read later is not counted twice.
    pub fn lists_probed(&self) -> usize {
        self.lists_probed
    }
    /// Allow-listed rows read and at least partially evaluated, **including**
    /// rows abandoned early. Blocked SQ arithmetic may also evaluate excluded
    /// lanes; those do not enter this logical counter.
    pub fn rows_scanned(&self) -> usize {
        self.rows_scanned
    }
    pub fn rows_committed(&self) -> usize {
        self.rows_committed
    }
    /// Rows the scan rejected against the abandon cutoff rather than evaluating
    /// into the band test, which for IVF-Flat under L2 means their distance is
    /// above the band's upper cut. IVF-SQ also counts estimates equal to that
    /// exclusive cut, using its blocked quantized-distance kernel.
    ///
    /// A diagnostic, not a work measure: a row is counted whether the kernel
    /// stopped at its first term or at its last, so this is not the number of
    /// rows whose evaluation was short-circuited. `rows_scanned` counts these
    /// rows too. Cosine and inner product never abandon partial sums. IVF-PQ
    /// and IVF-RQ always evaluate complete estimates, so their counts are zero.
    pub fn early_abandoned(&self) -> usize {
        self.early_abandoned
    }
}

/// Call-level statistics: the counters that are one-per-call rather than
/// per-query. Per-query counters live in [`RangeSearchStats`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RangeSearchCallStats {
    list_reads: usize,
}

impl RangeSearchCallStats {
    /// The number of **first read attempts of non-empty unique lists** (a
    /// logical measure): empty lists do not count, because the existing reader
    /// issues no payload I/O for them, and the several chunks of an oversized
    /// list count once. There is no re-reading, so this is also the actual
    /// number of list reads. IVF-SQ partition-cache hits do not count, since
    /// they issue no payload I/O.
    pub fn list_reads(&self) -> usize {
        self.list_reads
    }
}

/// One query's view of the result.
#[derive(Debug)]
pub struct QueryResult<'a> {
    pub labels: &'a [i64],
    /// Raw squared-L2, cosine-distance, or negative-inner-product values.
    pub raw_distances: &'a [f32],
    pub stats: &'a RangeSearchStats,
}

/// CSR-shaped batch result with offsets, labels, and explicitly raw distances.
/// Values remain in core units, including quantized estimates; public endpoint
/// conversion does not transform the output. There is no ambiguous distance
/// accessor:
///
/// ```compile_fail
/// use paimon_vindex_core::range::RangeSearchResult;
/// fn ambiguous(result: &RangeSearchResult) {
///     let _ = result.distances();
/// }
/// ```
///
/// Per-query views follow the same contract:
///
/// ```compile_fail
/// use paimon_vindex_core::range::QueryResult;
/// fn ambiguous(query: QueryResult<'_>) {
///     let _ = query.distances;
/// }
/// ```
#[derive(Debug)]
pub struct RangeSearchResult {
    lims: Vec<usize>,
    labels: Vec<i64>,
    raw_distances: Vec<f32>,
    stats: Vec<RangeSearchStats>,
    call_stats: RangeSearchCallStats,
}

impl RangeSearchResult {
    pub fn query_count(&self) -> usize {
        self.lims.len() - 1
    }

    /// Panics when `i >= query_count()`, rather than masking a caller's
    /// out-of-range index with an empty result.
    pub fn query(&self, i: usize) -> QueryResult<'_> {
        let (start, end) = (self.lims[i], self.lims[i + 1]);
        QueryResult {
            labels: &self.labels[start..end],
            raw_distances: &self.raw_distances[start..end],
            stats: &self.stats[i],
        }
    }

    pub fn call_stats(&self) -> &RangeSearchCallStats {
        &self.call_stats
    }

    pub fn lims(&self) -> &[usize] {
        &self.lims
    }

    pub fn labels(&self) -> &[i64] {
        &self.labels
    }

    /// Borrows raw squared-L2, cosine-distance, or negative-inner-product values.
    /// Use [`MetricType::public_distance`] for a public predicate value; do not
    /// compare these values directly with public L2 or inner-product endpoints.
    pub fn raw_distances(&self) -> &[f32] {
        &self.raw_distances
    }
}

/// Accumulates a CSR result.
///
/// Rows are staged **per query** and flattened in query order only in
/// [`build`]. A batch scan is list-major: one list fans out to several queries,
/// so different queries' rows are produced interleaved. Appending into a single
/// global array would let `query(0)` pick up another query's rows, so each query
/// gets its own bucket.
///
/// [`build`]: RangeResultBuilder::build
pub(crate) struct RangeResultBuilder {
    rows: Vec<Vec<(i64, f32)>>,
    stats: Vec<RangeSearchStats>,
    call_stats: RangeSearchCallStats,
}

impl RangeResultBuilder {
    pub(crate) fn new(nq: usize) -> Self {
        Self {
            rows: vec![Vec::new(); nq],
            stats: vec![RangeSearchStats::default(); nq],
            call_stats: RangeSearchCallStats::default(),
        }
    }

    /// Hands over one query's whole batch of rows, avoiding a row-by-row copy
    /// and a second resident allocation.
    pub(crate) fn take_rows(&mut self, query: usize, rows: Vec<(i64, f32)>) {
        self.stats[query].rows_committed += rows.len();
        if self.rows[query].is_empty() {
            self.rows[query] = rows; // the usual path: take ownership outright
        } else {
            self.rows[query].extend(rows);
        }
    }

    /// Records one **list read attempt** (a logical measure). The call-level
    /// fields are module-private, so other modules can only write them here.
    ///
    /// Accounting rules, each asserted by a test:
    /// * a unique list counts once even when it fans out to several queries
    /// * an oversized list read as several streamed chunks still counts once
    /// * empty lists do not count, since the existing reader issues no payload
    ///   I/O for them
    pub(crate) fn record_list_read(&mut self) {
        self.call_stats.list_reads += 1;
    }

    pub(crate) fn record_scanned(&mut self, query: usize, rows: usize) {
        self.stats[query].rows_scanned += rows;
    }

    pub(crate) fn record_early_abandoned(&mut self, query: usize, rows: usize) {
        self.stats[query].early_abandoned += rows;
    }

    pub(crate) fn record_lists_probed(&mut self, query: usize, lists: usize) {
        self.stats[query].lists_probed += lists;
    }

    /// Flattens into CSR in query order. No `finish_query` is needed: the order
    /// comes from the index into `rows` rather than from the order of calls, so
    /// a list-major parallel merge cannot cross queries.
    pub(crate) fn build(self) -> RangeSearchResult {
        let total: usize = self.rows.iter().map(Vec::len).sum();
        let mut lims = Vec::with_capacity(self.rows.len() + 1);
        let mut labels = Vec::with_capacity(total);
        let mut raw_distances = Vec::with_capacity(total);
        lims.push(0);
        for query_rows in &self.rows {
            for (id, distance) in query_rows {
                labels.push(*id);
                raw_distances.push(*distance);
            }
            lims.push(labels.len());
        }
        RangeSearchResult {
            lims,
            labels,
            raw_distances,
            stats: self.stats,
            call_stats: self.call_stats,
        }
    }
}

// Only `push_row` is gated behind cfg(test). The production path hands over a
// whole batch with `take_rows`, while `push_row` exists purely so the builder's
// unit tests can construct rows one at a time; a pub(crate) method referenced
// only from tests is dead code in the lib target and would fail
// `clippy --all-targets -- -D warnings`. The remaining methods (record_*,
// build) are all called from the production path and must **not** be gated, or
// the non-test lib target loses them.
#[cfg(test)]
impl RangeResultBuilder {
    fn push_row(&mut self, query: usize, id: i64, distance: f32) {
        self.rows[query].push((id, distance));
        self.stats[query].rows_committed += 1;
    }
}

/// Range search parameters. **Fields are private**: knobs introduced by later
/// work would break struct-literal construction if they were `pub` fields, while
/// declaring them now without implementing them would mean accepting a silently
/// ineffective parameter. Private fields plus `new()` remove that dilemma, and
/// a later width mode can arrive as an added constructor without breaking this
/// one.
///
/// The probe width is a plain `nprobe`. An automatically widening mode is
/// deliberately absent rather than present-but-unsupported: a variant every
/// well-formed value of which only ever returns `Unsupported` is public surface
/// with no working use.
#[derive(Debug, Clone, Copy)]
pub struct VectorRangeSearchParams {
    band: DistanceBand,
    nprobe: usize,
}

impl VectorRangeSearchParams {
    pub fn new(band: DistanceBand, nprobe: usize) -> Self {
        Self { band, nprobe }
    }

    pub fn band(&self) -> DistanceBand {
        self.band
    }

    /// Checks only the things that are caller bugs regardless of index or family,
    /// so a dispatcher can reject them **before** deciding whether the family
    /// supports range search at all.
    ///
    /// Without this, a family that returns `Unsupported` would swallow an
    /// invalid width: an FFI caller reading `Unsupported` as "fall back to a
    /// scan" would silently paper over its own bug.
    pub(crate) fn validate_shape(&self) -> io::Result<()> {
        if self.nprobe == 0 {
            return Err(invalid("range search nprobe must be greater than 0"));
        }
        Ok(())
    }

    /// Validates and returns the effective nprobe, clamped to `nlist`.
    ///
    /// Deliberately **not** public. Publishing it would freeze "a width
    /// resolves to one effective nprobe" as part of the API, which any future
    /// width that adapts per pass would then have to break. Top-K keeps the
    /// equivalent resolution private for the same reason. Callers get
    /// validation implicitly, by calling a search method.
    pub(crate) fn validate(&self, nlist: usize) -> io::Result<usize> {
        self.validate_shape()?;
        Ok(self.nprobe.min(nlist))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::MetricType;

    #[test]
    fn range_probe_lists_non_l2_preserves_order_and_ties_at_parallel_threshold() {
        let dimension = 64;
        let pools = [1, 4].map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        });
        for metric in [MetricType::Cosine, MetricType::InnerProduct] {
            for (query_count, nlist) in [(1, 128), (15, 128), (16, 128), (17, 128), (1, 2048)] {
                let mut centroids = vec![0.0; nlist * dimension];
                for list in 0..nlist {
                    centroids[list * dimension + (list / 2) % dimension] = 1.0;
                }
                let mut queries = vec![0.0; query_count * dimension];
                for query_index in 0..query_count {
                    queries[query_index * dimension + (query_index * 7 + 3) % dimension] = 1.0;
                }
                for nprobe in [1, 3, nlist + 1] {
                    let expected = (0..query_count)
                        .map(|query_index| {
                            let coordinate = (query_index * 7 + 3) % dimension;
                            (0..nlist)
                                .filter(|list| (list / 2) % dimension == coordinate)
                                .chain(
                                    (0..nlist).filter(|list| (list / 2) % dimension != coordinate),
                                )
                                .take(nprobe)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    for pool in &pools {
                        let actual = pool.install(|| {
                            range_probe_lists(
                                &queries, &centroids, dimension, nlist, nprobe, metric,
                            )
                            .unwrap()
                        });
                        assert_eq!(
                            actual, expected,
                            "{metric:?}, {query_count}, {nlist}, {nprobe}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn range_probe_lists_non_l2_parallel_batches_reject_invalid_distances() {
        let dimension = 64;
        let nlist = 128;
        let query_count = 16;
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for metric in [MetricType::Cosine, MetricType::InnerProduct] {
                for (late_query, far_centroid, diagnostic) in [
                    (0.0, f32::NAN, "non-finite IVF centroid"),
                    (0.0, f32::INFINITY, "non-finite IVF centroid"),
                    (0.0, f32::NEG_INFINITY, "non-finite IVF centroid"),
                    (1e20, 0.0, "non-finite query-centroid distance for list 0"),
                    (0.0, 1e20, "non-finite query-centroid distance for list 127"),
                ] {
                    let mut queries = vec![0.0f32; query_count * dimension];
                    queries[(query_count - 1) * dimension] = late_query;
                    let mut centroids = vec![0.0; nlist * dimension];
                    centroids[(nlist - 1) * dimension] = far_centroid;
                    assert!(queries.iter().all(|value| value.is_finite()));
                    if far_centroid.is_finite() {
                        assert!(centroids.iter().all(|value| value.is_finite()));
                    }
                    if late_query != 0.0 {
                        assert!(range_probe_lists(
                            &queries[..(query_count - 1) * dimension],
                            &centroids,
                            dimension,
                            nlist,
                            1,
                            metric,
                        )
                        .is_ok());
                    }
                    let error = pool
                        .install(|| {
                            range_probe_lists(&queries, &centroids, dimension, nlist, 1, metric)
                        })
                        .unwrap_err();
                    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
                    assert_eq!(error.to_string(), diagnostic);
                }
            }
        }
    }

    // --- Task 3: the band type -------------------------------------------

    #[test]
    fn a_finite_l2_band_is_left_closed_right_open() {
        let band =
            DistanceBand::from_raw(Bound::Finite(1.0), Bound::Finite(3.0), MetricType::L2).unwrap();
        assert!(!band.admit_raw(0.999));
        assert!(band.admit_raw(1.0), "the lower end is closed");
        assert!(band.admit_raw(2.999));
        assert!(!band.admit_raw(3.0), "the upper end is open");
    }

    #[test]
    fn unboundedness_is_structural_not_a_sentinel() {
        // Cosine's 1-cos can go slightly negative because distance.rs does not
        // clamp it, so a 0.0 sentinel would drop the most similar rows.
        let band = DistanceBand::from_raw(Bound::Unbounded, Bound::Finite(0.5), MetricType::Cosine)
            .unwrap();
        assert!(
            band.admit_raw(-1.1920929e-7),
            "an unbounded lower end must admit a slightly negative cosine distance"
        );
        // An inner-product internal distance can be exactly f32::MAX, which a
        // half-open interval with that sentinel as its upper cut would exclude.
        let band = DistanceBand::from_raw(
            Bound::Finite(0.0),
            Bound::Unbounded,
            MetricType::InnerProduct,
        )
        .unwrap();
        assert!(
            band.admit_raw(f32::MAX),
            "an unbounded upper end must admit f32::MAX"
        );
    }

    #[test]
    fn no_band_admits_a_non_finite_value() {
        // An unbounded side short-circuits its comparison to true, so without an
        // explicit check a whole-space band would call NaN a member. This is the
        // public membership authority; it answers "not a member" rather than
        // raising, which is not the same response as the collector's fail-loud
        // push, but neither may ever call such a value a match.
        let whole =
            DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
        let lower_open =
            DistanceBand::from_raw(Bound::Unbounded, Bound::Finite(1.0), MetricType::L2).unwrap();
        let upper_open =
            DistanceBand::from_raw(Bound::Finite(0.0), Bound::Unbounded, MetricType::L2).unwrap();
        let closed =
            DistanceBand::from_raw(Bound::Finite(0.0), Bound::Finite(1.0), MetricType::L2).unwrap();
        for band in [whole, lower_open, upper_open, closed] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(!band.admit_raw(value), "{band:?} must not admit {value}");
            }
        }
    }

    #[test]
    fn a_whole_space_band_admits_everything_finite() {
        let band =
            DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, MetricType::L2).unwrap();
        assert!(band.admit_raw(0.0) && band.admit_raw(f32::MAX) && !band.is_empty());
    }

    #[test]
    fn an_inverted_or_non_finite_band_fails_loud() {
        assert!(
            DistanceBand::from_raw(Bound::Finite(5.0), Bound::Finite(3.0), MetricType::L2).is_err()
        );
        assert!(DistanceBand::from_raw(
            Bound::Finite(f32::NAN),
            Bound::Finite(3.0),
            MetricType::L2
        )
        .is_err());
        assert!(DistanceBand::from_raw(
            Bound::Finite(0.0),
            Bound::Finite(f32::INFINITY),
            MetricType::L2
        )
        .is_err());
        assert!(
            DistanceBand::from_raw(Bound::Finite(-1.0), Bound::Finite(3.0), MetricType::L2)
                .is_err(),
            "L2 is a squared distance, so a negative cut is illegal"
        );
    }

    #[test]
    fn an_empty_band_is_legal_and_admits_nothing() {
        let band =
            DistanceBand::from_raw(Bound::Finite(2.0), Bound::Finite(2.0), MetricType::L2).unwrap();
        assert!(band.is_empty());
        assert!(!band.admit_raw(2.0));
    }

    #[test]
    fn all_metrics_accept_positive_probe_widths() {
        for metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
            let band = DistanceBand::from_raw(Bound::Unbounded, Bound::Unbounded, metric).unwrap();
            assert_eq!(
                VectorRangeSearchParams::new(band, 3).validate(2).unwrap(),
                2
            );
        }
    }

    // --- Task 4: deriving cuts from SQL endpoints -------------------------

    #[test]
    fn a_lower_side_rejects_upper_side_operators() {
        let ep = DistanceEndpoint {
            value: 0.5,
            op: CutOperator::Lt,
        };
        let err = DistanceBand::from_endpoints(Some(ep), None, MetricType::L2).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        let ep = DistanceEndpoint {
            value: 0.5,
            op: CutOperator::Ge,
        };
        assert!(DistanceBand::from_endpoints(None, Some(ep), MetricType::L2).is_err());
    }

    #[test]
    fn a_wrong_side_operator_is_invalid_for_every_metric() {
        for metric in [MetricType::Cosine, MetricType::InnerProduct] {
            let ep = DistanceEndpoint {
                value: 0.5,
                op: CutOperator::Lt,
            };
            let err = DistanceBand::from_endpoints(Some(ep), None, metric).unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "lower side with Lt on metric {metric:?}"
            );
            let ep = DistanceEndpoint {
                value: 0.5,
                op: CutOperator::Ge,
            };
            let err = DistanceBand::from_endpoints(None, Some(ep), metric).unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "upper side with Ge on metric {metric:?}"
            );
        }
    }

    #[test]
    fn a_missing_side_is_unbounded_not_a_sentinel() {
        let ep = DistanceEndpoint {
            value: 0.5,
            op: CutOperator::Lt,
        };
        let band = DistanceBand::from_endpoints(None, Some(ep), MetricType::L2).unwrap();
        assert_eq!(band.raw_lower(), Bound::Unbounded);
        let band = DistanceBand::from_endpoints(None, None, MetricType::L2).unwrap();
        assert_eq!(
            (band.raw_lower(), band.raw_upper()),
            (Bound::Unbounded, Bound::Unbounded)
        );
    }

    #[test]
    fn linear_endpoint_derivation_preserves_public_comparisons() {
        for metric in [MetricType::Cosine, MetricType::InnerProduct] {
            let ep = DistanceEndpoint {
                value: 0.5,
                op: CutOperator::Ge,
            };
            let band = DistanceBand::from_endpoints(Some(ep), None, metric).unwrap();
            for distance in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                assert_eq!(
                    band.admit_raw(distance),
                    metric.public_distance(distance) >= 0.5
                );
            }
        }
    }

    #[test]
    fn a_non_finite_endpoint_is_invalid_input_and_outranks_the_metric_check() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
                let ep = DistanceEndpoint {
                    value,
                    op: CutOperator::Ge,
                };
                let err = DistanceBand::from_endpoints(Some(ep), None, metric).unwrap_err();
                // A non-finite endpoint is a caller bug (InvalidInput) and that
                // outranks "this metric is not supported".
                assert_eq!(
                    err.kind(),
                    std::io::ErrorKind::InvalidInput,
                    "endpoint {value} / metric {metric:?}"
                );
            }
        }
    }

    // Do not assert that the `>` and `>=` cuts are adjacent, nor that the `>`
    // cut is strictly larger. The public value `sqrtf(stored)` has plateaus:
    // several adjacent squared f32 values round to the same public value, so
    // first_gt skips a whole plateau. Measured at endpoint 0.25 the cuts are
    // 0x3d800000 and 0x3d800002, two apart; at endpoint sqrt(0.2), which falls
    // between two public values, they coincide. The real property is that each
    // is minimal, plus the criterion distinguishing the two situations.
    //
    // The `0x3d80_0000` bit patterns below are **construction inputs**, not
    // frozen expected values: they are fed through `public()` to produce an
    // endpoint that is then handed to `from_endpoints`, and every assertion is a
    // relation between values computed at run time. That keeps them portable.
    // `f32::sqrt` is required by IEEE-754 to be exactly rounded, so `public()`
    // is bit-identical across targets -- unlike `fvec_l2sqr`, whose SIMD lane
    // grouping differs between AVX2 and NEON, which is exactly why distances
    // must never be asserted as bit patterns.

    fn cut_of(value: f64, op: CutOperator) -> f32 {
        let band = DistanceBand::from_endpoints(
            Some(DistanceEndpoint { value, op }),
            None,
            MetricType::L2,
        )
        .unwrap();
        let Bound::Finite(cut) = band.raw_lower() else {
            panic!("lower must be finite")
        };
        cut
    }

    fn public(bits: u32) -> f64 {
        f64::from(f32::from_bits(bits).sqrt())
    }

    #[test]
    fn first_ge_is_minimal() {
        for value in [0.25_f64, 0.5, 0.2_f64.sqrt(), 1.0, 3.0] {
            let bits = cut_of(value, CutOperator::Ge).to_bits();
            assert!(
                public(bits) >= value,
                "value {value}: the cut itself must be admitted"
            );
            assert!(
                bits == 0 || public(bits - 1) < value,
                "value {value}: the preceding f32 must be excluded, else the cut is not minimal"
            );
        }
    }

    #[test]
    fn first_gt_is_minimal() {
        for value in [0.25_f64, 0.5, 0.2_f64.sqrt(), 1.0, 3.0] {
            let bits = cut_of(value, CutOperator::Gt).to_bits();
            assert!(
                public(bits) > value,
                "value {value}: the cut itself must be strictly greater than the endpoint"
            );
            assert!(
                bits == 0 || public(bits - 1) <= value,
                "value {value}: the preceding f32 must not satisfy strict greater-than"
            );
        }
    }

    #[test]
    fn upper_side_cuts_are_minimal_excluded_values() {
        // The upper end is open, so its cut is the first *excluded* value:
        // Lt maps to first_ge and Le to first_gt. Testing only the lower side's
        // minimality would not catch an implementer swapping the two.
        let upper_cut = |value: f64, op: CutOperator| -> f32 {
            let band = DistanceBand::from_endpoints(
                None,
                Some(DistanceEndpoint { value, op }),
                MetricType::L2,
            )
            .unwrap();
            let Bound::Finite(cut) = band.raw_upper() else {
                panic!("upper must be finite")
            };
            cut
        };
        let on_plateau = public(0x3d80_0000);
        // For `< e` the first excluded value is the first whose public value is >= e.
        let lt = upper_cut(on_plateau, CutOperator::Lt).to_bits();
        assert!(public(lt) >= on_plateau && (lt == 0 || public(lt - 1) < on_plateau));
        // For `<= e` the first excluded value is the first whose public value is > e.
        let le = upper_cut(on_plateau, CutOperator::Le).to_bits();
        assert!(public(le) > on_plateau && (le == 0 || public(le - 1) <= on_plateau));
        assert!(
            le > lt,
            "with the endpoint on a plateau, `<=` must exclude later than `<`"
        );
    }

    #[test]
    fn a_two_sided_band_admits_exactly_the_intended_rows() {
        let lower = DistanceEndpoint {
            value: 0.5,
            op: CutOperator::Ge,
        };
        let upper = DistanceEndpoint {
            value: 1.5,
            op: CutOperator::Lt,
        };
        let band = DistanceBand::from_endpoints(Some(lower), Some(upper), MetricType::L2).unwrap();
        let (Bound::Finite(lo), Bound::Finite(hi)) = (band.raw_lower(), band.raw_upper()) else {
            panic!("both cuts must be finite")
        };
        // Just before the boundary, on it, and just before the upper cut.
        assert!(!band.admit_raw(f32::from_bits(lo.to_bits() - 1)));
        assert!(band.admit_raw(lo));
        assert!(band.admit_raw(f32::from_bits(hi.to_bits() - 1)));
        assert!(!band.admit_raw(hi));
    }

    #[test]
    fn gt_never_precedes_ge_and_they_coincide_off_plateau() {
        // An endpoint exactly equal to some public value means `>` has to skip
        // that value's whole plateau, so gt > ge.
        let on_plateau = public(0x3d80_0000);
        assert!(cut_of(on_plateau, CutOperator::Gt) > cut_of(on_plateau, CutOperator::Ge));

        // An endpoint between two public values has no public value equal to it,
        // so the first satisfier of `>=` and of `>` is the same.
        let between = public(0x3d80_0000) + (public(0x3d80_0002) - public(0x3d80_0000)) / 2.0;
        assert_eq!(
            cut_of(between, CutOperator::Gt),
            cut_of(between, CutOperator::Ge)
        );
    }

    #[test]
    fn an_endpoint_with_no_representable_cut_is_unsupported() {
        // Beyond the largest public value the meaning is "this predicate cannot
        // be expressed as a band, so do not push it down".
        let ep = DistanceEndpoint {
            value: 1.0e30,
            op: CutOperator::Ge,
        };
        let err = DistanceBand::from_endpoints(Some(ep), None, MetricType::L2).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn a_negative_endpoint_is_admitted_and_clamps_to_zero() {
        let ep = DistanceEndpoint {
            value: -1.0,
            op: CutOperator::Ge,
        };
        let band = DistanceBand::from_endpoints(Some(ep), None, MetricType::L2).unwrap();
        assert_eq!(
            band.raw_lower(),
            Bound::Finite(0.0),
            "every distance is >= a negative endpoint"
        );
    }

    // --- Task 5: the CSR result and statistics ---------------------------

    #[test]
    fn csr_slices_line_up_with_lims() {
        let mut builder = RangeResultBuilder::new(2);
        // Interleaved writes, mimicking a list-major batch scan.
        builder.push_row(0, 10, 1.5);
        builder.push_row(1, 20, 3.5);
        builder.push_row(0, 11, 2.5);
        let result = builder.build();
        assert_eq!(result.query(0).labels, &[10, 11]);
        assert_eq!(result.query(0).raw_distances, &[1.5, 2.5]);
        assert_eq!(result.query(1).labels, &[20]);
        assert_eq!(result.query(1).raw_distances, &[3.5]);
    }

    #[test]
    fn a_query_with_no_hits_yields_empty_slices() {
        let result = RangeResultBuilder::new(1).build();
        assert!(result.query(0).labels.is_empty() && result.query(0).raw_distances.is_empty());
        assert_eq!(result.call_stats().list_reads(), 0);
    }

    #[test]
    #[should_panic]
    fn an_out_of_range_query_index_panics() {
        let result = RangeResultBuilder::new(1).build();
        let _ = result.query(1);
    }

    #[test]
    fn per_query_stats_are_independent() {
        let mut builder = RangeResultBuilder::new(2);
        builder.record_scanned(0, 7);
        builder.record_scanned(1, 3);
        let result = builder.build();
        assert_eq!(result.query(0).stats.rows_scanned(), 7);
        assert_eq!(result.query(1).stats.rows_scanned(), 3);
    }

    // --- Task 7: width and params ----------------------------------------

    fn l2_band_for_params() -> DistanceBand {
        DistanceBand::from_raw(Bound::Finite(0.0), Bound::Finite(1.0), MetricType::L2).unwrap()
    }

    #[test]
    fn a_fixed_width_clamps_to_nlist_like_top_k_does() {
        let params = VectorRangeSearchParams::new(l2_band_for_params(), 4096);
        assert_eq!(
            params.validate(1024).unwrap(),
            1024,
            "matches the width.min(nlist) clamp in index.rs"
        );
    }

    #[test]
    fn a_zero_nprobe_is_invalid_input() {
        let params = VectorRangeSearchParams::new(l2_band_for_params(), 0);
        assert_eq!(
            params.validate(1024).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn a_zero_nprobe_is_invalid_for_every_metric() {
        for metric in [MetricType::Cosine, MetricType::InnerProduct] {
            let band =
                DistanceBand::from_raw(Bound::Finite(0.0), Bound::Finite(1.0), metric).unwrap();
            let err = VectorRangeSearchParams::new(band, 0)
                .validate(1024)
                .unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "nprobe == 0 on metric {metric:?}"
            );
        }
    }
}
