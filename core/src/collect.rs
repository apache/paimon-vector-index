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

//! Collection abstraction for list scans.
//!
//! A family's per-list scan kernel decides *which* rows it visits and what
//! value it computes for each. What happens to a visited row is a separate
//! concern, and this trait is the seam between the two, so a kernel is written
//! once rather than copied per consumer. Aligned with Faiss's
//! `InvertedListScanner`, which hands rows to a `ResultHandler&` in
//! `scan_codes`. Faiss splits the range case out into a separately named
//! `scan_codes_range` taking a `RangeQueryResult&`; one collector serves both
//! here, so there is a single kernel rather than a pair to keep in step.
//!
//! `ivfflat_io::ReaderTopKHeap`, `topk::TopKHeap`, and [`RangeCollector`]
//! implement the collection policies.

use std::io;

use crate::distance::MetricType;
use crate::range::{Bound, DistanceBand};

/// The interface a scan kernel uses to hand candidate rows to a collector.
pub(crate) trait Collector {
    const VALIDATE_COSINE_INPUTS: bool = false;

    /// Called when the scan kernel rejects a row against [`cutoff`] instead of
    /// delivering it.
    ///
    /// Such a row never reaches [`push`], so a collector that counts the rows a
    /// scan touched cannot recover that count from its `push` calls -- the
    /// kernel has to report the rejection explicitly. The default empty body
    /// keeps this free for collectors that do not track it, which is why the
    /// top-K heap does not implement it.
    ///
    /// "Early" is about where in the row the kernel stopped, and it is not
    /// guaranteed: a row can fail the cutoff on its very last term, or on the
    /// completed sum, and is reported here just the same. So this counts rows
    /// the cutoff excluded, not rows whose evaluation was cut short.
    ///
    /// [`cutoff`]: Collector::cutoff
    /// [`push`]: Collector::push
    #[inline]
    fn note_abandoned(&mut self) {}

    /// The admission threshold currently in force. A row may be abandoned as
    /// soon as its partially accumulated distance exceeds this value.
    ///
    /// `f32::INFINITY` means no pruning is possible, and callers must *not*
    /// enter the early-abandon kernel in that case: it would run a full SIMD
    /// pass that can never abandon anything.
    fn cutoff(&self) -> f32;

    /// Delivers one row, with the value the family's scan computed for it. For
    /// IVF-Flat that value is an exact distance; for IVF-RQ and IVF-SQ it is an
    /// estimate.
    ///
    /// Fallible because a collector may own a resource the scan cannot see: the
    /// oversized-list path streams chunks through a callback, and without a
    /// result type here a collector failure would have to panic to escape it.
    fn push(&mut self, id: i64, value: f32) -> io::Result<()>;
}

/// The threshold at which a partially accumulated L2 distance can be abandoned.
///
/// The **raw** upper cut, with no margin, because the scan prunes against the
/// very accumulation it goes on to commit -- see
/// [`distance::fvec_l2sqr_unless_exceeds`]. `f32::INFINITY` wherever pruning
/// cannot be justified, which only means no row is abandoned early.
///
/// [`distance::fvec_l2sqr_unless_exceeds`]: crate::distance::fvec_l2sqr_unless_exceeds
fn early_abandon_threshold(band: DistanceBand) -> f32 {
    // Only an L2 partial sum monotonically lower-bounds the full distance. A
    // partial inner product or cosine accumulation does not bound the final
    // value, because the remaining terms can be either sign.
    if band.metric() != MetricType::L2 {
        return f32::INFINITY;
    }
    match band.upper() {
        Bound::Finite(upper) => upper,
        Bound::Unbounded => f32::INFINITY,
    }
}

/// Collects the rows falling inside a band. It neither sorts nor truncates:
/// ordering is the caller's business and result caps arrive with later work.
pub(crate) struct RangeCollector {
    band: DistanceBand,
    /// Precomputed early-abandon threshold. Constant for a given band, while
    /// `cutoff()` is called once per row, so it is derived here rather than in
    /// the scan loop.
    cutoff: f32,
    rows: Vec<(i64, f32)>,
    scanned: usize,
    early_abandoned: usize,
}

impl RangeCollector {
    pub(crate) fn new(band: DistanceBand) -> Self {
        Self {
            cutoff: early_abandon_threshold(band),
            band,
            rows: Vec::new(),
            scanned: 0,
            early_abandoned: 0,
        }
    }

    /// Rows read and at least partially evaluated, **including** rows abandoned
    /// early.
    pub(crate) fn scanned(&self) -> usize {
        self.scanned
    }

    pub(crate) fn early_abandoned(&self) -> usize {
        self.early_abandoned
    }

    pub(crate) fn into_rows(self) -> Vec<(i64, f32)> {
        self.rows
    }

    pub(crate) fn merge(&mut self, mut other: Self) {
        self.scanned += other.scanned;
        self.early_abandoned += other.early_abandoned;
        self.rows.append(&mut other.rows);
    }
}

impl Collector for RangeCollector {
    const VALIDATE_COSINE_INPUTS: bool = true;

    #[inline]
    fn note_abandoned(&mut self) {
        self.scanned += 1;
        self.early_abandoned += 1;
    }

    #[inline]
    fn cutoff(&self) -> f32 {
        self.cutoff
    }

    #[inline]
    fn push(&mut self, id: i64, value: f32) -> io::Result<()> {
        self.scanned += 1;
        if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("non-finite distance {value} computed for row {id}"),
            ));
        }
        if self.band.admit(value) {
            self.rows.push((id, value));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::MetricType;
    use crate::range::{Bound, DistanceBand};

    fn l2_band(lower: f32, upper: f32) -> DistanceBand {
        DistanceBand::new(Bound::Finite(lower), Bound::Finite(upper), MetricType::L2).unwrap()
    }

    #[test]
    fn a_range_collector_keeps_only_in_band_rows() {
        let mut collector = RangeCollector::new(l2_band(1.0, 3.0));
        collector.push(10, 0.5).unwrap();
        collector.push(11, 1.0).unwrap();
        collector.push(12, 2.9).unwrap();
        collector.push(13, 3.0).unwrap();
        assert_eq!(collector.into_rows(), vec![(11, 1.0), (12, 2.9)]);
    }

    #[test]
    fn an_exact_l2_collector_prunes_at_the_raw_upper_cut() {
        // The scan prunes against the same accumulation it commits, so the cut
        // needs no margin. A cutoff *below* `upper` would drop in-band rows; one
        // above it would be dead headroom left over from a divergence that no
        // longer exists.
        // The subnormal cut is deliberate. It used to disable pruning outright,
        // because the widening was a relative margin and the relative-error
        // model does not survive gradual underflow. Monotonicity has no such
        // domain restriction -- adding a non-negative subnormal is still
        // non-decreasing -- so the cut is now used like any other, and the guard
        // that suppressed it is gone.
        for upper in [
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0e-20,
            0.5,
            3.0,
            1.0e30,
        ] {
            assert_eq!(
                RangeCollector::new(l2_band(0.0, upper)).cutoff().to_bits(),
                upper.to_bits(),
                "upper={upper:e}"
            );
        }
    }

    #[test]
    fn an_unbounded_upper_reports_no_cutoff() {
        let band = DistanceBand::new(Bound::Finite(1.0), Bound::Unbounded, MetricType::L2).unwrap();
        assert_eq!(RangeCollector::new(band).cutoff(), f32::INFINITY);
    }

    #[test]
    fn a_non_l2_metric_never_prunes_on_a_partial_sum() {
        // A partial cosine or inner-product accumulation does not bound the full
        // value, so the cutoff must stay infinite no matter what the band says.
        for metric in [MetricType::Cosine, MetricType::InnerProduct] {
            let band = DistanceBand::new(Bound::Finite(0.1), Bound::Finite(0.5), metric).unwrap();
            assert_eq!(
                RangeCollector::new(band).cutoff(),
                f32::INFINITY,
                "metric {metric:?} must not expose a finite cutoff"
            );
        }
    }

    #[test]
    fn the_collector_rejects_a_non_finite_computed_value() {
        // Fail loud on every family: under inner product -inf means "extremely
        // similar", so dropping it silently would erase a row that should have
        // matched, and the donor's "exact families may drop it" rule was only
        // ever derived for L2.
        //
        // Scope: this is the *collector's* contract, covering every path that
        // computes a full value. It deliberately does not cover early abandon,
        // which never reaches `push` -- an abandoned row provably satisfies
        // "full distance > upper", so classifying it out of band is the right
        // answer whether or not the computation overflowed.
        let mut collector = RangeCollector::new(l2_band(0.0, 10.0));
        assert!(collector.push(1, f32::NAN).is_err());
        assert!(collector.push(2, f32::INFINITY).is_err());
    }

    #[test]
    fn early_abandoned_rows_are_counted_as_scanned() {
        // rows_scanned means "rows read and at least partially evaluated", which
        // includes abandoned rows. Those never reach push, so the scan kernel
        // has to report them through note_abandoned.
        let mut collector = RangeCollector::new(l2_band(1.0, 2.0));
        collector.note_abandoned();
        collector.push(1, 1.5).unwrap();
        assert_eq!(collector.scanned(), 2);
        assert_eq!(collector.early_abandoned(), 1);
    }

    #[test]
    fn rows_are_counted_even_when_rejected() {
        let mut collector = RangeCollector::new(l2_band(1.0, 2.0));
        collector.push(1, 0.0).unwrap();
        collector.push(2, 1.5).unwrap();
        assert_eq!(collector.scanned(), 2);
    }
}
