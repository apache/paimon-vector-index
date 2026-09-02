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

use crate::blas::sgemm_a_bt;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetricType {
    L2 = 0,
    InnerProduct = 1,
    Cosine = 2,
}

impl MetricType {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(MetricType::L2),
            1 => Some(MetricType::InnerProduct),
            2 => Some(MetricType::Cosine),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MetricType::L2 => "l2",
            MetricType::InnerProduct => "inner_product",
            MetricType::Cosine => "cosine",
        }
    }
}

/// Squared L2 distance between two vectors.
#[inline]
pub fn fvec_l2sqr(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "fvec_l2sqr inputs must have the same length"
    );
    fvec_l2sqr_simd(a, b)
}

/// Squared L2 distances from one vector to four candidates.
///
/// Each output uses the same accumulation order as [`fvec_l2sqr`] so callers
/// can batch shared query loads without changing exact pruning decisions.
#[inline]
pub fn fvec_l2sqr_four(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    for candidate in [b0, b1, b2, b3] {
        assert_eq!(
            a.len(),
            candidate.len(),
            "fvec_l2sqr_four inputs must have the same length"
        );
    }
    fvec_l2sqr_four_simd(a, b0, b1, b2, b3)
}

/// Returns whether `scale * squared_l2` is strictly greater than `threshold`.
///
/// L2 components are non-negative, so the architecture-specific kernels can
/// stop after any completed SIMD block that already exceeds the threshold.
#[inline]
pub fn fvec_l2sqr_scaled_exceeds(a: &[f32], b: &[f32], scale: f32, threshold: f32) -> bool {
    assert_eq!(
        a.len(),
        b.len(),
        "fvec_l2sqr_scaled_exceeds inputs must have the same length"
    );
    assert!(
        scale.is_finite() && scale >= 0.0,
        "fvec_l2sqr_scaled_exceeds scale must be finite and non-negative"
    );
    if threshold < 0.0 {
        return true;
    }
    fvec_l2sqr_scaled_exceeds_simd(a, b, scale, threshold)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn fvec_l2sqr_simd(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { fvec_l2sqr_avx2(a, b) }
    } else {
        fvec_l2sqr_scalar(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn fvec_l2sqr_simd(a: &[f32], b: &[f32]) -> f32 {
    unsafe { fvec_l2sqr_neon(a, b) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn fvec_l2sqr_simd(a: &[f32], b: &[f32]) -> f32 {
    fvec_l2sqr_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn fvec_l2sqr_four_simd(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { fvec_l2sqr_four_avx2(a, b0, b1, b2, b3) }
    } else {
        fvec_l2sqr_four_scalar(a, b0, b1, b2, b3)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn fvec_l2sqr_four_simd(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    unsafe { fvec_l2sqr_four_neon(a, b0, b1, b2, b3) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn fvec_l2sqr_four_simd(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    fvec_l2sqr_four_scalar(a, b0, b1, b2, b3)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn fvec_l2sqr_scaled_exceeds_simd(a: &[f32], b: &[f32], scale: f32, threshold: f32) -> bool {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { fvec_l2sqr_scaled_exceeds_avx2(a, b, scale, threshold) }
    } else {
        fvec_l2sqr_scaled_exceeds_scalar(a, b, scale, threshold)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn fvec_l2sqr_scaled_exceeds_simd(a: &[f32], b: &[f32], scale: f32, threshold: f32) -> bool {
    unsafe { fvec_l2sqr_scaled_exceeds_neon(a, b, scale, threshold) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn fvec_l2sqr_scaled_exceeds_simd(a: &[f32], b: &[f32], scale: f32, threshold: f32) -> bool {
    fvec_l2sqr_scaled_exceeds_scalar(a, b, scale, threshold)
}

#[cfg(any(
    target_arch = "x86_64",
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
#[inline]
fn fvec_l2sqr_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(any(
    target_arch = "x86_64",
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
#[inline]
fn fvec_l2sqr_four_scalar(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    let candidates = [b0, b1, b2, b3];
    let mut sums = [0.0f32; 4];
    for i in 0..a.len() {
        let value = a[i];
        for candidate in 0..4 {
            let d = value - candidates[candidate][i];
            sums[candidate] += d * d;
        }
    }
    sums
}

#[cfg(any(
    target_arch = "x86_64",
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
#[inline]
fn fvec_l2sqr_scaled_exceeds_scalar(a: &[f32], b: &[f32], scale: f32, threshold: f32) -> bool {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
        if scale * sum > threshold {
            return true;
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fvec_l2sqr_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        i += 8;
    }

    let hi = _mm256_extractf128_ps::<1>(sum);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
    let mut result = _mm_cvtss_f32(sum32);

    while i < n {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        result += d * d;
        i += 1;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fvec_l2sqr_four_avx2(
    a: &[f32],
    b0: &[f32],
    b1: &[f32],
    b2: &[f32],
    b3: &[f32],
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let candidates = [b0, b1, b2, b3];
    let mut sums = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 8 <= a.len() {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        for candidate in 0..4 {
            let vb = unsafe { _mm256_loadu_ps(candidates[candidate].as_ptr().add(i)) };
            let diff = _mm256_sub_ps(va, vb);
            sums[candidate] = _mm256_add_ps(sums[candidate], _mm256_mul_ps(diff, diff));
        }
        i += 8;
    }

    macro_rules! horizontal_sum {
        ($sum:expr) => {{
            let hi = _mm256_extractf128_ps::<1>($sum);
            let lo = _mm256_castps256_ps128($sum);
            let sum128 = _mm_add_ps(lo, hi);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
            _mm_cvtss_f32(sum32)
        }};
    }
    let mut results = [
        horizontal_sum!(sums[0]),
        horizontal_sum!(sums[1]),
        horizontal_sum!(sums[2]),
        horizontal_sum!(sums[3]),
    ];
    while i < a.len() {
        let value = unsafe { *a.get_unchecked(i) };
        for candidate in 0..4 {
            let d = value - unsafe { *candidates[candidate].get_unchecked(i) };
            results[candidate] += d * d;
        }
        i += 1;
    }
    results
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fvec_l2sqr_scaled_exceeds_avx2(a: &[f32], b: &[f32], scale: f32, threshold: f32) -> bool {
    use std::arch::x86_64::*;

    let mut total = 0.0f32;
    let mut i = 0;
    while i + 128 <= a.len() {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        let block_end = i + 128;
        while i < block_end {
            for (offset, sum) in [
                (0, &mut sum0),
                (8, &mut sum1),
                (16, &mut sum2),
                (24, &mut sum3),
            ] {
                let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i + offset)) };
                let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i + offset)) };
                let diff = _mm256_sub_ps(va, vb);
                *sum = _mm256_add_ps(*sum, _mm256_mul_ps(diff, diff));
            }
            i += 32;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        let hi = _mm256_extractf128_ps::<1>(sum);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
        total += _mm_cvtss_f32(sum32);
        if scale * total > threshold {
            return true;
        }
    }

    let mut tail_sum = _mm256_setzero_ps();
    while i + 8 <= a.len() {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let diff = _mm256_sub_ps(va, vb);
        tail_sum = _mm256_add_ps(tail_sum, _mm256_mul_ps(diff, diff));
        i += 8;
    }
    let hi = _mm256_extractf128_ps::<1>(tail_sum);
    let lo = _mm256_castps256_ps128(tail_sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
    total += _mm_cvtss_f32(sum32);
    if scale * total > threshold {
        return true;
    }

    while i < a.len() {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        total += d * d;
        if scale * total > threshold {
            return true;
        }
        i += 1;
    }
    false
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fvec_l2sqr_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 8 <= n {
        let va0 = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        let vb0 = unsafe { vld1q_f32(b.as_ptr().add(i)) };
        let diff0 = vsubq_f32(va0, vb0);
        sum0 = vmlaq_f32(sum0, diff0, diff0);

        let va1 = unsafe { vld1q_f32(a.as_ptr().add(i + 4)) };
        let vb1 = unsafe { vld1q_f32(b.as_ptr().add(i + 4)) };
        let diff1 = vsubq_f32(va1, vb1);
        sum1 = vmlaq_f32(sum1, diff1, diff1);

        i += 8;
    }

    let mut result = vaddvq_f32(vaddq_f32(sum0, sum1));
    while i < n {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        result += d * d;
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fvec_l2sqr_four_neon(
    a: &[f32],
    b0: &[f32],
    b1: &[f32],
    b2: &[f32],
    b3: &[f32],
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let candidates = [b0, b1, b2, b3];
    let mut sums0 = [vdupq_n_f32(0.0); 4];
    let mut sums1 = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i + 8 <= a.len() {
        let va0 = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        let va1 = unsafe { vld1q_f32(a.as_ptr().add(i + 4)) };
        for candidate in 0..4 {
            let vb0 = unsafe { vld1q_f32(candidates[candidate].as_ptr().add(i)) };
            let diff0 = vsubq_f32(va0, vb0);
            sums0[candidate] = vmlaq_f32(sums0[candidate], diff0, diff0);

            let vb1 = unsafe { vld1q_f32(candidates[candidate].as_ptr().add(i + 4)) };
            let diff1 = vsubq_f32(va1, vb1);
            sums1[candidate] = vmlaq_f32(sums1[candidate], diff1, diff1);
        }
        i += 8;
    }

    let mut results = [
        vaddvq_f32(vaddq_f32(sums0[0], sums1[0])),
        vaddvq_f32(vaddq_f32(sums0[1], sums1[1])),
        vaddvq_f32(vaddq_f32(sums0[2], sums1[2])),
        vaddvq_f32(vaddq_f32(sums0[3], sums1[3])),
    ];
    while i < a.len() {
        let value = unsafe { *a.get_unchecked(i) };
        for candidate in 0..4 {
            let d = value - unsafe { *candidates[candidate].get_unchecked(i) };
            results[candidate] += d * d;
        }
        i += 1;
    }
    results
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fvec_l2sqr_scaled_exceeds_neon(a: &[f32], b: &[f32], scale: f32, threshold: f32) -> bool {
    use std::arch::aarch64::*;

    let mut total = 0.0f32;
    let mut i = 0;
    while i + 128 <= a.len() {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);
        let block_end = i + 128;
        while i < block_end {
            let va0 = unsafe { vld1q_f32(a.as_ptr().add(i)) };
            let vb0 = unsafe { vld1q_f32(b.as_ptr().add(i)) };
            let diff0 = vsubq_f32(va0, vb0);
            sum0 = vmlaq_f32(sum0, diff0, diff0);

            let va1 = unsafe { vld1q_f32(a.as_ptr().add(i + 4)) };
            let vb1 = unsafe { vld1q_f32(b.as_ptr().add(i + 4)) };
            let diff1 = vsubq_f32(va1, vb1);
            sum1 = vmlaq_f32(sum1, diff1, diff1);

            let va2 = unsafe { vld1q_f32(a.as_ptr().add(i + 8)) };
            let vb2 = unsafe { vld1q_f32(b.as_ptr().add(i + 8)) };
            let diff2 = vsubq_f32(va2, vb2);
            sum2 = vmlaq_f32(sum2, diff2, diff2);

            let va3 = unsafe { vld1q_f32(a.as_ptr().add(i + 12)) };
            let vb3 = unsafe { vld1q_f32(b.as_ptr().add(i + 12)) };
            let diff3 = vsubq_f32(va3, vb3);
            sum3 = vmlaq_f32(sum3, diff3, diff3);
            i += 16;
        }
        total += vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
        if scale * total > threshold {
            return true;
        }
    }

    let mut tail_sum0 = vdupq_n_f32(0.0);
    let mut tail_sum1 = vdupq_n_f32(0.0);
    while i + 8 <= a.len() {
        let va0 = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        let vb0 = unsafe { vld1q_f32(b.as_ptr().add(i)) };
        let diff0 = vsubq_f32(va0, vb0);
        tail_sum0 = vmlaq_f32(tail_sum0, diff0, diff0);

        let va1 = unsafe { vld1q_f32(a.as_ptr().add(i + 4)) };
        let vb1 = unsafe { vld1q_f32(b.as_ptr().add(i + 4)) };
        let diff1 = vsubq_f32(va1, vb1);
        tail_sum1 = vmlaq_f32(tail_sum1, diff1, diff1);
        i += 8;
    }
    total += vaddvq_f32(vaddq_f32(tail_sum0, tail_sum1));
    if scale * total > threshold {
        return true;
    }

    while i < a.len() {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        total += d * d;
        if scale * total > threshold {
            return true;
        }
        i += 1;
    }
    false
}

/// Squared L2 distance on sub-vectors.
pub fn fvec_l2sqr_sub(a: &[f32], a_off: usize, b: &[f32], b_off: usize, len: usize) -> f32 {
    fvec_l2sqr(&a[a_off..a_off + len], &b[b_off..b_off + len])
}

/// Inner product of two vectors.
#[inline]
pub fn fvec_inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    fvec_inner_product_simd(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn fvec_inner_product_simd(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { fvec_inner_product_avx2(a, b) }
    } else {
        fvec_inner_product_scalar(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn fvec_inner_product_simd(a: &[f32], b: &[f32]) -> f32 {
    unsafe { fvec_inner_product_neon(a, b) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn fvec_inner_product_simd(a: &[f32], b: &[f32]) -> f32 {
    fvec_inner_product_scalar(a, b)
}

#[inline]
#[cfg(not(target_arch = "aarch64"))]
fn fvec_inner_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fvec_inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
        i += 8;
    }

    let hi = _mm256_extractf128_ps::<1>(sum);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
    let mut result = _mm_cvtss_f32(sum32);

    while i < n {
        result += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fvec_inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 8 <= n {
        let va0 = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        let vb0 = unsafe { vld1q_f32(b.as_ptr().add(i)) };
        sum0 = vmlaq_f32(sum0, va0, vb0);

        let va1 = unsafe { vld1q_f32(a.as_ptr().add(i + 4)) };
        let vb1 = unsafe { vld1q_f32(b.as_ptr().add(i + 4)) };
        sum1 = vmlaq_f32(sum1, va1, vb1);

        i += 8;
    }

    let mut result = vaddvq_f32(vaddq_f32(sum0, sum1));
    while i < n {
        result += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
        i += 1;
    }
    result
}

/// Squared L2 norm of a vector.
#[inline]
pub fn fvec_norm_l2sqr(a: &[f32]) -> f32 {
    fvec_norm_l2sqr_simd(a)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn fvec_norm_l2sqr_simd(a: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { fvec_norm_l2sqr_avx2(a) }
    } else {
        fvec_norm_l2sqr_scalar(a)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn fvec_norm_l2sqr_simd(a: &[f32]) -> f32 {
    unsafe { fvec_norm_l2sqr_neon(a) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn fvec_norm_l2sqr_simd(a: &[f32]) -> f32 {
    fvec_norm_l2sqr_scalar(a)
}

#[inline]
#[cfg(not(target_arch = "aarch64"))]
fn fvec_norm_l2sqr_scalar(a: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in a {
        sum += v * v;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fvec_norm_l2sqr_avx2(a: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, va));
        i += 8;
    }

    let hi = _mm256_extractf128_ps::<1>(sum);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
    let mut result = _mm_cvtss_f32(sum32);

    while i < n {
        let v = unsafe { *a.get_unchecked(i) };
        result += v * v;
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fvec_norm_l2sqr_neon(a: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 8 <= n {
        let va0 = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        sum0 = vmlaq_f32(sum0, va0, va0);

        let va1 = unsafe { vld1q_f32(a.as_ptr().add(i + 4)) };
        sum1 = vmlaq_f32(sum1, va1, va1);

        i += 8;
    }

    let mut result = vaddvq_f32(vaddq_f32(sum0, sum1));
    while i < n {
        let v = unsafe { *a.get_unchecked(i) };
        result += v * v;
        i += 1;
    }
    result
}

/// Normalize a vector in-place to unit length. Returns the original norm.
pub fn fvec_normalize(v: &mut [f32]) -> f32 {
    let norm = fvec_norm_l2sqr(v).sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    norm
}

/// Distance used for ranking search results. Lower is better for all metrics.
pub fn fvec_distance(query: &[f32], vector: &[f32], metric: MetricType) -> f32 {
    match metric {
        MetricType::L2 => fvec_l2sqr(query, vector),
        MetricType::InnerProduct => -fvec_inner_product(query, vector),
        MetricType::Cosine => {
            let nq = fvec_norm_l2sqr(query).sqrt();
            let nv = fvec_norm_l2sqr(vector).sqrt();
            fvec_cosine_distance_with_norms(query, vector, nq, nv)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryDistance<'a> {
    query: &'a [f32],
    metric: MetricType,
    query_norm: f32,
}

impl<'a> QueryDistance<'a> {
    #[inline]
    pub(crate) fn new(query: &'a [f32], metric: MetricType) -> Self {
        let query_norm = if metric == MetricType::Cosine {
            fvec_norm_l2sqr(query).sqrt()
        } else {
            0.0
        };
        Self {
            query,
            metric,
            query_norm,
        }
    }

    #[inline]
    pub(crate) fn distance_to(&self, vector: &[f32], vector_norm: Option<f32>) -> f32 {
        match self.metric {
            MetricType::L2 => fvec_l2sqr(self.query, vector),
            MetricType::InnerProduct => -fvec_inner_product(self.query, vector),
            MetricType::Cosine => {
                let vector_norm = vector_norm.unwrap_or_else(|| fvec_norm_l2sqr(vector).sqrt());
                fvec_cosine_distance_with_norms(self.query, vector, self.query_norm, vector_norm)
            }
        }
    }
}

#[inline]
fn fvec_cosine_distance_with_norms(a: &[f32], b: &[f32], a_norm: f32, b_norm: f32) -> f32 {
    let denom = a_norm * b_norm;
    if denom > 0.0 {
        1.0 - fvec_inner_product(a, b) / denom
    } else {
        1.0
    }
}

pub fn preprocess_vectors(data: &[f32], n: usize, d: usize, metric: MetricType) -> Vec<f32> {
    let mut processed = data[..n * d].to_vec();
    if metric == MetricType::Cosine && d > 0 {
        if n > 1_000 {
            processed.par_chunks_mut(d).for_each(|vector| {
                fvec_normalize(vector);
            });
        } else {
            processed.chunks_mut(d).for_each(|vector| {
                fvec_normalize(vector);
            });
        }
    }
    processed
}

#[cfg(test)]
mod preprocess_tests {
    use super::*;

    #[test]
    fn test_preprocess_vectors_normalizes_cosine_only() {
        let data = vec![3.0, 4.0, 1.0, 2.0];

        assert_eq!(
            preprocess_vectors(&data, 1, 2, MetricType::L2),
            vec![3.0, 4.0]
        );
        assert_eq!(
            preprocess_vectors(&data, 1, 2, MetricType::Cosine),
            vec![0.6, 0.8]
        );
    }

    #[test]
    fn test_preprocess_vectors_normalizes_cosine_in_parallel() {
        let data = [3.0, 4.0].repeat(1_001);
        let processed = preprocess_vectors(&data, 1_001, 2, MetricType::Cosine);

        assert!(processed.chunks_exact(2).all(|vector| vector == [0.6, 0.8]));
    }

    #[test]
    fn test_preprocess_vectors_accepts_zero_dimension() {
        assert!(preprocess_vectors(&[], 1, 0, MetricType::Cosine).is_empty());
        assert!(preprocess_vectors(&[], 1_001, 0, MetricType::Cosine).is_empty());
    }
}

/// Compute result[i] = a[i] + bf * b[i]. Used for precomputed table merging.
/// Aligned with Faiss's fvec_madd.
pub fn fvec_madd(a: &[f32], b: &[f32], bf: f32, result: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), result.len());
    fvec_madd_simd(a, b, bf, result);
}

#[cfg(target_arch = "x86_64")]
fn fvec_madd_simd(a: &[f32], b: &[f32], bf: f32, result: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { fvec_madd_avx2(a, b, bf, result) };
    } else {
        fvec_madd_scalar(a, b, bf, result);
    }
}

#[cfg(target_arch = "aarch64")]
fn fvec_madd_simd(a: &[f32], b: &[f32], bf: f32, result: &mut [f32]) {
    unsafe { fvec_madd_neon(a, b, bf, result) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn fvec_madd_simd(a: &[f32], b: &[f32], bf: f32, result: &mut [f32]) {
    fvec_madd_scalar(a, b, bf, result);
}

#[inline]
#[allow(dead_code)]
fn fvec_madd_scalar(a: &[f32], b: &[f32], bf: f32, result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] + bf * b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fvec_madd_avx2(a: &[f32], b: &[f32], bf: f32, result: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let vbf = _mm256_set1_ps(bf);
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vr = _mm256_add_ps(va, _mm256_mul_ps(vbf, vb));
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
        i += 8;
    }
    while i < n {
        result[i] = a[i] + bf * b[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fvec_madd_neon(a: &[f32], b: &[f32], bf: f32, result: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = a.len();
    let vbf = vdupq_n_f32(bf);
    let mut i = 0;
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let vr = vmlaq_f32(va, vbf, vb);
        vst1q_f32(result.as_mut_ptr().add(i), vr);
        i += 4;
    }
    while i < n {
        result[i] = a[i] + bf * b[i];
        i += 1;
    }
}

/// SIMD-accelerated squared L2 distance for sub-vectors (used by PQ distance table).
pub fn fvec_l2sqr_batch(
    query_sub: &[f32],
    centroids: &[f32],
    dsub: usize,
    ksub: usize,
    result: &mut [f32],
) {
    debug_assert!(query_sub.len() >= dsub);
    debug_assert!(centroids.len() >= ksub * dsub);
    debug_assert!(result.len() >= ksub);

    if dsub >= 4 && ksub >= 8 {
        fvec_ip_batch(query_sub, centroids, dsub, ksub, result);
        let q_norm = fvec_norm_l2sqr(&query_sub[..dsub]);
        for j in 0..ksub {
            let c_off = j * dsub;
            let c_norm = fvec_norm_l2sqr(&centroids[c_off..c_off + dsub]);
            result[j] = (q_norm + c_norm - 2.0 * result[j]).max(0.0);
        }
    } else {
        for j in 0..ksub {
            let c_off = j * dsub;
            result[j] = fvec_l2sqr(&query_sub[..dsub], &centroids[c_off..c_off + dsub]);
        }
    }
}

/// SIMD-accelerated inner product for sub-vectors (used by PQ distance table).
pub fn fvec_ip_batch(
    query_sub: &[f32],
    centroids: &[f32],
    dsub: usize,
    ksub: usize,
    result: &mut [f32],
) {
    debug_assert!(query_sub.len() >= dsub);
    debug_assert!(centroids.len() >= ksub * dsub);
    debug_assert!(result.len() >= ksub);

    if dsub >= 4 && ksub >= 8 {
        sgemm_a_bt(
            1,
            ksub,
            dsub,
            1.0,
            &query_sub[..dsub],
            &centroids[..ksub * dsub],
            0.0,
            &mut result[..ksub],
        );
    } else {
        for j in 0..ksub {
            let c_off = j * dsub;
            result[j] = fvec_inner_product(&query_sub[..dsub], &centroids[c_off..c_off + dsub]);
        }
    }
}

/// Scan a batch of 4-bit PQ codes.
/// Approach (aligned with Lance/Faiss):
///   1. Compute first FLAT_NUM vectors with exact f32 (calibrate qmax)
///   2. Quantize distance table to u8
///   3. Accumulate distances in u8 domain via SIMD shuffle
///   4. Dequantize back to f32 at the end
///
/// codes: nibble-packed [count * (m/2)], row-major.
/// sim_table: [M * 16] f32 distance table.
pub fn scan_4bit_simd(sim_table: &[f32], codes: &[u8], count: usize, m: usize, dists: &mut [f32]) {
    const FLAT_NUM: usize = 200;

    let cs = m / 2; // code_size = m/2 bytes per vector

    // Step 1: Compute first FLAT_NUM vectors with f32 precision
    let flat_end = count.min(FLAT_NUM);
    for i in 0..flat_end {
        let base = i * cs;
        let mut d = 0.0f32;
        for pair in 0..cs {
            let byte = codes[base + pair];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            d += sim_table[(pair * 2) * 16 + lo];
            d += sim_table[(pair * 2 + 1) * 16 + hi];
        }
        dists[i] = d;
    }

    if count <= FLAT_NUM {
        return;
    }

    // Step 2: Determine qmax from the first FLAT_NUM distances
    let qmax = dists[..flat_end].iter().cloned().fold(f32::MIN, f32::max);

    // Quantize the entire distance table [M * 16] to u8
    let qmin = sim_table.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = (qmax - qmin).max(1e-10);
    let factor = 255.0 / range;

    let qtable: Vec<u8> = sim_table
        .iter()
        .map(|&d| ((d - qmin) * factor).clamp(0.0, 255.0) as u8)
        .collect();

    // Step 3: Scan remaining vectors in u8 domain
    // Use u16 accumulators to avoid overflow (M/2 pairs × max 255 per pair × 2 ≤ 65535 for M ≤ 256)
    let mut q_dists = vec![0u16; count];

    for pair in 0..cs {
        let qtab_lo = &qtable[(pair * 2) * 16..(pair * 2 + 1) * 16];
        let qtab_hi = &qtable[(pair * 2 + 1) * 16..(pair * 2 + 2) * 16];

        // SIMD-friendly inner loop: sequential code access, 16-entry table fits in register
        for i in flat_end..count {
            let byte = codes[i * cs + pair];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            q_dists[i] += qtab_lo[lo] as u16 + qtab_hi[hi] as u16;
        }
    }

    // Step 4: Dequantize back to f32
    let inv_factor = range / 255.0;
    let base_dist = qmin * m as f32; // M sub-quantizers each contribute at least qmin
    for i in flat_end..count {
        dists[i] = q_dists[i] as f32 * inv_factor + base_dist;
    }
}

/// Compute PQ distance from a precomputed distance table.
/// table layout: [M][ksub], codes: M bytes.
/// Each code[m] indexes into table[m * ksub + code[m]].
#[inline]
pub fn pq_distance_from_table(table: &[f32], codes: &[u8], m: usize, ksub: usize) -> f32 {
    pq_distance_from_table_simd(table, codes, m, ksub)
}

/// Process 4 codes at once for better instruction-level parallelism.
#[inline]
pub fn pq_distance_four_codes(
    table: &[f32],
    codes: &[u8],
    m: usize,
    ksub: usize,
    offsets: [usize; 4],
) -> [f32; 4] {
    let mut dists = [0.0f32; 4];
    for i in 0..m {
        let base = i * ksub;
        for j in 0..4 {
            dists[j] += table[base + codes[offsets[j] + i] as usize];
        }
    }
    dists
}

// SIMD-accelerated PQ distance table lookup.
#[cfg(target_arch = "x86_64")]
#[inline]
fn pq_distance_from_table_simd(table: &[f32], codes: &[u8], m: usize, ksub: usize) -> f32 {
    if is_x86_feature_detected!("avx2") && m >= 8 && ksub == 256 {
        unsafe { pq_distance_avx2(table, codes, m) }
    } else {
        pq_distance_scalar(table, codes, m, ksub)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn pq_distance_from_table_simd(table: &[f32], codes: &[u8], m: usize, ksub: usize) -> f32 {
    if ksub == 256 && m >= 4 {
        unsafe { pq_distance_neon(table, codes, m) }
    } else {
        pq_distance_scalar(table, codes, m, ksub)
    }
}

/// NEON-accelerated PQ distance with manual gather + vaddq_f32 accumulation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pq_distance_neon(table: &[f32], codes: &[u8], m: usize) -> f32 {
    use std::arch::aarch64::*;

    let ksub = 256usize;
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= m {
        let d0 = *table.get_unchecked(i * ksub + *codes.get_unchecked(i) as usize);
        let d1 = *table.get_unchecked((i + 1) * ksub + *codes.get_unchecked(i + 1) as usize);
        let d2 = *table.get_unchecked((i + 2) * ksub + *codes.get_unchecked(i + 2) as usize);
        let d3 = *table.get_unchecked((i + 3) * ksub + *codes.get_unchecked(i + 3) as usize);

        let arr = [d0, d1, d2, d3];
        let v = vld1q_f32(arr.as_ptr());
        sum = vaddq_f32(sum, v);
        i += 4;
    }

    let mut result = vaddvq_f32(sum);

    while i < m {
        result += *table.get_unchecked(i * ksub + *codes.get_unchecked(i) as usize);
        i += 1;
    }

    result
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn pq_distance_from_table_simd(table: &[f32], codes: &[u8], m: usize, ksub: usize) -> f32 {
    pq_distance_scalar(table, codes, m, ksub)
}

#[inline]
fn pq_distance_scalar(table: &[f32], codes: &[u8], m: usize, ksub: usize) -> f32 {
    let mut dist = 0.0f32;
    for i in 0..m {
        dist += table[i * ksub + codes[i] as usize];
    }
    dist
}

/// AVX2 PQ distance using gather instructions.
/// Aligned with Faiss's pq_code_distance-avx2.h.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pq_distance_avx2(table: &[f32], codes: &[u8], m: usize) -> f32 {
    use std::arch::x86_64::*;

    let ksub = 256usize;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 sub-quantizers at a time
    while i + 8 <= m {
        let offsets = _mm256_set_epi32(
            (7 * ksub + codes[i + 7] as usize) as i32,
            (6 * ksub + codes[i + 6] as usize) as i32,
            (5 * ksub + codes[i + 5] as usize) as i32,
            (4 * ksub + codes[i + 4] as usize) as i32,
            (3 * ksub + codes[i + 3] as usize) as i32,
            (2 * ksub + codes[i + 2] as usize) as i32,
            (ksub + codes[i + 1] as usize) as i32,
            (codes[i] as usize) as i32,
        );

        let tab_ptr = table.as_ptr().add(i * ksub);
        let gathered = _mm256_i32gather_ps::<4>(tab_ptr, offsets);
        sum = _mm256_add_ps(sum, gathered);
        i += 8;
    }

    // Horizontal sum of the 8 floats in sum
    let hi = _mm256_extractf128_ps::<1>(sum);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<1>(sum64, sum64));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remaining sub-quantizers
    while i < m {
        result += table[i * ksub + codes[i] as usize];
        i += 1;
    }

    result
}

/// Compute distance between query and a set of vectors, return top-k.
pub fn fvec_distances_batch(
    query: &[f32],
    vectors: &[f32],
    n: usize,
    d: usize,
    metric: MetricType,
    distances: &mut [f32],
) {
    let distance = QueryDistance::new(query, metric);
    for i in 0..n {
        let vec = &vectors[i * d..(i + 1) * d];
        distances[i] = distance.distance_to(vec, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2sqr() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((fvec_l2sqr(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2sqr_four_matches_individual_simd_kernels_exactly() {
        for len in [0, 1, 7, 8, 17, 67] {
            let query = (0..len)
                .map(|index| ((index * 29 + 7) % 41) as f32 * 0.125 - 2.0)
                .collect::<Vec<_>>();
            let candidates = std::array::from_fn::<_, 4, _>(|candidate| {
                (0..len)
                    .map(|index| {
                        ((index * (candidate + 11) + candidate * 17 + 3) % 53) as f32 * 0.0625 - 1.5
                    })
                    .collect::<Vec<_>>()
            });

            let actual = fvec_l2sqr_four(
                &query,
                &candidates[0],
                &candidates[1],
                &candidates[2],
                &candidates[3],
            );
            let expected =
                std::array::from_fn(|candidate| fvec_l2sqr(&query, &candidates[candidate]));

            assert_eq!(actual, expected, "length {len}");
        }
    }

    #[test]
    fn test_l2sqr_scaled_exceeds_uses_a_strict_threshold_across_multiple_blocks() {
        let a = [0.0f32; 129];
        let mut b = [0.0f32; 129];
        b[0] = 2.0;
        b[128] = 3.0;

        assert!(!fvec_l2sqr_scaled_exceeds(&a, &b, 2.0, 26.0));
        assert!(fvec_l2sqr_scaled_exceeds(&a, &b, 2.0, 25.0));
        assert!(fvec_l2sqr_scaled_exceeds(&a, &b, 1.0, 3.0));
        assert!(!fvec_l2sqr_scaled_exceeds(&a, &a, 2.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "fvec_l2sqr inputs must have the same length")]
    fn test_l2sqr_rejects_mismatched_lengths() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let _ = fvec_l2sqr(&a, &b);
    }

    #[test]
    fn test_inner_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((fvec_inner_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_and_norm_large_vector() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.25 - 3.0).collect();
        let b: Vec<f32> = (0..37).map(|i| 2.0 - i as f32 * 0.125).collect();

        let expected_dot: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
        let expected_norm: f32 = a.iter().map(|&x| x * x).sum();

        assert!((fvec_inner_product(&a, &b) - expected_dot).abs() < 1e-4);
        assert!((fvec_norm_l2sqr(&a) - expected_norm).abs() < 1e-4);
    }

    #[test]
    fn test_batch_distance_helpers_match_scalar() {
        let dsub = 5;
        let ksub = 9;
        let query: Vec<f32> = (0..dsub).map(|i| i as f32 * 0.3 - 0.7).collect();
        let centroids: Vec<f32> = (0..ksub * dsub).map(|i| i as f32 * 0.07 - 1.2).collect();

        let mut l2 = vec![0.0f32; ksub];
        let mut ip = vec![0.0f32; ksub];
        fvec_l2sqr_batch(&query, &centroids, dsub, ksub, &mut l2);
        fvec_ip_batch(&query, &centroids, dsub, ksub, &mut ip);

        for j in 0..ksub {
            let c = &centroids[j * dsub..(j + 1) * dsub];
            assert!((l2[j] - fvec_l2sqr(&query, c)).abs() < 1e-5);
            assert!((ip[j] - fvec_inner_product(&query, c)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fvec_distance_by_metric() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];

        assert!((fvec_distance(&a, &b, MetricType::L2) - 2.0).abs() < 1e-6);
        assert!((fvec_distance(&a, &b, MetricType::InnerProduct) - 0.0).abs() < 1e-6);
        assert!((fvec_distance(&a, &b, MetricType::Cosine) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metric_type_as_str() {
        assert_eq!(MetricType::L2.as_str(), "l2");
        assert_eq!(MetricType::InnerProduct.as_str(), "inner_product");
        assert_eq!(MetricType::Cosine.as_str(), "cosine");
    }

    #[test]
    fn test_normalize() {
        let mut v = [3.0, 4.0];
        fvec_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_pq_distance_scalar() {
        let table = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // 2 sub-q, 4 centroids
        let codes = [1u8, 3u8];
        let dist = pq_distance_scalar(&table, &codes, 2, 4);
        // table[0*4 + 1] + table[1*4 + 3] = 0.2 + 0.8 = 1.0
        assert!((dist - 1.0).abs() < 1e-6);
    }
}
