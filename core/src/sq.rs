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

use crate::distance::{fvec_norm_l2sqr, MetricType};

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarQuantizer {
    pub d: usize,
    pub min: f32,
    pub max: f32,
    pub mins: Vec<f32>,
    pub maxs: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarQuantizerDecodeLut {
    d: usize,
    values: Vec<f32>,
}

impl ScalarQuantizer {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            min: 0.0,
            max: 0.0,
            mins: vec![0.0; d],
            maxs: vec![0.0; d],
        }
    }

    pub fn with_bounds(d: usize, min: f32, max: f32) -> Self {
        Self {
            d,
            min,
            max,
            mins: vec![min; d],
            maxs: vec![max; d],
        }
    }

    pub fn with_dimension_bounds(d: usize, mins: Vec<f32>, maxs: Vec<f32>) -> Self {
        assert_eq!(mins.len(), d);
        assert_eq!(maxs.len(), d);
        let mut sq = Self {
            d,
            min: 0.0,
            max: 0.0,
            mins,
            maxs,
        };
        sq.refresh_global_bounds();
        sq
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        let len = n * self.d;
        let values = &data[..len];
        if n == 0 || self.d == 0 {
            self.min = 0.0;
            self.max = 0.0;
            self.mins.fill(0.0);
            self.maxs.fill(0.0);
            return;
        }

        self.ensure_bounds_len();
        self.mins.fill(f32::INFINITY);
        self.maxs.fill(f32::NEG_INFINITY);
        update_bounds_batch(values, n, self.d, &mut self.mins, &mut self.maxs);
        self.refresh_global_bounds();
    }

    pub fn code_size(&self) -> usize {
        self.d
    }

    pub fn encode_batch(&self, data: &[f32], n: usize, codes: &mut [u8]) {
        let len = n * self.d;
        assert!(data.len() >= len);
        assert!(codes.len() >= len);

        encode_batch_simd(data, n, self.d, &self.mins, &self.maxs, codes);
    }

    pub fn encode(&self, vector: &[f32], code: &mut [u8]) {
        self.encode_batch(vector, 1, code);
    }

    pub fn decode_batch(&self, codes: &[u8], n: usize, vectors: &mut [f32]) {
        let len = n * self.d;
        assert!(codes.len() >= len);
        assert!(vectors.len() >= len);

        for row in 0..n {
            let base = row * self.d;
            for dim in 0..self.d {
                vectors[base + dim] = self.decode_value(codes[base + dim], dim);
            }
        }
    }

    pub fn decode_batch_with_offset(
        &self,
        codes: &[u8],
        n: usize,
        offset: &[f32],
        vectors: &mut [f32],
    ) {
        let len = n * self.d;
        assert!(codes.len() >= len);
        assert!(offset.len() >= self.d);
        assert!(vectors.len() >= len);

        decode_batch_with_offset_simd(codes, n, self.d, &self.mins, &self.maxs, offset, vectors);
    }

    pub fn decode(&self, code: &[u8], vector: &mut [f32]) {
        self.decode_batch(code, 1, vector);
    }

    pub fn distance_to_code(&self, query: &[f32], code: &[u8], metric: MetricType) -> f32 {
        self.distance_to_code_with_context(query, code, self.distance_context(query, metric))
    }

    pub fn distance_context(&self, query: &[f32], metric: MetricType) -> DistanceContext {
        debug_assert!(query.len() >= self.d);
        DistanceContext::new(&query[..self.d], metric)
    }

    pub fn distance_to_code_with_context(
        &self,
        query: &[f32],
        code: &[u8],
        context: DistanceContext,
    ) -> f32 {
        self.distance_to_code_impl(query, code, &[], false, context)
    }

    pub fn distance_to_code_with_offset_with_context(
        &self,
        query: &[f32],
        code: &[u8],
        offset: &[f32],
        context: DistanceContext,
    ) -> f32 {
        debug_assert!(query.len() >= self.d);
        debug_assert!(code.len() >= self.d);
        debug_assert!(offset.len() >= self.d);

        self.distance_to_code_impl(query, code, offset, true, context)
    }

    pub fn distance_to_code_with_lut_with_context(
        &self,
        query: &[f32],
        code: &[u8],
        lut: &ScalarQuantizerDecodeLut,
        context: DistanceContext,
    ) -> f32 {
        self.distance_to_code_lut_impl(query, code, &[], false, lut, context)
    }

    pub fn distance_to_code_with_lut_offset_with_context(
        &self,
        query: &[f32],
        code: &[u8],
        offset: &[f32],
        lut: &ScalarQuantizerDecodeLut,
        context: DistanceContext,
    ) -> f32 {
        debug_assert!(query.len() >= self.d);
        debug_assert!(code.len() >= self.d);
        debug_assert!(offset.len() >= self.d);

        self.distance_to_code_lut_impl(query, code, offset, true, lut, context)
    }

    pub fn build_decode_lut(&self) -> ScalarQuantizerDecodeLut {
        let mut values = vec![0.0f32; self.d * 256];
        for dim in 0..self.d {
            let base = dim * 256;
            for code in 0..256 {
                values[base + code] = self.decode_value(code as u8, dim);
            }
        }
        ScalarQuantizerDecodeLut { d: self.d, values }
    }

    fn distance_to_code_impl(
        &self,
        query: &[f32],
        code: &[u8],
        offset: &[f32],
        use_offset: bool,
        context: DistanceContext,
    ) -> f32 {
        debug_assert!(query.len() >= self.d);
        debug_assert!(code.len() >= self.d);

        match context.metric {
            MetricType::L2 => {
                let mut sum = 0.0f32;
                for i in 0..self.d {
                    let diff =
                        query[i] - self.decode_value_with_offset(code[i], i, offset, use_offset);
                    sum += diff * diff;
                }
                sum
            }
            MetricType::InnerProduct => {
                let mut dot = 0.0f32;
                for i in 0..self.d {
                    dot += query[i] * self.decode_value_with_offset(code[i], i, offset, use_offset);
                }
                -dot
            }
            MetricType::Cosine => {
                let mut dot = 0.0f32;
                let mut vector_norm = 0.0f32;
                for i in 0..self.d {
                    let value = self.decode_value_with_offset(code[i], i, offset, use_offset);
                    dot += query[i] * value;
                    vector_norm += value * value;
                }
                let denom = context.query_norm * vector_norm.sqrt();
                if denom > 0.0 {
                    1.0 - dot / denom
                } else {
                    1.0
                }
            }
        }
    }

    fn distance_to_code_lut_impl(
        &self,
        query: &[f32],
        code: &[u8],
        offset: &[f32],
        use_offset: bool,
        lut: &ScalarQuantizerDecodeLut,
        context: DistanceContext,
    ) -> f32 {
        debug_assert!(query.len() >= self.d);
        debug_assert!(code.len() >= self.d);
        debug_assert_eq!(lut.d, self.d);

        match context.metric {
            MetricType::L2 => {
                let mut sum = 0.0f32;
                for i in 0..self.d {
                    let diff = query[i]
                        - decoded_lut_value_with_offset(lut, code[i], i, offset, use_offset);
                    sum += diff * diff;
                }
                sum
            }
            MetricType::InnerProduct => {
                let mut dot = 0.0f32;
                for i in 0..self.d {
                    dot += query[i]
                        * decoded_lut_value_with_offset(lut, code[i], i, offset, use_offset);
                }
                -dot
            }
            MetricType::Cosine => {
                let mut dot = 0.0f32;
                let mut vector_norm = 0.0f32;
                for i in 0..self.d {
                    let value = decoded_lut_value_with_offset(lut, code[i], i, offset, use_offset);
                    dot += query[i] * value;
                    vector_norm += value * value;
                }
                let denom = context.query_norm * vector_norm.sqrt();
                if denom > 0.0 {
                    1.0 - dot / denom
                } else {
                    1.0
                }
            }
        }
    }

    fn decode_value_with_offset(
        &self,
        code: u8,
        dim: usize,
        offset: &[f32],
        use_offset: bool,
    ) -> f32 {
        let value = self.decode_value(code, dim);
        if use_offset {
            value + offset[dim]
        } else {
            value
        }
    }

    fn decode_value(&self, code: u8, dim: usize) -> f32 {
        let min = self.mins[dim];
        let max = self.maxs[dim];
        if min >= max {
            min
        } else {
            min + code as f32 * (max - min) / 255.0
        }
    }

    fn ensure_bounds_len(&mut self) {
        if self.mins.len() != self.d {
            self.mins.resize(self.d, 0.0);
        }
        if self.maxs.len() != self.d {
            self.maxs.resize(self.d, 0.0);
        }
    }

    fn refresh_global_bounds(&mut self) {
        if self.d == 0 {
            self.min = 0.0;
            self.max = 0.0;
            return;
        }
        self.min = self.mins.iter().copied().fold(f32::INFINITY, f32::min);
        self.max = self.maxs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }
}

fn update_bounds_batch(data: &[f32], n: usize, d: usize, mins: &mut [f32], maxs: &mut [f32]) {
    update_bounds_batch_simd(data, n, d, mins, maxs);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn update_bounds_batch_simd(data: &[f32], n: usize, d: usize, mins: &mut [f32], maxs: &mut [f32]) {
    if is_x86_feature_detected!("avx2") && d >= 8 {
        unsafe { update_bounds_batch_avx2(data, n, d, mins, maxs) };
    } else {
        update_bounds_batch_scalar(data, n, d, mins, maxs);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn update_bounds_batch_simd(data: &[f32], n: usize, d: usize, mins: &mut [f32], maxs: &mut [f32]) {
    unsafe { update_bounds_batch_neon(data, n, d, mins, maxs) };
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn update_bounds_batch_simd(data: &[f32], n: usize, d: usize, mins: &mut [f32], maxs: &mut [f32]) {
    update_bounds_batch_scalar(data, n, d, mins, maxs);
}

#[cfg(not(target_arch = "aarch64"))]
fn update_bounds_batch_scalar(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &mut [f32],
    maxs: &mut [f32],
) {
    for vector in data[..n * d].chunks_exact(d) {
        for i in 0..d {
            mins[i] = mins[i].min(vector[i]);
            maxs[i] = maxs[i].max(vector[i]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_bounds_batch_avx2(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &mut [f32],
    maxs: &mut [f32],
) {
    use std::arch::x86_64::*;

    for row in 0..n {
        let base = row * d;
        let mut dim = 0;
        while dim + 8 <= d {
            let values = unsafe { _mm256_loadu_ps(data.as_ptr().add(base + dim)) };
            let current_min = unsafe { _mm256_loadu_ps(mins.as_ptr().add(dim)) };
            let current_max = unsafe { _mm256_loadu_ps(maxs.as_ptr().add(dim)) };
            unsafe {
                _mm256_storeu_ps(
                    mins.as_mut_ptr().add(dim),
                    _mm256_min_ps(current_min, values),
                );
                _mm256_storeu_ps(
                    maxs.as_mut_ptr().add(dim),
                    _mm256_max_ps(current_max, values),
                );
            }
            dim += 8;
        }
        while dim < d {
            let value = unsafe { *data.get_unchecked(base + dim) };
            let min_ref = unsafe { mins.get_unchecked_mut(dim) };
            *min_ref = min_ref.min(value);
            let max_ref = unsafe { maxs.get_unchecked_mut(dim) };
            *max_ref = max_ref.max(value);
            dim += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn update_bounds_batch_neon(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &mut [f32],
    maxs: &mut [f32],
) {
    use std::arch::aarch64::*;

    for row in 0..n {
        let base = row * d;
        let mut dim = 0;
        while dim + 4 <= d {
            let values = unsafe { vld1q_f32(data.as_ptr().add(base + dim)) };
            let current_min = unsafe { vld1q_f32(mins.as_ptr().add(dim)) };
            let current_max = unsafe { vld1q_f32(maxs.as_ptr().add(dim)) };
            unsafe {
                vst1q_f32(mins.as_mut_ptr().add(dim), vminq_f32(current_min, values));
                vst1q_f32(maxs.as_mut_ptr().add(dim), vmaxq_f32(current_max, values));
            }
            dim += 4;
        }
        while dim < d {
            let value = unsafe { *data.get_unchecked(base + dim) };
            let min_ref = unsafe { mins.get_unchecked_mut(dim) };
            *min_ref = min_ref.min(value);
            let max_ref = unsafe { maxs.get_unchecked_mut(dim) };
            *max_ref = max_ref.max(value);
            dim += 1;
        }
    }
}

fn encode_batch_simd(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    codes: &mut [u8],
) {
    encode_batch_simd_impl(data, n, d, mins, maxs, codes);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn encode_batch_simd_impl(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    codes: &mut [u8],
) {
    if is_x86_feature_detected!("avx2") && d >= 8 {
        unsafe { encode_batch_avx2(data, n, d, mins, maxs, codes) };
    } else {
        encode_batch_scalar(data, n, d, mins, maxs, codes);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn encode_batch_simd_impl(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    codes: &mut [u8],
) {
    unsafe { encode_batch_neon(data, n, d, mins, maxs, codes) };
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn encode_batch_simd_impl(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    codes: &mut [u8],
) {
    encode_batch_scalar(data, n, d, mins, maxs, codes);
}

#[cfg(not(target_arch = "aarch64"))]
fn encode_batch_scalar(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    codes: &mut [u8],
) {
    for row in 0..n {
        let base = row * d;
        for dim in 0..d {
            codes[base + dim] = encode_value(data[base + dim], mins[dim], maxs[dim]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn encode_batch_avx2(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    codes: &mut [u8],
) {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let max_code = _mm256_set1_ps(255.0);
    let mut scaled = [0.0f32; 8];
    for row in 0..n {
        let base = row * d;
        let mut dim = 0;
        while dim + 8 <= d {
            let values = unsafe { _mm256_loadu_ps(data.as_ptr().add(base + dim)) };
            let minv = unsafe { _mm256_loadu_ps(mins.as_ptr().add(dim)) };
            let maxv = unsafe { _mm256_loadu_ps(maxs.as_ptr().add(dim)) };
            let range = _mm256_sub_ps(maxv, minv);
            let valid = _mm256_cmp_ps::<_CMP_GT_OQ>(maxv, minv);
            let safe_range = _mm256_blendv_ps(one, range, valid);
            let scale = _mm256_blendv_ps(zero, _mm256_div_ps(max_code, safe_range), valid);
            let encoded = _mm256_min_ps(
                max_code,
                _mm256_max_ps(zero, _mm256_mul_ps(_mm256_sub_ps(values, minv), scale)),
            );
            unsafe { _mm256_storeu_ps(scaled.as_mut_ptr(), encoded) };
            for lane in 0..8 {
                codes[base + dim + lane] = scaled[lane].round() as u8;
            }
            dim += 8;
        }
        while dim < d {
            codes[base + dim] = encode_value(data[base + dim], mins[dim], maxs[dim]);
            dim += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_batch_neon(
    data: &[f32],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    codes: &mut [u8],
) {
    use std::arch::aarch64::*;

    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let max_code = vdupq_n_f32(255.0);
    let mut scaled = [0.0f32; 4];
    for row in 0..n {
        let base = row * d;
        let mut dim = 0;
        while dim + 4 <= d {
            let values = unsafe { vld1q_f32(data.as_ptr().add(base + dim)) };
            let minv = unsafe { vld1q_f32(mins.as_ptr().add(dim)) };
            let maxv = unsafe { vld1q_f32(maxs.as_ptr().add(dim)) };
            let range = vsubq_f32(maxv, minv);
            let valid = vcgtq_f32(maxv, minv);
            let safe_range = vbslq_f32(valid, range, one);
            let scale = vbslq_f32(valid, vdivq_f32(max_code, safe_range), zero);
            let encoded = vminq_f32(
                max_code,
                vmaxq_f32(zero, vmulq_f32(vsubq_f32(values, minv), scale)),
            );
            unsafe { vst1q_f32(scaled.as_mut_ptr(), encoded) };
            for lane in 0..4 {
                codes[base + dim + lane] = scaled[lane].round() as u8;
            }
            dim += 4;
        }
        while dim < d {
            codes[base + dim] = encode_value(data[base + dim], mins[dim], maxs[dim]);
            dim += 1;
        }
    }
}

#[inline]
fn encode_value(value: f32, min: f32, max: f32) -> u8 {
    if min >= max {
        0
    } else {
        ((value - min) * 255.0 / (max - min))
            .clamp(0.0, 255.0)
            .round() as u8
    }
}

fn decode_batch_with_offset_simd(
    codes: &[u8],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    offset: &[f32],
    vectors: &mut [f32],
) {
    decode_batch_with_offset_simd_impl(codes, n, d, mins, maxs, offset, vectors);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn decode_batch_with_offset_simd_impl(
    codes: &[u8],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    offset: &[f32],
    vectors: &mut [f32],
) {
    if is_x86_feature_detected!("avx2") && d >= 8 {
        unsafe { decode_batch_with_offset_avx2(codes, n, d, mins, maxs, offset, vectors) };
    } else {
        decode_batch_with_offset_scalar(codes, n, d, mins, maxs, offset, vectors);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn decode_batch_with_offset_simd_impl(
    codes: &[u8],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    offset: &[f32],
    vectors: &mut [f32],
) {
    unsafe { decode_batch_with_offset_neon(codes, n, d, mins, maxs, offset, vectors) };
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn decode_batch_with_offset_simd_impl(
    codes: &[u8],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    offset: &[f32],
    vectors: &mut [f32],
) {
    decode_batch_with_offset_scalar(codes, n, d, mins, maxs, offset, vectors);
}

#[cfg(not(target_arch = "aarch64"))]
fn decode_batch_with_offset_scalar(
    codes: &[u8],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    offset: &[f32],
    vectors: &mut [f32],
) {
    for row in 0..n {
        let base = row * d;
        for dim in 0..d {
            vectors[base + dim] =
                decode_value(codes[base + dim], mins[dim], maxs[dim]) + offset[dim];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_batch_with_offset_avx2(
    codes: &[u8],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    offset: &[f32],
    vectors: &mut [f32],
) {
    use std::arch::x86_64::*;

    let inv_255 = _mm256_set1_ps(1.0 / 255.0);
    for row in 0..n {
        let base = row * d;
        let mut dim = 0;
        while dim + 8 <= d {
            let code_bytes = unsafe { _mm_loadl_epi64(codes.as_ptr().add(base + dim).cast()) };
            let code_i32 = _mm256_cvtepu8_epi32(code_bytes);
            let code_f32 = _mm256_cvtepi32_ps(code_i32);
            let minv = unsafe { _mm256_loadu_ps(mins.as_ptr().add(dim)) };
            let maxv = unsafe { _mm256_loadu_ps(maxs.as_ptr().add(dim)) };
            let offsetv = unsafe { _mm256_loadu_ps(offset.as_ptr().add(dim)) };
            let decoded = _mm256_add_ps(
                offsetv,
                _mm256_add_ps(
                    minv,
                    _mm256_mul_ps(code_f32, _mm256_mul_ps(_mm256_sub_ps(maxv, minv), inv_255)),
                ),
            );
            let constant = _mm256_cmp_ps::<_CMP_GE_OQ>(minv, maxv);
            let constant_decoded = _mm256_add_ps(minv, offsetv);
            let result = _mm256_blendv_ps(decoded, constant_decoded, constant);
            unsafe { _mm256_storeu_ps(vectors.as_mut_ptr().add(base + dim), result) };
            dim += 8;
        }
        while dim < d {
            vectors[base + dim] =
                decode_value(codes[base + dim], mins[dim], maxs[dim]) + offset[dim];
            dim += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_batch_with_offset_neon(
    codes: &[u8],
    n: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    offset: &[f32],
    vectors: &mut [f32],
) {
    use std::arch::aarch64::*;

    let inv_255 = vdupq_n_f32(1.0 / 255.0);
    for row in 0..n {
        let base = row * d;
        let mut dim = 0;
        while dim + 8 <= d {
            let code_u8 = unsafe { vld1_u8(codes.as_ptr().add(base + dim)) };
            let code_u16 = vmovl_u8(code_u8);
            let low_u32 = vmovl_u16(vget_low_u16(code_u16));
            let high_u32 = vmovl_u16(vget_high_u16(code_u16));

            let min0 = unsafe { vld1q_f32(mins.as_ptr().add(dim)) };
            let max0 = unsafe { vld1q_f32(maxs.as_ptr().add(dim)) };
            let offset0 = unsafe { vld1q_f32(offset.as_ptr().add(dim)) };
            let decoded0 = vaddq_f32(
                offset0,
                vaddq_f32(
                    min0,
                    vmulq_f32(
                        vcvtq_f32_u32(low_u32),
                        vmulq_f32(vsubq_f32(max0, min0), inv_255),
                    ),
                ),
            );
            let constant0 = vcgeq_f32(min0, max0);
            let result0 = vbslq_f32(constant0, vaddq_f32(min0, offset0), decoded0);
            unsafe { vst1q_f32(vectors.as_mut_ptr().add(base + dim), result0) };

            let min1 = unsafe { vld1q_f32(mins.as_ptr().add(dim + 4)) };
            let max1 = unsafe { vld1q_f32(maxs.as_ptr().add(dim + 4)) };
            let offset1 = unsafe { vld1q_f32(offset.as_ptr().add(dim + 4)) };
            let decoded1 = vaddq_f32(
                offset1,
                vaddq_f32(
                    min1,
                    vmulq_f32(
                        vcvtq_f32_u32(high_u32),
                        vmulq_f32(vsubq_f32(max1, min1), inv_255),
                    ),
                ),
            );
            let constant1 = vcgeq_f32(min1, max1);
            let result1 = vbslq_f32(constant1, vaddq_f32(min1, offset1), decoded1);
            unsafe { vst1q_f32(vectors.as_mut_ptr().add(base + dim + 4), result1) };
            dim += 8;
        }
        while dim < d {
            vectors[base + dim] =
                decode_value(codes[base + dim], mins[dim], maxs[dim]) + offset[dim];
            dim += 1;
        }
    }
}

#[inline]
fn decode_value(code: u8, min: f32, max: f32) -> f32 {
    if min >= max {
        min
    } else {
        min + code as f32 * (max - min) / 255.0
    }
}

impl ScalarQuantizerDecodeLut {
    #[inline]
    pub fn decode_value(&self, code: u8, dim: usize) -> f32 {
        debug_assert!(dim < self.d);
        self.values[dim * 256 + code as usize]
    }

    pub fn dimension(&self) -> usize {
        self.d
    }
}

#[inline]
fn decoded_lut_value_with_offset(
    lut: &ScalarQuantizerDecodeLut,
    code: u8,
    dim: usize,
    offset: &[f32],
    use_offset: bool,
) -> f32 {
    let value = lut.decode_value(code, dim);
    if use_offset {
        value + offset[dim]
    } else {
        value
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DistanceContext {
    metric: MetricType,
    query_norm: f32,
}

impl DistanceContext {
    pub fn new(query: &[f32], metric: MetricType) -> Self {
        let query_norm = if metric == MetricType::Cosine {
            fvec_norm_l2sqr(query).sqrt()
        } else {
            0.0
        };
        Self { metric, query_norm }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantizer_round_trips_bounds() {
        let data = vec![-1.0, 0.0, 1.0, 3.0];
        let mut sq = ScalarQuantizer::new(2);

        sq.train(&data, 2);
        let mut codes = vec![0u8; data.len()];
        sq.encode_batch(&data, 2, &mut codes);
        let mut decoded = vec![0.0f32; data.len()];
        sq.decode_batch(&codes, 2, &mut decoded);

        assert_eq!(sq.min, -1.0);
        assert_eq!(sq.max, 3.0);
        assert_eq!(sq.mins, vec![-1.0, 0.0]);
        assert_eq!(sq.maxs, vec![1.0, 3.0]);
        assert_eq!(codes[0], 0);
        assert_eq!(codes[3], 255);
        assert!((decoded[0] + 1.0).abs() < 1e-6);
        assert!((decoded[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_quantizer_constant_input() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let mut sq = ScalarQuantizer::new(2);
        sq.train(&data, 2);

        let mut codes = vec![7u8; data.len()];
        sq.encode_batch(&data, 2, &mut codes);
        let mut decoded = vec![0.0f32; data.len()];
        sq.decode_batch(&codes, 2, &mut decoded);

        assert_eq!(codes, vec![0, 0, 0, 0]);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_scalar_quantizer_uses_per_dimension_bounds() {
        let data = vec![0.0, -100.0, 1.0, 100.0];
        let mut sq = ScalarQuantizer::new(2);
        sq.train(&data, 2);

        let mut codes = vec![0u8; data.len()];
        sq.encode_batch(&data, 2, &mut codes);
        let mut decoded = vec![0.0f32; data.len()];
        sq.decode_batch(&codes, 2, &mut decoded);

        assert_eq!(codes, vec![0, 0, 255, 255]);
        assert!((decoded[0] - 0.0).abs() < 1e-6);
        assert!((decoded[1] + 100.0).abs() < 1e-6);
        assert!((decoded[2] - 1.0).abs() < 1e-6);
        assert!((decoded[3] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_quantizer_wide_batch_paths() {
        let d = 9;
        let n = 5;
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 7 % 23) as f32) * 0.5 - 3.0)
            .collect();
        let mut sq = ScalarQuantizer::new(d);

        sq.train(&data, n);

        for dim in 0..d {
            let expected_min = (0..n)
                .map(|row| data[row * d + dim])
                .fold(f32::INFINITY, f32::min);
            let expected_max = (0..n)
                .map(|row| data[row * d + dim])
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(sq.mins[dim], expected_min);
            assert_eq!(sq.maxs[dim], expected_max);
        }

        let mut codes = vec![0u8; n * d];
        sq.encode_batch(&data, n, &mut codes);
        let mut decoded = vec![0.0f32; n * d];
        let offset: Vec<f32> = (0..d).map(|dim| dim as f32 * 0.25).collect();
        sq.decode_batch_with_offset(&codes, n, &offset, &mut decoded);

        for row in 0..n {
            for dim in 0..d {
                let expected = sq.decode_value(codes[row * d + dim], dim) + offset[dim];
                assert!((decoded[row * d + dim] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_scalar_quantizer_distance_to_code() {
        let sq = ScalarQuantizer::with_bounds(2, 0.0, 1.0);
        let mut code = vec![0u8; 2];
        sq.encode(&[1.0, 0.0], &mut code);

        let dist = sq.distance_to_code(&[1.0, 0.0], &code, MetricType::L2);

        assert!(dist < 1e-6);
    }

    #[test]
    fn test_scalar_quantizer_decode_batch_with_offset() {
        let sq = ScalarQuantizer::with_dimension_bounds(2, vec![0.0, -1.0], vec![1.0, 1.0]);
        let codes = vec![255, 0, 0, 255];
        let mut decoded = vec![0.0f32; 4];

        sq.decode_batch_with_offset(&codes, 2, &[10.0, 20.0], &mut decoded);

        assert!((decoded[0] - 11.0).abs() < 1e-6);
        assert!((decoded[1] - 19.0).abs() < 1e-6);
        assert!((decoded[2] - 10.0).abs() < 1e-6);
        assert!((decoded[3] - 21.0).abs() < 1e-6);
    }
}
