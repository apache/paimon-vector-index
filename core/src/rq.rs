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

pub const DEFAULT_RQ_ROTATION_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
pub const DEFAULT_RQ_ROTATION_ROUNDS: u32 = 3;
pub const RQ_BYTE_LUT_MIN_LIST_SIZE: usize = 64;
pub const DEFAULT_RQ_QUERY_BITS: usize = 0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RQCodeFactors {
    pub residual_norm_sqr: f32,
    pub vector_norm_sqr: f32,
    pub dp_multiplier: f32,
}

impl RQCodeFactors {
    pub fn zero() -> Self {
        Self {
            residual_norm_sqr: 0.0,
            vector_norm_sqr: 0.0,
            dp_multiplier: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RQRotation {
    d: usize,
    seed: u64,
    rounds: u32,
    ops: Vec<KacOp>,
}

#[derive(Debug, Clone, Copy)]
struct KacOp {
    i: usize,
    j: usize,
    cos: f32,
    sin: f32,
}

impl RQRotation {
    pub fn new(d: usize, seed: u64, rounds: u32) -> Self {
        let mut rng = SplitMix64::new(seed ^ (d as u64).rotate_left(17));
        let mut ops = Vec::new();
        if d >= 2 {
            for _ in 0..rounds {
                let mut order: Vec<usize> = (0..d).collect();
                for i in (1..d).rev() {
                    let j = rng.next_usize(i + 1);
                    order.swap(i, j);
                }
                for pair in order.chunks_exact(2) {
                    let angle = (rng.next_f32() * 2.0 - 1.0) * std::f32::consts::PI;
                    let (sin, cos) = angle.sin_cos();
                    ops.push(KacOp {
                        i: pair[0],
                        j: pair[1],
                        cos,
                        sin,
                    });
                }
            }
        }
        Self {
            d,
            seed,
            rounds,
            ops,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn rounds(&self) -> u32 {
        self.rounds
    }

    pub fn apply(&self, values: &mut [f32]) {
        debug_assert_eq!(values.len(), self.d);
        for op in &self.ops {
            let x = values[op.i];
            let y = values[op.j];
            values[op.i] = op.cos * x - op.sin * y;
            values[op.j] = op.sin * x + op.cos * y;
        }
    }
}

#[derive(Debug, Clone)]
pub struct RaBitQuantizer {
    d: usize,
    inv_sqrt_d: f32,
}

#[derive(Debug, Clone)]
pub struct RQDistanceContext {
    d: usize,
    code_size: usize,
    rotated_query_residual: Vec<f32>,
    query_residual_norm_sqr: f32,
    query_norm_sqr: f32,
    byte_signed_sums: Option<Vec<f32>>,
    quantized_query: Option<RQQuantizedQuery>,
}

#[derive(Debug, Clone)]
struct RQQuantizedQuery {
    scale: f32,
    sign_bits: Vec<u8>,
    magnitude_bit_planes: Vec<Vec<u8>>,
}

impl RaBitQuantizer {
    pub fn new(d: usize) -> Self {
        let inv_sqrt_d = if d == 0 { 1.0 } else { 1.0 / (d as f32).sqrt() };
        Self { d, inv_sqrt_d }
    }

    pub fn code_size(&self) -> usize {
        self.d.div_ceil(8)
    }

    pub fn encode(
        &self,
        rotated_residual: &[f32],
        vector_norm_sqr: f32,
        code: &mut [u8],
    ) -> RQCodeFactors {
        debug_assert_eq!(rotated_residual.len(), self.d);
        debug_assert!(code.len() >= self.code_size());
        code[..self.code_size()].fill(0);

        let (residual_norm_sqr, abs_sum) = fvec_norm_l2sqr_abs_sum(rotated_residual);
        for (byte_idx, chunk) in rotated_residual.chunks(8).enumerate() {
            let mut byte = 0u8;
            for (bit, &value) in chunk.iter().enumerate() {
                if value > 0.0 {
                    byte |= 1u8 << bit;
                }
            }
            code[byte_idx] = byte;
        }

        let dp_multiplier = if abs_sum > f32::EPSILON {
            residual_norm_sqr / (abs_sum * self.inv_sqrt_d)
        } else {
            0.0
        };
        RQCodeFactors {
            residual_norm_sqr,
            vector_norm_sqr,
            dp_multiplier,
        }
    }

    pub fn distance_to_code(
        &self,
        rotated_query_residual: &[f32],
        query: &[f32],
        code: &[u8],
        factors: RQCodeFactors,
        metric: MetricType,
    ) -> f32 {
        debug_assert_eq!(rotated_query_residual.len(), self.d);
        debug_assert_eq!(query.len(), self.d);

        let context = self.prepare_distance_context(rotated_query_residual.to_vec(), query, false);
        self.distance_to_code_prepared(&context, code, factors, metric)
    }

    pub fn prepare_distance_context(
        &self,
        rotated_query_residual: Vec<f32>,
        query: &[f32],
        build_byte_lut: bool,
    ) -> RQDistanceContext {
        self.prepare_distance_context_with_query_bits(
            rotated_query_residual,
            query,
            build_byte_lut,
            DEFAULT_RQ_QUERY_BITS,
        )
    }

    pub fn prepare_distance_context_with_query_bits(
        &self,
        rotated_query_residual: Vec<f32>,
        query: &[f32],
        build_byte_lut: bool,
        query_bits: usize,
    ) -> RQDistanceContext {
        debug_assert_eq!(rotated_query_residual.len(), self.d);
        debug_assert_eq!(query.len(), self.d);
        assert!(
            is_supported_query_bits(query_bits),
            "unsupported IVF-RQ query_bits {}; expected 0, 4, or 8",
            query_bits
        );

        let query_residual_norm_sqr = fvec_norm_l2sqr(&rotated_query_residual);
        let query_norm_sqr = fvec_norm_l2sqr(query);
        let quantized_query = if query_bits == DEFAULT_RQ_QUERY_BITS {
            None
        } else {
            Some(self.quantize_query(&rotated_query_residual, query_bits))
        };
        let byte_signed_sums = if quantized_query.is_none() && build_byte_lut {
            Some(self.build_byte_signed_sums(&rotated_query_residual))
        } else {
            None
        };

        RQDistanceContext {
            d: self.d,
            code_size: self.code_size(),
            rotated_query_residual,
            query_residual_norm_sqr,
            query_norm_sqr,
            byte_signed_sums,
            quantized_query,
        }
    }

    pub fn distance_to_code_prepared(
        &self,
        context: &RQDistanceContext,
        code: &[u8],
        factors: RQCodeFactors,
        metric: MetricType,
    ) -> f32 {
        debug_assert_eq!(context.d, self.d);
        debug_assert!(code.len() >= context.code_size);

        let signed_query_sum = self.signed_query_sum(context, code);
        let approx_ip = factors.dp_multiplier * signed_query_sum * self.inv_sqrt_d;
        let approx_l2 = (factors.residual_norm_sqr + context.query_residual_norm_sqr
            - 2.0 * approx_ip)
            .max(0.0);

        match metric {
            MetricType::L2 => approx_l2,
            MetricType::Cosine => 0.5 * approx_l2,
            MetricType::InnerProduct => {
                let base = factors.residual_norm_sqr - factors.vector_norm_sqr;
                let pre_dist = base + context.query_residual_norm_sqr - 2.0 * approx_ip;
                0.5 * (pre_dist - context.query_norm_sqr)
            }
        }
    }

    fn signed_query_sum(&self, context: &RQDistanceContext, code: &[u8]) -> f32 {
        if let Some(quantized_query) = &context.quantized_query {
            return quantized_query.signed_query_sum(code, context.code_size);
        }

        if let Some(byte_signed_sums) = &context.byte_signed_sums {
            let mut sum = 0.0f32;
            for byte_idx in 0..context.code_size {
                sum += byte_signed_sums[byte_idx * 256 + code[byte_idx] as usize];
            }
            return sum;
        }

        let mut sum = 0.0f32;
        for (dim, &value) in context.rotated_query_residual.iter().enumerate() {
            sum += if get_bit(code, dim) { value } else { -value };
        }
        sum
    }

    fn quantize_query(
        &self,
        rotated_query_residual: &[f32],
        query_bits: usize,
    ) -> RQQuantizedQuery {
        let magnitude_bits = query_bits - 1;
        let max_level = (1usize << magnitude_bits) - 1;
        let code_size = self.code_size();
        let max_abs = rotated_query_residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        let scale = if max_abs > f32::EPSILON {
            max_abs / max_level as f32
        } else {
            0.0
        };
        let mut sign_bits = vec![0u8; code_size];
        let mut magnitude_bit_planes = vec![vec![0u8; code_size]; magnitude_bits];

        if scale == 0.0 {
            return RQQuantizedQuery {
                scale,
                sign_bits,
                magnitude_bit_planes,
            };
        }

        for (dim, &value) in rotated_query_residual.iter().enumerate() {
            if value >= 0.0 {
                sign_bits[dim / 8] |= 1u8 << (dim % 8);
            }
            let level = (value.abs() / scale).round().clamp(0.0, max_level as f32) as usize;
            for (bit, plane) in magnitude_bit_planes.iter_mut().enumerate() {
                if (level >> bit) & 1 != 0 {
                    plane[dim / 8] |= 1u8 << (dim % 8);
                }
            }
        }

        RQQuantizedQuery {
            scale,
            sign_bits,
            magnitude_bit_planes,
        }
    }

    fn build_byte_signed_sums(&self, rotated_query_residual: &[f32]) -> Vec<f32> {
        let code_size = self.code_size();
        let mut byte_signed_sums = vec![0.0f32; code_size * 256];
        for byte_idx in 0..code_size {
            let dim_base = byte_idx * 8;
            let dim_end = (dim_base + 8).min(self.d);
            for pattern in 0..256usize {
                let mut sum = 0.0f32;
                for dim in dim_base..dim_end {
                    let bit = (pattern >> (dim - dim_base)) & 1;
                    let value = rotated_query_residual[dim];
                    sum += if bit != 0 { value } else { -value };
                }
                byte_signed_sums[byte_idx * 256 + pattern] = sum;
            }
        }
        byte_signed_sums
    }
}

impl RQQuantizedQuery {
    fn signed_query_sum(&self, code: &[u8], code_size: usize) -> f32 {
        if self.scale == 0.0 {
            return 0.0;
        }

        let mut signed_level_sum = 0i64;
        for (bit, plane) in self.magnitude_bit_planes.iter().enumerate() {
            let weight = 1i64 << bit;
            let mut plane_sum = 0i64;
            let mut offset = 0usize;

            while offset + 8 <= code_size {
                let selected = u64::from_le_bytes(plane[offset..offset + 8].try_into().unwrap());
                if selected != 0 {
                    let code_bits =
                        u64::from_le_bytes(code[offset..offset + 8].try_into().unwrap());
                    let sign_bits =
                        u64::from_le_bytes(self.sign_bits[offset..offset + 8].try_into().unwrap());
                    let same_sign = !(code_bits ^ sign_bits) & selected;
                    plane_sum += 2 * same_sign.count_ones() as i64 - selected.count_ones() as i64;
                }
                offset += 8;
            }

            while offset < code_size {
                let selected = plane[offset];
                if selected != 0 {
                    let same_sign = !(code[offset] ^ self.sign_bits[offset]) & selected;
                    plane_sum += 2 * same_sign.count_ones() as i64 - selected.count_ones() as i64;
                }
                offset += 1;
            }

            signed_level_sum += weight * plane_sum;
        }

        self.scale * signed_level_sum as f32
    }
}

#[inline]
pub fn is_supported_query_bits(query_bits: usize) -> bool {
    matches!(query_bits, 0 | 4 | 8)
}

fn get_bit(code: &[u8], dim: usize) -> bool {
    code[dim / 8] & (1u8 << (dim % 8)) != 0
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn fvec_norm_l2sqr_abs_sum(values: &[f32]) -> (f32, f32) {
    if is_x86_feature_detected!("avx2") && values.len() >= 8 {
        unsafe { fvec_norm_l2sqr_abs_sum_avx2(values) }
    } else {
        fvec_norm_l2sqr_abs_sum_scalar(values)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn fvec_norm_l2sqr_abs_sum(values: &[f32]) -> (f32, f32) {
    unsafe { fvec_norm_l2sqr_abs_sum_neon(values) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn fvec_norm_l2sqr_abs_sum(values: &[f32]) -> (f32, f32) {
    fvec_norm_l2sqr_abs_sum_scalar(values)
}

#[cfg(any(
    target_arch = "x86_64",
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
#[inline]
fn fvec_norm_l2sqr_abs_sum_scalar(values: &[f32]) -> (f32, f32) {
    let mut norm = 0.0f32;
    let mut abs_sum = 0.0f32;
    for &value in values {
        norm += value * value;
        abs_sum += value.abs();
    }
    (norm, abs_sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fvec_norm_l2sqr_abs_sum_avx2(values: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;

    let n = values.len();
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut norm_sum = _mm256_setzero_ps();
    let mut abs_sum_vec = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(i)) };
        norm_sum = _mm256_add_ps(norm_sum, _mm256_mul_ps(value, value));
        abs_sum_vec = _mm256_add_ps(abs_sum_vec, _mm256_and_ps(value, abs_mask));
        i += 8;
    }

    let norm_hi = _mm256_extractf128_ps::<1>(norm_sum);
    let norm_lo = _mm256_castps256_ps128(norm_sum);
    let norm_128 = _mm_add_ps(norm_lo, norm_hi);
    let norm_64 = _mm_add_ps(norm_128, _mm_movehl_ps(norm_128, norm_128));
    let norm_32 = _mm_add_ss(norm_64, _mm_shuffle_ps::<1>(norm_64, norm_64));
    let mut norm = _mm_cvtss_f32(norm_32);

    let abs_hi = _mm256_extractf128_ps::<1>(abs_sum_vec);
    let abs_lo = _mm256_castps256_ps128(abs_sum_vec);
    let abs_128 = _mm_add_ps(abs_lo, abs_hi);
    let abs_64 = _mm_add_ps(abs_128, _mm_movehl_ps(abs_128, abs_128));
    let abs_32 = _mm_add_ss(abs_64, _mm_shuffle_ps::<1>(abs_64, abs_64));
    let mut abs_sum = _mm_cvtss_f32(abs_32);

    while i < n {
        let value = unsafe { *values.get_unchecked(i) };
        norm += value * value;
        abs_sum += value.abs();
        i += 1;
    }
    (norm, abs_sum)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fvec_norm_l2sqr_abs_sum_neon(values: &[f32]) -> (f32, f32) {
    use std::arch::aarch64::*;

    let n = values.len();
    let mut norm_sum = vdupq_n_f32(0.0);
    let mut abs_sum_vec = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let value = unsafe { vld1q_f32(values.as_ptr().add(i)) };
        norm_sum = vmlaq_f32(norm_sum, value, value);
        abs_sum_vec = vaddq_f32(abs_sum_vec, vabsq_f32(value));
        i += 4;
    }

    let mut norm = vaddvq_f32(norm_sum);
    let mut abs_sum = vaddvq_f32(abs_sum_vec);
    while i < n {
        let value = unsafe { *values.get_unchecked(i) };
        norm += value * value;
        abs_sum += value.abs();
        i += 1;
    }
    (norm, abs_sum)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }

    fn next_f32(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as u32;
        mantissa as f32 / ((1u32 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rabit_quantizer_estimates_self_distance_as_zero() {
        let d = 8;
        let rotation = RQRotation::new(d, DEFAULT_RQ_ROTATION_SEED, DEFAULT_RQ_ROTATION_ROUNDS);
        let quantizer = RaBitQuantizer::new(d);
        let centroid = vec![1.0; d];
        let vector = vec![2.0, 1.5, 0.75, 3.0, 1.25, -1.0, 4.0, 2.5];
        let mut residual: Vec<f32> = vector
            .iter()
            .zip(centroid.iter())
            .map(|(&x, &c)| x - c)
            .collect();
        rotation.apply(&mut residual);

        let mut code = vec![0u8; quantizer.code_size()];
        let factors = quantizer.encode(&residual, fvec_norm_l2sqr(&vector), &mut code);
        let dist = quantizer.distance_to_code(&residual, &vector, &code, factors, MetricType::L2);

        assert!(
            dist <= 1e-5,
            "self distance should be close to zero: {dist}"
        );
    }

    #[test]
    fn distance_context_byte_lut_matches_scalar_path() {
        let d = 16;
        let quantizer = RaBitQuantizer::new(d);
        let rotated_residual: Vec<f32> = (0..d).map(|i| i as f32 * 0.25 - 1.5).collect();
        let query: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.125).collect();
        let rotated_query_residual: Vec<f32> = (0..d).map(|i| (i as f32 - 3.0) * 0.2).collect();

        let mut code = vec![0u8; quantizer.code_size()];
        let factors = quantizer.encode(&rotated_residual, fvec_norm_l2sqr(&query), &mut code);
        code[0] = 0b1010_0101;
        code[1] = 0b0101_1010;

        let scalar_context =
            quantizer.prepare_distance_context(rotated_query_residual.clone(), &query, false);
        let lut_context = quantizer.prepare_distance_context(rotated_query_residual, &query, true);

        for metric in [MetricType::L2, MetricType::Cosine, MetricType::InnerProduct] {
            let scalar =
                quantizer.distance_to_code_prepared(&scalar_context, &code, factors, metric);
            let lut = quantizer.distance_to_code_prepared(&lut_context, &code, factors, metric);
            assert!(
                (scalar - lut).abs() < 1e-5,
                "metric {:?}: scalar {} != lut {}",
                metric,
                scalar,
                lut
            );
        }
    }

    #[test]
    fn quantized_query_bit_planes_match_scalar_quantization() {
        let d = 24;
        let quantizer = RaBitQuantizer::new(d);
        let rotated_query_residual: Vec<f32> = (0..d).map(|i| (i as f32 - 11.0) * 0.17).collect();
        let query: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.03125).collect();
        let mut code = vec![0u8; quantizer.code_size()];
        for (byte_idx, byte) in code.iter_mut().enumerate() {
            *byte = if byte_idx % 2 == 0 {
                0b1010_1100
            } else {
                0b0101_0011
            };
        }

        for query_bits in [4, 8] {
            let context = quantizer.prepare_distance_context_with_query_bits(
                rotated_query_residual.clone(),
                &query,
                true,
                query_bits,
            );
            let quantized_query = context.quantized_query.as_ref().unwrap();
            let actual = quantized_query.signed_query_sum(&code, quantizer.code_size());
            let expected =
                scalar_quantized_signed_query_sum(&rotated_query_residual, &code, query_bits);

            assert!(
                (actual - expected).abs() < 1e-5,
                "query_bits {}: {} != {}",
                query_bits,
                actual,
                expected
            );
            assert!(context.byte_signed_sums.is_none());
        }
    }

    #[test]
    fn norm_l2sqr_abs_sum_helper_matches_expected_values() {
        let values = [-3.0f32, 4.0, 0.5, -0.25, 8.0, -2.0, 1.25, -6.0, 7.0];
        let (norm, abs_sum) = fvec_norm_l2sqr_abs_sum(&values);

        let expected_norm: f32 = values.iter().map(|value| value * value).sum();
        let expected_abs_sum: f32 = values.iter().map(|value| value.abs()).sum();
        assert!((norm - expected_norm).abs() < 1e-5);
        assert!((abs_sum - expected_abs_sum).abs() < 1e-5);
    }

    #[test]
    fn rotation_preserves_l2_norm() {
        let d = 16;
        let rotation = RQRotation::new(d, 11, 3);
        let mut vector: Vec<f32> = (0..d).map(|i| i as f32 - 3.0).collect();
        let before = fvec_norm_l2sqr(&vector);
        rotation.apply(&mut vector);
        let after = fvec_norm_l2sqr(&vector);
        assert!((before - after).abs() < 1e-4);
    }

    fn scalar_quantized_signed_query_sum(
        rotated_query_residual: &[f32],
        code: &[u8],
        query_bits: usize,
    ) -> f32 {
        let magnitude_bits = query_bits - 1;
        let max_level = (1usize << magnitude_bits) - 1;
        let max_abs = rotated_query_residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        if max_abs <= f32::EPSILON {
            return 0.0;
        }
        let scale = max_abs / max_level as f32;
        let mut sum = 0i64;
        for (dim, &value) in rotated_query_residual.iter().enumerate() {
            let code_sign = if get_bit(code, dim) { 1i64 } else { -1i64 };
            let query_sign = if value >= 0.0 { 1i64 } else { -1i64 };
            let level = (value.abs() / scale).round().clamp(0.0, max_level as f32) as i64;
            sum += code_sign * query_sign * level;
        }
        scale * sum as f32
    }
}
