// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::distance::MetricType;

pub const DEFAULT_RQ_BITS: usize = 4;
pub const DEFAULT_RQ_ROTATION_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
pub const DEFAULT_RQ_ROTATION_ROUNDS: u32 = 4;
pub const RQ_ROTATION_BLOCK_SIZE: usize = 64;
pub const RQ_SCAN_BLOCK_SIZE: usize = 32;

const QUANTIZATION_REFINEMENT_ROUNDS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RQCodeFactors {
    pub f_add: f32,
    pub f_rescale: f32,
    pub f_error: f32,
}

impl RQCodeFactors {
    pub fn zero() -> Self {
        Self {
            f_add: 0.0,
            f_rescale: 0.0,
            f_error: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RQVectorFactors {
    pub coarse: RQCodeFactors,
    pub full: RQCodeFactors,
}

impl RQVectorFactors {
    pub fn zero() -> Self {
        Self {
            coarse: RQCodeFactors::zero(),
            full: RQCodeFactors::zero(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RQRotation {
    d: usize,
    padded_d: usize,
    seed: u64,
    rounds: u32,
    sign_masks: Vec<Vec<u8>>,
    permutations: Vec<Vec<usize>>,
}

impl RQRotation {
    pub fn new(d: usize, seed: u64, rounds: u32) -> Self {
        let padded_d = padded_dimension(d);
        let mut rng = SplitMix64::new(seed ^ (d as u64).rotate_left(17));
        let mut sign_masks = Vec::with_capacity(rounds as usize);
        let mut permutations = Vec::with_capacity(rounds as usize);

        for _ in 0..rounds {
            let mut signs = vec![0u8; padded_d.div_ceil(8)];
            for value in &mut signs {
                *value = rng.next_u64() as u8;
            }
            sign_masks.push(signs);

            let mut permutation: Vec<usize> = (0..padded_d).collect();
            for i in (1..padded_d).rev() {
                let j = rng.next_usize(i + 1);
                permutation.swap(i, j);
            }
            permutations.push(permutation);
        }

        Self {
            d,
            padded_d,
            seed,
            rounds,
            sign_masks,
            permutations,
        }
    }

    pub fn dimension(&self) -> usize {
        self.d
    }

    pub fn padded_dimension(&self) -> usize {
        self.padded_d
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn rounds(&self) -> u32 {
        self.rounds
    }

    pub fn rotate(&self, input: &[f32], output: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(input.len(), self.d);
        debug_assert_eq!(output.len(), self.padded_d);
        debug_assert_eq!(scratch.len(), self.padded_d);

        output.fill(0.0);
        output[..self.d].copy_from_slice(input);
        self.apply_in_place(output, scratch);
    }

    pub fn apply_in_place(&self, values: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(values.len(), self.padded_d);
        debug_assert_eq!(scratch.len(), self.padded_d);

        for (signs, permutation) in self.sign_masks.iter().zip(&self.permutations) {
            for (dim, value) in values.iter_mut().enumerate() {
                if signs[dim / 8] & (1u8 << (dim % 8)) != 0 {
                    *value = -*value;
                }
            }
            for block in values.chunks_exact_mut(RQ_ROTATION_BLOCK_SIZE) {
                hadamard_64(block);
            }
            for (source, &destination) in permutation.iter().enumerate() {
                scratch[destination] = values[source];
            }
            values.copy_from_slice(scratch);
        }
    }
}

#[derive(Debug, Clone)]
pub struct RaBitQuantizer {
    d: usize,
    padded_d: usize,
    bits: usize,
    plane_size: usize,
}

#[derive(Debug)]
pub struct RQEncodeScratch {
    levels: Vec<u8>,
    centered: Vec<f32>,
    coarse_centered: Vec<f32>,
}

impl RQEncodeScratch {
    pub fn new(padded_d: usize) -> Self {
        Self {
            levels: vec![0; padded_d],
            centered: vec![0.0; padded_d],
            coarse_centered: vec![0.0; padded_d],
        }
    }
}

#[derive(Debug, Clone)]
pub struct RQQueryContext {
    rotated_query: Vec<f32>,
    sum: f32,
    byte_subset_sums: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct RQQueryTerms {
    pub g_add: f32,
    pub g_error: f32,
}

impl RaBitQuantizer {
    pub fn new(d: usize, bits: usize) -> Self {
        assert!(is_supported_rq_bits(bits), "RQ bits must be in 1..=8");
        let padded_d = padded_dimension(d);
        Self {
            d,
            padded_d,
            bits,
            plane_size: padded_d / 8,
        }
    }

    pub fn dimension(&self) -> usize {
        self.d
    }

    pub fn padded_dimension(&self) -> usize {
        self.padded_d
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn plane_size(&self) -> usize {
        self.plane_size
    }

    pub fn code_size(&self) -> usize {
        self.plane_size * self.bits
    }

    pub fn factor_fields(&self) -> usize {
        if self.bits == 1 {
            2
        } else {
            5
        }
    }

    pub fn encode(
        &self,
        rotated_residual: &[f32],
        rotated_centroid: &[f32],
        metric: MetricType,
        code: &mut [u8],
    ) -> RQVectorFactors {
        let mut scratch = RQEncodeScratch::new(self.padded_d);
        self.encode_with_scratch(
            rotated_residual,
            rotated_centroid,
            metric,
            code,
            &mut scratch,
        )
    }

    pub fn encode_with_scratch(
        &self,
        rotated_residual: &[f32],
        rotated_centroid: &[f32],
        metric: MetricType,
        code: &mut [u8],
        scratch: &mut RQEncodeScratch,
    ) -> RQVectorFactors {
        debug_assert_eq!(rotated_residual.len(), self.padded_d);
        debug_assert_eq!(rotated_centroid.len(), self.padded_d);
        debug_assert!(code.len() >= self.code_size());
        debug_assert_eq!(scratch.levels.len(), self.padded_d);
        code[..self.code_size()].fill(0);

        quantize_centered_levels(
            rotated_residual,
            self.bits,
            &mut scratch.levels,
            &mut scratch.centered,
        );
        for (dim, &level) in scratch.levels.iter().enumerate() {
            for stored_plane in 0..self.bits {
                let source_bit = self.bits - 1 - stored_plane;
                if level & (1u8 << source_bit) != 0 {
                    code[stored_plane * self.plane_size + dim / 8] |= 1u8 << (dim % 8);
                }
            }
            scratch.coarse_centered[dim] = if level & (1u8 << (self.bits - 1)) != 0 {
                0.5
            } else {
                -0.5
            };
        }

        let coarse = compute_factors(
            rotated_residual,
            rotated_centroid,
            &scratch.coarse_centered,
            metric,
        );
        let full = if self.bits == 1 {
            coarse
        } else {
            compute_factors(
                rotated_residual,
                rotated_centroid,
                &scratch.centered,
                metric,
            )
        };
        RQVectorFactors { coarse, full }
    }

    pub fn prepare_query(&self, rotated_query: Vec<f32>) -> RQQueryContext {
        debug_assert_eq!(rotated_query.len(), self.padded_d);
        let sum = rotated_query.iter().sum();
        let mut byte_subset_sums = vec![0.0f32; self.plane_size * 256];
        for byte_idx in 0..self.plane_size {
            let dim_base = byte_idx * 8;
            let lut = &mut byte_subset_sums[byte_idx * 256..(byte_idx + 1) * 256];
            for pattern in 1..256usize {
                let bit = pattern.trailing_zeros() as usize;
                let previous = pattern & (pattern - 1);
                lut[pattern] = lut[previous] + rotated_query[dim_base + bit];
            }
        }
        RQQueryContext {
            rotated_query,
            sum,
            byte_subset_sums,
        }
    }

    pub fn query_terms(
        &self,
        context: &RQQueryContext,
        rotated_centroid: &[f32],
        metric: MetricType,
    ) -> RQQueryTerms {
        debug_assert_eq!(rotated_centroid.len(), self.padded_d);
        let mut residual_norm_sqr = 0.0f32;
        let mut query_centroid_ip = 0.0f32;
        for (&query, &centroid) in context.rotated_query.iter().zip(rotated_centroid) {
            let residual = query - centroid;
            residual_norm_sqr += residual * residual;
            query_centroid_ip += query * centroid;
        }
        match metric {
            MetricType::L2 => RQQueryTerms {
                g_add: residual_norm_sqr,
                g_error: residual_norm_sqr.sqrt(),
            },
            MetricType::Cosine => RQQueryTerms {
                g_add: 0.5 * residual_norm_sqr,
                g_error: residual_norm_sqr.sqrt(),
            },
            MetricType::InnerProduct => RQQueryTerms {
                g_add: -query_centroid_ip,
                g_error: residual_norm_sqr.sqrt(),
            },
        }
    }

    pub fn unsigned_plane_inner_product(&self, context: &RQQueryContext, plane_code: &[u8]) -> f32 {
        debug_assert!(plane_code.len() >= self.plane_size);
        let mut result = 0.0f32;
        for byte_idx in 0..self.plane_size {
            result += context.byte_subset_sums[byte_idx * 256 + plane_code[byte_idx] as usize];
        }
        result
    }

    pub(crate) fn byte_subset_sum(
        &self,
        context: &RQQueryContext,
        byte_idx: usize,
        pattern: u8,
    ) -> f32 {
        context.byte_subset_sums[byte_idx * 256 + pattern as usize]
    }

    pub(crate) fn query_sum(&self, context: &RQQueryContext) -> f32 {
        context.sum
    }

    pub fn coarse_inner_product(&self, context: &RQQueryContext, code: &[u8]) -> f32 {
        self.unsigned_plane_inner_product(context, code) - 0.5 * context.sum
    }

    pub fn full_inner_product(&self, context: &RQQueryContext, code: &[u8]) -> f32 {
        debug_assert!(code.len() >= self.code_size());
        let mut unsigned = 0.0f32;
        for stored_plane in 0..self.bits {
            let weight = (1usize << (self.bits - 1 - stored_plane)) as f32;
            let start = stored_plane * self.plane_size;
            unsigned += weight
                * self.unsigned_plane_inner_product(context, &code[start..start + self.plane_size]);
        }
        let center = ((1usize << self.bits) - 1) as f32 * 0.5;
        unsigned - center * context.sum
    }

    pub fn estimate(
        &self,
        inner_product: f32,
        factors: RQCodeFactors,
        query_terms: RQQueryTerms,
    ) -> f32 {
        factors.f_add + query_terms.g_add + factors.f_rescale * inner_product
    }

    pub fn lower_bound(
        &self,
        estimate: f32,
        factors: RQCodeFactors,
        query_terms: RQQueryTerms,
    ) -> f32 {
        estimate - factors.f_error * query_terms.g_error
    }
}

pub fn padded_dimension(d: usize) -> usize {
    d.max(1).div_ceil(RQ_ROTATION_BLOCK_SIZE) * RQ_ROTATION_BLOCK_SIZE
}

#[inline]
pub fn is_supported_rq_bits(bits: usize) -> bool {
    (1..=8).contains(&bits)
}

fn quantize_centered_levels(
    residual: &[f32],
    bits: usize,
    levels: &mut [u8],
    centered: &mut [f32],
) {
    let max_code = (1usize << bits) - 1;
    let center = max_code as f32 * 0.5;
    let max_abs = residual
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    if max_abs <= f32::EPSILON {
        levels.fill(0);
        centered.fill(-center);
        return;
    }

    let mut scale = 2.0 * max_abs / max_code as f32;
    for _ in 0..QUANTIZATION_REFINEMENT_ROUNDS {
        let mut dot = 0.0f32;
        let mut norm_sqr = 0.0f32;
        for (dim, &value) in residual.iter().enumerate() {
            let level = (value / scale + center).round().clamp(0.0, max_code as f32) as u8;
            let centered_level = level as f32 - center;
            levels[dim] = level;
            centered[dim] = centered_level;
            dot += value * centered_level;
            norm_sqr += centered_level * centered_level;
        }
        if dot <= f32::EPSILON || norm_sqr <= f32::EPSILON {
            break;
        }
        scale = dot / norm_sqr;
    }
}

fn compute_factors(
    residual: &[f32],
    centroid: &[f32],
    centered_code: &[f32],
    metric: MetricType,
) -> RQCodeFactors {
    let mut residual_norm_sqr = 0.0f32;
    let mut residual_code_ip = 0.0f32;
    let mut centroid_code_ip = 0.0f32;
    let mut residual_centroid_ip = 0.0f32;
    for ((&value, &center), &code) in residual.iter().zip(centroid).zip(centered_code) {
        residual_norm_sqr += value * value;
        residual_code_ip += value * code;
        centroid_code_ip += center * code;
        residual_centroid_ip += value * center;
    }
    if residual_norm_sqr <= f32::EPSILON || residual_code_ip.abs() <= f32::EPSILON {
        return match metric {
            MetricType::L2 => RQCodeFactors {
                f_add: residual_norm_sqr,
                f_rescale: 0.0,
                f_error: 2.0 * residual_norm_sqr.sqrt(),
            },
            MetricType::Cosine => RQCodeFactors {
                f_add: 0.5 * residual_norm_sqr,
                f_rescale: 0.0,
                f_error: residual_norm_sqr.sqrt(),
            },
            MetricType::InnerProduct => RQCodeFactors {
                f_add: -residual_centroid_ip,
                f_rescale: 0.0,
                f_error: residual_norm_sqr.sqrt(),
            },
        };
    }

    let rescale = residual_norm_sqr / residual_code_ip;
    let mut reconstruction_error_sqr = 0.0f32;
    for (&value, &code) in residual.iter().zip(centered_code) {
        let error = value - rescale * code;
        reconstruction_error_sqr += error * error;
    }
    let reconstruction_error = reconstruction_error_sqr.sqrt();

    match metric {
        MetricType::L2 => RQCodeFactors {
            f_add: residual_norm_sqr + 2.0 * rescale * centroid_code_ip,
            f_rescale: -2.0 * rescale,
            f_error: 2.0 * reconstruction_error,
        },
        MetricType::Cosine => RQCodeFactors {
            f_add: 0.5 * residual_norm_sqr + rescale * centroid_code_ip,
            f_rescale: -rescale,
            f_error: reconstruction_error,
        },
        MetricType::InnerProduct => RQCodeFactors {
            f_add: -residual_centroid_ip + rescale * centroid_code_ip,
            f_rescale: -rescale,
            f_error: reconstruction_error,
        },
    }
}

fn hadamard_64(values: &mut [f32]) {
    debug_assert_eq!(values.len(), RQ_ROTATION_BLOCK_SIZE);
    let mut width = 1;
    while width < RQ_ROTATION_BLOCK_SIZE {
        for base in (0..RQ_ROTATION_BLOCK_SIZE).step_by(width * 2) {
            for offset in 0..width {
                let left = values[base + offset];
                let right = values[base + offset + width];
                values[base + offset] = left + right;
                values[base + offset + width] = left - right;
            }
        }
        width *= 2;
    }
    for value in values {
        *value *= 0.125;
    }
}

#[derive(Debug, Clone)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_deterministic_and_preserves_norm_with_padding() {
        let d = 70;
        let rotation = RQRotation::new(d, 17, DEFAULT_RQ_ROTATION_ROUNDS);
        let input: Vec<f32> = (0..d).map(|i| i as f32 * 0.25 - 3.0).collect();
        let mut first = vec![0.0; rotation.padded_dimension()];
        let mut second = vec![0.0; rotation.padded_dimension()];
        let mut scratch = vec![0.0; rotation.padded_dimension()];
        rotation.rotate(&input, &mut first, &mut scratch);
        rotation.rotate(&input, &mut second, &mut scratch);

        let before: f32 = input.iter().map(|value| value * value).sum();
        let after: f32 = first.iter().map(|value| value * value).sum();
        assert_eq!(first, second);
        assert!((before - after).abs() <= before * 1e-5);
    }

    #[test]
    fn rotation_spreads_a_basis_vector_across_fht_blocks() {
        let d = 128;
        let rotation = RQRotation::new(d, 91, 2);
        let mut input = vec![0.0; d];
        input[7] = 1.0;
        let mut output = vec![0.0; rotation.padded_dimension()];
        let mut scratch = vec![0.0; rotation.padded_dimension()];
        rotation.rotate(&input, &mut output, &mut scratch);

        let non_zero = output.iter().filter(|value| value.abs() > 1e-7).count();
        assert!(
            non_zero > RQ_ROTATION_BLOCK_SIZE,
            "two rounds must spread beyond one 64-dimension block, got {non_zero}"
        );
    }

    #[test]
    fn four_bit_estimator_is_exact_for_the_encoded_vector() {
        let d = 64;
        let quantizer = RaBitQuantizer::new(d, 4);
        let residual: Vec<f32> = (0..d)
            .map(|i| ((i * 17 % 31) as f32 - 15.0) * 0.13)
            .collect();
        let centroid: Vec<f32> = (0..d).map(|i| (i % 7) as f32 * 0.02).collect();
        let query: Vec<f32> = residual
            .iter()
            .zip(&centroid)
            .map(|(&residual, &centroid)| residual + centroid)
            .collect();
        let mut code = vec![0; quantizer.code_size()];
        let factors = quantizer.encode(&residual, &centroid, MetricType::L2, &mut code);
        let context = quantizer.prepare_query(query.clone());
        let terms = quantizer.query_terms(&context, &centroid, MetricType::L2);
        let estimate = quantizer.estimate(
            quantizer.full_inner_product(&context, &code),
            factors.full,
            terms,
        );

        assert!(estimate.abs() < 1e-4, "self distance was {estimate}");
    }

    #[test]
    fn more_stored_bits_reduce_reconstruction_error() {
        let d = 64;
        let residual: Vec<f32> = (0..d)
            .map(|i| ((i * 29 % 53) as f32 - 26.0) * 0.11)
            .collect();
        let centroid = vec![0.0; d];
        let mut errors = Vec::new();
        for bits in [1, 2, 4, 8] {
            let quantizer = RaBitQuantizer::new(d, bits);
            let mut code = vec![0; quantizer.code_size()];
            let factors = quantizer.encode(&residual, &centroid, MetricType::L2, &mut code);
            errors.push(factors.full.f_error);
        }

        assert!(errors.windows(2).all(|pair| pair[1] <= pair[0] + 1e-5));
        assert!(errors[3] < errors[0] * 0.1);
    }

    #[test]
    fn lower_bound_contains_exact_distance() {
        let d = 64;
        let quantizer = RaBitQuantizer::new(d, 4);
        let residual: Vec<f32> = (0..d)
            .map(|i| ((i * 11 % 37) as f32 - 18.0) * 0.09)
            .collect();
        let query: Vec<f32> = (0..d)
            .map(|i| ((i * 7 % 41) as f32 - 20.0) * 0.08)
            .collect();
        let centroid = vec![0.0; d];
        let mut code = vec![0; quantizer.code_size()];
        let factors = quantizer.encode(&residual, &centroid, MetricType::L2, &mut code);
        let context = quantizer.prepare_query(query.clone());
        let terms = quantizer.query_terms(&context, &centroid, MetricType::L2);
        let coarse = quantizer.estimate(
            quantizer.coarse_inner_product(&context, &code),
            factors.coarse,
            terms,
        );
        let lower = quantizer.lower_bound(coarse, factors.coarse, terms);
        let exact: f32 = residual
            .iter()
            .zip(query)
            .map(|(&left, right)| {
                let delta = left - right;
                delta * delta
            })
            .sum();

        assert!(lower <= exact + 1e-4, "lower={lower}, exact={exact}");
    }
}
