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
        for vector in values.chunks_exact(self.d) {
            for i in 0..self.d {
                self.mins[i] = self.mins[i].min(vector[i]);
                self.maxs[i] = self.maxs[i].max(vector[i]);
            }
        }
        self.refresh_global_bounds();
    }

    pub fn code_size(&self) -> usize {
        self.d
    }

    pub fn encode_batch(&self, data: &[f32], n: usize, codes: &mut [u8]) {
        let len = n * self.d;
        assert!(data.len() >= len);
        assert!(codes.len() >= len);

        for row in 0..n {
            let base = row * self.d;
            for dim in 0..self.d {
                let min = self.mins[dim];
                let max = self.maxs[dim];
                let out = base + dim;
                codes[out] = if min >= max {
                    0
                } else {
                    let scaled = ((data[out] - min) * 255.0 / (max - min)).clamp(0.0, 255.0);
                    scaled.round() as u8
                };
            }
        }
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
    fn test_scalar_quantizer_distance_to_code() {
        let sq = ScalarQuantizer::with_bounds(2, 0.0, 1.0);
        let mut code = vec![0u8; 2];
        sq.encode(&[1.0, 0.0], &mut code);

        let dist = sq.distance_to_code(&[1.0, 0.0], &code, MetricType::L2);

        assert!(dist < 1e-6);
    }
}
