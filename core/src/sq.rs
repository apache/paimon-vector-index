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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarQuantizer {
    pub d: usize,
    pub min: f32,
    pub max: f32,
}

impl ScalarQuantizer {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            min: 0.0,
            max: 0.0,
        }
    }

    pub fn with_bounds(d: usize, min: f32, max: f32) -> Self {
        Self { d, min, max }
    }

    pub fn train(&mut self, data: &[f32], n: usize) {
        let values = &data[..n * self.d];
        if values.is_empty() {
            self.min = 0.0;
            self.max = 0.0;
            return;
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &value in values {
            min = min.min(value);
            max = max.max(value);
        }
        self.min = min;
        self.max = max;
    }

    pub fn code_size(&self) -> usize {
        self.d
    }

    pub fn encode_batch(&self, data: &[f32], n: usize, codes: &mut [u8]) {
        let len = n * self.d;
        assert!(data.len() >= len);
        assert!(codes.len() >= len);

        if self.min >= self.max {
            codes[..len].fill(0);
            return;
        }

        let scale = 255.0 / (self.max - self.min);
        for i in 0..len {
            let scaled = ((data[i] - self.min) * scale).clamp(0.0, 255.0);
            codes[i] = scaled as u8;
        }
    }

    pub fn encode(&self, vector: &[f32], code: &mut [u8]) {
        self.encode_batch(vector, 1, code);
    }

    pub fn decode_batch(&self, codes: &[u8], n: usize, vectors: &mut [f32]) {
        let len = n * self.d;
        assert!(codes.len() >= len);
        assert!(vectors.len() >= len);

        if self.min >= self.max {
            vectors[..len].fill(self.min);
            return;
        }

        let scale = (self.max - self.min) / 255.0;
        for i in 0..len {
            vectors[i] = self.min + codes[i] as f32 * scale;
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
        debug_assert!(query.len() >= self.d);
        debug_assert!(code.len() >= self.d);

        match context.metric {
            MetricType::L2 => {
                let mut sum = 0.0f32;
                for i in 0..self.d {
                    let diff = query[i] - self.decode_value(code[i]);
                    sum += diff * diff;
                }
                sum
            }
            MetricType::InnerProduct => {
                let mut dot = 0.0f32;
                for i in 0..self.d {
                    dot += query[i] * self.decode_value(code[i]);
                }
                -dot
            }
            MetricType::Cosine => {
                let mut dot = 0.0f32;
                let mut vector_norm = 0.0f32;
                for i in 0..self.d {
                    let value = self.decode_value(code[i]);
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

    fn decode_value(&self, code: u8) -> f32 {
        if self.min >= self.max {
            self.min
        } else {
            self.min + code as f32 * (self.max - self.min) / 255.0
        }
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
    fn test_scalar_quantizer_distance_to_code() {
        let sq = ScalarQuantizer::with_bounds(2, 0.0, 1.0);
        let mut code = vec![0u8; 2];
        sq.encode(&[1.0, 0.0], &mut code);

        let dist = sq.distance_to_code(&[1.0, 0.0], &code, MetricType::L2);

        assert!(dist < 1e-6);
    }
}
