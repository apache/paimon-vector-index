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
use crate::distance::fvec_inner_product;
use crate::kmeans::KMeansConfig;
use crate::pq::ProductQuantizer;
use nalgebra::{DMatrix, SVD};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// OPQ (Optimized Product Quantization) rotation matrix.
/// Aligned with Faiss's OPQMatrix from VectorTransform.cpp.
///
/// Learns an orthogonal rotation R that minimizes PQ reconstruction error
/// via alternating Procrustes optimization.
pub struct OPQMatrix {
    pub d: usize,
    pub m: usize,
    pub niter: usize,
    pub niter_pq: usize,
    pub niter_pq_0: usize,
    pub max_train_points: usize,
    /// Rotation matrix [d * d], row-major. y = R * x.
    pub rotation: Vec<f32>,
    pub is_trained: bool,
}

impl OPQMatrix {
    pub fn new(d: usize, m: usize) -> Self {
        OPQMatrix {
            d,
            m,
            niter: 50,
            niter_pq: 4,
            niter_pq_0: 40,
            max_train_points: 65536,
            rotation: vec![0.0f32; d * d],
            is_trained: false,
        }
    }

    /// Train the OPQ rotation matrix.
    /// data: flat [n * d].
    pub fn train(&mut self, data: &[f32], n: usize, pq: &mut ProductQuantizer) {
        let d = self.d;
        let mut rng = StdRng::seed_from_u64(12345);

        // Subsample if needed
        let train_n = n.min(self.max_train_points);
        let mut train_data = if n > self.max_train_points {
            let mut sub = vec![0.0f32; train_n * d];
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..train_n {
                let j = rng.gen_range(i..n);
                indices.swap(i, j);
            }
            for (out_i, &src_i) in indices[..train_n].iter().enumerate() {
                sub[out_i * d..(out_i + 1) * d].copy_from_slice(&data[src_i * d..(src_i + 1) * d]);
            }
            sub
        } else {
            data[..n * d].to_vec()
        };

        // Center data (subtract mean) — aligned with Faiss OPQMatrix
        let mut mean = vec![0.0f32; d];
        for i in 0..train_n {
            for j in 0..d {
                mean[j] += train_data[i * d + j];
            }
        }
        let inv_n = 1.0 / train_n as f32;
        for j in 0..d {
            mean[j] *= inv_n;
        }
        for i in 0..train_n {
            for j in 0..d {
                train_data[i * d + j] -= mean[j];
            }
        }

        // Initialize with random orthogonal matrix via QR decomposition
        let random_mat: Vec<f32> = (0..d * d).map(|_| rng.gen::<f32>() - 0.5).collect();
        let mat = DMatrix::from_row_slice(d, d, &random_mat);
        let qr = mat.qr();
        let q = qr.q();
        for i in 0..d {
            for j in 0..d {
                self.rotation[i * d + j] = q[(i, j)];
            }
        }

        let mut projected = vec![0.0f32; train_n * d];
        let mut reconstructed = vec![0.0f32; train_n * d];
        let cs = pq.code_size();
        let mut codes = vec![0u8; train_n * cs];

        for iter in 0..self.niter {
            // 1. Project: projected = train_data * R^T
            self.apply_batch(&train_data, &mut projected, train_n);

            // 2. Train PQ on projected data (hot-start on iter >= 1)
            let pq_niter = if iter == 0 {
                self.niter_pq_0
            } else {
                self.niter_pq
            };
            let km_config = KMeansConfig {
                niter: pq_niter,
                ..KMeansConfig::default()
            };
            let hot_start = iter > 0;
            pq.train_hot_start(&projected, train_n, &km_config, hot_start);

            // 3. Encode and decode to get reconstructions
            pq.encode_batch(&projected, train_n, &mut codes);
            for i in 0..train_n {
                pq.decode(
                    &codes[i * cs..(i + 1) * cs],
                    &mut reconstructed[i * d..(i + 1) * d],
                );
            }

            // 4. Solve Procrustes: find R that minimizes ||X - Y*R^T||
            //    Solution: R = V * U^T where X^T * Y = U * S * V^T
            let x_mat = DMatrix::from_row_slice(train_n, d, &train_data);
            let y_mat = DMatrix::from_row_slice(train_n, d, &reconstructed);
            let cross_cov = x_mat.transpose() * &y_mat; // [d x d]

            let svd = SVD::new(cross_cov, true, true);
            if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
                // R = U * V^T
                let r = &u * &vt;
                for i in 0..d {
                    for j in 0..d {
                        self.rotation[i * d + j] = r[(i, j)];
                    }
                }
            }
        }

        // Final PQ training with the learned rotation
        self.apply_batch(&train_data, &mut projected, train_n);
        pq.train_with_config(&projected, train_n, &KMeansConfig::default());

        self.is_trained = true;
    }

    /// Apply rotation to a single vector: y = R * x.
    pub fn apply(&self, x: &[f32], y: &mut [f32]) {
        let d = self.d;
        for i in 0..d {
            y[i] = fvec_inner_product(&self.rotation[i * d..(i + 1) * d], x);
        }
    }

    /// Apply rotation to a batch of vectors.
    pub fn apply_batch(&self, data: &[f32], out: &mut [f32], n: usize) {
        let d = self.d;
        if n == 1 {
            self.apply(&data[..d], &mut out[..d]);
        } else if n > 1 {
            sgemm_a_bt(n, d, d, 1.0, data, &self.rotation, 0.0, out);
        }
    }

    /// Apply reverse rotation: x = R^T * y.
    pub fn apply_reverse(&self, y: &[f32], x: &mut [f32]) {
        let d = self.d;
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += self.rotation[j * d + i] * y[j];
            }
            x[i] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_orthogonality() {
        let d = 8;
        let m = 2;
        let n = 500;

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();

        let mut opq = OPQMatrix::new(d, m);
        opq.niter = 5; // Reduce for test speed
        let mut pq = ProductQuantizer::new(d, m);
        opq.train(&data, n, &mut pq);

        assert!(opq.is_trained);

        // Test that R * R^T ≈ I
        for i in 0..d {
            for j in 0..d {
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += opq.rotation[i * d + k] * opq.rotation[j * d + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "R*R^T[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_apply_reverse() {
        let d = 4;
        let mut opq = OPQMatrix::new(d, 2);
        // Set rotation to a simple permutation matrix
        opq.rotation = vec![
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];

        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 4];
        opq.apply(&x, &mut y);

        let mut x_back = [0.0f32; 4];
        opq.apply_reverse(&y, &mut x_back);

        for i in 0..d {
            assert!((x[i] - x_back[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_batch_matches_apply() {
        let d = 4;
        let mut opq = OPQMatrix::new(d, 2);
        opq.rotation = vec![
            0.5, 0.0, -0.5, 1.0, 1.0, 0.25, 0.0, -0.25, 0.0, 1.5, 0.5, 0.0, -1.0, 0.0, 0.75, 0.25,
        ];

        let n = 3;
        let data = vec![
            1.0, 2.0, 3.0, 4.0, -2.0, 0.5, 1.25, 3.5, 0.0, -1.0, 2.0, 0.75,
        ];
        let mut batch = vec![0.0f32; n * d];
        opq.apply_batch(&data, &mut batch, n);

        for i in 0..n {
            let mut single = vec![0.0f32; d];
            opq.apply(&data[i * d..(i + 1) * d], &mut single);
            for j in 0..d {
                assert!((batch[i * d + j] - single[j]).abs() < 1e-5);
            }
        }
    }
}
