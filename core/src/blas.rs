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

//! Matrix multiplication via `matrixmultiply` crate.
//! Pure Rust with SIMD + cache blocking, no system BLAS dependency.

/// C = alpha * A * B^T + beta * C
/// A: [m × k] row-major
/// B: [n × k] row-major (transposed in the multiply)
/// C: [m × n] row-major
pub fn sgemm_a_bt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    assert!(
        k <= isize::MAX as usize && n <= isize::MAX as usize,
        "SGEMM dimensions exceed isize"
    );
    let a_len = m.checked_mul(k).expect("a shape overflows usize");
    let b_len = n.checked_mul(k).expect("b shape overflows usize");
    let c_len = m.checked_mul(n).expect("c shape overflows usize");
    assert!(a.len() >= a_len, "a is shorter than m * k");
    assert!(b.len() >= b_len, "b is shorter than n * k");
    assert!(c.len() >= c_len, "c is shorter than m * n");

    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            k as isize, // row stride of A
            1,          // col stride of A
            b.as_ptr(),
            1,          // B^T: col stride = 1 means we read B row-wise as columns
            k as isize, // B^T: row stride = k means next "column of B^T" = next row of B
            beta,
            c.as_mut_ptr(),
            n as isize, // row stride of C
            1,          // col stride of C
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm_a_bt() {
        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // A * B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        //         = [[17, 23], [39, 53]]
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        sgemm_a_bt(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);

        assert!((c[0] - 17.0).abs() < 1e-5);
        assert!((c[1] - 23.0).abs() < 1e-5);
        assert!((c[2] - 39.0).abs() < 1e-5);
        assert!((c[3] - 53.0).abs() < 1e-5);
    }

    #[test]
    fn test_sgemm_rectangular() {
        // A = [[1, 2, 3]]   (1x3)
        // B = [[4, 5, 6], [7, 8, 9]]  (2x3)
        // A * B^T = [[1*4+2*5+3*6, 1*7+2*8+3*9]] = [[32, 50]]
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = [0.0f32; 2];

        sgemm_a_bt(1, 2, 3, 1.0, &a, &b, 0.0, &mut c);

        assert!((c[0] - 32.0).abs() < 1e-5);
        assert!((c[1] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_sgemm_accepts_larger_backing_slices() {
        let a = [1.0f32, 2.0, 99.0];
        let b = [3.0f32, 4.0, 99.0];
        let mut c = [0.0f32, 99.0];

        sgemm_a_bt(1, 1, 2, 1.0, &a, &b, 0.0, &mut c);

        assert_eq!(c, [11.0, 99.0]);
    }

    #[test]
    fn test_sgemm_rejects_short_slices() {
        let short_a = std::panic::catch_unwind(|| {
            let mut c = [0.0f32];
            sgemm_a_bt(1, 1, 2, 1.0, &[1.0], &[2.0, 3.0], 0.0, &mut c);
        });
        let short_b = std::panic::catch_unwind(|| {
            let mut c = [0.0f32];
            sgemm_a_bt(1, 1, 2, 1.0, &[1.0, 2.0], &[3.0], 0.0, &mut c);
        });
        let short_c = std::panic::catch_unwind(|| {
            sgemm_a_bt(1, 1, 2, 1.0, &[1.0, 2.0], &[3.0, 4.0], 0.0, &mut []);
        });

        assert!(short_a.is_err());
        assert!(short_b.is_err());
        assert!(short_c.is_err());
    }

    #[test]
    #[should_panic(expected = "SGEMM dimensions exceed isize")]
    fn test_sgemm_rejects_stride_overflow() {
        sgemm_a_bt(0, 0, usize::MAX, 1.0, &[], &[], 0.0, &mut []);
    }
}
