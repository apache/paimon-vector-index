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

//! FastScan: Faiss-style block layout (bbs=32) + vpshufb 32-way parallel lookup
//! for 4-bit PQ codes.
//!
//! Block layout: 32 vectors per block, codes interleaved by sub-quantizer pair.
//! Each block stores M/2 groups of 32 bytes (one byte per vector per sub-quant pair).
//!
//! Layout: [block0_pair0(32B), block0_pair1(32B), ..., block0_pairM/2(32B),
//!          block1_pair0(32B), ...]

/// Block size: 32 vectors per block (matches AVX2 register width).
pub const BBS: usize = 32;

/// Pack 4-bit codes from row-major [n][cs] into block layout.
/// Output layout: [num_blocks][cs][BBS] where cs = M/2.
/// Pads the last block with zeros if n is not a multiple of BBS.
pub fn pack_codes_block_layout(codes: &[u8], n: usize, cs: usize) -> Vec<u8> {
    let num_blocks = n.div_ceil(BBS);
    let block_size = cs * BBS; // bytes per block
    let mut packed = vec![0u8; num_blocks * block_size];

    for block in 0..num_blocks {
        let block_start = block * BBS;
        for pair in 0..cs {
            for vec_in_block in 0..BBS {
                let global_vec = block_start + vec_in_block;
                if global_vec < n {
                    packed[block * block_size + pair * BBS + vec_in_block] =
                        codes[global_vec * cs + pair];
                }
                // else: remains 0 (padding)
            }
        }
    }

    packed
}

/// Unpack block layout back to row-major [n][cs] (for compatibility).
pub fn unpack_codes_block_layout(packed: &[u8], n: usize, cs: usize) -> Vec<u8> {
    let num_blocks = n.div_ceil(BBS);
    let block_size = cs * BBS;
    let mut codes = vec![0u8; n * cs];

    for block in 0..num_blocks {
        let block_start = block * BBS;
        for pair in 0..cs {
            for vec_in_block in 0..BBS {
                let global_vec = block_start + vec_in_block;
                if global_vec < n {
                    codes[global_vec * cs + pair] =
                        packed[block * block_size + pair * BBS + vec_in_block];
                }
            }
        }
    }

    codes
}

/// Quantize a f32 distance table [M * 16] to u8 [M * 16].
/// Returns (qmin, qmax_used, quantized_table).
pub fn quantize_distance_table(table: &[f32], qmax_hint: f32) -> (f32, f32, Vec<u8>) {
    let qmin = table.iter().cloned().fold(f32::INFINITY, f32::min);
    let qmax = qmax_hint.max(table.iter().cloned().fold(f32::MIN, f32::max));
    let range = (qmax - qmin).max(1e-10);
    let factor = 255.0 / range;

    let qtable: Vec<u8> = table
        .iter()
        .map(|&d| ((d - qmin) * factor).clamp(0.0, 255.0) as u8)
        .collect();

    (qmin, qmax, qtable)
}

/// FastScan: scan block-layout 4-bit codes using SIMD.
/// codes: block layout [num_blocks][cs][BBS]
/// sim_table: [M * 16] f32 distance table
/// Returns f32 distances for `n` vectors.
pub fn fastscan_4bit(sim_table: &[f32], codes: &[u8], n: usize, m: usize, dists: &mut [f32]) {
    let cs = m / 2;

    // Step 1: f32 exact for first min(200, n) vectors as qmax calibration
    const FLAT_NUM: usize = 200;
    let flat_end = n.min(FLAT_NUM);
    let block_size = cs * BBS;

    for i in 0..flat_end {
        let block = i / BBS;
        let vec_in_block = i % BBS;
        let mut d = 0.0f32;
        for pair in 0..cs {
            let byte = codes[block * block_size + pair * BBS + vec_in_block];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            d += sim_table[(pair * 2) * 16 + lo];
            d += sim_table[(pair * 2 + 1) * 16 + hi];
        }
        dists[i] = d;
    }

    if n <= FLAT_NUM {
        return;
    }

    // Step 2: Quantize distance table
    let qmax = dists[..flat_end].iter().cloned().fold(f32::MIN, f32::max);
    let (qmin, _, qtable) = quantize_distance_table(sim_table, qmax);
    let range = (qmax - qmin).max(1e-10);

    // Step 3: Scan blocks using SIMD
    let num_blocks = n.div_ceil(BBS);
    let start_block = flat_end.div_ceil(BBS);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                fastscan_blocks_avx2(
                    &qtable,
                    codes,
                    cs,
                    start_block,
                    num_blocks,
                    block_size,
                    qmin,
                    range,
                    m,
                    dists,
                );
            }
            let partial_start = start_block * BBS;
            for i in flat_end..partial_start.min(n) {
                let block = i / BBS;
                let vec_in_block = i % BBS;
                let mut d = 0.0f32;
                for pair in 0..cs {
                    let byte = codes[block * block_size + pair * BBS + vec_in_block];
                    let lo = (byte & 0x0F) as usize;
                    let hi = ((byte >> 4) & 0x0F) as usize;
                    d += sim_table[(pair * 2) * 16 + lo];
                    d += sim_table[(pair * 2 + 1) * 16 + hi];
                }
                dists[i] = d;
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            fastscan_blocks_neon(
                &qtable,
                codes,
                cs,
                start_block,
                num_blocks,
                block_size,
                qmin,
                range,
                m,
                dists,
            );
        }
        let partial_start = start_block * BBS;
        for i in flat_end..partial_start.min(n) {
            let block = i / BBS;
            let vec_in_block = i % BBS;
            let mut d = 0.0f32;
            for pair in 0..cs {
                let byte = codes[block * block_size + pair * BBS + vec_in_block];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                d += sim_table[(pair * 2) * 16 + lo];
                d += sim_table[(pair * 2 + 1) * 16 + hi];
            }
            dists[i] = d;
        }
        return;
    }

    // Fallback: scalar scan on blocks (used when no SIMD available)
    #[allow(unreachable_code)]
    for block in start_block..num_blocks {
        let base_vec = block * BBS;
        let vecs_in_block = BBS.min(n - base_vec);

        let mut q_dists = [0u16; BBS];

        for pair in 0..cs {
            let qtab_lo = &qtable[(pair * 2) * 16..(pair * 2 + 1) * 16];
            let qtab_hi = &qtable[(pair * 2 + 1) * 16..(pair * 2 + 2) * 16];
            let block_codes = &codes[block * block_size + pair * BBS..];

            for v in 0..vecs_in_block {
                let byte = block_codes[v];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                q_dists[v] += qtab_lo[lo] as u16 + qtab_hi[hi] as u16;
            }
        }

        // Dequantize
        let inv_factor = range / 255.0;
        let base_dist = qmin * m as f32;
        for v in 0..vecs_in_block {
            dists[base_vec + v] = q_dists[v] as f32 * inv_factor + base_dist;
        }
    }

    // Fill gap between flat_end and start_block*BBS with exact computation
    let partial_start = start_block * BBS;
    for i in flat_end..partial_start.min(n) {
        let block = i / BBS;
        let vec_in_block = i % BBS;
        let mut d = 0.0f32;
        for pair in 0..cs {
            let byte = codes[block * block_size + pair * BBS + vec_in_block];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            d += sim_table[(pair * 2) * 16 + lo];
            d += sim_table[(pair * 2 + 1) * 16 + hi];
        }
        dists[i] = d;
    }
}

/// AVX2 block scan using vpshufb for 32-way parallel 4-bit lookup.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fastscan_blocks_avx2(
    qtable: &[u8],
    codes: &[u8],
    cs: usize,
    start_block: usize,
    num_blocks: usize,
    block_size: usize,
    qmin: f32,
    range: f32,
    m: usize,
    dists: &mut [f32],
) {
    use std::arch::x86_64::*;

    let inv_factor = range / 255.0;
    let base_dist = qmin * m as f32;

    for block in start_block..num_blocks {
        let base_vec = block * BBS;

        // u16 accumulators: 2 × __m256i = 32 × u16 values
        let mut accu_lo = _mm256_setzero_si256(); // vecs 0-15
        let mut accu_hi = _mm256_setzero_si256(); // vecs 16-31

        for pair in 0..cs {
            // Load 32-byte quantized LUT for this pair (lo + hi sub-quantizers)
            // LUT layout: [16 bytes for sub_lo, 16 bytes for sub_hi]
            let lut_lo_ptr = qtable.as_ptr().add((pair * 2) * 16);
            let lut_hi_ptr = qtable.as_ptr().add((pair * 2 + 1) * 16);

            // Broadcast 16-byte LUTs into 256-bit registers (same 16-byte table in both lanes)
            let lut_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut_lo_ptr as *const __m128i));
            let lut_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(lut_hi_ptr as *const __m128i));

            // Load 32 code bytes for this pair in this block
            let code_ptr = codes.as_ptr().add(block * block_size + pair * BBS);
            let code_vec = _mm256_loadu_si256(code_ptr as *const __m256i);

            // Split nibbles
            let mask = _mm256_set1_epi8(0x0F);
            let lo_nibbles = _mm256_and_si256(code_vec, mask);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(code_vec, 4), mask);

            // vpshufb: 32-way parallel lookup
            let dist_lo = _mm256_shuffle_epi8(lut_lo, lo_nibbles);
            let dist_hi = _mm256_shuffle_epi8(lut_hi, hi_nibbles);

            // Sum lo + hi distances (u8 + u8 → u8, saturating)
            let pair_dist = _mm256_adds_epu8(dist_lo, dist_hi);

            // Widen to u16 and accumulate (prevent u8 overflow across pairs)
            // Split into low 16 bytes and high 16 bytes
            let zero = _mm256_setzero_si256();
            let pair_lo = _mm256_unpacklo_epi8(pair_dist, zero); // 16 × u16 from bytes 0-15
            let pair_hi = _mm256_unpackhi_epi8(pair_dist, zero); // 16 × u16 from bytes 16-31

            accu_lo = _mm256_add_epi16(accu_lo, pair_lo);
            accu_hi = _mm256_add_epi16(accu_hi, pair_hi);
        }

        // Extract u16 values and dequantize to f32
        let mut q_vals = [0u16; BBS];
        _mm256_storeu_si256(q_vals.as_mut_ptr() as *mut __m256i, accu_lo);
        _mm256_storeu_si256(q_vals.as_mut_ptr().add(16) as *mut __m256i, accu_hi);

        for v in 0..BBS {
            let idx = base_vec + v;
            if idx < dists.len() {
                dists[idx] = q_vals[v] as f32 * inv_factor + base_dist;
            }
        }
    }
}

/// ARM NEON block scan (16-way per instruction, 2 passes per block).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fastscan_blocks_neon(
    qtable: &[u8],
    codes: &[u8],
    cs: usize,
    start_block: usize,
    num_blocks: usize,
    block_size: usize,
    qmin: f32,
    range: f32,
    m: usize,
    dists: &mut [f32],
) {
    use std::arch::aarch64::*;

    let inv_factor = range / 255.0;
    let base_dist = qmin * m as f32;

    for block in start_block..num_blocks {
        let base_vec = block * BBS;

        // u16 accumulators: 4 × uint16x8_t = 32 × u16
        let mut accu = [vdupq_n_u16(0); 4];

        for pair in 0..cs {
            let lut_lo = vld1q_u8(qtable.as_ptr().add((pair * 2) * 16));
            let lut_hi = vld1q_u8(qtable.as_ptr().add((pair * 2 + 1) * 16));

            let code_ptr = codes.as_ptr().add(block * block_size + pair * BBS);

            // Process 16 vectors at a time (NEON = 128-bit)
            for half in 0..2 {
                let code_vec = vld1q_u8(code_ptr.add(half * 16));

                let mask = vdupq_n_u8(0x0F);
                let lo_nib = vandq_u8(code_vec, mask);
                let hi_nib = vshrq_n_u8(code_vec, 4);

                // tbl: 16-way lookup
                let dist_lo = vqtbl1q_u8(lut_lo, lo_nib);
                let dist_hi = vqtbl1q_u8(lut_hi, hi_nib);

                let pair_dist = vqaddq_u8(dist_lo, dist_hi);

                // Widen to u16
                let lo_u16 = vmovl_u8(vget_low_u8(pair_dist));
                let hi_u16 = vmovl_u8(vget_high_u8(pair_dist));

                accu[half * 2] = vaddq_u16(accu[half * 2], lo_u16);
                accu[half * 2 + 1] = vaddq_u16(accu[half * 2 + 1], hi_u16);
            }
        }

        // Extract and dequantize
        let mut q_vals = [0u16; BBS];
        for i in 0..4 {
            vst1q_u16(q_vals.as_mut_ptr().add(i * 8), accu[i]);
        }

        for v in 0..BBS {
            let idx = base_vec + v;
            if idx < dists.len() {
                dists[idx] = q_vals[v] as f32 * inv_factor + base_dist;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let n = 100;
        let cs = 4; // m/2 = 4, i.e. m=8
        let codes: Vec<u8> = (0..n * cs).map(|i| (i % 256) as u8).collect();

        let packed = pack_codes_block_layout(&codes, n, cs);
        let unpacked = unpack_codes_block_layout(&packed, n, cs);

        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_fastscan_correctness() {
        let m = 8;
        let cs = m / 2;
        let n = 100;

        // Random-ish codes
        let codes_row: Vec<u8> = (0..n * cs).map(|i| ((i * 7 + 3) % 256) as u8).collect();

        // Random-ish distance table [M * 16]
        let sim_table: Vec<f32> = (0..m * 16).map(|i| (i as f32) * 0.1 + 0.5).collect();

        // Compute ground truth with scalar
        let mut expected = vec![0.0f32; n];
        for i in 0..n {
            for pair in 0..cs {
                let byte = codes_row[i * cs + pair];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;
                expected[i] += sim_table[(pair * 2) * 16 + lo];
                expected[i] += sim_table[(pair * 2 + 1) * 16 + hi];
            }
        }

        // Pack into block layout and scan
        let packed = pack_codes_block_layout(&codes_row, n, cs);
        let mut result = vec![0.0f32; n];
        fastscan_4bit(&sim_table, &packed, n, m, &mut result);

        // Check first 200 (f32 exact) — should match perfectly
        for i in 0..n.min(200) {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_fastscan_large() {
        let m = 16;
        let cs = m / 2;
        let n = 1000; // > 200, exercises quantized path

        let codes_row: Vec<u8> = (0..n * cs).map(|i| ((i * 13 + 7) % 256) as u8).collect();
        let sim_table: Vec<f32> = (0..m * 16).map(|i| (i as f32) * 0.05 + 1.0).collect();

        let packed = pack_codes_block_layout(&codes_row, n, cs);
        let mut result = vec![0.0f32; n];
        fastscan_4bit(&sim_table, &packed, n, m, &mut result);

        // Verify all distances are positive and reasonable
        for i in 0..n {
            assert!(
                result[i] > 0.0,
                "Distance at {} is non-positive: {}",
                i,
                result[i]
            );
            assert!(
                result[i] < 10000.0,
                "Distance at {} seems too large: {}",
                i,
                result[i]
            );
        }
    }
}
