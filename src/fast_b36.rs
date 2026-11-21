//! Division-free scalar and 4-wide AVX2 base36 encoders for 40-bit IDs (`u...`, `e...`, `k...`).

use crate::unscramble::B36_PAIR_LUT;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// `ceil(2^64 / 1296)` for exact 40-bit division by `1296` via 128-bit multiply-high.
/// Error `e = RECIP_1296_S64 * 1296 - 2^64 = 272`, so `x * e < 2^64` holds for all `x < 36^8` (including all 40-bit values).
const RECIP_1296_S64: u128 = 0x0032_9161_f9ad_d3c1;

/// `ceil(2^39 / 1296)` for exact division of `q1 < 2^30` by `1296` in 64-bit / AVX2 `mul_epu32`.
const RECIP_1296_S39: u64 = 424_194_301;

/// `ceil(2^30 / 1296)` for exact division of `q2 < 654,621` by `1296` in 64-bit / AVX2 `mul_epu32`.
const RECIP_1296_S30: u64 = 828_505;

/// `ceil(2^40 / 81)` for exact division of `y = (val >> 4) < 2^36` by `81` (`val / 1296 = (val >> 4) / 81`).
#[cfg(target_arch = "x86_64")]
const RECIP_81_S40_HI: u64 = 3;
#[cfg(target_arch = "x86_64")]
const RECIP_81_S40_LO: u64 = 689_315_739;

/// Branchlessly strips leading ASCII `'0'` characters from an 8-byte base36 word and writes
/// `[prefix, digits...]` into `dst[..9]`, returning total byte length (`2..=9`).
#[inline(always)]
unsafe fn write_b36_from_pairs(
    prefix: u8,
    p0: usize,
    p1: usize,
    p2: usize,
    p3: usize,
    dst: *mut u8,
) -> usize {
    let lut = B36_PAIR_LUT.0.as_ptr() as *const u16;
    let d0 = *lut.add(p0) as u64;
    let d1 = *lut.add(p1) as u64;
    let d2 = *lut.add(p2) as u64;
    let d3 = *lut.add(p3) as u64;
    let w = d0 | (d1 << 16) | (d2 << 32) | (d3 << 48);
    // Keep byte 7 even when all digits are '0' (val == 0 -> "<prefix>0").
    let skip_bits = ((w ^ 0x3030_3030_3030_3030u64) | 0x0100_0000_0000_0000u64).trailing_zeros();
    let skip_bytes = (skip_bits >> 3) as usize;
    *dst = prefix;
    (dst.add(1) as *mut u64).write_unaligned((w >> (skip_bytes * 8)).to_le());
    9 - skip_bytes
}

/// Division-free scalar base36 encoder for 40-bit values (`val < 2^40`, or `< 36^8`).
///
/// Formats `prefix` followed by lowercase base36 digits of `val` into `dst[..9]`
/// and returns the total byte length (`2..=9`), with zero hardware `div` instructions.
#[inline(always)]
pub fn encode_b36_nodiv(prefix: u8, val: u64, dst: &mut [u8; 9]) -> usize {
    debug_assert!(val < 2_821_109_907_456);
    let q1 = ((val as u128 * RECIP_1296_S64) >> 64) as u64;
    let p3 = (val - q1 * 1296) as usize;
    let q2 = (q1.wrapping_mul(RECIP_1296_S39)) >> 39;
    let p2 = (q1 - q2 * 1296) as usize;
    let q3 = (q2.wrapping_mul(RECIP_1296_S30)) >> 30;
    let p1 = (q2 - q3 * 1296) as usize;
    let p0 = q3 as usize;
    unsafe { write_b36_from_pairs(prefix, p0, p1, p2, p3, dst.as_mut_ptr()) }
}

/// Encodes 4 canonical `u...` user IDs simultaneously into `(out_data, out_offsets)`
/// using 4-wide AVX2 reciprocal multiply-high math (no hardware division).
///
/// `out_offsets` stores cumulative byte offsets starting at `0` (pushed once per encoded ID).
#[inline(always)]
pub fn encode_b36_x4(vals: [u64; 4], out_data: &mut Vec<u8>, out_offsets: &mut Vec<u32>) {
    if out_offsets.is_empty() {
        out_offsets.push(0);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        encode_b36_x4_avx2(vals, out_data, out_offsets);
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        out_data.reserve(36);
        out_offsets.reserve(4);
        let mut cur = out_data.len();
        for &v in &vals {
            let mut tmp = [0u8; 9];
            let len = encode_b36_nodiv(b'u', v, &mut tmp);
            unsafe {
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_data.as_mut_ptr().add(cur), len);
            }
            cur += len;
            out_offsets.push(cur as u32);
        }
        unsafe {
            out_data.set_len(cur);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
#[inline]
unsafe fn encode_b36_x4_avx2(vals: [u64; 4], out_data: &mut Vec<u8>, out_offsets: &mut Vec<u32>) {
    let v = _mm256_loadu_si256(vals.as_ptr() as *const __m256i);
    let c1296 = _mm256_set1_epi64x(1296);

    // q1 = val / 1296 = (val >> 4) / 81 for val < 2^40 (y = val >> 4 < 2^36).
    let y = _mm256_srli_epi64(v, 4);
    let y_hi = _mm256_srli_epi64(y, 32);
    let r1_lo = _mm256_set1_epi64x(RECIP_81_S40_LO as i64);
    let r1_hi = _mm256_set1_epi64x(RECIP_81_S40_HI as i64);

    let lo_lo = _mm256_mul_epu32(y, r1_lo);
    let hi_lo = _mm256_mul_epu32(y_hi, r1_lo);
    let lo_hi = _mm256_mul_epu32(y, r1_hi);
    let hi_hi = _mm256_mul_epu32(y_hi, r1_hi);

    let mid = _mm256_add_epi64(
        _mm256_srli_epi64(lo_lo, 32),
        _mm256_add_epi64(hi_lo, lo_hi),
    );
    let q1 = _mm256_add_epi64(_mm256_slli_epi64(hi_hi, 24), _mm256_srli_epi64(mid, 8));
    let p3 = _mm256_sub_epi64(v, _mm256_mul_epu32(q1, c1296));

    // q2 = q1 / 1296 (q1 < 2^30)
    let q2 = _mm256_srli_epi64(_mm256_mul_epu32(q1, _mm256_set1_epi64x(RECIP_1296_S39 as i64)), 39);
    let p2 = _mm256_sub_epi64(q1, _mm256_mul_epu32(q2, c1296));

    // q3 = q2 / 1296 (q2 < 654,621)
    let q3 = _mm256_srli_epi64(_mm256_mul_epu32(q2, _mm256_set1_epi64x(RECIP_1296_S30 as i64)), 30);
    let p1 = _mm256_sub_epi64(q2, _mm256_mul_epu32(q3, c1296));
    let p0 = q3;

    // Pack the 4 pairs [p0, p1, p2, p3] into one u64 per lane (4 x u16) and extract in registers
    // to avoid a 256-bit-store -> 16-bit-load store-to-load forwarding stall.
    let p01 = _mm256_or_si256(p0, _mm256_slli_epi64(p1, 16));
    let p23 = _mm256_or_si256(p2, _mm256_slli_epi64(p3, 16));
    let packed = _mm256_or_si256(p01, _mm256_slli_epi64(p23, 32));

    let lo128 = _mm256_castsi256_si128(packed);
    let hi128 = _mm256_extracti128_si256::<1>(packed);
    let lanes = [
        _mm_cvtsi128_si64(lo128) as u64,
        _mm_extract_epi64::<1>(lo128) as u64,
        _mm_cvtsi128_si64(hi128) as u64,
        _mm_extract_epi64::<1>(hi128) as u64,
    ];

    out_data.reserve(36);
    out_offsets.reserve(4);
    let mut cur = out_data.len();
    let base_ptr = out_data.as_mut_ptr();
    let off_len = out_offsets.len();
    let off_ptr = out_offsets.as_mut_ptr().add(off_len);
    for (k, lane) in lanes.into_iter().enumerate() {
        let len = write_b36_from_pairs(
            b'u',
            (lane & 0xFFFF) as usize,
            ((lane >> 16) & 0xFFFF) as usize,
            ((lane >> 32) & 0xFFFF) as usize,
            ((lane >> 48) & 0xFFFF) as usize,
            base_ptr.add(cur),
        );
        cur += len;
        *off_ptr.add(k) = cur as u32;
    }
    out_data.set_len(cur);
    out_offsets.set_len(off_len + 4);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unscramble::{encode_b36, MASK40, POW36};
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    fn test_encode_b36_nodiv_and_x4_parity() {
        let mut cases = vec![
            0u64,
            1,
            35,
            36,
            37,
            1295,
            1296,
            1297,
            POW36[3] - 1,
            POW36[3],
            POW36[4] - 1,
            POW36[4],
            POW36[5] - 1,
            POW36[5],
            POW36[6] - 1,
            POW36[6],
            POW36[7] - 1,
            POW36[7],
            POW36[7] + 1,
            MASK40 - 1,
            MASK40,
        ];
        // Deterministic PRNG for 100,000 40-bit values
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            cases.push(state & MASK40);
        }

        // Pad to multiple of 4 for encode_b36_x4
        while cases.len() % 4 != 0 {
            cases.push(0);
        }

        for &prefix in &[b'u', b'e', b'k'] {
            for &val in &cases {
                let mut ref_buf = [0u8; 9];
                let mut fast_buf = [0u8; 9];
                let ref_len = encode_b36(prefix, val, &mut ref_buf);
                let fast_len = encode_b36_nodiv(prefix, val, &mut fast_buf);
                assert_eq!(fast_len, ref_len, "len mismatch for prefix={} val={}", prefix as char, val);
                assert_eq!(
                    &fast_buf[..fast_len],
                    &ref_buf[..ref_len],
                    "bytes mismatch for prefix={} val={}",
                    prefix as char,
                    val
                );
            }
        }

        // Verify 4-wide AVX2 encode_b36_x4
        let mut out_data = Vec::new();
        let mut out_offsets = vec![0u32];
        for chunk in cases.chunks_exact(4) {
            encode_b36_x4(
                [chunk[0], chunk[1], chunk[2], chunk[3]],
                &mut out_data,
                &mut out_offsets,
            );
        }
        assert_eq!(out_offsets.len(), cases.len() + 1);
        for (i, &val) in cases.iter().enumerate() {
            let mut ref_buf = [0u8; 9];
            let ref_len = encode_b36(b'u', val, &mut ref_buf);
            let s = out_offsets[i] as usize;
            let e = out_offsets[i + 1] as usize;
            assert_eq!(&out_data[s..e], &ref_buf[..ref_len], "x4 mismatch at idx={i} val={val}");
        }
    }

    #[test]
    fn bench_encode_b36_766613() {
        const N: usize = 766_613;
        let mut vals = Vec::with_capacity(N);
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        for _ in 0..N {
            state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            vals.push((state ^ (state >> 23)) & MASK40);
        }

        let mut buf = [0u8; 9];
        let mut acc_ref = 0usize;
        let t0 = Instant::now();
        for &v in &vals {
            acc_ref += encode_b36(b'u', black_box(v), &mut buf);
        }
        let ref_ns = t0.elapsed().as_nanos() as f64;
        black_box(acc_ref);

        let mut acc_nodiv = 0usize;
        let t1 = Instant::now();
        for &v in &vals {
            acc_nodiv += encode_b36_nodiv(b'u', black_box(v), &mut buf);
        }
        let nodiv_ns = t1.elapsed().as_nanos() as f64;
        black_box(acc_nodiv);

        let mut out_data_ref = vec![0u8; N * 9];
        out_data_ref.clear();
        let mut out_offsets_ref = vec![0u32; N + 1];
        out_offsets_ref.clear();
        out_offsets_ref.push(0);
        let t1b = Instant::now();
        for &v in &vals {
            out_data_ref.reserve(9);
            let base = out_data_ref.len();
            unsafe {
                let dst = &mut *(out_data_ref.as_mut_ptr().add(base) as *mut [u8; 9]);
                let len = encode_b36(b'u', black_box(v), dst);
                out_data_ref.set_len(base + len);
                out_offsets_ref.push((base + len) as u32);
            }
        }
        let ref_vec_ns = t1b.elapsed().as_nanos() as f64;
        black_box(&out_data_ref);

        let mut out_data = vec![0u8; N * 9];
        out_data.clear();
        let mut out_offsets = vec![0u32; N + 1];
        out_offsets.clear();
        out_offsets.push(0);
        let t2 = Instant::now();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            #[target_feature(enable = "avx2,bmi2,lzcnt")]
            unsafe fn run_x4(vals: &[u64], out_data: &mut Vec<u8>, out_offsets: &mut Vec<u32>) {
                for chunk in vals.chunks_exact(4) {
                    encode_b36_x4(
                        black_box([chunk[0], chunk[1], chunk[2], chunk[3]]),
                        out_data,
                        out_offsets,
                    );
                }
            }
            run_x4(&vals, &mut out_data, &mut out_offsets);
        }
        #[cfg(not(target_arch = "x86_64"))]
        for chunk in vals.chunks_exact(4) {
            encode_b36_x4(
                black_box([chunk[0], chunk[1], chunk[2], chunk[3]]),
                &mut out_data,
                &mut out_offsets,
            );
        }
        for &v in vals.chunks_exact(4).remainder() {
            let len = encode_b36_nodiv(b'u', v, &mut buf);
            out_data.extend_from_slice(&buf[..len]);
            out_offsets.push(out_data.len() as u32);
        }
        let x4_ns = t2.elapsed().as_nanos() as f64;
        black_box(&out_data);

        assert_eq!(acc_ref, acc_nodiv);
        assert_eq!(out_offsets, out_offsets_ref);
        assert_eq!(out_data, out_data_ref);
        eprintln!(
            "[b36 bench {N} encodes] stack: ref(div)={:.2} ms ({:.2} ns/id) vs nodiv={:.2} ms ({:.2} ns/id, {:.2}x) | vec: ref(div)={:.2} ms vs avx2_x4={:.2} ms ({:.2}x)",
            ref_ns / 1e6,
            ref_ns / N as f64,
            nodiv_ns / 1e6,
            nodiv_ns / N as f64,
            ref_ns / nodiv_ns,
            ref_vec_ns / 1e6,
            x4_ns / 1e6,
            ref_vec_ns / x4_ns,
        );
    }
}
