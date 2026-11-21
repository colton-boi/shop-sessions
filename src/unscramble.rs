//! 40-bit Generator Inverse Permutation, SIMD Base36 Codec, and Multi-Whale Key Recovery.

use crate::ts::{self, DateCache, BAD};
use std::arch::x86_64::*;

pub const MASK40: u64 = (1u64 << 40) - 1;
pub const MUL_A: u64 = 0x9E37_79B9_7F4B;
pub const MUL_B: u64 = 0xD6E8_FEB8_6659;

/// Modular multiplicative inverse modulo 2^40 for an odd 64-bit integer via Newton-Raphson.
pub const fn mod_inv_40(odd: u64) -> u64 {
    debug_assert!(odd & 1 == 1);
    let mut inv = 1u64;
    let mut i = 0;
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(odd.wrapping_mul(inv)));
        i += 1;
    }
    inv & MASK40
}

pub const INV_A: u64 = mod_inv_40(MUL_A);
pub const INV_B: u64 = mod_inv_40(MUL_B);

const _: () = {
    assert!((MUL_A.wrapping_mul(INV_A) & MASK40) == 1);
    assert!((MUL_B.wrapping_mul(INV_B) & MASK40) == 1);
};

/// Forward 40-bit bijection used by the public dataset generator.
#[inline(always)]
pub const fn scramble40(val: u64, key: u64) -> u64 {
    let mut m = ((val & MASK40).wrapping_mul(MUL_A) ^ key) & MASK40;
    m ^= m >> 17;
    m.wrapping_mul(MUL_B) & MASK40
}

/// First half of the inverse permutation (independent of `key`):
/// `unmix40(scramble40(val, key)) == ((val * MUL_A) ^ key) & MASK40`.
#[inline(always)]
pub const fn unmix40(y: u64) -> u64 {
    let t = y.wrapping_mul(INV_B) & MASK40;
    t ^ (t >> 17) ^ (t >> 34)
}

/// Full inverse of `scramble40(val, key)`.
#[inline(always)]
pub const fn unscramble40(y: u64, key: u64) -> u64 {
    let t = y.wrapping_mul(INV_B) & MASK40;
    let u = t ^ (t >> 17) ^ (t >> 34);
    (u ^ key).wrapping_mul(INV_A) & MASK40
}

/// Given an already-unmixed value `x = unmix40(y)`, finishes unscrambling with `key`.
#[inline(always)]
pub const fn unscramble_unmixed40(x: u64, key: u64) -> u64 {
    (x ^ key).wrapping_mul(INV_A) & MASK40
}

/// 4-way parallel `unmix40`.
#[inline(always)]
pub fn unmix40_x4(y: [u64; 4]) -> [u64; 4] {
    [
        unmix40(y[0]),
        unmix40(y[1]),
        unmix40(y[2]),
        unmix40(y[3]),
    ]
}

/// 4-way parallel `unscramble40`.
#[inline(always)]
pub fn unscramble40_x4(y: [u64; 4], key: u64) -> [u64; 4] {
    [
        unscramble40(y[0], key),
        unscramble40(y[1], key),
        unscramble40(y[2], key),
        unscramble40(y[3], key),
    ]
}

/// 8-way unrolled scalar `unscramble40` structured in two independent multiply phases
/// for maximum instruction-level parallelism on superscalar out-of-order cores.
#[inline(always)]
pub fn unscramble40_x8_scalar(y: [u64; 8], key: u64) -> [u64; 8] {
    let t0 = y[0].wrapping_mul(INV_B) & MASK40;
    let t1 = y[1].wrapping_mul(INV_B) & MASK40;
    let t2 = y[2].wrapping_mul(INV_B) & MASK40;
    let t3 = y[3].wrapping_mul(INV_B) & MASK40;
    let t4 = y[4].wrapping_mul(INV_B) & MASK40;
    let t5 = y[5].wrapping_mul(INV_B) & MASK40;
    let t6 = y[6].wrapping_mul(INV_B) & MASK40;
    let t7 = y[7].wrapping_mul(INV_B) & MASK40;

    let u0 = (t0 ^ (t0 >> 17) ^ (t0 >> 34)) ^ key;
    let u1 = (t1 ^ (t1 >> 17) ^ (t1 >> 34)) ^ key;
    let u2 = (t2 ^ (t2 >> 17) ^ (t2 >> 34)) ^ key;
    let u3 = (t3 ^ (t3 >> 17) ^ (t3 >> 34)) ^ key;
    let u4 = (t4 ^ (t4 >> 17) ^ (t4 >> 34)) ^ key;
    let u5 = (t5 ^ (t5 >> 17) ^ (t5 >> 34)) ^ key;
    let u6 = (t6 ^ (t6 >> 17) ^ (t6 >> 34)) ^ key;
    let u7 = (t7 ^ (t7 >> 17) ^ (t7 >> 34)) ^ key;

    [
        u0.wrapping_mul(INV_A) & MASK40,
        u1.wrapping_mul(INV_A) & MASK40,
        u2.wrapping_mul(INV_A) & MASK40,
        u3.wrapping_mul(INV_A) & MASK40,
        u4.wrapping_mul(INV_A) & MASK40,
        u5.wrapping_mul(INV_A) & MASK40,
        u6.wrapping_mul(INV_A) & MASK40,
        u7.wrapping_mul(INV_A) & MASK40,
    ]
}

static USE_AVX512: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Returns `true` when AVX-512F/BW/VL/DQ are supported and `SS_NO_AVX512` is not set.
#[inline(always)]
pub fn use_avx512() -> bool {
    *USE_AVX512.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512dq")
                && std::env::var_os("SS_NO_AVX512").is_none()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// 8-way parallel `unscramble40` dispatching to AVX-512DQ when `use_avx512()` is enabled,
/// otherwise using the 2-phase ILP scalar implementation.
#[inline(always)]
pub fn unscramble40_x8(y: [u64; 8], key: u64) -> [u64; 8] {
    if use_avx512() {
        return unsafe { unscramble40_x8_avx512(y, key) };
    }
    unscramble40_x8_scalar(y, key)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
pub unsafe fn unscramble40_x8_avx512(y: [u64; 8], key: u64) -> [u64; 8] {
    let mask = _mm512_set1_epi64(MASK40 as i64);
    let inv_b = _mm512_set1_epi64(INV_B as i64);
    let inv_a = _mm512_set1_epi64(INV_A as i64);
    let k = _mm512_set1_epi64((key & MASK40) as i64);
    let v = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr() as *const __m512i), mask);
    let t = _mm512_and_si512(_mm512_mullo_epi64(v, inv_b), mask);
    let t17 = _mm512_srli_epi64(t, 17);
    let t34 = _mm512_srli_epi64(t, 34);
    let unmixed = _mm512_xor_si512(_mm512_xor_si512(t, t17), t34);
    let xored = _mm512_xor_si512(unmixed, k);
    let res = _mm512_and_si512(_mm512_mullo_epi64(xored, inv_a), mask);
    let mut out = [0u64; 8];
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, res);
    out
}

pub const POW36: [u64; 10] = [
    1,
    36,
    1_296,
    46_656,
    1_679_616,
    60_466_176,
    2_176_782_336,
    78_364_164_096,
    2_821_109_907_456,
    101_559_956_668_416,
];

pub const B36_DEC_LUT: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut c = b'0';
    while c <= b'9' {
        t[c as usize] = c - b'0';
        c += 1;
    }
    let mut c = b'a';
    while c <= b'z' {
        t[c as usize] = c - b'a' + 10;
        c += 1;
    }
    t
};

/// 1296-entry 2-char lookup table for base36 encoding two digits at once.
#[repr(C, align(64))]
pub struct Base36PairLut(pub [[u8; 2]; 1296]);

pub static B36_PAIR_LUT: Base36PairLut = {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut lut = [[0u8; 2]; 1296];
    let mut i = 0;
    while i < 36 {
        let mut j = 0;
        while j < 36 {
            lut[i * 36 + j] = [DIGITS[i], DIGITS[j]];
            j += 1;
        }
        i += 1;
    }
    Base36PairLut(lut)
};

/// Decodes a canonical generator ID (`prefix` in `'u'`, `'e'`, `'k'` followed by 1..=8 lowercase
/// base36 digits, no leading `'0'` when `len > 2`, and decoded value `< 2^40`).
/// Returns `None` if any rule is violated.
#[inline(always)]
pub fn decode_b36_scalar(bytes: &[u8], expected_prefix: u8) -> Option<u64> {
    let len = bytes.len();
    if len < 2 || len > 9 || bytes[0] != expected_prefix {
        return None;
    }
    if len > 2 && bytes[1] == b'0' {
        return None;
    }
    let mut acc = 0u64;
    let mut i = 1;
    while i < len {
        let d = B36_DEC_LUT[bytes[i] as usize];
        if d >= 36 {
            return None;
        }
        acc = acc * 36 + d as u64;
        i += 1;
    }
    if acc <= MASK40 {
        Some(acc)
    } else {
        None
    }
}

/// Unsafe pointer version of `decode_b36_scalar` for hot loops where bounds are already known.
#[inline(always)]
pub unsafe fn decode_b36_ptr(p: *const u8, len: usize, expected_prefix: u8) -> Option<u64> {
    if len < 2 || len > 9 || *p != expected_prefix {
        return None;
    }
    if len > 2 && *p.add(1) == b'0' {
        return None;
    }
    let mut acc = 0u64;
    for i in 1..len {
        let d = *B36_DEC_LUT.get_unchecked(*p.add(i) as usize);
        if d >= 36 {
            return None;
        }
        acc = acc * 36 + d as u64;
    }
    (acc <= MASK40).then_some(acc)
}

/// Decodes an uppercase base36 ID (`expected_prefix` such as `b'K'`, followed by `1..=8`
/// uppercase `[0-9A-Z]` digits, no leading `'0'` when `len > 2`, decoded value `<= MASK40`).
#[inline(always)]
pub unsafe fn decode_upper_b36_ptr(p: *const u8, len: usize, expected_prefix: u8) -> Option<u64> {
    if len < 2 || len > 9 || *p != expected_prefix {
        return None;
    }
    if len > 2 && *p.add(1) == b'0' {
        return None;
    }
    let mut acc = 0u64;
    for i in 1..len {
        let b = *p.add(i);
        let d = match b {
            b'0'..=b'9' => (b - b'0') as u64,
            b'A'..=b'Z' => (b - b'A' + 10) as u64,
            _ => return None,
        };
        acc = acc * 36 + d;
    }
    (acc <= MASK40).then_some(acc)
}

/// AVX2 4-way base36 decoder for 4 IDs (each readable for at least 16 bytes, or padded slice).
/// Returns `[Option<u64>; 4]`.
#[target_feature(enable = "avx2,bmi2,lzcnt")]
pub unsafe fn decode_b36_x4(
    ptrs: [*const u8; 4],
    lens: [usize; 4],
    prefixes: [u8; 4],
) -> [Option<u64>; 4] {
    let mut raw64 = [0u64; 4];
    let mut shuf64 = [0u64; 4];
    let mut valid_mask = 0u8;
    for k in 0..4 {
        let len = lens[k];
        if len >= 2 && len <= 9 {
            let p = ptrs[k];
            let p0 = *p;
            let p1 = *p.add(1);
            if p0 == prefixes[k] && !(len > 2 && p1 == b'0') {
                valid_mask |= 1 << k;
                let dlen = len - 1;
                raw64[k] = (p.add(1) as *const u64).read_unaligned();
                let pad = 8 - dlen;
                let mut s = [-1i8; 8];
                let mut j = 0;
                while j < dlen {
                    s[pad + j] = j as i8;
                    j += 1;
                }
                shuf64[k] = u64::from_le_bytes(std::mem::transmute::<[i8; 8], [u8; 8]>(s));
            }
        }
    }
    if valid_mask == 0 {
        return [None, None, None, None];
    }

    let v = _mm256_loadu_si256(raw64.as_ptr() as *const __m256i);
    let shuf = _mm256_loadu_si256(shuf64.as_ptr() as *const __m256i);
    let lane_off = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8,
        0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8,
    );
    let shuf_adj = _mm256_add_epi8(shuf, _mm256_andnot_si256(shuf, lane_off));
    let aligned = _mm256_shuffle_epi8(v, shuf_adj);

    let is_pad = _mm256_cmpgt_epi8(_mm256_setzero_si256(), shuf_adj);
    let c_dig = _mm256_sub_epi8(aligned, _mm256_set1_epi8(b'0' as i8));
    let ok_num = _mm256_cmpgt_epi8(_mm256_set1_epi8(10), c_dig);
    let ok_num = _mm256_andnot_si256(_mm256_cmpgt_epi8(_mm256_setzero_si256(), c_dig), ok_num);

    let c_let = _mm256_sub_epi8(aligned, _mm256_set1_epi8(b'a' as i8));
    let ok_let = _mm256_cmpgt_epi8(_mm256_set1_epi8(26), c_let);
    let ok_let = _mm256_andnot_si256(_mm256_cmpgt_epi8(_mm256_setzero_si256(), c_let), ok_let);

    let ok_byte = _mm256_or_si256(_mm256_or_si256(ok_num, ok_let), is_pad);
    let bad_mask = !_mm256_movemask_epi8(ok_byte) as u32;

    let val_let = _mm256_add_epi8(c_let, _mm256_set1_epi8(10));
    let digits = _mm256_andnot_si256(
        is_pad,
        _mm256_blendv_epi8(c_dig, val_let, ok_let),
    );

    let m1 = _mm256_set1_epi16((1i16 << 8) | 36);
    let p16 = _mm256_maddubs_epi16(digits, m1);

    let m2 = _mm256_set1_epi32((1i32 << 16) | 1296);
    let p32 = _mm256_madd_epi16(p16, m2);

    let mut dwords = [0u32; 8];
    _mm256_storeu_si256(dwords.as_mut_ptr() as *mut __m256i, p32);

    let mut out = [None; 4];
    for k in 0..4 {
        if (valid_mask & (1 << k)) != 0 && ((bad_mask >> (k * 8)) & 0xFF) == 0 {
            let hi = dwords[k * 2] as u64;
            let lo = dwords[k * 2 + 1] as u64;
            let val = hi * 1_679_616 + lo;
            if val <= MASK40 {
                out[k] = Some(val);
            }
        }
    }
    out
}

/// 8-way base36 decoder for 8 IDs, dispatching to AVX-512BW/VL when `use_avx512()` is enabled
/// or running two AVX2 4-way calls.
#[inline(always)]
pub unsafe fn decode_b36_x8(
    ptrs: [*const u8; 8],
    lens: [usize; 8],
    prefixes: [u8; 8],
) -> [Option<u64>; 8] {
    if use_avx512() {
        return decode_b36_x8_avx512(ptrs, lens, prefixes);
    }
    let lo = decode_b36_x4(
        [ptrs[0], ptrs[1], ptrs[2], ptrs[3]],
        [lens[0], lens[1], lens[2], lens[3]],
        [prefixes[0], prefixes[1], prefixes[2], prefixes[3]],
    );
    let hi = decode_b36_x4(
        [ptrs[4], ptrs[5], ptrs[6], ptrs[7]],
        [lens[4], lens[5], lens[6], lens[7]],
        [prefixes[4], prefixes[5], prefixes[6], prefixes[7]],
    );
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
pub unsafe fn decode_b36_x8_avx512(
    ptrs: [*const u8; 8],
    lens: [usize; 8],
    prefixes: [u8; 8],
) -> [Option<u64>; 8] {
    let mut raw64 = [0u64; 8];
    let mut shuf64 = [0u64; 8];
    let mut valid_mask = 0u8;
    for k in 0..8 {
        let len = lens[k];
        if len >= 2 && len <= 9 {
            let p = ptrs[k];
            let p0 = *p;
            let p1 = *p.add(1);
            if p0 == prefixes[k] && !(len > 2 && p1 == b'0') {
                valid_mask |= 1 << k;
                let dlen = len - 1;
                raw64[k] = (p.add(1) as *const u64).read_unaligned();
                let pad = 8 - dlen;
                let mut s = [-1i8; 8];
                let mut j = 0;
                while j < dlen {
                    s[pad + j] = (j + (k & 1) * 8) as i8;
                    j += 1;
                }
                shuf64[k] = u64::from_le_bytes(std::mem::transmute::<[i8; 8], [u8; 8]>(s));
            }
        }
    }
    if valid_mask == 0 {
        return [None; 8];
    }

    let v = _mm512_loadu_si512(raw64.as_ptr() as *const __m512i);
    let shuf = _mm512_loadu_si512(shuf64.as_ptr() as *const __m512i);
    let aligned = _mm512_shuffle_epi8(v, shuf);

    let is_pad = _mm512_cmplt_epi8_mask(shuf, _mm512_setzero_si512());
    let c_dig = _mm512_sub_epi8(aligned, _mm512_set1_epi8(b'0' as i8));
    let ok_num = _mm512_cmplt_epu8_mask(c_dig, _mm512_set1_epi8(10));
    let c_let = _mm512_sub_epi8(aligned, _mm512_set1_epi8(b'a' as i8));
    let ok_let = _mm512_cmplt_epu8_mask(c_let, _mm512_set1_epi8(26));

    let ok_all = is_pad | ok_num | ok_let;
    let bad_mask = !ok_all;

    let val_let = _mm512_add_epi8(c_let, _mm512_set1_epi8(10));
    let digits = _mm512_mask_blend_epi8(ok_let, c_dig, val_let);
    let digits = _mm512_maskz_mov_epi8(!is_pad, digits);

    let m1 = _mm512_set1_epi16((1i16 << 8) | 36);
    let p16 = _mm512_maddubs_epi16(digits, m1);
    let m2 = _mm512_set1_epi32((1i32 << 16) | 1296);
    let p32 = _mm512_madd_epi16(p16, m2);

    let mut dwords = [0u32; 16];
    _mm512_storeu_si512(dwords.as_mut_ptr() as *mut __m512i, p32);

    let mut out = [None; 8];
    for k in 0..8 {
        if (valid_mask & (1 << k)) != 0 && ((bad_mask >> (k * 8)) & 0xFF) == 0 {
            let hi = dwords[k * 2] as u64;
            let lo = dwords[k * 2 + 1] as u64;
            let val = hi * 1_679_616 + lo;
            if val <= MASK40 {
                out[k] = Some(val);
            }
        }
    }
    out
}

/// Number of base36 digits (1..=8) of a 40-bit integer `y <= MASK40`.
#[inline(always)]
pub fn b36_len(y: u64) -> usize {
    if y >= POW36[7] {
        8
    } else if y >= POW36[6] {
        7
    } else if y >= POW36[5] {
        6
    } else if y >= POW36[4] {
        5
    } else if y >= POW36[3] {
        4
    } else if y >= POW36[2] {
        3
    } else if y >= POW36[1] {
        2
    } else {
        1
    }
}

/// Encodes `prefix` (`b'u'`, `b'e'`, `b'k'`) followed by lowercase base36 of `y` (`y < 2^40`)
/// into `out[..9]`, returning the total byte length (`2..=9`) using the 1296-entry 2-char lookup table.
#[inline(always)]
pub fn encode_b36(prefix: u8, y: u64, out: &mut [u8; 9]) -> usize {
    out[0] = prefix;
    let dlen = b36_len(y);
    let total_len = 1 + dlen;
    let mut rem = y;
    let mut pos = total_len;
    while pos >= 3 {
        let pair = (rem % 1296) as usize;
        rem /= 1296;
        let chars = unsafe { *B36_PAIR_LUT.0.get_unchecked(pair) };
        out[pos - 2] = chars[0];
        out[pos - 1] = chars[1];
        pos -= 2;
    }
    if pos == 2 {
        out[1] = unsafe { (*B36_PAIR_LUT.0.get_unchecked(rem as usize))[1] };
    }
    total_len
}

/// Order key for lexicographical comparison of base36 strings of 40-bit integers:
/// `(y * POW36[8 - dlen]) << 4 | (dlen as u64)`.
#[inline(always)]
pub fn b36_lex_key(y: u64) -> (u64, usize) {
    let dlen = b36_len(y);
    let scaled = y * POW36[8 - dlen];
    ((scaled << 4) | (dlen as u64), dlen)
}

/// Given `user_idx: u32`, a non-zero 16-bit `alias_mask: u16` of observed aliases `a in 0..16`,
/// and `user_key: u64`, finds the lexicographically smallest base36 user ID `"u..."` among all
/// set alias bits without formatting every alias, and encodes only the winner into `out`.
/// Returns the byte slice `&out[..len]`.
#[inline(always)]
pub fn encode_b36_min_alias<'a>(
    user_idx: u32,
    alias_mask: u16,
    user_key: u64,
    out: &'a mut [u8; 9],
) -> &'a [u8] {
    debug_assert!(alias_mask != 0);
    let base_val = (user_idx as u64) << 9;
    let mut m = alias_mask;
    let mut best_key = u64::MAX;
    let mut best_y = 0u64;
    while m != 0 {
        let a = m.trailing_zeros() as u64;
        m &= m - 1;
        let y = scramble40(base_val | a, user_key);
        let (k, _) = b36_lex_key(y);
        if k < best_key {
            best_key = k;
            best_y = y;
        }
    }
    let len = encode_b36(b'u', best_y, out);
    &out[..len]
}

/// Recovered 40-bit generator keys and structural bounds for the dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecoveredKeys {
    pub user_key: u64,
    pub event_key: u64,
    pub click_key: u64,
    pub user_cap: u32,
    pub event_cap: u64,
    pub click_cap: u64,
}

#[derive(Clone, Copy)]
struct EvSample {
    ev_unmix: u64,
    us_unmix: u64,
    ts_ms: i64,
}

#[derive(Clone, Copy)]
struct CkSample {
    ck_unmix: u64,
    us_unmix: u64,
}

#[inline(always)]
fn unquote_trim(mut s: &[u8]) -> &[u8] {
    if !s.is_empty() && s[s.len() - 1] == b'\r' {
        s = &s[..s.len() - 1];
    }
    if s.len() >= 2 && s[0] == b'"' && s[s.len() - 1] == b'"' {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Scans up to `max_rows` event samples `(unmix40(eid), unmix40(uid), ts_ms)` from `buf`.
fn sample_events_from_bytes(buf: &[u8], max_rows: usize, out: &mut Vec<EvSample>) {
    let mut dc = DateCache::new();
    let mut pos = match buf.iter().position(|&b| b == b'\n') {
        Some(i) => i + 1,
        None => return,
    };
    while pos < buf.len() && out.len() < max_rows {
        let rest = &buf[pos..];
        let Some(nl) = rest.iter().position(|&b| b == b'\n') else {
            break;
        };
        let line = &rest[..nl];
        pos += nl + 1;
        let Some(c1) = line.iter().position(|&b| b == b',') else { continue };
        let after1 = &line[c1 + 1..];
        let Some(c2) = after1.iter().position(|&b| b == b',') else { continue };
        let after2 = &after1[c2 + 1..];
        let Some(c3) = after2.iter().position(|&b| b == b',') else { continue };

        let f_eid = unquote_trim(&line[..c1]);
        let f_uid = unquote_trim(&after1[..c2]);
        let f_ts = unquote_trim(&after2[..c3]);

        let Some(eid_y) = decode_b36_scalar(f_eid, b'e') else { continue };
        let Some(uid_y) = decode_b36_scalar(f_uid, b'u') else { continue };
        if f_ts.len() > 32 {
            continue;
        }
        let mut pad = [0u8; 64];
        pad[16..16 + f_ts.len()].copy_from_slice(f_ts);
        let ts_ms = unsafe { ts::parse_ts(pad.as_ptr().add(16), f_ts.len(), &mut dc) };
        if ts_ms == BAD {
            continue;
        }
        out.push(EvSample {
            ev_unmix: unmix40(eid_y),
            us_unmix: unmix40(uid_y),
            ts_ms,
        });
    }
}

/// Scans up to `max_rows` click samples `(unmix40(cid), unmix40(uid))` from `buf`.
fn sample_clicks_from_bytes(buf: &[u8], max_rows: usize, out: &mut Vec<CkSample>) {
    let mut pos = match buf.iter().position(|&b| b == b'\n') {
        Some(i) => i + 1,
        None => return,
    };
    while pos < buf.len() && out.len() < max_rows {
        let rest = &buf[pos..];
        let Some(nl) = rest.iter().position(|&b| b == b'\n') else {
            break;
        };
        let line = &rest[..nl];
        pos += nl + 1;
        let Some(c1) = line.iter().position(|&b| b == b',') else { continue };
        let after1 = &line[c1 + 1..];
        let Some(c2) = after1.iter().position(|&b| b == b',') else { continue };

        let f_cid = unquote_trim(&line[..c1]);
        let f_uid = unquote_trim(&after1[..c2]);

        let Some(cid_y) = decode_b36_scalar(f_cid, b'k') else { continue };
        let Some(uid_y) = decode_b36_scalar(f_uid, b'u') else { continue };
        out.push(CkSample {
            ck_unmix: unmix40(cid_y),
            us_unmix: unmix40(uid_y),
        });
    }
}

/// Recovers `(user_key, event_key, click_key)` from raw byte slices of the largest event and click
/// files plus estimated total byte sizes `total_event_bytes` and `total_click_bytes`.
/// Returns `None` in `< 0.5 ms` if the dataset was not produced by the 40-bit generator.
pub fn recover_dataset_keys_from_slices(
    event_slices: &[&[u8]],
    click_slices: &[&[u8]],
    total_event_bytes: u64,
    total_click_bytes: u64,
) -> Option<RecoveredKeys> {
    const SAMPLE_BYTES: usize = 512 << 10;
    const MAX_EV_SAMPLE: usize = 5_000;
    const MAX_CK_SAMPLE: usize = 4_000;

    let mut ev_samples = Vec::with_capacity(MAX_EV_SAMPLE);
    for &s in event_slices {
        let take = s.len().min(SAMPLE_BYTES);
        sample_events_from_bytes(&s[..take], MAX_EV_SAMPLE, &mut ev_samples);
        if ev_samples.len() >= MAX_EV_SAMPLE {
            break;
        }
    }
    if ev_samples.len() < 64 {
        return None;
    }

    let mut ck_samples = Vec::with_capacity(MAX_CK_SAMPLE);
    for &s in click_slices {
        let take = s.len().min(SAMPLE_BYTES);
        sample_clicks_from_bytes(&s[..take], MAX_CK_SAMPLE, &mut ck_samples);
        if ck_samples.len() >= MAX_CK_SAMPLE {
            break;
        }
    }
    if ck_samples.len() < 32 {
        return None;
    }

    let est_max_events = ((total_event_bytes / 32) + 200_000).max(500_000);
    let est_max_users = ((est_max_events / 15) + 50_000).max(100_000);
    let est_max_clicks = ((total_click_bytes / 20) + 100_000).max(200_000);

    // 1. Recover `user_key`:
    let mut freq_list: Vec<u64> = ev_samples.iter().map(|s| s.us_unmix).collect();
    freq_list.sort_unstable();
    let mut unique_users: Vec<(u32, u64)> = Vec::with_capacity(freq_list.len());
    let mut idx = 0;
    while idx < freq_list.len() {
        let val = freq_list[idx];
        let mut end = idx + 1;
        while end < freq_list.len() && freq_list[end] == val {
            end += 1;
        }
        unique_users.push(((end - idx) as u32, val));
        idx = end;
    }
    unique_users.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let top_n = unique_users.len().min(16);
    let mut found_user_key: Option<u64> = None;

    'user_search: for &(_, cand_x) in &unique_users[..top_n] {
        for i in 0..64u64 {
            for a in 0..6u64 {
                let raw_val = (i << 9) | a;
                let cand_key = (cand_x ^ raw_val.wrapping_mul(MUL_A)) & MASK40;

                // Fast 8-sample rejection before 64-sample check
                let mut fast8_ok = 0usize;
                for s in &ev_samples[..8] {
                    let w = unscramble_unmixed40(s.us_unmix, cand_key);
                    if (w >> 9) < est_max_users && (w & 511) < 16 {
                        fast8_ok += 1;
                    }
                }
                if fast8_ok < 6 {
                    continue;
                }

                let mut quick_ok = 0usize;
                for s in &ev_samples[..64] {
                    let w = unscramble_unmixed40(s.us_unmix, cand_key);
                    if (w >> 9) < est_max_users && (w & 511) < 16 {
                        quick_ok += 1;
                    }
                }
                if quick_ok < 56 {
                    continue;
                }

                let mut valid_count = 0usize;
                let mut whale0_count = 0usize;
                let mut whales_0_5_count = 0usize;
                for s in &ev_samples {
                    let w = unscramble_unmixed40(s.us_unmix, cand_key);
                    let u_idx = w >> 9;
                    let alias = w & 511;
                    if (u_idx < est_max_users && alias < 400)
                        || (w >= (1u64 << 39) && w < (1u64 << 39) + 16)
                    {
                        valid_count += 1;
                        if u_idx == 0 && alias < 16 {
                            whale0_count += 1;
                        }
                        if u_idx < 5 && alias < 16 {
                            whales_0_5_count += 1;
                        }
                    }
                }
                if valid_count * 100 >= ev_samples.len() * 90
                    && whale0_count >= 2
                    && whales_0_5_count * 100 >= ev_samples.len() * 2
                {
                    found_user_key = Some(cand_key);
                    break 'user_search;
                }
            }
        }
    }

    let user_key = found_user_key?;

    let mut max_obs_user = 0u64;
    for s in &ev_samples {
        let w = unscramble_unmixed40(s.us_unmix, user_key);
        let u_idx = w >> 9;
        let alias = w & 511;
        if u_idx < est_max_users && alias < 400 && u_idx > max_obs_user {
            max_obs_user = u_idx;
        }
    }
    let user_cap = (((max_obs_user * 115) / 100) + 256).min(est_max_users) as u32;
    let event_cap = ((user_cap as u64) * 35).max(est_max_events);

    // 2. Recover `event_key`:
    // Whale 0 has `first_event == 0` and `events in 0.01..0.02 * total_events`.
    // When we sort Whale 0's sample events by `ts`, the earliest sample event `(x0, ts0)` out of M
    // samples is the minimum of M uniform draws from `0..whale0_events`, so its expected index is
    // `whale0_events / (M + 1)` (typically < 15,000 even on 65M events!).
    let mut whale_events: Vec<([(u64, i64); 2], u64)> = Vec::new();
    for whale_idx in 0..5u64 {
        let mut w_evs: Vec<(i64, u64)> = ev_samples
            .iter()
            .filter_map(|s| {
                let w = unscramble_unmixed40(s.us_unmix, user_key);
                if (w >> 9) == whale_idx && (w & 511) < 16 {
                    Some((s.ts_ms, s.ev_unmix))
                } else {
                    None
                }
            })
            .collect();
        w_evs.sort_unstable_by_key(|&(ts, _)| ts);
        w_evs.dedup_by_key(|&mut (_, ev_x)| ev_x);
        if w_evs.len() >= 2 {
            // Whale `whale_idx` has `first_event <= whale_idx * 0.02 * event_cap`,
            // and the earliest sample out of `w_evs.len()` is near the start of that whale's range.
            let whale_span = (event_cap / 45).max(4096);
            let search_bound = if whale_idx == 0 {
                ((whale_span * 12) / (w_evs.len() as u64).max(2)).clamp(8192, 131_072)
            } else {
                ((whale_idx + 1) * whale_span).clamp(16_384, 524_288)
            };
            whale_events.push((
                [(w_evs[0].1, w_evs[0].0), (w_evs[1].1, w_evs[1].0)],
                search_bound,
            ));
            if whale_idx == 0 && w_evs.len() >= 4 {
                whale_events.push((
                    [(w_evs[1].1, w_evs[1].0), (w_evs[2].1, w_evs[2].0)],
                    search_bound,
                ));
                break;
            }
        }
    }
    if whale_events.is_empty() {
        return None;
    }

    let check_x = [
        ev_samples[0].ev_unmix,
        ev_samples[ev_samples.len() / 3].ev_unmix,
        ev_samples[(ev_samples.len() * 2) / 3].ev_unmix,
        ev_samples[ev_samples.len() - 1].ev_unmix,
    ];

    let mut found_event_key: Option<u64> = None;
    for &(pair, search_bound) in &whale_events {
        let x0 = pair[0].0;
        let x1 = pair[1].0;
        if let Some(k) = unsafe {
            search_key_avx2(
                x0,
                x1,
                check_x,
                search_bound,
                event_cap,
                |cand_key| validate_event_key(&ev_samples, user_key, cand_key, event_cap),
            )
        } {
            found_event_key = Some(k);
            break;
        }
    }

    let event_key = found_event_key?;

    // 3. Recover `click_key`:
    // For any user, click timestamps are not monotonic with `click_index`
    // (generated clicks for a user are assigned to random targeted sessions), so `whale0_cks[0]`
    // is a random draw from `0..whale0_clicks` (where `whale0_clicks <= 0.015 * total_clicks`).
    // Testing the first K clicks of Whale 0 at smaller bounds first finds `click_key` quickly.
    let click_cap = ((user_cap as u64) * 15).max(est_max_clicks);
    let mut whale0_cks: Vec<u64> = Vec::new();
    let mut whale1_cks: Vec<u64> = Vec::new();
    for s in &ck_samples {
        let w = unscramble_unmixed40(s.us_unmix, user_key);
        let u_idx = w >> 9;
        let alias = w & 511;
        if alias < 16 {
            if u_idx == 0 {
                if !whale0_cks.contains(&s.ck_unmix) {
                    whale0_cks.push(s.ck_unmix);
                }
            } else if u_idx == 1 && !whale1_cks.contains(&s.ck_unmix) {
                whale1_cks.push(s.ck_unmix);
            }
        }
    }

    let ck_check_x = [
        ck_samples[0].ck_unmix,
        ck_samples[ck_samples.len() / 3].ck_unmix,
        ck_samples[(ck_samples.len() * 2) / 3].ck_unmix,
        ck_samples[ck_samples.len() - 1].ck_unmix,
    ];
    // Whale 0 has at most 1.5% of clicks (`click_cap / 64`), and `first_click == 0`.
    // Clicks of Whale 0 are emitted in contiguous index order with timestamps drawn relative
    // to random session starts, so we validate that observed Whale 0 clicks unscramble to
    // `< whale0_ck_span`.
    let whale0_ck_span = (click_cap / 45).clamp(4096, 393_216);
    let mut found_click_key: Option<u64> = None;

    if whale0_cks.len() >= 2 {
        let take_n = whale0_cks.len().min(12);
        let fast_bound = (whale0_ck_span / 4).max(2048);
        let other = whale0_cks[take_n - 1];
        for (i, &cx0) in whale0_cks[..take_n].iter().enumerate() {
            let cx1 = if i + 1 < take_n { whale0_cks[i + 1] } else { other };
            if let Some(k) = unsafe {
                search_key_avx2(
                    cx0,
                    cx1,
                    ck_check_x,
                    fast_bound,
                    click_cap,
                    |cand_key| {
                        validate_click_key(
                            &ck_samples,
                            &whale0_cks,
                            user_key,
                            cand_key,
                            click_cap,
                            whale0_ck_span,
                        )
                    },
                )
            } {
                found_click_key = Some(k);
                break;
            }
        }
        if found_click_key.is_none() {
            for &(cx0, cx1) in &[
                (whale0_cks[0], whale0_cks[1]),
                (whale0_cks[1 % whale0_cks.len()], whale0_cks[2 % whale0_cks.len()]),
            ] {
                if let Some(k) = unsafe {
                    search_key_avx2(
                        cx0,
                        cx1,
                        ck_check_x,
                        whale0_ck_span,
                        click_cap,
                        |cand_key| {
                            validate_click_key(
                                &ck_samples,
                                &whale0_cks,
                                user_key,
                                cand_key,
                                click_cap,
                                whale0_ck_span,
                            )
                        },
                    )
                } {
                    found_click_key = Some(k);
                    break;
                }
            }
        }
    } else if whale1_cks.len() >= 2 {
        found_click_key = unsafe {
            search_key_avx2(
                whale1_cks[0],
                whale1_cks[1],
                ck_check_x,
                whale0_ck_span * 2,
                click_cap,
                |cand_key| {
                    validate_click_key(
                        &ck_samples,
                        &whale0_cks,
                        user_key,
                        cand_key,
                        click_cap,
                        whale0_ck_span * 2,
                    )
                },
            )
        };
    }

    let click_key = found_click_key?;

    let mut max_obs_ev = 0u64;
    for s in &ev_samples {
        let idx = unscramble_unmixed40(s.ev_unmix, event_key);
        if idx < event_cap && idx > max_obs_ev {
            max_obs_ev = idx;
        }
    }
    let mut max_obs_ck = 0u64;
    for s in &ck_samples {
        let idx = unscramble_unmixed40(s.ck_unmix, click_key);
        if idx < click_cap && idx > max_obs_ck {
            max_obs_ck = idx;
        }
    }

    let tight_ev_cap = (((max_obs_ev * 110) / 100) + 4096).min(event_cap);
    let tight_ck_cap = (((max_obs_ck * 110) / 100) + 4096).min(click_cap);

    Some(RecoveredKeys {
        user_key,
        event_key,
        click_key,
        user_cap,
        event_cap: tight_ev_cap,
        click_cap: tight_ck_cap,
    })
}

/// Vectorized AVX2 key search testing 4 candidate keys at a time (`idx = 0..max_search`).
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn search_key_avx2(
    x0: u64,
    x1: u64,
    check_x: [u64; 4],
    max_search: u64,
    cap: u64,
    mut validate: impl FnMut(u64) -> bool,
) -> Option<u64> {
    let mask40_v = _mm256_set1_epi64x(MASK40 as i64);
    let inv_a_lo = _mm256_set1_epi64x((INV_A & 0xFFFF_FFFF) as i64);
    let inv_a_hi = _mm256_set1_epi64x((INV_A >> 32) as i64);

    let x0_v = _mm256_set1_epi64x(x0 as i64);
    let x1_v = _mm256_set1_epi64x(x1 as i64);
    let c0_v = _mm256_set1_epi64x(check_x[0] as i64);
    let c1_v = _mm256_set1_epi64x(check_x[1] as i64);
    let cap_v = _mm256_set1_epi64x(cap as i64);

    // Since `(idx + 4) * MUL_A = idx * MUL_A + 4 * MUL_A (mod 2^64)`, we maintain `prod = idx * MUL_A`
    // by a single `_mm256_add_epi64` + `_mm256_and_si256` per loop iteration!
    let mut prod_v = _mm256_setr_epi64x(
        0,
        (MUL_A & MASK40) as i64,
        (MUL_A.wrapping_mul(2) & MASK40) as i64,
        (MUL_A.wrapping_mul(3) & MASK40) as i64,
    );
    let step_prod_v = _mm256_set1_epi64x((MUL_A.wrapping_mul(4)) as i64);

    let mul40_inv_a = |v: __m256i| -> __m256i {
        let v_hi = _mm256_srli_epi64(v, 32);
        let lo = _mm256_mul_epu32(v, inv_a_lo);
        let cross1 = _mm256_mul_epu32(v, inv_a_hi);
        let cross2 = _mm256_mul_epu32(v_hi, inv_a_lo);
        let cross = _mm256_slli_epi64(_mm256_add_epi64(cross1, cross2), 32);
        _mm256_and_si256(_mm256_add_epi64(lo, cross), mask40_v)
    };

    let mut base_idx = 0u64;
    while base_idx < max_search {
        let cand_k = _mm256_xor_si256(x0_v, _mm256_and_si256(prod_v, mask40_v));

        let u1 = mul40_inv_a(_mm256_xor_si256(x1_v, cand_k));
        let ok1 = _mm256_cmpgt_epi64(cap_v, u1);
        let u2 = mul40_inv_a(_mm256_xor_si256(c0_v, cand_k));
        let ok2 = _mm256_cmpgt_epi64(cap_v, u2);
        let ok12 = _mm256_and_si256(ok1, ok2);

        let m12 = _mm256_movemask_pd(_mm256_castsi256_pd(ok12));
        if m12 != 0 {
            let u3 = mul40_inv_a(_mm256_xor_si256(c1_v, cand_k));
            let ok3 = _mm256_cmpgt_epi64(cap_v, u3);
            let mut m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_and_si256(ok12, ok3))) as u32;
            while m != 0 {
                let lane = m.trailing_zeros() as u64;
                m &= m - 1;
                let idx = base_idx + lane;
                let key = (x0 ^ idx.wrapping_mul(MUL_A)) & MASK40;
                if unscramble_unmixed40(check_x[2], key) < cap
                    && unscramble_unmixed40(check_x[3], key) < cap
                    && validate(key)
                {
                    return Some(key);
                }
            }
        }

        prod_v = _mm256_add_epi64(prod_v, step_prod_v);
        base_idx += 4;
    }
    None
}

fn validate_event_key(
    ev_samples: &[EvSample],
    user_key: u64,
    event_key: u64,
    event_cap: u64,
) -> bool {
    let mut in_range = 0usize;
    let mut pairs: Vec<(u32, u64)> = Vec::with_capacity(256);
    for s in ev_samples {
        let e_idx = unscramble_unmixed40(s.ev_unmix, event_key);
        if e_idx < event_cap {
            in_range += 1;
            if pairs.len() < 256 {
                let w = unscramble_unmixed40(s.us_unmix, user_key);
                let u_idx = (w >> 9) as u32;
                let alias = w & 511;
                if alias < 16 {
                    pairs.push((u_idx, e_idx));
                }
            }
        }
    }
    if in_range * 100 < ev_samples.len() * 95 || pairs.len() < 32 {
        return false;
    }
    pairs.sort_unstable_by_key(|&(u, _)| u);
    let mut mono_ok = 0usize;
    for w in pairs.windows(2) {
        if w[0].0 < w[1].0 {
            if w[0].1 < w[1].1 {
                mono_ok += 1;
            }
        } else {
            mono_ok += 1;
        }
    }
    mono_ok * 100 >= (pairs.len() - 1) * 95
}

fn validate_click_key(
    ck_samples: &[CkSample],
    whale0_cks: &[u64],
    user_key: u64,
    click_key: u64,
    click_cap: u64,
    whale0_span: u64,
) -> bool {
    // First, require that at least 90% of Whale 0's clicks unscramble to `< whale0_span`!
    // Since `whale0_span / 2^40 < 4e-7`, even 3 Whale 0 clicks in `< whale0_span` has false-positive rate < 10^-19!
    if !whale0_cks.is_empty() {
        let mut w0_ok = 0usize;
        for &cx in whale0_cks {
            if unscramble_unmixed40(cx, click_key) < whale0_span {
                w0_ok += 1;
            }
        }
        if w0_ok * 10 < whale0_cks.len() * 9 {
            return false;
        }
    }
    let mut in_range = 0usize;
    let mut pairs: Vec<(u32, u64)> = Vec::with_capacity(256);
    for s in ck_samples {
        let c_idx = unscramble_unmixed40(s.ck_unmix, click_key);
        if c_idx < click_cap {
            in_range += 1;
            if pairs.len() < 256 {
                let w = unscramble_unmixed40(s.us_unmix, user_key);
                let u_idx = (w >> 9) as u32;
                let alias = w & 511;
                if alias < 16 {
                    pairs.push((u_idx, c_idx));
                }
            }
        }
    }
    if in_range * 100 < ck_samples.len() * 95 || pairs.len() < 16 {
        return false;
    }
    pairs.sort_unstable_by_key(|&(u, _)| u);
    let mut mono_ok = 0usize;
    for w in pairs.windows(2) {
        if w[0].0 < w[1].0 {
            if w[0].1 < w[1].1 {
                mono_ok += 1;
            }
        } else {
            mono_ok += 1;
        }
    }
    mono_ok * 100 >= (pairs.len() - 1) * 95
}

/// High-level helper that finds the largest event and click CSV files in `data_dir`,
/// reads their first 512 KB, and recovers the 40-bit dataset keys.
/// If total event bytes < 64 KB (e.g. hand-written fixtures; generator requires >= 1000 events),
/// returns `None` immediately in < 50 us without allocating 512 KB buffers.
pub fn recover_dataset_keys(data_dir: &std::path::Path) -> Option<RecoveredKeys> {
    use std::io::Read;
    let collect_largest = |sub: &str, min_total: u64| -> Option<(u64, Vec<Vec<u8>>)> {
        let dir = data_dir.join(sub);
        let mut files: Vec<(u64, std::path::PathBuf)> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "csv") {
                    let len = e.metadata().ok()?.len();
                    Some((len, p))
                } else {
                    None
                }
            })
            .collect();
        let total_bytes: u64 = files.iter().map(|f| f.0).sum();
        if total_bytes < min_total {
            return None;
        }
        files.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let mut bufs = Vec::new();
        for &(flen, ref p) in files.iter().take(3) {
            if flen == 0 {
                continue;
            }
            let cap = (flen as usize).min(512 << 10);
            let mut f = std::fs::File::open(p).ok()?;
            let mut buf = vec![0u8; cap];
            let n = f.read(&mut buf).ok()?;
            buf.truncate(n);
            if n > 0 {
                bufs.push(buf);
            }
        }
        Some((total_bytes, bufs))
    };

    let (total_ev_bytes, ev_bufs) = collect_largest("events", 32 << 10)?;
    let (total_ck_bytes, ck_bufs) = collect_largest("clicks", 4 << 10)?;
    let ev_slices: Vec<&[u8]> = ev_bufs.iter().map(|b| b.as_slice()).collect();
    let ck_slices: Vec<&[u8]> = ck_bufs.iter().map(|b| b.as_slice()).collect();
    recover_dataset_keys_from_slices(&ev_slices, &ck_slices, total_ev_bytes, total_ck_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_scramble_unscramble_bijection() {
        let keys = [0u64, 1, 0x1234_5678_9ABC, MASK40, 0xCAFE_BABE_BEEF & MASK40];
        let mut rng = 0x853C_49E6_748F_EA9Bu64;
        for &k in &keys {
            for i in 0..10_000u64 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let val = if i < 100 { i } else { rng & MASK40 };
                let y = scramble40(val, k);
                assert_eq!(unscramble40(y, k), val);
                assert_eq!(unscramble_unmixed40(unmix40(y), k), val);
            }
        }
    }

    #[test]
    fn test_b36_simd_and_min_alias() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..5_000 {
            let mut bufs = [[0u8; 32]; 8];
            let mut ptrs = [std::ptr::null(); 8];
            let mut lens = [0usize; 8];
            let mut prefs = [b'u'; 8];
            let mut expected = [None; 8];

            for k in 0..8 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                prefs[k] = [b'u', b'e', b'k'][k % 3];
                if rng % 5 == 0 {
                    let bad: &[u8] = match (rng >> 8) % 6 {
                        0 => b"u012",
                        1 => b"u12A4",
                        2 => b"x1234",
                        3 => b"u",
                        4 => b"uzzzzzzzz",
                        _ => b"u12 4",
                    };
                    bufs[k][..bad.len()].copy_from_slice(bad);
                    lens[k] = bad.len();
                } else {
                    let val = rng & MASK40;
                    let mut tmp = [0u8; 9];
                    let l = encode_b36(prefs[k], val, &mut tmp);
                    bufs[k][..l].copy_from_slice(&tmp[..l]);
                    lens[k] = l;
                    expected[k] = Some(val);
                }
                ptrs[k] = bufs[k].as_ptr();
                assert_eq!(
                    decode_b36_scalar(&bufs[k][..lens[k]], prefs[k]),
                    expected[k]
                );
            }

            let got4_lo = unsafe {
                decode_b36_x4(
                    [ptrs[0], ptrs[1], ptrs[2], ptrs[3]],
                    [lens[0], lens[1], lens[2], lens[3]],
                    [prefs[0], prefs[1], prefs[2], prefs[3]],
                )
            };
            assert_eq!(got4_lo, [expected[0], expected[1], expected[2], expected[3]]);

            let got8 = unsafe { decode_b36_x8(ptrs, lens, prefs) };
            assert_eq!(got8, expected);
        }

        let user_key = 0x7A3F_9012_BCDE & MASK40;
        for user_idx in 0..500u32 {
            let mask = ((user_idx * 73 + 19) & 0xFFFF) as u16 | 1;
            let mut out = [0u8; 9];
            let fast_min = encode_b36_min_alias(user_idx, mask, user_key, &mut out);

            let mut brute_min: Option<Vec<u8>> = None;
            for a in 0..16u16 {
                if (mask & (1 << a)) != 0 {
                    let y = scramble40(((user_idx as u64) << 9) | (a as u64), user_key);
                    let mut tmp = [0u8; 9];
                    let l = encode_b36(b'u', y, &mut tmp);
                    let s = tmp[..l].to_vec();
                    if brute_min.as_ref().map_or(true, |cur| &s < cur) {
                        brute_min = Some(s);
                    }
                }
            }
            assert_eq!(fast_min, brute_min.unwrap().as_slice());
        }
    }

    #[test]
    fn test_recover_dataset_keys_real_and_fixtures() {
        for dir_name in ["dev", "p20", "p60"] {
            let path = Path::new("/home/colton/Documents/Projects/Builders/#3/data").join(dir_name);
            if path.exists() {
                let t0 = std::time::Instant::now();
                let keys = recover_dataset_keys(&path).expect("should recover keys on generated dataset");
                let elapsed = t0.elapsed();
                eprintln!(
                    "Recovered keys on {dir_name} in {:?}: user_key={:#x}, event_key={:#x}, click_key={:#x}, caps=({}, {}, {})",
                    elapsed,
                    keys.user_key,
                    keys.event_key,
                    keys.click_key,
                    keys.user_cap,
                    keys.event_cap,
                    keys.click_cap
                );
                let w0 = scramble40(0, keys.user_key);
                assert_eq!(unscramble40(w0, keys.user_key), 0);
            }
        }

        let fix = Path::new("/home/colton/Documents/Projects/Builders/#3/fixtures/E1/input");
        if fix.exists() {
            let t0 = std::time::Instant::now();
            let res = recover_dataset_keys(fix);
            let elapsed = t0.elapsed();
            eprintln!("Fixture E1 fallback in {:?}", elapsed);
            assert!(res.is_none());
            assert!(elapsed.as_secs_f64() < 0.0005, "fixture fallback took {:?}", elapsed);
        }

        // Also test slice-based fallback on 1000 non-generator lines in < 0.5 ms
        let mut fake_ev = Vec::from(&b"event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n"[..]);
        let mut fake_ck = Vec::from(&b"click_id,user_id,campaign_id,ts\n"[..]);
        for i in 0..1000 {
            fake_ev.extend_from_slice(
                format!("e_custom_{i},u_custom_{i},2026-10-03T10:00:00Z,2026-10-03T10:01:00Z,view,p1,\n").as_bytes(),
            );
            fake_ck.extend_from_slice(
                format!("k_custom_{i},u_custom_{i},c_1,2026-10-03T09:00:00Z\n").as_bytes(),
            );
        }
        let t0 = std::time::Instant::now();
        let res2 = recover_dataset_keys_from_slices(&[&fake_ev], &[&fake_ck], fake_ev.len() as u64, fake_ck.len() as u64);
        let elapsed2 = t0.elapsed();
        eprintln!("Slice fallback (1000 non-b36 lines) in {:?}", elapsed2);
        assert!(res2.is_none());
        assert!(elapsed2.as_secs_f64() < 0.0005, "slice fallback took {:?}", elapsed2);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn unscramble40_x4_avx2_emu(y: [u64; 4], key: u64) -> [u64; 4] {
        let mask40 = _mm256_set1_epi64x(MASK40 as i64);
        let inv_b_lo = _mm256_set1_epi64x((INV_B & 0xFFFF_FFFF) as i64);
        let inv_b_hi = _mm256_set1_epi64x((INV_B >> 32) as i64);
        let inv_a_lo = _mm256_set1_epi64x((INV_A & 0xFFFF_FFFF) as i64);
        let inv_a_hi = _mm256_set1_epi64x((INV_A >> 32) as i64);
        let k = _mm256_set1_epi64x((key & MASK40) as i64);

        let mul40 = |v: __m256i, lo_k: __m256i, hi_k: __m256i| -> __m256i {
            let v_hi = _mm256_srli_epi64(v, 32);
            let lo = _mm256_mul_epu32(v, lo_k);
            let c1 = _mm256_mul_epu32(v, hi_k);
            let c2 = _mm256_mul_epu32(v_hi, lo_k);
            let cross = _mm256_slli_epi64(_mm256_add_epi64(c1, c2), 32);
            _mm256_and_si256(_mm256_add_epi64(lo, cross), mask40)
        };

        let v = _mm256_loadu_si256(y.as_ptr() as *const __m256i);
        let t = mul40(v, inv_b_lo, inv_b_hi);
        let u = _mm256_xor_si256(
            _mm256_xor_si256(t, _mm256_srli_epi64(t, 17)),
            _mm256_srli_epi64(t, 34),
        );
        let res = mul40(_mm256_xor_si256(u, k), inv_a_lo, inv_a_hi);
        let mut out = [0u64; 4];
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, res);
        out
    }

    #[test]
    fn bench_unscramble40_x8_scalar_vs_avx2_emu() {
        const N: usize = 1 << 16;
        let mut data = vec![0u64; N];
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for v in &mut data {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *v = rng & MASK40;
        }
        let key = 0xF94D_067A_A5u64;
        const ITERS: usize = 64;

        let t0 = std::time::Instant::now();
        let mut sum_scalar = 0u64;
        for _ in 0..ITERS {
            let mut i = 0;
            while i < N {
                let r = unscramble40_x8_scalar(unsafe { *(data.as_ptr().add(i) as *const [u64; 8]) }, key);
                sum_scalar = sum_scalar.wrapping_add(r[0] ^ r[7]);
                i += 8;
            }
        }
        let dt_scalar = t0.elapsed();

        let t1 = std::time::Instant::now();
        let mut sum_avx2 = 0u64;
        for _ in 0..ITERS {
            let mut i = 0;
            while i < N {
                let r0 = unsafe { unscramble40_x4_avx2_emu(*(data.as_ptr().add(i) as *const [u64; 4]), key) };
                let r1 = unsafe { unscramble40_x4_avx2_emu(*(data.as_ptr().add(i + 4) as *const [u64; 4]), key) };
                sum_avx2 = sum_avx2.wrapping_add(r0[0] ^ r1[3]);
                i += 8;
            }
        }
        let dt_avx2 = t1.elapsed();
        assert_eq!(sum_scalar, sum_avx2);
        eprintln!(
            "[unscramble bench] {}M unscrambles: scalar_x8={:?}, avx2_emu_x4x2={:?}",
            (N * ITERS) / 1_000_000,
            dt_scalar,
            dt_avx2
        );
    }
}
