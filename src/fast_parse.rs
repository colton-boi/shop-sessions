//! Width+Quote-Indexed Table-Driven SIMD CSV Parser & 64-Row Columnar Batch Validator.
//!
//! Solves the 20% random-quoting problem without branches by indexing `const fn`-generated
//! `#[repr(C, align(64))]` lookup tables by `(raw_field_width, has_leading_quote_bit)` (or
//! combined `(raw_width_0, quote_0, raw_width_1, quote_1)` for ID pairs).
//!
//! Any row that fails the SIMD width/quote/character check falls back per-row to an exact
//! scalar parser matching all spec rules (including `\r\n`, spaces, arbitrary timestamps,
//! long IDs, and malformed rows).

use crate::dims;
use crate::fast_links::encode_base36_token;
use crate::ids::tok_pair;
use crate::ts::{self, DateCache, DateLut, BAD, DAY_MS};
use crate::unscramble::{
    unscramble40, unscramble40_x8_avx512, unscramble40_x8_scalar, use_avx512, MASK40,
};
use crate::Shared;
use std::arch::x86_64::*;

pub const FAST_BATCH: usize = 64;
const WEEK_MS: i64 = 7 * DAY_MS;

/// Lookup table entry for a single base36 ID field (`u...`, `e...`, `k...`),
/// indexed by `((raw_width.min(15)) << 1) | (quote_bit & 1)` (32 entries).
#[repr(C, align(64))]
pub struct IdShuffleLut {
    /// For each `(raw_width, quote_bit)` index `0..32`, holds the 8-byte `_mm_shuffle_epi8` mask
    /// that right-aligns the base36 payload digits (excluding the `'u'`/`'e'`/`'k'` prefix and
    /// optional surrounding quotes) into an 8-byte lane, with `-1` (0x80) in padding positions.
    pub shuf: [[i8; 8]; 32],
    /// Offset of the prefix character (`0` if unquoted, `1` if quoted).
    pub prefix_off: [u8; 32],
    /// Offset of the first base36 digit (`1` if unquoted, `2` if quoted).
    pub first_digit_off: [u8; 32],
    /// Number of base36 payload digits (`1..=8`), or `0` if invalid `(raw_width, quote_bit)`.
    pub dlen: [u8; 32],
}

pub const ID_SHUFFLE_LUT: IdShuffleLut = {
    let mut shuf = [[-1i8; 8]; 32];
    let mut prefix_off = [0u8; 32];
    let mut first_digit_off = [0u8; 32];
    let mut dlen = [0u8; 32];

    let mut raw_w = 0usize;
    while raw_w < 16 {
        let mut q = 0usize;
        while q < 2 {
            let idx = (raw_w << 1) | q;
            let quote_bytes = q * 2;
            if raw_w >= 2 + quote_bytes && raw_w <= 9 + quote_bytes {
                let dl = raw_w - 1 - quote_bytes; // 1..=8
                let pad = 8 - dl;
                let start = 1 + q; // offset of first base36 digit from raw field start
                let mut j = 0;
                while j < dl {
                    shuf[idx][pad + j] = (start + j) as i8;
                    j += 1;
                }
                prefix_off[idx] = q as u8;
                first_digit_off[idx] = start as u8;
                dlen[idx] = dl as u8;
            }
            q += 1;
        }
        raw_w += 1;
    }
    IdShuffleLut {
        shuf,
        prefix_off,
        first_digit_off,
        dlen,
    }
};

/// Lookup table for decimal integer fields (`c_<N>` campaigns, `p<N>` products, and `price_cents`),
/// indexed by `((raw_width.min(15)) << 1) | (quote_bit & 1)` (32 entries).
#[repr(C, align(64))]
pub struct DecFieldLut {
    /// 8-byte `_mm_shuffle_epi8` mask right-aligning the digits into an 8-byte lane.
    pub shuf: [[i8; 8]; 32],
    /// Number of decimal digits, or `0` if invalid `(raw_width, quote_bit)`.
    pub dlen: [u8; 32],
    /// Offset of the first digit from raw field start.
    pub first_digit_off: [u8; 32],
}

const fn build_dec_lut(prefix_len: usize, max_digits: usize) -> DecFieldLut {
    let mut shuf = [[-1i8; 8]; 32];
    let mut dlen = [0u8; 32];
    let mut first_digit_off = [0u8; 32];

    let mut raw_w = 0usize;
    while raw_w < 16 {
        let mut q = 0usize;
        while q < 2 {
            let idx = (raw_w << 1) | q;
            let quote_bytes = q * 2;
            if raw_w >= prefix_len + 1 + quote_bytes
                && raw_w <= prefix_len + max_digits + quote_bytes
            {
                let dl = raw_w - prefix_len - quote_bytes;
                let pad = 8 - dl;
                let start = prefix_len + q;
                let mut j = 0;
                while j < dl {
                    shuf[idx][pad + j] = (start + j) as i8;
                    j += 1;
                }
                dlen[idx] = dl as u8;
                first_digit_off[idx] = start as u8;
            }
            q += 1;
        }
        raw_w += 1;
    }
    DecFieldLut {
        shuf,
        dlen,
        first_digit_off,
    }
}

/// Table for `c_<N>` campaign fields (`prefix_len = 2`, `1..=5` digits).
pub const CAMP_SHUFFLE_LUT: DecFieldLut = build_dec_lut(2, 5);
/// Table for `p<N>` product fields (`prefix_len = 1`, `1..=6` digits).
pub const PROD_SHUFFLE_LUT: DecFieldLut = build_dec_lut(1, 6);
/// Table for `price_cents` fields (`prefix_len = 0`, `1..=8` digits).
pub const PRICE_SHUFFLE_LUT: DecFieldLut = build_dec_lut(0, 8);

/// Timestamp width+quote lookup table indexed by `((raw_width.min(31)) << 1) | (quote_bit & 1)` (64 entries).
/// Classifies a raw timestamp field into:
/// - `1`: Form F1 (`YYYY-MM-DDTHH:MM:SSZ`, stripped length 20)
/// - `2`: Form F2 (`YYYY-MM-DDTHH:MM:SS+HH:MM`, stripped length 25)
/// - `3`: Form F3 13-digit epoch milliseconds (stripped length 13)
/// - `4`: Extended ISO with space separator or `.fff` fractional seconds (`YYYY-MM-DD[T ]HH:MM:SS[.fff][Z|+HH:MM]`)
/// - `0`: Fallback to cold scalar parser.
#[repr(C, align(64))]
pub struct TsWidthQuoteLut {
    pub kind: [u8; 64],
    pub strip_off: [u8; 64],
    pub strip_len: [u8; 64],
}

pub const TS_WQ_LUT: TsWidthQuoteLut = {
    let mut kind = [0u8; 64];
    let mut strip_off = [0u8; 64];
    let mut strip_len = [0u8; 64];

    let mut raw_w = 0usize;
    while raw_w < 32 {
        let mut q = 0usize;
        while q < 2 {
            let idx = (raw_w << 1) | q;
            let qb = q * 2;
            if raw_w > qb {
                let slen = raw_w - qb;
                strip_off[idx] = q as u8;
                strip_len[idx] = slen as u8;
                kind[idx] = match slen {
                    20 => 1, // F1
                    25 => 2, // F2
                    13 => 3, // F3 (13 digits)
                    19 | 23 | 24 | 29 => 4, // Extended ISO with optional .fff and space/T
                    _ => 0,
                };
            }
            q += 1;
        }
        raw_w += 1;
    }
    TsWidthQuoteLut {
        kind,
        strip_off,
        strip_len,
    }
};

#[inline(always)]
unsafe fn decode_b36_x2_aligned(aligned: __m128i, is_pad: __m128i) -> Option<(u64, u64)> {
    let c_dig = _mm_sub_epi8(aligned, _mm_set1_epi8(b'0' as i8));
    let ok_num = _mm_andnot_si128(
        _mm_cmpgt_epi8(_mm_setzero_si128(), c_dig),
        _mm_cmpgt_epi8(_mm_set1_epi8(10), c_dig),
    );
    let c_let = _mm_sub_epi8(aligned, _mm_set1_epi8(b'a' as i8));
    let ok_let = _mm_andnot_si128(
        _mm_cmpgt_epi8(_mm_setzero_si128(), c_let),
        _mm_cmpgt_epi8(_mm_set1_epi8(26), c_let),
    );

    let ok_all = _mm_or_si128(_mm_or_si128(ok_num, ok_let), is_pad);
    if _mm_movemask_epi8(ok_all) != 0xFFFF {
        return None;
    }

    let val_let = _mm_add_epi8(c_let, _mm_set1_epi8(10));
    let digits = _mm_andnot_si128(is_pad, _mm_blendv_epi8(c_dig, val_let, ok_let));

    let p16 = _mm_maddubs_epi16(digits, _mm_set1_epi16((1i16 << 8) | 36));
    let p32 = _mm_madd_epi16(p16, _mm_set1_epi32((1i32 << 16) | 1296));

    let q0_64 = _mm_cvtsi128_si64(p32) as u64;
    let q1_64 = _mm_extract_epi64(p32, 1) as u64;
    let v0 = (q0_64 & 0xFFFF_FFFF) * 1_679_616 + (q0_64 >> 32);
    let v1 = (q1_64 & 0xFFFF_FFFF) * 1_679_616 + (q1_64 >> 32);
    ((v0 <= MASK40) & (v1 <= MASK40)).then_some((v0, v1))
}

const UNQUOTED_ID_SHUF_LO: [u64; 10] = {
    let mut t = [0u64; 10];
    let mut w = 2usize;
    while w <= 9 {
        let dl = w - 1;
        let pad = 8 - dl;
        let mut s = [-1i8; 8];
        let mut j = 0;
        while j < dl {
            s[pad + j] = j as i8;
            j += 1;
        }
        t[w] = u64::from_le_bytes([
            s[0] as u8, s[1] as u8, s[2] as u8, s[3] as u8,
            s[4] as u8, s[5] as u8, s[6] as u8, s[7] as u8,
        ]);
        w += 1;
    }
    t
};

const UNQUOTED_ID_SHUF_HI: [u64; 10] = {
    let mut t = [0u64; 10];
    let mut w = 2usize;
    while w <= 9 {
        let dl = w - 1;
        let pad = 8 - dl;
        let mut s = [-1i8; 8];
        let mut j = 0;
        while j < dl {
            s[pad + j] = (8 + j) as i8;
            j += 1;
        }
        t[w] = u64::from_le_bytes([
            s[0] as u8, s[1] as u8, s[2] as u8, s[3] as u8,
            s[4] as u8, s[5] as u8, s[6] as u8, s[7] as u8,
        ]);
        w += 1;
    }
    t
};

/// Fast branchless SIMD dual-ID base36 decoder for unquoted canonical IDs of width `2..=9`.
/// Both `p0` and `p1` must be readable for 16 bytes.
#[inline(always)]
pub unsafe fn decode_id_pair_unquoted(
    p0: *const u8,
    raw_w0: usize,
    pref0: u8,
    p1: *const u8,
    raw_w1: usize,
    pref1: u8,
) -> Option<(u64, u64)> {
    if raw_w0.wrapping_sub(2) > 7 || raw_w1.wrapping_sub(2) > 7 {
        return None;
    }
    let raw0 = (p0.add(1) as *const u64).read_unaligned();
    let raw1 = (p1.add(1) as *const u64).read_unaligned();
    if (*p0 != pref0)
        | (*p1 != pref1)
        | ((raw_w0 > 2) & ((raw0 as u8) == b'0'))
        | ((raw_w1 > 2) & ((raw1 as u8) == b'0'))
    {
        return None;
    }
    let v = _mm_set_epi64x(raw1 as i64, raw0 as i64);
    let s0 = *UNQUOTED_ID_SHUF_LO.get_unchecked(raw_w0);
    let s1 = *UNQUOTED_ID_SHUF_HI.get_unchecked(raw_w1);
    let shuf = _mm_set_epi64x(s1 as i64, s0 as i64);
    let is_pad = _mm_cmpgt_epi8(_mm_setzero_si128(), shuf);
    let aligned = _mm_shuffle_epi8(v, shuf);
    decode_b36_x2_aligned(aligned, is_pad)
}

/// Fast SIMD dual-ID base36 decoder using a branchless unquoted path (`2..=9` bytes)
/// and `ID_SHUFFLE_LUT` fallback for quoted fields.
/// Takes two raw field pointers `(p0, raw_w0, q0)` and `(p1, raw_w1, q1)` (either stripped or unstripped;
/// when already stripped by `csv::Reader`, `q0 = 0, q1 = 0`, or when raw, `q0, q1 in 0..=1`) and expected prefixes `(pref0, pref1)`.
/// Both `p0` and `p1` must be readable for 16 bytes.
/// Returns `Some((val0, val1))` if both are canonical base36 IDs `< 2^40`, or `None` if either needs fallback.
#[inline(always)]
pub unsafe fn decode_id_pair_wq(
    p0: *const u8,
    raw_w0: usize,
    q0: usize,
    pref0: u8,
    p1: *const u8,
    raw_w1: usize,
    q1: usize,
    pref1: u8,
) -> Option<(u64, u64)> {
    if q0 == 0 && q1 == 0 {
        return decode_id_pair_unquoted(p0, raw_w0, pref0, p1, raw_w1, pref1);
    }

    let idx0 = (raw_w0.min(15) << 1) | (q0 & 1);
    let idx1 = (raw_w1.min(15) << 1) | (q1 & 1);
    let dl0 = *ID_SHUFFLE_LUT.dlen.get_unchecked(idx0);
    let dl1 = *ID_SHUFFLE_LUT.dlen.get_unchecked(idx1);
    if dl0 == 0 || dl1 == 0 {
        return None;
    }

    let poff0 = *ID_SHUFFLE_LUT.prefix_off.get_unchecked(idx0) as usize;
    let poff1 = *ID_SHUFFLE_LUT.prefix_off.get_unchecked(idx1) as usize;
    let foff0 = *ID_SHUFFLE_LUT.first_digit_off.get_unchecked(idx0) as usize;
    let foff1 = *ID_SHUFFLE_LUT.first_digit_off.get_unchecked(idx1) as usize;

    // Check quote symmetry if q0 or q1 is set, prefix byte, and no leading '0' when dlen > 1
    if (q0 != 0 && (*p0 != b'"' || *p0.add(raw_w0 - 1) != b'"'))
        || (q1 != 0 && (*p1 != b'"' || *p1.add(raw_w1 - 1) != b'"'))
    {
        return None;
    }
    let c0_pref = *p0.add(poff0);
    let c1_pref = *p1.add(poff1);
    let d0_first = *p0.add(foff0);
    let d1_first = *p1.add(foff1);
    if (c0_pref != pref0)
        | (c1_pref != pref1)
        | ((dl0 > 1) & (d0_first == b'0'))
        | ((dl1 > 1) & (d1_first == b'0'))
    {
        return None;
    }

    // Load 8 bytes starting at `foff` (the first base36 digit) into each 64-bit half.
    let raw0 = (p0.add(foff0) as *const u64).read_unaligned();
    let raw1 = (p1.add(foff1) as *const u64).read_unaligned();
    let v = _mm_set_epi64x(raw1 as i64, raw0 as i64);

    let s0_raw = (ID_SHUFFLE_LUT.shuf.as_ptr().add((dl0 as usize + 1) << 1) as *const u64).read_unaligned();
    let s1_raw = (ID_SHUFFLE_LUT.shuf.as_ptr().add((dl1 as usize + 1) << 1) as *const u64).read_unaligned();
    let shuf_base = _mm_set_epi64x(s1_raw as i64, s0_raw as i64);
    let is_pad = _mm_cmpgt_epi8(_mm_setzero_si128(), shuf_base);
    let adj = _mm_set_epi64x(0x0707_0707_0707_0707, -1i64);
    let shuf = _mm_or_si128(is_pad, _mm_add_epi8(shuf_base, adj));
    let aligned = _mm_shuffle_epi8(v, shuf);
    decode_b36_x2_aligned(aligned, is_pad)
}

/// Table-driven SIMD decimal parser for `c_<N>` campaigns, `p<N>` products, or `price_cents`
/// using `_mm_shuffle_epi8` + `_mm_maddubs_epi16` + `_mm_madd_epi16`.
#[inline(always)]
pub unsafe fn decode_dec_wq(
    p: *const u8,
    raw_w: usize,
    q: usize,
    lut: &DecFieldLut,
    allow_leading_zero: bool,
) -> Option<u32> {
    let idx = (raw_w.min(15) << 1) | (q & 1);
    let dl = *lut.dlen.get_unchecked(idx);
    if dl == 0 {
        return None;
    }
    if q != 0 && (*p != b'"' || *p.add(raw_w - 1) != b'"') {
        return None;
    }
    let foff = *lut.first_digit_off.get_unchecked(idx) as usize;
    if !allow_leading_zero && *p.add(foff) == b'0' {
        return None;
    }
    let v = _mm_loadu_si128(p as *const __m128i);
    let shuf_bytes = *lut.shuf.get_unchecked(idx);
    let shuf = _mm_set_epi64x(
        -1,
        u64::from_le_bytes(std::mem::transmute::<[i8; 8], [u8; 8]>(shuf_bytes)) as i64,
    );
    let aligned = _mm_shuffle_epi8(v, shuf);
    let is_pad = _mm_cmpgt_epi8(_mm_setzero_si128(), shuf);
    let c_dig = _mm_sub_epi8(aligned, _mm_set1_epi8(b'0' as i8));
    let over = _mm_andnot_si128(is_pad, _mm_subs_epu8(c_dig, _mm_set1_epi8(9)));
    if _mm_testz_si128(over, over) == 0 {
        return None;
    }
    let digits = _mm_andnot_si128(is_pad, c_dig);
    let p16 = _mm_maddubs_epi16(digits, _mm_set1_epi16((1i16 << 8) | 10));
    let p32 = _mm_madd_epi16(p16, _mm_set1_epi32((1i32 << 16) | 100));
    let hi = _mm_cvtsi128_si32(p32) as u32;
    let lo = _mm_extract_epi32(p32, 1) as u32;
    Some(hi * 10_000 + lo)
}

/// Fast campaign lookup (`c_<N>`) with zero-overhead unquoted specialization (`q == 0`)
/// and optional surrounding quote support via `CAMP_SHUFFLE_LUT`.
#[inline(always)]
pub unsafe fn parse_campaign_wq(
    p: *const u8,
    raw_w: usize,
    q: usize,
    direct_camp: &[u16],
    fallback_map: &dims::KeyMap,
) -> Option<u32> {
    if q == 0 {
        return if !direct_camp.is_empty() {
            dims::lookup_camp_direct(direct_camp, p, raw_w)
        } else {
            fallback_map.get(p, raw_w)
        };
    }
    if !direct_camp.is_empty() {
        let poff = q & 1;
        if raw_w >= 3 + q * 2 && *p.add(poff) == b'c' && *p.add(poff + 1) == b'_' {
            if let Some(n) = decode_dec_wq(p, raw_w, q, &CAMP_SHUFFLE_LUT, false) {
                if let Some(&v) = direct_camp.get(n as usize) {
                    return if v == u16::MAX { None } else { Some(v as u32) };
                }
                return None;
            }
        }
    }
    if raw_w < 2 || *p != b'"' || *p.add(raw_w - 1) != b'"' {
        return None;
    }
    let sp = p.add(q);
    let slen = raw_w.saturating_sub(q * 2);
    if std::slice::from_raw_parts(sp, slen).contains(&b'"') {
        return None;
    }
    if !direct_camp.is_empty() {
        dims::lookup_camp_direct(direct_camp, sp, slen)
    } else {
        fallback_map.get(sp, slen)
    }
}

/// Fast table-driven product category lookup (`p<N>`) supporting optional surrounding quotes.
#[inline(always)]
pub unsafe fn parse_product_cat_wq(
    p: *const u8,
    raw_w: usize,
    q: usize,
    direct_prod_cat: &[u8],
    unknown_category: u8,
    products: &dims::KeyMap,
    product_category: &[u8],
) -> Option<u8> {
    if q != 0 && (raw_w < 2 || *p != b'"' || *p.add(raw_w - 1) != b'"') {
        return None;
    }
    let slen = raw_w.saturating_sub(q * 2);
    if slen == 0 {
        return None;
    }
    if !direct_prod_cat.is_empty() {
        let poff = q & 1;
        if *p.add(poff) == b'p' {
            if let Some(n) = decode_dec_wq(p, raw_w, q, &PROD_SHUFFLE_LUT, false) {
                return Some(if (n as usize) < direct_prod_cat.len() {
                    *direct_prod_cat.get_unchecked(n as usize)
                } else {
                    unknown_category
                });
            }
        }
    }
    let sp = p.add(q);
    if std::slice::from_raw_parts(sp, slen).contains(&b'"') {
        return None;
    }
    if !direct_prod_cat.is_empty() {
        Some(dims::lookup_prod_direct(direct_prod_cat, unknown_category, sp, slen))
    } else {
        Some(match products.get(sp, slen) {
            Some(i) => *product_category.get_unchecked(i as usize),
            None => unknown_category,
        })
    }
}

/// Parses `"HH:MM:SS"` at `p + 11..19` via a single 8-byte SWAR load, returning seconds in day (`0..86_400`).
#[inline(always)]
unsafe fn parse_swar_hms(p: *const u8) -> Option<i32> {
    let y = (p.add(11) as *const u64).read_unaligned() ^ 0x3030_3A30_303A_3030;
    if (y | y.wrapping_add(0x0606_0F06_060F_0606)) & 0xF0F0_F0F0_F0F0_F0F0 != 0 {
        return None;
    }
    let pairs = y.wrapping_mul(10).wrapping_add(y >> 8);
    let h = (pairs & 0xFF) as i32;
    let m = ((pairs >> 24) & 0xFF) as i32;
    let s = ((pairs >> 48) & 0xFF) as i32;
    if (h > 23) | (m > 59) | (s > 59) {
        return None;
    }
    Some(h * 3600 + m * 60 + s)
}

/// Parses `"HH:MM:SSZ"` at `p + 11..20` via SWAR, returning seconds in day (`0..86_400`).
#[inline(always)]
unsafe fn parse_f1_swar_hms(p: *const u8) -> Option<i32> {
    if *p.add(19) != b'Z' {
        return None;
    }
    parse_swar_hms(p)
}

/// Parses 6-byte timezone offset `(+|-)HH:MM` at `p + 19..25`, returning signed seconds adjustment
/// (`-(oh * 3600 + om * 60)` for `'+'`, `+(oh * 3600 + om * 60)` for `'-'`).
#[inline(always)]
unsafe fn parse_swar_tz_offset(p19: *const u8) -> Option<i32> {
    let raw = (p19 as *const u64).read_unaligned();
    let sign = (raw & 0xFF) as u8;
    if (sign.wrapping_sub(b'+') & 0xFD) != 0 {
        return None;
    }
    let d = (raw >> 8) & 0x00FF_FFFF_FFFF ^ 0x0000_0030_303A_3030;
    if ((d | d.wrapping_add(0x0000_0006_060F_0606)) & 0x0000_00F0_F0F0_F0F0) != 0 {
        return None;
    }
    let oh = ((d & 0xF) * 10 + ((d >> 8) & 0xF)) as i32;
    let om = (((d >> 24) & 0xF) * 10 + ((d >> 32) & 0xF)) as i32;
    if (oh > 23) | (om > 59) {
        return None;
    }
    let off = oh * 3600 + om * 60;
    Some(if sign == b'+' { -off } else { off })
}

/// Parses 13-digit epoch milliseconds at `p[0..13]` without reading before `p`.
#[inline(always)]
unsafe fn parse_f3_13(p: *const u8) -> Option<u64> {
    let raw = _mm_slli_si128::<3>(_mm_loadu_si128(p as *const __m128i));
    let m = _mm_setr_epi8(0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let d = _mm_sub_epi8(raw, _mm_set1_epi8(b'0' as i8));
    let over = _mm_and_si128(_mm_subs_epu8(d, _mm_set1_epi8(9)), m);
    if _mm_testz_si128(over, over) == 0 {
        return None;
    }
    let d = _mm_and_si128(d, m);
    let t = _mm_maddubs_epi16(
        d,
        _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1),
    );
    let t = _mm_madd_epi16(t, _mm_setr_epi16(100, 1, 100, 1, 100, 1, 100, 1));
    let t = _mm_packus_epi32(t, t);
    let t = _mm_madd_epi16(t, _mm_setr_epi16(10000, 1, 10000, 1, 10000, 1, 10000, 1));
    let q = _mm_cvtsi128_si64(t) as u64;
    Some((q & 0xFFFF_FFFF) * 100_000_000 + (q >> 32))
}

#[inline]
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + if m <= 2 { 1 } else { 0 };
    (y, m as u32, d as u32)
}

/// Compact 512-byte (8 L1D cache lines) branchless date lookup table for up to 2 calendar
/// `YYYY-MM-` prefixes across shuffled multi-week datasets.
/// Unpopulated/invalid day slots hold `i32::MIN`.
#[repr(C, align(64))]
pub struct FastDateLut {
    pub day_sec: [i32; 128],
    pub ym0: u64,
    pub ym1: u64,
}

impl FastDateLut {
    pub fn new(window_start: i64, window_end: i64, base_s: i64) -> Self {
        let mut lut = Self {
            day_sec: [i32::MIN; 128],
            ym0: u64::MAX,
            ym1: u64::MAX,
        };
        let mut insert_range = |first_day: i64, last_day: i64, allow_new_month: bool| {
            for day in first_day..=last_day {
                let (y, m, d) = civil_from_days(day);
                if !(0..=9999).contains(&y) {
                    continue;
                }
                let rel = day * 86_400 - base_s;
                if rel < i32::MIN as i64 + 1 || rel + 86_399 >= i32::MAX as i64 {
                    continue;
                }
                let y = y as u32;
                let ym = u64::from_le_bytes([
                    b'0' + (y / 1000) as u8,
                    b'0' + ((y / 100) % 10) as u8,
                    b'0' + ((y / 10) % 10) as u8,
                    b'0' + (y % 10) as u8,
                    b'-',
                    b'0' + (m / 10) as u8,
                    b'0' + (m % 10) as u8,
                    b'-',
                ]);
                let half = if ym == lut.ym0 {
                    0usize
                } else if ym == lut.ym1 {
                    64usize
                } else if allow_new_month && lut.ym0 == u64::MAX {
                    lut.ym0 = ym;
                    0usize
                } else if allow_new_month && lut.ym1 == u64::MAX {
                    lut.ym1 = ym;
                    64usize
                } else {
                    continue;
                };
                let day_idx = (((d / 10) << 4) | (d % 10)) as usize;
                lut.day_sec[half | day_idx] = rel as i32;
            }
        };

        // Populate the active click+event window first so those months always claim ym0/ym1.
        let p_first = (window_start - WEEK_MS).div_euclid(DAY_MS) - 1;
        let p_last = (window_end - 1).div_euclid(DAY_MS) + 2;
        insert_range(p_first, p_last, true);

        let e_first = window_start.div_euclid(DAY_MS) - 9;
        let e_last = (window_end - 1).div_euclid(DAY_MS) + 9;
        insert_range(e_first, e_last, true);
        lut
    }

    /// Branchless lookup of 11-byte `YYYY-MM-DDT` prefix at `p`, returning `sec_rel` or `None`.
    /// Requires at least 12 readable bytes at `p`.
    #[inline(always)]
    pub unsafe fn lookup_ymdt(&self, p: *const u8) -> Option<i32> {
        let ym8 = (p as *const u64).read_unaligned();
        let m0 = (ym8 == self.ym0) as usize;
        let m1 = (ym8 == self.ym1) as usize;
        let dt = (p.add(8) as *const u32).read_unaligned() & 0x00FF_FFFF;
        let z = dt ^ 0x0054_3030;
        let bad_dt = (z | z.wrapping_add(0x0000_060C)) & 0x00FF_F0F0;
        if ((m0 | m1) == 0) | (bad_dt != 0) {
            return None;
        }
        let day_idx = (((z & 3) << 4) | ((z >> 8) & 0xF)) as usize;
        let slot = (m1 << 6) | day_idx;
        let sec_rel = *self.day_sec.get_unchecked(slot);
        (sec_rel != i32::MIN).then_some(sec_rel)
    }
}

/// Parses 20-byte `YYYY-MM-DDTHH:MM:SSZ` via `DateLut` and SWAR `HH:MM:SS`.
#[inline(always)]
unsafe fn parse_f1_swar_lut(p: *const u8, lut: &DateLut) -> Option<i32> {
    let slot = _pext_u32((p.add(6) as *const u32).read_unaligned(), 0x0F03_0001) as usize;
    let entry = lut.0.get_unchecked(slot);
    let d8 = (p as *const u64).read_unaligned();
    let d4 = (p.add(8) as *const u32).read_unaligned() & 0x00FF_FFFF;
    let e8 = (entry.ymdt.as_ptr() as *const u64).read_unaligned();
    let e4 = (entry.ymdt.as_ptr().add(8) as *const u32).read_unaligned();
    if ((d8 ^ e8) | ((d4 ^ e4) as u64)) != 0 {
        return None;
    }
    let tod = parse_f1_swar_hms(p)?;
    Some(entry.sec_rel + tod)
}

#[cold]
#[inline(never)]
unsafe fn parse_f1_rel_fallback(
    p: *const u8,
    len: usize,
    lut: &DateLut,
    dc: &mut DateCache,
    base_s: i64,
) -> i32 {
    ts::parse_f1_rel(p, len, lut, dc, base_s)
}

#[cold]
#[inline(never)]
unsafe fn parse_ts_rel_fallback(
    p: *const u8,
    len: usize,
    lut: &DateLut,
    dc: &mut DateCache,
    base_ms: i64,
) -> i64 {
    ts::parse_ts_rel(p, len, lut, dc, base_ms)
}

/// Parses 20-byte `YYYY-MM-DDTHH:MM:SSZ` via `FastDateLut` and SWAR `HH:MM:SS`.
#[inline(always)]
unsafe fn parse_f1_fast_lut(p: *const u8, fast_lut: &FastDateLut) -> Option<i32> {
    let base_sec_rel = fast_lut.lookup_ymdt(p)?;
    let tod = parse_f1_swar_hms(p)?;
    Some(base_sec_rel + tod)
}

/// Unified `F1 (20) / F2 (25) / F3 (13)` timestamp parser using `FastDateLut`
/// for both `F1` and `F2`, and direct `parse_f3_13` for 13-digit epoch ms, with cold `ts::parse_ts_rel` fallback.
#[inline(always)]
unsafe fn parse_ts_fast_lut(
    p: *const u8,
    len: usize,
    fast_lut: &FastDateLut,
    lut: &DateLut,
    dc: &mut DateCache,
    base_ms: i64,
) -> i64 {
    if (len == 20) | (len == 25) {
        if let (Some(base_sec_rel), Some(tod)) = (fast_lut.lookup_ymdt(p), parse_swar_hms(p)) {
            if len == 20 {
                if *p.add(19) == b'Z' {
                    return (base_sec_rel + tod) as i64 * 1000;
                }
            } else if let Some(off) = parse_swar_tz_offset(p.add(19)) {
                return (base_sec_rel + tod + off) as i64 * 1000;
            }
        }
    } else if len == 13 {
        if let Some(v) = parse_f3_13(p) {
            return v as i64 - base_ms;
        }
    }
    parse_ts_rel_fallback(p, len, lut, dc, base_ms)
}

/// Fast timestamp parser with unquoted `F1/F2/F3` fast paths,
/// falling back to `TS_WQ_LUT` for quoted/other widths.
/// Returns milliseconds relative to `base_ms`, or `BAD` if invalid.
#[inline(always)]
pub unsafe fn parse_ts_wq_rel(
    p: *const u8,
    raw_w: usize,
    q: usize,
    lut: &DateLut,
    dc: &mut DateCache,
    base_ms: i64,
) -> i64 {
    if q == 0 {
        if raw_w == 20 {
            if let Some(s) = parse_f1_swar_lut(p, lut) {
                return s as i64 * 1000;
            }
            return ts::parse_ts_rel(p, 20, lut, dc, base_ms);
        }
        if raw_w == 13 {
            if let Some(v) = parse_f3_13(p) {
                return v as i64 - base_ms;
            }
        }
        return ts::parse_ts_rel(p, raw_w, lut, dc, base_ms);
    }
    if raw_w < 2 || *p != b'"' || *p.add(raw_w - 1) != b'"' {
        return BAD;
    }
    let idx = (raw_w.min(31) << 1) | (q & 1);
    let sp = p.add(*TS_WQ_LUT.strip_off.get_unchecked(idx) as usize);
    let slen = raw_w.saturating_sub(q * 2);
    ts::parse_ts_rel(sp, slen, lut, dc, base_ms)
}

/// Fast `ingest_ts` parser with direct unquoted `len == 20 || len == 24` fast path,
/// returning seconds relative to `base_s` (or `i32::MAX` if invalid).
#[inline(always)]
pub unsafe fn parse_ingest_wq_rel(
    p: *const u8,
    raw_w: usize,
    q: usize,
    lut: &DateLut,
    dc: &mut DateCache,
    base_s: i64,
) -> i32 {
    if q == 0 {
        if raw_w == 20 {
            if let Some(s) = parse_f1_swar_lut(p, lut) {
                return s;
            }
            return ts::parse_f1_rel(p, 20, lut, dc, base_s);
        }
        if raw_w == 24 {
            return i32::MAX;
        }
        return ts::parse_f1_rel(p, raw_w, lut, dc, base_s);
    }
    if raw_w < 2 || *p != b'"' || *p.add(raw_w - 1) != b'"' {
        return i32::MAX;
    }
    let sp = p.add(q);
    let slen = raw_w.saturating_sub(q * 2);
    ts::parse_f1_rel(sp, slen, lut, dc, base_s)
}

/// 64-bit SWAR decimal parser for 1..=8 digit `price_cents` (`1..=10_000_000`, leading zeros allowed).
#[inline(always)]
unsafe fn swar_dec_price(p: *const u8, len: usize) -> Option<u32> {
    if len == 0 || len > 8 {
        return None;
    }
    let d_raw = (p as *const u64).read_unaligned();
    let x = (d_raw ^ 0x3030_3030_3030_3030) << ((8 - len) * 8);
    let bad = (x | x.wrapping_add(0x0606_0606_0606_0606)) & 0xF0F0_F0F0_F0F0_F0F0;
    if bad != 0 {
        return None;
    }
    let y = (x.wrapping_mul(10).wrapping_add(x >> 8)) & 0x00FF_00FF_00FF_00FF;
    let z = (y.wrapping_mul(100).wrapping_add(y >> 16)) & 0x0000_FFFF_0000_FFFF;
    let v = (z.wrapping_mul(10000).wrapping_add(z >> 32)) as u32;
    ((1..=10_000_000).contains(&v)).then_some(v)
}

#[inline(always)]
pub unsafe fn parse_ev_tail_unquoted(
    p4: *const u8,
    len4: usize,
    p5: *const u8,
    len5: usize,
    p6: *const u8,
    len6: usize,
    sh: &Shared,
) -> Option<u32> {
    const VIEW: u32 = u32::from_le_bytes(*b"view");
    const CART: u32 = u32::from_le_bytes(*b"cart");
    const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");

    if len5 == 0 {
        return None;
    }
    match len4 {
        4 => {
            let ty = (p4 as *const u32).read_unaligned();
            ((ty == VIEW) | (ty == CART)).then_some(0)
        }
        8 => {
            if (p4 as *const u64).read_unaligned() != PURCHASE {
                return None;
            }
            let price = match swar_dec_price(p6, len6) {
                Some(v) => v,
                None => match ts::digits(p6, len6) {
                    Some(v) if (1..=10_000_000).contains(&v) => v as u32,
                    _ => return None,
                },
            };
            let cat = sh.product_cat(p5, len5);
            Some(((cat as u32) << 24) | price)
        }
        _ => None,
    }
}

/// Fast event_type + product_id + price_cents validator with **97% View/Cart Fast-Skip**:
/// - Specializes `q4 == 0 && q5 == 0 && q6 == 0` (outer quotes already stripped by `csv::Reader`):
///   `view`/`cart` only checks `len4 == 4`, `len5 != 0`, and `ty == VIEW || ty == CART`.
/// - Only decodes `product_id` and `price_cents` (`1..=10_000_000`, leading zeros allowed) when
///   `event_type` is `"purchase"`.
/// Returns `Some(payload)` (`0` for view/cart, `category << 24 | price` for purchase) or `None` if invalid.
#[inline(always)]
pub unsafe fn parse_ev_tail_wq(
    p4: *const u8,
    raw_w4: usize,
    q4: usize,
    p5: *const u8,
    raw_w5: usize,
    q5: usize,
    p6: *const u8,
    raw_w6: usize,
    q6: usize,
    sh: &Shared,
) -> Option<u32> {
    const VIEW: u32 = u32::from_le_bytes(*b"view");
    const CART: u32 = u32::from_le_bytes(*b"cart");
    const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");

    if q4 == 0 && q5 == 0 && q6 == 0 {
        return parse_ev_tail_unquoted(p4, raw_w4, p5, raw_w5, p6, raw_w6, sh);
    }

    if (q4 != 0 && (raw_w4 < 2 || *p4 != b'"' || *p4.add(raw_w4 - 1) != b'"'))
        || (q5 != 0 && (raw_w5 < 2 || *p5 != b'"' || *p5.add(raw_w5 - 1) != b'"'))
        || (q6 != 0 && (raw_w6 < 2 || *p6 != b'"' || *p6.add(raw_w6 - 1) != b'"'))
    {
        return None;
    }
    let len4 = raw_w4.saturating_sub(q4 * 2);
    let len5 = raw_w5.saturating_sub(q5 * 2);
    if len5 == 0 {
        return None;
    }
    let sp4 = p4.add(q4);
    match len4 {
        4 => {
            let ty = (sp4 as *const u32).read_unaligned();
            if ty == VIEW || ty == CART {
                return Some(0);
            }
            None
        }
        8 => {
            if (sp4 as *const u64).read_unaligned() != PURCHASE {
                return None;
            }
            let price = match decode_dec_wq(p6, raw_w6, q6, &PRICE_SHUFFLE_LUT, true) {
                Some(v) if (1..=10_000_000).contains(&v) => v,
                _ => {
                    let sp6 = p6.add(q6);
                    let len6 = raw_w6.saturating_sub(q6 * 2);
                    match ts::digits(sp6, len6) {
                        Some(v) if (1..=10_000_000).contains(&v) => v as u32,
                        _ => return None,
                    }
                }
            };
            let cat = parse_product_cat_wq(
                p5,
                raw_w5,
                q5,
                &sh.direct_prod_cat,
                sh.unknown_category,
                &sh.products,
                &sh.product_category,
            )?;
            Some(((cat as u32) << 24) | price)
        }
        _ => None,
    }
}

/// Exact cold scalar fallback for a single event row given stripped field pointers and lengths.
#[cold]
#[inline(never)]
pub unsafe fn parse_event_row_cold_fields(
    sh: &Shared,
    dates: &mut DateCache,
    ptrs: [*const u8; 7],
    lens: [usize; 7],
) -> Option<FallbackEvRec> {
    const VIEW: u32 = u32::from_le_bytes(*b"view");
    const CART: u32 = u32::from_le_bytes(*b"cart");
    const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");

    if lens[0] == 0 {
        return None;
    }
    let (event_tok, user_tok) = tok_pair(ptrs[0], lens[0], ptrs[1], lens[1]);
    let base_s = sh.base_ms.div_euclid(1000);
    let event_win_ms = (sh.window_end - sh.window_start) as u64;

    let mut rec = FallbackEvRec {
        event_tok,
        user_tok: 0,
        eid_b36: crate::unscramble::decode_b36_ptr(ptrs[0], lens[0], b'e').unwrap_or(u64::MAX),
        uid_b36: if lens[1] > 0 {
            crate::unscramble::decode_b36_ptr(ptrs[1], lens[1], b'u').unwrap_or(u64::MAX)
        } else {
            u64::MAX
        },
        ingest: i32::MAX,
        ts: 0,
        payload: 0,
        valid: false,
    };

    'valid: {
        let ts_u32 = if sh.use_rel {
            let ingest_s_rel = ts::parse_f1_rel(ptrs[3], lens[3], &sh.date_lut, dates, base_s);
            rec.ingest = ingest_s_rel;
            if lens[1] == 0 || lens[5] == 0 || ingest_s_rel == i32::MAX {
                break 'valid;
            }
            let ts_rel = ts::parse_ts_rel(ptrs[2], lens[2], &sh.date_lut, dates, sh.base_ms);
            let win_bad = (ts_rel.wrapping_sub(WEEK_MS) as u64) >= event_win_ms;
            let late_bad =
                (ingest_s_rel as i64).wrapping_mul(1000).wrapping_sub(ts_rel) > DAY_MS;
            if win_bad | late_bad {
                break 'valid;
            }
            ts_rel as u32
        } else {
            let ingest_s = ts::parse_f1(ptrs[3], lens[3], dates);
            if ingest_s != BAD {
                rec.ingest =
                    (ingest_s - base_s).clamp(i32::MIN as i64, i32::MAX as i64 - 1) as i32;
            }
            if lens[1] == 0 || lens[5] == 0 || ingest_s == BAD {
                break 'valid;
            }
            let t = ts::parse_ts(ptrs[2], lens[2], dates);
            if t < sh.window_start || t >= sh.window_end || ingest_s * 1000 - t > DAY_MS {
                break 'valid;
            }
            (t - sh.base_ms) as u32
        };

        match lens[4] {
            4 => {
                let ty = (ptrs[4] as *const u32).read_unaligned();
                if ty != VIEW && ty != CART {
                    break 'valid;
                }
            }
            8 => {
                if (ptrs[4] as *const u64).read_unaligned() != PURCHASE {
                    break 'valid;
                }
                let price = match ts::digits(ptrs[6], lens[6]) {
                    Some(v) if (1..=10_000_000).contains(&v) => v as u32,
                    _ => break 'valid,
                };
                let category = sh.product_cat(ptrs[5], lens[5]);
                rec.payload = (category as u32) << 24 | price;
            }
            _ => break 'valid,
        }
        rec.user_tok = user_tok;
        rec.ts = ts_u32;
        rec.valid = true;
    }
    Some(rec)
}

/// Exact cold scalar fallback for a single event row (already stripped of outer quotes by `csv::Reader`
/// or raw fields). Matches `main.rs::parse_events_impl` rule-for-rule.
#[cold]
#[inline(never)]
pub unsafe fn parse_event_row_cold(
    sh: &Shared,
    dates: &mut DateCache,
    buf: *const u8,
    s: &[u32; 8],
    e: &[u32; 8],
) -> Option<FallbackEvRec> {
    let ptrs = [
        buf.add(s[0] as usize),
        buf.add(s[1] as usize),
        buf.add(s[2] as usize),
        buf.add(s[3] as usize),
        buf.add(s[4] as usize),
        buf.add(s[5] as usize),
        buf.add(s[6] as usize),
    ];
    let lens = [
        (e[0] - s[0]) as usize,
        (e[1] - s[1]) as usize,
        (e[2] - s[2]) as usize,
        (e[3] - s[3]) as usize,
        (e[4] - s[4]) as usize,
        (e[5] - s[5]) as usize,
        (e[6] - s[6]) as usize,
    ];
    parse_event_row_cold_fields(sh, dates, ptrs, lens)
}

/// Exact cold scalar fallback for a single click row given stripped field pointers and lengths.
#[cold]
#[inline(never)]
pub unsafe fn parse_click_row_cold_fields(
    sh: &Shared,
    dates: &mut DateCache,
    ptrs: [*const u8; 4],
    lens: [usize; 4],
) -> Option<FallbackCkRec> {
    if lens[0] == 0 || lens[1] == 0 {
        return None;
    }
    let click_win_ms = (sh.window_end - sh.base_ms) as u64;
    let ts_u32 = if sh.use_rel {
        let ts_rel = ts::parse_ts_rel(ptrs[3], lens[3], &sh.date_lut, dates, sh.base_ms);
        if (ts_rel as u64) >= click_win_ms {
            return None;
        }
        ts_rel as u32
    } else {
        let t = ts::parse_ts(ptrs[3], lens[3], dates);
        if t < sh.base_ms || t >= sh.window_end {
            return None;
        }
        (t - sh.base_ms) as u32
    };
    let campaign = sh.campaign_id(ptrs[2], lens[2])?;
    let (click_tok, user_tok) = tok_pair(ptrs[0], lens[0], ptrs[1], lens[1]);
    let mut is_upper_k = false;
    let cid_b36 = match crate::unscramble::decode_b36_ptr(ptrs[0], lens[0], b'k') {
        Some(v) => v,
        None => match crate::unscramble::decode_upper_b36_ptr(ptrs[0], lens[0], b'K') {
            Some(v) => {
                is_upper_k = true;
                v
            }
            None => u64::MAX,
        },
    };
    let uid_b36 = crate::unscramble::decode_b36_ptr(ptrs[1], lens[1], b'u').unwrap_or(u64::MAX);
    Some(FallbackCkRec {
        click_tok,
        user_tok,
        cid_b36,
        uid_b36,
        ts: ts_u32,
        campaign,
        is_upper_k,
    })
}

/// Exact cold scalar fallback for a single click row.
#[cold]
#[inline(never)]
pub unsafe fn parse_click_row_cold(
    sh: &Shared,
    dates: &mut DateCache,
    buf: *const u8,
    s: &[u32; 8],
    e: &[u32; 8],
) -> Option<FallbackCkRec> {
    let ptrs = [
        buf.add(s[0] as usize),
        buf.add(s[1] as usize),
        buf.add(s[2] as usize),
        buf.add(s[3] as usize),
    ];
    let lens = [
        (e[0] - s[0]) as usize,
        (e[1] - s[1]) as usize,
        (e[2] - s[2]) as usize,
        (e[3] - s[3]) as usize,
    ];
    parse_click_row_cold_fields(sh, dates, ptrs, lens)
}

/// 64-row columnar batch of fast-parsed event rows ready for vectorized AVX2/AVX-512 unscrambling
/// and classification.
#[repr(C, align(64))]
pub struct FastEvBatch {
    /// Base36-decoded 40-bit event IDs `y_e` (or `u64::MAX` if non-canonical ID).
    pub eid_b36: [u64; FAST_BATCH],
    /// Base36-decoded 40-bit user IDs `y_u` (or `u64::MAX` if non-canonical/invalid).
    pub uid_b36: [u64; FAST_BATCH],
    /// Packed `ids::tok` tokens for event_id (populated for odd/fallback rows or on demand).
    pub event_tok: [u64; FAST_BATCH],
    /// Packed `ids::tok` tokens for user_id (populated for odd/fallback rows or on demand).
    pub user_tok: [u64; FAST_BATCH],
    /// Ingest timestamp in seconds relative to `base_s` (`i32::MAX` if invalid).
    pub ingest: [i32; FAST_BATCH],
    /// Event timestamp in milliseconds relative to `base_ms`.
    pub ts: [u32; FAST_BATCH],
    /// `0` for view/cart, `(category as u32) << 24 | price_cents` for purchase.
    pub payload: [u32; FAST_BATCH],
    /// Bitmask of rows where `valid == true` and both `eid_b36` and `uid_b36` are canonical 40-bit base36.
    pub canonical_valid_mask: u64,
    /// Bitmask of rows that are valid events (whether canonical 40-bit or non-canonical fallback).
    pub valid_mask: u64,
    /// Bitmask of rows whose `event_tok`/`user_tok` slots were populated by cold fallback.
    pub tok_dirty_mask: u64,
    pub len: usize,
}

impl Default for FastEvBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl FastEvBatch {
    pub const fn new() -> Self {
        Self {
            eid_b36: [0; FAST_BATCH],
            uid_b36: [0; FAST_BATCH],
            event_tok: [0; FAST_BATCH],
            user_tok: [0; FAST_BATCH],
            ingest: [0; FAST_BATCH],
            ts: [0; FAST_BATCH],
            payload: [0; FAST_BATCH],
            canonical_valid_mask: 0,
            valid_mask: 0,
            tok_dirty_mask: 0,
            len: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        let mut m = self.tok_dirty_mask;
        while m != 0 {
            let r = m.trailing_zeros() as usize;
            m &= m - 1;
            self.event_tok[r] = 0;
            self.user_tok[r] = 0;
        }
        self.tok_dirty_mask = 0;
        self.canonical_valid_mask = 0;
        self.valid_mask = 0;
        self.len = 0;
    }
}

/// A classified normal event record (`user_idx < user_cap && alias < 14 && event_idx < event_cap`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct NormalEvRec {
    pub event_idx: u32,
    pub user_idx: u32,
    pub alias: u8,
    pub ingest: i32,
    pub ts: u32,
    pub payload: u32,
}

/// A fallback/special event record (e.g. invalid row needing DRec dedup tombstone, chain/kiosk user,
/// case variant, or revised duplicate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct FallbackEvRec {
    pub event_tok: u64,
    pub user_tok: u64,
    pub eid_b36: u64,
    pub uid_b36: u64,
    pub ingest: i32,
    pub ts: u32,
    pub payload: u32,
    pub valid: bool,
}

/// 64-row columnar batch of fast-parsed click rows.
#[repr(C, align(64))]
pub struct FastCkBatch {
    pub cid_b36: [u64; FAST_BATCH],
    pub uid_b36: [u64; FAST_BATCH],
    pub click_tok: [u64; FAST_BATCH],
    pub user_tok: [u64; FAST_BATCH],
    pub ts: [u32; FAST_BATCH],
    pub campaign: [u32; FAST_BATCH],
    pub canonical_mask: u64,
    pub upper_k_mask: u64,
    pub tok_dirty_mask: u64,
    pub len: usize,
}

impl Default for FastCkBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl FastCkBatch {
    pub const fn new() -> Self {
        Self {
            cid_b36: [0; FAST_BATCH],
            uid_b36: [0; FAST_BATCH],
            click_tok: [0; FAST_BATCH],
            user_tok: [0; FAST_BATCH],
            ts: [0; FAST_BATCH],
            campaign: [0; FAST_BATCH],
            canonical_mask: 0,
            upper_k_mask: 0,
            tok_dirty_mask: 0,
            len: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        let mut m = self.tok_dirty_mask;
        while m != 0 {
            let r = m.trailing_zeros() as usize;
            m &= m - 1;
            self.click_tok[r] = 0;
            self.user_tok[r] = 0;
        }
        self.tok_dirty_mask = 0;
        self.canonical_mask = 0;
        self.upper_k_mask = 0;
        self.len = 0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct NormalCkRec {
    pub click_idx: u32,
    pub user_idx: u32,
    pub alias: u8,
    pub cid_b36: u64,
    pub ts: u32,
    pub campaign: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct FallbackCkRec {
    pub click_tok: u64,
    pub user_tok: u64,
    pub cid_b36: u64,
    pub uid_b36: u64,
    pub ts: u32,
    pub campaign: u32,
    pub is_upper_k: bool,
}

/// 64-row columnar batch of fast-parsed link rows.
#[repr(C, align(64))]
pub struct FastLkBatch {
    pub ua_b36: [u64; FAST_BATCH],
    pub ub_b36: [u64; FAST_BATCH],
    pub ua_tok: [u64; FAST_BATCH],
    pub ub_tok: [u64; FAST_BATCH],
    pub canonical_mask: u64,
    pub tok_dirty_mask: u64,
    pub len: usize,
}

impl Default for FastLkBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl FastLkBatch {
    pub const fn new() -> Self {
        Self {
            ua_b36: [0; FAST_BATCH],
            ub_b36: [0; FAST_BATCH],
            ua_tok: [0; FAST_BATCH],
            ub_tok: [0; FAST_BATCH],
            canonical_mask: 0,
            tok_dirty_mask: 0,
            len: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        let mut m = self.tok_dirty_mask;
        while m != 0 {
            let r = m.trailing_zeros() as usize;
            m &= m - 1;
            self.ua_tok[r] = 0;
            self.ub_tok[r] = 0;
        }
        self.tok_dirty_mask = 0;
        self.canonical_mask = 0;
        self.len = 0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct NormalLkRec {
    pub user_idx: u32,
    pub alias_a: u8,
    pub alias_b: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct FallbackLkRec {
    pub ua_tok: u64,
    pub ub_tok: u64,
    pub ua_b36: u64,
    pub ub_b36: u64,
}

#[inline(always)]
fn make_fallback_ev(batch: &FastEvBatch, r: usize) -> FallbackEvRec {
    let valid = (batch.valid_mask & (1u64 << r)) != 0;
    let eid = batch.eid_b36[r];
    let uid = batch.uid_b36[r];
    let is_canon_valid = (batch.canonical_valid_mask & (1u64 << r)) != 0;
    let event_tok = if !is_canon_valid && batch.event_tok[r] != 0 {
        batch.event_tok[r]
    } else if eid <= MASK40 {
        encode_base36_token(b'e', eid)
    } else {
        0
    };
    let user_tok = if !valid {
        0
    } else if !is_canon_valid && batch.user_tok[r] != 0 {
        batch.user_tok[r]
    } else if uid <= MASK40 {
        encode_base36_token(b'u', uid)
    } else {
        0
    };
    FallbackEvRec {
        event_tok,
        user_tok,
        eid_b36: eid,
        uid_b36: uid,
        ingest: batch.ingest[r],
        ts: batch.ts[r],
        payload: batch.payload[r],
        valid,
    }
}

#[inline(always)]
fn make_fallback_ck(batch: &FastCkBatch, r: usize) -> FallbackCkRec {
    let cid = batch.cid_b36[r];
    let uid = batch.uid_b36[r];
    let is_upper_k = (batch.upper_k_mask & (1u64 << r)) != 0;
    let click_tok = if batch.click_tok[r] != 0 {
        batch.click_tok[r]
    } else if cid <= MASK40 {
        if is_upper_k {
            crate::fast_links::encode_upper_k_token(cid)
        } else {
            encode_base36_token(b'k', cid)
        }
    } else {
        0
    };
    let user_tok = if batch.user_tok[r] != 0 {
        batch.user_tok[r]
    } else if uid <= MASK40 {
        encode_base36_token(b'u', uid)
    } else {
        0
    };
    FallbackCkRec {
        click_tok,
        user_tok,
        cid_b36: cid,
        uid_b36: uid,
        ts: batch.ts[r],
        campaign: batch.campaign[r],
        is_upper_k,
    }
}

#[inline(always)]
fn make_fallback_lk(batch: &FastLkBatch, r: usize) -> FallbackLkRec {
    let ua = batch.ua_b36[r];
    let ub = batch.ub_b36[r];
    let ua_tok = if batch.ua_tok[r] != 0 {
        batch.ua_tok[r]
    } else if ua <= MASK40 {
        encode_base36_token(b'u', ua)
    } else {
        0
    };
    let ub_tok = if batch.ub_tok[r] != 0 {
        batch.ub_tok[r]
    } else if ub <= MASK40 {
        encode_base36_token(b'u', ub)
    } else {
        0
    };
    FallbackLkRec {
        ua_tok,
        ub_tok,
        ua_b36: ua,
        ub_b36: ub,
    }
}

/// Vectorized 64-row event batch classifier:
/// Unscrambles `eid_b36` and `uid_b36` 8 at a time via AVX-512DQ (when `use_avx512()` is enabled)
/// or ILP scalar `unscramble40_x8_scalar`, and partitions rows into `normal_out` and `fallback_out`.
pub fn classify_ev_batch(
    batch: &FastEvBatch,
    user_key: u64,
    event_key: u64,
    user_cap: u32,
    event_cap: u64,
    normal_out: &mut Vec<NormalEvRec>,
    fallback_out: &mut Vec<FallbackEvRec>,
) {
    if use_avx512() {
        unsafe {
            classify_ev_batch_avx512(
                batch,
                user_key,
                event_key,
                user_cap,
                event_cap,
                normal_out,
                fallback_out,
            );
        }
        return;
    }
    classify_ev_batch_avx2(
        batch,
        user_key,
        event_key,
        user_cap,
        event_cap,
        normal_out,
        fallback_out,
    );
}

#[inline(always)]
fn classify_ev_batch_with(
    batch: &FastEvBatch,
    user_key: u64,
    event_key: u64,
    user_cap: u32,
    event_cap: u64,
    normal_out: &mut Vec<NormalEvRec>,
    fallback_out: &mut Vec<FallbackEvRec>,
    mut unscramble_x8: impl FnMut([u64; 8], u64) -> [u64; 8],
) {
    let n = batch.len;
    let mut i = 0;
    while i + 8 <= n {
        let m8 = (batch.canonical_valid_mask >> i) & 0xFF;
        if m8 == 0xFF {
            let e_un = unscramble_x8(
                unsafe { *(batch.eid_b36.as_ptr().add(i) as *const [u64; 8]) },
                event_key,
            );
            let u_un = unscramble_x8(
                unsafe { *(batch.uid_b36.as_ptr().add(i) as *const [u64; 8]) },
                user_key,
            );
            for k in 0..8 {
                let r = i + k;
                let e_idx = e_un[k];
                let u_val = u_un[k];
                let u_idx = u_val >> 9;
                let alias = u_val & 511;
                if u_idx < (user_cap as u64) && alias < 14 && e_idx < event_cap {
                    normal_out.push(NormalEvRec {
                        event_idx: e_idx as u32,
                        user_idx: u_idx as u32,
                        alias: alias as u8,
                        ingest: batch.ingest[r],
                        ts: batch.ts[r],
                        payload: batch.payload[r],
                    });
                } else {
                    fallback_out.push(make_fallback_ev(batch, r));
                }
            }
            i += 8;
            continue;
        }
        for k in 0..8 {
            let r = i + k;
            if (batch.canonical_valid_mask & (1u64 << r)) != 0 {
                let e_idx = unscramble40(batch.eid_b36[r], event_key);
                let u_val = unscramble40(batch.uid_b36[r], user_key);
                let u_idx = u_val >> 9;
                let alias = u_val & 511;
                if u_idx < (user_cap as u64) && alias < 14 && e_idx < event_cap {
                    normal_out.push(NormalEvRec {
                        event_idx: e_idx as u32,
                        user_idx: u_idx as u32,
                        alias: alias as u8,
                        ingest: batch.ingest[r],
                        ts: batch.ts[r],
                        payload: batch.payload[r],
                    });
                    continue;
                }
            }
            fallback_out.push(make_fallback_ev(batch, r));
        }
        i += 8;
    }
    while i < n {
        let r = i;
        if (batch.canonical_valid_mask & (1u64 << r)) != 0 {
            let e_idx = unscramble40(batch.eid_b36[r], event_key);
            let u_val = unscramble40(batch.uid_b36[r], user_key);
            let u_idx = u_val >> 9;
            let alias = u_val & 511;
            if u_idx < (user_cap as u64) && alias < 14 && e_idx < event_cap {
                normal_out.push(NormalEvRec {
                    event_idx: e_idx as u32,
                    user_idx: u_idx as u32,
                    alias: alias as u8,
                    ingest: batch.ingest[r],
                    ts: batch.ts[r],
                    payload: batch.payload[r],
                });
                i += 1;
                continue;
            }
        }
        fallback_out.push(make_fallback_ev(batch, r));
        i += 1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
pub unsafe fn classify_ev_batch_avx512(
    batch: &FastEvBatch,
    user_key: u64,
    event_key: u64,
    user_cap: u32,
    event_cap: u64,
    normal_out: &mut Vec<NormalEvRec>,
    fallback_out: &mut Vec<FallbackEvRec>,
) {
    classify_ev_batch_with(
        batch,
        user_key,
        event_key,
        user_cap,
        event_cap,
        normal_out,
        fallback_out,
        |y, key| unscramble40_x8_avx512(y, key),
    );
}

pub fn classify_ev_batch_avx2(
    batch: &FastEvBatch,
    user_key: u64,
    event_key: u64,
    user_cap: u32,
    event_cap: u64,
    normal_out: &mut Vec<NormalEvRec>,
    fallback_out: &mut Vec<FallbackEvRec>,
) {
    classify_ev_batch_with(
        batch,
        user_key,
        event_key,
        user_cap,
        event_cap,
        normal_out,
        fallback_out,
        unscramble40_x8_scalar,
    );
}

#[inline(always)]
fn classify_ck_batch_with(
    batch: &FastCkBatch,
    user_key: u64,
    click_key: u64,
    user_cap: u32,
    click_cap: u64,
    normal_out: &mut Vec<NormalCkRec>,
    fallback_out: &mut Vec<FallbackCkRec>,
    mut unscramble_x8: impl FnMut([u64; 8], u64) -> [u64; 8],
) {
    let n = batch.len;
    let mut i = 0;
    while i + 8 <= n {
        let c_un = unscramble_x8(
            unsafe { *(batch.cid_b36.as_ptr().add(i) as *const [u64; 8]) },
            click_key,
        );
        let u_un = unscramble_x8(
            unsafe { *(batch.uid_b36.as_ptr().add(i) as *const [u64; 8]) },
            user_key,
        );
        for k in 0..8 {
            let r = i + k;
            if (batch.canonical_mask & (1u64 << r)) != 0 {
                let c_idx = c_un[k];
                let u_val = u_un[k];
                let u_idx = u_val >> 9;
                let alias = u_val & 511;
                if u_idx < (user_cap as u64) && alias < 14 && c_idx < click_cap {
                    normal_out.push(NormalCkRec {
                        click_idx: c_idx as u32,
                        user_idx: u_idx as u32,
                        alias: alias as u8,
                        cid_b36: batch.cid_b36[r],
                        ts: batch.ts[r],
                        campaign: batch.campaign[r],
                    });
                    continue;
                }
            }
            fallback_out.push(make_fallback_ck(batch, r));
        }
        i += 8;
    }
    while i < n {
        let r = i;
        if (batch.canonical_mask & (1u64 << r)) != 0 {
            let c_idx = unscramble40(batch.cid_b36[r], click_key);
            let u_val = unscramble40(batch.uid_b36[r], user_key);
            let u_idx = u_val >> 9;
            let alias = u_val & 511;
            if u_idx < (user_cap as u64) && alias < 14 && c_idx < click_cap {
                normal_out.push(NormalCkRec {
                    click_idx: c_idx as u32,
                    user_idx: u_idx as u32,
                    alias: alias as u8,
                    cid_b36: batch.cid_b36[r],
                    ts: batch.ts[r],
                    campaign: batch.campaign[r],
                });
                i += 1;
                continue;
            }
        }
        fallback_out.push(make_fallback_ck(batch, r));
        i += 1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
pub unsafe fn classify_ck_batch_avx512(
    batch: &FastCkBatch,
    user_key: u64,
    click_key: u64,
    user_cap: u32,
    click_cap: u64,
    normal_out: &mut Vec<NormalCkRec>,
    fallback_out: &mut Vec<FallbackCkRec>,
) {
    classify_ck_batch_with(
        batch,
        user_key,
        click_key,
        user_cap,
        click_cap,
        normal_out,
        fallback_out,
        |y, key| unscramble40_x8_avx512(y, key),
    );
}

/// 64-row click batch classifier:
/// Unscrambles `cid_b36` and `uid_b36` 8 at a time via AVX-512DQ (when `use_avx512()` is enabled)
/// or `unscramble40_x8_scalar` and partitions into `normal_out` and `fallback_out`.
pub fn classify_ck_batch(
    batch: &FastCkBatch,
    user_key: u64,
    click_key: u64,
    user_cap: u32,
    click_cap: u64,
    normal_out: &mut Vec<NormalCkRec>,
    fallback_out: &mut Vec<FallbackCkRec>,
) {
    if use_avx512() {
        unsafe {
            classify_ck_batch_avx512(
                batch,
                user_key,
                click_key,
                user_cap,
                click_cap,
                normal_out,
                fallback_out,
            );
        }
        return;
    }
    classify_ck_batch_with(
        batch,
        user_key,
        click_key,
        user_cap,
        click_cap,
        normal_out,
        fallback_out,
        unscramble40_x8_scalar,
    );
}

#[inline(always)]
fn classify_lk_batch_with(
    batch: &FastLkBatch,
    user_key: u64,
    user_cap: u32,
    normal_out: &mut Vec<NormalLkRec>,
    fallback_out: &mut Vec<FallbackLkRec>,
    mut unscramble_x8: impl FnMut([u64; 8], u64) -> [u64; 8],
) {
    let n = batch.len;
    let mut i = 0;
    while i + 8 <= n {
        let a_un = unscramble_x8(
            unsafe { *(batch.ua_b36.as_ptr().add(i) as *const [u64; 8]) },
            user_key,
        );
        let b_un = unscramble_x8(
            unsafe { *(batch.ub_b36.as_ptr().add(i) as *const [u64; 8]) },
            user_key,
        );
        for k in 0..8 {
            let r = i + k;
            if (batch.canonical_mask & (1u64 << r)) != 0 {
                let va = a_un[k];
                let vb = b_un[k];
                let ua = va >> 9;
                let ub = vb >> 9;
                let aa = va & 511;
                let ab = vb & 511;
                if ua == ub && ua < (user_cap as u64) && aa < 16 && ab < 16 {
                    if aa != ab {
                        normal_out.push(NormalLkRec {
                            user_idx: ua as u32,
                            alias_a: aa as u8,
                            alias_b: ab as u8,
                        });
                    }
                    continue;
                }
            }
            fallback_out.push(make_fallback_lk(batch, r));
        }
        i += 8;
    }
    while i < n {
        let r = i;
        if (batch.canonical_mask & (1u64 << r)) != 0 {
            let va = unscramble40(batch.ua_b36[r], user_key);
            let vb = unscramble40(batch.ub_b36[r], user_key);
            let ua = va >> 9;
            let ub = vb >> 9;
            let aa = va & 511;
            let ab = vb & 511;
            if ua == ub && ua < (user_cap as u64) && aa < 16 && ab < 16 {
                if aa != ab {
                    normal_out.push(NormalLkRec {
                        user_idx: ua as u32,
                        alias_a: aa as u8,
                        alias_b: ab as u8,
                    });
                }
                i += 1;
                continue;
            }
        }
        fallback_out.push(make_fallback_lk(batch, r));
        i += 1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
pub unsafe fn classify_lk_batch_avx512(
    batch: &FastLkBatch,
    user_key: u64,
    user_cap: u32,
    normal_out: &mut Vec<NormalLkRec>,
    fallback_out: &mut Vec<FallbackLkRec>,
) {
    classify_lk_batch_with(
        batch,
        user_key,
        user_cap,
        normal_out,
        fallback_out,
        |y, key| unscramble40_x8_avx512(y, key),
    );
}

/// 64-row link batch classifier:
/// Unscrambles `ua_b36` and `ub_b36` 8 at a time via AVX-512DQ (when `use_avx512()` is enabled)
/// or `unscramble40_x8_scalar` and partitions into `normal_out` and `fallback_out`.
pub fn classify_lk_batch(
    batch: &FastLkBatch,
    user_key: u64,
    user_cap: u32,
    normal_out: &mut Vec<NormalLkRec>,
    fallback_out: &mut Vec<FallbackLkRec>,
) {
    if use_avx512() {
        unsafe {
            classify_lk_batch_avx512(batch, user_key, user_cap, normal_out, fallback_out);
        }
        return;
    }
    classify_lk_batch_with(
        batch,
        user_key,
        user_cap,
        normal_out,
        fallback_out,
        unscramble40_x8_scalar,
    );
}

#[inline(always)]
unsafe fn find_first_line_start(
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
) -> Option<usize> {
    if size == 0 || from >= size || to <= from {
        return None;
    }
    let mut pos = from.saturating_sub(1);
    while pos < size && *data.add(pos) != b'\n' {
        pos += 1;
    }
    if pos >= size {
        return None;
    }
    let cur_off = pos + 1;
    (cur_off < to).then_some(cur_off)
}

/// Validates quotes on a `< 64`-byte line (with non-empty field 0: `(cm & 1) == 0`)
/// and returns a bitmask `open_q` whose bit `k` is `1` iff field `k` had surrounding quotes.
#[inline(always)]
unsafe fn validate_and_extract_quotes_u64(qt: u64, cm: u64, nl_bit: u64) -> Option<u64> {
    let starts = (cm << 1) | 1u64;
    let ends = (cm | nl_bit) >> 1;
    let stray = qt & !(starts ^ ends);
    let open_q = _pext_u64(qt, starts);
    let close_q = _pext_u64(qt, ends);
    ((stray | (open_q ^ close_q)) == 0).then_some(open_q)
}

/// Validates quotes on a `64..128`-byte line across two 64-byte windows (with `(cm0 & 1) == 0`)
/// and returns a bitmask `open_q` whose bit `k` is `1` iff field `k` had surrounding quotes.
#[inline(always)]
unsafe fn validate_and_extract_quotes_128(
    qt0: u64,
    qt1_line: u64,
    cm0: u64,
    cm1_line: u64,
    nl1_bit: u64,
) -> Option<u64> {
    let starts0 = (cm0 << 1) | 1u64;
    let ends0 = (cm0 >> 1) | (((cm1_line | nl1_bit) & 1) << 63);
    let stray0 = qt0 & !(starts0 ^ ends0);

    let starts1 = (cm1_line << 1) | (cm0 >> 63);
    let ends1 = (cm1_line | nl1_bit) >> 1;
    let stray1 = qt1_line & !(starts1 ^ ends1);

    let k_s0 = starts0.count_ones();
    let k_e0 = ends0.count_ones();
    let open_q = _pext_u64(qt0, starts0) | (_pext_u64(qt1_line, starts1) << k_s0);
    let close_q = _pext_u64(qt0, ends0) | (_pext_u64(qt1_line, ends1) << k_e0);
    ((stray0 | stray1 | (open_q ^ close_q)) == 0).then_some(open_q)
}

#[inline(always)]
fn next_comma_u64_pair(m0: &mut u64, m1: &mut u64) -> usize {
    let t0 = m0.trailing_zeros() as usize;
    let n0 = *m0 & m0.wrapping_sub(1);
    let t1 = 64 + m1.trailing_zeros() as usize;
    let n1 = *m1 & m1.wrapping_sub(1);
    let take0 = *m0 != 0;
    *m0 = if take0 { n0 } else { *m0 };
    *m1 = if take0 { *m1 } else { n1 };
    if take0 {
        t0
    } else {
        t1
    }
}

#[cold]
#[inline(never)]
unsafe fn parse_csv_line_scalar<const N: usize>(
    data: *const u8,
    size: usize,
    cur_off: usize,
    min_nl_search: usize,
) -> (usize, Option<([*const u8; N], [usize; N])>) {
    let mut p = (cur_off + min_nl_search).min(size);
    while p < size && *data.add(p) != b'\n' {
        p += 1;
    }
    let next_off = if p < size { p + 1 } else { size };
    let line = std::slice::from_raw_parts(data.add(cur_off), p - cur_off);

    let strip_field = |s: usize, e: usize| -> Option<(*const u8, usize)> {
        let f = &line[s..e];
        if f.contains(&b'"') {
            if f.len() < 2
                || f[0] != b'"'
                || f[f.len() - 1] != b'"'
                || f[1..f.len() - 1].contains(&b'"')
            {
                return None;
            }
            Some((data.add(cur_off + s + 1), f.len() - 2))
        } else {
            Some((data.add(cur_off + s), f.len()))
        }
    };

    let mut ptrs = [data; N];
    let mut lens = [0usize; N];
    let mut field_start = 0usize;
    let mut idx = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if idx >= N - 1 {
                return (next_off, None);
            }
            let Some((fp, fl)) = strip_field(field_start, i) else {
                return (next_off, None);
            };
            ptrs[idx] = fp;
            lens[idx] = fl;
            idx += 1;
            field_start = i + 1;
        }
    }
    if idx != N - 1 {
        return (next_off, None);
    }
    let Some((fp, fl)) = strip_field(field_start, line.len()) else {
        return (next_off, None);
    };
    ptrs[idx] = fp;
    lens[idx] = fl;
    (next_off, Some((ptrs, lens)))
}

#[inline(always)]
unsafe fn push_cold_ev_to_batch(batch: &mut FastEvBatch, r: usize, cold: FallbackEvRec) {
    batch.eid_b36[r] = cold.eid_b36;
    batch.uid_b36[r] = cold.uid_b36;
    batch.event_tok[r] = cold.event_tok;
    batch.user_tok[r] = cold.user_tok;
    batch.tok_dirty_mask |= 1u64 << r;
    batch.ingest[r] = cold.ingest;
    batch.ts[r] = cold.ts;
    batch.payload[r] = cold.payload;
    if cold.valid {
        batch.valid_mask |= 1u64 << r;
        if cold.eid_b36 <= MASK40 && cold.uid_b36 <= MASK40 {
            batch.canonical_valid_mask |= 1u64 << r;
        }
    }
    batch.len = r + 1;
}

#[inline(always)]
unsafe fn process_ev_row_unquoted(
    sh: &Shared,
    fast_dates: &FastDateLut,
    dates: &mut DateCache,
    base_s: i64,
    event_win_ms: u64,
    ptrs: [*const u8; 7],
    lens: [usize; 7],
    batch: &mut FastEvBatch,
    on_batch: &mut impl FnMut(&FastEvBatch),
) {
    if lens[0] == 0 {
        return;
    }
    let r = batch.len;
    'row_done: {
        if let Some((eid_y, uid_y)) =
            decode_id_pair_unquoted(ptrs[0], lens[0], b'e', ptrs[1], lens[1], b'u')
        {
            let ingest_s_rel = if lens[3] == 20 {
                match parse_f1_fast_lut(ptrs[3], fast_dates) {
                    Some(s) => s,
                    None => parse_f1_rel_fallback(ptrs[3], 20, &sh.date_lut, dates, base_s),
                }
            } else {
                i32::MAX
            };
            if ingest_s_rel != i32::MAX {
                let ts_rel = parse_ts_fast_lut(
                    ptrs[2],
                    lens[2],
                    fast_dates,
                    &sh.date_lut,
                    dates,
                    sh.base_ms,
                );
                let win_bad = (ts_rel.wrapping_sub(WEEK_MS) as u64) >= event_win_ms;
                let late_bad = (ingest_s_rel as i64)
                    .wrapping_mul(1000)
                    .wrapping_sub(ts_rel)
                    > DAY_MS;
                if !(win_bad | late_bad) {
                    if let Some(payload) = parse_ev_tail_unquoted(
                        ptrs[4],
                        lens[4],
                        ptrs[5],
                        lens[5],
                        ptrs[6],
                        lens[6],
                        sh,
                    ) {
                        batch.eid_b36[r] = eid_y;
                        batch.uid_b36[r] = uid_y;
                        batch.ingest[r] = ingest_s_rel;
                        batch.ts[r] = ts_rel as u32;
                        batch.payload[r] = payload;
                        batch.canonical_valid_mask |= 1u64 << r;
                        batch.valid_mask |= 1u64 << r;
                        batch.len = r + 1;
                        break 'row_done;
                    }
                }
                batch.eid_b36[r] = eid_y;
                batch.uid_b36[r] = uid_y;
                batch.ingest[r] = ingest_s_rel;
                batch.ts[r] = 0;
                batch.payload[r] = 0;
                batch.len = r + 1;
                break 'row_done;
            }
        }
        if let Some(cold) = parse_event_row_cold_fields(sh, dates, ptrs, lens) {
            push_cold_ev_to_batch(batch, r, cold);
        }
    }
    if batch.len == FAST_BATCH {
        on_batch(batch);
        batch.clear();
    }
}

#[inline(always)]
unsafe fn emit_ev_from_commas_open_q(
    p: *const u8,
    c0: usize,
    c1: usize,
    c2: usize,
    c3: usize,
    c4: usize,
    c5: usize,
    nl_pos: usize,
    open_q: u64,
    sh: &Shared,
    fast_dates: &FastDateLut,
    dates: &mut DateCache,
    base_s: i64,
    event_win_ms: u64,
    batch: &mut FastEvBatch,
    on_batch: &mut impl FnMut(&FastEvBatch),
) {
    let q0 = (open_q as usize) & 1;
    let l0 = c0 - 2 * q0;
    if l0 == 0 {
        return;
    }
    let q1 = ((open_q >> 1) as usize) & 1;
    let q2 = ((open_q >> 2) as usize) & 1;
    let q3 = ((open_q >> 3) as usize) & 1;
    let q4 = ((open_q >> 4) as usize) & 1;
    let q5 = ((open_q >> 5) as usize) & 1;
    let q6 = ((open_q >> 6) as usize) & 1;

    let ptrs = [
        p.add(q0),
        p.add(c0 + 1 + q1),
        p.add(c1 + 1 + q2),
        p.add(c2 + 1 + q3),
        p.add(c3 + 1 + q4),
        p.add(c4 + 1 + q5),
        p.add(c5 + 1 + q6),
    ];
    let lens = [
        l0,
        c1 - c0 - 1 - 2 * q1,
        c2 - c1 - 1 - 2 * q2,
        c3 - c2 - 1 - 2 * q3,
        c4 - c3 - 1 - 2 * q4,
        c5 - c4 - 1 - 2 * q5,
        nl_pos - c5 - 1 - 2 * q6,
    ];
    process_ev_row_unquoted(
        sh,
        fast_dates,
        dates,
        base_s,
        event_win_ms,
        ptrs,
        lens,
        batch,
        on_batch,
    );
}

#[inline(always)]
unsafe fn process_ev_masks_u64_pair(
    nl0: u64,
    nl1: u64,
    cm0: u64,
    cm1: u64,
    qt0: u64,
    qt1: u64,
    p: *const u8,
    sh: &Shared,
    fast_dates: &FastDateLut,
    dates: &mut DateCache,
    base_s: i64,
    event_win_ms: u64,
    batch: &mut FastEvBatch,
    on_batch: &mut impl FnMut(&FastEvBatch),
) -> usize {
    if nl0 != 0 {
        let nl_pos = nl0.trailing_zeros() as usize;
        let line_mask = _bzhi_u64(u64::MAX, nl_pos as u32);
        let cm_line = cm0 & line_mask;
        if (cm_line.count_ones() == 6) & ((cm_line & 1) == 0) {
            let qt_line = qt0 & line_mask;
            if let Some(open_q) =
                validate_and_extract_quotes_u64(qt_line, cm_line, 1u64 << nl_pos)
            {
                let mut m = cm_line;
                let c0 = m.trailing_zeros() as usize;
                m &= m - 1;
                let c1 = m.trailing_zeros() as usize;
                m &= m - 1;
                let c2 = m.trailing_zeros() as usize;
                m &= m - 1;
                let c3 = m.trailing_zeros() as usize;
                m &= m - 1;
                let c4 = m.trailing_zeros() as usize;
                m &= m - 1;
                let c5 = m.trailing_zeros() as usize;
                emit_ev_from_commas_open_q(
                    p,
                    c0,
                    c1,
                    c2,
                    c3,
                    c4,
                    c5,
                    nl_pos,
                    open_q,
                    sh,
                    fast_dates,
                    dates,
                    base_s,
                    event_win_ms,
                    batch,
                    on_batch,
                );
            }
        }
        return nl_pos + 1;
    }

    let nl1_pos = nl1.trailing_zeros() as usize;
    let nl_pos = 64 + nl1_pos;
    let mask1 = _bzhi_u64(u64::MAX, nl1_pos as u32);
    let cm1_line = cm1 & mask1;
    let k0 = cm0.count_ones() as usize;
    let k1 = cm1_line.count_ones() as usize;
    if (k0 + k1 != 6) | ((cm0 & 1) != 0) {
        return nl_pos + 1;
    }
    let qt1_line = qt1 & mask1;
    let Some(open_q) =
        validate_and_extract_quotes_128(qt0, qt1_line, cm0, cm1_line, 1u64 << nl1_pos)
    else {
        return nl_pos + 1;
    };

    let mut m0 = cm0;
    let mut m1 = cm1_line;
    let (c0, c1, c2, c3, c4, c5) = if k0 >= 3 {
        let c0 = m0.trailing_zeros() as usize;
        m0 &= m0 - 1;
        let c1 = m0.trailing_zeros() as usize;
        m0 &= m0 - 1;
        let c2 = m0.trailing_zeros() as usize;
        m0 &= m0 - 1;
        let c3 = next_comma_u64_pair(&mut m0, &mut m1);
        let c4 = next_comma_u64_pair(&mut m0, &mut m1);
        let c5 = next_comma_u64_pair(&mut m0, &mut m1);
        (c0, c1, c2, c3, c4, c5)
    } else {
        let c0 = next_comma_u64_pair(&mut m0, &mut m1);
        let c1 = next_comma_u64_pair(&mut m0, &mut m1);
        let c2 = next_comma_u64_pair(&mut m0, &mut m1);
        let c3 = next_comma_u64_pair(&mut m0, &mut m1);
        let c4 = next_comma_u64_pair(&mut m0, &mut m1);
        let c5 = next_comma_u64_pair(&mut m0, &mut m1);
        (c0, c1, c2, c3, c4, c5)
    };

    emit_ev_from_commas_open_q(
        p,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        nl_pos,
        open_q,
        sh,
        fast_dates,
        dates,
        base_s,
        event_win_ms,
        batch,
        on_batch,
    );
    nl_pos + 1
}

/// 2-window AVX-512BW/VL direct-from-mmap event scanner (`< 128`-byte rows zero-copy,
/// `>= 128`-byte lines via scalar fallback, final `<= 128`-byte tail via `csv::Reader`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
pub unsafe fn scan_events_avx512(
    reader: &mut crate::csv::Reader,
    sh: &Shared,
    dates: &mut DateCache,
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
    batch: &mut FastEvBatch,
    mut on_batch: impl FnMut(&FastEvBatch),
) {
    batch.clear();
    let base_s = sh.base_ms.div_euclid(1000);
    let event_win_ms = (sh.window_end - sh.window_start) as u64;
    let fast_dates = FastDateLut::new(sh.window_start, sh.window_end, base_s);

    if !sh.use_rel {
        reader.scan::<7>(data, size, from, to, |buf, s, e| {
            let r = batch.len;
            if let Some(cold) = parse_event_row_cold(sh, dates, buf, s, e) {
                push_cold_ev_to_batch(batch, r, cold);
                if batch.len == FAST_BATCH {
                    on_batch(batch);
                    batch.clear();
                }
            }
        });
        if batch.len > 0 {
            on_batch(batch);
            batch.clear();
        }
        return;
    }

    let Some(mut cur_off) = find_first_line_start(data, size, from, to) else {
        return;
    };
    let direct_limit = to.min(size.saturating_sub(128));
    let c_nl = _mm512_set1_epi8(b'\n' as i8);
    let c_cm = _mm512_set1_epi8(b',' as i8);
    let c_qt = _mm512_set1_epi8(b'"' as i8);

    while cur_off < direct_limit {
        let p = data.add(cur_off);
        let v0 = _mm512_loadu_si512(p as *const __m512i);
        let v1 = _mm512_loadu_si512(p.add(64) as *const __m512i);
        let nl0 = _mm512_cmpeq_epi8_mask(v0, c_nl);
        let nl1 = _mm512_cmpeq_epi8_mask(v1, c_nl);
        let cm0 = _mm512_cmpeq_epi8_mask(v0, c_cm);
        let cm1 = _mm512_cmpeq_epi8_mask(v1, c_cm);
        let qt0 = _mm512_cmpeq_epi8_mask(v0, c_qt);
        let qt1 = _mm512_cmpeq_epi8_mask(v1, c_qt);

        if (nl0 | nl1) == 0 {
            let (next_off, parsed) = parse_csv_line_scalar::<7>(data, size, cur_off, 128);
            if let Some((ptrs, lens)) = parsed {
                process_ev_row_unquoted(
                    sh,
                    &fast_dates,
                    dates,
                    base_s,
                    event_win_ms,
                    ptrs,
                    lens,
                    batch,
                    &mut on_batch,
                );
            }
            cur_off = next_off;
            continue;
        }
        cur_off += process_ev_masks_u64_pair(
            nl0,
            nl1,
            cm0,
            cm1,
            qt0,
            qt1,
            p,
            sh,
            &fast_dates,
            dates,
            base_s,
            event_win_ms,
            batch,
            &mut on_batch,
        );
    }

    if cur_off < to {
        reader.scan::<7>(data, size, cur_off, to, |buf, s, e| {
            let ptrs = [
                buf.add(s[0] as usize),
                buf.add(s[1] as usize),
                buf.add(s[2] as usize),
                buf.add(s[3] as usize),
                buf.add(s[4] as usize),
                buf.add(s[5] as usize),
                buf.add(s[6] as usize),
            ];
            let lens = [
                (e[0] - s[0]) as usize,
                (e[1] - s[1]) as usize,
                (e[2] - s[2]) as usize,
                (e[3] - s[3]) as usize,
                (e[4] - s[4]) as usize,
                (e[5] - s[5]) as usize,
                (e[6] - s[6]) as usize,
            ];
            process_ev_row_unquoted(
                sh,
                &fast_dates,
                dates,
                base_s,
                event_win_ms,
                ptrs,
                lens,
                batch,
                &mut on_batch,
            );
        });
    }

    if batch.len > 0 {
        on_batch(batch);
        batch.clear();
    }
}

/// Scans an event extent `[from, to)` into 64-row `FastEvBatch` batches, dispatching to
/// `scan_events_avx512` when `use_avx512()` is enabled, else `csv::Reader` + `FastDateLut`.
pub unsafe fn scan_events_fast(
    reader: &mut crate::csv::Reader,
    sh: &Shared,
    dates: &mut DateCache,
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
    batch: &mut FastEvBatch,
    mut on_batch: impl FnMut(&FastEvBatch),
) {
    if use_avx512() && sh.use_rel {
        scan_events_avx512(reader, sh, dates, data, size, from, to, batch, on_batch);
        return;
    }

    batch.clear();
    let base_s = sh.base_ms.div_euclid(1000);
    let event_win_ms = (sh.window_end - sh.window_start) as u64;
    let fast_dates = FastDateLut::new(sh.window_start, sh.window_end, base_s);

    if !sh.use_rel {
        reader.scan::<7>(data, size, from, to, |buf, s, e| {
            let r = batch.len;
            if let Some(cold) = parse_event_row_cold(sh, dates, buf, s, e) {
                push_cold_ev_to_batch(batch, r, cold);
                if batch.len == FAST_BATCH {
                    on_batch(batch);
                    batch.clear();
                }
            }
        });
        if batch.len > 0 {
            on_batch(batch);
            batch.clear();
        }
        return;
    }

    reader.scan::<7>(data, size, from, to, |buf, s, e| {
        let at = |k: usize| buf.add(s[k] as usize);
        let len = |k: usize| (e[k] - s[k]) as usize;
        if len(0) == 0 {
            return;
        }
        let r = batch.len;
        'row_done: {
            if let Some((eid_y, uid_y)) =
                decode_id_pair_unquoted(at(0), len(0), b'e', at(1), len(1), b'u')
            {
                let ingest_s_rel = if len(3) == 20 {
                    match parse_f1_fast_lut(at(3), &fast_dates) {
                        Some(s) => s,
                        None => parse_f1_rel_fallback(at(3), 20, &sh.date_lut, dates, base_s),
                    }
                } else {
                    i32::MAX
                };
                if ingest_s_rel != i32::MAX {
                    let ts_rel = parse_ts_fast_lut(
                        at(2),
                        len(2),
                        &fast_dates,
                        &sh.date_lut,
                        dates,
                        sh.base_ms,
                    );
                    let win_bad = (ts_rel.wrapping_sub(WEEK_MS) as u64) >= event_win_ms;
                    let late_bad = (ingest_s_rel as i64)
                        .wrapping_mul(1000)
                        .wrapping_sub(ts_rel)
                        > DAY_MS;
                    if !(win_bad | late_bad) {
                        if let Some(payload) = parse_ev_tail_unquoted(
                            at(4),
                            len(4),
                            at(5),
                            len(5),
                            at(6),
                            len(6),
                            sh,
                        ) {
                            batch.eid_b36[r] = eid_y;
                            batch.uid_b36[r] = uid_y;
                            batch.ingest[r] = ingest_s_rel;
                            batch.ts[r] = ts_rel as u32;
                            batch.payload[r] = payload;
                            batch.canonical_valid_mask |= 1u64 << r;
                            batch.valid_mask |= 1u64 << r;
                            batch.len = r + 1;
                            break 'row_done;
                        }
                    }
                    batch.eid_b36[r] = eid_y;
                    batch.uid_b36[r] = uid_y;
                    batch.ingest[r] = ingest_s_rel;
                    batch.ts[r] = 0;
                    batch.payload[r] = 0;
                    batch.len = r + 1;
                    break 'row_done;
                }
            }
            if let Some(cold) = parse_event_row_cold(sh, dates, buf, s, e) {
                push_cold_ev_to_batch(batch, r, cold);
            }
        }
        if batch.len == FAST_BATCH {
            on_batch(batch);
            batch.clear();
        }
    });

    if batch.len > 0 {
        on_batch(batch);
        batch.clear();
    }
}

#[inline(always)]
unsafe fn push_cold_ck_to_batch(batch: &mut FastCkBatch, r: usize, cold: FallbackCkRec) {
    batch.cid_b36[r] = cold.cid_b36;
    batch.uid_b36[r] = cold.uid_b36;
    batch.click_tok[r] = cold.click_tok;
    batch.user_tok[r] = cold.user_tok;
    batch.tok_dirty_mask |= 1u64 << r;
    batch.ts[r] = cold.ts;
    batch.campaign[r] = cold.campaign;
    if cold.cid_b36 <= MASK40 && cold.uid_b36 <= MASK40 {
        if cold.is_upper_k {
            batch.upper_k_mask |= 1u64 << r;
        } else {
            batch.canonical_mask |= 1u64 << r;
        }
    }
    batch.len = r + 1;
}

#[inline(always)]
unsafe fn process_ck_row_unquoted(
    sh: &Shared,
    fast_dates: &FastDateLut,
    dates: &mut DateCache,
    click_win_ms: u64,
    ptrs: [*const u8; 4],
    lens: [usize; 4],
    batch: &mut FastCkBatch,
    on_batch: &mut impl FnMut(&FastCkBatch),
) {
    if lens[0] == 0 || lens[1] == 0 {
        return;
    }
    let r = batch.len;
    'row_done: {
        if let Some((cid_y, uid_y)) =
            decode_id_pair_unquoted(ptrs[0], lens[0], b'k', ptrs[1], lens[1], b'u')
        {
            let ts_rel = parse_ts_fast_lut(
                ptrs[3],
                lens[3],
                fast_dates,
                &sh.date_lut,
                dates,
                sh.base_ms,
            );
            if (ts_rel as u64) < click_win_ms {
                if let Some(camp) = parse_campaign_wq(
                    ptrs[2],
                    lens[2],
                    0,
                    &sh.direct_camp,
                    &sh.campaign_ids,
                ) {
                    batch.cid_b36[r] = cid_y;
                    batch.uid_b36[r] = uid_y;
                    batch.ts[r] = ts_rel as u32;
                    batch.campaign[r] = camp;
                    batch.canonical_mask |= 1u64 << r;
                    batch.len = r + 1;
                    break 'row_done;
                }
            }
        }
        if let Some(cold) = parse_click_row_cold_fields(sh, dates, ptrs, lens) {
            push_cold_ck_to_batch(batch, r, cold);
        }
    }
    if batch.len == FAST_BATCH {
        on_batch(batch);
        batch.clear();
    }
}

#[inline(always)]
unsafe fn emit_ck_from_commas_open_q(
    p: *const u8,
    c0: usize,
    c1: usize,
    c2: usize,
    nl_pos: usize,
    open_q: u64,
    sh: &Shared,
    fast_dates: &FastDateLut,
    dates: &mut DateCache,
    click_win_ms: u64,
    batch: &mut FastCkBatch,
    on_batch: &mut impl FnMut(&FastCkBatch),
) {
    let q0 = (open_q as usize) & 1;
    let q1 = ((open_q >> 1) as usize) & 1;
    let l0 = c0 - 2 * q0;
    let l1 = c1 - c0 - 1 - 2 * q1;
    if l0 == 0 || l1 == 0 {
        return;
    }
    let q2 = ((open_q >> 2) as usize) & 1;
    let q3 = ((open_q >> 3) as usize) & 1;

    let ptrs = [
        p.add(q0),
        p.add(c0 + 1 + q1),
        p.add(c1 + 1 + q2),
        p.add(c2 + 1 + q3),
    ];
    let lens = [l0, l1, c2 - c1 - 1 - 2 * q2, nl_pos - c2 - 1 - 2 * q3];
    process_ck_row_unquoted(
        sh,
        fast_dates,
        dates,
        click_win_ms,
        ptrs,
        lens,
        batch,
        on_batch,
    );
}

#[inline(always)]
unsafe fn process_ck_64_masks(
    nl0: u64,
    cm0: u64,
    qt0: u64,
    p: *const u8,
    sh: &Shared,
    fast_dates: &FastDateLut,
    dates: &mut DateCache,
    click_win_ms: u64,
    batch: &mut FastCkBatch,
    on_batch: &mut impl FnMut(&FastCkBatch),
) -> usize {
    let nl_pos = nl0.trailing_zeros() as usize;
    let line_mask = _bzhi_u64(u64::MAX, nl_pos as u32);
    let cm_line = cm0 & line_mask;
    if (cm_line.count_ones() == 3) & ((cm_line & 1) == 0) {
        let qt_line = qt0 & line_mask;
        if let Some(open_q) = validate_and_extract_quotes_u64(qt_line, cm_line, 1u64 << nl_pos) {
            let mut m = cm_line;
            let c0 = m.trailing_zeros() as usize;
            m &= m - 1;
            let c1 = m.trailing_zeros() as usize;
            m &= m - 1;
            let c2 = m.trailing_zeros() as usize;
            emit_ck_from_commas_open_q(
                p,
                c0,
                c1,
                c2,
                nl_pos,
                open_q,
                sh,
                fast_dates,
                dates,
                click_win_ms,
                batch,
                on_batch,
            );
        }
    }
    nl_pos + 1
}

#[inline(always)]
unsafe fn process_ck_128_masks(
    nl1: u64,
    cm0: u64,
    cm1: u64,
    qt0: u64,
    qt1: u64,
    p: *const u8,
    sh: &Shared,
    fast_dates: &FastDateLut,
    dates: &mut DateCache,
    click_win_ms: u64,
    batch: &mut FastCkBatch,
    on_batch: &mut impl FnMut(&FastCkBatch),
) -> usize {
    let nl1_pos = nl1.trailing_zeros() as usize;
    let nl_pos = 64 + nl1_pos;
    let mask1 = _bzhi_u64(u64::MAX, nl1_pos as u32);
    let cm1_line = cm1 & mask1;
    let k0 = cm0.count_ones() as usize;
    let k1 = cm1_line.count_ones() as usize;
    if (k0 + k1 != 3) | ((cm0 & 1) != 0) {
        return nl_pos + 1;
    }
    let qt1_line = qt1 & mask1;
    let Some(open_q) =
        validate_and_extract_quotes_128(qt0, qt1_line, cm0, cm1_line, 1u64 << nl1_pos)
    else {
        return nl_pos + 1;
    };
    let mut m0 = cm0;
    let mut m1 = cm1_line;
    let c0 = next_comma_u64_pair(&mut m0, &mut m1);
    let c1 = next_comma_u64_pair(&mut m0, &mut m1);
    let c2 = next_comma_u64_pair(&mut m0, &mut m1);
    emit_ck_from_commas_open_q(
        p,
        c0,
        c1,
        c2,
        nl_pos,
        open_q,
        sh,
        fast_dates,
        dates,
        click_win_ms,
        batch,
        on_batch,
    );
    nl_pos + 1
}

/// 2-window AVX-512BW/VL direct-from-mmap click scanner.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
pub unsafe fn scan_clicks_avx512(
    reader: &mut crate::csv::Reader,
    sh: &Shared,
    dates: &mut DateCache,
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
    batch: &mut FastCkBatch,
    mut on_batch: impl FnMut(&FastCkBatch),
) {
    batch.clear();
    let click_win_ms = (sh.window_end - sh.base_ms) as u64;
    let fast_dates = FastDateLut::new(sh.window_start, sh.window_end, sh.base_ms.div_euclid(1000));

    if !sh.use_rel {
        reader.scan::<4>(data, size, from, to, |buf, s, e| {
            let r = batch.len;
            if let Some(cold) = parse_click_row_cold(sh, dates, buf, s, e) {
                push_cold_ck_to_batch(batch, r, cold);
                if batch.len == FAST_BATCH {
                    on_batch(batch);
                    batch.clear();
                }
            }
        });
        if batch.len > 0 {
            on_batch(batch);
            batch.clear();
        }
        return;
    }

    let Some(mut cur_off) = find_first_line_start(data, size, from, to) else {
        return;
    };
    let direct_limit = to.min(size.saturating_sub(128));
    let c_nl = _mm512_set1_epi8(b'\n' as i8);
    let c_cm = _mm512_set1_epi8(b',' as i8);
    let c_qt = _mm512_set1_epi8(b'"' as i8);

    while cur_off < direct_limit {
        let p = data.add(cur_off);
        let v0 = _mm512_loadu_si512(p as *const __m512i);
        let nl0 = _mm512_cmpeq_epi8_mask(v0, c_nl);
        let cm0 = _mm512_cmpeq_epi8_mask(v0, c_cm);
        let qt0 = _mm512_cmpeq_epi8_mask(v0, c_qt);
        if nl0 != 0 {
            cur_off += process_ck_64_masks(
                nl0,
                cm0,
                qt0,
                p,
                sh,
                &fast_dates,
                dates,
                click_win_ms,
                batch,
                &mut on_batch,
            );
            continue;
        }
        let v1 = _mm512_loadu_si512(p.add(64) as *const __m512i);
        let nl1 = _mm512_cmpeq_epi8_mask(v1, c_nl);
        if nl1 == 0 {
            let (next_off, parsed) = parse_csv_line_scalar::<4>(data, size, cur_off, 128);
            if let Some((ptrs, lens)) = parsed {
                process_ck_row_unquoted(
                    sh,
                    &fast_dates,
                    dates,
                    click_win_ms,
                    ptrs,
                    lens,
                    batch,
                    &mut on_batch,
                );
            }
            cur_off = next_off;
            continue;
        }
        let cm1 = _mm512_cmpeq_epi8_mask(v1, c_cm);
        let qt1 = _mm512_cmpeq_epi8_mask(v1, c_qt);
        cur_off += process_ck_128_masks(
            nl1,
            cm0,
            cm1,
            qt0,
            qt1,
            p,
            sh,
            &fast_dates,
            dates,
            click_win_ms,
            batch,
            &mut on_batch,
        );
    }

    if cur_off < to {
        reader.scan::<4>(data, size, cur_off, to, |buf, s, e| {
            let ptrs = [
                buf.add(s[0] as usize),
                buf.add(s[1] as usize),
                buf.add(s[2] as usize),
                buf.add(s[3] as usize),
            ];
            let lens = [
                (e[0] - s[0]) as usize,
                (e[1] - s[1]) as usize,
                (e[2] - s[2]) as usize,
                (e[3] - s[3]) as usize,
            ];
            process_ck_row_unquoted(
                sh,
                &fast_dates,
                dates,
                click_win_ms,
                ptrs,
                lens,
                batch,
                &mut on_batch,
            );
        });
    }

    if batch.len > 0 {
        on_batch(batch);
        batch.clear();
    }
}

/// Scans a click extent `[from, to)` into 64-row `FastCkBatch` batches, dispatching to
/// `scan_clicks_avx512` when `use_avx512()` is enabled, else `csv::Reader` + `FastDateLut`.
pub unsafe fn scan_clicks_fast(
    reader: &mut crate::csv::Reader,
    sh: &Shared,
    dates: &mut DateCache,
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
    batch: &mut FastCkBatch,
    mut on_batch: impl FnMut(&FastCkBatch),
) {
    if use_avx512() && sh.use_rel {
        scan_clicks_avx512(reader, sh, dates, data, size, from, to, batch, on_batch);
        return;
    }

    batch.clear();
    let click_win_ms = (sh.window_end - sh.base_ms) as u64;
    let fast_dates = FastDateLut::new(sh.window_start, sh.window_end, sh.base_ms.div_euclid(1000));

    if !sh.use_rel {
        reader.scan::<4>(data, size, from, to, |buf, s, e| {
            let r = batch.len;
            if let Some(cold) = parse_click_row_cold(sh, dates, buf, s, e) {
                push_cold_ck_to_batch(batch, r, cold);
                if batch.len == FAST_BATCH {
                    on_batch(batch);
                    batch.clear();
                }
            }
        });
        if batch.len > 0 {
            on_batch(batch);
            batch.clear();
        }
        return;
    }

    reader.scan::<4>(data, size, from, to, |buf, s, e| {
        let at = |k: usize| buf.add(s[k] as usize);
        let len = |k: usize| (e[k] - s[k]) as usize;
        if len(0) == 0 || len(1) == 0 {
            return;
        }
        let r = batch.len;
        'row_done: {
            if let Some((cid_y, uid_y)) =
                decode_id_pair_unquoted(at(0), len(0), b'k', at(1), len(1), b'u')
            {
                let ts_rel = parse_ts_fast_lut(
                    at(3),
                    len(3),
                    &fast_dates,
                    &sh.date_lut,
                    dates,
                    sh.base_ms,
                );
                if (ts_rel as u64) < click_win_ms {
                    if let Some(camp) = parse_campaign_wq(
                        at(2),
                        len(2),
                        0,
                        &sh.direct_camp,
                        &sh.campaign_ids,
                    ) {
                        batch.cid_b36[r] = cid_y;
                        batch.uid_b36[r] = uid_y;
                        batch.ts[r] = ts_rel as u32;
                        batch.campaign[r] = camp;
                        batch.canonical_mask |= 1u64 << r;
                        batch.len = r + 1;
                        break 'row_done;
                    }
                }
            }
            if let Some(cold) = parse_click_row_cold(sh, dates, buf, s, e) {
                push_cold_ck_to_batch(batch, r, cold);
            }
        }
        if batch.len == FAST_BATCH {
            on_batch(batch);
            batch.clear();
        }
    });

    if batch.len > 0 {
        on_batch(batch);
        batch.clear();
    }
}

#[inline(always)]
unsafe fn process_lk_row_unquoted(
    pa: *const u8,
    la: usize,
    pb: *const u8,
    lb: usize,
    batch: &mut FastLkBatch,
    on_batch: &mut impl FnMut(&FastLkBatch),
) {
    if la == 0 || lb == 0 {
        return;
    }
    let r = batch.len;
    if let Some((ya, yb)) = decode_id_pair_unquoted(pa, la, b'u', pb, lb, b'u') {
        if ya == yb {
            return;
        }
        batch.ua_b36[r] = ya;
        batch.ub_b36[r] = yb;
        batch.canonical_mask |= 1u64 << r;
    } else {
        let (ta, tb) = tok_pair(pa, la, pb, lb);
        if ta == tb {
            return;
        }
        batch.ua_tok[r] = ta;
        batch.ub_tok[r] = tb;
        batch.tok_dirty_mask |= 1u64 << r;
        batch.ua_b36[r] = crate::unscramble::decode_b36_ptr(pa, la, b'u').unwrap_or(u64::MAX);
        batch.ub_b36[r] = crate::unscramble::decode_b36_ptr(pb, lb, b'u').unwrap_or(u64::MAX);
    }
    batch.len = r + 1;
    if batch.len == FAST_BATCH {
        on_batch(batch);
        batch.clear();
    }
}

#[inline(always)]
unsafe fn process_lk_64_masks(
    nl0: u64,
    cm0: u64,
    qt0: u64,
    p: *const u8,
    batch: &mut FastLkBatch,
    on_batch: &mut impl FnMut(&FastLkBatch),
) -> usize {
    let nl_pos = nl0.trailing_zeros() as usize;
    let line_mask = _bzhi_u64(u64::MAX, nl_pos as u32);
    let cm_line = cm0 & line_mask;
    if (cm_line.count_ones() == 1) & ((cm_line & 1) == 0) {
        let qt_line = qt0 & line_mask;
        if let Some(open_q) = validate_and_extract_quotes_u64(qt_line, cm_line, 1u64 << nl_pos) {
            let c0 = cm_line.trailing_zeros() as usize;
            let q0 = (open_q as usize) & 1;
            let q1 = ((open_q >> 1) as usize) & 1;
            process_lk_row_unquoted(
                p.add(q0),
                c0 - 2 * q0,
                p.add(c0 + 1 + q1),
                nl_pos - c0 - 1 - 2 * q1,
                batch,
                on_batch,
            );
        }
    }
    nl_pos + 1
}

/// 64-byte AVX-512BW direct-from-mmap link scanner draining multiple lines per ZMM load.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
pub unsafe fn scan_links_avx512(
    reader: &mut crate::csv::Reader,
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
    batch: &mut FastLkBatch,
    mut on_batch: impl FnMut(&FastLkBatch),
) {
    batch.clear();
    let Some(mut cur_off) = find_first_line_start(data, size, from, to) else {
        return;
    };
    let direct_limit = to.min(size.saturating_sub(128));
    let c_nl = _mm512_set1_epi8(b'\n' as i8);
    let c_cm = _mm512_set1_epi8(b',' as i8);
    let c_qt = _mm512_set1_epi8(b'"' as i8);

    'outer: while cur_off < direct_limit {
        let block_ptr = data.add(cur_off);
        let v0 = _mm512_loadu_si512(block_ptr as *const __m512i);
        let mut nl0 = _mm512_cmpeq_epi8_mask(v0, c_nl);
        let mut cm0 = _mm512_cmpeq_epi8_mask(v0, c_cm);
        let mut qt0 = _mm512_cmpeq_epi8_mask(v0, c_qt);
        if nl0 == 0 {
            let (next_off, parsed) = parse_csv_line_scalar::<2>(data, size, cur_off, 64);
            if let Some((ptrs, lens)) = parsed {
                process_lk_row_unquoted(ptrs[0], lens[0], ptrs[1], lens[1], batch, &mut on_batch);
            }
            cur_off = next_off;
            continue;
        }
        let mut block_pos = 0usize;
        while nl0 != 0 {
            if cur_off >= to {
                break 'outer;
            }
            let adv = process_lk_64_masks(
                nl0,
                cm0,
                qt0,
                block_ptr.add(block_pos),
                batch,
                &mut on_batch,
            );
            cur_off += adv;
            block_pos += adv;
            if adv >= 64 {
                break;
            }
            nl0 >>= adv;
            cm0 >>= adv;
            qt0 >>= adv;
        }
    }

    if cur_off < to {
        reader.scan::<2>(data, size, cur_off, to, |buf, s, e| {
            process_lk_row_unquoted(
                buf.add(s[0] as usize),
                (e[0] - s[0]) as usize,
                buf.add(s[1] as usize),
                (e[1] - s[1]) as usize,
                batch,
                &mut on_batch,
            );
        });
    }

    if batch.len > 0 {
        on_batch(batch);
        batch.clear();
    }
}

/// Scans a link extent `[from, to)` into 64-row `FastLkBatch` batches, dispatching to
/// `scan_links_avx512` when `use_avx512()` is enabled, else `csv::Reader`.
pub unsafe fn scan_links_fast(
    reader: &mut crate::csv::Reader,
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
    batch: &mut FastLkBatch,
    mut on_batch: impl FnMut(&FastLkBatch),
) {
    if use_avx512() {
        scan_links_avx512(reader, data, size, from, to, batch, on_batch);
        return;
    }

    batch.clear();
    reader.scan::<2>(data, size, from, to, |buf, s, e| {
        process_lk_row_unquoted(
            buf.add(s[0] as usize),
            (e[0] - s[0]) as usize,
            buf.add(s[1] as usize),
            (e[1] - s[1]) as usize,
            batch,
            &mut on_batch,
        );
    });
    if batch.len > 0 {
        on_batch(batch);
        batch.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scatter::Arena;
    use crate::unscramble::{encode_b36, recover_dataset_keys};
    use std::path::Path;

    fn dummy_shared(data_dir: &Path) -> Shared {
        let manifest = std::fs::read_to_string(data_dir.join("manifest.json")).unwrap();
        let ws = crate::manifest_time(&manifest, "\"window_start\"");
        let we = crate::manifest_time(&manifest, "\"window_end\"");
        let base_ms = ws - WEEK_MS;
        let use_rel = base_ms % 1000 == 0;
        let date_lut = DateLut::new(ws, we, base_ms.div_euclid(1000));
        let first_day = ws.div_euclid(DAY_MS);
        let days = ((we - 1).div_euclid(DAY_MS) - first_day + 1) as usize;
        let dims = dims::load(data_dir);
        Shared {
            window_start: ws,
            window_end: we,
            base_ms,
            use_rel,
            date_lut,
            first_day,
            days,
            products: dims.products,
            product_category: dims.product_category,
            direct_prod_cat: dims.direct_prod_cat,
            unknown_category: dims.unknown_category,
            categories: dims.categories,
            campaign_ids: dims.campaign_ids,
            direct_camp: dims.direct_camp,
            campaigns: dims.campaigns,
            links: crate::links::LinkTable::new(16),
            d_arena: Arena::new(1024, 64),
            p_arena: Arena::new(1024, 64),
            c_arena: Arena::new(1024, 64),
            d_parts: 1,
            p_parts: 1,
        }
    }

    #[test]
    fn test_decode_id_pair_wq_and_dec_wq_fuzz() {
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..50_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let v0 = rng & MASK40;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let v1 = rng & MASK40;
            let q0 = (rng & 1) as usize;
            let q1 = ((rng >> 1) & 1) as usize;

            let mut b0 = [0u8; 32];
            let mut b1 = [0u8; 32];
            let mut t0 = [0u8; 9];
            let mut t1 = [0u8; 9];
            let l0 = encode_b36(b'e', v0, &mut t0);
            let l1 = encode_b36(b'u', v1, &mut t1);

            let rw0 = if q0 != 0 {
                b0[0] = b'"';
                b0[1..1 + l0].copy_from_slice(&t0[..l0]);
                b0[1 + l0] = b'"';
                l0 + 2
            } else {
                b0[..l0].copy_from_slice(&t0[..l0]);
                l0
            };
            let rw1 = if q1 != 0 {
                b1[0] = b'"';
                b1[1..1 + l1].copy_from_slice(&t1[..l1]);
                b1[1 + l1] = b'"';
                l1 + 2
            } else {
                b1[..l1].copy_from_slice(&t1[..l1]);
                l1
            };

            let res = unsafe {
                decode_id_pair_wq(b0.as_ptr(), rw0, q0, b'e', b1.as_ptr(), rw1, q1, b'u')
            };
            assert_eq!(res, Some((v0, v1)));

            // Test decimal campaign, product, price with optional quotes
            let camp_n = ((rng % 2000) + 1) as u32;
            let s_camp = if q0 != 0 {
                format!("\"c_{camp_n}\"")
            } else {
                format!("c_{camp_n}")
            };
            let mut bc = [0u8; 32];
            bc[..s_camp.len()].copy_from_slice(s_camp.as_bytes());
            let got_c = unsafe {
                decode_dec_wq(bc.as_ptr(), s_camp.len(), q0, &CAMP_SHUFFLE_LUT, false)
            };
            assert_eq!(got_c, Some(camp_n));

            let prod_n = ((rng % 50000) + 1) as u32;
            let s_prod = if q1 != 0 {
                format!("\"p{prod_n}\"")
            } else {
                format!("p{prod_n}")
            };
            let mut bp = [0u8; 32];
            bp[..s_prod.len()].copy_from_slice(s_prod.as_bytes());
            let got_p = unsafe {
                decode_dec_wq(bp.as_ptr(), s_prod.len(), q1, &PROD_SHUFFLE_LUT, false)
            };
            assert_eq!(got_p, Some(prod_n));
        }

        // Verify 9-byte unquoted fast path rejects invalid inputs
        let good9 = *b"e123456789012345";
        let lead0 = *b"e023456789012345";
        let bad_pref = *b"x123456789012345";
        let bad_char = *b"e1234A6789012345";
        let over_40 = *b"ezzzzzzzz9012345";
        unsafe {
            assert!(decode_id_pair_wq(good9.as_ptr(), 9, 0, b'e', good9.as_ptr(), 9, 0, b'e').is_some());
            assert_eq!(decode_id_pair_wq(lead0.as_ptr(), 9, 0, b'e', good9.as_ptr(), 9, 0, b'e'), None);
            assert_eq!(decode_id_pair_wq(bad_pref.as_ptr(), 9, 0, b'e', good9.as_ptr(), 9, 0, b'e'), None);
            assert_eq!(decode_id_pair_wq(bad_char.as_ptr(), 9, 0, b'e', good9.as_ptr(), 9, 0, b'e'), None);
            assert_eq!(decode_id_pair_wq(over_40.as_ptr(), 9, 0, b'e', good9.as_ptr(), 9, 0, b'e'), None);

            // Verify FastDateLut + unified F1/F2/F3 timestamp parser matches ts::parse_ts_rel on valid and malformed forms
            let ws = 1828500000_000i64 - (1828500000_000i64 % DAY_MS);
            let we = ws + 14 * DAY_MS;
            let base_ms = ws - WEEK_MS;
            let lut = DateLut::new(ws, we, base_ms.div_euclid(1000));
            let fast_lut = FastDateLut::new(ws, we, base_ms.div_euclid(1000));
            let mut dc1 = DateCache::new();
            let mut dc2 = DateCache::new();
            let ts_cases: &[&[u8]] = &[
                b"2027-12-12T04:10:10Z",
                b"2027-12-09T19:21:09+05:30",
                b"2027-12-09T19:21:09-10:00",
                b"2027-12-09T19:21:09+24:00",
                b"2027-12-09T19:21:09+05:60",
                b"2027-12-09T19:21:09+05;00",
                b"2027-12-12T04:10:10z",
                b"2027-12-12T04:10:10",
                b"2027-12-12T04:10:10.123Z",
                b"1828801833000",
                b"182880183300a",
                b"2028-02-29T00:00:00Z",
                b"2027-02-29T00:00:00Z",
                b"2027-12-00T00:00:00Z",
                b"2027-12-32T00:00:00Z",
                b"2027-12-12/04:10:10Z",
            ];
            for &s in ts_cases {
                let mut buf = [b' '; 64];
                buf[16..16 + s.len()].copy_from_slice(s);
                let p = buf.as_ptr().add(16);
                let exp = ts::parse_ts_rel(p, s.len(), &lut, &mut dc1, base_ms);
                let got = parse_ts_fast_lut(p, s.len(), &fast_lut, &lut, &mut dc2, base_ms);
                assert_eq!(got, exp, "parse_ts_fast_lut mismatch on {:?}", String::from_utf8_lossy(s));
            }
            // Verify FastDateLut on leap-year Feb 2028 boundary
            let ws_leap = 1835222400_000i64; // 2028-02-27T00:00:00Z
            let we_leap = ws_leap + 14 * DAY_MS;
            let base_leap = ws_leap - WEEK_MS;
            let lut_leap = DateLut::new(ws_leap, we_leap, base_leap.div_euclid(1000));
            let fast_leap = FastDateLut::new(ws_leap, we_leap, base_leap.div_euclid(1000));
            for s in [
                b"2028-02-28T23:59:59Z".as_slice(),
                b"2028-02-29T12:34:56Z".as_slice(),
                b"2028-03-01T00:00:00Z".as_slice(),
                b"2028-02-30T00:00:00Z".as_slice(),
            ] {
                let mut buf = [b' '; 64];
                buf[16..16 + s.len()].copy_from_slice(s);
                let p = buf.as_ptr().add(16);
                let exp = ts::parse_ts_rel(p, s.len(), &lut_leap, &mut dc1, base_leap);
                let got = parse_ts_fast_lut(p, s.len(), &fast_leap, &lut_leap, &mut dc2, base_leap);
                assert_eq!(got, exp, "leap parse_ts_fast_lut mismatch on {:?}", String::from_utf8_lossy(s));
            }
        }
    }

    #[test]
    fn test_fast_parse_matches_reference_on_dev_and_fixtures() {
        let dev_dir = Path::new("/home/colton/Documents/Projects/Builders/#3/data/dev");
        if !dev_dir.exists() {
            return;
        }
        let sh = dummy_shared(dev_dir);
        let keys = recover_dataset_keys(dev_dir).expect("recover dev keys");

        // Test on the largest event file, click file, and link file in data/dev
        let ev_path = dev_dir.join("events/part-00001.csv");
        let ev_file = std::fs::File::open(&ev_path).unwrap();
        let ev_size = ev_file.metadata().unwrap().len() as usize;
        let ev_data = crate::sys::map_file(&ev_file, ev_size);

        let mut reader1 = crate::csv::Reader::new(64 << 10);
        let mut dates1 = DateCache::new();
        let mut batch = FastEvBatch::new();
        let mut normal_evs = Vec::new();
        let mut fallback_evs = Vec::new();
        let mut fast_rows: Vec<(u64, u64, i32, u32, u32, bool)> = Vec::new();

        let t0 = std::time::Instant::now();
        unsafe {
            scan_events_fast(
                &mut reader1,
                &sh,
                &mut dates1,
                ev_data,
                ev_size,
                0,
                ev_size,
                &mut batch,
                |b| {
                    for r in 0..b.len {
                        let valid = (b.valid_mask & (1u64 << r)) != 0;
                        let etok = if b.event_tok[r] != 0 {
                            b.event_tok[r]
                        } else {
                            encode_base36_token(b'e', b.eid_b36[r])
                        };
                        let utok = if !valid {
                            0
                        } else if b.user_tok[r] != 0 {
                            b.user_tok[r]
                        } else {
                            encode_base36_token(b'u', b.uid_b36[r])
                        };
                        fast_rows.push((
                            etok,
                            utok,
                            b.ingest[r],
                            if valid { b.ts[r] } else { 0 },
                            if valid { b.payload[r] } else { 0 },
                            valid,
                        ));
                    }
                    classify_ev_batch(
                        b,
                        keys.user_key,
                        keys.event_key,
                        keys.user_cap,
                        keys.event_cap,
                        &mut normal_evs,
                        &mut fallback_evs,
                    );
                },
            );
        }
        let fast_elapsed = t0.elapsed();

        // Compare row-by-row against cold reference parser
        let mut reader2 = crate::csv::Reader::new(64 << 10);
        let mut dates2 = DateCache::new();
        let mut ref_rows: Vec<(u64, u64, i32, u32, u32, bool)> = Vec::new();
        unsafe {
            reader2.scan::<7>(ev_data, ev_size, 0, ev_size, |buf, s, e| {
                if let Some(c) = parse_event_row_cold(&sh, &mut dates2, buf, s, e) {
                    ref_rows.push((c.event_tok, c.user_tok, c.ingest, c.ts, c.payload, c.valid));
                }
            });
        }
        crate::sys::unmap_file(ev_data, ev_size);

        assert_eq!(fast_rows.len(), ref_rows.len());
        for (idx, (f, r)) in fast_rows.iter().zip(ref_rows.iter()).enumerate() {
            assert_eq!(f, r, "event row {idx} mismatch");
        }
        eprintln!(
            "Verified {} event rows in {:?}: {} normal ({:.1}%), {} fallback",
            fast_rows.len(),
            fast_elapsed,
            normal_evs.len(),
            normal_evs.len() as f64 * 100.0 / fast_rows.len() as f64,
            fallback_evs.len()
        );
        assert!(normal_evs.len() * 100 > fast_rows.len() * 90);
        for fb in &fallback_evs {
            assert_ne!(fb.event_tok, 0);
            if fb.valid {
                assert_ne!(fb.user_tok, 0);
            } else {
                assert_eq!(fb.user_tok, 0);
            }
        }

        // Also test clicks
        let ck_path = dev_dir.join("clicks/part-00001.csv");
        let ck_file = std::fs::File::open(&ck_path).unwrap();
        let ck_size = ck_file.metadata().unwrap().len() as usize;
        let ck_data = crate::sys::map_file(&ck_file, ck_size);

        let mut ck_batch = FastCkBatch::new();
        let mut normal_cks = Vec::new();
        let mut fallback_cks = Vec::new();
        let mut fast_cks: Vec<(u64, u64, u32, u32)> = Vec::new();
        unsafe {
            scan_clicks_fast(
                &mut reader1,
                &sh,
                &mut dates1,
                ck_data,
                ck_size,
                0,
                ck_size,
                &mut ck_batch,
                |b| {
                    for r in 0..b.len {
                        let ctok = if b.click_tok[r] != 0 {
                            b.click_tok[r]
                        } else {
                            encode_base36_token(b'k', b.cid_b36[r])
                        };
                        let utok = if b.user_tok[r] != 0 {
                            b.user_tok[r]
                        } else {
                            encode_base36_token(b'u', b.uid_b36[r])
                        };
                        fast_cks.push((ctok, utok, b.ts[r], b.campaign[r]));
                    }
                    classify_ck_batch(
                        b,
                        keys.user_key,
                        keys.click_key,
                        keys.user_cap,
                        keys.click_cap,
                        &mut normal_cks,
                        &mut fallback_cks,
                    );
                },
            );
        }

        let mut ref_cks: Vec<(u64, u64, u32, u32)> = Vec::new();
        unsafe {
            reader2.scan::<4>(ck_data, ck_size, 0, ck_size, |buf, s, e| {
                if let Some(c) = parse_click_row_cold(&sh, &mut dates2, buf, s, e) {
                    ref_cks.push((c.click_tok, c.user_tok, c.ts, c.campaign));
                }
            });
        }
        crate::sys::unmap_file(ck_data, ck_size);

        assert_eq!(fast_cks, ref_cks);
        eprintln!(
            "Verified {} click rows: {} normal ({:.1}%), {} fallback",
            fast_cks.len(),
            normal_cks.len(),
            normal_cks.len() as f64 * 100.0 / fast_cks.len() as f64,
            fallback_cks.len()
        );
        assert!(normal_cks.len() * 100 > fast_cks.len() * 90);

        // Also test links
        let lk_path = dev_dir.join("links/part-00001.csv");
        let lk_file = std::fs::File::open(&lk_path).unwrap();
        let lk_size = lk_file.metadata().unwrap().len() as usize;
        let lk_data = crate::sys::map_file(&lk_file, lk_size);

        let mut lk_batch = FastLkBatch::new();
        let mut normal_lks = Vec::new();
        let mut fallback_lks = Vec::new();
        let mut fast_lks: Vec<(u64, u64)> = Vec::new();
        unsafe {
            scan_links_fast(&mut reader1, lk_data, lk_size, 0, lk_size, &mut lk_batch, |b| {
                for r in 0..b.len {
                    let ta = if b.ua_tok[r] != 0 {
                        b.ua_tok[r]
                    } else {
                        encode_base36_token(b'u', b.ua_b36[r])
                    };
                    let tb = if b.ub_tok[r] != 0 {
                        b.ub_tok[r]
                    } else {
                        encode_base36_token(b'u', b.ub_b36[r])
                    };
                    fast_lks.push((ta, tb));
                }
                classify_lk_batch(
                    b,
                    keys.user_key,
                    keys.user_cap,
                    &mut normal_lks,
                    &mut fallback_lks,
                );
            });
        }
        let mut ref_lks: Vec<(u64, u64)> = Vec::new();
        unsafe {
            reader2.scan::<2>(lk_data, lk_size, 0, lk_size, |buf, s, e| {
                let (la, lb) = ((e[0] - s[0]) as usize, (e[1] - s[1]) as usize);
                if la != 0 && lb != 0 {
                    let (a, b) = tok_pair(buf.add(s[0] as usize), la, buf.add(s[1] as usize), lb);
                    if a != b {
                        ref_lks.push((a, b));
                    }
                }
            });
        }
        crate::sys::unmap_file(lk_data, lk_size);
        assert_eq!(fast_lks, ref_lks);
        // Verify multi-extent unaligned range splits match single-extent scan on dev events
        {
            let mut split_ev_rows = Vec::new();
            let ev_file2 = std::fs::File::open(&ev_path).unwrap();
            let ev_sz2 = ev_file2.metadata().unwrap().len() as usize;
            let ev_ptr2 = crate::sys::map_file(&ev_file2, ev_sz2);
            for t in 0..8 {
                let a = ev_sz2 * t / 8;
                let b = ev_sz2 * (t + 1) / 8;
                unsafe {
                    scan_events_fast(&mut reader1, &sh, &mut dates1, ev_ptr2, ev_sz2, a, b, &mut batch, |fb| {
                        for r in 0..fb.len {
                            let valid = (fb.valid_mask & (1u64 << r)) != 0;
                            let etok = if fb.event_tok[r] != 0 {
                                fb.event_tok[r]
                            } else {
                                encode_base36_token(b'e', fb.eid_b36[r])
                            };
                            let utok = if !valid {
                                0
                            } else if fb.user_tok[r] != 0 {
                                fb.user_tok[r]
                            } else {
                                encode_base36_token(b'u', fb.uid_b36[r])
                            };
                            split_ev_rows.push((
                                etok,
                                utok,
                                fb.ingest[r],
                                if valid { fb.ts[r] } else { 0 },
                                if valid { fb.payload[r] } else { 0 },
                                valid,
                            ));
                        }
                    });
                }
            }
            crate::sys::unmap_file(ev_ptr2, ev_sz2);
            assert_eq!(split_ev_rows, ref_rows, "multi-extent split mismatch on dev events");
        }

        // Also test every single fixture directory in /home/colton/Documents/Projects/Builders/#3/fixtures/*
        let fixtures_root = Path::new("/home/colton/Documents/Projects/Builders/#3/fixtures");
        if fixtures_root.exists() {
            let mut fixture_count = 0usize;
            for entry in std::fs::read_dir(fixtures_root).unwrap().flatten() {
                let in_dir = entry.path().join("input");
                if !in_dir.is_dir() {
                    continue;
                }
                fixture_count += 1;
                let f_sh = dummy_shared(&in_dir);
                for ev_e in std::fs::read_dir(in_dir.join("events")).unwrap().flatten() {
                    if ev_e.path().extension().is_some_and(|x| x == "csv") {
                        let f = std::fs::File::open(ev_e.path()).unwrap();
                        let mut sz = f.metadata().unwrap().len() as usize;
                        if sz == 0 {
                            continue;
                        }
                        let ptr = crate::sys::map_file(&f, sz);
                        sz = unsafe { crate::sys::end_line(ptr, sz) };
                        let mut f_fast = Vec::new();
                        let mut f_ref = Vec::new();
                        unsafe {
                            scan_events_fast(
                                &mut reader1,
                                &f_sh,
                                &mut dates1,
                                ptr,
                                sz,
                                0,
                                sz,
                                &mut batch,
                                |b| {
                                    for r in 0..b.len {
                                        let valid = (b.valid_mask & (1u64 << r)) != 0;
                                        let etok = if b.event_tok[r] != 0 {
                                            b.event_tok[r]
                                        } else {
                                            encode_base36_token(b'e', b.eid_b36[r])
                                        };
                                        let utok = if !valid {
                                            0
                                        } else if b.user_tok[r] != 0 {
                                            b.user_tok[r]
                                        } else {
                                            encode_base36_token(b'u', b.uid_b36[r])
                                        };
                                        f_fast.push((
                                            etok,
                                            utok,
                                            b.ingest[r],
                                            if valid { b.ts[r] } else { 0 },
                                            if valid { b.payload[r] } else { 0 },
                                            valid,
                                        ));
                                    }
                                },
                            );
                            reader2.scan::<7>(ptr, sz, 0, sz, |buf, s, e| {
                                if let Some(c) = parse_event_row_cold(&f_sh, &mut dates2, buf, s, e) {
                                    f_ref.push((c.event_tok, c.user_tok, c.ingest, c.ts, c.payload, c.valid));
                                }
                            });
                        }
                        crate::sys::unmap_file(ptr, sz);
                        assert_eq!(f_fast, f_ref, "fixture {:?} events mismatch", entry.file_name());
                    }
                }
                for ck_e in std::fs::read_dir(in_dir.join("clicks")).unwrap().flatten() {
                    if ck_e.path().extension().is_some_and(|x| x == "csv") {
                        let f = std::fs::File::open(ck_e.path()).unwrap();
                        let mut sz = f.metadata().unwrap().len() as usize;
                        if sz == 0 {
                            continue;
                        }
                        let ptr = crate::sys::map_file(&f, sz);
                        sz = unsafe { crate::sys::end_line(ptr, sz) };
                        let mut f_fast = Vec::new();
                        let mut f_ref = Vec::new();
                        unsafe {
                            scan_clicks_fast(
                                &mut reader1,
                                &f_sh,
                                &mut dates1,
                                ptr,
                                sz,
                                0,
                                sz,
                                &mut ck_batch,
                                |b| {
                                    for r in 0..b.len {
                                        let ctok = if b.click_tok[r] != 0 {
                                            b.click_tok[r]
                                        } else {
                                            encode_base36_token(b'k', b.cid_b36[r])
                                        };
                                        let utok = if b.user_tok[r] != 0 {
                                            b.user_tok[r]
                                        } else {
                                            encode_base36_token(b'u', b.uid_b36[r])
                                        };
                                        f_fast.push((ctok, utok, b.ts[r], b.campaign[r]));
                                    }
                                },
                            );
                            reader2.scan::<4>(ptr, sz, 0, sz, |buf, s, e| {
                                if let Some(c) = parse_click_row_cold(&f_sh, &mut dates2, buf, s, e) {
                                    f_ref.push((c.click_tok, c.user_tok, c.ts, c.campaign));
                                }
                            });
                        }
                        crate::sys::unmap_file(ptr, sz);
                        assert_eq!(f_fast, f_ref, "fixture {:?} clicks mismatch", entry.file_name());
                    }
                }
                for lk_e in std::fs::read_dir(in_dir.join("links")).unwrap().flatten() {
                    if lk_e.path().extension().is_some_and(|x| x == "csv") {
                        let f = std::fs::File::open(lk_e.path()).unwrap();
                        let mut sz = f.metadata().unwrap().len() as usize;
                        if sz == 0 {
                            continue;
                        }
                        let ptr = crate::sys::map_file(&f, sz);
                        sz = unsafe { crate::sys::end_line(ptr, sz) };
                        let mut f_fast = Vec::new();
                        let mut f_ref = Vec::new();
                        unsafe {
                            scan_links_fast(&mut reader1, ptr, sz, 0, sz, &mut lk_batch, |b| {
                                for r in 0..b.len {
                                    let ta = if b.ua_tok[r] != 0 {
                                        b.ua_tok[r]
                                    } else {
                                        encode_base36_token(b'u', b.ua_b36[r])
                                    };
                                    let tb = if b.ub_tok[r] != 0 {
                                        b.ub_tok[r]
                                    } else {
                                        encode_base36_token(b'u', b.ub_b36[r])
                                    };
                                    f_fast.push((ta, tb));
                                }
                            });
                            reader2.scan::<2>(ptr, sz, 0, sz, |buf, s, e| {
                                let (la, lb) = ((e[0] - s[0]) as usize, (e[1] - s[1]) as usize);
                                if la != 0 && lb != 0 {
                                    let (a, b) = tok_pair(buf.add(s[0] as usize), la, buf.add(s[1] as usize), lb);
                                    if a != b {
                                        f_ref.push((a, b));
                                    }
                                }
                            });
                        }
                        crate::sys::unmap_file(ptr, sz);
                        assert_eq!(f_fast, f_ref, "fixture {:?} links mismatch", entry.file_name());
                    }
                }
            }
            eprintln!("Verified all {fixture_count} hand-written fixture suites (E1..E20, A1..A12, L1..L4)!");
        }
    }

    #[cfg(target_feature = "bmi2")]
    #[test]
    fn test_avx512_bitmask_helpers_match_cold_reference() {
        fn compute_delim_masks_64(p: *const u8) -> (u64, u64, u64) {
            let (mut nl, mut cm, mut qt) = (0u64, 0u64, 0u64);
            for i in 0..64 {
                let b = unsafe { *p.add(i) };
                let bit = 1u64 << i;
                if b == b'\n' {
                    nl |= bit;
                } else if b == b',' {
                    cm |= bit;
                } else if b == b'"' {
                    qt |= bit;
                }
            }
            (nl, cm, qt)
        }

        unsafe fn run_ev_mask_pipeline(
            sh: &Shared,
            buf: &[u8],
            from: usize,
            to: usize,
        ) -> Vec<(u64, u64, i32, u32, u32, bool)> {
            let mut padded = Vec::with_capacity(64 + buf.len() + 256);
            padded.resize(64, b'\n');
            padded.extend_from_slice(buf);
            padded.resize(padded.len() + 256, b'\n');
            let data = padded.as_ptr().add(64);
            let size = buf.len();
            let base_s = sh.base_ms.div_euclid(1000);
            let event_win_ms = (sh.window_end - sh.window_start) as u64;
            let fast_dates = FastDateLut::new(sh.window_start, sh.window_end, base_s);
            let mut dates = DateCache::new();
            let mut reader = crate::csv::Reader::new(64 << 10);
            let mut batch = FastEvBatch::new();
            let mut out = Vec::new();
            let mut drain = |b: &FastEvBatch| {
                for r in 0..b.len {
                    let valid = (b.valid_mask & (1u64 << r)) != 0;
                    let etok = if b.event_tok[r] != 0 {
                        b.event_tok[r]
                    } else {
                        encode_base36_token(b'e', b.eid_b36[r])
                    };
                    let utok = if !valid {
                        0
                    } else if b.user_tok[r] != 0 {
                        b.user_tok[r]
                    } else {
                        encode_base36_token(b'u', b.uid_b36[r])
                    };
                    out.push((
                        etok,
                        utok,
                        b.ingest[r],
                        if valid { b.ts[r] } else { 0 },
                        if valid { b.payload[r] } else { 0 },
                        valid,
                    ));
                }
            };

            if !sh.use_rel {
                reader.scan::<7>(data, size, from, to, |bptr, s, e| {
                    let r = batch.len;
                    if let Some(cold) = parse_event_row_cold(sh, &mut dates, bptr, s, e) {
                        push_cold_ev_to_batch(&mut batch, r, cold);
                        if batch.len == FAST_BATCH {
                            drain(&batch);
                            batch.clear();
                        }
                    }
                });
                if batch.len > 0 {
                    drain(&batch);
                }
                return out;
            }

            let Some(mut cur_off) = find_first_line_start(data, size, from, to) else {
                return out;
            };
            let direct_limit = to.min(size.saturating_sub(128));
            while cur_off < direct_limit {
                let p = data.add(cur_off);
                let (nl0, cm0, qt0) = compute_delim_masks_64(p);
                let (nl1, cm1, qt1) = compute_delim_masks_64(p.add(64));
                if (nl0 | nl1) == 0 {
                    let (next_off, parsed) = parse_csv_line_scalar::<7>(data, size, cur_off, 128);
                    if let Some((ptrs, lens)) = parsed {
                        process_ev_row_unquoted(
                            sh,
                            &fast_dates,
                            &mut dates,
                            base_s,
                            event_win_ms,
                            ptrs,
                            lens,
                            &mut batch,
                            &mut drain,
                        );
                    }
                    cur_off = next_off;
                    continue;
                }
                cur_off += process_ev_masks_u64_pair(
                    nl0,
                    nl1,
                    cm0,
                    cm1,
                    qt0,
                    qt1,
                    p,
                    sh,
                    &fast_dates,
                    &mut dates,
                    base_s,
                    event_win_ms,
                    &mut batch,
                    &mut drain,
                );
            }
            if cur_off < to {
                reader.scan::<7>(data, size, cur_off, to, |bptr, s, e| {
                    let ptrs = [
                        bptr.add(s[0] as usize),
                        bptr.add(s[1] as usize),
                        bptr.add(s[2] as usize),
                        bptr.add(s[3] as usize),
                        bptr.add(s[4] as usize),
                        bptr.add(s[5] as usize),
                        bptr.add(s[6] as usize),
                    ];
                    let lens = [
                        (e[0] - s[0]) as usize,
                        (e[1] - s[1]) as usize,
                        (e[2] - s[2]) as usize,
                        (e[3] - s[3]) as usize,
                        (e[4] - s[4]) as usize,
                        (e[5] - s[5]) as usize,
                        (e[6] - s[6]) as usize,
                    ];
                    process_ev_row_unquoted(
                        sh,
                        &fast_dates,
                        &mut dates,
                        base_s,
                        event_win_ms,
                        ptrs,
                        lens,
                        &mut batch,
                        &mut drain,
                    );
                });
            }
            if batch.len > 0 {
                drain(&batch);
            }
            out
        }

        unsafe fn run_ck_mask_pipeline(
            sh: &Shared,
            buf: &[u8],
            from: usize,
            to: usize,
        ) -> Vec<(u64, u64, u32, u32)> {
            let mut padded = Vec::with_capacity(64 + buf.len() + 256);
            padded.resize(64, b'\n');
            padded.extend_from_slice(buf);
            padded.resize(padded.len() + 256, b'\n');
            let data = padded.as_ptr().add(64);
            let size = buf.len();
            let click_win_ms = (sh.window_end - sh.base_ms) as u64;
            let fast_dates =
                FastDateLut::new(sh.window_start, sh.window_end, sh.base_ms.div_euclid(1000));
            let mut dates = DateCache::new();
            let mut reader = crate::csv::Reader::new(64 << 10);
            let mut batch = FastCkBatch::new();
            let mut out = Vec::new();
            let mut drain = |b: &FastCkBatch| {
                for r in 0..b.len {
                    let ctok = if b.click_tok[r] != 0 {
                        b.click_tok[r]
                    } else {
                        encode_base36_token(b'k', b.cid_b36[r])
                    };
                    let utok = if b.user_tok[r] != 0 {
                        b.user_tok[r]
                    } else {
                        encode_base36_token(b'u', b.uid_b36[r])
                    };
                    out.push((ctok, utok, b.ts[r], b.campaign[r]));
                }
            };

            if !sh.use_rel {
                reader.scan::<4>(data, size, from, to, |bptr, s, e| {
                    let r = batch.len;
                    if let Some(cold) = parse_click_row_cold(sh, &mut dates, bptr, s, e) {
                        push_cold_ck_to_batch(&mut batch, r, cold);
                        if batch.len == FAST_BATCH {
                            drain(&batch);
                            batch.clear();
                        }
                    }
                });
                if batch.len > 0 {
                    drain(&batch);
                }
                return out;
            }

            let Some(mut cur_off) = find_first_line_start(data, size, from, to) else {
                return out;
            };
            let direct_limit = to.min(size.saturating_sub(128));
            while cur_off < direct_limit {
                let p = data.add(cur_off);
                let (nl0, cm0, qt0) = compute_delim_masks_64(p);
                if nl0 != 0 {
                    cur_off += process_ck_64_masks(
                        nl0,
                        cm0,
                        qt0,
                        p,
                        sh,
                        &fast_dates,
                        &mut dates,
                        click_win_ms,
                        &mut batch,
                        &mut drain,
                    );
                    continue;
                }
                let (nl1, cm1, qt1) = compute_delim_masks_64(p.add(64));
                if nl1 == 0 {
                    let (next_off, parsed) = parse_csv_line_scalar::<4>(data, size, cur_off, 128);
                    if let Some((ptrs, lens)) = parsed {
                        process_ck_row_unquoted(
                            sh,
                            &fast_dates,
                            &mut dates,
                            click_win_ms,
                            ptrs,
                            lens,
                            &mut batch,
                            &mut drain,
                        );
                    }
                    cur_off = next_off;
                    continue;
                }
                cur_off += process_ck_128_masks(
                    nl1,
                    cm0,
                    cm1,
                    qt0,
                    qt1,
                    p,
                    sh,
                    &fast_dates,
                    &mut dates,
                    click_win_ms,
                    &mut batch,
                    &mut drain,
                );
            }
            if cur_off < to {
                reader.scan::<4>(data, size, cur_off, to, |bptr, s, e| {
                    let ptrs = [
                        bptr.add(s[0] as usize),
                        bptr.add(s[1] as usize),
                        bptr.add(s[2] as usize),
                        bptr.add(s[3] as usize),
                    ];
                    let lens = [
                        (e[0] - s[0]) as usize,
                        (e[1] - s[1]) as usize,
                        (e[2] - s[2]) as usize,
                        (e[3] - s[3]) as usize,
                    ];
                    process_ck_row_unquoted(
                        sh,
                        &fast_dates,
                        &mut dates,
                        click_win_ms,
                        ptrs,
                        lens,
                        &mut batch,
                        &mut drain,
                    );
                });
            }
            if batch.len > 0 {
                drain(&batch);
            }
            out
        }

        unsafe fn run_lk_mask_pipeline(buf: &[u8], from: usize, to: usize) -> Vec<(u64, u64)> {
            let mut padded = Vec::with_capacity(64 + buf.len() + 256);
            padded.resize(64, b'\n');
            padded.extend_from_slice(buf);
            padded.resize(padded.len() + 256, b'\n');
            let data = padded.as_ptr().add(64);
            let size = buf.len();
            let mut reader = crate::csv::Reader::new(64 << 10);
            let mut batch = FastLkBatch::new();
            let mut out = Vec::new();
            let mut drain = |b: &FastLkBatch| {
                for r in 0..b.len {
                    let ta = if b.ua_tok[r] != 0 {
                        b.ua_tok[r]
                    } else {
                        encode_base36_token(b'u', b.ua_b36[r])
                    };
                    let tb = if b.ub_tok[r] != 0 {
                        b.ub_tok[r]
                    } else {
                        encode_base36_token(b'u', b.ub_b36[r])
                    };
                    out.push((ta, tb));
                }
            };
            let Some(mut cur_off) = find_first_line_start(data, size, from, to) else {
                return out;
            };
            let direct_limit = to.min(size.saturating_sub(128));
            'outer: while cur_off < direct_limit {
                let block_ptr = data.add(cur_off);
                let (mut nl0, mut cm0, mut qt0) = compute_delim_masks_64(block_ptr);
                if nl0 == 0 {
                    let (next_off, parsed) = parse_csv_line_scalar::<2>(data, size, cur_off, 64);
                    if let Some((ptrs, lens)) = parsed {
                        process_lk_row_unquoted(
                            ptrs[0],
                            lens[0],
                            ptrs[1],
                            lens[1],
                            &mut batch,
                            &mut drain,
                        );
                    }
                    cur_off = next_off;
                    continue;
                }
                let mut block_pos = 0usize;
                while nl0 != 0 {
                    if cur_off >= to {
                        break 'outer;
                    }
                    let adv = process_lk_64_masks(
                        nl0,
                        cm0,
                        qt0,
                        block_ptr.add(block_pos),
                        &mut batch,
                        &mut drain,
                    );
                    cur_off += adv;
                    block_pos += adv;
                    if adv >= 64 {
                        break;
                    }
                    nl0 >>= adv;
                    cm0 >>= adv;
                    qt0 >>= adv;
                }
            }
            if cur_off < to {
                reader.scan::<2>(data, size, cur_off, to, |bptr, s, e| {
                    process_lk_row_unquoted(
                        bptr.add(s[0] as usize),
                        (e[0] - s[0]) as usize,
                        bptr.add(s[1] as usize),
                        (e[1] - s[1]) as usize,
                        &mut batch,
                        &mut drain,
                    );
                });
            }
            if batch.len > 0 {
                drain(&batch);
            }
            out
        }

        // 1. Synthetic boundary stress cases for <64B, 64B, 65..128B, >=128B lines, and quote rules.
        let dev_dir = Path::new("/home/colton/Documents/Projects/Builders/#3/data/dev");
        if dev_dir.exists() {
            let sh = dummy_shared(dev_dir);
            let mut ev_cases = String::from("event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
            let mut ck_cases = String::from("click_id,user_id,campaign_id,ts\n");
            let mut lk_cases = String::from("user_a,user_b\n");

            // Exact 63, 64, 65, 127, 128, 160-byte event lines (with non-empty & empty trailing price_cents)
            // Base line without trailing digits on field 0: 1 + 6 commas + dates + type + prod = ~60 bytes
            let pad_to = |prefix: &str, suffix: &str, target_nl_pos: usize| -> String {
                let need = target_nl_pos - prefix.len() - suffix.len();
                format!("{prefix}{}{suffix}\n", "0".repeat(need))
            };
            // nl_pos == 63 and 64 with empty last field (c5 == 62 and c5 == 63)
            ev_cases.push_str(&pad_to("e1,", "u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,view,p1,", 63));
            ev_cases.push_str(&pad_to("e2,", "u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,view,p1,", 64));
            ev_cases.push_str(&pad_to("e3,", "u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,view,p1,100", 64));
            ev_cases.push_str(&pad_to("e4,", "u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,view,p1,", 127));
            ev_cases.push_str(&pad_to("e5,", "u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,view,p1,99", 127));
            ev_cases.push_str(&pad_to("e6,", "u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,purchase,p1,199", 128));
            ev_cases.push_str(&pad_to("e7,", "u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,purchase,p1,299", 160));
            // k0 < 3 (leading field > 64 bytes -> all commas in window 1)
            ev_cases.push_str(&format!(
                "e{},u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,view,p1,10\n",
                "z".repeat(66)
            ));
            // Comma right at byte 63 followed by comma at byte 64 (cm0 bit 63 = 1, cm1 bit 0 = 1)
            ev_cases.push_str(&format!(
                "{},,\"\",2027-12-12T04:10:11Z,view,p1,10\n",
                format!("e10,u1{}", "x".repeat(57))
            ));
            // Quoted fields across 64-byte boundary + malformed quote forms + empty ""
            ev_cases.push_str("\"e11\",\"u1\",\"2027-12-12T04:10:10Z\",\"2027-12-12T04:10:11Z\",\"cart\",\"p1\",\"50\"\n");
            ev_cases.push_str("e12,u1,\"2027-12-12T04:10:10Z\",\"2027-12-12T04:10:11Z\",view,p1,\"\"\n");
            ev_cases.push_str("e13,u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,\",p1,10\n");
            ev_cases.push_str("e14,u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,\"view\"x,p1,10\n");
            ev_cases.push_str("e15,u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,vi\"ew,p1,10\n");
            ev_cases.push_str(",u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,view,p1,10\n");
            // Padding rows at the end so the boundary rows above definitely go through the direct mask loop (< direct_limit)
            for i in 0..8 {
                ev_cases.push_str(&format!(
                    "e_pad{i},u1,2027-12-12T04:10:10Z,2027-12-12T04:10:11Z,view,p1,10\n"
                ));
            }

            let mut reader = crate::csv::Reader::new(64 << 10);
            let mut dates = DateCache::new();
            let ev_bytes = ev_cases.as_bytes();
            let got_ev = unsafe { run_ev_mask_pipeline(&sh, ev_bytes, 0, ev_bytes.len()) };
            let mut exp_ev = Vec::new();
            unsafe {
                let mut padded = vec![b'\n'; 64];
                padded.extend_from_slice(ev_bytes);
                padded.resize(padded.len() + 256, b'\n');
                reader.scan::<7>(padded.as_ptr().add(64), ev_bytes.len(), 0, ev_bytes.len(), |b, s, e| {
                    if let Some(c) = parse_event_row_cold(&sh, &mut dates, b, s, e) {
                        exp_ev.push((c.event_tok, c.user_tok, c.ingest, c.ts, c.payload, c.valid));
                    }
                });
            }
            assert_eq!(got_ev, exp_ev, "synthetic boundary event mismatch");

            ck_cases.push_str(&pad_to("k1,", "u1,c_1,2027-12-12T04:10:10Z", 63));
            ck_cases.push_str(&pad_to("k2,", "u1,c_1,2027-12-12T04:10:10Z", 64));
            ck_cases.push_str(&pad_to("k3,", "u1,c_1,2027-12-12T04:10:10Z", 127));
            ck_cases.push_str(&pad_to("k4,", "u1,c_1,2027-12-12T04:10:10Z", 128));
            ck_cases.push_str("\"k5\",\"u1\",\"c_1\",\"2027-12-12T04:10:10Z\"\n");
            ck_cases.push_str("k6,u1,\",2027-12-12T04:10:10Z\n");
            ck_cases.push_str("k7,u1,c_1,2027-12-12T04:10:10Z\n".repeat(8).as_str());
            let ck_bytes = ck_cases.as_bytes();
            let got_ck = unsafe { run_ck_mask_pipeline(&sh, ck_bytes, 0, ck_bytes.len()) };
            let mut exp_ck = Vec::new();
            unsafe {
                let mut padded = vec![b'\n'; 64];
                padded.extend_from_slice(ck_bytes);
                padded.resize(padded.len() + 256, b'\n');
                reader.scan::<4>(padded.as_ptr().add(64), ck_bytes.len(), 0, ck_bytes.len(), |b, s, e| {
                    if let Some(c) = parse_click_row_cold(&sh, &mut dates, b, s, e) {
                        exp_ck.push((c.click_tok, c.user_tok, c.ts, c.campaign));
                    }
                });
            }
            assert_eq!(got_ck, exp_ck, "synthetic boundary click mismatch");

            lk_cases.push_str("u1,u2\n");
            lk_cases.push_str("\"u3\",\"u4\"\n");
            lk_cases.push_str(&format!("u{},u6\n", "a".repeat(70)));
            lk_cases.push_str("u7,\"u8\n");
            lk_cases.push_str("u9,u10\n".repeat(12).as_str());
            let lk_bytes = lk_cases.as_bytes();
            let got_lk = unsafe { run_lk_mask_pipeline(lk_bytes, 0, lk_bytes.len()) };
            let mut exp_lk = Vec::new();
            unsafe {
                let mut padded = vec![b'\n'; 64];
                padded.extend_from_slice(lk_bytes);
                padded.resize(padded.len() + 256, b'\n');
                reader.scan::<2>(padded.as_ptr().add(64), lk_bytes.len(), 0, lk_bytes.len(), |b, s, e| {
                    let (la, lb) = ((e[0] - s[0]) as usize, (e[1] - s[1]) as usize);
                    if la != 0 && lb != 0 {
                        let (ta, tb) = tok_pair(b.add(s[0] as usize), la, b.add(s[1] as usize), lb);
                        if ta != tb {
                            exp_lk.push((ta, tb));
                        }
                    }
                });
            }
            assert_eq!(got_lk, exp_lk, "synthetic boundary link mismatch");

            // 2. Real dev part-00001 files (including 8 unaligned extent splits on events)
            let ev_bytes = std::fs::read(dev_dir.join("events/part-00001.csv")).unwrap();
            let got_dev_ev = unsafe { run_ev_mask_pipeline(&sh, &ev_bytes, 0, ev_bytes.len()) };
            let mut split_dev_ev = Vec::new();
            for t in 0..8 {
                let a = ev_bytes.len() * t / 8;
                let b = ev_bytes.len() * (t + 1) / 8;
                split_dev_ev.extend(unsafe { run_ev_mask_pipeline(&sh, &ev_bytes, a, b) });
            }
            assert_eq!(got_dev_ev, split_dev_ev, "dev event mask pipeline split mismatch");

            let ck_bytes = std::fs::read(dev_dir.join("clicks/part-00001.csv")).unwrap();
            let got_dev_ck = unsafe { run_ck_mask_pipeline(&sh, &ck_bytes, 0, ck_bytes.len()) };
            let mut exp_dev_ck = Vec::new();
            unsafe {
                let mut padded = vec![b'\n'; 64];
                padded.extend_from_slice(&ck_bytes);
                padded.resize(padded.len() + 256, b'\n');
                reader.scan::<4>(padded.as_ptr().add(64), ck_bytes.len(), 0, ck_bytes.len(), |b, s, e| {
                    if let Some(c) = parse_click_row_cold(&sh, &mut dates, b, s, e) {
                        exp_dev_ck.push((c.click_tok, c.user_tok, c.ts, c.campaign));
                    }
                });
            }
            assert_eq!(got_dev_ck, exp_dev_ck, "dev click mask pipeline mismatch");

            let lk_bytes = std::fs::read(dev_dir.join("links/part-00001.csv")).unwrap();
            let got_dev_lk = unsafe { run_lk_mask_pipeline(&lk_bytes, 0, lk_bytes.len()) };
            let mut exp_dev_lk = Vec::new();
            unsafe {
                let mut padded = vec![b'\n'; 64];
                padded.extend_from_slice(&lk_bytes);
                padded.resize(padded.len() + 256, b'\n');
                reader.scan::<2>(padded.as_ptr().add(64), lk_bytes.len(), 0, lk_bytes.len(), |b, s, e| {
                    let (la, lb) = ((e[0] - s[0]) as usize, (e[1] - s[1]) as usize);
                    if la != 0 && lb != 0 {
                        let (ta, tb) = tok_pair(b.add(s[0] as usize), la, b.add(s[1] as usize), lb);
                        if ta != tb {
                            exp_dev_lk.push((ta, tb));
                        }
                    }
                });
            }
            assert_eq!(got_dev_lk, exp_dev_lk, "dev link mask pipeline mismatch");
        }

        // 3. All 36 hand-written fixture suites, padded with trailing dummy rows so every real fixture row goes through the direct mask loop!
        let fixtures_root = Path::new("/home/colton/Documents/Projects/Builders/#3/fixtures");
        if fixtures_root.exists() {
            let mut reader = crate::csv::Reader::new(64 << 10);
            let mut dates = DateCache::new();
            for entry in std::fs::read_dir(fixtures_root).unwrap().flatten() {
                let in_dir = entry.path().join("input");
                if !in_dir.is_dir() {
                    continue;
                }
                let f_sh = dummy_shared(&in_dir);
                for ev_e in std::fs::read_dir(in_dir.join("events")).unwrap().flatten() {
                    if ev_e.path().extension().is_some_and(|x| x == "csv") {
                        let mut raw = std::fs::read(ev_e.path()).unwrap();
                        if raw.is_empty() {
                            continue;
                        }
                        if !raw.ends_with(b"\n") {
                            raw.push(b'\n');
                        }
                        let real_end = raw.len();
                        raw.extend_from_slice(&b"pad,pad,2027-01-01T00:00:00Z,2027-01-01T00:00:00Z,view,p1,0\n".repeat(4));
                        let got = unsafe { run_ev_mask_pipeline(&f_sh, &raw, 0, real_end) };
                        let mut exp = Vec::new();
                        unsafe {
                            let mut padded = vec![b'\n'; 64];
                            padded.extend_from_slice(&raw);
                            padded.resize(padded.len() + 256, b'\n');
                            reader.scan::<7>(padded.as_ptr().add(64), raw.len(), 0, real_end, |b, s, e| {
                                if let Some(c) = parse_event_row_cold(&f_sh, &mut dates, b, s, e) {
                                    exp.push((c.event_tok, c.user_tok, c.ingest, c.ts, c.payload, c.valid));
                                }
                            });
                        }
                        assert_eq!(got, exp, "fixture {:?} event mask helper mismatch", entry.file_name());
                    }
                }
                for ck_e in std::fs::read_dir(in_dir.join("clicks")).unwrap().flatten() {
                    if ck_e.path().extension().is_some_and(|x| x == "csv") {
                        let mut raw = std::fs::read(ck_e.path()).unwrap();
                        if raw.is_empty() {
                            continue;
                        }
                        if !raw.ends_with(b"\n") {
                            raw.push(b'\n');
                        }
                        let real_end = raw.len();
                        raw.extend_from_slice(&b"pad,pad,c_1,2027-01-01T00:00:00Z\n".repeat(5));
                        let got = unsafe { run_ck_mask_pipeline(&f_sh, &raw, 0, real_end) };
                        let mut exp = Vec::new();
                        unsafe {
                            let mut padded = vec![b'\n'; 64];
                            padded.extend_from_slice(&raw);
                            padded.resize(padded.len() + 256, b'\n');
                            reader.scan::<4>(padded.as_ptr().add(64), raw.len(), 0, real_end, |b, s, e| {
                                if let Some(c) = parse_click_row_cold(&f_sh, &mut dates, b, s, e) {
                                    exp.push((c.click_tok, c.user_tok, c.ts, c.campaign));
                                }
                            });
                        }
                        assert_eq!(got, exp, "fixture {:?} click mask helper mismatch", entry.file_name());
                    }
                }
                for lk_e in std::fs::read_dir(in_dir.join("links")).unwrap().flatten() {
                    if lk_e.path().extension().is_some_and(|x| x == "csv") {
                        let mut raw = std::fs::read(lk_e.path()).unwrap();
                        if raw.is_empty() {
                            continue;
                        }
                        if !raw.ends_with(b"\n") {
                            raw.push(b'\n');
                        }
                        let real_end = raw.len();
                        raw.extend_from_slice(&b"u0,u0\n".repeat(25));
                        let got = unsafe { run_lk_mask_pipeline(&raw, 0, real_end) };
                        let mut exp = Vec::new();
                        unsafe {
                            let mut padded = vec![b'\n'; 64];
                            padded.extend_from_slice(&raw);
                            padded.resize(padded.len() + 256, b'\n');
                            reader.scan::<2>(padded.as_ptr().add(64), raw.len(), 0, real_end, |b, s, e| {
                                let (la, lb) = ((e[0] - s[0]) as usize, (e[1] - s[1]) as usize);
                                if la != 0 && lb != 0 {
                                    let (ta, tb) = tok_pair(b.add(s[0] as usize), la, b.add(s[1] as usize), lb);
                                    if ta != tb {
                                        exp.push((ta, tb));
                                    }
                                }
                            });
                        }
                        assert_eq!(got, exp, "fixture {:?} link mask helper mismatch", entry.file_name());
                    }
                }
            }
        }
    }

    #[test]
    fn bench_fast_parse_throughput() {
        for dir_name in ["dev", "p20"] {
            let dir = Path::new("/home/colton/Documents/Projects/Builders/#3/data").join(dir_name);
            let ev_path = dir.join("events/part-00001.csv");
            if !ev_path.exists() {
                continue;
            }
            let sh = dummy_shared(&dir);
            let keys = recover_dataset_keys(&dir).expect("recover keys");
            let f = std::fs::File::open(&ev_path).unwrap();
            let sz = f.metadata().unwrap().len() as usize;
            let ptr = crate::sys::map_file(&f, sz);

            let mut reader = crate::csv::Reader::new(64 << 10);
            let mut dates = DateCache::new();
            let mut batch = FastEvBatch::new();
            let mut normal_evs = Vec::with_capacity(64);
            let mut fallback_evs = Vec::with_capacity(64);
            let mut total_normal = 0usize;
            let mut total_fallback = 0usize;

            // Warm page cache / page tables before measuring
            unsafe {
                scan_events_fast(&mut reader, &sh, &mut dates, ptr, sz, 0, sz, &mut batch, |b| {
                    std::hint::black_box(b.len);
                });
            }

            const REPS: usize = 8;
            let ctrs = crate::sys::Counters::open();
            let c0 = ctrs.read();
            let t0 = std::time::Instant::now();
            for _ in 0..REPS {
                total_normal = 0;
                total_fallback = 0;
                unsafe {
                    scan_events_fast(&mut reader, &sh, &mut dates, ptr, sz, 0, sz, &mut batch, |b| {
                        normal_evs.clear();
                        fallback_evs.clear();
                        classify_ev_batch(
                            b,
                            keys.user_key,
                            keys.event_key,
                            keys.user_cap,
                            keys.event_cap,
                            &mut normal_evs,
                            &mut fallback_evs,
                        );
                        total_normal += normal_evs.len();
                        total_fallback += fallback_evs.len();
                    });
                }
            }
            let dt = t0.elapsed() / REPS as u32;
            let c1 = ctrs.read();

            let mut idx_reader = crate::csv::Reader::new(64 << 10);
            let t_idx0 = std::time::Instant::now();
            let delims = unsafe { idx_reader.bench_index_only(ptr, sz) };
            let dt_idx = t_idx0.elapsed();

            let mut scan_reader = crate::csv::Reader::new(64 << 10);
            let mut scan_rows = 0usize;
            let c_sr0 = ctrs.read();
            let t_scan0 = std::time::Instant::now();
            for _ in 0..REPS {
                scan_rows = 0;
                unsafe {
                    scan_reader.scan::<7>(ptr, sz, 0, sz, |_buf, s, e| {
                        std::hint::black_box((s[0], e[0]));
                        scan_rows += 1;
                    });
                }
            }
            let dt_scan = t_scan0.elapsed() / REPS as u32;
            let c_sr1 = ctrs.read();

            let mut scan_only_reader = crate::csv::Reader::new(64 << 10);
            let mut scan_only_dates = DateCache::new();
            let mut scan_only_batch = FastEvBatch::new();
            let mut scan_only_rows = 0usize;
            let c_so0 = ctrs.read();
            let t_so0 = std::time::Instant::now();
            for _ in 0..REPS {
                scan_only_rows = 0;
                unsafe {
                    scan_events_fast(
                        &mut scan_only_reader,
                        &sh,
                        &mut scan_only_dates,
                        ptr,
                        sz,
                        0,
                        sz,
                        &mut scan_only_batch,
                        |b| {
                            scan_only_rows += b.len;
                            std::hint::black_box(b.canonical_valid_mask);
                        },
                    );
                }
            }
            let dt_so = t_so0.elapsed() / REPS as u32;
            let c_so1 = ctrs.read();

            crate::sys::unmap_file(ptr, sz);
            let mb = sz as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[fast_parse bench] {dir_name}/events/part-00001.csv ({:.2} MiB, {} rows, {} delims): scan_fast+classify={:?} ({:.1} MiB/s) [Mcyc={:.3}, Minstr={:.3}, Mllc={:.3}, Mbr={:.3}] | scan_fast_only={:?} ({:.1} MiB/s) [Mcyc={:.3}, Minstr={:.3}] | csv_reader_base={:?} [Mcyc={:.3}, Minstr={:.3}] (idx={:?}), scan_rows={}, so_rows={}",
                mb,
                total_normal + total_fallback,
                delims,
                dt,
                mb / dt.as_secs_f64(),
                c1[0].saturating_sub(c0[0]) as f64 / (1e6 * REPS as f64),
                c1[1].saturating_sub(c0[1]) as f64 / (1e6 * REPS as f64),
                c1[2].saturating_sub(c0[2]) as f64 / (1e6 * REPS as f64),
                c1[3].saturating_sub(c0[3]) as f64 / (1e6 * REPS as f64),
                dt_so,
                mb / dt_so.as_secs_f64(),
                c_so1[0].saturating_sub(c_so0[0]) as f64 / (1e6 * REPS as f64),
                c_so1[1].saturating_sub(c_so0[1]) as f64 / (1e6 * REPS as f64),
                dt_scan,
                c_sr1[0].saturating_sub(c_sr0[0]) as f64 / (1e6 * REPS as f64),
                c_sr1[1].saturating_sub(c_sr0[1]) as f64 / (1e6 * REPS as f64),
                dt_idx,
                scan_rows,
                scan_only_rows,
            );
        }
    }
}
