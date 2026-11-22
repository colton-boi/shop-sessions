use std::arch::x86_64::*;

pub const BAD: i64 = i64::MIN;
pub const DAY_MS: i64 = 86_400_000;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct DateSlot {
    pub ymdt: [u8; 12],
    pub sec_rel: i32,
}

#[repr(C, align(64))]
pub struct DateLut(pub [DateSlot; 128]);

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

impl DateLut {
    pub fn new(window_start: i64, window_end: i64, base_s: i64) -> Self {
        // C3: empty slots use '\n' (impossible inside CSV) and invert bit 0 of ymdt[6]
        // relative to `slot & 1` so NO 20-byte input (even raw '\n') can ever match an empty slot.
        let mut slots = [DateSlot { ymdt: [b'\n'; 12], sec_rel: 0 }; 128];
        for (i, s) in slots.iter_mut().enumerate() {
            s.ymdt[6] = b'\n' ^ ((i as u8 & 1) ^ 1);
        }
        let first_day = window_start.div_euclid(DAY_MS) - 9;
        let last_day = (window_end - 1).div_euclid(DAY_MS) + 9;
        for day in first_day..=last_day {
            let (y, m, d) = civil_from_days(day);
            if !(0..=9999).contains(&y) {
                continue;
            }
            let rel = day * 86_400 - base_s;
            if rel < i32::MIN as i64 || rel + 86_399 >= i32::MAX as i64 {
                continue;
            }
            let y = y as u32;
            let ymdt = [
                b'0' + (y / 1000) as u8,
                b'0' + ((y / 100) % 10) as u8,
                b'0' + ((y / 10) % 10) as u8,
                b'0' + (y % 10) as u8,
                b'-',
                b'0' + (m / 10) as u8,
                b'0' + (m % 10) as u8,
                b'-',
                b'0' + (d / 10) as u8,
                b'0' + (d % 10) as u8,
                b'T',
                0,
            ];
            let slot = unsafe { _pext_u32(u32::from_le_bytes([ymdt[6], ymdt[7], ymdt[8], ymdt[9]]), 0x0F03_0001) } as usize;
            if slots[slot].ymdt[0] == b'\n' {
                slots[slot] = DateSlot { ymdt, sec_rel: rel as i32 };
            }
        }
        DateLut(slots)
    }
}

/// Recently seen `YYYY-MM-DD` prefixes, so a date is validated once, not per row.
pub struct DateCache {
    k8: [u64; 64],
    k2: [u16; 64],
    days: [i64; 64],
}

impl DateCache {
    pub fn new() -> Self {
        DateCache { k8: [0; 64], k2: [0; 64], days: [0; 64] }
    }
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cold]
#[inline(never)]
unsafe fn date_slow(p: *const u8) -> Option<i64> {
    let b = std::slice::from_raw_parts(p, 10);
    if b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let mut v = [0i64; 10];
    for i in [0, 1, 2, 3, 5, 6, 8, 9] {
        if !b[i].is_ascii_digit() {
            return None;
        }
        v[i] = (b[i] - b'0') as i64;
    }
    let y = v[0] * 1000 + v[1] * 100 + v[2] * 10 + v[3];
    let m = v[5] * 10 + v[6];
    let d = v[8] * 10 + v[9];
    if !(1..=12).contains(&m) || d < 1 {
        return None;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let dim = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][(m - 1) as usize];
    if d > dim {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

/// `YYYY-MM-DDTHH:MM:SS` at `p` (19 readable bytes) as epoch seconds, or BAD.
#[inline(always)]
pub unsafe fn datetime19(p: *const u8, dc: &mut DateCache) -> i64 {
    let k8 = (p as *const u64).read_unaligned();
    let k2 = (p.add(8) as *const u16).read_unaligned();
    let slot = ((k2 as u32).wrapping_mul(0x9E37_79B1) >> 26) as usize;
    let days = if dc.k8[slot] == k8 && dc.k2[slot] == k2 {
        dc.days[slot]
    } else {
        match date_slow(p) {
            Some(d) => {
                dc.k8[slot] = k8;
                dc.k2[slot] = k2;
                dc.days[slot] = d;
                d
            }
            None => return BAD,
        }
    };
    if *p.add(10) != b'T' {
        return BAD;
    }
    // "HH:MM:SS": digits become 0..9 and the colons 0, anything else trips the nibble test.
    let y = (p.add(11) as *const u64).read_unaligned() ^ 0x3030_3A30_303A_3030;
    if (y | y.wrapping_add(0x0606_0F06_060F_0606)) & 0xF0F0_F0F0_F0F0_F0F0 != 0 {
        return BAD;
    }
    let pairs = y.wrapping_mul(10).wrapping_add(y >> 8);
    let h = (pairs & 0xFF) as i64;
    let m = ((pairs >> 24) & 0xFF) as i64;
    let s = ((pairs >> 48) & 0xFF) as i64;
    if h > 23 || m > 59 || s > 59 {
        return BAD;
    }
    days * 86_400 + h * 3600 + m * 60 + s
}

static TAIL_MASK: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255,
];

/// ASCII digits only, as a number. None when there is a non-digit, no digit at
/// all, or more than 16 digits after the leading zeros (too big for any caller).
/// The 16 bytes ending at `p + len` must be readable.
#[inline(always)]
pub unsafe fn digits(mut p: *const u8, mut len: usize) -> Option<u64> {
    if len == 0 {
        return None;
    }
    while len > 16 {
        if *p != b'0' {
            return None;
        }
        p = p.add(1);
        len -= 1;
    }
    let v = _mm_loadu_si128(p.add(len).sub(16) as *const __m128i);
    let m = _mm_loadu_si128(TAIL_MASK.as_ptr().add(len) as *const __m128i);
    let d = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
    let over = _mm_and_si128(_mm_subs_epu8(d, _mm_set1_epi8(9)), m);
    if _mm_testz_si128(over, over) == 0 {
        return None;
    }
    let d = _mm_and_si128(d, m);
    let t = _mm_maddubs_epi16(d, _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1));
    let t = _mm_madd_epi16(t, _mm_setr_epi16(100, 1, 100, 1, 100, 1, 100, 1));
    let t = _mm_packus_epi32(t, t);
    let t = _mm_madd_epi16(t, _mm_setr_epi16(10000, 1, 10000, 1, 10000, 1, 10000, 1));
    let hi = _mm_cvtsi128_si32(t) as u32 as u64;
    let lo = _mm_extract_epi32(t, 1) as u32 as u64;
    Some(hi * 100_000_000 + lo)
}

/// A `ts` field as epoch milliseconds, or BAD. Needs 16 readable bytes before
/// the field's end and 8 after its start.
#[inline(always)]
pub unsafe fn parse_ts(p: *const u8, len: usize, dc: &mut DateCache) -> i64 {
    if len == 20 && *p.add(19) == b'Z' {
        let s = datetime19(p, dc);
        return if s == BAD { BAD } else { s * 1000 };
    }
    if len == 25 && (*p.add(19) == b'+' || *p.add(19) == b'-') {
        return offset_form(p, dc);
    }
    match digits(p, len) {
        Some(v) => v as i64,
        None => BAD,
    }
}

#[inline(never)]
unsafe fn offset_form(p: *const u8, dc: &mut DateCache) -> i64 {
    let s = datetime19(p, dc);
    let o = std::slice::from_raw_parts(p.add(19), 6);
    if s == BAD || o[3] != b':' || ![o[1], o[2], o[4], o[5]].iter().all(u8::is_ascii_digit) {
        return BAD;
    }
    let oh = ((o[1] - b'0') * 10 + (o[2] - b'0')) as i64;
    let om = ((o[4] - b'0') * 10 + (o[5] - b'0')) as i64;
    if oh > 23 || om > 59 {
        return BAD;
    }
    let off = oh * 3600 + om * 60;
    (if o[0] == b'+' { s - off } else { s + off }) * 1000
}

/// `ingest_ts`: always form F1. Epoch seconds, or BAD.
#[inline(always)]
pub unsafe fn parse_f1(p: *const u8, len: usize, dc: &mut DateCache) -> i64 {
    if len == 20 && *p.add(19) == b'Z' {
        datetime19(p, dc)
    } else {
        BAD
    }
}

#[cold]
#[inline(never)]
unsafe fn parse_f1_slow_rel(p: *const u8, dc: &mut DateCache, base_s: i64) -> i32 {
    if *p.add(19) != b'Z' {
        return i32::MAX;
    }
    let s = datetime19(p, dc);
    if s == BAD {
        i32::MAX
    } else {
        (s - base_s).clamp(i32::MIN as i64, i32::MAX as i64 - 1) as i32
    }
}

#[inline(always)]
unsafe fn parse_f1_lut(p: *const u8, lut: &DateLut) -> Option<i32> {
    let slot = _pext_u32((p.add(6) as *const u32).read_unaligned(), 0x0F03_0001) as usize;
    let entry = lut.0.get_unchecked(slot);
    let d_bad = _mm_and_si128(
        _mm_xor_si128(_mm_loadu_si128(p as *const __m128i), _mm_load_si128(entry as *const DateSlot as *const __m128i)),
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0),
    );
    let c_dig = _mm_sub_epi8(
        _mm_loadu_si128(p.add(11) as *const __m128i),
        _mm_setr_epi8(48, 48, 58, 48, 48, 58, 48, 48, 90, 0, 0, 0, 0, 0, 0, 0),
    );
    let c_bad = _mm_subs_epu8(c_dig, _mm_setr_epi8(2, 9, 0, 5, 9, 0, 5, 9, 0, -1, -1, -1, -1, -1, -1, -1));
    let bad = _mm_or_si128(d_bad, c_bad);
    if _mm_testz_si128(bad, bad) == 0 {
        return None;
    }
    let t = _mm_madd_epi16(
        _mm_maddubs_epi16(c_dig, _mm_setr_epi8(10, 1, 0, 10, 1, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0)),
        _mm_setr_epi16(3600, 60, 60, 1, 0, 0, 0, 0),
    );
    let sec = _mm_cvtsi128_si32(_mm_add_epi32(t, _mm_shuffle_epi32(t, 1))) as u32;
    if sec >= 86_400 {
        return None;
    }
    Some(entry.sec_rel + sec as i32)
}

/// `ingest_ts`: always form F1. Seconds relative to `base_s` (clamped to `[i32::MIN, i32::MAX - 1]`),
/// or `i32::MAX` when BAD. Needs 27 readable bytes from `p` when `len == 20`.
#[inline(always)]
pub unsafe fn parse_f1_rel(p: *const u8, len: usize, lut: &DateLut, dc: &mut DateCache, base_s: i64) -> i32 {
    if len != 20 {
        return i32::MAX;
    }
    match parse_f1_lut(p, lut) {
        Some(s) => s,
        None => parse_f1_slow_rel(p, dc, base_s),
    }
}

/// Dual 256-bit AVX2 F1+F1 timestamp parse (`YYYY-MM-DDTHH:MM:SSZ` for both `p_ts` and `p_ing`).
/// Returns `Some((ts_s_rel, ingest_s_rel))` when both are valid 20-byte F1 timestamps in `lut`,
/// otherwise `None` (caller falls back to `parse_f1_rel` + `parse_ts_rel`).
/// Both `p_ts` and `p_ing` must be readable for at least 27 bytes.
#[inline(always)]
pub unsafe fn parse_dual_f1_rel(p_ts: *const u8, p_ing: *const u8, lut: &DateLut) -> Option<(i32, i32)> {
    let s0 = _pext_u32((p_ts.add(6) as *const u32).read_unaligned(), 0x0F03_0001) as usize;
    let s1 = _pext_u32((p_ing.add(6) as *const u32).read_unaligned(), 0x0F03_0001) as usize;
    let e0 = lut.0.get_unchecked(s0);
    let e1 = lut.0.get_unchecked(s1);
    let lut_vec = _mm256_set_m128i(
        _mm_load_si128(e1 as *const DateSlot as *const __m128i),
        _mm_load_si128(e0 as *const DateSlot as *const __m128i),
    );
    let d_in = _mm256_loadu2_m128i(p_ing as *const __m128i, p_ts as *const __m128i);
    let d_mask = _mm256_setr_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0,
    );
    let d_bad = _mm256_and_si256(_mm256_xor_si256(d_in, lut_vec), d_mask);

    let c_in = _mm256_loadu2_m128i(p_ing.add(11) as *const __m128i, p_ts.add(11) as *const __m128i);
    let c_sub = _mm256_setr_epi8(
        48, 48, 58, 48, 48, 58, 48, 48, 90, 0, 0, 0, 0, 0, 0, 0,
        48, 48, 58, 48, 48, 58, 48, 48, 90, 0, 0, 0, 0, 0, 0, 0,
    );
    let c_lim = _mm256_setr_epi8(
        2, 9, 0, 5, 9, 0, 5, 9, 0, -1, -1, -1, -1, -1, -1, -1,
        2, 9, 0, 5, 9, 0, 5, 9, 0, -1, -1, -1, -1, -1, -1, -1,
    );
    let c_dig = _mm256_sub_epi8(c_in, c_sub);
    let c_bad = _mm256_subs_epu8(c_dig, c_lim);
    let bad = _mm256_or_si256(d_bad, c_bad);
    if _mm256_testz_si256(bad, bad) == 0 {
        return None;
    }

    let mul_ub = _mm256_setr_epi8(
        10, 1, 0, 10, 1, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        10, 1, 0, 10, 1, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    );
    let mul_16 = _mm256_setr_epi16(
        3600, 60, 60, 1, 0, 0, 0, 0,
        3600, 60, 60, 1, 0, 0, 0, 0,
    );
    let t = _mm256_madd_epi16(_mm256_maddubs_epi16(c_dig, mul_ub), mul_16);
    let sum = _mm256_add_epi32(t, _mm256_shuffle_epi32(t, 1));
    let sec_ts = _mm256_cvtsi256_si32(sum) as u32;
    let sec_ing = _mm256_extract_epi32(sum, 4) as u32;
    if (sec_ts >= 86_400) | (sec_ing >= 86_400) {
        return None;
    }
    Some((e0.sec_rel + sec_ts as i32, e1.sec_rel + sec_ing as i32))
}

#[cold]
#[inline(never)]
unsafe fn parse_ts_slow_rel(p: *const u8, len: usize, dc: &mut DateCache, base_ms: i64) -> i64 {
    let t = parse_ts(p, len, dc);
    if t == BAD { BAD } else { t - base_ms }
}

static F3_13_MASK: [u8; 16] = [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

/// Exactly 13 ASCII digits at `p` as a number. Reads the 3 bytes before `p`.
#[inline(always)]
unsafe fn f3_13(p: *const u8) -> Option<u64> {
    let m = _mm_loadu_si128(F3_13_MASK.as_ptr() as *const __m128i);
    let d = _mm_sub_epi8(_mm_loadu_si128(p.sub(3) as *const __m128i), _mm_set1_epi8(b'0' as i8));
    let over = _mm_and_si128(_mm_subs_epu8(d, _mm_set1_epi8(9)), m);
    if _mm_testz_si128(over, over) == 0 {
        return None;
    }
    let d = _mm_and_si128(d, m);
    let t = _mm_maddubs_epi16(d, _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1));
    let t = _mm_madd_epi16(t, _mm_setr_epi16(100, 1, 100, 1, 100, 1, 100, 1));
    let t = _mm_packus_epi32(t, t);
    let t = _mm_madd_epi16(t, _mm_setr_epi16(10000, 1, 10000, 1, 10000, 1, 10000, 1));
    let hi = _mm_cvtsi128_si32(t) as u32 as u64;
    let lo = _mm_extract_epi32(t, 1) as u32 as u64;
    Some(hi * 100_000_000 + lo)
}

/// F2 `YYYY-MM-DDTHH:MM:SS+HH:MM` whose date is in `lut`, as seconds relative to `base_s`.
/// Reads 27 bytes from `p`.
#[inline(always)]
unsafe fn f2_lut(p: *const u8, lut: &DateLut) -> Option<i32> {
    let slot = _pext_u32((p.add(6) as *const u32).read_unaligned(), 0x0F03_0001) as usize;
    let entry = lut.0.get_unchecked(slot);
    let d_bad = _mm_and_si128(
        _mm_xor_si128(_mm_loadu_si128(p as *const __m128i), _mm_load_si128(entry as *const DateSlot as *const __m128i)),
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0),
    );
    // Bytes 11..25: HH:MM:SS, the sign (checked below), OH:OM.
    let c = _mm_sub_epi8(
        _mm_loadu_si128(p.add(11) as *const __m128i),
        _mm_setr_epi8(48, 48, 58, 48, 48, 58, 48, 48, 0, 48, 48, 58, 48, 48, 0, 0),
    );
    let c_bad = _mm_subs_epu8(c, _mm_setr_epi8(2, 9, 0, 5, 9, 0, 5, 9, -1, 2, 9, 0, 5, 9, -1, -1));
    let bad = _mm_or_si128(d_bad, c_bad);
    let sign = *p.add(19);
    if (_mm_testz_si128(bad, bad) == 0) | (sign.wrapping_sub(b'+') & 0xFD != 0) {
        return None;
    }
    let c = _mm_and_si128(c, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0));
    let t = _mm_madd_epi16(
        _mm_maddubs_epi16(c, _mm_setr_epi8(10, 1, 0, 10, 1, 0, 10, 1, 0, 10, 1, 0, 10, 1, 0, 0)),
        _mm_setr_epi16(3600, 60, 60, 1, 3600, 3600, 60, 0),
    );
    let sec = (_mm_cvtsi128_si32(t) + _mm_extract_epi32(t, 1)) as u32;
    let off = (_mm_extract_epi32(t, 2) + _mm_extract_epi32(t, 3)) as u32;
    if (sec >= 86_400) | (off >= 86_400) {
        return None;
    }
    let local = entry.sec_rel + sec as i32;
    Some(if sign == b'+' { local - off as i32 } else { local + off as i32 })
}

/// A `ts` field as milliseconds relative to `base_ms` (where `base_ms == base_s * 1000`), or BAD.
/// Needs 16 readable bytes before the field's end, 3 before `p` and 27 after it.
#[inline(always)]
pub unsafe fn parse_ts_rel(p: *const u8, len: usize, lut: &DateLut, dc: &mut DateCache, base_ms: i64) -> i64 {
    let fast = match len {
        20 => parse_f1_lut(p, lut).map(|s| s as i64 * 1000),
        13 => f3_13(p).map(|v| v as i64 - base_ms),
        25 => f2_lut(p, lut).map(|s| s as i64 * 1000),
        _ => None,
    };
    match fast {
        Some(t) => t,
        None => parse_ts_slow_rel(p, len, dc, base_ms),
    }
}

pub fn parse_ts_slice(b: &[u8]) -> i64 {
    let mut padded = [0u8; 96];
    if b.len() > 32 {
        return BAD;
    }
    padded[32..32 + b.len()].copy_from_slice(b);
    unsafe { parse_ts(padded.as_ptr().add(32), b.len(), &mut DateCache::new()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_civil_roundtrip_and_lut_collision_free() {
        for day in 0..50_000i64 {
            let (y, m, d) = civil_from_days(day);
            assert_eq!(days_from_civil(y, m as i64, d as i64), day);
        }
        // Check zero slot collisions across every 17-day window from 2024 to 2035 (incl. Feb 29, 2028)
        let start = days_from_civil(2024, 1, 1);
        let end = days_from_civil(2035, 12, 31);
        for d in start..=end {
            let ws = d * DAY_MS;
            let we = (d + 17) * DAY_MS;
            let base_s = (ws - 7 * DAY_MS) / 1000;
            let lut = DateLut::new(ws, we, base_s);
            let mut count = 0usize;
            for slot in &lut.0 {
                if slot.ymdt[0] != b'\n' {
                    count += 1;
                }
            }
            // first_day = d - 9, last_day = (d + 16) + 9 => 35 days total
            assert_eq!(count, 35, "collision in DateLut at day {d}");
        }
    }

    #[test]
    fn test_parse_ts_rel_fuzz() {
        let ws = days_from_civil(2028, 2, 20) * DAY_MS;
        let we = days_from_civil(2028, 3, 5) * DAY_MS;
        let base_ms = ws - 7 * DAY_MS;
        let lut = DateLut::new(ws, we, base_ms / 1000);
        let mut dc1 = DateCache::new();
        let mut dc2 = DateCache::new();
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alphabet = b"0123456789:-+TZz \r\n\0,\"./e";
        for i in 0..3_000_000u64 {
            let t = (base_ms + (rnd() % (30 * DAY_MS as u64)) as i64) / 1000;
            let day = t.div_euclid(86_400);
            let (y, m, d) = civil_from_days(day);
            let sod = t.rem_euclid(86_400);
            let mut s = match i % 3 {
                0 => format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", sod / 3600, sod / 60 % 60, sod % 60),
                1 => {
                    let oh = rnd() % 26;
                    let om = rnd() % 61;
                    let sg = if rnd() & 1 == 0 { '+' } else { '-' };
                    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}{sg}{oh:02}:{om:02}", sod / 3600, sod / 60 % 60, sod % 60)
                }
                _ => format!("{}", t * 1000 + (rnd() % 1000) as i64),
            }
            .into_bytes();
            let muts = rnd() % 4;
            for _ in 0..muts {
                if s.is_empty() {
                    break;
                }
                let k = (rnd() as usize) % s.len();
                match rnd() % 5 {
                    0 => s[k] = alphabet[(rnd() as usize) % alphabet.len()],
                    1 => s[k] = (rnd() & 0xFF) as u8,
                    2 => {
                        s.remove(k);
                    }
                    3 => s.insert(k, b'0'),
                    _ => s[k] = s[k].wrapping_add(1),
                }
            }
            if s.len() > 32 {
                continue;
            }
            let mut buf = [b'x'; 96];
            buf[32..32 + s.len()].copy_from_slice(&s);
            let p = unsafe { buf.as_ptr().add(32) };
            let ts_old = unsafe { parse_ts(p, s.len(), &mut dc1) };
            let ts_exp = if ts_old == BAD { BAD } else { ts_old - base_ms };
            let ts_new = unsafe { parse_ts_rel(p, s.len(), &lut, &mut dc2, base_ms) };
            assert_eq!(ts_new, ts_exp, "parse_ts_rel mismatch on {:?}", String::from_utf8_lossy(&s));
        }
    }

    #[test]
    fn test_parse_rel_equivalence() {
        let ws = days_from_civil(2028, 2, 20) * DAY_MS; // crosses 2028-02-29!
        let we = days_from_civil(2028, 3, 5) * DAY_MS;
        let base_ms = ws - 7 * DAY_MS;
        let base_s = base_ms / 1000;
        let lut = DateLut::new(ws, we, base_s);
        let mut dc1 = DateCache::new();
        let mut dc2 = DateCache::new();

        let cases: &[&[u8]] = &[
            b"2028-02-29T23:59:59Z",
            b"2028-02-28T00:00:00Z",
            b"2028-03-01T12:34:56Z",
            b"2028-02-13T00:00:00Z",
            b"2028-02-11T00:00:00Z",
            b"2028-03-13T23:59:59Z",
            b"2028-02-29T24:00:00Z",
            b"2028-02-29T23:60:00Z",
            b"2028-02-29T23:59:60Z",
            b"2027-02-29T00:00:00Z",
            b"2028-02-29T00:00:00z",
            b"2028-02-29 00:00:00Z",
            b"2028-02-29T00:00:00Z\r",
            b"\0\0\0\0\0\0\0\0\0\0\000:00:00Z",
            b"\n\n\n\n\n\n\n\n\n\n\n00:00:00Z",
            b"1970-01-01T00:00:00Z",
            b"1900-01-01T00:00:00Z",
            b"3000-01-01T00:00:00Z",
            b"2028-02-29T19:58:10-04:00",
            b"2028-02-29T19:58:10+00:00",
            b"2028-02-29T19:58:10-00:00",
            b"2028-02-29T19:58:10+23:59",
            b"2028-02-29T19:58:10+24:00",
            b"2028-02-29T19:58:10+05:60",
            b"2028-02-29T19:58:10+0200",
            b"2028-02-29T19:58:10*04:00",
            b"2028-02-29T19:58:10,04:00",
            b"2028-02-29T19:58:10+04;00",
            b"2028-02-29T19:58:10+4:000",
            b"2028-02-29T23:59:59-23:59",
            b"2028-02-21T00:00:00+12:00",
            b"2028-02-29T24:00:00+00:00",
            b"2028-02-29T19:60:10+00:00",
            b"2028-02-29T19:58:10+00:00\r",
            b"2027-02-29T19:58:10+00:00",
            b"2028-02-29T19:58:10Z+0:00",
            b"1791071890000",
            b"0791071890000",
            b"179107189000a",
            b"17910718900 0",
            b"1791071890000\r",
            b"0000000000000",
            b"9999999999999",
            b"1791071890000",
            b"00000001791071890000",
            b"0000000000001791071890000",
            b"86400000",
            b"9999999999999999999999999",
            b"1791071890000.0",
            b"-5",
            b"",
            "٢٥٠".as_bytes(),
        ];

        for &s in cases {
            let mut buf = [0u8; 96];
            buf[32..32 + s.len()].copy_from_slice(s);
            let p = unsafe { buf.as_ptr().add(32) };

            let f1_old = unsafe { parse_f1(p, s.len(), &mut dc1) };
            let f1_exp = if f1_old == BAD { i32::MAX } else { (f1_old - base_s).clamp(i32::MIN as i64, i32::MAX as i64 - 1) as i32 };
            let f1_new = unsafe { parse_f1_rel(p, s.len(), &lut, &mut dc2, base_s) };
            assert_eq!(f1_new, f1_exp, "parse_f1_rel mismatch on {:?}", String::from_utf8_lossy(s));

            let ts_old = unsafe { parse_ts(p, s.len(), &mut dc1) };
            let ts_exp = if ts_old == BAD { BAD } else { ts_old - base_ms };
            let ts_new = unsafe { parse_ts_rel(p, s.len(), &lut, &mut dc2, base_ms) };
            assert_eq!(ts_new, ts_exp, "parse_ts_rel mismatch on {:?}", String::from_utf8_lossy(s));
        }
    }

    #[test]
    fn test_parse_dual_f1_rel_exhaustive() {
        let ws = days_from_civil(2028, 2, 20) * DAY_MS; // crosses 2028-02-29
        let we = days_from_civil(2028, 3, 5) * DAY_MS;
        let base_ms = ws - 7 * DAY_MS;
        let base_s = base_ms / 1000;
        let lut = DateLut::new(ws, we, base_s);
        let mut dc = DateCache::new();

        let mut samples: Vec<Vec<u8>> = Vec::new();
        // Valid LUT dates at midnight boundaries and random times
        for day_off in -9..=23i64 {
            let day = ws.div_euclid(DAY_MS) + day_off;
            let (y, m, d) = civil_from_days(day);
            for (hh, mm, ss) in [(0, 0, 0), (0, 0, 1), (12, 34, 56), (23, 59, 58), (23, 59, 59)] {
                samples.push(format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z").into_bytes());
            }
        }
        // Invalid hours, mins, secs, delimiters, non-LUT dates, control chars
        let bad_cases: &[&[u8]] = &[
            b"2028-02-29T24:00:00Z",
            b"2028-02-29T25:00:00Z",
            b"2028-02-29T29:59:59Z",
            b"2028-02-29T30:00:00Z",
            b"2028-02-29T23:60:00Z",
            b"2028-02-29T23:99:00Z",
            b"2028-02-29T23:59:60Z",
            b"2028-02-29T23:59:99Z",
            b"2028-02-29T00:00:00z",
            b"2028-02-29 00:00:00Z",
            b"2028-02-29T00;00:00Z",
            b"2028-02-29T00:00;00Z",
            b"2028/02/29T00:00:00Z",
            b"2028-02-29T0a:00:00Z",
            b"2028-02-29T/0:00:00Z",
            b"2027-02-29T00:00:00Z",
            b"2028-02-10T00:00:00Z", // before LUT
            b"2028-03-15T00:00:00Z", // after LUT
            b"1970-01-01T00:00:00Z",
            b"\n\n\n\n\n\n\n\n\n\n\n00:00:00Z",
            b"\n\n\n\n\n\n\x0b\n\n\n\n00:00:00Z",
        ];
        for &c in bad_cases {
            samples.push(c.to_vec());
        }

        // Compute expected parse_f1_lut for each sample
        let mut expected: Vec<Option<i32>> = Vec::with_capacity(samples.len());
        for s in &samples {
            assert_eq!(s.len(), 20);
            let mut buf = [0u8; 64];
            buf[..20].copy_from_slice(s);
            let exp_lut = unsafe { parse_f1_lut(buf.as_ptr(), &lut) };
            // Also verify against scalar parse_f1_rel whenever LUT matches
            let exp_scalar = unsafe { parse_f1_rel(buf.as_ptr(), 20, &lut, &mut dc, base_s) };
            if let Some(v) = exp_lut {
                assert_eq!(v, exp_scalar, "LUT vs scalar mismatch on {:?}", String::from_utf8_lossy(s));
            }
            expected.push(exp_lut);
        }

        // Test every pair (i, j)
        for i in 0..samples.len() {
            let mut b0 = [0u8; 64];
            b0[..20].copy_from_slice(&samples[i]);
            for j in 0..samples.len() {
                let mut b1 = [0u8; 64];
                b1[..20].copy_from_slice(&samples[j]);
                let got = unsafe { parse_dual_f1_rel(b0.as_ptr(), b1.as_ptr(), &lut) };
                let exp = match (expected[i], expected[j]) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None,
                };
                assert_eq!(
                    got,
                    exp,
                    "dual mismatch on {:?} + {:?}",
                    String::from_utf8_lossy(&samples[i]),
                    String::from_utf8_lossy(&samples[j])
                );
            }
        }
    }
}
