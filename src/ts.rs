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

#[inline(never)]
unsafe fn offset_form_rel(p: *const u8, lut: &DateLut, dc: &mut DateCache, base_s: i64) -> i64 {
    let slot = _pext_u32((p.add(6) as *const u32).read_unaligned(), 0x0F03_0001) as usize;
    let entry = lut.0.get_unchecked(slot);
    let d_bad = _mm_and_si128(
        _mm_xor_si128(_mm_loadu_si128(p as *const __m128i), _mm_load_si128(entry as *const DateSlot as *const __m128i)),
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0),
    );
    let s_rel = if _mm_testz_si128(d_bad, d_bad) != 0 {
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
        entry.sec_rel as i64 + h * 3600 + m * 60 + s
    } else {
        let s = datetime19(p, dc);
        if s == BAD {
            return BAD;
        }
        s - base_s
    };
    let o = std::slice::from_raw_parts(p.add(19), 6);
    if o[3] != b':' || ![o[1], o[2], o[4], o[5]].iter().all(u8::is_ascii_digit) {
        return BAD;
    }
    let oh = ((o[1] - b'0') * 10 + (o[2] - b'0')) as i64;
    let om = ((o[4] - b'0') * 10 + (o[5] - b'0')) as i64;
    if oh > 23 || om > 59 {
        return BAD;
    }
    let off = oh * 3600 + om * 60;
    (if o[0] == b'+' { s_rel - off } else { s_rel + off }) * 1000
}

#[cold]
#[inline(never)]
unsafe fn parse_ts_slow_rel(p: *const u8, len: usize, dc: &mut DateCache, base_ms: i64) -> i64 {
    let t = parse_ts(p, len, dc);
    if t == BAD { BAD } else { t - base_ms }
}

/// A `ts` field as milliseconds relative to `base_ms` (where `base_ms == base_s * 1000`), or BAD.
/// Needs 16 readable bytes before the field's end and 16 after `p` / `p + 11` for F1/F2.
#[inline(always)]
pub unsafe fn parse_ts_rel(p: *const u8, len: usize, lut: &DateLut, dc: &mut DateCache, base_ms: i64) -> i64 {
    if len == 20 && *p.add(19) == b'Z' {
        return match parse_f1_lut(p, lut) {
            Some(s) => (s as i64) * 1000,
            None => parse_ts_slow_rel(p, len, dc, base_ms),
        };
    }
    if len == 25 && (*p.add(19) == b'+' || *p.add(19) == b'-') {
        return offset_form_rel(p, lut, dc, base_ms / 1000);
    }
    match digits(p, len) {
        Some(v) => (v as i64) - base_ms,
        None => BAD,
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
}
