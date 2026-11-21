use std::arch::x86_64::*;

pub const BAD: i64 = i64::MIN;
pub const DAY_MS: i64 = 86_400_000;

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

pub fn parse_ts_slice(b: &[u8]) -> i64 {
    let mut padded = [0u8; 96];
    if b.len() > 32 {
        return BAD;
    }
    padded[32..32 + b.len()].copy_from_slice(b);
    unsafe { parse_ts(padded.as_ptr().add(32), b.len(), &mut DateCache::new()) }
}
