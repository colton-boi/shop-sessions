use std::arch::x86_64::*;
use std::collections::HashMap;
use std::sync::Mutex;

/// An ID token is the ID itself when it is 1 to 9 ASCII bytes (7 bits each,
/// first byte most significant, so unsigned order is byte-wise order), and
/// otherwise `FALLBACK | index` into the interner.
pub const FALLBACK: u64 = 1 << 63;

struct Interner {
    map: HashMap<Box<[u8]>, u32>,
    list: Vec<Box<[u8]>>,
}

static INTERN: Mutex<Option<Interner>> = Mutex::new(None);

#[cold]
#[inline(never)]
fn intern(bytes: &[u8]) -> u64 {
    let mut guard = INTERN.lock().unwrap();
    let it = guard.get_or_insert_with(|| Interner { map: HashMap::new(), list: Vec::new() });
    if let Some(&i) = it.map.get(bytes) {
        return FALLBACK | i as u64;
    }
    let i = it.list.len() as u32;
    it.list.push(bytes.into());
    it.map.insert(bytes.into(), i);
    FALLBACK | i as u64
}

/// `p` must be readable for 9 bytes whatever `len` is. Returns 0 when the ID does not pack.
/// Branch-free: ID lengths 7 to 9 are mixed at random, so any branch on `len` mispredicts.
#[inline(always)]
unsafe fn pack(p: *const u8, len: usize) -> u64 {
    // Both loads are unconditional so the len == 9 choices compile to cmov, not a mispredicting branch.
    let raw = (p as *const u64).read_unaligned();
    let raw9 = *p.add(8) as u64;
    // BZHI saturates for an index of 64 or more, so len 8 and 9 keep all eight bytes.
    let keep = _bzhi_u64(u64::MAX, (len as u32) << 3);
    let x = raw & keep;
    let is9 = len == 9;
    let ninth = if is9 { raw9 } else { 0 };
    // Bytes past `len` read as 1 here, so the zero test sees only real bytes.
    let filled = x | (!keep & 0x0101_0101_0101_0101);
    let n9_bad = if is9 { raw9.wrapping_sub(1) | raw9 } else { 0 };
    let bad = ((filled.wrapping_sub(0x0101_0101_0101_0101) | filled) & 0x8080_8080_8080_8080) | (n9_bad & 0x80);
    if bad != 0 {
        return 0;
    }
    (_pext_u64(x.swap_bytes(), 0x7F7F_7F7F_7F7F_7F7F) << 7) | ninth
}

/// `p` must be readable for 9 bytes whatever `len` is; `len` is at least 1.
#[inline(always)]
pub unsafe fn tok(p: *const u8, len: usize) -> u64 {
    if len <= 9 {
        let t = pack(p, len);
        if t != 0 {
            return t;
        }
    }
    intern(std::slice::from_raw_parts(p, len))
}

/// `(tok(p0, l0), tok(p1, l1))`, except that an empty ID gives 0 instead of an interned token.
/// Both IDs pack in one AVX2 pass with no `pext`; any ID that does not pack (over 9 bytes,
/// a NUL or a non-ASCII byte) takes the scalar path. `p0` and `p1` must be readable for 16 bytes.
#[inline(always)]
pub unsafe fn tok_pair(p0: *const u8, l0: usize, p1: *const u8, l1: usize) -> (u64, u64) {
    let raw = _mm256_loadu2_m128i(p1 as *const __m128i, p0 as *const __m128i);
    let lens = _mm256_setr_m128i(_mm_set1_epi8(l0.min(16) as i8), _mm_set1_epi8(l1.min(16) as i8));
    let idx = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let keep = _mm256_cmpgt_epi8(lens, idx);
    let x = _mm256_and_si256(raw, keep);
    // Bytes past the ID read as 1, so a 0 or a high bit anywhere is a byte that does not pack.
    let filled = _mm256_or_si256(x, _mm256_andnot_si256(keep, _mm256_set1_epi8(1)));
    let bad = _mm256_movemask_epi8(_mm256_or_si256(filled, _mm256_cmpeq_epi8(filled, _mm256_setzero_si256())));
    if (bad != 0) | (l0.max(l1) > 9) {
        return tok_pair_slow(p0, l0, p1, l1);
    }
    // Words (b0, b1): b0 << 7 | b1; then pairs of words: w0 << 14 | w1, so each dword packs four bytes.
    let w = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(x, _mm256_set1_epi16(0xFF)), 7), _mm256_srli_epi16(x, 8));
    let d = _mm256_madd_epi16(w, _mm256_setr_epi16(16384, 1, 16384, 1, 16384, 1, 16384, 1, 16384, 1, 16384, 1, 16384, 1, 16384, 1));
    // Lane qword 0 holds (bytes 0-3, bytes 4-7) as two 28-bit dwords; byte 8 sits in qword 1.
    let hi = _mm256_slli_epi64(d, 35);
    let lo = _mm256_slli_epi64(_mm256_srli_epi64(d, 32), 7);
    let ninth = _mm256_shuffle_epi8(x, _mm256_setr_epi8(8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    let t = _mm256_or_si256(_mm256_or_si256(hi, lo), ninth);
    (_mm256_extract_epi64(t, 0) as u64, _mm256_extract_epi64(t, 2) as u64)
}

#[cold]
#[inline(never)]
unsafe fn tok_pair_slow(p0: *const u8, l0: usize, p1: *const u8, l1: usize) -> (u64, u64) {
    (if l0 == 0 { 0 } else { tok(p0, l0) }, if l1 == 0 { 0 } else { tok(p1, l1) })
}

pub fn tok_bytes(t: u64, out: &mut Vec<u8>) {
    out.clear();
    if t & FALLBACK != 0 {
        let guard = INTERN.lock().unwrap();
        out.extend_from_slice(&guard.as_ref().unwrap().list[(t & 0xFFFF_FFFF) as usize]);
        return;
    }
    for i in 0..9 {
        let c = ((t >> (7 * (8 - i))) & 0x7F) as u8;
        if c == 0 {
            break;
        }
        out.push(c);
    }
}

#[cold]
#[inline(never)]
fn less_slow(a: u64, b: u64) -> bool {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    tok_bytes(a, &mut x);
    tok_bytes(b, &mut y);
    x < y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tok_pair_matches_tok() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alpha = b"abcdefghijklmnopqrstuvwxyz0123456789ekuU _\r";
        for i in 0..2_000_000u64 {
            let mut buf = [0u8; 64];
            for b in buf.iter_mut() {
                *b = match rnd() % 8 {
                    0 => (rnd() & 0xFF) as u8,
                    1 => 0,
                    2 => 0x7F,
                    3 => 0x80,
                    _ => alpha[(rnd() as usize) % alpha.len()],
                };
            }
            if i % 2 == 0 {
                for b in buf.iter_mut() {
                    *b = alpha[(*b as usize) % alpha.len()];
                }
            }
            let l0 = (rnd() % 13) as usize;
            let l1 = (rnd() % 13) as usize;
            let (p0, p1) = unsafe { (buf.as_ptr(), buf.as_ptr().add(20)) };
            let (a, b) = unsafe { tok_pair(p0, l0, p1, l1) };
            let ea = if l0 == 0 { 0 } else { unsafe { tok(p0, l0) } };
            let eb = if l1 == 0 { 0 } else { unsafe { tok(p1, l1) } };
            assert_eq!((a, b), (ea, eb), "l0 {l0} l1 {l1} {:?}", &buf[..36]);
        }
    }
}

/// Byte-wise order of the two IDs.
#[inline(always)]
pub fn tok_less(a: u64, b: u64) -> bool {
    if (a | b) & FALLBACK == 0 {
        a < b
    } else {
        less_slow(a, b)
    }
}
