use std::arch::x86_64::{_bzhi_u64, _pext_u64};
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

/// Byte-wise order of the two IDs.
#[inline(always)]
pub fn tok_less(a: u64, b: u64) -> bool {
    if (a | b) & FALLBACK == 0 {
        a < b
    } else {
        less_slow(a, b)
    }
}
