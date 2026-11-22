use crate::scatter::Arena;
use crate::sys::{map_zeroed, map_zeroed_huge, mix};
use std::arch::x86_64::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{
    fence, AtomicBool, AtomicPtr, AtomicU64, AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst},
};
use std::sync::OnceLock;

/// Token tag bits (bits 63..62):
/// - `00` / `01` (bit 63 == 0): 1..9 ASCII bytes packed 7 bits each, MSB-first (`0 < t < 2^63`).
/// - `10` (`TAG_HEX`): 33-byte hex ID `[eukEUK]` + 32 lowercase hex digits `0-9a-f`.
/// - `11` (`TAG_FALLBACK`): arbitrary byte string in lock-free `FallbackTable`.
pub const TAG_HEX: u64 = 2 << 62;
pub const TAG_FALLBACK: u64 = 3 << 62;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Hex128 {
    pub hi: u64,
    pub lo: u64,
}

#[derive(Clone, Copy, Default)]
pub struct SideWriter {
    next: u32,
    end: u32,
}

impl SideWriter {
    #[inline(always)]
    pub unsafe fn push(&mut self, arena: &Arena<Hex128>, hi: u64, lo: u64) -> u32 {
        if self.next == self.end {
            self.refill(arena);
        }
        let at = self.next;
        self.next = at + 1;
        arena.base.add(at as usize).write(Hex128 { hi, lo });
        at
    }

    #[cold]
    #[inline(never)]
    fn refill(&mut self, arena: &Arena<Hex128>) {
        let at = arena.take();
        self.next = at as u32;
        self.end = (at + arena.chunk()) as u32;
    }
}

/// Validates and decodes a 33-byte hex ID (`[eukEUK]` + 32 lowercase hex digits `0-9a-f`)
/// into `(prefix, hi, lo)` using AVX2 SIMD.
#[inline(always)]
pub unsafe fn parse_hex33(p: *const u8, len: usize) -> Option<(u8, u64, u64)> {
    if len != 33 {
        return None;
    }
    let prefix = *p;
    let lower = prefix | 0x20;
    if lower != b'e' && lower != b'u' && lower != b'k' {
        return None;
    }
    let v = _mm256_loadu_si256(p.add(1) as *const __m256i);
    // Range-check every byte against '0'..='9' and 'a'..='f'.
    let d1 = _mm256_subs_epu8(_mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8)), _mm256_set1_epi8(9));
    let d2 = _mm256_subs_epu8(_mm256_sub_epi8(v, _mm256_set1_epi8(b'a' as i8)), _mm256_set1_epi8(5));
    let err = _mm256_min_epu8(d1, d2);
    if _mm256_testz_si256(err, err) == 0 {
        return None;
    }
    // Convert ASCII hex chars to nibbles 0..=15.
    let lo_nib = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
    let is_alpha = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(b'9' as i8));
    let nibs = _mm256_add_epi8(lo_nib, _mm256_and_si256(is_alpha, _mm256_set1_epi8(9)));
    // Pair adjacent nibbles: byte[2k] * 16 + byte[2k+1].
    let bytes = _mm256_maddubs_epi16(nibs, _mm256_set1_epi16(0x0110));
    let pack_bswap = _mm256_setr_epi8(
        14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1,
        14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let packed = _mm256_shuffle_epi8(bytes, pack_bswap);
    let hi = _mm256_extract_epi64(packed, 0) as u64;
    let lo = _mm256_extract_epi64(packed, 2) as u64;
    Some((prefix, hi, lo))
}

/// Encodes `(prefix, hi, lo)` as 33 ASCII bytes (`prefix` + 32 lowercase hex digits)
/// into `out` using AVX2 `vpshufb` nibble lookup (zero branches).
#[inline(always)]
pub unsafe fn encode_hex33(prefix: u8, hi: u64, lo: u64, out: *mut u8) {
    *out = prefix;
    let x = _mm256_setr_epi64x(hi as i64, 0, lo as i64, 0);
    let dup_mask = _mm256_setr_epi8(
        7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0,
        7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0,
    );
    let dup = _mm256_shuffle_epi8(x, dup_mask);
    let hi_nibs = _mm256_and_si256(_mm256_srli_epi16(dup, 4), _mm256_set1_epi16(0x000F));
    let lo_nibs = _mm256_and_si256(dup, _mm256_set1_epi16(0x0F00));
    let nibs = _mm256_or_si256(hi_nibs, lo_nibs);
    let hex_lut = _mm256_setr_epi8(
        b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8,
        b'4' as i8, b'5' as i8, b'6' as i8, b'7' as i8,
        b'8' as i8, b'9' as i8, b'a' as i8, b'b' as i8,
        b'c' as i8, b'd' as i8, b'e' as i8, b'f' as i8,
        b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8,
        b'4' as i8, b'5' as i8, b'6' as i8, b'7' as i8,
        b'8' as i8, b'9' as i8, b'a' as i8, b'b' as i8,
        b'c' as i8, b'd' as i8, b'e' as i8, b'f' as i8,
    );
    let ascii = _mm256_shuffle_epi8(hex_lut, nibs);
    _mm256_storeu_si256(out.add(1) as *mut __m256i, ascii);
}

#[inline(always)]
fn hex_char(nib: u8) -> u64 {
    (if nib < 10 { b'0' + nib } else { b'a' - 10 + nib }) as u64
}

/// Returns the 63-bit packed representation of the first 9 bytes of a 33B hex ID.
#[inline(always)]
pub fn head9_of_hex(prefix: u8, hi: u64) -> u64 {
    let mut acc = (prefix as u64) << 56;
    acc |= hex_char(((hi >> 60) & 0xF) as u8) << 49;
    acc |= hex_char(((hi >> 56) & 0xF) as u8) << 42;
    acc |= hex_char(((hi >> 52) & 0xF) as u8) << 35;
    acc |= hex_char(((hi >> 48) & 0xF) as u8) << 28;
    acc |= hex_char(((hi >> 44) & 0xF) as u8) << 21;
    acc |= hex_char(((hi >> 40) & 0xF) as u8) << 14;
    acc |= hex_char(((hi >> 36) & 0xF) as u8) << 7;
    acc |= hex_char(((hi >> 32) & 0xF) as u8);
    acc
}

#[inline(always)]
fn hash_hex(prefix: u8, hi: u64, lo: u64) -> u64 {
    mix(hi ^ lo.rotate_left(23) ^ (prefix as u64).wrapping_mul(0x517C_C1B7_2722_0A95))
}

#[inline(always)]
fn make_order_hex_tok(prefix: u8, hi: u64, idx: u32) -> u64 {
    TAG_HEX | ((prefix as u64) << 55) | ((hi >> 41) << 32) | (idx as u64)
}

#[repr(C, align(64))]
struct AlignedUsize(AtomicUsize);

#[repr(C, align(64))]
struct AlignedBool(AtomicBool);

#[repr(C)]
struct HexSlot {
    /// 0 = empty, 1 = writing, (h & !0x3FF) | ((prefix as u64) << 2) | 2 = ready
    state: AtomicU64,
    hi: UnsafeCell<u64>,
    lo: UnsafeCell<u64>,
}

struct HexSegment {
    slots: *mut HexSlot,
    cap: usize,
    mask: usize,
    base_idx: usize,
    max_load: usize,
    seal_mask: usize,
    count: [AlignedUsize; 64],
    in_flight: [AlignedUsize; 64],
    sealed: AlignedBool,
    next: AtomicPtr<HexSegment>,
}
unsafe impl Sync for HexSegment {}
unsafe impl Send for HexSegment {}

impl HexSegment {
    fn new(cap: usize, base_idx: usize) -> Self {
        debug_assert!(cap.is_power_of_two());
        let bytes = cap * std::mem::size_of::<HexSlot>();
        let slots = if bytes >= (2 << 20) { map_zeroed_huge(bytes) } else { map_zeroed(bytes) } as *mut HexSlot;
        HexSegment {
            slots,
            cap,
            mask: cap - 1,
            base_idx,
            max_load: cap * 3 / 4,
            seal_mask: if cap >= 16384 { 63 } else { 0 },
            count: std::array::from_fn(|_| AlignedUsize(AtomicUsize::new(0))),
            in_flight: std::array::from_fn(|_| AlignedUsize(AtomicUsize::new(0))),
            sealed: AlignedBool(AtomicBool::new(false)),
            next: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

/// Lock-free fixed-width open-addressed table mapping `(prefix, hi, lo)` -> `u32`
/// with zero string copies or variable-length comparisons.
pub struct HexUserTable {
    head: HexSegment,
}
unsafe impl Sync for HexUserTable {}
unsafe impl Send for HexUserTable {}

impl HexUserTable {
    fn with_cap(cap: usize) -> Self {
        HexUserTable { head: HexSegment::new(cap.next_power_of_two().max(65536), 0) }
    }

    fn for_bytes(total_input_bytes: usize) -> Self {
        let cap = (total_input_bytes / 1536).next_power_of_two().clamp(1 << 16, 1 << 23);
        Self::with_cap(cap)
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub unsafe fn intern(&self, prefix: u8, hi: u64, lo: u64) -> u32 {
        self.intern_tid(prefix, hi, lo, 0)
    }

    #[inline(always)]
    pub unsafe fn intern_tid(&self, prefix: u8, hi: u64, lo: u64, tid: usize) -> u32 {
        let h = hash_hex(prefix, hi, lo);
        let ready_tag = (h & !0x3FF) | ((prefix as u64) << 2) | 2;
        let mut seg = &self.head;
        loop {
            let mut idx = (h as usize) & seg.mask;
            loop {
                let slot = &*seg.slots.add(idx);
                let mut st = slot.state.load(Acquire);
                while st == 1 {
                    std::hint::spin_loop();
                    st = slot.state.load(Acquire);
                }
                if st == 0 {
                    break;
                }
                if st == ready_tag && *slot.hi.get() == hi && *slot.lo.get() == lo {
                    return (seg.base_idx + idx) as u32;
                }
                idx = (idx + 1) & seg.mask;
            }
            if let Some(found) = Self::insert_in_segment(seg, hi, lo, ready_tag, idx, tid) {
                return found;
            }
            seg = Self::next_segment(seg);
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn insert_in_segment(
        seg: &HexSegment,
        hi: u64,
        lo: u64,
        ready_tag: u64,
        mut idx: usize,
        tid: usize,
    ) -> Option<u32> {
        let shard = tid & 63;
        seg.in_flight[shard].0.fetch_add(1, SeqCst);
        if !seg.sealed.0.load(Acquire) {
            loop {
                let slot = &*seg.slots.add(idx);
                let mut st = slot.state.load(Acquire);
                while st == 1 {
                    std::hint::spin_loop();
                    st = slot.state.load(Acquire);
                }
                if st == 0 {
                    match slot.state.compare_exchange_weak(0, 1, AcqRel, Acquire) {
                        Ok(_) => {
                            *slot.hi.get() = hi;
                            *slot.lo.get() = lo;
                            slot.state.store(ready_tag, Release);
                            let c = seg.count[shard].0.fetch_add(1, Relaxed) + 1;
                            if (c & seg.seal_mask) == 0 {
                                let total: usize = seg.count.iter().map(|s| s.0.load(Relaxed)).sum();
                                if total >= seg.max_load {
                                    seg.sealed.0.store(true, SeqCst);
                                }
                            }
                            seg.in_flight[shard].0.fetch_sub(1, Release);
                            return Some((seg.base_idx + idx) as u32);
                        }
                        Err(_) => continue,
                    }
                }
                if st == ready_tag && *slot.hi.get() == hi && *slot.lo.get() == lo {
                    seg.in_flight[shard].0.fetch_sub(1, Release);
                    return Some((seg.base_idx + idx) as u32);
                }
                idx = (idx + 1) & seg.mask;
            }
        }
        seg.in_flight[shard].0.fetch_sub(1, Release);
        fence(SeqCst);
        for inf in &seg.in_flight {
            while inf.0.load(Acquire) != 0 {
                std::hint::spin_loop();
            }
        }
        // Zero writers can now touch `seg`; final read-only probe from `idx`.
        loop {
            let slot = &*seg.slots.add(idx);
            let st = slot.state.load(Acquire);
            if st == 0 {
                return None;
            }
            if st == ready_tag && *slot.hi.get() == hi && *slot.lo.get() == lo {
                return Some((seg.base_idx + idx) as u32);
            }
            idx = (idx + 1) & seg.mask;
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn next_segment(seg: &HexSegment) -> &HexSegment {
        let p = seg.next.load(Acquire);
        if !p.is_null() {
            return &*p;
        }
        let next = Box::into_raw(Box::new(HexSegment::new(seg.cap * 2, seg.base_idx + seg.cap)));
        match seg.next.compare_exchange(std::ptr::null_mut(), next, AcqRel, Acquire) {
            Ok(_) => &*next,
            Err(existing) => {
                // Losing race: drop unused allocation cleanly.
                let boxed = Box::from_raw(next);
                crate::sys::release(boxed.slots as *mut u8, boxed.cap * std::mem::size_of::<HexSlot>());
                &*existing
            }
        }
    }

    #[inline(always)]
    pub unsafe fn get(&self, global_idx: u32) -> (u8, u64, u64) {
        let g = global_idx as usize;
        if g < self.head.cap {
            let s = &*self.head.slots.add(g);
            let prefix = ((s.state.load(Relaxed) >> 2) & 0xFF) as u8;
            return (prefix, *s.hi.get(), *s.lo.get());
        }
        self.get_slow(g)
    }

    #[cold]
    #[inline(never)]
    unsafe fn get_slow(&self, g: usize) -> (u8, u64, u64) {
        let mut cur = &self.head;
        while g >= cur.base_idx + cur.cap {
            cur = &*cur.next.load(Acquire);
        }
        let s = &*cur.slots.add(g - cur.base_idx);
        let prefix = ((s.state.load(Relaxed) >> 2) & 0xFF) as u8;
        (prefix, *s.hi.get(), *s.lo.get())
    }
}

static HEX_USERS: OnceLock<HexUserTable> = OnceLock::new();

pub fn init_hex_user_table(total_input_bytes: usize) {
    let _ = HEX_USERS.set(HexUserTable::for_bytes(total_input_bytes));
}

#[inline(always)]
fn hex_users() -> &'static HexUserTable {
    HEX_USERS.get_or_init(|| HexUserTable::with_cap(1 << 16))
}

#[repr(C)]
struct FbSlot {
    state: AtomicU64, // 0 = empty, 1 = writing, (h & !3) | 2 = ready
    offset: UnsafeCell<u32>,
    len: UnsafeCell<u32>,
}

struct FbSegment {
    slots: *mut FbSlot,
    cap: usize,
    mask: usize,
    base_idx: usize,
    max_load: usize,
    count: AlignedUsize,
    in_flight: AlignedUsize,
    sealed: AlignedBool,
    next: AtomicPtr<FbSegment>,
}
unsafe impl Sync for FbSegment {}
unsafe impl Send for FbSegment {}

impl FbSegment {
    fn new(cap: usize, base_idx: usize) -> Self {
        let slots = map_zeroed(cap * std::mem::size_of::<FbSlot>()) as *mut FbSlot;
        FbSegment {
            slots,
            cap,
            mask: cap - 1,
            base_idx,
            max_load: cap * 3 / 4,
            count: AlignedUsize(AtomicUsize::new(0)),
            in_flight: AlignedUsize(AtomicUsize::new(0)),
            sealed: AlignedBool(AtomicBool::new(false)),
            next: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

struct FallbackTable {
    arena: *mut u8,
    arena_cap: usize,
    arena_next: AtomicUsize,
    head: FbSegment,
}
unsafe impl Sync for FallbackTable {}
unsafe impl Send for FallbackTable {}

impl FallbackTable {
    fn new() -> Self {
        let arena_cap = 256 << 20;
        FallbackTable {
            arena: map_zeroed(arena_cap),
            arena_cap,
            arena_next: AtomicUsize::new(0),
            head: FbSegment::new(1 << 16, 0),
        }
    }

    fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325u64 ^ (bytes.len() as u64);
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01B3);
        }
        mix(h)
    }

    unsafe fn alloc_bytes(&self, bytes: &[u8]) -> u32 {
        let off = self.arena_next.fetch_add(bytes.len().max(1), Relaxed);
        assert!(off + bytes.len() <= self.arena_cap, "fallback arena overflow");
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.arena.add(off), bytes.len());
        off as u32
    }

    unsafe fn intern(&self, bytes: &[u8]) -> u32 {
        let h = Self::hash_bytes(bytes);
        let ready_tag = (h & !3) | 2;
        let mut seg = &self.head;
        loop {
            let mut idx = (h as usize) & seg.mask;
            loop {
                let slot = &*seg.slots.add(idx);
                let mut st = slot.state.load(Acquire);
                while st == 1 {
                    std::hint::spin_loop();
                    st = slot.state.load(Acquire);
                }
                if st == 0 {
                    break;
                }
                if st == ready_tag {
                    let s = std::slice::from_raw_parts(self.arena.add(*slot.offset.get() as usize), *slot.len.get() as usize);
                    if s == bytes {
                        return (seg.base_idx + idx) as u32;
                    }
                }
                idx = (idx + 1) & seg.mask;
            }
            if let Some(found) = self.insert_in_segment(seg, bytes, ready_tag, idx) {
                return found;
            }
            seg = Self::next_segment(seg);
        }
    }

    unsafe fn insert_in_segment(&self, seg: &FbSegment, bytes: &[u8], ready_tag: u64, mut idx: usize) -> Option<u32> {
        seg.in_flight.0.fetch_add(1, SeqCst);
        if !seg.sealed.0.load(Acquire) {
            loop {
                let slot = &*seg.slots.add(idx);
                let mut st = slot.state.load(Acquire);
                while st == 1 {
                    std::hint::spin_loop();
                    st = slot.state.load(Acquire);
                }
                if st == 0 {
                    match slot.state.compare_exchange_weak(0, 1, AcqRel, Acquire) {
                        Ok(_) => {
                            let off = self.alloc_bytes(bytes);
                            *slot.offset.get() = off;
                            *slot.len.get() = bytes.len() as u32;
                            slot.state.store(ready_tag, Release);
                            let c = seg.count.0.fetch_add(1, Relaxed) + 1;
                            if c >= seg.max_load {
                                seg.sealed.0.store(true, Release);
                            }
                            seg.in_flight.0.fetch_sub(1, Release);
                            return Some((seg.base_idx + idx) as u32);
                        }
                        Err(_) => continue,
                    }
                }
                if st == ready_tag {
                    let s = std::slice::from_raw_parts(self.arena.add(*slot.offset.get() as usize), *slot.len.get() as usize);
                    if s == bytes {
                        seg.in_flight.0.fetch_sub(1, Release);
                        return Some((seg.base_idx + idx) as u32);
                    }
                }
                idx = (idx + 1) & seg.mask;
            }
        }
        seg.in_flight.0.fetch_sub(1, Release);
        fence(SeqCst);
        while seg.in_flight.0.load(Acquire) != 0 {
            std::hint::spin_loop();
        }
        loop {
            let slot = &*seg.slots.add(idx);
            let st = slot.state.load(Acquire);
            if st == 0 {
                return None;
            }
            if st == ready_tag {
                let s = std::slice::from_raw_parts(self.arena.add(*slot.offset.get() as usize), *slot.len.get() as usize);
                if s == bytes {
                    return Some((seg.base_idx + idx) as u32);
                }
            }
            idx = (idx + 1) & seg.mask;
        }
    }

    unsafe fn next_segment(seg: &FbSegment) -> &FbSegment {
        let p = seg.next.load(Acquire);
        if !p.is_null() {
            return &*p;
        }
        let next = Box::into_raw(Box::new(FbSegment::new(seg.cap * 2, seg.base_idx + seg.cap)));
        match seg.next.compare_exchange(std::ptr::null_mut(), next, AcqRel, Acquire) {
            Ok(_) => &*next,
            Err(existing) => {
                let boxed = Box::from_raw(next);
                crate::sys::release(boxed.slots as *mut u8, boxed.cap * std::mem::size_of::<FbSlot>());
                &*existing
            }
        }
    }

    unsafe fn get(&self, global_idx: u32) -> &[u8] {
        let g = global_idx as usize;
        let mut cur = &self.head;
        while g >= cur.base_idx + cur.cap {
            cur = &*cur.next.load(Acquire);
        }
        let s = &*cur.slots.add(g - cur.base_idx);
        std::slice::from_raw_parts(self.arena.add(*s.offset.get() as usize), *s.len.get() as usize)
    }
}

static FALLBACK_TABLE: OnceLock<FallbackTable> = OnceLock::new();

#[cold]
#[inline(never)]
unsafe fn fallback_intern(p: *const u8, len: usize) -> u64 {
    let bytes = std::slice::from_raw_parts(p, len);
    let idx = FALLBACK_TABLE.get_or_init(FallbackTable::new).intern(bytes);
    TAG_FALLBACK | idx as u64
}

#[cold]
#[inline(never)]
fn fallback_bytes(idx: u32) -> &'static [u8] {
    unsafe { FALLBACK_TABLE.get().unwrap().get(idx) }
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

/// Attempts to pack two `<= 9B` ASCII IDs in one AVX2 pass with no `pext`.
/// `p0` and `p1` must be readable for 16 bytes; empty ID gives 0.
#[inline(always)]
unsafe fn try_pack_pair(p0: *const u8, l0: usize, p1: *const u8, l1: usize) -> Option<(u64, u64)> {
    let raw = _mm256_loadu2_m128i(p1 as *const __m128i, p0 as *const __m128i);
    let lens = _mm256_setr_m128i(_mm_set1_epi8(l0.min(16) as i8), _mm_set1_epi8(l1.min(16) as i8));
    let idx = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    );
    let keep = _mm256_cmpgt_epi8(lens, idx);
    let x = _mm256_and_si256(raw, keep);
    // Bytes past the ID read as 1, so a 0 or a high bit anywhere is a byte that does not pack.
    let filled = _mm256_or_si256(x, _mm256_andnot_si256(keep, _mm256_set1_epi8(1)));
    let bad = _mm256_movemask_epi8(_mm256_or_si256(filled, _mm256_cmpeq_epi8(filled, _mm256_setzero_si256())));
    if (bad != 0) | (l0.max(l1) > 9) {
        return None;
    }
    // Words (b0, b1): b0 << 7 | b1; then pairs of words: w0 << 14 | w1, so each dword packs four bytes.
    let w = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(x, _mm256_set1_epi16(0xFF)), 7), _mm256_srli_epi16(x, 8));
    let d = _mm256_madd_epi16(w, _mm256_setr_epi16(16384, 1, 16384, 1, 16384, 1, 16384, 1, 16384, 1, 16384, 1, 16384, 1, 16384, 1));
    // Lane qword 0 holds (bytes 0-3, bytes 4-7) as two 28-bit dwords; byte 8 sits in qword 1.
    let hi = _mm256_slli_epi64(d, 35);
    let lo = _mm256_slli_epi64(_mm256_srli_epi64(d, 32), 7);
    let ninth = _mm256_shuffle_epi8(
        x,
        _mm256_setr_epi8(
            8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ),
    );
    let t = _mm256_or_si256(_mm256_or_si256(hi, lo), ninth);
    Some((_mm256_extract_epi64(t, 0) as u64, _mm256_extract_epi64(t, 2) as u64))
}

/// Tokenizes a `user_id` (`p` must be readable for 9 bytes; `len >= 1`).
#[allow(dead_code)]
#[inline(always)]
pub unsafe fn tok(p: *const u8, len: usize) -> u64 {
    tok_tid(p, len, 0)
}

/// Tokenizes a `user_id` using worker `tid` for sharded `HexUserTable` accounting.
/// (`p` must be readable for 9 bytes; `len >= 1`).
#[inline(always)]
pub unsafe fn tok_tid(p: *const u8, len: usize, tid: usize) -> u64 {
    if len <= 9 {
        let t = pack(p, len);
        if t != 0 {
            return t;
        }
    } else if let Some((prefix, hi, lo)) = parse_hex33(p, len) {
        let idx = hex_users().intern_tid(prefix, hi, lo, tid);
        return make_order_hex_tok(prefix, hi, idx);
    }
    fallback_intern(p, len)
}

/// `(tok(p0, l0), tok(p1, l1))`, except that an empty ID gives 0 instead of an interned token.
/// `p0` and `p1` must be readable for 16 bytes.
#[allow(dead_code)]
#[inline(always)]
pub unsafe fn tok_pair(p0: *const u8, l0: usize, p1: *const u8, l1: usize) -> (u64, u64) {
    tok_pair_tid(p0, l0, p1, l1, 0)
}

/// `(tok_tid(p0, l0, tid), tok_tid(p1, l1, tid))`, except that an empty ID gives 0 instead of an interned token.
/// `p0` and `p1` must be readable for 16 bytes.
#[inline(always)]
pub unsafe fn tok_pair_tid(p0: *const u8, l0: usize, p1: *const u8, l1: usize, tid: usize) -> (u64, u64) {
    if let Some(pair) = try_pack_pair(p0, l0, p1, l1) {
        return pair;
    }
    tok_pair_long_tid(p0, l0, p1, l1, tid)
}

#[inline(never)]
unsafe fn tok_pair_long_tid(p0: *const u8, l0: usize, p1: *const u8, l1: usize, tid: usize) -> (u64, u64) {
    (if l0 == 0 { 0 } else { tok_tid(p0, l0, tid) }, if l1 == 0 { 0 } else { tok_tid(p1, l1, tid) })
}

/// Tokenizes an `event_id` without any hash table or mutex: 33B hex IDs push `(hi, lo)`
/// into `side`/`arena` and pack a 27-bit hash + 3-bit prefix code + 32-bit side index.
#[inline(always)]
pub unsafe fn tok_event(p: *const u8, len: usize, side: &mut SideWriter, arena: &Arena<Hex128>) -> u64 {
    if len <= 9 {
        let t = pack(p, len);
        if t != 0 {
            return t;
        }
    } else if let Some((prefix, hi, lo)) = parse_hex33(p, len) {
        let idx = side.push(arena, hi, lo);
        let pcode = ((prefix as u64) & 0x38) >> 3;
        let h27 = (hash_hex(prefix, hi, lo) >> 37) & 0x07FF_FFFF;
        return TAG_HEX | (pcode << 59) | (h27 << 32) | (idx as u64);
    }
    fallback_intern(p, len)
}

/// Packs `(event, user)` in one AVX2 pass when both are `<= 9B` ASCII; otherwise tokenizes
/// `event` and returns `user = 0` so the caller only interns `user` if the event row is valid.
#[inline(always)]
pub unsafe fn try_pack_event_user(
    p0: *const u8,
    l0: usize,
    p1: *const u8,
    l1: usize,
    e_side: &mut SideWriter,
    e_arena: &Arena<Hex128>,
) -> (u64, u64) {
    if let Some(pair) = try_pack_pair(p0, l0, p1, l1) {
        return pair;
    }
    (tok_event(p0, l0, e_side, e_arena), 0)
}

/// Canonical 64-bit hash key for an `event_id` token (independent of side-table index).
#[inline(always)]
pub fn event_canon_key(event: u64) -> u64 {
    if (event >> 62) == 2 {
        event >> 32
    } else {
        event
    }
}

/// Exact equality of two `event_id` tokens.
#[inline(always)]
pub unsafe fn event_eq(e_side: *const Hex128, a: u64, b: u64) -> bool {
    if (a >> 62) == 2 {
        if (a >> 32) != (b >> 32) {
            return false;
        }
        let ha = &*e_side.add(a as u32 as usize);
        let hb = &*e_side.add(b as u32 as usize);
        ha.hi == hb.hi && ha.lo == hb.lo
    } else {
        a == b
    }
}

/// Tokenizes a `click_id` without any hash table or mutex: 33B hex IDs push `(hi, lo)`
/// into `side`/`arena` and preserve `(prefix, top 23 bits of hi)` in the token.
#[inline(always)]
pub unsafe fn tok_click(p: *const u8, len: usize, side: &mut SideWriter, arena: &Arena<Hex128>) -> u64 {
    if len <= 9 {
        let t = pack(p, len);
        if t != 0 {
            return t;
        }
    } else if let Some((prefix, hi, lo)) = parse_hex33(p, len) {
        let idx = side.push(arena, hi, lo);
        return make_order_hex_tok(prefix, hi, idx);
    }
    fallback_intern(p, len)
}

/// Tokenizes `(click_id, user_id)` in one AVX2 pass when both are `<= 9B` ASCII.
#[allow(dead_code)]
#[inline(always)]
pub unsafe fn tok_click_user_pair(
    p0: *const u8,
    l0: usize,
    p1: *const u8,
    l1: usize,
    c_side: &mut SideWriter,
    c_arena: &Arena<Hex128>,
) -> (u64, u64) {
    tok_click_user_pair_tid(p0, l0, p1, l1, c_side, c_arena, 0)
}

/// Tokenizes `(click_id, user_id)` in one AVX2 pass when both are `<= 9B` ASCII,
/// using worker `tid` for sharded `HexUserTable` accounting.
#[inline(always)]
pub unsafe fn tok_click_user_pair_tid(
    p0: *const u8,
    l0: usize,
    p1: *const u8,
    l1: usize,
    c_side: &mut SideWriter,
    c_arena: &Arena<Hex128>,
    tid: usize,
) -> (u64, u64) {
    if let Some(pair) = try_pack_pair(p0, l0, p1, l1) {
        return pair;
    }
    tok_click_user_long_tid(p0, l0, p1, l1, c_side, c_arena, tid)
}

#[inline(never)]
unsafe fn tok_click_user_long_tid(
    p0: *const u8,
    l0: usize,
    p1: *const u8,
    l1: usize,
    c_side: &mut SideWriter,
    c_arena: &Arena<Hex128>,
    tid: usize,
) -> (u64, u64) {
    (
        if l0 == 0 { 0 } else { tok_click(p0, l0, c_side, c_arena) },
        if l1 == 0 { 0 } else { tok_tid(p1, l1, tid) },
    )
}

#[inline(always)]
fn unpack_9_buf(t: u64, buf: &mut [u8; 33]) -> usize {
    let mut n = 0;
    for i in 0..9 {
        let c = ((t >> (7 * (8 - i))) & 0x7F) as u8;
        if c == 0 {
            break;
        }
        buf[n] = c;
        n += 1;
    }
    n
}

#[inline(always)]
pub fn append_tok_bytes(t: u64, out: &mut Vec<u8>) {
    let tag = t >> 62;
    if tag < 2 {
        unsafe {
            let b0_7 = _pdep_u64(t >> 7, 0x7F7F_7F7F_7F7F_7F7F).swap_bytes();
            let b8 = (t & 0x7F) as u8;
            let len = if b8 != 0 { 9 } else { 8 - (b0_7.leading_zeros() >> 3) as usize };
            let old = out.len();
            out.reserve(9);
            let dst = out.as_mut_ptr().add(old);
            (dst as *mut u64).write_unaligned(b0_7);
            *dst.add(8) = b8;
            out.set_len(old + len);
        }
        return;
    }
    if tag == 2 {
        unsafe {
            let (prefix, hi, lo) = hex_users().get(t as u32);
            let old = out.len();
            out.reserve(33);
            encode_hex33(prefix, hi, lo, out.as_mut_ptr().add(old));
            out.set_len(old + 33);
        }
        return;
    }
    tok_bytes_fallback(t, out);
}

#[cfg(test)]
#[inline(always)]
pub fn tok_bytes(t: u64, out: &mut Vec<u8>) {
    out.clear();
    append_tok_bytes(t, out);
}

#[cold]
#[inline(never)]
fn tok_bytes_fallback(t: u64, out: &mut Vec<u8>) {
    out.extend_from_slice(fallback_bytes(t as u32));
}

#[cold]
#[inline(never)]
unsafe fn less_fallback(a: u64, b: u64, get_hex: &dyn Fn(u32) -> (u64, u64)) -> bool {
    let mut ba = [0u8; 33];
    let mut bb = [0u8; 33];
    let slice_of = |t: u64, buf: &mut [u8; 33]| -> &[u8] {
        let tag = t >> 62;
        if tag < 2 {
            let n = unpack_9_buf(t, buf);
            std::slice::from_raw_parts(buf.as_ptr(), n)
        } else if tag == 2 {
            let p = ((t >> 55) & 0x7F) as u8;
            let (hi, lo) = get_hex(t as u32);
            encode_hex33(p, hi, lo, buf.as_mut_ptr());
            std::slice::from_raw_parts(buf.as_ptr(), 33)
        } else {
            fallback_bytes(t as u32)
        }
    };
    slice_of(a, &mut ba) < slice_of(b, &mut bb)
}

#[inline(always)]
unsafe fn less_slow_with(a: u64, b: u64, get_hex: impl Fn(u32) -> (u64, u64)) -> bool {
    let ta = a >> 62;
    let tb = b >> 62;
    if ta == 2 && tb == 2 {
        let top_a = a >> 32;
        let top_b = b >> 32;
        if top_a != top_b {
            return top_a < top_b;
        }
        let (ha, la) = get_hex(a as u32);
        let (hb, lb) = get_hex(b as u32);
        return (ha, la) < (hb, lb);
    }
    if ta < 2 && tb == 2 {
        let a0 = a >> 56;
        let b0 = (b >> 55) & 0x7F;
        if a0 != b0 {
            return a0 < b0;
        }
        let (bhi, _) = get_hex(b as u32);
        return a <= head9_of_hex(b0 as u8, bhi);
    }
    if ta == 2 && tb < 2 {
        let a0 = (a >> 55) & 0x7F;
        let b0 = b >> 56;
        if a0 != b0 {
            return a0 < b0;
        }
        let (ahi, _) = get_hex(a as u32);
        return head9_of_hex(a0 as u8, ahi) < b;
    }
    less_fallback(a, b, &get_hex)
}

#[inline(never)]
fn less_slow_user(a: u64, b: u64) -> bool {
    unsafe {
        less_slow_with(a, b, |idx| {
            let (_, hi, lo) = hex_users().get(idx);
            (hi, lo)
        })
    }
}

/// Byte-wise order of two `user_id` tokens.
#[inline(always)]
pub fn tok_less(a: u64, b: u64) -> bool {
    if (a | b) >> 62 == 0 {
        a < b
    } else {
        less_slow_user(a, b)
    }
}

#[inline(never)]
unsafe fn less_slow_click(c_side: *const Hex128, a: u64, b: u64) -> bool {
    less_slow_with(a, b, |idx| {
        let h = &*c_side.add(idx as usize);
        (h.hi, h.lo)
    })
}

/// Byte-wise order of two `click_id` tokens.
#[inline(always)]
pub unsafe fn click_less(c_side: *const Hex128, a: u64, b: u64) -> bool {
    if (a | b) >> 62 == 0 {
        a < b
    } else {
        less_slow_click(c_side, a, b)
    }
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

    #[test]
    fn test_hex33_roundtrip_and_validation() {
        let prefixes = [b'e', b'E', b'u', b'U', b'k', b'K'];
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for &pfx in &prefixes {
            for i in 0..10_000 {
                let (hi, lo) = if i == 0 {
                    (0, 0)
                } else if i == 1 {
                    (u64::MAX, u64::MAX)
                } else {
                    (rnd(), rnd())
                };
                let expected = format!("{}{hi:016x}{lo:016x}", pfx as char);
                let mut out = [0u8; 33];
                unsafe { encode_hex33(pfx, hi, lo, out.as_mut_ptr()) };
                assert_eq!(&out[..], expected.as_bytes());
                let parsed = unsafe { parse_hex33(out.as_ptr(), 33) };
                assert_eq!(parsed, Some((pfx, hi, lo)));
                // Check head9_of_hex matches pack(out, 9)
                let packed9 = unsafe { pack(out.as_ptr(), 9) };
                assert_eq!(head9_of_hex(pfx, hi), packed9);
            }
        }
        // Invalid prefixes, lengths, uppercase hex, non-hex
        let mut bad = *b"x0123456789abcdef0123456789abcdef";
        assert_eq!(unsafe { parse_hex33(bad.as_ptr(), 33) }, None);
        bad[0] = b'u';
        assert!(unsafe { parse_hex33(bad.as_ptr(), 33) }.is_some());
        assert_eq!(unsafe { parse_hex33(bad.as_ptr(), 32) }, None);
        assert_eq!(unsafe { parse_hex33(bad.as_ptr(), 34) }, None);
        for pos in 1..33 {
            let mut v = bad;
            v[pos] = b'A'; // uppercase hex rejected
            assert_eq!(unsafe { parse_hex33(v.as_ptr(), 33) }, None);
            v[pos] = b'g'; // non-hex rejected
            assert_eq!(unsafe { parse_hex33(v.as_ptr(), 33) }, None);
            v[pos] = b'/'; // below '0'
            assert_eq!(unsafe { parse_hex33(v.as_ptr(), 33) }, None);
            v[pos] = b':'; // above '9'
            assert_eq!(unsafe { parse_hex33(v.as_ptr(), 33) }, None);
            v[pos] = b'`'; // below 'a'
            assert_eq!(unsafe { parse_hex33(v.as_ptr(), 33) }, None);
            v[pos] = 0x80; // high bit
            assert_eq!(unsafe { parse_hex33(v.as_ptr(), 33) }, None);
            v[pos] = 0x00; // NUL
            assert_eq!(unsafe { parse_hex33(v.as_ptr(), 33) }, None);
        }
    }

    #[test]
    fn test_bytewise_ordering_packed_hex33_fallback() {
        let samples: Vec<&[u8]> = vec![
            b"U0",
            b"U01234567",
            b"U0123456789abcdef0123456789abcdef",
            b"Uffffffffffffffffffffffffffffffff",
            b"a_very_long_fallback_id",
            b"u0",
            b"u0123",
            b"u01234567",
            b"u01234567000000000000000000000000",
            b"u01234567000000000000000000000001",
            b"u0123456789abcdef0123456789abcdef",
            b"u01234567ffffffffffffffffffffffff",
            b"u01234568",
            b"u0123456g",
            b"u12345678",
            b"u12345678000000000000000000000000",
            b"u123456789",
            b"u_fallback_non_hex_exceeding_9_bytes",
            b"uffffffffffffffffffffffffffffffff",
            b"uzzzzzzzz",
            b"x0123456789abcdef0123456789abcdef",
        ];
        // Verify tok and click_id ordering for every pair
        let arena = Arena::<Hex128>::new(1024, 256);
        let mut sw = SideWriter::default();
        let mut user_toks = Vec::new();
        let mut click_toks = Vec::new();
        for &s in &samples {
            let mut buf = [0u8; 64];
            buf[..s.len()].copy_from_slice(s);
            let ut = unsafe { tok(buf.as_ptr(), s.len()) };
            let ct = unsafe { tok_click(buf.as_ptr(), s.len(), &mut sw, &arena) };
            let mut out = Vec::new();
            tok_bytes(ut, &mut out);
            assert_eq!(out.as_slice(), s);
            user_toks.push(ut);
            click_toks.push(ct);
        }
        for i in 0..samples.len() {
            for j in 0..samples.len() {
                let expected = samples[i] < samples[j];
                assert_eq!(
                    tok_less(user_toks[i], user_toks[j]),
                    expected,
                    "user tok_less mismatch: {:?} vs {:?}",
                    std::str::from_utf8(samples[i]),
                    std::str::from_utf8(samples[j])
                );
                assert_eq!(
                    unsafe { click_less(arena.base, click_toks[i], click_toks[j]) },
                    expected,
                    "click_less mismatch: {:?} vs {:?}",
                    std::str::from_utf8(samples[i]),
                    std::str::from_utf8(samples[j])
                );
            }
        }
    }

    #[test]
    fn test_hex_user_table_multithreaded_and_segment_overflow() {
        // Force segment overflow with tiny 64-slot initial segment across 4 threads.
        let tbl = HexUserTable { head: HexSegment::new(64, 0) };
        let items: Vec<(u8, u64, u64)> = (0..500u64)
            .map(|i| (b'u', i.wrapping_mul(0x9E37_79B9_7F4A_7C15), i ^ 0xDEAD_BEEF))
            .collect();
        let mut results = vec![Vec::new(); 4];
        std::thread::scope(|sc| {
            for (tid, out) in results.iter_mut().enumerate() {
                let tbl_ref = &tbl;
                let items_ref = &items;
                sc.spawn(move || {
                    for k in 0..items_ref.len() {
                        let idx = (k + tid * 37) % items_ref.len();
                        let (p, hi, lo) = items_ref[idx];
                        let id = unsafe { tbl_ref.intern(p, hi, lo) };
                        out.push((idx, id));
                    }
                });
            }
        });
        for r in &mut results {
            r.sort_by_key(|x| x.0);
        }
        for k in 0..items.len() {
            let id0 = results[0][k].1;
            for t in 1..4 {
                assert_eq!(results[t][k].1, id0);
            }
            assert_eq!(unsafe { tbl.get(id0) }, items[k]);
        }
    }

    #[test]
    fn test_event_tok_canon_and_eq() {
        let arena = Arena::<Hex128>::new(1024, 256);
        let mut sw1 = SideWriter::default();
        let mut sw2 = SideWriter::default();
        let samples: Vec<&[u8]> = vec![
            b"e1",
            b"e12345678",
            b"e0123456789abcdef0123456789abcdef",
            b"E0123456789abcdef0123456789abcdef",
            b"e0123456789abcdef0123456789abcdee",
            b"e_non_hex_long_event_id_123456",
        ];
        let mut toks1 = Vec::new();
        let mut toks2 = Vec::new();
        for &s in &samples {
            let mut buf = [0u8; 64];
            buf[..s.len()].copy_from_slice(s);
            toks1.push(unsafe { tok_event(buf.as_ptr(), s.len(), &mut sw1, &arena) });
            toks2.push(unsafe { tok_event(buf.as_ptr(), s.len(), &mut sw2, &arena) });
        }
        for i in 0..samples.len() {
            assert_eq!(event_canon_key(toks1[i]), event_canon_key(toks2[i]));
            for j in 0..samples.len() {
                assert_eq!(
                    unsafe { event_eq(arena.base, toks1[i], toks2[j]) },
                    i == j,
                    "event_eq mismatch for {} vs {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_fallback_table_segment_overflow() {
        let mut tbl = FallbackTable::new();
        tbl.head = FbSegment::new(64, 0);
        let items: Vec<Vec<u8>> = (0..400).map(|i| format!("fb_non_hex_id_{i:06}").into_bytes()).collect();
        let mut results = vec![Vec::new(); 4];
        std::thread::scope(|sc| {
            for (tid, out) in results.iter_mut().enumerate() {
                let tbl_ref = &tbl;
                let items_ref = &items;
                sc.spawn(move || {
                    for k in 0..items_ref.len() {
                        let idx = (k + tid * 29) % items_ref.len();
                        let id = unsafe { tbl_ref.intern(&items_ref[idx]) };
                        out.push((idx, id));
                    }
                });
            }
        });
        for r in &mut results {
            r.sort_by_key(|x| x.0);
        }
        for k in 0..items.len() {
            let id0 = results[0][k].1;
            for t in 1..4 {
                assert_eq!(results[t][k].1, id0);
            }
            assert_eq!(unsafe { tbl.get(id0) }, items[k].as_slice());
        }
    }

    #[test]
    fn test_hex_user_table_sharded_tid() {
        let tbl_sized = HexUserTable::for_bytes(3usize << 30);
        assert_eq!(tbl_sized.head.cap, 1 << 21);
        let tbl = HexUserTable::with_cap(1 << 16);
        let items: Vec<(u8, u64, u64)> = (0..2000u64)
            .map(|i| (if i % 2 == 0 { b'u' } else { b'U' }, i.wrapping_mul(0x9E37_79B9_7F4A_7C15), i + 1))
            .collect();
        let mut results = vec![Vec::new(); 8];
        std::thread::scope(|sc| {
            for (tid, out) in results.iter_mut().enumerate() {
                let tbl_ref = &tbl;
                let items_ref = &items;
                sc.spawn(move || {
                    for k in 0..items_ref.len() {
                        let idx = (k + tid * 37) % items_ref.len();
                        let (p, hi, lo) = items_ref[idx];
                        let id = unsafe { tbl_ref.intern_tid(p, hi, lo, tid * 9) };
                        out.push((idx, id));
                    }
                });
            }
        });
        for r in &mut results {
            r.sort_by_key(|x| x.0);
        }
        for k in 0..items.len() {
            let id0 = results[0][k].1;
            for t in 1..8 {
                assert_eq!(results[t][k].1, id0);
            }
            assert_eq!(unsafe { tbl.get(id0) }, items[k]);
        }
    }

    #[test]
    fn test_append_tok_bytes_matches_unpack_9_buf() {
        let mut x = 0x1234_5678_9ABC_DEF0u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut out = Vec::new();
        let mut buf = [0u8; 33];
        for len in 1..=9 {
            // Test boundary characters (0x01, 0x7F, '0', 'z') and random 7-bit non-zero ASCII.
            for case in 0..2000 {
                let mut src = [0u8; 16];
                for i in 0..len {
                    src[i] = match case {
                        0 => 0x01,
                        1 => 0x7F,
                        2 => b'u',
                        3 => if i == len - 1 { 0x01 } else { 0x7F },
                        4 => if i == len - 1 { 0x7F } else { 0x01 },
                        _ => (rnd() % 127 + 1) as u8,
                    };
                }
                let t = unsafe { pack(src.as_ptr(), len) };
                assert_ne!(t, 0);
                assert!(t >> 62 < 2);
                let expected_len = unpack_9_buf(t, &mut buf);
                assert_eq!(expected_len, len);

                // Verify empty out and non-empty out (prefix preservation).
                out.clear();
                append_tok_bytes(t, &mut out);
                assert_eq!(out.as_slice(), &buf[..expected_len], "len={len} case={case}");

                let prefix = b"pfx:";
                out.clear();
                out.extend_from_slice(prefix);
                append_tok_bytes(t, &mut out);
                assert_eq!(&out[..prefix.len()], prefix);
                assert_eq!(&out[prefix.len()..], &buf[..expected_len], "prefix len={len} case={case}");
            }
        }
    }
}
