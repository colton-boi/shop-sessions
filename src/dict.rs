use crate::ids::{
    append_tok_bytes, encode_hex33, hash_hex, head9_of_hex, hex_user_get, pack,
    tok_less as ids_tok_less, tok_tid, TAG_HEX,
};
#[cfg(test)]
use crate::ids::parse_hex33;
use crate::links::LinkTable;
use crate::sys::{map_zeroed_huge, mix, par, release};
use std::arch::x86_64::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{
    AtomicBool, AtomicU32, AtomicU64, AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};

pub const MAX_PIDS: usize = (1 << 22) - 4096;
const LEASE_BATCH: u32 = 1024;
const U_MAX_PROBE: usize = 512;
const H_MAX_PROBE: usize = 64;
const G_MAX_PROBE: usize = 512;

#[repr(C, align(16))]
struct Align16([i8; 16]);

static BASE37_SHUF: [Align16; 9] = [
    Align16([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
    Align16([-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1]),
    Align16([-1, -1, -1, -1, -1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1]),
    Align16([-1, -1, -1, -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1]),
    Align16([-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1]),
    Align16([-1, -1, -1, 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1]),
    Align16([-1, -1, 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1]),
    Align16([-1, 0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1]),
    Align16([0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1]),
];

/// Exact branchless SIMD bijection for `prefix` + 1..=8 lowercase `[0-9a-z]` into `1..37^8` (`< 2^42`).
/// Caller must ensure `p` is readable for at least 9 bytes.
#[inline(always)]
pub unsafe fn pack_prefix_base37(prefix: u8, p: *const u8, len: usize) -> Option<u64> {
    if len < 2 || len > 9 || *p != prefix {
        return None;
    }
    let n = len - 1;
    let shuf = _mm_load_si128(BASE37_SHUF.as_ptr().add(n) as *const __m128i);
    let raw = _mm_loadl_epi64(p.add(1) as *const __m128i);
    let v = _mm_shuffle_epi8(raw, shuf);
    let keep = _mm_cmpgt_epi8(shuf, _mm_set1_epi8(-1));
    let d = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
    let a = _mm_sub_epi8(v, _mm_set1_epi8(b'a' as i8));
    let bad_d = _mm_subs_epu8(d, _mm_set1_epi8(9));
    let bad_a = _mm_subs_epu8(a, _mm_set1_epi8(25));
    let bad = _mm_and_si128(_mm_min_epu8(bad_d, bad_a), keep);
    if _mm_testz_si128(bad, bad) == 0 {
        return None;
    }
    let is_d = _mm_cmpeq_epi8(bad_d, _mm_setzero_si128());
    let code_d = _mm_add_epi8(d, _mm_set1_epi8(1));
    let code_a = _mm_add_epi8(a, _mm_set1_epi8(11));
    let code = _mm_and_si128(_mm_blendv_epi8(code_a, code_d, is_d), keep);
    let pairs = _mm_maddubs_epi16(code, _mm_set1_epi16(0x0125));
    let quads = _mm_madd_epi16(pairs, _mm_set1_epi32(0x0001_0559));
    let hi = _mm_cvtsi128_si32(quads) as u32 as u64;
    let lo = _mm_extract_epi32(quads, 1) as u32 as u64;
    Some(hi * 1_874_161 + lo)
}

#[inline(always)]
pub unsafe fn pack_u_base37(p: *const u8, len: usize) -> Option<u64> {
    pack_prefix_base37(b'u', p, len)
}

/// Exact branchless 256-bit AVX2 SIMD bijection packing two `[prefix + 1..=8 lowercase [0-9a-z]]`
/// IDs into `(u64, u64)` in `1..37^8` (`< 2^42`).
/// Caller must ensure `p0` and `p1` are each readable for at least 9 bytes.
#[inline(always)]
pub unsafe fn pack_pair_base37<const P0: u8, const P1: u8>(
    p0: *const u8,
    l0: usize,
    p1: *const u8,
    l1: usize,
) -> Option<(u64, u64)> {
    if (l0.wrapping_sub(2) | l1.wrapping_sub(2)) > 7 || *p0 != P0 || *p1 != P1 {
        return None;
    }
    let shuf0 = _mm_load_si128(BASE37_SHUF.as_ptr().add(l0 - 1) as *const __m128i);
    let shuf1 = _mm_load_si128(BASE37_SHUF.as_ptr().add(l1 - 1) as *const __m128i);
    let shuf = _mm256_set_m128i(shuf1, shuf0);
    let raw = _mm256_set_m128i(
        _mm_loadl_epi64(p1.add(1) as *const __m128i),
        _mm_loadl_epi64(p0.add(1) as *const __m128i),
    );
    let v = _mm256_shuffle_epi8(raw, shuf);
    let keep = _mm256_cmpgt_epi8(shuf, _mm256_set1_epi8(-1));
    let d = _mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8));
    let a = _mm256_sub_epi8(v, _mm256_set1_epi8(b'a' as i8));
    let bad_d = _mm256_subs_epu8(d, _mm256_set1_epi8(9));
    let bad_a = _mm256_subs_epu8(a, _mm256_set1_epi8(25));
    let bad = _mm256_and_si256(_mm256_min_epu8(bad_d, bad_a), keep);
    if _mm256_testz_si256(bad, bad) == 0 {
        return None;
    }
    let is_d = _mm256_cmpeq_epi8(bad_d, _mm256_setzero_si256());
    let code_d = _mm256_add_epi8(d, _mm256_set1_epi8(1));
    let code_a = _mm256_add_epi8(a, _mm256_set1_epi8(11));
    let code = _mm256_and_si256(_mm256_blendv_epi8(code_a, code_d, is_d), keep);
    let pairs = _mm256_maddubs_epi16(code, _mm256_set1_epi16(0x0125));
    let quads = _mm256_madd_epi16(pairs, _mm256_set1_epi32(0x0001_0559));
    let q0 = _mm256_castsi256_si128(quads);
    let q1 = _mm256_extracti128_si256::<1>(quads);
    let hi0 = _mm_cvtsi128_si32(q0) as u32 as u64;
    let lo0 = _mm_extract_epi32(q0, 1) as u32 as u64;
    let hi1 = _mm_cvtsi128_si32(q1) as u32 as u64;
    let lo1 = _mm_extract_epi32(q1, 1) as u32 as u64;
    Some((hi0 * 1_874_161 + lo0, hi1 * 1_874_161 + lo1))
}

/// Packs `'e'` `event_id` and `'u'` `user_id` together in one 256-bit AVX2 vector.
#[inline(always)]
pub unsafe fn pack_eu_base37_pair(
    p0: *const u8,
    l0: usize,
    p1: *const u8,
    l1: usize,
) -> Option<(u64, u64)> {
    pack_pair_base37::<b'e', b'u'>(p0, l0, p1, l1)
}

#[repr(C, align(64))]
pub struct HexLine {
    pub k: [[u64; 2]; 3],
    pub pid: [u32; 3],
    pub ctrl: AtomicU32,
}

#[repr(C)]
struct GenSlot {
    tok: AtomicU64,
    pid: UnsafeCell<u32>,
    _pad: u32,
}

#[derive(Clone, Copy, Default)]
pub struct PidLease {
    cur: u32,
    end: u32,
}

impl PidLease {
    #[inline(always)]
    fn peek(&mut self, next_pid: &AtomicU32, overflow: &AtomicBool) -> u32 {
        if self.cur == self.end {
            return self.refill(next_pid, overflow);
        }
        self.cur
    }

    #[cold]
    #[inline(never)]
    fn refill(&mut self, next_pid: &AtomicU32, overflow: &AtomicBool) -> u32 {
        if overflow.load(Relaxed) || next_pid.load(Relaxed) >= MAX_PIDS as u32 {
            overflow.store(true, Relaxed);
            return 0;
        }
        let base = next_pid.fetch_add(LEASE_BATCH, Relaxed);
        if base >= MAX_PIDS as u32 {
            overflow.store(true, Relaxed);
            return 0;
        }
        self.cur = base;
        self.end = base.saturating_add(LEASE_BATCH).min(MAX_PIDS as u32);
        self.cur
    }

    #[inline(always)]
    fn consume(&mut self) {
        self.cur += 1;
    }
}

/// Lock-free dense-PID user dictionary.
/// `pid == 0` is reserved for empty/invalid; valid PIDs are in `1..MAX_PIDS`.
pub struct UserDict {
    uslots: *mut AtomicU64,
    umask: usize,
    ucap: usize,
    pub hlines: *mut HexLine,
    hmask: usize,
    hcap: usize,
    gslots: *mut GenSlot,
    gmask: usize,
    gcap: usize,
    pub pid_name: *mut u64,
    next_pid: AtomicU32,
    overflow: AtomicBool,
    g_used: AtomicBool,
}

unsafe impl Sync for UserDict {}
unsafe impl Send for UserDict {}

impl UserDict {
    #[allow(dead_code)]
    pub fn new(event_bytes: usize) -> Self {
        Self::new_with_links(event_bytes, 0)
    }

    pub fn new_with_links(event_bytes: usize, link_bytes: usize) -> Self {
        let sized = event_bytes.max(link_bytes.saturating_mul(16));
        let ucap = (sized / 450).next_power_of_two().clamp(1 << 16, 1 << 23);
        let hcap = (sized / 8192).next_power_of_two().clamp(1 << 12, 1 << 20);
        let gcap = 1 << 18;
        let uslots = map_zeroed_huge(ucap * std::mem::size_of::<AtomicU64>()) as *mut AtomicU64;
        let hlines = map_zeroed_huge(hcap * std::mem::size_of::<HexLine>()) as *mut HexLine;
        let gslots = map_zeroed_huge(gcap * std::mem::size_of::<GenSlot>()) as *mut GenSlot;
        let pid_name = map_zeroed_huge(MAX_PIDS * std::mem::size_of::<u64>()) as *mut u64;
        Self {
            uslots,
            umask: ucap - 1,
            ucap,
            hlines,
            hmask: hcap - 1,
            hcap,
            gslots,
            gmask: gcap - 1,
            gcap,
            pid_name,
            next_pid: AtomicU32::new(1),
            overflow: AtomicBool::new(false),
            g_used: AtomicBool::new(false),
        }
    }

    #[inline(always)]
    pub fn is_overflow(&self) -> bool {
        self.overflow.load(Relaxed)
    }

    #[inline(always)]
    pub fn max_pid(&self) -> u32 {
        self.next_pid.load(Relaxed).saturating_sub(1).min((MAX_PIDS - 1) as u32)
    }

    #[inline(always)]
    pub unsafe fn pid_tok(&self, pid: u32) -> u64 {
        *self.pid_name.add(pid as usize)
    }

    #[inline(always)]
    pub unsafe fn prefetch_u(&self, key42: u64) -> usize {
        let idx = (mix(key42) as usize) & self.umask;
        _mm_prefetch(self.uslots.add(idx) as *const i8, _MM_HINT_T0);
        idx
    }

    #[inline(always)]
    pub unsafe fn get_or_insert_u_prehashed(
        &self,
        key42: u64,
        idx: usize,
        p: *const u8,
        len: usize,
        lease: &mut PidLease,
    ) -> u32 {
        let v = (*self.uslots.add(idx)).load(Relaxed);
        if (v >> 22) == key42 {
            return (v & 0x3F_FFFF) as u32;
        }
        self.get_or_insert_u_slow(key42, idx, p, len, lease)
    }

    #[cold]
    #[inline(never)]
    unsafe fn get_or_insert_u_slow(
        &self,
        key42: u64,
        mut idx: usize,
        p: *const u8,
        len: usize,
        lease: &mut PidLease,
    ) -> u32 {
        for _ in 0..U_MAX_PROBE {
            let slot = &*self.uslots.add(idx);
            let mut v = slot.load(Relaxed);
            loop {
                if (v >> 22) == key42 {
                    return (v & 0x3F_FFFF) as u32;
                }
                if v == 0 {
                    let pid = lease.peek(&self.next_pid, &self.overflow);
                    if pid == 0 {
                        return 0;
                    }
                    let entry = (key42 << 22) | (pid as u64);
                    match slot.compare_exchange_weak(0, entry, AcqRel, Relaxed) {
                        Ok(_) => {
                            lease.consume();
                            *self.pid_name.add(pid as usize) = pack(p, len);
                            return pid;
                        }
                        Err(now) => {
                            v = now;
                            continue;
                        }
                    }
                }
                break;
            }
            idx = (idx + 1) & self.umask;
        }
        self.overflow.store(true, Relaxed);
        0
    }

    #[inline(always)]
    pub unsafe fn prefetch_hex(&self, hi: u64, lo: u64) -> usize {
        let idx = (hash_hex(b'u', hi, lo) as usize) & self.hmask;
        _mm_prefetch(self.hlines.add(idx) as *const i8, _MM_HINT_T0);
        idx
    }

    #[inline(always)]
    pub unsafe fn get_or_insert_hex_prehashed(
        &self,
        hi: u64,
        lo: u64,
        idx: usize,
        lease: &mut PidLease,
        tid: usize,
    ) -> u32 {
        let line = &*self.hlines.add(idx);
        let ctrl = line.ctrl.load(Acquire);
        let ready = (ctrl >> 8) & 7;
        if (ready & 1) != 0 && line.k[0] == [hi, lo] {
            return line.pid[0];
        }
        if (ready & 2) != 0 && line.k[1] == [hi, lo] {
            return line.pid[1];
        }
        if (ready & 4) != 0 && line.k[2] == [hi, lo] {
            return line.pid[2];
        }
        self.get_or_insert_hex_slow(hi, lo, idx, lease, tid)
    }

    #[cold]
    #[inline(never)]
    unsafe fn get_or_insert_hex_slow(
        &self,
        hi: u64,
        lo: u64,
        mut line_idx: usize,
        lease: &mut PidLease,
        tid: usize,
    ) -> u32 {
        for _ in 0..H_MAX_PROBE {
            let line = &mut *self.hlines.add(line_idx);
            let mut c = line.ctrl.load(Acquire);
            loop {
                while (c & 7) != ((c >> 8) & 7) {
                    std::hint::spin_loop();
                    c = line.ctrl.load(Acquire);
                }
                let ready = (c >> 8) & 7;
                if (ready & 1) != 0 && line.k[0] == [hi, lo] {
                    return line.pid[0];
                }
                if (ready & 2) != 0 && line.k[1] == [hi, lo] {
                    return line.pid[1];
                }
                if (ready & 4) != 0 && line.k[2] == [hi, lo] {
                    return line.pid[2];
                }
                if (c & 7) == 7 {
                    break;
                }
                let lane = ((!c) & 7).trailing_zeros() as usize;
                let pid = lease.peek(&self.next_pid, &self.overflow);
                if pid == 0 {
                    return 0;
                }
                match line
                    .ctrl
                    .compare_exchange_weak(c, c | (1u32 << lane), AcqRel, Acquire)
                {
                    Ok(_) => {
                        lease.consume();
                        line.k[lane] = [hi, lo];
                        line.pid[lane] = pid;
                        // Bits 55..61 are guaranteed zero here because line_idx < 2^20.
                        *self.pid_name.add(pid as usize) =
                            TAG_HEX | ((line_idx as u64) << 2) | (lane as u64);
                        line.ctrl.fetch_or(1u32 << (8 + lane), Release);
                        return pid;
                    }
                    Err(now) => c = now,
                }
            }
            line_idx = (line_idx + 1) & self.hmask;
        }
        // Spill into general table using exact 33B ASCII bytes so both lookup and insert agree.
        let mut buf = [0u8; 33];
        encode_hex33(b'u', hi, lo, buf.as_mut_ptr());
        self.get_or_insert_gen(buf.as_ptr(), 33, lease, tid)
    }

    #[cold]
    #[inline(never)]
    pub unsafe fn get_or_insert_gen(
        &self,
        p: *const u8,
        len: usize,
        lease: &mut PidLease,
        tid: usize,
    ) -> u32 {
        let tok = tok_tid(p, len, tid);
        self.get_or_insert_gen_tok(tok, lease)
    }

    #[inline(never)]
    unsafe fn get_or_insert_gen_tok(&self, tok: u64, lease: &mut PidLease) -> u32 {
        debug_assert!(tok > 1);
        let mut idx = (mix(tok) as usize) & self.gmask;
        for _ in 0..G_MAX_PROBE {
            let slot = &*self.gslots.add(idx);
            let mut t = slot.tok.load(Acquire);
            loop {
                if t == tok {
                    return *slot.pid.get();
                }
                if t == 1 {
                    std::hint::spin_loop();
                    t = slot.tok.load(Acquire);
                    continue;
                }
                if t == 0 {
                    let pid = lease.peek(&self.next_pid, &self.overflow);
                    if pid == 0 {
                        return 0;
                    }
                    match slot.tok.compare_exchange_weak(0, 1, AcqRel, Acquire) {
                        Ok(_) => {
                            lease.consume();
                            *slot.pid.get() = pid;
                            *self.pid_name.add(pid as usize) = tok;
                            self.g_used.store(true, Relaxed);
                            slot.tok.store(tok, Release);
                            return pid;
                        }
                        Err(now) => {
                            t = now;
                            continue;
                        }
                    }
                }
                break;
            }
            idx = (idx + 1) & self.gmask;
        }
        self.overflow.store(true, Relaxed);
        0
    }

    /// Byte-wise comparison of two tokens stored in `self.pid_name`.
    #[inline(always)]
    pub unsafe fn tok_less(&self, a: u64, b: u64) -> bool {
        if ((a | b) >> 62) == 0 {
            return a < b;
        }
        self.tok_less_slow(a, b)
    }

    #[cold]
    #[inline(never)]
    unsafe fn tok_less_slow(&self, a: u64, b: u64) -> bool {
        let a_is_hl = (a >> 62) == 2 && ((a >> 55) & 0x7F) == 0;
        let b_is_hl = (b >> 62) == 2 && ((b >> 55) & 0x7F) == 0;
        if !a_is_hl && !b_is_hl {
            return ids_tok_less(a, b);
        }
        if a_is_hl && b_is_hl {
            let la = &*self.hlines.add(((a >> 2) & 0xF_FFFF) as usize);
            let lb = &*self.hlines.add(((b >> 2) & 0xF_FFFF) as usize);
            let [ha, loa] = la.k[(a & 3) as usize];
            let [hb, lob] = lb.k[(b & 3) as usize];
            return (ha, loa) < (hb, lob);
        }
        if a_is_hl && (b >> 62) < 2 {
            let b0 = b >> 56;
            if b'u' as u64 != b0 {
                return (b'u' as u64) < b0;
            }
            let [ahi, _] = (*self.hlines.add(((a >> 2) & 0xF_FFFF) as usize)).k[(a & 3) as usize];
            return head9_of_hex(b'u', ahi) < b;
        }
        if b_is_hl && (a >> 62) < 2 {
            let a0 = a >> 56;
            if a0 != b'u' as u64 {
                return a0 < b'u' as u64;
            }
            let [bhi, _] = (*self.hlines.add(((b >> 2) & 0xF_FFFF) as usize)).k[(b & 3) as usize];
            return a <= head9_of_hex(b'u', bhi);
        }
        // One is hlines ('u' + 32 hex) and the other is hex_users() or Fallback.
        let mut ba = Vec::with_capacity(64);
        let mut bb = Vec::with_capacity(64);
        let push_dict_tok = |t: u64, is_hl: bool, out: &mut Vec<u8>| {
            if is_hl {
                let [hi, lo] = (*self.hlines.add(((t >> 2) & 0xF_FFFF) as usize)).k[(t & 3) as usize];
                out.resize(33, 0);
                encode_hex33(b'u', hi, lo, out.as_mut_ptr());
            } else {
                append_tok_bytes(t, out);
            }
        };
        push_dict_tok(a, a_is_hl, &mut ba);
        push_dict_tok(b, b_is_hl, &mut bb);
        ba < bb
    }

    /// Resolves connected components over `temp_uid = 1..=max_pid()` directly using a compact
    /// `AtomicU32` parent array, assigns dense canonical PIDs `1..=num_roots`, and remaps
    /// `uslots`, `hlines`, and `gslots` in parallel.
    pub unsafe fn resolve_links_dsu(&self, pair_lists: &[&[(u32, u32)]], threads: usize) {
        let n_uids = self.max_pid() as usize;
        if n_uids == 0 {
            return;
        }

        let mut parent: Vec<AtomicU32> = Vec::with_capacity(n_uids + 1);
        std::ptr::write_bytes(parent.as_mut_ptr(), 0, n_uids + 1);
        parent.set_len(n_uids + 1);

        #[inline(always)]
        unsafe fn dsu_find(parent: &[AtomicU32], mut x: u32) -> u32 {
            loop {
                let p = parent.get_unchecked(x as usize).load(Acquire);
                if p == 0 {
                    return x;
                }
                let g = parent.get_unchecked(p as usize).load(Acquire);
                if g == 0 {
                    return p;
                }
                let _ = parent
                    .get_unchecked(x as usize)
                    .compare_exchange_weak(p, g, AcqRel, Acquire);
                x = g;
            }
        }

        const LINK_BLOCK: usize = 2048;
        let blocks: Vec<(usize, usize)> = pair_lists
            .iter()
            .enumerate()
            .flat_map(|(t, p)| (0..p.len()).step_by(LINK_BLOCK).map(move |from| (t, from)))
            .collect();
        if !blocks.is_empty() {
            let next = AtomicUsize::new(0);
            par(threads, |_| {
                const AHEAD: usize = 8;
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    let Some(&(t, from)) = blocks.get(i) else { break };
                    let list = pair_lists[t];
                    let slice = &list[from..(from + LINK_BLOCK).min(list.len())];
                    let pre = slice.len().min(AHEAD);
                    for k in 0..pre {
                        let (a, b) = *slice.get_unchecked(k);
                        _mm_prefetch(parent.as_ptr().add(a as usize) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(parent.as_ptr().add(b as usize) as *const i8, _MM_HINT_T0);
                    }
                    for k in 0..slice.len() {
                        if k + AHEAD < slice.len() {
                            let (na, nb) = *slice.get_unchecked(k + AHEAD);
                            _mm_prefetch(parent.as_ptr().add(na as usize) as *const i8, _MM_HINT_T0);
                            _mm_prefetch(parent.as_ptr().add(nb as usize) as *const i8, _MM_HINT_T0);
                        }
                        let (ua, ub) = *slice.get_unchecked(k);
                        if ua == ub {
                            continue;
                        }
                        let mut ra = ua;
                        let mut rb = ub;
                        loop {
                            ra = dsu_find(&parent, ra);
                            rb = dsu_find(&parent, rb);
                            if ra == rb {
                                break;
                            }
                            if self.tok_less(self.pid_tok(rb), self.pid_tok(ra)) {
                                std::mem::swap(&mut ra, &mut rb);
                            }
                            let p_rb = parent.get_unchecked(rb as usize);
                            if p_rb.load(Relaxed) == 0
                                && p_rb.compare_exchange_weak(0, ra, AcqRel, Acquire).is_ok()
                            {
                                break;
                            }
                        }
                    }
                }
            });
        }

        // Parallel path compression + root count per chunk.
        const CHUNK: usize = 16_384;
        let n_chunks = n_uids.div_ceil(CHUNK);
        let mut chunk_roots = vec![0u32; n_chunks];
        let chunk_roots_addr = chunk_roots.as_mut_ptr() as usize;
        let next = AtomicUsize::new(0);
        par(threads, |_| loop {
            let c = next.fetch_add(1, Relaxed);
            if c >= n_chunks {
                break;
            }
            let from = 1 + c * CHUNK;
            let to = (from + CHUNK).min(n_uids + 1);
            let mut roots = 0u32;
            for u in from..to {
                if *self.pid_name.add(u) == 0 {
                    continue;
                }
                let p = parent.get_unchecked(u).load(Relaxed);
                if p == 0 {
                    roots += 1;
                } else {
                    let mut cur = p;
                    loop {
                        let np = parent.get_unchecked(cur as usize).load(Relaxed);
                        if np == 0 {
                            break;
                        }
                        cur = np;
                    }
                    parent.get_unchecked(u).store(cur, Relaxed);
                }
            }
            *(chunk_roots_addr as *mut u32).add(c) = roots;
        });

        let mut sum = 1u32;
        for c in 0..n_chunks {
            let r = chunk_roots[c];
            chunk_roots[c] = sum;
            sum += r;
        }
        let num_roots = (sum - 1) as usize;
        if num_roots == 0 {
            self.next_pid.store(1, Relaxed);
            return;
        }

        let mut uid_to_pid: Vec<u32> = Vec::with_capacity(n_uids + 1);
        uid_to_pid.set_len(n_uids + 1);
        let uid_to_pid_addr = uid_to_pid.as_mut_ptr() as usize;

        let mut new_roots: Vec<u64> = Vec::with_capacity(num_roots + 1);
        new_roots.set_len(num_roots + 1);
        let new_roots_addr = new_roots.as_mut_ptr() as usize;

        // Assign dense canonical PIDs `1..=num_roots` to roots and populate `new_roots`.
        let next = AtomicUsize::new(0);
        par(threads, |_| loop {
            let c = next.fetch_add(1, Relaxed);
            if c >= n_chunks {
                break;
            }
            let from = 1 + c * CHUNK;
            let to = (from + CHUNK).min(n_uids + 1);
            let mut pid = chunk_roots[c];
            let utp = uid_to_pid_addr as *mut u32;
            let nr = new_roots_addr as *mut u64;
            for u in from..to {
                let tok = *self.pid_name.add(u);
                if tok != 0 && parent.get_unchecked(u).load(Relaxed) == 0 {
                    *utp.add(u) = pid;
                    *nr.add(pid as usize) = tok;
                    pid += 1;
                }
            }
        });

        // Fill `uid_to_pid` for non-roots and copy `new_roots` back into `pid_name[1..=num_roots]`.
        let copy_chunks = num_roots.div_ceil(CHUNK);
        let next_u = AtomicUsize::new(0);
        let next_r = AtomicUsize::new(0);
        par(threads, |_| {
            let utp = uid_to_pid_addr as *mut u32;
            loop {
                let c = next_u.fetch_add(1, Relaxed);
                if c >= n_chunks {
                    break;
                }
                let from = 1 + c * CHUNK;
                let to = (from + CHUNK).min(n_uids + 1);
                for u in from..to {
                    let r = parent.get_unchecked(u).load(Relaxed);
                    if r != 0 {
                        *utp.add(u) = *utp.add(r as usize);
                    }
                }
            }
            loop {
                let c = next_r.fetch_add(1, Relaxed);
                if c >= copy_chunks {
                    break;
                }
                let from = 1 + c * CHUNK;
                let to = (from + CHUNK).min(num_roots + 1);
                std::ptr::copy_nonoverlapping(
                    new_roots.as_ptr().add(from),
                    self.pid_name.add(from),
                    to - from,
                );
            }
        });
        self.next_pid.store(num_roots as u32 + 1, Relaxed);

        // One parallel sequential sweep over uslots, hlines, and (if used) gslots remapping temp_uid -> canonical pid.
        const SWEEP: usize = 1 << 15;
        let u_sweeps = self.ucap.div_ceil(SWEEP);
        let h_sweeps = self.hcap.div_ceil(SWEEP);
        let g_sweeps = if self.g_used.load(Relaxed) {
            self.gcap.div_ceil(SWEEP)
        } else {
            0
        };
        let total_sweeps = u_sweeps + h_sweeps + g_sweeps;
        let next = AtomicUsize::new(0);
        par(threads, |_| {
            let utp = uid_to_pid.as_ptr();
            loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= total_sweeps {
                    break;
                }
                if i < u_sweeps {
                    let from = i * SWEEP;
                    let to = (from + SWEEP).min(self.ucap);
                    for s in from..to {
                        let slot = &*self.uslots.add(s);
                        let v = slot.load(Relaxed);
                        if v != 0 {
                            let new_pid = *utp.add((v & 0x3F_FFFF) as usize);
                            slot.store((v & !0x3F_FFFF) | (new_pid as u64), Relaxed);
                        }
                    }
                } else if i < u_sweeps + h_sweeps {
                    let hi_i = i - u_sweeps;
                    let from = hi_i * SWEEP;
                    let to = (from + SWEEP).min(self.hcap);
                    for s in from..to {
                        let line = &mut *self.hlines.add(s);
                        let ready = (line.ctrl.load(Relaxed) >> 8) & 7;
                        if ready != 0 {
                            if (ready & 1) != 0 {
                                line.pid[0] = *utp.add(line.pid[0] as usize);
                            }
                            if (ready & 2) != 0 {
                                line.pid[1] = *utp.add(line.pid[1] as usize);
                            }
                            if (ready & 4) != 0 {
                                line.pid[2] = *utp.add(line.pid[2] as usize);
                            }
                        }
                    }
                } else {
                    let gi = i - u_sweeps - h_sweeps;
                    let from = gi * SWEEP;
                    let to = (from + SWEEP).min(self.gcap);
                    for s in from..to {
                        let slot = &*self.gslots.add(s);
                        if slot.tok.load(Relaxed) > 1 {
                            let old_uid = *slot.pid.get();
                            *slot.pid.get() = *utp.add(old_uid as usize);
                        }
                    }
                }
            }
        });
    }

    #[cfg(test)]
    #[inline(always)]
    pub unsafe fn intern_user(
        &self,
        p: *const u8,
        len: usize,
        lease: &mut PidLease,
        tid: usize,
    ) -> u32 {
        if let Some(key42) = pack_u_base37(p, len) {
            let idx = self.prefetch_u(key42);
            return self.get_or_insert_u_prehashed(key42, idx, p, len, lease);
        }
        if len == 33 && *p == b'u' {
            if let Some((_, hi, lo)) = parse_hex33(p, len) {
                let idx = self.prefetch_hex(hi, lo);
                return self.get_or_insert_hex_prehashed(hi, lo, idx, lease, tid);
            }
        }
        self.get_or_insert_gen(p, len, lease, tid)
    }

    #[allow(dead_code)]
    unsafe fn seed_u(&self, key42: u64, pid: u32) {
        let mut idx = (mix(key42) as usize) & self.umask;
        let entry = (key42 << 22) | (pid as u64);
        for _ in 0..U_MAX_PROBE {
            let slot = &*self.uslots.add(idx);
            let mut v = slot.load(Relaxed);
            loop {
                if (v >> 22) == key42 {
                    return;
                }
                if v == 0 {
                    match slot.compare_exchange_weak(0, entry, AcqRel, Relaxed) {
                        Ok(_) => return,
                        Err(now) => {
                            v = now;
                            continue;
                        }
                    }
                }
                break;
            }
            idx = (idx + 1) & self.umask;
        }
        self.overflow.store(true, Relaxed);
    }

    #[allow(dead_code)]
    unsafe fn seed_hex_or_g(&self, hi: u64, lo: u64, k_tok: u64, pid: u32) {
        let mut line_idx = (hash_hex(b'u', hi, lo) as usize) & self.hmask;
        for _ in 0..H_MAX_PROBE {
            let line = &mut *self.hlines.add(line_idx);
            let mut c = line.ctrl.load(Acquire);
            loop {
                while (c & 7) != ((c >> 8) & 7) {
                    std::hint::spin_loop();
                    c = line.ctrl.load(Acquire);
                }
                let ready = (c >> 8) & 7;
                if ((ready & 1) != 0 && line.k[0] == [hi, lo])
                    || ((ready & 2) != 0 && line.k[1] == [hi, lo])
                    || ((ready & 4) != 0 && line.k[2] == [hi, lo])
                {
                    return;
                }
                if (c & 7) == 7 {
                    break;
                }
                let lane = ((!c) & 7).trailing_zeros() as usize;
                match line
                    .ctrl
                    .compare_exchange_weak(c, c | (1u32 << lane), AcqRel, Acquire)
                {
                    Ok(_) => {
                        line.k[lane] = [hi, lo];
                        line.pid[lane] = pid;
                        line.ctrl.fetch_or(1u32 << (8 + lane), Release);
                        return;
                    }
                    Err(now) => c = now,
                }
            }
            line_idx = (line_idx + 1) & self.hmask;
        }
        self.seed_g(k_tok, pid);
    }

    #[allow(dead_code)]
    unsafe fn seed_g(&self, tok: u64, pid: u32) {
        let mut idx = (mix(tok) as usize) & self.gmask;
        for _ in 0..G_MAX_PROBE {
            let slot = &*self.gslots.add(idx);
            let mut t = slot.tok.load(Acquire);
            loop {
                if t == tok {
                    return;
                }
                if t == 1 {
                    std::hint::spin_loop();
                    t = slot.tok.load(Acquire);
                    continue;
                }
                if t == 0 {
                    match slot.tok.compare_exchange_weak(0, 1, AcqRel, Acquire) {
                        Ok(_) => {
                            *slot.pid.get() = pid;
                            slot.tok.store(tok, Release);
                            return;
                        }
                        Err(now) => {
                            t = now;
                            continue;
                        }
                    }
                }
                break;
            }
            idx = (idx + 1) & self.gmask;
        }
        self.overflow.store(true, Relaxed);
    }

    /// Seeds all linked IDs from `LinkTable` into `UserDict`, assigning dense PIDs `1..`
    /// to every root. Returns `false` if PID/table capacity overflowed (caller restores `LinkTable`).
    #[allow(dead_code)]
    pub unsafe fn seed_from_links(&self, links: &LinkTable, threads: usize) -> bool {
        const SWEEP: usize = 1 << 14;
        let total = links.slots();
        let sweeps = total.div_ceil(SWEEP);

        // Pass 1: assign dense pid to every root (k != 0 && v == k) and record pid in LinkTable val.
        let next = std::sync::atomic::AtomicUsize::new(0);
        par(threads, |_| {
            let mut lease = PidLease::default();
            loop {
                let i = next.fetch_add(1, Relaxed);
                let from = i * SWEEP;
                if from >= total {
                    break;
                }
                let to = (from + SWEEP).min(total);
                for s in from..to {
                    let (k, v) = links.slot_kv_relaxed(s);
                    if k != 0 && v == k {
                        let pid = lease.peek(&self.next_pid, &self.overflow);
                        if pid == 0 {
                            return;
                        }
                        lease.consume();
                        *self.pid_name.add(pid as usize) = k;
                        links.set_slot_val_relaxed(s, pid as u64);
                    }
                }
            }
        });
        if self.is_overflow() {
            return false;
        }

        // Pass 2: insert every (k -> root_pid) into uslots/hlines/gslots in parallel with prefetch.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let _ = sweeps;
        par(threads, |_| {
            let mut buf = [0u8; 16];
            let mut batch = [(0u64, 0u64, 0u64); 16]; // (k, v, hv)
            let mut nb = 0usize;
            let flush = |batch: &[(u64, u64, u64)], buf: &mut [u8; 16]| {
                for &(k, v, hv) in batch {
                    let pid = if (v >> 56) == 0 {
                        v as u32
                    } else {
                        links.person_hashed(v, hv) as u32
                    };
                    let tag = k >> 62;
                    if tag < 2 {
                        let b0_7 = _pdep_u64(k >> 7, 0x7F7F_7F7F_7F7F_7F7F).swap_bytes();
                        let b8 = (k & 0x7F) as u8;
                        let len = if b8 != 0 {
                            9
                        } else {
                            8 - (b0_7.leading_zeros() >> 3) as usize
                        };
                        (buf.as_mut_ptr() as *mut u64).write_unaligned(b0_7);
                        buf[8] = b8;
                        if let Some(key42) = pack_u_base37(buf.as_ptr(), len) {
                            self.seed_u(key42, pid);
                        } else {
                            self.seed_g(k, pid);
                        }
                    } else if tag == 2 {
                        let (prefix, hi, lo) = hex_user_get(k as u32);
                        if prefix == b'u' {
                            self.seed_hex_or_g(hi, lo, k, pid);
                        } else {
                            self.seed_g(k, pid);
                        }
                    } else {
                        self.seed_g(k, pid);
                    }
                }
            };
            loop {
                let i = next.fetch_add(1, Relaxed);
                let from = i * SWEEP;
                if from >= total {
                    break;
                }
                let to = (from + SWEEP).min(total);
                for s in from..to {
                    let (k, v) = links.slot_kv_relaxed(s);
                    if k == 0 {
                        continue;
                    }
                    let hv = if (v >> 56) != 0 {
                        let h = mix(v);
                        links.prefetch_hashed(h);
                        h
                    } else {
                        0
                    };
                    batch[nb] = (k, v, hv);
                    nb += 1;
                    if nb == 16 {
                        flush(&batch, &mut buf);
                        nb = 0;
                    }
                }
            }
            if nb > 0 {
                flush(&batch[..nb], &mut buf);
            }
        });
        !self.is_overflow()
    }

    pub fn populate_regions(&self) -> [(usize, usize); 3] {
        [
            (self.uslots as usize, self.ucap * std::mem::size_of::<AtomicU64>()),
            (self.hlines as usize, self.hcap * std::mem::size_of::<HexLine>()),
            (
                self.pid_name as usize,
                (self.ucap / 2).min(MAX_PIDS) * std::mem::size_of::<u64>(),
            ),
        ]
    }

    pub fn release(&self) {
        release(self.uslots as *mut u8, self.ucap * std::mem::size_of::<AtomicU64>());
        release(self.hlines as *mut u8, self.hcap * std::mem::size_of::<HexLine>());
        release(self.gslots as *mut u8, self.gcap * std::mem::size_of::<GenSlot>());
        release(self.pid_name as *mut u8, MAX_PIDS * std::mem::size_of::<u64>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::tok;
    use std::collections::HashSet;

    #[test]
    fn base37_bijection_and_range() {
        let mut seen = HashSet::new();
        let samples = [
            "u0", "u00", "u000", "u1", "u01", "ua", "uz", "u0z", "uz0", "u00000000", "uzzzzzzzz",
            "u12345678",
        ];
        for s in samples {
            let mut buf = [0u8; 32];
            buf[..s.len()].copy_from_slice(s.as_bytes());
            let v = unsafe { pack_u_base37(buf.as_ptr(), s.len()) }.unwrap();
            assert!(v >= 1 && v < (1u64 << 42));
            assert!(seen.insert(v), "collision for {s}");
        }
        // Invalid inputs
        for bad in ["u", "a123", "U123", "u_123", "u123456789", "u-1", "uA"] {
            let mut buf = [0u8; 32];
            buf[..bad.len()].copy_from_slice(bad.as_bytes());
            assert_eq!(unsafe { pack_u_base37(buf.as_ptr(), bad.len()) }, None, "{bad}");
        }
    }

    #[test]
    fn user_dict_multithreaded_and_links_seed() {
        unsafe {
            let links = LinkTable::new(1024);
            let mut buf = [0u8; 64];
            let mut mk = |s: &[u8]| {
                buf[..s.len()].copy_from_slice(s);
                tok(buf.as_ptr(), s.len())
            };
            let u1 = mk(b"u100");
            let u2 = mk(b"u200");
            let hx1 = mk(b"u0123456789abcdef0123456789abcdef");
            let hx2 = mk(b"u0123456789abcdef0123456789abcde0");
            let gen1 = mk(b"U_UPPER_CASE_ID");
            let gen2 = mk(b"x_other_id");
            let pairs = [(u2, u1), (hx1, hx2), (gen1, gen2)];
            links.link_all(&pairs);
            links.compress_and_resolve(0, links.slots());

            let dict = UserDict::new(1 << 20);
            assert!(dict.seed_from_links(&links, 4));
            assert!(!dict.is_overflow());

            let mut l = PidLease::default();
            let p_u1 = dict.intern_user(b"u100".as_ptr(), 4, &mut l, 0);
            let p_u2 = dict.intern_user(b"u200".as_ptr(), 4, &mut l, 0);
            assert_eq!(p_u1, p_u2);
            assert_ne!(p_u1, 0);

            let p_hx1 = dict.intern_user(b"u0123456789abcdef0123456789abcdef".as_ptr(), 33, &mut l, 0);
            let p_hx2 = dict.intern_user(b"u0123456789abcdef0123456789abcde0".as_ptr(), 33, &mut l, 0);
            assert_eq!(p_hx1, p_hx2);

            let p_g1 = dict.intern_user(b"U_UPPER_CASE_ID".as_ptr(), 15, &mut l, 0);
            let p_g2 = dict.intern_user(b"x_other_id".as_ptr(), 10, &mut l, 0);
            assert_eq!(p_g1, p_g2);

            // Concurrent inserts of fresh users
            let pids: Vec<Vec<u32>> = std::thread::scope(|sc| {
                let mut handles = Vec::new();
                for t in 0..4 {
                    let d = &dict;
                    handles.push(sc.spawn(move || {
                        let mut lease = PidLease::default();
                        let mut out = Vec::new();
                        for i in 0..2000u32 {
                            let s = format!("u{i:07}          ");
                            out.push(d.intern_user(s.as_ptr(), 8, &mut lease, t));
                        }
                        out
                    }));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for t in 1..4 {
                assert_eq!(pids[0], pids[t]);
            }
            dict.release();
            links.release(0, links.slots());
        }
    }

    #[test]
    fn pack_eu_base37_pair_exhaustive_parity() {
        let mut x = 0x517C_C1B7_2722_0A95u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let alpha = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/\0\x7f\x80";
        for l0 in 1..=10 {
            for l1 in 1..=10 {
                for iter in 0..400u32 {
                    let mut b0 = [0u8; 32];
                    let mut b1 = [0u8; 32];
                    for k in 0..l0 {
                        b0[k] = if iter < 300 {
                            let c = alpha[(rnd() as usize) % 36]; // valid base37
                            if k == 0 && iter < 250 { b'e' } else { c }
                        } else {
                            alpha[(rnd() as usize) % alpha.len()]
                        };
                    }
                    for k in 0..l1 {
                        b1[k] = if iter < 300 {
                            let c = alpha[(rnd() as usize) % 36];
                            if k == 0 && iter < 250 { b'u' } else { c }
                        } else {
                            alpha[(rnd() as usize) % alpha.len()]
                        };
                    }
                    // Boundary edge cases for iter == 0..3
                    if iter == 0 {
                        b0[0] = b'e';
                        b1[0] = b'u';
                        for k in 1..l0 { b0[k] = b'0'; }
                        for k in 1..l1 { b1[k] = b'z'; }
                    } else if iter == 1 {
                        b0[0] = b'e';
                        b1[0] = b'u';
                        for k in 1..l0 { b0[k] = b'z'; }
                        for k in 1..l1 { b1[k] = b'0'; }
                    } else if iter == 2 {
                        // Bad char just past valid range ('9'+1=':', 'a'-1='`', 'z'+1='{', '0'-1='/')
                        b0[0] = b'e';
                        b1[0] = b'u';
                        if l0 > 1 { b0[l0 - 1] = b':'; }
                        if l1 > 1 { b1[l1 - 1] = b'{'; }
                    }
                    let got = unsafe { pack_eu_base37_pair(b0.as_ptr(), l0, b1.as_ptr(), l1) };
                    let e0 = unsafe { pack_prefix_base37(b'e', b0.as_ptr(), l0) };
                    let e1 = unsafe { pack_u_base37(b1.as_ptr(), l1) };
                    let exp = match (e0, e1) {
                        (Some(a), Some(b)) => Some((a, b)),
                        _ => None,
                    };
                    assert_eq!(got, exp, "l0={l0} l1={l1} iter={iter}");
                }
            }
        }
    }

    #[test]
    fn user_dict_direct_dsu_matches_link_table_multithreaded() {
        unsafe {
            let mut raw_ids: Vec<Vec<u8>> = Vec::new();
            // Chain of 300 short IDs + 1 smallest root
            for i in 0..300 {
                raw_ids.push(format!("u{i:05}").into_bytes());
            }
            // Hub of 600 hex33 IDs + short + uppercase + fallback mixed
            for i in 0..600 {
                raw_ids.push(format!("u{i:032x}").into_bytes());
            }
            raw_ids.push(b"U_UPPER_CASE_HUB".to_vec());
            raw_ids.push(b"A_SMALLEST_UPPER_ID".to_vec());
            raw_ids.push(b"a_smallest_fallback_id_exceeding_9".to_vec());
            raw_ids.push(b"u0".to_vec()); // smaller than u00000? Wait: 'u','0' < 'u','0','0'...
            raw_ids.push(b"u0123456789abcdef0123456789abcdef".to_vec());
            raw_ids.push(b"u_mixed_long_fallback_user_identifier".to_vec());
            // Unlinked solo IDs
            raw_ids.push(b"u_solo_a".to_vec());
            raw_ids.push(b"U_SOLO_B".to_vec());

            let mut edge_indices: Vec<(usize, usize)> = Vec::new();
            // Chain 0..300
            for i in 0..299 {
                edge_indices.push((i, i + 1));
            }
            // Duplicate & reverse & self edges in chain
            for i in (0..299).step_by(7) {
                edge_indices.push((i + 1, i));
                edge_indices.push((i, i));
            }
            // Hub at 300..900 linked with U_UPPER_CASE_HUB (900) and u0 (903) and mixed hex/fallback
            for i in 301..900 {
                edge_indices.push((i, 300));
            }
            edge_indices.push((300, 900)); // U_UPPER_CASE_HUB
            edge_indices.push((450, 903)); // u0
            edge_indices.push((904, 905)); // hex + fallback
            edge_indices.push((905, 901)); // + A_SMALLEST_UPPER_ID
            // Link chain to a_smallest_fallback_id_exceeding_9 (902)
            edge_indices.push((150, 902));

            // Reference: LinkTable
            let links = LinkTable::new(edge_indices.len() * 2);
            let mut buf = [0u8; 64];
            let mut toks = Vec::with_capacity(raw_ids.len());
            for id in &raw_ids {
                buf[..id.len()].copy_from_slice(id);
                toks.push(tok(buf.as_ptr(), id.len()));
            }
            let lt_pairs: Vec<(u64, u64)> = edge_indices.iter().map(|&(a, b)| (toks[a], toks[b])).collect();
            std::thread::scope(|sc| {
                for chunk in lt_pairs.chunks(128) {
                    sc.spawn(|| links.link_all(chunk));
                }
            });
            links.compress_and_resolve(0, links.slots());

            // Subject: direct UserDict + resolve_links_dsu across 4 threads
            let dict = UserDict::new_with_links(1 << 20, 1 << 20);
            let per_thread_pairs: Vec<Vec<(u32, u32)>> = std::thread::scope(|sc| {
                let mut handles = Vec::new();
                for (t, chunk) in edge_indices.chunks(edge_indices.len().div_ceil(4)).enumerate() {
                    let d = &dict;
                    let ids_ref = &raw_ids;
                    handles.push(sc.spawn(move || {
                        let mut lease = PidLease::default();
                        let mut pairs = Vec::with_capacity(chunk.len());
                        for &(ia, ib) in chunk {
                            let mut ba = [0u8; 64];
                            let mut bb = [0u8; 64];
                            ba[..ids_ref[ia].len()].copy_from_slice(&ids_ref[ia]);
                            bb[..ids_ref[ib].len()].copy_from_slice(&ids_ref[ib]);
                            let ua = d.intern_user(ba.as_ptr(), ids_ref[ia].len(), &mut lease, t);
                            let ub = d.intern_user(bb.as_ptr(), ids_ref[ib].len(), &mut lease, t);
                            assert_ne!(ua, 0);
                            assert_ne!(ub, 0);
                            if ua != ub {
                                pairs.push((ua, ub));
                            }
                        }
                        pairs
                    }));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let pair_slices: Vec<&[(u32, u32)]> = per_thread_pairs.iter().map(|v| v.as_slice()).collect();
            dict.resolve_links_dsu(&pair_slices, 4);
            assert!(!dict.is_overflow());

            // Verify every linked ID resolves to the exact same canonical root string as LinkTable!
            let mut lease = PidLease::default();
            let mut out_lt = Vec::new();
            let mut out_ud = Vec::new();
            let decode_ud = |tok: u64, out: &mut Vec<u8>| {
                out.clear();
                if (tok >> 62) == 2 && ((tok >> 55) & 0x7F) == 0 {
                    let line_idx = ((tok >> 2) & 0xF_FFFF) as usize;
                    let lane = (tok & 3) as usize;
                    let [hi, lo] = (*dict.hlines.add(line_idx)).k[lane];
                    out.resize(33, 0);
                    encode_hex33(b'u', hi, lo, out.as_mut_ptr());
                } else {
                    append_tok_bytes(tok, out);
                }
            };
            for (idx, id) in raw_ids.iter().enumerate() {
                let mut b = [0u8; 64];
                b[..id.len()].copy_from_slice(id);
                let pid = dict.intern_user(b.as_ptr(), id.len(), &mut lease, 0);
                assert_ne!(pid, 0);
                assert!(pid <= dict.max_pid());
                decode_ud(dict.pid_tok(pid), &mut out_ud);

                let lt_root = links.person_hashed(toks[idx], mix(toks[idx]));
                out_lt.clear();
                append_tok_bytes(lt_root, &mut out_lt);
                assert_eq!(out_ud, out_lt, "root mismatch for {:?}", String::from_utf8_lossy(id));
            }

            dict.release();
            links.release(0, links.slots());
        }
    }
}
