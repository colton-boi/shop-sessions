use crate::ids::{encode_hex33, hash_hex, hex_user_get, pack, tok_tid, TAG_HEX};
#[cfg(test)]
use crate::ids::parse_hex33;
use crate::links::LinkTable;
use crate::sys::{map_zeroed_huge, mix, par, release};
use std::arch::x86_64::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{
    AtomicBool, AtomicU32, AtomicU64,
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
}

unsafe impl Sync for UserDict {}
unsafe impl Send for UserDict {}

impl UserDict {
    pub fn new(event_bytes: usize) -> Self {
        let ucap = (event_bytes / 450).next_power_of_two().clamp(1 << 16, 1 << 23);
        let hcap = (event_bytes / 8192).next_power_of_two().clamp(1 << 12, 1 << 20);
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
}
