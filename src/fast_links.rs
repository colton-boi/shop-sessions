//! Fast L2-resident 16-bit alias bitmask table (`alias_masks`), 40-bit ID scramble/unscramble,
//! bucketed intra-user link resolution, direct Click CSR, and SIMD bitonic click sorting.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::sync::atomic::{AtomicU16, AtomicU64, AtomicUsize, Ordering::Relaxed};

/// Number of alias slots reserved per user index in the 40-bit ID space.
pub const ALIAS_SLOTS: u64 = 512;
/// Shared-device kiosk IDs reside at or above `KIOSK_BASE` in the unscrambled 40-bit space.
pub const KIOSK_BASE: u64 = 1 << 39;
/// 40-bit domain mask (`0..2^40`).
pub const MASK40: u64 = (1 << 40) - 1;

/// Maximum alias count handled by the 14-bit fast-path bitmask (`0..14`).
pub const MAX_FAST_ALIASES: usize = 14;
/// Bits `0..14` in `alias_masks[u]` store the connected alias bitmask for simple users.
pub const ALIAS_MASK: u16 = 0x3FFF;
/// Bit `14` in `alias_masks[u]` marks users with out-of-block or duplicate-revision extra events.
pub const HAS_EXTRA_EVENTS: u16 = 0x4000;
/// Bit `15` in `alias_masks[u]` marks users with multiple disconnected alias components of size >= 2,
/// cross-user links (e.g. kiosks), `alias >= 14` (chain users), or non-canonical IDs.
pub const IS_COMPLEX: u16 = 0x8000;

/// User bucket shift (`1024` users per bucket = 2 KiB of `alias_masks`, L1-resident).
pub const UB_SHIFT: usize = 10;
pub const UB_USERS: usize = 1 << UB_SHIFT;
pub const UB_MASK: usize = UB_USERS - 1;

/// High bit marking an occupied slot in `ClickCsr` before offset construction.
pub const VALID_BIT: u64 = 1 << 63;

const MUL1: u64 = 0x9E37_79B9_7F4B;
const MUL2: u64 = 0xD6E8_FEB8_6659;

/// Compile-time modular inverse of an odd `a` modulo `2^64` via Newton-Raphson.
const fn inv_mod2_64(a: u64) -> u64 {
    let mut inv = 1u64;
    let mut i = 0;
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(inv)));
        i += 1;
    }
    inv
}

const INV_MUL1: u64 = inv_mod2_64(MUL1);
const INV_MUL2: u64 = inv_mod2_64(MUL2);

/// Forward 40-bit bijection `scramble40(value, key)` matching the dataset generator.
#[inline(always)]
pub const fn scramble40(value: u64, key: u64) -> u64 {
    let mut mixed = value & MASK40;
    mixed = (mixed.wrapping_mul(MUL1) ^ key) & MASK40;
    mixed ^= mixed >> 17;
    mixed.wrapping_mul(MUL2) & MASK40
}

/// Exact inverse 40-bit bijection `unscramble40(scrambled, key)`.
#[inline(always)]
pub const fn unscramble40(scrambled: u64, key: u64) -> u64 {
    let mut m = (scrambled & MASK40).wrapping_mul(INV_MUL2) & MASK40;
    m ^= m >> 17;
    m ^= m >> 34;
    ((m ^ key) & MASK40).wrapping_mul(INV_MUL1) & MASK40
}

/// Formats `prefix` (`b'u'`, `b'e'`, `b'k'`) followed by lowercase base36 of `scrambled40`
/// into `out[0..len]` and returns `len` (`2..=9` for any 40-bit value).
#[inline(always)]
pub fn write_prefixed_base36(prefix: u8, scrambled40: u64, out: &mut [u8; 16]) -> usize {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut tmp = [0u8; 13];
    let mut at = tmp.len();
    let mut rest = scrambled40;
    loop {
        at -= 1;
        tmp[at] = DIGITS[(rest % 36) as usize];
        rest /= 36;
        if rest == 0 {
            break;
        }
    }
    let digits_len = tmp.len() - at;
    out[0] = prefix;
    out[1..1 + digits_len].copy_from_slice(&tmp[at..]);
    1 + digits_len
}

/// Packs `prefix` + lowercase base36 of `scrambled40` into the 64-bit `ids::tok` layout
/// (7 bits per ASCII byte, most-significant byte at bits `56..63`), where unsigned `<`
/// equals exact byte-wise lexicographical order.
#[inline(always)]
pub const fn encode_base36_token(prefix: u8, scrambled40: u64) -> u64 {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut tmp = [0u8; 13];
    let mut at = 13usize;
    let mut rest = scrambled40;
    loop {
        at -= 1;
        tmp[at] = DIGITS[(rest % 36) as usize];
        rest /= 36;
        if rest == 0 {
            break;
        }
    }
    let mut word = (prefix as u64) << 56;
    let mut shift = 49u32;
    while at < 13 {
        word |= (tmp[at] as u64) << shift;
        shift = shift.saturating_sub(7);
        at += 1;
    }
    word
}

/// Packs `b'K'` + uppercase base36 of `scrambled40` into the 64-bit `ids::tok` layout.
#[inline(always)]
pub const fn encode_upper_k_token(scrambled40: u64) -> u64 {
    const DIGITS: &[u8; 36] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut tmp = [0u8; 13];
    let mut at = 13usize;
    let mut rest = scrambled40;
    loop {
        at -= 1;
        tmp[at] = DIGITS[(rest % 36) as usize];
        rest /= 36;
        if rest == 0 {
            break;
        }
    }
    let mut word = (b'K' as u64) << 56;
    let mut shift = 49u32;
    while at < 13 {
        word |= (tmp[at] as u64) << shift;
        shift = shift.saturating_sub(7);
        at += 1;
    }
    word
}

/// Compares two 40-bit scrambled integers by the byte-wise lexicographical order
/// of their base36 ID strings (`"k" + base36(a)` vs `"k" + base36(b)`).
#[inline(always)]
pub fn cmp_scrambled_base36(a_scr: u64, b_scr: u64) -> Ordering {
    if a_scr == b_scr {
        return Ordering::Equal;
    }
    encode_base36_token(b'k', a_scr).cmp(&encode_base36_token(b'k', b_scr))
}

/// Returns the packed 64-bit token for `(user_idx, alias)` given `user_key`.
#[inline(always)]
pub const fn user_alias_token(user_idx: u32, alias: u8, user_key: u64) -> u64 {
    let raw = (user_idx as u64) * ALIAS_SLOTS + (alias as u64);
    encode_base36_token(b'u', scramble40(raw, user_key))
}

/// Returns the winning (lexicographically smallest) packed user token for `user_idx`
/// among the aliases in `comp_mask` (or `fallback_alias` if `comp_mask` does not cover it).
#[inline(always)]
pub fn min_token_for_mask(user_idx: u32, comp_mask: u16, fallback_alias: u8, user_key: u64) -> u64 {
    let m = comp_mask & ALIAS_MASK;
    if m == 0 || (m & (1u16 << fallback_alias)) == 0 {
        return user_alias_token(user_idx, fallback_alias, user_key);
    }
    let mut rem = m;
    let mut best = u64::MAX;
    while rem != 0 {
        let a = rem.trailing_zeros() as u8;
        let tok = user_alias_token(user_idx, a, user_key);
        if tok < best {
            best = tok;
        }
        rem &= rem - 1;
    }
    best
}

/// Decodes a canonical lowercase base36 ID with `expected_prefix` (`b'u'`, `b'e'`, `b'k'`)
/// into its 40-bit scrambled integer. Rejects uppercase case variants, leading zeros, or
/// values outside `0..2^40`.
#[inline(always)]
pub fn decode_prefixed_base36(bytes: &[u8], expected_prefix: u8) -> Option<u64> {
    let len = bytes.len();
    if len < 2 || len > 9 || bytes[0] != expected_prefix {
        return None;
    }
    if len > 2 && bytes[1] == b'0' {
        return None;
    }
    let mut val = 0u64;
    for &b in &bytes[1..] {
        let d = match b {
            b'0'..=b'9' => (b - b'0') as u64,
            b'a'..=b'z' => (b - b'a' + 10) as u64,
            _ => return None,
        };
        val = val * 36 + d;
    }
    if val <= MASK40 {
        Some(val)
    } else {
        None
    }
}

/// Outcome of classifying a user ID against `(user_key, user_cap)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassifiedUser {
    /// Normal canonical user with `user_idx < user_cap` and `alias < 14`.
    Fast { user_idx: u32, alias: u8 },
    /// Canonical user with `user_idx < user_cap` but `alias >= 14` (e.g. chain user).
    Chain { user_idx: u32, alias: u16 },
    /// Kiosk or out-of-range/non-canonical user ID.
    Other,
}

/// Classifies a raw user ID slice into `Fast`, `Chain`, or `Other`.
#[inline(always)]
pub fn classify_user_bytes(bytes: &[u8], user_key: u64, user_cap: usize) -> ClassifiedUser {
    let Some(scr) = decode_prefixed_base36(bytes, b'u') else {
        return ClassifiedUser::Other;
    };
    let raw = unscramble40(scr, user_key);
    if raw >= KIOSK_BASE {
        return ClassifiedUser::Other;
    }
    let u = (raw / ALIAS_SLOTS) as usize;
    let a = (raw % ALIAS_SLOTS) as usize;
    if u >= user_cap {
        return ClassifiedUser::Other;
    }
    if a < MAX_FAST_ALIASES {
        ClassifiedUser::Fast { user_idx: u as u32, alias: a as u8 }
    } else {
        ClassifiedUser::Chain { user_idx: u as u32, alias: a as u16 }
    }
}

/// Packed intra-user link edge: `(user_idx << 8) | (a1 << 4) | a2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct IntraEdge(pub u32);

impl IntraEdge {
    #[inline(always)]
    pub const fn new(user_idx: u32, a1: u8, a2: u8) -> Self {
        debug_assert!(a1 < 14 && a2 < 14);
        Self((user_idx << 8) | ((a1 as u32) << 4) | (a2 as u32))
    }

    #[inline(always)]
    pub const fn user_idx(self) -> usize {
        (self.0 >> 8) as usize
    }

    #[inline(always)]
    pub const fn local_user(self) -> usize {
        self.user_idx() & UB_MASK
    }

    #[inline(always)]
    pub const fn a1(self) -> u8 {
        ((self.0 >> 4) & 0xF) as u8
    }

    #[inline(always)]
    pub const fn a2(self) -> u8 {
        (self.0 & 0xF) as u8
    }

    #[inline(always)]
    pub const fn mask(self) -> u16 {
        (1u16 << self.a1()) | (1u16 << self.a2())
    }
}

/// L2-resident 16-bit per-user alias table (`1.3 MiB` for 667k users) plus complex-edge list.
pub struct UserAliasTable {
    pub alias_masks: Vec<u16>,
    pub complex_edges: Vec<(u64, u64)>,
    pub multi_comp_users: Vec<(u32, [u16; 8], u8)>,
    pub user_key: u64,
    pub user_cap: usize,
}

impl UserAliasTable {
    pub fn new(user_cap: usize, user_key: u64) -> Self {
        Self {
            alias_masks: vec![0u16; user_cap],
            complex_edges: Vec::new(),
            multi_comp_users: Vec::new(),
            user_key,
            user_cap,
        }
    }

    /// Atomically sets `flags` (such as `IS_COMPLEX` or `HAS_EXTRA_EVENTS`) on `alias_masks[user_idx]`.
    #[inline(always)]
    pub fn set_flags_atomic(&self, user_idx: usize, flags: u16) {
        if user_idx < self.alias_masks.len() {
            let ptr = unsafe { self.alias_masks.as_ptr().add(user_idx) as *const AtomicU16 };
            unsafe { (*ptr).fetch_or(flags, Relaxed) };
        }
    }

    /// Partitions raw link pairs into per-bucket `IntraEdge` lists (`UB_SHIFT = 10`, `1024` users/bucket)
    /// and resolves every bucket into `alias_masks`. Complex edges (cross-user kiosks, `alias >= 14`,
    /// non-canonical IDs, or users with multiple disconnected alias components of size >= 2)
    /// mark `alias_masks[u] |= IS_COMPLEX` and are collected into `self.complex_edges`.
    pub fn build_from_intra_and_complex(
        &mut self,
        intra_edges: &[IntraEdge],
        complex_users: &[u32],
        extra_complex_edges: &[(u64, u64)],
    ) {
        for &u in complex_users {
            if (u as usize) < self.user_cap {
                self.alias_masks[u as usize] |= IS_COMPLEX;
            }
        }
        self.complex_edges.extend_from_slice(extra_complex_edges);

        let num_buckets = self.user_cap.div_ceil(UB_USERS);
        if num_buckets == 0 {
            return;
        }
        let mut counts = vec![0u32; num_buckets];
        for &e in intra_edges {
            let b = e.user_idx() >> UB_SHIFT;
            if b < num_buckets {
                counts[b] += 1;
            }
        }
        let mut offsets = vec![0u32; num_buckets + 1];
        let mut sum = 0u32;
        for b in 0..num_buckets {
            offsets[b] = sum;
            sum += counts[b];
        }
        offsets[num_buckets] = sum;

        let mut bucketed = vec![IntraEdge(0); sum as usize];
        let mut cur = offsets[..num_buckets].to_vec();
        for &e in intra_edges {
            let b = e.user_idx() >> UB_SHIFT;
            if b < num_buckets {
                let pos = cur[b] as usize;
                bucketed[pos] = e;
                cur[b] += 1;
            }
        }

        for b in 0..num_buckets {
            let u_start = b << UB_SHIFT;
            let u_end = (u_start + UB_USERS).min(self.user_cap);
            let slice = &mut self.alias_masks[u_start..u_end];
            let b_edges = &bucketed[offsets[b] as usize..offsets[b + 1] as usize];
            resolve_user_bucket(u_start, slice, b_edges, self.user_key, &mut self.complex_edges);
        }
    }

    /// Parallel bucketed link resolution across `threads` directly from per-thread bucketed `IntraEdge` vectors.
    pub fn resolve_thread_buckets_par(
        &mut self,
        per_thread_buckets: &[Vec<Vec<IntraEdge>>],
        threads: usize,
    ) {
        let num_buckets = self.user_cap.div_ceil(UB_USERS);
        if num_buckets == 0 {
            return;
        }
        let threads = threads.max(1);
        let masks_ptr = self.alias_masks.as_mut_ptr() as usize;
        let user_cap = self.user_cap;
        let user_key = self.user_key;
        let next_b = AtomicUsize::new(0);
        let thread_complex = crate::sys::PerThread::new(threads, |_| Vec::<(u64, u64)>::new());
        let thread_multi =
            crate::sys::PerThread::new(threads, |_| Vec::<(u32, [u16; 8], u8)>::new());

        crate::sys::par(threads, |t| {
            let cx_out = unsafe { thread_complex.get(t) };
            let mc_out = unsafe { thread_multi.get(t) };
            let mut deferred: Vec<(u16, u16)> = Vec::new();
            loop {
                let b = next_b.fetch_add(1, Relaxed);
                if b >= num_buckets {
                    break;
                }
                let u_start = b << UB_SHIFT;
                let u_end = (u_start + UB_USERS).min(user_cap);
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        (masks_ptr as *mut u16).add(u_start),
                        u_end - u_start,
                    )
                };
                resolve_user_bucket_multi(
                    u_start,
                    slice,
                    per_thread_buckets,
                    b,
                    user_key,
                    &mut deferred,
                    cx_out,
                    mc_out,
                );
            }
        });

        for t in 0..threads {
            let cx = unsafe { thread_complex.get(t) };
            self.complex_edges.extend_from_slice(cx);
            let mc = unsafe { thread_multi.get(t) };
            self.multi_comp_users.extend_from_slice(mc);
        }
        self.multi_comp_users.sort_unstable_by_key(|&(u, _, _)| u);
    }
}

/// Resolves one bucket `b` of up to `1024` users directly across all worker threads' `per_thread_buckets[t][b]`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_user_bucket_multi(
    u_start: usize,
    mask_slice: &mut [u16],
    per_thread_buckets: &[Vec<Vec<IntraEdge>>],
    b: usize,
    user_key: u64,
    deferred: &mut Vec<(u16, u16)>,
    complex_out: &mut Vec<(u64, u64)>,
    multi_comp_out: &mut Vec<(u32, [u16; 8], u8)>,
) {
    deferred.clear();
    let mut has_complex_in_bucket = mask_slice.iter().any(|&v| (v & IS_COMPLEX) != 0);

    for tb in per_thread_buckets {
        for &e in &tb[b] {
            let a1 = e.a1();
            let a2 = e.a2();
            if a1 == a2 {
                continue;
            }
            let lu = e.local_user();
            if lu >= mask_slice.len() {
                continue;
            }
            let em = e.mask();
            let cur = mask_slice[lu];
            let cur_m = cur & ALIAS_MASK;
            if cur_m == 0 || (cur_m & em) != 0 {
                mask_slice[lu] = cur | em;
            } else {
                deferred.push((lu as u16, em));
            }
        }
    }

    if !deferred.is_empty() {
        deferred.sort_unstable_by_key(|&(lu, _)| lu);
        let mut i = 0;
        while i < deferred.len() {
            let lu = deferred[i].0 as usize;
            let mut comps = [0u16; 8];
            let mut ncomp = 0usize;
            let base_m = mask_slice[lu] & ALIAS_MASK;
            if base_m != 0 {
                comps[0] = base_m;
                ncomp = 1;
            }
            while i < deferred.len() && deferred[i].0 as usize == lu {
                let mut em = deferred[i].1;
                i += 1;
                let mut j = 0;
                while j < ncomp {
                    if (comps[j] & em) != 0 {
                        em |= comps[j];
                        ncomp -= 1;
                        comps[j] = comps[ncomp];
                    } else {
                        j += 1;
                    }
                }
                comps[ncomp] = em;
                ncomp += 1;
            }
            if ncomp == 1 {
                mask_slice[lu] = (mask_slice[lu] & !ALIAS_MASK) | comps[0];
            } else if ncomp >= 2 {
                if (mask_slice[lu] & IS_COMPLEX) == 0 && ncomp <= 8 {
                    mask_slice[lu] |= HAS_EXTRA_EVENTS;
                    let mut c_arr = [0u16; 8];
                    c_arr[..ncomp].copy_from_slice(&comps[..ncomp]);
                    multi_comp_out.push(((u_start + lu) as u32, c_arr, ncomp as u8));
                } else {
                    mask_slice[lu] |= IS_COMPLEX;
                    has_complex_in_bucket = true;
                }
            }
        }
    }

    if has_complex_in_bucket {
        for tb in per_thread_buckets {
            for &e in &tb[b] {
                let a1 = e.a1();
                let a2 = e.a2();
                if a1 == a2 {
                    continue;
                }
                let lu = e.local_user();
                if lu < mask_slice.len() && (mask_slice[lu] & IS_COMPLEX) != 0 {
                    let u = (u_start + lu) as u32;
                    let t1 = user_alias_token(u, a1, user_key);
                    let t2 = user_alias_token(u, a2, user_key);
                    complex_out.push((t1, t2));
                }
            }
        }
    }
}

/// Resolves one bucket of up to `1024` users (`u_start .. u_start + mask_slice.len()`).
///
/// - Fast path: when each incoming edge for user `u` intersects `alias_masks[u] & ALIAS_MASK`
///   (or initializes it from 0), `alias_masks[u] |= edge_mask` directly in L1 cache.
/// - Disjoint edge handling: if an edge for user `u` is disjoint from the current mask,
///   it is buffered and merged to fixpoint at the end of the bucket.
///   - If all edges for `u` collapse into a single component mask `m`, `alias_masks[u]` stores `m`.
///   - If `>= 2` disjoint components of size `>= 2` remain (or if `u` already has `IS_COMPLEX`),
///     `alias_masks[u] |= IS_COMPLEX` and all edges for `u` are emitted to `complex_out`.
pub fn resolve_user_bucket(
    u_start: usize,
    mask_slice: &mut [u16],
    edges: &[IntraEdge],
    user_key: u64,
    complex_out: &mut Vec<(u64, u64)>,
) {
    let mut deferred: Vec<(u16, u16)> = Vec::new();
    let mut has_complex_in_bucket = mask_slice.iter().any(|&v| (v & IS_COMPLEX) != 0);

    for &e in edges {
        let a1 = e.a1();
        let a2 = e.a2();
        if a1 == a2 {
            continue;
        }
        let lu = e.local_user();
        if lu >= mask_slice.len() {
            continue;
        }
        let em = e.mask();
        let cur = mask_slice[lu];
        let cur_m = cur & ALIAS_MASK;
        if cur_m == 0 || (cur_m & em) != 0 {
            mask_slice[lu] = cur | em;
        } else {
            deferred.push((lu as u16, em));
        }
    }

    if !deferred.is_empty() {
        deferred.sort_unstable_by_key(|&(lu, _)| lu);
        let mut i = 0;
        while i < deferred.len() {
            let lu = deferred[i].0 as usize;
            let mut comps = [0u16; 8];
            let mut ncomp = 0usize;
            let base_m = mask_slice[lu] & ALIAS_MASK;
            if base_m != 0 {
                comps[0] = base_m;
                ncomp = 1;
            }
            while i < deferred.len() && deferred[i].0 as usize == lu {
                let mut em = deferred[i].1;
                i += 1;
                let mut j = 0;
                while j < ncomp {
                    if (comps[j] & em) != 0 {
                        em |= comps[j];
                        ncomp -= 1;
                        comps[j] = comps[ncomp];
                    } else {
                        j += 1;
                    }
                }
                comps[ncomp] = em;
                ncomp += 1;
            }
            if ncomp == 1 {
                mask_slice[lu] = (mask_slice[lu] & !ALIAS_MASK) | comps[0];
            } else if ncomp >= 2 {
                mask_slice[lu] |= IS_COMPLEX;
                has_complex_in_bucket = true;
            }
        }
    }

    if has_complex_in_bucket {
        for &e in edges {
            let a1 = e.a1();
            let a2 = e.a2();
            if a1 == a2 {
                continue;
            }
            let lu = e.local_user();
            if lu < mask_slice.len() && (mask_slice[lu] & IS_COMPLEX) != 0 {
                let u = (u_start + lu) as u32;
                let t1 = user_alias_token(u, a1, user_key);
                let t2 = user_alias_token(u, a2, user_key);
                complex_out.push((t1, t2));
            }
        }
    }
}

/// Bit `31` of `ClickSlot::key_and_alias` records an uppercase `'K...'` click ID
/// for tie-breaking and complex-user fallback without marking canonical users complex.
pub const UPPER_K_KEY_BIT: u64 = 1 << 31;

/// 16-byte Click CSR slot:
/// - `key_and_alias`: `(rel_ts << 32) | (upper_k << 31) | (camp << 4) | alias`
/// - `user_or_cid`: during parse, `(u as u64) | VALID_BIT` (or packed `(VALID_BIT | (u << 40) | cid_b36)`);
///   after CSR offset pass, `cid_b36` (`0..=MASK40`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ClickSlot {
    pub key_and_alias: u64,
    pub user_or_cid: u64,
}

impl ClickSlot {
    #[inline(always)]
    pub const fn new_parsed(rel_ts: u32, camp: u32, alias: u8, user_idx: u32) -> Self {
        Self {
            key_and_alias: ((rel_ts as u64) << 32)
                | (((camp as u64) & 0x07FF_FFFF) << 4)
                | ((alias as u64) & 0xF),
            user_or_cid: (user_idx as u64) | VALID_BIT,
        }
    }

    #[inline(always)]
    pub const fn new_resolved(rel_ts: u32, camp: u32, alias: u8, scrambled_cid: u64) -> Self {
        Self {
            key_and_alias: ((rel_ts as u64) << 32)
                | (((camp as u64) & 0x07FF_FFFF) << 4)
                | ((alias as u64) & 0xF),
            user_or_cid: scrambled_cid,
        }
    }

    #[inline(always)]
    pub const fn ts(&self) -> u32 {
        (self.key_and_alias >> 32) as u32
    }

    #[inline(always)]
    pub const fn camp(&self) -> u32 {
        ((self.key_and_alias >> 4) & 0x07FF_FFFF) as u32
    }

    #[inline(always)]
    pub const fn is_upper_k(&self) -> bool {
        (self.key_and_alias & UPPER_K_KEY_BIT) != 0
    }

    #[inline(always)]
    pub const fn alias(&self) -> u8 {
        (self.key_and_alias & 0xF) as u8
    }

    #[inline(always)]
    pub const fn sort_key60(&self) -> u64 {
        (self.key_and_alias & !UPPER_K_KEY_BIT) >> 4
    }

    #[inline(always)]
    pub const fn is_valid_parsed(&self) -> bool {
        (self.user_or_cid & VALID_BIT) != 0
    }

    #[inline(always)]
    pub const fn parsed_user(&self) -> usize {
        (self.user_or_cid & !VALID_BIT) as usize
    }
}

/// Direct Click CSR indexed by `click_index = unscramble40(cid, ckey)`.
/// Eliminates click hash-dedup and scatter streams by placing clicks directly at `csr[click_idx]`,
/// then compacting valid clicks in-place into per-user slices `csr[csr_off[u]..csr_end[u]]`.
pub struct ClickCsr {
    pub csr: Vec<ClickSlot>,
    pub csr_off: Vec<u32>,
    pub csr_end: Vec<u32>,
    pub user_cap: usize,
}

impl ClickCsr {
    pub fn with_capacity(user_cap: usize, click_cap: usize) -> Self {
        Self {
            csr: vec![ClickSlot::default(); click_cap],
            csr_off: vec![0u32; user_cap],
            csr_end: vec![0u32; user_cap],
            user_cap,
        }
    }

    /// Writes a parsed click into `csr[click_idx]`. Duplicate clicks with the same `click_idx`
    /// naturally overwrite the same slot with identical values, deduplicating in O(1).
    #[inline(always)]
    pub fn put_click(&mut self, click_idx: usize, rel_ts: u32, camp: u32, alias: u8, user_idx: u32) {
        if click_idx < self.csr.len() && (user_idx as usize) < self.user_cap && (alias as usize) < MAX_FAST_ALIASES {
            self.csr[click_idx] = ClickSlot::new_parsed(rel_ts, camp, alias, user_idx);
        }
    }

    /// Lock-free concurrent write of a parsed click into `csr[click_idx]`, packing
    /// `(VALID_BIT | (user_idx << 40) | cid_b36)` into `user_or_cid` so compaction
    /// never needs to re-scramble `click_idx`.
    #[inline(always)]
    pub unsafe fn put_click_concurrent(
        &self,
        click_idx: usize,
        rel_ts: u32,
        camp: u32,
        alias: u8,
        user_idx: u32,
        cid_b36: u64,
    ) {
        self.put_click_concurrent_cased(click_idx, rel_ts, camp, alias, user_idx, cid_b36, false);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn put_click_concurrent_cased(
        &self,
        click_idx: usize,
        rel_ts: u32,
        camp: u32,
        alias: u8,
        user_idx: u32,
        cid_b36: u64,
        is_upper_k: bool,
    ) {
        if click_idx < self.csr.len()
            && (user_idx as usize) < self.user_cap
            && (user_idx as usize) < (1 << 23)
            && (alias as usize) < MAX_FAST_ALIASES
        {
            let upper_bit = if is_upper_k { UPPER_K_KEY_BIT } else { 0 };
            let key_and_alias = ((rel_ts as u64) << 32)
                | upper_bit
                | (((camp as u64) & 0x07FF_FFFF) << 4)
                | ((alias as u64) & 0xF);
            let user_or_cid =
                VALID_BIT | (((user_idx as u64) & 0x7F_FFFF) << 40) | (cid_b36 & MASK40);
            let p = self.csr.as_ptr().add(click_idx) as *const AtomicU64;
            (*p).store(key_and_alias, Relaxed);
            (*p.add(1)).store(user_or_cid, Relaxed);
        }
    }

    /// Verifies that `u` is non-decreasing across occupied slots in `csr`, replaces `user_or_cid`
    /// with `scramble40(click_idx, click_key)`, builds `csr_off`/`csr_end` (compacting any
    /// unoccupied gaps in-place), and sorts every user's click slice using the fast bitonic sorter.
    pub fn finalize_and_sort(&mut self, click_key: u64) -> bool {
        self.csr_off.fill(0);
        self.csr_end.fill(0);
        let mut prev_u_opt: Option<usize> = None;
        let mut u_start = 0usize;
        let mut write_idx = 0usize;

        for click_idx in 0..self.csr.len() {
            let slot = self.csr[click_idx];
            if !slot.is_valid_parsed() {
                continue;
            }
            let u = slot.parsed_user();
            if u >= self.user_cap {
                return false;
            }
            match prev_u_opt {
                None => {
                    prev_u_opt = Some(u);
                }
                Some(pu) if u != pu => {
                    if u < pu {
                        return false;
                    }
                    self.csr_off[pu] = u_start as u32;
                    self.csr_end[pu] = write_idx as u32;
                    if write_idx - u_start > 1 {
                        sort_click_slice(&mut self.csr[u_start..write_idx]);
                    }
                    u_start = write_idx;
                    prev_u_opt = Some(u);
                }
                _ => {}
            }
            self.csr[write_idx] = ClickSlot {
                key_and_alias: slot.key_and_alias,
                user_or_cid: scramble40(click_idx as u64, click_key),
            };
            write_idx += 1;
        }
        if let Some(last_u) = prev_u_opt {
            self.csr_off[last_u] = u_start as u32;
            self.csr_end[last_u] = write_idx as u32;
            if write_idx - u_start > 1 {
                sort_click_slice(&mut self.csr[u_start..write_idx]);
            }
        }
        true
    }

    /// Single-pass in-place parallel compaction, `csr_off`/`csr_end` construction, and L1-hot
    /// click slice sorting across `threads` (zero extra `ClickSlot` allocation).
    /// Reads `(user_idx, cid_b36)` directly from `user_or_cid` (`VALID_BIT | (user_idx << 40) | cid_b36`).
    pub fn finalize_and_sort_par(&mut self, _click_key: u64, threads: usize) -> bool {
        use std::sync::atomic::AtomicU32;
        let n = self.csr.len();
        let user_cap = self.user_cap;
        if user_cap == 0 || n == 0 {
            self.csr_off.fill(0);
            self.csr_end.fill(0);
            self.csr.clear();
            return true;
        }
        let threads = threads.max(1);
        let num_ranges = (threads * 16).clamp(16, 128).min(n);

        // 1. Find user-aligned cut points in parallel so no user's clicks span two ranges.
        let cuts_atomic: Vec<AtomicUsize> = (0..=num_ranges).map(|_| AtomicUsize::new(0)).collect();
        cuts_atomic[num_ranges].store(n, Relaxed);
        let csr_ref = &self.csr;
        let next_cut = AtomicUsize::new(1);
        crate::sys::par(threads, |_| loop {
            let p = next_cut.fetch_add(1, Relaxed);
            if p >= num_ranges {
                break;
            }
            let start = p * n / num_ranges;
            let mut idx = start;
            while idx < n && !csr_ref[idx].is_valid_parsed() {
                idx += 1;
            }
            if idx >= n {
                cuts_atomic[p].store(n, Relaxed);
                continue;
            }
            let u0 = ((csr_ref[idx].user_or_cid >> 40) & 0x7F_FFFF) as usize;
            idx += 1;
            while idx < n {
                let s = csr_ref[idx];
                if s.is_valid_parsed() && (((s.user_or_cid >> 40) & 0x7F_FFFF) as usize) != u0 {
                    break;
                }
                idx += 1;
            }
            cuts_atomic[p].store(idx, Relaxed);
        });
        let cuts: Vec<usize> = cuts_atomic.iter().map(|a| a.load(Relaxed)).collect();

        // 2. Single-pass in-place compaction + `csr_off`/`csr_end` + L1-hot sort in disjoint ranges.
        //    Whales (`len > 2048`) are deferred to a parallel sort step right below.
        #[derive(Clone, Copy, Default)]
        struct RangeCheck {
            first_u: usize,
            last_u: usize,
            has_any: bool,
            ok: bool,
        }
        let mut checks: Vec<RangeCheck> = vec![RangeCheck::default(); num_ranges];
        let mut range_whales: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_ranges];
        let checks_ptr = checks.as_mut_ptr() as usize;
        let range_whales_ptr = range_whales.as_mut_ptr() as usize;
        let csr_ptr = self.csr.as_mut_ptr() as usize;
        let off_ptr = self.csr_off.as_mut_ptr() as usize;
        let end_ptr = self.csr_end.as_mut_ptr() as usize;
        let cuts_ref = &cuts;
        let next_r = AtomicUsize::new(0);

        crate::sys::par(threads, |_| {
            let csr_base = csr_ptr as *mut ClickSlot;
            let off_base = off_ptr as *const AtomicU32;
            let end_base = end_ptr as *const AtomicU32;
            loop {
                let r = next_r.fetch_add(1, Relaxed);
                if r >= num_ranges {
                    break;
                }
                let raw_start = cuts_ref[r];
                let raw_end = cuts_ref[r + 1];
                let mut write_idx = raw_start;
                let mut u_start = raw_start;
                let mut prev_u_opt: Option<usize> = None;
                let mut first_u = usize::MAX;
                let mut ok = true;
                let mut local_whales: Vec<(usize, usize)> = Vec::new();

                for read_idx in raw_start..raw_end {
                    let slot = unsafe { *csr_base.add(read_idx) };
                    if !slot.is_valid_parsed() {
                        continue;
                    }
                    let u = ((slot.user_or_cid >> 40) & 0x7F_FFFF) as usize;
                    if u >= user_cap {
                        ok = false;
                        break;
                    }
                    match prev_u_opt {
                        None => {
                            first_u = u;
                            prev_u_opt = Some(u);
                        }
                        Some(pu) if u != pu => {
                            if u < pu {
                                ok = false;
                                break;
                            }
                            unsafe {
                                (*off_base.add(pu)).store(u_start as u32, Relaxed);
                                (*end_base.add(pu)).store(write_idx as u32, Relaxed);
                            }
                            let len = write_idx - u_start;
                            if len > 2048 {
                                local_whales.push((u_start, write_idx));
                            } else if len > 1 {
                                let slice = unsafe {
                                    std::slice::from_raw_parts_mut(csr_base.add(u_start), len)
                                };
                                sort_click_slice(slice);
                            }
                            u_start = write_idx;
                            prev_u_opt = Some(u);
                        }
                        _ => {}
                    }
                    unsafe {
                        *csr_base.add(write_idx) = ClickSlot {
                            key_and_alias: slot.key_and_alias,
                            user_or_cid: slot.user_or_cid & MASK40,
                        };
                    }
                    write_idx += 1;
                }

                if ok {
                    if let Some(last_u) = prev_u_opt {
                        unsafe {
                            (*off_base.add(last_u)).store(u_start as u32, Relaxed);
                            (*end_base.add(last_u)).store(write_idx as u32, Relaxed);
                        }
                        let len = write_idx - u_start;
                        if len > 2048 {
                            local_whales.push((u_start, write_idx));
                        } else if len > 1 {
                            let slice = unsafe {
                                std::slice::from_raw_parts_mut(csr_base.add(u_start), len)
                            };
                            sort_click_slice(slice);
                        }
                    }
                }

                unsafe {
                    *(checks_ptr as *mut RangeCheck).add(r) = RangeCheck {
                        first_u,
                        last_u: prev_u_opt.unwrap_or(0),
                        has_any: prev_u_opt.is_some(),
                        ok,
                    };
                    if !local_whales.is_empty() {
                        *(range_whales_ptr as *mut Vec<(usize, usize)>).add(r) = local_whales;
                    }
                }
            }
        });

        let mut prev_last_u: Option<usize> = None;
        for &c in &checks {
            if !c.ok {
                return false;
            }
            if c.has_any {
                if let Some(pl) = prev_last_u {
                    if c.first_u <= pl {
                        return false;
                    }
                }
                prev_last_u = Some(c.last_u);
            }
        }

        // 3. Parallel sort for any whale user slices (`len > 2048`).
        let mut all_whales: Vec<(usize, usize)> = Vec::new();
        for rw in &mut range_whales {
            if !rw.is_empty() {
                all_whales.append(rw);
            }
        }
        if !all_whales.is_empty() {
            let next_w = AtomicUsize::new(0);
            let whales_ref = &all_whales;
            crate::sys::par(threads, |_| {
                let csr_base = csr_ptr as *mut ClickSlot;
                loop {
                    let wi = next_w.fetch_add(1, Relaxed);
                    let Some(&(s, e)) = whales_ref.get(wi) else {
                        break;
                    };
                    let slice = unsafe { std::slice::from_raw_parts_mut(csr_base.add(s), e - s) };
                    sort_click_slice(slice);
                }
            });
        }
        true
    }

    fn finalize_offsets(&mut self, click_key: u64) -> bool {
        self.csr_off.fill(0);
        self.csr_end.fill(0);
        let mut prev_u_opt: Option<usize> = None;
        let mut u_start = 0usize;
        let mut write_idx = 0usize;
        for click_idx in 0..self.csr.len() {
            let slot = self.csr[click_idx];
            if !slot.is_valid_parsed() {
                continue;
            }
            let u = slot.parsed_user();
            if u >= self.user_cap {
                return false;
            }
            match prev_u_opt {
                None => prev_u_opt = Some(u),
                Some(pu) if u != pu => {
                    if u < pu {
                        return false;
                    }
                    self.csr_off[pu] = u_start as u32;
                    self.csr_end[pu] = write_idx as u32;
                    u_start = write_idx;
                    prev_u_opt = Some(u);
                }
                _ => {}
            }
            self.csr[write_idx] = ClickSlot {
                key_and_alias: slot.key_and_alias,
                user_or_cid: scramble40(click_idx as u64, click_key),
            };
            write_idx += 1;
        }
        if let Some(last_u) = prev_u_opt {
            self.csr_off[last_u] = u_start as u32;
            self.csr_end[last_u] = write_idx as u32;
        }
        true
    }

    #[inline(always)]
    pub fn csr_slice(&self, user_idx: usize) -> &[ClickSlot] {
        let start = self.csr_off[user_idx] as usize;
        let end = self.csr_end[user_idx] as usize;
        &self.csr[start..end]
    }
}

/// Canonical comparator for two resolved `ClickSlot`s:
/// 1. Primary: `ts` ascending (`key_and_alias >> 32`).
/// 2. When `ts` is identical and `camp` differs: orders by base36 `click_id` DESCENDING
///    (largest base36 `click_id` first, smallest base36 `click_id` LAST) so that the rightmost
///    (latest) click at timestamp `ts` (`hi - 1` or highest set bit `31 - win_mask.leading_zeros()`)
///    is ALWAYS the winning click with the smallest base36 `click_id`!
/// 3. When `ts` and `camp` are both identical: orders by `key_and_alias >> 4`, then base36 `click_id` descending.
#[inline(always)]
pub fn cmp_click_slot(a: &ClickSlot, b: &ClickSlot) -> Ordering {
    let ta = a.ts();
    let tb = b.ts();
    match ta.cmp(&tb) {
        Ordering::Equal => {
            // Order smallest click_id LAST (`'K...' < 'k...'`), so latest-click lookup
            // selects the smallest base36 click_id.
            let cid_ord = a.is_upper_k().cmp(&b.is_upper_k()).then_with(|| {
                cmp_scrambled_base36(b.user_or_cid & MASK40, a.user_or_cid & MASK40)
            });
            if a.camp() != b.camp() {
                cid_ord
            } else {
                cid_ord.then_with(|| a.alias().cmp(&b.alias()))
            }
        }
        ord => ord,
    }
}

/// Checks if a timestamp-sorted `ClickSlot` slice has any adjacent pair with identical `ts`
/// and different `camp`, and resolves any equal-`ts` runs using `cmp_click_slot`.
#[inline(always)]
fn fix_same_ts_ties(slice: &mut [ClickSlot]) {
    let n = slice.len();
    if n <= 1 {
        return;
    }
    let mut i = 1usize;
    while i < n {
        let prev = slice[i - 1];
        let cur = slice[i];
        if prev.ts() == cur.ts() && prev.camp() != cur.camp() {
            let t = prev.ts();
            let mut start = i - 1;
            while start > 0 && slice[start - 1].ts() == t {
                start -= 1;
            }
            let mut end = i + 1;
            while end < n && slice[end].ts() == t {
                end += 1;
            }
            // Insertion sort on the small equal-ts run (almost always length 2).
            let run = &mut slice[start..end];
            for j in 1..run.len() {
                let key = run[j];
                let mut k = j;
                while k > 0 && cmp_click_slot(&key, &run[k - 1]) == Ordering::Less {
                    run[k] = run[k - 1];
                    k -= 1;
                }
                run[k] = key;
            }
            i = end;
        } else {
            i += 1;
        }
    }
}

#[inline(always)]
fn cas_slot(slice: &mut [ClickSlot], i: usize, j: usize) {
    if slice[i].sort_key60() > slice[j].sort_key60() {
        slice.swap(i, j);
    }
}

/// Ultra-fast click slice sorter:
/// - `n == 2, 3, 4`: optimal register comparison networks on `key_and_alias >> 4`.
/// - `n <= 8` and `n <= 16`: SIMD bitonic sorting network (AVX-512 on Cascade Lake, AVX2 on Zen 3)
///   packing `orig_idx (0..16)` into the low 4 bits of `(key_and_alias & !0xF) | orig_idx`.
/// - `n > 16`: unstable sort on `key_and_alias >> 4`.
/// - Post-pass: only invokes `fix_same_ts_ties` (base36 `click_id` comparison) when adjacent
///   sorted clicks have identical `ts` and different `camp`.
#[inline(always)]
pub fn sort_click_slice(slice: &mut [ClickSlot]) {
    let n = slice.len();
    match n {
        0 | 1 => return,
        2 => {
            cas_slot(slice, 0, 1);
            if slice[0].ts() == slice[1].ts() && slice[0].camp() != slice[1].camp() {
                if cmp_click_slot(&slice[0], &slice[1]) == Ordering::Greater {
                    slice.swap(0, 1);
                }
            }
            return;
        }
        3 => {
            cas_slot(slice, 0, 2);
            cas_slot(slice, 0, 1);
            cas_slot(slice, 1, 2);
            fix_same_ts_ties(slice);
            return;
        }
        4 => {
            cas_slot(slice, 0, 2);
            cas_slot(slice, 1, 3);
            cas_slot(slice, 0, 1);
            cas_slot(slice, 2, 3);
            cas_slot(slice, 1, 2);
            fix_same_ts_ties(slice);
            return;
        }
        5..=8 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
                    sort_clicks_le8_avx512(slice);
                } else {
                    sort_clicks_le8_avx2(slice);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            slice.sort_unstable_by_key(|s| s.sort_key60());
            fix_same_ts_ties(slice);
            return;
        }
        9..=16 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
                    sort_clicks_le16_avx512(slice);
                } else {
                    sort_clicks_le16_avx2(slice);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            slice.sort_unstable_by_key(|s| s.sort_key60());
            fix_same_ts_ties(slice);
            return;
        }
        _ => {
            slice.sort_unstable_by_key(|s| s.sort_key60());
            fix_same_ts_ties(slice);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod simd_bitonic {
    use super::ClickSlot;
    use std::arch::x86_64::*;

    const SIGN64: i64 = i64::MIN;

    /// Unsigned 64-bit `(min, max)` on two 4-lane `__m256i` vectors in AVX2.
    #[inline(always)]
    unsafe fn minmax_epu64_avx2(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
        let sign = _mm256_set1_epi64x(SIGN64);
        let a_s = _mm256_xor_si256(a, sign);
        let b_s = _mm256_xor_si256(b, sign);
        let gt = _mm256_cmpgt_epi64(a_s, b_s);
        let min = _mm256_blendv_epi8(a, b, gt);
        let max = _mm256_blendv_epi8(b, a, gt);
        (min, max)
    }

    /// Sort 4 unsigned `u64` lanes inside a single `__m256i` ascending using a 3-step bitonic network.
    #[inline(always)]
    unsafe fn sort4_u64_avx2(mut v: __m256i) -> __m256i {
        // Step 1: compare (0,1) and (2,3), perm = [1,0,3,2] = 0xB1, max at lanes {1,3} = 0xA (blendpd 0b1010)
        let p1 = _mm256_permute4x64_epi64::<0xB1>(v);
        let (min1, max1) = minmax_epu64_avx2(v, p1);
        v = _mm256_castpd_si256(_mm256_blend_pd::<0b1010>(
            _mm256_castsi256_pd(min1),
            _mm256_castsi256_pd(max1),
        ));

        // Step 2: compare (0,3) and (1,2), perm = [3,2,1,0] = 0x1B, max at lanes {2,3} = 0b1100
        let p2 = _mm256_permute4x64_epi64::<0x1B>(v);
        let (min2, max2) = minmax_epu64_avx2(v, p2);
        v = _mm256_castpd_si256(_mm256_blend_pd::<0b1100>(
            _mm256_castsi256_pd(min2),
            _mm256_castsi256_pd(max2),
        ));

        // Step 3: compare (0,1) and (2,3), perm = [1,0,3,2] = 0xB1, max at lanes {1,3} = 0b1010
        let p3 = _mm256_permute4x64_epi64::<0xB1>(v);
        let (min3, max3) = minmax_epu64_avx2(v, p3);
        _mm256_castpd_si256(_mm256_blend_pd::<0b1010>(
            _mm256_castsi256_pd(min3),
            _mm256_castsi256_pd(max3),
        ))
    }

    /// Bitonic merge of a 4-lane bitonic `__m256i` (distance-2 then distance-1).
    #[inline(always)]
    unsafe fn merge4_u64_avx2(mut v: __m256i) -> __m256i {
        // Distance 2: compare (0,2) and (1,3), perm = [2,3,0,1] = 0x4E, max at lanes {2,3} = 0b1100
        let p1 = _mm256_permute4x64_epi64::<0x4E>(v);
        let (min1, max1) = minmax_epu64_avx2(v, p1);
        v = _mm256_castpd_si256(_mm256_blend_pd::<0b1100>(
            _mm256_castsi256_pd(min1),
            _mm256_castsi256_pd(max1),
        ));

        // Distance 1: compare (0,1) and (2,3), perm = [1,0,3,2] = 0xB1, max at lanes {1,3} = 0b1010
        let p2 = _mm256_permute4x64_epi64::<0xB1>(v);
        let (min2, max2) = minmax_epu64_avx2(v, p2);
        _mm256_castpd_si256(_mm256_blend_pd::<0b1010>(
            _mm256_castsi256_pd(min2),
            _mm256_castsi256_pd(max2),
        ))
    }

    /// Sort 8 unsigned `u64` lanes in two `__m256i` registers `(a0, a1)` ascending.
    #[inline(always)]
    unsafe fn sort8_u64_avx2(mut a0: __m256i, mut a1: __m256i) -> (__m256i, __m256i) {
        a0 = sort4_u64_avx2(a0);
        a1 = sort4_u64_avx2(a1);
        // Reverse a1 and compare-exchange with a0: (0,7), (1,6), (2,5), (3,4)
        let a1_rev = _mm256_permute4x64_epi64::<0x1B>(a1);
        let (lo, hi) = minmax_epu64_avx2(a0, a1_rev);
        (merge4_u64_avx2(lo), merge4_u64_avx2(hi))
    }

    #[target_feature(enable = "avx2,bmi2,lzcnt")]
    pub unsafe fn sort_clicks_le8_avx2(slice: &mut [ClickSlot]) {
        let n = slice.len();
        let mut keys = [u64::MAX; 8];
        let mut orig = [ClickSlot::default(); 8];
        for i in 0..n {
            orig[i] = *slice.get_unchecked(i);
            keys[i] = (orig[i].key_and_alias & !0xFu64) | (i as u64);
        }
        let a0 = _mm256_loadu_si256(keys.as_ptr() as *const __m256i);
        let a1 = _mm256_loadu_si256(keys.as_ptr().add(4) as *const __m256i);
        let (s0, s1) = sort8_u64_avx2(a0, a1);
        _mm256_storeu_si256(keys.as_mut_ptr() as *mut __m256i, s0);
        _mm256_storeu_si256(keys.as_mut_ptr().add(4) as *mut __m256i, s1);
        for i in 0..n {
            let idx = (keys[i] & 0xF) as usize;
            *slice.get_unchecked_mut(i) = orig[idx];
        }
    }

    #[target_feature(enable = "avx2,bmi2,lzcnt")]
    pub unsafe fn sort_clicks_le16_avx2(slice: &mut [ClickSlot]) {
        let n = slice.len();
        let mut keys = [u64::MAX; 16];
        let mut orig = [ClickSlot::default(); 16];
        for i in 0..n {
            orig[i] = *slice.get_unchecked(i);
            keys[i] = (orig[i].key_and_alias & !0xFu64) | (i as u64);
        }
        let a0 = _mm256_loadu_si256(keys.as_ptr() as *const __m256i);
        let a1 = _mm256_loadu_si256(keys.as_ptr().add(4) as *const __m256i);
        let b0 = _mm256_loadu_si256(keys.as_ptr().add(8) as *const __m256i);
        let b1 = _mm256_loadu_si256(keys.as_ptr().add(12) as *const __m256i);

        let (a0, a1) = sort8_u64_avx2(a0, a1);
        let (b0, b1) = sort8_u64_avx2(b0, b1);

        // Step 7: compare (0..8) with reversed (8..16)
        let b1_rev = _mm256_permute4x64_epi64::<0x1B>(b1);
        let b0_rev = _mm256_permute4x64_epi64::<0x1B>(b0);
        let (lo0, hi1) = minmax_epu64_avx2(a0, b1_rev);
        let (lo1, hi0) = minmax_epu64_avx2(a1, b0_rev);

        // Step 8: distance-4 compare within each 8-element half
        let (lo0, lo1) = minmax_epu64_avx2(lo0, lo1);
        let (hi0, hi1) = minmax_epu64_avx2(hi0, hi1);

        // Steps 9 & 10: distance-2 and distance-1 merge on each 4-lane vector
        let lo0 = merge4_u64_avx2(lo0);
        let lo1 = merge4_u64_avx2(lo1);
        let hi0 = merge4_u64_avx2(hi0);
        let hi1 = merge4_u64_avx2(hi1);

        _mm256_storeu_si256(keys.as_mut_ptr() as *mut __m256i, lo0);
        _mm256_storeu_si256(keys.as_mut_ptr().add(4) as *mut __m256i, lo1);
        _mm256_storeu_si256(keys.as_mut_ptr().add(8) as *mut __m256i, hi0);
        _mm256_storeu_si256(keys.as_mut_ptr().add(12) as *mut __m256i, hi1);

        for i in 0..n {
            let idx = (keys[i] & 0xF) as usize;
            *slice.get_unchecked_mut(i) = orig[idx];
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
    unsafe fn cas8_u64_avx512(v: __m512i, perm: __m512i, max_mask: __mmask8) -> __m512i {
        let p = _mm512_permutexvar_epi64(perm, v);
        let lo = _mm512_min_epu64(v, p);
        let hi = _mm512_max_epu64(v, p);
        _mm512_mask_mov_epi64(lo, max_mask, hi)
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
    unsafe fn sort8_u64_avx512(mut v: __m512i) -> __m512i {
        let p_d1 = _mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6);
        let p_rev4 = _mm512_setr_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        let p_rev8 = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        let p_d2 = _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5);

        v = cas8_u64_avx512(v, p_d1, 0xAA);
        v = cas8_u64_avx512(v, p_rev4, 0xCC);
        v = cas8_u64_avx512(v, p_d1, 0xAA);
        v = cas8_u64_avx512(v, p_rev8, 0xF0);
        v = cas8_u64_avx512(v, p_d2, 0xCC);
        cas8_u64_avx512(v, p_d1, 0xAA)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
    pub unsafe fn sort_clicks_le8_avx512(slice: &mut [ClickSlot]) {
        let n = slice.len();
        let mut keys = [u64::MAX; 8];
        let mut orig = [ClickSlot::default(); 8];
        for i in 0..n {
            orig[i] = *slice.get_unchecked(i);
            keys[i] = (orig[i].key_and_alias & !0xFu64) | (i as u64);
        }
        let v = _mm512_loadu_si512(keys.as_ptr() as *const __m512i);
        let sorted = sort8_u64_avx512(v);
        _mm512_storeu_si512(keys.as_mut_ptr() as *mut __m512i, sorted);
        for i in 0..n {
            let idx = (keys[i] & 0xF) as usize;
            *slice.get_unchecked_mut(i) = orig[idx];
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
    pub unsafe fn sort_clicks_le16_avx512(slice: &mut [ClickSlot]) {
        let n = slice.len();
        let mut keys = [u64::MAX; 16];
        let mut orig = [ClickSlot::default(); 16];
        for i in 0..n {
            orig[i] = *slice.get_unchecked(i);
            keys[i] = (orig[i].key_and_alias & !0xFu64) | (i as u64);
        }
        let v0 = sort8_u64_avx512(_mm512_loadu_si512(keys.as_ptr() as *const __m512i));
        let v1 = sort8_u64_avx512(_mm512_loadu_si512(keys.as_ptr().add(8) as *const __m512i));

        let p_rev8 = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        let p_d4 = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);
        let p_d2 = _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5);
        let p_d1 = _mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6);

        let v1_rev = _mm512_permutexvar_epi64(p_rev8, v1);
        let mut lo = _mm512_min_epu64(v0, v1_rev);
        let mut hi = _mm512_max_epu64(v0, v1_rev);

        lo = cas8_u64_avx512(lo, p_d4, 0xF0);
        hi = cas8_u64_avx512(hi, p_d4, 0xF0);
        lo = cas8_u64_avx512(lo, p_d2, 0xCC);
        hi = cas8_u64_avx512(hi, p_d2, 0xCC);
        lo = cas8_u64_avx512(lo, p_d1, 0xAA);
        hi = cas8_u64_avx512(hi, p_d1, 0xAA);

        _mm512_storeu_si512(keys.as_mut_ptr() as *mut __m512i, lo);
        _mm512_storeu_si512(keys.as_mut_ptr().add(8) as *mut __m512i, hi);
        for i in 0..n {
            let idx = (keys[i] & 0xF) as usize;
            *slice.get_unchecked_mut(i) = orig[idx];
        }
    }
}

#[cfg(target_arch = "x86_64")]
use simd_bitonic::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scramble_unscramble_bijection_and_tokens() {
        let key = 0x517C_C1B7_2722_0A95u64 & MASK40;
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for i in 0..50_000u64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let v = if i < 1024 { i } else { rng & MASK40 };
            let s = scramble40(v, key);
            assert_eq!(unscramble40(s, key), v, "failed roundtrip for {v}");

            let mut buf = [0u8; 16];
            let len = write_prefixed_base36(b'u', s, &mut buf);
            let decoded = decode_prefixed_base36(&buf[..len], b'u');
            assert_eq!(decoded, Some(s));

            let tok_fast = encode_base36_token(b'u', s);
            let tok_ref = unsafe { crate::ids::tok(buf.as_ptr(), len) };
            assert_eq!(tok_fast, tok_ref);
        }
    }

    #[test]
    fn test_bucketed_link_resolution_simple_deferred_and_complex() {
        let user_key = 0x3141_5926_5358_9793u64 & MASK40;
        let mut tab = UserAliasTable::new(2048, user_key);

        // User 0: simple connected star (0-1, 0-2, 2-3) -> mask 0b1111
        // User 1: out-of-order bridge: (0,1), then disjoint (2,3), then bridge (1,2) -> merges to 0b1111!
        // User 2: genuinely disconnected components of size >= 2: (0,1) and (2,3) -> IS_COMPLEX!
        // User 1025 (bucket 1): single edge (4, 7) -> mask (1<<4)|(1<<7)
        // User 1026 (bucket 1): marked complex externally (e.g. kiosk) with edge (0,1)
        let intra = vec![
            IntraEdge::new(0, 0, 1),
            IntraEdge::new(0, 0, 2),
            IntraEdge::new(0, 2, 3),
            IntraEdge::new(1, 0, 1),
            IntraEdge::new(1, 2, 3),
            IntraEdge::new(1, 1, 2),
            IntraEdge::new(2, 0, 1),
            IntraEdge::new(2, 2, 3),
            IntraEdge::new(1025, 4, 7),
            IntraEdge::new(1026, 0, 1),
        ];
        let kiosk_tok = encode_base36_token(b'u', scramble40(KIOSK_BASE + 1, user_key));
        let u1026_a0 = user_alias_token(1026, 0, user_key);
        tab.build_from_intra_and_complex(&intra, &[1026], &[(u1026_a0, kiosk_tok)]);

        assert_eq!(tab.alias_masks[0], 0b1111);
        assert_eq!(tab.alias_masks[1], 0b1111);
        assert_ne!(tab.alias_masks[2] & IS_COMPLEX, 0);
        assert_eq!(tab.alias_masks[1025], (1u16 << 4) | (1u16 << 7));
        assert_ne!(tab.alias_masks[1026] & IS_COMPLEX, 0);
        // Complex edges should contain the 1 external kiosk edge + 2 edges from user 2 + 1 edge from user 1026 = 4 edges
        assert_eq!(tab.complex_edges.len(), 4);
    }

    #[test]
    fn test_click_csr_and_bitonic_sort_vs_scalar() {
        let ckey = 0x2718_2818_2845_9045u64 & MASK40;
        let mut rng = 0xCAFE_BABE_DEAD_BEEFu64;
        let mut next_u64 = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        // Test every click slice length from 0 to 25 across many random trials, including same-ts ties!
        for len in 0..=25 {
            for trial in 0..200 {
                let mut slice = Vec::with_capacity(len);
                for k in 0..len {
                    // Force timestamp ties frequently on even trials
                    let ts = if trial % 2 == 0 {
                        (next_u64() % 4) as u32 * 100
                    } else {
                        (next_u64() % 1_000_000) as u32
                    };
                    let camp = (next_u64() % 8) as u32;
                    let alias = (next_u64() % 6) as u8;
                    let cid = scramble40((trial as u64) * 64 + (k as u64), ckey);
                    slice.push(ClickSlot::new_resolved(ts, camp, alias, cid));
                }
                let mut fast = slice.clone();
                sort_click_slice(&mut fast);

                // Verify timestamps are monotonically non-decreasing
                for w in fast.windows(2) {
                    assert!(w[0].ts() <= w[1].ts(), "ts out of order for len={len}");
                }

                // Verify that for every distinct timestamp `t` present in the slice,
                // the LAST click with `ts == t` has the campaign of the click with the
                // lexicographically smallest base36 click_id among all clicks at `ts == t`!
                let mut i = 0;
                while i < len {
                    let t = fast[i].ts();
                    let mut j = i + 1;
                    while j < len && fast[j].ts() == t {
                        j += 1;
                    }
                    let expected_winner = slice
                        .iter()
                        .filter(|c| c.ts() == t)
                        .min_by(|a, b| cmp_scrambled_base36(a.user_or_cid, b.user_or_cid))
                        .unwrap();
                    assert_eq!(
                        fast[j - 1].camp(),
                        expected_winner.camp(),
                        "wrong tie-break winner at len={len}, trial={trial}"
                    );
                    i = j;
                }
            }
        }

        // Also test full ClickCsr pipeline with sparse holes and multiple users
        let mut csr = ClickCsr::with_capacity(4, 16);
        csr.put_click(0, 500, 3, 0, 0);
        csr.put_click(1, 200, 1, 1, 0);
        csr.put_click(2, 200, 2, 0, 0);
        // slot 3 left empty (simulating dropped bad click)
        csr.put_click(4, 900, 4, 2, 2);
        csr.put_click(5, 100, 1, 0, 2);
        csr.put_click(6, 700, 5, 1, 2);
        csr.put_click(7, 300, 2, 0, 2);
        csr.put_click(8, 400, 3, 3, 2);
        assert!(csr.finalize_and_sort(ckey));

        assert_eq!(csr.csr_slice(0).len(), 3);
        assert_eq!(csr.csr_slice(1).len(), 0);
        assert_eq!(csr.csr_slice(2).len(), 5);
        assert_eq!(csr.csr_slice(3).len(), 0);
        let u2_ts: Vec<u32> = csr.csr_slice(2).iter().map(|c| c.ts()).collect();
        assert_eq!(u2_ts, vec![100, 300, 400, 700, 900]);

        // Also test concurrent put_click_concurrent + parallel finalize_and_sort_par across sparse gaps
        let mut p_csr = ClickCsr::with_capacity(64, 512);
        unsafe {
            for idx in 0..400usize {
                if idx % 7 == 3 {
                    continue;
                }
                let u = (idx / 10) as u32;
                let cid = scramble40(idx as u64, ckey);
                p_csr.put_click_concurrent(idx, ((400 - idx) * 10) as u32, (idx % 5) as u32, (idx % 3) as u8, u, cid);
            }
        }
        assert!(p_csr.finalize_and_sort_par(ckey, 4));
        for u in 0..40usize {
            let sl = p_csr.csr_slice(u);
            assert!(!sl.is_empty());
            for w in sl.windows(2) {
                assert!(w[0].ts() <= w[1].ts());
            }
        }
        assert_eq!(p_csr.csr_slice(50).len(), 0);

        // Also test multi_comp_users in resolve_thread_buckets_par
        let ukey = 0x3141_5926_5358_9793u64 & MASK40;
        let mut mtab = UserAliasTable::new(1024, ukey);
        let b0 = vec![vec![IntraEdge::new(5, 0, 1)]];
        let b1 = vec![vec![IntraEdge::new(5, 2, 3), IntraEdge::new(5, 3, 4)]];
        mtab.resolve_thread_buckets_par(&[b0, b1], 2);
        assert_eq!(mtab.alias_masks[5] & IS_COMPLEX, 0);
        assert_ne!(mtab.alias_masks[5] & HAS_EXTRA_EVENTS, 0);
        assert_eq!(mtab.multi_comp_users.len(), 1);
        assert_eq!(mtab.multi_comp_users[0].0, 5);
        assert_eq!(mtab.multi_comp_users[0].2, 2);

        // Test uppercase 'K...' click encoding/decoding and equal-ts tie breaking ('K...' < 'k...')
        let tok_to_vec = |t: u64| {
            let mut v = Vec::new();
            crate::ids::tok_bytes(t, &mut v);
            v
        };
        let k_low = scramble40(42, ckey);
        let k_up = scramble40(99, ckey);
        let tok_up = encode_upper_k_token(k_up);
        let tok_up_bytes = tok_to_vec(tok_up);
        assert_eq!(tok_up_bytes[0], b'K');
        let decoded = unsafe {
            crate::unscramble::decode_upper_b36_ptr(
                tok_up_bytes.as_ptr(),
                tok_up_bytes.len(),
                b'K',
            )
        };
        assert_eq!(decoded, Some(k_up));

        for len in [2usize, 4, 8, 12, 20] {
            let mut ucsr = ClickCsr::with_capacity(2, len + 2);
            unsafe {
                ucsr.put_click_concurrent_cased(0, 500, 1, 0, 0, k_low, false);
                ucsr.put_click_concurrent_cased(1, 500, 2, 0, 0, k_up, true);
                for idx in 2..len {
                    let cid = scramble40(idx as u64 + 1000, ckey);
                    ucsr.put_click_concurrent_cased(
                        idx,
                        500,
                        ((idx % 5) + 3) as u32,
                        (idx % 4) as u8,
                        0,
                        cid,
                        idx % 3 == 0,
                    );
                }
            }
            assert!(ucsr.finalize_and_sort_par(ckey, 2));
            let sl = ucsr.csr_slice(0);
            assert_eq!(sl.len(), len);
            let winner = sl.last().unwrap();
            let expected_winner_cid = sl
                .iter()
                .map(|c| {
                    let tok = if c.is_upper_k() {
                        encode_upper_k_token(c.user_or_cid)
                    } else {
                        encode_base36_token(b'k', c.user_or_cid)
                    };
                    tok_to_vec(tok)
                })
                .min()
                .unwrap();
            let actual_winner_tok = if winner.is_upper_k() {
                encode_upper_k_token(winner.user_or_cid)
            } else {
                encode_base36_token(b'k', winner.user_or_cid)
            };
            assert_eq!(tok_to_vec(actual_winner_tok), expected_winner_cid);
        }
    }
}
