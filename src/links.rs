use crate::ids::tok_less;
use crate::sys::{map_zeroed_huge, mix, release};
use std::arch::x86_64::{
    __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_load_si256, _mm256_movemask_pd,
    _mm256_set1_epi64x, _mm256_setzero_si256, _mm_prefetch, _MM_HINT_T0,
};
use std::sync::atomic::{
    AtomicU64,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicBool;

const SWEEP_PREFETCH_BUCKETS: usize = 4;

#[repr(C, align(64))]
struct Bucket {
    keys: [AtomicU64; 4],
    /// While building: parent slot + 1 (`v != 0 && (v >> 56) == 0`), or `0` for a root.
    /// After `compress_and_resolve`: the person's token (`(v >> 56) != 0`).
    vals: [AtomicU64; 4],
}

/// Lock-free 64 B SoA 4-slot ID table plus union-find. A root always carries the
/// byte-wise smallest ID of its set, because a union only ever points the
/// larger-named root at the smaller-named one.
pub struct LinkTable {
    buckets: *const Bucket,
    bucket_mask: usize,
    #[cfg(debug_assertions)]
    dead: AtomicBool,
}
unsafe impl Sync for LinkTable {}

impl LinkTable {
    pub fn new(pairs: usize) -> Self {
        let cap = (pairs.saturating_mul(5) / 2).next_power_of_two().max(4);
        Self::with_buckets(cap >> 2)
    }

    fn with_buckets(num_buckets: usize) -> Self {
        debug_assert!(num_buckets.is_power_of_two() && num_buckets >= 1);
        let buckets = map_zeroed_huge(num_buckets * std::mem::size_of::<Bucket>()) as *const Bucket;
        debug_assert_eq!(buckets as usize & 63, 0);
        LinkTable {
            buckets,
            bucket_mask: num_buckets - 1,
            #[cfg(debug_assertions)]
            dead: AtomicBool::new(false),
        }
    }

    /// Marks the table dead (`DROP_LINKS`): no lookup or union may touch it afterwards.
    pub fn mark_dead(&self) {
        #[cfg(debug_assertions)]
        self.dead.store(true, Release);
    }

    /// Exposes the dead table's backing pages as mutable bytes for slab recycling during `dedup`.
    /// Safety: caller must guarantee no thread touches `LinkTable` lookups/unions in or after `dedup`.
    pub unsafe fn as_mut_bytes(&self) -> &'static mut [u8] {
        self.mark_dead();
        let bytes = (self.bucket_mask + 1) * std::mem::size_of::<Bucket>();
        std::slice::from_raw_parts_mut(self.buckets as *mut u8, bytes)
    }

    /// Frees the table's pages in `[from, to)` slot ranges once no lookup can happen.
    pub fn release(&self, from: usize, to: usize) {
        release(self.buckets.wrapping_add(from >> 2) as *mut u8, (to - from) * 16);
    }

    pub fn slots(&self) -> usize {
        (self.bucket_mask + 1) << 2
    }

    #[inline(always)]
    unsafe fn slot_key(&self, s: usize) -> &AtomicU64 {
        &(*self.buckets.add(s >> 2)).keys[s & 3]
    }

    #[inline(always)]
    unsafe fn slot_val(&self, s: usize) -> &AtomicU64 {
        &(*self.buckets.add(s >> 2)).vals[s & 3]
    }

    /// Prefetches the 64 B bucket of a token whose `mix` is `h`.
    #[inline(always)]
    pub fn prefetch_hashed(&self, h: u64) {
        #[cfg(debug_assertions)]
        debug_assert!(!self.dead.load(Relaxed), "LinkTable accessed after DROP_LINKS");
        unsafe { _mm_prefetch(self.buckets.add(h as usize & self.bucket_mask) as *const i8, _MM_HINT_T0) }
    }

    /// Inserts `token` with precomputed `h = mix(token)`.
    /// Lanes 0..4 are filled monotonically before spilling to the next bucket so
    /// `keys[3] == 0` strictly proves no insert at this bucket ever overflowed.
    #[inline(always)]
    unsafe fn insert_hashed(&self, token: u64, h: u64) -> usize {
        debug_assert!(token >> 56 != 0);
        let mut b_idx = h as usize & self.bucket_mask;
        loop {
            let b = &*self.buckets.add(b_idx);
            for lane in 0..4 {
                let k = b.keys[lane].load(Acquire);
                if k == token {
                    return (b_idx << 2) | lane;
                }
                if k == 0 {
                    match b.keys[lane].compare_exchange(0, token, AcqRel, Acquire) {
                        Ok(_) => return (b_idx << 2) | lane,
                        Err(now) if now == token => return (b_idx << 2) | lane,
                        Err(_) => {}
                    }
                }
            }
            b_idx = (b_idx + 1) & self.bucket_mask;
        }
    }

    #[inline(always)]
    unsafe fn find(&self, mut i: usize) -> usize {
        loop {
            let p = self.slot_val(i).load(Acquire);
            if p == 0 {
                return i;
            }
            let p_slot = p as usize - 1;
            let g = self.slot_val(p_slot).load(Acquire);
            if g == 0 {
                return p_slot;
            }
            let _ = self.slot_val(i).compare_exchange(p, g, AcqRel, Acquire);
            i = g as usize - 1;
        }
    }

    #[inline(always)]
    pub unsafe fn link_hashed(&self, a: u64, ha: u64, b: u64, hb: u64) {
        if a == b {
            return;
        }
        let (mut x, mut y) = (self.insert_hashed(a, ha), self.insert_hashed(b, hb));
        loop {
            x = self.find(x);
            y = self.find(y);
            if x == y {
                return;
            }
            if tok_less(self.slot_key(x).load(Acquire), self.slot_key(y).load(Acquire)) {
                std::mem::swap(&mut x, &mut y);
            }
            if self.slot_val(x).load(Relaxed) == 0
                && self.slot_val(x).compare_exchange(0, y as u64 + 1, AcqRel, Acquire).is_ok()
            {
                return;
            }
        }
    }

    /// Links every pair, prefetching and caching the `mix()` hashes 8 pairs ahead.
    pub unsafe fn link_all(&self, pairs: &[(u64, u64)]) {
        const AHEAD: usize = 8;
        let mut hashes = [(0u64, 0u64); AHEAD];
        let pre = pairs.len().min(AHEAD);
        for k in 0..pre {
            let (a, b) = *pairs.get_unchecked(k);
            let (ha, hb) = (mix(a), mix(b));
            hashes[k] = (ha, hb);
            self.prefetch_hashed(ha);
            self.prefetch_hashed(hb);
        }
        for k in 0..pairs.len() {
            let (a, b) = *pairs.get_unchecked(k);
            let (ha, hb) = hashes[k & (AHEAD - 1)];
            if k + AHEAD < pairs.len() {
                let (na, nb) = *pairs.get_unchecked(k + AHEAD);
                let (nha, nhb) = (mix(na), mix(nb));
                hashes[k & (AHEAD - 1)] = (nha, nhb);
                self.prefetch_hashed(nha);
                self.prefetch_hashed(nhb);
            }
            self.link_hashed(a, ha, b, hb);
        }
    }

    /// Single-pass lock-free path compression and root-token resolution using the bit-56
    /// invariant: valid tokens satisfy `(t >> 56) != 0`, whereas `parent_slot + 1` has
    /// `(v >> 56) == 0` and `0` marks an unresolved root. Strictly iterative.
    pub unsafe fn compress_and_resolve(&self, from: usize, to: usize) {
        let from_b = from >> 2;
        let to_b = to >> 2;
        for b_idx in from_b..to_b {
            if b_idx + SWEEP_PREFETCH_BUCKETS < to_b {
                let pb = &*self.buckets.add(b_idx + SWEEP_PREFETCH_BUCKETS);
                for lane in 0..4 {
                    let p = pb.vals[lane].load(Relaxed);
                    if p != 0 && (p >> 56) == 0 {
                        _mm_prefetch(self.buckets.add((p as usize - 1) >> 2) as *const i8, _MM_HINT_T0);
                    }
                }
            }
            let b = &*self.buckets.add(b_idx);
            for lane in 0..4 {
                let k = b.keys[lane].load(Relaxed);
                if k == 0 {
                    // insert_hashed fills lanes 0..4 monotonically, so remaining lanes are empty.
                    break;
                }
                let v = b.vals[lane].load(Acquire);
                if (v >> 56) != 0 {
                    continue;
                }
                if v == 0 {
                    b.vals[lane].store(k, Release);
                    continue;
                }
                let first_p = v as usize - 1;
                let mut cur = first_p;
                let root_tok = loop {
                    let nv = self.slot_val(cur).load(Acquire);
                    if nv == 0 {
                        let kt = self.slot_key(cur).load(Relaxed);
                        self.slot_val(cur).store(kt, Release);
                        break kt;
                    }
                    if (nv >> 56) != 0 {
                        break nv;
                    }
                    cur = nv as usize - 1;
                };
                b.vals[lane].store(root_tok, Release);
                if cur != first_p {
                    let mut w = first_p;
                    loop {
                        let nv = self.slot_val(w).load(Acquire);
                        if nv == 0 || (nv >> 56) != 0 {
                            break;
                        }
                        self.slot_val(w).store(root_tok, Release);
                        w = nv as usize - 1;
                    }
                }
            }
        }
    }

    /// The person an ID belongs to, given `h = mix(token)`; an ID no link
    /// mentions is its own person. Read-only during `parse`.
    #[inline(always)]
    pub unsafe fn person_hashed(&self, token: u64, h: u64) -> u64 {
        #[cfg(debug_assertions)]
        debug_assert!(!self.dead.load(Relaxed), "LinkTable accessed after DROP_LINKS");
        debug_assert!(token >> 56 != 0);
        let target = _mm256_set1_epi64x(token as i64);
        let zero = _mm256_setzero_si256();
        let mut b_idx = h as usize & self.bucket_mask;
        loop {
            let b = self.buckets.add(b_idx);
            let keys = _mm256_load_si256(b as *const __m256i);
            let eq = _mm256_cmpeq_epi64(keys, target);
            let m = _mm256_movemask_pd(_mm256_castsi256_pd(eq)) as u32;
            let z = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(keys, zero))) as u32;
            if (m | (z & 8)) != 0 {
                let lane = (m.trailing_zeros() & 3) as usize;
                let v = (*b).vals.get_unchecked(lane).load(Relaxed);
                return if m != 0 { v } else { token };
            }
            b_idx = (b_idx + 1) & self.bucket_mask;
        }
    }

    #[inline(always)]
    pub unsafe fn slot_kv_relaxed(&self, s: usize) -> (u64, u64) {
        let b = &*self.buckets.add(s >> 2);
        let lane = s & 3;
        (b.keys[lane].load(Relaxed), b.vals[lane].load(Relaxed))
    }

    #[inline(always)]
    pub unsafe fn set_slot_val_relaxed(&self, s: usize, val: u64) {
        let b = &*self.buckets.add(s >> 2);
        b.vals[s & 3].store(val, Relaxed);
    }

    /// Restores root `vals` back to their root tokens if V6 falls back to V5 after seeding.
    pub unsafe fn restore_roots(&self, from: usize, to: usize) {
        for s in from..to {
            let b = &*self.buckets.add(s >> 2);
            let lane = s & 3;
            let k = b.keys[lane].load(Relaxed);
            if k != 0 && (b.vals[lane].load(Relaxed) >> 56) == 0 {
                b.vals[lane].store(k, Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::tok;

    unsafe fn make_tok(s: &[u8]) -> u64 {
        let mut buf = [0u8; 64];
        buf[..s.len()].copy_from_slice(s);
        let t = tok(buf.as_ptr(), s.len());
        assert!(t >> 56 != 0, "token must have nonzero high byte");
        t
    }

    #[test]
    fn overflow_heavy_tiny_buckets_multithreaded() {
        // Force extreme overflow: 2 buckets = 8 slots, 7 distinct IDs (87.5% load factor),
        // multiple threads hammering insert/link concurrently.
        let t = LinkTable::with_buckets(2);
        let ids: Vec<u64> = unsafe {
            ["u9000", "u8000", "u7000", "u6000", "u5000", "u4000", "u3000"]
                .iter()
                .map(|s| make_tok(s.as_bytes()))
                .collect()
        };
        let mut pairs = Vec::new();
        for w in ids.windows(2) {
            pairs.push((w[0], w[1]));
            pairs.push((w[1], w[0]));
            pairs.push((w[0], w[0])); // self-link
        }
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| unsafe { t.link_all(&pairs) });
            }
        });
        unsafe {
            t.compress_and_resolve(0, 4);
            t.compress_and_resolve(4, 8);
            for &id in &ids {
                assert_eq!(t.person_hashed(id, mix(id)), ids[6]);
            }
            let absent = make_tok(b"u9999");
            assert_eq!(t.person_hashed(absent, mix(absent)), absent);
        }
    }

    #[test]
    fn chain_400_and_hub_5k_and_fallback() {
        let mut ids = Vec::new();
        unsafe {
            for i in 0..400 {
                let s = format!("u_{i:04}");
                ids.push(make_tok(s.as_bytes()));
            }
            // Add >9B FALLBACK ID that sorts lexicographically smallest ('a...' < 'u...')
            let fb = make_tok(b"a_very_long_fallback_id_exceeding_9_bytes");
            let mut pairs = Vec::new();
            for i in 0..399 {
                pairs.push((ids[i], ids[i + 1]));
            }
            pairs.push((ids[200], fb));
            // Add a 5k-ID hub
            let hub_min = make_tok(b"h_00000");
            let mut hub_ids = vec![hub_min];
            for i in 1..5000 {
                let s = format!("h_{i:05}");
                let x = make_tok(s.as_bytes());
                hub_ids.push(x);
                pairs.push((x, hub_min));
            }
            let table = LinkTable::new(pairs.len());
            std::thread::scope(|sc| {
                for chunk in pairs.chunks(1024) {
                    sc.spawn(|| table.link_all(chunk));
                }
            });
            let total_slots = table.slots();
            let tref = &table;
            std::thread::scope(|sc| {
                for from in (0..total_slots).step_by(16384) {
                    let to = (from + 16384).min(total_slots);
                    sc.spawn(move || tref.compress_and_resolve(from, to));
                }
            });
            for &id in &ids {
                assert_eq!(table.person_hashed(id, mix(id)), fb);
            }
            assert_eq!(table.person_hashed(fb, mix(fb)), fb);
            for &id in &hub_ids {
                assert_eq!(table.person_hashed(id, mix(id)), hub_min);
            }
            let solo = make_tok(b"solo_id");
            assert_eq!(table.person_hashed(solo, mix(solo)), solo);
        }
    }
}
