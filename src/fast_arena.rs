//! Lock-free 2 MiB hugepage slab pool, 4 KiB linked-chunk partitioned arena,
//! non-temporal AVX2 streaming scatter, and zero-allocation `BoundarySlab`.

#![allow(dead_code)]

use crate::fast_sess::BoundaryEv;
use crate::sys;
use std::marker::PhantomData;
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

extern "C" {
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

/// 2 MiB super-slab size matching x86-64 hugepages.
pub const SLAB_BYTES: usize = 2 << 20;
/// 4 KiB linked chunk size carved from a 2 MiB slab.
pub const CHUNK_BYTES: usize = 4096;
/// Number of 4 KiB chunks per 2 MiB slab.
pub const CHUNKS_PER_SLAB: usize = SLAB_BYTES / CHUNK_BYTES;
/// 64-byte cache-line header at the front of each `LinkedChunk<T>`.
pub const CHUNK_HEADER_BYTES: usize = 64;
/// Cache-line-aligned payload capacity inside each 4 KiB `LinkedChunk<T>`.
pub const CHUNK_DATA_BYTES: usize = CHUNK_BYTES - CHUNK_HEADER_BYTES;

const DEFAULT_RESERVE_BYTES: usize = 2 << 30;
const PTR_TAG_MASK: usize = CHUNK_BYTES - 1;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct CacheLine([u8; 64]);

/// Contiguous virtual address reservation that hands out 2 MiB slabs lock-freely
/// and lazily commits each slab on first claim via `MADV_POPULATE_WRITE`.
pub struct SlabPool {
    base: *mut u8,
    cap_bytes: usize,
    next_offset: AtomicUsize,
}

unsafe impl Send for SlabPool {}
unsafe impl Sync for SlabPool {}

impl SlabPool {
    /// Reserves `reserve_bytes` (rounded up to `SLAB_BYTES`) using `sys::map_zeroed_huge`
    /// with zero upfront physical page population.
    pub fn new(reserve_bytes: usize) -> Self {
        let cap_bytes = (reserve_bytes.max(SLAB_BYTES) + SLAB_BYTES - 1) & !(SLAB_BYTES - 1);
        let base = sys::map_zeroed_huge(cap_bytes);
        debug_assert!((base as usize) & PTR_TAG_MASK == 0);
        Self {
            base,
            cap_bytes,
            next_offset: AtomicUsize::new(0),
        }
    }

    /// Reserves the default 2 GiB virtual arena.
    pub fn with_default_reserve() -> Self {
        Self::new(DEFAULT_RESERVE_BYTES)
    }

    /// Claims one 2 MiB slab and populates its physical pages on demand.
    #[cold]
    #[inline(never)]
    pub fn alloc_slab(&self) -> *mut u8 {
        let off = self.next_offset.fetch_add(SLAB_BYTES, Ordering::Relaxed);
        assert!(off + SLAB_BYTES <= self.cap_bytes, "SlabPool capacity exhausted");
        let ptr = unsafe { self.base.add(off) };
        sys::populate(ptr, SLAB_BYTES);
        ptr
    }

    #[inline(always)]
    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    #[inline(always)]
    pub fn used_bytes(&self) -> usize {
        self.next_offset.load(Ordering::Relaxed).min(self.cap_bytes)
    }

    #[inline(always)]
    pub fn slab_count(&self) -> usize {
        self.used_bytes() / SLAB_BYTES
    }
}

impl Drop for SlabPool {
    fn drop(&mut self) {
        if !self.base.is_null() && self.cap_bytes > 0 {
            unsafe {
                munmap(self.base as *mut c_void, self.cap_bytes);
            }
            self.base = std::ptr::null_mut();
        }
    }
}

const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

const fn lcm(a: usize, b: usize) -> usize {
    (a / gcd(a, b)) * b
}

/// 4 KiB linked chunk whose payload starts on a 64-byte cache-line boundary,
/// enabling aligned `_mm256_stream_si256` stores without header/data false sharing.
///
/// For 16-byte records (`FastEvRec`, `ClickSlot`), `CAP = 252` (`63 * 4` records = `4,032` bytes),
/// making `size_of::<LinkedChunk<T>>() == 4,096` bytes with zero wasted cache lines.
#[repr(C, align(64))]
pub struct LinkedChunk<T> {
    pub next: *mut LinkedChunk<T>,
    pub count: u32,
    pub _pad: u32,
    _header_pad: [u64; 6],
    payload: [CacheLine; CHUNK_DATA_BYTES / 64],
    _marker: PhantomData<T>,
}

impl<T: Copy> LinkedChunk<T> {
    /// Number of 64-byte cache lines per staging flush unit.
    pub const FLUSH_LINES: usize = lcm(64, std::mem::size_of::<T>()) / 64;
    /// Records per staging flush unit.
    pub const FLUSH_UNIT: usize = (Self::FLUSH_LINES * 64) / std::mem::size_of::<T>();
    /// Maximum records stored in one 4 KiB chunk (always a multiple of `FLUSH_UNIT`).
    pub const CAP: usize =
        (CHUNK_DATA_BYTES / (Self::FLUSH_LINES * 64)) * Self::FLUSH_UNIT;

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        let n = (self.count as usize).min(Self::CAP);
        unsafe { std::slice::from_raw_parts(self.data_ptr(), n) }
    }

    #[inline(always)]
    pub unsafe fn data_ptr(&self) -> *const T {
        self.payload.as_ptr() as *const T
    }

    #[inline(always)]
    pub unsafe fn data_mut_ptr(&mut self) -> *mut T {
        self.payload.as_mut_ptr() as *mut T
    }
}

/// Lock-free partitioned arena of 4 KiB `LinkedChunk<T>` lists backed by `SlabPool`.
pub struct LinkedChunkArena<T> {
    pool: SlabPool,
    heads: Vec<AtomicPtr<LinkedChunk<T>>>,
    total_chunks: AtomicUsize,
}

unsafe impl<T: Send> Send for LinkedChunkArena<T> {}
unsafe impl<T: Sync> Sync for LinkedChunkArena<T> {}

impl<T: Copy> LinkedChunkArena<T> {
    /// Creates a partitioned arena with `num_parts` atomic list heads and the default 2 GiB reserve.
    pub fn new(num_parts: usize) -> Self {
        Self::with_reserve(num_parts, DEFAULT_RESERVE_BYTES)
    }

    /// Creates a partitioned arena with `num_parts` atomic list heads and `reserve_bytes` virtual reserve.
    pub fn with_reserve(num_parts: usize, reserve_bytes: usize) -> Self {
        assert!(
            std::mem::size_of::<T>() > 0
                && std::mem::size_of::<T>() <= CHUNK_DATA_BYTES
                && std::mem::align_of::<T>() <= 64
                && LinkedChunk::<T>::CAP > 0
                && LinkedChunk::<T>::CAP <= PTR_TAG_MASK,
            "unsupported record layout for LinkedChunk<T>"
        );
        let mut heads = Vec::with_capacity(num_parts);
        for _ in 0..num_parts {
            heads.push(AtomicPtr::new(std::ptr::null_mut()));
        }
        Self {
            pool: SlabPool::new(reserve_bytes),
            heads,
            total_chunks: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    pub fn num_parts(&self) -> usize {
        self.heads.len()
    }

    #[inline(always)]
    pub fn pool(&self) -> &SlabPool {
        &self.pool
    }

    #[inline(always)]
    pub fn total_chunks(&self) -> usize {
        self.total_chunks.load(Ordering::Relaxed)
    }

    /// Prepends a completed chunk to partition `part`'s lock-free singly-linked list.
    #[inline(always)]
    pub fn push_head(&self, part: usize, chunk: *mut LinkedChunk<T>) {
        debug_assert!(part < self.heads.len());
        let head = unsafe { self.heads.get_unchecked(part) };
        let mut cur = head.load(Ordering::Relaxed);
        loop {
            unsafe { (*chunk).next = cur };
            match head.compare_exchange_weak(
                cur,
                chunk,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Returns the atomic list head for partition `part` with acquire ordering.
    #[inline(always)]
    pub fn partition_head(&self, part: usize) -> *mut LinkedChunk<T> {
        self.heads[part].load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn is_partition_empty(&self, part: usize) -> bool {
        self.heads[part].load(Ordering::Acquire).is_null()
    }

    /// Iterates all chunks in partition `part`, prefetching `chunk.next` into L1.
    #[inline(always)]
    pub fn chunks(&self, part: usize) -> ChunkIter<'_, T> {
        let cur = self.heads[part].load(Ordering::Acquire);
        ChunkIter {
            cur,
            _marker: PhantomData,
        }
    }

    /// Visits every contiguous record slice in partition `part`.
    #[inline]
    pub fn for_each_chunk_slice(&self, part: usize, mut f: impl FnMut(&[T])) {
        for chunk in self.chunks(part) {
            f(chunk.as_slice());
        }
    }
}

/// Iterator over a partition's linked chunks with `next`-pointer L1 prefetching.
pub struct ChunkIter<'a, T> {
    cur: *mut LinkedChunk<T>,
    _marker: PhantomData<&'a LinkedChunk<T>>,
}

impl<'a, T: Copy> Iterator for ChunkIter<'a, T> {
    type Item = &'a LinkedChunk<T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.cur;
        if ptr.is_null() {
            return None;
        }
        let chunk = unsafe { &*ptr };
        let next = chunk.next;
        if !next.is_null() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    next as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        self.cur = next;
        Some(chunk)
    }
}

#[inline(always)]
unsafe fn stream_lines_avx2(dst: *mut u8, src: *const u8, lines: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use std::arch::x86_64::*;
        let mut d = dst as *mut __m256i;
        let mut s = src as *const __m256i;
        for _ in 0..lines {
            let v0 = _mm256_load_si256(s);
            let v1 = _mm256_load_si256(s.add(1));
            _mm256_stream_si256(d, v0);
            _mm256_stream_si256(d.add(1), v1);
            d = d.add(2);
            s = s.add(2);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        std::ptr::copy_nonoverlapping(src, dst, lines * 64);
    }
}

#[inline(always)]
fn sfence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::sync::atomic::fence(Ordering::Release);
    }
}

/// Per-thread non-temporal streaming scatter into a `LinkedChunkArena<T>`.
///
/// Stores one packed `(chunk_ptr | count)` word and one 64-byte cache-line staging
/// buffer per partition (for 16-byte records). Full cache lines flush directly to DRAM
/// via `_mm256_stream_si256`, keeping scattered writes out of L1/L2/L3 cache.
pub struct LinkedScatter<T: Copy> {
    /// Packed `chunk_ptr | count`, where `chunk_ptr` is 4 KiB-aligned and `count <= CAP`.
    /// Initialized to `CAP` (`chunk_ptr == 0`) so the first push triggers lazy allocation.
    heads: Vec<usize>,
    staging: Vec<CacheLine>,
    slab_ptr: *mut u8,
    slab_off: usize,
    chunks_used: usize,
    slabs_used: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Send> Send for LinkedScatter<T> {}

impl<T: Copy> LinkedScatter<T> {
    pub const LINES: usize = LinkedChunk::<T>::FLUSH_LINES;
    pub const UNIT: usize = LinkedChunk::<T>::FLUSH_UNIT;
    pub const CAP: usize = LinkedChunk::<T>::CAP;

    pub fn new() -> Self {
        Self {
            heads: Vec::new(),
            staging: Vec::new(),
            slab_ptr: std::ptr::null_mut(),
            slab_off: SLAB_BYTES,
            chunks_used: 0,
            slabs_used: 0,
            _marker: PhantomData,
        }
    }

    /// Resets thread-local staging and partition cursors for `parts` partitions.
    pub fn reset(&mut self, parts: usize) {
        self.heads.clear();
        self.heads.resize(parts, Self::CAP);
        self.staging.clear();
        self.staging.resize(parts * Self::LINES, CacheLine([0; 64]));
        self.slab_ptr = std::ptr::null_mut();
        self.slab_off = SLAB_BYTES;
        self.chunks_used = 0;
        self.slabs_used = 0;
    }

    #[inline(always)]
    pub fn chunks_used(&self) -> usize {
        self.chunks_used
    }

    #[inline(always)]
    pub fn slabs_used(&self) -> usize {
        self.slabs_used
    }

    /// Appends one record to partition `part`, streaming full 64-byte lines via `_mm256_stream_si256`.
    #[inline(always)]
    pub unsafe fn push(&mut self, arena: &LinkedChunkArena<T>, part: usize, rec: T) {
        debug_assert!(part < self.heads.len());
        let mut p = *self.heads.get_unchecked(part);
        let mut idx = p & PTR_TAG_MASK;
        if idx == Self::CAP {
            p = self.rollover_chunk(arena, part, p);
            idx = 0;
        }
        let staged = idx & (Self::UNIT - 1);
        let line_ptr = self.staging.as_mut_ptr().add(part * Self::LINES) as *mut T;
        line_ptr.add(staged).write(rec);

        let next_idx = idx + 1;
        *self.heads.get_unchecked_mut(part) = p + 1;

        if (next_idx & (Self::UNIT - 1)) == 0 {
            let chunk = (p & !PTR_TAG_MASK) as *mut LinkedChunk<T>;
            let dst = (*chunk).data_mut_ptr().add(next_idx - Self::UNIT) as *mut u8;
            stream_lines_avx2(dst, line_ptr as *const u8, Self::LINES);
        }
    }

    /// Pushes up to 8 records selected by `mask`, prefetching target staging lines
    /// and partition heads up front to overlap L2/L3 misses.
    #[inline(always)]
    pub unsafe fn push_batch_8(
        &mut self,
        arena: &LinkedChunkArena<T>,
        parts: [usize; 8],
        recs: [T; 8],
        mask: u8,
    ) {
        if mask == 0 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            if mask == 0xFF {
                for k in 0..8 {
                    let part = *parts.get_unchecked(k);
                    debug_assert!(part < self.heads.len());
                    _mm_prefetch(
                        self.staging.as_ptr().add(part * Self::LINES) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                for k in 0..8 {
                    self.push(arena, *parts.get_unchecked(k), *recs.get_unchecked(k));
                }
                return;
            }
            let mut pm = mask;
            while pm != 0 {
                let k = pm.trailing_zeros() as usize;
                pm &= pm - 1;
                let part = *parts.get_unchecked(k);
                debug_assert!(part < self.heads.len());
                _mm_prefetch(
                    self.staging.as_ptr().add(part * Self::LINES) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }
        let mut m = mask;
        while m != 0 {
            let k = m.trailing_zeros() as usize;
            m &= m - 1;
            self.push(arena, *parts.get_unchecked(k), *recs.get_unchecked(k));
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn rollover_chunk(
        &mut self,
        arena: &LinkedChunkArena<T>,
        part: usize,
        prev_packed: usize,
    ) -> usize {
        let prev_chunk = (prev_packed & !PTR_TAG_MASK) as *mut LinkedChunk<T>;
        if !prev_chunk.is_null() {
            (*prev_chunk).count = Self::CAP as u32;
            sfence();
            arena.push_head(part, prev_chunk);
        }
        if self.slab_off + CHUNK_BYTES > SLAB_BYTES {
            self.slab_ptr = arena.pool.alloc_slab();
            self.slab_off = 0;
            self.slabs_used += 1;
        }
        let new_chunk = self.slab_ptr.add(self.slab_off) as *mut LinkedChunk<T>;
        self.slab_off += CHUNK_BYTES;
        self.chunks_used += 1;
        (*new_chunk).next = std::ptr::null_mut();
        (*new_chunk).count = 0;
        (*new_chunk)._pad = 0;
        new_chunk as usize
    }

    /// Flushes partial staging lines, links all remaining active chunks into `arena`,
    /// and clears per-partition cursors.
    pub unsafe fn finish(&mut self, arena: &LinkedChunkArena<T>) {
        sfence();
        for part in 0..self.heads.len() {
            let p = *self.heads.get_unchecked(part);
            let chunk = (p & !PTR_TAG_MASK) as *mut LinkedChunk<T>;
            if chunk.is_null() {
                continue;
            }
            let idx = p & PTR_TAG_MASK;
            let staged = idx & (Self::UNIT - 1);
            if staged > 0 {
                let src = self.staging.as_ptr().add(part * Self::LINES) as *const T;
                let dst = (*chunk).data_mut_ptr().add(idx - staged);
                std::ptr::copy_nonoverlapping(src, dst, staged);
            }
            if idx > 0 {
                (*chunk).count = idx as u32;
                arena.push_head(part, chunk);
            }
        }
        arena
            .total_chunks
            .fetch_add(self.chunks_used, Ordering::Relaxed);
        self.heads.clear();
    }
}

impl<T: Copy> Default for LinkedScatter<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Head boundary flag (`r == 0` in a block).
pub const SEG_FLAG_HEAD: u8 = 1 << 0;
/// Tail boundary flag (`r == nrun - 1` in a block).
pub const SEG_FLAG_TAIL: u8 = 1 << 1;
/// Segment requires timestamp sorting in Stage 3B.
pub const SEG_FLAG_UNSORTED: u8 = 1 << 2;

/// Flat descriptor referencing a contiguous slice `[ev_offset .. ev_offset + ev_len]`
/// inside a `BoundarySlab::events` buffer (zero per-segment heap allocations).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct FlatBoundarySeg {
    pub user_idx: u32,
    pub block_idx: u32,
    pub ev_offset: u32,
    pub ev_len: u32,
    pub alias_mask: u16,
    pub flags: u8,
    pub _pad: u8,
}

impl FlatBoundarySeg {
    #[inline(always)]
    pub const fn make_flags(is_head: bool, is_tail: bool, unsorted: bool) -> u8 {
        (if is_head { SEG_FLAG_HEAD } else { 0 })
            | (if is_tail { SEG_FLAG_TAIL } else { 0 })
            | (if unsorted { SEG_FLAG_UNSORTED } else { 0 })
    }

    #[inline(always)]
    pub const fn is_head(&self) -> bool {
        (self.flags & SEG_FLAG_HEAD) != 0
    }

    #[inline(always)]
    pub const fn is_tail(&self) -> bool {
        (self.flags & SEG_FLAG_TAIL) != 0
    }

    #[inline(always)]
    pub const fn is_unsorted(&self) -> bool {
        (self.flags & SEG_FLAG_UNSORTED) != 0
    }
}

/// Zero-copy reference into a slice of per-thread `BoundarySlab`s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlatSegRef {
    pub user_idx: u32,
    pub block_idx: u32,
    pub slab_idx: u16,
    pub seg_idx: u32,
}

/// Per-thread flat boundary-segment slab replacing `Vec<Vec<BoundaryEv>>`.
///
/// Pre-allocates one contiguous `events: Vec<BoundaryEv>` (`131,072` elements = `1.5 MiB`)
/// and one `segs: Vec<FlatBoundarySeg>`, eliminating ~2,500 heap allocations per run.
pub struct BoundarySlab {
    pub events: Vec<BoundaryEv>,
    pub segs: Vec<FlatBoundarySeg>,
}

impl BoundarySlab {
    pub const DEFAULT_EVENT_CAP: usize = 131_072;
    pub const DEFAULT_SEG_CAP: usize = 4_096;

    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_EVENT_CAP, Self::DEFAULT_SEG_CAP)
    }

    pub fn with_capacity(ev_cap: usize, seg_cap: usize) -> Self {
        Self {
            events: Vec::with_capacity(ev_cap),
            segs: Vec::with_capacity(seg_cap),
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.events.clear();
        self.segs.clear();
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.segs.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    #[inline(always)]
    pub fn event_len(&self) -> usize {
        self.events.len()
    }

    /// Appends a segment from an existing `BoundaryEv` slice without allocating.
    #[inline]
    pub fn push_seg(
        &mut self,
        user_idx: u32,
        block_idx: u32,
        alias_mask: u16,
        flags: u8,
        evs: &[BoundaryEv],
    ) {
        let ev_offset = self.events.len() as u32;
        let ev_len = evs.len() as u32;
        self.events.extend_from_slice(evs);
        self.segs.push(FlatBoundarySeg {
            user_idx,
            block_idx,
            ev_offset,
            ev_len,
            alias_mask,
            flags,
            _pad: 0,
        });
    }

    /// Converts packed `w0` event words directly into `self.events` and appends a segment descriptor.
    #[inline]
    pub fn push_seg_from_w0(
        &mut self,
        user_idx: u32,
        block_idx: u32,
        alias_mask: u16,
        flags: u8,
        w0_slice: &[u64],
    ) {
        let ev_offset = self.events.len() as u32;
        let ev_len = w0_slice.len() as u32;
        self.events.reserve(w0_slice.len());
        for &w0 in w0_slice {
            self.events.push(BoundaryEv::from_w0(w0));
        }
        self.segs.push(FlatBoundarySeg {
            user_idx,
            block_idx,
            ev_offset,
            ev_len,
            alias_mask,
            flags,
            _pad: 0,
        });
    }

    /// Returns the `BoundaryEv` slice referenced by `seg`.
    #[inline(always)]
    pub fn seg_events(&self, seg: &FlatBoundarySeg) -> &[BoundaryEv] {
        let s = seg.ev_offset as usize;
        let e = s + seg.ev_len as usize;
        &self.events[s..e]
    }

    /// Sorts segment descriptors by `(user_idx, block_idx)` without moving `events`.
    #[inline]
    pub fn sort_by_user_block(&mut self) {
        self.segs
            .sort_unstable_by_key(|s| (s.user_idx, s.block_idx));
    }

    /// Appends another slab's events and rebased segment descriptors in one contiguous copy.
    pub fn append_from(&mut self, other: &BoundarySlab) {
        if other.segs.is_empty() {
            return;
        }
        let base_off = self.events.len() as u32;
        self.events.extend_from_slice(&other.events);
        self.segs.reserve(other.segs.len());
        for &seg in &other.segs {
            self.segs.push(FlatBoundarySeg {
                ev_offset: seg.ev_offset + base_off,
                ..seg
            });
        }
    }

    /// Collects and sorts zero-copy segment references across all per-thread `slabs`
    /// by `(user_idx, block_idx)` without copying `BoundaryEv` payloads.
    pub fn collect_sorted_refs(slabs: &[BoundarySlab]) -> Vec<FlatSegRef> {
        let total: usize = slabs.iter().map(|s| s.segs.len()).sum();
        let mut refs = Vec::with_capacity(total);
        for (slab_idx, slab) in slabs.iter().enumerate() {
            for (seg_idx, seg) in slab.segs.iter().enumerate() {
                refs.push(FlatSegRef {
                    user_idx: seg.user_idx,
                    block_idx: seg.block_idx,
                    slab_idx: slab_idx as u16,
                    seg_idx: seg_idx as u32,
                });
            }
        }
        refs.sort_unstable_by_key(|r| (r.user_idx, r.block_idx));
        refs
    }
}

impl Default for BoundarySlab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast_links::ClickSlot;
    use crate::fast_sess::{BoundarySeg, FastEvRec};
    use crate::scatter::{by_partition, Arena, Scatter};
    use crate::sys::{self, par, PerThread};
    use std::mem::{align_of, offset_of, size_of};
    use std::time::Instant;

    #[inline(always)]
    fn xorshift64(s: &mut u64) -> u64 {
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *s = x;
        x
    }

    #[test]
    fn test_layout_and_slab_pool() {
        assert_eq!(size_of::<LinkedChunk<FastEvRec>>(), 4096);
        assert_eq!(align_of::<LinkedChunk<FastEvRec>>(), 64);
        assert_eq!(offset_of!(LinkedChunk<FastEvRec>, payload), 64);
        assert_eq!(LinkedChunk::<FastEvRec>::CAP, 252);
        assert_eq!(LinkedChunk::<FastEvRec>::FLUSH_UNIT, 4);
        assert_eq!(LinkedChunk::<FastEvRec>::FLUSH_LINES, 1);

        assert_eq!(size_of::<LinkedChunk<ClickSlot>>(), 4096);
        assert_eq!(LinkedChunk::<ClickSlot>::CAP, 252);

        let pool = SlabPool::new(16 << 20);
        assert_eq!(pool.cap_bytes(), 16 << 20);
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.slab_count(), 0);

        let s0 = pool.alloc_slab();
        let s1 = pool.alloc_slab();
        assert_eq!((s0 as usize) & PTR_TAG_MASK, 0);
        assert_eq!(s1 as usize - s0 as usize, SLAB_BYTES);
        assert_eq!(pool.slab_count(), 2);
        assert_eq!(pool.used_bytes(), 2 * SLAB_BYTES);
    }

    #[test]
    fn test_chunk_boundaries_and_empty_partitions() {
        let counts = [0usize, 1, 3, 4, 5, 251, 252, 253, 504, 505];
        let arena = LinkedChunkArena::<FastEvRec>::with_reserve(counts.len(), 16 << 20);
        let mut sc = LinkedScatter::<FastEvRec>::new();
        sc.reset(counts.len());

        let mut expected_sum = vec![0u128; counts.len()];
        for (p, &n) in counts.iter().enumerate() {
            for i in 0..n {
                let rec = FastEvRec {
                    w0: ((p as u64) << 32) | (i as u64 + 1),
                    w1: (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                };
                expected_sum[p] = expected_sum[p].wrapping_add((rec.w0 as u128) ^ ((rec.w1 as u128) << 64));
                unsafe { sc.push(&arena, p, rec) };
            }
        }
        unsafe { sc.finish(&arena) };

        assert!(arena.is_partition_empty(0));
        assert_eq!(arena.chunks(0).count(), 0);

        for (p, &n) in counts.iter().enumerate() {
            let mut got_n = 0usize;
            let mut got_sum = 0u128;
            for chunk in arena.chunks(p) {
                assert!(chunk.len() > 0 && chunk.len() <= LinkedChunk::<FastEvRec>::CAP);
                for &rec in chunk.as_slice() {
                    got_n += 1;
                    got_sum = got_sum.wrapping_add((rec.w0 as u128) ^ ((rec.w1 as u128) << 64));
                }
            }
            assert_eq!(got_n, n, "partition {p} count mismatch");
            assert_eq!(got_sum, expected_sum[p], "partition {p} checksum mismatch");
        }
    }

    #[test]
    fn test_concurrent_scatter_2m_672_parts() {
        const THREADS: usize = 8;
        const PARTS: usize = 672;
        const PER_THREAD: usize = 250_000;
        const TOTAL: usize = THREADS * PER_THREAD;

        let arena = LinkedChunkArena::<FastEvRec>::with_reserve(PARTS, 128 << 20);
        let per_thread_expected = PerThread::new(THREADS, |_| vec![(0usize, 0u64, 0u64); PARTS]);

        par(THREADS, |tid| unsafe {
            let mut sc = LinkedScatter::<FastEvRec>::new();
            sc.reset(PARTS);
            let exp = per_thread_expected.get(tid);
            let mut rng = 0x1234_5678_9ABC_DEF0u64 ^ ((tid as u64 + 1) * 0x9E37_79B9_7F4A_7C15);

            // First half via scalar push, second half via push_batch_8 (mixed full and partial masks).
            let half = PER_THREAD / 2;
            for i in 0..half {
                let part = (xorshift64(&mut rng) as usize) % PARTS;
                let rec = FastEvRec {
                    w0: ((tid as u64) << 48) | (i as u64),
                    w1: xorshift64(&mut rng),
                };
                exp[part].0 += 1;
                exp[part].1 ^= rec.w0;
                exp[part].2 = exp[part].2.wrapping_add(rec.w1);
                sc.push(&arena, part, rec);
            }

            let mut i = half;
            while i < PER_THREAD {
                let remain = (PER_THREAD - i).min(8);
                let mut parts = [0usize; 8];
                let mut recs = [FastEvRec::default(); 8];
                let mut mask = 0u8;
                for k in 0..remain {
                    let part = (xorshift64(&mut rng) as usize) % PARTS;
                    let rec = FastEvRec {
                        w0: ((tid as u64) << 48) | ((i + k) as u64),
                        w1: xorshift64(&mut rng),
                    };
                    parts[k] = part;
                    recs[k] = rec;
                    mask |= 1u8 << k;
                    exp[part].0 += 1;
                    exp[part].1 ^= rec.w0;
                    exp[part].2 = exp[part].2.wrapping_add(rec.w1);
                }
                sc.push_batch_8(&arena, parts, recs, mask);
                i += remain;
            }
            sc.finish(&arena);
        });

        let mut total_got = 0usize;
        for p in 0..PARTS {
            let mut exp_cnt = 0usize;
            let mut exp_xor = 0u64;
            let mut exp_sum = 0u64;
            for t in 0..THREADS {
                let e = unsafe { per_thread_expected.get(t)[p] };
                exp_cnt += e.0;
                exp_xor ^= e.1;
                exp_sum = exp_sum.wrapping_add(e.2);
            }

            let mut got_cnt = 0usize;
            let mut got_xor = 0u64;
            let mut got_sum = 0u64;
            for chunk in arena.chunks(p) {
                for &rec in chunk.as_slice() {
                    got_cnt += 1;
                    got_xor ^= rec.w0;
                    got_sum = got_sum.wrapping_add(rec.w1);
                }
            }
            assert_eq!(got_cnt, exp_cnt, "partition {p} count mismatch");
            assert_eq!(got_xor, exp_xor, "partition {p} w0 xor mismatch");
            assert_eq!(got_sum, exp_sum, "partition {p} w1 sum mismatch");
            total_got += got_cnt;
        }
        assert_eq!(total_got, TOTAL);
    }

    #[test]
    fn test_boundary_slab_zero_alloc() {
        const THREADS: usize = 8;
        const SEGS_PER_THREAD: usize = 320;
        let slabs = PerThread::new(THREADS, |_| BoundarySlab::new());

        par(THREADS, |tid| unsafe {
            let slab = slabs.get(tid);
            for s in 0..SEGS_PER_THREAD {
                let user_idx = ((s * 17 + tid * 7) % 1024) as u32;
                let block_idx = (tid * 100 + s) as u32;
                let n = (s % 19) + 1;
                let w0s: Vec<u64> = (0..n)
                    .map(|k| {
                        BoundaryEv {
                            rel_ts_ms: (block_idx * 1000 + k as u32) & 0x1FFF_FFFF,
                            price_cents: (k as u32 + 1) * 100,
                            cat: (k % 32) as u8,
                            alias: (k % 14) as u8,
                        }
                        .to_w0()
                    })
                    .collect();
                let flags = FlatBoundarySeg::make_flags(s % 2 == 0, s % 2 == 1, s % 5 == 0);
                slab.push_seg_from_w0(user_idx, block_idx, 1u16 << (tid % 14), flags, &w0s);
            }
        });

        let owned_slabs: Vec<BoundarySlab> = (0..THREADS)
            .map(|t| std::mem::take(unsafe { slabs.get(t) }))
            .collect();

        let refs = BoundarySlab::collect_sorted_refs(&owned_slabs);
        assert_eq!(refs.len(), THREADS * SEGS_PER_THREAD);

        let mut merged = BoundarySlab::with_capacity(64_000, refs.len());
        for s in &owned_slabs {
            merged.append_from(s);
        }
        merged.sort_by_user_block();
        assert_eq!(merged.len(), refs.len());

        for (r, seg) in refs.iter().zip(merged.segs.iter()) {
            assert_eq!(r.user_idx, seg.user_idx);
            assert_eq!(r.block_idx, seg.block_idx);
            let ref_seg = &owned_slabs[r.slab_idx as usize].segs[r.seg_idx as usize];
            assert_eq!(
                owned_slabs[r.slab_idx as usize].seg_events(ref_seg),
                merged.seg_events(seg)
            );
        }
    }

    #[test]
    fn bench_linked_scatter_vs_arena_4m() {
        const THREADS: usize = 8;
        const PARTS: usize = 1_343;
        const TOTAL: usize = 4_000_000;
        const PER_THREAD: usize = TOTAL / THREADS;
        const OLD_CHUNK: usize = 512;

        // Pre-generate deterministic (part, FastEvRec) inputs per thread so the benchmark
        // measures scatter + allocation + iteration rather than PRNG time.
        let thread_inputs: Vec<Vec<(u16, FastEvRec)>> = (0..THREADS)
            .map(|tid| {
                let mut rng = 0xCAFE_BABE_1234_5678u64 ^ ((tid as u64 + 1) * 0x9E37_79B9_7F4A_7C15);
                let mut v = Vec::with_capacity(PER_THREAD);
                for i in 0..PER_THREAD {
                    let p = ((xorshift64(&mut rng) >> 32) as usize % PARTS) as u16;
                    let rec = FastEvRec {
                        w0: ((tid as u64) << 40) | (i as u64),
                        w1: xorshift64(&mut rng),
                    };
                    v.push((p, rec));
                }
                v
            })
            .collect();

        // 1. Baseline: Arena<FastEvRec> + upfront populate + Scatter<FastEvRec, 4> + by_partition
        let t_old_total = Instant::now();
        let old_cap = TOTAL + (THREADS * PARTS + 2) * OLD_CHUNK;
        let old_arena = Arena::<FastEvRec>::new(old_cap, OLD_CHUNK);
        let pop_bytes = ((TOTAL + THREADS * PARTS * OLD_CHUNK / 2) * 16).min(old_arena.cap_bytes());
        let t_old_pop = Instant::now();
        {
            const PIECE: usize = 4 << 20;
            let pieces: Vec<(usize, usize)> = (0..pop_bytes)
                .step_by(PIECE)
                .map(|at| ((old_arena.base as usize) + at, PIECE.min(pop_bytes - at)))
                .collect();
            let next = AtomicUsize::new(0);
            par(THREADS, |_| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(at, n)) = pieces.get(i) else { break };
                sys::populate(at as *mut u8, n);
            });
        }
        let old_pop_ms = t_old_pop.elapsed().as_secs_f64() * 1e3;

        let old_scatters = PerThread::new(THREADS, |_| Scatter::<FastEvRec, 4>::new());
        let t_old_scat = Instant::now();
        par(THREADS, |tid| unsafe {
            let sc = old_scatters.get(tid);
            sc.reset(PARTS);
            for &(p, rec) in &thread_inputs[tid] {
                sc.push(&old_arena, p as usize, rec);
            }
            sc.finish(&old_arena);
        });
        let old_scat_ms = t_old_scat.elapsed().as_secs_f64() * 1e3;

        let t_old_iter = Instant::now();
        let old_logs: Vec<&Vec<(u32, u32, u32)>> = (0..THREADS)
            .map(|t| unsafe { &old_scatters.get(t).log })
            .collect();
        let old_by_part = by_partition(&old_logs, PARTS);
        let mut old_cnt = 0usize;
        let mut old_xor = 0u64;
        for chunks in &old_by_part {
            for &(first, count) in chunks {
                let slice = unsafe {
                    std::slice::from_raw_parts(old_arena.base.add(first as usize), count as usize)
                };
                for rec in slice {
                    old_cnt += 1;
                    old_xor ^= rec.w0 ^ rec.w1;
                }
            }
        }
        let old_iter_ms = t_old_iter.elapsed().as_secs_f64() * 1e3;
        let old_total_ms = t_old_total.elapsed().as_secs_f64() * 1e3;
        sys::release(old_arena.base as *mut u8, old_arena.cap_bytes());

        // 2. New: LinkedChunkArena<FastEvRec> + LinkedScatter<FastEvRec> (scalar push)
        let t_new_total = Instant::now();
        let new_arena = LinkedChunkArena::<FastEvRec>::with_reserve(PARTS, 256 << 20);
        let new_scatters = PerThread::new(THREADS, |_| LinkedScatter::<FastEvRec>::new());
        let t_new_scat = Instant::now();
        par(THREADS, |tid| unsafe {
            let sc = new_scatters.get(tid);
            sc.reset(PARTS);
            for &(p, rec) in &thread_inputs[tid] {
                sc.push(&new_arena, p as usize, rec);
            }
            sc.finish(&new_arena);
        });
        let new_scat_ms = t_new_scat.elapsed().as_secs_f64() * 1e3;

        let t_new_iter = Instant::now();
        let mut new_cnt = 0usize;
        let mut new_xor = 0u64;
        for p in 0..PARTS {
            for chunk in new_arena.chunks(p) {
                for rec in chunk.as_slice() {
                    new_cnt += 1;
                    new_xor ^= rec.w0 ^ rec.w1;
                }
            }
        }
        let new_iter_ms = t_new_iter.elapsed().as_secs_f64() * 1e3;
        let new_total_ms = t_new_total.elapsed().as_secs_f64() * 1e3;
        let new_slabs = new_arena.pool().slab_count();
        let new_used_mib = new_arena.pool().used_bytes() as f64 / (1024.0 * 1024.0);
        let new_chunks = new_arena.total_chunks();
        drop(new_scatters);
        drop(new_arena);

        // 3. New: LinkedChunkArena<FastEvRec> + LinkedScatter<FastEvRec>::push_batch_8
        let t_batch_total = Instant::now();
        let batch_arena = LinkedChunkArena::<FastEvRec>::with_reserve(PARTS, 256 << 20);
        let batch_scatters = PerThread::new(THREADS, |_| LinkedScatter::<FastEvRec>::new());
        let t_batch_scat = Instant::now();
        par(THREADS, |tid| unsafe {
            let sc = batch_scatters.get(tid);
            sc.reset(PARTS);
            let inp = &thread_inputs[tid];
            let mut i = 0;
            while i + 8 <= inp.len() {
                let parts = [
                    inp[i].0 as usize,
                    inp[i + 1].0 as usize,
                    inp[i + 2].0 as usize,
                    inp[i + 3].0 as usize,
                    inp[i + 4].0 as usize,
                    inp[i + 5].0 as usize,
                    inp[i + 6].0 as usize,
                    inp[i + 7].0 as usize,
                ];
                let recs = [
                    inp[i].1,
                    inp[i + 1].1,
                    inp[i + 2].1,
                    inp[i + 3].1,
                    inp[i + 4].1,
                    inp[i + 5].1,
                    inp[i + 6].1,
                    inp[i + 7].1,
                ];
                sc.push_batch_8(&batch_arena, parts, recs, 0xFF);
                i += 8;
            }
            while i < inp.len() {
                sc.push(&batch_arena, inp[i].0 as usize, inp[i].1);
                i += 1;
            }
            sc.finish(&batch_arena);
        });
        let batch_scat_ms = t_batch_scat.elapsed().as_secs_f64() * 1e3;
        let mut batch_cnt = 0usize;
        let mut batch_xor = 0u64;
        for p in 0..PARTS {
            for chunk in batch_arena.chunks(p) {
                for rec in chunk.as_slice() {
                    batch_cnt += 1;
                    batch_xor ^= rec.w0 ^ rec.w1;
                }
            }
        }
        let batch_total_ms = t_batch_total.elapsed().as_secs_f64() * 1e3;

        assert_eq!(old_cnt, TOTAL);
        assert_eq!(new_cnt, TOTAL);
        assert_eq!(batch_cnt, TOTAL);
        assert_eq!(new_xor, old_xor);
        assert_eq!(batch_xor, old_xor);

        // 4. BoundarySlab vs Vec<BoundarySeg> microbenchmark (2,500 segments x 12 events)
        const SEGS: usize = 2_500;
        const EVS_PER_SEG: usize = 12;
        let sample_evs: Vec<BoundaryEv> = (0..EVS_PER_SEG)
            .map(|k| BoundaryEv {
                rel_ts_ms: k as u32 * 1000,
                price_cents: 1999,
                cat: 3,
                alias: 1,
            })
            .collect();

        let t_vec_seg = Instant::now();
        let mut vec_segs: Vec<BoundarySeg> = Vec::new();
        for i in 0..SEGS {
            vec_segs.push(BoundarySeg {
                block_idx: i as u32,
                is_head: true,
                is_tail: false,
                user_idx: (i * 13) as u32,
                alias_mask: 1,
                unsorted: false,
                events: sample_evs.clone(),
            });
        }
        vec_segs.sort_unstable_by_key(|s| (s.user_idx, s.block_idx));
        let vec_seg_ms = t_vec_seg.elapsed().as_secs_f64() * 1e3;
        drop(vec_segs);

        let mut flat_slab = BoundarySlab::new();
        // Warm once (as FastSessWorker::new does at worker init before Stage 3A), then measure.
        flat_slab.push_seg(0, 0, 1, SEG_FLAG_HEAD, &sample_evs);
        flat_slab.clear();
        let t_flat_seg = Instant::now();
        for i in 0..SEGS {
            flat_slab.push_seg(
                (i * 13) as u32,
                i as u32,
                1,
                SEG_FLAG_HEAD,
                &sample_evs,
            );
        }
        flat_slab.sort_by_user_block();
        let flat_seg_ms = t_flat_seg.elapsed().as_secs_f64() * 1e3;

        println!(
            "\n=== fast_arena 4M-record (16B) x 8-thread x {PARTS}-partition benchmark ==="
        );
        println!(
            "Old Arena+Scatter:      total={old_total_ms:.2} ms (populate={old_pop_ms:.2} ms, scatter={old_scat_ms:.2} ms, regroup+iter={old_iter_ms:.2} ms, arena_used={:.1} MiB)",
            old_arena.used_bytes() as f64 / (1024.0 * 1024.0)
        );
        println!(
            "New LinkedScatter(push):total={new_total_ms:.2} ms (populate=0.00 ms upfront, scatter+lazy_slab={new_scat_ms:.2} ms, iter={new_iter_ms:.2} ms, slabs={new_slabs} ({new_used_mib:.1} MiB), chunks={new_chunks})"
        );
        println!(
            "New LinkedScatter(b8):  total={batch_total_ms:.2} ms (scatter+lazy_slab={batch_scat_ms:.2} ms, slabs={} ({:.1} MiB), chunks={})",
            batch_arena.pool().slab_count(),
            batch_arena.pool().used_bytes() as f64 / (1024.0 * 1024.0),
            batch_arena.total_chunks()
        );
        println!(
            "BoundarySeg (2500 segs): Vec<Vec>={vec_seg_ms:.3} ms vs BoundarySlab={flat_seg_ms:.3} ms"
        );
    }
}
