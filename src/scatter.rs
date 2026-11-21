use crate::sys::map_zeroed;
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// A shared region of fixed-width records, handed out in chunks.
pub struct Arena<T> {
    pub base: *mut T,
    cap: usize,
    next: AtomicUsize,
    chunk: usize,
}
unsafe impl<T> Sync for Arena<T> {}

impl<T> Arena<T> {
    pub fn new(records: usize, chunk: usize) -> Self {
        let cap = records.div_ceil(chunk) * chunk;
        assert!(cap < u32::MAX as usize, "input too large for 32-bit record indexes");
        Arena { base: map_zeroed(cap * std::mem::size_of::<T>()) as *mut T, cap, next: AtomicUsize::new(0), chunk }
    }

    fn take(&self) -> usize {
        let at = self.next.fetch_add(self.chunk, Relaxed);
        assert!(at + self.chunk <= self.cap, "record arena overflow");
        at
    }
}

#[derive(Clone, Copy)]
struct Head {
    /// Arena index the next record goes to; equal to `end` when a chunk is needed.
    next: u32,
    end: u32,
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([u8; 64]);

/// One thread's write heads into an arena, one per partition. Records collect
/// in a cache-resident staging block per partition and leave as whole cache
/// lines with non-temporal stores, so the arena is never read while filling.
/// `UNIT` records must fill a whole number of cache lines.
pub struct Scatter<T, const UNIT: usize> {
    heads: Vec<Head>,
    staging: Vec<Line>,
    /// (partition, first record, record count) of every chunk this thread filled.
    pub log: Vec<(u32, u32, u32)>,
    _t: std::marker::PhantomData<T>,
}

impl<T: Copy, const UNIT: usize> Scatter<T, UNIT> {
    const LINES: usize = UNIT * std::mem::size_of::<T>() / 64;

    pub fn new() -> Self {
        assert!(UNIT.is_power_of_two() && UNIT * std::mem::size_of::<T>() % 64 == 0);
        Scatter { heads: Vec::new(), staging: Vec::new(), log: Vec::new(), _t: std::marker::PhantomData }
    }

    pub fn reset(&mut self, parts: usize) {
        self.heads = vec![Head { next: 0, end: 0 }; parts];
        self.staging = vec![Line([0; 64]); parts * Self::LINES];
    }

    /// Appends to a partition and returns the record's arena index.
    #[inline(always)]
    pub unsafe fn push(&mut self, arena: &Arena<T>, part: usize, v: T) -> u32 {
        if self.heads.get_unchecked(part).next == self.heads.get_unchecked(part).end {
            self.refill(arena, part);
        }
        let h = self.heads.get_unchecked_mut(part);
        let at = h.next;
        let block = self.staging.as_mut_ptr().add(part * Self::LINES) as *mut T;
        block.add(at as usize & (UNIT - 1)).write(v);
        h.next = at + 1;
        if (at as usize + 1) & (UNIT - 1) == 0 {
            let dst = arena.base.add(at as usize + 1 - UNIT) as *mut __m256i;
            let src = block as *const __m256i;
            for k in 0..Self::LINES * 2 {
                _mm256_stream_si256(dst.add(k), _mm256_load_si256(src.add(k)));
            }
        }
        at
    }

    #[cold]
    #[inline(never)]
    unsafe fn refill(&mut self, arena: &Arena<T>, part: usize) {
        let h = &mut self.heads[part];
        if h.end != 0 {
            self.log.push((part as u32, h.end - arena.chunk as u32, arena.chunk as u32));
        }
        debug_assert!(arena.chunk % UNIT == 0);
        // Index 0 would read as "no chunk yet", so the arena's first chunk is burned
        // by whichever thread gets it.
        let mut at = arena.take();
        if at == 0 {
            at = arena.take();
        }
        h.next = at as u32;
        h.end = (at + arena.chunk) as u32;
    }

    pub unsafe fn finish(&mut self, arena: &Arena<T>) {
        _mm_sfence();
        for (part, h) in self.heads.iter().enumerate() {
            if h.end == 0 {
                continue;
            }
            let first = h.end - arena.chunk as u32;
            let staged = h.next as usize & (UNIT - 1);
            let block = self.staging.as_ptr().add(part * Self::LINES) as *const T;
            std::ptr::copy_nonoverlapping(block, arena.base.add(h.next as usize - staged), staged);
            if h.next > first {
                self.log.push((part as u32, first, h.next - first));
            }
        }
        self.heads.clear();
    }
}

/// The partition of a record keyed by `key`. Any fixed function works; this
/// one is a multiply, where `sys::mix` is five operations. Its top bits are
/// independent of the low `mix` bits that the per-partition hash tables use.
#[inline(always)]
pub fn part_of_key(key: u64, parts: usize) -> usize {
    (((key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) * parts as u64) >> 32) as usize
}

/// Every thread's chunk log, regrouped by partition.
pub fn by_partition(logs: &[&Vec<(u32, u32, u32)>], parts: usize) -> Vec<Vec<(u32, u32)>> {
    let mut out = vec![Vec::new(); parts];
    for log in logs {
        for &(part, first, count) in log.iter() {
            out[part as usize].push((first, count));
        }
    }
    out
}
