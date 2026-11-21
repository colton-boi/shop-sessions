use crate::ids::tok_less;
use crate::sys::{map_zeroed, mix};
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[repr(C)]
struct Slot {
    key: AtomicU64,
    /// While building: parent slot + 1, or 0 for a root. After `resolve`: the person's token.
    val: AtomicU64,
}

/// Lock-free ID table plus union-find. A root always carries the byte-wise
/// smallest ID of its set, because a union only ever points the larger-named
/// root at the smaller-named one.
pub struct LinkTable {
    slots: *const Slot,
    mask: usize,
}
unsafe impl Sync for LinkTable {}

impl LinkTable {
    pub fn new(pairs: usize) -> Self {
        let cap = (pairs * 3).next_power_of_two().max(4);
        LinkTable { slots: map_zeroed(cap * 16) as *const Slot, mask: cap - 1 }
    }

    pub fn slots(&self) -> usize {
        self.mask + 1
    }

    #[inline(always)]
    unsafe fn slot(&self, i: usize) -> &Slot {
        &*self.slots.add(i)
    }

    /// Prefetches the slot of a token whose `mix` is `h`.
    #[inline(always)]
    pub fn prefetch_hashed(&self, h: u64) {
        unsafe { _mm_prefetch(self.slots.add(h as usize & self.mask) as *const i8, _MM_HINT_T0) }
    }

    unsafe fn insert(&self, token: u64) -> usize {
        let mut i = mix(token) as usize & self.mask;
        loop {
            let s = self.slot(i);
            let k = s.key.load(Relaxed);
            if k == token {
                return i;
            }
            if k == 0 {
                match s.key.compare_exchange(0, token, Relaxed, Relaxed) {
                    Ok(_) => return i,
                    Err(now) if now == token => return i,
                    Err(_) => {}
                }
            }
            i = (i + 1) & self.mask;
        }
    }

    unsafe fn find(&self, mut i: usize) -> usize {
        loop {
            let p = self.slot(i).val.load(Relaxed);
            if p == 0 {
                return i;
            }
            let g = self.slot(p as usize - 1).val.load(Relaxed);
            if g == 0 {
                return p as usize - 1;
            }
            let _ = self.slot(i).val.compare_exchange(p, g, Relaxed, Relaxed);
            i = g as usize - 1;
        }
    }

    pub unsafe fn link(&self, a: u64, b: u64) {
        let (mut x, mut y) = (self.insert(a), self.insert(b));
        loop {
            x = self.find(x);
            y = self.find(y);
            if x == y {
                return;
            }
            if tok_less(self.slot(x).key.load(Relaxed), self.slot(y).key.load(Relaxed)) {
                std::mem::swap(&mut x, &mut y);
            }
            if self.slot(x).val.compare_exchange(0, y as u64 + 1, Relaxed, Relaxed).is_ok() {
                return;
            }
        }
    }

    /// First sweep after all links: point every slot straight at its root.
    pub unsafe fn compress(&self, from: usize, to: usize) {
        for i in from..to {
            if self.slot(i).key.load(Relaxed) != 0 {
                let r = self.find(i);
                if r != i {
                    self.slot(i).val.store(r as u64 + 1, Relaxed);
                }
            }
        }
    }

    /// Second sweep: replace the root index by the root's token.
    pub unsafe fn resolve(&self, from: usize, to: usize) {
        for i in from..to {
            let s = self.slot(i);
            let k = s.key.load(Relaxed);
            if k != 0 {
                let v = s.val.load(Relaxed);
                let name = if v == 0 { k } else { self.slot(v as usize - 1).key.load(Relaxed) };
                s.val.store(name, Relaxed);
            }
        }
    }

    /// The person an ID belongs to, given `h = mix(token)`; an ID no link
    /// mentions is its own person.
    #[inline(always)]
    pub unsafe fn person_hashed(&self, token: u64, h: u64) -> u64 {
        let mut i = h as usize & self.mask;
        loop {
            let s = self.slot(i);
            let k = s.key.load(Relaxed);
            if k == token {
                return s.val.load(Relaxed);
            }
            if k == 0 {
                return token;
            }
            i = (i + 1) & self.mask;
        }
    }
}
