use crate::csv::Reader;
use crate::dict::{
    pack_eu_base37_pair, pack_pair_base37, pack_prefix_base37, pack_u_base37, PidLease, UserDict,
};
use crate::group::{sort_u64_clicks, Cell, Grouper, V6ClickFb, V6ClickU64};
use crate::ids::{self, event_canon_key, event_eq, hash_hex, parse_hex33, tok_click, tok_event, SideWriter};
use crate::scan4k::{self, CsvRow, PreadBuf, RowHandler};
use crate::sys::{self, map_zeroed_huge, mix, par, release, PerThread};
use crate::ts::{self, DateCache, BAD, DAY_MS};
use crate::{Extent, Input, Mapped, Shared, WEEK_MS};
use std::arch::x86_64::*;
use std::os::unix::io::RawFd;
use std::path::Path;
use std::sync::atomic::{
    AtomicBool, AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed},
};
use std::sync::Mutex;

const SLAB_BYTES: usize = 16 << 20;
const BLOCK_BYTES: usize = 4160;
const BATCH: usize = 32;

const RAW_B: usize = 256;
const RAW_SIDE_BIT: u64 = 1 << 63;
const RAW_VALID_BIT: u64 = 1 << 62;
const RAW_EID_SIDE_BIT: u64 = 1 << 40;
const RAW_REM_MASK: u64 = (1 << 34) - 1;
const RAW_ING_MAX: i32 = (1 << 21) - 1;

const LONG_ING_MAX: u64 = (1 << 40) - 1;
const LONG_VALID_BIT: u64 = 1;

#[inline(always)]
pub fn applicable(sh: &Shared, event_bytes: u64) -> bool {
    sh.use_rel
        && sh.days <= 45
        && (sh.window_end - sh.base_ms) < (1 << 31)
        && sh.campaigns.len() <= 4095
        && sh.categories.len() <= 127
        && (event_bytes / 30) < (1 << 27)
        && std::env::var_os("SS_FORCE_V5").is_none()
}

#[inline(always)]
const fn mix42(x: u64) -> u64 {
    const MASK42: u64 = (1u64 << 42) - 1;
    let x = (x ^ (x >> 21)).wrapping_mul(0x224B_A937_15B7) & MASK42;
    x ^ (x >> 21)
}

#[repr(C, align(64))]
pub struct Block<T: Copy, const CAP: usize> {
    pub next: *mut Block<T, CAP>,
    pub count: u32,
    pub _pad: [u8; 64 - 12],
    pub items: [T; CAP],
}

struct SlabState {
    slabs: Vec<(*mut u8, usize)>,
    cur_ptr: *mut u8,
    cur_rem: usize,
}
unsafe impl Send for SlabState {}

pub struct BlockSlabPool {
    state: Mutex<SlabState>,
}

impl BlockSlabPool {
    pub fn with_initial(init_bytes: usize) -> (Self, (usize, usize)) {
        let aligned = if init_bytes == 0 {
            0
        } else {
            init_bytes.div_ceil(2 << 20) * (2 << 20)
        };
        let mut slabs = Vec::new();
        let (cur_ptr, cur_rem) = if aligned > 0 {
            let p = map_zeroed_huge(aligned);
            slabs.push((p, aligned));
            (p, aligned / BLOCK_BYTES)
        } else {
            (std::ptr::null_mut(), 0)
        };
        (
            Self {
                state: Mutex::new(SlabState {
                    slabs,
                    cur_ptr,
                    cur_rem,
                }),
            },
            (cur_ptr as usize, aligned),
        )
    }

    #[cold]
    #[inline(never)]
    fn take_batch(&self, want: usize) -> (*mut u8, usize) {
        let mut st = self.state.lock().unwrap();
        if st.cur_rem == 0 {
            let p = map_zeroed_huge(SLAB_BYTES);
            st.slabs.push((p, SLAB_BYTES));
            st.cur_ptr = p;
            st.cur_rem = SLAB_BYTES / BLOCK_BYTES;
        }
        let n = st.cur_rem.min(want);
        let ptr = st.cur_ptr;
        st.cur_ptr = unsafe { ptr.add(n * BLOCK_BYTES) };
        st.cur_rem -= n;
        (ptr, n)
    }

    pub fn release(&self) {
        let mut st = self.state.lock().unwrap();
        for (p, len) in st.slabs.drain(..) {
            release(p, len);
        }
        st.cur_ptr = std::ptr::null_mut();
        st.cur_rem = 0;
    }
}

pub struct LocalBlockAlloc {
    free_head: *mut u8,
    batch_ptr: *mut u8,
    batch_rem: usize,
}

impl LocalBlockAlloc {
    pub fn new() -> Self {
        Self {
            free_head: std::ptr::null_mut(),
            batch_ptr: std::ptr::null_mut(),
            batch_rem: 0,
        }
    }

    #[inline(always)]
    unsafe fn alloc<T: Copy, const CAP: usize>(&mut self, pool: &BlockSlabPool) -> *mut Block<T, CAP> {
        if !self.free_head.is_null() {
            let p = self.free_head as *mut Block<T, CAP>;
            self.free_head = *(p as *const *mut u8);
            (*p).next = std::ptr::null_mut();
            (*p).count = 0;
            return p;
        }
        self.alloc_fresh(pool)
    }

    #[cold]
    #[inline(never)]
    unsafe fn alloc_fresh<T: Copy, const CAP: usize>(
        &mut self,
        pool: &BlockSlabPool,
    ) -> *mut Block<T, CAP> {
        if self.batch_rem == 0 {
            (self.batch_ptr, self.batch_rem) = pool.take_batch(64);
        }
        let p = self.batch_ptr as *mut Block<T, CAP>;
        self.batch_ptr = self.batch_ptr.add(BLOCK_BYTES);
        self.batch_rem -= 1;
        (*p).next = std::ptr::null_mut();
        (*p).count = 0;
        p
    }

    #[inline(always)]
    unsafe fn recycle<T: Copy, const CAP: usize>(&mut self, mut head: *mut Block<T, CAP>) {
        while !head.is_null() {
            let next = (*head).next;
            *(head as *mut *mut u8) = self.free_head;
            self.free_head = head as *mut u8;
            head = next;
        }
    }
}

#[repr(C, align(64))]
struct WcArr<T: Copy, const B: usize, const WC: usize>([[T; WC]; B]);

pub struct PartitionedStore<T: Copy, const CAP: usize, const B: usize> {
    threads: usize,
    heads: Vec<*mut Block<T, CAP>>,
    counts: Vec<u32>,
}
unsafe impl<T: Copy, const CAP: usize, const B: usize> Sync for PartitionedStore<T, CAP, B> {}
unsafe impl<T: Copy, const CAP: usize, const B: usize> Send for PartitionedStore<T, CAP, B> {}

impl<T: Copy, const CAP: usize, const B: usize> PartitionedStore<T, CAP, B> {
    pub fn new(threads: usize) -> Self {
        Self {
            threads,
            heads: vec![std::ptr::null_mut(); B * threads],
            counts: vec![0; B * threads],
        }
    }

    #[inline(always)]
    unsafe fn publish(&self, b: usize, tid: usize, head: *mut Block<T, CAP>, count: u32) {
        let idx = b * self.threads + tid;
        let hp = self.heads.as_ptr() as *mut *mut Block<T, CAP>;
        let cp = self.counts.as_ptr() as *mut u32;
        *hp.add(idx) = head;
        *cp.add(idx) = count;
    }

    #[inline(always)]
    fn bucket_total(&self, b: usize) -> usize {
        let base = b * self.threads;
        let mut s = 0usize;
        for t in 0..self.threads {
            s += self.counts[base + t] as usize;
        }
        s
    }

    #[inline(always)]
    fn thread_head(&self, b: usize, t: usize) -> *mut Block<T, CAP> {
        self.heads[b * self.threads + t]
    }

    fn total_items(&self) -> usize {
        self.counts.iter().map(|&c| c as usize).sum()
    }
}

pub struct StoreWriter<T: Copy, const CAP: usize, const B: usize, const WC: usize> {
    wc: Box<WcArr<T, B, WC>>,
    wc_len: [u8; B],
    cur_block: [*mut Block<T, CAP>; B],
    cur_pos: [u16; B],
    heads: [*mut Block<T, CAP>; B],
    counts: [u32; B],
}

impl<T: Copy, const CAP: usize, const B: usize, const WC: usize> StoreWriter<T, CAP, B, WC> {
    pub fn new() -> Self {
        assert_eq!(std::mem::size_of::<Block<T, CAP>>(), BLOCK_BYTES);
        assert_eq!(WC * std::mem::size_of::<T>(), 64);
        let z: T = unsafe { std::mem::zeroed() };
        Self {
            wc: Box::new(WcArr([[z; WC]; B])),
            wc_len: [0; B],
            cur_block: [std::ptr::null_mut(); B],
            cur_pos: [CAP as u16; B],
            heads: [std::ptr::null_mut(); B],
            counts: [0; B],
        }
    }

    #[inline(always)]
    unsafe fn push(&mut self, b: usize, item: T, alloc: &mut LocalBlockAlloc, pool: &BlockSlabPool) {
        let len = *self.wc_len.get_unchecked(b) as usize;
        *self.wc.0.get_unchecked_mut(b).get_unchecked_mut(len) = item;
        if len + 1 < WC {
            *self.wc_len.get_unchecked_mut(b) = (len + 1) as u8;
        } else {
            let mut pos = *self.cur_pos.get_unchecked(b) as usize;
            if pos == CAP {
                self.swap_block(b, alloc, pool);
                pos = 0;
            }
            let blk = *self.cur_block.get_unchecked(b);
            let src = self.wc.0.as_ptr().add(b) as *const __m256i;
            let dst = (*blk).items.as_mut_ptr().add(pos) as *mut __m256i;
            let v0 = _mm256_load_si256(src);
            let v1 = _mm256_load_si256(src.add(1));
            _mm256_stream_si256(dst, v0);
            _mm256_stream_si256(dst.add(1), v1);
            *self.cur_pos.get_unchecked_mut(b) = (pos + WC) as u16;
            *self.wc_len.get_unchecked_mut(b) = 0;
            *self.counts.get_unchecked_mut(b) += WC as u32;
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn swap_block(&mut self, b: usize, alloc: &mut LocalBlockAlloc, pool: &BlockSlabPool) {
        let old = self.cur_block[b];
        if !old.is_null() {
            (*old).count = CAP as u32;
            (*old).next = self.heads[b];
            self.heads[b] = old;
        }
        self.cur_block[b] = alloc.alloc(pool);
        self.cur_pos[b] = 0;
    }

    unsafe fn finish(
        &mut self,
        store: &PartitionedStore<T, CAP, B>,
        tid: usize,
        alloc: &mut LocalBlockAlloc,
        pool: &BlockSlabPool,
    ) {
        for b in 0..B {
            let rem = self.wc_len[b] as usize;
            if rem > 0 {
                let mut pos = self.cur_pos[b] as usize;
                if pos == CAP {
                    self.swap_block(b, alloc, pool);
                    pos = 0;
                }
                let blk = self.cur_block[b];
                let dst = (*blk).items.as_mut_ptr().add(pos);
                std::ptr::copy_nonoverlapping(self.wc.0[b].as_ptr(), dst, rem);
                self.cur_pos[b] = (pos + rem) as u16;
                self.counts[b] += rem as u32;
                self.wc_len[b] = 0;
            }
            let blk = self.cur_block[b];
            if !blk.is_null() {
                (*blk).count = self.cur_pos[b] as u32;
                (*blk).next = self.heads[b];
                self.heads[b] = blk;
                self.cur_block[b] = std::ptr::null_mut();
                self.cur_pos[b] = CAP as u16;
            }
            store.publish(b, tid, self.heads[b], self.counts[b]);
            self.heads[b] = std::ptr::null_mut();
            self.counts[b] = 0;
        }
        _mm_sfence();
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RawEvent16 {
    pub w0: u64,
    pub w1: u64,
}

impl RawEvent16 {
    #[inline(always)]
    fn is_side(self) -> bool {
        (self.w0 & RAW_SIDE_BIT) != 0
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn is_valid(self) -> bool {
        (self.w0 & RAW_VALID_BIT) != 0
    }
    #[inline(always)]
    fn eid_is_side(self) -> bool {
        (self.w0 & RAW_EID_SIDE_BIT) != 0
    }
    #[inline(always)]
    fn ing_rel21(self) -> u32 {
        ((self.w0 >> 41) & (RAW_ING_MAX as u64)) as u32
    }
    #[inline(always)]
    fn rem34(self) -> u64 {
        self.w0 & RAW_REM_MASK
    }
    #[inline(always)]
    fn inline_ts(self) -> u32 {
        (self.w1 >> 32) as u32
    }
    #[inline(always)]
    fn inline_pid(self) -> u32 {
        ((self.w1 >> 8) & 0x3F_FFFF) as u32
    }
    #[inline(always)]
    fn side_tid(self) -> usize {
        (self.w1 >> 26) as usize
    }
    #[inline(always)]
    fn side_idx(self) -> usize {
        (self.w1 & ((1 << 26) - 1)) as usize
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct LongEv32 {
    pub k_lo: u64,
    pub k_hi: u64,
    pub ing_pid: u64,
    pub payload: u64,
}

impl LongEv32 {
    #[inline(always)]
    fn ing40(self) -> u64 {
        self.ing_pid >> 24
    }
    #[inline(always)]
    fn pid(self) -> u32 {
        (self.ing_pid & 0x3F_FFFF) as u32
    }
    #[inline(always)]
    fn ts_rel_ms(self) -> u32 {
        (self.payload >> 32) as u32
    }
    #[inline(always)]
    fn price(self) -> u32 {
        ((self.payload >> 8) & 0xFF_FFFF) as u32
    }
    #[inline(always)]
    fn cat_id(self) -> u32 {
        ((self.payload >> 1) & 0x7F) as u32
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn is_valid(self) -> bool {
        (self.payload & LONG_VALID_BIT) != 0
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CompactClick {
    pub click_id: u64,
    pub meta: u64,
}

impl CompactClick {
    #[inline(always)]
    fn pid_hi(self) -> usize {
        (self.meta >> 44) as usize
    }
    #[inline(always)]
    fn ts_rel_ms(self) -> u32 {
        ((self.meta >> 12) & 0xFFFF_FFFF) as u32
    }
    #[inline(always)]
    fn camp(self) -> u32 {
        (self.meta & 0xFFF) as u32
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CompactPersonEv8 {
    pub pid_hi: u32,
    pub ts_rel_ms: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CompactPurchase {
    pub pid_hi: u32,
    pub ts_rel_ms: u32,
    pub price: u32,
    pub cat_id: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SideEvent {
    pub eid_tok: u64,
    pub ingest_s: i64,
    pub ts_rel_ms: u32,
    pub pid: u32,
    pub price: u32,
    pub cat_id: u8,
    pub valid: u8,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RawUPre {
    u_ptr: *const u8,
    key42: u64,
    w0: u64,
    ts_rel_ms: u32,
    u_slot: u32,
    eb: u16,
    u_len: u16,
}

impl Default for RawUPre {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RawHPre {
    hi: u64,
    lo: u64,
    w0: u64,
    ts_rel_ms: u32,
    u_slot: u32,
    eb: u16,
    _pad: u16,
}

impl Default for RawHPre {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct SidePre {
    u_ptr: *const u8,
    u_k1: u64,
    u_k2: u64,
    w0: u64,
    side_idx: u32,
    u_slot: u32,
    eb: u16,
    u_len: u16,
    u_kind: u8,
}

impl Default for SidePre {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct LongPre {
    u_ptr: *const u8,
    u_k1: u64,
    u_k2: u64,
    k_hi: u64,
    k_lo: u64,
    ing40: u64,
    payload: u64,
    u_slot: u32,
    u_len: u16,
    u_kind: u8,
    lb: u8,
}

impl Default for LongPre {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Clone, Copy)]
struct CkUPre {
    u_ptr: *const u8,
    u_len: u16,
    u_slot: u32,
    key42: u64,
    click_id: u64,
    ts_rel_ms: u32,
    camp: u32,
}

impl Default for CkUPre {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[derive(Clone, Copy)]
struct CkHPre {
    hi: u64,
    lo: u64,
    u_slot: u32,
    click_id: u64,
    ts_rel_ms: u32,
    camp: u32,
}

impl Default for CkHPre {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

struct ParseWorker {
    reader: Reader,
    pbuf: Option<PreadBuf>,
    dates: DateCache,
    lease: PidLease,
    alloc: LocalBlockAlloc,
    raw_writer: StoreWriter<RawEvent16, 256, RAW_B, 4>,
    long_writer: StoreWriter<LongEv32, 128, 128, 2>,
    c_writer: StoreWriter<CompactClick, 256, 256, 4>,
    e_side: SideWriter,
    c_side: SideWriter,
    side_events: Vec<SideEvent>,
    total_event_lines: u64,
}

impl ParseWorker {
    fn new(read_buf: usize, pread: bool) -> Self {
        Self {
            reader: Reader::lazy(read_buf),
            pbuf: pread.then(PreadBuf::new),
            dates: DateCache::new(),
            lease: PidLease::default(),
            alloc: LocalBlockAlloc::new(),
            raw_writer: StoreWriter::new(),
            long_writer: StoreWriter::new(),
            c_writer: StoreWriter::new(),
            e_side: SideWriter::default(),
            c_side: SideWriter::default(),
            side_events: Vec::new(),
            total_event_lines: 0,
        }
    }
}

#[cold]
#[inline(never)]
unsafe fn parse_purchase_cold(
    sh: &Shared,
    p_ty: *const u8,
    p_prod: *const u8,
    l_prod: usize,
    p_price: *const u8,
    l_price: usize,
) -> Option<(u8, u32)> {
    const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");
    if (p_ty as *const u64).read_unaligned() != PURCHASE {
        return None;
    }
    let price = match ts::digits(p_price, l_price) {
        Some(v) if (1..=10_000_000).contains(&v) => v as u32,
        _ => return None,
    };
    let cat = sh.product_cat(p_prod, l_prod);
    Some((cat, price))
}

#[cold]
#[inline(never)]
unsafe fn exact_ingest_s_cold(p: *const u8, len: usize, dates: &mut DateCache) -> i64 {
    let s = ts::parse_f1(p, len, dates);
    if s == BAD {
        i64::MAX
    } else {
        s
    }
}

#[cold]
#[inline(never)]
unsafe fn tok_event_cold(
    p: *const u8,
    len: usize,
    e_side: &mut SideWriter,
    sh: &Shared,
) -> (usize, u64, u64) {
    let tok = tok_event(p, len, e_side, &sh.e_side_arena);
    let canon = event_canon_key(tok);
    let h = mix(canon);
    let eb = (h >> 56) as usize;
    let rem34 = (h >> 22) & RAW_REM_MASK;
    (eb, rem34, tok)
}

#[cold]
#[inline(never)]
unsafe fn tok_click_cold(
    p: *const u8,
    len: usize,
    c_side: &mut SideWriter,
    sh: &Shared,
) -> u64 {
    tok_click(p, len, c_side, &sh.c_side_arena)
}

#[inline(always)]
unsafe fn prep_user_prefetch(dict: &UserDict, p: *const u8, len: usize) -> (u8, usize, u64, u64) {
    if let Some(key42) = pack_u_base37(p, len) {
        let slot = dict.prefetch_u(key42);
        return (1, slot, key42, 0);
    }
    if len == 33 && *p == b'u' {
        if let Some((_, hi, lo)) = parse_hex33(p, len) {
            let slot = dict.prefetch_hex(hi, lo);
            return (2, slot, hi, lo);
        }
    }
    (3, 0, 0, 0)
}

#[inline(always)]
unsafe fn resolve_user_parts(
    dict: &UserDict,
    u_kind: u8,
    u_slot: usize,
    u_k1: u64,
    u_k2: u64,
    u_ptr: *const u8,
    u_len: usize,
    lease: &mut PidLease,
    tid: usize,
) -> u32 {
    if u_kind == 1 {
        dict.get_or_insert_u_prehashed(u_k1, u_slot, u_ptr, u_len, lease)
    } else if u_kind == 2 {
        dict.get_or_insert_hex_prehashed(u_k1, u_k2, u_slot, lease, tid)
    } else {
        dict.get_or_insert_gen(u_ptr, u_len, lease, tid)
    }
}

/// Flushes strictly valid short non-side events (zero `RAW_SIDE_BIT` check).
#[inline(never)]
unsafe fn flush_raw_u_batch(
    dict: &UserDict,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    batch: &[RawUPre],
) {
    for pre in batch {
        let pid = dict.get_or_insert_u_prehashed(
            pre.key42,
            pre.u_slot as usize,
            pre.u_ptr,
            pre.u_len as usize,
            &mut w.lease,
        );
        let w1 = ((pre.ts_rel_ms as u64) << 32) | ((pid as u64) << 8);
        w.raw_writer
            .push(pre.eb as usize, RawEvent16 { w0: pre.w0, w1 }, &mut w.alloc, pool);
    }
}

/// Flushes strictly valid hex non-side events (zero `RAW_SIDE_BIT` check).
#[inline(never)]
unsafe fn flush_raw_h_batch(
    dict: &UserDict,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    tid: usize,
    batch: &[RawHPre],
) {
    for pre in batch {
        let pid = dict.get_or_insert_hex_prehashed(
            pre.hi,
            pre.lo,
            pre.u_slot as usize,
            &mut w.lease,
            tid,
        );
        let w1 = ((pre.ts_rel_ms as u64) << 32) | ((pid as u64) << 8);
        w.raw_writer
            .push(pre.eb as usize, RawEvent16 { w0: pre.w0, w1 }, &mut w.alloc, pool);
    }
}

/// Flushes valid side/purchase raw events.
#[inline(never)]
unsafe fn flush_side_batch(
    dict: &UserDict,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    tid: usize,
    batch: &[SidePre],
) {
    for pre in batch {
        let pid = resolve_user_parts(
            dict,
            pre.u_kind,
            pre.u_slot as usize,
            pre.u_k1,
            pre.u_k2,
            pre.u_ptr,
            pre.u_len as usize,
            &mut w.lease,
            tid,
        );
        w.side_events.get_unchecked_mut(pre.side_idx as usize).pid = pid;
        let w1 = ((tid as u64) << 26) | (pre.side_idx as u64);
        w.raw_writer
            .push(pre.eb as usize, RawEvent16 { w0: pre.w0, w1 }, &mut w.alloc, pool);
    }
}

#[inline(never)]
unsafe fn flush_long_batch(
    dict: &UserDict,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    tid: usize,
    batch: &[LongPre],
) {
    for pre in batch {
        let pid = resolve_user_parts(
            dict,
            pre.u_kind,
            pre.u_slot as usize,
            pre.u_k1,
            pre.u_k2,
            pre.u_ptr,
            pre.u_len as usize,
            &mut w.lease,
            tid,
        );
        w.long_writer.push(
            pre.lb as usize,
            LongEv32 {
                k_lo: pre.k_lo,
                k_hi: pre.k_hi,
                ing_pid: (pre.ing40 << 24) | (pid as u64),
                payload: pre.payload,
            },
            &mut w.alloc,
            pool,
        );
    }
}

#[cold]
#[inline(never)]
unsafe fn parse_ts_pair_slow(
    sh: &Shared,
    dates: &mut DateCache,
    base_s: i64,
    event_win_ms: u64,
    p2: *const u8,
    l2: usize,
    p3: *const u8,
    l3: usize,
    fields_ok: bool,
) -> (i32, u32, bool) {
    let ingest_s_rel = ts::parse_f1_rel(p3, l3, &sh.date_lut, dates, base_s);
    if !fields_ok || ingest_s_rel == i32::MAX {
        return (ingest_s_rel, 0, false);
    }
    let ts_rel = ts::parse_ts_rel(p2, l2, &sh.date_lut, dates, sh.base_ms);
    let win_bad = (ts_rel.wrapping_sub(WEEK_MS) as u64) >= event_win_ms;
    let late_bad = (ingest_s_rel as i64).wrapping_mul(1000).wrapping_sub(ts_rel) > DAY_MS;
    if win_bad | late_bad {
        (ingest_s_rel, 0, false)
    } else {
        (ingest_s_rel, ts_rel as u32, true)
    }
}

#[cold]
#[inline(never)]
unsafe fn push_side_raw_ev(
    dict: &UserDict,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    tid: usize,
    side_overflow: &AtomicBool,
    side_batch: &mut [SidePre; BATCH],
    side_nb: &mut usize,
    eb: usize,
    rem_w0_bits: u64,
    eid_tok: u64,
    p1: *const u8,
    l1: usize,
    known_u_key42: Option<u64>,
    p3: *const u8,
    l3: usize,
    ingest_s_rel: i32,
    base_s: i64,
    ts_rel_ms: u32,
    price: u32,
    cat_id: u8,
    valid: bool,
) {
    let exact_s = if ingest_s_rel != i32::MIN && ingest_s_rel < i32::MAX - 1 {
        base_s + ingest_s_rel as i64
    } else {
        exact_ingest_s_cold(p3, l3, &mut w.dates)
    };
    let clamped_rel21 = if exact_s == i64::MAX {
        RAW_ING_MAX as u32
    } else {
        (exact_s - (base_s + 604_800 - (1 << 19))).clamp(0, RAW_ING_MAX as i64) as u32
    };
    let side_idx = w.side_events.len();
    if side_idx >= (1 << 26) {
        side_overflow.store(true, Relaxed);
        return;
    }
    w.side_events.push(SideEvent {
        eid_tok,
        ingest_s: exact_s,
        ts_rel_ms,
        pid: 0,
        price,
        cat_id,
        valid: valid as u8,
    });
    let w0 = rem_w0_bits
        | RAW_SIDE_BIT
        | ((clamped_rel21 as u64) << 41)
        | if valid { RAW_VALID_BIT } else { 0 };
    if !valid {
        let w1 = ((tid as u64) << 26) | (side_idx as u64);
        w.raw_writer
            .push(eb, RawEvent16 { w0, w1 }, &mut w.alloc, pool);
        return;
    }
    let (u_kind, u_slot, u_k1, u_k2) = match known_u_key42 {
        Some(key42) => (1u8, dict.prefetch_u(key42), key42, 0u64),
        None => prep_user_prefetch(dict, p1, l1),
    };
    let pre = side_batch.get_unchecked_mut(*side_nb);
    pre.u_ptr = p1;
    pre.u_len = l1 as u16;
    pre.u_kind = u_kind;
    pre.u_slot = u_slot as u32;
    pre.u_k1 = u_k1;
    pre.u_k2 = u_k2;
    pre.w0 = w0;
    pre.side_idx = side_idx as u32;
    pre.eb = eb as u16;
    *side_nb += 1;
    if *side_nb == BATCH {
        flush_side_batch(dict, w, pool, tid, side_batch);
        *side_nb = 0;
    }
}

struct EventRowParser<'a> {
    sh: &'a Shared,
    dict: &'a UserDict,
    w: &'a mut ParseWorker,
    pool: &'a BlockSlabPool,
    tid: usize,
    side_overflow: &'a AtomicBool,
    base_s: i64,
    event_win_ms: u64,
    u_batch: [RawUPre; BATCH],
    u_nb: usize,
    h_batch: [RawHPre; BATCH],
    h_nb: usize,
    side_batch: [SidePre; BATCH],
    side_nb: usize,
    long_batch: [LongPre; BATCH],
    long_nb: usize,
}

impl<'a> EventRowParser<'a> {
    #[inline(always)]
    unsafe fn flush_all(&mut self) {
        if self.u_nb > 0 {
            flush_raw_u_batch(self.dict, self.w, self.pool, &self.u_batch[..self.u_nb]);
            self.u_nb = 0;
        }
        if self.h_nb > 0 {
            flush_raw_h_batch(
                self.dict,
                self.w,
                self.pool,
                self.tid,
                &self.h_batch[..self.h_nb],
            );
            self.h_nb = 0;
        }
        if self.side_nb > 0 {
            flush_side_batch(
                self.dict,
                self.w,
                self.pool,
                self.tid,
                &self.side_batch[..self.side_nb],
            );
            self.side_nb = 0;
        }
        if self.long_nb > 0 {
            flush_long_batch(
                self.dict,
                self.w,
                self.pool,
                self.tid,
                &self.long_batch[..self.long_nb],
            );
            self.long_nb = 0;
        }
    }
}

impl<'a> RowHandler for EventRowParser<'a> {
    #[inline(always)]
    unsafe fn before_refill(&mut self) {
        self.flush_all();
    }

    #[inline(always)]
    unsafe fn on_row<R: CsvRow>(&mut self, row: R) {
        const VIEW: u32 = u32::from_le_bytes(*b"view");
        const CART: u32 = u32::from_le_bytes(*b"cart");
        let (p0, l0) = row.field(0);
        if l0 == 0 {
            return;
        }
        self.w.total_event_lines += 1;
        let (p1, l1) = row.field(1);
        let (p2, l2) = row.field(2);
        let (p3, l3) = row.field(3);
        let (p4, l4) = row.field(4);
        let (p5, l5) = row.field(5);

        let fields_ok = (l1 != 0) & (l5 != 0);
        let (ingest_s_rel, ts_rel_ms, mut valid) = if l2 == 20 && l3 == 20 {
            if let Some((ts_s_rel, ing_s_rel)) = ts::parse_dual_f1_rel(p2, p3, &self.sh.date_lut)
            {
                let ts_rel = (ts_s_rel as i64) * 1000;
                let win_bad = (ts_rel.wrapping_sub(WEEK_MS) as u64) >= self.event_win_ms;
                let late_bad = (ing_s_rel as i64 - ts_s_rel as i64) > 86_400;
                (ing_s_rel, ts_rel as u32, fields_ok & !win_bad & !late_bad)
            } else {
                parse_ts_pair_slow(
                    self.sh,
                    &mut self.w.dates,
                    self.base_s,
                    self.event_win_ms,
                    p2,
                    l2,
                    p3,
                    l3,
                    fields_ok,
                )
            }
        } else {
            parse_ts_pair_slow(
                self.sh,
                &mut self.w.dates,
                self.base_s,
                self.event_win_ms,
                p2,
                l2,
                p3,
                l3,
                fields_ok,
            )
        };

        let mut price = 0u32;
        let mut cat_id = 0u8;
        if valid {
            if l4 == 4 {
                let ty = (p4 as *const u32).read_unaligned();
                valid &= !((ty != VIEW) & (ty != CART));
            } else if l4 == 8 {
                let (p6, l6) = row.field(6);
                match parse_purchase_cold(self.sh, p4, p5, l5, p6, l6) {
                    Some((c, pr)) => {
                        cat_id = c;
                        price = pr;
                    }
                    None => valid = false,
                }
            } else {
                valid = false;
            }
        }

        if let Some((eid42, u_key42)) = pack_eu_base37_pair(p0, l0, p1, l1) {
            let h42 = mix42(eid42);
            let eb = (h42 >> 34) as usize;
            let rem34 = h42 & RAW_REM_MASK;
            let ing_off = ingest_s_rel - (604_800 - (1 << 19));
            let in_range_ing =
                ingest_s_rel != i32::MAX && ((ing_off - 1) as u32) < (RAW_ING_MAX as u32 - 1);
            if in_range_ing && price == 0 {
                if !valid {
                    let w0 = ((ing_off as u64) << 41) | rem34;
                    self.w
                        .raw_writer
                        .push(eb, RawEvent16 { w0, w1: 0 }, &mut self.w.alloc, self.pool);
                    return;
                }
                let u_slot = self.dict.prefetch_u(u_key42);
                let w0 = ((ing_off as u64) << 41) | rem34 | RAW_VALID_BIT;
                let pre = self.u_batch.get_unchecked_mut(self.u_nb);
                pre.u_ptr = p1;
                pre.u_len = l1 as u16;
                pre.key42 = u_key42;
                pre.u_slot = u_slot as u32;
                pre.w0 = w0;
                pre.ts_rel_ms = ts_rel_ms;
                pre.eb = eb as u16;
                self.u_nb += 1;
                if self.u_nb == BATCH {
                    flush_raw_u_batch(self.dict, self.w, self.pool, &self.u_batch);
                    self.u_nb = 0;
                }
            } else {
                push_side_raw_ev(
                    self.dict,
                    self.w,
                    self.pool,
                    self.tid,
                    self.side_overflow,
                    &mut self.side_batch,
                    &mut self.side_nb,
                    eb,
                    rem34,
                    0,
                    p1,
                    l1,
                    Some(u_key42),
                    p3,
                    l3,
                    ingest_s_rel,
                    self.base_s,
                    ts_rel_ms,
                    price,
                    cat_id,
                    valid,
                );
            }
            return;
        }

        if let Some(eid42) = pack_prefix_base37(b'e', p0, l0) {
            let h42 = mix42(eid42);
            let eb = (h42 >> 34) as usize;
            let rem34 = h42 & RAW_REM_MASK;
            let ing_off = ingest_s_rel - (604_800 - (1 << 19));
            let in_range_ing =
                ingest_s_rel != i32::MAX && ((ing_off - 1) as u32) < (RAW_ING_MAX as u32 - 1);
            if in_range_ing && price == 0 {
                if !valid {
                    let w0 = ((ing_off as u64) << 41) | rem34;
                    self.w
                        .raw_writer
                        .push(eb, RawEvent16 { w0, w1: 0 }, &mut self.w.alloc, self.pool);
                    return;
                }
                let w0 = ((ing_off as u64) << 41) | rem34 | RAW_VALID_BIT;
                if l1 == 33 && *p1 == b'u' {
                    if let Some((_, hi, lo)) = parse_hex33(p1, l1) {
                        let u_slot = self.dict.prefetch_hex(hi, lo);
                        let pre = self.h_batch.get_unchecked_mut(self.h_nb);
                        pre.hi = hi;
                        pre.lo = lo;
                        pre.u_slot = u_slot as u32;
                        pre.w0 = w0;
                        pre.ts_rel_ms = ts_rel_ms;
                        pre.eb = eb as u16;
                        self.h_nb += 1;
                        if self.h_nb == BATCH {
                            flush_raw_h_batch(
                                self.dict,
                                self.w,
                                self.pool,
                                self.tid,
                                &self.h_batch,
                            );
                            self.h_nb = 0;
                        }
                        return;
                    }
                }
                let pid = self
                    .dict
                    .get_or_insert_gen(p1, l1, &mut self.w.lease, self.tid);
                let w1 = ((ts_rel_ms as u64) << 32) | ((pid as u64) << 8);
                self.w
                    .raw_writer
                    .push(eb, RawEvent16 { w0, w1 }, &mut self.w.alloc, self.pool);
            } else {
                push_side_raw_ev(
                    self.dict,
                    self.w,
                    self.pool,
                    self.tid,
                    self.side_overflow,
                    &mut self.side_batch,
                    &mut self.side_nb,
                    eb,
                    rem34,
                    0,
                    p1,
                    l1,
                    None,
                    p3,
                    l3,
                    ingest_s_rel,
                    self.base_s,
                    ts_rel_ms,
                    price,
                    cat_id,
                    valid,
                );
            }
            return;
        }

        if l0 == 33 && *p0 == b'e' {
            if let Some((_, hi, lo)) = parse_hex33(p0, 33) {
                let lb = (hash_hex(b'e', hi, lo) >> 57) as usize;
                let ing40 = if ingest_s_rel != i32::MIN && ingest_s_rel < i32::MAX - 1 {
                    ((ingest_s_rel as i64) + (1i64 << 39)) as u64
                } else {
                    let exact_s = exact_ingest_s_cold(p3, l3, &mut self.w.dates);
                    if exact_s == i64::MAX {
                        LONG_ING_MAX
                    } else {
                        (exact_s - self.base_s + (1i64 << 39))
                            .clamp(0, (LONG_ING_MAX - 1) as i64) as u64
                    }
                };
                let payload = ((ts_rel_ms as u64) << 32)
                    | ((price as u64) << 8)
                    | ((cat_id as u64) << 1)
                    | (valid as u64);
                if !valid {
                    self.w.long_writer.push(
                        lb,
                        LongEv32 {
                            k_lo: lo,
                            k_hi: hi,
                            ing_pid: ing40 << 24,
                            payload,
                        },
                        &mut self.w.alloc,
                        self.pool,
                    );
                    return;
                }
                let (u_kind, u_slot, u_k1, u_k2) = prep_user_prefetch(self.dict, p1, l1);
                let pre = self.long_batch.get_unchecked_mut(self.long_nb);
                pre.u_ptr = p1;
                pre.u_len = l1 as u16;
                pre.u_kind = u_kind;
                pre.u_slot = u_slot as u32;
                pre.u_k1 = u_k1;
                pre.u_k2 = u_k2;
                pre.k_hi = hi;
                pre.k_lo = lo;
                pre.ing40 = ing40;
                pre.payload = payload;
                pre.lb = lb as u8;
                self.long_nb += 1;
                if self.long_nb == BATCH {
                    flush_long_batch(self.dict, self.w, self.pool, self.tid, &self.long_batch);
                    self.long_nb = 0;
                }
                return;
            }
        }

        let (eb, rem34, tok) = tok_event_cold(p0, l0, &mut self.w.e_side, self.sh);
        push_side_raw_ev(
            self.dict,
            self.w,
            self.pool,
            self.tid,
            self.side_overflow,
            &mut self.side_batch,
            &mut self.side_nb,
            eb,
            rem34 | RAW_EID_SIDE_BIT,
            tok,
            p1,
            l1,
            None,
            p3,
            l3,
            ingest_s_rel,
            self.base_s,
            ts_rel_ms,
            price,
            cat_id,
            valid,
        );
    }
}

#[inline(never)]
unsafe fn parse_events_v6(
    sh: &Shared,
    dict: &UserDict,
    Mapped(data, size): Mapped,
    fd: Option<RawFd>,
    x: &Extent,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    tid: usize,
    side_overflow: &AtomicBool,
) {
    let base_s = sh.base_ms.div_euclid(1000);
    let event_win_ms = (sh.window_end - sh.window_start) as u64;
    let mut reader = std::mem::replace(&mut w.reader, Reader::lazy(0));
    let mut pbuf = w.pbuf.take();
    {
        let mut parser = EventRowParser {
            sh,
            dict,
            w,
            pool,
            tid,
            side_overflow,
            base_s,
            event_win_ms,
            u_batch: [RawUPre::default(); BATCH],
            u_nb: 0,
            h_batch: [RawHPre::default(); BATCH],
            h_nb: 0,
            side_batch: [SidePre::default(); BATCH],
            side_nb: 0,
            long_batch: [LongPre::default(); BATCH],
            long_nb: 0,
        };
        if let Some(fd) = fd {
            let pb = pbuf.get_or_insert_with(PreadBuf::new);
            scan4k::scan_rows_pread::<7, _>(
                fd,
                size,
                x.from as usize,
                x.to as usize,
                &mut reader,
                pb,
                &mut parser,
            );
        } else {
            scan4k::scan_rows::<7, _>(
                data,
                size,
                x.from as usize,
                x.to as usize,
                &mut reader,
                &mut parser,
            );
        }
        parser.flush_all();
    }
    w.reader = reader;
    w.pbuf = pbuf;
}

/// Flushes short base37 click user batch (zero `u_kind` branch).
#[inline(never)]
unsafe fn flush_ck_u_batch(
    dict: &UserDict,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    batch: &[CkUPre],
) {
    for pre in batch {
        let pid = dict.get_or_insert_u_prehashed(
            pre.key42,
            pre.u_slot as usize,
            pre.u_ptr,
            pre.u_len as usize,
            &mut w.lease,
        );
        if pid == 0 {
            continue;
        }
        let meta = (((pid as u64) >> 8) << 44)
            | ((pre.ts_rel_ms as u64) << 12)
            | (pre.camp as u64 & 0xFFF);
        w.c_writer.push(
            (pid & 255) as usize,
            CompactClick {
                click_id: pre.click_id,
                meta,
            },
            &mut w.alloc,
            pool,
        );
    }
}

/// Flushes 33B hex click user batch (zero `u_kind` branch).
#[inline(never)]
unsafe fn flush_ck_h_batch(
    dict: &UserDict,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    tid: usize,
    batch: &[CkHPre],
) {
    for pre in batch {
        let pid = dict.get_or_insert_hex_prehashed(
            pre.hi,
            pre.lo,
            pre.u_slot as usize,
            &mut w.lease,
            tid,
        );
        if pid == 0 {
            continue;
        }
        let meta = (((pid as u64) >> 8) << 44)
            | ((pre.ts_rel_ms as u64) << 12)
            | (pre.camp as u64 & 0xFFF);
        w.c_writer.push(
            (pid & 255) as usize,
            CompactClick {
                click_id: pre.click_id,
                meta,
            },
            &mut w.alloc,
            pool,
        );
    }
}

struct ClickRowParser<'a> {
    sh: &'a Shared,
    dict: &'a UserDict,
    w: &'a mut ParseWorker,
    pool: &'a BlockSlabPool,
    tid: usize,
    click_win_ms: u64,
    u_batch: [CkUPre; BATCH],
    u_nb: usize,
    h_batch: [CkHPre; BATCH],
    h_nb: usize,
}

impl<'a> ClickRowParser<'a> {
    #[inline(always)]
    unsafe fn flush_all(&mut self) {
        if self.u_nb > 0 {
            flush_ck_u_batch(self.dict, self.w, self.pool, &self.u_batch[..self.u_nb]);
            self.u_nb = 0;
        }
        if self.h_nb > 0 {
            flush_ck_h_batch(
                self.dict,
                self.w,
                self.pool,
                self.tid,
                &self.h_batch[..self.h_nb],
            );
            self.h_nb = 0;
        }
    }
}

impl<'a> RowHandler for ClickRowParser<'a> {
    #[inline(always)]
    unsafe fn before_refill(&mut self) {
        self.flush_all();
    }

    #[inline(always)]
    unsafe fn on_row<R: CsvRow>(&mut self, row: R) {
        let (p0, l0) = row.field(0);
        let (p1, l1) = row.field(1);
        if l0 == 0 || l1 == 0 {
            return;
        }
        let (p3, l3) = row.field(3);
        let ts_rel = ts::parse_ts_rel(p3, l3, &self.sh.date_lut, &mut self.w.dates, self.sh.base_ms);
        if (ts_rel as u64) >= self.click_win_ms {
            return;
        }
        let (p2, l2) = row.field(2);
        let Some(camp) = self.sh.campaign_id(p2, l2) else {
            return;
        };
        let click_id = if l0 <= 9 {
            let t = ids::pack(p0, l0);
            if t != 0 {
                t
            } else {
                tok_click_cold(p0, l0, &mut self.w.c_side, self.sh)
            }
        } else {
            tok_click_cold(p0, l0, &mut self.w.c_side, self.sh)
        };
        let ts_rel_ms = ts_rel as u32;
        if let Some(key42) = pack_u_base37(p1, l1) {
            let u_slot = self.dict.prefetch_u(key42);
            let pre = self.u_batch.get_unchecked_mut(self.u_nb);
            pre.click_id = click_id;
            pre.ts_rel_ms = ts_rel_ms;
            pre.camp = camp;
            pre.u_ptr = p1;
            pre.u_len = l1 as u16;
            pre.key42 = key42;
            pre.u_slot = u_slot as u32;
            self.u_nb += 1;
            if self.u_nb == BATCH {
                flush_ck_u_batch(self.dict, self.w, self.pool, &self.u_batch);
                self.u_nb = 0;
            }
        } else if l1 == 33 && *p1 == b'u' {
            if let Some((_, hi, lo)) = parse_hex33(p1, l1) {
                let u_slot = self.dict.prefetch_hex(hi, lo);
                let pre = self.h_batch.get_unchecked_mut(self.h_nb);
                pre.click_id = click_id;
                pre.ts_rel_ms = ts_rel_ms;
                pre.camp = camp;
                pre.hi = hi;
                pre.lo = lo;
                pre.u_slot = u_slot as u32;
                self.h_nb += 1;
                if self.h_nb == BATCH {
                    flush_ck_h_batch(self.dict, self.w, self.pool, self.tid, &self.h_batch);
                    self.h_nb = 0;
                }
                return;
            }
            let pid = self
                .dict
                .get_or_insert_gen(p1, l1, &mut self.w.lease, self.tid);
            if pid != 0 {
                let meta = (((pid as u64) >> 8) << 44)
                    | ((ts_rel_ms as u64) << 12)
                    | (camp as u64 & 0xFFF);
                self.w.c_writer.push(
                    (pid & 255) as usize,
                    CompactClick { click_id, meta },
                    &mut self.w.alloc,
                    self.pool,
                );
            }
        } else {
            let pid = self
                .dict
                .get_or_insert_gen(p1, l1, &mut self.w.lease, self.tid);
            if pid != 0 {
                let meta = (((pid as u64) >> 8) << 44)
                    | ((ts_rel_ms as u64) << 12)
                    | (camp as u64 & 0xFFF);
                self.w.c_writer.push(
                    (pid & 255) as usize,
                    CompactClick { click_id, meta },
                    &mut self.w.alloc,
                    self.pool,
                );
            }
        }
    }
}

#[inline(never)]
unsafe fn parse_clicks_v6(
    sh: &Shared,
    dict: &UserDict,
    Mapped(data, size): Mapped,
    fd: Option<RawFd>,
    x: &Extent,
    w: &mut ParseWorker,
    pool: &BlockSlabPool,
    tid: usize,
) {
    let click_win_ms = (sh.window_end - sh.base_ms) as u64;
    let mut reader = std::mem::replace(&mut w.reader, Reader::lazy(0));
    let mut pbuf = w.pbuf.take();
    {
        let mut parser = ClickRowParser {
            sh,
            dict,
            w,
            pool,
            tid,
            click_win_ms,
            u_batch: [CkUPre::default(); BATCH],
            u_nb: 0,
            h_batch: [CkHPre::default(); BATCH],
            h_nb: 0,
        };
        if let Some(fd) = fd {
            let pb = pbuf.get_or_insert_with(PreadBuf::new);
            scan4k::scan_rows_pread::<4, _>(
                fd,
                size,
                x.from as usize,
                x.to as usize,
                &mut reader,
                pb,
                &mut parser,
            );
        } else {
            scan4k::scan_rows::<4, _>(
                data,
                size,
                x.from as usize,
                x.to as usize,
                &mut reader,
                &mut parser,
            );
        }
        parser.flush_all();
    }
    w.reader = reader;
    w.pbuf = pbuf;
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn mix42_exact_bijection_and_invertibility() {
        const MASK42: u64 = (1u64 << 42) - 1;
        const MUL: u64 = 0x224B_A937_15B7;
        // Modular inverse of MUL mod 2^42 via Newton/Hensel lifting
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(MUL.wrapping_mul(inv))) & MASK42;
        }
        assert_eq!(MUL.wrapping_mul(inv) & MASK42, 1);

        let unmix42 = |mut y: u64| -> u64 {
            y ^= y >> 21;
            y = y.wrapping_mul(inv) & MASK42;
            y ^= y >> 21;
            y
        };

        // Test boundary & random values across full 42-bit range
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for i in 0..500_000u64 {
            let v = match i {
                0 => 0,
                1 => 1,
                2 => MASK42,
                3..=44 => 1u64 << (i - 3),
                _ => rnd() & MASK42,
            };
            let h = mix42(v);
            assert!(h <= MASK42);
            assert_eq!(unmix42(h), v, "mix42 failed inverse for {v}");
        }

        // Dense uniqueness check over 1..=200_000
        let mut seen = std::collections::HashSet::with_capacity(200_000);
        for v in 1..=200_000u64 {
            assert!(seen.insert(mix42(v)), "collision at {v}");
        }
    }
}

#[inline(always)]
fn fastrange32(h: u32, cap: usize) -> usize {
    ((h as u64 * cap as u64) >> 32) as usize
}

#[inline(always)]
unsafe fn same_raw_event(
    a: RawEvent16,
    b: RawEvent16,
    sides: &[&[SideEvent]],
    e_side_base: *const ids::Hex128,
) -> bool {
    let sa = a.eid_is_side();
    let sb = b.eid_is_side();
    if !sa & !sb {
        return a.rem34() == b.rem34();
    }
    if sa & sb {
        let ta = sides.get_unchecked(a.side_tid()).get_unchecked(a.side_idx()).eid_tok;
        let tb = sides.get_unchecked(b.side_tid()).get_unchecked(b.side_idx()).eid_tok;
        return event_eq(e_side_base, ta, tb);
    }
    false
}

#[inline(always)]
unsafe fn raw_ingest_smaller(a: RawEvent16, b: RawEvent16, sides: &[&[SideEvent]]) -> bool {
    let ia = a.ing_rel21();
    let ib = b.ing_rel21();
    if ia != ib {
        return ia < ib;
    }
    if a.is_side() & b.is_side() {
        let sa = sides.get_unchecked(a.side_tid()).get_unchecked(a.side_idx()).ingest_s;
        let sb = sides.get_unchecked(b.side_tid()).get_unchecked(b.side_idx()).ingest_s;
        return sa < sb;
    }
    false
}

struct DedupWorker {
    bits_pair: Vec<[u64; 2]>,
    ht8: Vec<u64>,
    cand_raw: Vec<RawEvent16>,
    cand_long: Vec<(u64, LongEv32)>,
    short_win: Vec<RawEvent16>,
    long_win: Vec<LongEv32>,
    p_writer: StoreWriter<CompactPersonEv8, 512, 256, 8>,
    pu_writer: StoreWriter<CompactPurchase, 256, 256, 4>,
    alloc: LocalBlockAlloc,
}

impl DedupWorker {
    fn new() -> Self {
        Self {
            bits_pair: Vec::new(),
            ht8: Vec::new(),
            cand_raw: Vec::new(),
            cand_long: Vec::new(),
            short_win: Vec::new(),
            long_win: Vec::new(),
            p_writer: StoreWriter::new(),
            pu_writer: StoreWriter::new(),
            alloc: LocalBlockAlloc::new(),
        }
    }

    #[inline(always)]
    unsafe fn emit_raw(
        &mut self,
        w: RawEvent16,
        sides: &[&[SideEvent]],
        pool: &BlockSlabPool,
    ) {
        let flags = w.w0 >> 62;
        if flags == 1 {
            let pid = w.inline_pid();
            let ts = w.inline_ts();
            self.p_writer.push(
                (pid & 255) as usize,
                CompactPersonEv8 {
                    pid_hi: pid >> 8,
                    ts_rel_ms: ts,
                },
                &mut self.alloc,
                pool,
            );
        } else if flags == 3 {
            self.emit_raw_side(w, sides, pool);
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn emit_raw_side(
        &mut self,
        w: RawEvent16,
        sides: &[&[SideEvent]],
        pool: &BlockSlabPool,
    ) {
        let se = *sides.get_unchecked(w.side_tid()).get_unchecked(w.side_idx());
        if se.valid != 0 {
            self.p_writer.push(
                (se.pid & 255) as usize,
                CompactPersonEv8 {
                    pid_hi: se.pid >> 8,
                    ts_rel_ms: se.ts_rel_ms,
                },
                &mut self.alloc,
                pool,
            );
            if se.price > 0 {
                self.pu_writer.push(
                    (se.pid & 255) as usize,
                    CompactPurchase {
                        pid_hi: se.pid >> 8,
                        ts_rel_ms: se.ts_rel_ms,
                        price: se.price,
                        cat_id: se.cat_id as u32,
                    },
                    &mut self.alloc,
                    pool,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn emit_long(&mut self, w: LongEv32, pool: &BlockSlabPool) {
        if !w.is_valid() {
            return;
        }
        let pid = w.pid();
        let ts = w.ts_rel_ms();
        self.p_writer.push(
            (pid & 255) as usize,
            CompactPersonEv8 {
                pid_hi: pid >> 8,
                ts_rel_ms: ts,
            },
            &mut self.alloc,
            pool,
        );
        let price = w.price();
        if price > 0 {
            self.pu_writer.push(
                (pid & 255) as usize,
                CompactPurchase {
                    pid_hi: pid >> 8,
                    ts_rel_ms: ts,
                    price,
                    cat_id: w.cat_id(),
                },
                &mut self.alloc,
                pool,
            );
        }
    }

    unsafe fn dedup_short_bucket(
        &mut self,
        b: usize,
        threads: usize,
        raw_store: &PartitionedStore<RawEvent16, 256, RAW_B>,
        sides: &[&[SideEvent]],
        e_side_base: *const ids::Hex128,
        pool: &BlockSlabPool,
    ) {
        const AHEAD: usize = 8;
        let nb = raw_store.bucket_total(b);
        if nb == 0 {
            return;
        }
        let bits = (nb * 8).next_power_of_two().max(64);
        let mask = bits - 1;
        self.bits_pair.clear();
        self.bits_pair.resize(bits / 64, [0; 2]);
        let bp_ptr = self.bits_pair.as_mut_ptr();

        // Pass 1: 2-bit interleaved bloom filter to isolate ~14% potential duplicates.
        for t in 0..threads {
            let mut blk = raw_store.thread_head(b, t);
            while !blk.is_null() {
                let cnt = (*blk).count as usize;
                let items = (*blk).items.as_ptr();
                let pre = cnt.min(AHEAD);
                for j in 0..pre {
                    let bp = ((*items.add(j)).rem34() as usize) & mask;
                    _mm_prefetch(bp_ptr.add(bp >> 6) as *const i8, _MM_HINT_T0);
                }
                for j in 0..cnt {
                    if j + AHEAD < cnt {
                        let bp = ((*items.add(j + AHEAD)).rem34() as usize) & mask;
                        _mm_prefetch(bp_ptr.add(bp >> 6) as *const i8, _MM_HINT_T0);
                    }
                    let bp = ((*items.add(j)).rem34() as usize) & mask;
                    let pair = &mut *bp_ptr.add(bp >> 6);
                    let bit = 1u64 << (bp & 63);
                    pair[1] |= pair[0] & bit;
                    pair[0] |= bit;
                }
                blk = (*blk).next;
            }
        }

        // Pass 2: emit unique events directly (~86%), collect candidates (~14%), recycle blocks.
        self.cand_raw.clear();
        self.cand_raw.reserve((nb / 4).max(64));
        for t in 0..threads {
            let head = raw_store.thread_head(b, t);
            let mut blk = head;
            while !blk.is_null() {
                let cnt = (*blk).count as usize;
                let items = (*blk).items.as_ptr();
                let pre = cnt.min(AHEAD);
                for j in 0..pre {
                    let bp = ((*items.add(j)).rem34() as usize) & mask;
                    _mm_prefetch(bp_ptr.add(bp >> 6) as *const i8, _MM_HINT_T0);
                }
                for j in 0..cnt {
                    if j + AHEAD < cnt {
                        let bp = ((*items.add(j + AHEAD)).rem34() as usize) & mask;
                        _mm_prefetch(bp_ptr.add(bp >> 6) as *const i8, _MM_HINT_T0);
                    }
                    let r = *items.add(j);
                    let bp = (r.rem34() as usize) & mask;
                    let is_cand = (((*bp_ptr.add(bp >> 6))[1] >> (bp & 63)) & 1) != 0;
                    if !is_cand {
                        self.emit_raw(r, sides, pool);
                    } else {
                        self.cand_raw.push(r);
                    }
                }
                blk = (*blk).next;
            }
            self.alloc.recycle(head);
        }

        let nc = self.cand_raw.len();
        if nc == 0 {
            return;
        }
        let hcap = nc + (nc >> 1) + 64;
        self.ht8.clear();
        self.ht8.resize(hcap, 0);
        self.short_win.clear();
        self.short_win.reserve(nc);

        let ht8 = self.ht8.as_mut_ptr();
        let cand_ptr = self.cand_raw.as_ptr();
        let pre = nc.min(AHEAD);
        for i in 0..pre {
            let s = fastrange32((*cand_ptr.add(i)).rem34() as u32, hcap);
            _mm_prefetch(ht8.add(s) as *const i8, _MM_HINT_T0);
        }
        for i in 0..nc {
            if i + AHEAD < nc {
                let s = fastrange32((*cand_ptr.add(i + AHEAD)).rem34() as u32, hcap);
                _mm_prefetch(ht8.add(s) as *const i8, _MM_HINT_T0);
            }
            let r = *cand_ptr.add(i);
            let key35 = ((r.eid_is_side() as u64) << 34) | r.rem34();
            let mut slot = fastrange32(r.rem34() as u32, hcap);
            loop {
                let entry = *ht8.add(slot);
                if entry == 0 {
                    let idx = self.short_win.len();
                    self.short_win.push(r);
                    *ht8.add(slot) = (key35 << 24) | ((idx + 1) as u64);
                    break;
                }
                if (entry >> 24) == key35 {
                    let cur = self.short_win.get_unchecked_mut((entry as usize & 0xFF_FFFF) - 1);
                    if !r.eid_is_side() || same_raw_event(r, *cur, sides, e_side_base) {
                        if raw_ingest_smaller(r, *cur, sides) {
                            *cur = r;
                        }
                        break;
                    }
                }
                slot += 1;
                if slot == hcap {
                    slot = 0;
                }
            }
        }

        let nwin = self.short_win.len();
        for i in 0..nwin {
            let w = *self.short_win.get_unchecked(i);
            self.emit_raw(w, sides, pool);
        }
    }

    unsafe fn dedup_long_bucket(
        &mut self,
        b: usize,
        threads: usize,
        long_store: &PartitionedStore<LongEv32, 128, 128>,
        pool: &BlockSlabPool,
    ) {
        const AHEAD: usize = 8;
        let nb = long_store.bucket_total(b);
        if nb == 0 {
            return;
        }
        let bits = (nb * 8).next_power_of_two().max(64);
        let mask = bits - 1;
        self.bits_pair.clear();
        self.bits_pair.resize(bits / 64, [0; 2]);
        let bp_ptr = self.bits_pair.as_mut_ptr();

        #[inline(always)]
        fn fast_h_long(k_hi: u64, k_lo: u64) -> u64 {
            k_lo ^ k_hi.rotate_left(23)
        }

        for t in 0..threads {
            let mut blk = long_store.thread_head(b, t);
            while !blk.is_null() {
                let cnt = (*blk).count as usize;
                let items = (*blk).items.as_ptr();
                let pre = cnt.min(AHEAD);
                for j in 0..pre {
                    let r = &*items.add(j);
                    let bp = (fast_h_long(r.k_hi, r.k_lo) as usize) & mask;
                    _mm_prefetch(bp_ptr.add(bp >> 6) as *const i8, _MM_HINT_T0);
                }
                for j in 0..cnt {
                    if j + AHEAD < cnt {
                        let nr = &*items.add(j + AHEAD);
                        let bp = (fast_h_long(nr.k_hi, nr.k_lo) as usize) & mask;
                        _mm_prefetch(bp_ptr.add(bp >> 6) as *const i8, _MM_HINT_T0);
                    }
                    let r = &*items.add(j);
                    let h = fast_h_long(r.k_hi, r.k_lo);
                    let bp = (h as usize) & mask;
                    let pair = &mut *bp_ptr.add(bp >> 6);
                    let bit = 1u64 << (bp & 63);
                    pair[1] |= pair[0] & bit;
                    pair[0] |= bit;
                }
                blk = (*blk).next;
            }
        }

        self.cand_long.clear();
        self.cand_long.reserve((nb / 4).max(64));
        for t in 0..threads {
            let head = long_store.thread_head(b, t);
            let mut blk = head;
            while !blk.is_null() {
                let cnt = (*blk).count as usize;
                let items = (*blk).items.as_ptr();
                let pre = cnt.min(AHEAD);
                for j in 0..pre {
                    let r = &*items.add(j);
                    let bp = (fast_h_long(r.k_hi, r.k_lo) as usize) & mask;
                    _mm_prefetch(bp_ptr.add(bp >> 6) as *const i8, _MM_HINT_T0);
                }
                for j in 0..cnt {
                    if j + AHEAD < cnt {
                        let nr = &*items.add(j + AHEAD);
                        let bp = (fast_h_long(nr.k_hi, nr.k_lo) as usize) & mask;
                        _mm_prefetch(bp_ptr.add(bp >> 6) as *const i8, _MM_HINT_T0);
                    }
                    let r = *items.add(j);
                    let h = fast_h_long(r.k_hi, r.k_lo);
                    let bp = (h as usize) & mask;
                    let is_cand = (((*bp_ptr.add(bp >> 6))[1] >> (bp & 63)) & 1) != 0;
                    if !is_cand {
                        self.emit_long(r, pool);
                    } else {
                        self.cand_long.push((h, r));
                    }
                }
                blk = (*blk).next;
            }
            self.alloc.recycle(head);
        }

        let nc = self.cand_long.len();
        if nc == 0 {
            return;
        }
        let hcap = nc + (nc >> 1) + 64;
        self.ht8.clear();
        self.ht8.resize(hcap, 0);
        self.long_win.clear();
        self.long_win.reserve(nc);

        let ht8 = self.ht8.as_mut_ptr();
        let cand_ptr = self.cand_long.as_ptr();
        let pre = nc.min(AHEAD);
        for i in 0..pre {
            let h = (*cand_ptr.add(i)).0;
            let s = fastrange32((h >> 25) as u32, hcap);
            _mm_prefetch(ht8.add(s) as *const i8, _MM_HINT_T0);
        }
        for i in 0..nc {
            if i + AHEAD < nc {
                let h = (*cand_ptr.add(i + AHEAD)).0;
                let s = fastrange32((h >> 25) as u32, hcap);
                _mm_prefetch(ht8.add(s) as *const i8, _MM_HINT_T0);
            }
            let (h, r) = *cand_ptr.add(i);
            let tag35 = (h >> 22) & ((1u64 << 35) - 1);
            let mut slot = fastrange32((h >> 25) as u32, hcap);
            loop {
                let entry = *ht8.add(slot);
                if entry == 0 {
                    let idx = self.long_win.len();
                    self.long_win.push(r);
                    *ht8.add(slot) = (tag35 << 24) | ((idx + 1) as u64);
                    break;
                }
                if (entry >> 24) == tag35 {
                    let cur = self.long_win.get_unchecked_mut((entry as usize & 0xFF_FFFF) - 1);
                    if r.k_lo == cur.k_lo && r.k_hi == cur.k_hi {
                        if r.ing40() < cur.ing40() {
                            *cur = r;
                        }
                        break;
                    }
                }
                slot += 1;
                if slot == hcap {
                    slot = 0;
                }
            }
        }

        let nwin = self.long_win.len();
        for i in 0..nwin {
            let w = *self.long_win.get_unchecked(i);
            self.emit_long(w, pool);
        }
    }
}

struct SessionScratch {
    ev_count: Vec<u32>,
    pu_count: Vec<u32>,
    ck_count: Vec<u32>,
    ev_u32: Vec<u32>,
    purchases: Vec<(u64, u32)>,
    ck_ids: Vec<u64>,
    ck_packed: Vec<V6ClickU64>,
    ck_fb: Vec<V6ClickFb>,
}

impl SessionScratch {
    fn new() -> Self {
        Self {
            ev_count: Vec::new(),
            pu_count: Vec::new(),
            ck_count: Vec::new(),
            ev_u32: Vec::new(),
            purchases: Vec::new(),
            ck_ids: Vec::new(),
            ck_packed: Vec::new(),
            ck_fb: Vec::new(),
        }
    }

    unsafe fn run_bucket(
        &mut self,
        b: usize,
        threads: usize,
        sh: &Shared,
        dict: &UserDict,
        num_hi: usize,
        p_store: &PartitionedStore<CompactPersonEv8, 512, 256>,
        pu_store: &PartitionedStore<CompactPurchase, 256, 256>,
        c_store: &PartitionedStore<CompactClick, 256, 256>,
        g: &mut Grouper,
    ) {
        self.ev_count.clear();
        self.ev_count.resize(num_hi, 0);
        let mut ev_total = 0usize;
        for t in 0..threads {
            let mut blk = p_store.thread_head(b, t);
            while !blk.is_null() {
                let cnt = (*blk).count as usize;
                let items = (*blk).items.as_ptr();
                for j in 0..cnt {
                    let phi = (*items.add(j)).pid_hi as usize;
                    *self.ev_count.get_unchecked_mut(phi) += 1;
                }
                ev_total += cnt;
                blk = (*blk).next;
            }
        }
        if ev_total == 0 {
            return;
        }

        self.pu_count.clear();
        self.pu_count.resize(num_hi, 0);
        let mut pu_total = 0usize;
        for t in 0..threads {
            let mut blk = pu_store.thread_head(b, t);
            while !blk.is_null() {
                let cnt = (*blk).count as usize;
                let items = (*blk).items.as_ptr();
                for j in 0..cnt {
                    let phi = (*items.add(j)).pid_hi as usize;
                    *self.pu_count.get_unchecked_mut(phi) += 1;
                }
                pu_total += cnt;
                blk = (*blk).next;
            }
        }

        self.ck_count.clear();
        self.ck_count.resize(num_hi, 0);
        let mut ck_total = 0usize;
        for t in 0..threads {
            let mut blk = c_store.thread_head(b, t);
            while !blk.is_null() {
                let cnt = (*blk).count as usize;
                let items = (*blk).items.as_ptr();
                for j in 0..cnt {
                    let phi = (*items.add(j)).pid_hi();
                    if *self.ev_count.get_unchecked(phi) != 0 {
                        *self.ck_count.get_unchecked_mut(phi) += 1;
                        ck_total += 1;
                    }
                }
                blk = (*blk).next;
            }
        }

        // Prefix sums (arrays turn into end-cursors during Pass 2 scatter).
        let mut sum = 0u32;
        for c in self.ev_count.iter_mut() {
            (*c, sum) = (sum, sum + *c);
        }
        let mut sum = 0u32;
        for c in self.pu_count.iter_mut() {
            (*c, sum) = (sum, sum + *c);
        }
        let mut sum = 0u32;
        for c in self.ck_count.iter_mut() {
            (*c, sum) = (sum, sum + *c);
        }

        // Pass 2 scatter events (+ 8 trailing u32 pad for 1-YMM small sessionizer).
        self.ev_u32.clear();
        self.ev_u32.resize(ev_total + 8, 0);
        let ev_ptr = self.ev_u32.as_mut_ptr();
        for t in 0..threads {
            let mut blk = p_store.thread_head(b, t);
            while !blk.is_null() {
                let cnt = (*blk).count as usize;
                let items = (*blk).items.as_ptr();
                for j in 0..cnt {
                    let ev = *items.add(j);
                    let slot = self.ev_count.get_unchecked_mut(ev.pid_hi as usize);
                    *ev_ptr.add(*slot as usize) = ev.ts_rel_ms;
                    *slot += 1;
                }
                blk = (*blk).next;
            }
        }

        self.purchases.clear();
        self.purchases.resize(pu_total, (0, 0));
        if pu_total > 0 {
            let pu_ptr = self.purchases.as_mut_ptr();
            for t in 0..threads {
                let mut blk = pu_store.thread_head(b, t);
                while !blk.is_null() {
                    let cnt = (*blk).count as usize;
                    let items = (*blk).items.as_ptr();
                    for j in 0..cnt {
                        let pu = *items.add(j);
                        let slot = self.pu_count.get_unchecked_mut(pu.pid_hi as usize);
                        *pu_ptr.add(*slot as usize) =
                            (pu.ts_rel_ms as u64, (pu.cat_id << 24) | pu.price);
                        *slot += 1;
                    }
                    blk = (*blk).next;
                }
            }
        }

        let compact_clicks = ck_total <= 65_535;
        if compact_clicks {
            self.ck_ids.clear();
            self.ck_ids.reserve(ck_total);
            self.ck_packed.clear();
            self.ck_packed.resize(ck_total, V6ClickU64(0));
            if ck_total > 0 {
                let ck_ptr = self.ck_packed.as_mut_ptr();
                let ck_ids_ptr = self.ck_ids.as_mut_ptr();
                let mut n_ck = 0usize;
                for t in 0..threads {
                    let mut blk = c_store.thread_head(b, t);
                    while !blk.is_null() {
                        let cnt = (*blk).count as usize;
                        let items = (*blk).items.as_ptr();
                        for j in 0..cnt {
                            let ck = *items.add(j);
                            let phi = ck.pid_hi();
                            let prev_ev = if phi == 0 { 0 } else { *self.ev_count.get_unchecked(phi - 1) };
                            if *self.ev_count.get_unchecked(phi) > prev_ev {
                                *ck_ids_ptr.add(n_ck) = ck.click_id;
                                let slot = self.ck_count.get_unchecked_mut(phi);
                                *ck_ptr.add(*slot as usize) = V6ClickU64(
                                    ((ck.ts_rel_ms() as u64) << 32)
                                        | ((ck.camp() as u64) << 16)
                                        | (n_ck as u64),
                                );
                                *slot += 1;
                                n_ck += 1;
                            }
                        }
                        blk = (*blk).next;
                    }
                }
                self.ck_ids.set_len(n_ck);
            }
        } else {
            self.ck_fb.clear();
            self.ck_fb.resize(
                ck_total,
                V6ClickFb {
                    key: 0,
                    campaign: 0,
                    click_id: 0,
                },
            );
            let ck_ptr = self.ck_fb.as_mut_ptr();
            for t in 0..threads {
                let mut blk = c_store.thread_head(b, t);
                while !blk.is_null() {
                    let cnt = (*blk).count as usize;
                    let items = (*blk).items.as_ptr();
                    for j in 0..cnt {
                        let ck = *items.add(j);
                        let phi = ck.pid_hi();
                        let prev_ev = if phi == 0 { 0 } else { *self.ev_count.get_unchecked(phi - 1) };
                        if *self.ev_count.get_unchecked(phi) > prev_ev {
                            let slot = self.ck_count.get_unchecked_mut(phi);
                            *ck_ptr.add(*slot as usize) = V6ClickFb {
                                key: ((ck.ts_rel_ms() as u64) << 32) | (phi as u64),
                                campaign: ck.camp(),
                                click_id: ck.click_id,
                            };
                            *slot += 1;
                        }
                    }
                    blk = (*blk).next;
                }
            }
        }

        let (mut ev_from, mut pu_from, mut ck_from) = (0usize, 0usize, 0usize);
        for phi in 0..num_hi {
            let ev_end = *self.ev_count.get_unchecked(phi) as usize;
            let pu_end = *self.pu_count.get_unchecked(phi) as usize;
            let ck_end = *self.ck_count.get_unchecked(phi) as usize;
            if ev_end > ev_from {
                let ev = &mut self.ev_u32[ev_from..ev_end];
                g.sort_ev(ev);
                let pu = &mut self.purchases[pu_from..pu_end];
                if pu.len() > 1 {
                    pu.sort_unstable_by_key(|x| x.0);
                }
                let pid = ((phi as u32) << 8) | (b as u32);
                let person_tok = dict.pid_tok(pid);
                if compact_clicks {
                    let ck_raw = std::slice::from_raw_parts_mut(
                        self.ck_packed.as_mut_ptr().add(ck_from) as *mut u64,
                        ck_end - ck_from,
                    );
                    sort_u64_clicks(ck_raw);
                    let ck = &self.ck_packed[ck_from..ck_end];
                    g.run_person_v6(sh, person_tok, ev, pu, ck, &self.ck_ids);
                } else {
                    let ck = &mut self.ck_fb[ck_from..ck_end];
                    ck.sort_unstable_by_key(|c| c.key >> 32);
                    g.run_person_v6_fallback(sh, person_tok, ev, pu, ck);
                }
            }
            (ev_from, pu_from, ck_from) = (ev_end, pu_end, ck_end);
        }
    }
}

pub struct V6Metrics {
    pub event_lines: u64,
    pub valid_events: u64,
    pub clicks: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct LinkPre {
    p0: *const u8,
    p1: *const u8,
    a0: u64,
    b0: u64,
    a1: u64,
    b1: u64,
    s0: u32,
    s1: u32,
    l0: u16,
    l1: u16,
    k0: u8,
    k1: u8,
}

impl Default for LinkPre {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[inline(never)]
unsafe fn flush_link_batch(
    dict: &UserDict,
    lease: &mut PidLease,
    tid: usize,
    batch: &[LinkPre],
    pairs: &mut Vec<(u32, u32)>,
) {
    for pre in batch {
        if dict.is_overflow() {
            return;
        }
        let ua = resolve_user_parts(
            dict,
            pre.k0,
            pre.s0 as usize,
            pre.a0,
            pre.b0,
            pre.p0,
            pre.l0 as usize,
            lease,
            tid,
        );
        let ub = resolve_user_parts(
            dict,
            pre.k1,
            pre.s1 as usize,
            pre.a1,
            pre.b1,
            pre.p1,
            pre.l1 as usize,
            lease,
            tid,
        );
        if ua == 0 || ub == 0 {
            return;
        }
        if ua != ub {
            pairs.push((ua, ub));
        }
    }
}

struct LinkRowParser<'a> {
    dict: &'a UserDict,
    lease: &'a mut PidLease,
    tid: usize,
    pairs: &'a mut Vec<(u32, u32)>,
    batch: [LinkPre; BATCH],
    nb: usize,
}

impl<'a> LinkRowParser<'a> {
    #[inline(always)]
    unsafe fn flush_all(&mut self) {
        if self.nb > 0 {
            flush_link_batch(
                self.dict,
                self.lease,
                self.tid,
                &self.batch[..self.nb],
                self.pairs,
            );
            self.nb = 0;
        }
    }
}

impl<'a> RowHandler for LinkRowParser<'a> {
    #[inline(always)]
    unsafe fn before_refill(&mut self) {
        self.flush_all();
    }

    #[inline(always)]
    unsafe fn on_row<R: CsvRow>(&mut self, row: R) {
        let (p0, l0) = row.field(0);
        let (p1, l1) = row.field(1);
        if l0 == 0 || l1 == 0 {
            return;
        }
        let (k0, s0, a0, b0, k1, s1, a1, b1) =
            if let Some((ka, kb)) = pack_pair_base37::<b'u', b'u'>(p0, l0, p1, l1) {
                (
                    1u8,
                    self.dict.prefetch_u(ka),
                    ka,
                    0u64,
                    1u8,
                    self.dict.prefetch_u(kb),
                    kb,
                    0u64,
                )
            } else {
                let (k0, s0, a0, b0) = prep_user_prefetch(self.dict, p0, l0);
                let (k1, s1, a1, b1) = prep_user_prefetch(self.dict, p1, l1);
                (k0, s0, a0, b0, k1, s1, a1, b1)
            };
        let pre = self.batch.get_unchecked_mut(self.nb);
        pre.p0 = p0;
        pre.l0 = l0 as u16;
        pre.k0 = k0;
        pre.s0 = s0 as u32;
        pre.a0 = a0;
        pre.b0 = b0;
        pre.p1 = p1;
        pre.l1 = l1 as u16;
        pre.k1 = k1;
        pre.s1 = s1 as u32;
        pre.a1 = a1;
        pre.b1 = b1;
        self.nb += 1;
        if self.nb == BATCH {
            flush_link_batch(self.dict, self.lease, self.tid, &self.batch, self.pairs);
            self.nb = 0;
        }
    }
}

/// Executes V6 fast path directly from `link_input`, `event_input`, and `click_input`.
/// Returns `Some(metrics)` on success, or `None` if `UserDict` / `SideEvent` overflowed
/// (after releasing V6 allocations for V5 fallback).
pub fn run(
    sh: &Shared,
    link_input: &mut Input,
    event_input: &mut Input,
    click_input: &mut Input,
    threads: usize,
    read_buf: usize,
    user_daily_dir: &Path,
    campaign_daily_dir: &Path,
    verbose: bool,
    log_mark: &mut dyn FnMut(&'static str),
) -> Option<V6Metrics> {
    for i in 0..link_input.files.len() {
        link_input.end_line(i);
    }

    // Guardrail C5: all mmap()s (UserDict + BlockSlabPool) completed BEFORE any MADV_POPULATE_WRITE.
    let dict = UserDict::new_with_links(event_input.bytes as usize, link_input.bytes as usize);
    let est_ev = (event_input.bytes as usize / 72).max(1);
    let est_ck = (click_input.bytes as usize / 45).max(1);
    let init_pool_bytes = est_ev * 18 + est_ck * 16;
    let (pool, pool_reg) = BlockSlabPool::with_initial(init_pool_bytes);

    let pop_pieces: Vec<(usize, usize)> = {
        const PIECE: usize = 4 << 20;
        let dict_regs = dict.populate_regions();
        let all_regs = [dict_regs[0], dict_regs[1], dict_regs[2], pool_reg];
        let max_len = all_regs.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let mut pieces =
            Vec::with_capacity(all_regs.iter().map(|&(_, n)| n.div_ceil(PIECE)).sum());
        for at in (0..max_len).step_by(PIECE) {
            for &(b, n) in &all_regs {
                if at < n {
                    pieces.push((b + at, PIECE.min(n - at)));
                }
            }
        }
        pieces
    };
    let pop_next = AtomicUsize::new(0);
    let overlap_pop = std::env::var("SS_OVERLAP_POP").ok().as_deref() != Some("0");

    let pread = scan4k::use_pread();
    let parse_workers = PerThread::new(threads, |_| ParseWorker::new(read_buf, pread));
    let per_thread_pairs = PerThread::new(threads, |_| Vec::<(u32, u32)>::new());

    // Phase 0: check event/click trailing newlines, parse links directly into UserDict,
    // and opportunistically drain `pop_pieces` on idle threads.
    let tails = event_input.files.len() + click_input.files.len();
    let next_tail = AtomicUsize::new(0);
    let unended: Vec<AtomicBool> = (0..tails).map(|_| AtomicBool::new(false)).collect();
    let next_link = AtomicUsize::new(0);
    let active0 = AtomicUsize::new(threads);
    par(threads, |t| unsafe {
        loop {
            let i = next_tail.fetch_add(1, Relaxed);
            if i >= tails {
                break;
            }
            let lacks = match i.checked_sub(event_input.files.len()) {
                None => event_input.lacks_newline(i),
                Some(k) => click_input.lacks_newline(k),
            };
            unended[i].store(lacks, Relaxed);
        }

        let w = parse_workers.get(t);
        let pairs = per_thread_pairs.get(t);
        let mut reader = std::mem::replace(&mut w.reader, Reader::lazy(0));
        let mut pbuf = w.pbuf.take();
        {
            let mut parser = LinkRowParser {
                dict: &dict,
                lease: &mut w.lease,
                tid: t,
                pairs,
                batch: [LinkPre::default(); BATCH],
                nb: 0,
            };
            loop {
                if dict.is_overflow() {
                    break;
                }
                let i = next_link.fetch_add(1, Relaxed);
                let Some(x) = link_input.extents.get(i) else { break };
                let Mapped(data, size) = link_input.files[x.file];
                if let Some(fd) = link_input.file_fd(x.file) {
                    let pb = pbuf.get_or_insert_with(PreadBuf::new);
                    scan4k::scan_rows_pread::<2, _>(
                        fd,
                        size,
                        x.from as usize,
                        x.to as usize,
                        &mut reader,
                        pb,
                        &mut parser,
                    );
                    parser.flush_all();
                } else {
                    scan4k::scan_rows::<2, _>(
                        data,
                        size,
                        x.from as usize,
                        x.to as usize,
                        &mut reader,
                        &mut parser,
                    );
                }
                link_input.done(x);
            }
            parser.flush_all();
        }
        w.reader = reader;
        w.pbuf = pbuf;
        w.lease = PidLease::default();

        active0.fetch_sub(1, AcqRel);
        if overlap_pop {
            while active0.load(Acquire) > 0 {
                let i = pop_next.fetch_add(1, Relaxed);
                let Some(&(at, n)) = pop_pieces.get(i) else { break };
                sys::populate(at as *mut u8, n);
            }
        }
    });

    for (i, u) in unended.iter().enumerate() {
        if u.load(Relaxed) {
            match i.checked_sub(event_input.files.len()) {
                None => event_input.end_line(i),
                Some(k) => click_input.end_line(k),
            }
        }
    }

    if dict.is_overflow() {
        pool.release();
        dict.release();
        return None;
    }
    log_mark("links parse");

    unsafe {
        let pair_slices: Vec<&[(u32, u32)]> = (0..threads)
            .map(|t| per_thread_pairs.get(t).as_slice())
            .collect();
        dict.resolve_links_dsu(&pair_slices, threads);
        for t in 0..threads {
            *per_thread_pairs.get(t) = Vec::new();
        }
    }
    log_mark("links join");

    if pop_next.load(Relaxed) < pop_pieces.len() {
        par(threads, |_| loop {
            let i = pop_next.fetch_add(1, Relaxed);
            let Some(&(at, n)) = pop_pieces.get(i) else { break };
            sys::populate(at as *mut u8, n);
        });
    }
    log_mark("populate");

    let raw_store: PartitionedStore<RawEvent16, 256, RAW_B> = PartitionedStore::new(threads);
    let long_store: PartitionedStore<LongEv32, 128, 128> = PartitionedStore::new(threads);
    let c_store: PartitionedStore<CompactClick, 256, 256> = PartitionedStore::new(threads);
    let p_store: PartitionedStore<CompactPersonEv8, 512, 256> = PartitionedStore::new(threads);
    let pu_store: PartitionedStore<CompactPurchase, 256, 256> = PartitionedStore::new(threads);

    let side_overflow = AtomicBool::new(false);

    let mut work: Vec<(bool, usize)> = (0..event_input.extents.len())
        .map(|i| (true, i))
        .chain((0..click_input.extents.len()).map(|i| (false, i)))
        .collect();
    let extent_len = |&(is_event, i): &(bool, usize)| {
        let x = if is_event {
            &event_input.extents[i]
        } else {
            &click_input.extents[i]
        };
        x.to - x.from
    };
    work.sort_by_key(|w| std::cmp::Reverse(extent_len(w)));

    let next = AtomicUsize::new(0);
    par(threads, |t| unsafe {
        let w = parse_workers.get(t);
        loop {
            if dict.is_overflow() || side_overflow.load(Relaxed) {
                break;
            }
            let i = next.fetch_add(1, Relaxed);
            let Some(&(is_event, idx)) = work.get(i) else { break };
            if is_event {
                let x = &event_input.extents[idx];
                parse_events_v6(
                    sh,
                    &dict,
                    event_input.files[x.file],
                    event_input.file_fd(x.file),
                    x,
                    w,
                    &pool,
                    t,
                    &side_overflow,
                );
                event_input.done(x);
            } else {
                let x = &click_input.extents[idx];
                parse_clicks_v6(
                    sh,
                    &dict,
                    click_input.files[x.file],
                    click_input.file_fd(x.file),
                    x,
                    w,
                    &pool,
                    t,
                );
                click_input.done(x);
            }
        }
        w.raw_writer.finish(&raw_store, t, &mut w.alloc, &pool);
        w.long_writer.finish(&long_store, t, &mut w.alloc, &pool);
        w.c_writer.finish(&c_store, t, &mut w.alloc, &pool);
    });

    if dict.is_overflow() || side_overflow.load(Relaxed) {
        pool.release();
        dict.release();
        return None;
    }

    let release_mem = std::env::var_os("SS_NORELEASE").is_none()
        && sys::host_profile().thp == sys::ThpPolicy::NoHuge;
    log_mark("parse");

    let event_lines: u64 = (0..threads)
        .map(|t| unsafe { parse_workers.get(t).total_event_lines })
        .sum();
    let side_slices: Vec<&[SideEvent]> = (0..threads)
        .map(|t| unsafe { parse_workers.get(t).side_events.as_slice() })
        .collect();

    let mut dedup_work: Vec<(bool, usize, usize)> = (0..RAW_B)
        .map(|b| (false, b, raw_store.bucket_total(b)))
        .chain((0..128).map(|b| (true, b, long_store.bucket_total(b))))
        .collect();
    dedup_work.sort_by_key(|x| std::cmp::Reverse(x.2));

    let dedup_workers = PerThread::new(threads, |_| DedupWorker::new());
    let next = AtomicUsize::new(0);
    par(threads, |t| unsafe {
        let dw = dedup_workers.get(t);
        loop {
            let i = next.fetch_add(1, Relaxed);
            let Some(&(is_long, b, _)) = dedup_work.get(i) else { break };
            if is_long {
                dw.dedup_long_bucket(b, threads, &long_store, &pool);
            } else {
                dw.dedup_short_bucket(
                    b,
                    threads,
                    &raw_store,
                    &side_slices,
                    sh.e_side_arena.base,
                    &pool,
                );
            }
        }
        dw.p_writer.finish(&p_store, t, &mut dw.alloc, &pool);
        dw.pu_writer.finish(&pu_store, t, &mut dw.alloc, &pool);
    });
    log_mark("dedup");

    let valid_events = p_store.total_items() as u64;
    let clicks = c_store.total_items() as u64;

    let mut p_order: Vec<usize> = (0..256).collect();
    p_order.sort_by_key(|&b| std::cmp::Reverse(p_store.bucket_total(b) + c_store.bucket_total(b)));

    let num_hi = (dict.max_pid() as usize >> 8) + 1;
    let per_thread_cells = PerThread::new(threads, |_| Vec::<Cell>::new());
    let next = AtomicUsize::new(0);
    par(threads, |t| unsafe {
        let mut g = Grouper::new(
            user_daily_dir.join(format!("part-{t:03}.parquet")),
            (sh.campaigns.len() + 1) * sh.days,
            &sh.categories,
        );
        g.out.set_v6_hlines(dict.hlines);
        g.ensure_covered(sh.campaigns.len());
        let mut scratch = SessionScratch::new();
        loop {
            let i = next.fetch_add(1, Relaxed);
            let Some(&b) = p_order.get(i) else { break };
            scratch.run_bucket(
                b,
                threads,
                sh,
                &dict,
                num_hi,
                &p_store,
                &pu_store,
                &c_store,
                &mut g,
            );
        }
        *per_thread_cells.get(t) = g.finish(t == 0);
    });
    log_mark("sessions");

    let cells: Vec<Vec<Cell>> = (0..threads)
        .map(|t| std::mem::take(unsafe { per_thread_cells.get(t) }))
        .collect();
    crate::write_campaigns(sh, &campaign_daily_dir.join("part-000.parquet"), &cells);
    log_mark("campaigns");

    if verbose {
        eprintln!(
            "v6 stats: threads {threads}  max_pid {}  event_lines {event_lines}  valid {valid_events}  clicks {clicks}  peak rss {} MB",
            dict.max_pid(),
            sys::peak_rss_kb() / 1024
        );
    }

    if release_mem {
        pool.release();
        dict.release();
    }
    Some(V6Metrics {
        event_lines,
        valid_events,
        clicks,
    })
}
