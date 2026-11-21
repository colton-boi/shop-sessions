//! L2-Resident 32K-Slot Direct-Index Event Dedup + In-Cache Vectorized Sessionizer
//! (`process_block_32k`) and Stage 3B Cross-Block Boundary Stitcher.

#![allow(dead_code)]

use crate::fast_links::{ClickCsr, ClickSlot, ALIAS_MASK, HAS_EXTRA_EVENTS, IS_COMPLEX};
#[cfg(test)]
use crate::fast_links::min_token_for_mask;
use crate::group::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Each event block covers `32,768` consecutive `event_index` values (`512 KiB` of 16-byte slots).
pub const BLOCK_SHIFT: usize = 15;
pub const BLOCK_EVENTS: usize = 1 << BLOCK_SHIFT;
pub const BLOCK_MASK: usize = BLOCK_EVENTS - 1;

/// Session inactivity threshold: strictly greater than 30 minutes (`1,800,000` ms) starts a new session.
pub const SESSION_GAP_MS: u32 = 1_800_000;
/// 7-day click attribution reach (`604,800,000` ms).
pub const WEEK_MS: u32 = 604_800_000;
/// 1 day in milliseconds (`86,400,000 = 1024 * 84,375`).
pub const DAY_MS_U32: u32 = 86_400_000;
/// Magic multiplier for exact division by `84,375`:
/// `ceil(2^39 / 84_375) = 6_515_625`, so `(y * 6_515_625) >> 39 == y / 84_375` for `0 <= day <= 143`.
pub const DAY_MAGIC_MUL: u64 = 6_515_625;
pub const DAY_MAGIC_SHIFT: u32 = 39;

/// Sentinel `w0` for an invalid event row that still has a valid `event_id` and participates in dedup.
pub const INVALID_W0: u64 = u64::MAX;
/// Bit 63 in `FastEvRec::w1` marks a valid canonical-`eid` event whose `user_id` is odd/complex
/// (chain user `alias >= 14`, kiosk `>= KIOSK_BASE`, or uppercase case variant).
/// Bits `37..63` of `w1` then store the 26-bit `odd_handle` (`(tid << 20) | seq`).
pub const ODD_USER_BIT: u64 = 1u64 << 63;

/// 16-byte packed event record for 32K-block direct-index dedup and sessionization:
/// - `w0`: `(rel_ts_ms << 33) | (price_cents << 9) | (cat << 4) | alias` (or `u64::MAX` if invalid)
/// - `w1`: `[bit 63: ODD_USER_BIT] | ((user_idx_or_handle & 0x3FF_FFFF) << 37) | (ingest_rel_sec << 16) | slot_with_gen`
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct FastEvRec {
    pub w0: u64,
    pub w1: u64,
}

impl FastEvRec {
    #[inline(always)]
    pub const fn new_valid(
        rel_ts_ms: u32,
        price_cents: u32,
        cat: u8,
        alias: u8,
        user_idx: u32,
        ingest_rel_sec: u32,
        slot: u16,
    ) -> Self {
        debug_assert!(price_cents <= 10_000_000);
        debug_assert!(cat < 32);
        debug_assert!(alias < 16);
        let ing = if ingest_rel_sec > 0x1F_FFFF { 0x1F_FFFF } else { ingest_rel_sec };
        let w0 = ((rel_ts_ms as u64) << 33)
            | (((price_cents as u64) & 0xFF_FFFF) << 9)
            | (((cat as u64) & 0x1F) << 4)
            | ((alias as u64) & 0xF);
        let w1 = (((user_idx as u64) & 0x3FF_FFFF) << 37)
            | ((ing as u64) << 16)
            | ((slot as u64) & 0x7FFF);
        Self { w0, w1 }
    }

    #[inline(always)]
    pub const fn new_odd_user(
        rel_ts_ms: u32,
        price_cents: u32,
        cat: u8,
        odd_handle: u32,
        ingest_rel_sec: u32,
        slot: u16,
    ) -> Self {
        let ing = if ingest_rel_sec > 0x1F_FFFF { 0x1F_FFFF } else { ingest_rel_sec };
        let w0 = ((rel_ts_ms as u64) << 33)
            | (((price_cents as u64) & 0xFF_FFFF) << 9)
            | (((cat as u64) & 0x1F) << 4);
        let w1 = ODD_USER_BIT
            | (((odd_handle as u64) & 0x3FF_FFFF) << 37)
            | ((ing as u64) << 16)
            | ((slot as u64) & 0x7FFF);
        Self { w0, w1 }
    }

    #[inline(always)]
    pub const fn new_invalid(user_idx: u32, ingest_rel_sec: u32, slot: u16) -> Self {
        let ing = if ingest_rel_sec > 0x1F_FFFF { 0x1F_FFFF } else { ingest_rel_sec };
        let w1 = (((user_idx as u64) & 0x3FF_FFFF) << 37)
            | ((ing as u64) << 16)
            | ((slot as u64) & 0x7FFF);
        Self { w0: INVALID_W0, w1 }
    }

    #[inline(always)]
    pub const fn is_valid(&self) -> bool {
        self.w0 != INVALID_W0
    }

    #[inline(always)]
    pub const fn is_odd_user(&self) -> bool {
        (self.w1 & ODD_USER_BIT) != 0
    }

    #[inline(always)]
    pub const fn odd_handle(&self) -> u32 {
        ((self.w1 >> 37) & 0x3FF_FFFF) as u32
    }

    #[inline(always)]
    pub const fn rel_ts_ms(&self) -> u32 {
        (self.w0 >> 33) as u32
    }

    #[inline(always)]
    pub const fn price_cents(&self) -> u32 {
        ((self.w0 >> 9) & 0xFF_FFFF) as u32
    }

    #[inline(always)]
    pub const fn cat(&self) -> u8 {
        ((self.w0 >> 4) & 0x1F) as u8
    }

    #[inline(always)]
    pub const fn alias(&self) -> u8 {
        (self.w0 & 0xF) as u8
    }

    #[inline(always)]
    pub const fn user_idx(&self) -> u32 {
        ((self.w1 >> 37) & 0x3FF_FFFF) as u32
    }

    #[inline(always)]
    pub const fn ingest_rel_sec(&self) -> u32 {
        ((self.w1 >> 16) & 0x1F_FFFF) as u32
    }

    #[inline(always)]
    pub const fn slot(&self) -> usize {
        (self.w1 & 0x7FFF) as usize
    }
}

/// Computes the 0-based window day from a timestamp `rel_ts_ms` and `ws_day_offset`
/// (`rel_ts_ms + ws_day_offset` = milliseconds since `window_start`) via exact magic multiplication:
/// `(((sts + ws_day_offset) >> 10) * 6_516_709) >> 39`.
#[inline(always)]
pub const fn fast_session_day(rel_ts_ms: u32, ws_day_offset: i64) -> u32 {
    let ms_from_ws = (rel_ts_ms as i64 + ws_day_offset) as u64;
    let y = ms_from_ws >> 10;
    ((y.wrapping_mul(DAY_MAGIC_MUL)) >> DAY_MAGIC_SHIFT) as u32
}

/// Identified run of contiguous events for a single `user_idx` inside a 32K block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UserRun {
    pub user_idx: u32,
    pub ev_start: u16,
    pub ev_end: u16,
    pub sess_start: u32,
    pub sess_end: u32,
    pub alias_mask: u16,
    pub unsorted: bool,
}

/// Compact event stored in a `BoundarySeg` for cross-block stitching.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundaryEv {
    pub rel_ts_ms: u32,
    pub price_cents: u32,
    pub cat: u8,
    pub alias: u8,
}

impl BoundaryEv {
    #[inline(always)]
    pub const fn from_w0(w0: u64) -> Self {
        Self {
            rel_ts_ms: (w0 >> 33) as u32,
            price_cents: ((w0 >> 9) & 0xFF_FFFF) as u32,
            cat: ((w0 >> 4) & 0x1F) as u8,
            alias: (w0 & 0xF) as u8,
        }
    }

    #[inline(always)]
    pub const fn to_w0(self) -> u64 {
        ((self.rel_ts_ms as u64) << 33)
            | (((self.price_cents as u64) & 0xFF_FFFF) << 9)
            | (((self.cat as u64) & 0x1F) << 4)
            | ((self.alias as u64) & 0xF)
    }
}

/// Boundary user segment emitted at the head (`r == 0`) or tail (`r == nrun - 1`) of a 32K block
/// (or for an interior run that was unsorted) to be stitched in Stage 3B.
#[derive(Clone, Debug, Default)]
pub struct BoundarySeg {
    pub block_idx: u32,
    pub is_head: bool,
    pub is_tail: bool,
    pub user_idx: u32,
    pub alias_mask: u16,
    pub unsorted: bool,
    pub events: Vec<BoundaryEv>,
}

/// Columnar `user_daily` output row produced by the fast sessionizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastUserDailyRow {
    pub person_idx: u32,
    pub day: i32,
    pub sessions: u32,
    pub events: u32,
    pub revenue_cents: u64,
    pub top_category: Option<u8>,
}

/// Per-thread columnar output buffer for `user_daily` rows and winning `user_id` strings.
pub struct FastUserBatch {
    /// Winning packed 64-bit `ids::tok` token per emitted person (`person_idx` indexes into `person_tokens`).
    pub person_tokens: Vec<u64>,
    /// Columnar `user_daily` rows referencing `person_idx` in `person_tokens`.
    pub rows: Vec<FastUserDailyRow>,
    /// Direct columnar arrays for `FastUserDailyWriter` (populated alongside or instead of `rows`).
    pub names_data: Vec<u8>,
    pub names_offsets: Vec<u32>,
    pub person_keys: Vec<u32>,
    pub day: Vec<i32>,
    pub sessions: Vec<i32>,
    pub events: Vec<i32>,
    pub revenue_cents: Vec<i32>,
    pub top_category: Vec<u8>,
}

impl Default for FastUserBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl FastUserBatch {
    pub fn new() -> Self {
        Self {
            person_tokens: Vec::new(),
            rows: Vec::new(),
            names_data: Vec::with_capacity(256 << 10),
            names_offsets: vec![0],
            person_keys: Vec::with_capacity(32 << 10),
            day: Vec::with_capacity(32 << 10),
            sessions: Vec::with_capacity(32 << 10),
            events: Vec::with_capacity(32 << 10),
            revenue_cents: Vec::with_capacity(32 << 10),
            top_category: Vec::with_capacity(32 << 10),
        }
    }

    pub fn clear(&mut self) {
        self.person_tokens.clear();
        self.rows.clear();
        self.names_data.clear();
        self.names_offsets.clear();
        self.names_offsets.push(0);
        self.person_keys.clear();
        self.day.clear();
        self.sessions.clear();
        self.events.clear();
        self.revenue_cents.clear();
        self.top_category.clear();
    }

    #[inline(always)]
    pub fn push_person_name(&mut self, user_idx: u32, eff_mask: u16, user_key: u64) -> u32 {
        let pidx = (self.names_offsets.len() - 1) as u32;
        let base_val = (user_idx as u64) << 9;
        let best_y = if (eff_mask & (eff_mask - 1)) == 0 {
            let a = eff_mask.trailing_zeros() as u64;
            crate::unscramble::scramble40(base_val | a, user_key)
        } else {
            let mut m = eff_mask;
            let mut best_key = u64::MAX;
            let mut best_y = 0u64;
            while m != 0 {
                let a = m.trailing_zeros() as u64;
                m &= m - 1;
                let y = crate::unscramble::scramble40(base_val | a, user_key);
                let (k, _) = crate::unscramble::b36_lex_key(y);
                if k < best_key {
                    best_key = k;
                    best_y = y;
                }
            }
            best_y
        };
        self.names_data.reserve(9);
        let base_len = self.names_data.len();
        unsafe {
            let dst = &mut *(self.names_data.as_mut_ptr().add(base_len) as *mut [u8; 9]);
            let len = crate::unscramble::encode_b36(b'u', best_y, dst);
            self.names_data.set_len(base_len + len);
            self.names_offsets.push((base_len + len) as u32);
        }
        pidx
    }

    pub fn as_cols<'a>(
        &'a self,
        cat_dict_data: &'a [u8],
        cat_dict_offsets: &'a [u32],
    ) -> crate::fast_parquet::UserDailyCols<'a> {
        crate::fast_parquet::UserDailyCols {
            names_data: &self.names_data,
            names_offsets: &self.names_offsets,
            person_keys: &self.person_keys,
            day: &self.day,
            sessions: &self.sessions,
            events: &self.events,
            revenue_cents: &self.revenue_cents,
            top_category: &self.top_category,
            cat_dict_data,
            cat_dict_offsets,
        }
    }
}

/// Shared read-only configuration for Stage 3A (`process_block_32k`) and Stage 3B (`stitch_and_process_boundaries`).
pub struct FastSessConfig<'a> {
    pub alias_masks: &'a [u16],
    pub multi_comp_users: &'a [(u32, [u16; 8], u8)],
    pub click_csr: &'a ClickCsr,
    pub claimed_users: &'a [AtomicU64],
    pub user_key: u64,
    /// Offset added to `rel_ts_ms` to obtain milliseconds since `window_start` (`-604_800_000` when `base_ms = window_start - 7 days`).
    pub ws_day_offset: i64,
    pub first_day: i32,
    pub days: usize,
    pub num_campaigns: usize,
}

impl<'a> FastSessConfig<'a> {
    /// Atomically claims `user_idx` in `claimed_users`, returning `true` if this thread won the claim.
    #[inline(always)]
    pub fn try_claim_user(&self, user_idx: u32) -> bool {
        let u = user_idx as usize;
        let word = u >> 6;
        let bit = 1u64 << (u & 63);
        if word >= self.claimed_users.len() {
            return false;
        }
        (self.claimed_users[word].fetch_or(bit, Relaxed) & bit) == 0
    }

    #[inline(always)]
    pub fn is_user_claimed(&self, user_idx: u32) -> bool {
        let u = user_idx as usize;
        let word = u >> 6;
        let bit = 1u64 << (u & 63);
        if word >= self.claimed_users.len() {
            return false;
        }
        (self.claimed_users[word].load(Relaxed) & bit) != 0
    }
}

const TILE_FLUSH: usize = 1024;
const TILE_CAP: usize = TILE_FLUSH + 16;

/// L1-resident tile buffer (`~9.1 KiB`) used to fuse slot compaction with vectorized
/// run/session boundary scanning, avoiding 288 KiB per-block `q_user`/`q_ts`/`q_alias` arrays.
#[repr(C, align(32))]
struct RunTileBuf {
    user: [u32; TILE_CAP],
    ts: [u32; TILE_CAP],
    alias: [u8; TILE_CAP],
}

impl Default for RunTileBuf {
    fn default() -> Self {
        Self {
            user: [0u32; TILE_CAP],
            ts: [0u32; TILE_CAP],
            alias: [0u8; TILE_CAP],
        }
    }
}

/// Per-thread L2-resident workspace for Stage 3A 32K-block event deduplication and vectorized sessionization.
pub struct FastSessWorker {
    /// 512 KiB L2-resident direct-index dedup table (`32,768` x 16-byte slots).
    pub slots: Box<[FastEvRec; BLOCK_EVENTS]>,
    /// Generation counter incremented by `8` before each block so `slots` never needs `memset` between blocks.
    pub gen: u16,
    /// Compacted live event `w0` words (`(rel_ts_ms << 33) | (price_cents << 9) | (cat << 4) | alias`).
    pub q_w0: Box<[u64; BLOCK_EVENTS]>,
    /// L1-resident tile buffer for fused compaction and boundary detection.
    tile: Box<RunTileBuf>,
    /// 64-bit purchase bitmask array (`1` bit per compacted live event in `0..k`: bit `i & 63` of `pkm[i >> 6]` is 1 iff `price_cents > 0`).
    /// Includes 1 extra padding word so 4-bit unaligned writes never cross past the end.
    pub pkm: [u64; BLOCK_EVENTS / 64 + 1],
    /// Session start timestamps `sts[0..nsess]`.
    pub sts: Vec<u32>,
    /// Session event start offsets `s_ev_start[0..=nsess]` in `q[0..k]`.
    pub s_ev_start: Vec<u16>,
    /// Session day indices `sday[0..nsess]`.
    pub sday: Vec<u32>,
    /// Identified user runs in the current 32K block.
    pub runs: Vec<UserRun>,
    /// Per-thread `campaign_daily` cells (`(num_campaigns + 1) * days`).
    pub cells: Vec<Cell>,
    /// Separate per-thread `assisted_sessions` table (`num_campaigns * days`) for cache-friendly click assist updates.
    pub assisted: Vec<u32>,
    /// Generation-tagged coverage array (`num_campaigns`) for the `> 16` clicks two-pointer assist loop.
    covered: Vec<u64>,
    assist_gen: u32,
    /// Per-thread columnar `user_daily` output batch.
    pub user_out: FastUserBatch,
    /// Boundary segments collected across processed 32K blocks for Stage 3B stitching.
    pub boundary_segs: Vec<BoundarySeg>,
    /// Deduplicated canonical-`eid` events whose `user_id` was odd/complex (`(odd_handle, w0)`).
    pub kept_odd_events: Vec<(u32, u64)>,
}

impl FastSessWorker {
    pub fn new(num_campaigns: usize, days: usize) -> Self {
        let total_cells = (num_campaigns + 1) * days;
        // Allocate 512 KiB arrays directly on the heap without overflowing the stack.
        let slots: Box<[FastEvRec; BLOCK_EVENTS]> = vec![FastEvRec::default(); BLOCK_EVENTS]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        let q_w0: Box<[u64; BLOCK_EVENTS]> = vec![0u64; BLOCK_EVENTS]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        let tile = Box::<RunTileBuf>::default();

        Self {
            slots,
            gen: 0,
            q_w0,
            tile,
            pkm: [0u64; BLOCK_EVENTS / 64 + 1],
            sts: Vec::with_capacity(8192),
            s_ev_start: Vec::with_capacity(8192),
            sday: Vec::with_capacity(8192),
            runs: Vec::with_capacity(2048),
            cells: vec![Cell::default(); total_cells],
            assisted: vec![0u32; total_cells],
            covered: vec![0u64; num_campaigns.max(1)],
            assist_gen: 0,
            user_out: FastUserBatch::new(),
            boundary_segs: Vec::new(),
            kept_odd_events: Vec::new(),
        }
    }

    /// Merges `assisted` into `cells` and returns the thread's `campaign_daily` cells.
    pub fn finalize_cells(&mut self) -> &[Cell] {
        for (c, &a) in self.cells.iter_mut().zip(&self.assisted) {
            c.assisted = a;
        }
        &self.cells
    }

    /// Advances the block generation counter by 8, only zeroing `slots` on 16-bit wraparound
    /// (every 8,191 blocks = 268M events).
    #[inline(always)]
    fn next_generation(&mut self) -> u16 {
        if self.gen >= u16::MAX - 16 {
            self.slots.fill(FastEvRec::default());
            self.gen = 8;
        } else {
            self.gen = self.gen.wrapping_add(8);
        }
        self.gen
    }

    #[inline(always)]
    unsafe fn dedup_one(slots_ptr: *mut FastEvRec, rec: FastEvRec, gen: u16) {
        let idx = rec.slot();
        let tag = ((idx as u16) & 7).wrapping_add(gen);
        let slot = slots_ptr.add(idx);
        if ((*slot).w1 as u16) != tag || rec.ingest_rel_sec() < (*slot).ingest_rel_sec() {
            *slot = FastEvRec {
                w0: rec.w0,
                w1: (rec.w1 & !0xFFFFu64) | (tag as u64),
            };
        }
    }

    #[inline(always)]
    unsafe fn dedup_slice(slots_ptr: *mut FastEvRec, slice: &[FastEvRec], gen: u16) {
        const PREFETCH_AHEAD: usize = 12;
        let n = slice.len();
        let s_ptr = slice.as_ptr();
        let main_n = n.saturating_sub(PREFETCH_AHEAD);
        for i in 0..main_n {
            #[cfg(target_arch = "x86_64")]
            {
                let ahead_slot = (*s_ptr.add(i + PREFETCH_AHEAD)).slot();
                std::arch::x86_64::_mm_prefetch(
                    slots_ptr.add(ahead_slot) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
            Self::dedup_one(slots_ptr, *s_ptr.add(i), gen);
        }
        for i in main_n..n {
            Self::dedup_one(slots_ptr, *s_ptr.add(i), gen);
        }
    }

    /// Stage 3A (chunked arena variant): Processes one block of up to 32,768 consecutive `event_index` slots
    /// directly from `Arena<FastEvRec>` chunks without copying into an intermediate vector.
    pub unsafe fn process_block_32k_chunks(
        &mut self,
        block_idx: u32,
        arena_base: *const FastEvRec,
        chunks: &[(u32, u32)],
        cfg: &FastSessConfig<'_>,
    ) {
        if chunks.is_empty() {
            return;
        }
        let gen = self.next_generation();
        let slots_ptr = self.slots.as_mut_ptr();
        for &(first, count) in chunks {
            let slice = std::slice::from_raw_parts(arena_base.add(first as usize), count as usize);
            Self::dedup_slice(slots_ptr, slice, gen);
        }
        self.finish_block_32k_after_dedup(block_idx, gen, cfg);
    }

    /// Stage 3A: Processes one block of up to 32,768 consecutive `event_index` slots:
    /// 1. Direct-index L2 dedup in `self.slots` using generation counter `self.gen`.
    /// 2. L1-tiled compaction of live valid events into `q_w0` + `pkm` and fused boundary pass.
    /// 3. In-cache session attribution and columnar `user_daily` / `campaign_daily` aggregation for interior runs.
    /// 4. Saving head (`r == 0`) and tail (`r == nrun - 1`) boundary runs as `BoundarySeg` for Stage 3B.
    pub fn process_block_32k(
        &mut self,
        block_idx: u32,
        events: &[FastEvRec],
        cfg: &FastSessConfig<'_>,
    ) {
        if events.is_empty() {
            return;
        }
        let gen = self.next_generation();
        let slots_ptr = self.slots.as_mut_ptr();
        unsafe {
            Self::dedup_slice(slots_ptr, events, gen);
        }
        self.finish_block_32k_after_dedup(block_idx, gen, cfg);
    }

    fn finish_block_32k_after_dedup(
        &mut self,
        block_idx: u32,
        gen: u16,
        cfg: &FastSessConfig<'_>,
    ) {
        // 2. Compact live valid events from `slots[0..32768]` and identify user runs + sessions in L1 tiles
        let k = self.compact_and_identify(gen);
        let nrun = self.runs.len();
        if k == 0 || nrun == 0 {
            return;
        }

        // 3. Process interior user runs (`r in 1..nrun-1`)
        if nrun >= 3 {
            for r in 1..(nrun - 1) {
                let run = self.runs[r];
                let u = run.user_idx as usize;
                let ut = if u < cfg.alias_masks.len() { cfg.alias_masks[u] } else { IS_COMPLEX };
                if run.unsorted || (ut & (IS_COMPLEX | HAS_EXTRA_EVENTS)) != 0 {
                    // Save complex/unsorted interior run for Stage 3B fallback
                    self.push_boundary_seg(block_idx, false, false, run);
                    continue;
                }
                let comp_mask = ut & ALIAS_MASK;
                let class = if (run.alias_mask & !comp_mask) == 0 {
                    comp_mask
                } else if run.alias_mask.is_power_of_two() && (run.alias_mask & comp_mask) == 0 {
                    run.alias_mask
                } else {
                    // Events in this run span multiple disconnected alias components
                    self.push_boundary_seg(block_idx, false, false, run);
                    continue;
                };

                if !cfg.try_claim_user(run.user_idx) {
                    continue;
                }
                self.attribute_and_emit_run(run, class, cfg);
            }
        }

        // 4. Save boundary runs (`r == 0` and `r == nrun - 1`) for Stage 3B cross-block stitching
        if nrun == 1 {
            let run0 = self.runs[0];
            self.push_boundary_seg(block_idx, true, true, run0);
        } else {
            let first = self.runs[0];
            let last = self.runs[nrun - 1];
            self.push_boundary_seg(block_idx, true, false, first);
            self.push_boundary_seg(block_idx, false, true, last);
        }
    }

    fn push_boundary_seg(&mut self, block_idx: u32, is_head: bool, is_tail: bool, run: UserRun) {
        let ev0 = run.ev_start as usize;
        let ev1 = run.ev_end as usize;
        let mut evs = Vec::with_capacity(ev1 - ev0);
        for i in ev0..ev1 {
            evs.push(BoundaryEv::from_w0(self.q_w0[i]));
        }
        self.boundary_segs.push(BoundarySeg {
            block_idx,
            is_head,
            is_tail,
            user_idx: run.user_idx,
            alias_mask: run.alias_mask,
            unsorted: run.unsorted,
            events: evs,
        });
    }

    /// Fuses compaction of occupied valid slots from `self.slots[0..32768]` into `q_w0` + `pkm`
    /// with L1-tiled run/session boundary detection (`sts`, `s_ev_start`, `runs`).
    #[inline(always)]
    fn compact_and_identify(&mut self, gen: u16) -> usize {
        self.sts.clear();
        self.s_ev_start.clear();
        self.runs.clear();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512dq")
            {
                return self.compact_and_identify_avx512(gen);
            }
            if is_x86_feature_detected!("avx2") {
                return self.compact_and_identify_avx2(gen);
            }
        }
        self.compact_and_identify_scalar(gen)
    }

    #[inline]
    fn flush_tile_scalar(
        tile: &mut RunTileBuf,
        sts: &mut Vec<u32>,
        s_ev_start: &mut Vec<u16>,
        runs: &mut Vec<UserRun>,
        total_k: usize,
        t: usize,
        cur_run: &mut UserRun,
        has_started: &mut bool,
    ) {
        let end = 1 + t;
        let mut i = if !*has_started {
            *cur_run = UserRun {
                user_idx: tile.user[1],
                ev_start: total_k as u16,
                ev_end: 0,
                sess_start: sts.len() as u32,
                sess_end: 0,
                alias_mask: 1u16 << tile.alias[1],
                unsorted: false,
            };
            sts.push(tile.ts[1]);
            s_ev_start.push(total_k as u16);
            *has_started = true;
            2usize
        } else {
            1usize
        };

        while i < end {
            let g_idx = (total_k + i - 1) as u16;
            let u = tile.user[i];
            let ts = tile.ts[i];
            let prev_ts = tile.ts[i - 1];
            let alias = tile.alias[i];
            if u != cur_run.user_idx {
                cur_run.ev_end = g_idx;
                cur_run.sess_end = sts.len() as u32;
                runs.push(*cur_run);

                *cur_run = UserRun {
                    user_idx: u,
                    ev_start: g_idx,
                    ev_end: 0,
                    sess_start: sts.len() as u32,
                    sess_end: 0,
                    alias_mask: 1u16 << alias,
                    unsorted: false,
                };
                sts.push(ts);
                s_ev_start.push(g_idx);
            } else {
                cur_run.alias_mask |= 1u16 << alias;
                if ts < prev_ts {
                    cur_run.unsorted = true;
                } else if ts - prev_ts > SESSION_GAP_MS {
                    sts.push(ts);
                    s_ev_start.push(g_idx);
                }
            }
            i += 1;
        }

        tile.user[0] = tile.user[t];
        tile.ts[0] = tile.ts[t];
    }

    fn compact_and_identify_scalar(&mut self, gen: u16) -> usize {
        self.pkm.fill(0);
        let mut total_k = 0usize;
        let mut t = 0usize;
        let mut cur_run = UserRun::default();
        let mut has_started = false;

        let tile = &mut *self.tile;
        for idx in 0..BLOCK_EVENTS {
            let tag = ((idx as u16) & 7).wrapping_add(gen);
            let rec = self.slots[idx];
            if (rec.w1 as u16) == tag && rec.w0 != INVALID_W0 {
                if rec.is_odd_user() {
                    self.kept_odd_events.push((rec.odd_handle(), rec.w0));
                    continue;
                }
                let k = total_k + t;
                self.q_w0[k] = rec.w0;
                tile.user[1 + t] = rec.user_idx();
                tile.ts[1 + t] = rec.rel_ts_ms();
                tile.alias[1 + t] = rec.alias();
                if rec.price_cents() != 0 {
                    self.pkm[k >> 6] |= 1u64 << (k & 63);
                }
                t += 1;
                if t >= TILE_FLUSH {
                    Self::flush_tile_scalar(
                        tile,
                        &mut self.sts,
                        &mut self.s_ev_start,
                        &mut self.runs,
                        total_k,
                        t,
                        &mut cur_run,
                        &mut has_started,
                    );
                    total_k += t;
                    t = 0;
                }
            }
        }

        if t > 0 {
            Self::flush_tile_scalar(
                tile,
                &mut self.sts,
                &mut self.s_ev_start,
                &mut self.runs,
                total_k,
                t,
                &mut cur_run,
                &mut has_started,
            );
            total_k += t;
        }
        if has_started {
            cur_run.ev_end = total_k as u16;
            cur_run.sess_end = self.sts.len() as u32;
            self.runs.push(cur_run);
            self.s_ev_start.push(total_k as u16);
        }
        total_k
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi2,lzcnt")]
    unsafe fn flush_tile_avx2(
        tile: &mut RunTileBuf,
        sts: &mut Vec<u32>,
        s_ev_start: &mut Vec<u16>,
        runs: &mut Vec<UserRun>,
        total_k: usize,
        t: usize,
        cur_run: &mut UserRun,
        has_started: &mut bool,
    ) {
        use std::arch::x86_64::*;
        let end = 1 + t;
        let mut i = if !*has_started {
            *cur_run = UserRun {
                user_idx: *tile.user.get_unchecked(1),
                ev_start: total_k as u16,
                ev_end: 0,
                sess_start: sts.len() as u32,
                sess_end: 0,
                alias_mask: 1u16 << (*tile.alias.get_unchecked(1)),
                unsorted: false,
            };
            sts.push(*tile.ts.get_unchecked(1));
            s_ev_start.push(total_k as u16);
            *has_started = true;
            2usize
        } else {
            1usize
        };

        let v_gap = _mm256_set1_epi32(SESSION_GAP_MS as i32);
        let u_ptr = tile.user.as_ptr();
        let ts_ptr = tile.ts.as_ptr();
        let alias_ptr = tile.alias.as_ptr();

        while i + 8 <= end {
            let u_cur = _mm256_loadu_si256(u_ptr.add(i) as *const __m256i);
            let u_prev = _mm256_loadu_si256(u_ptr.add(i - 1) as *const __m256i);
            let u_eq = _mm256_cmpeq_epi32(u_cur, u_prev);
            let u_start_mask = (!_mm256_movemask_ps(_mm256_castsi256_ps(u_eq))) as u32 & 0xFF;

            let ts_cur = _mm256_loadu_si256(ts_ptr.add(i) as *const __m256i);
            let ts_prev = _mm256_loadu_si256(ts_ptr.add(i - 1) as *const __m256i);

            let min_ts = _mm256_min_epu32(ts_cur, ts_prev);
            let ge_prev = _mm256_cmpeq_epi32(min_ts, ts_prev);
            let lt_mask = (!_mm256_movemask_ps(_mm256_castsi256_ps(ge_prev))) as u32 & 0xFF;
            let unsorted_mask = lt_mask & !u_start_mask;

            let diff = _mm256_sub_epi32(ts_cur, ts_prev);
            let le_gap = _mm256_cmpeq_epi32(_mm256_min_epu32(diff, v_gap), diff);
            let gt_gap_mask = (!_mm256_movemask_ps(_mm256_castsi256_ps(le_gap))) as u32 & 0xFF & !lt_mask;

            let any_boundary = u_start_mask | gt_gap_mask | unsorted_mask;
            if any_boundary == 0 {
                let aliases = (alias_ptr.add(i) as *const u64).read_unaligned();
                let a0 = *alias_ptr.add(i);
                if aliases == (a0 as u64).wrapping_mul(0x0101_0101_0101_0101) {
                    cur_run.alias_mask |= 1u16 << a0;
                } else {
                    let mut m = cur_run.alias_mask;
                    for lane in 0..8 {
                        m |= 1u16 << (*alias_ptr.add(i + lane));
                    }
                    cur_run.alias_mask = m;
                }
            } else {
                for lane in 0..8 {
                    let idx = i + lane;
                    let g_idx = (total_k + idx - 1) as u16;
                    let bit = 1u32 << lane;
                    let alias = *alias_ptr.add(idx);
                    if (u_start_mask & bit) != 0 {
                        cur_run.ev_end = g_idx;
                        cur_run.sess_end = sts.len() as u32;
                        runs.push(*cur_run);

                        let u = *u_ptr.add(idx);
                        let ts = *ts_ptr.add(idx);
                        *cur_run = UserRun {
                            user_idx: u,
                            ev_start: g_idx,
                            ev_end: 0,
                            sess_start: sts.len() as u32,
                            sess_end: 0,
                            alias_mask: 1u16 << alias,
                            unsorted: false,
                        };
                        sts.push(ts);
                        s_ev_start.push(g_idx);
                    } else {
                        cur_run.alias_mask |= 1u16 << alias;
                        if (unsorted_mask & bit) != 0 {
                            cur_run.unsorted = true;
                        } else if (gt_gap_mask & bit) != 0 {
                            sts.push(*ts_ptr.add(idx));
                            s_ev_start.push(g_idx);
                        }
                    }
                }
            }
            i += 8;
        }

        while i < end {
            let g_idx = (total_k + i - 1) as u16;
            let u = *u_ptr.add(i);
            let ts = *ts_ptr.add(i);
            let prev_ts = *ts_ptr.add(i - 1);
            let alias = *alias_ptr.add(i);
            if u != cur_run.user_idx {
                cur_run.ev_end = g_idx;
                cur_run.sess_end = sts.len() as u32;
                runs.push(*cur_run);

                *cur_run = UserRun {
                    user_idx: u,
                    ev_start: g_idx,
                    ev_end: 0,
                    sess_start: sts.len() as u32,
                    sess_end: 0,
                    alias_mask: 1u16 << alias,
                    unsorted: false,
                };
                sts.push(ts);
                s_ev_start.push(g_idx);
            } else {
                cur_run.alias_mask |= 1u16 << alias;
                if ts < prev_ts {
                    cur_run.unsorted = true;
                } else if ts - prev_ts > SESSION_GAP_MS {
                    sts.push(ts);
                    s_ev_start.push(g_idx);
                }
            }
            i += 1;
        }

        tile.user[0] = *u_ptr.add(t);
        tile.ts[0] = *ts_ptr.add(t);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi2,lzcnt")]
    unsafe fn compact_and_identify_avx2(&mut self, gen: u16) -> usize {
        use std::arch::x86_64::*;
        self.pkm.fill(0);
        let mut total_k = 0usize;
        let mut t = 0usize;
        let mut cur_run = UserRun::default();
        let mut has_started = false;

        // After `_mm256_unpacklo/hi_epi64(v0, v1)`, lanes are ordered `[0, 2, 1, 3]`.
        let tags_lo = _mm256_setr_epi64x(
            gen as i64,
            gen.wrapping_add(2) as i64,
            gen.wrapping_add(1) as i64,
            gen.wrapping_add(3) as i64,
        );
        let tags_hi = _mm256_setr_epi64x(
            gen.wrapping_add(4) as i64,
            gen.wrapping_add(6) as i64,
            gen.wrapping_add(5) as i64,
            gen.wrapping_add(7) as i64,
        );
        let mask16 = _mm256_set1_epi64x(0xFFFF);
        let inv_w0 = _mm256_set1_epi64x(-1);
        let mask_f = _mm256_set1_epi64x(0xF);
        let price_mask = _mm256_set1_epi64x((0xFF_FFFFu64 << 9) as i64);
        let zero = _mm256_setzero_si256();
        let zero128 = _mm_setzero_si128();

        let slots_ptr = self.slots.as_ptr() as *const __m256i;
        let q_w0_ptr = self.q_w0.as_mut_ptr();
        let pkm_byte_ptr = self.pkm.as_mut_ptr() as *mut u8;
        let tile = &mut *self.tile;

        for idx in (0..BLOCK_EVENTS).step_by(4) {
            let v0 = _mm256_loadu_si256(slots_ptr.add(idx >> 1));
            let v1 = _mm256_loadu_si256(slots_ptr.add((idx >> 1) + 1));
            let w0_perm = _mm256_unpacklo_epi64(v0, v1); // [w0_0, w0_2, w0_1, w0_3]
            let w1_perm = _mm256_unpackhi_epi64(v0, v1); // [w1_0, w1_2, w1_1, w1_3]

            let tags = if (idx & 4) == 0 { tags_lo } else { tags_hi };
            let slot_tags = _mm256_and_si256(w1_perm, mask16);
            let occ_eq = _mm256_cmpeq_epi64(slot_tags, tags);
            let inv_eq = _mm256_cmpeq_epi64(w0_perm, inv_w0);
            let live_vec = _mm256_andnot_si256(inv_eq, occ_eq);
            let live_perm = _mm256_movemask_pd(_mm256_castsi256_pd(live_vec)) as u32;
            if live_perm == 0 {
                continue;
            }

            let odd_perm = _mm256_movemask_pd(_mm256_castsi256_pd(w1_perm)) as u32;
            if live_perm == 0xF && odd_perm == 0 {
                let k = total_k + t;
                let tp = 1 + t;
                let w0 = _mm256_permute4x64_epi64(w0_perm, 0xD8);
                let w1 = _mm256_permute4x64_epi64(w1_perm, 0xD8);

                _mm256_storeu_si256(q_w0_ptr.add(k) as *mut __m256i, w0);

                let u64v = _mm256_srli_epi64(w1, 37);
                let u32v = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(u64v, 0x08), 0xD8);
                _mm_storeu_si128(tile.user.as_mut_ptr().add(tp) as *mut __m128i, _mm256_castsi256_si128(u32v));

                let ts64 = _mm256_srli_epi64(w0, 33);
                let ts32 = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(ts64, 0x08), 0xD8);
                _mm_storeu_si128(tile.ts.as_mut_ptr().add(tp) as *mut __m128i, _mm256_castsi256_si128(ts32));

                let a64 = _mm256_and_si256(w0, mask_f);
                let a32 = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(a64, 0x08), 0xD8);
                let a16 = _mm_packus_epi32(_mm256_castsi256_si128(a32), zero128);
                let a8 = _mm_packus_epi16(a16, zero128);
                (tile.alias.as_mut_ptr().add(tp) as *mut u32)
                    .write_unaligned((_mm_cvtsi128_si32(a8) as u32).to_le());

                let p_nonz = _mm256_and_si256(w0, price_mask);
                let p_zero = _mm256_cmpeq_epi64(p_nonz, zero);
                let purch_mask = (!_mm256_movemask_pd(_mm256_castsi256_pd(p_zero)) as u16) & 0xF;
                let pkm_u16 = pkm_byte_ptr.add(k >> 3) as *mut u16;
                pkm_u16.write_unaligned(pkm_u16.read_unaligned() | (purch_mask << (k & 7)));

                t += 4;
            } else {
                // Convert `[0, 2, 1, 3]` mask bits into `[0, 1, 2, 3]` slot order.
                let mut m = (live_perm & 1)
                    | ((live_perm & 4) >> 1)
                    | ((live_perm & 2) << 1)
                    | (live_perm & 8);
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let rec = *self.slots.get_unchecked(idx + lane);
                    if rec.is_odd_user() {
                        self.kept_odd_events.push((rec.odd_handle(), rec.w0));
                        continue;
                    }
                    let k = total_k + t;
                    let tp = 1 + t;
                    *q_w0_ptr.add(k) = rec.w0;
                    *tile.user.get_unchecked_mut(tp) = rec.user_idx();
                    *tile.ts.get_unchecked_mut(tp) = rec.rel_ts_ms();
                    *tile.alias.get_unchecked_mut(tp) = rec.alias();
                    let is_purch = ((rec.w0 >> 9) & 0xFF_FFFF) != 0;
                    *self.pkm.get_unchecked_mut(k >> 6) |= (is_purch as u64) << (k & 63);
                    t += 1;
                }
            }

            if t >= TILE_FLUSH {
                Self::flush_tile_avx2(
                    tile,
                    &mut self.sts,
                    &mut self.s_ev_start,
                    &mut self.runs,
                    total_k,
                    t,
                    &mut cur_run,
                    &mut has_started,
                );
                total_k += t;
                t = 0;
            }
        }

        if t > 0 {
            Self::flush_tile_avx2(
                tile,
                &mut self.sts,
                &mut self.s_ev_start,
                &mut self.runs,
                total_k,
                t,
                &mut cur_run,
                &mut has_started,
            );
            total_k += t;
        }
        if has_started {
            cur_run.ev_end = total_k as u16;
            cur_run.sess_end = self.sts.len() as u32;
            self.runs.push(cur_run);
            self.s_ev_start.push(total_k as u16);
        }
        total_k
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
    unsafe fn flush_tile_avx512(
        tile: &mut RunTileBuf,
        sts: &mut Vec<u32>,
        s_ev_start: &mut Vec<u16>,
        runs: &mut Vec<UserRun>,
        total_k: usize,
        t: usize,
        cur_run: &mut UserRun,
        has_started: &mut bool,
    ) {
        use std::arch::x86_64::*;
        let end = 1 + t;
        let mut i = if !*has_started {
            *cur_run = UserRun {
                user_idx: *tile.user.get_unchecked(1),
                ev_start: total_k as u16,
                ev_end: 0,
                sess_start: sts.len() as u32,
                sess_end: 0,
                alias_mask: 1u16 << (*tile.alias.get_unchecked(1)),
                unsorted: false,
            };
            sts.push(*tile.ts.get_unchecked(1));
            s_ev_start.push(total_k as u16);
            *has_started = true;
            2usize
        } else {
            1usize
        };

        let v_gap = _mm512_set1_epi32(SESSION_GAP_MS as i32);
        let u_ptr = tile.user.as_ptr();
        let ts_ptr = tile.ts.as_ptr();
        let alias_ptr = tile.alias.as_ptr();

        while i + 16 <= end {
            let u_cur = _mm512_loadu_si512(u_ptr.add(i) as *const __m512i);
            let u_prev = _mm512_loadu_si512(u_ptr.add(i - 1) as *const __m512i);
            let u_start_mask = _mm512_cmpneq_epi32_mask(u_cur, u_prev) as u32;

            let ts_cur = _mm512_loadu_si512(ts_ptr.add(i) as *const __m512i);
            let ts_prev = _mm512_loadu_si512(ts_ptr.add(i - 1) as *const __m512i);

            let lt_mask = _mm512_cmplt_epu32_mask(ts_cur, ts_prev) as u32;
            let unsorted_mask = lt_mask & !u_start_mask;

            let diff = _mm512_sub_epi32(ts_cur, ts_prev);
            let gt_gap_mask = (_mm512_cmpgt_epu32_mask(diff, v_gap) as u32) & !lt_mask;

            let any_boundary = u_start_mask | gt_gap_mask | unsorted_mask;
            if any_boundary == 0 {
                let ap = alias_ptr.add(i);
                let a0 = *ap;
                let rep = (a0 as u64).wrapping_mul(0x0101_0101_0101_0101);
                if (ap as *const u64).read_unaligned() == rep
                    && (ap.add(8) as *const u64).read_unaligned() == rep
                {
                    cur_run.alias_mask |= 1u16 << a0;
                } else {
                    let mut m = cur_run.alias_mask;
                    for lane in 0..16 {
                        m |= 1u16 << (*alias_ptr.add(i + lane));
                    }
                    cur_run.alias_mask = m;
                }
            } else {
                for lane in 0..16 {
                    let idx = i + lane;
                    let g_idx = (total_k + idx - 1) as u16;
                    let bit = 1u32 << lane;
                    let alias = *alias_ptr.add(idx);
                    if (u_start_mask & bit) != 0 {
                        cur_run.ev_end = g_idx;
                        cur_run.sess_end = sts.len() as u32;
                        runs.push(*cur_run);

                        let u = *u_ptr.add(idx);
                        let ts = *ts_ptr.add(idx);
                        *cur_run = UserRun {
                            user_idx: u,
                            ev_start: g_idx,
                            ev_end: 0,
                            sess_start: sts.len() as u32,
                            sess_end: 0,
                            alias_mask: 1u16 << alias,
                            unsorted: false,
                        };
                        sts.push(ts);
                        s_ev_start.push(g_idx);
                    } else {
                        cur_run.alias_mask |= 1u16 << alias;
                        if (unsorted_mask & bit) != 0 {
                            cur_run.unsorted = true;
                        } else if (gt_gap_mask & bit) != 0 {
                            sts.push(*ts_ptr.add(idx));
                            s_ev_start.push(g_idx);
                        }
                    }
                }
            }
            i += 16;
        }

        while i < end {
            let g_idx = (total_k + i - 1) as u16;
            let u = *u_ptr.add(i);
            let ts = *ts_ptr.add(i);
            let prev_ts = *ts_ptr.add(i - 1);
            let alias = *alias_ptr.add(i);
            if u != cur_run.user_idx {
                cur_run.ev_end = g_idx;
                cur_run.sess_end = sts.len() as u32;
                runs.push(*cur_run);

                *cur_run = UserRun {
                    user_idx: u,
                    ev_start: g_idx,
                    ev_end: 0,
                    sess_start: sts.len() as u32,
                    sess_end: 0,
                    alias_mask: 1u16 << alias,
                    unsorted: false,
                };
                sts.push(ts);
                s_ev_start.push(g_idx);
            } else {
                cur_run.alias_mask |= 1u16 << alias;
                if ts < prev_ts {
                    cur_run.unsorted = true;
                } else if ts - prev_ts > SESSION_GAP_MS {
                    sts.push(ts);
                    s_ev_start.push(g_idx);
                }
            }
            i += 1;
        }

        tile.user[0] = *u_ptr.add(t);
        tile.ts[0] = *ts_ptr.add(t);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
    unsafe fn compact_and_identify_avx512(&mut self, gen: u16) -> usize {
        use std::arch::x86_64::*;
        self.pkm.fill(0);
        let mut total_k = 0usize;
        let mut t = 0usize;
        let mut cur_run = UserRun::default();
        let mut has_started = false;

        let tags = _mm512_setr_epi64(
            0u16.wrapping_add(gen) as i64,
            1u16.wrapping_add(gen) as i64,
            2u16.wrapping_add(gen) as i64,
            3u16.wrapping_add(gen) as i64,
            4u16.wrapping_add(gen) as i64,
            5u16.wrapping_add(gen) as i64,
            6u16.wrapping_add(gen) as i64,
            7u16.wrapping_add(gen) as i64,
        );
        let mask16 = _mm512_set1_epi64(0xFFFF);
        let inv_w0 = _mm512_set1_epi64(-1);
        let idx_even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let idx_odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        let mut w0_buf = [0u64; 8];
        let mut w1_buf = [0u64; 8];

        let slots_ptr = self.slots.as_ptr() as *const __m512i;
        let tile = &mut *self.tile;
        for chunk in 0..(BLOCK_EVENTS / 8) {
            let v0 = _mm512_loadu_si512(slots_ptr.add(chunk * 2));
            let v1 = _mm512_loadu_si512(slots_ptr.add(chunk * 2 + 1));
            let w0 = _mm512_permutex2var_epi64(v0, idx_even, v1);
            let w1 = _mm512_permutex2var_epi64(v0, idx_odd, v1);

            let slot_tags = _mm512_and_si512(w1, mask16);
            let occupied = _mm512_cmpeq_epi64_mask(slot_tags, tags);
            let valid = _mm512_cmpneq_epi64_mask(w0, inv_w0);
            let live = occupied & valid;
            if live == 0 {
                continue;
            }
            let cnt = live.count_ones() as usize;
            _mm512_mask_compressstoreu_epi64(w0_buf.as_mut_ptr() as *mut i64, live, w0);
            _mm512_mask_compressstoreu_epi64(w1_buf.as_mut_ptr() as *mut i64, live, w1);

            for i in 0..cnt {
                let cw0 = w0_buf[i];
                let cw1 = w1_buf[i];
                if (cw1 & ODD_USER_BIT) != 0 {
                    let handle = ((cw1 >> 37) & 0x3FF_FFFF) as u32;
                    self.kept_odd_events.push((handle, cw0));
                    continue;
                }
                let k = total_k + t;
                let tp = 1 + t;
                *self.q_w0.get_unchecked_mut(k) = cw0;
                *tile.user.get_unchecked_mut(tp) = ((cw1 >> 37) & 0x3FF_FFFF) as u32;
                *tile.ts.get_unchecked_mut(tp) = (cw0 >> 33) as u32;
                *tile.alias.get_unchecked_mut(tp) = (cw0 & 0xF) as u8;
                let is_purch = ((cw0 >> 9) & 0xFF_FFFF) != 0;
                *self.pkm.get_unchecked_mut(k >> 6) |= (is_purch as u64) << (k & 63);
                t += 1;
            }
            if t >= TILE_FLUSH {
                Self::flush_tile_avx512(
                    tile,
                    &mut self.sts,
                    &mut self.s_ev_start,
                    &mut self.runs,
                    total_k,
                    t,
                    &mut cur_run,
                    &mut has_started,
                );
                total_k += t;
                t = 0;
            }
        }

        if t > 0 {
            Self::flush_tile_avx512(
                tile,
                &mut self.sts,
                &mut self.s_ev_start,
                &mut self.runs,
                total_k,
                t,
                &mut cur_run,
                &mut has_started,
            );
            total_k += t;
        }
        if has_started {
            cur_run.ev_end = total_k as u16;
            cur_run.sess_end = self.sts.len() as u32;
            self.runs.push(cur_run);
            self.s_ev_start.push(total_k as u16);
        }
        total_k
    }

    /// Attributes sessions `[run.sess_start..run.sess_end)` for `run.user_idx`, aggregates purchase
    /// events via `self.pkm`, updates `campaign_daily` counters, and appends `user_daily` rows.
    fn attribute_and_emit_run(
        &mut self,
        run: UserRun,
        class: u16,
        cfg: &FastSessConfig<'_>,
    ) {
        let s0 = run.sess_start as usize;
        let s1 = run.sess_end as usize;
        let nsess = s1 - s0;
        if nsess == 0 {
            return;
        }

        let eff_mask = if (class & ALIAS_MASK) != 0 && (class & run.alias_mask) != 0 {
            class & ALIAS_MASK
        } else {
            1u16 << (run.alias_mask.trailing_zeros() as u8)
        };
        let person_idx = self.user_out.push_person_name(run.user_idx, eff_mask, cfg.user_key);
        #[cfg(test)]
        {
            let fallback_alias = run.alias_mask.trailing_zeros() as u8;
            let winner_tok = min_token_for_mask(run.user_idx, class, fallback_alias, cfg.user_key);
            self.user_out.person_tokens.push(winner_tok);
        }

        // Compute session days via exact magic multiply `(((sts[s] + ws_day_offset) >> 10) * 6_515_625) >> 39`
        self.sday.clear();
        self.sday.extend(
            (s0..s1).map(|s| fast_session_day(unsafe { *self.sts.get_unchecked(s) }, cfg.ws_day_offset)),
        );

        let raw_clicks = if (run.user_idx as usize) < cfg.click_csr.user_cap {
            cfg.click_csr.csr_slice(run.user_idx as usize)
        } else {
            &[]
        };

        // Check if all clicks in `raw_clicks` belong to `class`
        let all_in_class = raw_clicks.iter().all(|c| (class & (1u16 << c.alias())) != 0);
        let filtered_buf: Vec<ClickSlot>;
        let clicks: &[ClickSlot] = if all_in_class {
            raw_clicks
        } else {
            filtered_buf = raw_clicks
                .iter()
                .copied()
                .filter(|c| (class & (1u16 << c.alias())) != 0)
                .collect();
            &filtered_buf
        };

        let nck = clicks.len();
        let days = cfg.days;
        let organic = cfg.num_campaigns;

        // Precompute same_camp_mask[0..16] for the 1..=16 clicks vectorized fast path
        let mut ck_ts16 = [u32::MAX; 16];
        let mut ck_camp16 = [0u32; 16];
        let mut same_camp_mask16 = [0u16; 16];
        if (1..=16).contains(&nck) {
            for i in 0..nck {
                ck_ts16[i] = clicks[i].ts();
                ck_camp16[i] = clicks[i].camp();
            }
            for i in 0..nck {
                let mut m = 0u16;
                for j in i..nck {
                    if ck_camp16[j] == ck_camp16[i] {
                        m |= 1u16 << j;
                    }
                }
                same_camp_mask16[i] = m;
            }
        }

        let mut cur_day = self.sday[0];
        let mut day_sessions = 0u32;
        let mut day_events = 0u32;
        let mut day_rev = 0u64;
        let mut cat_rev = [0u64; 32];
        let mut cat_mask = 0u32;

        let mut next_click = 0usize;

        for local_s in 0..nsess {
            let s = s0 + local_s;
            let s_day = self.sday[local_s];
            let s_ts = self.sts[s];
            let ev_start = self.s_ev_start[s] as usize;
            let ev_end = self.s_ev_start[s + 1] as usize;

            if s_day != cur_day {
                self.emit_user_day_row(
                    person_idx,
                    cfg.first_day + cur_day as i32,
                    day_sessions,
                    day_events,
                    day_rev,
                    &mut cat_rev,
                    &mut cat_mask,
                );
                cur_day = s_day;
                day_sessions = 0;
                day_events = 0;
                day_rev = 0;
            }

            // Aggregate purchases in session `[ev_start..ev_end)` using `self.pkm` bitmask
            let mut sess_rev = 0u64;
            let mut w_idx = ev_start >> 6;
            let end_w_idx = (ev_end.saturating_sub(1)) >> 6;
            if ev_start < ev_end {
                while w_idx <= end_w_idx {
                    let mut word = self.pkm[w_idx];
                    if w_idx == (ev_start >> 6) {
                        word &= !0u64 << (ev_start & 63);
                    }
                    if w_idx == end_w_idx && (ev_end & 63) != 0 {
                        word &= (1u64 << (ev_end & 63)) - 1;
                    }
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        let ev_idx = (w_idx << 6) + bit;
                        let w0 = self.q_w0[ev_idx];
                        let price = ((w0 >> 9) & 0xFF_FFFF) as u64;
                        let cat = ((w0 >> 4) & 0x1F) as usize;
                        sess_rev += price;
                        cat_rev[cat] += price;
                        cat_mask |= 1u32 << cat;
                        word &= word - 1;
                    }
                    w_idx += 1;
                }
            }

            // Attribute session against clicks (0 clicks, 1..=16 clicks vectorized, >16 clicks two-pointer)
            let credited_camp = if nck == 0 {
                organic
            } else if nck <= 16 {
                let win_mask = compute_click_window_mask_16(s_ts, nck, &ck_ts16);
                if win_mask == 0 {
                    organic
                } else {
                    // Record all distinct campaigns in the 7-day window as assisted
                    let mut m = win_mask;
                    while m != 0 {
                        let k = m.trailing_zeros() as usize;
                        let c = ck_camp16[k] as usize;
                        self.assisted[c * days + s_day as usize] += 1;
                        m &= !same_camp_mask16[k];
                    }
                    // Highest set bit in `win_mask` is the latest click <= s_ts
                    // (ties at the same ts already have smallest base36 click_id placed last by ClickCsr!)
                    let latest = 31 - (win_mask as u32).leading_zeros() as usize;
                    ck_camp16[latest] as usize
                }
            } else {
                while next_click < nck && clicks[next_click].ts() <= s_ts {
                    next_click += 1;
                }
                if next_click > 0 && s_ts - clicks[next_click - 1].ts() <= WEEK_MS {
                    clicks[next_click - 1].camp() as usize
                } else {
                    organic
                }
            };

            let cell = &mut self.cells[credited_camp * days + s_day as usize];
            cell.sessions += 1;
            cell.purchasing += (sess_rev > 0) as u32;
            cell.revenue += sess_rev;

            day_sessions += 1;
            day_events += (ev_end - ev_start) as u32;
            day_rev += sess_rev;
        }

        self.emit_user_day_row(
            person_idx,
            cfg.first_day + cur_day as i32,
            day_sessions,
            day_events,
            day_rev,
            &mut cat_rev,
            &mut cat_mask,
        );

        // For > 16 clicks, run the two-pointer window [lo, hi) over `sts[s0..s1]` for assisted sessions
        if nck > 16 {
            self.attribute_assists_gt16(s0, nsess, days, clicks);
        }
    }

    #[inline(always)]
    fn emit_user_day_row(
        &mut self,
        person_idx: u32,
        day: i32,
        sessions: u32,
        events: u32,
        revenue_cents: u64,
        cat_rev: &mut [u64; 32],
        cat_mask: &mut u32,
    ) {
        let mut top_category = None;
        if *cat_mask != 0 {
            let mut m = *cat_mask;
            let mut best_rev = 0u64;
            let mut best_cat = 0u8;
            while m != 0 {
                let c = m.trailing_zeros() as usize;
                let rev = cat_rev[c];
                cat_rev[c] = 0;
                // Ascending bit iteration visits smaller category indices first,
                // so strict `>` automatically tie-breaks in favor of the smallest category index!
                if rev > best_rev {
                    best_rev = rev;
                    best_cat = c as u8;
                }
                m &= m - 1;
            }
            *cat_mask = 0;
            top_category = Some(best_cat);
        }
        #[cfg(test)]
        self.user_out.rows.push(FastUserDailyRow {
            person_idx,
            day,
            sessions,
            events,
            revenue_cents,
            top_category,
        });
        self.user_out.person_keys.push(person_idx);
        self.user_out.day.push(day);
        self.user_out.sessions.push(sessions as i32);
        self.user_out.events.push(events as i32);
        self.user_out.revenue_cents.push(revenue_cents as i32);
        self.user_out
            .top_category
            .push(top_category.unwrap_or(crate::fast_parquet::NULL_CAT));
    }

    /// Two-pointer `[lo, hi)` assisted-session attribution for users with `> 16` clicks.
    fn attribute_assists_gt16(&mut self, s0: usize, nsess: usize, days: usize, clicks: &[ClickSlot]) {
        self.assist_gen = self.assist_gen.wrapping_add(1);
        let gen_tag = (self.assist_gen as u64) << 32;
        let mut lo = 0usize;
        let mut hi = 0usize;
        for c in clicks {
            let ts = c.ts();
            while lo < nsess && self.sts[s0 + lo] < ts {
                lo += 1;
            }
            let reach = ts as u64 + WEEK_MS as u64;
            while hi < nsess && (self.sts[s0 + hi] as u64) <= reach {
                hi += 1;
            }
            let camp = c.camp() as usize;
            if camp >= self.covered.len() {
                self.covered.resize(camp + 1, 0);
            }
            let cov = &mut self.covered[camp];
            let from = if (*cov & 0xFFFF_FFFF_0000_0000) == gen_tag {
                lo.max((*cov as u32) as usize)
            } else {
                lo
            };
            *cov = gen_tag | (hi as u64);
            for s_idx in from..hi {
                let d = self.sday[s_idx] as usize;
                self.assisted[camp * days + d] += 1;
            }
        }
    }

    /// Stage 3B: Stitches `BoundarySeg`s across adjacent 32K blocks and processes all simple
    /// boundary users through the exact same session attribution and columnar output pipeline.
    /// Returns any remaining complex (`IS_COMPLEX` or multi-component) boundary segments.
    pub fn stitch_and_process_boundaries(
        &mut self,
        mut segs: Vec<BoundarySeg>,
        cfg: &FastSessConfig<'_>,
    ) -> Vec<BoundarySeg> {
        if segs.is_empty() {
            return Vec::new();
        }
        // Sort by (user_idx, block_idx) so all boundary pieces of a user across adjacent blocks are contiguous
        segs.sort_by_key(|s| (s.user_idx, s.block_idx));

        let mut complex_leftovers = Vec::new();
        let mut i = 0usize;
        while i < segs.len() {
            let u = segs[i].user_idx;
            let mut j = i + 1;
            while j < segs.len() && segs[j].user_idx == u {
                j += 1;
            }
            self.process_stitched_user_group(&segs[i..j], cfg, &mut complex_leftovers);
            i = j;
        }
        complex_leftovers
    }

    /// Processes all `BoundarySeg`s belonging to a single `user_idx` (already sorted by `block_idx`).
    pub fn process_stitched_user_group(
        &mut self,
        user_segs: &[BoundarySeg],
        cfg: &FastSessConfig<'_>,
        complex_leftovers: &mut Vec<BoundarySeg>,
    ) {
        if user_segs.is_empty() {
            return;
        }
        let u = user_segs[0].user_idx;
        if cfg.is_user_claimed(u) {
            return;
        }

        let mut alias_mask = 0u16;
        let mut unsorted = false;
        let mut total_evs = 0usize;
        for seg in user_segs {
            alias_mask |= seg.alias_mask;
            unsorted |= seg.unsorted;
            total_evs += seg.events.len();
        }

        let mut merged_evs = Vec::with_capacity(total_evs);
        let mut prev_ts = 0u32;
        for seg in user_segs {
            for &ev in &seg.events {
                if !merged_evs.is_empty() && ev.rel_ts_ms < prev_ts {
                    unsorted = true;
                }
                prev_ts = ev.rel_ts_ms;
                merged_evs.push(ev);
            }
        }
        if unsorted {
            merged_evs.sort_by_key(|e| e.rel_ts_ms);
        }

        let ut = if (u as usize) < cfg.alias_masks.len() {
            cfg.alias_masks[u as usize]
        } else {
            IS_COMPLEX
        };

        let comp_mask = ut & ALIAS_MASK;
        let multi_idx = if (ut & IS_COMPLEX) == 0 && !cfg.multi_comp_users.is_empty() {
            cfg.multi_comp_users.binary_search_by_key(&u, |&(uid, _, _)| uid).ok()
        } else {
            None
        };
        if let Some(mi) = multi_idx {
            if cfg.try_claim_user(u) {
                let (_, comps, ncomp) = cfg.multi_comp_users[mi];
                let mut union_mask = 0u16;
                for &cm in &comps[..ncomp as usize] {
                    union_mask |= cm;
                    if (alias_mask & cm) != 0 {
                        let sub_evs: Vec<BoundaryEv> = merged_evs
                            .iter()
                            .copied()
                            .filter(|e| (cm & (1u16 << e.alias)) != 0)
                            .collect();
                        self.process_stitched_simple_user(
                            u,
                            cm,
                            alias_mask & cm,
                            &sub_evs,
                            cfg,
                        );
                    }
                }
                let mut singletons = alias_mask & !union_mask;
                while singletons != 0 {
                    let a = singletons.trailing_zeros() as u8;
                    let s_mask = 1u16 << a;
                    let sub_evs: Vec<BoundaryEv> = merged_evs
                        .iter()
                        .copied()
                        .filter(|e| e.alias == a)
                        .collect();
                    self.process_stitched_simple_user(u, s_mask, s_mask, &sub_evs, cfg);
                    singletons &= singletons - 1;
                }
            }
            return;
        }

        let simple_class = if (ut & IS_COMPLEX) != 0 {
            None
        } else if (alias_mask & !comp_mask) == 0 {
            Some(comp_mask)
        } else if alias_mask.is_power_of_two() && (alias_mask & comp_mask) == 0 {
            Some(alias_mask)
        } else {
            None
        };

        if let Some(class) = simple_class {
            if cfg.try_claim_user(u) {
                self.process_stitched_simple_user(u, class, alias_mask, &merged_evs, cfg);
            }
        } else if (ut & IS_COMPLEX) == 0 {
            // Non-complex user whose link between aliases was dropped by CSV corruption:
            // `comp_mask` is the single connected component of size >= 2 (if any), and
            // each bit in `alias_mask & !comp_mask` is an unlinked singleton person!
            if cfg.try_claim_user(u) {
                if (alias_mask & comp_mask) != 0 {
                    let sub_evs: Vec<BoundaryEv> = merged_evs
                        .iter()
                        .copied()
                        .filter(|e| (comp_mask & (1u16 << e.alias)) != 0)
                        .collect();
                    self.process_stitched_simple_user(
                        u,
                        comp_mask,
                        alias_mask & comp_mask,
                        &sub_evs,
                        cfg,
                    );
                }
                let mut singletons = alias_mask & !comp_mask;
                while singletons != 0 {
                    let a = singletons.trailing_zeros() as u8;
                    let s_mask = 1u16 << a;
                    let sub_evs: Vec<BoundaryEv> = merged_evs
                        .iter()
                        .copied()
                        .filter(|e| e.alias == a)
                        .collect();
                    self.process_stitched_simple_user(u, s_mask, s_mask, &sub_evs, cfg);
                    singletons &= singletons - 1;
                }
            }
        } else {
            complex_leftovers.push(BoundarySeg {
                block_idx: user_segs[0].block_idx,
                is_head: user_segs[0].is_head,
                is_tail: user_segs[user_segs.len() - 1].is_tail,
                user_idx: u,
                alias_mask,
                unsorted: false,
                events: merged_evs,
            });
        }
    }

    /// Processes a stitched simple user (even if they have `> 32,768` events across multiple blocks)
    /// by streaming or loading their events into `q_w0` and running `attribute_and_emit_run`.
    fn process_stitched_simple_user(
        &mut self,
        user_idx: u32,
        class: u16,
        alias_mask: u16,
        events: &[BoundaryEv],
        cfg: &FastSessConfig<'_>,
    ) {
        if events.is_empty() {
            return;
        }
        // If `events.len() <= BLOCK_EVENTS`, use the in-cache `q_w0` and `pkm` arrays directly!
        if events.len() <= BLOCK_EVENTS {
            let k = events.len();
            let words = (k + 63) >> 6;
            self.pkm[..words].fill(0);
            self.sts.clear();
            self.s_ev_start.clear();

            let mut prev_ts = events[0].rel_ts_ms;
            self.sts.push(prev_ts);
            self.s_ev_start.push(0);

            for (idx, &ev) in events.iter().enumerate() {
                self.q_w0[idx] = ev.to_w0();
                if ev.price_cents != 0 {
                    self.pkm[idx >> 6] |= 1u64 << (idx & 63);
                }
                if idx > 0 && ev.rel_ts_ms - prev_ts > SESSION_GAP_MS {
                    self.sts.push(ev.rel_ts_ms);
                    self.s_ev_start.push(idx as u16);
                }
                prev_ts = ev.rel_ts_ms;
            }
            self.s_ev_start.push(k as u16);

            let run = UserRun {
                user_idx,
                ev_start: 0,
                ev_end: k as u16,
                sess_start: 0,
                sess_end: self.sts.len() as u32,
                alias_mask,
                unsorted: false,
            };
            self.attribute_and_emit_run(run, class, cfg);
        } else {
            // Whale user with > 32,768 events: split at session boundaries into <= 16,384-event chunks
            // and process through the exact same attribution logic.
            self.process_whale_simple_user(user_idx, class, alias_mask, events, cfg);
        }
    }

    fn process_whale_simple_user(
        &mut self,
        user_idx: u32,
        class: u16,
        alias_mask: u16,
        events: &[BoundaryEv],
        cfg: &FastSessConfig<'_>,
    ) {
        let fallback_alias = alias_mask.trailing_zeros() as u8;
        let eff_mask = if (class & ALIAS_MASK) != 0 && (class & alias_mask) != 0 {
            class & ALIAS_MASK
        } else {
            1u16 << fallback_alias
        };
        let person_idx = self.user_out.push_person_name(user_idx, eff_mask, cfg.user_key);
        #[cfg(test)]
        {
            let winner_tok = min_token_for_mask(user_idx, class, fallback_alias, cfg.user_key);
            self.user_out.person_tokens.push(winner_tok);
        }

        let raw_clicks = if (user_idx as usize) < cfg.click_csr.user_cap {
            cfg.click_csr.csr_slice(user_idx as usize)
        } else {
            &[]
        };
        let filtered_buf: Vec<ClickSlot>;
        let clicks: &[ClickSlot] = if raw_clicks.iter().all(|c| (class & (1u16 << c.alias())) != 0) {
            raw_clicks
        } else {
            filtered_buf = raw_clicks
                .iter()
                .copied()
                .filter(|c| (class & (1u16 << c.alias())) != 0)
                .collect();
            &filtered_buf
        };

        self.sts.clear();
        self.sday.clear();
        let nck = clicks.len();
        let days = cfg.days;
        let organic = cfg.num_campaigns;
        let mut next_click = 0usize;

        let mut cur_day = u32::MAX;
        let mut day_sessions = 0u32;
        let mut day_events = 0u32;
        let mut day_rev = 0u64;
        let mut cat_rev = [0u64; 32];
        let mut cat_mask = 0u32;

        let mut i = 0usize;
        let n = events.len();
        while i < n {
            let s_ts = events[i].rel_ts_ms;
            let s_day = fast_session_day(s_ts, cfg.ws_day_offset);
            if s_day != cur_day {
                if cur_day != u32::MAX {
                    self.emit_user_day_row(
                        person_idx,
                        cfg.first_day + cur_day as i32,
                        day_sessions,
                        day_events,
                        day_rev,
                        &mut cat_rev,
                        &mut cat_mask,
                    );
                }
                cur_day = s_day;
                day_sessions = 0;
                day_events = 0;
                day_rev = 0;
            }
            self.sts.push(s_ts);
            self.sday.push(s_day);

            let mut prev = s_ts;
            let mut count = 0u32;
            let mut sess_rev = 0u64;
            while i < n {
                let ev = events[i];
                if ev.rel_ts_ms - prev > SESSION_GAP_MS {
                    break;
                }
                prev = ev.rel_ts_ms;
                count += 1;
                i += 1;
                if ev.price_cents != 0 {
                    let price = ev.price_cents as u64;
                    let cat = ev.cat as usize;
                    sess_rev += price;
                    cat_rev[cat] += price;
                    cat_mask |= 1u32 << cat;
                }
            }

            while next_click < nck && clicks[next_click].ts() <= s_ts {
                next_click += 1;
            }
            let credited = if next_click > 0 && s_ts - clicks[next_click - 1].ts() <= WEEK_MS {
                clicks[next_click - 1].camp() as usize
            } else {
                organic
            };
            let cell = &mut self.cells[credited * days + s_day as usize];
            cell.sessions += 1;
            cell.purchasing += (sess_rev > 0) as u32;
            cell.revenue += sess_rev;

            day_sessions += 1;
            day_events += count;
            day_rev += sess_rev;
        }
        if cur_day != u32::MAX {
            self.emit_user_day_row(
                person_idx,
                cfg.first_day + cur_day as i32,
                day_sessions,
                day_events,
                day_rev,
                &mut cat_rev,
                &mut cat_mask,
            );
        }
        if nck > 0 {
            let nsess = self.sts.len();
            self.attribute_assists_gt16(0, nsess, days, clicks);
        }
    }
}

/// Computes the 16-bit mask of clicks `k in 0..nck` (`nck <= 16`) that satisfy
/// `0 <= sess_ts - ck_ts[k] <= WEEK_MS` (i.e. `sess_ts.wrapping_sub(ck_ts[k]) <= WEEK_MS`).
/// Dispatches to AVX-512 on Cascade Lake and AVX2 on Zen 3.
#[inline(always)]
pub fn compute_click_window_mask_16(sess_ts: u32, nck: usize, ck_ts: &[u32; 16]) -> u16 {
    let valid_lanes = ((1u32 << nck) - 1) as u16;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx512f") {
            return compute_click_window_mask_16_avx512(sess_ts, ck_ts) & valid_lanes;
        }
        if is_x86_feature_detected!("avx2") {
            return compute_click_window_mask_16_avx2(sess_ts, ck_ts) & valid_lanes;
        }
    }
    let mut m = 0u16;
    for k in 0..nck {
        if sess_ts.wrapping_sub(ck_ts[k]) <= WEEK_MS {
            m |= 1u16 << k;
        }
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
unsafe fn compute_click_window_mask_16_avx512(sess_ts: u32, ck_ts: &[u32; 16]) -> u16 {
    use std::arch::x86_64::*;
    let v_sess = _mm512_set1_epi32(sess_ts as i32);
    let v_ck = _mm512_loadu_si512(ck_ts.as_ptr() as *const __m512i);
    let diff = _mm512_sub_epi32(v_sess, v_ck);
    let v_week = _mm512_set1_epi32(WEEK_MS as i32);
    _mm512_cmple_epu32_mask(diff, v_week)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn compute_click_window_mask_16_avx2(sess_ts: u32, ck_ts: &[u32; 16]) -> u16 {
    use std::arch::x86_64::*;
    let v_sess = _mm256_set1_epi32(sess_ts as i32);
    let v_week = _mm256_set1_epi32(WEEK_MS as i32);

    let ck0 = _mm256_loadu_si256(ck_ts.as_ptr() as *const __m256i);
    let ck1 = _mm256_loadu_si256(ck_ts.as_ptr().add(8) as *const __m256i);

    let d0 = _mm256_sub_epi32(v_sess, ck0);
    let d1 = _mm256_sub_epi32(v_sess, ck1);

    // Unsigned `d <= WEEK_MS` in AVX2 is `min_epu32(d, WEEK_MS) == d`!
    let le0 = _mm256_cmpeq_epi32(_mm256_min_epu32(d0, v_week), d0);
    let le1 = _mm256_cmpeq_epi32(_mm256_min_epu32(d1, v_week), d1);

    let m0 = _mm256_movemask_ps(_mm256_castsi256_ps(le0)) as u16;
    let m1 = _mm256_movemask_ps(_mm256_castsi256_ps(le1)) as u16;
    m0 | (m1 << 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast_links::{
        cmp_scrambled_base36, scramble40, user_alias_token, ClickCsr, IntraEdge, UserAliasTable,
        MASK40,
    };

    #[test]
    fn test_magic_multiply_session_day_exact() {
        let ws_offset = -(WEEK_MS as i64);
        for day in 0..30u32 {
            let day_start = WEEK_MS + day * DAY_MS_U32;
            let day_end = day_start + DAY_MS_U32 - 1;
            assert_eq!(fast_session_day(day_start, ws_offset), day);
            assert_eq!(fast_session_day(day_start + 123_456, ws_offset), day);
            assert_eq!(fast_session_day(day_end, ws_offset), day);
        }
    }

    #[test]
    fn test_32k_block_dedup_and_e5_invalid_original() {
        let ukey = 0x1111_2222_3333u64 & MASK40;
        let ckey = 0x4444_5555_6666u64 & MASK40;
        let tab = UserAliasTable::new(8, ukey);
        let mut csr = ClickCsr::with_capacity(8, 0);
        assert!(csr.finalize_and_sort(ckey));
        let claimed = [AtomicU64::new(0)];

        let cfg = FastSessConfig {
            alias_masks: &tab.alias_masks,
            multi_comp_users: &tab.multi_comp_users,
            click_csr: &csr,
            claimed_users: &claimed,
            user_key: ukey,
            ws_day_offset: -(WEEK_MS as i64),
            first_day: 20_000,
            days: 14,
            num_campaigns: 4,
        };

        let mut worker = FastSessWorker::new(4, 14);
        let base_ts = WEEK_MS + 10_000;

        // Build 3 user runs in the block so user 1 is an interior run (`r == 1`):
        // - User 0 (slot 0): head boundary run
        // - User 1 (slots 10, 11, 12): interior run
        //   - Slot 10: duplicate event_id where later ingest (200) has price=500, earlier ingest (100) has price=250 -> 250 must win!
        //   - Slot 11: fixture E5! Earliest ingest (50) is INVALID, later ingest (90) is VALID -> must be killed!
        //   - Slot 12: valid view event at base_ts + 1000
        // - User 2 (slot 20): tail boundary run
        let events = vec![
            FastEvRec::new_valid(base_ts, 0, 0, 0, 0, 100, 0),
            FastEvRec::new_valid(base_ts + 100, 500, 2, 0, 1, 200, 10),
            FastEvRec::new_valid(base_ts + 100, 250, 1, 0, 1, 100, 10),
            FastEvRec::new_valid(base_ts + 500, 999, 3, 0, 1, 90, 11),
            FastEvRec::new_invalid(1, 50, 11),
            FastEvRec::new_valid(base_ts + 1000, 0, 0, 0, 1, 100, 12),
            FastEvRec::new_valid(base_ts + 2000, 0, 0, 0, 2, 100, 20),
        ];

        worker.process_block_32k(0, &events, &cfg);

        // User 1 should be claimed and emitted immediately as an interior run
        assert!(cfg.is_user_claimed(1));
        assert!(!cfg.is_user_claimed(0));
        assert!(!cfg.is_user_claimed(2));
        assert_eq!(worker.user_out.rows.len(), 1);
        let row = worker.user_out.rows[0];
        assert_eq!(row.day, 20_000);
        assert_eq!(row.sessions, 1);
        // Slot 10 (valid winner) + Slot 12 (valid) = 2 events (Slot 11 was killed by invalid earliest copy!)
        assert_eq!(row.events, 2);
        assert_eq!(row.revenue_cents, 250);
        assert_eq!(row.top_category, Some(1));
    }

    #[test]
    fn test_session_boundary_7day_attribution_and_category_tie_break() {
        let ukey = 0x7777_8888_9999u64 & MASK40;
        let ckey = 0xAAAA_BBBB_CCCCu64 & MASK40;
        let mut tab = UserAliasTable::new(4, ukey);
        // Connect aliases 0 and 1 for user 1
        tab.build_from_intra_and_complex(&[IntraEdge::new(1, 0, 1)], &[], &[]);

        // Set up ClickCsr for user 1:
        // Click 0 & Click 1 share the exact same timestamp `t_click = WEEK_MS + 1000` with DIFFERENT campaigns (1 vs 2)
        // Click 2 is for campaign 1 again at `WEEK_MS + 1500` (to test duplicate campaign assist deduplication!)
        // Click 3 is at `WEEK_MS + 2000`, campaign 3, and we'll test a session exactly 7 days later (`+ WEEK_MS`) vs `7 days + 1 ms`!
        let mut csr = ClickCsr::with_capacity(4, 4);
        let t_click = WEEK_MS + 1_000;
        csr.put_click(0, t_click, 1, 0, 1);
        csr.put_click(1, t_click, 2, 1, 1);
        csr.put_click(2, t_click + 500, 1, 0, 1);
        csr.put_click(3, t_click + 2_000_000, 3, 1, 1);
        assert!(csr.finalize_and_sort(ckey));

        // Determine which of click_idx 0 (camp 1) or click_idx 1 (camp 2) has the smaller base36 click_id
        let cid0 = scramble40(0, ckey);
        let cid1 = scramble40(1, ckey);
        let tie_winner_camp = if cmp_scrambled_base36(cid0, cid1) == std::cmp::Ordering::Less {
            1usize
        } else {
            2usize
        };

        let claimed = [AtomicU64::new(0)];
        let cfg = FastSessConfig {
            alias_masks: &tab.alias_masks,
            multi_comp_users: &tab.multi_comp_users,
            click_csr: &csr,
            claimed_users: &claimed,
            user_key: ukey,
            ws_day_offset: -(WEEK_MS as i64),
            first_day: 20_000,
            days: 14,
            num_campaigns: 4,
        };

        let mut worker = FastSessWorker::new(4, 14);
        // Session 1 for user 1:
        // - event at `t_click + 100` (alias 0, purchase 300 cents in cat 5)
        // - event at `t_click + 100 + 1_800_000` (exact 30-min gap -> STILL SESSION 1! alias 1, purchase 300 cents in cat 2)
        //   -> Session 1 has tied category revenue (300 in cat 5, 300 in cat 2) -> cat 2 MUST win tie-break!
        //   -> Session 1 is at `t_click + 100`, so only Click 0 and Click 1 are `<= s_ts` (at `t_click`).
        //      Credited campaign MUST be `tie_winner_camp`, and both camp 1 and camp 2 get +1 assisted!
        // Session 2 for user 1:
        // - event at `t_click + 100 + 1_800_000 + 1_800_001` (30m + 1ms gap -> NEW SESSION 2 on Day 0!)
        //   -> At `t_click + 3_600_101`, Clicks 0, 1, 2, 3 are all in window!
        //      Credited campaign is Click 3 (camp 3). Assisted campaigns on Day 0: camp 1 (deduped across Click 0 & 2!), camp 2, camp 3.
        // Session 3 for user 1:
        // - event at `(t_click + 2_000_000) + WEEK_MS` (Day 7: EXACTLY 7 days after Click 3 -> Click 3 still reaches!)
        // Session 4 for user 1:
        // - event at `(t_click + 2_000_000) + WEEK_MS + 1_800_001` (Day 7: > 7 days after all clicks -> ORGANIC!)
        let s1_e0 = t_click + 100;
        let s1_e1 = s1_e0 + SESSION_GAP_MS;
        let s2_e0 = s1_e1 + SESSION_GAP_MS + 1;
        let s3_e0 = (t_click + 2_000_000) + WEEK_MS;
        let s4_e0 = s3_e0 + SESSION_GAP_MS + 1;

        let events = vec![
            FastEvRec::new_valid(t_click, 0, 0, 0, 0, 100, 0),
            FastEvRec::new_valid(s1_e0, 300, 5, 0, 1, 100, 1),
            FastEvRec::new_valid(s1_e1, 300, 2, 1, 1, 100, 2),
            FastEvRec::new_valid(s2_e0, 0, 0, 0, 1, 100, 3),
            FastEvRec::new_valid(s3_e0, 100, 4, 0, 1, 100, 4),
            FastEvRec::new_valid(s4_e0, 0, 0, 1, 1, 100, 5),
            FastEvRec::new_valid(s4_e0 + 1000, 0, 0, 0, 2, 100, 6),
        ];

        worker.process_block_32k(0, &events, &cfg);

        // Verify winning user_id token for user 1 is min(token(1, 0), token(1, 1))
        let expected_tok = user_alias_token(1, 0, ukey).min(user_alias_token(1, 1, ukey));
        assert_eq!(worker.user_out.person_tokens, vec![expected_tok]);

        // Verify user_daily rows for user 1 (Day 0 and Day 7)
        assert_eq!(worker.user_out.rows.len(), 2);
        let r_day0 = worker.user_out.rows[0];
        assert_eq!(r_day0.day, 20_000);
        assert_eq!(r_day0.sessions, 2);
        assert_eq!(r_day0.events, 3);
        assert_eq!(r_day0.revenue_cents, 600);
        assert_eq!(r_day0.top_category, Some(2)); // cat 2 beats cat 5 on 300 vs 300 tie!

        let r_day7 = worker.user_out.rows[1];
        assert_eq!(r_day7.day, 20_007);
        assert_eq!(r_day7.sessions, 2);
        assert_eq!(r_day7.events, 2);
        assert_eq!(r_day7.revenue_cents, 100);
        assert_eq!(r_day7.top_category, Some(4));

        // Verify campaign_daily cells
        let cells = worker.finalize_cells();
        let cell_at = |camp: usize, day: usize| cells[camp * 14 + day];

        // On Day 0:
        // Session 1 credited to `tie_winner_camp` (purchasing=1, rev=600); assisted: camp 1 and camp 2
        // Session 2 credited to camp 3 (purchasing=0, rev=0); assisted: camp 1 (deduped!), camp 2, camp 3
        assert_eq!(cell_at(tie_winner_camp, 0).sessions, 1);
        assert_eq!(cell_at(tie_winner_camp, 0).purchasing, 1);
        assert_eq!(cell_at(tie_winner_camp, 0).revenue, 600);
        assert_eq!(cell_at(1, 0).assisted, 2);
        assert_eq!(cell_at(2, 0).assisted, 2);
        assert_eq!(cell_at(3, 0).sessions, 1);
        assert_eq!(cell_at(3, 0).assisted, 1);

        // On Day 7:
        // Session 3 credited to camp 3 (exact 7-day reach!), purchasing=1, rev=100, assisted=1
        // Session 4 credited to organic (camp 4), sessions=1, purchasing=0
        assert_eq!(cell_at(3, 7).sessions, 1);
        assert_eq!(cell_at(3, 7).purchasing, 1);
        assert_eq!(cell_at(3, 7).revenue, 100);
        assert_eq!(cell_at(3, 7).assisted, 1);
        assert_eq!(cell_at(4, 7).sessions, 1);
    }

    #[test]
    fn test_cross_block_boundary_stitching() {
        let ukey = 0x1357_9BDF_2468u64 & MASK40;
        let ckey = 0x2468_ACE0_1357u64 & MASK40;
        let mut tab = UserAliasTable::new(4, ukey);
        tab.build_from_intra_and_complex(&[IntraEdge::new(1, 0, 1)], &[], &[]);
        let mut csr = ClickCsr::with_capacity(4, 0);
        assert!(csr.finalize_and_sort(ckey));
        let claimed = [AtomicU64::new(0)];

        let cfg = FastSessConfig {
            alias_masks: &tab.alias_masks,
            multi_comp_users: &tab.multi_comp_users,
            click_csr: &csr,
            claimed_users: &claimed,
            user_key: ukey,
            ws_day_offset: -(WEEK_MS as i64),
            first_day: 20_000,
            days: 14,
            num_campaigns: 2,
        };

        let mut worker = FastSessWorker::new(2, 14);
        let t0 = WEEK_MS + 5_000;

        // Block 0: user 0 (head) and user 1 (tail: 2 events at t0 and t0 + 10_000)
        let block0 = vec![
            FastEvRec::new_valid(t0, 0, 0, 0, 0, 100, 0),
            FastEvRec::new_valid(t0, 150, 3, 0, 1, 100, 32766),
            FastEvRec::new_valid(t0 + 10_000, 0, 0, 1, 1, 100, 32767),
        ];
        // Block 1: user 1 continues at head (1 event at t0 + 20_000, within 30 min of block 0's tail!) and user 2 (tail)
        let block1 = vec![
            FastEvRec::new_valid(t0 + 20_000, 250, 3, 1, 1, 100, 0),
            FastEvRec::new_valid(t0 + 50_000, 0, 0, 0, 2, 100, 1),
        ];

        worker.process_block_32k(0, &block0, &cfg);
        worker.process_block_32k(1, &block1, &cfg);

        // Neither block 0 nor block 1 had interior runs, so 0 rows emitted before Stage 3B
        assert_eq!(worker.user_out.rows.len(), 0);

        let segs = std::mem::take(&mut worker.boundary_segs);
        let leftovers = worker.stitch_and_process_boundaries(segs, &cfg);
        assert!(leftovers.is_empty());

        // Now users 0, 1, 2 are all claimed and stitched!
        // User 1's 3 events across block 0 and block 1 merged into a SINGLE session with 3 events and 400 revenue_cents!
        assert!(cfg.is_user_claimed(0));
        assert!(cfg.is_user_claimed(1));
        assert!(cfg.is_user_claimed(2));
        assert_eq!(worker.user_out.rows.len(), 3);
        let u1_row = worker.user_out.rows[1];
        assert_eq!(u1_row.sessions, 1);
        assert_eq!(u1_row.events, 3);
        assert_eq!(u1_row.revenue_cents, 400);
        assert_eq!(u1_row.top_category, Some(3));
    }

    #[test]
    fn test_gt16_clicks_and_full_pipeline_benchmark() {
        let ukey = 0x9999_AAAA_BBBBu64 & MASK40;
        let ckey = 0xCCCC_DDDD_EEEEu64 & MASK40;
        let num_users = 2048usize;
        let mut tab = UserAliasTable::new(num_users, ukey);
        let mut intra = Vec::with_capacity(num_users * 2);
        for u in 0..num_users as u32 {
            intra.push(IntraEdge::new(u, 0, 1));
            intra.push(IntraEdge::new(u, 1, 2));
        }
        let t0 = std::time::Instant::now();
        tab.build_from_intra_and_complex(&intra, &[], &[]);
        let link_us = t0.elapsed().as_secs_f64() * 1e6;

        // Give user 1 exactly 20 clicks (> 16 clicks path) and the rest 4 clicks each
        let total_clicks = 20 + (num_users - 1) * 4;
        let mut csr = ClickCsr::with_capacity(num_users, total_clicks);
        let mut c_idx = 0usize;
        for u in 0..num_users as u32 {
            let nck = if u == 1 { 20 } else { 4 };
            for k in 0..nck {
                let ts = WEEK_MS + (k as u32) * 60_000;
                let camp = (k % 5) as u32;
                csr.put_click(c_idx, ts, camp, (k % 3) as u8, u);
                c_idx += 1;
            }
        }
        let t1 = std::time::Instant::now();
        assert!(csr.finalize_and_sort(ckey));
        let csr_us = t1.elapsed().as_secs_f64() * 1e6;

        // Build one full 32,768-event block across users 0..2048 (16 events per user)
        let mut block_evs = Vec::with_capacity(BLOCK_EVENTS);
        for u in 0..num_users as u32 {
            for e in 0..16u32 {
                let slot = (u * 16 + e) as u16;
                // 2 sessions per user (8 events in session 0, 8 events in session 1)
                let ts = WEEK_MS + if e < 8 { e * 10_000 } else { 3_600_000 + e * 10_000 };
                let price = if e == 3 || e == 11 { 500 } else { 0 };
                let cat = ((u + e) % 10) as u8;
                let alias = (e % 3) as u8;
                block_evs.push(FastEvRec::new_valid(ts, price, cat, alias, u, 100, slot));
            }
        }

        let claimed: Vec<AtomicU64> = (0..(num_users.div_ceil(64))).map(|_| AtomicU64::new(0)).collect();
        let cfg = FastSessConfig {
            alias_masks: &tab.alias_masks,
            multi_comp_users: &tab.multi_comp_users,
            click_csr: &csr,
            claimed_users: &claimed,
            user_key: ukey,
            ws_day_offset: -(WEEK_MS as i64),
            first_day: 20_000,
            days: 14,
            num_campaigns: 8,
        };

        let mut worker = FastSessWorker::new(8, 14);
        let t2 = std::time::Instant::now();
        worker.process_block_32k(0, &block_evs, &cfg);
        let segs = std::mem::take(&mut worker.boundary_segs);
        let leftovers = worker.stitch_and_process_boundaries(segs, &cfg);
        let sess_us = t2.elapsed().as_secs_f64() * 1e6;

        assert!(leftovers.is_empty());
        assert_eq!(worker.user_out.rows.len(), num_users);
        let cells = worker.finalize_cells();
        let total_sessions: u32 = cells.iter().map(|c| c.sessions).sum();
        assert_eq!(total_sessions, (num_users * 2) as u32);
        eprintln!(
            "[fast-core bench] 2048 users: link_resolve={link_us:.1} us, click_csr_sort({total_clicks} clicks)={csr_us:.1} us, 32K_block_dedup_and_sessionize={sess_us:.1} us"
        );
    }

    #[test]
    fn test_compact_and_identify_avx2_and_push_person_name_exact() {
        let mut w_scalar = FastSessWorker::new(4, 14);
        let mut w_avx2 = FastSessWorker::new(4, 14);
        let gen = 8u16;
        w_scalar.gen = gen;
        w_avx2.gen = gen;

        // Populate > 1024 live events to exercise multi-tile flushing across L1 tile boundaries:
        // - slots 0..1200: dense normal valid events (hits `live_perm == 0xF && odd_perm == 0` fast path,
        //   multiple users, session gaps, unsorted timestamps, and purchase bits across u64 pkm boundaries)
        // - slots 1200..1500: sparse pattern (holes, INVALID_W0, odd-user events, stale-gen tags)
        // - slots 32760..32768: tail dense 4-slot groups
        for idx in 0..BLOCK_EVENTS {
            let tag = ((idx as u16) & 7).wrapping_add(gen);
            let rec = if idx < 1200 || idx >= 32760 {
                let price = if idx % 5 == 0 { 199 + idx as u32 } else { 0 };
                let u = (idx / 5) as u32;
                let ts_off = if idx % 17 == 13 { 0 } else { (idx as u32) * 400_000 };
                let mut r = FastEvRec::new_valid(
                    WEEK_MS + ts_off,
                    price,
                    (idx % 17) as u8,
                    (idx % 14) as u8,
                    u,
                    100,
                    idx as u16,
                );
                r.w1 = (r.w1 & !0xFFFF) | (tag as u64);
                r
            } else if idx < 1500 {
                match idx % 7 {
                    0 => {
                        let mut r = FastEvRec::new_invalid(idx as u32, 50, idx as u16);
                        r.w1 = (r.w1 & !0xFFFF) | (tag as u64);
                        r
                    }
                    1 => {
                        let mut r = FastEvRec::new_odd_user(WEEK_MS + 1000, 500, 3, idx as u32, 60, idx as u16);
                        r.w1 = (r.w1 & !0xFFFF) | (tag as u64);
                        r
                    }
                    2 | 3 => {
                        let mut r = FastEvRec::new_valid(
                            WEEK_MS + (idx as u32) * 50_000,
                            if idx % 2 == 0 { 250 } else { 0 },
                            2,
                            1,
                            (idx / 3) as u32,
                            70,
                            idx as u16,
                        );
                        r.w1 = (r.w1 & !0xFFFF) | (tag as u64);
                        r
                    }
                    _ => FastEvRec::default(), // stale gen
                }
            } else {
                FastEvRec::default()
            };
            w_scalar.slots[idx] = rec;
            w_avx2.slots[idx] = rec;
        }

        w_scalar.sts.clear();
        w_scalar.s_ev_start.clear();
        w_scalar.runs.clear();
        w_avx2.sts.clear();
        w_avx2.s_ev_start.clear();
        w_avx2.runs.clear();

        let k_scalar = w_scalar.compact_and_identify_scalar(gen);
        #[cfg(target_arch = "x86_64")]
        let k_avx2 = unsafe { w_avx2.compact_and_identify_avx2(gen) };
        #[cfg(not(target_arch = "x86_64"))]
        let k_avx2 = w_avx2.compact_and_identify_scalar(gen);

        assert!(k_scalar > TILE_FLUSH, "expected >1 tile to verify boundary continuity");
        assert_eq!(k_avx2, k_scalar);
        assert_eq!(&w_avx2.q_w0[..k_avx2], &w_scalar.q_w0[..k_scalar]);
        let words = k_scalar.div_ceil(64);
        assert_eq!(&w_avx2.pkm[..words], &w_scalar.pkm[..words]);
        assert_eq!(w_avx2.kept_odd_events, w_scalar.kept_odd_events);
        assert_eq!(w_avx2.sts, w_scalar.sts);
        assert_eq!(w_avx2.s_ev_start, w_scalar.s_ev_start);
        assert_eq!(w_avx2.runs, w_scalar.runs);

        // Verify FastUserBatch::push_person_name against encode_b36_min_alias
        let ukey = 0x1234_5678_9ABC_DEF0u64 & MASK40;
        let mut batch = FastUserBatch::new();
        for (i, &mask) in [1u16, 2, 8, 1 << 13, 0b11, 0b10101, 0x3FFF].iter().enumerate() {
            let pidx = batch.push_person_name((i as u32) * 12345 + 7, mask, ukey);
            assert_eq!(pidx as usize, i);
            let mut ref_buf = [0u8; 9];
            let expected = crate::unscramble::encode_b36_min_alias((i as u32) * 12345 + 7, mask, ukey, &mut ref_buf);
            let s = batch.names_offsets[i] as usize;
            let e = batch.names_offsets[i + 1] as usize;
            assert_eq!(&batch.names_data[s..e], expected);
        }
    }
}
