use crate::csv::Reader;
use crate::diag;
use crate::dims::Dims;
use crate::fast_arena::{LinkedChunkArena, LinkedScatter};
use crate::fast_links::{
    encode_base36_token, encode_upper_k_token, user_alias_token, ClickBucketPrep, ClickCsr,
    ClickSlot, IntraEdge, UserAliasTable, HAS_EXTRA_EVENTS, IS_COMPLEX, MASK40, UB_SHIFT,
    UB_USERS,
};
use crate::fast_parquet::write_campaigns_sharded;
use crate::fast_parse::{
    scan_clicks_fast, scan_events_fast, scan_links_fast, FallbackEvRec, FastCkBatch, FastEvBatch,
    FastLkBatch,
};
use crate::fast_sess::{
    BoundaryEv, BoundarySeg, FastEvRec, FastSessConfig, FastSessWorker, BLOCK_MASK, BLOCK_SHIFT,
};
use crate::group::{self, build_cat_dict, Cell, Grouper};
use crate::links::LinkTable;
use crate::scatter::{by_partition, part_of_key, Arena, Scatter};
use crate::sys::{self, mix, par, PerThread};
use crate::ts::{DateCache, DateLut, DAY_MS};
use crate::unscramble::{recover_dataset_keys_from_slices, unscramble40, unscramble40_x8};
use crate::{CRec, Input, Mapped, PRec, Shared, C_CHUNK, P_CHUNK};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

const CX_PARTS: usize = 64;

struct FastParseWorker {
    reader: Reader,
    dates: DateCache,
    ev_batch: FastEvBatch,
    ck_batch: FastCkBatch,
    lk_batch: FastLkBatch,
    intra_buckets: Vec<Vec<IntraEdge>>,
    complex_edges: Vec<(u64, u64)>,
    fallback_clicks: Vec<CRec>,
    ev_scatter: LinkedScatter<FastEvRec>,
    ck_scatter: LinkedScatter<ClickSlot>,
    odd_user_tokens: Vec<u64>,
    odd_eid_events: Vec<FallbackEvRec>,
    parsed_ev_rows: u64,
    valid_ev_rows: u64,
    parsed_ck_rows: u64,
}

impl FastParseWorker {
    fn new(read_buf: usize, num_user_buckets: usize) -> Self {
        Self {
            reader: Reader::new(read_buf),
            dates: DateCache::new(),
            ev_batch: FastEvBatch::new(),
            ck_batch: FastCkBatch::new(),
            lk_batch: FastLkBatch::new(),
            intra_buckets: vec![Vec::new(); num_user_buckets],
            complex_edges: Vec::new(),
            fallback_clicks: Vec::new(),
            ev_scatter: LinkedScatter::new(),
            ck_scatter: LinkedScatter::new(),
            odd_user_tokens: Vec::new(),
            odd_eid_events: Vec::new(),
            parsed_ev_rows: 0,
            valid_ev_rows: 0,
            parsed_ck_rows: 0,
        }
    }
}

#[derive(Clone, Copy)]
enum WorkKind {
    Link(usize),
    Click(usize),
    Event(usize),
}

type Mark<'a> = (&'a str, f64, (f64, f64, u64), [u64; 4]);

fn log_stage(marks: &[Mark]) {
    let (name, at, ru, _) = marks[marks.len() - 1];
    let (prev_at, prev_ru) = if marks.len() > 1 {
        (marks[marks.len() - 2].1, marks[marks.len() - 2].2)
    } else {
        (0.0, (0.0, 0.0, 0))
    };
    diag::stage(name, at - prev_at, ru.0 - prev_ru.0, ru.1 - prev_ru.1);
}

#[inline(always)]
fn mark_user_complex(user_alias_table: &UserAliasTable, b36: u64, user_key: u64, user_cap: u32) {
    if b36 <= MASK40 {
        let u = (unscramble40(b36, user_key) >> 9) as usize;
        if u < user_cap as usize {
            user_alias_table.set_flags_atomic(u, IS_COMPLEX);
        }
    }
}

#[inline(always)]
unsafe fn push_odd_user_or_fallback(
    w: &mut FastParseWorker,
    tid: usize,
    ev_arena: &LinkedChunkArena<FastEvRec>,
    block: usize,
    slot: u16,
    ts: u32,
    payload: u32,
    ing_u32: u32,
    utok: u64,
    eid_b36: u64,
    uid_b36: u64,
    etok_existing: u64,
    ingest: i32,
) {
    let seq = w.odd_user_tokens.len();
    if tid < 32 && seq < (1 << 20) {
        w.odd_user_tokens.push(utok);
        let odd_handle = ((tid as u32) << 20) | (seq as u32);
        let price = payload & 0xFF_FFFF;
        let cat = (payload >> 24) as u8;
        w.ev_scatter.push(
            ev_arena,
            block,
            FastEvRec::new_odd_user(ts, price, cat, odd_handle, ing_u32, slot),
        );
    } else {
        let etok = if etok_existing != 0 {
            etok_existing
        } else {
            encode_base36_token(b'e', eid_b36)
        };
        w.odd_eid_events.push(FallbackEvRec {
            event_tok: etok,
            user_tok: utok,
            eid_b36,
            uid_b36,
            ingest,
            ts,
            payload,
            valid: true,
        });
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn push_canonical_valid_ev(
    b: &FastEvBatch,
    r: usize,
    e_idx: u64,
    u_val: u64,
    w: &mut FastParseWorker,
    tid: usize,
    ev_arena: &LinkedChunkArena<FastEvRec>,
    user_alias_table: &UserAliasTable,
    user_key: u64,
    user_cap: u32,
    event_cap: u64,
) {
    let ingest = b.ingest[r];
    if ingest == i32::MAX {
        return;
    }
    let ts = b.ts[r];
    let payload = b.payload[r];
    if e_idx < event_cap {
        let block = (e_idx >> BLOCK_SHIFT) as usize;
        let slot = (e_idx & (BLOCK_MASK as u64)) as u16;
        let ing_u32 = (ingest as i64 - 518_400).clamp(0, 0x1F_FFFF) as u32;
        let u_idx = u_val >> 9;
        let alias = u_val & 511;
        if u_idx < user_cap as u64 && alias < 14 {
            let price = payload & 0xFF_FFFF;
            let cat = (payload >> 24) as u8;
            w.ev_scatter.push(
                ev_arena,
                block,
                FastEvRec::new_valid(ts, price, cat, alias as u8, u_idx as u32, ing_u32, slot),
            );
        } else {
            mark_user_complex(user_alias_table, b.uid_b36[r], user_key, user_cap);
            let utok = if b.user_tok[r] != 0 {
                b.user_tok[r]
            } else {
                encode_base36_token(b'u', b.uid_b36[r])
            };
            push_odd_user_or_fallback(
                w,
                tid,
                ev_arena,
                block,
                slot,
                ts,
                payload,
                ing_u32,
                utok,
                b.eid_b36[r],
                b.uid_b36[r],
                b.event_tok[r],
                ingest,
            );
        }
    } else {
        let etok = if b.event_tok[r] != 0 {
            b.event_tok[r]
        } else {
            encode_base36_token(b'e', b.eid_b36[r])
        };
        let utok = if b.user_tok[r] != 0 {
            b.user_tok[r]
        } else {
            encode_base36_token(b'u', b.uid_b36[r])
        };
        w.odd_eid_events.push(FallbackEvRec {
            event_tok: etok,
            user_tok: utok,
            eid_b36: b.eid_b36[r],
            uid_b36: b.uid_b36[r],
            ingest,
            ts,
            payload,
            valid: true,
        });
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn process_ev_row_unscrambled(
    b: &FastEvBatch,
    r: usize,
    e_idx: u64,
    u_val: u64,
    w: &mut FastParseWorker,
    tid: usize,
    ev_arena: &LinkedChunkArena<FastEvRec>,
    user_alias_table: &UserAliasTable,
    user_key: u64,
    user_cap: u32,
    event_cap: u64,
) {
    let valid = (b.valid_mask & (1u64 << r)) != 0;
    if valid {
        w.valid_ev_rows += 1;
    }
    let eid_b36 = b.eid_b36[r];
    let uid_b36 = b.uid_b36[r];
    let ingest = b.ingest[r];
    if ingest == i32::MAX {
        return;
    }
    if eid_b36 <= MASK40 && e_idx < event_cap {
        let block = (e_idx >> BLOCK_SHIFT) as usize;
        let slot = (e_idx & (BLOCK_MASK as u64)) as u16;
        let ing_u32 = (ingest as i64 - 518_400).clamp(0, 0x1F_FFFF) as u32;
        if !valid {
            w.ev_scatter
                .push(ev_arena, block, FastEvRec::new_invalid(0, ing_u32, slot));
        } else if uid_b36 <= MASK40 {
            let u_idx = u_val >> 9;
            let alias = u_val & 511;
            let ts = b.ts[r];
            let payload = b.payload[r];
            if u_idx < user_cap as u64 && alias < 14 {
                let price = payload & 0xFF_FFFF;
                let cat = (payload >> 24) as u8;
                w.ev_scatter.push(
                    ev_arena,
                    block,
                    FastEvRec::new_valid(
                        ts,
                        price,
                        cat,
                        alias as u8,
                        u_idx as u32,
                        ing_u32,
                        slot,
                    ),
                );
            } else {
                mark_user_complex(user_alias_table, uid_b36, user_key, user_cap);
                let utok = if b.user_tok[r] != 0 {
                    b.user_tok[r]
                } else {
                    encode_base36_token(b'u', uid_b36)
                };
                push_odd_user_or_fallback(
                    w,
                    tid,
                    ev_arena,
                    block,
                    slot,
                    ts,
                    payload,
                    ing_u32,
                    utok,
                    eid_b36,
                    uid_b36,
                    b.event_tok[r],
                    ingest,
                );
            }
        } else {
            let utok = b.user_tok[r];
            push_odd_user_or_fallback(
                w,
                tid,
                ev_arena,
                block,
                slot,
                b.ts[r],
                b.payload[r],
                ing_u32,
                utok,
                eid_b36,
                uid_b36,
                b.event_tok[r],
                ingest,
            );
        }
        return;
    }

    let etok = if b.event_tok[r] != 0 {
        b.event_tok[r]
    } else {
        encode_base36_token(b'e', eid_b36)
    };
    let utok = if !valid {
        0
    } else if b.user_tok[r] != 0 {
        b.user_tok[r]
    } else if uid_b36 <= MASK40 {
        encode_base36_token(b'u', uid_b36)
    } else {
        0
    };
    w.odd_eid_events.push(FallbackEvRec {
        event_tok: etok,
        user_tok: utok,
        eid_b36,
        uid_b36,
        ingest,
        ts: b.ts[r],
        payload: b.payload[r],
        valid,
    });
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn process_ev_row_scalar(
    b: &FastEvBatch,
    r: usize,
    w: &mut FastParseWorker,
    tid: usize,
    ev_arena: &LinkedChunkArena<FastEvRec>,
    user_alias_table: &UserAliasTable,
    user_key: u64,
    event_key: u64,
    user_cap: u32,
    event_cap: u64,
) {
    let eid_b36 = b.eid_b36[r];
    let uid_b36 = b.uid_b36[r];
    let e_idx = if eid_b36 <= MASK40 {
        unscramble40(eid_b36, event_key)
    } else {
        u64::MAX
    };
    let u_val = if uid_b36 <= MASK40 {
        unscramble40(uid_b36, user_key)
    } else {
        u64::MAX
    };
    process_ev_row_unscrambled(
        b,
        r,
        e_idx,
        u_val,
        w,
        tid,
        ev_arena,
        user_alias_table,
        user_key,
        user_cap,
        event_cap,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn process_ck_row_unscrambled(
    b: &FastCkBatch,
    r: usize,
    c_idx: u64,
    u_val: u64,
    w: &mut FastParseWorker,
    ck_arena: &LinkedChunkArena<ClickSlot>,
    ck_shift: u32,
    user_alias_table: &UserAliasTable,
    user_key: u64,
    user_cap: u32,
    click_cap: u64,
) {
    let is_upper_k = (b.upper_k_mask & (1u64 << r)) != 0;
    if ((b.canonical_mask & (1u64 << r)) != 0)
        || (is_upper_k && b.cid_b36[r] <= MASK40 && b.uid_b36[r] <= MASK40)
    {
        let u_idx = u_val >> 9;
        let alias = u_val & 511;
        if u_idx < user_cap as u64 && alias < 14 && c_idx < click_cap {
            w.ck_scatter.push(
                ck_arena,
                (u_idx as usize) >> ck_shift,
                ClickSlot::new_parsed_cased(
                    b.ts[r],
                    b.campaign[r],
                    alias as u8,
                    u_idx as u32,
                    b.cid_b36[r],
                    is_upper_k,
                ),
            );
            return;
        }
    }
    mark_user_complex(user_alias_table, b.uid_b36[r], user_key, user_cap);
    let ctok = if b.click_tok[r] != 0 {
        b.click_tok[r]
    } else if is_upper_k {
        encode_upper_k_token(b.cid_b36[r])
    } else {
        encode_base36_token(b'k', b.cid_b36[r])
    };
    let utok = if b.user_tok[r] != 0 {
        b.user_tok[r]
    } else {
        encode_base36_token(b'u', b.uid_b36[r])
    };
    w.fallback_clicks.push(CRec {
        person: utok,
        click: ctok,
        ts: b.ts[r],
        campaign: b.campaign[r],
    });
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn process_ck_row_scalar(
    b: &FastCkBatch,
    r: usize,
    w: &mut FastParseWorker,
    ck_arena: &LinkedChunkArena<ClickSlot>,
    ck_shift: u32,
    user_alias_table: &UserAliasTable,
    user_key: u64,
    click_key: u64,
    user_cap: u32,
    click_cap: u64,
) {
    let c_idx = if b.cid_b36[r] <= MASK40 {
        unscramble40(b.cid_b36[r], click_key)
    } else {
        u64::MAX
    };
    let u_val = if b.uid_b36[r] <= MASK40 {
        unscramble40(b.uid_b36[r], user_key)
    } else {
        u64::MAX
    };
    process_ck_row_unscrambled(
        b,
        r,
        c_idx,
        u_val,
        w,
        ck_arena,
        ck_shift,
        user_alias_table,
        user_key,
        user_cap,
        click_cap,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn process_lk_row_unscrambled(
    b: &FastLkBatch,
    r: usize,
    va: u64,
    vb: u64,
    w: &mut FastParseWorker,
    user_alias_table: &UserAliasTable,
    user_key: u64,
    user_cap: u32,
) {
    if (b.canonical_mask & (1u64 << r)) != 0 {
        let ua = va >> 9;
        let ub = vb >> 9;
        let aa = va & 511;
        let ab = vb & 511;
        if ua == ub && ua < user_cap as u64 && aa < 14 && ab < 14 {
            if aa != ab {
                w.intra_buckets[(ua as usize) >> UB_SHIFT].push(IntraEdge::new(
                    ua as u32,
                    aa as u8,
                    ab as u8,
                ));
            }
            return;
        }
    }
    mark_user_complex(user_alias_table, b.ua_b36[r], user_key, user_cap);
    mark_user_complex(user_alias_table, b.ub_b36[r], user_key, user_cap);
    let ta = if b.ua_tok[r] != 0 {
        b.ua_tok[r]
    } else {
        encode_base36_token(b'u', b.ua_b36[r])
    };
    let tb = if b.ub_tok[r] != 0 {
        b.ub_tok[r]
    } else {
        encode_base36_token(b'u', b.ub_b36[r])
    };
    if ta != tb {
        w.complex_edges.push((ta, tb));
    }
}

#[inline(always)]
fn process_lk_row_scalar(
    b: &FastLkBatch,
    r: usize,
    w: &mut FastParseWorker,
    user_alias_table: &UserAliasTable,
    user_key: u64,
    user_cap: u32,
) {
    let va = if b.ua_b36[r] <= MASK40 {
        unscramble40(b.ua_b36[r], user_key)
    } else {
        u64::MAX
    };
    let vb = if b.ub_b36[r] <= MASK40 {
        unscramble40(b.ub_b36[r], user_key)
    } else {
        u64::MAX
    };
    process_lk_row_unscrambled(b, r, va, vb, w, user_alias_table, user_key, user_cap);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_run_fast(
    t0: Instant,
    output: &Path,
    threads: usize,
    verbose: bool,
    counters: Option<&sys::Counters>,
    window_start: i64,
    window_end: i64,
    base_ms: i64,
    first_day: i64,
    days: usize,
    read_buf: usize,
    dims_loader: &mut Option<std::thread::JoinHandle<Dims>>,
    dims_fallback: &mut Option<Dims>,
    link_input: &mut Input,
    event_input: &mut Input,
    click_input: &mut Input,
) -> bool {
    // Step 0: Eligibility & Key Recovery (< 50 us fallback on non-generator datasets)
    if base_ms % 1000 != 0
        || days > 30
        || threads > 64
        || std::env::var_os("SS_NOFAST").is_some()
        || event_input.bytes < (32 << 10)
        || click_input.bytes < (4 << 10)
        || event_input.files.is_empty()
        || click_input.files.is_empty()
    {
        return false;
    }

    const SAMPLE_BYTES: usize = 512 << 10;
    let largest_slices = |files: &[Mapped]| -> Vec<&[u8]> {
        let mut by_sz: Vec<(usize, *const u8)> =
            files.iter().filter(|m| m.1 > 0).map(|m| (m.1, m.0)).collect();
        by_sz.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        by_sz
            .into_iter()
            .take(3)
            .map(|(sz, p)| unsafe { std::slice::from_raw_parts(p, sz.min(SAMPLE_BYTES)) })
            .collect()
    };
    let ev_samples = largest_slices(&event_input.files);
    let ck_samples = largest_slices(&click_input.files);
    let Some(keys) = recover_dataset_keys_from_slices(
        &ev_samples,
        &ck_samples,
        event_input.bytes,
        click_input.bytes,
    ) else {
        return false;
    };

    // Join dims now; if >32 categories (5-bit FastEvRec limit), preserve dims and fall back.
    let dims = dims_loader
        .take()
        .expect("dims_loader")
        .join()
        .expect("loading products and campaigns");
    if dims.categories.len() > 32 {
        *dims_fallback = Some(dims);
        return false;
    }

    let event_cap = keys.event_cap.max(1 << 16).min(1u64 << 30);
    let user_cap = keys.user_cap.max(1 << 16).min((1 << 23) - 1);
    let click_cap = keys.click_cap.max(1 << 16).min(1u64 << 29);
    let num_ev_blocks = ((event_cap as usize) + BLOCK_MASK) >> BLOCK_SHIFT;
    let num_user_buckets = (user_cap as usize).div_ceil(UB_USERS);

    let count = || counters.as_ref().map_or([0; 4], |c| c.read());
    let mark = |name| (name, t0.elapsed().as_secs_f64(), sys::rusage(), count());
    let mut marks: Vec<Mark> = vec![mark("setup")];
    log_stage(&marks);

    // Step 1: End-of-line checks, Shared setup, and pre-population
    for i in 0..link_input.files.len() {
        link_input.end_line(i);
    }
    let tails = event_input.files.len() + click_input.files.len();
    let next_tail = AtomicUsize::new(0);
    let unended: Vec<AtomicBool> = (0..tails).map(|_| AtomicBool::new(false)).collect();
    par(threads, |_| {
        loop {
            let i = next_tail.fetch_add(1, Relaxed);
            if i >= tails {
                break;
            }
            let lacks = match i.checked_sub(event_input.files.len()) {
                None => event_input.lacks_newline(i),
                Some(k) => click_input.lacks_newline(k),
            };
            if lacks {
                unended[i].store(true, Relaxed);
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

    let date_lut = DateLut::new(window_start, window_end, base_ms.div_euclid(1000));
    let mut sh = Shared {
        window_start,
        window_end,
        base_ms,
        use_rel: true,
        date_lut,
        first_day,
        days,
        products: dims.products,
        product_category: dims.product_category,
        direct_prod_cat: dims.direct_prod_cat,
        unknown_category: dims.unknown_category,
        categories: dims.categories,
        campaign_ids: dims.campaign_ids,
        direct_camp: dims.direct_camp,
        campaigns: dims.campaigns,
        links: LinkTable::new(16),
        d_arena: Arena::new(64, 64),
        p_arena: Arena::new(64, 64),
        c_arena: Arena::new(64, 64),
        d_parts: 1,
        p_parts: 1,
    };

    let user_alias_table = UserAliasTable::new(user_cap as usize, keys.user_key);
    let ck_shift: u32 = if (user_cap as usize) <= (1 << 20) {
        UB_SHIFT as u32
    } else {
        (user_cap as usize)
            .div_ceil(256)
            .next_power_of_two()
            .trailing_zeros()
            .max(UB_SHIFT as u32)
    };
    let ck_buckets = (user_cap as usize).div_ceil(1 << ck_shift).max(1);
    let ck_reserve = (click_input.bytes as usize * 2).max(64 << 20) + threads * (8 << 20);
    let ck_arena = LinkedChunkArena::<ClickSlot>::with_reserve(ck_buckets, ck_reserve);
    let ck_csr_cap = (click_input.bytes as usize / 32).max(1 << 16);
    let ck_csr_base =
        sys::map_zeroed_huge(ck_csr_cap * std::mem::size_of::<ClickSlot>()) as *mut ClickSlot;
    let ev_reserve = (event_input.bytes as usize * 2).max(64 << 20) + threads * (8 << 20);
    let ev_arena = LinkedChunkArena::<FastEvRec>::with_reserve(num_ev_blocks, ev_reserve);
    marks.push(mark("populate"));
    log_stage(&marks);

    // Step 2: Unified Stage 1 Parse (ALL Links first, then ALL Clicks sorted desc, then ALL Events sorted desc)
    // Overlap ClickCsr bucket construction with the tail of Stage 1 as soon as Click scatter finishes.
    let mut lk_work: Vec<(u64, WorkKind)> = (0..link_input.extents.len())
        .map(|i| {
            let x = &link_input.extents[i];
            (x.to - x.from, WorkKind::Link(i))
        })
        .collect();
    lk_work.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let mut ck_work: Vec<(u64, WorkKind)> = (0..click_input.extents.len())
        .map(|i| {
            let x = &click_input.extents[i];
            (x.to - x.from, WorkKind::Click(i))
        })
        .collect();
    ck_work.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let mut ev_work: Vec<(u64, WorkKind)> = (0..event_input.extents.len())
        .map(|i| {
            let x = &event_input.extents[i];
            (x.to - x.from, WorkKind::Event(i))
        })
        .collect();
    ev_work.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut work: Vec<WorkKind> = Vec::with_capacity(lk_work.len() + ck_work.len() + ev_work.len());
    work.extend(lk_work.into_iter().map(|(_, k)| k));
    work.extend(ck_work.into_iter().map(|(_, k)| k));
    let num_lk_ck = work.len();
    work.extend(ev_work.into_iter().map(|(_, k)| k));

    struct SyncCell<T>(std::cell::UnsafeCell<T>);
    unsafe impl<T> Sync for SyncCell<T> {}
    impl<T> SyncCell<T> {
        #[inline(always)]
        unsafe fn set(&self, v: T) {
            *self.0.get() = v;
        }
        #[inline(always)]
        unsafe fn as_ref(&self) -> &T {
            &*self.0.get()
        }
    }

    let parse_workers = PerThread::new(threads, |_| FastParseWorker::new(read_buf, num_user_buckets));
    let next_work = AtomicUsize::new(0);
    let ck_flushed_threads = AtomicUsize::new(0);
    let ck_ready = AtomicBool::new(false);
    let ck_prep_cell: SyncCell<Option<ClickBucketPrep>> = SyncCell(std::cell::UnsafeCell::new(None));
    let ck_prep_ref = &ck_prep_cell;
    let ck_csr_addr = ck_csr_base as usize;
    let next_ck_job = AtomicUsize::new(0);
    let ck_n_jobs = AtomicUsize::new(0);
    let ck_worker_whales: Vec<std::sync::Mutex<Vec<(usize, usize)>>> =
        (0..threads).map(|_| std::sync::Mutex::new(Vec::new())).collect();

    par(threads, |tid| unsafe {
        let w = parse_workers.get(tid);
        w.ev_scatter.reset(num_ev_blocks);
        w.ck_scatter.reset(ck_buckets);
        let mut ck_flushed_local = false;
        let mut ck_counts: Vec<u32> = Vec::new();
        let mut local_ck_whales: Vec<(usize, usize)> = Vec::new();

        let flush_clicks = |w: &mut FastParseWorker, ck_flushed_local: &mut bool| {
            if !*ck_flushed_local {
                w.ck_scatter.finish(&ck_arena);
                *ck_flushed_local = true;
                if ck_flushed_threads.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1 == threads {
                    let prep = ClickCsr::prepare_contiguous_buckets_linked(
                        &ck_arena,
                        ck_shift,
                        user_cap as usize,
                        ck_csr_cap,
                    );
                    ck_n_jobs.store(prep.bucket_order.len(), Relaxed);
                    ck_prep_ref.set(Some(prep));
                    ck_ready.store(true, std::sync::atomic::Ordering::Release);
                }
            }
        };

        loop {
            let wi = next_work.fetch_add(1, Relaxed);
            if wi >= num_lk_ck && !ck_flushed_local {
                flush_clicks(w, &mut ck_flushed_local);
            }
            let Some(&kind) = work.get(wi) else {
                if ck_ready.load(std::sync::atomic::Ordering::Acquire) {
                    if let Some(prep) = ck_prep_ref.as_ref().as_ref() {
                        let nj = ck_n_jobs.load(Relaxed);
                        if ck_counts.is_empty() && nj > 0 {
                            ck_counts.resize(1usize << ck_shift, 0);
                        }
                        loop {
                            let ji = next_ck_job.fetch_add(1, Relaxed);
                            if ji >= nj {
                                break;
                            }
                            ClickCsr::process_contiguous_bucket_linked(
                                prep,
                                ji,
                                &ck_arena,
                                ck_csr_addr as *mut ClickSlot,
                                ck_shift,
                                user_cap as usize,
                                &mut ck_counts,
                                &mut local_ck_whales,
                            );
                        }
                    }
                }
                break;
            };
            match kind {
                WorkKind::Link(i) => {
                    let x = &link_input.extents[i];
                    let Mapped(data, size) = link_input.files[x.file];
                    let mut reader = std::mem::replace(&mut w.reader, Reader::new(0));
                    let mut lk_batch = std::mem::take(&mut w.lk_batch);
                    scan_links_fast(
                        &mut reader,
                        data,
                        size,
                        x.from as usize,
                        x.to as usize,
                        &mut lk_batch,
                        |b| {
                            let n = b.len;
                            let mut idx = 0;
                            while idx + 8 <= n {
                                let a_un = unscramble40_x8(
                                    *(b.ua_b36.as_ptr().add(idx) as *const [u64; 8]),
                                    keys.user_key,
                                );
                                let b_un = unscramble40_x8(
                                    *(b.ub_b36.as_ptr().add(idx) as *const [u64; 8]),
                                    keys.user_key,
                                );
                                if ((b.canonical_mask >> idx) & 0xFF) == 0xFF {
                                    for k in 0..8 {
                                        let r = idx + k;
                                        let va = a_un[k];
                                        let vb = b_un[k];
                                        let ua = va >> 9;
                                        let ub = vb >> 9;
                                        let aa = va & 511;
                                        let ab = vb & 511;
                                        if ua == ub && ua < user_cap as u64 && aa < 14 && ab < 14 {
                                            if aa != ab {
                                                w.intra_buckets[(ua as usize) >> UB_SHIFT].push(
                                                    IntraEdge::new(ua as u32, aa as u8, ab as u8),
                                                );
                                            }
                                        } else {
                                            mark_user_complex(
                                                &user_alias_table,
                                                b.ua_b36[r],
                                                keys.user_key,
                                                user_cap,
                                            );
                                            mark_user_complex(
                                                &user_alias_table,
                                                b.ub_b36[r],
                                                keys.user_key,
                                                user_cap,
                                            );
                                            let ta = if b.ua_tok[r] != 0 {
                                                b.ua_tok[r]
                                            } else {
                                                encode_base36_token(b'u', b.ua_b36[r])
                                            };
                                            let tb = if b.ub_tok[r] != 0 {
                                                b.ub_tok[r]
                                            } else {
                                                encode_base36_token(b'u', b.ub_b36[r])
                                            };
                                            if ta != tb {
                                                w.complex_edges.push((ta, tb));
                                            }
                                        }
                                    }
                                } else {
                                    for k in 0..8 {
                                        process_lk_row_unscrambled(
                                            b,
                                            idx + k,
                                            a_un[k],
                                            b_un[k],
                                            w,
                                            &user_alias_table,
                                            keys.user_key,
                                            user_cap,
                                        );
                                    }
                                }
                                idx += 8;
                            }
                            while idx < n {
                                process_lk_row_scalar(
                                    b,
                                    idx,
                                    w,
                                    &user_alias_table,
                                    keys.user_key,
                                    user_cap,
                                );
                                idx += 1;
                            }
                        },
                    );
                    w.reader = reader;
                    w.lk_batch = lk_batch;
                    let _ = x;
                }
                WorkKind::Click(i) => {
                    let x = &click_input.extents[i];
                    let Mapped(data, size) = click_input.files[x.file];
                    let mut reader = std::mem::replace(&mut w.reader, Reader::new(0));
                    let mut dates = std::mem::replace(&mut w.dates, DateCache::new());
                    let mut ck_batch = std::mem::take(&mut w.ck_batch);
                    scan_clicks_fast(
                        &mut reader,
                        &sh,
                        &mut dates,
                        data,
                        size,
                        x.from as usize,
                        x.to as usize,
                        &mut ck_batch,
                        |b| {
                            let n = b.len;
                            w.parsed_ck_rows += n as u64;
                            let mut idx = 0;
                            while idx + 8 <= n {
                                let c_un = unscramble40_x8(
                                    *(b.cid_b36.as_ptr().add(idx) as *const [u64; 8]),
                                    keys.click_key,
                                );
                                let u_un = unscramble40_x8(
                                    *(b.uid_b36.as_ptr().add(idx) as *const [u64; 8]),
                                    keys.user_key,
                                );
                                if ((b.canonical_mask >> idx) & 0xFF) == 0xFF {
                                    for k in 0..8 {
                                        let r = idx + k;
                                        let c_idx = c_un[k];
                                        let u_val = u_un[k];
                                        let u_idx = u_val >> 9;
                                        let alias = u_val & 511;
                                        if u_idx < user_cap as u64
                                            && alias < 14
                                            && c_idx < click_cap
                                        {
                                            w.ck_scatter.push(
                                                &ck_arena,
                                                (u_idx as usize) >> ck_shift,
                                                ClickSlot::new_parsed_cased(
                                                    b.ts[r],
                                                    b.campaign[r],
                                                    alias as u8,
                                                    u_idx as u32,
                                                    b.cid_b36[r],
                                                    false,
                                                ),
                                            );
                                        } else {
                                            mark_user_complex(
                                                &user_alias_table,
                                                b.uid_b36[r],
                                                keys.user_key,
                                                user_cap,
                                            );
                                            let ctok = if b.click_tok[r] != 0 {
                                                b.click_tok[r]
                                            } else {
                                                encode_base36_token(b'k', b.cid_b36[r])
                                            };
                                            let utok = if b.user_tok[r] != 0 {
                                                b.user_tok[r]
                                            } else {
                                                encode_base36_token(b'u', b.uid_b36[r])
                                            };
                                            w.fallback_clicks.push(CRec {
                                                person: utok,
                                                click: ctok,
                                                ts: b.ts[r],
                                                campaign: b.campaign[r],
                                            });
                                        }
                                    }
                                } else {
                                    for k in 0..8 {
                                        process_ck_row_unscrambled(
                                            b,
                                            idx + k,
                                            c_un[k],
                                            u_un[k],
                                            w,
                                            &ck_arena,
                                            ck_shift,
                                            &user_alias_table,
                                            keys.user_key,
                                            user_cap,
                                            click_cap,
                                        );
                                    }
                                }
                                idx += 8;
                            }
                            while idx < n {
                                process_ck_row_scalar(
                                    b,
                                    idx,
                                    w,
                                    &ck_arena,
                                    ck_shift,
                                    &user_alias_table,
                                    keys.user_key,
                                    keys.click_key,
                                    user_cap,
                                    click_cap,
                                );
                                idx += 1;
                            }
                        },
                    );
                    w.reader = reader;
                    w.dates = dates;
                    w.ck_batch = ck_batch;
                }
                WorkKind::Event(i) => {
                    let x = &event_input.extents[i];
                    let Mapped(data, size) = event_input.files[x.file];
                    let mut reader = std::mem::replace(&mut w.reader, Reader::new(0));
                    let mut dates = std::mem::replace(&mut w.dates, DateCache::new());
                    let mut ev_batch = std::mem::take(&mut w.ev_batch);
                    scan_events_fast(
                        &mut reader,
                        &sh,
                        &mut dates,
                        data,
                        size,
                        x.from as usize,
                        x.to as usize,
                        &mut ev_batch,
                        |b| {
                            let n = b.len;
                            w.parsed_ev_rows += n as u64;
                            let mut idx = 0;
                            while idx + 8 <= n {
                                let e_un = unscramble40_x8(
                                    *(b.eid_b36.as_ptr().add(idx) as *const [u64; 8]),
                                    keys.event_key,
                                );
                                let u_un = unscramble40_x8(
                                    *(b.uid_b36.as_ptr().add(idx) as *const [u64; 8]),
                                    keys.user_key,
                                );
                                for k in 0..8 {
                                    let r = idx + k;
                                    if (b.canonical_valid_mask & (1u64 << r)) != 0 {
                                        w.valid_ev_rows += 1;
                                        push_canonical_valid_ev(
                                            b,
                                            r,
                                            e_un[k],
                                            u_un[k],
                                            w,
                                            tid,
                                            &ev_arena,
                                            &user_alias_table,
                                            keys.user_key,
                                            user_cap,
                                            event_cap,
                                        );
                                    } else {
                                        process_ev_row_unscrambled(
                                            b,
                                            r,
                                            e_un[k],
                                            u_un[k],
                                            w,
                                            tid,
                                            &ev_arena,
                                            &user_alias_table,
                                            keys.user_key,
                                            user_cap,
                                            event_cap,
                                        );
                                    }
                                }
                                idx += 8;
                            }
                            while idx < n {
                                process_ev_row_scalar(
                                    b,
                                    idx,
                                    w,
                                    tid,
                                    &ev_arena,
                                    &user_alias_table,
                                    keys.user_key,
                                    keys.event_key,
                                    user_cap,
                                    event_cap,
                                );
                                idx += 1;
                            }
                        },
                    );
                    w.reader = reader;
                    w.dates = dates;
                    w.ev_batch = ev_batch;
                }
            }
        }
        w.ev_scatter.finish(&ev_arena);
        flush_clicks(w, &mut ck_flushed_local);
        if !local_ck_whales.is_empty() {
            *ck_worker_whales[tid].lock().unwrap() = local_ck_whales;
        }
    });
    marks.push(mark("parse"));
    log_stage(&marks);

    // Step 3: Stage 2 Links + Clicks + Odd-EID Dedup
    let mut user_alias_table = user_alias_table;
    let mut all_odd_eid: Vec<FallbackEvRec> = Vec::new();
    let mut total_ev_rows = 0u64;
    let mut total_valid_rows = 0u64;
    let mut total_ck_rows = 0u64;
    for t in 0..threads {
        let w = unsafe { parse_workers.get(t) };
        total_ev_rows += w.parsed_ev_rows;
        total_valid_rows += w.valid_ev_rows;
        total_ck_rows += w.parsed_ck_rows;
        if !w.odd_eid_events.is_empty() {
            all_odd_eid.extend_from_slice(&w.odd_eid_events);
            w.odd_eid_events = Vec::new();
        }
        if !w.complex_edges.is_empty() {
            user_alias_table.complex_edges.extend_from_slice(&w.complex_edges);
            w.complex_edges = Vec::new();
        }
    }

    let mut deduped_odd_eid_events: Vec<FallbackEvRec> = Vec::new();
    if !all_odd_eid.is_empty() {
        all_odd_eid.sort_unstable_by_key(|r| (r.event_tok, r.ingest));
        let mut i = 0;
        while i < all_odd_eid.len() {
            let winner = all_odd_eid[i];
            if winner.valid {
                deduped_odd_eid_events.push(winner);
            }
            let tok = winner.event_tok;
            i += 1;
            while i < all_odd_eid.len() && all_odd_eid[i].event_tok == tok {
                i += 1;
            }
        }
    }
    drop(all_odd_eid);

    let per_thread_buckets: Vec<Vec<Vec<IntraEdge>>> = (0..threads)
        .map(|t| std::mem::take(unsafe { &mut parse_workers.get(t).intra_buckets }))
        .collect();
    user_alias_table.resolve_thread_buckets_par(&per_thread_buckets, threads);
    drop(per_thread_buckets);

    // Route deduped valid odd-eid events with canonical fast users into Stage 3B via HAS_EXTRA_EVENTS
    // instead of marking those users IS_COMPLEX (avoids sending entire whales to the complex Grouper).
    let mut extra_odd_segs: Vec<BoundarySeg> = Vec::new();
    let mut cx_odd_eid_events: Vec<FallbackEvRec> = Vec::new();
    for ev in deduped_odd_eid_events {
        if ev.uid_b36 <= MASK40 {
            let u_val = unscramble40(ev.uid_b36, keys.user_key);
            let u_idx = u_val >> 9;
            let alias = u_val & 511;
            if u_idx < user_cap as u64
                && alias < 14
                && (user_alias_table.alias_masks[u_idx as usize] & IS_COMPLEX) == 0
            {
                user_alias_table.set_flags_atomic(u_idx as usize, HAS_EXTRA_EVENTS);
                extra_odd_segs.push(BoundarySeg {
                    block_idx: u32::MAX,
                    is_head: false,
                    is_tail: false,
                    user_idx: u_idx as u32,
                    alias_mask: 1u16 << alias,
                    unsorted: true,
                    events: vec![BoundaryEv {
                        rel_ts_ms: ev.ts,
                        price_cents: ev.payload & 0xFF_FFFF,
                        cat: (ev.payload >> 24) as u8,
                        alias: alias as u8,
                    }],
                });
                continue;
            }
        }
        mark_user_complex(&user_alias_table, ev.uid_b36, keys.user_key, user_cap);
        cx_odd_eid_events.push(ev);
    }
    let deduped_odd_eid_events = cx_odd_eid_events;
    marks.push(mark("links"));
    log_stage(&marks);

    let ck_prep = ck_prep_cell
        .0
        .into_inner()
        .expect("ClickBucketPrep initialized in Stage 1");
    let ck_nj = ck_prep.bucket_order.len();
    if next_ck_job.load(Relaxed) < ck_nj {
        let prep_ref = &ck_prep;
        let max_u_len = 1usize << ck_shift;
        let rem_whales: Vec<std::sync::Mutex<Vec<(usize, usize)>>> =
            (0..threads).map(|_| std::sync::Mutex::new(Vec::new())).collect();
        let ck_arena_ref = &ck_arena;
        par(threads, |tid| unsafe {
            let mut counts = vec![0u32; max_u_len];
            let mut local_whales: Vec<(usize, usize)> = Vec::new();
            loop {
                let ji = next_ck_job.fetch_add(1, Relaxed);
                if ji >= ck_nj {
                    break;
                }
                ClickCsr::process_contiguous_bucket_linked(
                    prep_ref,
                    ji,
                    ck_arena_ref,
                    ck_csr_addr as *mut ClickSlot,
                    ck_shift,
                    user_cap as usize,
                    &mut counts,
                    &mut local_whales,
                );
            }
            if !local_whales.is_empty() {
                *rem_whales[tid].lock().unwrap() = local_whales;
            }
        });
        for rw in rem_whales {
            ck_worker_whales[0]
                .lock()
                .unwrap()
                .extend(rw.into_inner().unwrap());
        }
    }
    let mut all_ck_whales: Vec<(usize, usize)> = Vec::new();
    for w in ck_worker_whales {
        all_ck_whales.extend(w.into_inner().unwrap());
    }
    let click_csr = unsafe {
        ClickCsr::finish_contiguous_buckets(
            ck_prep,
            ck_csr_base,
            user_cap as usize,
            &all_ck_whales,
            threads,
        )
    };
    marks.push(mark("clicks"));
    log_stage(&marks);

    // Step 4: Stage 3A (16K L2-Resident Block Dedup & Sessionize) + Stage 3B (Parallel Boundary Stitch)
    let claimed_users: Vec<std::sync::atomic::AtomicU64> = (0..(user_cap as usize).div_ceil(64))
        .map(|_| std::sync::atomic::AtomicU64::new(0))
        .collect();

    let cfg = FastSessConfig {
        alias_masks: &user_alias_table.alias_masks,
        multi_comp_users: &user_alias_table.multi_comp_users,
        click_csr: &click_csr,
        claimed_users: &claimed_users,
        user_key: keys.user_key,
        ws_day_offset: base_ms - first_day * DAY_MS,
        first_day: first_day as i32,
        days,
        num_campaigns: sh.campaigns.len(),
    };

    let sess_workers = PerThread::new(threads, |_| FastSessWorker::new(sh.campaigns.len(), days));
    let ev_arena_ref = &ev_arena;
    let next_block = AtomicUsize::new(0);
    par(threads, |t| unsafe {
        let sw = sess_workers.get(t);
        loop {
            let b = next_block.fetch_add(1, Relaxed);
            if b >= num_ev_blocks {
                break;
            }
            if !ev_arena_ref.is_partition_empty(b) {
                sw.process_block_linked(b as u32, ev_arena_ref.chunks(b), &cfg);
            }
        }
    });
    marks.push(mark("sess3a"));
    log_stage(&marks);

    let mut all_segs: Vec<BoundarySeg> = extra_odd_segs;
    for t in 0..threads {
        let sw = unsafe { sess_workers.get(t) };
        if !sw.boundary_segs.is_empty() {
            all_segs.append(&mut sw.boundary_segs);
        }
    }
    all_segs.sort_unstable_by_key(|s| (s.user_idx, s.block_idx));

    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    {
        let mut i = 0usize;
        while i < all_segs.len() {
            let u = all_segs[i].user_idx;
            let mut tot = all_segs[i].events.len();
            let mut j = i + 1;
            while j < all_segs.len() && all_segs[j].user_idx == u {
                tot += all_segs[j].events.len();
                j += 1;
            }
            groups.push((tot, i, j));
            i = j;
        }
    }
    groups.sort_unstable_by_key(|&(tot, _, _)| std::cmp::Reverse(tot));

    let per_thread_leftovers = PerThread::new(threads, |_| Vec::<BoundarySeg>::new());
    let next_group = AtomicUsize::new(0);
    par(threads, |t| unsafe {
        let sw = sess_workers.get(t);
        let cx_left = per_thread_leftovers.get(t);
        loop {
            let i = next_group.fetch_add(1, Relaxed);
            let Some(&(_, s, e)) = groups.get(i) else { break };
            sw.process_stitched_user_group(&all_segs[s..e], &cfg, cx_left);
        }
    });
    drop(all_segs);
    drop(groups);
    marks.push(mark("sess3b"));
    log_stage(&marks);

    // Step 5: Complex Users (< 0.5%) + Parallel Parquet Output
    let complex_links = LinkTable::new(user_alias_table.complex_edges.len());
    if !user_alias_table.complex_edges.is_empty() {
        const LINK_BLOCK: usize = 2048;
        let blocks: Vec<usize> = (0..user_alias_table.complex_edges.len())
            .step_by(LINK_BLOCK)
            .collect();
        let next_lb = AtomicUsize::new(0);
        par(threads, |_| loop {
            let i = next_lb.fetch_add(1, Relaxed);
            let Some(&from) = blocks.get(i) else { break };
            let to = (from + LINK_BLOCK).min(user_alias_table.complex_edges.len());
            unsafe { complex_links.link_all(&user_alias_table.complex_edges[from..to]) };
        });
        const SWEEP: usize = 1 << 14;
        let sweeps = complex_links.slots().div_ceil(SWEEP);
        let next_sw = AtomicUsize::new(0);
        par(threads, |_| loop {
            let i = next_sw.fetch_add(1, Relaxed);
            if i >= sweeps {
                break;
            }
            unsafe {
                complex_links.compress_and_resolve(i * SWEEP, ((i + 1) * SWEEP).min(complex_links.slots()));
            }
        });
    }

    let mut cx_ev_count = deduped_odd_eid_events.len();
    let mut cx_ck_count = 0usize;
    for t in 0..threads {
        unsafe {
            cx_ev_count += per_thread_leftovers
                .get(t)
                .iter()
                .map(|s| s.events.len())
                .sum::<usize>();
            cx_ev_count += sess_workers.get(t).kept_odd_events.len();
            cx_ck_count += parse_workers.get(t).fallback_clicks.len();
        }
    }

    let mut complex_click_users: Vec<u32> = Vec::new();
    for u in 0..user_cap as usize {
        if (user_alias_table.alias_masks[u] & IS_COMPLEX) != 0 {
            let n = click_csr.csr_slice(u).len();
            if n > 0 {
                complex_click_users.push(u as u32);
                cx_ck_count += n;
            }
        }
    }

    let cx_p_arena = Arena::<PRec>::new(cx_ev_count + (threads * CX_PARTS + 2) * P_CHUNK, P_CHUNK);
    let cx_c_arena = Arena::<CRec>::new(cx_ck_count + (threads * CX_PARTS + 2) * C_CHUNK, C_CHUNK);

    let cx_p_scatters = PerThread::new(threads, |_| Scatter::<PRec, 4>::new());
    let cx_c_scatters = PerThread::new(threads, |_| Scatter::<CRec, 8>::new());

    if cx_ev_count > 0 || cx_ck_count > 0 {
        let next_cu = AtomicUsize::new(0);
        let odd_eid_chunk = (deduped_odd_eid_events.len().div_ceil(threads)).max(1);
        par(threads, |t| unsafe {
            let p_sc = cx_p_scatters.get(t);
            let c_sc = cx_c_scatters.get(t);
            p_sc.reset(CX_PARTS);
            c_sc.reset(CX_PARTS);

            for seg in per_thread_leftovers.get(t).iter() {
                for ev in &seg.events {
                    let raw_tok = user_alias_token(seg.user_idx, ev.alias, keys.user_key);
                    let person = complex_links.person_hashed(raw_tok, mix(raw_tok));
                    let payload = if ev.price_cents == 0 {
                        0
                    } else {
                        ((ev.cat as u32) << 24) | ev.price_cents
                    };
                    p_sc.push(
                        &cx_p_arena,
                        part_of_key(person, CX_PARTS),
                        PRec { person, ts: ev.rel_ts_ms, payload },
                    );
                }
            }

            for &(odd_handle, w0) in &sess_workers.get(t).kept_odd_events {
                let src_tid = (odd_handle >> 20) as usize;
                let seq = (odd_handle & 0xF_FFFF) as usize;
                let raw_tok = parse_workers.get(src_tid).odd_user_tokens[seq];
                let person = complex_links.person_hashed(raw_tok, mix(raw_tok));
                let price = ((w0 >> 9) & 0xFF_FFFF) as u32;
                let cat = ((w0 >> 4) & 0x1F) as u8;
                let payload = if price == 0 { 0 } else { ((cat as u32) << 24) | price };
                p_sc.push(
                    &cx_p_arena,
                    part_of_key(person, CX_PARTS),
                    PRec { person, ts: (w0 >> 33) as u32, payload },
                );
            }

            let oe_from = (t * odd_eid_chunk).min(deduped_odd_eid_events.len());
            let oe_to = ((t + 1) * odd_eid_chunk).min(deduped_odd_eid_events.len());
            for ev in &deduped_odd_eid_events[oe_from..oe_to] {
                let person = complex_links.person_hashed(ev.user_tok, mix(ev.user_tok));
                p_sc.push(
                    &cx_p_arena,
                    part_of_key(person, CX_PARTS),
                    PRec { person, ts: ev.ts, payload: ev.payload },
                );
            }

            for c in &parse_workers.get(t).fallback_clicks {
                let person = complex_links.person_hashed(c.person, mix(c.person));
                c_sc.push(
                    &cx_c_arena,
                    part_of_key(person, CX_PARTS),
                    CRec { person, click: c.click, ts: c.ts, campaign: c.campaign },
                );
            }

            loop {
                let ci = next_cu.fetch_add(1, Relaxed);
                let Some(&u) = complex_click_users.get(ci) else { break };
                for slot in click_csr.csr_slice(u as usize) {
                    let raw_tok = user_alias_token(u, slot.alias(), keys.user_key);
                    let person = complex_links.person_hashed(raw_tok, mix(raw_tok));
                    let click = if slot.is_upper_k() {
                        encode_upper_k_token(slot.user_or_cid)
                    } else {
                        encode_base36_token(b'k', slot.user_or_cid)
                    };
                    c_sc.push(
                        &cx_c_arena,
                        part_of_key(person, CX_PARTS),
                        CRec { person, click, ts: slot.ts(), campaign: slot.camp() },
                    );
                }
            }

            p_sc.finish(&cx_p_arena);
            c_sc.finish(&cx_c_arena);
        });
    }

    sh.links = complex_links;
    sh.p_arena = cx_p_arena;
    sh.c_arena = cx_c_arena;
    sh.p_parts = CX_PARTS;

    let cx_p_logs: Vec<&Vec<(u32, u32, u32)>> = (0..threads)
        .map(|t| unsafe { &cx_p_scatters.get(t).log })
        .collect();
    let cx_c_logs: Vec<&Vec<(u32, u32, u32)>> = (0..threads)
        .map(|t| unsafe { &cx_c_scatters.get(t).log })
        .collect();
    let cx_p_chunks = by_partition(&cx_p_logs, CX_PARTS);
    let cx_c_chunks = by_partition(&cx_c_logs, CX_PARTS);

    let size_of = |chunks: &Vec<(u32, u32)>| chunks.iter().map(|c| c.1 as u64).sum::<u64>();
    let mut cx_order: Vec<usize> = (0..CX_PARTS).collect();
    cx_order.sort_unstable_by_key(|&i| {
        std::cmp::Reverse(size_of(&cx_p_chunks[i]) + size_of(&cx_c_chunks[i]))
    });

    let (cat_data, cat_off) = build_cat_dict(&sh.categories);
    let groupers = PerThread::new(threads, |t| {
        Grouper::new(
            output.join("user_daily").join(format!("part-{t:03}.parquet")),
            (sh.campaigns.len() + 1) * days,
            &sh.categories,
        )
    });

    let next_cx = AtomicUsize::new(0);
    par(threads, |t| unsafe {
        let g = groupers.get(t);
        if cx_ev_count > 0 {
            loop {
                let i = next_cx.fetch_add(1, Relaxed);
                let Some(&part) = cx_order.get(i) else { break };
                g.partition(&sh, &cx_p_chunks[part], &cx_c_chunks[part]);
            }
        }
        let out = std::mem::replace(&mut g.out, group::UserOut::new(PathBuf::new()));
        let extra = sess_workers.get(t).user_out.as_cols(&cat_data, &cat_off);
        out.finish_with_extra(&extra, t == 0);
    });
    marks.push(mark("complex"));
    log_stage(&marks);

    enum CellRef<'a> {
        Borrowed(&'a [Cell]),
        Owned(Vec<Cell>),
    }
    impl AsRef<[Cell]> for CellRef<'_> {
        #[inline(always)]
        fn as_ref(&self) -> &[Cell] {
            match self {
                CellRef::Borrowed(s) => s,
                CellRef::Owned(v) => v,
            }
        }
    }
    let mut all_cells: Vec<CellRef<'_>> = Vec::with_capacity(threads * 2);
    for tid in 0..threads {
        unsafe {
            all_cells.push(CellRef::Borrowed(sess_workers.get(tid).finalize_cells()));
            if cx_ev_count > 0 {
                all_cells.push(CellRef::Owned(groupers.get(tid).take_cells()));
            }
        }
    }
    write_campaigns_sharded(
        &sh.campaigns,
        sh.first_day as i32,
        sh.days,
        &output.join("campaign_daily"),
        &all_cells,
        threads,
    );
    marks.push(mark("campaigns"));
    log_stage(&marks);

    if verbose {
        let (mut prev, mut prev_ru, mut prev_n) = (0.0, (0.0, 0.0, 0), marks[0].3);
        for (name, at, ru, n) in &marks {
            let d: Vec<f64> = (0..4).map(|k| (n[k] - prev_n[k]) as f64 / 1e6).collect();
            eprintln!(
                "{name:12} wall {:7.1} ms  user {:7.1}  sys {:6.1} ms  kfault {:6.0}  Mcyc {:7.1}  Minstr {:7.1}  Mllc {:6.2}  Mbr {:6.2}",
                (at - prev) * 1e3,
                (ru.0 - prev_ru.0) * 1e3,
                (ru.1 - prev_ru.1) * 1e3,
                (ru.2 - prev_ru.2) as f64 / 1e3,
                d[0],
                d[1],
                d[2],
                d[3]
            );
            (prev, prev_ru, prev_n) = (*at, *ru, *n);
        }
        eprintln!(
            "threads {threads}  ev_blocks {num_ev_blocks}  user_cap {user_cap}  events lines {total_ev_rows}  valid {total_valid_rows}  clicks {total_ck_rows}  cx_ev {cx_ev_count}  cx_ck {cx_ck_count}  peak rss {} MB",
            sys::peak_rss_kb() / 1024
        );
        let ru = sys::rusage();
        eprintln!(
            "total wall {:.1} ms  user {:.1} ms  sys {:.1} ms",
            t0.elapsed().as_secs_f64() * 1e3,
            ru.0 * 1e3,
            ru.1 * 1e3
        );
    }
    let stages: Vec<(&str, f64)> = marks
        .iter()
        .scan(0.0, |prev, m| {
            let d = m.1 - *prev;
            *prev = m.1;
            Some((m.0, d))
        })
        .collect();
    let ru = sys::rusage();
    diag::summary(
        t0.elapsed().as_secs_f64(),
        &stages,
        ru.0,
        ru.1,
        [total_ev_rows, total_valid_rows, total_ck_rows],
    );

    sys::prof::finish();
    sys::exit_now();
}
