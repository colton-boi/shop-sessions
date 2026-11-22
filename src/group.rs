use crate::fast_parquet::{FastUserDailyWriter, UserDailyCols, NULL_CAT};
use crate::ids::{append_tok_bytes, click_less, event_canon_key, event_eq};
use crate::ts::DAY_MS;
use crate::sys::mix;
use crate::{CRec, DRec, PRec, Shared, DEAD, NONE};
use std::arch::x86_64::*;
use std::path::PathBuf;

const SESSION_GAP: u32 = 1_800_000;
const WEEK: u32 = 604_800_000;
const DAY_MS_U32: u32 = DAY_MS as u32;

const FLUSH_ROWS: usize = 65_536;

/// Appends `person_tok` directly as a Parquet `PLAIN` `BYTE_ARRAY` entry (`[len_le32, bytes...]`)
/// into `names_plain`, returning its dense dictionary index and incrementing `names_count`.
#[inline(always)]
fn append_plain_tok(person_tok: u64, names_plain: &mut Vec<u8>, names_count: &mut u32) -> u32 {
    let old = names_plain.len();
    names_plain.reserve(4 + 33);
    unsafe { names_plain.set_len(old + 4) };
    append_tok_bytes(person_tok, names_plain);
    let tok_len = (names_plain.len() - old - 4) as u32;
    unsafe {
        (names_plain.as_mut_ptr().add(old) as *mut u32).write_unaligned(tok_len.to_le());
    }
    let key = *names_count;
    *names_count = key + 1;
    key
}

/// One thread's `user_daily` file, directly split into zero-revenue and non-zero live buffers
/// and streamed as Parquet v1 `RLE_DICTIONARY` row groups at `FLUSH_ROWS`.
pub struct UserOut {
    path: PathBuf,
    writer: Option<FastUserDailyWriter>,
    spill_idx: u32,

    z_names_plain: Vec<u8>,
    z_names_count: u32,
    z_person_keys: Vec<u32>,
    z_day: Vec<i32>,
    z_sessions: Vec<i32>,
    z_events: Vec<i32>,
    z_cap_min: usize,
    cur_person_z_key: u32,

    nz_names_plain: Vec<u8>,
    nz_names_count: u32,
    nz_person_keys: Vec<u32>,
    nz_day: Vec<i32>,
    nz_sessions: Vec<i32>,
    nz_events: Vec<i32>,
    nz_revenue: Vec<i32>,
    nz_revenue_i64: Vec<u64>,
    nz_revenue_overflow: bool,
    nz_top_category: Vec<u8>,
    cur_person_nz_key: u32,

    cat_dict_data: Vec<u8>,
    cat_dict_offsets: Vec<u32>,
}

pub fn build_cat_dict(categories: &[Vec<u8>]) -> (Vec<u8>, Vec<u32>) {
    let mut cat_dict_data = Vec::new();
    let mut cat_dict_offsets = Vec::with_capacity(categories.len() + 1);
    cat_dict_offsets.push(0);
    for c in categories {
        cat_dict_data.extend_from_slice(c);
        cat_dict_offsets.push(cat_dict_data.len() as u32);
    }
    (cat_dict_data, cat_dict_offsets)
}

impl UserOut {
    #[allow(dead_code)]
    pub fn new(path: PathBuf) -> Self {
        Self::with_categories(path, &[])
    }

    pub fn with_categories(path: PathBuf, categories: &[Vec<u8>]) -> Self {
        let (cat_dict_data, cat_dict_offsets) = build_cat_dict(categories);
        let z_cap = if path.as_os_str().is_empty() { 0 } else { FLUSH_ROWS + 256 };
        let nz_cap = if path.as_os_str().is_empty() { 0 } else { 16_384 };
        let z_person_keys = Vec::with_capacity(z_cap);
        let z_day = Vec::with_capacity(z_cap);
        let z_sessions = Vec::with_capacity(z_cap);
        let z_events = Vec::with_capacity(z_cap);
        let z_cap_min = z_person_keys
            .capacity()
            .min(z_day.capacity())
            .min(z_sessions.capacity())
            .min(z_events.capacity());
        UserOut {
            path,
            writer: None,
            spill_idx: 0,

            z_names_plain: Vec::with_capacity(if z_cap > 0 { 256 << 10 } else { 0 }),
            z_names_count: 0,
            z_person_keys,
            z_day,
            z_sessions,
            z_events,
            z_cap_min,
            cur_person_z_key: u32::MAX,

            nz_names_plain: Vec::with_capacity(if nz_cap > 0 { 64 << 10 } else { 0 }),
            nz_names_count: 0,
            nz_person_keys: Vec::with_capacity(nz_cap),
            nz_day: Vec::with_capacity(nz_cap),
            nz_sessions: Vec::with_capacity(nz_cap),
            nz_events: Vec::with_capacity(nz_cap),
            nz_revenue: Vec::with_capacity(nz_cap),
            nz_revenue_i64: Vec::new(),
            nz_revenue_overflow: false,
            nz_top_category: Vec::with_capacity(nz_cap),
            cur_person_nz_key: u32::MAX,

            cat_dict_data,
            cat_dict_offsets,
        }
    }

    #[inline(always)]
    pub fn begin_person(&mut self) {
        self.cur_person_z_key = u32::MAX;
        self.cur_person_nz_key = u32::MAX;
        if self.z_day.len() >= FLUSH_ROWS {
            self.flush_zero();
        }
        if self.nz_day.len() >= FLUSH_ROWS {
            self.flush_nonz();
        }
    }

    #[cold]
    #[inline(never)]
    fn grow_zero(&mut self) {
        self.z_person_keys.reserve(256);
        self.z_day.reserve(256);
        self.z_sessions.reserve(256);
        self.z_events.reserve(256);
        self.z_cap_min = self
            .z_person_keys
            .capacity()
            .min(self.z_day.capacity())
            .min(self.z_sessions.capacity())
            .min(self.z_events.capacity());
    }

    #[cold]
    #[inline(never)]
    fn promote_nz_revenue_i64(&mut self) {
        self.nz_revenue_overflow = true;
        self.nz_revenue_i64.reserve(self.nz_revenue.len() + 1);
        self.nz_revenue_i64
            .extend(self.nz_revenue.iter().map(|&r| r as u32 as u64));
    }

    fn spill_path(&self, idx: u32) -> PathBuf {
        let stem = self.path.file_stem().and_then(|s| s.to_str()).unwrap_or("part");
        let name = if idx == 1 {
            format!("{stem}.i64.parquet")
        } else {
            format!("{stem}.i64.{idx}.parquet")
        };
        match self.path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.join(name),
            _ => PathBuf::from(name),
        }
    }

    #[cold]
    #[inline(never)]
    fn flush_zero(&mut self) {
        if self.z_day.is_empty() {
            return;
        }
        let i64_mode = self.nz_revenue_overflow;
        let w = self.writer.get_or_insert_with(|| {
            let mut w = FastUserDailyWriter::create(&self.path);
            if i64_mode {
                w.set_revenue_i64(true);
            }
            w
        });
        w.write_zero_rev_direct(
            &self.z_names_plain,
            self.z_names_count as usize,
            &self.z_person_keys,
            &self.z_day,
            &self.z_sessions,
            &self.z_events,
        );
        self.z_names_plain.clear();
        self.z_names_count = 0;
        self.z_person_keys.clear();
        self.z_day.clear();
        self.z_sessions.clear();
        self.z_events.clear();
    }

    #[cold]
    #[inline(never)]
    fn flush_nonz(&mut self) {
        if self.nz_day.is_empty() {
            return;
        }
        if self.nz_revenue_overflow {
            let need_spill = matches!(
                self.writer.as_ref(),
                Some(w) if !w.is_revenue_i64() && w.total_rows() > 0
            );
            if need_spill {
                if let Some(old_w) = self.writer.take() {
                    old_w.finish();
                }
                self.spill_idx += 1;
                let spill_path = self.spill_path(self.spill_idx);
                let mut w = FastUserDailyWriter::create(&spill_path);
                w.set_revenue_i64(true);
                self.writer = Some(w);
            } else {
                let w = self
                    .writer
                    .get_or_insert_with(|| FastUserDailyWriter::create(&self.path));
                if !w.is_revenue_i64() {
                    w.set_revenue_i64(true);
                }
            }
            let w = self.writer.as_mut().unwrap();
            w.write_nonz_i64_direct(
                &self.nz_names_plain,
                self.nz_names_count as usize,
                &self.nz_person_keys,
                &self.nz_day,
                &self.nz_sessions,
                &self.nz_events,
                &self.nz_revenue_i64,
                &self.nz_top_category,
                &self.cat_dict_data,
                &self.cat_dict_offsets,
            );
        } else {
            let w = self
                .writer
                .get_or_insert_with(|| FastUserDailyWriter::create(&self.path));
            w.write_nonz_direct(
                &self.nz_names_plain,
                self.nz_names_count as usize,
                &self.nz_person_keys,
                &self.nz_day,
                &self.nz_sessions,
                &self.nz_events,
                &self.nz_revenue,
                &self.nz_top_category,
                &self.cat_dict_data,
                &self.cat_dict_offsets,
            );
        }
        self.nz_names_plain.clear();
        self.nz_names_count = 0;
        self.nz_person_keys.clear();
        self.nz_day.clear();
        self.nz_sessions.clear();
        self.nz_events.clear();
        self.nz_revenue.clear();
        self.nz_revenue_i64.clear();
        self.nz_revenue_overflow = false;
        self.nz_top_category.clear();
    }

    #[inline(always)]
    fn push_zero_raw(&mut self, pk: u32, day: i32, sessions: u32, events: u32) {
        let len = self.z_day.len();
        if len == self.z_cap_min {
            self.grow_zero();
        }
        unsafe {
            *self.z_person_keys.as_mut_ptr().add(len) = pk;
            *self.z_day.as_mut_ptr().add(len) = day;
            *self.z_sessions.as_mut_ptr().add(len) = sessions as i32;
            *self.z_events.as_mut_ptr().add(len) = events as i32;
            self.z_person_keys.set_len(len + 1);
            self.z_day.set_len(len + 1);
            self.z_sessions.set_len(len + 1);
            self.z_events.set_len(len + 1);
        }
    }

    #[inline(always)]
    pub fn add_zero(&mut self, person_tok: u64, day: i32, sessions: u32, events: u32) {
        let pk = if self.cur_person_z_key == u32::MAX {
            let k = append_plain_tok(person_tok, &mut self.z_names_plain, &mut self.z_names_count);
            self.cur_person_z_key = k;
            k
        } else {
            self.cur_person_z_key
        };
        self.push_zero_raw(pk, day, sessions, events);
    }

    #[inline(always)]
    pub fn add_zero_single(&mut self, person_tok: u64, day: i32, sessions: u32, events: u32) {
        if self.z_day.len() >= FLUSH_ROWS {
            self.flush_zero();
        }
        let pk = append_plain_tok(person_tok, &mut self.z_names_plain, &mut self.z_names_count);
        self.push_zero_raw(pk, day, sessions, events);
    }

    #[inline(always)]
    pub fn add(
        &mut self,
        person_tok: u64,
        day: i32,
        sessions: u32,
        events: u32,
        revenue: u64,
        category_idx: Option<u8>,
    ) {
        if revenue == 0 && category_idx.is_none() {
            self.add_zero(person_tok, day, sessions, events);
        } else {
            let pk = if self.cur_person_nz_key == u32::MAX {
                let k = append_plain_tok(person_tok, &mut self.nz_names_plain, &mut self.nz_names_count);
                self.cur_person_nz_key = k;
                k
            } else {
                self.cur_person_nz_key
            };
            self.nz_person_keys.push(pk);
            self.nz_day.push(day);
            self.nz_sessions.push(sessions as i32);
            self.nz_events.push(events as i32);
            if revenue > i32::MAX as u64 && !self.nz_revenue_overflow {
                self.promote_nz_revenue_i64();
            }
            if self.nz_revenue_overflow {
                self.nz_revenue_i64.push(revenue);
            }
            self.nz_revenue.push(revenue as i32);
            self.nz_top_category.push(category_idx.unwrap_or(NULL_CAT));
        }
    }

    /// An empty table still needs one file to carry the schema.
    pub fn finish(mut self, even_if_empty: bool) {
        if self.z_day.is_empty() && self.nz_day.is_empty() {
            match self.writer.take() {
                Some(w) => w.finish(),
                None => {
                    if even_if_empty {
                        FastUserDailyWriter::create(&self.path).finish();
                    }
                }
            }
            return;
        }
        if !self.z_day.is_empty() {
            self.flush_zero();
        }
        if !self.nz_day.is_empty() {
            self.flush_nonz();
        }
        if let Some(w) = self.writer.take() {
            w.finish();
        }
    }

    /// Writes both `extra` (from the fast-path sessionizer) and `self` (from complex users)
    /// into a single `user_daily/part-*.parquet` file.
    #[allow(dead_code)]
    pub fn finish_with_extra(mut self, extra: &UserDailyCols<'_>, even_if_empty: bool) {
        if !extra.is_empty() {
            let i64_mode = self.nz_revenue_overflow;
            let w = self.writer.get_or_insert_with(|| {
                let mut w = FastUserDailyWriter::create(&self.path);
                if i64_mode {
                    w.set_revenue_i64(true);
                }
                w
            });
            w.write_batches_split(extra, FLUSH_ROWS);
        }
        self.finish(even_if_empty);
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Cell {
    pub revenue: u64,
    pub sessions: u32,
    pub purchasing: u32,
    pub assisted: u32,
}

/// Fallback click record used when a partition exceeds 65,536 clicks or 65,536 campaigns.
#[derive(Clone, Copy)]
struct Click {
    key: u64,
    campaign: u32,
    at: u32,
}

trait ClickItem: Copy {
    fn ts(self) -> u32;
    fn campaign(self) -> usize;
    unsafe fn credit(sh: &Shared, clicks: &[Self], ck_at: &[u32]) -> usize;
}

impl ClickItem for u64 {
    #[inline(always)]
    fn ts(self) -> u32 {
        (self >> 32) as u32
    }

    #[inline(always)]
    fn campaign(self) -> usize {
        ((self >> 16) & 0xFFFF) as usize
    }

    #[inline(always)]
    unsafe fn credit(sh: &Shared, clicks: &[Self], ck_at: &[u32]) -> usize {
        credit_u64(sh, clicks, ck_at)
    }
}

impl ClickItem for Click {
    #[inline(always)]
    fn ts(self) -> u32 {
        (self.key >> 32) as u32
    }

    #[inline(always)]
    fn campaign(self) -> usize {
        self.campaign as usize
    }

    #[inline(always)]
    unsafe fn credit(sh: &Shared, clicks: &[Self], _ck_at: &[u32]) -> usize {
        credit_click(sh, clicks)
    }
}

struct DayRow {
    day: u32,
    first: u32,
    end: u32,
}

struct SessEmitState {
    day_origin: u32,
    cur_day: u32,
    next_day_start: u32,
    day: u32,
    day_sessions: u32,
    day_events: u32,
    day_revenue: u64,
    day_first: u32,
    session_index: u32,
    next_click: usize,
    cached_nc: usize,
    cached_camp: usize,
    next_purchase: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortMode {
    Avx2,
    Radix,
}

fn read_sort_mode() -> SortMode {
    match std::env::var("SS_SORT_MODE").as_deref() {
        Ok("radix") => SortMode::Radix,
        _ => SortMode::Avx2,
    }
}

/// One thread's working memory for turning a person partition into rows.
pub struct Grouper {
    keys: Vec<u64>,
    ids: Vec<u32>,
    dirty_keys: Vec<u32>,
    people: Vec<u64>,
    ev_count: Vec<u32>,
    pu_count: Vec<u32>,
    ck_count: Vec<u32>,
    /// Unscattered event person IDs and raw timestamps.
    ev_pid: Vec<u32>,
    ev_ts: Vec<u32>,
    /// Scattered per-person raw event timestamps `ts` (`< 2^31`).
    ev_u32: Vec<u32>,
    /// Scratch buffer for large per-person L1 radix sort.
    ev_tmp: Vec<u32>,
    /// Whole-partition radix buffers (used when `SS_SORT_MODE=radix`).
    ev_a: Vec<u64>,
    ev_b: Vec<u64>,
    hist: Box<[[u32; RADIX]; 2]>,
    /// Compact click scatter buffers (`ts << 32 | campaign << 16 | local_idx`).
    ck_pid: Vec<u32>,
    ck_packed_in: Vec<u64>,
    ck_packed: Vec<u64>,
    ck_at: Vec<u32>,
    /// Fallback click buffers for >65,536 campaigns or >65,536 clicks in one partition.
    ck_a: Vec<Click>,
    ck_b: Vec<Click>,
    /// (sort key `(id << 32) | ts`, payload) of every purchase, in sort-key order once sorted.
    purchases: Vec<(u64, u32)>,
    starts: Vec<u32>,
    day_rows: Vec<DayRow>,
    sess_row: Vec<u32>,
    covered: Vec<u64>,
    generation: u32,
    cat_sum: [u64; 256],
    cats: Vec<u8>,
    pub out: UserOut,
    /// Hot cell array (4B/cell): fits in L2 even under SMT.
    cell_sessions: Vec<u32>,
    /// Cold purchasing cell array (16B/cell): touched ONLY when `revenue > 0`.
    cell_purch: Vec<(u64, u32)>,
    /// Kept apart from `cell_sessions`/`cell_purch` so `assists()` touches 4B/cell.
    assisted: Vec<u32>,
    sort_mode: SortMode,
}

impl Grouper {
    pub fn new(path: PathBuf, cells: usize, categories: &[Vec<u8>]) -> Self {
        Grouper {
            keys: vec![0; 2048],
            ids: vec![0; 2048],
            dirty_keys: Vec::with_capacity(512),
            people: Vec::new(),
            ev_count: Vec::new(),
            pu_count: Vec::new(),
            ck_count: Vec::new(),
            ev_pid: Vec::new(),
            ev_ts: Vec::new(),
            ev_u32: Vec::new(),
            ev_tmp: Vec::new(),
            ev_a: Vec::new(),
            ev_b: Vec::new(),
            hist: Box::new([[0; RADIX]; 2]),
            ck_pid: Vec::new(),
            ck_packed_in: Vec::new(),
            ck_packed: Vec::new(),
            ck_at: Vec::new(),
            ck_a: Vec::new(),
            ck_b: Vec::new(),
            purchases: Vec::new(),
            starts: Vec::new(),
            day_rows: Vec::new(),
            sess_row: Vec::new(),
            covered: Vec::new(),
            generation: 0,
            cat_sum: [0; 256],
            cats: Vec::new(),
            out: UserOut::with_categories(path, categories),
            cell_sessions: vec![0; cells],
            cell_purch: vec![(0, 0); cells],
            assisted: vec![0; cells],
            sort_mode: read_sort_mode(),
        }
    }

    fn grow(&mut self) {
        let cap = self.keys.len() * 2;
        self.keys = vec![0; cap];
        self.ids = vec![0; cap];
        self.dirty_keys.clear();
        for (id, &person) in self.people.iter().enumerate() {
            let mut at = slot_of(person, cap - 1);
            while self.keys[at] != 0 {
                at = (at + 1) & (cap - 1);
            }
            self.keys[at] = person;
            self.ids[at] = id as u32;
            self.dirty_keys.push(at as u32);
        }
    }

    #[inline(always)]
    unsafe fn person_id(&mut self, person: u64, insert: bool) -> u32 {
        let mask = self.keys.len() - 1;
        let mut at = slot_of(person, mask);
        loop {
            let k = *self.keys.get_unchecked(at);
            if k == person {
                return *self.ids.get_unchecked(at);
            }
            if k == 0 {
                break;
            }
            at = (at + 1) & mask;
        }
        if !insert {
            return u32::MAX;
        }
        let id = self.people.len() as u32;
        self.keys[at] = person;
        self.ids[at] = id;
        self.dirty_keys.push(at as u32);
        self.people.push(person);
        self.ev_count.push(0);
        self.pu_count.push(0);
        if self.people.len() * 4 > self.keys.len() {
            self.grow();
        }
        id
    }

    /// Group one partition's live events and clicks by person, sort each
    /// person's records by ts, then emit each person.
    pub unsafe fn partition(&mut self, sh: &Shared, events: &[(u32, u32)], clicks: &[(u32, u32)]) {
        if self.covered.len() < sh.campaigns.len() {
            self.covered.resize(sh.campaigns.len(), 0);
        }
        if self.dirty_keys.len() * 6 <= self.keys.len() {
            for &at in &self.dirty_keys {
                *self.keys.get_unchecked_mut(at as usize) = 0;
            }
        } else {
            self.keys.fill(0);
        }
        self.dirty_keys.clear();
        self.people.clear();
        self.ev_count.clear();
        self.pu_count.clear();
        self.purchases.clear();

        let ev_total: usize = events.iter().map(|c| c.1 as usize).sum();
        self.ev_pid.clear();
        self.ev_pid.reserve(ev_total);
        self.ev_ts.clear();
        self.ev_ts.reserve(ev_total);
        let pid_ptr = self.ev_pid.as_mut_ptr();
        let ts_ptr = self.ev_ts.as_mut_ptr();
        let mut ev_n = 0usize;
        for &(first, count) in events {
            let recs: &[PRec] = std::slice::from_raw_parts(sh.p_arena.base.add(first as usize), count as usize);
            for r in recs {
                if r.person == DEAD {
                    continue;
                }
                let id = self.person_id(r.person, true);
                *self.ev_count.get_unchecked_mut(id as usize) += 1;
                *pid_ptr.add(ev_n) = id;
                *ts_ptr.add(ev_n) = r.ts;
                ev_n += 1;
                if r.payload != 0 {
                    *self.pu_count.get_unchecked_mut(id as usize) += 1;
                    self.purchases.push(((id as u64) << 32 | r.ts as u64, r.payload));
                }
            }
        }
        self.ev_pid.set_len(ev_n);
        self.ev_ts.set_len(ev_n);

        let people = self.people.len();
        if people == 0 {
            return;
        }

        // Scatter or radix-sort events into `self.ev_u32` (holding raw `r.ts`).
        self.ev_u32.clear();
        self.ev_u32.reserve(ev_n);
        self.ev_u32.set_len(ev_n);
        if self.sort_mode == SortMode::Avx2 {
            let mut sum = 0u32;
            for c in self.ev_count.iter_mut() {
                (*c, sum) = (sum, sum + *c);
            }
            let ppid = self.ev_pid.as_ptr();
            let pts = self.ev_ts.as_ptr();
            let pe = self.ev_u32.as_mut_ptr();
            for i in 0..ev_n {
                let id = *ppid.add(i) as usize;
                let slot = self.ev_count.get_unchecked_mut(id);
                *pe.add(*slot as usize) = *pts.add(i);
                *slot += 1;
            }
        } else {
            self.radix_partition_events(ev_n);
        }

        self.purchases.sort_unstable_by_key(|p| p.0);

        let ck_total: usize = clicks.iter().map(|c| c.1 as usize).sum();
        if sh.campaigns.len() <= 65_536 && ck_total <= 65_536 {
            self.partition_compact_clicks(sh, clicks, people, ck_total);
        } else {
            self.partition_fallback_clicks(sh, clicks, people, ck_total);
        }
    }

    unsafe fn radix_partition_events(&mut self, ev_n: usize) {
        let [h0, h1] = &mut *self.hist;
        h0.fill(0);
        h1.fill(0);
        self.ev_a.clear();
        self.ev_a.reserve(ev_n);
        self.ev_a.set_len(ev_n);
        self.ev_b.clear();
        self.ev_b.reserve(ev_n);
        self.ev_b.set_len(ev_n);
        let ppid = self.ev_pid.as_ptr();
        let pts = self.ev_ts.as_ptr();
        let pa = self.ev_a.as_mut_ptr();
        let pb = self.ev_b.as_mut_ptr();
        for i in 0..ev_n {
            let id = *ppid.add(i);
            let ts = *pts.add(i);
            *pa.add(i) = ((id as u64) << 32) | (ts as u64);
            *h0.get_unchecked_mut(digit0_u32(ts)) += 1;
            *h1.get_unchecked_mut(digit1_u32(ts)) += 1;
        }
        for h in [&mut *h0, &mut *h1] {
            let mut sum = 0u32;
            for c in h.iter_mut() {
                (*c, sum) = (sum, sum + *c);
            }
        }
        let mut sum = 0u32;
        for c in self.ev_count.iter_mut() {
            (*c, sum) = (sum, sum + *c);
        }
        for i in 0..ev_n {
            let x = *pa.add(i);
            let slot = h0.get_unchecked_mut(digit0_u32(x as u32));
            *pb.add(*slot as usize) = x;
            *slot += 1;
        }
        for i in 0..ev_n {
            let x = *pb.add(i);
            let slot = h1.get_unchecked_mut(digit1_u32(x as u32));
            *pa.add(*slot as usize) = x;
            *slot += 1;
        }
        let pe = self.ev_u32.as_mut_ptr();
        for i in 0..ev_n {
            let x = *pa.add(i);
            let slot = self.ev_count.get_unchecked_mut((x >> 32) as usize);
            *pe.add(*slot as usize) = x as u32;
            *slot += 1;
        }
    }

    unsafe fn partition_compact_clicks(
        &mut self,
        sh: &Shared,
        clicks: &[(u32, u32)],
        people: usize,
        ck_total: usize,
    ) {
        self.ck_at.clear();
        self.ck_at.reserve(ck_total);
        self.ck_pid.clear();
        self.ck_pid.reserve(ck_total);
        self.ck_packed_in.clear();
        self.ck_packed_in.reserve(ck_total);
        self.ck_count.clear();
        self.ck_count.resize(people, 0);

        let ck_at_ptr = self.ck_at.as_mut_ptr();
        let ck_pid_ptr = self.ck_pid.as_mut_ptr();
        let ck_in_ptr = self.ck_packed_in.as_mut_ptr();
        let mut ck_n = 0usize;
        for &(first, count) in clicks {
            let recs: &[CRec] = std::slice::from_raw_parts(sh.c_arena.base.add(first as usize), count as usize);
            for (k, r) in recs.iter().enumerate() {
                let id = self.person_id(r.person, false);
                if id == u32::MAX {
                    continue;
                }
                *self.ck_count.get_unchecked_mut(id as usize) += 1;
                *ck_at_ptr.add(ck_n) = first + k as u32;
                *ck_pid_ptr.add(ck_n) = id;
                *ck_in_ptr.add(ck_n) = ((r.ts as u64) << 32) | ((r.campaign as u64) << 16) | (ck_n as u64);
                ck_n += 1;
            }
        }
        self.ck_at.set_len(ck_n);
        self.ck_pid.set_len(ck_n);
        self.ck_packed_in.set_len(ck_n);

        self.ck_packed.clear();
        self.ck_packed.reserve(ck_n);
        self.ck_packed.set_len(ck_n);
        {
            let mut sum = 0u32;
            for c in self.ck_count.iter_mut() {
                (*c, sum) = (sum, sum + *c);
            }
            let ppid = self.ck_pid.as_ptr();
            let pin = self.ck_packed_in.as_ptr();
            let pout = self.ck_packed.as_mut_ptr();
            for i in 0..ck_n {
                let id = *ppid.add(i) as usize;
                let slot = self.ck_count.get_unchecked_mut(id);
                *pout.add(*slot as usize) = *pin.add(i);
                *slot += 1;
            }
        }

        let mut events = std::mem::take(&mut self.ev_u32);
        let mut ck_arr = std::mem::take(&mut self.ck_packed);
        let ck_at = std::mem::take(&mut self.ck_at);
        let purchases = std::mem::take(&mut self.purchases);
        let use_avx2 = self.sort_mode == SortMode::Avx2;
        let (mut ev_from, mut ck_from, mut pu_from) = (0usize, 0usize, 0usize);
        for id in 0..people {
            let (ev_end, ck_end) = (
                *self.ev_count.get_unchecked(id) as usize,
                *self.ck_count.get_unchecked(id) as usize,
            );
            let pu_end = pu_from + *self.pu_count.get_unchecked(id) as usize;
            let ev = &mut events[ev_from..ev_end];
            if use_avx2 {
                sort_u32_avx2(ev, &mut self.ev_tmp);
            } else {
                fix_order_u32(ev);
            }
            let ck = &mut ck_arr[ck_from..ck_end];
            sort_u64_clicks(ck);
            self.person(sh, *self.people.get_unchecked(id), ev, &purchases[pu_from..pu_end], ck, &ck_at);
            (ev_from, ck_from, pu_from) = (ev_end, ck_end, pu_end);
        }
        self.ev_u32 = events;
        self.ck_packed = ck_arr;
        self.ck_at = ck_at;
        self.purchases = purchases;
    }

    #[cold]
    #[inline(never)]
    unsafe fn partition_fallback_clicks(
        &mut self,
        sh: &Shared,
        clicks: &[(u32, u32)],
        people: usize,
        ck_total: usize,
    ) {
        self.ck_a.clear();
        self.ck_a.reserve(ck_total);
        self.ck_count.clear();
        self.ck_count.resize(people, 0);
        for &(first, count) in clicks {
            let recs: &[CRec] = std::slice::from_raw_parts(sh.c_arena.base.add(first as usize), count as usize);
            for (k, r) in recs.iter().enumerate() {
                let id = self.person_id(r.person, false);
                if id == u32::MAX {
                    continue;
                }
                *self.ck_count.get_unchecked_mut(id as usize) += 1;
                let key = ((r.ts as u64) << 32) | (id as u64);
                let n = self.ck_a.len();
                *self.ck_a.as_mut_ptr().add(n) = Click {
                    key,
                    campaign: r.campaign,
                    at: first + k as u32,
                };
                self.ck_a.set_len(n + 1);
            }
        }
        let ck_n = self.ck_a.len();
        self.ck_b.clear();
        self.ck_b.reserve(ck_n);
        self.ck_b.set_len(ck_n);
        {
            let mut sum = 0u32;
            for c in self.ck_count.iter_mut() {
                (*c, sum) = (sum, sum + *c);
            }
            let pa = self.ck_a.as_ptr();
            let pb = self.ck_b.as_mut_ptr();
            for i in 0..ck_n {
                let x = *pa.add(i);
                let slot = self.ck_count.get_unchecked_mut(x.key as u32 as usize);
                *pb.add(*slot as usize) = x;
                *slot += 1;
            }
        }

        let mut events = std::mem::take(&mut self.ev_u32);
        let mut ck_arr = std::mem::take(&mut self.ck_b);
        let purchases = std::mem::take(&mut self.purchases);
        let use_avx2 = self.sort_mode == SortMode::Avx2;
        let (mut ev_from, mut ck_from, mut pu_from) = (0usize, 0usize, 0usize);
        for id in 0..people {
            let (ev_end, ck_end) = (
                *self.ev_count.get_unchecked(id) as usize,
                *self.ck_count.get_unchecked(id) as usize,
            );
            let pu_end = pu_from + *self.pu_count.get_unchecked(id) as usize;
            let ev = &mut events[ev_from..ev_end];
            if use_avx2 {
                sort_u32_avx2(ev, &mut self.ev_tmp);
            } else {
                fix_order_u32(ev);
            }
            let ck = &mut ck_arr[ck_from..ck_end];
            ck.sort_unstable_by_key(|c| c.ts());
            self.person(sh, *self.people.get_unchecked(id), ev, &purchases[pu_from..pu_end], ck, &[]);
            (ev_from, ck_from, pu_from) = (ev_end, ck_end, pu_end);
        }
        self.ev_u32 = events;
        self.ck_b = ck_arr;
        self.purchases = purchases;
    }

    fn build_cells(&self) -> Vec<Cell> {
        let n = self.cell_sessions.len();
        let mut cells = Vec::with_capacity(n);
        for i in 0..n {
            let (revenue, purchasing) = self.cell_purch[i];
            cells.push(Cell {
                revenue,
                sessions: self.cell_sessions[i],
                purchasing,
                assisted: self.assisted[i],
            });
        }
        cells
    }

    #[allow(dead_code)]
    pub fn take_cells(&mut self) -> Vec<Cell> {
        self.build_cells()
    }

    pub fn finish(self, even_if_empty: bool) -> Vec<Cell> {
        let cells = self.build_cells();
        self.out.finish(even_if_empty);
        cells
    }

    #[inline(always)]
    fn flush_day<const HAS_PURCH: bool>(
        &mut self,
        sh: &Shared,
        person: u64,
        day: u32,
        sessions: u32,
        events: u32,
        revenue: u64,
    ) {
        let day_i32 = (sh.first_day + day as i64) as i32;
        if HAS_PURCH {
            let mut best: Option<u8> = None;
            for &c in &self.cats {
                let sum = self.cat_sum[c as usize];
                // Category indexes are in byte-wise name order, so the smaller index wins a tie.
                best = match best {
                    Some(b) if (self.cat_sum[b as usize], std::cmp::Reverse(b)) >= (sum, std::cmp::Reverse(c)) => Some(b),
                    _ => Some(c),
                };
            }
            for &c in &self.cats {
                self.cat_sum[c as usize] = 0;
            }
            self.cats.clear();
            self.out.add(person, day_i32, sessions, events, revenue, best);
        } else {
            self.out.add_zero(person, day_i32, sessions, events);
        }
    }

    #[inline(always)]
    unsafe fn person<C: ClickItem>(
        &mut self,
        sh: &Shared,
        person: u64,
        events: &[u32],
        purchases: &[(u64, u32)],
        clicks: &[C],
        ck_at: &[u32],
    ) {
        match (!purchases.is_empty(), !clicks.is_empty()) {
            (false, false) => self.person_impl::<false, false, C>(sh, person, events, purchases, clicks, ck_at),
            (true, false) => self.person_impl::<true, false, C>(sh, person, events, purchases, clicks, ck_at),
            (false, true) => self.person_impl::<false, true, C>(sh, person, events, purchases, clicks, ck_at),
            (true, true) => self.person_impl::<true, true, C>(sh, person, events, purchases, clicks, ck_at),
        }
    }

    unsafe fn person_impl<const HAS_PURCH: bool, const HAS_CLICKS: bool, C: ClickItem>(
        &mut self,
        sh: &Shared,
        person: u64,
        events: &[u32],
        purchases: &[(u64, u32)],
        clicks: &[C],
        ck_at: &[u32],
    ) {
        if events.is_empty() {
            return;
        }
        if !HAS_CLICKS && !HAS_PURCH && events.len() == 1 {
            let day_origin = (sh.first_day * DAY_MS - sh.base_ms) as u32;
            let start = *events.get_unchecked(0);
            let day = (start - day_origin) / DAY_MS_U32;
            *self
                .cell_sessions
                .get_unchecked_mut(sh.campaigns.len() * sh.days + day as usize) += 1;
            self.out
                .add_zero_single(person, (sh.first_day + day as i64) as i32, 1, 1);
            return;
        }
        if events.len() >= 9 {
            self.person_impl_simd::<HAS_PURCH, HAS_CLICKS, C>(sh, person, events, purchases, clicks, ck_at);
        } else {
            self.person_impl_scalar::<HAS_PURCH, HAS_CLICKS, C>(sh, person, events, purchases, clicks, ck_at);
        }
    }

    #[inline(always)]
    unsafe fn emit_session<const HAS_PURCH: bool, const HAS_CLICKS: bool, C: ClickItem>(
        &mut self,
        sh: &Shared,
        person: u64,
        events: &[u32],
        purchases: &[(u64, u32)],
        clicks: &[C],
        ck_at: &[u32],
        st: &mut SessEmitState,
        sess_start: usize,
        sess_end: usize,
        next_sess_ts: u32,
    ) {
        let days = sh.days;
        let organic = sh.campaigns.len();
        let start = *events.get_unchecked(sess_start);
        let start_day = if start < st.next_day_start {
            st.cur_day
        } else {
            let next2 = st.next_day_start + DAY_MS_U32;
            if start < next2 {
                st.cur_day += 1;
                st.next_day_start = next2;
                st.cur_day
            } else {
                let d = (start - st.day_origin) / DAY_MS_U32;
                st.cur_day = d;
                st.next_day_start = st.day_origin + (d + 1) * DAY_MS_U32;
                d
            }
        };
        if start_day != st.day {
            if st.day != u32::MAX {
                self.flush_day::<HAS_PURCH>(sh, person, st.day, st.day_sessions, st.day_events, st.day_revenue);
                if HAS_CLICKS {
                    self.day_rows.push(DayRow { day: st.day, first: st.day_first, end: st.session_index });
                }
            }
            st.day = start_day;
            st.day_sessions = 0;
            st.day_events = 0;
            st.day_revenue = 0;
            if HAS_CLICKS {
                st.day_first = st.session_index;
            }
        }
        let campaign = if HAS_CLICKS {
            let nck = clicks.len();
            while st.next_click < nck && clicks.get_unchecked(st.next_click).ts() <= start {
                st.next_click += 1;
            }
            if st.next_click > 0 && start - clicks.get_unchecked(st.next_click - 1).ts() <= WEEK {
                if st.next_click != st.cached_nc {
                    st.cached_nc = st.next_click;
                    st.cached_camp = C::credit(sh, &clicks[..st.next_click], ck_at);
                }
                st.cached_camp
            } else {
                organic
            }
        } else {
            organic
        };
        let cell_idx = campaign * days + start_day as usize;
        _mm_prefetch(self.cell_sessions.as_ptr().add(cell_idx) as *const i8, _MM_HINT_T0);
        let count = (sess_end - sess_start) as u32;
        let mut revenue = 0u64;
        if HAS_PURCH {
            while st.next_purchase < purchases.len()
                && ((*purchases.get_unchecked(st.next_purchase)).0 as u32) < next_sess_ts
            {
                let payload = purchases.get_unchecked(st.next_purchase).1;
                st.next_purchase += 1;
                let (cat, price) = ((payload >> 24) as u8, (payload & 0xFF_FFFF) as u64);
                revenue += price;
                if self.cat_sum[cat as usize] == 0 {
                    self.cats.push(cat);
                }
                self.cat_sum[cat as usize] += price;
            }
        }
        st.day_sessions += 1;
        st.day_events += count;
        *self.cell_sessions.get_unchecked_mut(cell_idx) += 1;
        if HAS_PURCH {
            st.day_revenue += revenue;
            if revenue > 0 {
                let p = self.cell_purch.get_unchecked_mut(cell_idx);
                p.0 += revenue;
                p.1 += 1;
            }
        }
        if HAS_CLICKS {
            st.session_index += 1;
            self.starts.push(start);
            self.sess_row.push(self.day_rows.len() as u32);
        }
    }

    unsafe fn person_impl_simd<const HAS_PURCH: bool, const HAS_CLICKS: bool, C: ClickItem>(
        &mut self,
        sh: &Shared,
        person: u64,
        events: &[u32],
        purchases: &[(u64, u32)],
        clicks: &[C],
        ck_at: &[u32],
    ) {
        self.out.begin_person();
        if HAS_CLICKS {
            self.starts.clear();
            self.sess_row.clear();
            self.day_rows.clear();
        }
        let days = sh.days;
        let n = events.len();
        let day_origin = (sh.first_day * DAY_MS - sh.base_ms) as u32;
        let mut st = SessEmitState {
            day_origin,
            cur_day: 0,
            next_day_start: day_origin + DAY_MS_U32,
            day: u32::MAX,
            day_sessions: 0,
            day_events: 0,
            day_revenue: 0,
            day_first: 0,
            session_index: 0,
            next_click: 0,
            cached_nc: 0,
            cached_camp: 0,
            next_purchase: 0,
        };

        let mut sess_start = 0usize;
        let ts0 = *events.get_unchecked(0);
        let mut prev_v = _mm256_set1_epi32(ts0 as i32);
        let gap_v = _mm256_set1_epi32(SESSION_GAP as i32);
        let ptr = events.as_ptr();
        let mut j = 1usize;
        while j + 8 <= n {
            // 3-op AVX2 session boundary detection: load cur, align prev, sub, cmpgt, movemask.
            let ts_cur = _mm256_loadu_si256(ptr.add(j) as *const __m256i);
            let hi_lo = _mm256_permute2x128_si256(prev_v, ts_cur, 0x21);
            let ts_prev = _mm256_alignr_epi8(ts_cur, hi_lo, 12);
            prev_v = ts_cur;

            let diff = _mm256_sub_epi32(ts_cur, ts_prev);
            let cmp = _mm256_cmpgt_epi32(diff, gap_v);
            let mut mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32;
            while mask != 0 {
                let k = mask.trailing_zeros() as usize;
                let boundary = j + k;
                let next_ts = *events.get_unchecked(boundary);
                self.emit_session::<HAS_PURCH, HAS_CLICKS, C>(
                    sh, person, events, purchases, clicks, ck_at, &mut st, sess_start, boundary, next_ts,
                );
                sess_start = boundary;
                mask &= mask - 1;
            }
            j += 8;
        }
        let mut prev_ts = *events.get_unchecked(j - 1);
        while j < n {
            let t = *events.get_unchecked(j);
            if t - prev_ts > SESSION_GAP {
                self.emit_session::<HAS_PURCH, HAS_CLICKS, C>(
                    sh, person, events, purchases, clicks, ck_at, &mut st, sess_start, j, t,
                );
                sess_start = j;
            }
            prev_ts = t;
            j += 1;
        }
        self.emit_session::<HAS_PURCH, HAS_CLICKS, C>(
            sh, person, events, purchases, clicks, ck_at, &mut st, sess_start, n, u32::MAX,
        );

        self.flush_day::<HAS_PURCH>(sh, person, st.day, st.day_sessions, st.day_events, st.day_revenue);
        if HAS_CLICKS {
            self.day_rows.push(DayRow { day: st.day, first: st.day_first, end: st.session_index });
            self.starts.push(u32::MAX);
            self.assists(days, clicks);
        }
    }

    unsafe fn person_impl_scalar<const HAS_PURCH: bool, const HAS_CLICKS: bool, C: ClickItem>(
        &mut self,
        sh: &Shared,
        person: u64,
        events: &[u32],
        purchases: &[(u64, u32)],
        clicks: &[C],
        ck_at: &[u32],
    ) {
        self.out.begin_person();
        if HAS_CLICKS {
            self.starts.clear();
            self.sess_row.clear();
            self.day_rows.clear();
        }
        let days = sh.days;
        let n = events.len();
        let day_origin = (sh.first_day * DAY_MS - sh.base_ms) as u32;
        let mut st = SessEmitState {
            day_origin,
            cur_day: 0,
            next_day_start: day_origin + DAY_MS_U32,
            day: u32::MAX,
            day_sessions: 0,
            day_events: 0,
            day_revenue: 0,
            day_first: 0,
            session_index: 0,
            next_click: 0,
            cached_nc: 0,
            cached_camp: 0,
            next_purchase: 0,
        };

        let mut sess_start = 0usize;
        let mut prev_ts = *events.get_unchecked(0);
        let mut j = 1usize;
        while j < n {
            let t = *events.get_unchecked(j);
            if t - prev_ts > SESSION_GAP {
                self.emit_session::<HAS_PURCH, HAS_CLICKS, C>(
                    sh, person, events, purchases, clicks, ck_at, &mut st, sess_start, j, t,
                );
                sess_start = j;
            }
            prev_ts = t;
            j += 1;
        }
        self.emit_session::<HAS_PURCH, HAS_CLICKS, C>(
            sh, person, events, purchases, clicks, ck_at, &mut st, sess_start, n, u32::MAX,
        );

        self.flush_day::<HAS_PURCH>(sh, person, st.day, st.day_sessions, st.day_events, st.day_revenue);
        if HAS_CLICKS {
            self.day_rows.push(DayRow { day: st.day, first: st.day_first, end: st.session_index });
            self.starts.push(u32::MAX);
            self.assists(days, clicks);
        }
    }

    /// Click k assists the sessions [lo_k, hi_k) with start in [ts, ts + 7 days].
    /// Both ends only grow with ts, so for one campaign everything below the
    /// previous click's hi is already counted.
    /// Requires `self.starts` to be terminated with a `u32::MAX` sentinel after the `ns` real starts.
    #[inline(always)]
    unsafe fn assists<C: ClickItem>(&mut self, days: usize, clicks: &[C]) {
        if self.day_rows.len() == 1 {
            self.assists_impl::<true, C>(days, clicks);
        } else {
            self.assists_impl::<false, C>(days, clicks);
        }
    }

    unsafe fn assists_impl<const SINGLE_DAY: bool, C: ClickItem>(&mut self, days: usize, clicks: &[C]) {
        debug_assert!(!self.starts.is_empty());
        debug_assert_eq!(*self.starts.last().unwrap(), u32::MAX);
        self.generation += 1;
        let gen = (self.generation as u64) << 32;
        let s = self.starts.as_ptr();
        let ns = self.starts.len() - 1;
        let d0 = if SINGLE_DAY {
            debug_assert_eq!(self.day_rows.len(), 1);
            self.day_rows.get_unchecked(0).day as usize
        } else {
            0
        };
        let (mut lo, mut hi) = (0usize, 0usize);
        for &c in clicks {
            let ts = c.ts();
            while *s.add(lo) < ts {
                lo += 1;
            }
            if lo == ns {
                break;
            }
            let reach = ts as u64 + WEEK as u64;
            while (*s.add(hi) as u64) <= reach {
                hi += 1;
            }
            let camp = c.campaign();
            let slot = self.covered.get_unchecked_mut(camp);
            let from = if *slot >> 32 == gen >> 32 { lo.max(*slot as u32 as usize) } else { lo };
            *slot = gen | hi as u64;
            if from >= hi {
                continue;
            }
            let base = camp * days;
            if SINGLE_DAY {
                *self.assisted.get_unchecked_mut(base + d0) += (hi - from) as u32;
            } else {
                let mut r = *self.sess_row.get_unchecked(from) as usize;
                loop {
                    let row = self.day_rows.get_unchecked(r);
                    let end = (row.end as usize).min(hi);
                    *self.assisted.get_unchecked_mut(base + row.day as usize) +=
                        (end - from.max(row.first as usize)) as u32;
                    if end == hi {
                        break;
                    }
                    r += 1;
                }
            }
        }
    }
}

#[inline(always)]
fn slot_of(person: u64, mask: usize) -> usize {
    (person.wrapping_mul(0xD6E8_FEB8_6659_FD93) >> 40) as usize & mask
}

const DIGIT: u32 = 11;
const RADIX: usize = 1 << DIGIT;
/// 10 + 11 + 11 = 32 bits: two 11-bit digits cover bits 10..32 of any `u32`,
/// leaving only inversions within a 1024-value bucket for `fix_order_u32`.
const TS_SKIP: u32 = 10;

#[inline(always)]
fn digit0_u32(ts: u32) -> usize {
    ((ts >> TS_SKIP) as usize) & (RADIX - 1)
}

#[inline(always)]
fn digit1_u32(ts: u32) -> usize {
    ((ts >> (TS_SKIP + DIGIT)) as usize) & (RADIX - 1)
}

/// Repairs the rare inversions inside a `1 << TS_SKIP` bucket after 2-pass 11-bit radix sort.
fn fix_order_u32(v: &mut [u32]) {
    let n = v.len();
    let mut i = 1;
    while i < n {
        if v[i] >= v[i - 1] {
            i += 1;
            continue;
        }
        let b = v[i] >> TS_SKIP;
        let mut from = i - 1;
        while from > 0 && (v[from - 1] >> TS_SKIP) == b {
            from -= 1;
        }
        let mut to = i + 1;
        while to < n && (v[to] >> TS_SKIP) == b {
            to += 1;
        }
        v[from..to].sort_unstable();
        i = to;
    }
}

#[repr(C, align(32))]
struct MaskTable32([[i32; 8]; 9]);

static LANE_MASKS_32: MaskTable32 = MaskTable32([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1],
]);

#[inline(always)]
unsafe fn lane_mask_32(k: usize) -> __m256i {
    _mm256_load_si256(LANE_MASKS_32.0.as_ptr().add(k) as *const __m256i)
}

#[repr(C, align(32))]
struct MaskTable64([[i64; 4]; 5]);

static LANE_MASKS_64: MaskTable64 = MaskTable64([
    [0, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, -1, 0, 0],
    [-1, -1, -1, 0],
    [-1, -1, -1, -1],
]);

#[inline(always)]
unsafe fn lane_mask_64(k: usize) -> __m256i {
    _mm256_load_si256(LANE_MASKS_64.0.as_ptr().add(k) as *const __m256i)
}

#[inline(always)]
unsafe fn cmp_swap_1<const IMM8: i32>(v: __m256i) -> __m256i {
    let p = _mm256_shuffle_epi32(v, 0xB1);
    _mm256_blend_epi32(_mm256_min_epu32(v, p), _mm256_max_epu32(v, p), IMM8)
}

#[inline(always)]
unsafe fn cmp_swap_2<const IMM8: i32>(v: __m256i) -> __m256i {
    let p = _mm256_shuffle_epi32(v, 0x4E);
    _mm256_blend_epi32(_mm256_min_epu32(v, p), _mm256_max_epu32(v, p), IMM8)
}

#[inline(always)]
unsafe fn cmp_swap_4<const IMM8: i32>(v: __m256i) -> __m256i {
    let p = _mm256_permute2x128_si256(v, v, 0x01);
    _mm256_blend_epi32(_mm256_min_epu32(v, p), _mm256_max_epu32(v, p), IMM8)
}

#[inline(always)]
unsafe fn bitonic_merge_8_asc(mut v: __m256i) -> __m256i {
    v = cmp_swap_4::<0xF0>(v);
    v = cmp_swap_2::<0xCC>(v);
    cmp_swap_1::<0xAA>(v)
}

#[inline(always)]
unsafe fn bitonic_merge_8_desc(mut v: __m256i) -> __m256i {
    v = cmp_swap_4::<0x0F>(v);
    v = cmp_swap_2::<0x33>(v);
    cmp_swap_1::<0x55>(v)
}

#[inline(always)]
unsafe fn bitonic_pre_8(mut v: __m256i) -> __m256i {
    v = cmp_swap_1::<0x66>(v);
    v = cmp_swap_2::<0x3C>(v);
    cmp_swap_1::<0x5A>(v)
}

#[inline(always)]
unsafe fn sort_8_asc(v: __m256i) -> __m256i {
    bitonic_merge_8_asc(bitonic_pre_8(v))
}

#[inline(always)]
unsafe fn sort_8_desc(v: __m256i) -> __m256i {
    bitonic_merge_8_desc(bitonic_pre_8(v))
}

#[inline(always)]
unsafe fn cmp_swap_vec(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    (_mm256_min_epu32(a, b), _mm256_max_epu32(a, b))
}

/// 1-YMM branchless AVX2 bitonic sort handling `n = 5..=8`.
#[inline(always)]
unsafe fn bitonic_8_pad(ptr: *mut u32, n: usize) {
    let m0 = lane_mask_32(n);
    let all_max = _mm256_set1_epi32(-1);
    let l0 = _mm256_maskload_epi32(ptr as *const i32, m0);
    let v0 = sort_8_asc(_mm256_blendv_epi8(all_max, l0, m0));
    _mm256_maskstore_epi32(ptr as *mut i32, m0, v0);
}

/// 2-YMM branchless AVX2 bitonic sort handling `n = 9..=16`.
#[inline(always)]
unsafe fn bitonic_16_pad(ptr: *mut u32, n: usize) {
    let m1 = lane_mask_32(n - 8);
    let all_max = _mm256_set1_epi32(-1);
    let mut v0 = sort_8_asc(_mm256_loadu_si256(ptr as *const __m256i));
    let l1 = _mm256_maskload_epi32(ptr.add(8) as *const i32, m1);
    let mut v1 = sort_8_desc(_mm256_blendv_epi8(all_max, l1, m1));
    (v0, v1) = cmp_swap_vec(v0, v1);
    v0 = bitonic_merge_8_asc(v0);
    v1 = bitonic_merge_8_asc(v1);
    _mm256_storeu_si256(ptr as *mut __m256i, v0);
    _mm256_maskstore_epi32(ptr.add(8) as *mut i32, m1, v1);
}

/// 4-YMM branchless AVX2 bitonic sort handling `n = 17..=32`.
#[inline(always)]
unsafe fn bitonic_32_pad(ptr: *mut u32, n: usize) {
    let m2 = lane_mask_32((n - 16).min(8));
    let m3 = lane_mask_32(n.saturating_sub(24));
    let all_max = _mm256_set1_epi32(-1);
    let mut v0 = sort_8_asc(_mm256_loadu_si256(ptr as *const __m256i));
    let mut v1 = sort_8_desc(_mm256_loadu_si256(ptr.add(8) as *const __m256i));
    let l2 = _mm256_maskload_epi32(ptr.add(16) as *const i32, m2);
    let mut v2 = sort_8_asc(_mm256_blendv_epi8(all_max, l2, m2));
    let l3 = _mm256_maskload_epi32(ptr.add(24) as *const i32, m3);
    let mut v3 = sort_8_desc(_mm256_blendv_epi8(all_max, l3, m3));

    (v0, v1) = cmp_swap_vec(v0, v1);
    (v3, v2) = cmp_swap_vec(v2, v3);
    v0 = bitonic_merge_8_asc(v0);
    v1 = bitonic_merge_8_asc(v1);
    v2 = bitonic_merge_8_desc(v2);
    v3 = bitonic_merge_8_desc(v3);

    (v0, v2) = cmp_swap_vec(v0, v2);
    (v1, v3) = cmp_swap_vec(v1, v3);
    (v0, v1) = cmp_swap_vec(v0, v1);
    (v2, v3) = cmp_swap_vec(v2, v3);
    v0 = bitonic_merge_8_asc(v0);
    v1 = bitonic_merge_8_asc(v1);
    v2 = bitonic_merge_8_asc(v2);
    v3 = bitonic_merge_8_asc(v3);

    _mm256_storeu_si256(ptr as *mut __m256i, v0);
    _mm256_storeu_si256(ptr.add(8) as *mut __m256i, v1);
    _mm256_maskstore_epi32(ptr.add(16) as *mut i32, m2, v2);
    _mm256_maskstore_epi32(ptr.add(24) as *mut i32, m3, v3);
}

/// 8-YMM branchless AVX2 bitonic sort handling `n = 33..=64`.
#[inline(always)]
unsafe fn bitonic_64_pad(ptr: *mut u32, n: usize) {
    let rem = n - 32;
    let m4 = lane_mask_32(rem.min(8));
    let m5 = lane_mask_32(rem.saturating_sub(8).min(8));
    let m6 = lane_mask_32(rem.saturating_sub(16).min(8));
    let m7 = lane_mask_32(rem.saturating_sub(24));
    let all_max = _mm256_set1_epi32(-1);

    let mut v0 = sort_8_asc(_mm256_loadu_si256(ptr as *const __m256i));
    let mut v1 = sort_8_desc(_mm256_loadu_si256(ptr.add(8) as *const __m256i));
    let mut v2 = sort_8_asc(_mm256_loadu_si256(ptr.add(16) as *const __m256i));
    let mut v3 = sort_8_desc(_mm256_loadu_si256(ptr.add(24) as *const __m256i));
    let l4 = _mm256_maskload_epi32(ptr.add(32) as *const i32, m4);
    let mut v4 = sort_8_asc(_mm256_blendv_epi8(all_max, l4, m4));
    let l5 = _mm256_maskload_epi32(ptr.add(40) as *const i32, m5);
    let mut v5 = sort_8_desc(_mm256_blendv_epi8(all_max, l5, m5));
    let l6 = _mm256_maskload_epi32(ptr.add(48) as *const i32, m6);
    let mut v6 = sort_8_asc(_mm256_blendv_epi8(all_max, l6, m6));
    let l7 = _mm256_maskload_epi32(ptr.add(56) as *const i32, m7);
    let mut v7 = sort_8_desc(_mm256_blendv_epi8(all_max, l7, m7));

    // Size-16 merge: (v0,v1) asc, (v2,v3) desc, (v4,v5) asc, (v6,v7) desc.
    (v0, v1) = cmp_swap_vec(v0, v1);
    (v3, v2) = cmp_swap_vec(v2, v3);
    (v4, v5) = cmp_swap_vec(v4, v5);
    (v7, v6) = cmp_swap_vec(v6, v7);
    v0 = bitonic_merge_8_asc(v0);
    v1 = bitonic_merge_8_asc(v1);
    v2 = bitonic_merge_8_desc(v2);
    v3 = bitonic_merge_8_desc(v3);
    v4 = bitonic_merge_8_asc(v4);
    v5 = bitonic_merge_8_asc(v5);
    v6 = bitonic_merge_8_desc(v6);
    v7 = bitonic_merge_8_desc(v7);

    // Size-32 merge: (v0..v3) asc, (v4..v7) desc.
    (v0, v2) = cmp_swap_vec(v0, v2);
    (v1, v3) = cmp_swap_vec(v1, v3);
    (v0, v1) = cmp_swap_vec(v0, v1);
    (v2, v3) = cmp_swap_vec(v2, v3);
    v0 = bitonic_merge_8_asc(v0);
    v1 = bitonic_merge_8_asc(v1);
    v2 = bitonic_merge_8_asc(v2);
    v3 = bitonic_merge_8_asc(v3);

    (v6, v4) = cmp_swap_vec(v4, v6);
    (v7, v5) = cmp_swap_vec(v5, v7);
    (v5, v4) = cmp_swap_vec(v4, v5);
    (v7, v6) = cmp_swap_vec(v6, v7);
    v4 = bitonic_merge_8_desc(v4);
    v5 = bitonic_merge_8_desc(v5);
    v6 = bitonic_merge_8_desc(v6);
    v7 = bitonic_merge_8_desc(v7);

    // Size-64 merge: (v0..v7) asc.
    (v0, v4) = cmp_swap_vec(v0, v4);
    (v1, v5) = cmp_swap_vec(v1, v5);
    (v2, v6) = cmp_swap_vec(v2, v6);
    (v3, v7) = cmp_swap_vec(v3, v7);
    (v0, v2) = cmp_swap_vec(v0, v2);
    (v1, v3) = cmp_swap_vec(v1, v3);
    (v4, v6) = cmp_swap_vec(v4, v6);
    (v5, v7) = cmp_swap_vec(v5, v7);
    (v0, v1) = cmp_swap_vec(v0, v1);
    (v2, v3) = cmp_swap_vec(v2, v3);
    (v4, v5) = cmp_swap_vec(v4, v5);
    (v6, v7) = cmp_swap_vec(v6, v7);
    v0 = bitonic_merge_8_asc(v0);
    v1 = bitonic_merge_8_asc(v1);
    v2 = bitonic_merge_8_asc(v2);
    v3 = bitonic_merge_8_asc(v3);
    v4 = bitonic_merge_8_asc(v4);
    v5 = bitonic_merge_8_asc(v5);
    v6 = bitonic_merge_8_asc(v6);
    v7 = bitonic_merge_8_asc(v7);

    _mm256_storeu_si256(ptr as *mut __m256i, v0);
    _mm256_storeu_si256(ptr.add(8) as *mut __m256i, v1);
    _mm256_storeu_si256(ptr.add(16) as *mut __m256i, v2);
    _mm256_storeu_si256(ptr.add(24) as *mut __m256i, v3);
    _mm256_maskstore_epi32(ptr.add(32) as *mut i32, m4, v4);
    _mm256_maskstore_epi32(ptr.add(40) as *mut i32, m5, v5);
    _mm256_maskstore_epi32(ptr.add(48) as *mut i32, m6, v6);
    _mm256_maskstore_epi32(ptr.add(56) as *mut i32, m7, v7);
}

#[cold]
#[inline(never)]
fn radix_u32_2pass(ev: &mut [u32], tmp: &mut Vec<u32>) {
    let n = ev.len();
    if n <= 256 {
        ev.sort_unstable();
        return;
    }
    let mut h0 = [0u32; RADIX];
    let mut h1 = [0u32; RADIX];
    for &ts in ev.iter() {
        h0[digit0_u32(ts)] += 1;
        h1[digit1_u32(ts)] += 1;
    }
    for h in [&mut h0, &mut h1] {
        let mut sum = 0u32;
        for c in h.iter_mut() {
            (*c, sum) = (sum, sum + *c);
        }
    }
    tmp.clear();
    tmp.reserve(n);
    unsafe {
        tmp.set_len(n);
        let pe = ev.as_mut_ptr();
        let pt = tmp.as_mut_ptr();
        for i in 0..n {
            let x = *pe.add(i);
            let slot = h0.get_unchecked_mut(digit0_u32(x));
            *pt.add(*slot as usize) = x;
            *slot += 1;
        }
        for i in 0..n {
            let x = *pt.add(i);
            let slot = h1.get_unchecked_mut(digit1_u32(x));
            *pe.add(*slot as usize) = x;
            *slot += 1;
        }
    }
    fix_order_u32(ev);
}

/// Branchless AVX2 register bitonic sort for per-person `u32` event timestamps (`<= 64`),
/// falling back to `sort_unstable` / 2-pass L1 11-bit radix + `fix_order_u32` for larger slices.
#[inline(always)]
pub(crate) fn sort_u32_avx2(ev: &mut [u32], tmp: &mut Vec<u32>) {
    let n = ev.len();
    if n <= 1 {
        return;
    }
    let ptr = ev.as_mut_ptr();
    unsafe {
        if n == 2 {
            let (a, b) = (*ptr, *ptr.add(1));
            *ptr = a.min(b);
            *ptr.add(1) = a.max(b);
        } else if n == 3 {
            let (mut a, mut b, mut c) = (*ptr, *ptr.add(1), *ptr.add(2));
            (a, b) = (a.min(b), a.max(b));
            (b, c) = (b.min(c), b.max(c));
            (a, b) = (a.min(b), a.max(b));
            *ptr = a;
            *ptr.add(1) = b;
            *ptr.add(2) = c;
        } else if n == 4 {
            let (mut a, mut b, mut c, mut d) = (*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3));
            (a, b) = (a.min(b), a.max(b));
            (c, d) = (c.min(d), c.max(d));
            (a, c) = (a.min(c), a.max(c));
            (b, d) = (b.min(d), b.max(d));
            (b, c) = (b.min(c), b.max(c));
            *ptr = a;
            *ptr.add(1) = b;
            *ptr.add(2) = c;
            *ptr.add(3) = d;
        } else if n <= 8 {
            bitonic_8_pad(ptr, n);
        } else if n <= 16 {
            bitonic_16_pad(ptr, n);
        } else if n <= 32 {
            bitonic_32_pad(ptr, n);
        } else if n <= 64 {
            bitonic_64_pad(ptr, n);
        } else {
            radix_u32_2pass(ev, tmp);
        }
    }
}

#[inline(always)]
unsafe fn minmax_i64(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let gt = _mm256_cmpgt_epi64(a, b);
    (_mm256_blendv_epi8(a, b, gt), _mm256_blendv_epi8(b, a, gt))
}

#[inline(always)]
unsafe fn stride1_i64<const BLEND: i32>(x: __m256i) -> __m256i {
    let y = _mm256_shuffle_epi32(x, 0x4E);
    let (min, max) = minmax_i64(x, y);
    _mm256_blend_epi32(min, max, BLEND)
}

#[inline(always)]
unsafe fn stride2_i64<const BLEND: i32>(x: __m256i) -> __m256i {
    let y = _mm256_permute4x64_epi64(x, 0x4E);
    let (min, max) = minmax_i64(x, y);
    _mm256_blend_epi32(min, max, BLEND)
}

/// In-place branchless sort for compact `u64` clicks (`ts << 32 | campaign << 16 | local_idx`,
/// where `ts < 2^31` so signed `i64` order == unsigned `u64` order).
#[inline(always)]
unsafe fn sort_u64_clicks(ck: &mut [u64]) {
    let n = ck.len();
    if n <= 1 {
        return;
    }
    let ptr = ck.as_mut_ptr();
    if n == 2 {
        let (a, b) = (*ptr, *ptr.add(1));
        *ptr = a.min(b);
        *ptr.add(1) = a.max(b);
        return;
    }
    if n == 3 {
        let (mut a, mut b, mut c) = (*ptr, *ptr.add(1), *ptr.add(2));
        (a, b) = (a.min(b), a.max(b));
        (b, c) = (b.min(c), b.max(c));
        (a, b) = (a.min(b), a.max(b));
        *ptr = a;
        *ptr.add(1) = b;
        *ptr.add(2) = c;
        return;
    }
    if n == 4 {
        let (mut a, mut b, mut c, mut d) = (*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3));
        (a, b) = (a.min(b), a.max(b));
        (c, d) = (c.min(d), c.max(d));
        (a, c) = (a.min(c), a.max(c));
        (b, d) = (b.min(d), b.max(d));
        (b, c) = (b.min(c), b.max(c));
        *ptr = a;
        *ptr.add(1) = b;
        *ptr.add(2) = c;
        *ptr.add(3) = d;
        return;
    }
    if n <= 8 {
        let m1 = lane_mask_64(n - 4);
        let all_max = _mm256_set1_epi64x(i64::MAX);
        let mut r0 = _mm256_loadu_si256(ptr as *const __m256i);
        let l1 = _mm256_maskload_epi64(ptr.add(4) as *const i64, m1);
        let mut r1 = _mm256_blendv_epi8(all_max, l1, m1);
        r0 = stride1_i64::<0x3C>(r0);
        r1 = stride1_i64::<0x3C>(r1);
        r0 = stride2_i64::<0xF0>(r0);
        r1 = stride2_i64::<0x0F>(r1);
        r0 = stride1_i64::<0xCC>(r0);
        r1 = stride1_i64::<0x33>(r1);
        (r0, r1) = minmax_i64(r0, r1);
        r0 = stride2_i64::<0xF0>(r0);
        r1 = stride2_i64::<0xF0>(r1);
        r0 = stride1_i64::<0xCC>(r0);
        r1 = stride1_i64::<0xCC>(r1);
        _mm256_storeu_si256(ptr as *mut __m256i, r0);
        _mm256_maskstore_epi64(ptr.add(4) as *mut i64, m1, r1);
        return;
    }
    if n <= 16 {
        let m2 = lane_mask_64((n - 8).min(4));
        let m3 = lane_mask_64(n.saturating_sub(12));
        let all_max = _mm256_set1_epi64x(i64::MAX);
        let mut r0 = _mm256_loadu_si256(ptr as *const __m256i);
        let mut r1 = _mm256_loadu_si256(ptr.add(4) as *const __m256i);
        let l2 = _mm256_maskload_epi64(ptr.add(8) as *const i64, m2);
        let mut r2 = _mm256_blendv_epi8(all_max, l2, m2);
        let l3 = _mm256_maskload_epi64(ptr.add(12) as *const i64, m3);
        let mut r3 = _mm256_blendv_epi8(all_max, l3, m3);

        r0 = stride1_i64::<0x3C>(r0);
        r1 = stride1_i64::<0x3C>(r1);
        r2 = stride1_i64::<0x3C>(r2);
        r3 = stride1_i64::<0x3C>(r3);

        r0 = stride2_i64::<0xF0>(r0);
        r1 = stride2_i64::<0x0F>(r1);
        r2 = stride2_i64::<0xF0>(r2);
        r3 = stride2_i64::<0x0F>(r3);
        r0 = stride1_i64::<0xCC>(r0);
        r1 = stride1_i64::<0x33>(r1);
        r2 = stride1_i64::<0xCC>(r2);
        r3 = stride1_i64::<0x33>(r3);

        (r0, r1) = minmax_i64(r0, r1);
        (r3, r2) = minmax_i64(r2, r3);
        r0 = stride2_i64::<0xF0>(r0);
        r1 = stride2_i64::<0xF0>(r1);
        r2 = stride2_i64::<0x0F>(r2);
        r3 = stride2_i64::<0x0F>(r3);
        r0 = stride1_i64::<0xCC>(r0);
        r1 = stride1_i64::<0xCC>(r1);
        r2 = stride1_i64::<0x33>(r2);
        r3 = stride1_i64::<0x33>(r3);

        (r0, r2) = minmax_i64(r0, r2);
        (r1, r3) = minmax_i64(r1, r3);
        (r0, r1) = minmax_i64(r0, r1);
        (r2, r3) = minmax_i64(r2, r3);
        r0 = stride2_i64::<0xF0>(r0);
        r1 = stride2_i64::<0xF0>(r1);
        r2 = stride2_i64::<0xF0>(r2);
        r3 = stride2_i64::<0xF0>(r3);
        r0 = stride1_i64::<0xCC>(r0);
        r1 = stride1_i64::<0xCC>(r1);
        r2 = stride1_i64::<0xCC>(r2);
        r3 = stride1_i64::<0xCC>(r3);

        _mm256_storeu_si256(ptr as *mut __m256i, r0);
        _mm256_storeu_si256(ptr.add(4) as *mut __m256i, r1);
        _mm256_maskstore_epi64(ptr.add(8) as *mut i64, m2, r2);
        _mm256_maskstore_epi64(ptr.add(12) as *mut i64, m3, r3);
        return;
    }
    ck.sort_unstable();
}

/// The campaign of the latest click; on a tie in ts, the smallest click_id.
#[inline(always)]
unsafe fn credit_u64(sh: &Shared, clicks: &[u64], ck_at: &[u32]) -> usize {
    let last = clicks.len() - 1;
    let c_last = *clicks.get_unchecked(last);
    let ts = c_last.ts();
    if last == 0 || (*clicks.get_unchecked(last - 1)).ts() != ts {
        return c_last.campaign();
    }
    credit_tie_u64(sh, clicks, ck_at, last, ts)
}

#[cold]
#[inline(never)]
unsafe fn credit_tie_u64(sh: &Shared, clicks: &[u64], ck_at: &[u32], last: usize, ts: u32) -> usize {
    let camp0 = (*clicks.get_unchecked(last)).campaign();
    let mut start = last - 1;
    let mut all_same = (*clicks.get_unchecked(start)).campaign() == camp0;
    while start > 0 && (*clicks.get_unchecked(start - 1)).ts() == ts {
        start -= 1;
        all_same &= (*clicks.get_unchecked(start)).campaign() == camp0;
    }
    if all_same {
        return camp0;
    }
    let id = |packed: u64| {
        let local_idx = (packed & 0xFFFF) as usize;
        let at = *ck_at.get_unchecked(local_idx);
        (*sh.c_arena.base.add(at as usize)).click
    };
    let mut best = last;
    let mut best_id = id(*clicks.get_unchecked(last));
    let mut k = last;
    while k > start {
        k -= 1;
        let ck_id = id(*clicks.get_unchecked(k));
        if click_less(sh.c_side_arena.base, ck_id, best_id) {
            best = k;
            best_id = ck_id;
        }
    }
    (*clicks.get_unchecked(best)).campaign()
}

/// Fallback `credit` for `Click` struct when a partition exceeds 65,536 clicks or campaigns.
#[inline(always)]
unsafe fn credit_click(sh: &Shared, clicks: &[Click]) -> usize {
    let last = clicks.len() - 1;
    let c_last = clicks.get_unchecked(last);
    let ts = c_last.ts();
    if last == 0 || clicks.get_unchecked(last - 1).ts() != ts {
        return c_last.campaign as usize;
    }
    credit_tie_click(sh, clicks, last, ts)
}

#[cold]
#[inline(never)]
unsafe fn credit_tie_click(sh: &Shared, clicks: &[Click], last: usize, ts: u32) -> usize {
    let camp0 = clicks.get_unchecked(last).campaign;
    let mut start = last - 1;
    let mut all_same = clicks.get_unchecked(start).campaign == camp0;
    while start > 0 && clicks.get_unchecked(start - 1).ts() == ts {
        start -= 1;
        all_same &= clicks.get_unchecked(start).campaign == camp0;
    }
    if all_same {
        return camp0 as usize;
    }
    let id = |c: &Click| (*sh.c_arena.base.add(c.at as usize)).click;
    let mut best = last;
    let mut best_id = id(clicks.get_unchecked(last));
    let mut k = last;
    while k > start {
        k -= 1;
        let ck_id = id(clicks.get_unchecked(k));
        if click_less(sh.c_side_arena.base, ck_id, best_id) {
            best = k;
            best_id = ck_id;
        }
    }
    clicks.get_unchecked(best).campaign as usize
}

/// One thread's working memory for event_id dedup.
#[derive(Default)]
pub struct Dedup {
    bits_pair: Vec<[u64; 2]>,
    buckets: Vec<u32>,
    candidates: Vec<u32>,
    table: Vec<u64>,
}

#[inline(always)]
fn dedup_bucket(event: u64, mask: usize) -> usize {
    ((event_canon_key(event).wrapping_mul(0xD6E8_FEB8_6659_FD93) >> 32) as usize) & mask
}

impl Dedup {
    /// Keeps, per event_id, the copy with the smallest ingest_ts and kills the
    /// payload record of every other copy. Two interleaved bitmaps over a hash
    /// of the ID mark the buckets hit more than once; only records in those
    /// buckets, which include every copy of every duplicate, go through the
    /// exact table.
    pub unsafe fn partition(&mut self, sh: &Shared, chunks: &[(u32, u32)]) {
        const PF_AHEAD: usize = 8;
        let total: usize = chunks.iter().map(|c| c.1 as usize).sum();
        let bits = (total * 8).next_power_of_two().max(64);
        let mask = bits - 1;
        self.bits_pair.clear();
        self.bits_pair.resize(bits / 64, [0; 2]);
        self.buckets.clear();
        self.buckets.reserve(total);
        let base = sh.d_arena.base;
        let (bits_pair, buckets) = (self.bits_pair.as_mut_ptr(), self.buckets.as_mut_ptr());
        let mut k = 0;
        for &(first, count) in chunks {
            let count = count as usize;
            let chunk = base.add(first as usize);
            if count > PF_AHEAD {
                let mut ring = [0usize; PF_AHEAD];
                for j in 0..PF_AHEAD {
                    let b = dedup_bucket((*chunk.add(j)).event, mask);
                    ring[j] = b;
                    _mm_prefetch(bits_pair.add(b / 64) as *const i8, _MM_HINT_T0);
                }
                for j in 0..count - PF_AHEAD {
                    let b = ring[j & (PF_AHEAD - 1)];
                    let nb = dedup_bucket((*chunk.add(j + PF_AHEAD)).event, mask);
                    ring[j & (PF_AHEAD - 1)] = nb;
                    _mm_prefetch(bits_pair.add(nb / 64) as *const i8, _MM_HINT_T0);
                    *buckets.add(k) = b as u32;
                    k += 1;
                    let (w, bit) = (b / 64, 1u64 << (b % 64));
                    let pair = &mut *bits_pair.add(w);
                    pair[1] |= pair[0] & bit;
                    pair[0] |= bit;
                }
                for j in (count - PF_AHEAD)..count {
                    let b = ring[j & (PF_AHEAD - 1)];
                    *buckets.add(k) = b as u32;
                    k += 1;
                    let (w, bit) = (b / 64, 1u64 << (b % 64));
                    let pair = &mut *bits_pair.add(w);
                    pair[1] |= pair[0] & bit;
                    pair[0] |= bit;
                }
            } else {
                for j in 0..count {
                    let b = dedup_bucket((*chunk.add(j)).event, mask);
                    *buckets.add(k) = b as u32;
                    k += 1;
                    let (w, bit) = (b / 64, 1u64 << (b % 64));
                    let pair = &mut *bits_pair.add(w);
                    pair[1] |= pair[0] & bit;
                    pair[0] |= bit;
                }
            }
        }
        self.candidates.clear();
        self.candidates.reserve(total);
        let cands = self.candidates.as_mut_ptr();
        let (mut k, mut nc) = (0usize, 0usize);
        for &(first, count) in chunks {
            for at in first..first + count {
                if k + PF_AHEAD < total {
                    let nb = *buckets.add(k + PF_AHEAD) as usize;
                    _mm_prefetch(bits_pair.add(nb / 64) as *const i8, _MM_HINT_T0);
                }
                let b = *buckets.add(k) as usize;
                k += 1;
                // ~14% of records are candidates at random: a branch here mispredicts.
                *cands.add(nc) = at;
                nc += ((*bits_pair.add(b / 64))[1] >> (b % 64)) as usize & 1;
            }
        }
        self.candidates.set_len(nc);
        let cap = (self.candidates.len() * 2).next_power_of_two().max(16);
        self.table.clear();
        self.table.resize(cap, 0);
        let mask = cap - 1;
        let table = self.table.as_mut_ptr();
        let kill = |r: &DRec| {
            if r.payload != NONE {
                (*sh.p_arena.base.add(r.payload as usize)).person = DEAD;
            }
        };
        const AHEAD: usize = 16;
        let cands = &self.candidates;
        for i in 0..nc {
            if i + AHEAD < nc {
                _mm_prefetch(base.add(*cands.get_unchecked(i + AHEAD) as usize) as *const i8, _MM_HINT_T0);
            }
            let at = *cands.get_unchecked(i);
            let r = &*base.add(at as usize);
            let h = mix(event_canon_key(r.event)).rotate_left(32);
            let tag = ((h >> 20) as u32 | 1) as u64;
            let mut slot = h as usize & mask;
            loop {
                let entry = *table.add(slot);
                if entry == 0 {
                    *table.add(slot) = tag << 32 | at as u64;
                    break;
                }
                if entry >> 32 == tag {
                    let other = &*base.add(entry as u32 as usize);
                    if event_eq(sh.e_side_arena.base, other.event, r.event) {
                        if r.ingest < other.ingest {
                            kill(other);
                            *table.add(slot) = tag << 32 | at as u64;
                        } else {
                            kill(r);
                        }
                        break;
                    }
                }
                slot = (slot + 1) & mask;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::tok;
    use std::path::Path;
    use std::process::Command;

    fn make_tok(s: &str) -> u64 {
        let mut buf = [0u8; 32];
        let b = s.as_bytes();
        buf[..b.len()].copy_from_slice(b);
        unsafe { tok(buf.as_ptr(), b.len()) }
    }

    #[test]
    fn test_user_out_empty_even_if_empty() {
        let tmp_dir = std::env::temp_dir();
        let skip_path = tmp_dir.join("user_out_empty_skip.parquet");
        let keep_path = tmp_dir.join("user_out_empty_keep.parquet");
        let _ = std::fs::remove_file(&skip_path);
        let _ = std::fs::remove_file(&keep_path);

        UserOut::new(skip_path.clone()).finish(false);
        assert!(!skip_path.exists(), "even_if_empty=false must not create file");

        UserOut::new(keep_path.clone()).finish(true);
        assert!(keep_path.exists(), "even_if_empty=true must create schema-only file");

        let venv_py = Path::new("/home/colton/Documents/Projects/Builders/#3/.venv/bin/python");
        if venv_py.exists() {
            let out = Command::new(venv_py)
                .arg("-c")
                .arg("import sys, polars as pl; df = pl.read_parquet(sys.argv[1]); assert df.height == 0 and df.columns == ['user_id', 'day', 'sessions', 'events', 'revenue_cents', 'top_category']")
                .arg(keep_path.to_str().unwrap())
                .output()
                .unwrap();
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        }

        let _ = std::fs::remove_file(&keep_path);
    }

    #[test]
    fn test_user_out_streaming_65k_and_i64_overflow() {
        let tmp_dir = std::env::temp_dir().join("user_out_stream_test");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let cats = vec![b"books".to_vec(), b"games".to_vec(), b"music".to_vec()];
        let stream_path = tmp_dir.join("stream.parquet");

        // 1. Stream >65,536 zero rows AND >65,536 nonz rows via UserOut
        let mut exp_rows = 0usize;
        let mut exp_nulls = 0usize;
        let mut exp_rev = 0u64;
        let mut exp_sess = 0u64;
        let mut exp_ev = 0u64;
        {
            let mut out = UserOut::with_categories(stream_path.clone(), &cats);
            for p in 0..70_000u32 {
                let pt = make_tok(&format!("u{p:06}"));
                out.begin_person();
                // 1 zero-rev day + 1 nonz day per person => triggers both 65,536 flushes
                out.add(pt, 20727 + (p % 10) as i32, 1, 2, 0, None);
                exp_rows += 1;
                exp_nulls += 1;
                exp_sess += 1;
                exp_ev += 2;

                let rev = 100 + (p % 500) as u64;
                let cat = if p % 5 == 0 { None } else { Some((p % 3) as u8) };
                out.add(pt, 20728 + (p % 10) as i32, 2, 5, rev, cat);
                exp_rows += 1;
                if cat.is_none() {
                    exp_nulls += 1;
                }
                exp_rev += rev;
                exp_sess += 2;
                exp_ev += 5;
            }
            out.finish(true);
        }

        // 2. Overflow > i32::MAX BEFORE any flush (single file)
        let ovf_single_dir = tmp_dir.join("ovf_single");
        std::fs::create_dir_all(&ovf_single_dir).unwrap();
        let ovf_single_path = ovf_single_dir.join("part-000.parquet");
        let big_rev = (i32::MAX as u64) + 987_654_321;
        {
            let mut out = UserOut::with_categories(ovf_single_path.clone(), &cats);
            let p0 = make_tok("whale0");
            out.begin_person();
            out.add(p0, 20727, 1, 1, 0, None);
            out.add(p0, 20728, 1, 2, 500, Some(0));
            out.add(p0, 20729, 3, 9, big_rev, Some(1));
            out.finish(true);
        }

        // 3. Overflow > i32::MAX AFTER >65,536 zero-rev rows already flushed as INT32
        let ovf_spill_dir = tmp_dir.join("ovf_spill");
        std::fs::create_dir_all(&ovf_spill_dir).unwrap();
        let ovf_spill_path = ovf_spill_dir.join("part-000.parquet");
        {
            let mut out = UserOut::with_categories(ovf_spill_path.clone(), &cats);
            for p in 0..66_000u32 {
                let pt = make_tok(&format!("z{p:06}"));
                out.begin_person();
                out.add(pt, 20727, 1, 1, 0, None);
            }
            let pw = make_tok("whale_late");
            out.begin_person();
            out.add(pw, 20730, 4, 12, big_rev, Some(2));
            out.finish(true);
        }

        let venv_py = Path::new("/home/colton/Documents/Projects/Builders/#3/.venv/bin/python");
        if venv_py.exists() {
            let script = r#"
import sys
import polars as pl
import duckdb

p_stream, p_single, d_spill = sys.argv[1], sys.argv[2], sys.argv[3]
df_s = pl.read_parquet(p_stream)
print(f"STREAM rows={df_s.height} nulls={df_s['top_category'].null_count()} rev={df_s['revenue_cents'].sum()} sess={df_s['sessions'].sum()} ev={df_s['events'].sum()}")

df_o1 = pl.read_parquet(p_single)
print(f"OVF1 rows={df_o1.height} rev={df_o1['revenue_cents'].sum()}")

duck_o2 = duckdb.sql(f"SELECT COUNT(*), SUM(revenue_cents) FROM read_parquet('{d_spill}/*.parquet')").fetchall()[0]
print(f"OVF2 rows={duck_o2[0]} rev={duck_o2[1]}")
"#;
            let out = Command::new(venv_py)
                .arg("-c")
                .arg(script)
                .arg(stream_path.to_str().unwrap())
                .arg(ovf_single_path.to_str().unwrap())
                .arg(ovf_spill_dir.to_str().unwrap())
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "Python verifier failed:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let stdout = String::from_utf8_lossy(&out.stdout);
            let lines: Vec<&str> = stdout.lines().collect();
            assert_eq!(
                lines[0],
                format!("STREAM rows={exp_rows} nulls={exp_nulls} rev={exp_rev} sess={exp_sess} ev={exp_ev}")
            );
            assert_eq!(lines[1], format!("OVF1 rows=3 rev={}", 500 + big_rev));
            assert_eq!(lines[2], format!("OVF2 rows=66001 rev={big_rev}"));
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn sort_u32_avx2_exhaustive_0_to_150_and_stress() {
        fn verify_sorted(mut buf: Vec<u32>, n: usize, tmp: &mut Vec<u32>) {
            const SENTINEL: u32 = 0xDEAD_BEEF;
            let mut expected = buf[..n].to_vec();
            expected.sort_unstable();
            buf.resize(n + 32, SENTINEL);
            sort_u32_avx2(&mut buf[..n], tmp);
            assert_eq!(&buf[..n], expected.as_slice(), "mismatch for n={n}");
            assert!(buf[n..].iter().all(|&x| x == SENTINEL), "sentinel clobbered for n={n}");
        }

        let mut tmp = Vec::new();
        let mut pat = [0u32; 16];
        for n in 2..=14usize {
            let total = 1usize << n;
            for mask in 0..total {
                for i in 0..n {
                    pat[i] = ((mask >> i) & 1) as u32;
                }
                let mut buf = pat[..n].to_vec();
                let ones = (mask as u32).count_ones() as usize;
                sort_u32_avx2(&mut buf, &mut tmp);
                assert!(buf[..n - ones].iter().all(|&x| x == 0), "n={n} mask={mask}");
                assert!(buf[n - ones..].iter().all(|&x| x == 1), "n={n} mask={mask}");
            }
        }

        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_u64 = || -> u64 {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            rng.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        // Fuzz EVERY length n in 0..=150 plus larger boundary sizes for radix_u32_2pass.
        let mut lengths: Vec<usize> = (0..=150).collect();
        lengths.extend_from_slice(&[255, 256, 257, 512, 1025]);
        for &n in &lengths {
            verify_sorted(vec![0; n], n, &mut tmp);
            verify_sorted(vec![u32::MAX - 2; n], n, &mut tmp);
            verify_sorted(vec![u32::MAX; n], n, &mut tmp);
            verify_sorted((0..n as u32).collect(), n, &mut tmp);
            verify_sorted((0..n as u32).rev().collect(), n, &mut tmp);
            let edge: Vec<u32> = (0..n)
                .map(|i| match i % 4 {
                    0 => 0,
                    1 => u32::MAX - 2,
                    2 => u32::MAX,
                    _ => 42,
                })
                .collect();
            verify_sorted(edge, n, &mut tmp);

            let iters = if n <= 64 { 200 } else { 40 };
            for iter in 0..iters {
                let mode = iter % 5;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    let x = match mode {
                        0 => next_u64() as u32,
                        1 => match next_u64() % 4 {
                            0 => 0,
                            1 => u32::MAX - 2,
                            2 => u32::MAX,
                            _ => (next_u64() & 0xFF) as u32,
                        },
                        2 => (next_u64() % 7) as u32,
                        3 => (next_u64() >> 33) as u32, // valid ts < 2^31
                        _ => (next_u64() as u32) & 0x00FF_00FF,
                    };
                    v.push(x);
                }
                verify_sorted(v, n, &mut tmp);
            }
        }
    }

    #[test]
    fn sort_u64_clicks_exhaustive() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_u64 = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };
        const SENTINEL: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut lengths: Vec<usize> = (0..=32).collect();
        lengths.push(64);
        for n in lengths {
            for case in 0..200 {
                let mut v = Vec::with_capacity(n + 8);
                for idx in 0..n {
                    let ts: u32 = match case {
                        0 => (idx as u32) * 100,
                        1 => ((n - 1 - idx) as u32) * 100,
                        2 => 1_073_741_824,
                        3 => (1u32 << 31) - 1,
                        4 => (idx as u32 % 3) * 12345,
                        _ => (next_u64() >> 33) as u32,
                    };
                    let camp = (idx as u64) % 7;
                    v.push(((ts as u64) << 32) | (camp << 16) | (idx as u64));
                }
                let mut expected = v.clone();
                expected.sort_unstable();

                v.resize(n + 8, SENTINEL);
                unsafe { sort_u64_clicks(&mut v[..n]) };

                assert_eq!(&v[..n], expected.as_slice(), "n={n} case={case}");
                assert!(v[n..].iter().all(|&x| x == SENTINEL), "sentinel clobbered for n={n}");
            }
        }
    }

    fn make_test_shared(days: usize, num_campaigns: usize, arena_cap: usize) -> Shared {
        let first_day = 20_000i64;
        let window_start = first_day * DAY_MS;
        let window_end = window_start + (days as i64) * DAY_MS;
        let base_ms = window_start - crate::WEEK_MS;
        Shared {
            window_start,
            window_end,
            base_ms,
            use_rel: true,
            date_lut: crate::ts::DateLut::new(window_start, window_end, base_ms.div_euclid(1000)),
            first_day,
            days,
            products: crate::dims::KeyMap::new(&[]),
            product_category: Vec::new(),
            direct_prod_cat: Vec::new(),
            unknown_category: 0,
            categories: (0..8).map(|c| format!("cat{c}").into_bytes()).collect(),
            campaign_ids: crate::dims::KeyMap::new(&[]),
            direct_camp: Vec::new(),
            campaigns: (0..num_campaigns)
                .map(|c| (format!("camp{c}").into_bytes(), b"ch".to_vec()))
                .collect(),
            links: crate::links::LinkTable::new(0),
            d_arena: crate::scatter::Arena::new(arena_cap, 64),
            p_arena: crate::scatter::Arena::new(arena_cap, 64),
            c_arena: crate::scatter::Arena::new(arena_cap, 64),
            e_side_arena: crate::scatter::Arena::new(arena_cap, 64),
            c_side_arena: crate::scatter::Arena::new(arena_cap, 64),
            d_parts: 1,
            p_parts: 1,
        }
    }

    /// Simple textbook scalar reference for one person's sorted events, purchases, and clicks.
    unsafe fn ref_person_scalar(
        g: &mut Grouper,
        sh: &Shared,
        person: u64,
        events: &[u32],
        purchases: &[(u64, u32)],
        clicks: &[Click],
    ) {
        if events.is_empty() {
            return;
        }
        g.out.begin_person();
        let nck = clicks.len();
        let mut next_purchase = 0usize;
        g.starts.clear();
        g.sess_row.clear();
        g.day_rows.clear();
        let days = sh.days;
        let organic = sh.campaigns.len();
        let n = events.len();
        let (mut i, mut next_click) = (0usize, 0usize);
        let day_origin = (sh.first_day * DAY_MS - sh.base_ms) as u32;
        let mut day = u32::MAX;
        let (mut day_sessions, mut day_events, mut day_revenue, mut day_first) = (0u32, 0u32, 0u64, 0u32);
        let mut session_index = 0u32;
        while i < n {
            let start = events[i];
            let start_day = (start - day_origin) / DAY_MS_U32;
            if start_day != day {
                if day != u32::MAX {
                    if day_revenue > 0 {
                        g.flush_day::<true>(sh, person, day, day_sessions, day_events, day_revenue);
                    } else {
                        g.flush_day::<false>(sh, person, day, day_sessions, day_events, 0);
                    }
                    g.day_rows.push(DayRow { day, first: day_first, end: session_index });
                }
                day = start_day;
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
                day_first = session_index;
            }
            while next_click < nck && clicks[next_click].ts() <= start {
                next_click += 1;
            }
            let campaign = if next_click > 0 && start - clicks[next_click - 1].ts() <= WEEK {
                credit_click(sh, &clicks[..next_click])
            } else {
                organic
            };
            let mut prev = start;
            let mut count = 0u32;
            while i < n && events[i] - prev <= SESSION_GAP {
                prev = events[i];
                count += 1;
                i += 1;
            }
            let next_sess_ts = if i < n { events[i] } else { u32::MAX };
            let mut revenue = 0u64;
            while next_purchase < purchases.len() && (purchases[next_purchase].0 as u32) < next_sess_ts {
                let payload = purchases[next_purchase].1;
                next_purchase += 1;
                let (cat, price) = ((payload >> 24) as u8, (payload & 0xFF_FFFF) as u64);
                revenue += price;
                if g.cat_sum[cat as usize] == 0 {
                    g.cats.push(cat);
                }
                g.cat_sum[cat as usize] += price;
            }
            day_sessions += 1;
            day_events += count;
            day_revenue += revenue;
            let cell_idx = campaign * days + start_day as usize;
            g.cell_sessions[cell_idx] += 1;
            if revenue > 0 {
                g.cell_purch[cell_idx].0 += revenue;
                g.cell_purch[cell_idx].1 += 1;
            }
            session_index += 1;
            if nck > 0 {
                g.starts.push(start);
                g.sess_row.push(g.day_rows.len() as u32);
            }
        }
        if day_revenue > 0 {
            g.flush_day::<true>(sh, person, day, day_sessions, day_events, day_revenue);
        } else {
            g.flush_day::<false>(sh, person, day, day_sessions, day_events, 0);
        }
        if nck > 0 {
            g.day_rows.push(DayRow { day, first: day_first, end: session_index });
            g.starts.push(u32::MAX);
            g.assists(days, clicks);
        }
    }

    #[test]
    fn test_simd_sess_matches_scalar_20k() {
        let days = 30usize;
        let num_campaigns = 5usize;
        let sh = make_test_shared(days, num_campaigns, 256);
        let day_origin = (sh.first_day * DAY_MS - sh.base_ms) as u32;
        let cats: Vec<Vec<u8>> = sh.categories.clone();
        let cells_len = (num_campaigns + 1) * days;

        let mut ck_at = Vec::with_capacity(128);
        for idx in 0..128usize {
            let click_tok = make_tok(&format!("clk{:04}", 1000 - (idx * 7) % 101));
            unsafe {
                *sh.c_arena.base.add(idx) = CRec {
                    person: 1,
                    click: click_tok,
                    ts: 0,
                    campaign: (idx % num_campaigns) as u32,
                };
            }
            ck_at.push(idx as u32);
        }

        let mut g_simd = Grouper::new(PathBuf::new(), cells_len, &cats);
        let mut g_scal = Grouper::new(PathBuf::new(), cells_len, &cats);
        g_simd.covered.resize(num_campaigns, 0);
        g_scal.covered.resize(num_campaigns, 0);

        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut next_u64 = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };

        for stream in 0..20_000usize {
            let person = make_tok(&format!("p{stream:06}"));
            let n_ev = match stream % 20 {
                0 => 9,
                1 => 10,
                2 => 16,
                3 => 17,
                4 => 33,
                5 => 1,
                _ => 1 + (next_u64() as usize % 40),
            };
            let n_ck = match stream % 7 {
                0 => 0,
                1 => 1,
                _ => (next_u64() as usize) % 12,
            };

            let mut clicks_u64 = Vec::with_capacity(n_ck);
            let mut clicks_fb = Vec::with_capacity(n_ck);
            let mut ck_ts = day_origin.saturating_sub(crate::WEEK_MS as u32) + (next_u64() as u32 % (3 * DAY_MS_U32));
            for c_idx in 0..n_ck {
                let step = match next_u64() % 5 {
                    0 => 0,
                    1 => DAY_MS_U32,
                    _ => next_u64() as u32 % (2 * DAY_MS_U32),
                };
                ck_ts = ck_ts.saturating_add(step).min(day_origin + (days as u32 - 1) * DAY_MS_U32);
                let camp = (next_u64() as usize % num_campaigns) as u32;
                let at = ((stream * 13 + c_idx) % 128) as u32;
                unsafe {
                    (*sh.c_arena.base.add(at as usize)).campaign = camp;
                }
                clicks_u64.push(((ck_ts as u64) << 32) | ((camp as u64) << 16) | (at as u64));
                clicks_fb.push(Click {
                    key: ((ck_ts as u64) << 32) | (c_idx as u64),
                    campaign: camp,
                    at,
                });
            }
            clicks_u64.sort_unstable();
            clicks_fb.sort_by_key(|c| c.ts());

            let mut events = Vec::with_capacity(n_ev);
            let mut purchases = Vec::new();
            let mut ts = day_origin + (next_u64() as u32 % (5 * DAY_MS_U32));
            let max_ts = day_origin + (days as u32 - 1) * DAY_MS_U32;
            for ev_idx in 0..n_ev {
                let delta = match (stream + ev_idx) % 8 {
                    0 => 0,
                    1 => SESSION_GAP,
                    2 => SESSION_GAP + 1,
                    3 => DAY_MS_U32,
                    _ => (next_u64() as u32) % (3 * SESSION_GAP),
                };
                if ts + delta <= max_ts {
                    ts += delta;
                }
                events.push(ts);
                if next_u64() % 4 == 0 {
                    let cat = (next_u64() as u32 % 8) << 24;
                    let price = 1 + (next_u64() as u32 % 100_000);
                    purchases.push(((1u64 << 32) | (ts as u64), cat | price));
                }
            }

            unsafe {
                g_simd.person(&sh, person, &events, &purchases, &clicks_u64, &ck_at);
                ref_person_scalar(&mut g_scal, &sh, person, &events, &purchases, &clicks_fb);
            }

            if stream % 500 == 499 || stream == 19_999 {
                assert_eq!(g_simd.assisted, g_scal.assisted, "stream {stream}: assisted mismatch");
                assert_eq!(g_simd.cell_sessions, g_scal.cell_sessions, "stream {stream}: cell_sessions");
                assert_eq!(g_simd.cell_purch, g_scal.cell_purch, "stream {stream}: cell_purch");
                assert_eq!(g_simd.out.z_names_plain, g_scal.out.z_names_plain, "stream {stream}: z_names_plain");
                assert_eq!(g_simd.out.z_names_count, g_scal.out.z_names_count, "stream {stream}: z_names_count");
                assert_eq!(g_simd.out.z_person_keys, g_scal.out.z_person_keys, "stream {stream}: z_person_keys");
                assert_eq!(g_simd.out.z_day, g_scal.out.z_day, "stream {stream}: z_day");
                assert_eq!(g_simd.out.z_sessions, g_scal.out.z_sessions, "stream {stream}: z_sessions");
                assert_eq!(g_simd.out.z_events, g_scal.out.z_events, "stream {stream}: z_events");
                assert_eq!(g_simd.out.nz_names_plain, g_scal.out.nz_names_plain, "stream {stream}: nz_names_plain");
                assert_eq!(g_simd.out.nz_names_count, g_scal.out.nz_names_count, "stream {stream}: nz_names_count");
                assert_eq!(g_simd.out.nz_person_keys, g_scal.out.nz_person_keys, "stream {stream}: nz_person_keys");
                assert_eq!(g_simd.out.nz_day, g_scal.out.nz_day, "stream {stream}: nz_day");
                assert_eq!(g_simd.out.nz_sessions, g_scal.out.nz_sessions, "stream {stream}: nz_sessions");
                assert_eq!(g_simd.out.nz_events, g_scal.out.nz_events, "stream {stream}: nz_events");
                assert_eq!(g_simd.out.nz_revenue, g_scal.out.nz_revenue, "stream {stream}: nz_revenue");
                assert_eq!(g_simd.out.nz_top_category, g_scal.out.nz_top_category, "stream {stream}: nz_top_category");

                for out in [&mut g_simd.out, &mut g_scal.out] {
                    out.z_names_plain.clear();
                    out.z_names_count = 0;
                    out.z_person_keys.clear();
                    out.z_day.clear();
                    out.z_sessions.clear();
                    out.z_events.clear();
                    out.nz_names_plain.clear();
                    out.nz_names_count = 0;
                    out.nz_person_keys.clear();
                    out.nz_day.clear();
                    out.nz_sessions.clear();
                    out.nz_events.clear();
                    out.nz_revenue.clear();
                    out.nz_top_category.clear();
                }
            }
        }
    }

    #[test]
    fn test_assists_sentinel_boundary() {
        let days = 4usize;
        let num_campaigns = 2usize;
        let boundary_ts: u32 = (i32::MAX as u32) - WEEK;
        let max_ts: u32 = i32::MAX as u32;

        // 1. Single-day fast path at boundary_ts
        let mut g = Grouper::new(PathBuf::new(), (num_campaigns + 1) * days, &[]);
        g.covered.resize(num_campaigns, 0);
        g.starts = vec![boundary_ts - 1, boundary_ts, max_ts, u32::MAX];
        g.day_rows = vec![DayRow { day: 2, first: 0, end: 3 }];
        g.sess_row = vec![0, 0, 0];
        let clicks: Vec<u64> = vec![
            ((boundary_ts as u64) << 32) | (0u64 << 16) | 0,
            ((max_ts as u64) << 32) | (1u64 << 16) | 1,
        ];
        unsafe { g.assists(days, &clicks) };
        assert_eq!(g.assisted[0 * days + 2], 2);
        assert_eq!(g.assisted[1 * days + 2], 1);

        // 2. Multi-day path with click past all sessions (lo == ns hits u32::MAX sentinel)
        g.assisted.fill(0);
        g.starts = vec![boundary_ts - 10, boundary_ts, max_ts - 1, u32::MAX];
        g.day_rows = vec![
            DayRow { day: 0, first: 0, end: 1 },
            DayRow { day: 1, first: 1, end: 3 },
        ];
        g.sess_row = vec![0, 1, 1];
        let clicks2: Vec<u64> = vec![
            ((boundary_ts as u64) << 32) | (0u64 << 16) | 0,
            ((max_ts as u64) << 32) | (0u64 << 16) | 1,
        ];
        unsafe { g.assists(days, &clicks2) };
        assert_eq!(g.assisted[0 * days + 0], 0);
        assert_eq!(g.assisted[0 * days + 1], 2);

        // 3. Empty real sessions (ns == 0, only sentinel u32::MAX)
        g.assisted.fill(0);
        g.starts = vec![u32::MAX];
        g.day_rows = vec![DayRow { day: 0, first: 0, end: 0 }];
        g.sess_row.clear();
        unsafe { g.assists(days, &clicks2) };
        assert!(g.assisted.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_partition_avx2_vs_radix() {
        let days = 10usize;
        let num_campaigns = 4usize;
        let sh = make_test_shared(days, num_campaigns, 4096);
        let day_origin = (sh.first_day * DAY_MS - sh.base_ms) as u32;
        let cells_len = (num_campaigns + 1) * days;

        let mut seed = 0xCAFE_BABE_1234_5678u64;
        let mut next_u64 = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };

        let people: Vec<u64> = (0..120).map(|i| make_tok(&format!("part_u{i:04}"))).collect();
        let mut p_len = 0usize;
        let mut c_len = 0usize;

        for idx in 0..800usize {
            let p_idx = if idx < 300 { 0 } else if idx < 400 { 1 } else { idx % people.len() };
            let person = if idx % 19 == 0 { DEAD } else { people[p_idx] };
            let ts = day_origin + (next_u64() as u32 % (days as u32 * DAY_MS_U32));
            let payload = if next_u64() % 5 == 0 {
                let cat = (next_u64() as u32 % 8) << 24;
                let price = 1 + (next_u64() as u32 % 10_000);
                cat | price
            } else {
                0
            };
            unsafe {
                *sh.p_arena.base.add(p_len) = PRec { person, ts, payload };
            }
            p_len += 1;
        }

        for idx in 0..250usize {
            let p_idx = if idx % 3 == 0 { 0 } else { idx % people.len() };
            // Include deliberate equal-ts multi-campaign ties to exercise Guardrail C2.
            let ts = if idx % 10 < 3 {
                day_origin + 123_456
            } else {
                day_origin.saturating_sub(WEEK / 2) + (next_u64() as u32 % (days as u32 * DAY_MS_U32))
            };
            let click = make_tok(&format!("ck_{:04}", (idx * 37) % 97));
            let campaign = (next_u64() as usize % num_campaigns) as u32;
            unsafe {
                *sh.c_arena.base.add(c_len) = CRec {
                    person: people[p_idx],
                    click,
                    ts,
                    campaign,
                };
            }
            c_len += 1;
        }

        let ev_chunks = [(0u32, p_len as u32)];
        let ck_chunks = [(0u32, c_len as u32)];

        let mut g_avx2 = Grouper::new(PathBuf::new(), cells_len, &sh.categories);
        g_avx2.sort_mode = SortMode::Avx2;
        let mut g_radix = Grouper::new(PathBuf::new(), cells_len, &sh.categories);
        g_radix.sort_mode = SortMode::Radix;

        unsafe {
            g_avx2.partition(&sh, &ev_chunks, &ck_chunks);
            g_radix.partition(&sh, &ev_chunks, &ck_chunks);
        }

        let c_avx2 = g_avx2.take_cells();
        let c_radix = g_radix.take_cells();
        for (i, (a, r)) in c_avx2.iter().zip(&c_radix).enumerate() {
            assert_eq!(a.revenue, r.revenue, "cell {i} revenue");
            assert_eq!(a.sessions, r.sessions, "cell {i} sessions");
            assert_eq!(a.purchasing, r.purchasing, "cell {i} purchasing");
            assert_eq!(a.assisted, r.assisted, "cell {i} assisted");
        }
        assert_eq!(g_avx2.out.z_names_plain, g_radix.out.z_names_plain);
        assert_eq!(g_avx2.out.z_day, g_radix.out.z_day);
        assert_eq!(g_avx2.out.z_sessions, g_radix.out.z_sessions);
        assert_eq!(g_avx2.out.z_events, g_radix.out.z_events);
        assert_eq!(g_avx2.out.nz_names_plain, g_radix.out.nz_names_plain);
        assert_eq!(g_avx2.out.nz_day, g_radix.out.nz_day);
        assert_eq!(g_avx2.out.nz_sessions, g_radix.out.nz_sessions);
        assert_eq!(g_avx2.out.nz_events, g_radix.out.nz_events);
        assert_eq!(g_avx2.out.nz_revenue, g_radix.out.nz_revenue);
        assert_eq!(g_avx2.out.nz_top_category, g_radix.out.nz_top_category);
    }
}
