use crate::fast_parquet::{FastUserDailyWriter, UserDailyCols, NULL_CAT};
use crate::ids::{append_tok_bytes, click_less, event_canon_key, event_eq};
use crate::ts::DAY_MS;
use crate::sys::mix;
use crate::{CRec, DRec, PRec, Shared, DEAD, NONE};
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
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
    names_plain.extend_from_slice(&[0u8; 4]);
    append_tok_bytes(person_tok, names_plain);
    let tok_len = (names_plain.len() - old - 4) as u32;
    names_plain[old..old + 4].copy_from_slice(&tok_len.to_le_bytes());
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
        UserOut {
            path,
            writer: None,
            spill_idx: 0,

            z_names_plain: Vec::with_capacity(if z_cap > 0 { 256 << 10 } else { 0 }),
            z_names_count: 0,
            z_person_keys: Vec::with_capacity(z_cap),
            z_day: Vec::with_capacity(z_cap),
            z_sessions: Vec::with_capacity(z_cap),
            z_events: Vec::with_capacity(z_cap),
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
            let pk = if self.cur_person_z_key == u32::MAX {
                let k = append_plain_tok(person_tok, &mut self.z_names_plain, &mut self.z_names_count);
                self.cur_person_z_key = k;
                k
            } else {
                self.cur_person_z_key
            };
            self.z_person_keys.push(pk);
            self.z_day.push(day);
            self.z_sessions.push(sessions as i32);
            self.z_events.push(events as i32);
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

/// A click in the sort: `key` is local person << 32 | ts << 1, `at` its CRec index.
#[derive(Clone, Copy)]
struct Click {
    key: u64,
    campaign: u32,
    at: u32,
}

impl Click {
    #[inline(always)]
    fn ts(&self) -> u32 {
        self.key as u32 >> 1
    }
}

struct DayRow {
    day: u32,
    first: u32,
    end: u32,
}

/// One thread's working memory for turning a person partition into rows.
pub struct Grouper {
    keys: Vec<u64>,
    ids: Vec<u32>,
    people: Vec<u64>,
    ev_count: Vec<u32>,
    ck_count: Vec<u32>,
    /// Event sort keys: local person << 32 | ts << 1 | is_purchase.
    ev_a: Vec<u64>,
    ev_b: Vec<u64>,
    ck_a: Vec<Click>,
    ck_b: Vec<Click>,
    /// (sort key, payload) of every purchase, in sort-key order once sorted.
    purchases: Vec<(u64, u32)>,
    hist: Box<[[u32; RADIX]; 4]>,
    starts: Vec<u32>,
    day_rows: Vec<DayRow>,
    sess_row: Vec<u32>,
    covered: Vec<u64>,
    generation: u32,
    cat_sum: [u64; 256],
    cats: Vec<u8>,
    pub out: UserOut,
    cells: Vec<Cell>,
    /// Kept apart from `cells` so the per-click updates touch a smaller table.
    assisted: Vec<u32>,
}

impl Grouper {
    pub fn new(path: PathBuf, cells: usize, categories: &[Vec<u8>]) -> Self {
        Grouper {
            keys: vec![0; 4096],
            ids: vec![0; 4096],
            people: Vec::new(),
            ev_count: Vec::new(),
            ck_count: Vec::new(),
            ev_a: Vec::new(),
            ev_b: Vec::new(),
            ck_a: Vec::new(),
            ck_b: Vec::new(),
            purchases: Vec::new(),
            hist: Box::new([[0; RADIX]; 4]),
            starts: Vec::new(),
            day_rows: Vec::new(),
            sess_row: Vec::new(),
            covered: Vec::new(),
            generation: 0,
            cat_sum: [0; 256],
            cats: Vec::new(),
            out: UserOut::with_categories(path, categories),
            cells: vec![Cell::default(); cells],
            assisted: vec![0; cells],
        }
    }

    fn grow(&mut self) {
        let cap = self.keys.len() * 2;
        self.keys = vec![0; cap];
        self.ids = vec![0; cap];
        for (id, &person) in self.people.iter().enumerate() {
            let mut at = slot_of(person, cap - 1);
            while self.keys[at] != 0 {
                at = (at + 1) & (cap - 1);
            }
            self.keys[at] = person;
            self.ids[at] = id as u32;
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
        self.people.push(person);
        self.ev_count.push(0);
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
        self.keys.fill(0);
        self.people.clear();
        self.ev_count.clear();
        self.purchases.clear();
        for h in self.hist.iter_mut() {
            h.fill(0);
        }
        let total: usize = events.iter().map(|c| c.1 as usize).sum();
        self.ev_a.clear();
        self.ev_a.reserve(total);
        for &(first, count) in events {
            let recs: &[PRec] = std::slice::from_raw_parts(sh.p_arena.base.add(first as usize), count as usize);
            for r in recs {
                if r.person == DEAD {
                    continue;
                }
                let id = self.person_id(r.person, true);
                *self.ev_count.get_unchecked_mut(id as usize) += 1;
                let key = (id as u64) << 32 | (r.ts as u64) << 1 | (r.payload != 0) as u64;
                let n = self.ev_a.len();
                *self.ev_a.as_mut_ptr().add(n) = key;
                self.ev_a.set_len(n + 1);
                *self.hist[0].get_unchecked_mut(digit0(key)) += 1;
                *self.hist[1].get_unchecked_mut(digit1(key)) += 1;
                if r.payload != 0 {
                    self.purchases.push((key, r.payload));
                }
            }
        }
        let people = self.people.len();
        if people == 0 {
            return;
        }
        let total: usize = clicks.iter().map(|c| c.1 as usize).sum();
        self.ck_a.clear();
        self.ck_a.reserve(total);
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
                let key = (id as u64) << 32 | (r.ts as u64) << 1;
                let n = self.ck_a.len();
                *self.ck_a.as_mut_ptr().add(n) = Click { key, campaign: r.campaign, at: first + k as u32 };
                self.ck_a.set_len(n + 1);
                *self.hist[2].get_unchecked_mut(digit0(key)) += 1;
                *self.hist[3].get_unchecked_mut(digit1(key)) += 1;
            }
        }
        let [h0, h1, h2, h3] = &mut *self.hist;
        radix3(&mut self.ev_a, &mut self.ev_b, h0, h1, &mut self.ev_count, |&k| k);
        radix3(&mut self.ck_a, &mut self.ck_b, h2, h3, &mut self.ck_count, |c| c.key);
        self.purchases.sort_unstable_by_key(|p| p.0);
        let mut events = std::mem::take(&mut self.ev_b);
        let mut clicks = std::mem::take(&mut self.ck_b);
        let purchases = std::mem::take(&mut self.purchases);
        let (mut ev_from, mut ck_from, mut pu_from) = (0usize, 0usize, 0usize);
        for id in 0..people {
            // After the scatter each count slot holds its person's end offset.
            let (ev_end, ck_end) = (self.ev_count[id] as usize, self.ck_count[id] as usize);
            let ev = &mut events[ev_from..ev_end];
            let pu_end = pu_from + ev.iter().filter(|&&k| k & 1 != 0).count();
            let ck = &mut clicks[ck_from..ck_end];
            fix_order(ck, |c| c.key);
            fix_order(ev, |&k| k);
            self.person(sh, self.people[id], ev, &purchases[pu_from..pu_end], ck);
            (ev_from, ck_from, pu_from) = (ev_end, ck_end, pu_end);
        }
        self.ev_b = events;
        self.ck_b = clicks;
        self.purchases = purchases;
    }

    #[allow(dead_code)]
    pub fn take_cells(&mut self) -> Vec<Cell> {
        for (c, &a) in self.cells.iter_mut().zip(&self.assisted) {
            c.assisted = a;
        }
        std::mem::take(&mut self.cells)
    }

    pub fn finish(mut self, even_if_empty: bool) -> Vec<Cell> {
        self.out.finish(even_if_empty);
        for (c, &a) in self.cells.iter_mut().zip(&self.assisted) {
            c.assisted = a;
        }
        self.cells
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
        let best = if HAS_PURCH {
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
            best
        } else {
            None
        };
        self.out.add(person, (sh.first_day + day as i64) as i32, sessions, events, revenue, best);
    }

    #[inline(always)]
    unsafe fn person(&mut self, sh: &Shared, person: u64, events: &[u64], purchases: &[(u64, u32)], clicks: &[Click]) {
        if purchases.is_empty() {
            self.person_impl::<false>(sh, person, events, purchases, clicks);
        } else {
            self.person_impl::<true>(sh, person, events, purchases, clicks);
        }
    }

    unsafe fn person_impl<const HAS_PURCH: bool>(
        &mut self,
        sh: &Shared,
        person: u64,
        events: &[u64],
        purchases: &[(u64, u32)],
        clicks: &[Click],
    ) {
        self.out.begin_person();
        let nck = clicks.len();
        let mut next_purchase = 0;
        self.starts.clear();
        self.sess_row.clear();
        self.day_rows.clear();
        let days = sh.days;
        let organic = sh.campaigns.len();
        let n = events.len();
        let (mut i, mut next_click) = (0usize, 0usize);
        let (mut cached_nc, mut cached_camp) = (0usize, 0usize);
        let day_origin = (sh.first_day * DAY_MS - sh.base_ms) as u32;
        let mut cur_day = 0u32;
        let mut next_day_start = day_origin + DAY_MS_U32;
        let mut day = u32::MAX;
        let (mut day_sessions, mut day_events, mut day_revenue, mut day_first) = (0u32, 0u32, 0u64, 0u32);
        let mut session_index = 0u32;
        while i < n {
            let start = *events.get_unchecked(i) as u32 >> 1;
            let start_day = if start < next_day_start {
                cur_day
            } else {
                let next2 = next_day_start + DAY_MS_U32;
                if start < next2 {
                    cur_day += 1;
                    next_day_start = next2;
                    cur_day
                } else {
                    let d = (start - day_origin) / DAY_MS_U32;
                    cur_day = d;
                    next_day_start = day_origin + (d + 1) * DAY_MS_U32;
                    d
                }
            };
            if start_day != day {
                if day != u32::MAX {
                    self.flush_day::<HAS_PURCH>(sh, person, day, day_sessions, day_events, day_revenue);
                    self.day_rows.push(DayRow { day, first: day_first, end: session_index });
                }
                day = start_day;
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
                day_first = session_index;
            }
            while next_click < nck && clicks.get_unchecked(next_click).ts() <= start {
                next_click += 1;
            }
            let campaign = if next_click > 0 && start - clicks.get_unchecked(next_click - 1).ts() <= WEEK {
                if next_click != cached_nc {
                    cached_nc = next_click;
                    cached_camp = credit(sh, &clicks[..next_click]);
                }
                cached_camp
            } else {
                organic
            };
            let cell = self.cells.as_ptr().add(campaign * days + start_day as usize);
            _mm_prefetch(cell as *const i8, _MM_HINT_T0);
            let (mut prev, mut count, mut revenue) = (start, 0u32, 0u64);
            while i < n {
                let e = *events.get_unchecked(i);
                let t = e as u32 >> 1;
                if t - prev > SESSION_GAP {
                    break;
                }
                prev = t;
                count += 1;
                i += 1;
                if HAS_PURCH && (e & 1 != 0) {
                    let payload = purchases.get_unchecked(next_purchase).1;
                    next_purchase += 1;
                    let (cat, price) = ((payload >> 24) as u8, (payload & 0xFF_FFFF) as u64);
                    revenue += price;
                    if self.cat_sum[cat as usize] == 0 {
                        self.cats.push(cat);
                    }
                    self.cat_sum[cat as usize] += price;
                }
            }
            day_sessions += 1;
            day_events += count;
            let cell = self.cells.get_unchecked_mut(campaign * days + start_day as usize);
            cell.sessions += 1;
            if HAS_PURCH {
                day_revenue += revenue;
                cell.purchasing += (revenue > 0) as u32;
                cell.revenue += revenue;
            }
            session_index += 1;
            if nck > 0 {
                self.starts.push(start);
                self.sess_row.push(self.day_rows.len() as u32);
            }
        }
        self.flush_day::<HAS_PURCH>(sh, person, day, day_sessions, day_events, day_revenue);
        if nck == 0 {
            return;
        }
        self.day_rows.push(DayRow { day, first: day_first, end: session_index });
        self.assists(days, clicks);
    }

    /// Click k assists the sessions [lo_k, hi_k) with start in [ts, ts + 7 days].
    /// Both ends only grow with ts, so for one campaign everything below the
    /// previous click's hi is already counted.
    fn assists(&mut self, days: usize, clicks: &[Click]) {
        self.generation += 1;
        let gen = (self.generation as u64) << 32;
        let starts = &self.starts;
        let ns = starts.len();
        let (mut lo, mut hi) = (0usize, 0usize);
        for c in clicks {
            let ts = c.ts();
            while lo < ns && starts[lo] < ts {
                lo += 1;
            }
            if lo == ns {
                break;
            }
            let reach = ts as u64 + WEEK as u64;
            while hi < ns && starts[hi] as u64 <= reach {
                hi += 1;
            }
            let slot = &mut self.covered[c.campaign as usize];
            let from = if *slot >> 32 == gen >> 32 { lo.max(*slot as u32 as usize) } else { lo };
            *slot = gen | hi as u64;
            if from >= hi {
                continue;
            }
            let base = c.campaign as usize * days;
            let mut r = self.sess_row[from] as usize;
            loop {
                let row = &self.day_rows[r];
                let end = (row.end as usize).min(hi);
                self.assisted[base + row.day as usize] += (end - from.max(row.first as usize)) as u32;
                if end == hi {
                    break;
                }
                r += 1;
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
/// Sort keys hold ts << 1 in bits 1..32. The two ts digits skip its low
/// TS_SKIP bits; `fix_order` repairs the rare inversions that leaves.
const TS_SKIP: u32 = 9;

#[inline(always)]
fn digit0(key: u64) -> usize {
    (key >> (1 + TS_SKIP)) as usize & (RADIX - 1)
}

#[inline(always)]
fn digit1(key: u64) -> usize {
    (key >> (1 + TS_SKIP + DIGIT)) as usize & (RADIX - 1)
}

/// Stable LSD passes from `a` into `b`: two ts digits, then the local person
/// (key >> 32) by its counts. `h0`/`h1` are digit histograms; on return
/// `counts[p]` is person p's end offset in `b`.
unsafe fn radix3<T: Copy>(a: &mut Vec<T>, b: &mut Vec<T>, h0: &mut [u32; RADIX], h1: &mut [u32; RADIX], counts: &mut [u32], key: impl Fn(&T) -> u64) {
    let n = a.len();
    b.clear();
    b.reserve(n);
    b.set_len(n);
    for h in [&mut *h0, &mut *h1] {
        let mut sum = 0;
        for c in h.iter_mut() {
            (*c, sum) = (sum, sum + *c);
        }
    }
    let mut sum = 0;
    for c in counts.iter_mut() {
        (*c, sum) = (sum, sum + *c);
    }
    let (pa, pb) = (a.as_mut_ptr(), b.as_mut_ptr());
    for i in 0..n {
        let x = *pa.add(i);
        let slot = h0.get_unchecked_mut(digit0(key(&x)));
        *pb.add(*slot as usize) = x;
        *slot += 1;
    }
    for i in 0..n {
        let x = *pb.add(i);
        let slot = h1.get_unchecked_mut(digit1(key(&x)));
        *pa.add(*slot as usize) = x;
        *slot += 1;
    }
    for i in 0..n {
        let x = *pa.add(i);
        let slot = counts.get_unchecked_mut((key(&x) >> 32) as usize);
        *pb.add(*slot as usize) = x;
        *slot += 1;
    }
}

/// Sorts by ts a slice already sorted by its radix digits: only runs that share
/// those digits can be out of order, so each such run is sorted on its own.
fn fix_order<T: Copy>(v: &mut [T], key: impl Fn(&T) -> u64) {
    let ts = |x: &T| key(x) as u32 >> 1;
    let bucket = |x: &T| key(x) as u32 >> (1 + TS_SKIP);
    let n = v.len();
    let mut i = 1;
    while i < n {
        if ts(&v[i]) >= ts(&v[i - 1]) {
            i += 1;
            continue;
        }
        let b = bucket(&v[i]);
        let mut from = i - 1;
        while from > 0 && bucket(&v[from - 1]) == b {
            from -= 1;
        }
        let mut to = i + 1;
        while to < n && bucket(&v[to]) == b {
            to += 1;
        }
        v[from..to].sort_unstable_by_key(|x| ts(x));
        i = to;
    }
}

/// The campaign of the latest click; on a tie in ts, the smallest click_id.
#[inline(always)]
unsafe fn credit(sh: &Shared, clicks: &[Click]) -> usize {
    let last = clicks.len() - 1;
    let c_last = clicks.get_unchecked(last);
    let ts = c_last.ts();
    if last == 0 || clicks.get_unchecked(last - 1).ts() != ts {
        return c_last.campaign as usize;
    }
    credit_tie(sh, clicks, last, ts)
}

#[cold]
#[inline(never)]
unsafe fn credit_tie(sh: &Shared, clicks: &[Click], last: usize, ts: u32) -> usize {
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
}
