use crate::fast_parquet::{FastUserDailyWriter, UserDailyCols, NULL_CAT};
use crate::ids::{tok_bytes, tok_less};
use crate::parquet::{Col, Kind, Writer, BYTES, INT32, INT64};
use crate::ts::DAY_MS;
use crate::sys::mix;
use crate::{CRec, DRec, PRec, Shared, DEAD, NONE};
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::path::PathBuf;

const SESSION_GAP: u32 = 1_800_000;
const WEEK: u32 = 604_800_000;

pub static USER_COLS: [Col; 6] = [
    Col { name: "user_id", ty: BYTES, kind: Kind::Utf8 },
    Col { name: "day", ty: INT32, kind: Kind::Date },
    Col { name: "sessions", ty: INT32, kind: Kind::Plain },
    Col { name: "events", ty: INT32, kind: Kind::Plain },
    Col { name: "revenue_cents", ty: INT64, kind: Kind::Plain },
    Col { name: "top_category", ty: BYTES, kind: Kind::Utf8 },
];

const FLUSH_ROWS: usize = 4_096;

/// One thread's `user_daily` file, buffered in columnar vectors and written with
/// `RLE_DICTIONARY` + zero-revenue split row groups.
pub struct UserOut {
    path: PathBuf,
    names_data: Vec<u8>,
    names_offsets: Vec<u32>,
    person_keys: Vec<u32>,
    day: Vec<i32>,
    sessions: Vec<i32>,
    events: Vec<i32>,
    revenue_cents: Vec<i32>,
    revenue_i64_overflow: bool,
    revenue_i64_raw: Vec<u64>,
    top_category: Vec<u8>,
    cat_dict_data: Vec<u8>,
    cat_dict_offsets: Vec<u32>,
    cur_person_started: bool,
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
    pub fn new(path: PathBuf) -> Self {
        Self::with_categories(path, &[])
    }

    pub fn with_categories(path: PathBuf, categories: &[Vec<u8>]) -> Self {
        let (cat_dict_data, cat_dict_offsets) = build_cat_dict(categories);
        UserOut {
            path,
            names_data: Vec::with_capacity(64 << 10),
            names_offsets: vec![0],
            person_keys: Vec::with_capacity(FLUSH_ROWS),
            day: Vec::with_capacity(FLUSH_ROWS),
            sessions: Vec::with_capacity(FLUSH_ROWS),
            events: Vec::with_capacity(FLUSH_ROWS),
            revenue_cents: Vec::with_capacity(FLUSH_ROWS),
            revenue_i64_overflow: false,
            revenue_i64_raw: Vec::new(),
            top_category: Vec::with_capacity(FLUSH_ROWS),
            cat_dict_data,
            cat_dict_offsets,
            cur_person_started: false,
        }
    }

    #[inline(always)]
    pub fn begin_person(&mut self) {
        self.cur_person_started = false;
    }

    #[inline(always)]
    fn add(&mut self, user: &[u8], day: i32, sessions: u32, events: u32, revenue: u64, category_idx: Option<u8>) {
        if !self.cur_person_started {
            self.names_data.extend_from_slice(user);
            self.names_offsets.push(self.names_data.len() as u32);
            self.cur_person_started = true;
        }
        let pk = (self.names_offsets.len() - 2) as u32;
        self.person_keys.push(pk);
        self.day.push(day);
        self.sessions.push(sessions as i32);
        self.events.push(events as i32);
        if revenue > i32::MAX as u64 {
            if !self.revenue_i64_overflow {
                self.revenue_i64_overflow = true;
                self.revenue_i64_raw.extend(self.revenue_cents.iter().map(|&r| r as u64));
            }
        }
        if self.revenue_i64_overflow {
            self.revenue_i64_raw.push(revenue);
        }
        self.revenue_cents.push(revenue as i32);
        self.top_category.push(category_idx.unwrap_or(NULL_CAT));
    }

    /// An empty table still needs one file to carry the schema.
    pub fn finish(self, even_if_empty: bool) {
        let rows = self.day.len();
        if rows == 0 && !even_if_empty {
            return;
        }
        if !self.revenue_i64_overflow {
            let mut w = FastUserDailyWriter::create(&self.path);
            if rows > 0 {
                let cols = UserDailyCols {
                    names_data: &self.names_data,
                    names_offsets: &self.names_offsets,
                    person_keys: &self.person_keys,
                    day: &self.day,
                    sessions: &self.sessions,
                    events: &self.events,
                    revenue_cents: &self.revenue_cents,
                    top_category: &self.top_category,
                    cat_dict_data: &self.cat_dict_data,
                    cat_dict_offsets: &self.cat_dict_offsets,
                };
                w.write_batches_split(&cols, FLUSH_ROWS);
            }
            w.finish();
        } else {
            let mut w = Writer::create(&self.path, &USER_COLS);
            let mut u_buf = Vec::with_capacity(rows * 12);
            let mut d_buf = Vec::with_capacity(rows * 4);
            let mut s_buf = Vec::with_capacity(rows * 4);
            let mut e_buf = Vec::with_capacity(rows * 4);
            let mut r_buf = Vec::with_capacity(rows * 8);
            let mut c_buf = Vec::new();
            let mut pres = vec![0u8; rows.div_ceil(8)];
            for i in 0..rows {
                let pk = self.person_keys[i] as usize;
                let uname = &self.names_data[self.names_offsets[pk] as usize..self.names_offsets[pk + 1] as usize];
                u_buf.extend_from_slice(&(uname.len() as u32).to_le_bytes());
                u_buf.extend_from_slice(uname);
                d_buf.extend_from_slice(&self.day[i].to_le_bytes());
                s_buf.extend_from_slice(&self.sessions[i].to_le_bytes());
                e_buf.extend_from_slice(&self.events[i].to_le_bytes());
                let rev = self.revenue_i64_raw.get(i).copied().unwrap_or(self.revenue_cents[i] as u64);
                r_buf.extend_from_slice(&rev.to_le_bytes());
                let cat = self.top_category[i];
                if cat != NULL_CAT {
                    let cs = &self.cat_dict_data[self.cat_dict_offsets[cat as usize] as usize..self.cat_dict_offsets[cat as usize + 1] as usize];
                    c_buf.extend_from_slice(&(cs.len() as u32).to_le_bytes());
                    c_buf.extend_from_slice(cs);
                    pres[i / 8] |= 1 << (i % 8);
                }
            }
            w.row_group(
                rows,
                &[
                    (&u_buf, None),
                    (&d_buf, None),
                    (&s_buf, None),
                    (&e_buf, None),
                    (&r_buf, None),
                    (&c_buf, Some(&pres)),
                ],
            );
            w.finish();
        }
    }

    /// Writes both `extra` (from the fast-path sessionizer) and `self` (from complex users)
    /// into a single `user_daily/part-*.parquet` file.
    pub fn finish_with_extra(self, extra: &UserDailyCols<'_>, even_if_empty: bool) {
        let rows = self.day.len();
        if rows == 0 && extra.is_empty() && !even_if_empty {
            return;
        }
        if !self.revenue_i64_overflow {
            let mut w = FastUserDailyWriter::create(&self.path);
            if !extra.is_empty() {
                w.write_batches_split(extra, FLUSH_ROWS);
            }
            if rows > 0 {
                let cols = UserDailyCols {
                    names_data: &self.names_data,
                    names_offsets: &self.names_offsets,
                    person_keys: &self.person_keys,
                    day: &self.day,
                    sessions: &self.sessions,
                    events: &self.events,
                    revenue_cents: &self.revenue_cents,
                    top_category: &self.top_category,
                    cat_dict_data: &self.cat_dict_data,
                    cat_dict_offsets: &self.cat_dict_offsets,
                };
                w.write_batches_split(&cols, FLUSH_ROWS);
            }
            w.finish();
        } else {
            let mut w = Writer::create(&self.path, &USER_COLS);
            let write_cols_plain = |w: &mut Writer,
                                    n: usize,
                                    names_data: &[u8],
                                    names_offsets: &[u32],
                                    person_keys: &[u32],
                                    day: &[i32],
                                    sessions: &[i32],
                                    events: &[i32],
                                    rev_i32: &[i32],
                                    rev_i64: Option<&[u64]>,
                                    top_category: &[u8],
                                    cat_data: &[u8],
                                    cat_off: &[u32]| {
                if n == 0 {
                    return;
                }
                let mut u_buf = Vec::with_capacity(n * 12);
                let mut d_buf = Vec::with_capacity(n * 4);
                let mut s_buf = Vec::with_capacity(n * 4);
                let mut e_buf = Vec::with_capacity(n * 4);
                let mut r_buf = Vec::with_capacity(n * 8);
                let mut c_buf = Vec::new();
                let mut pres = vec![0u8; n.div_ceil(8)];
                for i in 0..n {
                    let pk = person_keys[i] as usize;
                    let uname = &names_data[names_offsets[pk] as usize..names_offsets[pk + 1] as usize];
                    u_buf.extend_from_slice(&(uname.len() as u32).to_le_bytes());
                    u_buf.extend_from_slice(uname);
                    d_buf.extend_from_slice(&day[i].to_le_bytes());
                    s_buf.extend_from_slice(&sessions[i].to_le_bytes());
                    e_buf.extend_from_slice(&events[i].to_le_bytes());
                    let rev = rev_i64
                        .and_then(|v| v.get(i).copied())
                        .unwrap_or(rev_i32[i] as u64);
                    r_buf.extend_from_slice(&rev.to_le_bytes());
                    let cat = top_category[i];
                    if cat != NULL_CAT {
                        let cs = &cat_data[cat_off[cat as usize] as usize..cat_off[cat as usize + 1] as usize];
                        c_buf.extend_from_slice(&(cs.len() as u32).to_le_bytes());
                        c_buf.extend_from_slice(cs);
                        pres[i / 8] |= 1 << (i % 8);
                    }
                }
                w.row_group(
                    n,
                    &[
                        (&u_buf, None),
                        (&d_buf, None),
                        (&s_buf, None),
                        (&e_buf, None),
                        (&r_buf, None),
                        (&c_buf, Some(&pres)),
                    ],
                );
            };
            write_cols_plain(
                &mut w,
                extra.len(),
                extra.names_data,
                extra.names_offsets,
                extra.person_keys,
                extra.day,
                extra.sessions,
                extra.events,
                extra.revenue_cents,
                None,
                extra.top_category,
                extra.cat_dict_data,
                extra.cat_dict_offsets,
            );
            write_cols_plain(
                &mut w,
                rows,
                &self.names_data,
                &self.names_offsets,
                &self.person_keys,
                &self.day,
                &self.sessions,
                &self.events,
                &self.revenue_cents,
                Some(&self.revenue_i64_raw),
                &self.top_category,
                &self.cat_dict_data,
                &self.cat_dict_offsets,
            );
            w.finish();
        }
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
    name: Vec<u8>,
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
            name: Vec::new(),
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

    pub fn take_cells(&mut self) -> Vec<Cell> {
        for (c, &a) in self.cells.iter_mut().zip(&self.assisted) {
            c.assisted = a;
        }
        std::mem::take(&mut self.cells)
    }

    fn flush_day(&mut self, sh: &Shared, day: u32, sessions: u32, events: u32, revenue: u64) {
        let mut best: Option<u8> = None;
        for &c in &self.cats {
            let sum = self.cat_sum[c as usize];
            // Category indexes are in byte-wise name order, so the smaller index wins a tie.
            best = match best {
                Some(b) if (self.cat_sum[b as usize], std::cmp::Reverse(b)) >= (sum, std::cmp::Reverse(c)) => Some(b),
                _ => Some(c),
            };
        }
        self.out.add(&self.name, (sh.first_day + day as i64) as i32, sessions, events, revenue, best);
        for &c in &self.cats {
            self.cat_sum[c as usize] = 0;
        }
        self.cats.clear();
    }

    unsafe fn person(&mut self, sh: &Shared, person: u64, events: &[u64], purchases: &[(u64, u32)], clicks: &[Click]) {
        tok_bytes_fast(person, &mut self.name);
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
        let mut day = u32::MAX;
        let (mut day_sessions, mut day_events, mut day_revenue, mut day_first) = (0u32, 0u32, 0u64, 0u32);
        let mut session_index = 0u32;
        while i < n {
            let start = *events.get_unchecked(i) as u32 >> 1;
            let start_day = ((sh.base_ms + start as i64).div_euclid(DAY_MS) - sh.first_day) as u32;
            if start_day != day {
                if day != u32::MAX {
                    self.flush_day(sh, day, day_sessions, day_events, day_revenue);
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
                credit(sh, &clicks[..next_click])
            } else {
                organic
            };
            let cell = self.cells.as_ptr().add(campaign * days + start_day as usize);
            std::arch::x86_64::_mm_prefetch(cell as *const i8, std::arch::x86_64::_MM_HINT_T0);
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
                if e & 1 != 0 {
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
            day_revenue += revenue;
            let cell = self.cells.get_unchecked_mut(campaign * days + start_day as usize);
            cell.sessions += 1;
            cell.purchasing += (revenue > 0) as u32;
            cell.revenue += revenue;
            session_index += 1;
            if nck > 0 {
                self.starts.push(start);
                self.sess_row.push(self.day_rows.len() as u32);
            }
        }
        self.flush_day(sh, day, day_sessions, day_events, day_revenue);
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
    let ts = clicks[last].ts();
    let mut best = last;
    let mut k = last;
    let id = |c: &Click| (*sh.c_arena.base.add(c.at as usize)).click;
    while k > 0 && clicks[k - 1].ts() == ts {
        k -= 1;
        if tok_less(id(&clicks[k]), id(&clicks[best])) {
            best = k;
        }
    }
    clicks[best].campaign as usize
}

#[inline(always)]
fn tok_bytes_fast(t: u64, out: &mut Vec<u8>) {
    tok_bytes(t, out)
}

/// One thread's working memory for event_id dedup.
#[derive(Default)]
pub struct Dedup {
    once: Vec<u64>,
    twice: Vec<u64>,
    buckets: Vec<u32>,
    candidates: Vec<u32>,
    table: Vec<u64>,
}

impl Dedup {
    /// Keeps, per event_id, the copy with the smallest ingest_ts and kills the
    /// payload record of every other copy. Two bitmaps over a hash of the ID
    /// mark the buckets hit more than once; only records in those buckets,
    /// which include every copy of every duplicate, go through the exact table.
    pub unsafe fn partition(&mut self, sh: &Shared, chunks: &[(u32, u32)]) {
        let total: usize = chunks.iter().map(|c| c.1 as usize).sum();
        let bits = (total * 8).next_power_of_two().max(64);
        let mask = bits - 1;
        self.once.clear();
        self.once.resize(bits / 64, 0);
        self.twice.clear();
        self.twice.resize(bits / 64, 0);
        self.buckets.clear();
        self.buckets.reserve(total);
        let base = sh.d_arena.base;
        let (once, twice, buckets) = (self.once.as_mut_ptr(), self.twice.as_mut_ptr(), self.buckets.as_mut_ptr());
        let mut k = 0;
        for &(first, count) in chunks {
            for at in first..first + count {
                let b = mix((*base.add(at as usize)).event) as usize & mask;
                *buckets.add(k) = b as u32;
                k += 1;
                let (w, bit) = (b / 64, 1u64 << (b % 64));
                *twice.add(w) |= *once.add(w) & bit;
                *once.add(w) |= bit;
            }
        }
        self.candidates.clear();
        self.candidates.reserve(total);
        let cands = self.candidates.as_mut_ptr();
        let (mut k, mut nc) = (0usize, 0usize);
        for &(first, count) in chunks {
            for at in first..first + count {
                let b = *buckets.add(k) as usize;
                k += 1;
                // ~14% of records are candidates at random: a branch here mispredicts.
                *cands.add(nc) = at;
                nc += (*twice.add(b / 64) >> (b % 64)) as usize & 1;
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
            let h = mix(r.event).rotate_left(32);
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
                    if other.event == r.event {
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
