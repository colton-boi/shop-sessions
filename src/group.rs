use crate::ids::{tok_bytes, tok_less};
use crate::parquet::{Col, Kind, Writer, BYTES, INT32, INT64};
use crate::sys::mix;
use crate::ts::DAY_MS;
use crate::{CRec, PRec, Shared, DEAD};
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

const FLUSH_ROWS: usize = 20_000;

/// One thread's `user_daily` file, written a row group at a time.
pub struct UserOut {
    path: PathBuf,
    writer: Option<Writer>,
    user: Vec<u8>,
    day: Vec<u8>,
    sessions: Vec<u8>,
    events: Vec<u8>,
    revenue: Vec<u8>,
    category: Vec<u8>,
    present: Vec<u8>,
    rows: usize,
}

impl UserOut {
    pub fn new(path: PathBuf) -> Self {
        UserOut {
            path,
            writer: None,
            user: Vec::new(),
            day: Vec::new(),
            sessions: Vec::new(),
            events: Vec::new(),
            revenue: Vec::new(),
            category: Vec::new(),
            present: Vec::new(),
            rows: 0,
        }
    }

    #[inline(always)]
    fn add(&mut self, user: &[u8], day: i32, sessions: u32, events: u32, revenue: u64, category: Option<&[u8]>) {
        self.user.extend_from_slice(&(user.len() as u32).to_le_bytes());
        self.user.extend_from_slice(user);
        self.day.extend_from_slice(&day.to_le_bytes());
        self.sessions.extend_from_slice(&sessions.to_le_bytes());
        self.events.extend_from_slice(&events.to_le_bytes());
        self.revenue.extend_from_slice(&revenue.to_le_bytes());
        if self.rows % 8 == 0 {
            self.present.push(0);
        }
        if let Some(c) = category {
            self.category.extend_from_slice(&(c.len() as u32).to_le_bytes());
            self.category.extend_from_slice(c);
            *self.present.last_mut().unwrap() |= 1 << (self.rows % 8);
        }
        self.rows += 1;
        if self.rows == FLUSH_ROWS {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.rows == 0 {
            return;
        }
        let w = self.writer.get_or_insert_with(|| Writer::create(&self.path, &USER_COLS));
        w.row_group(
            self.rows,
            &[
                (&self.user, None),
                (&self.day, None),
                (&self.sessions, None),
                (&self.events, None),
                (&self.revenue, None),
                (&self.category, Some(&self.present)),
            ],
        );
        for b in [&mut self.user, &mut self.day, &mut self.sessions, &mut self.events, &mut self.revenue, &mut self.category, &mut self.present] {
            b.clear();
        }
        self.rows = 0;
    }

    /// An empty table still needs one file to carry the schema.
    pub fn finish(mut self, even_if_empty: bool) {
        self.flush();
        if even_if_empty && self.writer.is_none() {
            self.writer = Some(Writer::create(&self.path, &USER_COLS));
        }
        if let Some(w) = self.writer {
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

#[derive(Clone, Copy)]
struct Click {
    id: u64,
    ts: u32,
    campaign: u32,
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
    ev_person: Vec<u32>,
    ev_value: Vec<u64>,
    ev_sorted: Vec<u64>,
    ck_person: Vec<u32>,
    ck_value: Vec<Click>,
    ck_sorted: Vec<Click>,
    starts: Vec<u32>,
    day_rows: Vec<DayRow>,
    by_campaign: Vec<u64>,
    cat_sum: [u64; 256],
    cats: Vec<u8>,
    name: Vec<u8>,
    pub out: UserOut,
    pub cells: Vec<Cell>,
}

impl Grouper {
    pub fn new(path: PathBuf, cells: usize) -> Self {
        Grouper {
            keys: vec![0; 1024],
            ids: vec![0; 1024],
            people: Vec::new(),
            ev_count: Vec::new(),
            ck_count: Vec::new(),
            ev_person: Vec::new(),
            ev_value: Vec::new(),
            ev_sorted: Vec::new(),
            ck_person: Vec::new(),
            ck_value: Vec::new(),
            ck_sorted: Vec::new(),
            starts: Vec::new(),
            day_rows: Vec::new(),
            by_campaign: Vec::new(),
            cat_sum: [0; 256],
            cats: Vec::new(),
            name: Vec::new(),
            out: UserOut::new(path),
            cells: vec![Cell::default(); cells],
        }
    }

    fn grow(&mut self) {
        let cap = self.keys.len() * 2;
        self.keys = vec![0; cap];
        self.ids = vec![0; cap];
        for (id, &person) in self.people.iter().enumerate() {
            let mut at = mix(person) as usize & (cap - 1);
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
        let mut at = mix(person) as usize & mask;
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
        if self.people.len() * 2 > self.keys.len() {
            self.grow();
        }
        id
    }

    /// Group one partition's live events and clicks by person, then emit each person.
    pub unsafe fn partition(&mut self, sh: &Shared, events: &[(u32, u32)], clicks: &[(u32, u32)]) {
        self.keys.fill(0);
        self.people.clear();
        self.ev_count.clear();
        self.ev_person.clear();
        self.ev_value.clear();
        for &(first, count) in events {
            let recs: &[PRec] = std::slice::from_raw_parts(sh.p_arena.base.add(first as usize), count as usize);
            for r in recs {
                if r.person == DEAD {
                    continue;
                }
                let id = self.person_id(r.person, true);
                *self.ev_count.get_unchecked_mut(id as usize) += 1;
                self.ev_person.push(id);
                self.ev_value.push((r.ts as u64) << 32 | r.payload as u64);
            }
        }
        let people = self.people.len();
        if people == 0 {
            return;
        }
        self.ck_count.clear();
        self.ck_count.resize(people, 0);
        self.ck_person.clear();
        self.ck_value.clear();
        for &(first, count) in clicks {
            let recs: &[CRec] = std::slice::from_raw_parts(sh.c_arena.base.add(first as usize), count as usize);
            for r in recs {
                let id = self.person_id(r.person, false);
                if id == u32::MAX {
                    continue;
                }
                *self.ck_count.get_unchecked_mut(id as usize) += 1;
                self.ck_person.push(id);
                self.ck_value.push(Click { id: r.click, ts: r.ts, campaign: r.campaign });
            }
        }
        // Counts become end offsets, and the scatter below walks them down to start offsets.
        let (mut ev_total, mut ck_total) = (0u32, 0u32);
        for id in 0..people {
            ev_total += self.ev_count[id];
            self.ev_count[id] = ev_total;
            ck_total += self.ck_count[id];
            self.ck_count[id] = ck_total;
        }
        self.ev_sorted.resize(ev_total as usize, 0);
        for i in 0..ev_total as usize {
            let slot = self.ev_count.get_unchecked_mut(*self.ev_person.get_unchecked(i) as usize);
            *slot -= 1;
            *self.ev_sorted.get_unchecked_mut(*slot as usize) = *self.ev_value.get_unchecked(i);
        }
        self.ck_sorted.resize(ck_total as usize, Click { id: 0, ts: 0, campaign: 0 });
        for i in 0..ck_total as usize {
            let slot = self.ck_count.get_unchecked_mut(*self.ck_person.get_unchecked(i) as usize);
            *slot -= 1;
            *self.ck_sorted.get_unchecked_mut(*slot as usize) = *self.ck_value.get_unchecked(i);
        }
        let mut events = std::mem::take(&mut self.ev_sorted);
        let mut clicks = std::mem::take(&mut self.ck_sorted);
        for id in 0..people {
            let ev_end = if id + 1 < people { self.ev_count[id + 1] } else { ev_total } as usize;
            let ck_end = if id + 1 < people { self.ck_count[id + 1] } else { ck_total } as usize;
            let (ev_from, ck_from) = (self.ev_count[id] as usize, self.ck_count[id] as usize);
            self.person(sh, self.people[id], &mut events[ev_from..ev_end], &mut clicks[ck_from..ck_end]);
        }
        self.ev_sorted = events;
        self.ck_sorted = clicks;
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
        let category = best.map(|b| sh.categories[b as usize].as_slice());
        self.out.add(&self.name, (sh.first_day + day as i64) as i32, sessions, events, revenue, category);
        for &c in &self.cats {
            self.cat_sum[c as usize] = 0;
        }
        self.cats.clear();
    }

    unsafe fn person(&mut self, sh: &Shared, person: u64, events: &mut [u64], clicks: &mut [Click]) {
        events.sort_unstable();
        tok_bytes_fast(person, &mut self.name);
        let nck = clicks.len();
        if nck > 0 {
            self.by_campaign.clear();
            self.by_campaign.extend(clicks.iter().map(|c| (c.campaign as u64) << 32 | c.ts as u64));
            clicks.sort_unstable_by_key(|c| c.ts);
            // The last click of each equal-ts run takes the campaign of the run's
            // smallest click_id, so crediting only ever looks at one click.
            let mut i = 0;
            while i < nck {
                let (mut j, mut winner) = (i + 1, i);
                while j < nck && clicks[j].ts == clicks[i].ts {
                    if tok_less(clicks[j].id, clicks[winner].id) {
                        winner = j;
                    }
                    j += 1;
                }
                clicks[j - 1].campaign = clicks[winner].campaign;
                i = j;
            }
        }
        self.starts.clear();
        self.day_rows.clear();
        let days = sh.days;
        let organic = sh.campaigns.len();
        let n = events.len();
        let (mut i, mut next_click) = (0usize, 0usize);
        let mut day = u32::MAX;
        let (mut day_sessions, mut day_events, mut day_revenue, mut day_first) = (0u32, 0u32, 0u64, 0u32);
        let mut session_index = 0u32;
        while i < n {
            let start = (*events.get_unchecked(i) >> 32) as u32;
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
            while next_click < nck && clicks.get_unchecked(next_click).ts <= start {
                next_click += 1;
            }
            let campaign = if next_click > 0 && start - clicks.get_unchecked(next_click - 1).ts <= WEEK {
                clicks.get_unchecked(next_click - 1).campaign as usize
            } else {
                organic
            };
            let (mut prev, mut count, mut revenue) = (start, 0u32, 0u64);
            while i < n {
                let e = *events.get_unchecked(i);
                let t = (e >> 32) as u32;
                if t - prev > SESSION_GAP {
                    break;
                }
                prev = t;
                count += 1;
                i += 1;
                let payload = e as u32;
                if payload != 0 {
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
            }
        }
        self.flush_day(sh, day, day_sessions, day_events, day_revenue);
        if nck == 0 {
            return;
        }
        self.day_rows.push(DayRow { day, first: day_first, end: session_index });
        self.assists(days);
    }

    /// For each campaign, the union of [click, click + 7 days] covers exactly the
    /// session starts it assists; count the sessions under each stretch per day.
    fn assists(&mut self, days: usize) {
        self.by_campaign.sort_unstable();
        let keys = &self.by_campaign;
        let n = keys.len();
        let mut i = 0;
        while i < n {
            let campaign = (keys[i] >> 32) as usize;
            let from = keys[i] as u32;
            let mut to = from as u64 + WEEK as u64;
            i += 1;
            while i < n && (keys[i] >> 32) as usize == campaign && keys[i] as u32 as u64 <= to {
                to = keys[i] as u32 as u64 + WEEK as u64;
                i += 1;
            }
            let lo = self.starts.partition_point(|&s| s < from) as u32;
            let hi = self.starts.partition_point(|&s| s as u64 <= to) as u32;
            if lo >= hi {
                continue;
            }
            for row in &self.day_rows {
                let (a, b) = (lo.max(row.first), hi.min(row.end));
                if a < b {
                    self.cells[campaign * days + row.day as usize].assisted += b - a;
                }
            }
        }
    }
}

#[inline(always)]
fn tok_bytes_fast(t: u64, out: &mut Vec<u8>) {
    tok_bytes(t, out)
}
