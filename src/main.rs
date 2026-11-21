mod csv;
mod diag;
mod dims;
mod group;
mod ids;
mod links;
mod parquet;
mod scatter;
mod sys;
mod ts;

use csv::Reader;
use dims::KeyMap;
use group::{Cell, Grouper};
use ids::tok_pair;
use links::LinkTable;
use parquet::{Col, Kind, Writer, BYTES, INT32, INT64};
use scatter::{by_partition, part_of_key, Arena, Scatter};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
use sys::{mix, par, PerThread};
use ts::{DateCache, DateLut, BAD, DAY_MS};

/// Dedup stream: one per parsed event line, partitioned by event_id.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DRec {
    event: u64,
    ingest: i32,
    /// Index of this line's PRec, or NONE when the line is invalid.
    payload: u32,
}

/// Payload stream: one per valid event line, partitioned by person.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PRec {
    person: u64,
    ts: u32,
    /// 0 for a view or cart, else category << 24 | price_cents.
    payload: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CRec {
    person: u64,
    click: u64,
    ts: u32,
    campaign: u32,
}

const NONE: u32 = u32::MAX;
pub const DEAD: u64 = u64::MAX;
const WEEK_MS: i64 = 7 * DAY_MS;

const READ_BUF: usize = 64 << 10;
const EXTENT: u64 = 2 << 20;
const D_CHUNK: usize = 512;
const P_CHUNK: usize = 512;
const C_CHUNK: usize = 256;
const BATCH: usize = 32;

pub struct Shared {
    window_start: i64,
    window_end: i64,
    /// Every stored `ts` is milliseconds after this: window_start - 7 days.
    pub base_ms: i64,
    use_rel: bool,
    date_lut: DateLut,
    pub first_day: i64,
    pub days: usize,
    products: KeyMap,
    product_category: Vec<u8>,
    direct_prod_cat: Vec<u8>,
    unknown_category: u8,
    pub categories: Vec<Vec<u8>>,
    campaign_ids: KeyMap,
    direct_camp: Vec<u16>,
    pub campaigns: Vec<(Vec<u8>, Vec<u8>)>,
    links: LinkTable,
    d_arena: Arena<DRec>,
    pub p_arena: Arena<PRec>,
    pub c_arena: Arena<CRec>,
    d_parts: usize,
    p_parts: usize,
}

impl Shared {
    #[inline(always)]
    unsafe fn product_cat(&self, p: *const u8, len: usize) -> u8 {
        if !self.direct_prod_cat.is_empty() {
            dims::lookup_prod_direct(&self.direct_prod_cat, self.unknown_category, p, len)
        } else {
            match self.products.get(p, len) {
                Some(i) => *self.product_category.get_unchecked(i as usize),
                None => self.unknown_category,
            }
        }
    }

    #[inline(always)]
    unsafe fn campaign_id(&self, p: *const u8, len: usize) -> Option<u32> {
        if !self.direct_camp.is_empty() {
            dims::lookup_camp_direct(&self.direct_camp, p, len)
        } else {
            self.campaign_ids.get(p, len)
        }
    }
}

#[derive(Clone, Copy)]
struct PendingEvent {
    event: u64,
    user: u64,
    /// `mix(user)`, computed once for the prefetch and the lookup.
    user_hash: u64,
    ingest: i32,
    ts: u32,
    payload: u32,
    valid: bool,
}

#[derive(Clone, Copy)]
struct PendingClick {
    click: u64,
    user: u64,
    user_hash: u64,
    ts: u32,
    campaign: u32,
}

struct Worker {
    reader: Reader,
    dates: DateCache,
    pairs: Vec<(u64, u64)>,
    d: Scatter<DRec, 4>,
    p: Scatter<PRec, 4>,
    c: Scatter<CRec, 8>,
    /// Per stage: on-CPU, runqueue-wait and idle seconds.
    busy: [[f64; 3]; 8],
}

struct Extent {
    file: usize,
    from: u64,
    to: u64,
}

/// A mapped input file: see `sys::map_file`.
#[derive(Clone, Copy)]
struct Mapped(*const u8, usize);
unsafe impl Sync for Mapped {}
unsafe impl Send for Mapped {}

struct Input {
    files: Vec<Mapped>,
    /// Extents of each file not yet parsed; the last one to finish unmaps the file.
    remaining: Vec<AtomicUsize>,
    extents: Vec<Extent>,
    bytes: u64,
}

impl Input {
    fn done(&self, x: &Extent) {
        if self.remaining[x.file].fetch_sub(1, Relaxed) == 1 {
            sys::unmap_file(self.files[x.file].0, self.files[x.file].1);
        }
    }

    /// Makes file `i` end with a newline, as `csv::Reader` needs.
    fn end_line(&mut self, i: usize) {
        let Mapped(data, size) = self.files[i];
        self.files[i].1 = unsafe { sys::end_line(data, size) };
    }

    fn lacks_newline(&self, i: usize) -> bool {
        let Mapped(data, size) = self.files[i];
        size != 0 && unsafe { *data.add(size - 1) } != b'\n'
    }
}

fn list_csv(dir: &Path, extent: u64) -> Input {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "csv") && p.is_file())
            .collect(),
        Err(_) => Vec::new(),
    };
    paths.sort();
    let mut input = Input { files: Vec::new(), remaining: Vec::new(), extents: Vec::new(), bytes: 0 };
    for p in paths {
        let file = File::open(&p).expect("open input");
        let size = file.metadata().unwrap().len();
        let index = input.files.len();
        let mut from = 0;
        let first = input.extents.len();
        while from < size {
            let to = (from + extent).min(size);
            input.extents.push(Extent { file: index, from, to });
            from = to;
        }
        input.remaining.push(AtomicUsize::new(input.extents.len() - first));
        input.bytes += size;
        input.files.push(Mapped(sys::map_file(&file, size as usize), size as usize));
    }
    input
}

fn manifest_time(text: &str, key: &str) -> i64 {
    let at = text.find(key).unwrap_or_else(|| panic!("manifest lacks {key}"));
    let rest = &text[at + key.len()..];
    let open = rest.find(':').and_then(|c| rest[c..].find('"').map(|q| c + q + 1)).expect("manifest value");
    let close = open + rest[open..].find('"').expect("manifest value");
    let ms = ts::parse_ts_slice(rest[open..close].as_bytes());
    assert!(ms != BAD, "bad {key} in manifest");
    ms
}

unsafe fn parse_links(sh_files: &Input, next: &AtomicUsize, w: &mut Worker) {
    let Worker { reader, pairs, .. } = w;
    loop {
        let i = next.fetch_add(1, Relaxed);
        let Some(x) = sh_files.extents.get(i) else { break };
        let Mapped(data, size) = sh_files.files[x.file];
        reader.scan::<2>(data, size, x.from as usize, x.to as usize, |buf, s, e| {
            let (la, lb) = ((e[0] - s[0]) as usize, (e[1] - s[1]) as usize);
            if la != 0 && lb != 0 {
                let (a, b) = tok_pair(buf.add(s[0] as usize), la, buf.add(s[1] as usize), lb);
                if a != b {
                    pairs.push((a, b));
                }
            }
        });
        sh_files.done(x);
    }
}

#[inline(never)]
unsafe fn flush_events(sh: &Shared, d: &mut Scatter<DRec, 4>, p: &mut Scatter<PRec, 4>, batch: &[PendingEvent]) {
    for e in batch {
        let payload = if e.valid {
            let person = sh.links.person_hashed(e.user, e.user_hash);
            p.push(&sh.p_arena, part_of_key(person, sh.p_parts), PRec { person, ts: e.ts, payload: e.payload })
        } else {
            NONE
        };
        d.push(&sh.d_arena, part_of_key(e.event, sh.d_parts), DRec { event: e.event, ingest: e.ingest, payload });
    }
}

unsafe fn parse_events(sh: &Shared, file: Mapped, x: &Extent, w: &mut Worker) {
    if sh.use_rel {
        parse_events_impl::<true>(sh, file, x, w);
    } else {
        parse_events_impl::<false>(sh, file, x, w);
    }
}

#[inline(never)]
unsafe fn parse_events_impl<const REL: bool>(sh: &Shared, Mapped(data, size): Mapped, x: &Extent, w: &mut Worker) {
    const VIEW: u32 = u32::from_le_bytes(*b"view");
    const CART: u32 = u32::from_le_bytes(*b"cart");
    const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");
    let Worker { reader, dates, d, p, .. } = w;
    let mut batch = [PendingEvent { event: 0, user: 0, user_hash: 0, ingest: 0, ts: 0, payload: 0, valid: false }; BATCH];
    let mut pending = 0;
    let base_s = sh.base_ms.div_euclid(1000);
    let event_win_ms = (sh.window_end - sh.window_start) as u64;
    reader.scan::<7>(data, size, x.from as usize, x.to as usize, |buf, s, e| {
        let at = |k: usize| buf.add(s[k] as usize);
        let len = |k: usize| (e[k] - s[k]) as usize;
        if len(0) == 0 {
            return;
        }
        let (event, user) = tok_pair(at(0), len(0), at(1), len(1));
        let mut out = PendingEvent { event, user: 0, user_hash: 0, ingest: i32::MAX, ts: 0, payload: 0, valid: false };
        'valid: {
            let ts_u32 = if REL {
                let ingest_s_rel = ts::parse_f1_rel(at(3), len(3), &sh.date_lut, dates, base_s);
                out.ingest = ingest_s_rel;
                if len(1) == 0 || len(5) == 0 || ingest_s_rel == i32::MAX {
                    break 'valid;
                }
                let ts_rel = ts::parse_ts_rel(at(2), len(2), &sh.date_lut, dates, sh.base_ms);
                let win_bad = (ts_rel.wrapping_sub(WEEK_MS) as u64) >= event_win_ms;
                let late_bad = (ingest_s_rel as i64).wrapping_mul(1000).wrapping_sub(ts_rel) > DAY_MS;
                if win_bad | late_bad {
                    break 'valid;
                }
                ts_rel as u32
            } else {
                let ingest_s = ts::parse_f1(at(3), len(3), dates);
                if ingest_s != BAD {
                    // Only the order of copies matters here; the lateness test below uses the exact value.
                    out.ingest = (ingest_s - base_s).clamp(i32::MIN as i64, i32::MAX as i64 - 1) as i32;
                }
                if len(1) == 0 || len(5) == 0 || ingest_s == BAD {
                    break 'valid;
                }
                let t = ts::parse_ts(at(2), len(2), dates);
                if t < sh.window_start || t >= sh.window_end || ingest_s * 1000 - t > DAY_MS {
                    break 'valid;
                }
                (t - sh.base_ms) as u32
            };
            match len(4) {
                4 => {
                    let ty = (at(4) as *const u32).read_unaligned();
                    if ty != VIEW && ty != CART {
                        break 'valid;
                    }
                }
                8 => {
                    if (at(4) as *const u64).read_unaligned() != PURCHASE {
                        break 'valid;
                    }
                    let price = match ts::digits(at(6), len(6)) {
                        Some(v) if (1..=10_000_000).contains(&v) => v as u32,
                        _ => break 'valid,
                    };
                    let category = sh.product_cat(at(5), len(5));
                    out.payload = (category as u32) << 24 | price;
                }
                _ => break 'valid,
            }
            out.user = user;
            out.user_hash = mix(out.user);
            sh.links.prefetch_hashed(out.user_hash);
            out.ts = ts_u32;
            out.valid = true;
        }
        *batch.get_unchecked_mut(pending) = out;
        pending += 1;
        if pending == BATCH {
            flush_events(sh, d, p, &batch);
            pending = 0;
        }
    });
    flush_events(sh, d, p, &batch[..pending]);
}

#[inline(never)]
unsafe fn flush_clicks(sh: &Shared, c: &mut Scatter<CRec, 8>, batch: &[PendingClick]) {
    for k in batch {
        let person = sh.links.person_hashed(k.user, k.user_hash);
        let rec = CRec { person, click: k.click, ts: k.ts, campaign: k.campaign };
        c.push(&sh.c_arena, part_of_key(person, sh.p_parts), rec);
    }
}

unsafe fn parse_clicks(sh: &Shared, file: Mapped, x: &Extent, w: &mut Worker) {
    if sh.use_rel {
        parse_clicks_impl::<true>(sh, file, x, w);
    } else {
        parse_clicks_impl::<false>(sh, file, x, w);
    }
}

#[inline(never)]
unsafe fn parse_clicks_impl<const REL: bool>(sh: &Shared, Mapped(data, size): Mapped, x: &Extent, w: &mut Worker) {
    let Worker { reader, dates, c, .. } = w;
    let mut batch = [PendingClick { click: 0, user: 0, user_hash: 0, ts: 0, campaign: 0 }; BATCH];
    let mut pending = 0;
    let click_win_ms = (sh.window_end - sh.base_ms) as u64;
    reader.scan::<4>(data, size, x.from as usize, x.to as usize, |buf, s, e| {
        let at = |k: usize| buf.add(s[k] as usize);
        let len = |k: usize| (e[k] - s[k]) as usize;
        if len(0) == 0 || len(1) == 0 {
            return;
        }
        let ts_u32 = if REL {
            let ts_rel = ts::parse_ts_rel(at(3), len(3), &sh.date_lut, dates, sh.base_ms);
            if (ts_rel as u64) >= click_win_ms {
                return;
            }
            ts_rel as u32
        } else {
            let t = ts::parse_ts(at(3), len(3), dates);
            if t < sh.base_ms || t >= sh.window_end {
                return;
            }
            (t - sh.base_ms) as u32
        };
        let Some(campaign) = sh.campaign_id(at(2), len(2)) else { return };
        let (click, user) = tok_pair(at(0), len(0), at(1), len(1));
        let user_hash = mix(user);
        sh.links.prefetch_hashed(user_hash);
        *batch.get_unchecked_mut(pending) =
            PendingClick { click, user, user_hash, ts: ts_u32, campaign };
        pending += 1;
        if pending == BATCH {
            flush_clicks(sh, c, &batch);
            pending = 0;
        }
    });
    flush_clicks(sh, c, &batch[..pending]);
}

static CAMPAIGN_COLS: [Col; 7] = [
    Col { name: "campaign_id", ty: BYTES, kind: Kind::Utf8 },
    Col { name: "channel", ty: BYTES, kind: Kind::Utf8 },
    Col { name: "day", ty: INT32, kind: Kind::Date },
    Col { name: "sessions", ty: INT64, kind: Kind::Plain },
    Col { name: "purchasing_sessions", ty: INT64, kind: Kind::Plain },
    Col { name: "revenue_cents", ty: INT64, kind: Kind::Plain },
    Col { name: "assisted_sessions", ty: INT64, kind: Kind::Plain },
];

fn write_campaigns(sh: &Shared, path: &Path, per_thread: &[Vec<Cell>]) {
    let mut cols: [Vec<u8>; 7] = Default::default();
    let mut rows = 0;
    let string = |b: &mut Vec<u8>, s: &[u8]| {
        b.extend_from_slice(&(s.len() as u32).to_le_bytes());
        b.extend_from_slice(s);
    };
    for campaign in 0..=sh.campaigns.len() {
        let (id, channel): (&[u8], &[u8]) = match sh.campaigns.get(campaign) {
            Some((id, channel)) => (id, channel),
            None => (b"organic", b"organic"),
        };
        for day in 0..sh.days {
            let (mut sessions, mut purchasing, mut assisted, mut revenue) = (0u64, 0u64, 0u64, 0u64);
            for cells in per_thread {
                let c = &cells[campaign * sh.days + day];
                sessions += c.sessions as u64;
                purchasing += c.purchasing as u64;
                assisted += c.assisted as u64;
                revenue += c.revenue;
            }
            if sessions == 0 && assisted == 0 {
                continue;
            }
            string(&mut cols[0], id);
            string(&mut cols[1], channel);
            cols[2].extend_from_slice(&((sh.first_day + day as i64) as i32).to_le_bytes());
            cols[3].extend_from_slice(&sessions.to_le_bytes());
            cols[4].extend_from_slice(&purchasing.to_le_bytes());
            cols[5].extend_from_slice(&revenue.to_le_bytes());
            cols[6].extend_from_slice(&assisted.to_le_bytes());
            rows += 1;
        }
    }
    let mut w = Writer::create(path, &CAMPAIGN_COLS);
    if rows > 0 {
        let data: Vec<(&[u8], Option<&[u8]>)> = cols.iter().map(|c| (c.as_slice(), None)).collect();
        w.row_group(rows, &data);
    }
    w.finish();
}

type Mark<'a> = (&'a str, f64, (f64, f64, u64), [u64; 4]);

fn log_stage(marks: &[Mark]) {
    let (name, at, ru, _) = marks[marks.len() - 1];
    let (prev_at, prev_ru) = if marks.len() > 1 { (marks[marks.len() - 2].1, marks[marks.len() - 2].2) } else { (0.0, (0.0, 0.0, 0)) };
    diag::stage(name, at - prev_at, ru.0 - prev_ru.0, ru.1 - prev_ru.1);
}

fn main() {
    let t0 = Instant::now();
    sys::prof::start();
    let args: Vec<String> = std::env::args().collect();
    let dir = |var: &str, back: usize, default: &str| -> PathBuf {
        if args.len() >= 3 {
            return PathBuf::from(&args[args.len() - back]);
        }
        PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.to_string()))
    };
    let data = dir("DATA_DIR", 2, "/data");
    let output = dir("OUTPUT_DIR", 1, "/output");
    let verbose = std::env::var_os("SS_STATS").is_some();
    let threads = std::env::var("SS_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()))
        .clamp(1, 256);
    sys::init_host_profile(&data);
    // Counters must open before the pool starts so every worker thread inherits them.
    let counters = verbose.then(sys::Counters::open);
    sys::start_pool(threads);
    diag::host(threads);

    let manifest = std::fs::read_to_string(data.join("manifest.json")).expect("manifest.json");
    let window_start = manifest_time(&manifest, "\"window_start\"");
    let window_end = manifest_time(&manifest, "\"window_end\"");
    let base_ms = window_start - WEEK_MS;
    assert!(window_end > window_start, "empty window");
    assert!(window_end - base_ms < 1 << 31, "window too long for 32-bit offsets");
    let first_day = window_start.div_euclid(DAY_MS);
    let days = ((window_end - 1).div_euclid(DAY_MS) - first_day + 1) as usize;

    if verbose { eprintln!("  manifest {:.2} ms", t0.elapsed().as_secs_f64()*1e3); }
    // Products and campaigns are first needed by the event parse; load them beside the links.
    let dims_loader = {
        let data = data.clone();
        std::thread::spawn(move || dims::load(&data))
    };
    let knob = |name: &str, default: u64| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let extent = knob("SS_EXTENT", EXTENT);
    let read_buf = knob("SS_READ_BUF", READ_BUF as u64) as usize;
    let mut link_input = list_csv(&data.join("links"), extent.min(256 << 10));
    let mut event_input = list_csv(&data.join("events"), extent);
    let mut click_input = list_csv(&data.join("clicks"), extent);
    // The links are parsed first; the other files' last bytes are read in parallel during that parse.
    for i in 0..link_input.files.len() {
        link_input.end_line(i);
    }
    std::fs::create_dir_all(output.join("user_daily")).expect("create output");
    std::fs::create_dir_all(output.join("campaign_daily")).expect("create output");

    if verbose { eprintln!("  listed {:.2} ms", t0.elapsed().as_secs_f64()*1e3); }
    let workers = PerThread::new(threads, |_| Worker {
        reader: Reader::new(read_buf),
        dates: DateCache::new(),
        pairs: Vec::new(),
        d: Scatter::new(),
        p: Scatter::new(),
        c: Scatter::new(),
        busy: [[0.0; 3]; 8],
    });
    let count = || counters.as_ref().map_or([0; 4], |c| c.read());
    let mark = |name| (name, t0.elapsed().as_secs_f64(), sys::rusage(), count());
    let mut marks: Vec<Mark> = vec![mark("setup")];
    log_stage(&marks);
    let phase = |stage: usize, f: &(dyn Fn(usize) + Sync)| {
        let begun = Instant::now();
        par(threads, |t| {
            let (cpu, wait) = if verbose { sys::schedstat() } else { (0.0, 0.0) };
            f(t);
            if verbose {
                let (cpu2, wait2) = sys::schedstat();
                let b = unsafe { &mut workers.get(t).busy[stage] };
                b[0] += cpu2 - cpu;
                b[1] += wait2 - wait;
            }
        });
        if verbose {
            let wall = begun.elapsed().as_secs_f64();
            for t in 0..threads {
                let b = unsafe { &mut workers.get(t).busy[stage] };
                b[2] += wall - (b[0] + b[1]).min(wall);
            }
        }
    };

    let next = AtomicUsize::new(0);
    let tails = event_input.files.len() + click_input.files.len();
    let next_tail = AtomicUsize::new(0);
    let unended: Vec<std::sync::atomic::AtomicBool> = (0..tails).map(|_| Default::default()).collect();
    phase(0, &|t| unsafe {
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
        parse_links(&link_input, &next, workers.get(t))
    });
    // Rare, and done before the populate so no mapping change contends with it for the mmap lock.
    for (i, u) in unended.iter().enumerate() {
        if u.load(Relaxed) {
            match i.checked_sub(event_input.files.len()) {
                None => event_input.end_line(i),
                Some(k) => click_input.end_line(k),
            }
        }
    }
    marks.push(mark("links parse"));
    log_stage(&marks);

    let pair_lists: Vec<&[(u64, u64)]> = (0..threads).map(|t| unsafe { workers.get(t).pairs.as_slice() }).collect();
    let pairs: usize = pair_lists.iter().map(|p| p.len()).sum();
    let links = LinkTable::new(pairs);
    const LINK_BLOCK: usize = 2048;
    let blocks: Vec<(usize, usize)> = pair_lists
        .iter()
        .enumerate()
        .flat_map(|(t, p)| (0..p.len()).step_by(LINK_BLOCK).map(move |from| (t, from)))
        .collect();
    let next = AtomicUsize::new(0);
    phase(1, &|_| loop {
        let i = next.fetch_add(1, Relaxed);
        let Some(&(t, from)) = blocks.get(i) else { break };
        let list = pair_lists[t];
        unsafe { links.link_all(&list[from..(from + LINK_BLOCK).min(list.len())]) };
    });
    const SWEEP: usize = 1 << 14;
    let sweeps = links.slots().div_ceil(SWEEP);
    let sweep = |f: &(dyn Fn(usize, usize) + Sync)| {
        let next = AtomicUsize::new(0);
        phase(2, &|_| loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= sweeps {
                break;
            }
            f(i * SWEEP, ((i + 1) * SWEEP).min(links.slots()));
        });
    };
    sweep(&|from, to| unsafe { links.compress_and_resolve(from, to) });
    marks.push(mark("links join"));
    log_stage(&marks);

    let rows_guess = (event_input.bytes / 78) as usize;
    let parts = |per: usize| (rows_guess / per).next_power_of_two().clamp(1, 1024);
    let (d_parts, p_parts) = (parts(16_384), parts(16_384));
    let (d_parts, p_parts) = (knob("SS_DPARTS", d_parts as u64) as usize, knob("SS_PPARTS", p_parts as u64) as usize);
    let dims = dims_loader.join().expect("loading products and campaigns");
    let d_recs = event_input.bytes as usize / 8 + (threads * d_parts + 2) * D_CHUNK;
    let p_recs = event_input.bytes as usize / 32 + (threads * p_parts + 2) * P_CHUNK;
    let c_recs = click_input.bytes as usize / 8 + (threads * p_parts + 2) * C_CHUNK;
    let use_rel = base_ms % 1000 == 0;
    let date_lut = DateLut::new(window_start, window_end, base_ms.div_euclid(1000));
    let sh = Shared {
        window_start,
        window_end,
        base_ms,
        use_rel,
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
        links,
        d_arena: Arena::new(d_recs, D_CHUNK),
        p_arena: Arena::new(p_recs, P_CHUNK),
        c_arena: Arena::new(c_recs, C_CHUNK),
        d_parts,
        p_parts,
    };

    // Committing the expected arena in 4 MB blocks costs about a third of the kernel
    // time that eight threads faulting it in page by page take during the parse.
    let populated = [rows_guess * 17, rows_guess * 17, click_input.bytes as usize / 46 * 25];
    {
        const PIECE: usize = 4 << 20;
        let regions = [
            (sh.d_arena.base as usize, populated[0]),
            (sh.p_arena.base as usize, populated[1]),
            (sh.c_arena.base as usize, populated[2]),
        ];
        let pieces: Vec<(usize, usize)> = regions
            .iter()
            .flat_map(|&(b, n)| (0..n).step_by(PIECE).map(move |at| (b + at, PIECE.min(n - at))))
            .collect();
        let next = AtomicUsize::new(0);
        phase(6, &|_| loop {
            let i = next.fetch_add(1, Relaxed);
            let Some(&(at, n)) = pieces.get(i) else { break };
            sys::populate(at as *mut u8, n);
        });
        marks.push(mark("populate"));
    log_stage(&marks);
    }
    // Largest work first, so the tail of the phase is made of small pieces.
    let mut work: Vec<(bool, usize)> = (0..event_input.extents.len())
        .map(|i| (true, i))
        .chain((0..click_input.extents.len()).map(|i| (false, i)))
        .collect();
    let extent_len = |&(is_event, i): &(bool, usize)| {
        let x = if is_event { &event_input.extents[i] } else { &click_input.extents[i] };
        x.to - x.from
    };
    work.sort_by_key(|w| std::cmp::Reverse(extent_len(w)));
    let next = AtomicUsize::new(0);
    phase(3, &|t| unsafe {
        let w = workers.get(t);
        w.d.reset(sh.d_parts);
        w.p.reset(sh.p_parts);
        w.c.reset(sh.p_parts);
        loop {
            let i = next.fetch_add(1, Relaxed);
            let Some(&(is_event, i)) = work.get(i) else { break };
            if is_event {
                let x = &event_input.extents[i];
                parse_events(&sh, event_input.files[x.file], x, w);
                event_input.done(x);
            } else {
                let x = &click_input.extents[i];
                parse_clicks(&sh, click_input.files[x.file], x, w);
                click_input.done(x);
            }
        }
        w.d.finish(&sh.d_arena);
        w.p.finish(&sh.p_arena);
        w.c.finish(&sh.c_arena);
    });
    marks.push(mark("parse"));
    log_stage(&marks);

    let logs = |f: &dyn Fn(&Worker) -> &Vec<(u32, u32, u32)>| -> Vec<&Vec<(u32, u32, u32)>> {
        (0..threads).map(|t| f(unsafe { workers.get(t) })).collect()
    };
    let d_chunks = by_partition(&logs(&|w| &w.d.log), sh.d_parts);
    let p_chunks = by_partition(&logs(&|w| &w.p.log), sh.p_parts);
    let c_chunks = by_partition(&logs(&|w| &w.c.log), sh.p_parts);

    let next = AtomicUsize::new(0);
    phase(4, &|_| {
        let mut work = group::Dedup::default();
        loop {
            let i = next.fetch_add(1, Relaxed);
            let Some(chunks) = d_chunks.get(i) else { break };
            unsafe { work.partition(&sh, chunks) };
        }
    });
    marks.push(mark("dedup"));
    log_stage(&marks);

    let size_of = |chunks: &Vec<(u32, u32)>| chunks.iter().map(|c| c.1 as u64).sum::<u64>();
    let mut order: Vec<usize> = (0..sh.p_parts).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(size_of(&p_chunks[i]) + size_of(&c_chunks[i])));
    // Tearing memory down at exit is serial and on the clock, so it is freed in parallel: what is dead
    // by now (links table, dedup arena) by threads that run out of sessions work, the rest while
    // thread 0 writes the campaign table. Only pages ever touched are walked.
    const PIECE: usize = 32 << 20;
    let touched = |base: usize, used: usize, populated: usize, cap: usize| (base, used.max(populated).min(cap));
    let split = |regions: &[(usize, usize)]| -> Vec<(usize, usize)> {
        regions.iter().flat_map(|&(b, n)| (0..n).step_by(PIECE).map(move |at| (b + at, PIECE.min(n - at)))).collect()
    };
    let dead_pieces = split(&[touched(sh.d_arena.base as usize, sh.d_arena.used_bytes(), populated[0], sh.d_arena.cap_bytes())]);
    let live_pieces = split(&[
        touched(sh.p_arena.base as usize, sh.p_arena.used_bytes(), populated[1], sh.p_arena.cap_bytes()),
        touched(sh.c_arena.base as usize, sh.c_arena.used_bytes(), populated[2], sh.c_arena.cap_bytes()),
    ]);
    let release = std::env::var_os("SS_NORELEASE").is_none();
    let (dead_next, live_next) = (AtomicUsize::new(0), AtomicUsize::new(0));
    const REL_SWEEP: usize = 1 << 16;
    let rel_sweeps = sh.links.slots().div_ceil(REL_SWEEP);
    let drain_dead = || {
        while release {
            let i = dead_next.fetch_add(1, Relaxed);
            if i < rel_sweeps {
                sh.links.release(i * REL_SWEEP, ((i + 1) * REL_SWEEP).min(sh.links.slots()));
                continue;
            }
            let Some(&(at, n)) = dead_pieces.get(i - rel_sweeps) else { break };
            sys::release(at as *mut u8, n);
        }
    };
    let groupers = PerThread::new(threads, |t| {
        Grouper::new(output.join("user_daily").join(format!("part-{t:03}.parquet")), (sh.campaigns.len() + 1) * days)
    });
    let next = AtomicUsize::new(0);
    phase(5, &|t| unsafe {
        let g = groupers.get(t);
        loop {
            let i = next.fetch_add(1, Relaxed);
            let Some(&part) = order.get(i) else { break };
            g.partition(&sh, &p_chunks[part], &c_chunks[part]);
        }
        std::mem::replace(&mut g.out, group::UserOut::new(PathBuf::new())).finish(t == 0);
        drain_dead();
    });
    marks.push(mark("sessions"));
    log_stage(&marks);

    phase(7, &|t| {
        if t == 0 {
            let cells: Vec<Vec<Cell>> = (0..threads).map(|t| unsafe { groupers.get(t).take_cells() }).collect();
            write_campaigns(&sh, &output.join("campaign_daily").join("part-000.parquet"), &cells);
        }
        drain_dead();
        while release {
            let Some(&(at, n)) = live_pieces.get(live_next.fetch_add(1, Relaxed)) else { break };
            sys::release(at as *mut u8, n);
        }
    });
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
        for (stage, name) in ["links parse", "links join", "links sweep", "parse", "dedup", "sessions"].iter().enumerate() {
            let cell = |k: usize| -> String {
                (0..threads).map(|t| format!("{:4.0}", unsafe { workers.get(t).busy[stage][k] } * 1e3)).collect::<Vec<_>>().join("")
            };
            eprintln!("{name:12} cpu  {}", cell(0));
            eprintln!("{:12} wait {}", "", cell(1));
            eprintln!("{:12} idle {}", "", cell(2));
        }
        eprintln!(
            "threads {threads}  d_parts {}  p_parts {}  events lines {}  valid {}  clicks {}  peak rss {} MB",
            sh.d_parts,
            sh.p_parts,
            d_chunks.iter().map(size_of).sum::<u64>(),
            p_chunks.iter().map(size_of).sum::<u64>(),
            c_chunks.iter().map(size_of).sum::<u64>(),
            sys::peak_rss_kb() / 1024
        );
        let ru = sys::rusage();
        eprintln!("total wall {:.1} ms  user {:.1} ms  sys {:.1} ms", t0.elapsed().as_secs_f64() * 1e3, ru.0 * 1e3, ru.1 * 1e3);
    }
    let stages: Vec<(&str, f64)> = marks.iter().scan(0.0, |prev, m| { let d = m.1 - *prev; *prev = m.1; Some((m.0, d)) }).collect();
    let ru = sys::rusage();
    let rows = [&d_chunks, &p_chunks, &c_chunks].map(|c| c.iter().map(size_of).sum::<u64>());
    diag::summary(t0.elapsed().as_secs_f64(), &stages, ru.0, ru.1, rows);
    sys::prof::finish();
    std::process::exit(0);
}
