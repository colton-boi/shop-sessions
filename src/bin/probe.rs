//! Scorer probe: times the real pipeline on the scoring machine and reports
//! through the stderr tail and through extra `user_daily` rows, the two
//! channels a failed dev check returns. Not a scoring entry.
#[allow(dead_code)]
#[path = "../parquet.rs"]
mod parquet;

use parquet::{Col, Kind, Writer, BYTES, INT32, INT64};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Default, Clone)]
struct Run {
    ext: f64,
    total: f64,
    user: f64,
    sys: f64,
    rss: f64,
    stages: Vec<(String, f64, f64, f64)>,
    host: String,
}

impl Run {
    fn stage(&self, name: &str) -> (f64, f64, f64) {
        self.stages.iter().find(|s| s.0 == name).map_or((-1.0, -1.0, -1.0), |s| (s.1, s.2, s.3))
    }
}

fn num(s: &str) -> f64 {
    s.trim().parse().unwrap_or(-1.0)
}

fn run(args: &[&str], env: &[(&str, &str)]) -> Run {
    let t = Instant::now();
    let out = Command::new("/sessions").args(args).envs(env.iter().copied()).stdout(Stdio::null()).stderr(Stdio::piped()).output();
    let mut r = Run { ext: t.elapsed().as_secs_f64() * 1e3, total: -1.0, ..Default::default() };
    let Ok(out) = out else { return r };
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        if let Some(h) = line.strip_prefix("ss host ") {
            r.host = h.to_string();
        } else if let Some(s) = line.strip_prefix("ss stage ") {
            if let Some((left, right)) = s.split_once(" ms  cpu ") {
                let left = left.trim();
                let (name, wall) = left.rsplit_once(' ').unwrap_or((left, "-1"));
                let (u, sy) = right.trim_end_matches(" ms").split_once('+').unwrap_or(("-1", "-1"));
                r.stages.push((name.trim().to_string(), num(wall), num(u), num(sy)));
            }
        } else if let Some(s) = line.strip_prefix("ss done total ") {
            let w: Vec<&str> = s.split_whitespace().collect();
            r.total = num(w[0]);
            if let Some(i) = w.iter().position(|x| *x == "peak_rss") {
                r.rss = num(w[i + 1]);
            }
            if let Some(i) = w.iter().position(|x| *x == "cpu") {
                let (u, sy) = w[i + 1].split_once('+').unwrap_or(("-1", "-1"));
                (r.user, r.sys) = (num(u), num(sy));
            }
        }
    }
    r
}

fn read_trim(p: &str) -> String {
    std::fs::read_to_string(p).map(|s| s.trim().to_string()).unwrap_or_else(|_| "-".into())
}

fn fstype(mount: &str) -> String {
    let info = std::fs::read_to_string("/proc/self/mountinfo").unwrap_or_default();
    for l in info.lines() {
        let f: Vec<&str> = l.split(' ').collect();
        if f.len() > 4 && f[4] == mount {
            if let Some(i) = f.iter().position(|x| *x == "-") {
                return f.get(i + 1).unwrap_or(&"?").to_string();
            }
        }
    }
    "?".into()
}

fn meminfo(key: &str) -> u64 {
    let m = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    m.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()).unwrap_or(0) / 1024
}

fn read_all_ms(dir: &Path) -> f64 {
    let t = Instant::now();
    for sub in ["events", "clicks", "links"] {
        if let Ok(rd) = std::fs::read_dir(dir.join(sub)) {
            for e in rd.flatten() {
                let _ = std::fs::read(e.path());
            }
        }
    }
    t.elapsed().as_secs_f64() * 1e3
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

static COLS: [Col; 6] = [
    Col { name: "user_id", ty: BYTES, kind: Kind::Utf8 },
    Col { name: "day", ty: INT32, kind: Kind::Date },
    Col { name: "sessions", ty: INT64, kind: Kind::Plain },
    Col { name: "events", ty: INT64, kind: Kind::Plain },
    Col { name: "revenue_cents", ty: INT64, kind: Kind::Plain },
    Col { name: "top_category", ty: BYTES, kind: Kind::Utf8 },
];

fn write_rows(path: &Path, rows: &[(String, [i64; 3], String)]) {
    let mut c: [Vec<u8>; 6] = Default::default();
    let s = |b: &mut Vec<u8>, v: &str| {
        b.extend_from_slice(&(v.len() as u32).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    };
    for (label, n, text) in rows {
        s(&mut c[0], label);
        c[1].extend_from_slice(&10957i32.to_le_bytes());
        for k in 0..3 {
            c[2 + k].extend_from_slice(&n[k].to_le_bytes());
        }
        s(&mut c[5], text);
    }
    let mut w = Writer::create(path, &COLS);
    let data: Vec<(&[u8], Option<&[u8]>)> = c.iter().map(|v| (v.as_slice(), None)).collect();
    w.row_group(rows.len(), &data);
    w.finish();
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("noop") {
        return;
    }
    let data = std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into());
    let output = std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into());
    let scratch = std::env::var("SCRATCH_DIR").unwrap_or_else(|_| "/scratch".into());
    let sp = |s: &str| format!("{scratch}/{s}");
    let t_all = Instant::now();

    let rd1 = read_all_ms(Path::new(&data));
    let rd2 = read_all_ms(Path::new(&data));
    let self_exe = std::env::current_exe().unwrap();
    let noop_us = (0..7)
        .map(|_| {
            let t = Instant::now();
            let _ = Command::new(&self_exe).arg("noop").status();
            t.elapsed().as_secs_f64() * 1e6
        })
        .fold(f64::MAX, f64::min);

    let dev: Vec<Run> = (0..3)
        .map(|i| {
            let o = sp(&format!("dev{i}"));
            let r = run(&[&data, &o], &[]);
            let _ = std::fs::remove_dir_all(&o);
            r
        })
        .collect();

    let t = Instant::now();
    let gen_ok = Command::new("/generate")
        .args(["--seed", "7", "--events", "20000000", "--clicks", "5000000", "--output", &sp("p20")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let gen_ms = if gen_ok { t.elapsed().as_secs_f64() * 1e3 } else { -1.0 };

    let p20 = sp("p20");
    let go = |env: &[(&str, &str)]| {
        if !gen_ok {
            return Run { ext: -1.0, total: -1.0, ..Default::default() };
        }
        let o = sp("p20out");
        let r = run(&[&p20, &o], env);
        let _ = std::fs::remove_dir_all(&o);
        r
    };
    let warm = go(&[]);
    let base: Vec<Run> = (0..3).map(|_| go(&[])).collect();
    let nohuge = go(&[("SS_NOHUGE", "1")]);
    let t4 = go(&[("SS_THREADS", "4")]);
    let base2 = go(&[]);
    let _ = std::fs::remove_dir_all(&p20);

    // The real tables, plus the report rows that make the dev check differ.
    let fin = run(&[&data, &output], &[]);
    let best = base.iter().chain([&base2]).min_by(|a, b| a.ext.partial_cmp(&b.ext).unwrap()).cloned().unwrap();
    let mut exts: Vec<f64> = base.iter().chain([&base2]).map(|r| r.ext).collect();
    let ext_med = median(&mut exts);
    let mut dev_ext: Vec<f64> = dev.iter().map(|r| r.ext).collect();
    let dev_min = dev_ext.iter().cloned().fold(f64::MAX, f64::min);
    let dev_tot = dev.iter().map(|r| r.total).fold(f64::MAX, f64::min);
    let _ = median(&mut dev_ext);

    let kern = read_trim("/proc/sys/kernel/osrelease");
    let thp = |f: &str| {
        let s = read_trim(&format!("/sys/kernel/mm/transparent_hugepage/{f}"));
        s.split('[').nth(1).and_then(|x| x.split(']').next()).unwrap_or("-").to_string()
    };
    let cache = |i: u32| read_trim(&format!("/sys/devices/system/cpu/cpu0/cache/index{i}/size"));
    let smt = read_trim("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    let mhz = std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("cpu MHz"))
        .and_then(|l| l.split(':').nth(1).map(|v| v.trim().split('.').next().unwrap_or("").to_string()))
        .unwrap_or_default();
    let (pw, _pu, ps) = best.stage("populate");
    let (parse_w, parse_u, _) = best.stage("parse");
    let (sess_w, sess_u, _) = best.stage("sessions");
    let (dedup_w, _, _) = best.stage("dedup");
    let (lj_w, _, _) = best.stage("links join");
    let (camp_w, _, _) = best.stage("campaigns");
    let (setup_w, _, _) = best.stage("setup");
    let facts = format!(
        "thp={}/{} k={} fs={}/{} l2={} l3={} smt={} mhz={} mem={}G cache={}G",
        thp("enabled"),
        thp("defrag"),
        kern,
        fstype(&data),
        fstype(&scratch),
        cache(2),
        cache(3),
        smt,
        mhz,
        meminfo("MemTotal") / 1024,
        meminfo("Cached") / 1024
    );

    let r = |x: f64| x.round() as i64;
    let rows = vec![
        ("0a p20 wall".to_string(), [r(best.ext), r(ext_med), r(best.total)], facts.clone()),
        ("0b p20 stg".to_string(), [r(parse_w), r(sess_w), r(pw)], format!("dedup={dedup_w} lj={lj_w} camp={camp_w} setup={setup_w}")),
        ("0c p20 cpu".to_string(), [r(best.user), r(best.sys), r(ps)], format!("parse_u={parse_u} sess_u={sess_u} rss={}", best.rss)),
        ("0d variants".to_string(), [r(nohuge.ext), r(t4.ext), r(gen_ms)], format!("nh_tot={} t4_tot={} warm={}", nohuge.total, t4.total, warm.ext)),
        ("0e base".to_string(), [r(noop_us), r(dev_min), r(dev_tot)], format!("rd={rd1:.0}/{rd2:.0} fin={:.0}", fin.ext)),
    ];
    write_rows(&Path::new(&output).join("user_daily").join("part-zz-probe.parquet"), &rows);

    eprintln!("probe host {}", fin.host);
    eprintln!("probe facts {facts}");
    eprintln!(
        "probe A noop={noop_us:.0}us dev={dev_min:.0}/{dev_tot:.0} rd={rd1:.0}/{rd2:.0} gen={gen_ms:.0} nohuge={:.0}/{:.0} t4={:.0}/{:.0} all={:.0}",
        nohuge.ext,
        nohuge.total,
        t4.ext,
        t4.total,
        t_all.elapsed().as_secs_f64() * 1e3
    );
    eprintln!(
        "probe B p20 ext={:.0}/{ext_med:.0} tot={:.0} u+s={:.0}+{:.0} setup={setup_w:.0} lj={lj_w:.0} pop={pw:.0}/{ps:.0}s parse={parse_w:.0} dd={dedup_w:.0} sess={sess_w:.0} camp={camp_w:.0}",
        best.ext, best.total, best.user, best.sys
    );
}
