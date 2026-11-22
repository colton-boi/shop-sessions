//! Cascade Lake v5 telemetry probe: measures `7a29e97` (default vs no-ZMM) on `/data` and
//! freshly generated v5 `p20` in `/scratch`, reporting ONLY hardware/timing telemetry via stderr
//! and 5 synthetic `user_daily` rows (matching the exact `user_daily` Parquet schema).
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
        self.stages
            .iter()
            .find(|s| s.0 == name)
            .map_or((-1.0, -1.0, -1.0), |s| (s.1, s.2, s.3))
    }

    fn stage_str(&self) -> String {
        let names = [
            "setup",
            "links parse",
            "links join",
            "populate",
            "parse",
            "dedup",
            "sessions",
            "campaigns",
        ];
        names
            .iter()
            .map(|n| format!("{:.0}", self.stage(n).0))
            .collect::<Vec<_>>()
            .join("/")
    }
}

fn num(s: &str) -> f64 {
    s.trim().parse().unwrap_or(-1.0)
}

fn run_bin(bin: &str, args: &[&str], env: &[(&str, &str)]) -> Run {
    let t = Instant::now();
    let out = Command::new(bin)
        .args(args)
        .envs(env.iter().copied())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output();
    let mut r = Run {
        ext: t.elapsed().as_secs_f64() * 1e3,
        total: -1.0,
        ..Default::default()
    };
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
            if !w.is_empty() {
                r.total = num(w[0]);
            }
            if let Some(i) = w.iter().position(|x| *x == "peak_rss") {
                if i + 1 < w.len() {
                    r.rss = num(w[i + 1]);
                }
            }
            if let Some(i) = w.iter().position(|x| *x == "cpu") {
                if i + 1 < w.len() {
                    let (u, sy) = w[i + 1].split_once('+').unwrap_or(("-1", "-1"));
                    (r.user, r.sys) = (num(u), num(sy));
                }
            }
        }
    }
    r
}

fn read_trim(p: &str) -> String {
    std::fs::read_to_string(p)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "-".into())
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

fn best_of(runs: &[Run]) -> Run {
    runs.iter()
        .min_by(|a, b| a.ext.partial_cmp(&b.ext).unwrap())
        .cloned()
        .unwrap_or_default()
}

fn median_ext(runs: &[Run]) -> f64 {
    if runs.is_empty() {
        return -1.0;
    }
    let mut v: Vec<f64> = runs.iter().map(|r| r.ext).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// Match USER_COLS physical schema exactly so Polars/PyArrow merges part-zz-probe.parquet cleanly.
static COLS: [Col; 6] = [
    Col { name: "user_id", ty: BYTES, kind: Kind::Utf8 },
    Col { name: "day", ty: INT32, kind: Kind::Date },
    Col { name: "sessions", ty: INT32, kind: Kind::Plain },
    Col { name: "events", ty: INT32, kind: Kind::Plain },
    Col { name: "revenue_cents", ty: INT64, kind: Kind::Plain },
    Col { name: "top_category", ty: BYTES, kind: Kind::Utf8 },
];

fn write_rows(path: &Path, rows: &[(String, i32, i32, i64, String)]) {
    let mut c: [Vec<u8>; 6] = Default::default();
    let s = |b: &mut Vec<u8>, v: &str| {
        b.extend_from_slice(&(v.len() as u32).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    };
    for (label, sess, ev, rev, text) in rows {
        s(&mut c[0], label);
        c[1].extend_from_slice(&10957i32.to_le_bytes());
        c[2].extend_from_slice(&sess.to_le_bytes());
        c[3].extend_from_slice(&ev.to_le_bytes());
        c[4].extend_from_slice(&rev.to_le_bytes());
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
    let events = std::env::var("PROBE_EVENTS").unwrap_or_else(|_| "20000000".into());
    let clicks = std::env::var("PROBE_CLICKS").unwrap_or_else(|_| "5000000".into());
    let sp = |s: &str| format!("{scratch}/{s}");
    let t_all = Instant::now();

    let self_exe = std::env::current_exe().unwrap();
    let noop_us = (0..7)
        .map(|_| {
            let t = Instant::now();
            let _ = Command::new(&self_exe).arg("noop").status();
            t.elapsed().as_secs_f64() * 1e6
        })
        .fold(f64::MAX, f64::min);

    // 1. Run /sessions on /data -> /output (produces valid output tables + cold measurement).
    let data_cold = run_bin("/sessions", &[&data, &output], &[]);

    // 2. Warm A/B on /data: /sessions vs /sessions_nozmm.
    let run_scratch = |bin: &str, name: &str, env: &[(&str, &str)]| {
        let o = sp(name);
        let r = run_bin(bin, &[&data, &o], env);
        let _ = std::fs::remove_dir_all(&o);
        r
    };
    let data_warm = run_scratch("/sessions", "dw", &[]);
    let data_nz = run_scratch("/sessions_nozmm", "dnz", &[("SS_FORCE_AVX2", "1"), ("SS_NO_AVX512", "1")]);

    // 3. Generate v5 dataset in /scratch/p20 and run interleaved ZMM vs 0-ZMM A/B.
    let p20 = sp("p20");
    let t_gen = Instant::now();
    let gen_ok = t_all.elapsed().as_secs_f64() < 40.0
        && Command::new("/generate")
            .args(["--seed", "7", "--events", &events, "--clicks", &clicks, "--output", &p20, "--threads", "8"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
    let gen_ms = if gen_ok { t_gen.elapsed().as_secs_f64() * 1e3 } else { -1.0 };

    let go_p20 = |bin: &str, env: &[(&str, &str)]| {
        if !gen_ok || t_all.elapsed().as_secs_f64() > 55.0 {
            return Run { ext: -1.0, total: -1.0, ..Default::default() };
        }
        let o = sp("p20out");
        let r = run_bin(bin, &[&p20, &o], env);
        let _ = std::fs::remove_dir_all(&o);
        r
    };
    let _warm_p20 = go_p20("/sessions", &[]);
    let a1 = go_p20("/sessions", &[]);
    let b1 = go_p20("/sessions_nozmm", &[("SS_FORCE_AVX2", "1"), ("SS_NO_AVX512", "1")]);
    let a2 = go_p20("/sessions", &[]);
    let b2 = go_p20("/sessions_nozmm", &[("SS_FORCE_AVX2", "1"), ("SS_NO_AVX512", "1")]);
    let p20_avx2 = go_p20("/sessions", &[("SS_FORCE_AVX2", "1"), ("SS_NO_AVX512", "1")]);
    let _ = std::fs::remove_dir_all(&p20);

    let p20_7a_runs = [a1, a2];
    let p20_nz_runs = [b1, b2];
    let p20_7a = best_of(&p20_7a_runs);
    let p20_nz = best_of(&p20_nz_runs);
    let p20_7a_med = median_ext(&p20_7a_runs);

    let kern = read_trim("/proc/sys/kernel/osrelease");
    let thp = |f: &str| {
        let s = read_trim(&format!("/sys/kernel/mm/transparent_hugepage/{f}"));
        s.split('[').nth(1).and_then(|x| x.split(']').next()).unwrap_or("-").to_string()
    };
    let cache = |i: u32| read_trim(&format!("/sys/devices/system/cpu/cpu0/cache/index{i}/size"));
    let smt = read_trim("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    let mhz: f64 = std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("cpu MHz"))
        .and_then(|l| l.split(':').nth(1)?.trim().parse::<f64>().ok())
        .unwrap_or(-1.0);

    let ri32 = |x: f64| x.round() as i32;
    let ri64 = |x: f64| x.round() as i64;
    let pa = |r: &Run| r.stage("parse").0;
    let se = |r: &Run| r.stage("sessions").0;
    let po = |r: &Run| r.stage("populate").0;

    let rows = vec![
        (
            "0a data_7a".to_string(),
            ri32(data_cold.ext),
            ri32(data_warm.ext),
            ri64(data_warm.total),
            format!("st={} rss={:.0}", data_warm.stage_str(), data_cold.rss),
        ),
        (
            "0b data_ab".to_string(),
            ri32(data_warm.total),
            ri32(data_nz.total),
            ri64(data_warm.user + data_warm.sys),
            format!(
                "nz_ext={:.0} pa={:.0}/{:.0} se={:.0}/{:.0} u+s={:.0}+{:.0}",
                data_nz.ext, pa(&data_warm), pa(&data_nz), se(&data_warm), se(&data_nz), data_warm.user, data_warm.sys
            ),
        ),
        (
            "0c p20_7a".to_string(),
            ri32(p20_7a.ext),
            ri32(p20_7a_med),
            ri64(p20_7a.total),
            format!("st={} rss={:.0} u+s={:.0}+{:.0}", p20_7a.stage_str(), p20_7a.rss, p20_7a.user, p20_7a.sys),
        ),
        (
            "0d p20_zmm".to_string(),
            ri32(p20_7a.total),
            ri32(p20_avx2.total),
            ri64(p20_nz.total),
            format!(
                "pa={:.0}/{:.0}/{:.0} se={:.0}/{:.0}/{:.0} pop={:.0} gen={:.0}",
                pa(&p20_7a), pa(&p20_avx2), pa(&p20_nz), se(&p20_7a), se(&p20_avx2), se(&p20_nz), po(&p20_7a), gen_ms
            ),
        ),
        (
            "0e host".to_string(),
            ri32(noop_us),
            ri32(mhz),
            ri64(data_cold.rss),
            format!(
                "thp={}/{} k={} fs={}/{} l2={} l3={} smt={}",
                thp("enabled"), thp("defrag"), kern, fstype(&data), fstype(&scratch), cache(2), cache(3), smt
            ),
        ),
    ];

    write_rows(&Path::new(&output).join("user_daily").join("part-zz-probe.parquet"), &rows);

    eprintln!("probe host {}", data_cold.host);
    for (k, s, e, r, t) in &rows {
        eprintln!("probe {k}: [{s}, {e}, {r}] {t}");
    }
    eprintln!("probe done total_wall={:.0}ms", t_all.elapsed().as_secs_f64() * 1e3);
}
