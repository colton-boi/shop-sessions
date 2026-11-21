//! Diagnostic runner for stage timings, CPU/RSS usage, thread scaling, THP state,
//! and tuning knobs on dev + generated 20M-event datasets.
#![allow(dead_code)]

#[path = "../parquet.rs"]
mod parquet;

use parquet::{Col, Kind, Writer, BYTES, INT32, INT64};
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

extern "C" {
    fn wait4(pid: i32, status: *mut i32, options: i32, rusage: *mut [i64; 18]) -> i32;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
}

#[derive(Default, Clone)]
struct StageStats {
    name: String,
    wall_ms: f64,
    user_ms: f64,
    sys_ms: f64,
    kfault: f64,
    mcyc: f64,
    minstr: f64,
    mllc: f64,
    mbr: f64,
}

#[derive(Default, Clone)]
struct RunResult {
    ext_ms: f64,
    in_total_ms: f64,
    user_ms: f64,
    sys_ms: f64,
    peak_rss_mb: f64,
    kfault_total: f64,
    cx_ev: i64,
    cx_ck: i64,
    stages: Vec<StageStats>,
    host: String,
}

impl RunResult {
    fn stage(&self, name: &str) -> StageStats {
        self.stages
            .iter()
            .find(|s| s.name == name)
            .cloned()
            .unwrap_or_else(|| StageStats {
                name: name.to_string(),
                wall_ms: -1.0,
                user_ms: -1.0,
                sys_ms: -1.0,
                ..Default::default()
            })
    }
}

fn parse_num(s: &str) -> f64 {
    s.trim().parse().unwrap_or(-1.0)
}

fn key_val(words: &[&str], key: &str) -> Option<f64> {
    words
        .iter()
        .position(|&w| w == key)
        .and_then(|i| words.get(i + 1))
        .and_then(|v| v.parse::<f64>().ok())
}

fn physical_core_cpus(limit: usize) -> Vec<usize> {
    let mut seen_groups: Vec<String> = Vec::new();
    let mut phys: Vec<usize> = Vec::new();
    for cpu in 0..256usize {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let Ok(raw) = std::fs::read_to_string(&path) else {
            break;
        };
        let norm = raw.trim().to_string();
        if !seen_groups.contains(&norm) {
            seen_groups.push(norm);
            phys.push(cpu);
            if phys.len() == limit {
                return phys;
            }
        }
    }
    if phys.is_empty() {
        return (0..limit).collect();
    }
    for cpu in 0..256usize {
        if phys.len() == limit {
            break;
        }
        if !phys.contains(&cpu) {
            phys.push(cpu);
        }
    }
    phys
}

fn run_sessions(
    bin: &str,
    args: &[&str],
    env: &[(&str, &str)],
    pin_cpus: Option<&[usize]>,
    sample_ahp: bool,
) -> (RunResult, u64) {
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .envs(env.iter().copied())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if let Some(cpus) = pin_cpus {
        let mut mask = [0u8; 128];
        for &cpu in cpus {
            if cpu < 1024 {
                mask[cpu / 8] |= 1u8 << (cpu % 8);
            }
        }
        unsafe {
            cmd.pre_exec(move || {
                let rc = sched_setaffinity(0, mask.len(), mask.as_ptr());
                if rc != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    let ahp_stop = Arc::new(AtomicBool::new(false));
    let ahp_peak_kb = Arc::new(AtomicU64::new(0));
    let ahp_base_kb = meminfo_kb("AnonHugePages:");
    let ahp_thread = if sample_ahp {
        let stop = Arc::clone(&ahp_stop);
        let peak = Arc::clone(&ahp_peak_kb);
        Some(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let cur = meminfo_kb("AnonHugePages:");
                peak.fetch_max(cur, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(15));
            }
        }))
    } else {
        None
    };

    let t0 = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => {
            ahp_stop.store(true, Ordering::Relaxed);
            if let Some(h) = ahp_thread {
                let _ = h.join();
            }
            return (
                RunResult {
                    ext_ms: -1.0,
                    in_total_ms: -1.0,
                    ..Default::default()
                },
                0,
            );
        }
    };

    let mut stderr_bytes = Vec::with_capacity(4096);
    if let Some(mut se) = child.stderr.take() {
        let _ = se.read_to_end(&mut stderr_bytes);
    }

    let mut status = 0i32;
    let mut ru = [0i64; 18];
    let rc = unsafe { wait4(child.id() as i32, &mut status, 0, &mut ru) };
    let ext_ms = t0.elapsed().as_secs_f64() * 1e3;

    ahp_stop.store(true, Ordering::Relaxed);
    if let Some(h) = ahp_thread {
        let _ = h.join();
    }
    let ahp_delta_mb = ahp_peak_kb.load(Ordering::Relaxed).saturating_sub(ahp_base_kb) / 1024;

    let mut res = RunResult {
        ext_ms,
        in_total_ms: -1.0,
        ..Default::default()
    };
    if rc > 0 {
        res.user_ms = (ru[0] as f64) * 1e3 + (ru[1] as f64) * 1e-3;
        res.sys_ms = (ru[2] as f64) * 1e3 + (ru[3] as f64) * 1e-3;
        res.peak_rss_mb = (ru[4] as f64) / 1024.0;
    }

    let stderr_str = String::from_utf8_lossy(&stderr_bytes);
    for line in stderr_str.lines() {
        if let Some(h) = line.strip_prefix("ss host ") {
            res.host = h.to_string();
        } else if let Some(s) = line.strip_prefix("ss stage ") {
            if let Some((left, right)) = s.split_once(" ms  cpu ") {
                let left = left.trim();
                let (name, wall) = left.rsplit_once(' ').unwrap_or((left, "-1"));
                let (u, sy) = right
                    .trim_end_matches(" ms")
                    .split_once('+')
                    .unwrap_or(("-1", "-1"));
                let name = name.trim().to_string();
                if !res.stages.iter().any(|st| st.name == name) {
                    res.stages.push(StageStats {
                        name,
                        wall_ms: parse_num(wall),
                        user_ms: parse_num(u),
                        sys_ms: parse_num(sy),
                        ..Default::default()
                    });
                }
            }
        } else if let Some(s) = line.strip_prefix("ss done total ") {
            let w: Vec<&str> = s.split_whitespace().collect();
            if let Some(tot) = w.first() {
                res.in_total_ms = parse_num(tot);
            }
            if let Some(rss) = key_val(&w, "peak_rss") {
                res.peak_rss_mb = rss;
            }
            if let Some(i) = w.iter().position(|&x| x == "cpu") {
                if let Some(pair) = w.get(i + 1) {
                    if let Some((u, sy)) = pair.split_once('+') {
                        res.user_ms = parse_num(u);
                        res.sys_ms = parse_num(sy);
                    }
                }
            }
        } else if line.contains(" wall ") && line.contains(" kfault ") && line.contains(" Mcyc ") {
            let w: Vec<&str> = line.split_whitespace().collect();
            if let Some(wall_idx) = w.iter().position(|&x| x == "wall") {
                let name = w[..wall_idx].join(" ");
                let wall_ms = key_val(&w, "wall").unwrap_or(-1.0);
                let user_ms = key_val(&w, "user").unwrap_or(-1.0);
                let sys_ms = key_val(&w, "sys").unwrap_or(-1.0);
                let kfault = key_val(&w, "kfault").unwrap_or(0.0);
                let mcyc = key_val(&w, "Mcyc").unwrap_or(0.0);
                let minstr = key_val(&w, "Minstr").unwrap_or(0.0);
                let mllc = key_val(&w, "Mllc").unwrap_or(0.0);
                let mbr = key_val(&w, "Mbr").unwrap_or(0.0);
                res.kfault_total += kfault.max(0.0);
                if let Some(existing) = res.stages.iter_mut().find(|st| st.name == name) {
                    existing.wall_ms = wall_ms;
                    existing.user_ms = user_ms;
                    existing.sys_ms = sys_ms;
                    existing.kfault = kfault;
                    existing.mcyc = mcyc;
                    existing.minstr = minstr;
                    existing.mllc = mllc;
                    existing.mbr = mbr;
                } else {
                    res.stages.push(StageStats {
                        name,
                        wall_ms,
                        user_ms,
                        sys_ms,
                        kfault,
                        mcyc,
                        minstr,
                        mllc,
                        mbr,
                    });
                }
            }
        } else if line.starts_with("threads ") && line.contains(" cx_ev ") {
            let w: Vec<&str> = line.split_whitespace().collect();
            if let Some(v) = key_val(&w, "cx_ev") {
                res.cx_ev = v.round() as i64;
            }
            if let Some(v) = key_val(&w, "cx_ck") {
                res.cx_ck = v.round() as i64;
            }
            if let Some(v) = key_val(&w, "rss") {
                res.peak_rss_mb = v;
            }
        } else if line.starts_with("total wall ") {
            let w: Vec<&str> = line.split_whitespace().collect();
            if let Some(v) = key_val(&w, "wall") {
                res.in_total_ms = v;
            }
            if let Some(v) = key_val(&w, "user") {
                res.user_ms = v;
            }
            if let Some(v) = key_val(&w, "sys") {
                res.sys_ms = v;
            }
        }
    }

    (res, ahp_delta_mb)
}

fn meminfo_kb(key: &str) -> u64 {
    let m = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    m.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        .unwrap_or(0)
}

fn cpu_mhz() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("cpu MHz"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().split('.').next())
        .unwrap_or("-")
        .to_string()
}

fn cgroup_val(file: &str) -> String {
    let read_trim = |p: &str| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string());
    let own = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("0::").map(str::to_string)));
    if let Some(p) = own {
        if let Some(v) = read_trim(&format!("/sys/fs/cgroup{}/{file}", p.trim_end_matches('/'))) {
            return v;
        }
    }
    read_trim(&format!("/sys/fs/cgroup/{file}")).unwrap_or_else(|| "-".into())
}

fn dir_bytes(dir: &Path) -> u64 {
    let mut sum = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Ok(m) = e.metadata() {
                if m.is_file() {
                    sum += m.len();
                }
            }
        }
    }
    sum
}

static COLS: [Col; 6] = [
    Col { name: "user_id", ty: BYTES, kind: Kind::Utf8 },
    Col { name: "day", ty: INT32, kind: Kind::Date },
    Col { name: "sessions", ty: INT64, kind: Kind::Plain },
    Col { name: "events", ty: INT64, kind: Kind::Plain },
    Col { name: "revenue_cents", ty: INT64, kind: Kind::Plain },
    Col { name: "top_category", ty: BYTES, kind: Kind::Utf8 },
];

fn write_probe_rows(path: &Path, rows: &[(String, [i64; 3], String)]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut c: [Vec<u8>; 6] = Default::default();
    let push_str = |b: &mut Vec<u8>, v: &str| {
        b.extend_from_slice(&(v.len() as u32).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    };
    for (label, vals, text) in rows {
        push_str(&mut c[0], label);
        c[1].extend_from_slice(&10957i32.to_le_bytes());
        for k in 0..3 {
            c[2 + k].extend_from_slice(&vals[k].to_le_bytes());
        }
        push_str(&mut c[5], text);
    }
    let mut w = Writer::create(path, &COLS);
    let data: Vec<(&[u8], Option<&[u8]>)> = c.iter().map(|v| (v.as_slice(), None)).collect();
    w.row_group(rows.len(), &data);
    w.finish();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir_arg = |var: &str, back: usize, default: &str| -> String {
        if args.len() >= 3 {
            return args[args.len() - back].clone();
        }
        std::env::var(var).unwrap_or_else(|_| default.to_string())
    };
    let data = dir_arg("DATA_DIR", 2, "/data");
    let output = dir_arg("OUTPUT_DIR", 1, "/output");
    let scratch = std::env::var("SCRATCH_DIR").unwrap_or_else(|_| "/scratch".into());
    let sessions_bin = std::env::var("PROBE_SESSIONS").unwrap_or_else(|_| "/sessions".into());
    let generate_bin = std::env::var("PROBE_GENERATE").unwrap_or_else(|_| "/generate".into());

    // 1. Always write normal output on DATA_DIR -> OUTPUT_DIR first.
    let (dev_run, _) = run_sessions(&sessions_bin, &[&data, &output], &[], None, false);

    // Only run the /scratch/p20 diagnostic matrix on dev-sized inputs (< 250 MB events).
    let ev_bytes = dir_bytes(&Path::new(&data).join("events"));
    if ev_bytes >= 250_000_000 {
        return;
    }

    // 2. Generate /scratch/p20 if not already present.
    let p20_dir = PathBuf::from(&scratch).join("p20");
    let p20_str = p20_dir.to_string_lossy().into_owned();
    let mut generated_here = false;
    if !p20_dir.join("manifest.json").exists() {
        let gen_ok = Command::new(&generate_bin)
            .args([
                "--seed",
                "7",
                "--events",
                "20000000",
                "--clicks",
                "5000000",
                "--output",
                &p20_str,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !gen_ok {
            return;
        }
        generated_here = true;
    }

    let p20_out = PathBuf::from(&scratch).join("p20out");
    let p20_out_str = p20_out.to_string_lossy().into_owned();
    let run_p20 = |env: &[(&str, &str)], pin: Option<&[usize]>, ahp: bool| -> (RunResult, u64) {
        let _ = std::fs::remove_dir_all(&p20_out);
        let res = run_sessions(&sessions_bin, &[&p20_str, &p20_out_str], env, pin, ahp);
        let _ = std::fs::remove_dir_all(&p20_out);
        res
    };

    // 3. Matrix on /scratch/p20 with SS_STATS=1
    let (t16_cold, _) = run_p20(&[("SS_STATS", "1")], None, false);
    let (t16, _) = run_p20(&[("SS_STATS", "1")], None, false);
    let phys8 = physical_core_cpus(8);
    let (t8, _) = run_p20(&[("SS_STATS", "1"), ("SS_THREADS", "8")], Some(&phys8), false);
    let (t12, _) = run_p20(&[("SS_STATS", "1"), ("SS_THREADS", "12")], None, false);
    let (nofast, _) = run_p20(&[("SS_STATS", "1"), ("SS_NOFAST", "1")], None, false);
    let (norel, ahp_mb) = run_p20(&[("SS_STATS", "1"), ("SS_NORELEASE", "1")], None, true);
    let (ch256, _) = run_p20(&[("SS_STATS", "1"), ("SS_EV_CHUNK", "256")], None, false);
    let (ch512, _) = run_p20(&[("SS_STATS", "1"), ("SS_EV_CHUNK", "512")], None, false);
    let (ch1024, _) = run_p20(&[("SS_STATS", "1"), ("SS_EV_CHUNK", "1024")], None, false);
    let (ch2048, _) = run_p20(&[("SS_STATS", "1"), ("SS_EV_CHUNK", "2048")], None, false);

    if generated_here {
        let _ = std::fs::remove_dir_all(&p20_dir);
    }

    let mhz = cpu_mhz();
    let avail_par = std::thread::available_parallelism().map_or(0, |n| n.get());
    let cset = cgroup_val("cpuset.cpus.effective");
    let cmax = cgroup_val("cpu.max").replace(' ', "/");
    let r = |x: f64| x.round() as i64;

    let set16 = t16.stage("setup");
    let pop16 = t16.stage("populate");
    let par16 = t16.stage("parse");
    let lk16 = t16.stage("links");
    let ck16 = t16.stage("clicks");
    let s3a16 = t16.stage("sess3a");
    let s3b16 = t16.stage("sess3b");
    let cx16 = t16.stage("complex");
    let cmp16 = t16.stage("campaigns");

    let pop8 = t8.stage("populate");
    let par8 = t8.stage("parse");
    let lk8 = t8.stage("links");
    let ck8 = t8.stage("clicks");
    let s3a8 = t8.stage("sess3a");
    let s3b8 = t8.stage("sess3b");
    let cx8 = t8.stage("complex");
    let cmp8 = t8.stage("campaigns");

    let best_knob_wall = [ch256.ext_ms, ch512.ext_ms, ch1024.ext_ms, ch2048.ext_ms, norel.ext_ms]
        .into_iter()
        .filter(|&v| v > 0.0)
        .fold(f64::MAX, f64::min);
    let best_knob_i64 = if best_knob_wall == f64::MAX { -1 } else { r(best_knob_wall) };

    let row_a = format!(
        "t16_c={:.0} t16_w={:.0} t16_in={:.0} t8_w={:.0} t8_in={:.0} t12_w={:.0} nofast={:.0} norel={:.0} mhz={mhz} ahp={ahp_mb}MB cpus={avail_par} cset={cset} cmax={cmax}",
        t16_cold.ext_ms, t16.ext_ms, t16.in_total_ms, t8.ext_ms, t8.in_total_ms, t12.ext_ms, nofast.ext_ms, norel.ext_ms
    );
    let row_b = format!(
        "t16 wall: set={:.0} pop={:.0} par={:.0} lk={:.0} ck={:.0} s3a={:.0} s3b={:.0} cx={:.0} cmp={:.0} tot={:.0}",
        set16.wall_ms, pop16.wall_ms, par16.wall_ms, lk16.wall_ms, ck16.wall_ms, s3a16.wall_ms, s3b16.wall_ms, cx16.wall_ms, cmp16.wall_ms, t16.in_total_ms
    );
    let row_c = format!(
        "u/s: pop={:.0}/{:.0} par={:.0}/{:.0} lk={:.0}/{:.0} ck={:.0}/{:.0} s3a={:.0}/{:.0} s3b={:.0}/{:.0} cx={:.0}/{:.0} cmp={:.0}/{:.0} kf={:.0}",
        pop16.user_ms, pop16.sys_ms,
        par16.user_ms, par16.sys_ms,
        lk16.user_ms, lk16.sys_ms,
        ck16.user_ms, ck16.sys_ms,
        s3a16.user_ms, s3a16.sys_ms,
        s3b16.user_ms, s3b16.sys_ms,
        cx16.user_ms, cx16.sys_ms,
        cmp16.user_ms, cmp16.sys_ms,
        t16.kfault_total
    );
    let row_d = format!(
        "t8 wall: pop={:.0} par={:.0} lk={:.0} ck={:.0} s3a={:.0} s3b={:.0} cx={:.0} cmp={:.0} tot={:.0} u={:.0} s={:.0}",
        pop8.wall_ms, par8.wall_ms, lk8.wall_ms, ck8.wall_ms, s3a8.wall_ms, s3b8.wall_ms, cx8.wall_ms, cmp8.wall_ms, t8.in_total_ms, t8.user_ms, t8.sys_ms
    );
    let row_e = format!(
        "ch256={:.0} ch512={:.0} ch1024={:.0} ch2048={:.0} norel={:.0} p_cyc={:.0} p_ins={:.0} p_llc={:.1} p_br={:.1} s_llc={:.1} s_cyc={:.0} s_ins={:.0} c_cyc={:.0}",
        ch256.ext_ms, ch512.ext_ms, ch1024.ext_ms, ch2048.ext_ms, norel.ext_ms,
        par16.mcyc, par16.minstr, par16.mllc, par16.mbr, s3a16.mllc, s3a16.mcyc, s3a16.minstr, ck16.mcyc
    );

    let rows = vec![
        ("0a p20 wall".to_string(), [r(t16.ext_ms), r(t8.ext_ms), r(nofast.ext_ms)], row_a.clone()),
        ("0b p20 stg16".to_string(), [r(par16.wall_ms), r(s3a16.wall_ms), r(ck16.wall_ms)], row_b.clone()),
        ("0c p20 cpu16".to_string(), [r(t16.user_ms), r(t16.sys_ms), r(t16.peak_rss_mb)], row_c.clone()),
        ("0d p20 stg8".to_string(), [r(par8.wall_ms), r(s3a8.wall_ms), r(t8.user_ms)], row_d.clone()),
        ("0e p20 knobs".to_string(), [best_knob_i64, t16.cx_ev / 1000, t16.cx_ck], row_e.clone()),
    ];

    let probe_pq = Path::new(&output).join("user_daily").join("part-zz-probe.parquet");
    write_probe_rows(&probe_pq, &rows);

    eprintln!("probe host {}", if !t16.host.is_empty() { &t16.host } else { &dev_run.host });
    eprintln!("0a p20 wall: {row_a}");
    eprintln!("0b p20 stg16: {row_b}");
    eprintln!("0c p20 cpu16: {row_c}");
    eprintln!("0d p20 stg8: {row_d}");
    eprintln!("0e p20 knobs: {row_e}");
}
