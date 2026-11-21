//! Always-on run log on stderr: one host line at startup, one line per stage,
//! and a summary line last. Scorer logs often keep only the tail, so the last
//! line repeats every stage's wall time. Everything here costs well under 1 ms.

use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
use std::io::Write;
use std::os::raw::c_int;

extern "C" {
    fn sched_getaffinity(pid: c_int, size: usize, mask: *mut u8) -> c_int;
    fn sysconf(name: c_int) -> i64;
}

fn cpu_brand() -> String {
    let mut b = Vec::with_capacity(48);
    if __cpuid(0x8000_0000).eax < 0x8000_0004 {
        return "?".into();
    }
    for leaf in 0x8000_0002..=0x8000_0004u32 {
        let r = __cpuid(leaf);
        for v in [r.eax, r.ebx, r.ecx, r.edx] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    let s = String::from_utf8_lossy(&b).trim_matches(|c: char| c == '\0' || c == ' ').to_string();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// AVX-512 subsets the CPU reports, and whether the OS saves ZMM state.
fn avx512() -> String {
    let (l7, l1) = (__cpuid_count(7, 0), __cpuid(1));
    let mut have: Vec<&str> = Vec::new();
    for (bit, reg, name) in [(16, l7.ebx, "f"), (30, l7.ebx, "bw"), (31, l7.ebx, "vl"), (1, l7.ecx, "vbmi"), (6, l7.ecx, "vbmi2")] {
        if reg >> bit & 1 != 0 {
            have.push(name);
        }
    }
    if have.is_empty() {
        return "none".into();
    }
    let osxsave = l1.ecx >> 27 & 1 != 0;
    let zmm = osxsave && unsafe { _xgetbv(0) } & 0xE6 == 0xE6;
    format!("{}{}", have.join(","), if zmm { "" } else { "(os-off)" })
}

fn affinity() -> (usize, String) {
    let mut mask = [0u8; 128];
    if unsafe { sched_getaffinity(0, mask.len(), mask.as_mut_ptr()) } != 0 {
        return (0, "?".into());
    }
    let on = |i: usize| mask[i / 8] >> (i % 8) & 1 != 0;
    let (mut ranges, mut count, mut i) = (Vec::new(), 0, 0);
    while i < 1024 {
        if !on(i) {
            i += 1;
            continue;
        }
        let from = i;
        while i < 1024 && on(i) {
            i += 1;
        }
        count += i - from;
        ranges.push(if i - from == 1 { format!("{from}") } else { format!("{from}-{}", i - 1) });
    }
    (count, ranges.join(","))
}

fn read_trim(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// cgroup v2 limits of this process's group, falling back to the namespace root.
fn cgroup(file: &str) -> String {
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

fn thp(file: &str) -> String {
    read_trim(&format!("/sys/kernel/mm/transparent_hugepage/{file}"))
        .and_then(|s| Some(s.split('[').nth(1)?.split(']').next()?.to_string()))
        .unwrap_or_else(|| "-".into())
}

fn emit(line: String) {
    let _ = std::io::stderr().write_all(line.as_bytes());
}

pub fn host(threads: usize) {
    let (n_aff, aff) = affinity();
    let online = unsafe { sysconf(84) }; // _SC_NPROCESSORS_ONLN
    let (cpu_max, mem_max) = (cgroup("cpu.max"), cgroup("memory.max"));
    let mem = mem_max.parse::<u64>().map_or(mem_max.clone(), |b| format!("{}M", b >> 20));
    let hp = crate::sys::host_profile();
    emit(format!(
        "ss host cpu=\"{}\" avx512={} nproc={online} affinity={aff}({n_aff}) threads={threads} cpu.max={} memory.max={mem} thp={}/{} kern={}.{} fs={:#x} o9={} simd={:?} thp_pol={:?} fhuge={}\n",
        cpu_brand(),
        avx512(),
        cpu_max.replace(' ', "/"),
        thp("enabled"),
        thp("defrag"),
        hp.kern_major,
        hp.kern_minor,
        hp.fs_magic,
        hp.order9_free,
        hp.simd,
        hp.thp,
        hp.file_huge_ok,
    ));
}

/// Wall and CPU of one finished stage, from the previous mark to this one.
pub fn stage(name: &str, wall_s: f64, user_s: f64, sys_s: f64) {
    emit(format!("ss stage {name:<11} {:7.1} ms  cpu {:.0}+{:.0} ms\n", wall_s * 1e3, user_s * 1e3, sys_s * 1e3));
}

pub fn summary(total_s: f64, stages: &[(&str, f64)], user_s: f64, sys_s: f64, rows: [u64; 3]) {
    let parts: Vec<String> = stages.iter().map(|(n, s)| format!("{}={:.0}", n.replace(' ', "_"), s * 1e3)).collect();
    emit(format!(
        "ss done total {:.1} ms  peak_rss {} MB  cpu {:.0}+{:.0} ms  events {} valid {} clicks {}  stages {}\n",
        total_s * 1e3,
        crate::sys::peak_rss_kb() / 1024,
        user_s * 1e3,
        sys_s * 1e3,
        rows[0],
        rows[1],
        rows[2],
        parts.join(" ")
    ));
}
