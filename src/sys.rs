use std::cell::UnsafeCell;
use std::os::raw::{c_int, c_void};

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
    #[cfg(target_env = "gnu")]
    fn mallopt(param: c_int, value: c_int) -> c_int;
    fn syscall(number: i64, ...) -> i64;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn getrusage(who: c_int, usage: *mut [i64; 36]) -> c_int;
    fn sched_getaffinity(pid: c_int, size: usize, mask: *mut u8) -> c_int;
    fn sched_setaffinity(pid: c_int, size: usize, mask: *const u8) -> c_int;
}

/// Process user seconds, system seconds and minor faults.
pub fn rusage() -> (f64, f64, u64) {
    let mut r = [0i64; 36];
    unsafe { getrusage(0, &mut r) };
    (r[0] as f64 + r[1] as f64 * 1e-6, r[2] as f64 + r[3] as f64 * 1e-6, r[8] as u64)
}

/// Calling thread's on-CPU and runqueue-wait seconds.
pub fn schedstat() -> (f64, f64) {
    let s = std::fs::read_to_string("/proc/thread-self/schedstat").unwrap_or_default();
    let mut it = s.split_whitespace().map(|v| v.parse::<u64>().unwrap_or(0) as f64 * 1e-9);
    (it.next().unwrap_or(0.0), it.next().unwrap_or(0.0))
}

/// Hardware counters for the stats report: cycles, instructions, cache misses, branch misses.
pub struct Counters([c_int; 4]);

impl Counters {
    pub fn open() -> Self {
        let mut fds = [-1; 4];
        for (fd, config) in fds.iter_mut().zip([0u64, 1, 3, 5]) {
            let mut attr = [0u64; 16];
            attr[0] = 128 << 32; // type PERF_TYPE_HARDWARE, size 128
            attr[1] = config;
            attr[5] = 1 << 1 | 1 << 5; // inherit, exclude_kernel
            *fd = unsafe { syscall(298, attr.as_ptr(), 0, -1, -1, 0) } as c_int;
        }
        Counters(fds)
    }

    pub fn read(&self) -> [u64; 4] {
        let mut out = [0u64; 4];
        for (v, &fd) in out.iter_mut().zip(&self.0) {
            if fd >= 0 {
                unsafe { read(fd, v as *mut u64 as *mut c_void, 8) };
            }
        }
        out
    }
}

const HUGE: usize = 2 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimdTier {
    Avx2,
    ClxAvx512,
    IcxAvx512,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ThpPolicy {
    DefaultHuge,
    AdviseHuge,
    NoHuge,
}

/// What the host offers, probed once at startup in well under 100 us.
#[derive(Clone, Debug)]
pub struct HostProfile {
    pub simd: SimdTier,
    pub thp: ThpPolicy,
    pub thp_enabled: String,
    pub affinity: (usize, String),
    pub file_huge_ok: bool,
    pub fs_magic: i64,
    pub kern_major: u32,
    pub kern_minor: u32,
    pub order9_free: u32,
}

static HOST_PROFILE: std::sync::OnceLock<HostProfile> = std::sync::OnceLock::new();

extern "C" {
    fn statfs(path: *const std::os::raw::c_char, buf: *mut [i64; 16]) -> c_int;
}

fn probe_affinity() -> (usize, String) {
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

fn probe_simd() -> SimdTier {
    use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
    let (l1, l7) = (__cpuid(1), __cpuid_count(7, 0));
    let zmm_os = (l1.ecx >> 27) & 1 != 0 && (unsafe { _xgetbv(0) } & 0xE6) == 0xE6;
    let bit = |reg: u32, b: u32| (reg >> b) & 1 != 0;
    let clx = zmm_os && bit(l7.ebx, 16) && bit(l7.ebx, 17) && bit(l7.ebx, 30) && bit(l7.ebx, 31);
    let icx = clx && bit(l7.ecx, 6) && bit(l7.ecx, 10) && bit(l7.ecx, 12);
    if icx {
        SimdTier::IcxAvx512
    } else if clx {
        SimdTier::ClxAvx512
    } else {
        SimdTier::Avx2
    }
}

/// Free order-9 (2 MiB) pages, or `u32::MAX` when `/proc/buddyinfo` cannot be read.
fn order9_free() -> u32 {
    let Ok(s) = std::fs::read_to_string("/proc/buddyinfo") else { return u32::MAX };
    let mut total = 0u64;
    let mut any = false;
    for line in s.lines() {
        // "Node 0, zone   Normal  a b c d e f g h i j k": order 9 is the 10th count.
        if let Some(v) = line.split_whitespace().skip(4).nth(9).and_then(|v| v.parse::<u64>().ok()) {
            total += v;
            any = true;
        }
    }
    if any {
        total.min(u32::MAX as u64 - 1) as u32
    } else {
        u32::MAX
    }
}

fn probe_host(data_dir: &std::path::Path) -> HostProfile {
    let en = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").unwrap_or_default();
    let thp_enabled = en
        .split('[')
        .nth(1)
        .and_then(|s| s.split(']').next())
        .unwrap_or("-")
        .to_string();
    let o9 = if std::env::var_os("SS_STATS").is_some() {
        order9_free()
    } else {
        u32::MAX
    };
    let thp = if std::env::var_os("SS_NOHUGE").is_some() {
        ThpPolicy::NoHuge
    } else if std::env::var_os("SS_THP").is_some_and(|v| v == "advise") {
        // Models the scorer's thp=madvise on a host whose THP mode is `always`.
        ThpPolicy::AdviseHuge
    } else {
        match thp_enabled.as_str() {
            // Without MADV_HUGEPAGE the kernel allocates huge pages opportunistically and never stalls to compact.
            "always" => ThpPolicy::DefaultHuge,
            // The scorer's mode: MADV_HUGEPAGE is the only way to get huge pages (turning THP off costs +148 ms).
            "madvise" => ThpPolicy::AdviseHuge,
            "never" => ThpPolicy::NoHuge,
            _ => ThpPolicy::AdviseHuge,
        }
    };
    let rel = std::fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
    let mut it = rel.trim().split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse::<u32>().unwrap_or(0));
    let (kern_major, kern_minor) = (it.next().unwrap_or(0), it.next().unwrap_or(0));
    let mut fs_magic = 0i64;
    if let Ok(c) = std::ffi::CString::new(data_dir.as_os_str().as_encoded_bytes()) {
        let mut buf = [0i64; 16];
        if unsafe { statfs(c.as_ptr(), &mut buf) } == 0 {
            fs_magic = buf[0];
        }
    }
    let kern = (kern_major, kern_minor);
    let file_huge_ok = (fs_magic == 0x0102_1994 && kern >= (5, 15))
        || (fs_magic == 0x5846_5342 && kern >= (5, 18))
        || (fs_magic == 0xEF53 && kern >= (6, 8));
    HostProfile {
        simd: probe_simd(),
        thp,
        thp_enabled,
        affinity: probe_affinity(),
        file_huge_ok,
        fs_magic,
        kern_major,
        kern_minor,
        order9_free: o9,
    }
}

pub fn init_host_profile(data_dir: &std::path::Path) -> &'static HostProfile {
    HOST_PROFILE.get_or_init(|| probe_host(data_dir))
}

pub fn host_profile() -> &'static HostProfile {
    HOST_PROFILE.get_or_init(|| probe_host(std::path::Path::new("/data")))
}

#[allow(dead_code)]
pub fn use_avx512() -> bool {
    host_profile().simd != SimdTier::Avx2 && std::env::var_os("SS_FORCE_AVX2").is_none()
}

const PAGE: usize = 4096;

/// A whole file mapped read-only, with a zero page before it (`ts::digits` loads up to 15 bytes
/// backward) and at least one after it. Files of 2 MiB or more start on a 2 MiB boundary so a page
/// cache of huge folios can be mapped with PMDs. Nothing is read here: see `end_line`.
#[allow(dead_code)]
pub fn map_file(file: &std::fs::File, size: usize) -> *const u8 {
    use std::os::unix::io::AsRawFd;
    if size == 0 {
        return std::ptr::null();
    }
    let align = if size >= HUGE { HUGE } else { PAGE };
    let span = PAGE + size.div_ceil(PAGE) * PAGE + PAGE + align;
    unsafe {
        // PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
        let base = mmap(std::ptr::null_mut(), span, 1, 0x02 | 0x20 | 0x4000, -1, 0) as usize;
        assert!(base as isize != -1, "mmap reserve failed");
        let data = (base + PAGE + align - 1) & !(align - 1);
        // MAP_PRIVATE | MAP_FIXED over the reserved span.
        let p = mmap(data as *mut c_void, size, 1, 0x02 | 0x10, file.as_raw_fd(), 0);
        assert!(p as usize == data, "mmap of input failed");
        data as *const u8
    }
}

/// Contiguous 2 MiB-aligned virtual region covering multiple input CSV files.
pub struct ContiguousMappedFile {
    pub ptr: *const u8,
    #[allow(dead_code)]
    pub rel_offset: u64,
    pub size: usize,
}
unsafe impl Send for ContiguousMappedFile {}
unsafe impl Sync for ContiguousMappedFile {}

#[allow(dead_code)]
pub struct ContiguousInputRegion {
    pub reserve_base: *mut u8,
    pub reserve_len: usize,
    pub region_base: *const u8,
    pub total_span: usize,
    pub files: Vec<ContiguousMappedFile>,
}
unsafe impl Send for ContiguousInputRegion {}
unsafe impl Sync for ContiguousInputRegion {}

struct SyncSlots<T>(Vec<UnsafeCell<T>>);
unsafe impl<T: Send> Sync for SyncSlots<T> {}
impl<T> SyncSlots<T> {
    #[allow(clippy::mut_from_ref)]
    unsafe fn slot(&self, i: usize) -> &mut T {
        &mut *self.0[i].get()
    }
}

/// Reserves one anonymous virtual range covering all `files` padded to 2 MiB boundaries
/// (with a leading 2 MiB zero guard) and maps each non-empty file into its 2 MiB-aligned slot
/// using `MAP_SHARED | MAP_FIXED`.
#[allow(dead_code)]
pub fn map_files_contiguous(files: &[std::path::PathBuf]) -> ContiguousInputRegion {
    map_files_contiguous_par(files, 1)
}

/// Parallel version of `map_files_contiguous`: parallelizes `open`+`fstat` and `mmap(MAP_SHARED | MAP_FIXED)`
/// across `threads` workers while preserving exact file ordering and the 2 MiB leading zero guard.
pub fn map_files_contiguous_par(files: &[std::path::PathBuf], threads: usize) -> ContiguousInputRegion {
    use std::os::unix::io::AsRawFd;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    if files.is_empty() {
        return ContiguousInputRegion {
            reserve_base: std::ptr::null_mut(),
            reserve_len: 0,
            region_base: std::ptr::null(),
            total_span: 0,
            files: Vec::new(),
        };
    }
    let n = files.len();
    let par_threads = threads.min(n).max(1);
    let opened: SyncSlots<Option<(std::fs::File, usize)>> =
        SyncSlots((0..n).map(|_| UnsafeCell::new(None)).collect());
    let open_one = |i: usize| {
        let f = std::fs::File::open(&files[i]).expect("open input file");
        let size = f.metadata().expect("stat input file").len() as usize;
        unsafe { *opened.slot(i) = Some((f, size)) };
    };
    if par_threads > 1 {
        let next = AtomicUsize::new(0);
        par(par_threads, |_| loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= n {
                break;
            }
            open_one(i);
        });
    } else {
        for i in 0..n {
            open_one(i);
        }
    }

    let mut offsets = Vec::with_capacity(n);
    let mut rel_offset = 0usize;
    for i in 0..n {
        let size = unsafe { (*opened.slot(i)).as_ref().unwrap().1 };
        offsets.push(rel_offset);
        // Leave at least one trailing zero page after each file and align the next slot to 2 MiB.
        let slot_span = ((size + PAGE).max(1) + HUGE - 1) & !(HUGE - 1);
        rel_offset += slot_span;
    }
    let total_span = rel_offset;
    let reserve_len = HUGE + total_span + HUGE;
    let file_huge_ok = host_profile().file_huge_ok;
    unsafe {
        // PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
        let raw = mmap(std::ptr::null_mut(), reserve_len, 1, 0x02 | 0x20 | 0x4000, -1, 0);
        assert!(raw as isize != -1, "contiguous mmap reserve failed");
        let region_addr = (raw as usize + HUGE) & !(HUGE - 1);
        let mapped: SyncSlots<ContiguousMappedFile> = SyncSlots(
            (0..n)
                .map(|_| {
                    UnsafeCell::new(ContiguousMappedFile {
                        ptr: std::ptr::null(),
                        rel_offset: 0,
                        size: 0,
                    })
                })
                .collect(),
        );
        let map_one = |i: usize| {
            let (f, size) = opened.slot(i).take().unwrap();
            let off = offsets[i];
            let slot_addr = region_addr + off;
            if size > 0 {
                // MAP_SHARED | MAP_FIXED (0x01 | 0x10)
                let p = mmap(slot_addr as *mut c_void, size, 1, 0x01 | 0x10, f.as_raw_fd(), 0);
                assert!(p as usize == slot_addr, "contiguous MAP_FIXED input mmap failed");
                if size >= HUGE && file_huge_ok {
                    madvise(slot_addr as *mut c_void, size & !(HUGE - 1), 14);
                }
            }
            drop(f);
            *mapped.slot(i) = ContiguousMappedFile {
                ptr: slot_addr as *const u8,
                rel_offset: off as u64,
                size,
            };
        };
        if par_threads > 1 {
            let next = AtomicUsize::new(0);
            par(par_threads, |_| loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                map_one(i);
            });
        } else {
            for i in 0..n {
                map_one(i);
            }
        }
        ContiguousInputRegion {
            reserve_base: raw as *mut u8,
            reserve_len,
            region_base: region_addr as *const u8,
            total_span,
            files: mapped.0.into_iter().map(UnsafeCell::into_inner).collect(),
        }
    }
}

/// Thread-safe huge-page-backed bump allocator over a `map_zeroed` virtual arena.
#[allow(dead_code)]
pub struct HugeBump {
    base: *mut u8,
    cap: usize,
    next: std::sync::atomic::AtomicUsize,
}
unsafe impl Send for HugeBump {}
unsafe impl Sync for HugeBump {}

#[allow(dead_code)]
impl HugeBump {
    pub fn new(cap_bytes: usize) -> Self {
        let cap = (cap_bytes.max(1) + HUGE - 1) & !(HUGE - 1);
        HugeBump {
            base: map_zeroed(cap),
            cap,
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    pub fn alloc_aligned(&self, bytes: usize, align: usize) -> *mut u8 {
        debug_assert!(align.is_power_of_two());
        let mask = align - 1;
        let mut cur = self.next.load(std::sync::atomic::Ordering::Relaxed);
        loop {
            let aligned = (cur + mask) & !mask;
            let end = aligned + bytes;
            assert!(end <= self.cap, "HugeBump arena overflow");
            match self.next.compare_exchange_weak(
                cur,
                end,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => return unsafe { self.base.add(aligned) },
                Err(actual) => cur = actual,
            }
        }
    }

    pub fn used_bytes(&self) -> usize {
        self.next.load(std::sync::atomic::Ordering::Relaxed).min(self.cap)
    }

    pub fn release_used(&self) {
        let used = (self.used_bytes() + PAGE - 1) & !(PAGE - 1);
        if used > 0 {
            release(self.base, used);
        }
    }
}

/// The length of a `map_file` mapping once it ends with a newline: when the file lacks one, a
/// newline is written after its last byte (into a private anonymous copy of that tail page).
pub unsafe fn end_line(data: *const u8, size: usize) -> usize {
    if size == 0 || *data.add(size - 1) == b'\n' {
        return size;
    }
    let at = data.add(size) as usize;
    let page_base = at & !(PAGE - 1);
    let in_page = at - page_base;
    if in_page == 0 {
        // `at` is on the anonymous zero page immediately following the file mapping.
        // PROT_READ | PROT_WRITE
        assert!(mprotect(page_base as *mut c_void, PAGE, 3) == 0, "mprotect failed");
        *(at as *mut u8) = b'\n';
    } else {
        // Remap only the last partial page as MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED so this
        // works safely on read-only MAP_SHARED file mappings without SIGSEGV/EACCES.
        let mut tmp = [0u8; PAGE];
        std::ptr::copy_nonoverlapping(page_base as *const u8, tmp.as_mut_ptr(), in_page);
        tmp[in_page] = b'\n';
        let p = mmap(page_base as *mut c_void, PAGE, 3, 0x02 | 0x20 | 0x10, -1, 0);
        assert!(p as usize == page_base, "end_line anonymous remap failed");
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), page_base as *mut u8, in_page + 1);
    }
    size + 1
}

/// Drops the page tables of the first `size` bytes of a `map_file` mapping.
/// MADV_DONTNEED needs only the read side of the mmap lock, where munmap would
/// stall every other thread's page faults; the empty mapping then costs
/// nothing at exit.
pub fn unmap_file(data: *const u8, size: usize) {
    if size != 0 {
        unsafe { madvise(data as *mut c_void, size.div_ceil(PAGE) * PAGE, 4) };
    }
}

/// Commits `bytes` at `p` up front (MADV_POPULATE_WRITE), so the first writes take no faults.
pub fn populate(p: *mut u8, bytes: usize) {
    if unsafe { madvise(p as *mut c_void, bytes, 23) } != 0 {
        let mut at = 0;
        while at < bytes {
            unsafe { std::ptr::write_volatile(p.add(at), 0) };
            at += 4096;
        }
    }
}

/// Frees the pages of a dead `map_zeroed` region of `bytes` bytes now, so exit has nothing left to tear down.
pub fn release(p: *mut u8, bytes: usize) {
    unsafe { madvise(p as *mut c_void, bytes, 4) };
}

#[inline]
fn apply_thp_advice(ptr: *mut u8, len: usize) {
    match host_profile().thp {
        ThpPolicy::AdviseHuge => unsafe { madvise(ptr as *mut c_void, len, 14) },
        ThpPolicy::NoHuge => unsafe { madvise(ptr as *mut c_void, len, 15) },
        ThpPolicy::DefaultHuge => 0,
    };
}

/// Zero-filled anonymous memory that is only committed where touched.
pub fn map_zeroed(bytes: usize) -> *mut u8 {
    let len = (bytes.max(1) + HUGE - 1) & !(HUGE - 1);
    // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE
    let p = unsafe { mmap(std::ptr::null_mut(), len, 3, 0x02 | 0x20 | 0x4000, -1, 0) };
    if p as isize == -1 {
        panic!("mmap of {len} bytes failed");
    }
    if bytes >= HUGE {
        apply_thp_advice(p as *mut u8, len);
    }
    p as *mut u8
}

/// Zero-filled 2 MiB-aligned anonymous hugepage arena (`MADV_HUGEPAGE` when supported).
pub fn map_zeroed_huge(bytes: usize) -> *mut u8 {
    let len = (bytes.max(1) + HUGE - 1) & !(HUGE - 1);
    // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE
    let p = unsafe { mmap(std::ptr::null_mut(), len, 3, 0x02 | 0x20 | 0x4000, -1, 0) };
    if p as isize == -1 {
        panic!("mmap of {len} bytes failed");
    }
    let aligned = if (p as usize) & (HUGE - 1) == 0 {
        p as *mut u8
    } else {
        unsafe { munmap(p, len) };
        let raw = unsafe { mmap(std::ptr::null_mut(), len + HUGE, 3, 0x02 | 0x20 | 0x4000, -1, 0) };
        assert!(raw as isize != -1, "mmap of {} bytes failed", len + HUGE);
        let aligned_addr = (raw as usize + HUGE - 1) & !(HUGE - 1);
        let head_slack = aligned_addr - raw as usize;
        if head_slack > 0 {
            unsafe { munmap(raw, head_slack) };
        }
        let tail_slack = (raw as usize + len + HUGE) - (aligned_addr + len);
        if tail_slack > 0 {
            unsafe { munmap((aligned_addr + len) as *mut c_void, tail_slack) };
        }
        aligned_addr as *mut u8
    };
    apply_thp_advice(aligned, len);
    aligned
}

fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                for c in lo..=hi.min(1023) {
                    out.push(c);
                }
            }
        } else if let Ok(c) = part.parse::<usize>() {
            if c < 1024 {
                out.push(c);
            }
        }
    }
    out
}

/// Allowed CPUs ordered so SMT siblings on the same physical core are paired consecutively:
/// `(core0_t0, core0_t1, core1_t0, core1_t1, ...)`.
pub fn cpu_order() -> Vec<usize> {
    let mut mask = [0u8; 128];
    let mut allowed = Vec::new();
    if unsafe { sched_getaffinity(0, mask.len(), mask.as_mut_ptr()) } == 0 {
        for cpu in 0..1024 {
            if (mask[cpu / 8] >> (cpu % 8)) & 1 != 0 {
                allowed.push(cpu);
            }
        }
    }
    if allowed.is_empty() {
        let n = std::thread::available_parallelism().map_or(8, |n| n.get()).min(1024);
        return (0..n).collect();
    }
    let mut in_mask = [false; 1024];
    for &c in &allowed {
        in_mask[c] = true;
    }
    let mut placed = [false; 1024];
    let mut ordered = Vec::with_capacity(allowed.len());
    for &cpu in &allowed {
        if placed[cpu] {
            continue;
        }
        placed[cpu] = true;
        ordered.push(cpu);
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        if let Ok(raw) = std::fs::read_to_string(path) {
            let mut sibs = parse_cpu_list(&raw);
            sibs.sort_unstable();
            sibs.dedup();
            for s in sibs {
                if s < 1024 && in_mask[s] && !placed[s] {
                    placed[s] = true;
                    ordered.push(s);
                }
            }
        }
    }
    ordered
}

fn pin_to_cpu(cpu: usize) {
    if cpu < 1024 {
        let mut mask = [0u8; 128];
        mask[cpu / 8] |= 1u8 << (cpu % 8);
        unsafe { sched_setaffinity(0, mask.len(), mask.as_ptr()) };
    }
}

/// Persistent workers for `par`: spawning threads for every stage costs 0.1 to 0.2 ms each time plus wake-up
/// latency, so the workers start once in a binary tree, spin briefly between stages and then sleep on a futex.
mod pool {
    use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering::*};
    use std::sync::OnceLock;

    type Job<'a> = &'a (dyn Fn(usize) + Sync);

    pub struct Pool {
        workers: usize,
        job: AtomicPtr<Job<'static>>,
        /// Generation << 10 | thread count of the current job, in one word so a worker never pairs a count
        /// with the wrong generation.
        epoch: AtomicU32,
        pending: AtomicU32,
        /// Threads inside `futex_wait` or about to enter it; a wake is a syscall, skipped when none.
        sleepers: AtomicU32,
        spin: u32,
        pin_caller: AtomicBool,
    }

    static POOL: OnceLock<Pool> = OnceLock::new();
    static CPU_ORDER: OnceLock<Vec<usize>> = OnceLock::new();

    pub fn cached_cpu_order() -> &'static [usize] {
        CPU_ORDER.get_or_init(super::cpu_order)
    }

    fn futex_wait(word: &AtomicU32, value: u32) {
        unsafe { super::syscall(202, word as *const AtomicU32, 128, value, std::ptr::null::<u8>()) };
    }

    /// Wakes the waiters on `word`, which the caller has just changed with a SeqCst store or RMW.
    fn futex_wake(pool: &Pool, word: &AtomicU32) {
        // Pairs with the SeqCst increment in `wait_change`: either the waiter sees the new value
        // before sleeping, or this load sees the waiter.
        if pool.sleepers.load(SeqCst) != 0 {
            unsafe { super::syscall(202, word as *const AtomicU32, 129, i32::MAX) };
        }
    }

    /// Waits until `word` differs from `value`.
    fn wait_change(pool: &Pool, word: &AtomicU32, value: u32) {
        for _ in 0..pool.spin {
            if word.load(Acquire) != value {
                return;
            }
            std::hint::spin_loop();
        }
        while word.load(Acquire) == value {
            pool.sleepers.fetch_add(1, SeqCst);
            if word.load(SeqCst) == value {
                futex_wait(word, value);
            }
            pool.sleepers.fetch_sub(1, SeqCst);
        }
    }

    /// PAUSE takes about 140 cycles on Intel since Skylake and about 65 on Zen, so the spin before
    /// sleeping is about the same time on both (~0.1 ms at 3 GHz).
    fn spin_count() -> u32 {
        let v = std::arch::x86_64::__cpuid(0);
        let intel = (v.ebx, v.edx, v.ecx) == (0x756e_6547, 0x4965_6e69, 0x6c65_746e);
        if intel { 2048 } else { 1 << 14 }
    }

    fn spawn_worker(t: usize, total: usize, pin: bool) {
        std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || {
                let c1 = 2 * t + 1;
                if c1 < total {
                    spawn_worker(c1, total, pin);
                }
                let c2 = 2 * t + 2;
                if c2 < total {
                    spawn_worker(c2, total, pin);
                }
                if pin {
                    if let Some(&cpu) = cached_cpu_order().get(t) {
                        super::pin_to_cpu(cpu);
                    }
                }
                let pool = POOL.wait();
                let mut seen = 0;
                loop {
                    wait_change(pool, &pool.epoch, seen);
                    seen = pool.epoch.load(Acquire);
                    if (t as u32) < seen & 1023 {
                        let job = unsafe { *pool.job.load(Acquire) };
                        job(t);
                        if pool.pending.fetch_sub(1, SeqCst) == 1 {
                            futex_wake(pool, &pool.pending);
                        }
                    }
                }
            })
            .unwrap();
    }

    /// Starts `workers` threads in a binary tree, once; later calls keep the first pool.
    pub fn start(workers: usize) -> &'static Pool {
        POOL.get_or_init(|| {
            let pin = std::env::var_os("SS_NOPIN").is_none();
            let has_cpus = pin && !cached_cpu_order().is_empty();
            let pool = Pool {
                workers,
                job: AtomicPtr::new(std::ptr::null_mut()),
                epoch: AtomicU32::new(0),
                pending: AtomicU32::new(0),
                sleepers: AtomicU32::new(0),
                spin: spin_count(),
                pin_caller: AtomicBool::new(has_cpus),
            };
            let total = workers + 1;
            if 1 < total {
                spawn_worker(1, total, pin);
            }
            if 2 < total {
                spawn_worker(2, total, pin);
            }
            pool
        })
    }

    /// Runs `f(0..n)` with `f(0)` on the caller; false when the pool is too small.
    pub fn run(n: usize, f: Job) -> bool {
        let Some(pool) = POOL.get() else { return false };
        if n - 1 > pool.workers || n > 1023 {
            return false;
        }
        if pool.pin_caller.swap(false, Relaxed) {
            if let Some(&cpu) = cached_cpu_order().first() {
                super::pin_to_cpu(cpu);
            }
        }
        let job: *const Job = &f;
        pool.job.store(job as *mut Job<'static>, Release);
        pool.pending.store(n as u32 - 1, Release);
        let epoch = pool.epoch.load(Relaxed);
        pool.epoch.store(((epoch >> 10).wrapping_add(1) << 10) | n as u32, SeqCst);
        futex_wake(pool, &pool.epoch);
        f(0);
        loop {
            let left = pool.pending.load(Acquire);
            if left == 0 {
                break;
            }
            wait_change(pool, &pool.pending, left);
        }
        true
    }
}

/// Starts the persistent workers that `par` uses for up to `threads` threads.
pub fn start_pool(threads: usize) {
    if threads > 1 && std::env::var_os("SS_NOPOOL").is_none() {
        pool::start(threads - 1);
    }
}

pub fn par<F: Fn(usize) + Sync>(n: usize, f: F) {
    if n == 1 {
        f(0);
        return;
    }
    if pool::run(n, &f) {
        return;
    }
    let pin = std::env::var_os("SS_NOPIN").is_none();
    let cpus = pin.then(|| pool::cached_cpu_order());
    std::thread::scope(|s| {
        for t in 1..n {
            let f = &f;
            let cpu = cpus.and_then(|c| c.get(t).copied());
            std::thread::Builder::new()
                .stack_size(1 << 20)
                .spawn_scoped(s, move || {
                    if let Some(c) = cpu {
                        pin_to_cpu(c);
                    }
                    f(t);
                })
                .unwrap();
        }
        f(0);
    });
}

struct SyncCell<T>(UnsafeCell<T>);
unsafe impl<T: Send> Sync for SyncCell<T> {}
impl<T> SyncCell<T> {
    #[inline(always)]
    fn get(&self) -> *mut T {
        self.0.get()
    }
}

/// Runs two heterogeneous setup closures in parallel on the thread pool (must not nest `par`).
#[allow(dead_code)]
pub fn join2<A, B, RA, RB>(threads: usize, a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    if threads <= 1 {
        return (a(), b());
    }
    let fa = SyncCell(UnsafeCell::new(Some(a)));
    let fb = SyncCell(UnsafeCell::new(Some(b)));
    let ra: SyncCell<Option<RA>> = SyncCell(UnsafeCell::new(None));
    let rb: SyncCell<Option<RB>> = SyncCell(UnsafeCell::new(None));
    par(2, |t| unsafe {
        if t == 0 {
            *ra.get() = Some(((*fa.get()).take().unwrap())());
        } else {
            *rb.get() = Some(((*fb.get()).take().unwrap())());
        }
    });
    (ra.0.into_inner().unwrap(), rb.0.into_inner().unwrap())
}

/// Runs three heterogeneous setup closures in parallel on the thread pool (must not nest `par`).
#[allow(dead_code)]
pub fn join3<A, B, C, RA, RB, RC>(threads: usize, a: A, b: B, c: C) -> (RA, RB, RC)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    C: FnOnce() -> RC + Send,
    RA: Send,
    RB: Send,
    RC: Send,
{
    if threads <= 1 {
        return (a(), b(), c());
    }
    let fa = SyncCell(UnsafeCell::new(Some(a)));
    let fb = SyncCell(UnsafeCell::new(Some(b)));
    let fc = SyncCell(UnsafeCell::new(Some(c)));
    let ra: SyncCell<Option<RA>> = SyncCell(UnsafeCell::new(None));
    let rb: SyncCell<Option<RB>> = SyncCell(UnsafeCell::new(None));
    let rc: SyncCell<Option<RC>> = SyncCell(UnsafeCell::new(None));
    let next = std::sync::atomic::AtomicUsize::new(0);
    par(threads.min(3), |_| loop {
        match next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) {
            0 => unsafe { *ra.get() = Some(((*fa.get()).take().unwrap())()) },
            1 => unsafe { *rb.get() = Some(((*fb.get()).take().unwrap())()) },
            2 => unsafe { *rc.get() = Some(((*fc.get()).take().unwrap())()) },
            _ => break,
        }
    });
    (
        ra.0.into_inner().unwrap(),
        rb.0.into_inner().unwrap(),
        rc.0.into_inner().unwrap(),
    )
}

/// One slot per worker thread; each worker only ever touches its own slot.
pub struct PerThread<T>(Vec<UnsafeCell<T>>);
unsafe impl<T> Sync for PerThread<T> {}
impl<T> PerThread<T> {
    pub fn new(n: usize, mut f: impl FnMut(usize) -> T) -> Self {
        PerThread((0..n).map(|i| UnsafeCell::new(f(i))).collect())
    }
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self, i: usize) -> &mut T {
        &mut *self.0[i].get()
    }
}

#[inline(always)]
pub fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let x = (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

/// Large buffers come from the heap and stay there when freed, instead of a fresh mmap (faults,
/// zeroing) and a munmap (TLB shootdown) each time a growing Vec moves.
pub fn tune_malloc() {
    #[cfg(target_env = "gnu")]
    {
        if std::env::var_os("SS_MALLOC_DEFAULT").is_some() {
            return;
        }
        unsafe {
            mallopt(-3, 32 << 20); // M_MMAP_THRESHOLD, at its maximum
            mallopt(-1, -1); // M_TRIM_THRESHOLD: never trim
            mallopt(-2, 16 << 20); // M_TOP_PAD
        }
    }
}

/// Ends the process at once: every output file is written and closed by now, and nothing else
/// needs flushing, so libc's exit handlers are skipped.
pub fn exit_now() -> ! {
    // A PGO training build writes its profile from an exit handler.
    if cfg!(pgo_gen) {
        std::process::exit(0);
    }
    unsafe { syscall(231, 0) };
    unreachable!()
}

pub fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

/// SIGPROF sampler for `SS_PROF=<file>`: records the interrupted user RIP of
/// whichever thread is running, so samples follow CPU time.
pub mod prof {
    use std::alloc::{alloc_zeroed, Layout};
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};

    extern "C" {
        fn sigaction(sig: c_int, act: *const [u64; 19], old: *mut [u64; 19]) -> c_int;
        fn setitimer(which: c_int, new: *const [i64; 4], old: *mut [i64; 4]) -> c_int;
    }

    const CAP: usize = 1 << 22;
    static SAMPLES: AtomicPtr<AtomicU64> = AtomicPtr::new(std::ptr::null_mut());
    static COUNT: AtomicUsize = AtomicUsize::new(0);

    extern "C" fn on_prof(_sig: c_int, _info: *mut c_void, ctx: *mut c_void) {
        let p = SAMPLES.load(Acquire);
        if p.is_null() {
            return;
        }
        let rip = unsafe { *(ctx as *const u8).add(168).cast::<u64>() };
        let i = COUNT.fetch_add(1, Relaxed);
        if i < CAP {
            unsafe { (*p.add(i)).store(rip, Relaxed) };
        }
    }

    pub fn start() {
        if std::env::var_os("SS_PROF").is_none() {
            return;
        }
        let p = unsafe { alloc_zeroed(Layout::array::<AtomicU64>(CAP).unwrap()) as *mut AtomicU64 };
        assert!(!p.is_null(), "SS_PROF buffer alloc failed");
        SAMPLES.store(p, Release);
        let mut act = [0u64; 19];
        act[0] = on_prof as *const () as u64;
        act[17] = 0x4 | 0x1000_0000; // SA_SIGINFO | SA_RESTART
        unsafe {
            sigaction(27, &act, std::ptr::null_mut());
            setitimer(2, &[0, 200, 0, 200], std::ptr::null_mut());
        }
    }

    pub fn finish() {
        let Some(path) = std::env::var_os("SS_PROF") else { return };
        unsafe { setitimer(2, &[0; 4], std::ptr::null_mut()) };
        let p = SAMPLES.load(Acquire);
        if p.is_null() {
            return;
        }
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let exe = std::fs::read_link("/proc/self/exe").unwrap_or_default();
        let base = maps
            .lines()
            .find(|l| l.ends_with(exe.to_str().unwrap_or("?")))
            .and_then(|l| u64::from_str_radix(l.split('-').next()?, 16).ok())
            .unwrap_or(0);
        let n = COUNT.load(Relaxed).min(CAP);
        let mut out = format!("{base:x}\n");
        let samples = unsafe { std::slice::from_raw_parts(p, n) };
        for s in samples {
            out.push_str(&format!("{:x}\n", s.load(Relaxed)));
        }
        std::fs::write(path, out).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpu_list_and_cpu_order() {
        assert_eq!(parse_cpu_list("0-1"), vec![0, 1]);
        assert_eq!(parse_cpu_list("0,8"), vec![0, 8]);
        assert_eq!(parse_cpu_list("0-1, 8-9, 12"), vec![0, 1, 8, 9, 12]);

        let order = cpu_order();
        assert!(!order.is_empty());
        let mut uniq = order.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), order.len(), "cpu_order must not repeat CPUs");

        // Verify consecutive SMT sibling pairing when siblings are both in the affinity mask.
        if let Ok(raw) = std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
            order[0]
        )) {
            let sibs: Vec<usize> = parse_cpu_list(&raw)
                .into_iter()
                .filter(|c| order.contains(c))
                .collect();
            if sibs.len() >= 2 {
                assert!(sibs.contains(&order[1]), "expected sibling paired at index 1: {order:?}");
            }
        }
    }

    #[test]
    fn test_map_zeroed_huge_and_contiguous_padding() {
        let p = map_zeroed_huge(4096);
        assert_eq!((p as usize) & (HUGE - 1), 0, "map_zeroed_huge must be 2 MiB aligned");
        unsafe {
            assert_eq!(*p, 0);
            *p = 0xAB;
            assert_eq!(*p, 0xAB);
        }
        release(p, HUGE);

        let dir = std::env::temp_dir().join(format!("ss_sys_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p1 = dir.join("exact_page.csv");
        let p2 = dir.join("empty.csv");
        let p3 = dir.join("tail.csv");
        let exact = vec![b'x'; PAGE];
        std::fs::write(&p1, &exact).unwrap();
        std::fs::write(&p2, b"").unwrap();
        std::fs::write(&p3, b"a,b\n1,2").unwrap();

        let region = map_files_contiguous_par(&[p1.clone(), p2.clone(), p3.clone()], 4);
        assert_eq!(region.files.len(), 3);
        for f in &region.files {
            assert_eq!((f.ptr as usize) & (HUGE - 1), 0);
            // At least 64 bytes of readable zero padding after every file for SIMD over-reads.
            let pad = unsafe { std::slice::from_raw_parts(f.ptr.add(f.size), 64) };
            assert!(pad.iter().all(|&b| b == 0));
            // Verify C1: leading zero guard page before mapped file.
            let lead = unsafe { std::slice::from_raw_parts(f.ptr.sub(PAGE), PAGE) };
            assert!(lead.iter().all(|&b| b == 0));
        }
        unsafe {
            assert_eq!(end_line(region.files[0].ptr, region.files[0].size), PAGE + 1);
            assert_eq!(*region.files[0].ptr.add(PAGE), b'\n');
            assert_eq!(end_line(region.files[1].ptr, region.files[1].size), 0);
            assert_eq!(end_line(region.files[2].ptr, region.files[2].size), 8);
            let s3 = std::slice::from_raw_parts(region.files[2].ptr, 8);
            assert_eq!(s3, b"a,b\n1,2\n");
            let pad3 = std::slice::from_raw_parts(region.files[2].ptr.add(8), 64);
            assert!(pad3.iter().all(|&b| b == 0));
        }
        assert_eq!(std::fs::read(&p3).unwrap(), b"a,b\n1,2", "MAP_SHARED end_line must not modify disk");
        release(region.reserve_base, region.reserve_len);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_join_helpers() {
        assert_eq!(join2(1, || 10, || "ok"), (10, "ok"));
        assert_eq!(join2(2, || 21, || 21 * 2), (21, 42));
        assert_eq!(join3(1, || 1, || 2, || 3), (1, 2, 3));
        assert_eq!(join3(2, || 10, || 20, || 30), (10, 20, 30));
        assert_eq!(join3(4, || "a", || "b", || "c"), ("a", "b", "c"));
    }
}
