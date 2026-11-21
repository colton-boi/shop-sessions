use std::cell::UnsafeCell;
use std::os::raw::{c_int, c_void};

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
    fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    fn syscall(number: i64, ...) -> i64;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn getrusage(who: c_int, usage: *mut [i64; 18]) -> c_int;
}

/// Process user seconds, system seconds and minor faults.
pub fn rusage() -> (f64, f64, u64) {
    let mut r = [0i64; 18];
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

/// A whole file mapped copy-on-write, with a zero page of slack on both sides
/// and a newline written after the last byte when the file lacks one.
pub fn map_file(file: &std::fs::File, size: usize) -> (*const u8, usize) {
    use std::os::unix::io::AsRawFd;
    const PAGE: usize = 4096;
    if size == 0 {
        return (std::ptr::null(), 0);
    }
    let span = (size + PAGE - 1) / PAGE * PAGE + 2 * PAGE;
    unsafe {
        let base = mmap(std::ptr::null_mut(), span, 3, 0x02 | 0x20 | 0x4000, -1, 0) as *mut u8;
        assert!(base as isize != -1, "mmap reserve failed");
        let data = base.add(PAGE);
        // MAP_PRIVATE | MAP_FIXED over the reserved span.
        let p = mmap(data as *mut c_void, size, 3, 0x02 | 0x10, file.as_raw_fd(), 0);
        assert!(p as *mut u8 == data, "mmap of input failed");
        if *data.add(size - 1) != b'\n' {
            *data.add(size) = b'\n';
            return (data, size + 1);
        }
        (data, size)
    }
}

/// Drops the page tables of a `map_file` mapping of a file of `size` bytes.
/// MADV_DONTNEED needs only the read side of the mmap lock, where munmap would
/// stall every other thread's page faults; the empty mapping then costs
/// nothing at exit.
pub fn unmap_file(data: *const u8, size: usize) {
    const PAGE: usize = 4096;
    if size != 0 {
        unsafe { madvise(data.sub(PAGE) as *mut c_void, (size + PAGE - 1) / PAGE * PAGE + 2 * PAGE, 4) };
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

/// Zero-filled anonymous memory that is only committed where touched.
pub fn map_zeroed(bytes: usize) -> *mut u8 {
    let len = (bytes.max(1) + HUGE - 1) & !(HUGE - 1);
    // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE
    let p = unsafe { mmap(std::ptr::null_mut(), len, 3, 0x02 | 0x20 | 0x4000, -1, 0) };
    if p as isize == -1 {
        panic!("mmap of {len} bytes failed");
    }
    if bytes >= HUGE {
        // MADV_HUGEPAGE; SS_NOHUGE=1 forces 4 KB pages (MADV_NOHUGEPAGE) to model a host without THP.
        unsafe { madvise(p, len, if std::env::var_os("SS_NOHUGE").is_some() { 15 } else { 14 }) };
    }
    p as *mut u8
}

/// Persistent workers for `par`: spawning threads for every stage costs 0.1 to 0.2 ms each time plus wake-up
/// latency, so the workers start once, spin briefly between stages and then sleep on a futex.
mod pool {
    use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering::*};
    use std::sync::OnceLock;

    type Job<'a> = &'a (dyn Fn(usize) + Sync);

    pub struct Pool {
        workers: usize,
        job: AtomicPtr<Job<'static>>,
        /// Generation << 10 | thread count of the current job, in one word so a worker never pairs a count
        /// with the wrong generation.
        epoch: AtomicU32,
        pending: AtomicU32,
    }

    static POOL: OnceLock<Pool> = OnceLock::new();
    const SPIN: u32 = 1 << 14;

    fn futex_wait(word: &AtomicU32, value: u32) {
        unsafe { super::syscall(202, word as *const AtomicU32, 128, value, std::ptr::null::<u8>()) };
    }

    fn futex_wake(word: &AtomicU32) {
        unsafe { super::syscall(202, word as *const AtomicU32, 129, i32::MAX) };
    }

    /// Waits until `word` differs from `value`.
    fn wait_change(word: &AtomicU32, value: u32) {
        for _ in 0..SPIN {
            if word.load(Acquire) != value {
                return;
            }
            std::hint::spin_loop();
        }
        while word.load(Acquire) == value {
            futex_wait(word, value);
        }
    }

    /// Starts `workers` threads, once; later calls keep the first pool.
    pub fn start(workers: usize) -> &'static Pool {
        POOL.get_or_init(|| {
            let pool = Pool {
                workers,
                job: AtomicPtr::new(std::ptr::null_mut()),
                epoch: AtomicU32::new(0),
                pending: AtomicU32::new(0),
            };
            for t in 1..=workers {
                std::thread::Builder::new()
                    .stack_size(2 << 20)
                    .spawn(move || {
                        let pool = POOL.wait();
                        let mut seen = 0;
                        loop {
                            wait_change(&pool.epoch, seen);
                            seen = pool.epoch.load(Acquire);
                            if (t as u32) < seen & 1023 {
                                let job = unsafe { *pool.job.load(Acquire) };
                                job(t);
                                if pool.pending.fetch_sub(1, AcqRel) == 1 {
                                    futex_wake(&pool.pending);
                                }
                            }
                        }
                    })
                    .unwrap();
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
        let job: *const Job = &f;
        pool.job.store(job as *mut Job<'static>, Release);
        pool.pending.store(n as u32 - 1, Release);
        let epoch = pool.epoch.load(Relaxed);
        pool.epoch.store(((epoch >> 10).wrapping_add(1) << 10) | n as u32, Release);
        futex_wake(&pool.epoch);
        f(0);
        loop {
            let left = pool.pending.load(Acquire);
            if left == 0 {
                break;
            }
            wait_change(&pool.pending, left);
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
    std::thread::scope(|s| {
        for t in 1..n {
            let f = &f;
            std::thread::Builder::new().stack_size(1 << 20).spawn_scoped(s, move || f(t)).unwrap();
        }
        f(0);
    });
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
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

    extern "C" {
        fn sigaction(sig: c_int, act: *const [u64; 19], old: *mut [u64; 19]) -> c_int;
        fn setitimer(which: c_int, new: *const [i64; 4], old: *mut [i64; 4]) -> c_int;
    }

    const CAP: usize = 1 << 22;
    static SAMPLES: [AtomicU64; CAP] = [const { AtomicU64::new(0) }; CAP];
    static COUNT: AtomicUsize = AtomicUsize::new(0);

    extern "C" fn on_prof(_sig: c_int, _info: *mut c_void, ctx: *mut c_void) {
        let rip = unsafe { *(ctx as *const u8).add(168).cast::<u64>() };
        let i = COUNT.fetch_add(1, Relaxed);
        if i < CAP {
            SAMPLES[i].store(rip, Relaxed);
        }
    }

    pub fn start() {
        if std::env::var_os("SS_PROF").is_none() {
            return;
        }
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
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let exe = std::fs::read_link("/proc/self/exe").unwrap_or_default();
        let base = maps
            .lines()
            .find(|l| l.ends_with(exe.to_str().unwrap_or("?")))
            .and_then(|l| u64::from_str_radix(l.split('-').next()?, 16).ok())
            .unwrap_or(0);
        let n = COUNT.load(Relaxed).min(CAP);
        let mut out = format!("{base:x}\n");
        for s in &SAMPLES[..n] {
            out.push_str(&format!("{:x}\n", s.load(Relaxed)));
        }
        std::fs::write(path, out).unwrap();
    }
}
