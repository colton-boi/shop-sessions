use crate::csv::Reader;
use std::arch::x86_64::*;
use std::os::unix::io::RawFd;
use std::sync::OnceLock;

const DEFAULT_PFD: usize = 1024;
const PREAD_LEAD: usize = 4096;
const PREAD_TRAIL: usize = 64;
const PREAD_BUF_BYTES: usize = 132 << 10;
const PREAD_CHUNK: usize = PREAD_BUF_BYTES - PREAD_LEAD - PREAD_TRAIL;

#[inline]
pub fn use_pread() -> bool {
    static PREAD: OnceLock<bool> = OnceLock::new();
    *PREAD.get_or_init(|| std::env::var("SS_PREAD").ok().as_deref() == Some("1"))
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Align64([u8; 64]);

/// Reusable per-worker 132 KiB 64B-aligned buffer with 4 KiB leading zeros and >= 64 B trailing zeros.
pub struct PreadBuf {
    buf: Vec<Align64>,
}

impl PreadBuf {
    pub fn new() -> Self {
        Self {
            buf: vec![Align64([0u8; 64]); PREAD_BUF_BYTES / 64],
        }
    }

    #[inline(always)]
    fn byte_len(&self) -> usize {
        self.buf.len() * 64
    }

    #[inline(always)]
    fn data_cap(&self) -> usize {
        self.byte_len() - PREAD_LEAD - PREAD_TRAIL
    }

    #[inline(always)]
    fn data_ptr(&mut self) -> *mut u8 {
        unsafe { (self.buf.as_mut_ptr() as *mut u8).add(PREAD_LEAD) }
    }

    #[cold]
    #[inline(never)]
    fn grow_for(&mut self, min_data: usize) {
        let need_bytes = (PREAD_LEAD + min_data + PREAD_TRAIL).next_power_of_two();
        let need_blocks = need_bytes.div_ceil(64);
        if need_blocks > self.buf.len() {
            self.buf.resize(need_blocks, Align64([0u8; 64]));
        }
    }
}

pub trait CsvRow {
    unsafe fn field(&self, k: usize) -> (*const u8, usize);
}

#[derive(Clone, Copy)]
pub struct FastRow {
    pub bptr: *const u8,
    pub seps: *const u16,
    pub q_word: u64,
}

impl CsvRow for FastRow {
    #[inline(always)]
    unsafe fn field(&self, k: usize) -> (*const u8, usize) {
        let q = ((self.q_word >> k) & 1) as usize;
        let s = (*self.seps.add(k)).wrapping_add(1) as usize + q;
        let e = (*self.seps.add(k + 1)) as usize - q;
        (self.bptr.add(s), e - s)
    }
}

pub struct FallbackRow {
    pub buf: *const u8,
    pub s: [u32; 8],
    pub e: [u32; 8],
}

impl CsvRow for FallbackRow {
    #[inline(always)]
    unsafe fn field(&self, k: usize) -> (*const u8, usize) {
        let sk = *self.s.get_unchecked(k) as usize;
        let ek = *self.e.get_unchecked(k) as usize;
        (self.buf.add(sk), ek - sk)
    }
}

pub trait RowHandler {
    unsafe fn on_row<R: CsvRow>(&mut self, row: R);
    #[inline(always)]
    unsafe fn before_refill(&mut self) {}
}

#[inline]
pub fn prefetch_distance() -> usize {
    static PFD: OnceLock<usize> = OnceLock::new();
    *PFD.get_or_init(|| {
        std::env::var("SS_PFD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_PFD)
    })
}

#[repr(C, align(64))]
pub struct BlockIdx {
    pub seps: [u16; 4160],
    pub nl_bits: [u64; 66],
    pub q_bits: [u64; 66],
}

impl BlockIdx {
    #[allow(dead_code)]
    #[inline]
    pub fn new() -> Self {
        Self {
            seps: [0; 4160],
            nl_bits: [0; 66],
            q_bits: [0; 66],
        }
    }
}

#[inline(always)]
unsafe fn find_nl(data: *const u8, mut pos: usize, size: usize) -> usize {
    let nl = _mm256_set1_epi8(b'\n' as i8);
    while pos + 32 <= size {
        let v = _mm256_loadu_si256(data.add(pos) as *const __m256i);
        let m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, nl)) as u32;
        if m != 0 {
            return pos + m.trailing_zeros() as usize;
        }
        pos += 32;
    }
    while pos < size {
        if *data.add(pos) == b'\n' {
            return pos;
        }
        pos += 1;
    }
    size
}

#[cold]
#[inline(never)]
unsafe fn dense_seps(dst: *mut u16, base: u16, mut rem: u64) {
    let mut k = 8usize;
    while rem != 0 {
        *dst.add(k) = base + rem.trailing_zeros() as u16;
        rem &= rem.wrapping_sub(1);
        k += 1;
    }
}

#[inline(always)]
unsafe fn step_block(
    idx: &mut BlockIdx,
    bi: usize,
    cm: u64,
    nm: u64,
    qm: u64,
    total_seps: &mut usize,
    nl_acc: &mut u64,
    q_acc: &mut u64,
    widx: &mut usize,
    bpos: &mut u32,
    s_carry: &mut u64,
    rq_carry: &mut u64,
    inq_mask: &mut u64,
    any_bad: &mut u64,
) {
    let s = cm | nm;
    let fs = (s << 1) | *s_carry;
    *s_carry = s >> 63;

    let rq = qm & !fs;
    let cq = (rq << 1) | *rq_carry;
    *rq_carry = rq >> 63;

    let ones = _mm_set1_epi8(-1);
    let px_v = _mm_clmulepi64_si128::<0x00>(_mm_cvtsi64_si128(qm as i64), ones);
    let px = (_mm_cvtsi128_si64(px_v) as u64) ^ *inq_mask;
    *inq_mask = ((px as i64) >> 63) as u64;

    *any_bad |= (cq & !s) | (px & s);

    let pop = s.count_ones() as u32;
    let nl_pext = _pext_u64(nm, s);
    let q_pext = _pext_u64(cq, s);

    let bp = *bpos;
    *nl_acc |= nl_pext << bp;
    *q_acc |= q_pext << bp;
    let next_bp = bp + pop;
    if next_bp >= 64 {
        let wi = *widx;
        *idx.nl_bits.get_unchecked_mut(wi) = *nl_acc;
        *idx.q_bits.get_unchecked_mut(wi) = *q_acc;
        *widx = wi + 1;
        *bpos = next_bp - 64;
        let sh = 64 - bp;
        *nl_acc = if sh < 64 { nl_pext >> sh } else { 0 };
        *q_acc = if sh < 64 { q_pext >> sh } else { 0 };
    } else {
        *bpos = next_bp;
    }

    let ts = *total_seps;
    let dst = idx.seps.as_mut_ptr().add(ts);
    let base = (bi << 6) as u64;
    let mut rem = s;
    std::arch::asm!(
        "tzcnt {t}, {s}",
        "lea {v:e}, [{base:r} + {t}]",
        "mov word ptr [{dst}], {v:x}",
        "blsr {s}, {s}",
        "tzcnt {t}, {s}",
        "lea {v:e}, [{base:r} + {t}]",
        "mov word ptr [{dst} + 2], {v:x}",
        "blsr {s}, {s}",
        "tzcnt {t}, {s}",
        "lea {v:e}, [{base:r} + {t}]",
        "mov word ptr [{dst} + 4], {v:x}",
        "blsr {s}, {s}",
        "tzcnt {t}, {s}",
        "lea {v:e}, [{base:r} + {t}]",
        "mov word ptr [{dst} + 6], {v:x}",
        "blsr {s}, {s}",
        "tzcnt {t}, {s}",
        "lea {v:e}, [{base:r} + {t}]",
        "mov word ptr [{dst} + 8], {v:x}",
        "blsr {s}, {s}",
        "tzcnt {t}, {s}",
        "lea {v:e}, [{base:r} + {t}]",
        "mov word ptr [{dst} + 10], {v:x}",
        "blsr {s}, {s}",
        "tzcnt {t}, {s}",
        "lea {v:e}, [{base:r} + {t}]",
        "mov word ptr [{dst} + 12], {v:x}",
        "blsr {s}, {s}",
        "tzcnt {t}, {s}",
        "lea {v:e}, [{base:r} + {t}]",
        "mov word ptr [{dst} + 14], {v:x}",
        "blsr {s}, {s}",
        s = inout(reg) rem,
        t = out(reg) _,
        v = out(reg) _,
        dst = in(reg) dst,
        base = in(reg) base,
        options(nostack),
    );
    if rem != 0 {
        dense_seps(dst, base as u16, rem);
    }
    *total_seps = ts + pop as usize;
}

#[allow(dead_code)]
#[inline(never)]
unsafe fn index_block(
    idx: &mut BlockIdx,
    bptr: *const u8,
    blen: usize,
    size_rem: usize,
    pfd: usize,
) -> (usize, bool) {
    if cfg!(all(target_feature = "avx512bw", target_feature = "avx512vl"))
        || crate::sys::use_avx512()
    {
        return index_block_avx512vl(idx, bptr, blen, size_rem, pfd);
    }
    index_block_avx2(idx, bptr, blen, size_rem, pfd)
}

#[target_feature(enable = "avx512bw,avx512vl,bmi2,lzcnt,pclmulqdq")]
unsafe fn index_block_avx512vl(
    idx: &mut BlockIdx,
    bptr: *const u8,
    blen: usize,
    size_rem: usize,
    pfd: usize,
) -> (usize, bool) {
    idx.seps[0] = u16::MAX;

    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let quote = _mm256_set1_epi8(b'"' as i8);

    let mut total_seps = 1usize;
    let mut nl_acc = 1u64;
    let mut q_acc = 0u64;
    let mut widx = 0usize;
    let mut bpos = 1u32;
    let mut s_carry = 1u64;
    let mut rq_carry = 0u64;
    let mut inq_mask = 0u64;
    let mut any_bad = 0u64;

    let nblks = (blen + 63) >> 6;
    let full_blks = blen >> 6;
    for bi in 0..full_blks {
        let pf_off = ((bi << 6) + pfd).min(size_rem);
        _mm_prefetch(bptr.add(pf_off) as *const i8, _MM_HINT_T0);
        let p = bptr.add(bi << 6);
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let cm = (_mm256_cmpeq_epi8_mask(a, comma) as u64)
            | ((_mm256_cmpeq_epi8_mask(b, comma) as u64) << 32);
        let nm = (_mm256_cmpeq_epi8_mask(a, nl) as u64)
            | ((_mm256_cmpeq_epi8_mask(b, nl) as u64) << 32);
        let qm = (_mm256_cmpeq_epi8_mask(a, quote) as u64)
            | ((_mm256_cmpeq_epi8_mask(b, quote) as u64) << 32);
        step_block(
            idx,
            bi,
            cm,
            nm,
            qm,
            &mut total_seps,
            &mut nl_acc,
            &mut q_acc,
            &mut widx,
            &mut bpos,
            &mut s_carry,
            &mut rq_carry,
            &mut inq_mask,
            &mut any_bad,
        );
    }
    if full_blks < nblks {
        let bi = full_blks;
        let rem = (blen & 63) as u32;
        let p = bptr.add(bi << 6);
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let cm = _bzhi_u64(
            (_mm256_cmpeq_epi8_mask(a, comma) as u64)
                | ((_mm256_cmpeq_epi8_mask(b, comma) as u64) << 32),
            rem,
        );
        let nm = _bzhi_u64(
            (_mm256_cmpeq_epi8_mask(a, nl) as u64)
                | ((_mm256_cmpeq_epi8_mask(b, nl) as u64) << 32),
            rem,
        );
        let qm = _bzhi_u64(
            (_mm256_cmpeq_epi8_mask(a, quote) as u64)
                | ((_mm256_cmpeq_epi8_mask(b, quote) as u64) << 32),
            rem,
        );
        step_block(
            idx,
            bi,
            cm,
            nm,
            qm,
            &mut total_seps,
            &mut nl_acc,
            &mut q_acc,
            &mut widx,
            &mut bpos,
            &mut s_carry,
            &mut rq_carry,
            &mut inq_mask,
            &mut any_bad,
        );
    }
    *idx.nl_bits.get_unchecked_mut(widx) = nl_acc;
    *idx.q_bits.get_unchecked_mut(widx) = q_acc;
    *idx.nl_bits.get_unchecked_mut(widx + 1) = 0;
    *idx.q_bits.get_unchecked_mut(widx + 1) = 0;
    (total_seps, any_bad != 0)
}

#[target_feature(enable = "avx2,bmi2,lzcnt,pclmulqdq")]
unsafe fn index_block_avx2(
    idx: &mut BlockIdx,
    bptr: *const u8,
    blen: usize,
    size_rem: usize,
    pfd: usize,
) -> (usize, bool) {
    idx.seps[0] = u16::MAX;

    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let quote = _mm256_set1_epi8(b'"' as i8);

    let mut total_seps = 1usize;
    let mut nl_acc = 1u64;
    let mut q_acc = 0u64;
    let mut widx = 0usize;
    let mut bpos = 1u32;
    let mut s_carry = 1u64;
    let mut rq_carry = 0u64;
    let mut inq_mask = 0u64;
    let mut any_bad = 0u64;

    let nblks = (blen + 63) >> 6;
    let full_blks = blen >> 6;
    for bi in 0..full_blks {
        let pf_off = ((bi << 6) + pfd).min(size_rem);
        _mm_prefetch(bptr.add(pf_off) as *const i8, _MM_HINT_T0);
        let p = bptr.add(bi << 6);
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let cm = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, comma)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, comma)) as u32 as u64) << 32);
        let nm = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, nl)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, nl)) as u32 as u64) << 32);
        let qm = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, quote)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, quote)) as u32 as u64) << 32);
        step_block(
            idx,
            bi,
            cm,
            nm,
            qm,
            &mut total_seps,
            &mut nl_acc,
            &mut q_acc,
            &mut widx,
            &mut bpos,
            &mut s_carry,
            &mut rq_carry,
            &mut inq_mask,
            &mut any_bad,
        );
    }
    if full_blks < nblks {
        let bi = full_blks;
        let rem = (blen & 63) as u32;
        let p = bptr.add(bi << 6);
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let cm = _bzhi_u64(
            (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, comma)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, comma)) as u32 as u64) << 32),
            rem,
        );
        let nm = _bzhi_u64(
            (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, nl)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, nl)) as u32 as u64) << 32),
            rem,
        );
        let qm = _bzhi_u64(
            (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, quote)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, quote)) as u32 as u64) << 32),
            rem,
        );
        step_block(
            idx,
            bi,
            cm,
            nm,
            qm,
            &mut total_seps,
            &mut nl_acc,
            &mut q_acc,
            &mut widx,
            &mut bpos,
            &mut s_carry,
            &mut rq_carry,
            &mut inq_mask,
            &mut any_bad,
        );
    }
    *idx.nl_bits.get_unchecked_mut(widx) = nl_acc;
    *idx.q_bits.get_unchecked_mut(widx) = q_acc;
    *idx.nl_bits.get_unchecked_mut(widx + 1) = 0;
    *idx.q_bits.get_unchecked_mut(widx + 1) = 0;
    (total_seps, any_bad != 0)
}

#[inline(always)]
fn last_nl_sep(nl_bits: &[u64; 66], total_seps: usize) -> usize {
    let last_w = (total_seps - 1) >> 6;
    let last_bit = (total_seps - 1) & 63;
    let mask = u64::MAX >> (63 - last_bit);
    let mut w = last_w;
    let mut word = nl_bits[w] & mask;
    loop {
        if w == 0 {
            word &= !1u64;
        }
        if word != 0 {
            let bit_in_w = 63 - word.leading_zeros() as usize;
            return (w << 6) | bit_in_w;
        }
        if w == 0 {
            return 0;
        }
        w -= 1;
        word = nl_bits[w];
    }
}

#[cold]
#[inline(never)]
unsafe fn fallback_block<const N: usize, F: RowHandler>(
    reader: &mut Reader,
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
    handler: &mut F,
) {
    reader.scan::<N>(data, size, from, to, |buf, s32, e32| {
        handler.on_row(FallbackRow {
            buf,
            s: *s32,
            e: *e32,
        });
    });
}

#[inline(always)]
unsafe fn scan_buf_rows<const N: usize, F: RowHandler>(
    data: *const u8,
    size: usize,
    mut cur: usize,
    b: usize,
    index_fn: unsafe fn(&mut BlockIdx, *const u8, usize, usize, usize) -> (usize, bool),
    idx: &mut BlockIdx,
    pfd: usize,
    fallback: &mut Reader,
    handler: &mut F,
) {
    let nl_mask = (1u64 << N) - 1;
    let want_nl = 1u64 << (N - 1);

    while cur < b && cur < size {
        let size_rem = size - cur;
        let blen = 4096usize.min(size_rem);
        let bptr = data.add(cur);
        let (total_seps, any_bad) = index_fn(idx, bptr, blen, size_rem, pfd);
        let last_nl = last_nl_sep(&idx.nl_bits, total_seps);
        if last_nl == 0 {
            // Line exceeds 4096 bytes: delegate up to its newline to csv::Reader.
            let nl_at = find_nl(data, cur, size);
            let next = nl_at + 1;
            fallback_block::<N, F>(fallback, data, size, cur, next.min(b), handler);
            cur = next;
            continue;
        }
        let next_cur = cur + *idx.seps.get_unchecked(last_nl) as usize + 1;
        if any_bad {
            fallback_block::<N, F>(fallback, data, size, cur, next_cur.min(b), handler);
            cur = next_cur;
            continue;
        }

        let mut d_idx = 0usize;
        while d_idx < last_nl {
            let line_start = (*idx.seps.get_unchecked(d_idx)).wrapping_add(1) as usize;
            if cur + line_start >= b {
                return;
            }
            let bit = d_idx + 1;
            let byte_at = bit >> 3;
            let bit_off = bit & 7;
            let nl_word = ((idx.nl_bits.as_ptr() as *const u8).add(byte_at) as *const u64)
                .read_unaligned()
                >> bit_off;
            if (nl_word & nl_mask) != want_nl {
                let mut w = bit >> 6;
                let rem = *idx.nl_bits.get_unchecked(w) >> (bit & 63);
                if rem != 0 {
                    d_idx = bit + rem.trailing_zeros() as usize;
                } else {
                    w += 1;
                    while *idx.nl_bits.get_unchecked(w) == 0 {
                        w += 1;
                    }
                    d_idx = (w << 6) + idx.nl_bits.get_unchecked(w).trailing_zeros() as usize;
                }
                continue;
            }
            let q_word = ((idx.q_bits.as_ptr() as *const u8).add(byte_at) as *const u64)
                .read_unaligned()
                >> bit_off;
            let seps = idx.seps.as_ptr().add(d_idx);
            d_idx += N;
            handler.on_row(FastRow { bptr, seps, q_word });
        }
        cur = next_cur;
    }
}

/// Calls `handler.on_row(...)` for every well-formed line of `N` fields (`1 <= N <= 7`)
/// that starts in `a..b` of the mapped file `(data, size)`. The header line is skipped.
/// `data` must end with `b'\n'` when `size > 0` and have >= 64 B readable zero padding past `size`.
pub unsafe fn scan_rows<const N: usize, F: RowHandler>(
    data: *const u8,
    size: usize,
    a: usize,
    b: usize,
    fallback: &mut Reader,
    handler: &mut F,
) {
    if size == 0 || a >= b || a >= size {
        return;
    }
    let first_nl = find_nl(data, a.saturating_sub(1), size);
    let cur = first_nl + 1;
    if cur >= b || cur >= size {
        return;
    }
    let pfd = prefetch_distance();
    let index_fn: unsafe fn(&mut BlockIdx, *const u8, usize, usize, usize) -> (usize, bool) =
        if cfg!(all(target_feature = "avx512bw", target_feature = "avx512vl"))
            || crate::sys::use_avx512()
        {
            index_block_avx512vl
        } else {
            index_block_avx2
        };
    let mut idx_uninit = std::mem::MaybeUninit::<BlockIdx>::uninit();
    let idx = &mut *idx_uninit.as_mut_ptr();
    scan_buf_rows::<N, F>(data, size, cur, b, index_fn, idx, pfd, fallback, handler);
}

unsafe fn pread_all(fd: RawFd, dst: *mut u8, max_len: usize, off: usize) -> usize {
    extern "C" {
        fn pread(
            fd: std::os::raw::c_int,
            buf: *mut std::os::raw::c_void,
            count: usize,
            offset: i64,
        ) -> isize;
    }
    let mut got = 0usize;
    while got < max_len {
        let r = pread(
            fd,
            dst.add(got) as *mut std::os::raw::c_void,
            max_len - got,
            (off + got) as i64,
        );
        if r > 0 {
            got += r as usize;
        } else if r == 0 {
            break;
        } else {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            panic!("pread failed: {err}");
        }
    }
    got
}

/// `SS_PREAD=1` variant of `scan_rows`: reads chunks via `libc::pread` into `pbuf`
/// (132 KiB 64B-aligned buffer with 4 KiB leading zeros + 64 B trailing zeros).
pub unsafe fn scan_rows_pread<const N: usize, F: RowHandler>(
    fd: RawFd,
    file_size: usize,
    a: usize,
    b: usize,
    fallback: &mut Reader,
    pbuf: &mut PreadBuf,
    handler: &mut F,
) {
    if file_size == 0 || a >= b || a >= file_size {
        return;
    }
    let pfd = prefetch_distance();
    let index_fn: unsafe fn(&mut BlockIdx, *const u8, usize, usize, usize) -> (usize, bool) =
        if cfg!(all(target_feature = "avx512bw", target_feature = "avx512vl"))
            || crate::sys::use_avx512()
        {
            index_block_avx512vl
        } else {
            index_block_avx2
        };
    let mut idx_uninit = std::mem::MaybeUninit::<BlockIdx>::uninit();
    let idx = &mut *idx_uninit.as_mut_ptr();

    let mut file_off = a.saturating_sub(1);
    let mut data_ptr = pbuf.data_ptr();
    let mut n = pread_all(
        fd,
        data_ptr,
        (pbuf.data_cap() - 1).min(file_size - file_off),
        file_off,
    );
    let mut buf_cur;
    loop {
        if file_off + n >= file_size && (n == 0 || *data_ptr.add(n - 1) != b'\n') {
            *data_ptr.add(n) = b'\n';
            n += 1;
        }
        let nl = find_nl(data_ptr, 0, n);
        if nl < n {
            buf_cur = nl + 1;
            break;
        }
        file_off += n;
        if file_off >= b || file_off >= file_size {
            return;
        }
        n = pread_all(
            fd,
            data_ptr,
            (pbuf.data_cap() - 1).min(file_size - file_off),
            file_off,
        );
    }
    if file_off + buf_cur >= b || file_off + buf_cur >= file_size {
        return;
    }

    loop {
        if file_off + n >= file_size && (n == 0 || *data_ptr.add(n - 1) != b'\n') {
            *data_ptr.add(n) = b'\n';
            n += 1;
        }
        let mut last_nl_pos = n;
        let mut found_nl = false;
        while last_nl_pos > buf_cur {
            last_nl_pos -= 1;
            if *data_ptr.add(last_nl_pos) == b'\n' {
                found_nl = true;
                break;
            }
        }
        if !found_nl {
            let keep_from = buf_cur - 1;
            let kept = n - keep_from;
            std::ptr::copy(data_ptr.add(keep_from), data_ptr, kept);
            file_off += keep_from;
            buf_cur = 1;
            n = kept;
            if n >= pbuf.data_cap() - 1 {
                pbuf.grow_for(pbuf.data_cap() * 2);
                data_ptr = pbuf.data_ptr();
            }
            let rem_file = file_size.saturating_sub(file_off + n);
            let max_read = (pbuf.data_cap() - 1 - n).min(rem_file);
            if max_read > 0 {
                n += pread_all(fd, data_ptr.add(n), max_read, file_off + n);
            }
            continue;
        }

        let chunk_end = last_nl_pos + 1;
        std::ptr::write_bytes(data_ptr.add(chunk_end), 0, PREAD_TRAIL);
        let local_b = b.saturating_sub(file_off).min(chunk_end);
        scan_buf_rows::<N, F>(
            data_ptr,
            chunk_end,
            buf_cur,
            local_b,
            index_fn,
            idx,
            pfd,
            fallback,
            handler,
        );

        let next_file_off = file_off + chunk_end;
        if next_file_off >= b || next_file_off >= file_size {
            return;
        }
        handler.before_refill();
        file_off = next_file_off - 1;
        let want = PREAD_CHUNK.min(pbuf.data_cap() - 1).min(file_size - file_off);
        n = pread_all(fd, data_ptr, want, file_off);
        buf_cur = 1;
    }
}

/// Calls `f(base_ptr, starts, ends)` for every well-formed line of `N` fields (`1 <= N <= 7`)
/// that starts in `a..b` of the mapped file `(data, size)`. The header line is skipped.
/// `data` must end with `b'\n'` when `size > 0` and have >= 64 B readable zero padding past `size`.
#[allow(dead_code)]
pub unsafe fn scan<const N: usize>(
    data: *const u8,
    size: usize,
    a: usize,
    b: usize,
    fallback: &mut Reader,
    f: impl FnMut(*const u8, &[u16; 8], &[u16; 8]),
) {
    struct LegacyHandler<const N: usize, F: FnMut(*const u8, &[u16; 8], &[u16; 8])>(F);
    impl<const N: usize, F: FnMut(*const u8, &[u16; 8], &[u16; 8])> RowHandler for LegacyHandler<N, F> {
        #[inline(always)]
        unsafe fn on_row<R: CsvRow>(&mut self, row: R) {
            let mut s = [0u16; 8];
            let mut e = [0u16; 8];
            let mut base = std::ptr::null();
            for k in 0..N {
                let (p, l) = row.field(k);
                if k == 0 {
                    base = p;
                }
                let off = p.offset_from(base) as u16;
                s[k] = off;
                e[k] = off + l as u16;
            }
            (self.0)(base, &s, &e);
        }
    }
    let mut h = LegacyHandler::<N, _>(f);
    scan_rows::<N, _>(data, size, a, b, fallback, &mut h);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys::{end_line, map_files_contiguous_par, release};

    unsafe fn run_both<const N: usize>(raw: &[u8], splits: &[usize]) {
        use std::os::unix::io::AsRawFd;
        let dir = std::env::temp_dir().join(format!(
            "ss_scan4k_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("t.csv");
        std::fs::write(&p, raw).unwrap();
        let file = std::fs::File::open(&p).unwrap();
        let raw_len = raw.len();
        let region = map_files_contiguous_par(&[p], 1);
        let ptr = region.files[0].ptr;
        let size = end_line(ptr, region.files[0].size);

        struct Collect<const N: usize>(Vec<Vec<Vec<u8>>>);
        impl<const N: usize> RowHandler for Collect<N> {
            unsafe fn on_row<R: CsvRow>(&mut self, row: R) {
                let mut r = Vec::with_capacity(N);
                for k in 0..N {
                    let (p, l) = row.field(k);
                    r.push(std::slice::from_raw_parts(p, l).to_vec());
                }
                self.0.push(r);
            }
        }

        let mut pbuf = PreadBuf::new();
        for w in splits.windows(2) {
            let (a, b) = (w[0], w[1]);
            let mut got = Vec::new();
            let mut fb = Reader::lazy(16 << 10);
            scan::<N>(ptr, size, a, b, &mut fb, |buf, s, e| {
                let mut row = Vec::with_capacity(N);
                for k in 0..N {
                    row.push(
                        std::slice::from_raw_parts(
                            buf.add(s[k] as usize),
                            (e[k] - s[k]) as usize,
                        )
                        .to_vec(),
                    );
                }
                got.push(row);
            });

            let mut got_pread = Collect::<N>(Vec::new());
            let mut fb_pread = Reader::lazy(16 << 10);
            scan_rows_pread::<N, _>(
                file.as_raw_fd(),
                raw_len,
                a,
                b,
                &mut fb_pread,
                &mut pbuf,
                &mut got_pread,
            );

            let mut exp = Vec::new();
            let mut r = Reader::new(16 << 10);
            r.scan::<N>(ptr, size, a, b, |buf, s, e| {
                let mut row = Vec::with_capacity(N);
                for k in 0..N {
                    row.push(
                        std::slice::from_raw_parts(
                            buf.add(s[k] as usize),
                            (e[k] - s[k]) as usize,
                        )
                        .to_vec(),
                    );
                }
                exp.push(row);
            });

            assert_eq!(got, exp, "scan mismatch on split {a}..{b}");
            assert_eq!(got_pread.0, exp, "pread mismatch on split {a}..{b}");
        }
        release(region.reserve_base, region.reserve_len);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan4k_pread_multi_chunk_and_dense() {
        let mut data = Vec::new();
        data.extend_from_slice(b"c0,c1,c2,c3,c4,c5,c6\n");
        // > 300 KiB across multiple 132 KiB pread chunks with quoted/unquoted & fallback lines
        for i in 0..5500 {
            if i == 2100 {
                data.extend_from_slice(b"a\"b,u_stray,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,view,p1,0\n");
            } else if i % 100 == 0 {
                data.extend_from_slice(
                    format!("\"e_{i}\",\"u_{i}\",2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,purchase,p1,99\n")
                        .as_bytes(),
                );
            } else {
                data.extend_from_slice(
                    format!("e_{i},u_{i},2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,view,p1,0\n")
                        .as_bytes(),
                );
            }
        }
        data.extend_from_slice(b"e_tail,u_tail,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,view,p1,0");
        let n = data.len();
        unsafe {
            run_both::<7>(&data, &[0, n]);
            run_both::<7>(&data, &[0, 1024, 131_000, 131_008, 131_072, 262_144, n]);
        }
    }

    #[test]
    fn scan4k_matches_csv_reader_tricky_dialects() {
        let mut data = Vec::new();
        data.extend_from_slice(b"c0,c1,c2,c3,c4,c5,c6\n");
        data.extend_from_slice(b"e1,u1,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,view,p1,0\n");
        // Empty fields & quoted empty fields
        data.extend_from_slice(b",\"\",,,,,\n");
        data.extend_from_slice(b"\"e2\",\"u2\",\"2026-01-01T00:00:01Z\",\"2026-01-01T00:00:01Z\",\"purchase\",\"p2\",\"123\"\n");
        // Stray quotes, comma inside quotes, single quote, wrong column counts, CR
        data.extend_from_slice(b"\"a,b\",u3,ts,ing,view,p3,0\n");
        data.extend_from_slice(b"a\"b,u4,ts,ing,view,p4,0\n");
        data.extend_from_slice(b"\",u5,ts,ing,view,p5,0\n");
        data.extend_from_slice(b"\"unclosed,u6,ts,ing,view,p6,0\n");
        data.extend_from_slice(b"only,two\n");
        data.extend_from_slice(b"a,b,c,d,e,f,g,extra\n");
        data.extend_from_slice(b"e3,u7,ts,ing,cart,p7,0\r\n");
        data.extend_from_slice(b"\"\"\"\",u8,ts,ing,view,p8,0\n");
        // Long line > 4096 bytes
        data.extend_from_slice(b"e_long,");
        data.extend(std::iter::repeat_n(b'x', 4500));
        data.extend_from_slice(b",ts,ing,view,p9,0\n");
        // Fill past 10 KiB across multiple 4K boundaries without trailing newline
        for i in 0..200 {
            let line = format!("e_{i},u_{i},2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,view,p1,0\n");
            data.extend_from_slice(line.as_bytes());
        }
        data.extend_from_slice(b"e_last,u_last,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,view,p1,0");

        let n = data.len();
        unsafe {
            run_both::<7>(&data, &[0, n]);
            // Mid-line, mid-quote, mid-4K, and exact newline boundary splits
            run_both::<7>(&data, &[0, 1, 22, 64, 150, 4095, 4096, 4097, 8192, n - 5, n]);
        }
    }

    #[test]
    fn scan4k_4col_clicks_boundary_splits() {
        let mut data = Vec::new();
        data.extend_from_slice(b"click_id,user_id,campaign_id,ts\n");
        for i in 0..500 {
            if i % 17 == 0 {
                data.extend_from_slice(format!("\"k_{i}\",\"u_{i}\",\"c_1\",\"2026-01-01T00:00:00Z\"\n").as_bytes());
            } else if i % 23 == 0 {
                data.extend_from_slice(b"\"\",u_empty,c_2,2026-01-01T00:00:00Z\n");
            } else {
                data.extend_from_slice(format!("k_{i},u_{i},c_1,2026-01-01T00:00:00Z\n").as_bytes());
            }
        }
        let n = data.len();
        let mut splits = Vec::new();
        let mut p = 0;
        while p < n {
            splits.push(p);
            p += 317;
        }
        splits.push(n);
        unsafe {
            run_both::<4>(&data, &splits);
        }
    }
}
