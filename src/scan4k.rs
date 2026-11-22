use crate::csv::Reader;
use std::arch::x86_64::*;
use std::sync::OnceLock;

const DEFAULT_PFD: usize = 1024;

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
unsafe fn zero_bits(idx: &mut BlockIdx) {
    let z = _mm256_setzero_si256();
    let nl = idx.nl_bits.as_mut_ptr() as *mut __m256i;
    let qb = idx.q_bits.as_mut_ptr() as *mut __m256i;
    for i in 0..16 {
        _mm256_storeu_si256(nl.add(i), z);
        _mm256_storeu_si256(qb.add(i), z);
    }
    _mm_storeu_si128(idx.nl_bits.as_mut_ptr().add(64) as *mut __m128i, _mm_setzero_si128());
    _mm_storeu_si128(idx.q_bits.as_mut_ptr().add(64) as *mut __m128i, _mm_setzero_si128());
    idx.nl_bits[0] = 1;
}

#[inline(always)]
unsafe fn step_block(
    idx: &mut BlockIdx,
    bi: usize,
    cm: u64,
    nm: u64,
    qm: u64,
    total_seps: &mut usize,
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

    let pop = s.count_ones() as usize;
    let nl_pext = _pext_u64(nm, s);
    let q_pext = _pext_u64(cq, s);

    let ts = *total_seps;
    let dst_nl = idx.nl_bits.as_mut_ptr().add(ts >> 6) as *mut u128;
    dst_nl.write_unaligned(dst_nl.read_unaligned() | ((nl_pext as u128) << (ts & 63)));
    let dst_q = idx.q_bits.as_mut_ptr().add(ts >> 6) as *mut u128;
    dst_q.write_unaligned(dst_q.read_unaligned() | ((q_pext as u128) << (ts & 63)));

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
    *total_seps = ts + pop;
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
    zero_bits(idx);

    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let quote = _mm256_set1_epi8(b'"' as i8);

    let mut total_seps = 1usize;
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
            &mut s_carry,
            &mut rq_carry,
            &mut inq_mask,
            &mut any_bad,
        );
    }
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
    zero_bits(idx);

    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let quote = _mm256_set1_epi8(b'"' as i8);

    let mut total_seps = 1usize;
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
            &mut s_carry,
            &mut rq_carry,
            &mut inq_mask,
            &mut any_bad,
        );
    }
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
unsafe fn fallback_block<const N: usize>(
    reader: &mut Reader,
    data: *const u8,
    size: usize,
    from: usize,
    to: usize,
    mut f: impl FnMut(*const u8, &[u16; 8], &[u16; 8]),
) {
    reader.scan::<N>(data, size, from, to, |buf, s32, e32| {
        let mut s = [0u16; 8];
        let mut e = [0u16; 8];
        for k in 0..N {
            s[k] = s32[k] as u16;
            e[k] = e32[k] as u16;
        }
        f(buf, &s, &e);
    });
}

/// Calls `f(base_ptr, starts, ends)` for every well-formed line of `N` fields (`1 <= N <= 7`)
/// that starts in `a..b` of the mapped file `(data, size)`. The header line is skipped.
/// `data` must end with `b'\n'` when `size > 0` and have >= 64 B readable zero padding past `size`.
pub unsafe fn scan<const N: usize>(
    data: *const u8,
    size: usize,
    a: usize,
    b: usize,
    fallback: &mut Reader,
    mut f: impl FnMut(*const u8, &[u16; 8], &[u16; 8]),
) {
    if size == 0 || a >= b || a >= size {
        return;
    }
    let first_nl = find_nl(data, a.saturating_sub(1), size);
    let mut cur = first_nl + 1;
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
            fallback_block::<N>(fallback, data, size, cur, next.min(b), &mut f);
            cur = next;
            continue;
        }
        let next_cur = cur + *idx.seps.get_unchecked(last_nl) as usize + 1;
        if any_bad {
            fallback_block::<N>(fallback, data, size, cur, next_cur.min(b), &mut f);
            cur = next_cur;
            continue;
        }

        let bit_lut = _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128);
        let ones = _mm_set1_epi16(1);
        let twos = _mm_set1_epi16(2);
        let zero = _mm_setzero_si128();
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
            let before = _mm_loadu_si128(idx.seps.as_ptr().add(d_idx) as *const __m128i);
            let after = _mm_loadu_si128(idx.seps.as_ptr().add(d_idx + 1) as *const __m128i);
            let qv = _mm_and_si128(_mm_set1_epi16(q_word as i16), bit_lut);
            let qmask = _mm_cmpeq_epi16(qv, zero);
            let starts = _mm_add_epi16(before, _mm_add_epi16(twos, qmask));
            let ends = _mm_sub_epi16(after, _mm_add_epi16(ones, qmask));
            let mut s = [0u16; 8];
            let mut e = [0u16; 8];
            _mm_storeu_si128(s.as_mut_ptr() as *mut __m128i, starts);
            _mm_storeu_si128(e.as_mut_ptr() as *mut __m128i, ends);
            d_idx += N;
            f(bptr, &s, &e);
        }
        cur = next_cur;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys::{end_line, map_files_contiguous_par, release};

    unsafe fn run_both<const N: usize>(raw: &[u8], splits: &[usize]) {
        let dir = std::env::temp_dir().join(format!(
            "ss_scan4k_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("t.csv");
        std::fs::write(&p, raw).unwrap();
        let region = map_files_contiguous_par(&[p], 1);
        let ptr = region.files[0].ptr;
        let size = end_line(ptr, region.files[0].size);

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

            assert_eq!(got, exp, "mismatch on split {a}..{b}");
        }
        release(region.reserve_base, region.reserve_len);
        let _ = std::fs::remove_dir_all(&dir);
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
