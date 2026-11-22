use std::arch::x86_64::*;

/// Indexes a mapped CSV file a cache-sized window at a time and hands out the
/// field bounds of every well-formed line.
///
/// The index holds, per comma or newline, its position and the four bytes from
/// the one before it: whether its field closed with a quote, whether it ends a
/// line, and whether the next field opens with a quote. Every window ends after
/// a newline, so every indexed line is complete.
pub struct Reader {
    window: usize,
    /// The low 32 bits of each delimiter's address; entry 0 stands for the
    /// newline before the window.
    ent: Vec<u32>,
    /// The four bytes at each delimiter's position - 1.
    flags: Vec<u32>,
    /// Positions of quotes that neither open nor close their field.
    strays: Vec<u32>,
    count: usize,
}

impl Reader {
    #[allow(dead_code)]
    pub fn new(window: usize) -> Self {
        let mut r = Self::lazy(window);
        r.ensure();
        r
    }

    pub fn lazy(window: usize) -> Self {
        let window = window.max(64);
        assert!(window < (u32::MAX >> 1) as usize, "line too long");
        Reader { window, ent: Vec::new(), flags: Vec::new(), strays: Vec::new(), count: 0 }
    }

    #[inline(always)]
    fn ensure(&mut self) {
        if self.ent.is_empty() {
            self.resize(self.window);
        }
    }

    fn resize(&mut self, window: usize) {
        assert!(window < (u32::MAX >> 1) as usize, "line too long");
        self.window = window;
        self.ent.resize(window + 128, 0);
        self.flags.resize(window + 128, 0);
    }

    /// Indexes the `n` bytes at `buf`, which start at a line start and end
    /// with a newline. Dispatches to 256-bit AVX-512VL when compiled with
    /// AVX-512BW+VL or supported and enabled at runtime, else AVX2.
    /// Kept out-of-line so PGO profiles generated on x86-64-v3 stay valid for `scan`.
    #[inline(never)]
    unsafe fn index(&mut self, buf: *const u8, n: usize) {
        if cfg!(all(target_feature = "avx512bw", target_feature = "avx512vl"))
            || crate::sys::use_avx512()
        {
            self.index_avx512vl(buf, n);
            return;
        }
        self.index_avx2(buf, n);
    }

    #[inline(always)]
    unsafe fn index_step<const FULL: bool>(
        &mut self,
        buf: *const u8,
        p: *const u8,
        left: usize,
        mut d: u64,
        mut q: u64,
        dst: &mut *mut u32,
        fdst: &mut *mut u32,
        pd: &mut u64,
        pq: &mut u64,
        pfs: &mut u64,
        carry: &mut u64,
    ) {
        if !FULL {
            d = _bzhi_u64(d, left as u32);
            q = _bzhi_u64(q, left as u32);
        }
        let s: u64;
        // Entries hold the low 32 bits of the delimiter's address. Seven
        // delimiters cover all but 0.04% of words. Hand-written because
        // LLVM's generic tuning adds a dependency-breaking xor and splits
        // the lea, three more instructions per delimiter.
        std::arch::asm!(
            "tzcnt {t}, {s}",
            "lea {v:e}, [{src} + {t}]",
            "mov dword ptr [{dst}], {v:e}",
            "mov {v:e}, dword ptr [{src} + {t} - 1]",
            "mov dword ptr [{fdst}], {v:e}",
            "blsr {s}, {s}",
            "tzcnt {t}, {s}",
            "lea {v:e}, [{src} + {t}]",
            "mov dword ptr [{dst} + 4], {v:e}",
            "mov {v:e}, dword ptr [{src} + {t} - 1]",
            "mov dword ptr [{fdst} + 4], {v:e}",
            "blsr {s}, {s}",
            "tzcnt {t}, {s}",
            "lea {v:e}, [{src} + {t}]",
            "mov dword ptr [{dst} + 8], {v:e}",
            "mov {v:e}, dword ptr [{src} + {t} - 1]",
            "mov dword ptr [{fdst} + 8], {v:e}",
            "blsr {s}, {s}",
            "tzcnt {t}, {s}",
            "lea {v:e}, [{src} + {t}]",
            "mov dword ptr [{dst} + 12], {v:e}",
            "mov {v:e}, dword ptr [{src} + {t} - 1]",
            "mov dword ptr [{fdst} + 12], {v:e}",
            "blsr {s}, {s}",
            "tzcnt {t}, {s}",
            "lea {v:e}, [{src} + {t}]",
            "mov dword ptr [{dst} + 16], {v:e}",
            "mov {v:e}, dword ptr [{src} + {t} - 1]",
            "mov dword ptr [{fdst} + 16], {v:e}",
            "blsr {s}, {s}",
            "tzcnt {t}, {s}",
            "lea {v:e}, [{src} + {t}]",
            "mov dword ptr [{dst} + 20], {v:e}",
            "mov {v:e}, dword ptr [{src} + {t} - 1]",
            "mov dword ptr [{fdst} + 20], {v:e}",
            "blsr {s}, {s}",
            "tzcnt {t}, {s}",
            "lea {v:e}, [{src} + {t}]",
            "mov dword ptr [{dst} + 24], {v:e}",
            "mov {v:e}, dword ptr [{src} + {t} - 1]",
            "mov dword ptr [{fdst} + 24], {v:e}",
            "blsr {s}, {s}",
            s = inout(reg) d => s,
            t = out(reg) _,
            v = out(reg) _,
            dst = in(reg) *dst,
            fdst = in(reg) *fdst,
            src = in(reg) p,
            options(nostack),
        );
        let found = d.count_ones() as usize;
        if found > 7 {
            Self::dense(*dst, *fdst, p, s, found);
        }
        let stray = *pq & !(*pfs ^ (*pd >> 1 | d << 63));
        if stray != 0 {
            self.note(stray, p.sub(64).offset_from(buf) as u32);
        }
        *pfs = d << 1 | *carry;
        *carry = d >> 63;
        *pd = d;
        *pq = q;
        *dst = dst.add(found);
        *fdst = fdst.add(found);
    }

    #[target_feature(enable = "avx512bw,avx512vl,bmi2,lzcnt")]
    unsafe fn index_avx512vl(&mut self, buf: *const u8, n: usize) {
        let ent = self.ent.as_mut_ptr();
        let flags = self.flags.as_mut_ptr();
        let comma = _mm256_set1_epi8(b',' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let quote = _mm256_set1_epi8(b'"' as i8);
        *ent = (buf as u32).wrapping_sub(1);
        *flags = (b'\n' as u32) << 8 | (*buf as u32) << 16 | (*buf.add(1) as u32) << 24;
        self.strays.clear();
        let (mut dst, mut fdst) = (ent.add(1), flags.add(1));
        let (mut pd, mut pq, mut pfs) = (0u64, 0u64, 0u64);
        let mut carry = 1u64;
        let mut p = buf;
        let full_end = buf.add(n & !63);
        let end = buf.add(n);
        while p < full_end {
            let a = _mm256_loadu_si256(p as *const __m256i);
            let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
            let da = _mm256_cmpeq_epi8_mask(a, comma) | _mm256_cmpeq_epi8_mask(a, newline);
            let db = _mm256_cmpeq_epi8_mask(b, comma) | _mm256_cmpeq_epi8_mask(b, newline);
            let d = (da as u64) | ((db as u64) << 32);
            let q = (_mm256_cmpeq_epi8_mask(a, quote) as u64)
                | ((_mm256_cmpeq_epi8_mask(b, quote) as u64) << 32);
            self.index_step::<true>(
                buf, p, 64, d, q, &mut dst, &mut fdst, &mut pd, &mut pq, &mut pfs, &mut carry,
            );
            p = p.add(64);
        }
        if p < end {
            // `scan` guarantees >= 64 bytes of readable padding past `buf + n`, matching `index_avx2`.
            let a = _mm256_loadu_si256(p as *const __m256i);
            let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
            let da = _mm256_cmpeq_epi8_mask(a, comma) | _mm256_cmpeq_epi8_mask(a, newline);
            let db = _mm256_cmpeq_epi8_mask(b, comma) | _mm256_cmpeq_epi8_mask(b, newline);
            let d = (da as u64) | ((db as u64) << 32);
            let q = (_mm256_cmpeq_epi8_mask(a, quote) as u64)
                | ((_mm256_cmpeq_epi8_mask(b, quote) as u64) << 32);
            let left = end.offset_from(p) as usize;
            self.index_step::<false>(
                buf, p, left, d, q, &mut dst, &mut fdst, &mut pd, &mut pq, &mut pfs, &mut carry,
            );
            p = p.add(64);
        }
        let stray = pq & !(pfs ^ (pd >> 1));
        if stray != 0 {
            self.note(stray, p.sub(64).offset_from(buf) as u32);
        }
        self.count = dst.offset_from(ent) as usize;
        if !self.strays.is_empty() {
            self.patch(buf as u32);
        }
    }

    unsafe fn index_avx2(&mut self, buf: *const u8, n: usize) {
        let ent = self.ent.as_mut_ptr();
        let flags = self.flags.as_mut_ptr();
        let comma = _mm256_set1_epi8(b',' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let quote = _mm256_set1_epi8(b'"' as i8);
        *ent = (buf as u32).wrapping_sub(1);
        *flags = (b'\n' as u32) << 8 | (*buf as u32) << 16 | (*buf.add(1) as u32) << 24;
        self.strays.clear();
        let (mut dst, mut fdst) = (ent.add(1), flags.add(1));
        // A word's stray test needs the next word's first delimiter bit, so it
        // runs one word late on these.
        let (mut pd, mut pq, mut pfs) = (0u64, 0u64, 0u64);
        let mut carry = 1u64;
        let mut p = buf;
        let full_end = buf.add(n & !63);
        let end = buf.add(n);
        while p < full_end {
            let a = _mm256_loadu_si256(p as *const __m256i);
            let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
            let da = _mm256_or_si256(_mm256_cmpeq_epi8(a, comma), _mm256_cmpeq_epi8(a, newline));
            let db = _mm256_or_si256(_mm256_cmpeq_epi8(b, comma), _mm256_cmpeq_epi8(b, newline));
            let d = (_mm256_movemask_epi8(da) as u32 as u64) | (_mm256_movemask_epi8(db) as u32 as u64) << 32;
            let q = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, quote)) as u32 as u64)
                | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, quote)) as u32 as u64) << 32;
            self.index_step::<true>(
                buf, p, 64, d, q, &mut dst, &mut fdst, &mut pd, &mut pq, &mut pfs, &mut carry,
            );
            p = p.add(64);
        }
        if p < end {
            let a = _mm256_loadu_si256(p as *const __m256i);
            let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
            let da = _mm256_or_si256(_mm256_cmpeq_epi8(a, comma), _mm256_cmpeq_epi8(a, newline));
            let db = _mm256_or_si256(_mm256_cmpeq_epi8(b, comma), _mm256_cmpeq_epi8(b, newline));
            let d = (_mm256_movemask_epi8(da) as u32 as u64) | (_mm256_movemask_epi8(db) as u32 as u64) << 32;
            let q = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, quote)) as u32 as u64)
                | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, quote)) as u32 as u64) << 32;
            let left = end.offset_from(p) as usize;
            self.index_step::<false>(
                buf, p, left, d, q, &mut dst, &mut fdst, &mut pd, &mut pq, &mut pfs, &mut carry,
            );
            p = p.add(64);
        }
        let stray = pq & !(pfs ^ (pd >> 1));
        if stray != 0 {
            self.note(stray, p.sub(64).offset_from(buf) as u32);
        }
        self.count = dst.offset_from(ent) as usize;
        if !self.strays.is_empty() {
            self.patch(buf as u32);
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn dense(dst: *mut u32, fdst: *mut u32, p: *const u8, mut s: u64, found: usize) {
        for k in 7..found {
            let t = s.trailing_zeros();
            *dst.add(k) = (p as u32).wrapping_add(t);
            *fdst.add(k) = (p.add(t as usize).sub(1) as *const u32).read_unaligned();
            s &= s.wrapping_sub(1);
        }
    }

    /// Records quotes that are neither at a field's start nor at its end, or
    /// are both (a field that is one quote).
    #[cold]
    #[inline(never)]
    fn note(&mut self, mut stray: u64, base: u32) {
        while stray != 0 {
            self.strays.push(base + stray.trailing_zeros());
            stray &= stray - 1;
        }
    }

    /// Spoils the close byte of each stray quote's field so that it disagrees
    /// with the open byte, which makes the line malformed.
    #[cold]
    #[inline(never)]
    fn patch(&mut self, base: u32) {
        let ent = &self.ent[..self.count];
        for &pos in &self.strays {
            let i = 1 + ent[1..].partition_point(|&e| e.wrapping_sub(base) < pos);
            let open = (self.flags[i - 1] >> 16) as u8 == b'"';
            let close = if open { 0 } else { b'"' as u32 };
            self.flags[i] = (self.flags[i] & !0xff) | close;
        }
    }

    /// Calls `f(buffer, starts, ends)` for every well-formed line of `N` fields
    /// (N <= 7) that starts in `a..b` of the file, quotes already stripped. The
    /// first line of the file is the header and is never passed on. `data`
    /// holds the whole file, ends with a newline, and may be over-read by 64
    /// bytes on either side.
    pub unsafe fn scan<const N: usize>(
        &mut self,
        data: *const u8,
        size: usize,
        a: usize,
        b: usize,
        mut f: impl FnMut(*const u8, &[u32; 8], &[u32; 8]),
    ) {
        self.ensure();
        // Skipping through the first newline drops the header when a == 0 and
        // the tail of the previous range's last line otherwise.
        let mut start = a.saturating_sub(1);
        let mut skip = true;
        let quote = _mm256_set1_epi8(b'"' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let open_byte = _mm256_set1_epi32(0x00ff_0000);
        let open_quote = _mm256_set1_epi32(0x0022_0000);
        // Lane k of a line's flags is its delimiter k (0: the one before it).
        // Byte 0 is the close quote of field k - 1, byte 1 the delimiter, byte 2
        // the open quote of field k.
        let opens: u32 = (0..N).map(|k| 1u32 << (4 * k + 2)).sum();
        let newlines: u32 = (1..=N).map(|k| 1u32 << (4 * k + 1)).sum();
        let last_newline: u32 = 1 << (4 * N + 1);
        while start < size {
            let mut n = self.window.min(size - start);
            let eof = start + n == size;
            let buf = data.add(start);
            if !eof {
                let mut m = n;
                while m > 0 && *buf.add(m - 1) != b'\n' {
                    m -= 1;
                }
                if m == 0 {
                    // No newline in the window: one line longer than the window.
                    if skip {
                        start += n;
                    } else {
                        self.resize(self.window * 2);
                    }
                    continue;
                }
                n = m;
            }
            self.index(buf, n);
            let ent = self.ent.as_ptr();
            let flags = self.flags.as_ptr();
            let count = self.count;
            let is_newline = |at: usize| (*flags.add(at) >> 8) as u8 == b'\n';
            // `at` is the entry of the delimiter before the current line.
            let mut at = 0usize;
            if skip {
                at = 1;
                while !is_newline(at) {
                    at += 1;
                }
                skip = false;
            }
            let base = buf as u32;
            let first = _mm256_set1_epi32(1i32.wrapping_sub(base as i32));
            let last = _mm256_set1_epi32(base as i32);
            let line_start = |at: usize| (*ent.add(at)).wrapping_sub(base).wrapping_add(1);
            let limit = b.saturating_sub(start).min(u32::MAX as usize) as u32;
            while at + 1 < count {
                if line_start(at) >= limit {
                    return;
                }
                let fl = _mm256_loadu_si256(flags.add(at) as *const __m256i);
                let nl = _mm256_movemask_epi8(_mm256_cmpeq_epi8(fl, newline)) as u32;
                if nl & newlines != last_newline {
                    // Wrong field count: move on to the line's own newline.
                    at += 1;
                    while !is_newline(at) {
                        at += 1;
                    }
                    continue;
                }
                let q = _mm256_movemask_epi8(_mm256_cmpeq_epi8(fl, quote)) as u32;
                let line = at;
                at += N;
                if (q ^ q >> 2) & opens != 0 {
                    continue;
                }
                let before = _mm256_loadu_si256(ent.add(line) as *const __m256i);
                let after = _mm256_loadu_si256(ent.add(line + 1) as *const __m256i);
                let opened = _mm256_cmpeq_epi32(_mm256_and_si256(fl, open_byte), open_quote);
                let starts = _mm256_add_epi32(_mm256_sub_epi32(before, opened), first);
                let ends = _mm256_sub_epi32(_mm256_add_epi32(after, opened), last);
                let (mut s, mut e) = ([0u32; 8], [0u32; 8]);
                _mm256_storeu_si256(s.as_mut_ptr() as *mut __m256i, starts);
                _mm256_storeu_si256(e.as_mut_ptr() as *mut __m256i, ends);
                f(buf, &s, &e);
            }
            let next = line_start(at) as usize;
            if eof || start + next >= b {
                return;
            }
            start += next;
        }
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub unsafe fn bench_index_only(&mut self, data: *const u8, size: usize) -> usize {
        self.ensure();
        let mut start = 0usize;
        let mut total_delims = 0usize;
        while start < size {
            let mut n = self.window.min(size - start);
            let eof = start + n == size;
            let buf = data.add(start);
            if !eof {
                let mut m = n;
                while m > 0 && *buf.add(m - 1) != b'\n' {
                    m -= 1;
                }
                if m == 0 {
                    self.resize(self.window * 2);
                    continue;
                }
                n = m;
            }
            self.index(buf, n);
            total_delims += self.count;
            start += n;
        }
        total_delims
    }
}
