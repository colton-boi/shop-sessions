use std::arch::x86_64::*;

// One index entry per comma or newline: its position + 1 in the low bits, and
// what the quotes around it look like in the high bits.
const OPENS: u32 = 1 << 31; // a quote follows it, opening the next field
const CLOSES: u32 = 1 << 30; // a quote precedes it, closing its field
const NEWLINE: u32 = 1 << 29;
const STRAY: u32 = 1 << 28; // its field holds a quote that neither opens nor closes it
const POS: u32 = STRAY - 1;

/// Indexes a mapped CSV file a cache-sized window at a time and hands out the
/// field bounds of every well-formed line.
pub struct Reader {
    window: usize,
    entries: Vec<u32>,
    /// Entries up to and including the last newline of the indexed data.
    complete: usize,
}

impl Reader {
    pub fn new(window: usize) -> Self {
        let mut r = Reader { window: 0, entries: Vec::new(), complete: 0 };
        r.resize(window.max(64));
        r
    }

    fn resize(&mut self, window: usize) {
        assert!(window < POS as usize, "line too long");
        self.window = window;
        self.entries.resize(window + 128, 0);
    }

    /// Builds the entries for the `n` bytes at `buf`, which start at a line
    /// start. Entry 0 stands for the delimiter before the first byte.
    unsafe fn index(&mut self, buf: *const u8, n: usize) {
        let out = self.entries.as_mut_ptr();
        let comma = _mm256_set1_epi8(b',' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let quote = _mm256_set1_epi8(b'"' as i8);
        *out = if *buf == b'"' { OPENS } else { 0 };
        let mut count = 1usize;
        let mut complete = 0usize;
        // Carries between words: the previous byte was a delimiter, was a quote,
        // and a stray quote is still waiting for its field's delimiter.
        let (mut after_delim, mut after_quote, mut stray_carry) = (1u64, 0u64, false);
        let words = n.div_ceil(64);
        for w in 0..words {
            let p = buf.add(w * 64);
            let a = _mm256_loadu_si256(p as *const __m256i);
            let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
            let mask = |c: __m256i| {
                (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, c)) as u32 as u64)
                    | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, c)) as u32 as u64) << 32)
            };
            let (mut mc, mut mn, mut mq) = (mask(comma), mask(newline), mask(quote));
            let next = *p.add(64);
            let (mut quote_next, mut delim_next) = ((next == b'"') as u64, (next == b',' || next == b'\n') as u64);
            if w == words - 1 {
                if n % 64 != 0 {
                    let keep = (1u64 << (n % 64)) - 1;
                    mc &= keep;
                    mn &= keep;
                    mq &= keep;
                }
                quote_next = 0;
                delim_next = 0;
            }
            let delims = mc | mn;
            let opening = mq & (delims << 1 | after_delim);
            let closing = mq & (delims >> 1 | delim_next << 63);
            let stray = mq & !(opening | closing);
            // Adding a stray quote into the run of non-delimiters above it carries
            // up to the run's end: the delimiter that closes the quote's field.
            let (sum, c1) = (!delims).overflowing_add(stray);
            let (sum, c2) = sum.overflowing_add(stray_carry as u64);
            stray_carry = c1 | c2;
            let f_open = delims & (mq >> 1 | quote_next << 63);
            let f_close = delims & (mq << 1 | after_quote);
            let f_stray = delims & sum;
            after_delim = delims >> 63;
            after_quote = mq >> 63;

            let flags = _pdep_u64(_pext_u64(f_open, delims), 0x8080_8080_8080_8080)
                | _pdep_u64(_pext_u64(f_close, delims), 0x4040_4040_4040_4040)
                | _pdep_u64(_pext_u64(mn, delims), 0x2020_2020_2020_2020)
                | _pdep_u64(_pext_u64(f_stray, delims), 0x1010_1010_1010_1010);
            let base = (w * 64) as u32 + 1;
            let found = delims.count_ones() as usize;
            let dst = out.add(count);
            let mut s = delims;
            for k in 0..8 {
                *dst.add(k) = base + s.trailing_zeros();
                s &= s.wrapping_sub(1);
            }
            let wide = _mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(flags as i64)), 24);
            let d = dst as *mut __m256i;
            _mm256_storeu_si256(d, _mm256_or_si256(_mm256_loadu_si256(d), wide));
            if found > 8 {
                Self::index_dense(dst, base, delims, [f_open, f_close, mn, f_stray]);
            }
            if mn != 0 {
                let last = 63 - mn.leading_zeros();
                complete = count + (delims & (u64::MAX >> (63 - last))).count_ones() as usize;
            }
            count += found;
        }
        self.complete = complete;
    }

    #[cold]
    #[inline(never)]
    unsafe fn index_dense(dst: *mut u32, base: u32, delims: u64, flags: [u64; 4]) {
        let mut s = delims;
        let mut k = 0;
        while s != 0 {
            let bit = s & s.wrapping_neg();
            let mut e = base + s.trailing_zeros();
            for (f, flag) in flags.iter().zip([OPENS, CLOSES, NEWLINE, STRAY]) {
                if f & bit != 0 {
                    e |= flag;
                }
            }
            *dst.add(k) = e;
            k += 1;
            s ^= bit;
        }
    }

    /// Calls `f(buffer, starts, ends)` for every well-formed line of `N` fields
    /// (N <= 8) that starts in `a..b` of the file, quotes already stripped. The
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
        // Skipping through the first newline drops the header when a == 0 and
        // the tail of the previous range's last line otherwise.
        let mut start = a.saturating_sub(1);
        let mut skip = true;
        let fields = (1u32 << N) - 1;
        while start < size {
            let n = self.window.min(size - start);
            let eof = start + n == size;
            let buf = data.add(start);
            self.index(buf, n);
            let entries = self.entries.as_ptr();
            let complete = self.complete;
            if complete == 0 {
                // No newline in the window: one line longer than the window.
                if skip {
                    start += n;
                } else {
                    self.resize(self.window * 2);
                }
                continue;
            }
            // `at` is the entry of the delimiter before the current line.
            let mut at = 0usize;
            if skip {
                at = 1;
                while *entries.add(at) & NEWLINE == 0 {
                    at += 1;
                }
                skip = false;
            }
            let limit = b.saturating_sub(start).min(u32::MAX as usize) as u32;
            let pos = _mm256_set1_epi32(POS as i32);
            let one = _mm256_set1_epi32(1);
            let mut line_start = *entries.add(at) & POS;
            while at + 1 < complete {
                if line_start >= limit {
                    return;
                }
                let before = _mm256_loadu_si256(entries.add(at) as *const __m256i);
                let after = _mm256_loadu_si256(entries.add(at + 1) as *const __m256i);
                let bits = |v: __m256i, shift: i32| {
                    _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_sll_epi32(v, _mm_cvtsi32_si128(shift)))) as u32
                };
                let newlines = bits(after, 2);
                if newlines & fields != 1 << (N - 1) {
                    // Wrong field count: move on to the line's own newline.
                    at += 1;
                    while *entries.add(at) & NEWLINE == 0 {
                        at += 1;
                    }
                    line_start = *entries.add(at) & POS;
                    continue;
                }
                let opens = _mm256_srli_epi32(before, 31);
                let closes = _mm256_and_si256(_mm256_srli_epi32(after, 30), one);
                let starts = _mm256_add_epi32(_mm256_and_si256(before, pos), opens);
                let ends = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_and_si256(after, pos), one), closes);
                // Bad when a field has a stray quote, an unmatched one, or is a lone quote.
                let bad = bits(after, 3)
                    | !(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(opens, closes))) as u32)
                    | _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(starts, ends))) as u32;
                at += N;
                line_start = *entries.add(at) & POS;
                if bad & fields == 0 {
                    let (mut s, mut e) = ([0u32; 8], [0u32; 8]);
                    _mm256_storeu_si256(s.as_mut_ptr() as *mut __m256i, starts);
                    _mm256_storeu_si256(e.as_mut_ptr() as *mut __m256i, ends);
                    f(buf, &s, &e);
                }
            }
            if eof || line_start as usize + start >= b {
                return;
            }
            start += line_start as usize;
        }
    }
}
