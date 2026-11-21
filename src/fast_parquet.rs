use std::arch::x86_64::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub const NULL_CAT: u8 = 0xFF;

const TYPE_INT32: i32 = 1;
const TYPE_BYTE_ARRAY: i32 = 6;

const REP_REQUIRED: i32 = 0;
const REP_OPTIONAL: i32 = 1;

const P_DATA: i32 = 0;
const P_DICT: i32 = 2;

const E_PLAIN: i32 = 0;
const E_RLE: i32 = 3;
const E_RLE_DICT: i32 = 8;

const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BINARY: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;

/// Columnar view over a batch of `user_daily` rows.
///
/// - `names_data` and `names_offsets` store unique person names (length `num_persons + 1`,
///   where person `k`'s name is `names_data[names_offsets[k]..names_offsets[k + 1]]`).
/// - `person_keys` holds the person index `k` for each row `0..n` (monotonically non-decreasing
///   before zero-revenue splitting, or arbitrary indices within `lo_person..=hi_person`).
/// - `day`, `sessions`, `events`, `revenue_cents` hold the per-row `i32` values.
/// - `top_category` holds the category index `0..num_cats` or `0xFF` (`NULL_CAT`) for null.
/// - `cat_dict_data` and `cat_dict_offsets` store the category strings (`0..num_cats`).
#[derive(Clone, Copy)]
pub struct UserDailyCols<'a> {
    pub names_data: &'a [u8],
    pub names_offsets: &'a [u32],
    pub person_keys: &'a [u32],
    pub day: &'a [i32],
    pub sessions: &'a [i32],
    pub events: &'a [i32],
    pub revenue_cents: &'a [i32],
    pub top_category: &'a [u8],
    pub cat_dict_data: &'a [u8],
    pub cat_dict_offsets: &'a [u32],
}

impl<'a> UserDailyCols<'a> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.day.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.day.is_empty()
    }

    #[inline(always)]
    pub fn slice(&self, from: usize, to: usize) -> Self {
        UserDailyCols {
            names_data: self.names_data,
            names_offsets: self.names_offsets,
            person_keys: &self.person_keys[from..to],
            day: &self.day[from..to],
            sessions: &self.sessions[from..to],
            events: &self.events[from..to],
            revenue_cents: &self.revenue_cents[from..to],
            top_category: &self.top_category[from..to],
            cat_dict_data: self.cat_dict_data,
            cat_dict_offsets: self.cat_dict_offsets,
        }
    }
}

/// Reusable scratch buffers for zero-revenue row-group splitting.
#[derive(Default)]
pub struct SplitScratch {
    pub zero_names_data: Vec<u8>,
    pub zero_names_offsets: Vec<u32>,
    pub zero_person: Vec<u32>,
    pub zero_day: Vec<i32>,
    pub zero_sessions: Vec<i32>,
    pub zero_events: Vec<i32>,
    pub zero_revenue: Vec<i32>,
    pub zero_category: Vec<u8>,

    pub nonz_names_data: Vec<u8>,
    pub nonz_names_offsets: Vec<u32>,
    pub nonz_person: Vec<u32>,
    pub nonz_day: Vec<i32>,
    pub nonz_sessions: Vec<i32>,
    pub nonz_events: Vec<i32>,
    pub nonz_revenue: Vec<i32>,
    pub nonz_category: Vec<u8>,
}

impl SplitScratch {
    pub fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self, cap: usize) {
        self.zero_names_data.clear();
        self.zero_names_offsets.clear();
        self.zero_names_offsets.push(0);
        self.zero_person.clear();
        self.zero_day.clear();
        self.zero_sessions.clear();
        self.zero_events.clear();
        self.zero_revenue.clear();
        self.zero_category.clear();

        self.nonz_names_data.clear();
        self.nonz_names_offsets.clear();
        self.nonz_names_offsets.push(0);
        self.nonz_person.clear();
        self.nonz_day.clear();
        self.nonz_sessions.clear();
        self.nonz_events.clear();
        self.nonz_revenue.clear();
        self.nonz_category.clear();

        self.zero_person.reserve(cap);
        self.zero_day.reserve(cap);
        self.zero_sessions.reserve(cap);
        self.zero_events.reserve(cap);
        self.zero_revenue.reserve(cap);
        self.zero_category.reserve(cap);

        self.nonz_person.reserve(cap);
        self.nonz_day.reserve(cap);
        self.nonz_sessions.reserve(cap);
        self.nonz_events.reserve(cap);
        self.nonz_revenue.reserve(cap);
        self.nonz_category.reserve(cap);
    }
}

/// Compact Thrift encoder writing directly into a caller-owned byte buffer.
struct ThriftWriter<'a> {
    buf: &'a mut Vec<u8>,
    last_id: i16,
    stack: [i16; 8],
    depth: usize,
}

impl<'a> ThriftWriter<'a> {
    #[inline(always)]
    fn new(buf: &'a mut Vec<u8>) -> Self {
        ThriftWriter { buf, last_id: 0, stack: [0; 8], depth: 0 }
    }

    #[inline(always)]
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }

    #[inline(always)]
    fn zigzag(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }

    #[inline(always)]
    fn field(&mut self, id: i16, ty: u8) {
        let delta = id - self.last_id;
        if (1..=15).contains(&delta) {
            self.buf.push(((delta as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            self.zigzag(id as i64);
        }
        self.last_id = id;
    }

    #[inline(always)]
    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, T_I32);
        self.zigzag(v as i64);
    }

    #[inline(always)]
    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, T_I64);
        self.zigzag(v);
    }

    #[inline(always)]
    fn binary(&mut self, id: i16, v: &[u8]) {
        self.field(id, T_BINARY);
        self.varint(v.len() as u64);
        self.buf.extend_from_slice(v);
    }

    #[inline(always)]
    fn list(&mut self, id: i16, elem_ty: u8, len: usize) {
        self.field(id, T_LIST);
        if len < 15 {
            self.buf.push(((len as u8) << 4) | elem_ty);
        } else {
            self.buf.push(0xF0 | elem_ty);
            self.varint(len as u64);
        }
    }

    #[inline(always)]
    fn begin_struct(&mut self) {
        debug_assert!(self.depth < self.stack.len());
        self.stack[self.depth] = self.last_id;
        self.depth += 1;
        self.last_id = 0;
    }

    #[inline(always)]
    fn begin_field(&mut self, id: i16) {
        self.field(id, T_STRUCT);
        self.begin_struct();
    }

    #[inline(always)]
    fn end_struct(&mut self) {
        self.buf.push(0);
        if self.depth > 0 {
            self.depth -= 1;
            self.last_id = self.stack[self.depth];
        } else {
            self.last_id = 0;
        }
    }
}

#[inline(always)]
fn push_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline(always)]
fn varint_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

#[inline(always)]
pub fn bit_width(max_val: u32) -> u8 {
    if max_val == 0 {
        1
    } else {
        (32 - max_val.leading_zeros()) as u8
    }
}

/// SWAR-packs 8 byte lanes in `x` (each `< (1 << w)`, `1 <= w <= 8`) into the low `8 * w` bits
/// of a `u64`, LSB-first.
#[inline(always)]
fn swar_pack8_u8(x: u64, w: usize) -> u64 {
    let lo1 = x & 0x00FF_00FF_00FF_00FF;
    let hi1 = (x >> 8) & 0x00FF_00FF_00FF_00FF;
    let p1 = lo1 | (hi1 << w);

    let lo2 = p1 & 0x0000_FFFF_0000_FFFF;
    let hi2 = (p1 >> 16) & 0x0000_FFFF_0000_FFFF;
    let p2 = lo2 | (hi2 << (2 * w));

    let lo3 = p2 & 0xFFFF_FFFF;
    let hi3 = p2 >> 32;
    lo3 | (hi3 << (4 * w))
}

/// Bit-packs `values: &[u8]` (each `< (1 << width)`, `1 <= width <= 8`) into Parquet
/// RLE/Bit-Packing Hybrid bit-packed runs (groups of 8 values, LSB-first).
/// Appends the bit-packed header varint `((num_groups << 1) | 1)` and packed bytes to `out`.
pub fn encode_bitpacked_u8(values: &[u8], width: u8, out: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let w = width as usize;
    let num_groups = values.len().div_ceil(8);
    push_varint(out, ((num_groups as u64) << 1) | 1);
    if w == 0 {
        return;
    }
    debug_assert!(w <= 8);
    let total_bytes = num_groups * w;
    out.reserve(total_bytes + 8);
    let start_len = out.len();
    let src = values.as_ptr();
    let mut dst = unsafe { out.as_mut_ptr().add(start_len) };
    let full_groups = values.len() / 8;
    for g in 0..full_groups {
        let x = unsafe { (src.add(g * 8) as *const u64).read_unaligned().to_le() };
        let packed = swar_pack8_u8(x, w);
        unsafe {
            (dst as *mut u64).write_unaligned(packed.to_le());
            dst = dst.add(w);
        }
    }
    let rem = values.len() - full_groups * 8;
    if rem > 0 {
        let mut tail = [0u8; 8];
        unsafe {
            std::ptr::copy_nonoverlapping(src.add(full_groups * 8), tail.as_mut_ptr(), rem);
            let x = u64::from_le_bytes(tail);
            let packed = swar_pack8_u8(x, w);
            (dst as *mut u64).write_unaligned(packed.to_le());
        }
    }
    unsafe {
        out.set_len(start_len + total_bytes);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn encode_bitpacked_u32_w8_avx2(values: &[u32], width: usize, out: &mut Vec<u8>) {
    let num_groups = values.len().div_ceil(8);
    let total_bytes = num_groups * width;
    out.reserve(total_bytes + 8);
    let start_len = out.len();
    let src = values.as_ptr();
    let mut dst = out.as_mut_ptr().add(start_len);
    let zero = _mm256_setzero_si256();
    let zero128 = _mm_setzero_si128();
    let full_groups = values.len() / 8;
    for g in 0..full_groups {
        let v = _mm256_loadu_si256(src.add(g * 8) as *const __m256i);
        let p16 = _mm256_packus_epi32(v, zero);
        let p16_ord = _mm256_permute4x64_epi64(p16, 0xD8);
        let p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16_ord), zero128);
        let x = (_mm_cvtsi128_si64(p8) as u64).to_le();
        let packed = swar_pack8_u8(x, width);
        (dst as *mut u64).write_unaligned(packed.to_le());
        dst = dst.add(width);
    }
    let rem = values.len() - full_groups * 8;
    if rem > 0 {
        let mut x = 0u64;
        for i in 0..rem {
            x |= (*src.add(full_groups * 8 + i) as u64) << (i * 8);
        }
        let packed = swar_pack8_u8(x, width);
        (dst as *mut u64).write_unaligned(packed.to_le());
    }
    out.set_len(start_len + total_bytes);
}

/// Bit-packs `values` (each `< (1 << width)`) into Parquet RLE/Bit-Packing Hybrid bit-packed runs
/// (groups of 8 values, LSB-first). Appends the bit-packed header varint `((num_groups << 1) | 1)`
/// and the packed bytes to `out`.
pub fn encode_bitpacked_u32(values: &[u32], width: u8, out: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let w = width as usize;
    let num_groups = values.len().div_ceil(8);
    push_varint(out, ((num_groups as u64) << 1) | 1);
    if w == 0 {
        return;
    }
    if w <= 8 {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                encode_bitpacked_u32_w8_avx2(values, w, out);
            }
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let total_bytes = num_groups * w;
            out.reserve(total_bytes + 8);
            let start_len = out.len();
            let mut dst = unsafe { out.as_mut_ptr().add(start_len) };
            for g in 0..num_groups {
                let base = g * 8;
                let rem = (values.len() - base).min(8);
                let mut x = 0u64;
                for i in 0..rem {
                    x |= (unsafe { *values.get_unchecked(base + i) } as u64) << (i * 8);
                }
                let packed = swar_pack8_u8(x, w);
                unsafe {
                    (dst as *mut u64).write_unaligned(packed.to_le());
                    dst = dst.add(w);
                }
            }
            unsafe {
                out.set_len(start_len + total_bytes);
            }
            return;
        }
    }

    let total_bytes = num_groups * w;
    out.reserve(total_bytes + 8);
    let start_len = out.len();
    let mut dst = unsafe { out.as_mut_ptr().add(start_len) };
    for g in 0..num_groups {
        let base = g * 8;
        let rem = values.len() - base;
        let mut vals = [0u64; 8];
        if rem >= 8 {
            for i in 0..8 {
                vals[i] = unsafe { *values.get_unchecked(base + i) as u64 };
            }
        } else {
            for i in 0..rem {
                vals[i] = values[base + i] as u64;
            }
        }
        let mut bit_buf = 0u128;
        let mut bits_in_buf = 0usize;
        for i in 0..8 {
            bit_buf |= (vals[i] as u128) << bits_in_buf;
            bits_in_buf += w;
            while bits_in_buf >= 64 {
                unsafe {
                    (dst as *mut u64).write_unaligned((bit_buf as u64).to_le());
                    dst = dst.add(8);
                }
                bit_buf >>= 64;
                bits_in_buf -= 64;
            }
        }
        if bits_in_buf > 0 {
            debug_assert_eq!(bits_in_buf % 8, 0);
            let nbytes = bits_in_buf / 8;
            let le = (bit_buf as u64).to_le_bytes();
            unsafe {
                std::ptr::copy_nonoverlapping(le.as_ptr(), dst, nbytes);
                dst = dst.add(nbytes);
            }
        }
    }
    unsafe {
        out.set_len(start_len + total_bytes);
    }
}

/// Scans forward from `start` while `values[j] == target`, checking 8 `u32`s per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
#[inline]
unsafe fn scan_run_len_avx2(ptr: *const u32, start: usize, n: usize, target: u32) -> usize {
    let tvec = _mm256_set1_epi32(target as i32);
    let mut j = start;
    while j + 8 <= n {
        let v = _mm256_loadu_si256(ptr.add(j) as *const __m256i);
        let eq = _mm256_cmpeq_epi32(v, tvec);
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask != 0xFFFF_FFFF {
            return j - start + ((!mask).trailing_zeros() >> 2) as usize;
        }
        j += 8;
    }
    while j < n && *ptr.add(j) == target {
        j += 1;
    }
    j - start
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn scan_run_len_u64(values: &[u32], start: usize, target: u32) -> usize {
    let n = values.len();
    let ptr = values.as_ptr();
    let target64 = (target as u64) | ((target as u64) << 32);
    let mut j = start;
    while j + 8 <= n {
        let a = unsafe { (ptr.add(j) as *const u64).read_unaligned() };
        let b = unsafe { (ptr.add(j + 2) as *const u64).read_unaligned() };
        let c = unsafe { (ptr.add(j + 4) as *const u64).read_unaligned() };
        let d = unsafe { (ptr.add(j + 6) as *const u64).read_unaligned() };
        if (a ^ target64) | (b ^ target64) | (c ^ target64) | (d ^ target64) != 0 {
            break;
        }
        j += 8;
    }
    while j < n && unsafe { *ptr.add(j) } == target {
        j += 1;
    }
    j - start
}

#[inline(always)]
fn emit_rle_run(val: u32, run_len: usize, val_bytes: usize, out: &mut Vec<u8>) {
    push_varint(out, (run_len as u64) << 1);
    let le = val.to_le_bytes();
    out.extend_from_slice(&le[..val_bytes]);
}

/// Encodes `values` (each `< (1 << width)`) into Parquet RLE/Bit-Packing Hybrid runs
/// (without the leading 1-byte `width`), using vectorized 8-wide run detection.
///
/// Long runs (`>= 8` elements, or `>= 16` mid-bitpack at `width == 1`) are emitted as
/// RLE runs, while non-run regions are coalesced into bit-packed runs.
pub fn encode_rle_u32(values: &[u32], width: u8, out: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    if width == 0 {
        push_varint(out, (values.len() as u64) << 1);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { encode_rle_u32_avx2(values, width, out) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        encode_rle_u32_scalar(values, width, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn encode_rle_u32_avx2(values: &[u32], width: u8, out: &mut Vec<u8>) {
    let n = values.len();
    let ptr = values.as_ptr();
    let val_bytes = (width as usize).div_ceil(8);
    let mut i = 0usize;
    let mut lit_start = 0usize;

    while i + 8 <= n {
        let v = *ptr.add(i);
        let tvec = _mm256_set1_epi32(v as i32);
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let eq = _mm256_cmpeq_epi32(chunk, tvec);
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask == 0xFFFF_FFFF {
            let run_len = 8 + scan_run_len_avx2(ptr, i + 8, n, v);
            if width == 1 && lit_start < i && run_len < 16 {
                i += 8;
            } else {
                if lit_start < i {
                    encode_bitpacked_u32(&values[lit_start..i], width, out);
                }
                emit_rle_run(v, run_len, val_bytes, out);
                i += run_len;
                lit_start = i;
            }
        } else {
            i += 8;
        }
    }

    if lit_start < i {
        encode_bitpacked_u32(&values[lit_start..n], width, out);
    } else if i < n {
        let v = *ptr.add(i);
        let mut all_eq = true;
        for j in (i + 1)..n {
            if *ptr.add(j) != v {
                all_eq = false;
                break;
            }
        }
        if all_eq {
            emit_rle_run(v, n - i, val_bytes, out);
        } else {
            encode_bitpacked_u32(&values[i..n], width, out);
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn encode_rle_u32_scalar(values: &[u32], width: u8, out: &mut Vec<u8>) {
    let n = values.len();
    let val_bytes = (width as usize).div_ceil(8);
    let mut i = 0usize;
    let mut lit_start = 0usize;

    while i + 8 <= n {
        let v = values[i];
        let run_len = scan_run_len_u64(values, i, v);
        if run_len >= 8 && !(width == 1 && lit_start < i && run_len < 16) {
            if lit_start < i {
                encode_bitpacked_u32(&values[lit_start..i], width, out);
            }
            emit_rle_run(v, run_len, val_bytes, out);
            i += run_len;
            lit_start = i;
        } else {
            i += 8;
        }
    }

    if lit_start < i {
        encode_bitpacked_u32(&values[lit_start..n], width, out);
    } else if i < n {
        let v = values[i];
        if values[i + 1..n].iter().all(|&x| x == v) {
            emit_rle_run(v, n - i, val_bytes, out);
        } else {
            encode_bitpacked_u32(&values[i..n], width, out);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn count_runs_u32_avx2(keys: &[u32]) -> usize {
    let n = keys.len();
    if n <= 1 {
        return n;
    }
    let ptr = keys.as_ptr();
    let mut runs = 1usize;
    let mut i = 1usize;
    while i + 8 <= n {
        let cur = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let prev = _mm256_loadu_si256(ptr.add(i - 1) as *const __m256i);
        let eq = _mm256_cmpeq_epi32(cur, prev);
        let eq_mask = _mm256_movemask_ps(_mm256_castsi256_ps(eq)) as u32;
        runs += ((!eq_mask) & 0xFF).count_ones() as usize;
        i += 8;
    }
    while i < n {
        if *ptr.add(i) != *ptr.add(i - 1) {
            runs += 1;
        }
        i += 1;
    }
    runs
}

/// Subtracts `base` from each element in `src` and stores `(src[i] - base) as u32` in `dst`.
#[inline]
fn sub_min_to_u32(src: &[i32], base: i32, dst: &mut Vec<u32>) {
    let n = src.len();
    dst.clear();
    dst.reserve(n);
    #[cfg(target_arch = "x86_64")]
    unsafe {
        sub_min_to_u32_avx2(src.as_ptr(), base, dst.as_mut_ptr(), n);
        dst.set_len(n);
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        dst.extend(src.iter().map(|&v| (v - base) as u32));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn sub_min_to_u32_avx2(src: *const i32, base: i32, dst: *mut u32, n: usize) {
    let bvec = _mm256_set1_epi32(base);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
        let d = _mm256_sub_epi32(v, bvec);
        _mm256_storeu_si256(dst.add(i) as *mut __m256i, d);
        i += 8;
    }
    while i < n {
        *dst.add(i) = (*src.add(i) - base) as u32;
        i += 1;
    }
}

/// Encodes `keys` (relative to `base_key`) using Parquet RLE/Bit-Packing Hybrid (`RLE_DICTIONARY`
/// payload after the 1-byte bit-width). Chooses the smaller of pure RLE runs and pure bit-packing.
pub fn encode_rle_or_bitpack_keys(keys: &[u32], base_key: u32, width: u8, out: &mut Vec<u8>) {
    out.push(width);
    if keys.is_empty() {
        return;
    }
    let val_bytes = (width as usize).div_ceil(8);
    #[cfg(target_arch = "x86_64")]
    let runs = unsafe { count_runs_u32_avx2(keys) };
    #[cfg(not(target_arch = "x86_64"))]
    let runs = {
        let mut r = 1usize;
        for i in 1..keys.len() {
            if keys[i] != keys[i - 1] {
                r += 1;
            }
        }
        r
    };
    let rle_est = runs * (1 + val_bytes);
    let bp_groups = keys.len().div_ceil(8);
    let bp_est = 2 + bp_groups * (width as usize);

    if rle_est <= bp_est {
        let mut i = 0;
        let n = keys.len();
        let ptr = keys.as_ptr();
        while i < n {
            let k = unsafe { *ptr.add(i) };
            #[cfg(target_arch = "x86_64")]
            let run_len = 1 + unsafe { scan_run_len_avx2(ptr, i + 1, n, k) };
            #[cfg(not(target_arch = "x86_64"))]
            let run_len = scan_run_len_u64(keys, i, k);
            emit_rle_run(k - base_key, run_len, val_bytes, out);
            i += run_len;
        }
    } else {
        let num_groups = keys.len().div_ceil(8);
        push_varint(out, ((num_groups as u64) << 1) | 1);
        let w = width as usize;
        let total_bytes = num_groups * w;
        out.reserve(total_bytes + 8);
        let start_len = out.len();
        let mut dst = unsafe { out.as_mut_ptr().add(start_len) };
        for g in 0..num_groups {
            let base = g * 8;
            let rem = keys.len() - base;
            let mut vals = [0u64; 8];
            if rem >= 8 {
                for j in 0..8 {
                    vals[j] = (unsafe { *keys.get_unchecked(base + j) } - base_key) as u64;
                }
            } else {
                for j in 0..rem {
                    vals[j] = (keys[base + j] - base_key) as u64;
                }
            }
            if w <= 8 {
                let mut acc = 0u64;
                for j in 0..8 {
                    acc |= vals[j] << (j * w);
                }
                unsafe {
                    (dst as *mut u64).write_unaligned(acc.to_le());
                    dst = dst.add(w);
                }
            } else {
                let mut bit_buf = 0u128;
                let mut bits_in_buf = 0usize;
                for j in 0..8 {
                    bit_buf |= (vals[j] as u128) << bits_in_buf;
                    bits_in_buf += w;
                    while bits_in_buf >= 64 {
                        unsafe {
                            (dst as *mut u64).write_unaligned((bit_buf as u64).to_le());
                            dst = dst.add(8);
                        }
                        bit_buf >>= 64;
                        bits_in_buf -= 64;
                    }
                }
                if bits_in_buf > 0 {
                    let nbytes = bits_in_buf / 8;
                    let le = (bit_buf as u64).to_le_bytes();
                    unsafe {
                        std::ptr::copy_nonoverlapping(le.as_ptr(), dst, nbytes);
                        dst = dst.add(nbytes);
                    }
                }
            }
        }
        unsafe {
            out.set_len(start_len + total_bytes);
        }
    }
}

/// Fast check whether `cats` contains at least one non-null (`!= NULL_CAT`) byte.
#[inline]
fn has_non_null_u8(cats: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { has_non_null_u8_avx2(cats) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        cats.iter().any(|&c| c != NULL_CAT)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn has_non_null_u8_avx2(cats: &[u8]) -> bool {
    let n = cats.len();
    let ptr = cats.as_ptr();
    let null_vec = _mm256_set1_epi8(-1);
    let mut i = 0usize;
    while i + 32 <= n {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let eq = _mm256_cmpeq_epi8(v, null_vec);
        if _mm256_movemask_epi8(eq) != -1 {
            return true;
        }
        i += 32;
    }
    while i < n {
        if *ptr.add(i) != NULL_CAT {
            return true;
        }
        i += 1;
    }
    false
}

/// Encodes the `top_category` data-page payload (1-bit bit-packed definition levels followed by
/// `w`-bit bit-packed non-null `u8` dictionary indices) into `out`.
pub fn encode_top_category_chunk(
    cats: &[u8],
    num_cats: usize,
    cat_u8_buf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) {
    let n = cats.len();
    let num_groups = n.div_ceil(8);
    let def_hdr = ((num_groups as u64) << 1) | 1;
    let def_block_len = varint_len(def_hdr) + num_groups;

    out.extend_from_slice(&(def_block_len as u32).to_le_bytes());
    push_varint(out, def_hdr);
    out.reserve(num_groups + 4);

    cat_u8_buf.clear();
    cat_u8_buf.reserve(n);

    let def_start = out.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        encode_top_category_defs_avx2(
            cats.as_ptr(),
            n,
            out.as_mut_ptr().add(def_start),
            cat_u8_buf,
        );
        out.set_len(def_start + num_groups);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for g in 0..num_groups {
            let base = g * 8;
            let rem = (n - base).min(8);
            let mut mask = 0u8;
            for b in 0..rem {
                let c = cats[base + b];
                if c != NULL_CAT {
                    mask |= 1 << b;
                    cat_u8_buf.push(c);
                }
            }
            out.push(mask);
        }
    }

    let w = bit_width(num_cats.saturating_sub(1) as u32).max(5);
    out.push(w);
    encode_bitpacked_u8(cat_u8_buf, w, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn encode_top_category_defs_avx2(
    cats_ptr: *const u8,
    n: usize,
    mut def_dst: *mut u8,
    cat_u8_buf: &mut Vec<u8>,
) {
    let null_vec = _mm256_set1_epi8(-1);
    let mut cat_dst = cat_u8_buf.as_mut_ptr();
    let mut i = 0usize;
    while i + 32 <= n {
        let v = _mm256_loadu_si256(cats_ptr.add(i) as *const __m256i);
        let is_null = _mm256_cmpeq_epi8(v, null_vec);
        let present = !(_mm256_movemask_epi8(is_null) as u32);
        (def_dst as *mut u32).write_unaligned(present.to_le());
        def_dst = def_dst.add(4);
        if present == 0xFFFF_FFFF {
            _mm256_storeu_si256(cat_dst as *mut __m256i, v);
            cat_dst = cat_dst.add(32);
        } else if present != 0 {
            let mut m = present;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                *cat_dst = *cats_ptr.add(i + b);
                cat_dst = cat_dst.add(1);
                m &= m - 1;
            }
        }
        i += 32;
    }
    while i < n {
        let rem = (n - i).min(8);
        let mut mask = 0u8;
        for b in 0..rem {
            let c = *cats_ptr.add(i + b);
            if c != NULL_CAT {
                mask |= 1 << b;
                *cat_dst = c;
                cat_dst = cat_dst.add(1);
            }
        }
        *def_dst = mask;
        def_dst = def_dst.add(1);
        i += 8;
    }
    let cat_len = cat_dst.offset_from(cat_u8_buf.as_ptr()) as usize;
    cat_u8_buf.set_len(cat_len);
}

/// Checks whether a slice of `i32` has identical min and max (and returns `(min, max)`).
/// Uses 256-bit AVX2 on x86_64 (avoiding 512-bit `zmm` frequency downclocking).
#[inline(always)]
pub fn min_max_i32(slice: &[i32]) -> (i32, i32) {
    debug_assert!(!slice.is_empty());
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { min_max_i32_avx2(slice) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut lo = slice[0];
        let mut hi = slice[0];
        for &v in &slice[1..] {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        (lo, hi)
    }
}

#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn min_max_i32_avx2(slice: &[i32]) -> (i32, i32) {
    let n = slice.len();
    let ptr = slice.as_ptr();
    let mut i = 0usize;
    let first = *ptr;
    let mut vmin = _mm256_set1_epi32(first);
    let mut vmax = vmin;
    while i + 8 <= n {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        vmin = _mm256_min_epi32(vmin, v);
        vmax = _mm256_max_epi32(vmax, v);
        i += 8;
    }
    let mut tmp_min = [0i32; 8];
    let mut tmp_max = [0i32; 8];
    _mm256_storeu_si256(tmp_min.as_mut_ptr() as *mut __m256i, vmin);
    _mm256_storeu_si256(tmp_max.as_mut_ptr() as *mut __m256i, vmax);
    let mut lo = first;
    let mut hi = first;
    for k in 0..8 {
        if tmp_min[k] < lo {
            lo = tmp_min[k];
        }
        if tmp_max[k] > hi {
            hi = tmp_max[k];
        }
    }
    while i < n {
        let v = *ptr.add(i);
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
        i += 1;
    }
    (lo, hi)
}

/// Computes an 8-bit mask of rows in `0..len` (`len <= 8`) that have `rev == 0 && cat == 0xFF`.
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn zero_rev_mask8_avx2(rev: *const i32, cat: *const u8, len: usize) -> u8 {
    if len == 8 {
        let r = _mm256_loadu_si256(rev as *const __m256i);
        let r_eq = _mm256_cmpeq_epi32(r, _mm256_setzero_si256());
        let r_mask = _mm256_movemask_ps(_mm256_castsi256_ps(r_eq)) as u8;
        let c_u64 = (cat as *const u64).read_unaligned();
        let c_vec = _mm_set1_epi64x(c_u64 as i64);
        let c_eq = _mm_cmpeq_epi8(c_vec, _mm_set1_epi8(-1));
        let c_mask = (_mm_movemask_epi8(c_eq) as u16 & 0xFF) as u8;
        r_mask & c_mask
    } else {
        let mut m = 0u8;
        for i in 0..len {
            if *rev.add(i) == 0 && *cat.add(i) == NULL_CAT {
                m |= 1 << i;
            }
        }
        m
    }
}

/// Compacts `person_keys` in place into dense `0..num_unique` keys and populates `out_names_data`
/// and `out_names_offsets` with only the person names actually referenced in `person_keys`.
fn compact_person_dict(
    src_names_data: &[u8],
    src_names_offsets: &[u32],
    person_keys: &mut [u32],
    out_names_data: &mut Vec<u8>,
    out_names_offsets: &mut Vec<u32>,
) {
    out_names_data.clear();
    out_names_offsets.clear();
    out_names_offsets.push(0);
    if person_keys.is_empty() {
        return;
    }
    let mut prev_src = u32::MAX;
    let mut cur_dense = 0u32;
    for k in person_keys.iter_mut() {
        let src_k = *k;
        if src_k != prev_src {
            prev_src = src_k;
            cur_dense = (out_names_offsets.len() - 1) as u32;
            let s = src_names_offsets[src_k as usize] as usize;
            let e = src_names_offsets[src_k as usize + 1] as usize;
            out_names_data.extend_from_slice(&src_names_data[s..e]);
            out_names_offsets.push(out_names_data.len() as u32);
        }
        *k = cur_dense;
    }
}

/// Partitions `cols` into two `UserDailyCols` views inside `scratch`:
/// 1. `zero_rev`: rows where `revenue_cents == 0 && top_category == 0xFF`
/// 2. `non_zero`: rows where `revenue_cents != 0 || top_category != 0xFF`
///
/// Uses 256-bit AVX2 (avoiding 512-bit `zmm` frequency downclocking).
pub fn split_zero_revenue_chunks<'a>(
    cols: &UserDailyCols<'a>,
    scratch: &'a mut SplitScratch,
) -> (UserDailyCols<'a>, UserDailyCols<'a>) {
    let n = cols.len();
    scratch.prepare(n);
    unsafe { split_zero_revenue_avx2(cols, scratch) };
    compact_person_dict(
        cols.names_data,
        cols.names_offsets,
        &mut scratch.zero_person,
        &mut scratch.zero_names_data,
        &mut scratch.zero_names_offsets,
    );
    compact_person_dict(
        cols.names_data,
        cols.names_offsets,
        &mut scratch.nonz_person,
        &mut scratch.nonz_names_data,
        &mut scratch.nonz_names_offsets,
    );
    let zero = UserDailyCols {
        names_data: &scratch.zero_names_data,
        names_offsets: &scratch.zero_names_offsets,
        person_keys: &scratch.zero_person,
        day: &scratch.zero_day,
        sessions: &scratch.zero_sessions,
        events: &scratch.zero_events,
        revenue_cents: &scratch.zero_revenue,
        top_category: &scratch.zero_category,
        cat_dict_data: cols.cat_dict_data,
        cat_dict_offsets: cols.cat_dict_offsets,
    };
    let nonz = UserDailyCols {
        names_data: &scratch.nonz_names_data,
        names_offsets: &scratch.nonz_names_offsets,
        person_keys: &scratch.nonz_person,
        day: &scratch.nonz_day,
        sessions: &scratch.nonz_sessions,
        events: &scratch.nonz_events,
        revenue_cents: &scratch.nonz_revenue,
        top_category: &scratch.nonz_category,
        cat_dict_data: cols.cat_dict_data,
        cat_dict_offsets: cols.cat_dict_offsets,
    };
    (zero, nonz)
}

#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn split_zero_revenue_avx2(cols: &UserDailyCols<'_>, scratch: &mut SplitScratch) {
    let n = cols.len();
    let mut i = 0usize;
    let mut z_len = 0usize;
    let mut nz_len = 0usize;

    let zp = scratch.zero_person.as_mut_ptr();
    let zd = scratch.zero_day.as_mut_ptr();
    let zs = scratch.zero_sessions.as_mut_ptr();
    let ze = scratch.zero_events.as_mut_ptr();

    let nzp = scratch.nonz_person.as_mut_ptr();
    let nzd = scratch.nonz_day.as_mut_ptr();
    let nzs = scratch.nonz_sessions.as_mut_ptr();
    let nze = scratch.nonz_events.as_mut_ptr();
    let nzr = scratch.nonz_revenue.as_mut_ptr();
    let nzc = scratch.nonz_category.as_mut_ptr();

    while i + 8 <= n {
        let zmask = zero_rev_mask8_avx2(cols.revenue_cents.as_ptr().add(i), cols.top_category.as_ptr().add(i), 8);
        if zmask == 0xFF {
            std::ptr::copy_nonoverlapping(cols.person_keys.as_ptr().add(i), zp.add(z_len), 8);
            std::ptr::copy_nonoverlapping(cols.day.as_ptr().add(i), zd.add(z_len), 8);
            std::ptr::copy_nonoverlapping(cols.sessions.as_ptr().add(i), zs.add(z_len), 8);
            std::ptr::copy_nonoverlapping(cols.events.as_ptr().add(i), ze.add(z_len), 8);
            z_len += 8;
        } else {
            for k in 0..8 {
                let idx = i + k;
                if (zmask >> k) & 1 != 0 {
                    *zp.add(z_len) = *cols.person_keys.get_unchecked(idx);
                    *zd.add(z_len) = *cols.day.get_unchecked(idx);
                    *zs.add(z_len) = *cols.sessions.get_unchecked(idx);
                    *ze.add(z_len) = *cols.events.get_unchecked(idx);
                    z_len += 1;
                } else {
                    *nzp.add(nz_len) = *cols.person_keys.get_unchecked(idx);
                    *nzd.add(nz_len) = *cols.day.get_unchecked(idx);
                    *nzs.add(nz_len) = *cols.sessions.get_unchecked(idx);
                    *nze.add(nz_len) = *cols.events.get_unchecked(idx);
                    *nzr.add(nz_len) = *cols.revenue_cents.get_unchecked(idx);
                    *nzc.add(nz_len) = *cols.top_category.get_unchecked(idx);
                    nz_len += 1;
                }
            }
        }
        i += 8;
    }

    while i < n {
        let rev = *cols.revenue_cents.get_unchecked(i);
        let cat = *cols.top_category.get_unchecked(i);
        if rev == 0 && cat == NULL_CAT {
            *zp.add(z_len) = *cols.person_keys.get_unchecked(i);
            *zd.add(z_len) = *cols.day.get_unchecked(i);
            *zs.add(z_len) = *cols.sessions.get_unchecked(i);
            *ze.add(z_len) = *cols.events.get_unchecked(i);
            z_len += 1;
        } else {
            *nzp.add(nz_len) = *cols.person_keys.get_unchecked(i);
            *nzd.add(nz_len) = *cols.day.get_unchecked(i);
            *nzs.add(nz_len) = *cols.sessions.get_unchecked(i);
            *nze.add(nz_len) = *cols.events.get_unchecked(i);
            *nzr.add(nz_len) = rev;
            *nzc.add(nz_len) = cat;
            nz_len += 1;
        }
        i += 1;
    }

    scratch.zero_person.set_len(z_len);
    scratch.zero_day.set_len(z_len);
    scratch.zero_sessions.set_len(z_len);
    scratch.zero_events.set_len(z_len);
    scratch.zero_revenue.resize(z_len, 0);
    scratch.zero_category.resize(z_len, NULL_CAT);

    scratch.nonz_person.set_len(nz_len);
    scratch.nonz_day.set_len(nz_len);
    scratch.nonz_sessions.set_len(nz_len);
    scratch.nonz_events.set_len(nz_len);
    scratch.nonz_revenue.set_len(nz_len);
    scratch.nonz_category.set_len(nz_len);
}

#[derive(Clone, Copy)]
struct ColChunkMeta {
    file_offset: u64,
    total_size: u64,
    data_page_offset: u64,
    dict_page_offset: Option<u64>,
    uses_dict: bool,
}

struct RowGroupMeta {
    num_rows: u64,
    cols: [ColChunkMeta; 6],
}

/// High-speed columnar Parquet v1 writer for `user_daily` using `RLE_DICTIONARY`
/// and optional zero-revenue row-group splitting.
///
/// Accumulates the entire Parquet file in a single `Vec<u8>` buffer (pre-allocated to 4 MiB on the
/// first non-empty row group) and flushes it in one `write_all` syscall in `finish()`.
pub struct FastUserDailyWriter {
    file: File,
    buf: Vec<u8>,
    total_rows: u64,
    groups: Vec<RowGroupMeta>,
    page_buf: Vec<u8>,
    dict_buf: Vec<u8>,
    idx_buf: Vec<u32>,
    cat_u8_buf: Vec<u8>,
    split_scratch: SplitScratch,
}

impl FastUserDailyWriter {
    pub fn create(path: &Path) -> Self {
        let file = File::create(path).expect("create fast_parquet output");
        let mut buf = Vec::with_capacity(512);
        buf.extend_from_slice(b"PAR1");
        FastUserDailyWriter {
            file,
            buf,
            total_rows: 0,
            groups: Vec::new(),
            page_buf: Vec::with_capacity(64 << 10),
            dict_buf: Vec::with_capacity(64 << 10),
            idx_buf: Vec::with_capacity(4096),
            cat_u8_buf: Vec::with_capacity(4096),
            split_scratch: SplitScratch::new(),
        }
    }

    /// Writes `cols` in chunks of `chunk_rows` (e.g. 4096), splitting each chunk into a
    /// zero-revenue row group (`rev == 0 && cat == 0xFF`) and a non-zero row group.
    pub fn write_batches_split(&mut self, cols: &UserDailyCols<'_>, chunk_rows: usize) {
        let step = chunk_rows.max(1);
        let mut from = 0usize;
        while from < cols.len() {
            let to = (from + step).min(cols.len());
            let sub = cols.slice(from, to);
            let mut scratch = std::mem::take(&mut self.split_scratch);
            let (zero_rg, nonz_rg) = split_zero_revenue_chunks(&sub, &mut scratch);
            if !zero_rg.is_empty() {
                self.write_row_group(&zero_rg);
            }
            if !nonz_rg.is_empty() {
                self.write_row_group(&nonz_rg);
            }
            self.split_scratch = scratch;
            from = to;
        }
    }

    /// Writes `cols` directly as a single Parquet row group without splitting.
    pub fn write_row_group(&mut self, cols: &UserDailyCols<'_>) {
        let n = cols.len();
        if n == 0 {
            return;
        }
        if self.buf.capacity() < (4 << 20) {
            self.buf.reserve((4 << 20) - self.buf.len());
        }
        let c0 = self.write_user_id_col(cols);
        let c1 = self.write_day_col(cols.day);
        let c2 = self.write_sessions_col(cols.sessions);
        let c3 = self.write_events_col(cols.events);
        let c4 = self.write_int32_const_or_plain(cols.revenue_cents);
        let c5 = self.write_top_category_col(cols);

        self.total_rows += n as u64;
        self.groups.push(RowGroupMeta {
            num_rows: n as u64,
            cols: [c0, c1, c2, c3, c4, c5],
        });
    }

    fn emit_dict_page(&mut self, num_values: usize, payload: &[u8]) -> (u64, u64) {
        let off = self.buf.len() as u64;
        {
            let mut t = ThriftWriter::new(&mut self.buf);
            t.i32(1, P_DICT);
            t.i32(2, payload.len() as i32);
            t.i32(3, payload.len() as i32);
            t.begin_field(7); // DictionaryPageHeader
            t.i32(1, num_values as i32);
            t.i32(2, E_PLAIN);
            t.end_struct();
            self.buf.push(0);
        }
        self.buf.extend_from_slice(payload);
        let written = self.buf.len() as u64 - off;
        (off, written)
    }

    fn emit_data_page(&mut self, num_values: usize, encoding: i32, payload: &[u8]) -> (u64, u64) {
        let off = self.buf.len() as u64;
        {
            let mut t = ThriftWriter::new(&mut self.buf);
            t.i32(1, P_DATA);
            t.i32(2, payload.len() as i32);
            t.i32(3, payload.len() as i32);
            t.begin_field(5); // DataPageHeader
            t.i32(1, num_values as i32);
            t.i32(2, encoding);
            t.i32(3, E_RLE); // definition_level_encoding
            t.i32(4, E_RLE); // repetition_level_encoding
            t.end_struct();
            self.buf.push(0);
        }
        self.buf.extend_from_slice(payload);
        let written = self.buf.len() as u64 - off;
        (off, written)
    }

    /// 1. `user_id` (`BYTE_ARRAY`, `REQUIRED`):
    /// Dictionary page of `lo_person..=hi_person` + RLE/Bit-Packed `RLE_DICTIONARY` data page.
    fn write_user_id_col(&mut self, cols: &UserDailyCols<'_>) -> ColChunkMeta {
        let keys = cols.person_keys;
        let mut lo = keys[0];
        let mut hi = keys[0];
        for &k in &keys[1..] {
            if k < lo {
                lo = k;
            }
            if k > hi {
                hi = k;
            }
        }
        let dict_count = (hi - lo + 1) as usize;
        self.dict_buf.clear();
        for p in lo..=hi {
            let start = cols.names_offsets[p as usize] as usize;
            let end = cols.names_offsets[p as usize + 1] as usize;
            let name = &cols.names_data[start..end];
            self.dict_buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            self.dict_buf.extend_from_slice(name);
        }
        let dict_payload = std::mem::take(&mut self.dict_buf);
        let (dict_off, dict_sz) = self.emit_dict_page(dict_count, &dict_payload);
        self.dict_buf = dict_payload;

        let w = bit_width(hi - lo);
        let mut page_payload = std::mem::take(&mut self.page_buf);
        page_payload.clear();
        encode_rle_or_bitpack_keys(keys, lo, w, &mut page_payload);
        let (data_off, data_sz) = self.emit_data_page(keys.len(), E_RLE_DICT, &page_payload);
        self.page_buf = page_payload;

        ColChunkMeta {
            file_offset: dict_off,
            total_size: dict_sz + data_sz,
            data_page_offset: data_off,
            dict_page_offset: Some(dict_off),
            uses_dict: true,
        }
    }

    /// Shared helper for `INT32` columns whose span `max_v - min_v < 256`:
    /// dense `INT32` dictionary page (`min_v..=max_v`) + RLE or bit-packed `RLE_DICTIONARY` page.
    fn write_dense_i32_dict_col(
        &mut self,
        values: &[i32],
        min_v: i32,
        max_v: i32,
        min_bit_width: u8,
        use_rle: bool,
    ) -> ColChunkMeta {
        let span = (max_v as i64 - min_v as i64) as u32;
        let dict_count = (span + 1) as usize;
        let mut dict_payload = std::mem::take(&mut self.dict_buf);
        dict_payload.clear();
        for v in min_v..=max_v {
            dict_payload.extend_from_slice(&v.to_le_bytes());
        }
        let (dict_off, dict_sz) = self.emit_dict_page(dict_count, &dict_payload);
        self.dict_buf = dict_payload;

        let w = bit_width(span).max(min_bit_width);
        let mut page_payload = std::mem::take(&mut self.page_buf);
        page_payload.clear();
        page_payload.push(w);
        sub_min_to_u32(values, min_v, &mut self.idx_buf);
        if use_rle {
            encode_rle_u32(&self.idx_buf, w, &mut page_payload);
        } else {
            encode_bitpacked_u32(&self.idx_buf, w, &mut page_payload);
        }
        let (data_off, data_sz) = self.emit_data_page(values.len(), E_RLE_DICT, &page_payload);
        self.page_buf = page_payload;

        ColChunkMeta {
            file_offset: dict_off,
            total_size: dict_sz + data_sz,
            data_page_offset: data_off,
            dict_page_offset: Some(dict_off),
            uses_dict: true,
        }
    }

    /// 2. `day` (`INT32`, `DATE`, `REQUIRED`):
    /// Dense dictionary page of `min_day..=max_day` + bit-packed `RLE_DICTIONARY` data page.
    fn write_day_col(&mut self, days: &[i32]) -> ColChunkMeta {
        let (min_d, max_d) = min_max_i32(days);
        let span = (max_d as i64 - min_d as i64) as u64;
        if span == 0 {
            self.write_int32_const_or_plain(days)
        } else if span < 256 {
            self.write_dense_i32_dict_col(days, min_d, max_d, 4, false)
        } else {
            self.write_plain_i32(days)
        }
    }

    /// 3. `sessions` (`INT32`, `REQUIRED`):
    /// When `max_sessions - min_sessions < 256`, dense `INT32` dictionary + vectorized RLE/bit-packed
    /// `RLE_DICTIONARY`; otherwise `PLAIN`.
    fn write_sessions_col(&mut self, sessions: &[i32]) -> ColChunkMeta {
        let (min_s, max_s) = min_max_i32(sessions);
        let span = (max_s as i64 - min_s as i64) as u64;
        if span == 0 {
            self.write_int32_const_or_plain(sessions)
        } else if span < 256 {
            self.write_dense_i32_dict_col(sessions, min_s, max_s, 1, true)
        } else {
            self.write_plain_i32(sessions)
        }
    }

    /// 4. `events` (`INT32`, `REQUIRED`):
    /// When all values in the row group are identical, write a 1-entry `INT32` dictionary page
    /// (4 bytes) + a single RLE run `(n << 1, 0x00)`; when `max - min < 256`, use a dense
    /// `INT32` dictionary + vectorized RLE/bit-packed `RLE_DICTIONARY`; otherwise `PLAIN`.
    fn write_events_col(&mut self, values: &[i32]) -> ColChunkMeta {
        let (min_v, max_v) = min_max_i32(values);
        let span = (max_v as i64 - min_v as i64) as u64;
        if span == 0 {
            self.write_int32_const_or_plain(values)
        } else if span < 256 {
            self.write_dense_i32_dict_col(values, min_v, max_v, 1, true)
        } else {
            self.write_plain_i32(values)
        }
    }

    /// 5. `revenue_cents` (`INT32`, `REQUIRED`):
    /// When all values in the row group are identical (e.g. `revenue_cents == 0`), write a 1-entry
    /// `INT32` dictionary page (4 bytes) + a single RLE run `(n << 1, 0x00)`; otherwise `PLAIN`.
    fn write_int32_const_or_plain(&mut self, values: &[i32]) -> ColChunkMeta {
        let (min_v, max_v) = min_max_i32(values);
        if min_v == max_v {
            let dict_bytes = min_v.to_le_bytes();
            let (dict_off, dict_sz) = self.emit_dict_page(1, &dict_bytes);

            let mut page_payload = std::mem::take(&mut self.page_buf);
            page_payload.clear();
            page_payload.push(1); // bit-width = 1
            push_varint(&mut page_payload, (values.len() as u64) << 1);
            page_payload.push(0); // dictionary index 0
            let (data_off, data_sz) = self.emit_data_page(values.len(), E_RLE_DICT, &page_payload);
            self.page_buf = page_payload;

            ColChunkMeta {
                file_offset: dict_off,
                total_size: dict_sz + data_sz,
                data_page_offset: data_off,
                dict_page_offset: Some(dict_off),
                uses_dict: true,
            }
        } else {
            self.write_plain_i32(values)
        }
    }

    fn write_plain_i32(&mut self, values: &[i32]) -> ColChunkMeta {
        let raw_bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
        };
        let (data_off, data_sz) = self.emit_data_page(values.len(), E_PLAIN, raw_bytes);
        ColChunkMeta {
            file_offset: data_off,
            total_size: data_sz,
            data_page_offset: data_off,
            dict_page_offset: None,
            uses_dict: false,
        }
    }

    /// 6. `top_category` (`BYTE_ARRAY`, `OPTIONAL`):
    /// - If 0 non-null categories: `PLAIN` data page with all-zero RLE definition level `(n << 1, 0x00)`
    ///   and 0 value bytes.
    /// - Otherwise: Dictionary Page of category strings + `E_RLE_DICT` Data Page encoded via
    ///   `encode_top_category_chunk`.
    fn write_top_category_col(&mut self, cols: &UserDailyCols<'_>) -> ColChunkMeta {
        let cats = cols.top_category;
        let n = cats.len();

        if !has_non_null_u8(cats) {
            let mut page_payload = std::mem::take(&mut self.page_buf);
            page_payload.clear();
            let hdr = (n as u64) << 1;
            let def_len = varint_len(hdr) + 1;
            page_payload.extend_from_slice(&(def_len as u32).to_le_bytes());
            push_varint(&mut page_payload, hdr);
            page_payload.push(0); // def level 0 (NULL) repeated n times
            let (data_off, data_sz) = self.emit_data_page(n, E_PLAIN, &page_payload);
            self.page_buf = page_payload;

            ColChunkMeta {
                file_offset: data_off,
                total_size: data_sz,
                data_page_offset: data_off,
                dict_page_offset: None,
                uses_dict: false,
            }
        } else {
            let num_cats = cols.cat_dict_offsets.len().saturating_sub(1);
            let mut dict_payload = std::mem::take(&mut self.dict_buf);
            dict_payload.clear();
            for c in 0..num_cats {
                let s = cols.cat_dict_offsets[c] as usize;
                let e = cols.cat_dict_offsets[c + 1] as usize;
                let bytes = &cols.cat_dict_data[s..e];
                dict_payload.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                dict_payload.extend_from_slice(bytes);
            }
            let (dict_off, dict_sz) = self.emit_dict_page(num_cats, &dict_payload);
            self.dict_buf = dict_payload;

            let mut page_payload = std::mem::take(&mut self.page_buf);
            page_payload.clear();
            let mut cat_u8_buf = std::mem::take(&mut self.cat_u8_buf);
            encode_top_category_chunk(cats, num_cats, &mut cat_u8_buf, &mut page_payload);
            self.cat_u8_buf = cat_u8_buf;

            let (data_off, data_sz) = self.emit_data_page(n, E_RLE_DICT, &page_payload);
            self.page_buf = page_payload;

            ColChunkMeta {
                file_offset: dict_off,
                total_size: dict_sz + data_sz,
                data_page_offset: data_off,
                dict_page_offset: Some(dict_off),
                uses_dict: true,
            }
        }
    }

    pub fn finish(mut self) {
        #[derive(Clone, Copy)]
        struct ColSpec {
            name: &'static [u8],
            ty: i32,
            rep: i32,
            converted: Option<i32>,
            logical_tag: Option<i16>,
        }

        const SPECS: [ColSpec; 6] = [
            ColSpec { name: b"user_id", ty: TYPE_BYTE_ARRAY, rep: REP_REQUIRED, converted: Some(0), logical_tag: Some(1) },
            ColSpec { name: b"day", ty: TYPE_INT32, rep: REP_REQUIRED, converted: Some(6), logical_tag: Some(6) },
            ColSpec { name: b"sessions", ty: TYPE_INT32, rep: REP_REQUIRED, converted: None, logical_tag: None },
            ColSpec { name: b"events", ty: TYPE_INT32, rep: REP_REQUIRED, converted: None, logical_tag: None },
            ColSpec { name: b"revenue_cents", ty: TYPE_INT32, rep: REP_REQUIRED, converted: None, logical_tag: None },
            ColSpec { name: b"top_category", ty: TYPE_BYTE_ARRAY, rep: REP_OPTIONAL, converted: Some(0), logical_tag: Some(1) },
        ];

        let footer_start = self.buf.len();
        {
            let mut t = ThriftWriter::new(&mut self.buf);
            t.i32(1, 1); // version = 1
            t.list(2, T_STRUCT, SPECS.len() + 1);

            // Root schema element
            t.begin_struct();
            t.binary(4, b"schema");
            t.i32(5, SPECS.len() as i32);
            t.end_struct();

            for spec in &SPECS {
                t.begin_struct();
                t.i32(1, spec.ty);
                t.i32(3, spec.rep);
                t.binary(4, spec.name);
                if let Some(conv) = spec.converted {
                    t.i32(6, conv);
                }
                if let Some(ltag) = spec.logical_tag {
                    t.begin_field(10);
                    t.begin_field(ltag);
                    t.end_struct();
                    t.end_struct();
                }
                t.end_struct();
            }

            t.i64(3, self.total_rows as i64);
            t.list(4, T_STRUCT, self.groups.len());

            for g in &self.groups {
                t.begin_struct();
                t.list(1, T_STRUCT, SPECS.len());
                let mut rg_bytes = 0u64;
                for (spec, chunk) in SPECS.iter().zip(&g.cols) {
                    rg_bytes += chunk.total_size;
                    t.begin_struct();
                    t.i64(2, chunk.file_offset as i64);
                    t.begin_field(3); // ColumnMetaData
                    t.i32(1, spec.ty);
                    if chunk.uses_dict {
                        t.list(2, T_I32, 3);
                        t.zigzag(E_PLAIN as i64);
                        t.zigzag(E_RLE as i64);
                        t.zigzag(E_RLE_DICT as i64);
                    } else {
                        t.list(2, T_I32, 2);
                        t.zigzag(E_PLAIN as i64);
                        t.zigzag(E_RLE as i64);
                    }
                    t.list(3, T_BINARY, 1);
                    t.varint(spec.name.len() as u64);
                    t.buf.extend_from_slice(spec.name);
                    t.i32(4, 0); // UNCOMPRESSED
                    t.i64(5, g.num_rows as i64);
                    t.i64(6, chunk.total_size as i64);
                    t.i64(7, chunk.total_size as i64);
                    t.i64(9, chunk.data_page_offset as i64);
                    if let Some(doff) = chunk.dict_page_offset {
                        t.i64(11, doff as i64);
                    }
                    t.end_struct();
                    t.end_struct();
                }
                t.i64(2, rg_bytes as i64);
                t.i64(3, g.num_rows as i64);
                t.end_struct();
            }

            t.buf.push(0); // Stop FileMetaData
        }
        let footer_len = (self.buf.len() - footer_start) as u32;
        self.buf.extend_from_slice(&footer_len.to_le_bytes());
        self.buf.extend_from_slice(b"PAR1");
        self.file.write_all(&self.buf).expect("write Parquet output");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn build_sample_dataset(num_rows: usize) -> (
        Vec<u8>,
        Vec<u32>,
        Vec<u32>,
        Vec<i32>,
        Vec<i32>,
        Vec<i32>,
        Vec<i32>,
        Vec<u8>,
        Vec<u8>,
        Vec<u32>,
    ) {
        let mut names_data = Vec::new();
        let mut names_offsets = vec![0u32];
        let num_persons = (num_rows / 4).max(8);
        for p in 0..num_persons {
            let name = format!("u_{p:06}");
            names_data.extend_from_slice(name.as_bytes());
            names_offsets.push(names_data.len() as u32);
        }

        let categories = [
            "apparel", "automotive", "beauty", "books", "electronics",
            "food", "furniture", "garden", "grocery", "health",
            "home", "jewelry", "kids", "movies", "music",
            "office", "outdoors", "pets", "shoes", "sports",
            "tools", "toys", "travel", "unknown", "video_games",
        ];
        let mut cat_dict_data = Vec::new();
        let mut cat_dict_offsets = vec![0u32];
        for c in &categories {
            cat_dict_data.extend_from_slice(c.as_bytes());
            cat_dict_offsets.push(cat_dict_data.len() as u32);
        }

        let mut person_keys = Vec::with_capacity(num_rows);
        let mut day = Vec::with_capacity(num_rows);
        let mut sessions = Vec::with_capacity(num_rows);
        let mut events = Vec::with_capacity(num_rows);
        let mut revenue_cents = Vec::with_capacity(num_rows);
        let mut top_category = Vec::with_capacity(num_rows);

        let base_day = 20727i32; // 2026-10-01
        let mut cur_person = 0u32;
        let mut run = 0usize;
        for i in 0..num_rows {
            person_keys.push(cur_person);
            run += 1;
            if run >= ((i % 7) + 1) && (cur_person as usize + 1) < num_persons {
                cur_person += 1;
                run = 0;
            }
            day.push(base_day + (i % 15) as i32);
            sessions.push(1 + (i % 11) as i32);
            events.push(1 + (i % 47) as i32);
            // ~85% zero revenue + NULL category, ~15% non-zero revenue + valid category
            if i % 20 < 17 {
                revenue_cents.push(0);
                top_category.push(NULL_CAT);
            } else {
                revenue_cents.push(((i as i32 * 37) % 50_000) + 100);
                top_category.push((i % categories.len()) as u8);
            }
        }

        (
            names_data,
            names_offsets,
            person_keys,
            day,
            sessions,
            events,
            revenue_cents,
            top_category,
            cat_dict_data,
            cat_dict_offsets,
        )
    }

    fn verify_with_polars_and_duckdb(path: &Path, cols: &UserDailyCols<'_>, categories: &[&str]) {
        let venv_py = Path::new("/home/colton/Documents/Projects/Builders/#3/.venv/bin/python");
        if !venv_py.exists() {
            return;
        }
        let script = r#"
import sys
import polars as pl
import pyarrow.parquet as pq
import duckdb

path = sys.argv[1]
# 1. Read with PyArrow
table = pq.read_table(path)
# 2. Read with Polars
df = pl.read_parquet(path)
# 3. Read with DuckDB
rel = duckdb.sql(f"SELECT user_id, day, sessions, events, revenue_cents, top_category FROM read_parquet('{path}')").fetchall()

assert df.height == len(rel), f"height mismatch {df.height} vs {len(rel)}"
assert table.num_rows == df.height
assert str(df.schema["day"]) == "Date", f"bad day schema: {df.schema['day']}"

# Print summary line for Rust test to verify exact aggregates and rows
null_cats = df["top_category"].null_count()
rev_sum = df["revenue_cents"].sum()
sess_sum = df["sessions"].sum()
ev_sum = df["events"].sum()
print(f"OK rows={df.height} nulls={null_cats} rev={rev_sum} sess={sess_sum} ev={ev_sum}")
"#;
        let out = Command::new(venv_py)
            .arg("-c")
            .arg(script)
            .arg(path.to_str().unwrap())
            .output()
            .expect("run python verifier");
        assert!(
            out.status.success(),
            "Python verifier failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let exp_rows = cols.len();
        let exp_nulls = cols.top_category.iter().filter(|&&c| c == NULL_CAT).count();
        let exp_rev: i64 = cols.revenue_cents.iter().map(|&v| v as i64).sum();
        let exp_sess: i64 = cols.sessions.iter().map(|&v| v as i64).sum();
        let exp_ev: i64 = cols.events.iter().map(|&v| v as i64).sum();
        let expected_line = format!(
            "OK rows={exp_rows} nulls={exp_nulls} rev={exp_rev} sess={exp_sess} ev={exp_ev}"
        );
        assert!(
            stdout.trim() == expected_line,
            "Mismatch: got {:?}, expected {:?} (cats={})",
            stdout.trim(),
            expected_line,
            categories.len()
        );
    }

    #[test]
    fn test_fast_parquet_split_and_unsplit_readers() {
        let (nd, no, pk, day, sess, ev, rev, cat, cd, co) = build_sample_dataset(10_000);
        let cols = UserDailyCols {
            names_data: &nd,
            names_offsets: &no,
            person_keys: &pk,
            day: &day,
            sessions: &sess,
            events: &ev,
            revenue_cents: &rev,
            top_category: &cat,
            cat_dict_data: &cd,
            cat_dict_offsets: &co,
        };

        let tmp_dir = std::env::temp_dir();
        let unsplit_path = tmp_dir.join("fast_pq_test_unsplit.parquet");
        let split_path = tmp_dir.join("fast_pq_test_split.parquet");

        // Write unsplit (4,096-row row groups)
        {
            let mut w = FastUserDailyWriter::create(&unsplit_path);
            let mut at = 0;
            while at < cols.len() {
                let end = (at + 4096).min(cols.len());
                w.write_row_group(&cols.slice(at, end));
                at = end;
            }
            w.finish();
        }

        // Write split (4,096-row batches split into zero-revenue and non-zero row groups)
        {
            let mut w = FastUserDailyWriter::create(&split_path);
            w.write_batches_split(&cols, 4096);
            w.finish();
        }

        let unsplit_size = std::fs::metadata(&unsplit_path).unwrap().len();
        let split_size = std::fs::metadata(&split_path).unwrap().len();
        // Plain uncompressed 10,000 rows in src/parquet.rs is ~385 KB (12 B user_id + 24 B numbers + ~5 B cat/levels);
        // split RLE_DICTIONARY is ~101 KB (>73% smaller than plain, and smaller than unsplit!).
        assert!(
            split_size < unsplit_size,
            "expected split_size ({split_size}) < unsplit_size ({unsplit_size})"
        );
        assert!(
            split_size < 120_000,
            "expected split_size ({split_size}) < 120 KB for 10k rows (>65% smaller than ~385 KB plain)"
        );

        let dummy_cats = ["apparel"];
        verify_with_polars_and_duckdb(&unsplit_path, &cols, &dummy_cats);
        verify_with_polars_and_duckdb(&split_path, &cols, &dummy_cats);

        // Also verify row-by-row exact equality between unsplit and split via Polars
        let venv_py = Path::new("/home/colton/Documents/Projects/Builders/#3/.venv/bin/python");
        if venv_py.exists() {
            let out = Command::new(venv_py)
                .arg("-c")
                .arg(
                    "import sys, polars as pl; \
                     a = pl.read_parquet(sys.argv[1]).sort(['user_id', 'day']); \
                     b = pl.read_parquet(sys.argv[2]).sort(['user_id', 'day']); \
                     assert a.equals(b, null_equal=True), 'unsplit != split'",
                )
                .arg(unsplit_path.to_str().unwrap())
                .arg(split_path.to_str().unwrap())
                .output()
                .unwrap();
            assert!(out.status.success(), "Polars equality check failed: {}", String::from_utf8_lossy(&out.stderr));
        }

        let _ = std::fs::remove_file(&unsplit_path);
        let _ = std::fs::remove_file(&split_path);
    }

    #[test]
    fn test_fast_parquet_edge_cases_all_null_all_const_wide_sessions() {
        let (mut nd, mut no, mut pk, mut day, mut sess, mut ev, mut rev, mut cat, cd, co) =
            build_sample_dataset(512);
        // Case 1: All-null categories, constant events, constant revenue, constant day, constant sessions
        for i in 0..512 {
            day[i] = 20730;
            sess[i] = 3;
            ev[i] = 7;
            rev[i] = 0;
            cat[i] = NULL_CAT;
        }
        // Case 2: Append 512 rows where sessions span > 256 (triggers PLAIN fallback) and all categories non-null
        for i in 0..512 {
            let p = (no.len() - 1) as u32;
            let name = format!("z_whale_{i:04}");
            nd.extend_from_slice(name.as_bytes());
            no.push(nd.len() as u32);
            pk.push(p);
            day.push(20730 + (i as i32 % 300)); // day span >= 256 -> PLAIN fallback
            sess.push(1 + (i as i32 * 3)); // session span >= 256 -> PLAIN fallback
            ev.push(10 + i as i32);
            rev.push(500 + i as i32);
            cat.push((i % 25) as u8);
        }

        let cols = UserDailyCols {
            names_data: &nd,
            names_offsets: &no,
            person_keys: &pk,
            day: &day,
            sessions: &sess,
            events: &ev,
            revenue_cents: &rev,
            top_category: &cat,
            cat_dict_data: &cd,
            cat_dict_offsets: &co,
        };

        let tmp_path = std::env::temp_dir().join("fast_pq_test_edges.parquet");
        {
            let mut w = FastUserDailyWriter::create(&tmp_path);
            w.write_row_group(&cols.slice(0, 512));
            w.write_row_group(&cols.slice(512, 1024));
            w.finish();
        }
        verify_with_polars_and_duckdb(&tmp_path, &cols, &[]);
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_contiguous_file_mapping_and_huge_bump() {
        let tmp_dir = std::env::temp_dir();
        let p1 = tmp_dir.join("contig_test_1.csv");
        let p2 = tmp_dir.join("contig_test_2.csv");
        let data1 = vec![b'a'; (2 << 20) + 12345];
        let data2 = b"event_id,user_id\ne1,u1\n".to_vec();
        std::fs::write(&p1, &data1).unwrap();
        std::fs::write(&p2, &data2).unwrap();

        let region = crate::sys::map_files_contiguous(&[p1.clone(), p2.clone()]);
        assert_eq!(region.files.len(), 2);
        assert_eq!(region.region_base as usize & ((2 << 20) - 1), 0, "region_base must be 2 MiB-aligned");
        for f in &region.files {
            assert_eq!(f.ptr as usize & ((2 << 20) - 1), 0, "each file slot must be 2 MiB-aligned");
            assert_eq!(
                unsafe { region.region_base.add(f.rel_offset as usize) },
                f.ptr,
                "every byte offset is addressable as a u64 offset from region_base"
            );
        }
        let s1 = unsafe { std::slice::from_raw_parts(region.files[0].ptr, region.files[0].size) };
        let s2 = unsafe { std::slice::from_raw_parts(region.files[1].ptr, region.files[1].size) };
        assert_eq!(s1, data1.as_slice());
        assert_eq!(s2, data2.as_slice());

        crate::sys::release(region.reserve_base, region.reserve_len);
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);

        let bump = crate::sys::HugeBump::new(4 << 20);
        let a = bump.alloc_aligned(1024, 64);
        let b = bump.alloc_aligned(2048, 4096);
        assert_eq!(a as usize & 63, 0);
        assert_eq!(b as usize & 4095, 0);
        unsafe {
            std::ptr::write_bytes(a, 0x42, 1024);
            std::ptr::write_bytes(b, 0x99, 2048);
        }
        assert!(bump.used_bytes() >= 4096 + 2048);
        bump.release_used();
    }

    fn decode_rle_hybrid(encoded: &[u8], width: u8, count: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(count);
        if count == 0 {
            return out;
        }
        let w = width as usize;
        let val_bytes = w.div_ceil(8);
        let mask = if w == 32 { u32::MAX } else { (1u32 << w) - 1 };
        let mut pos = 0usize;
        while out.len() < count {
            let mut hdr = 0u64;
            let mut shift = 0u32;
            loop {
                let b = encoded[pos];
                pos += 1;
                hdr |= ((b & 0x7F) as u64) << shift;
                if b & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            if hdr & 1 == 0 {
                let run_len = (hdr >> 1) as usize;
                let mut val = 0u32;
                for b in 0..val_bytes {
                    val |= (encoded[pos + b] as u32) << (b * 8);
                }
                pos += val_bytes;
                let take = run_len.min(count - out.len());
                out.resize(out.len() + take, val);
            } else {
                let num_groups = (hdr >> 1) as usize;
                let nvals = num_groups * 8;
                let byte_len = num_groups * w;
                let slice = &encoded[pos..pos + byte_len];
                pos += byte_len;
                let mut bit_pos = 0usize;
                for _ in 0..nvals {
                    let byte_idx = bit_pos / 8;
                    let bit_off = bit_pos % 8;
                    let mut raw = 0u64;
                    for b in 0..5 {
                        if byte_idx + b < slice.len() {
                            raw |= (slice[byte_idx + b] as u64) << (b * 8);
                        }
                    }
                    if out.len() < count {
                        out.push(((raw >> bit_off) as u32) & mask);
                    }
                    bit_pos += w;
                }
            }
        }
        assert_eq!(pos, encoded.len(), "trailing bytes after decoding RLE/BP stream");
        out
    }

    #[test]
    fn test_encode_rle_u32_and_bitpacked_roundtrip() {
        let mut patterns: Vec<(u8, Vec<u32>)> = Vec::new();
        // Empty
        patterns.push((4, vec![]));
        // 95% zeros (sessions-like) across various lengths and widths
        for &len in &[1usize, 7, 8, 9, 15, 16, 17, 63, 64, 65, 255, 4096] {
            for &w in &[1u8, 2, 3, 5, 8, 12] {
                let max_v = (1u32 << w) - 1;
                let mut vals = Vec::with_capacity(len);
                for i in 0..len {
                    if i % 20 == 7 {
                        vals.push((i as u32 % max_v).max(1).min(max_v));
                    } else {
                        vals.push(0);
                    }
                }
                patterns.push((w, vals));
            }
        }
        // Long runs interspersed with non-run segments and short tails
        let mut mixed = Vec::new();
        mixed.extend(std::iter::repeat_n(3u32, 40));
        mixed.extend(0..16u32);
        mixed.extend(std::iter::repeat_n(7u32, 23));
        mixed.extend([1u32, 2, 1, 2, 3]);
        patterns.push((4, mixed));

        // Pure alternating (no runs)
        patterns.push((5, (0..131).map(|i| (i * 7) % 31).collect()));
        // All constant
        patterns.push((1, vec![0u32; 500]));
        patterns.push((6, vec![42u32; 500]));

        let mut buf = Vec::new();
        for (w, vals) in &patterns {
            buf.clear();
            encode_rle_u32(vals, *w, &mut buf);
            let decoded = decode_rle_hybrid(&buf, *w, vals.len());
            assert_eq!(&decoded, vals, "encode_rle_u32 roundtrip mismatch for w={w}, len={}", vals.len());

            buf.clear();
            encode_bitpacked_u32(vals, *w, &mut buf);
            let decoded_bp = decode_rle_hybrid(&buf, *w, vals.len());
            assert_eq!(&decoded_bp, vals, "encode_bitpacked_u32 roundtrip mismatch for w={w}, len={}", vals.len());
        }
    }

    #[test]
    fn test_encode_top_category_chunk_roundtrip() {
        let mut cat_buf = Vec::new();
        let mut out = Vec::new();
        for &len in &[1usize, 7, 8, 9, 31, 32, 33, 64, 99, 4096] {
            for mode in 0..3 {
                let cats: Vec<u8> = (0..len)
                    .map(|i| match mode {
                        0 => (i % 25) as u8, // all non-null
                        1 => if i % 7 == 0 { (i % 25) as u8 } else { NULL_CAT }, // sparse
                        _ => if (i / 32) % 2 == 0 { (i % 25) as u8 } else { NULL_CAT }, // mixed 32-byte blocks
                    })
                    .collect();
                out.clear();
                encode_top_category_chunk(&cats, 25, &mut cat_buf, &mut out);

                let def_len = u32::from_le_bytes(out[0..4].try_into().unwrap()) as usize;
                let defs = decode_rle_hybrid(&out[4..4 + def_len], 1, len);
                let expected_defs: Vec<u32> = cats.iter().map(|&c| u32::from(c != NULL_CAT)).collect();
                assert_eq!(defs, expected_defs, "def mismatch len={len} mode={mode}");

                let w = out[4 + def_len];
                assert_eq!(w, 5);
                let non_null_count = cats.iter().filter(|&&c| c != NULL_CAT).count();
                let vals = decode_rle_hybrid(&out[5 + def_len..], w, non_null_count);
                let expected_vals: Vec<u32> = cats.iter().copied().filter(|&c| c != NULL_CAT).map(u32::from).collect();
                assert_eq!(vals, expected_vals, "val mismatch len={len} mode={mode}");
            }
        }
    }

    #[test]
    fn test_fast_parquet_empty_file_and_realistic_throughput() {
        let tmp_dir = std::env::temp_dir();
        let empty_path = tmp_dir.join("fast_pq_test_empty.parquet");
        {
            let w = FastUserDailyWriter::create(&empty_path);
            w.finish();
        }
        let venv_py = Path::new("/home/colton/Documents/Projects/Builders/#3/.venv/bin/python");
        if venv_py.exists() {
            let out = Command::new(venv_py)
                .arg("-c")
                .arg("import sys, polars as pl; df = pl.read_parquet(sys.argv[1]); assert df.height == 0 and len(df.columns) == 6")
                .arg(empty_path.to_str().unwrap())
                .output()
                .unwrap();
            assert!(out.status.success(), "empty parquet read failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        let _ = std::fs::remove_file(&empty_path);

        // Realistic 100k-row dataset (~95% sessions==1, long events runs, ~85% zero-revenue)
        let n = 100_000usize;
        let (nd, no, pk, day, mut sess, mut ev, rev, cat, cd, co) = build_sample_dataset(n);
        for i in 0..n {
            sess[i] = if i % 20 == 0 { 2 + (i % 3) as i32 } else { 1 };
            ev[i] = 1 + ((i / 25) % 8) as i32;
        }
        let cols = UserDailyCols {
            names_data: &nd,
            names_offsets: &no,
            person_keys: &pk,
            day: &day,
            sessions: &sess,
            events: &ev,
            revenue_cents: &rev,
            top_category: &cat,
            cat_dict_data: &cd,
            cat_dict_offsets: &co,
        };
        let bench_path = tmp_dir.join("fast_pq_test_100k.parquet");
        let t0 = std::time::Instant::now();
        {
            let mut w = FastUserDailyWriter::create(&bench_path);
            w.write_batches_split(&cols, 4096);
            w.finish();
        }
        let elapsed = t0.elapsed();
        let sz = std::fs::metadata(&bench_path).unwrap().len();
        eprintln!("100k split write: {:?}, size={} bytes ({:.2} B/row)", elapsed, sz, sz as f64 / n as f64);
        verify_with_polars_and_duckdb(&bench_path, &cols, &[]);
        let _ = std::fs::remove_file(&bench_path);
    }
}
