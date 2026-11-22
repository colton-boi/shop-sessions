use crate::group::Cell;
use std::arch::x86_64::*;
use std::fs::File;
use std::io::{IoSlice, Write};
use std::path::Path;

pub const NULL_CAT: u8 = 0xFF;

const TYPE_INT32: i32 = 1;
const TYPE_INT64: i32 = 2;
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

        // +8 slack allows full 8-lane AVX-512VL compressed stores without trailing bounds checks.
        let rcap = cap + 8;
        self.zero_person.reserve(rcap);
        self.zero_day.reserve(rcap);
        self.zero_sessions.reserve(rcap);
        self.zero_events.reserve(rcap);
        self.zero_revenue.reserve(rcap);
        self.zero_category.reserve(rcap);

        self.nonz_person.reserve(rcap);
        self.nonz_day.reserve(rcap);
        self.nonz_sessions.reserve(rcap);
        self.nonz_events.reserve(rcap);
        self.nonz_revenue.reserve(rcap);
        self.nonz_category.reserve(rcap);
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
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
unsafe fn count_runs_u32_avx512vl(keys: &[u32]) -> usize {
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
        let neq = _mm256_cmpneq_epi32_mask(cur, prev);
        runs += (neq as u32).count_ones() as usize;
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

/// Fused `(src[i] - base)` + 8-lane SWAR bit-packing (`1 <= width <= 8`) directly into `out`,
/// avoiding an intermediate `Vec<u32>` pass.
fn encode_sub_min_bitpacked_u8(values: &[i32], base: i32, width: u8, out: &mut Vec<u8>) {
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
    #[cfg(target_arch = "x86_64")]
    unsafe {
        encode_sub_min_bitpacked_avx2(values, base, w, out);
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let total_bytes = num_groups * w;
        out.reserve(total_bytes + 8);
        let start_len = out.len();
        let mut dst = unsafe { out.as_mut_ptr().add(start_len) };
        for g in 0..num_groups {
            let off = g * 8;
            let rem = (values.len() - off).min(8);
            let mut x = 0u64;
            for i in 0..rem {
                let v = (values[off + i] - base) as u8 as u64;
                x |= v << (i * 8);
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
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn encode_sub_min_bitpacked_avx2(values: &[i32], base: i32, width: usize, out: &mut Vec<u8>) {
    let n = values.len();
    let num_groups = n.div_ceil(8);
    let total_bytes = num_groups * width;
    out.reserve(total_bytes + 8);
    let start_len = out.len();
    let src = values.as_ptr();
    let mut dst = out.as_mut_ptr().add(start_len);
    let bvec = _mm256_set1_epi32(base);
    let zero = _mm256_setzero_si256();
    let zero128 = _mm_setzero_si128();
    let full_groups = n / 8;
    for g in 0..full_groups {
        let v = _mm256_sub_epi32(_mm256_loadu_si256(src.add(g * 8) as *const __m256i), bvec);
        let p16 = _mm256_packus_epi32(v, zero);
        let p16_ord = _mm256_permute4x64_epi64(p16, 0xD8);
        let p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16_ord), zero128);
        let x = (_mm_cvtsi128_si64(p8) as u64).to_le();
        let packed = swar_pack8_u8(x, width);
        (dst as *mut u64).write_unaligned(packed.to_le());
        dst = dst.add(width);
    }
    let rem = n - full_groups * 8;
    if rem > 0 {
        let mut x = 0u64;
        for i in 0..rem {
            let d = (*src.add(full_groups * 8 + i) - base) as u8 as u64;
            x |= d << (i * 8);
        }
        let packed = swar_pack8_u8(x, width);
        (dst as *mut u64).write_unaligned(packed.to_le());
    }
    out.set_len(start_len + total_bytes);
}

#[inline]
fn count_runs_u32(keys: &[u32]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512() {
            return unsafe { count_runs_u32_avx512vl(keys) };
        }
        unsafe { count_runs_u32_avx2(keys) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if keys.is_empty() {
            return 0;
        }
        let mut r = 1usize;
        for i in 1..keys.len() {
            if keys[i] != keys[i - 1] {
                r += 1;
            }
        }
        r
    }
}

/// Encodes `keys` (relative to `base_key`) using Parquet RLE/Bit-Packing Hybrid (`RLE_DICTIONARY`
/// payload after the 1-byte bit-width). Chooses the smaller of pure RLE runs and pure bit-packing.
pub fn encode_rle_or_bitpack_keys(keys: &[u32], base_key: u32, width: u8, out: &mut Vec<u8>) {
    if keys.is_empty() {
        out.push(width);
        return;
    }
    let runs = count_runs_u32(keys);
    encode_rle_or_bitpack_keys_with_runs(keys, base_key, width, runs, out);
}

/// Bit-packs `keys` (relative to `base_key`) at `width` bits per key with the Parquet RLE-hybrid
/// bit-packed header (`(num_groups << 1) | 1`).
fn encode_bitpacked_keys(keys: &[u32], base_key: u32, width: u8, out: &mut Vec<u8>) {
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

/// Encodes `keys` when `runs` (the number of contiguous equal-value runs in `keys`) is already known.
fn encode_rle_or_bitpack_keys_with_runs(
    keys: &[u32],
    base_key: u32,
    width: u8,
    runs: usize,
    out: &mut Vec<u8>,
) {
    out.push(width);
    if keys.is_empty() {
        return;
    }
    let val_bytes = (width as usize).div_ceil(8);
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
        encode_bitpacked_keys(keys, base_key, width, out);
    }
}

/// Zero-scan RLE/Bit-Packing encoding of dense dictionary keys `0..run_lens.len()` using
/// pre-tracked per-person run lengths.
fn encode_rle_or_bitpack_keys_from_run_lens(
    keys: &[u32],
    run_lens: &[u32],
    width: u8,
    out: &mut Vec<u8>,
) {
    out.push(width);
    if keys.is_empty() {
        return;
    }
    let val_bytes = (width as usize).div_ceil(8);
    let rle_est = run_lens.len() * (1 + val_bytes);
    let bp_groups = keys.len().div_ceil(8);
    let bp_est = 2 + bp_groups * (width as usize);

    if rle_est <= bp_est {
        for (k, &rl) in run_lens.iter().enumerate() {
            emit_rle_run(k as u32, rl as usize, val_bytes, out);
        }
    } else {
        encode_bitpacked_keys(keys, 0, width, out);
    }
}

/// Fast check whether `cats` contains at least one non-null (`!= NULL_CAT`) byte.
#[inline]
fn has_non_null_u8(cats: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512() {
            return unsafe { has_non_null_u8_avx512vl(cats) };
        }
        return unsafe { has_non_null_u8_avx2(cats) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        cats.iter().any(|&c| c != NULL_CAT)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
unsafe fn has_non_null_u8_avx512vl(cats: &[u8]) -> bool {
    let n = cats.len();
    let ptr = cats.as_ptr();
    let null_vec = _mm256_set1_epi8(-1);
    let mut i = 0usize;
    while i + 64 <= n {
        let v0 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let v1 = _mm256_loadu_si256(ptr.add(i + 32) as *const __m256i);
        if (_mm256_cmpneq_epi8_mask(v0, null_vec) | _mm256_cmpneq_epi8_mask(v1, null_vec)) != 0 {
            return true;
        }
        i += 64;
    }
    while i + 32 <= n {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        if _mm256_cmpneq_epi8_mask(v, null_vec) != 0 {
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
    cat_u8_buf.reserve(n + 16);

    let def_start = out.len();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if use_avx512() {
            encode_top_category_defs_avx512vl(
                cats.as_ptr(),
                n,
                out.as_mut_ptr().add(def_start),
                cat_u8_buf,
            );
        } else {
            encode_top_category_defs_avx2(
                cats.as_ptr(),
                n,
                out.as_mut_ptr().add(def_start),
                cat_u8_buf,
            );
        }
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
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
unsafe fn encode_top_category_defs_avx512vl(
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
        let present = _mm256_cmpneq_epi8_mask(v, null_vec);
        (def_dst as *mut u32).write_unaligned(present.to_le());
        def_dst = def_dst.add(4);
        if present == 0xFFFF_FFFF {
            _mm256_storeu_si256(cat_dst as *mut __m256i, v);
            cat_dst = cat_dst.add(32);
        } else if present != 0 {
            for q in 0..4 {
                let m8 = ((present >> (q * 8)) & 0xFF) as __mmask8;
                if m8 != 0 {
                    let v8 = _mm_loadl_epi64(cats_ptr.add(i + q * 8) as *const __m128i);
                    let v32 = _mm256_cvtepu8_epi32(v8);
                    let comp = _mm256_maskz_compress_epi32(m8, v32);
                    let packed = _mm256_cvtepi32_epi8(comp);
                    _mm_storel_epi64(cat_dst as *mut __m128i, packed);
                    cat_dst = cat_dst.add(m8.count_ones() as usize);
                }
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

/// Returns `true` when 256-bit AVX-512VL should be used.
/// When compiled with `-C target-cpu=cascadelake` (`avx512f+avx512bw+avx512vl`),
/// this folds to a compile-time constant `true` with zero runtime overhead unless `SS_NO_AVX512` is set.
#[inline(always)]
fn use_avx512() -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vl"
    ))]
    {
        true
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vl"
        ))
    ))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && std::env::var_os("SS_NO_AVX512").is_none()
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Checks whether a slice of `i32` has identical min and max (and returns `(min, max)`).
/// Uses 256-bit AVX-512VL when available (or compiled with AVX-512 target features), else AVX2.
/// Kept out-of-line so PGO profiles generated on x86-64-v3 stay valid for column writers.
#[inline(never)]
pub fn min_max_i32(slice: &[i32]) -> (i32, i32) {
    debug_assert!(!slice.is_empty());
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512() {
            return unsafe { min_max_i32_avx512vl(slice) };
        }
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
unsafe fn min_max_i32_avx512vl(slice: &[i32]) -> (i32, i32) {
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
    let min128 = _mm_min_epi32(_mm256_castsi256_si128(vmin), _mm256_extracti128_si256(vmin, 1));
    let min64 = _mm_min_epi32(min128, _mm_shuffle_epi32(min128, 0x4E));
    let min32 = _mm_min_epi32(min64, _mm_shuffle_epi32(min64, 0xB1));
    let mut lo = _mm_cvtsi128_si32(min32);

    let max128 = _mm_max_epi32(_mm256_castsi256_si128(vmax), _mm256_extracti128_si256(vmax, 1));
    let max64 = _mm_max_epi32(max128, _mm_shuffle_epi32(max128, 0x4E));
    let max32 = _mm_max_epi32(max64, _mm_shuffle_epi32(max64, 0xB1));
    let mut hi = _mm_cvtsi128_si32(max32);

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
/// Uses 256-bit AVX-512VL `_mm256_maskz_compress_epi32` when available, else 256-bit AVX2.
pub fn split_zero_revenue_chunks<'a>(
    cols: &UserDailyCols<'a>,
    scratch: &'a mut SplitScratch,
) -> (UserDailyCols<'a>, UserDailyCols<'a>) {
    split_zero_revenue_impl::<true>(cols, scratch)
}

#[inline(never)]
fn split_zero_revenue_impl<'a, const FILL_ZERO_CONSTS: bool>(
    cols: &UserDailyCols<'a>,
    scratch: &'a mut SplitScratch,
) -> (UserDailyCols<'a>, UserDailyCols<'a>) {
    let n = cols.len();
    scratch.prepare(n);
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512() {
            unsafe { split_zero_revenue_avx512vl::<FILL_ZERO_CONSTS>(cols, scratch) };
        } else {
            unsafe { split_zero_revenue_avx2::<FILL_ZERO_CONSTS>(cols, scratch) };
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        unsafe { split_zero_revenue_avx2::<FILL_ZERO_CONSTS>(cols, scratch) };
    }
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,bmi2,lzcnt")]
unsafe fn split_zero_revenue_avx512vl<const FILL_ZERO_CONSTS: bool>(
    cols: &UserDailyCols<'_>,
    scratch: &mut SplitScratch,
) {
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

    let zero32 = _mm256_setzero_si256();
    let null8 = _mm_set1_epi8(-1);

    while i + 8 <= n {
        let r = _mm256_loadu_si256(cols.revenue_cents.as_ptr().add(i) as *const __m256i);
        let r_zero: __mmask8 = _mm256_cmpeq_epi32_mask(r, zero32);
        let c = _mm_loadl_epi64(cols.top_category.as_ptr().add(i) as *const __m128i);
        let c_null = (_mm_cmpeq_epi8_mask(c, null8) & 0xFF) as __mmask8;
        let zmask: __mmask8 = r_zero & c_null;

        let vp = _mm256_loadu_si256(cols.person_keys.as_ptr().add(i) as *const __m256i);
        let vd = _mm256_loadu_si256(cols.day.as_ptr().add(i) as *const __m256i);
        let vs = _mm256_loadu_si256(cols.sessions.as_ptr().add(i) as *const __m256i);
        let ve = _mm256_loadu_si256(cols.events.as_ptr().add(i) as *const __m256i);

        if zmask == 0xFF {
            _mm256_storeu_si256(zp.add(z_len) as *mut __m256i, vp);
            _mm256_storeu_si256(zd.add(z_len) as *mut __m256i, vd);
            _mm256_storeu_si256(zs.add(z_len) as *mut __m256i, vs);
            _mm256_storeu_si256(ze.add(z_len) as *mut __m256i, ve);
            z_len += 8;
        } else {
            let z_cnt = zmask.count_ones() as usize;
            let nz_cnt = 8 - z_cnt;
            if z_cnt > 0 {
                _mm256_storeu_si256(zp.add(z_len) as *mut __m256i, _mm256_maskz_compress_epi32(zmask, vp));
                _mm256_storeu_si256(zd.add(z_len) as *mut __m256i, _mm256_maskz_compress_epi32(zmask, vd));
                _mm256_storeu_si256(zs.add(z_len) as *mut __m256i, _mm256_maskz_compress_epi32(zmask, vs));
                _mm256_storeu_si256(ze.add(z_len) as *mut __m256i, _mm256_maskz_compress_epi32(zmask, ve));
                z_len += z_cnt;
            }
            let nzmask: __mmask8 = !zmask;
            _mm256_storeu_si256(nzp.add(nz_len) as *mut __m256i, _mm256_maskz_compress_epi32(nzmask, vp));
            _mm256_storeu_si256(nzd.add(nz_len) as *mut __m256i, _mm256_maskz_compress_epi32(nzmask, vd));
            _mm256_storeu_si256(nzs.add(nz_len) as *mut __m256i, _mm256_maskz_compress_epi32(nzmask, vs));
            _mm256_storeu_si256(nze.add(nz_len) as *mut __m256i, _mm256_maskz_compress_epi32(nzmask, ve));
            _mm256_storeu_si256(nzr.add(nz_len) as *mut __m256i, _mm256_maskz_compress_epi32(nzmask, r));
            let mut m = nzmask;
            let mut dst_c = nzc.add(nz_len);
            while m != 0 {
                let bit = m.trailing_zeros() as usize;
                *dst_c = *cols.top_category.get_unchecked(i + bit);
                dst_c = dst_c.add(1);
                m &= m - 1;
            }
            nz_len += nz_cnt;
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
    if FILL_ZERO_CONSTS {
        scratch.zero_revenue.resize(z_len, 0);
        scratch.zero_category.resize(z_len, NULL_CAT);
    }

    scratch.nonz_person.set_len(nz_len);
    scratch.nonz_day.set_len(nz_len);
    scratch.nonz_sessions.set_len(nz_len);
    scratch.nonz_events.set_len(nz_len);
    scratch.nonz_revenue.set_len(nz_len);
    scratch.nonz_category.set_len(nz_len);
}

#[target_feature(enable = "avx2,bmi2,lzcnt")]
unsafe fn split_zero_revenue_avx2<const FILL_ZERO_CONSTS: bool>(
    cols: &UserDailyCols<'_>,
    scratch: &mut SplitScratch,
) {
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
    if FILL_ZERO_CONSTS {
        scratch.zero_revenue.resize(z_len, 0);
        scratch.zero_category.resize(z_len, NULL_CAT);
    }

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
/// Streams each row group directly to `file` while tracking `file_offset` so memory stays bounded.
pub struct FastUserDailyWriter {
    file: File,
    file_offset: u64,
    buf: Vec<u8>,
    total_rows: u64,
    revenue_i64: bool,
    groups: Vec<RowGroupMeta>,
    page_buf: Vec<u8>,
    dict_buf: Vec<u8>,
    idx_buf: Vec<u32>,
    cat_u8_buf: Vec<u8>,
    split_scratch: SplitScratch,
}

fn emit_dict_page_into(
    base_offset: u64,
    buf: &mut Vec<u8>,
    num_values: usize,
    payload: &[u8],
) -> (u64, u64) {
    let off = buf.len() as u64;
    {
        let mut t = ThriftWriter::new(buf);
        t.i32(1, P_DICT);
        t.i32(2, payload.len() as i32);
        t.i32(3, payload.len() as i32);
        t.begin_field(7); // DictionaryPageHeader
        t.i32(1, num_values as i32);
        t.i32(2, E_PLAIN);
        t.end_struct();
        buf.push(0);
    }
    buf.extend_from_slice(payload);
    let written = buf.len() as u64 - off;
    (base_offset + off, written)
}

fn emit_data_page_into(
    base_offset: u64,
    buf: &mut Vec<u8>,
    num_values: usize,
    encoding: i32,
    payload: &[u8],
) -> (u64, u64) {
    let off = buf.len() as u64;
    {
        let mut t = ThriftWriter::new(buf);
        t.i32(1, P_DATA);
        t.i32(2, payload.len() as i32);
        t.i32(3, payload.len() as i32);
        t.begin_field(5); // DataPageHeader
        t.i32(1, num_values as i32);
        t.i32(2, encoding);
        t.i32(3, E_RLE); // definition_level_encoding
        t.i32(4, E_RLE); // repetition_level_encoding
        t.end_struct();
        buf.push(0);
    }
    buf.extend_from_slice(payload);
    let written = buf.len() as u64 - off;
    (base_offset + off, written)
}

fn write_two_vectored(file: &mut File, mut a: &[u8], mut b: &[u8]) {
    while !a.is_empty() || !b.is_empty() {
        let res = if !a.is_empty() && !b.is_empty() {
            let iov = [IoSlice::new(a), IoSlice::new(b)];
            file.write_vectored(&iov)
        } else if !a.is_empty() {
            file.write(a)
        } else {
            file.write(b)
        };
        match res {
            Ok(0) => panic!("write Parquet output returned 0"),
            Ok(n) => {
                if n <= a.len() {
                    a = &a[n..];
                } else {
                    let rem = n - a.len();
                    a = &[];
                    b = &b[rem..];
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => panic!("write Parquet output failed: {e}"),
        }
    }
}

impl FastUserDailyWriter {
    pub fn create(path: &Path) -> Self {
        let file = File::create(path).expect("create fast_parquet output");
        let mut buf = Vec::with_capacity(512);
        buf.extend_from_slice(b"PAR1");
        FastUserDailyWriter {
            file,
            file_offset: 0,
            buf,
            total_rows: 0,
            revenue_i64: false,
            groups: Vec::new(),
            page_buf: Vec::with_capacity(64 << 10),
            dict_buf: Vec::with_capacity(64 << 10),
            idx_buf: Vec::with_capacity(4096),
            cat_u8_buf: Vec::with_capacity(4096),
            split_scratch: SplitScratch::new(),
        }
    }

    #[inline(always)]
    pub fn is_revenue_i64(&self) -> bool {
        self.revenue_i64
    }

    #[inline(always)]
    pub fn set_revenue_i64(&mut self, v: bool) {
        assert!(self.total_rows == 0 || self.revenue_i64 == v);
        self.revenue_i64 = v;
    }

    #[inline(always)]
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    #[inline(always)]
    fn flush_buf(&mut self) {
        if !self.buf.is_empty() {
            self.file.write_all(&self.buf).expect("write Parquet row group");
            self.file_offset += self.buf.len() as u64;
            self.buf.clear();
        }
    }

    /// Streams a pre-split zero-revenue (`revenue_cents == 0`, `top_category == NULL`) chunk
    /// directly as one Parquet row group using pre-encoded `PLAIN` user_id dictionary bytes.
    pub fn write_zero_rev_direct(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
    ) {
        self.write_zero_rev_direct_impl(names_plain, names_count, person_keys, None, day, sessions, events);
    }

    /// Streams a pre-split zero-revenue chunk using pre-tracked `run_lens` (zero key scan).
    #[allow(clippy::too_many_arguments)]
    pub fn write_zero_rev_direct_with_runs(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        run_lens: &[u32],
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
    ) {
        self.write_zero_rev_direct_impl(
            names_plain,
            names_count,
            person_keys,
            Some(run_lens),
            day,
            sessions,
            events,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_zero_rev_direct_impl(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        run_lens: Option<&[u32]>,
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
    ) {
        let n = day.len();
        if n == 0 {
            return;
        }
        debug_assert_eq!(person_keys.len(), n);
        debug_assert_eq!(sessions.len(), n);
        debug_assert_eq!(events.len(), n);
        self.buf.reserve(n.saturating_mul(4));

        let c0 = self.write_user_id_plain_dict_col_impl(names_plain, names_count, person_keys, run_lens);
        let c1 = self.write_day_col(day);
        let c2 = self.write_sessions_col(sessions);
        let c3 = self.write_events_col(events);
        let c4 = if self.revenue_i64 {
            self.write_const_i64_col(n, 0)
        } else {
            self.write_const_i32_col(n, 0)
        };
        let c5 = self.write_all_null_category_col(n);

        self.total_rows += n as u64;
        self.groups.push(RowGroupMeta {
            num_rows: n as u64,
            cols: [c0, c1, c2, c3, c4, c5],
        });
        self.flush_buf();
    }

    /// Streams a pre-split non-zero chunk directly as one Parquet row group using pre-encoded
    /// `PLAIN` user_id dictionary bytes (`i32` revenue).
    #[allow(clippy::too_many_arguments)]
    pub fn write_nonz_direct(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
        revenue_cents: &[i32],
        top_category: &[u8],
        cat_dict_data: &[u8],
        cat_dict_offsets: &[u32],
    ) {
        self.write_nonz_direct_impl(
            names_plain,
            names_count,
            person_keys,
            None,
            day,
            sessions,
            events,
            revenue_cents,
            top_category,
            cat_dict_data,
            cat_dict_offsets,
        );
    }

    /// Streams a pre-split non-zero chunk using pre-tracked `run_lens` (zero key scan, `i32` revenue).
    #[allow(clippy::too_many_arguments)]
    pub fn write_nonz_direct_with_runs(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        run_lens: &[u32],
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
        revenue_cents: &[i32],
        top_category: &[u8],
        cat_dict_data: &[u8],
        cat_dict_offsets: &[u32],
    ) {
        self.write_nonz_direct_impl(
            names_plain,
            names_count,
            person_keys,
            Some(run_lens),
            day,
            sessions,
            events,
            revenue_cents,
            top_category,
            cat_dict_data,
            cat_dict_offsets,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_nonz_direct_impl(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        run_lens: Option<&[u32]>,
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
        revenue_cents: &[i32],
        top_category: &[u8],
        cat_dict_data: &[u8],
        cat_dict_offsets: &[u32],
    ) {
        let n = day.len();
        if n == 0 {
            return;
        }
        debug_assert_eq!(person_keys.len(), n);
        debug_assert_eq!(sessions.len(), n);
        debug_assert_eq!(events.len(), n);
        debug_assert_eq!(revenue_cents.len(), n);
        debug_assert_eq!(top_category.len(), n);
        self.buf.reserve(n.saturating_mul(6));

        let c0 = self.write_user_id_plain_dict_col_impl(names_plain, names_count, person_keys, run_lens);
        let c1 = self.write_day_col(day);
        let c2 = self.write_sessions_col(sessions);
        let c3 = self.write_events_col(events);
        let c4 = if self.revenue_i64 {
            self.write_i32_as_i64_const_or_plain(revenue_cents)
        } else {
            self.write_int32_const_or_plain(revenue_cents)
        };
        let c5 = self.write_top_category_slice(top_category, cat_dict_data, cat_dict_offsets);

        self.total_rows += n as u64;
        self.groups.push(RowGroupMeta {
            num_rows: n as u64,
            cols: [c0, c1, c2, c3, c4, c5],
        });
        self.flush_buf();
    }

    /// Streams a pre-split non-zero chunk directly as one Parquet row group with `INT64` revenue.
    #[allow(clippy::too_many_arguments)]
    pub fn write_nonz_i64_direct(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
        revenue_i64: &[u64],
        top_category: &[u8],
        cat_dict_data: &[u8],
        cat_dict_offsets: &[u32],
    ) {
        self.write_nonz_i64_direct_impl(
            names_plain,
            names_count,
            person_keys,
            None,
            day,
            sessions,
            events,
            revenue_i64,
            top_category,
            cat_dict_data,
            cat_dict_offsets,
        );
    }

    /// Streams a pre-split non-zero chunk using pre-tracked `run_lens` (zero key scan, `INT64` revenue).
    #[allow(clippy::too_many_arguments)]
    pub fn write_nonz_i64_direct_with_runs(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        run_lens: &[u32],
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
        revenue_i64: &[u64],
        top_category: &[u8],
        cat_dict_data: &[u8],
        cat_dict_offsets: &[u32],
    ) {
        self.write_nonz_i64_direct_impl(
            names_plain,
            names_count,
            person_keys,
            Some(run_lens),
            day,
            sessions,
            events,
            revenue_i64,
            top_category,
            cat_dict_data,
            cat_dict_offsets,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_nonz_i64_direct_impl(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        run_lens: Option<&[u32]>,
        day: &[i32],
        sessions: &[i32],
        events: &[i32],
        revenue_i64: &[u64],
        top_category: &[u8],
        cat_dict_data: &[u8],
        cat_dict_offsets: &[u32],
    ) {
        let n = day.len();
        if n == 0 {
            return;
        }
        assert!(self.revenue_i64 || self.total_rows == 0);
        self.revenue_i64 = true;
        debug_assert_eq!(person_keys.len(), n);
        debug_assert_eq!(sessions.len(), n);
        debug_assert_eq!(events.len(), n);
        debug_assert_eq!(revenue_i64.len(), n);
        debug_assert_eq!(top_category.len(), n);
        self.buf.reserve(n.saturating_mul(6));

        let c0 = self.write_user_id_plain_dict_col_impl(names_plain, names_count, person_keys, run_lens);
        let c1 = self.write_day_col(day);
        let c2 = self.write_sessions_col(sessions);
        let c3 = self.write_events_col(events);
        let c4 = self.write_u64_const_or_plain(revenue_i64);
        let c5 = self.write_top_category_slice(top_category, cat_dict_data, cat_dict_offsets);

        self.total_rows += n as u64;
        self.groups.push(RowGroupMeta {
            num_rows: n as u64,
            cols: [c0, c1, c2, c3, c4, c5],
        });
        self.flush_buf();
    }

    /// Writes `cols` in chunks of `chunk_rows` (e.g. 4096), splitting each chunk into a
    /// zero-revenue row group (`rev == 0 && cat == 0xFF`) and a non-zero row group.
    pub fn write_batches_split(&mut self, cols: &UserDailyCols<'_>, chunk_rows: usize) {
        let rows = cols.len();
        if rows == 0 {
            return;
        }
        let step = chunk_rows.max(1);
        self.buf.reserve(step.min(rows).saturating_mul(6));
        let mut from = 0usize;
        while from < rows {
            let to = (from + step).min(rows);
            let sub = cols.slice(from, to);
            let mut scratch = std::mem::take(&mut self.split_scratch);
            let (zero_rg, nonz_rg) = split_zero_revenue_impl::<false>(&sub, &mut scratch);
            let z_n = zero_rg.day.len();
            if z_n > 0 {
                self.write_zero_rev_row_group(&zero_rg, z_n);
            }
            if !nonz_rg.is_empty() {
                self.write_row_group(&nonz_rg);
            }
            self.split_scratch = scratch;
            from = to;
        }
    }

    fn write_zero_rev_row_group(&mut self, cols: &UserDailyCols<'_>, n: usize) {
        let c0 = self.write_user_id_col(cols);
        let c1 = self.write_day_col(cols.day);
        let c2 = self.write_sessions_col(cols.sessions);
        let c3 = self.write_events_col(cols.events);
        let c4 = if self.revenue_i64 {
            self.write_const_i64_col(n, 0)
        } else {
            self.write_const_i32_col(n, 0)
        };
        let c5 = self.write_all_null_category_col(n);

        self.total_rows += n as u64;
        self.groups.push(RowGroupMeta {
            num_rows: n as u64,
            cols: [c0, c1, c2, c3, c4, c5],
        });
        self.flush_buf();
    }

    /// Writes `cols` directly as a single Parquet row group without splitting.
    pub fn write_row_group(&mut self, cols: &UserDailyCols<'_>) {
        let n = cols.len();
        if n == 0 {
            return;
        }
        self.buf.reserve(n.saturating_mul(6));
        let c0 = self.write_user_id_col(cols);
        let c1 = self.write_day_col(cols.day);
        let c2 = self.write_sessions_col(cols.sessions);
        let c3 = self.write_events_col(cols.events);
        let c4 = if self.revenue_i64 {
            self.write_i32_as_i64_const_or_plain(cols.revenue_cents)
        } else {
            self.write_int32_const_or_plain(cols.revenue_cents)
        };
        let c5 = self.write_top_category_col(cols);

        self.total_rows += n as u64;
        self.groups.push(RowGroupMeta {
            num_rows: n as u64,
            cols: [c0, c1, c2, c3, c4, c5],
        });
        self.flush_buf();
    }

    /// 1a. `user_id` (`BYTE_ARRAY`, `REQUIRED`) directly from pre-encoded `PLAIN` dictionary bytes.
    fn write_user_id_plain_dict_col_impl(
        &mut self,
        names_plain: &[u8],
        names_count: usize,
        person_keys: &[u32],
        run_lens: Option<&[u32]>,
    ) -> ColChunkMeta {
        debug_assert!(names_count > 0);
        debug_assert!(!person_keys.is_empty());

        let dict_off = self.file_offset + self.buf.len() as u64;
        let hdr_start = self.buf.len();
        {
            let mut t = ThriftWriter::new(&mut self.buf);
            t.i32(1, P_DICT);
            t.i32(2, names_plain.len() as i32);
            t.i32(3, names_plain.len() as i32);
            t.begin_field(7); // DictionaryPageHeader
            t.i32(1, names_count as i32);
            t.i32(2, E_PLAIN);
            t.end_struct();
            self.buf.push(0);
        }
        let hdr_len = (self.buf.len() - hdr_start) as u64;
        let dict_sz = hdr_len + names_plain.len() as u64;
        if names_plain.len() >= 4096 {
            write_two_vectored(&mut self.file, &self.buf, names_plain);
            self.file_offset += (self.buf.len() + names_plain.len()) as u64;
            self.buf.clear();
        } else {
            self.buf.extend_from_slice(names_plain);
        }

        let w = bit_width((names_count - 1) as u32);
        self.page_buf.clear();
        if let Some(rl) = run_lens {
            #[cfg(debug_assertions)]
            {
                assert_eq!(rl.len(), names_count);
                let mut pos = 0usize;
                for (k, &run_len) in rl.iter().enumerate() {
                    assert!(run_len > 0);
                    for _ in 0..run_len {
                        assert_eq!(person_keys[pos], k as u32);
                        pos += 1;
                    }
                }
                assert_eq!(pos, person_keys.len());
                assert_eq!(names_count, count_runs_u32(person_keys));
            }
            encode_rle_or_bitpack_keys_from_run_lens(person_keys, rl, w, &mut self.page_buf);
        } else {
            debug_assert_eq!(names_count, count_runs_u32(person_keys));
            encode_rle_or_bitpack_keys_with_runs(person_keys, 0, w, names_count, &mut self.page_buf);
        }
        let (data_off, data_sz) =
            emit_data_page_into(self.file_offset, &mut self.buf, person_keys.len(), E_RLE_DICT, &self.page_buf);

        ColChunkMeta {
            file_offset: dict_off,
            total_size: dict_sz + data_sz,
            data_page_offset: data_off,
            dict_page_offset: Some(dict_off),
            uses_dict: true,
        }
    }

    /// 1b. `user_id` (`BYTE_ARRAY`, `REQUIRED`):
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
        let data_span = cols.names_offsets[hi as usize + 1] - cols.names_offsets[lo as usize];
        self.dict_buf.reserve(dict_count * 4 + data_span as usize);
        for p in lo..=hi {
            let start = cols.names_offsets[p as usize] as usize;
            let end = cols.names_offsets[p as usize + 1] as usize;
            let name = &cols.names_data[start..end];
            self.dict_buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
            self.dict_buf.extend_from_slice(name);
        }
        let (dict_off, dict_sz) = emit_dict_page_into(self.file_offset, &mut self.buf, dict_count, &self.dict_buf);

        let w = bit_width(hi - lo);
        self.page_buf.clear();
        encode_rle_or_bitpack_keys(keys, lo, w, &mut self.page_buf);
        let (data_off, data_sz) =
            emit_data_page_into(self.file_offset, &mut self.buf, keys.len(), E_RLE_DICT, &self.page_buf);

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
        self.dict_buf.clear();
        self.dict_buf.reserve(dict_count * 4);
        for v in min_v..=max_v {
            self.dict_buf.extend_from_slice(&v.to_le_bytes());
        }
        let (dict_off, dict_sz) = emit_dict_page_into(self.file_offset, &mut self.buf, dict_count, &self.dict_buf);

        let w = bit_width(span).max(min_bit_width);
        self.page_buf.clear();
        self.page_buf.push(w);
        if use_rle {
            sub_min_to_u32(values, min_v, &mut self.idx_buf);
            encode_rle_u32(&self.idx_buf, w, &mut self.page_buf);
        } else {
            encode_sub_min_bitpacked_u8(values, min_v, w, &mut self.page_buf);
        }
        let (data_off, data_sz) =
            emit_data_page_into(self.file_offset, &mut self.buf, values.len(), E_RLE_DICT, &self.page_buf);

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
            self.write_const_i32_col(days.len(), min_d)
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
            self.write_const_i32_col(sessions.len(), min_s)
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
            self.write_const_i32_col(values.len(), min_v)
        } else if span < 256 {
            self.write_dense_i32_dict_col(values, min_v, max_v, 1, true)
        } else {
            self.write_plain_i32(values)
        }
    }

    fn write_const_i32_col(&mut self, n: usize, val: i32) -> ColChunkMeta {
        let dict_bytes = val.to_le_bytes();
        let (dict_off, dict_sz) = emit_dict_page_into(self.file_offset, &mut self.buf, 1, &dict_bytes);

        self.page_buf.clear();
        self.page_buf.push(1); // bit-width = 1
        push_varint(&mut self.page_buf, (n as u64) << 1);
        self.page_buf.push(0); // dictionary index 0
        let (data_off, data_sz) =
            emit_data_page_into(self.file_offset, &mut self.buf, n, E_RLE_DICT, &self.page_buf);

        ColChunkMeta {
            file_offset: dict_off,
            total_size: dict_sz + data_sz,
            data_page_offset: data_off,
            dict_page_offset: Some(dict_off),
            uses_dict: true,
        }
    }

    fn write_const_i64_col(&mut self, n: usize, val: u64) -> ColChunkMeta {
        let dict_bytes = val.to_le_bytes();
        let (dict_off, dict_sz) = emit_dict_page_into(self.file_offset, &mut self.buf, 1, &dict_bytes);

        self.page_buf.clear();
        self.page_buf.push(1); // bit-width = 1
        push_varint(&mut self.page_buf, (n as u64) << 1);
        self.page_buf.push(0); // dictionary index 0
        let (data_off, data_sz) =
            emit_data_page_into(self.file_offset, &mut self.buf, n, E_RLE_DICT, &self.page_buf);

        ColChunkMeta {
            file_offset: dict_off,
            total_size: dict_sz + data_sz,
            data_page_offset: data_off,
            dict_page_offset: Some(dict_off),
            uses_dict: true,
        }
    }

    /// 5. `revenue_cents` (`INT32`, `REQUIRED`):
    /// When all values in the row group are identical (e.g. `revenue_cents == 0`), write a 1-entry
    /// `INT32` dictionary page (4 bytes) + a single RLE run `(n << 1, 0x00)`; otherwise `PLAIN`.
    fn write_int32_const_or_plain(&mut self, values: &[i32]) -> ColChunkMeta {
        let (min_v, max_v) = min_max_i32(values);
        if min_v == max_v {
            self.write_const_i32_col(values.len(), min_v)
        } else {
            self.write_plain_i32(values)
        }
    }

    fn write_i32_as_i64_const_or_plain(&mut self, values: &[i32]) -> ColChunkMeta {
        let (min_v, max_v) = min_max_i32(values);
        if min_v == max_v {
            self.write_const_i64_col(values.len(), min_v as i64 as u64)
        } else {
            self.page_buf.clear();
            self.page_buf.reserve(values.len() * 8);
            for &v in values {
                self.page_buf.extend_from_slice(&(v as i64).to_le_bytes());
            }
            let (data_off, data_sz) =
                emit_data_page_into(self.file_offset, &mut self.buf, values.len(), E_PLAIN, &self.page_buf);
            ColChunkMeta {
                file_offset: data_off,
                total_size: data_sz,
                data_page_offset: data_off,
                dict_page_offset: None,
                uses_dict: false,
            }
        }
    }

    fn write_u64_const_or_plain(&mut self, values: &[u64]) -> ColChunkMeta {
        let first = values[0];
        if values[1..].iter().all(|&v| v == first) {
            self.write_const_i64_col(values.len(), first)
        } else {
            let raw_bytes = unsafe {
                std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8)
            };
            let (data_off, data_sz) =
                emit_data_page_into(self.file_offset, &mut self.buf, values.len(), E_PLAIN, raw_bytes);
            ColChunkMeta {
                file_offset: data_off,
                total_size: data_sz,
                data_page_offset: data_off,
                dict_page_offset: None,
                uses_dict: false,
            }
        }
    }

    fn write_plain_i32(&mut self, values: &[i32]) -> ColChunkMeta {
        let raw_bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
        };
        let (data_off, data_sz) =
            emit_data_page_into(self.file_offset, &mut self.buf, values.len(), E_PLAIN, raw_bytes);
        ColChunkMeta {
            file_offset: data_off,
            total_size: data_sz,
            data_page_offset: data_off,
            dict_page_offset: None,
            uses_dict: false,
        }
    }

    fn write_all_null_category_col(&mut self, n: usize) -> ColChunkMeta {
        self.page_buf.clear();
        let hdr = (n as u64) << 1;
        let def_len = varint_len(hdr) + 1;
        self.page_buf.extend_from_slice(&(def_len as u32).to_le_bytes());
        push_varint(&mut self.page_buf, hdr);
        self.page_buf.push(0); // def level 0 (NULL) repeated n times
        let (data_off, data_sz) =
            emit_data_page_into(self.file_offset, &mut self.buf, n, E_PLAIN, &self.page_buf);

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
        self.write_top_category_slice(cols.top_category, cols.cat_dict_data, cols.cat_dict_offsets)
    }

    fn write_top_category_slice(
        &mut self,
        cats: &[u8],
        cat_dict_data: &[u8],
        cat_dict_offsets: &[u32],
    ) -> ColChunkMeta {
        let n = cats.len();

        if !has_non_null_u8(cats) {
            self.write_all_null_category_col(n)
        } else {
            let num_cats = cat_dict_offsets.len().saturating_sub(1);
            self.dict_buf.clear();
            self.dict_buf
                .reserve(num_cats * 4 + cat_dict_data.len());
            for c in 0..num_cats {
                let s = cat_dict_offsets[c] as usize;
                let e = cat_dict_offsets[c + 1] as usize;
                let bytes = &cat_dict_data[s..e];
                self.dict_buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                self.dict_buf.extend_from_slice(bytes);
            }
            let (dict_off, dict_sz) = emit_dict_page_into(self.file_offset, &mut self.buf, num_cats, &self.dict_buf);

            self.page_buf.clear();
            encode_top_category_chunk(cats, num_cats, &mut self.cat_u8_buf, &mut self.page_buf);

            let (data_off, data_sz) =
                emit_data_page_into(self.file_offset, &mut self.buf, n, E_RLE_DICT, &self.page_buf);

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

        let rev_ty = if self.revenue_i64 { TYPE_INT64 } else { TYPE_INT32 };
        let specs: [ColSpec; 6] = [
            ColSpec { name: b"user_id", ty: TYPE_BYTE_ARRAY, rep: REP_REQUIRED, converted: Some(0), logical_tag: Some(1) },
            ColSpec { name: b"day", ty: TYPE_INT32, rep: REP_REQUIRED, converted: Some(6), logical_tag: Some(6) },
            ColSpec { name: b"sessions", ty: TYPE_INT32, rep: REP_REQUIRED, converted: None, logical_tag: None },
            ColSpec { name: b"events", ty: TYPE_INT32, rep: REP_REQUIRED, converted: None, logical_tag: None },
            ColSpec { name: b"revenue_cents", ty: rev_ty, rep: REP_REQUIRED, converted: None, logical_tag: None },
            ColSpec { name: b"top_category", ty: TYPE_BYTE_ARRAY, rep: REP_OPTIONAL, converted: Some(0), logical_tag: Some(1) },
        ];

        let footer_start = self.buf.len();
        {
            let mut t = ThriftWriter::new(&mut self.buf);
            t.i32(1, 1); // version = 1
            t.list(2, T_STRUCT, specs.len() + 1);

            // Root schema element
            t.begin_struct();
            t.binary(4, b"schema");
            t.i32(5, specs.len() as i32);
            t.end_struct();

            for spec in &specs {
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
                t.list(1, T_STRUCT, specs.len());
                let mut rg_bytes = 0u64;
                for (spec, chunk) in specs.iter().zip(&g.cols) {
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

/// Reduces `per_thread` campaign-day `Cell` slices for shard `shard_idx` of `num_shards`
/// (covering campaigns `c0..c1` out of `campaigns.len() + 1` including `"organic"`) and writes
/// `out_dir/part-{shard_idx:03}.parquet` in a single `write_all` syscall.
pub fn write_campaigns_shard<S: AsRef<[Cell]>>(
    campaigns: &[(Vec<u8>, Vec<u8>)],
    first_day: i32,
    days: usize,
    out_dir: &Path,
    per_thread: &[S],
    shard_idx: usize,
    num_shards: usize,
) {
    let total_campaigns = campaigns.len() + 1;
    let c0 = shard_idx * total_campaigns / num_shards;
    let c1 = (shard_idx + 1) * total_campaigns / num_shards;

    let mut cols: [Vec<u8>; 7] = Default::default();
    let est_rows = (c1 - c0) * days;
    cols[0].reserve(est_rows * 16);
    cols[1].reserve(est_rows * 12);
    cols[2].reserve(est_rows * 4);
    cols[3].reserve(est_rows * 8);
    cols[4].reserve(est_rows * 8);
    cols[5].reserve(est_rows * 8);
    cols[6].reserve(est_rows * 8);

    let need_cells = total_campaigns * days;
    for cells in per_thread {
        assert!(cells.as_ref().len() >= need_cells);
    }

    let mut rows = 0usize;
    for campaign in c0..c1 {
        let (id, channel): (&[u8], &[u8]) = match campaigns.get(campaign) {
            Some((id, ch)) => (id.as_slice(), ch.as_slice()),
            None => (b"organic", b"organic"),
        };
        let camp_base = campaign * days;
        for day in 0..days {
            let idx = camp_base + day;
            let (mut sessions, mut purchasing, mut assisted, mut revenue) = (0u64, 0u64, 0u64, 0u64);
            for cells in per_thread {
                let c = unsafe { cells.as_ref().get_unchecked(idx) };
                sessions += c.sessions as u64;
                purchasing += c.purchasing as u64;
                assisted += c.assisted as u64;
                revenue += c.revenue;
            }
            if sessions == 0 && assisted == 0 {
                continue;
            }
            cols[0].extend_from_slice(&(id.len() as u32).to_le_bytes());
            cols[0].extend_from_slice(id);
            cols[1].extend_from_slice(&(channel.len() as u32).to_le_bytes());
            cols[1].extend_from_slice(channel);
            cols[2].extend_from_slice(&(first_day + day as i32).to_le_bytes());
            cols[3].extend_from_slice(&sessions.to_le_bytes());
            cols[4].extend_from_slice(&purchasing.to_le_bytes());
            cols[5].extend_from_slice(&revenue.to_le_bytes());
            cols[6].extend_from_slice(&assisted.to_le_bytes());
            rows += 1;
        }
    }

    if rows == 0 && shard_idx > 0 {
        return;
    }

    let total_payload: usize = cols.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(total_payload + 1024);
    buf.extend_from_slice(b"PAR1");

    let mut chunks: [(u64, u64); 7] = [(0, 0); 7];
    if rows > 0 {
        let mut levels = [0u8; 16];
        let mut lvl_len = 0usize;
        let mut v = (rows as u64) << 1;
        while v >= 0x80 {
            levels[lvl_len] = (v as u8) | 0x80;
            lvl_len += 1;
            v >>= 7;
        }
        levels[lvl_len] = v as u8;
        lvl_len += 1;
        levels[lvl_len] = 1; // all values present (def_level = 1)
        lvl_len += 1;

        for (i, col_bytes) in cols.iter().enumerate() {
            let off = buf.len() as u64;
            let page_len = 4 + lvl_len + col_bytes.len();
            {
                let mut t = ThriftWriter::new(&mut buf);
                t.i32(1, P_DATA);
                t.i32(2, page_len as i32);
                t.i32(3, page_len as i32);
                t.begin_field(5);
                t.i32(1, rows as i32);
                t.i32(2, E_PLAIN);
                t.i32(3, E_RLE);
                t.i32(4, E_RLE);
                t.end_struct();
                buf.push(0);
            }
            buf.extend_from_slice(&(lvl_len as u32).to_le_bytes());
            buf.extend_from_slice(&levels[..lvl_len]);
            buf.extend_from_slice(col_bytes);
            let sz = buf.len() as u64 - off;
            chunks[i] = (off, sz);
        }
    }

    #[derive(Clone, Copy)]
    struct CampSpec {
        name: &'static [u8],
        ty: i32,
        converted: Option<i32>,
        logical_tag: Option<i16>,
    }

    const CAMP_SPECS: [CampSpec; 7] = [
        CampSpec { name: b"campaign_id", ty: TYPE_BYTE_ARRAY, converted: Some(0), logical_tag: Some(1) },
        CampSpec { name: b"channel", ty: TYPE_BYTE_ARRAY, converted: Some(0), logical_tag: Some(1) },
        CampSpec { name: b"day", ty: TYPE_INT32, converted: Some(6), logical_tag: Some(6) },
        CampSpec { name: b"sessions", ty: TYPE_INT64, converted: None, logical_tag: None },
        CampSpec { name: b"purchasing_sessions", ty: TYPE_INT64, converted: None, logical_tag: None },
        CampSpec { name: b"revenue_cents", ty: TYPE_INT64, converted: None, logical_tag: None },
        CampSpec { name: b"assisted_sessions", ty: TYPE_INT64, converted: None, logical_tag: None },
    ];

    let footer_start = buf.len();
    {
        let mut t = ThriftWriter::new(&mut buf);
        t.i32(1, 1);
        t.list(2, T_STRUCT, CAMP_SPECS.len() + 1);
        t.begin_struct();
        t.binary(4, b"schema");
        t.i32(5, CAMP_SPECS.len() as i32);
        t.end_struct();
        for spec in &CAMP_SPECS {
            t.begin_struct();
            t.i32(1, spec.ty);
            t.i32(3, REP_OPTIONAL);
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
        t.i64(3, rows as i64);
        let num_rg = usize::from(rows > 0);
        t.list(4, T_STRUCT, num_rg);
        if rows > 0 {
            t.begin_struct();
            t.list(1, T_STRUCT, CAMP_SPECS.len());
            let mut rg_bytes = 0u64;
            for (spec, &(off, sz)) in CAMP_SPECS.iter().zip(&chunks) {
                rg_bytes += sz;
                t.begin_struct();
                t.i64(2, off as i64);
                t.begin_field(3);
                t.i32(1, spec.ty);
                t.list(2, T_I32, 2);
                t.zigzag(E_PLAIN as i64);
                t.zigzag(E_RLE as i64);
                t.list(3, T_BINARY, 1);
                t.varint(spec.name.len() as u64);
                t.buf.extend_from_slice(spec.name);
                t.i32(4, 0);
                t.i64(5, rows as i64);
                t.i64(6, sz as i64);
                t.i64(7, sz as i64);
                t.i64(9, off as i64);
                t.end_struct();
                t.end_struct();
            }
            t.i64(2, rg_bytes as i64);
            t.i64(3, rows as i64);
            t.end_struct();
        }
        buf.push(0);
    }
    let footer_len = (buf.len() - footer_start) as u32;
    buf.extend_from_slice(&footer_len.to_le_bytes());
    buf.extend_from_slice(b"PAR1");

    let part_path = out_dir.join(format!("part-{shard_idx:03}.parquet"));
    let mut file = File::create(&part_path).expect("create campaign_daily shard");
    file.write_all(&buf).expect("write campaign_daily shard");
}

/// Reduces `per_thread` `Cell` slices in parallel across `threads` shards, writing
/// `out_dir/part-{t:03}.parquet` for disjoint campaign slices `c0..c1`.
pub fn write_campaigns_sharded<S: AsRef<[Cell]> + Sync>(
    campaigns: &[(Vec<u8>, Vec<u8>)],
    first_day: i32,
    days: usize,
    out_dir: &Path,
    per_thread: &[S],
    threads: usize,
) {
    let _ = std::fs::create_dir_all(out_dir);
    let total_campaigns = campaigns.len() + 1;
    let num_shards = threads.max(1).min(total_campaigns);
    if num_shards <= 1 {
        write_campaigns_shard(campaigns, first_day, days, out_dir, per_thread, 0, 1);
    } else {
        crate::sys::par(num_shards, |t| {
            write_campaigns_shard(campaigns, first_day, days, out_dir, per_thread, t, num_shards);
        });
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

    #[test]
    fn test_split_zero_revenue_and_min_max_parity() {
        for &len in &[1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 513] {
            let vals: Vec<i32> = (0..len).map(|i| ((i as i32 * 37) % 199) - 99).collect();
            let exp_min = *vals.iter().min().unwrap();
            let exp_max = *vals.iter().max().unwrap();
            assert_eq!(min_max_i32(&vals), (exp_min, exp_max));

            let (nd, no, pk, day, sess, ev, rev, cat, cd, co) = build_sample_dataset(len);
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
            let mut scratch = SplitScratch::new();
            let (z, nz) = split_zero_revenue_chunks(&cols, &mut scratch);
            assert_eq!(z.len() + nz.len(), len);
            assert!(z.revenue_cents.iter().all(|&r| r == 0));
            assert!(z.top_category.iter().all(|&c| c == NULL_CAT));
            for i in 0..nz.len() {
                assert!(nz.revenue_cents[i] != 0 || nz.top_category[i] != NULL_CAT);
            }
        }
    }

    #[test]
    fn test_write_campaigns_sharded_pyarrow_polars_duckdb() {
        let campaigns: Vec<(Vec<u8>, Vec<u8>)> = (0..53)
            .map(|c| {
                let ch = match c % 4 {
                    0 => b"paid_search".to_vec(),
                    1 => b"social".to_vec(),
                    2 => b"email".to_vec(),
                    _ => b"display".to_vec(),
                };
                (format!("camp_{c:03}").into_bytes(), ch)
            })
            .collect();
        let days = 15usize;
        let first_day = 20727i32;
        let total_campaigns = campaigns.len() + 1;
        let threads = 6usize;

        let mut per_thread = vec![vec![Cell::default(); total_campaigns * days]; threads];
        for t in 0..threads {
            for c in 0..total_campaigns {
                // Leave some campaigns completely empty to exercise empty-shard/row skipping
                if c % 11 == 3 {
                    continue;
                }
                for d in 0..days {
                    if (c + d + t) % 3 == 0 {
                        continue;
                    }
                    let cell = &mut per_thread[t][c * days + d];
                    cell.sessions = ((c + 1) * (t + 1) + d) as u32;
                    cell.purchasing = ((c + t) % 5) as u32;
                    cell.assisted = ((c * 3 + d + t) % 7) as u32;
                    cell.revenue = ((c * 1000 + d * 100 + t * 17) as u64) * 10;
                }
            }
        }

        let tmp_dir = std::env::temp_dir().join("fast_pq_campaigns_sharded_test");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let sharded_dir = tmp_dir.join("sharded");
        let single_dir = tmp_dir.join("single");

        write_campaigns_sharded(&campaigns, first_day, days, &sharded_dir, &per_thread, 8);
        write_campaigns_sharded(&campaigns, first_day, days, &single_dir, &per_thread, 1);

        let part_count = std::fs::read_dir(&sharded_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "parquet"))
            .count();
        assert!(part_count >= 2, "expected multiple shards, got {part_count}");

        // Compute exact expected aggregates in Rust
        let mut exp_rows = 0usize;
        let mut exp_sess = 0i64;
        let mut exp_purch = 0i64;
        let mut exp_rev = 0i64;
        let mut exp_assist = 0i64;
        for c in 0..total_campaigns {
            for d in 0..days {
                let (mut s, mut p, mut a, mut r) = (0u64, 0u64, 0u64, 0u64);
                for cells in &per_thread {
                    let cell = cells[c * days + d];
                    s += cell.sessions as u64;
                    p += cell.purchasing as u64;
                    a += cell.assisted as u64;
                    r += cell.revenue;
                }
                if s == 0 && a == 0 {
                    continue;
                }
                exp_rows += 1;
                exp_sess += s as i64;
                exp_purch += p as i64;
                exp_rev += r as i64;
                exp_assist += a as i64;
            }
        }

        let venv_py = Path::new("/home/colton/Documents/Projects/Builders/#3/.venv/bin/python");
        if venv_py.exists() {
            let script = r#"
import glob, sys
import pyarrow.parquet as pq
import polars as pl
import duckdb

sharded_glob = sys.argv[1] + "/*.parquet"
single_path = sys.argv[2] + "/part-000.parquet"

pa_sh = pq.read_table(sys.argv[1])
pa_sg = pq.read_table(single_path)
assert pa_sh.schema.equals(pa_sg.schema), f"schema mismatch: {pa_sh.schema} vs {pa_sg.schema}"

df_sh = pl.read_parquet(sharded_glob).sort(["campaign_id", "day"])
df_sg = pl.read_parquet(single_path).sort(["campaign_id", "day"])
assert df_sh.equals(df_sg), "sharded != single"
assert str(df_sh.schema["day"]) == "Date"

duck_rows = duckdb.sql(f"SELECT COUNT(*), SUM(sessions), SUM(purchasing_sessions), SUM(revenue_cents), SUM(assisted_sessions) FROM read_parquet('{sharded_glob}')").fetchall()[0]
print(f"OK rows={duck_rows[0]} sess={duck_rows[1]} purch={duck_rows[2]} rev={duck_rows[3]} assist={duck_rows[4]}")
"#;
            let out = Command::new(venv_py)
                .arg("-c")
                .arg(script)
                .arg(sharded_dir.to_str().unwrap())
                .arg(single_dir.to_str().unwrap())
                .output()
                .expect("run python verifier for campaign_daily");
            assert!(
                out.status.success(),
                "Campaign verifier failed:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let stdout = String::from_utf8_lossy(&out.stdout);
            let expected = format!(
                "OK rows={exp_rows} sess={exp_sess} purch={exp_purch} rev={exp_rev} assist={exp_assist}"
            );
            assert_eq!(stdout.trim(), expected);
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    fn encode_plain_names(names: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        for name in names {
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
        }
        out
    }

    #[test]
    fn test_fast_parquet_direct_streaming_and_i64_overflow() {
        let tmp_dir = std::env::temp_dir();
        let stream_path = tmp_dir.join("fast_pq_test_direct_stream.parquet");
        let i64_path = tmp_dir.join("fast_pq_test_direct_i64.parquet");

        let categories = ["apparel", "books", "electronics"];
        let mut cat_data = Vec::new();
        let mut cat_off = vec![0u32];
        for c in &categories {
            cat_data.extend_from_slice(c.as_bytes());
            cat_off.push(cat_data.len() as u32);
        }

        // 1. Stream >65,536 zero-rev rows + >65,536 nonz rows across multiple direct row groups.
        let mut exp_rows = 0usize;
        let mut exp_nulls = 0usize;
        let mut exp_rev = 0i64;
        let mut exp_sess = 0i64;
        let mut exp_ev = 0i64;
        {
            let mut w = FastUserDailyWriter::create(&stream_path);
            for chunk in 0..2 {
                let n = 66_000usize;
                let names: Vec<String> = (0..n / 2).map(|i| format!("uz_{chunk}_{i:06}")).collect();
                let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
                let names_plain = encode_plain_names(&name_refs);
                let mut pk = Vec::with_capacity(n);
                let mut day = Vec::with_capacity(n);
                let mut sess = Vec::with_capacity(n);
                let mut ev = Vec::with_capacity(n);
                for i in 0..n {
                    pk.push((i / 2) as u32);
                    day.push(20727 + (i % 15) as i32);
                    sess.push(1 + (i % 3) as i32);
                    ev.push(1 + (i % 10) as i32);
                }
                exp_rows += n;
                exp_nulls += n;
                exp_sess += sess.iter().map(|&v| v as i64).sum::<i64>();
                exp_ev += ev.iter().map(|&v| v as i64).sum::<i64>();
                w.write_zero_rev_direct(&names_plain, name_refs.len(), &pk, &day, &sess, &ev);
            }

            let nz_n = 70_000usize;
            let nz_names: Vec<String> = (0..nz_n).map(|i| format!("unz_{i:06}")).collect();
            let nz_refs: Vec<&str> = nz_names.iter().map(|s| s.as_str()).collect();
            let nz_plain = encode_plain_names(&nz_refs);
            let mut pk = Vec::with_capacity(nz_n);
            let mut day = Vec::with_capacity(nz_n);
            let mut sess = Vec::with_capacity(nz_n);
            let mut ev = Vec::with_capacity(nz_n);
            let mut rev = Vec::with_capacity(nz_n);
            let mut cat = Vec::with_capacity(nz_n);
            for i in 0..nz_n {
                pk.push(i as u32);
                day.push(20727 + (i % 10) as i32);
                sess.push(1 + (i % 5) as i32);
                ev.push(2 + (i % 7) as i32);
                if i % 4 == 0 {
                    rev.push(0);
                    cat.push((i % categories.len()) as u8);
                } else {
                    rev.push(100 + (i % 5000) as i32);
                    cat.push(if i % 3 == 0 { NULL_CAT } else { (i % categories.len()) as u8 });
                }
            }
            exp_rows += nz_n;
            exp_nulls += cat.iter().filter(|&&c| c == NULL_CAT).count();
            exp_rev += rev.iter().map(|&v| v as i64).sum::<i64>();
            exp_sess += sess.iter().map(|&v| v as i64).sum::<i64>();
            exp_ev += ev.iter().map(|&v| v as i64).sum::<i64>();
            w.write_nonz_direct(
                &nz_plain,
                nz_refs.len(),
                &pk,
                &day,
                &sess,
                &ev,
                &rev,
                &cat,
                &cat_data,
                &cat_off,
            );
            w.finish();
        }

        // 2. Test INT64 revenue overflow via write_nonz_i64_direct + write_zero_rev_direct
        let i64_revs = [100u64, (i32::MAX as u64) + 12345, 9_000_000_000u64];
        {
            let mut w = FastUserDailyWriter::create(&i64_path);
            let nz_plain = encode_plain_names(&["whale_1", "whale_2", "whale_3"]);
            w.write_nonz_i64_direct(
                &nz_plain,
                3,
                &[0, 1, 2],
                &[20727, 20728, 20729],
                &[1, 2, 3],
                &[10, 20, 30],
                &i64_revs,
                &[0, NULL_CAT, 2],
                &cat_data,
                &cat_off,
            );
            let z_plain = encode_plain_names(&["zero_1", "zero_2"]);
            w.write_zero_rev_direct(&z_plain, 2, &[0, 1], &[20727, 20728], &[1, 1], &[2, 3]);
            w.finish();
        }

        let venv_py = Path::new("/home/colton/Documents/Projects/Builders/#3/.venv/bin/python");
        if venv_py.exists() {
            let script = r#"
import sys
import polars as pl
import pyarrow.parquet as pq
import duckdb

p1, p2 = sys.argv[1], sys.argv[2]
for p in (p1, p2):
    pq.read_table(p)
    duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchall()

df1 = pl.read_parquet(p1)
print(f"STREAM rows={df1.height} nulls={df1['top_category'].null_count()} rev={df1['revenue_cents'].sum()} sess={df1['sessions'].sum()} ev={df1['events'].sum()}")
df2 = pl.read_parquet(p2)
print(f"I64 rows={df2.height} nulls={df2['top_category'].null_count()} rev={df2['revenue_cents'].sum()} sess={df2['sessions'].sum()} ev={df2['events'].sum()}")
"#;
            let out = Command::new(venv_py)
                .arg("-c")
                .arg(script)
                .arg(stream_path.to_str().unwrap())
                .arg(i64_path.to_str().unwrap())
                .output()
                .expect("run python verifier");
            assert!(
                out.status.success(),
                "Python verifier failed:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let stdout = String::from_utf8_lossy(&out.stdout);
            let lines: Vec<&str> = stdout.lines().collect();
            assert_eq!(
                lines[0],
                format!("STREAM rows={exp_rows} nulls={exp_nulls} rev={exp_rev} sess={exp_sess} ev={exp_ev}")
            );
            let exp_i64_rev: u64 = i64_revs.iter().sum();
            assert_eq!(
                lines[1],
                format!("I64 rows=5 nulls=3 rev={exp_i64_rev} sess=8 ev=65")
            );
        }

        let _ = std::fs::remove_file(&stream_path);
        let _ = std::fs::remove_file(&i64_path);
    }

    #[test]
    fn test_with_runs_byte_identical_parity() {
        let tmp_dir = std::env::temp_dir();
        let (cat_data, cat_off) = crate::group::build_cat_dict(&[b"books".to_vec(), b"music".to_vec()]);

        // Test both RLE-selected (avg run >= 2) and bitpack-selected (avg run == 1)
        for &(n_users, repeats) in &[(1usize, 5usize), (200, 3), (300, 1)] {
            let names: Vec<String> = (0..n_users).map(|i| format!("u{i:04}")).collect();
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let names_plain = encode_plain_names(&name_refs);
            let mut pk = Vec::new();
            let mut run_lens = Vec::new();
            for u in 0..n_users {
                let r = if repeats == 1 { 1 } else { 1 + (u % repeats) };
                run_lens.push(r as u32);
                for _ in 0..r {
                    pk.push(u as u32);
                }
            }
            let n = pk.len();
            let day: Vec<i32> = (0..n).map(|i| 20700 + (i % 10) as i32).collect();
            let sess: Vec<i32> = vec![1; n];
            let ev: Vec<i32> = (0..n).map(|i| 1 + (i % 4) as i32).collect();
            let rev32: Vec<i32> = (0..n).map(|i| 100 + i as i32).collect();
            let rev64: Vec<u64> = (0..n).map(|i| (i32::MAX as u64) + 1 + i as u64).collect();
            let cat: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();

            let p_ref = tmp_dir.join(format!("runs_ref_{n_users}_{repeats}.parquet"));
            let p_run = tmp_dir.join(format!("runs_new_{n_users}_{repeats}.parquet"));

            let mut w1 = FastUserDailyWriter::create(&p_ref);
            w1.write_zero_rev_direct(&names_plain, n_users, &pk, &day, &sess, &ev);
            w1.write_nonz_direct(&names_plain, n_users, &pk, &day, &sess, &ev, &rev32, &cat, &cat_data, &cat_off);
            w1.finish();

            let mut w2 = FastUserDailyWriter::create(&p_run);
            w2.write_zero_rev_direct_with_runs(&names_plain, n_users, &pk, &run_lens, &day, &sess, &ev);
            w2.write_nonz_direct_with_runs(&names_plain, n_users, &pk, &run_lens, &day, &sess, &ev, &rev32, &cat, &cat_data, &cat_off);
            w2.finish();

            assert_eq!(std::fs::read(&p_ref).unwrap(), std::fs::read(&p_run).unwrap());

            let mut w3 = FastUserDailyWriter::create(&p_ref);
            w3.write_nonz_i64_direct(&names_plain, n_users, &pk, &day, &sess, &ev, &rev64, &cat, &cat_data, &cat_off);
            w3.finish();

            let mut w4 = FastUserDailyWriter::create(&p_run);
            w4.write_nonz_i64_direct_with_runs(&names_plain, n_users, &pk, &run_lens, &day, &sess, &ev, &rev64, &cat, &cat_data, &cat_off);
            w4.finish();

            assert_eq!(std::fs::read(&p_ref).unwrap(), std::fs::read(&p_run).unwrap());

            let _ = std::fs::remove_file(&p_ref);
            let _ = std::fs::remove_file(&p_run);
        }
    }
}
