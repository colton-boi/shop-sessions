use crate::sys::mix;
use std::collections::HashMap;
use std::path::Path;

/// Byte-string to index map with a probe on the raw 8 bytes for short keys.
pub struct KeyMap {
    slots: Vec<(u64, u32, u32)>,
    mask: usize,
    long: HashMap<Vec<u8>, u32>,
}

#[inline(always)]
fn short_key(word: u64, len: usize) -> u64 {
    word & (u64::MAX >> (64 - 8 * len))
}

impl KeyMap {
    pub fn empty() -> Self {
        KeyMap { slots: vec![(0, 0, 0); 1], mask: 0, long: HashMap::new() }
    }

    pub fn new(keys: &[&[u8]]) -> Self {
        let cap = (keys.len() * 2).next_power_of_two().max(16);
        let mut m = KeyMap { slots: vec![(0, 0, 0); cap], mask: cap - 1, long: HashMap::new() };
        for (i, k) in keys.iter().enumerate() {
            if k.is_empty() || k.len() > 8 {
                m.long.insert(k.to_vec(), i as u32);
                continue;
            }
            let mut padded = [0u8; 8];
            padded[..k.len()].copy_from_slice(k);
            let key = u64::from_le_bytes(padded);
            let mut at = mix(key ^ k.len() as u64) as usize & m.mask;
            while m.slots[at].1 != 0 {
                at = (at + 1) & m.mask;
            }
            m.slots[at] = (key, k.len() as u32, i as u32);
        }
        m
    }

    /// `p` must be readable for 8 bytes whatever `len` is.
    #[inline(always)]
    pub unsafe fn get(&self, p: *const u8, len: usize) -> Option<u32> {
        if len == 0 || len > 8 {
            return self.get_long(p, len);
        }
        let key = short_key((p as *const u64).read_unaligned(), len);
        let mut at = mix(key ^ len as u64) as usize & self.mask;
        loop {
            let s = self.slots.get_unchecked(at);
            if s.1 == 0 {
                return None;
            }
            if s.0 == key && s.1 == len as u32 {
                return Some(s.2);
            }
            at = (at + 1) & self.mask;
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn get_long(&self, p: *const u8, len: usize) -> Option<u32> {
        self.long.get(std::slice::from_raw_parts(p, len)).copied()
    }
}

fn unquote(f: &[u8]) -> Option<&[u8]> {
    let f = if f.len() >= 2 && f[0] == b'"' && f[f.len() - 1] == b'"' { &f[1..f.len() - 1] } else { f };
    (!f.contains(&b'"')).then_some(f)
}

/// The data lines of a small two-column CSV, by the same dialect as the big files.
fn two_columns(bytes: &[u8]) -> Vec<(&[u8], &[u8])> {
    let mut rows = Vec::new();
    let mut lines = bytes.split(|&c| c == b'\n');
    lines.next();
    if bytes.last() == Some(&b'\n') {
        lines.next_back();
    }
    for line in lines {
        let Some(comma) = line.iter().position(|&c| c == b',') else { continue };
        let (a, b) = (&line[..comma], &line[comma + 1..]);
        if b.contains(&b',') {
            continue;
        }
        if let (Some(a), Some(b)) = (unquote(a), unquote(b)) {
            rows.push((a, b));
        }
    }
    rows
}

pub struct Dims {
    pub products: KeyMap,
    pub product_category: Vec<u8>,
    pub direct_prod_cat: Vec<u8>,
    pub unknown_category: u8,
    /// Category names in byte-wise order, so a smaller index is a smaller name.
    pub categories: Vec<Vec<u8>>,
    pub campaign_ids: KeyMap,
    pub direct_camp: Vec<u16>,
    pub campaigns: Vec<(Vec<u8>, Vec<u8>)>,
}

fn parse_canonical_id(key: &[u8], prefix: &[u8], max_val: usize) -> Option<usize> {
    let d = key.strip_prefix(prefix)?;
    if d.is_empty() || d[0] < b'1' || d[0] > b'9' {
        return None;
    }
    let mut n = 0usize;
    for &b in d {
        if !b.is_ascii_digit() {
            return None;
        }
        n = n.checked_mul(10)?.checked_add((b - b'0') as usize)?;
        if n > max_val {
            return None;
        }
    }
    Some(n)
}

#[inline(always)]
unsafe fn swar_dec(d_raw: u64, dlen: usize, prefix_bad: u64) -> Option<usize> {
    let x = (d_raw ^ 0x3030_3030_3030_3030) << ((8 - dlen) * 8);
    let bad = (x | x.wrapping_add(0x0606_0606_0606_0606)) & 0xF0F0_F0F0_F0F0_F0F0;
    if prefix_bad | (((d_raw as u8) == b'0') as u64) | bad != 0 {
        return None;
    }
    let y = (x.wrapping_mul(10).wrapping_add(x >> 8)) & 0x00FF_00FF_00FF_00FF;
    let z = (y.wrapping_mul(100).wrapping_add(y >> 16)) & 0x0000_FFFF_0000_FFFF;
    Some(((z.wrapping_mul(10000).wrapping_add(z >> 32)) as u32) as usize)
}

/// Lookup for `c_<N>` (canonical `c_[1-9][0-9]* <= 65535`). `p` must be readable for 8 bytes.
#[inline(always)]
pub unsafe fn lookup_camp_direct(lut: &[u16], p: *const u8, len: usize) -> Option<u32> {
    if len.wrapping_sub(3) > 4 {
        return None;
    }
    let w = (p as *const u64).read_unaligned();
    let n = swar_dec(w >> 16, len - 2, (w & 0xFFFF) ^ 0x5F63)?;
    let v = *lut.get(n)?;
    if v == u16::MAX { None } else { Some(v as u32) }
}

/// Lookup for `p<N>` (canonical `p[1-9][0-9]* <= 262144`). `p` must be readable for 8 bytes.
#[inline(always)]
pub unsafe fn lookup_prod_direct(lut: &[u8], unknown: u8, p: *const u8, len: usize) -> u8 {
    if len.wrapping_sub(2) > 5 {
        return unknown;
    }
    let w = (p as *const u64).read_unaligned();
    let Some(n) = swar_dec(w >> 8, len - 1, (w & 0xFF) ^ (b'p' as u64)) else {
        return unknown;
    };
    if n < lut.len() { *lut.get_unchecked(n) } else { unknown }
}

pub fn load(dir: &Path) -> Dims {
    let products_csv = std::fs::read(dir.join("products.csv")).unwrap_or_default();
    let rows = two_columns(&products_csv);
    let mut first_seen: HashMap<&[u8], u8> = HashMap::new();
    let mut names: Vec<&[u8]> = vec![b"unknown"];
    first_seen.insert(b"unknown", 0);
    let provisional: Vec<u8> = rows
        .iter()
        .map(|r| {
            *first_seen.entry(r.1).or_insert_with(|| {
                names.push(r.1);
                assert!(names.len() <= 256, "more than 256 categories");
                (names.len() - 1) as u8
            })
        })
        .collect();
    let mut order: Vec<u8> = (0..names.len() as u8).collect();
    order.sort_by_key(|&i| names[i as usize]);
    let mut rank = vec![0u8; names.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i as usize] = r as u8;
    }
    let product_category: Vec<u8> = provisional.iter().map(|&p| rank[p as usize]).collect();
    let unknown_category = rank[0];
    let mut direct_prod_cat = Vec::new();
    if !rows.is_empty() {
        let mut parsed = Vec::with_capacity(rows.len());
        let mut max_id = 0usize;
        let mut ok = true;
        for (i, r) in rows.iter().enumerate() {
            if let Some(id) = parse_canonical_id(r.0, b"p", 262_144) {
                max_id = max_id.max(id);
                parsed.push((id, product_category[i]));
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            let mut lut = vec![unknown_category; max_id + 1];
            let mut seen = vec![false; max_id + 1];
            for &(id, cat) in &parsed {
                if seen[id] {
                    ok = false;
                    break;
                }
                seen[id] = true;
                lut[id] = cat;
            }
            if ok {
                direct_prod_cat = lut;
            }
        }
    }
    let products = if direct_prod_cat.is_empty() {
        KeyMap::new(&rows.iter().map(|r| r.0).collect::<Vec<_>>())
    } else {
        KeyMap::empty()
    };

    let campaigns_csv = std::fs::read(dir.join("campaigns.csv")).unwrap_or_default();
    let campaigns: Vec<(Vec<u8>, Vec<u8>)> = two_columns(&campaigns_csv).iter().map(|r| (r.0.to_vec(), r.1.to_vec())).collect();
    let mut direct_camp = Vec::new();
    if !campaigns.is_empty() && campaigns.len() < u16::MAX as usize {
        let mut parsed = Vec::with_capacity(campaigns.len());
        let mut max_id = 0usize;
        let mut ok = true;
        for (i, (id, _)) in campaigns.iter().enumerate() {
            if let Some(n) = parse_canonical_id(id, b"c_", 65_535) {
                max_id = max_id.max(n);
                parsed.push((n, i as u16));
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            let mut lut = vec![u16::MAX; max_id + 1];
            for &(n, idx) in &parsed {
                if lut[n] != u16::MAX {
                    ok = false;
                    break;
                }
                lut[n] = idx;
            }
            if ok {
                direct_camp = lut;
            }
        }
    }
    let campaign_ids = if direct_camp.is_empty() {
        KeyMap::new(&campaigns.iter().map(|r| r.0.as_slice()).collect::<Vec<_>>())
    } else {
        KeyMap::empty()
    };

    Dims {
        products,
        product_category,
        direct_prod_cat,
        unknown_category,
        categories: order.iter().map(|&i| names[i as usize].to_vec()).collect(),
        campaign_ids,
        direct_camp,
        campaigns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad(s: &[u8]) -> [u8; 32] {
        let mut buf = [b'x'; 32];
        buf[..s.len()].copy_from_slice(s);
        buf
    }

    #[test]
    fn test_direct_camp_and_prod() {
        let mut camp_lut = vec![u16::MAX; 65536];
        camp_lut[1] = 0;
        camp_lut[42] = 7;
        camp_lut[65535] = 1999;

        let valid_c: &[(&[u8], Option<u32>)] = &[
            (b"c_1", Some(0)),
            (b"c_42", Some(7)),
            (b"c_65535", Some(1999)),
            (b"c_2", None),
            (b"c_0", None),
            (b"c_01", None),
            (b"c_042", None),
            (b"C_1", None),
            (b"c_ 1", None),
            (b"c_1 ", None),
            (b" c_1", None),
            (b"c_1\r", None),
            (b"c_1\x00", None),
            (b"c_", None),
            (b"", None),
            (b"c_65536", None),
            (b"c_123456", None),
            (b"c_-1", None),
            (b"c_+1", None),
            ("c_١".as_bytes(), None),
        ];
        for &(k, expected) in valid_c {
            let b = pad(k);
            let got = unsafe { lookup_camp_direct(&camp_lut, b.as_ptr(), k.len()) };
            assert_eq!(got, expected, "camp {:?}", std::str::from_utf8(k));
        }

        let unk = 99u8;
        let mut prod_lut = vec![unk; 262_145];
        prod_lut[1] = 3;
        prod_lut[50000] = 11;
        prod_lut[262144] = 17;

        let valid_p: &[(&[u8], u8)] = &[
            (b"p1", 3),
            (b"p50000", 11),
            (b"p262144", 17),
            (b"p2", unk),
            (b"p0", unk),
            (b"p01", unk),
            (b"p0050000", unk),
            (b"P1", unk),
            (b"p 1", unk),
            (b"p1 ", unk),
            (b" p1", unk),
            (b"p1\r", unk),
            (b"p1\x00", unk),
            (b"p", unk),
            (b"", unk),
            (b"p262145", unk),
            (b"p1234567", unk),
            (b"p-1", unk),
            ("p٢".as_bytes(), unk),
        ];
        for &(k, expected) in valid_p {
            let b = pad(k);
            let got = unsafe { lookup_prod_direct(&prod_lut, unk, b.as_ptr(), k.len()) };
            assert_eq!(got, expected, "prod {:?}", std::str::from_utf8(k));
        }
    }
}
