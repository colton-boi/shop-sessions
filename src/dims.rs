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
    pub unknown_category: u8,
    /// Category names in byte-wise order, so a smaller index is a smaller name.
    pub categories: Vec<Vec<u8>>,
    pub campaign_ids: KeyMap,
    pub campaigns: Vec<(Vec<u8>, Vec<u8>)>,
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
    let campaigns_csv = std::fs::read(dir.join("campaigns.csv")).unwrap_or_default();
    let campaigns: Vec<(Vec<u8>, Vec<u8>)> = two_columns(&campaigns_csv).iter().map(|r| (r.0.to_vec(), r.1.to_vec())).collect();
    Dims {
        products: KeyMap::new(&rows.iter().map(|r| r.0).collect::<Vec<_>>()),
        product_category: provisional.iter().map(|&p| rank[p as usize]).collect(),
        unknown_category: rank[0],
        categories: order.iter().map(|&i| names[i as usize].to_vec()).collect(),
        campaign_ids: KeyMap::new(&campaigns.iter().map(|r| r.0.as_slice()).collect::<Vec<_>>()),
        campaigns,
    }
}
