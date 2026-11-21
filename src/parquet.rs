use std::fs::File;
use std::io::Write;
use std::path::Path;

pub const INT32: u8 = 1;
pub const INT64: u8 = 2;
pub const BYTES: u8 = 6;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Plain,
    Utf8,
    Date,
}

pub struct Col {
    pub name: &'static str,
    pub ty: u8,
    pub kind: Kind,
}

/// Thrift compact protocol, the little of it a Parquet footer needs.
struct Thrift {
    b: Vec<u8>,
    last: i16,
    stack: Vec<i16>,
}

const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BINARY: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;

impl Thrift {
    fn new() -> Self {
        Thrift { b: Vec::new(), last: 0, stack: Vec::new() }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.b.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.b.push(v as u8);
    }
    fn zigzag(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn field(&mut self, id: i16, ty: u8) {
        let delta = id - self.last;
        if (1..=15).contains(&delta) {
            self.b.push((delta as u8) << 4 | ty);
        } else {
            self.b.push(ty);
            self.zigzag(id as i64);
        }
        self.last = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, T_I32);
        self.zigzag(v as i64);
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, T_I64);
        self.zigzag(v);
    }
    fn binary(&mut self, id: i16, v: &[u8]) {
        self.field(id, T_BINARY);
        self.varint(v.len() as u64);
        self.b.extend_from_slice(v);
    }
    fn list(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, T_LIST);
        if n < 15 {
            self.b.push((n as u8) << 4 | elem);
        } else {
            self.b.push(0xF0 | elem);
            self.varint(n as u64);
        }
    }
    fn begin(&mut self) {
        self.stack.push(self.last);
        self.last = 0;
    }
    fn begin_field(&mut self, id: i16) {
        self.field(id, T_STRUCT);
        self.begin();
    }
    fn end(&mut self) {
        self.b.push(0);
        self.last = self.stack.pop().unwrap();
    }
}

struct Chunk {
    offset: u64,
    size: u64,
}

struct Group {
    rows: u64,
    chunks: Vec<Chunk>,
}

/// One Parquet file: uncompressed, PLAIN, one page per column per row group,
/// every column optional.
pub struct Writer {
    file: File,
    offset: u64,
    cols: &'static [Col],
    groups: Vec<Group>,
    rows: u64,
    head: Vec<u8>,
}

impl Writer {
    pub fn create(path: &Path, cols: &'static [Col]) -> Self {
        let mut file = File::create(path).expect("create output file");
        file.write_all(b"PAR1").unwrap();
        Writer { file, offset: 4, cols, groups: Vec::new(), rows: 0, head: Vec::new() }
    }

    /// `data[i]` is column i's PLAIN-encoded non-null values, with a validity
    /// bitmap (bit set = present) when the column has nulls.
    pub fn row_group(&mut self, rows: usize, data: &[(&[u8], Option<&[u8]>)]) {
        let mut chunks = Vec::with_capacity(data.len());
        for (values, validity) in data {
            let mut levels = Thrift::new();
            match validity {
                None => {
                    levels.varint((rows as u64) << 1);
                    levels.b.push(1);
                }
                Some(bits) => {
                    let groups = rows.div_ceil(8);
                    levels.varint((groups as u64) << 1 | 1);
                    levels.b.extend_from_slice(&bits[..groups]);
                }
            }
            let page = 4 + levels.b.len() + values.len();
            let mut t = Thrift::new();
            t.i32(1, 0);
            t.i32(2, page as i32);
            t.i32(3, page as i32);
            t.begin_field(5);
            t.i32(1, rows as i32);
            t.i32(2, 0);
            t.i32(3, 3);
            t.i32(4, 3);
            t.end();
            t.b.push(0);
            self.head.clear();
            self.head.extend_from_slice(&t.b);
            self.head.extend_from_slice(&(levels.b.len() as u32).to_le_bytes());
            self.head.extend_from_slice(&levels.b);
            self.file.write_all(&self.head).unwrap();
            self.file.write_all(values).unwrap();
            let size = (self.head.len() + values.len()) as u64;
            chunks.push(Chunk { offset: self.offset, size });
            self.offset += size;
        }
        self.rows += rows as u64;
        self.groups.push(Group { rows: rows as u64, chunks });
    }

    pub fn finish(mut self) {
        let mut t = Thrift::new();
        t.i32(1, 1);
        t.list(2, T_STRUCT, self.cols.len() + 1);
        t.begin();
        t.binary(4, b"schema");
        t.i32(5, self.cols.len() as i32);
        t.end();
        for c in self.cols {
            t.begin();
            t.i32(1, c.ty as i32);
            t.i32(3, 1);
            t.binary(4, c.name.as_bytes());
            match c.kind {
                Kind::Plain => {}
                Kind::Utf8 => {
                    t.i32(6, 0);
                    t.begin_field(10);
                    t.begin_field(1);
                    t.end();
                    t.end();
                }
                Kind::Date => {
                    t.i32(6, 6);
                    t.begin_field(10);
                    t.begin_field(6);
                    t.end();
                    t.end();
                }
            }
            t.end();
        }
        t.i64(3, self.rows as i64);
        t.list(4, T_STRUCT, self.groups.len());
        for g in &self.groups {
            t.begin();
            t.list(1, T_STRUCT, g.chunks.len());
            let mut total = 0;
            for (c, chunk) in self.cols.iter().zip(&g.chunks) {
                total += chunk.size;
                t.begin();
                t.i64(2, chunk.offset as i64);
                t.begin_field(3);
                t.i32(1, c.ty as i32);
                t.list(2, T_I32, 2);
                t.zigzag(0);
                t.zigzag(3);
                t.list(3, T_BINARY, 1);
                t.varint(c.name.len() as u64);
                t.b.extend_from_slice(c.name.as_bytes());
                t.i32(4, 0);
                t.i64(5, g.rows as i64);
                t.i64(6, chunk.size as i64);
                t.i64(7, chunk.size as i64);
                t.i64(9, chunk.offset as i64);
                t.end();
                t.end();
            }
            t.i64(2, total as i64);
            t.i64(3, g.rows as i64);
            t.end();
        }
        t.b.push(0);
        let len = t.b.len() as u32;
        t.b.extend_from_slice(&len.to_le_bytes());
        t.b.extend_from_slice(b"PAR1");
        self.file.write_all(&t.b).unwrap();
    }
}
