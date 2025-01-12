use std::ops::Index;

pub const D_VOCAB: usize = 50257;
pub const D_BYTES: usize = 320827;
pub const UNKNOWN: &str = "�";

#[derive(Debug)]
#[repr(C)]
pub struct Span {
    pub offset: u32,
    pub len: u32,
}

#[derive(Debug)]
#[repr(C)]
pub struct Decoder {
    pub spans: Vec<Span>,
    pub bytes: Vec<u8>,
}

pub trait Parse<T = Self> {
    fn parse(buf: &[u8]) -> T;
}

impl Parse for Span {
    fn parse(buf: &[u8]) -> Self {
        let offset = u32::from_le_bytes(buf[0..4].try_into().expect("Incorrect number of bytes"));
        let len = u32::from_le_bytes(buf[4..8].try_into().expect("Incorrect number of bytes"));

        Self { offset, len }
    }
}

impl Parse for Vec<Span> {
    fn parse(buf: &[u8]) -> Self {
        let mut acc = Vec::new();

        for idx in 0..D_VOCAB {
            let offset = idx * 8;
            let buffer = &buf[offset..offset + 8];
            acc.push(Span::parse(buffer));
        }

        acc
    }
}

impl Parse for Decoder {
    fn parse(buf: &[u8]) -> Self {
        Self {
            spans: <Vec<Span> as Parse>::parse(buf),
            bytes: buf[buf.len() - D_BYTES..].to_vec(),
        }
    }
}

impl Index<u16> for Decoder {
    type Output = str;

    fn index(&self, index: u16) -> &Self::Output {
        let Span { offset, len } = self.spans[index as usize];
        let (start, end) = (offset as usize, (offset + len) as usize);
        std::str::from_utf8(&self.bytes[start..end]).unwrap_or(UNKNOWN)
    }
}
