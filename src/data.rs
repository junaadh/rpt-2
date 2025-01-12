use crate::decoder::Parse;

#[derive(Debug)]
pub struct Data {
    pub inner: Vec<u16>,
}

impl Data {}

impl Parse for Data {
    fn parse(buf: &[u8]) -> Self {
        assert!(
            buf.len() % 2 == 0,
            "corrupted tokens file, tokens file len should be divisible by 2"
        );
        let mut inner = Vec::new();

        for i in 0..(buf.len() / 2) {
            let offset = i * 2;
            inner.push(u16::from_le_bytes(
                buf[offset..offset + 2]
                    .try_into()
                    .expect("Corrupted tokens file"),
            ));
        }

        Self { inner }
    }
}
