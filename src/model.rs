use crate::decoder::ParseTensor;

#[derive(Debug)]
pub struct W {
    pub weight: Vec<f32>,
}

impl W {
    pub fn new(w: Vec<f32>) -> Self {
        Self { weight: w.to_vec() }
    }
}

impl ParseTensor for W {
    fn tf_parse(json: &str, raw: &[u8], key: Option<&str>) -> Self {
        let size = Self::tf_get_offsets_and_shape(json, key.unwrap());
        let res = Self::new(
            raw[size.0..size.1]
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|b| f32::from_le_bytes(b.try_into().expect("size of buffer slice incorrect")))
                .collect(),
        );
        assert_eq!(
            res.weight.len(),
            size.2 * std::mem::size_of::<f32>(),
            "weight expected to be in shape {} found {}",
            size.2,
            res.weight.len()
        );
        res
    }
}

#[derive(Debug)]
pub struct WB {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl WB {
    pub fn new(w: Vec<f32>, b: Vec<f32>) -> Self {
        Self { weight: w, bias: b }
    }
}

impl ParseTensor for WB {
    fn tf_parse(json: &str, raw: &[u8], key: Option<&str>) -> Self {
        let key = key.unwrap();
        let wsize = Self::tf_get_offsets_and_shape(json, &format!("{key}.weight"));
        let bsize = Self::tf_get_offsets_and_shape(json, &format!("{key}.bias"));

        let w = raw[wsize.0..wsize.1]
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|b| f32::from_le_bytes(b.try_into().expect("size of buffer slice incorrect")))
            .collect::<Vec<_>>();
        assert_eq!(
            w.len(),
            wsize.2 * std::mem::size_of::<f32>(),
            "{}.weight expected to be in shape {} found {}",
            key,
            wsize.2,
            w.len()
        );
        let b = raw[bsize.0..bsize.1]
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|b| f32::from_le_bytes(b.try_into().expect("size of buffer slice incorrect")))
            .collect::<Vec<_>>();
        assert_eq!(
            b.len(),
            bsize.2 * std::mem::size_of::<f32>(),
            "{}.bias expected to be in shape {} found {}",
            key,
            bsize.2,
            b.len()
        );

        Self::new(w, b)
    }
}

#[derive(Debug)]
pub struct Attn {
    pub bias: Vec<f32>,
    pub c_attn: WB,
    pub c_proj: WB,
}

impl Attn {
    pub fn new(b: Vec<f32>, attn: WB, proj: WB) -> Self {
        Self {
            bias: b,
            c_attn: attn,
            c_proj: proj,
        }
    }
}

impl ParseTensor for Attn {
    fn tf_parse(json: &str, raw: &[u8], key: Option<&str>) -> Self {
        let key = key.unwrap();
        let bsize = Self::tf_get_offsets_and_shape(json, &format!("{key}.bias"));
        let b = raw[bsize.0..bsize.1]
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|b| f32::from_le_bytes(b.try_into().expect("size of buffer slice incorrect")))
            .collect::<Vec<_>>();
        assert_eq!(
            b.len(),
            bsize.2 * std::mem::size_of::<f32>(),
            "{}.bias expected to be in shape {} found {}",
            key,
            bsize.2,
            b.len()
        );

        let c_attn = WB::tf_parse(json, raw, Some(&format!("{key}.c_attn")));
        let c_proj = WB::tf_parse(json, raw, Some(&format!("{key}.c_proj")));

        Self::new(b, c_attn, c_proj)
    }
}

#[derive(Debug)]
pub struct Mlp {
    pub c_fc: WB,
    pub c_proj: WB,
}

impl Mlp {
    pub fn new(c_fc: WB, c_proj: WB) -> Self {
        Self { c_fc, c_proj }
    }
}

impl ParseTensor for Mlp {
    fn tf_parse(json: &str, raw: &[u8], key: Option<&str>) -> Self {
        let key = key.unwrap();
        let c_fc = WB::tf_parse(json, raw, Some(&format!("{key}.c_fc")));
        let c_proj = WB::tf_parse(json, raw, Some(&format!("{key}.c_proj")));

        Self::new(c_fc, c_proj)
    }
}

#[derive(Debug)]
pub struct H {
    pub attn: Attn,
    pub ln_1: WB,
    pub ln_2: WB,
    pub mlp: Mlp,
}

impl H {
    pub fn new(attn: Attn, ln_1: WB, ln_2: WB, mlp: Mlp) -> Self {
        Self {
            attn,
            ln_1,
            ln_2,
            mlp,
        }
    }
}

impl ParseTensor for H {
    fn tf_parse(json: &str, raw: &[u8], key: Option<&str>) -> Self {
        let key = key.unwrap();
        let attn = Attn::tf_parse(json, raw, Some(&format!("{key}.attn")));
        let ln_1 = WB::tf_parse(json, raw, Some(&format!("{key}.ln_1")));
        let ln_2 = WB::tf_parse(json, raw, Some(&format!("{key}.ln_2")));
        let mlp = Mlp::tf_parse(json, raw, Some(&format!("{key}.mlp")));

        Self::new(attn, ln_1, ln_2, mlp)
    }
}

#[derive(Debug)]
pub struct Tensor {
    pub wpe: W,
    pub wte: W,
    pub h: Vec<H>,
    pub ln_f: WB,
}

impl Tensor {
    pub fn new(wpe: W, wte: W, h: Vec<H>, ln_f: WB) -> Self {
        Self { wpe, wte, h, ln_f }
    }
}

impl ParseTensor for Tensor {
    fn tf_parse(json: &str, raw: &[u8], _: Option<&str>) -> Self {
        let wpe = W::tf_parse(json, raw, Some("wpe"));
        let wte = W::tf_parse(json, raw, Some("wte"));
        let h = (0..12)
            .map(|i| H::tf_parse(json, raw, Some(&format!("h.{i}"))))
            .collect::<Vec<_>>();
        let ln_f = WB::tf_parse(json, raw, Some("ln_f"));
        Self::new(wpe, wte, h, ln_f)
    }
}
