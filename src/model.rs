#[derive(Debug)]
pub struct W {
    pub weight: Vec<f32>,
}

impl W {
    pub fn new(w: Vec<f32>) -> Self {
        Self { weight: w.to_vec() }
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

#[derive(Debug)]
pub struct Tensors {
    pub wpe: W,
    pub wte: W,
    pub h: Vec<H>,
    pub ln_f: WB,
}

impl Tensors {
    pub fn new(wpe: W, wte: W, h: Vec<H>, ln_f: WB) -> Self {
        Self { wpe, wte, h, ln_f }
    }
}
