use std::{fs, io::Read, os::unix::fs::FileExt};

use rpt2::{
    data::Data,
    decoder::{Decoder, Parse, ParseTensor, UNKNOWN},
    model::Tensor,
};

fn main() {
    let mut buffer = Vec::new();
    fs::File::open("target/enc")
        .expect("Failed to open file")
        .read_to_end(&mut buffer)
        .expect("Failed to read file");

    let _decoder = Decoder::parse(&buffer);

    buffer.clear();
    fs::File::open("target/tokens")
        .expect("Failed to open tokens")
        .read_to_end(&mut buffer)
        .expect("Failed to read tokens");

    let _data = Data::parse(&buffer);

    buffer.clear();
    let mut model = fs::File::open("assets/model.safetensors").expect("Failed to open model");
    let mut json_size = [0; 8];

    model
        .read_exact_at(&mut json_size, 0)
        .expect("Error reading header");

    let json_size = u64::from_le_bytes(json_size);

    let mut json = vec![0; json_size as usize];
    model
        .read_exact_at(&mut json, 8)
        .expect("Failed to read json");

    let json = std::str::from_utf8(&json).unwrap_or(UNKNOWN);

    model
        .read_to_end(&mut buffer)
        .expect("failed to read raw data");

    let buffer = &buffer[8 + json_size as usize..];

    let tensor = Tensor::tf_parse(json, buffer, None);

    println!("{}", tensor);
}
