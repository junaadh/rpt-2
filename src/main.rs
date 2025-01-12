use std::{fs, io::Read, os::unix::fs::FileExt};

use rpt2::{
    data::Data,
    decoder::{Decoder, Parse, UNKNOWN},
    model::W,
};

fn main() {
    let mut buffer = Vec::new();
    fs::File::open("target/enc")
        .expect("Failed to open file")
        .read_to_end(&mut buffer)
        .expect("Failed to read file");

    let decoder = Decoder::parse(&buffer);
    // for span in decoder.spans.iter() {
    //     println!(
    //         "offset: {}, size: {}, char: {} - {:?}",
    //         span.offset,
    //         span.len,
    //         // String::from_utf8_lossy(
    //         std::str::from_utf8(
    //             &decoder.bytes[span.offset as usize..(span.offset + span.len) as usize]
    //         )
    //         .unwrap_or("�"),
    //         &decoder.bytes[span.offset as usize..(span.offset + span.len) as usize]
    //     );
    // }

    buffer.clear();
    fs::File::open("target/tokens")
        .expect("Failed to open tokens")
        .read_to_end(&mut buffer)
        .expect("Failed to read tokens");

    let data = Data::parse(&buffer);

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
    // println!("{json}");

    model
        .read_to_end(&mut buffer)
        .expect("failed to read raw data");

    let buffer = &buffer[8 + json_size as usize..];

    let (start, end, shape) = tf_get_offsets_and_shape(json, "wpe.weight");

    let wpe = W::new(
        buffer[start..end]
            .chunks_exact(8)
            .map(|x| {
                let bytes = x.try_into().expect("expected a 4 byte byte array");
                f32::from_le_bytes(bytes)
            })
            .collect(),
    );
    assert_eq!(wpe.weight.len(), shape);

    let (start, end, shape) = tf_get_offsets_and_shape(json, "wte.weight");
    let wte = W::new(
        buffer[start..end]
            .chunks_exact(8)
            .map(|x| {
                let bytes = x.try_into().expect("expected a 4 byte byte array");
                f32::from_le_bytes(bytes)
            })
            .collect(),
    );
    assert_eq!(wte.weight.len(), shape);

    // println!("{},{}, {}", start, end, end - start);

    // assert_eq!(end - start, shape * std::mem::size_of::<f32>());
    // let mut float_buf = Vec::with_capacity((end - start) / 4);

    // for float in buffer[start..end].chunks_exact(4) {
    //     // println!("{float:?}");

    //     let float = float.try_into().unwrap();
    //     float_buf.push(f32::from_le_bytes(float));
    // }

    // println!("{:?}", float_buf);

    // for token in data.inner {
    //     print!("{}", &decoder[token]);
    // }
}

fn tf_get_offsets_and_shape(json: &str, key: &str) -> (usize, usize, usize) {
    let label = "\"data_offsets\":";

    let start_ = json
        .find(key)
        .unwrap_or_else(|| panic!("key: {key} not found in json"));
    println!("{}", &json[start_..start_ + 100]);
    let start = json[start_..]
        .find(label)
        .unwrap_or_else(|| panic!("label: {label} not found in json {key}"))
        + start_
        + label.len()
        + 1;

    let end = json[start..]
        .find("]")
        .unwrap_or_else(|| panic!("found array end"))
        + start;

    // println!("{}", &json[start..end]);

    let value = json[start..end]
        .split_once(',')
        .map(|(s, e)| {
            let s1 = s.parse::<usize>().expect("failed to parse start to usize");
            let e1 = e
                .parse::<usize>()
                .expect("Failed to parse end offset to usize");
            (s1, e1)
        })
        .expect("Expected offset array to be seperated by comma");

    let start = json[start_..]
        .find("\"shape\":[")
        .unwrap_or_else(|| panic!("shape not found in json"))
        + "\"shape\":[".len()
        + start_;
    let end = json[start..]
        .find("]")
        .unwrap_or_else(|| panic!("closing delimeter not found"))
        + start;

    let shape = json[start..end]
        .split(',')
        .map(|x| x.parse::<usize>().expect("failed to parse shape to usize"))
        .fold(1, |acc, x| {
            println!("{acc}*{x}={}", acc * x);
            acc * x
        });

    println!("value: {shape}");

    (value.0, value.1, shape)
}
