# pyright: basic
import sys
import os
import tiktoken
import struct


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run encoder <file|text>\n")
        exit(1)

    contents = sys.argv[1]

    if os.path.exists(contents) and not os.path.isdir(contents):
        file = open(contents, "r")
        contents = file.read()

    enc = tiktoken.encoding_for_model("gpt2")
    tokens = enc.encode(contents)

    spans = []
    buf = b""

    for idx in range(50257):
        word = enc.decode_single_token_bytes(idx)  # ([idx])

        spans.append(len(buf))
        spans.append(len(word))

        buf += word  # bytes(word, "utf-8")

    out_file = open("target/enc", "wb")

    for sp in spans:
        value = struct.pack("I", sp)
        out_file.write(value)

    out_file.write(buf)

    with open("target/tokens", "wb") as tok_buf:
        for token in tokens:
            value = struct.pack("H", token)
            tok_buf.write(value)


if __name__ == "__main__":
    main()
