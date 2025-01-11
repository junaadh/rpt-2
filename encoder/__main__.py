# pyright: basic
import sys
import os
import tiktoken
import struct


def main():
    if len(sys.argv) != 2:
        print("Usage: encoder <filename> | <arg>")
        exit(0)

    arg = sys.argv[1]

    contents = arg

    if os.path.exists(arg) and not os.path.isdir(arg):
        with open(arg, "r") as file:
            contents = file.read()

    enc = tiktoken.encoding_for_model("gpt2")
    enc.encode(contents)

    spans = []
    buf = b""

    for idx in range(len(enc.token_byte_values()) + 1):
        word = enc.decode([idx])

        spans.append(len(buf))
        spans.append(len(word))

        buf += bytes(word, "utf-8")

    out_file = open(".zig-cache/enc", "wb")

    for sp in spans:
        value = struct.pack("I", sp)
        out_file.write(value)

    out_file.write(buf)


if __name__ == "__main__":
    main()
