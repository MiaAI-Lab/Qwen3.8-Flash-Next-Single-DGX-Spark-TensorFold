#!/usr/bin/env python3
"""Image input (VISION=1): draws a red circle and a blue square into a PNG, sends it as a data URL and
checks the model names both.

Usage: tools/visioncheck.py      (API_URL / PORT as in bench.py). Exit code 1 on failure.
"""
import base64
import json
import struct
import sys
import urllib.request
import zlib

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing bench
from bench import URL, open_url  # noqa: E402

W, H = 480, 240


def png() -> bytes:
    """A white 480 x 240 RGB image: a red circle on the left, a blue square on the right (stdlib only)."""

    rows = []
    for y in range(H):
        row = bytearray([0])                          # filter: none
        for x in range(W):
            if (x - 120) ** 2 + (y - 120) ** 2 <= 80 ** 2:
                row += bytes((220, 30, 30))
            elif 280 <= x < 440 and 40 <= y < 200:
                row += bytes((30, 60, 220))
            else:
                row += bytes((255, 255, 255))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b""))


def main() -> None:
    url = "data:image/png;base64," + base64.b64encode(png()).decode()
    body = {"model": "Qwen3.8-Flash-Next", "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Which shapes are in this image, and what colour is each? One sentence."}]}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(open_url(req, 600))          # a server without VISION=1 answers 400 (image input off)
    text = (out["choices"][0]["message"].get("content") or "").strip()
    print(f"reply ({out.get('usage', {}).get('prompt_tokens')} prompt tokens): {text}")
    low = text.lower()
    ok = all(word in low for word in ("red", "circle", "blue", "square"))
    print("visioncheck:", "OK: the model sees the red circle and the blue square" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
