#!/usr/bin/env python3
"""Image reuse: a conversation whose pictures are re-sent each turn resumes most of its prompt from the cache.

Usage: tools/image_reuse.py [pictures] [size] [--text TOKENS] [--min-reuse 0.9] [--max-tokens 48]
  pictures  pictures in the conversation (default 7); size their side in pixels (default 896, whose patch grid is
            like a phone screenshot's).
  --text    the prose before the pictures, in tokens (default 14000).
  API_URL / PORT / MODEL as in client.py; VISION=1 on the server; thinking off, greedy. The server needs the sliced
  fills (TF_FLASHNEXT_FILL=1 or TF_FLASHNEXT_FILL_FAIR=1): that is where image prompts are kept (patches/0016).

The conversation's second turn is sent three times:
  fresh    must not resume (cached 0): nothing with these pictures is stored yet
  again    must resume >= min-reuse of its prompt at its history mark, and its reply must equal the fresh one
  changed  one picture replaced by same-size, different bytes: must not resume (cached 0)
Each turn prints prompt tokens, cached tokens, the share resumed, the server's prefill seconds and the wall time.
When the server reports no cached-token count the timings are printed and the result is UNCHECKED (exit 3), never a
pass. Exit 0 pass, 1 fail, 2 could not run, 3 unchecked.
"""
import argparse
import base64
import hashlib
import random
import struct
import sys
import zlib

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
import client  # noqa: E402


def png(side: int, seed: int) -> str:
    """A side x side RGB PNG in a data URL: rows carry the seed, so same-size pictures differ in every pixel."""
    row = bytes((x * 7 + seed * 53) & 0xFF for x in range(side * 3))
    raw = b"".join(b"\x00" + row for _ in range(side))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    blob = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(blob).decode()


def part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def sha(r: "client.Reply") -> str:
    return hashlib.sha256((r.content + "\x00" + r.reasoning).encode()).hexdigest()[:12]


def fmt(v) -> str:
    return "n/a" if v is None else f"{v:.2f} s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pictures", nargs="?", type=int, default=7)
    p.add_argument("size", nargs="?", type=int, default=896)
    p.add_argument("--text", type=int, default=14000, help="tokens of prose before the pictures")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--min-reuse", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0, help="text seed; 0 draws one per run, so the prompt is fresh")
    a = p.parse_args()
    if a.pictures < 1 or a.size < 64:
        p.error("pictures is at least 1, size at least 64")
    seed = a.seed or random.randrange(1, 1 << 30)
    cpt, measured = client.calibrate(client.sentence, seed)
    prose = " ".join(client.filler_units(client.sentence, a.text, seed, cpt))
    print(f"seed {seed}: ~{a.text} prose tokens (chars/token {cpt}{'' if measured else ', fallback'}), "
          f"{a.pictures} pictures of {a.size}px", flush=True)

    urls = [png(a.size, i + 1) for i in range(a.pictures)]
    swapped = list(urls)
    swapped[0] = png(a.size, 1000)                   # same size and patch grid, different pixels
    head = lambda pics: [{"role": "user", "content": [{"type": "text", "text": f"Field log {seed}.\n{prose}"}]
                                           + [part(u) for u in pics]}]
    tail = [{"role": "assistant", "content": "OK"},
            {"role": "user", "content": "In one sentence, describe what you looked at."}]
    convo, changed = head(urls) + tail, head(swapped) + tail

    fresh = client.chat(convo, max_tokens=a.max_tokens)
    print(f"turn 2 (fresh):   prompt {fresh.prompt_tokens} tokens, cached {fresh.cached_tokens}, "
          f"prefill {fmt(fresh.prefill_seconds)}, wall {fresh.seconds:.1f} s", flush=True)
    again = client.chat(convo, max_tokens=a.max_tokens)
    share = again.cached_tokens / again.prompt_tokens \
        if again.cached_tokens is not None and again.prompt_tokens else None
    print(f"turn 2 (again):   prompt {again.prompt_tokens} tokens, cached {again.cached_tokens}"
          + (f" ({share:.1%})" if share is not None else "")
          + f", prefill {fmt(again.prefill_seconds)}, wall {again.seconds:.1f} s", flush=True)
    diff = client.chat(changed, max_tokens=a.max_tokens)
    print(f"turn 2 (changed): prompt {diff.prompt_tokens} tokens, cached {diff.cached_tokens}, "
          f"prefill {fmt(diff.prefill_seconds)}, wall {diff.seconds:.1f} s", flush=True)

    if fresh.cached_tokens is None or again.cached_tokens is None or diff.cached_tokens is None \
            or not again.prompt_tokens:
        print("image_reuse: UNCHECKED: the server reports no cached-token count; compare the prefill times above "
              f"with the fresh turn's ({fmt(fresh.prefill_seconds)})")
        return client.UNCHECKED
    bad = []
    if share < a.min_reuse:
        bad.append(f"the resumed turn cached only {share:.1%} of its prompt (want >= {a.min_reuse:.0%}); does the "
                   f"server fill prompts (TF_FLASHNEXT_FILL=1 or TF_FLASHNEXT_FILL_FAIR=1)?")
    if sha(fresh) != sha(again) or (fresh.token_sha and again.token_sha and fresh.token_sha != again.token_sha):
        bad.append(f"the resumed reply differs from the fresh one ({sha(fresh)} vs {sha(again)})")
    if diff.cached_tokens > 0.5 * (diff.prompt_tokens or 0):
        bad.append(f"one changed picture still resumed {diff.cached_tokens} of {diff.prompt_tokens} tokens")
    if bad:
        print("image_reuse: FAIL: " + "; ".join(bad))
        return client.FAIL
    print(f"image_reuse: PASS: the resumed turn cached {share:.1%} of its {again.prompt_tokens}-token prompt "
          f"({fmt(fresh.prefill_seconds)} -> {fmt(again.prefill_seconds)} prefill), replies equal ({sha(again)}), "
          f"a changed picture cached {diff.cached_tokens}")
    return client.PASS


if __name__ == "__main__":
    client.run(main)
