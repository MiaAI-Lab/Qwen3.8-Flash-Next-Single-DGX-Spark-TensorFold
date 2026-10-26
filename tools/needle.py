#!/usr/bin/env python3
"""Needle in a haystack: a passphrase hidden in ~size tokens of generated prose, asked for greedily with thinking off.

Usage: tools/needle.py [label] [size]
  size: the prompt's length in tokens (default 200000; '128k', '1M' = 1,000,000 accepted; up to the server's window).
  NEEDLE_DEPTH (default 0.6) where the passphrase sits, SEED (default 7) the text, API_URL / PORT / MODEL as in
  client.py. The filler is sized with the server's /tokenize when it has one (else ~4 characters a token, said so).
Prints the label, the prompt tokens the server counted, the prefill seconds it reported, and found yes/no.
Exit 0 found, 1 not found, 2 could not run.
"""
import os
import random
import sys

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
import client  # noqa: E402

QUESTION = "\n\nWhat is the secret passphrase mentioned in the text above? Reply with the passphrase only."
OVERHEAD = 64                            # the question, the needle and the chat template, in tokens (about)


def passphrase(seed: int) -> str:
    rng = random.Random(f"needle-{seed}")
    return f"{rng.choice(client._ADJS)}-{rng.choice(client._NOUNS)}-{rng.randint(1000, 9999)}"


def build(size: int, seed: int, depth: float):
    secret = passphrase(seed)
    cpt, measured = client.calibrate(client.sentence, seed)
    hay = client.filler_units(client.sentence, max(size - OVERHEAD, 16), seed, cpt)
    at = min(len(hay), max(0, int(len(hay) * depth)))
    hay.insert(at, f"Remember this: the secret passphrase is {secret}.")
    return " ".join(hay) + QUESTION, secret, cpt, measured


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "needle"
    size = client.parse_size(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    if size > 1_048_576:
        raise client.ApiError(f"size {size} is above 1,048,576 tokens, the largest window this recipe serves")
    seed = int(os.environ.get("SEED", "7"))
    depth = float(os.environ.get("NEEDLE_DEPTH", "0.6"))
    prompt, secret, cpt, measured = build(size, seed, depth)
    if not measured:
        print(f"{label}: note: sized at {cpt} characters a token (no /tokenize answer, or CHARS_PER_TOKEN set)",
              file=sys.stderr)
    r = client.chat([{"role": "user", "content": prompt}], max_tokens=64, thinking=False, temperature=0.0, seed=1234)
    found = secret.lower() in r.content.lower()
    prefill = f"{r.prefill_seconds:.2f} s" if r.prefill_seconds is not None else "not reported"
    print(f"{label}: prompt {r.prompt_tokens} tokens (asked ~{size}), needle at {depth:.0%}, prefill {prefill}, "
          f"total {r.seconds:.1f} s, answer {r.content.strip()[-80:]!r}, found {'yes' if found else 'no'}", flush=True)
    return client.PASS if found else client.FAIL


if __name__ == "__main__":
    client.run(main)
