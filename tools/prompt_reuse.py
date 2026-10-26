#!/usr/bin/env python3
"""Prompt reuse: a conversation takes more turns, and each turn resumes most of its prompt from the server's cache.

Usage: tools/prompt_reuse.py [size] [turns] [--other] [--min-reuse 0.9]
  size   the first turn's prompt in tokens (default 30000; '128k' accepted), turns the follow-up turns (default 4).
  --other  before each follow-up, one request of another conversation with the same system prompt (a sub-agent);
           useful with PARALLEL above 1.
  API_URL / PORT / MODEL as in client.py; thinking off, greedy.
Each turn prints prompt tokens, cached tokens (usage.prompt_tokens_details.cached_tokens, or the server's spelling),
the share resumed, the prefill seconds the server reported and the wall time.
Pass: every follow-up resumed >= min-reuse of its prompt. When the server reports no cached-token count, the timings
are printed and the result is UNCHECKED (exit 3), never a pass.
Exit 0 pass, 1 fail, 2 could not run, 3 unchecked.
"""
import argparse
import random
import sys

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
import client  # noqa: E402


def text(tokens: int, seed: int, cpt: float) -> str:
    return " ".join(client.filler_units(client.sentence, tokens, seed, cpt))


def fmt(v, spec="{:.2f} s"):
    return spec.format(v) if v is not None else "n/a"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("size", nargs="?", default="30000")
    p.add_argument("turns", nargs="?", type=int, default=4)
    p.add_argument("--other", action="store_true", help="interleave another conversation with the same system prompt")
    p.add_argument("--min-reuse", type=float, default=0.9)
    a = p.parse_args()
    if a.turns < 1:
        p.error("turns is at least 1")
    size = client.parse_size(a.size)
    cpt, _ = client.calibrate(client.sentence, 11)
    system = {"role": "system", "content": "You are a careful assistant. Read the material, then answer in one short "
                                           "sentence. " + text(1500, 1, cpt)}
    convo = [system, {"role": "user", "content": "Here is a long field log. Summarize it in one line.\n\n"
                                                 + text(size, 2, cpt)}]
    r = client.chat(convo, max_tokens=48)
    cold_prefill = r.prefill_seconds
    print(f"turn 0 (cold): prompt {r.prompt_tokens} tokens, cached {r.cached_tokens}, prefill "
          f"{fmt(cold_prefill)}, wall {r.seconds:.1f} s", flush=True)
    shares, unreported = [], False
    rng = random.Random(5)
    for turn in range(1, a.turns + 1):
        if a.other:
            o = client.chat([system, {"role": "user", "content": f"Side question {turn}: name a use of a "
                                                                 f"{rng.choice(client._NOUNS)}."}], max_tokens=32)
            print(f"  other conversation: prompt {o.prompt_tokens} tokens, cached {o.cached_tokens}")
        convo += [{"role": "assistant", "content": r.content},
                  {"role": "user", "content": f"Note {turn}: {text(150, 100 + turn, cpt)} What changed in note {turn}?"}]
        r = client.chat(convo, max_tokens=48)
        cached, prompt = r.cached_tokens, r.prompt_tokens
        if cached is None or not prompt:
            unreported = True
            share = None
        else:
            share = cached / prompt
            shares.append(share)
        print(f"turn {turn}: prompt {prompt} tokens, cached {cached}"
              + (f" ({share:.1%})" if share is not None else " (not reported)")
              + f", prefill {fmt(r.prefill_seconds)}, wall {r.seconds:.1f} s", flush=True)
    if unreported:
        print("prompt_reuse: UNCHECKED: the server reports no cached-token count; compare the prefill times above "
              f"with the cold turn's ({fmt(cold_prefill)})")
        return client.UNCHECKED
    worst = min(shares)
    if worst >= a.min_reuse:
        print(f"prompt_reuse: PASS: every follow-up resumed >= {a.min_reuse:.0%} of its prompt (worst {worst:.1%})")
        return client.PASS
    print(f"prompt_reuse: FAIL: a follow-up resumed only {worst:.1%} of its prompt (want >= {a.min_reuse:.0%})")
    return client.FAIL


if __name__ == "__main__":
    client.run(main)
