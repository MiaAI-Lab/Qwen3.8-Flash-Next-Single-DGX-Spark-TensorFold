#!/usr/bin/env python3
"""Decode (prose and code, 1..5 concurrent streams) and prefill (4k..128k) speeds against a server, using tools/bench.py for the request, the prose generator and the stats names.

usage: bench.py URL [--decode-only] [--prefill-only] [--streams 1,2,3,4,5] [--sizes 5000,10000,...] (sizes count generated words: about 0.8 prompt tokens each, so the defaults give 4k to 126k tokens) [--tokens 256]
Decode: total tok/s = sum of the server's per-stream decode speeds, greedy, median of --reps runs.
Prefill: fresh prose each run (nothing resumed from the prompt cache but the chat template's opening, a few dozen tokens every prompt shares and
the row shows as "cached"), prompt tokens the server prefilled (prompt minus cached) / its prefill seconds.
"""
import argparse
import os
import statistics
import sys
import threading
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402

PROSE = "Explain why the sky is blue in a few paragraphs."
CODE = "Write a Python quicksort with docstring and tests."


def decode(prompt, streams, tokens):
    out = [None] * streams

    def one(i):
        out[i] = B.run(prompt, tokens, 0)

    t0 = time.time()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(streams)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    per = [r["decode_tps"] or (r["completion_tokens"] / r["decode_s"] if r["decode_s"] else 0) for r in out]
    return sum(per), statistics.mean(per), statistics.median(r["ttft"] for r in out), sum(r["completion_tokens"] for r in out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--decode-only", action="store_true")
    p.add_argument("--prefill-only", action="store_true")
    p.add_argument("--streams", default="1,2,3,4,5")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--sizes", default="5000,10000,20000,40000,80000,160000")
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    B.API_URL = a.url.rstrip("/")
    B.URL = B.API_URL + "/v1/chat/completions"
    if not a.prefill_only:
        for name, prompt in (("prose", PROSE), ("code", CODE)):
            decode(prompt, 1, 16)    # warm up
            for s in map(int, a.streams.split(",")):
                rs = [decode(prompt, s, a.tokens) for _ in range(a.reps)]
                total = statistics.median(r[0] for r in rs)
                each = statistics.median(r[1] for r in rs)
                wall = statistics.median(r[2] for r in rs)*1000
                print(f"decode {name:5s} {s} stream{'s' if s > 1 else ' '}: {total:6.1f} tok/s total, {each:5.1f} per stream,"
                      f" TTFT {wall:5.0f} ms, {rs[0][3]} tok (median of {a.reps})", flush=True)
    if not a.decode_only:
        seed = int(time.time())
        for n in map(int, a.sizes.split(",")):
          rr=[]
          for k in range(a.reps):
            r = B.run(B.prose(n, seed + n + k*7919) + "\n\nSummarize the text above in one sentence.", 16)
            rr.append(r)
          B.check_fresh(rr)
          if 1:
            r=sorted(rr,key=lambda x:x["prefill_s"])[len(rr)//2]
            print(f"prefill {r['prompt_tokens']:>7,} tok: {r['prefill_s']:7.2f} s  {B.prefilled(r) / r['prefill_s']:6.0f} tok/s"
                  f"  TTFT {r['ttft']:6.2f} s  ({r['cached']} cached)", flush=True)


if __name__ == "__main__":
    main()
