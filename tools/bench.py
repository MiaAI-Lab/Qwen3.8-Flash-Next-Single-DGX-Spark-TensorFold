#!/usr/bin/env python3
"""Prefill and decode benchmark against the running server.

Every prompt is fresh random prose (a new seed per run), so the prompt cache never resumes more than the chat template's
opening, which every prompt shares (44 tokens on v1.1.4), and each prompt looks up n-gram rows it has not touched before.
Reports the server's own prefill seconds, the prefill rate over the tokens it did prefill (prompt minus cached), and decode tokens/s.

Usage: tools/bench.py [label]      (API_URL, default http://127.0.0.1:$PORT with PORT 8888; DECODE_ONLY=1 skips prefill)
"""
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

API_URL = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")).rstrip("/")
URL = API_URL + "/v1/chat/completions"
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()
SIZES = [(8_000, 1), (16_000, 1), (32_000, 1), (64_000, 1), (128_000, 1)]
# The engine resumes a new conversation from the chat template's opening, which every prompt shares (44 tokens on v1.1.4).
# A prompt of random prose that reuses more than this has hit the prompt cache, and its prefill time says nothing about prefill.
MAX_SHARED_PREFIX = 256


def prose(tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    while len(out) < tokens * 0.72:          # ~1.14 tokens a word: the prompt comes out at ~0.82 x tokens
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
        out += sentence[:-1] + [sentence[-1] + "."]
    return " ".join(out)


def open_url(req: urllib.request.Request, timeout: float):
    """urlopen, with a one-line message instead of a traceback when the server is not there or refuses."""

    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        sys.exit(f"the server at {API_URL} answered {exc.code}: {exc.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach the server at {API_URL} ({exc.reason}): is it running? (./start.sh)")


def check_fresh(rs: list) -> None:
    """Exit if a prompt of `rs` (results of run) reused more than the template's opening from the prompt cache."""
    assert all(r["cached"] is not None for r in rs), "the server's reply carries no cached-token count: cannot tell whether the prompts are fresh"
    hits = [r["cached"] for r in rs if r["cached"] > MAX_SHARED_PREFIX]
    assert not hits, f"prompt cache hit ({hits[0]} tokens reused, more than the {MAX_SHARED_PREFIX} the chat template shares): prompts are not fresh"


def prefilled(r: dict) -> int:
    """The prompt tokens the server prefilled: all but the cached ones."""
    return r["prompt_tokens"] - r["cached"]


def run(prompt: str, max_tokens: int, temperature=None) -> dict:
    body = {"model": os.environ.get("MODEL", "Qwen3.8-Flash-Next"), "stream": True, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start, first, tf, usage = time.time(), None, {}, {}
    for line in open_url(req, 3600):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        chunk = json.loads(line[5:])
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        if first is None and (delta.get("content") or delta.get("reasoning_content")):
            first = time.time()
        # The stats object and `usage` can arrive in different chunks (1.0.0 sends usage last): keep each on its own.
        tf = chunk.get("tensorfold") or tf
        usage = chunk.get("usage") or usage
    return normalize(tf, usage, (first or time.time()) - start)


def normalize(tf: dict, usage: dict, ttft: float) -> dict:
    """One flat record from the server's `tensorfold` stats and `usage`: the 1.0.0 names, the 0.6.x ones as fallback."""

    def pick(*names):
        return next((tf[n] for n in names if tf.get(n) is not None), None)

    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = tf.get("cached")
    return {"prompt_tokens": usage.get("prompt_tokens", tf.get("prompt_tokens")),
            "completion_tokens": usage.get("completion_tokens", tf.get("completion_tokens")),
            "cached": cached,
            "prefill_s": pick("prefill_seconds", "prefill_s"),
            "decode_tps": pick("tokens_per_second", "decode_tps"),
            "decode_s": pick("decode_seconds", "decode_s"),
            "ttft": ttft}


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    seed = int(time.time())
    print(f"== {label}")
    for tokens, runs in ([] if os.environ.get("DECODE_ONLY") else SIZES):
        rs = [run(prose(tokens, seed + i * 7919 + tokens) + "\n\nSummarize the text above in one sentence.", 16)
              for i in range(runs)]
        missing = [k for k in ("prompt_tokens", "cached", "prefill_s") if any(r[k] is None for r in rs)]
        if missing:
            sys.exit(f"the server's reply carries no {', '.join(missing)}: this script does not know its stats layout")
        check_fresh(rs)
        n = statistics.median(r["prompt_tokens"] for r in rs)
        done = statistics.median(prefilled(r) for r in rs)
        shared = statistics.median(r["cached"] for r in rs)
        pre = statistics.median(r["prefill_s"] for r in rs)
        ttft = statistics.median(r["ttft"] for r in rs)
        each = ", ".join("%.2f" % r["prefill_s"] for r in rs)
        print(f"prefill {n:>7,.0f} tok: {pre:7.2f} s  {done / pre:6.0f} tok/s  TTFT {ttft:6.2f} s"
              f"  ({shared:,.0f} cached; {runs} run{'s' if runs > 1 else ''}, each {each} s)")
    for name, prompt, temp in (("code greedy", "Write a Python quicksort with docstring and tests.", 0),
                               ("chat sampled", "Explain why the sky is blue in a few paragraphs.", None)):
        rs = [run(prompt, 256, temp) for _ in range(5)]
        if any(not (r["decode_tps"] or (r["completion_tokens"] and r["decode_s"])) for r in rs):
            sys.exit("the server's reply carries no decode speed: this script does not know its stats layout")
        tps = [r["decode_tps"] or r["completion_tokens"] / r["decode_s"] for r in rs]   # parallel mode: no decode_tps
        print(f"decode {name:12s}: {statistics.median(tps):5.1f} tok/s (median of 5)")


if __name__ == "__main__":
    main()
