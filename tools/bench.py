#!/usr/bin/env python3
"""Prefill and decode benchmark against the running server.

Smoke (default): sequential fresh synthetic prompts, then a short decode check. Extended driver: fixed
manifest/seeds, barrier-started concurrent streams, JSONL records, thinking vs visible-answer timing.

Usage:
  tools/bench.py [label]                 smoke (DECODE_ONLY=1 skips prefill)
  tools/bench.py --suite --seed 1 --jsonl run.jsonl --clients 1,2,4,5
  tools/bench.py --manifest prompts.json --jsonl run.jsonl --clients 5

API_URL, default http://127.0.0.1:$PORT with PORT 8888.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import threading
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
SIZES = [(1_000, 3), (4_000, 3), (16_000, 2), (64_000, 1)]      # (approx prompt tokens, runs)
NATURAL = (
    "Explain how a lock-free ring buffer can pass messages between one writer and one reader without a mutex.",
    "A librarian has to shelve a cart of books that are already in call-number order. Describe a practical way "
    "to merge them into the stacks and what goes wrong if two carts arrive at once.",
    "Write a few paragraphs on why weather forecasts get worse further into the future, using the idea of "
    "sensitive dependence rather than a list of model names.",
    "You are editing a short memo that already contains this sentence: The spare key is taped under the third "
    "drawer. Quote that sentence, then rewrite the memo so a contractor can find the key without reading the rest.",
)


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


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def complete(prompt: str, max_tokens: int, *, temperature=None, seed=None, thinking=None,
             draft=None, return_token_ids=False, timeout=3600, name="") -> dict:
    """One streamed request. Empty output, HTTP and SSE errors are failures, not a substituted TTFT."""

    body = {"model": "Qwen3.8-Flash-Next", "stream": True, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = int(seed)
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    if draft is not None:
        body["draft"] = bool(draft)
    if return_token_ids:
        body["return_token_ids"] = True
    rec = {"name": name, "ok": False, "error": None, "prompt": prompt, "max_tokens": max_tokens,
           "temperature": temperature, "seed": seed, "thinking": thinking, "draft": draft,
           "ttft": None, "answer_ttft": None, "elapsed": None, "content": "", "reasoning_content": "",
           "finish_reason": None, "emissions": [], "token_ids": None, "output_sha": None,
           "cached": None, "prompt_tokens": None, "completion_tokens": None}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        fp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        rec["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"
        rec["elapsed"] = time.perf_counter() - start
        return rec
    except urllib.error.URLError as exc:
        rec["error"] = f"URL error: {exc.reason}"
        rec["elapsed"] = time.perf_counter() - start
        return rec
    stats, usage, first, first_answer = {}, {}, None, None
    content, reasoning = [], []
    try:
        for raw in fp:
            now = time.perf_counter()
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                chunk = json.loads(line[5:])
            except json.JSONDecodeError:
                rec["error"] = f"bad SSE JSON: {line[:200]}"
                rec["elapsed"] = now - start
                return rec
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            rec["finish_reason"] = rec["finish_reason"] or choice.get("finish_reason")
            rdelta = delta.get("reasoning_content") or ""
            cdelta = delta.get("content") or ""
            if rdelta or cdelta:
                rec["emissions"].append({"t": round(now - start, 6), "reasoning": len(rdelta), "content": len(cdelta)})
            if rdelta:
                reasoning.append(rdelta)
                if first is None:
                    first = now
            if cdelta:
                content.append(cdelta)
                if first is None:
                    first = now
                if first_answer is None:
                    first_answer = now
            if "tensorfold" in chunk:
                stats = chunk["tensorfold"]
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("token_ids") is not None:
                rec["token_ids"] = chunk["token_ids"]
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["elapsed"] = time.perf_counter() - start
        return rec
    rec["elapsed"] = time.perf_counter() - start
    rec["content"] = "".join(content)
    rec["reasoning_content"] = "".join(reasoning)
    rec["output_sha"] = _sha(rec["reasoning_content"] + "\n" + rec["content"])
    rec.update(usage)
    rec.update(stats)
    rec["cached"] = stats.get("cached", usage.get("prompt_tokens_cached"))
    rec["prompt_tokens"] = usage.get("prompt_tokens", stats.get("prompt_tokens"))
    rec["completion_tokens"] = usage.get("completion_tokens", stats.get("completion_tokens"))
    rec["token_sha"] = stats.get("token_sha")
    rec["ttft"] = None if first is None else first - start
    rec["answer_ttft"] = None if first_answer is None else first_answer - start
    if first is None:
        rec["error"] = rec["error"] or "empty output"
        return rec
    rec["ok"] = True
    return rec


def run(prompt: str, max_tokens: int, temperature=None) -> dict:
    """Smoke-test helper: streamed stats plus TTFT. Empty/error responses raise SystemExit."""

    rec = complete(prompt, max_tokens, temperature=temperature)
    if not rec["ok"]:
        sys.exit(f"the server at {API_URL} failed: {rec.get('error')}")
    stats = {k: rec[k] for k in rec if k not in
             ("prompt", "content", "reasoning_content", "emissions", "token_ids", "name", "ok", "error")}
    return stats


def concurrent(reqs: list[dict]) -> list[dict]:
    """Barrier-start every request, including queued load above serving capacity."""

    barrier, out = threading.Barrier(len(reqs)), [None] * len(reqs)

    def worker(i, kwargs):
        try:
            barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            out[i] = {**kwargs, "ok": False, "error": "start barrier broken"}
            return
        out[i] = complete(**kwargs)

    threads = [threading.Thread(target=worker, args=(i, kw), daemon=True) for i, kw in enumerate(reqs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def _write_jsonl(path: str, rows: list[dict]) -> None:
    slim = []
    for r in rows:
        item = {k: v for k, v in r.items() if k != "prompt" or len(str(v)) < 4000}
        if len(r.get("prompt") or "") >= 4000:
            item["prompt_sha"] = _sha(r["prompt"])
            item["prompt_chars"] = len(r["prompt"])
        slim.append(item)
    with open(path, "a", encoding="utf-8") as f:
        for item in slim:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _summarize(label: str, rows: list[dict], wall0: float | None = None, wall1: float | None = None) -> None:
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    if bad:
        print(f"{label}: {len(bad)}/{len(rows)} failed: " + "; ".join(str(r.get("error")) for r in bad[:3]))
    if not ok:
        return
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    answers = [r["answer_ttft"] for r in ok if r.get("answer_ttft") is not None]
    elap = [r["elapsed"] for r in ok if r.get("elapsed")]
    comps = [int(r["completion_tokens"] or 0) for r in ok]
    prompts = [int(r["prompt_tokens"] or 0) for r in ok]
    cached = [int(r.get("cached") or 0) for r in ok]
    drafted = [int(r.get("drafted") or 0) for r in ok]
    accepted = [int(r.get("accepted") or 0) for r in ok]
    decode_s = [float(r["decode_s"]) for r in ok if r.get("decode_s")]
    gaps = []
    for r in ok:
        ts = [e["t"] for e in r.get("emissions") or []]
        gaps.extend(b - a for a, b in zip(ts, ts[1:]))
    gen = sum(comps)
    if wall0 is not None and wall1 is not None and wall1 > wall0 and gen:
        agg = gen / (wall1 - wall0)
    else:
        agg = None
    per = []
    for r, ntok in zip(ok, comps):
        d = r.get("decode_s")
        if d and float(d) > 0 and ntok:
            per.append(ntok / float(d))
    acc = (sum(accepted) / sum(drafted)) if sum(drafted) else None
    yield_r = None
    rounds = sum(int(r.get("rounds") or 0) for r in ok)
    if rounds:
        yield_r = 1 + (sum(accepted) / rounds)
    bits = [f"{label}: {len(ok)} ok"]
    if prompts:
        bits.append(f"prompt {statistics.median(prompts):.0f} tok cached {statistics.median(cached):.0f}")
    if ttfts:
        bits.append(f"TTFT {_pct(ttfts, 50)*1e3:.0f}/{_pct(ttfts, 95)*1e3:.0f} ms p50/p95")
    if answers:
        bits.append(f"answer {_pct(answers, 50)*1e3:.0f} ms")
    if agg:
        bits.append(f"aggregate {agg:.1f} tok/s")
    if per:
        bits.append(f"per-request {statistics.median(per):.1f} tok/s")
    if decode_s and comps:
        e2e = [c / e for c, e in zip(comps, elap) if e]
        if e2e:
            bits.append(f"e2e {statistics.median(e2e):.1f} tok/s")
    if acc is not None:
        bits.append(f"accept {acc:.3f}")
    if yield_r is not None:
        bits.append(f"yield {yield_r:.3f}")
    if gaps:
        bits.append(f"gap p50/p95 {_pct(gaps, 50)*1e3:.0f}/{_pct(gaps, 95)*1e3:.0f} ms")
    print("  ".join(bits))


def _load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data["requests"]


def _default_suite(seed: int, max_tokens: int, thinking: bool | None) -> list[dict]:
    reqs = []
    for i, text in enumerate(NATURAL[:-1]):
        reqs.append({"name": f"natural-{i}", "prompt": text, "max_tokens": max_tokens,
                     "temperature": None, "seed": seed + i, "thinking": thinking})
    reqs.append({"name": "copy-quote", "prompt": NATURAL[-1], "max_tokens": max_tokens,
                 "temperature": None, "seed": seed + 50, "thinking": thinking})
    reqs.append({"name": "synthetic-short",
                 "prompt": prose(400, seed + 99) + "\n\nSummarize the text above in two paragraphs.",
                 "max_tokens": max_tokens, "temperature": None, "seed": seed + 99, "thinking": thinking})
    return reqs


def _parse_clients(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def smoke(label: str, seed: int | None) -> None:
    seed = int(time.time()) if seed is None else seed
    print(f"== {label} seed={seed}")
    for tokens, runs in ([] if os.environ.get("DECODE_ONLY") else SIZES):
        rs = [run(prose(tokens, seed + i * 7919 + tokens) + "\n\nSummarize the text above in one sentence.", 16)
              for i in range(runs)]
        cached = [int(r.get("cached") or 0) for r in rs]
        if any(c != 0 for c in cached):
            print(f"note: engine prefix cached={cached} (not a page-cache or n-gram guarantee)")
        n = statistics.median(r["prompt_tokens"] for r in rs)
        pre = statistics.median(r["prefill_s"] for r in rs)
        ttft = statistics.median(r["ttft"] for r in rs)
        each = ", ".join("%.2f" % r["prefill_s"] for r in rs)
        print(f"prefill {n:>7,.0f} tok: {pre:7.2f} s  {n / pre:6.0f} tok/s  TTFT {ttft:6.2f} s"
              f"  ({runs} run{'s' if runs > 1 else ''}, each {each} s)")
    for name, prompt, temp in (("code greedy", "Write a Python quicksort with docstring and tests.", 0),
                               ("chat sampled", "Explain why the sky is blue in a few paragraphs.", None)):
        rs = [run(prompt, 256, temp) for _ in range(5)]
        tps = [r.get("decode_tps") or r["completion_tokens"] / r["decode_s"] for r in rs]
        print(f"decode {name:12s}: {statistics.median(tps):5.1f} tok/s (median of 5)")


def suite(args) -> None:
    thinking = None if args.thinking == "default" else args.thinking == "on"
    if args.manifest:
        reqs = _load_manifest(args.manifest)
        jobs = []
        for i, r in enumerate(reqs):
            jobs.append({"name": r.get("id") or r.get("name") or f"m{i}", "prompt": r["prompt"],
                         "max_tokens": int(r.get("max_tokens") or args.max_tokens),
                         "temperature": r.get("temperature"), "seed": r.get("seed", args.seed),
                         "thinking": r.get("thinking", thinking), "draft": r.get("draft"),
                         "return_token_ids": args.return_token_ids})
    else:
        jobs = [{**r, "return_token_ids": args.return_token_ids} for r in
                _default_suite(args.seed or 1, args.max_tokens, thinking)]
        if args.greedy:
            for j in jobs:
                j["temperature"] = 0
    if args.write_manifest:
        with open(args.write_manifest, "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "requests": [
                {k: v for k, v in j.items() if k != "return_token_ids"} for j in jobs]}, f, indent=2)
            f.write("\n")
        print(f"wrote {args.write_manifest} ({len(jobs)} requests)")
        if not (args.suite or args.manifest or args.jsonl or args.clients):
            return
    clients = _parse_clients(args.clients) if args.clients else [1]
    print(f"== {args.label} seed={args.seed} thinking={args.thinking} clients={clients}")
    pool = list(jobs)
    if args.repeat > 1:
        pool = [b for b in pool for _ in range(args.repeat)]
    for c in clients:
        if c <= 1 and not args.queued:
            wall0 = time.perf_counter()
            rows = [complete(**j) for j in pool]
            wall1 = time.perf_counter()
        else:
            need = c + args.queued
            batch = (pool * ((need + len(pool) - 1) // len(pool)))[:need]
            wall0 = time.perf_counter()
            rows = concurrent(batch)
            wall1 = time.perf_counter()
        if args.jsonl:
            _write_jsonl(args.jsonl, [{**r, "clients": c, "label": args.label} for r in rows])
        _summarize(f"C={c}", rows, wall0, wall1)
        shas = [r.get("output_sha") for r in rows if r.get("ok")]
        if shas:
            print(f"  output_sha {len(set(shas))} distinct / {len(shas)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label", nargs="?", default="run")
    p.add_argument("--seed", type=int, default=None, help="fixed request/prompt seed (smoke: wall time if omitted)")
    p.add_argument("--suite", action="store_true", help="held-out natural + synthetic decode suite")
    p.add_argument("--manifest", help="JSON list or {requests:[...]} of prompts")
    p.add_argument("--write-manifest", help="write the requests this run would send")
    p.add_argument("--jsonl", help="append one JSON object per request")
    p.add_argument("--clients", default=None, help="concurrency arms, e.g. 1,2,4,5")
    p.add_argument("--queued", type=int, default=0, help="extra requests started with the barrier (above capacity)")
    p.add_argument("--max-tokens", type=int, default=512, help="decode length for --suite / --manifest")
    p.add_argument("--thinking", choices=("on", "off", "default"), default="default")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--return-token-ids", action="store_true")
    p.add_argument("--repeat", type=int, default=1, help="repeat each suite prompt this many times")
    args = p.parse_args()
    if args.suite or args.manifest or args.jsonl or args.clients or args.write_manifest:
        if args.seed is None:
            args.seed = 1
        suite(args)
        return
    smoke(args.label, args.seed)


if __name__ == "__main__":
    main()
