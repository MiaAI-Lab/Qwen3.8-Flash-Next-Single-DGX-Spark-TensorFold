#!/usr/bin/env python3
"""Synthetic long-prefill interference probe; stdlib only, never starts a server.

Run only against an authorized isolated server with at least --active + 1 slots.
Records content-event gaps (not token ITL), output text, usage and relative times.
No endpoint, headers, filesystem paths or wall-clock timestamps enter the report.
"""

import argparse
import json
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def payload(model, text, tokens, seed):
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "max_tokens": tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def events(response):
    """Parse SSE data fields, including multiline events, comments and CRLF."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def stream(url, body, epoch, timeout, ready=None):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    result = {"sent_s": time.perf_counter() - epoch, "events": [], "usage": {}, "finish_reason": None}
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for data in events(response):
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise ValueError("server returned an error event")
            if chunk.get("usage"):
                result["usage"] = chunk["usage"]
            if chunk.get("tensorfold") is not None:
                # Copy only the cache counter, not arbitrary server metadata.
                result["cached"] = chunk["tensorfold"].get("cached")
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    raise ValueError("thinking was requested off but reasoning content arrived")
                if delta.get("content"):
                    result["events"].append([time.perf_counter() - epoch, delta["content"]])
                    if ready is not None:
                        ready.set()
                if choice.get("finish_reason") is not None:
                    result["finish_reason"] = choice["finish_reason"]
    result["end_s"] = time.perf_counter() - epoch
    if not done or not result["events"] or not result["finish_reason"] or not result["usage"].get("completion_tokens"):
        raise ValueError("incomplete SSE response (content, usage, finish reason and DONE required)")
    result["text"] = "".join(text for _, text in result["events"])
    return result


def summarize(active, long, injection):
    first = long["events"][0][0]
    if any(r["events"][0][0] > injection or r["end_s"] <= injection for r in active):
        raise ValueError("an active stream was not streaming at injection; increase --tokens")
    cached = long.get("cached")
    if cached is None:
        cached = long["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
    if cached != 0:
        raise ValueError("long prompt cache counter is missing or nonzero; use a fresh server/seed")
    gaps = [
        b[0] - a[0] for r in active for a, b in zip(r["events"], r["events"][1:]) if b[0] >= injection and a[0] <= first
    ]
    if not gaps:
        raise ValueError("no overlapping content intervals; increase the active reply length")
    return {
        "injection_s": injection,
        "worst_overlapping_content_gap_s": max(gaps),
        "long_ttft_s": first - long["sent_s"],
        "long_wall_s": long["end_s"] - long["sent_s"],
        "active_finished_before_long_first": [r["end_s"] < first for r in active],
        "active": active,
        "long": long,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", help="server base URL, without /v1")
    p.add_argument("model")
    p.add_argument("--active", type=int, default=3)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--words", type=int, default=120000, help="synthetic words, NOT tokenizer tokens")
    p.add_argument("--seed", type=int, default=7319)
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()
    if min(args.active, args.tokens, args.words, args.timeout) <= 0:
        p.error("counts and timeout must be positive")
    rng = random.Random(args.seed)
    prompt = f"Synthetic trial {args.seed}.\n" + " ".join(
        rng.choices(
            ["river", "garden", "engine", "cloud", "paper", "stone", "window", "meadow", "copper", "violet"],
            k=args.words,
        )
    )
    prompt += "\nReply with exactly PREFILL_OK and nothing else."
    epoch = time.perf_counter()
    url = args.base.rstrip("/") + "/v1/chat/completions"
    with ThreadPoolExecutor(max_workers=args.active + 1) as pool:
        gates = [threading.Event() for _ in range(args.active)]
        futures = [
            pool.submit(
                stream,
                url,
                payload(
                    args.model,
                    f"Synthetic task {i}: write a long numbered guide to imaginary gardens.",
                    args.tokens,
                    args.seed,
                ),
                epoch,
                args.timeout,
                gate,
            )
            for i, gate in enumerate(gates)
        ]
        for gate in gates:
            if not gate.wait(args.timeout):
                raise RuntimeError("active stream did not deliver content before timeout")
        if any(f.done() for f in futures):
            raise RuntimeError("an active request already finished; increase --tokens")
        injection = time.perf_counter() - epoch
        long_body = payload(args.model, prompt, 16, args.seed)
        long_body["ignore_eos"] = False
        long = stream(url, long_body, epoch, args.timeout)
        active = [f.result() for f in futures]
    if long["text"].strip() != "PREFILL_OK":
        raise ValueError("long prompt marker mismatch")
    report = summarize(active, long, injection)
    report["parameters"] = {"active": args.active, "tokens": args.tokens, "words": args.words, "seed": args.seed}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
