#!/usr/bin/env python3
"""Paired Dutch/English/code benchmark using the existing SSE/statistics client.

Run once per server restart with --arm A|B|serial and the same --block.
Fresh per-block/client/request markers avoid exact prompt cache reuse, while
paired arms send identical requests. Warmups use separate markers and are discarded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from bench import complete, concurrent, _write_jsonl, _summarize


def jobs(requests, block, clients, repeat, serial=False, warmup=False):
    result = []
    for index, request in enumerate(requests):
        prompt = (f"[Meetcode {block}-{clients}-{repeat}-{index}-{'warmup' if warmup else 'meting'}; "
                  "negeer deze code in je antwoord.]\n" + request["prompt"])
        result.append({"name": request["id"], "prompt": prompt,
                       "max_tokens": request.get("max_tokens", 1024), "temperature": 0,
                       "seed": 42, "thinking": False, "draft": not serial, "return_token_ids": True})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("A", "B", "serial"), required=True)
    ap.add_argument("--block", required=True, help="same identifier for paired restarts, e.g. 01")
    ap.add_argument("--manifest", type=Path, default=Path("tools/nl_benchmark.json"))
    ap.add_argument("--clients", default="1,2,4")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--jsonl", type=Path, required=True)
    args = ap.parse_args()
    clients = [int(c) for c in args.clients.split(",")]
    if not clients or min(clients) < 1 or args.repeat < 1:
        ap.error("clients and repeat must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requests = manifest["requests"]
    if not requests or len({r["id"] for r in requests}) != len(requests):
        ap.error("manifest needs unique IDs and at least one request")
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    groups = {r["id"]: r["group"] for r in requests}
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl.exists():
        ap.error("output already exists; use a fresh file for each restart")
    for c in clients:
        warmup = jobs(requests[:c], args.block, c, 0, args.arm == "serial", warmup=True)
        for r in concurrent(warmup):
            if not r["ok"]:
                raise RuntimeError(f"warmup failed: {r.get('error')}")
        for repeat in range(args.repeat):
            batch = jobs(requests, args.block, c, repeat, args.arm == "serial")
            # Send every held-out prompt at each concurrency, not just the first c.
            for start in range(0, len(batch), c):
                subset = batch[start:start+c]
                t0 = time.perf_counter()
                rows = concurrent(subset)
                t1 = time.perf_counter()
                _write_jsonl(str(args.jsonl), [{**r, "arm": args.arm, "block": args.block,
                             "clients": c, "actual_clients": len(subset), "repeat": repeat,
                             "group": groups[r["name"]], "manifest_sha256": digest,
                             "request_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest(),
                             "batch_wall_s": t1-t0} for r in rows])
                _summarize(f"{args.arm} C={c} repeat={repeat}", rows, t0, t1)
                if any(not r["ok"] for r in rows):
                    raise RuntimeError("request failed; raw record retained, block is invalid")


if __name__ == "__main__":
    main()
