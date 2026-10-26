#!/usr/bin/env python3
"""tools/bench.py reads the stats of a 1.0.0 stream (issue #38), against a local fake server.

1.0.0 sends the `tensorfold` stats object in one chunk and `usage` in a later one, with the 1.0.0 field names
(`prefill_seconds`, `tokens_per_second`, `usage.prompt_tokens_details.cached_tokens`). The 0.6.x layout (one chunk,
`prefill_s`, `cached`) must keep working. Also checks that a prompt-cache hit is still caught.

Usage: tools/test_bench_stats.py        (no GPU, no Docker, no model)
"""
import http.server
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LAYOUTS = {
    "1.0.0": [
        {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
         "tensorfold": {"prefill_seconds": 2.0, "tokens_per_second": 50.0, "seconds": 0.32, "time_to_first_token": 2.1}},
        {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 16, "total_tokens": 1016,
                                  "prompt_tokens_details": {"cached_tokens": 0}}},
    ],
    "0.6.x": [
        {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
        {"choices": [], "tensorfold": {"prefill_s": 2.0, "decode_tps": 50.0, "cached": 0},
         "usage": {"prompt_tokens": 1000, "completion_tokens": 16}},
    ],
    "1.0.0 cache hit": [
        {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
        {"choices": [], "tensorfold": {"prefill_seconds": 0.1, "tokens_per_second": 50.0}},
        {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 16,
                                  "prompt_tokens_details": {"cached_tokens": 900}}},
    ],
}


def serve(chunks):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    failures = 0
    for name, chunks in LAYOUTS.items():
        server = serve(chunks)
        os.environ["API_URL"] = f"http://127.0.0.1:{server.server_address[1]}"
        import importlib
        import bench
        bench = importlib.reload(bench)
        try:
            r = bench.run("hello", 16)
            want = {"prompt_tokens": 1000, "prefill_s": 2.0 if name != "1.0.0 cache hit" else 0.1, "decode_tps": 50.0,
                    "cached": 900 if name == "1.0.0 cache hit" else 0}
            bad = {k: (r.get(k), v) for k, v in want.items() if r.get(k) != v}
            print(("ok  " if not bad else "FAIL") + f" {name}: " + (str(bad) if bad else "stats read"))
            failures += bool(bad)
        finally:
            server.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
