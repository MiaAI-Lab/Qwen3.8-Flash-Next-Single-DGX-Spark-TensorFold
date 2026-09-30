"""Offline tests for the synthetic SSE probe; no server/model/GPU dependency."""

import importlib.util
import io
import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).resolve().parents[1] / "tools/bench_prefill_fairness.py"
)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def response(*chunks, done=True):
    raw = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    return io.BytesIO((raw + ("data: [DONE]\n\n" if done else "")).encode())


class ProbeTests(unittest.TestCase):
    def test_cli_against_synthetic_loopback_sse_fixture(self):
        # Protocol test only: the fixture is not an inference engine/benchmark result.
        injected, progressed, finished = (threading.Event() for _ in range(3))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(content, terminal=False):
                    chunk: dict = {
                        "choices": [{"delta": {"content": content}, "finish_reason": "stop" if terminal else None}]
                    }
                    if terminal:
                        chunk.update(usage={"prompt_tokens": 10, "completion_tokens": 2}, tensorfold={"cached": 0})
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    if terminal:
                        self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

                if body["max_tokens"] == 16:
                    injected.set()
                    if not progressed.wait(5):
                        return
                    send("PREFILL_OK", terminal=True)
                    finished.set()
                else:
                    send("first ")
                    if not injected.wait(5):
                        return
                    send("second")
                    progressed.set()
                    if not finished.wait(5):
                        return
                    send("", terminal=True)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            assert probe.__file__ is not None
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(probe.__file__)),
                    f"http://127.0.0.1:{server.server_port}",
                    "synthetic-fixture",
                    "--active",
                    "1",
                    "--tokens",
                    "2",
                    "--words",
                    "10",
                    "--timeout",
                    "5",
                ],
                text=True,
                capture_output=True,
                timeout=15,
                check=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["long"]["text"], "PREFILL_OK")
            self.assertEqual(report["active"][0]["text"], "first second")
            self.assertGreater(report["worst_overlapping_content_gap_s"], 0)
            self.assertNotIn("127.0.0.1", result.stdout)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(5)

    def test_multiline_crlf_comments(self):
        raw = io.BytesIO(b': keepalive\r\ndata: {"a":\r\ndata: 1}\r\n\r\ndata: [DONE]\r\n\r\n')
        self.assertEqual(list(probe.events(raw)), ['{"a":\n1}', "[DONE]"])

    def test_usage_after_finish_and_bundled_content(self):
        data = response(
            {"choices": [{"delta": {"content": "two words"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 2}, "tensorfold": {"cached": 0}},
        )
        with patch.object(probe.urllib.request, "urlopen", return_value=data):
            result = probe.stream("http://localhost/v1/chat/completions", {}, 0, 1)
        self.assertEqual(result["text"], "two words")
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertEqual(len(result["events"]), 1)  # never equate chunks with tokens
        self.assertEqual(result["cached"], 0)

    def test_incomplete_and_error_streams_fail(self):
        for data in (
            response({"error": {"message": "failure"}}),
            response({"choices": [{"delta": {"content": "text"}}]}, done=False),
            response({"choices": [{"delta": {"reasoning_content": "thinking"}}]}),
        ):
            with (
                self.subTest(),
                patch.object(probe.urllib.request, "urlopen", return_value=data),
                self.assertRaises(ValueError),
            ):
                probe.stream("http://localhost/v1/chat/completions", {}, 0, 1)

    def test_gap_window_cache_and_active_guards(self):
        active = [{"events": [[1, "a"], [3, "b"], [4, "c"]], "end_s": 4.1}]
        long = {"events": [[5, "PREFILL_OK"]], "sent_s": 2, "end_s": 5.1, "cached": 0}
        result = probe.summarize(active, long, 2)
        self.assertEqual(result["worst_overlapping_content_gap_s"], 2)
        self.assertEqual(result["active_finished_before_long_first"], [True])
        for cached in (None, 1):
            with self.subTest(cached=cached), self.assertRaises(ValueError):
                probe.summarize(active, {**long, "cached": cached, "usage": {}}, 2)
        with self.assertRaises(ValueError):
            probe.summarize(active, long, 4.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
