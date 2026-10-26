#!/usr/bin/env python3
"""The OpenAI-compatible client the checks in tools/ share (Python standard library only).

Server: API_URL (default http://127.0.0.1:8888, or http://127.0.0.1:$PORT); a trailing /v1 is accepted. API_KEY, when
set, is sent as a bearer token. MODEL overrides the model id, which is otherwise the first id of /v1/models.

As a command:   tools/client.py [--stream] [--think] [--max-tokens N] [--temperature T] "message"
prints the reply (and, on stderr, the prompt/cached/reply token counts and the server's timings).

Thinking is off unless asked (chat_template_kwargs {"enable_thinking": false}). Errors are one line on stderr, never
a traceback. Exit codes used by every tool here: 0 checked and passed, 1 checked and failed, 2 could not run (server
unreachable, HTTP error, bad reply), 3 ran but a check could not be verified (reported as UNCHECKED, never as a pass).

client.py, needle.py, toolcheck.py and prompt_reuse.py are MiaAI-Lab's, adapted from MiaAI-Lab's tools of the same
names in https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (tools/); long_context.py and
exact.py are new for this recipe.
"""
import http.client
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

PASS, FAIL, CANNOT_RUN, UNCHECKED = 0, 1, 2, 3


def _base_url() -> str:
    url = os.environ.get("API_URL") or "http://127.0.0.1:" + os.environ.get("PORT", "8888")
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


API_URL = _base_url()


class ApiError(Exception):
    """A request that did not give a usable reply; the message is one line."""


def _headers() -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    key = os.environ.get("API_KEY")
    if key:
        h["Authorization"] = "Bearer " + key
    return h


def _one_line(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _error_text(raw: bytes) -> str:
    try:
        body = json.loads(raw)
        err = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(err, dict):
            return err.get("message") or json.dumps(err)
        return str(err)
    except ValueError:
        return raw.decode(errors="replace")


def _open(path: str, body=None, timeout: float = 3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API_URL + path, data, _headers(), method="GET" if body is None else "POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ApiError(f"{path} answered HTTP {exc.code}: {_one_line(_error_text(exc.read()))}") from None
    except urllib.error.URLError as exc:
        raise ApiError(f"cannot reach the server at {API_URL} ({_one_line(exc.reason)}): is it running? (./start.sh)") from None
    except (OSError, http.client.HTTPException) as exc:
        raise ApiError(f"{path}: {_one_line(exc) or type(exc).__name__} (server {API_URL})") from None


def get_json(path: str, timeout: float = 30):
    with _open(path, None, timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw)
    except ValueError:
        raise ApiError(f"{path} did not answer JSON: {_one_line(raw.decode(errors='replace'))}") from None


def post_json(path: str, body: dict, timeout: float = 3600):
    try:
        with _open(path, body, timeout) as resp:
            raw = resp.read()
    except (OSError, http.client.HTTPException) as exc:
        raise ApiError(f"{path}: {_one_line(exc) or type(exc).__name__} while reading the reply") from None
    try:
        return json.loads(raw)
    except ValueError:
        raise ApiError(f"{path} did not answer JSON: {_one_line(raw.decode(errors='replace'))}") from None


_MODEL = None


def model_id() -> str:
    """MODEL, else the first id the server lists at /v1/models."""
    global _MODEL
    if _MODEL is None:
        _MODEL = os.environ.get("MODEL")
        if not _MODEL:
            listing = get_json("/v1/models")
            try:
                _MODEL = listing["data"][0]["id"]
            except (KeyError, IndexError, TypeError):
                raise ApiError(f"/v1/models lists no model: {_one_line(json.dumps(listing))}") from None
    return _MODEL


def count_tokens(text: str):
    """The server's token count of a plain text (vLLM-style /tokenize), or None when it has no such route."""
    try:
        reply = post_json("/tokenize", {"prompt": text, "add_special_tokens": False}, 600)
    except ApiError:
        return None
    if isinstance(reply, dict) and isinstance(reply.get("count"), int):
        return reply["count"]
    if isinstance(reply, dict) and isinstance(reply.get("tokens"), list):
        return len(reply["tokens"])
    return None


def window():
    """The server's context window when it says (/tokenize's max_model_len), else None."""
    try:
        reply = post_json("/tokenize", {"prompt": "a", "add_special_tokens": False}, 60)
    except ApiError:
        return None
    n = reply.get("max_model_len") if isinstance(reply, dict) else None
    return n if isinstance(n, int) and n > 0 else None


class Reply:
    """One chat completion, streamed or not, in one shape."""

    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.tool_calls = []          # [{"id", "name", "arguments" (the raw string)}]
        self.finish_reason = None
        self.usage = {}
        self.tensorfold = {}          # the TensorFold server's runtime extras (prefill_seconds, token_sha, drafts, ...)
        self.speculative = {}
        self.seconds = 0.0            # wall time of the request
        self.first_token_s = None     # streamed: wall time to the first content/reasoning/tool delta
        self.raw = None

    @property
    def prompt_tokens(self):
        return self.usage.get("prompt_tokens")

    @property
    def cached_tokens(self):
        """usage.prompt_tokens_details.cached_tokens, or a server's other spelling; None when it reports none."""
        details = self.usage.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            return details["cached_tokens"]
        for k in ("cached_tokens", "prompt_cache_hit_tokens", "cache_read_input_tokens"):
            if isinstance(self.usage.get(k), int):
                return self.usage[k]
        return None

    @property
    def prefill_seconds(self):
        v = self.tensorfold.get("prefill_seconds")
        if v is None:
            v = self.tensorfold.get("prefill_s")
        return v if isinstance(v, (int, float)) else None

    @property
    def token_sha(self):
        v = self.tensorfold.get("token_sha")
        return v if isinstance(v, str) else None


def chat_body(messages, max_tokens=256, thinking=False, temperature=0.0, stream=False, **extra) -> dict:
    body = {"model": model_id(), "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    body.update(extra)
    return body


def chat(messages, max_tokens=256, thinking=False, temperature=0.0, stream=False, timeout=3600, **extra) -> Reply:
    """POST /v1/chat/completions. extra fields (tools, seed, draft, ...) go into the body as given."""
    body = chat_body(messages, max_tokens, thinking, temperature, stream, **extra)
    t0 = time.time()
    if not stream:
        raw = post_json("/v1/chat/completions", body, timeout)
        r = _parse_whole(raw)
    else:
        r = _stream("/v1/chat/completions", body, timeout, t0)
    r.seconds = time.time() - t0
    return r


def _parse_whole(raw) -> Reply:
    r = Reply()
    r.raw = raw
    try:
        choice = raw["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise ApiError(f"the reply has no choices[0].message: {_one_line(json.dumps(raw))}") from None
    r.content = msg.get("content") or ""
    r.reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        r.tool_calls.append({"id": c.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments")})
    r.finish_reason = choice.get("finish_reason")
    r.usage = raw.get("usage") or {}
    r.tensorfold = raw.get("tensorfold") or {}
    r.speculative = raw.get("speculative") or {}
    return r


def _stream(path, body, timeout, t0) -> Reply:
    r = Reply()
    calls = {}                            # index -> {"id", "name", "arguments"}
    events = 0
    try:
        resp = _open(path, body, timeout)
        with resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    raise ApiError(f"a stream event is not JSON: {_one_line(data)}") from None
                if "error" in chunk:
                    raise ApiError(f"the stream reported an error: {_one_line(_error_text(data.encode()))}")
                events += 1
                for key, attr in (("usage", "usage"), ("tensorfold", "tensorfold"), ("speculative", "speculative")):
                    if isinstance(chunk.get(key), dict):
                        setattr(r, attr, chunk[key])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    got = False
                    if delta.get("content"):
                        r.content += delta["content"]
                        got = True
                    for k in ("reasoning_content", "reasoning"):
                        if delta.get(k):
                            r.reasoning += delta[k]
                            got = True
                    for c in delta.get("tool_calls") or []:
                        slot = calls.setdefault(c.get("index", len(calls)), {"id": None, "name": None, "arguments": ""})
                        if c.get("id"):
                            slot["id"] = c["id"]
                        fn = c.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = (slot["name"] or "") + fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                        got = True
                    if got and r.first_token_s is None:
                        r.first_token_s = time.time() - t0
                    if choice.get("finish_reason"):
                        r.finish_reason = choice["finish_reason"]
    except (OSError, http.client.HTTPException) as exc:
        raise ApiError(f"{path}: the stream broke ({_one_line(exc) or type(exc).__name__})") from None
    if events == 0:
        raise ApiError(f"{path}: the stream carried no events")
    r.tool_calls = [calls[i] for i in sorted(calls)]
    return r


# ---- deterministic filler text (generated; no outside text) ----

_NOUNS = ("river mountain lantern harbor meadow orchard village bridge window garden engine letter signal market "
          "teacher farmer sailor painter traveler merchant scholar shepherd weaver miller baker child neighbor "
          "forest valley island road kitchen library tower field winter summer morning evening storm cloud "
          "candle basket wagon ladder kettle quilt fence stream pebble feather shadow chapel tavern mill").split()
_ADJS = ("quiet old narrow bright gray patient distant gentle crooked steady golden cold warm early late "
         "small wide heavy simple careful humble faded green silver hollow").split()
_VERBS = ("watched carried followed remembered crossed repaired described opened gathered visited painted "
          "mended counted studied lifted wrapped measured noticed planted guarded").split()
_ENDS = ("before the rain came", "after the long harvest", "while the bells rang", "at the edge of the town",
         "under a pale sky", "beside the slow water", "as the lamps were lit", "for the rest of the season",
         "without a single word", "near the end of the road", "when the frost had gone", "in the usual way")


def sentence(rng: random.Random) -> str:
    """One sentence of plain generated prose: words only, no digits (so a needle's code never occurs by chance)."""
    a, b = rng.choice(_NOUNS), rng.choice(_NOUNS)
    pick = rng.random()
    if pick < 0.4:
        s = f"the {rng.choice(_ADJS)} {a} {rng.choice(_VERBS)} the {b} {rng.choice(_ENDS)}"
    elif pick < 0.7:
        s = f"a {a} and a {rng.choice(_ADJS)} {b} were {rng.choice(_ADJS)} {rng.choice(_ENDS)}"
    else:
        s = (f"in the {rng.choice(_ADJS)} {a} someone {rng.choice(_VERBS)} every {b}, "
             f"and the {rng.choice(_NOUNS)} {rng.choice(_VERBS)} it {rng.choice(_ENDS)}")
    return s[0].upper() + s[1:] + "."


def calibrate(make_unit, seed: int, sample_units: int = 3000, fallback: float = 4.0):
    """Characters per token of a unit generator, measured with the server's /tokenize on a seeded sample.
    Returns (chars_per_token, measured: bool). CHARS_PER_TOKEN pins it (for A/B runs against servers without
    /tokenize)."""
    pinned = os.environ.get("CHARS_PER_TOKEN")
    if pinned:
        return float(pinned), False
    rng = random.Random(seed)
    sample = " ".join(make_unit(rng) for _ in range(sample_units))
    n = count_tokens(sample)
    if not n:
        return fallback, False
    return round(len(sample) / n, 3), True


def filler_units(make_unit, tokens: int, seed: int, chars_per_token: float, joiner: str = " ") -> list:
    """Seeded units (sentences or lines) until they hold ~tokens tokens at chars_per_token."""
    rng = random.Random(seed)
    want = int(tokens * chars_per_token)
    out, have = [], 0
    while have < want:
        u = make_unit(rng)
        out.append(u)
        have += len(u) + len(joiner)
    return out


def parse_size(text: str) -> int:
    """'200000', '128k' (128,000) or '1M' (1,000,000)."""
    t = text.strip().lower().replace("_", "").replace(",", "")
    mult = 1
    if t.endswith("k"):
        mult, t = 1000, t[:-1]
    elif t.endswith("m"):
        mult, t = 1_000_000, t[:-1]
    try:
        n = int(float(t) * mult)
    except ValueError:
        raise ApiError(f"not a size: {text!r} (use e.g. 200000, 128k or 1M)") from None
    if n <= 0:
        raise ApiError(f"not a size: {text!r}")
    return n


def run(main) -> None:
    """Run a tool's main with one-line errors: ApiError exits 2, Ctrl-C exits 130; main's return is the exit code."""
    try:
        code = main()
    except ApiError as exc:
        print(f"{os.path.basename(sys.argv[0])}: {exc}", file=sys.stderr, flush=True)
        sys.exit(CANNOT_RUN)
    except KeyboardInterrupt:
        print(f"{os.path.basename(sys.argv[0])}: interrupted", file=sys.stderr, flush=True)
        sys.exit(130)
    sys.exit(code or 0)


def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(description="One chat request to the served model (thinking off unless --think).")
    p.add_argument("message")
    p.add_argument("--stream", action="store_true", help="stream the reply")
    p.add_argument("--think", action="store_true", help="enable thinking")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    a = p.parse_args()
    r = chat([{"role": "user", "content": a.message}], a.max_tokens, a.think, a.temperature, a.stream)
    if r.reasoning:
        print("[reasoning] " + r.reasoning.strip())
    print(r.content)
    for c in r.tool_calls:
        print(f"[tool call] {c['name']}({c['arguments']})")
    tf = r.tensorfold
    print(f"[{model_id()}] prompt {r.prompt_tokens} tok, cached {r.cached_tokens}, reply "
          f"{r.usage.get('completion_tokens')} tok, finish {r.finish_reason}, {r.seconds:.2f} s"
          + (f", prefill {r.prefill_seconds:.3f} s" if r.prefill_seconds is not None else "")
          + (f", {tf['tokens_per_second']:.1f} tok/s" if isinstance(tf.get("tokens_per_second"), (int, float)) else "")
          + (f", first token {r.first_token_s:.2f} s" if r.first_token_s is not None else ""), file=sys.stderr)
    return PASS


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    run(_cli)
