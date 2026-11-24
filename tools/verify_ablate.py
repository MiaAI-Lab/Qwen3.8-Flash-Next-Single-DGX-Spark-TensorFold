#!/usr/bin/env python3
"""Per-request abliteration (patches 0006 to 0008), against a running server.

Usage: tools/verify_ablate.py [model id ...]   (default: SERVED_NAME and SERVED_NAME-abliterated; API_URL / PORT / API_KEY as
       in client.py, CONTAINER for the prompt-cache log lines)
Checks, each printed PASS or FAIL:
  own state  each id is sent one ~12k-token prompt twice, in turn. Its cold pass resumes under 10% of the prompt (no other id's
             state, though the previous id has just kept the same text) and its second pass at least 90% (its own state), with
             the same greedy reply. Unchecked when the server reports no cached-token count.
  mixed      short greedy prompts for every id, one at a time, then all at once so stock and identity rows share decode rounds:
             each concurrent reply equals its one-at-a-time one.
Exit 0 pass, 1 fail, 2 could not run, 3 a check could not be verified.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
import client  # noqa: E402

SERVED = os.environ.get("SERVED_NAME", "Qwen3.8-Flash-Next")
MODELS = sys.argv[1:] or [SERVED, SERVED + "-abliterated"]
CONTAINER = os.environ.get("CONTAINER", "qwen38-flash-next-tf")
FILLER = " ".join(f"Record {i}: the valve on line {i % 97} was inspected at {i % 24:02d}:00 and logged as nominal." for i in range(450))
LONG = [
    {"role": "system", "content": "You are a terse maintenance assistant. Answer in one short sentence."},
    {"role": "user", "content": "Here is the inspection log.\n" + FILLER},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "Which line number appears in record 133? Reply with the number only."},
]
SHORT = ["Name the three primary colours of light.", "What is 17 times 23? Reply with the number only.",
         "Write one sentence about the sea."]


def ask(model: str, messages: list, max_tokens: int) -> tuple[str, int, int | None]:
    """The reply's text, the prompt tokens and the cached ones (None when the server does not say)."""
    r = client.chat(messages, max_tokens=max_tokens, model=model)
    return r.content, r.prompt_tokens or 0, r.cached_tokens


def cache_line() -> str:
    logs = subprocess.run(["docker", "logs", "--tail", "40", CONTAINER], capture_output=True, text=True)
    lines = [line for line in (logs.stdout + logs.stderr).splitlines() if "prompt cache:" in line]
    return lines[-1][:170] if lines else "(no prompt cache line)"


def main() -> int:
    fails, unchecked = 0, 0
    for model in MODELS:
        cold, prompt, cold_cached = ask(model, LONG, 24)
        print(f"{model}: cold  prompt={prompt} cached={cold_cached} reply={cold!r}\n    {cache_line()}")
        again, prompt, again_cached = ask(model, LONG, 24)
        print(f"{model}: again prompt={prompt} cached={again_cached} reply={again!r}\n    {cache_line()}")
        if cold != again:
            fails += 1
            print(f"FAIL own state {model}: the resumed reply differs from the cold one")
        elif cold_cached is None or again_cached is None:
            unchecked += 1
            print(f"UNCHECKED own state {model}: the server reports no cached-token count")
        elif cold_cached >= 0.1 * prompt or again_cached < 0.9 * prompt:
            fails += 1
            print(f"FAIL own state {model}: cold resumed {cold_cached}, again {again_cached} of {prompt} tokens")
        else:
            print(f"PASS own state {model}: cold resumed {cold_cached}, again {again_cached} of {prompt}, same reply")

    jobs = [(model, text) for model in MODELS for text in SHORT]
    serial = {job: ask(job[0], [{"role": "user", "content": job[1]}], 48)[0] for job in jobs}
    with ThreadPoolExecutor(len(jobs)) as pool:
        mixed = dict(zip(jobs, pool.map(lambda job: ask(job[0], [{"role": "user", "content": job[1]}], 48)[0], jobs)))
    differ = [job for job in jobs if mixed[job] != serial[job]]
    for model, text in differ:
        print(f"  {model} / {text!r}: alone {serial[(model, text)]!r}, mixed {mixed[(model, text)]!r}")
    fails += bool(differ)
    print(f"{'FAIL' if differ else 'PASS'} mixed: {len(jobs) - len(differ)} of {len(jobs)} concurrent replies "
          f"({len(MODELS)} ids) equal their one-at-a-time ones")
    differ_ids = len({serial[(m, t)] for m in MODELS for t in SHORT}) > len(SHORT)
    print(f"(the ids' replies {'differ' if differ_ids else 'are all the same'} on these prompts)")

    print("verify_ablate:", "FAIL" if fails else "UNCHECKED" if unchecked else "PASS")
    return 1 if fails else 3 if unchecked else 0


if __name__ == "__main__":
    client.run(main)
