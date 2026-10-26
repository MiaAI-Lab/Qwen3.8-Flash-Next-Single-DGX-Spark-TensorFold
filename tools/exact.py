#!/usr/bin/env python3
"""Exactness at the API: greedy replies must not depend on drafting or on what else runs at the same time.

Usage: tools/exact.py [--max-tokens 160] [--prompts N] [--pad TOKENS]     (API_URL / PORT / MODEL as in client.py;
       thinking off). --pad puts ~TOKENS of generated prose (e.g. 100, 1k, 3k) before each question, a different
       text per prompt, so the prompt path is checked at that length too.
Sends the same greedy requests (temperature 0, fixed seed) three ways:
  serial   one at a time, drafts as the server serves them
  nodraft  one at a time with "draft": false
  conc     all at once (threads), drafts as served
and prints, per prompt, the sha256 of each reply's text (12 hex) and the server's token sha when it reports one.
Checks: nodraft == serial and conc == serial (text, token sha when reported, finish reason).
The draft check is UNCHECKED, not passed, when the server says it served without drafts (or does not say), since then
it compared serial with serial; "draft": false that the server reports as drafted is a FAIL.
Exit 0 pass, 1 a reply differs, 2 could not run, 3 a check could not be verified.
"""
import argparse
import hashlib
import sys
import threading

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
import client  # noqa: E402

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in four sentences why the sky looks blue during the day and red at sunset.",
    "List the first twelve prime numbers, then give their sum and show the addition.",
    "Translate into French: 'The old bridge was repaired before the winter storms came to the valley.'",
    "Write a short JSON object describing a library book with title, author, year and three subject tags.",
    "A train leaves at 09:40 and arrives at 13:15. How long is the trip? Show the steps.",
]


def text_sha(r: client.Reply) -> str:
    blob = r.content + "\x00" + r.reasoning + "\x00" + "".join(f"{c['name']}({c['arguments']})" for c in r.tool_calls)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def ask(prompt: str, max_tokens: int, **extra) -> client.Reply:
    return client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, thinking=False, temperature=0.0,
                       seed=4321, **extra)


def same(a: client.Reply, b: client.Reply) -> bool:
    if text_sha(a) != text_sha(b) or a.finish_reason != b.finish_reason:
        return False
    if a.token_sha is not None and b.token_sha is not None and a.token_sha != b.token_sha:
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--prompts", type=int, default=len(PROMPTS), help=f"how many of the {len(PROMPTS)} prompts")
    p.add_argument("--pad", default="0", help="~tokens of prose before each question (100, 1k, 3k; default none)")
    a = p.parse_args()
    prompts = PROMPTS[:max(1, min(a.prompts, len(PROMPTS)))]
    client.model_id()
    pad = client.parse_size(a.pad) if a.pad not in ("", "0") else 0
    if pad:
        cpt, _ = client.calibrate(client.sentence, 7)
        prompts = [" ".join(client.filler_units(client.sentence, pad, 50 + i, cpt)) + "\n\n" + q
                   for i, q in enumerate(prompts)]
        sizes = [client.count_tokens(q) for q in prompts]
        print(f"prompts padded to ~{pad} tokens: {sizes}")
    try:
        lanes = client.get_json("/health").get("max_batch_size")
    except (client.ApiError, AttributeError):
        lanes = None

    serial = [ask(q, a.max_tokens) for q in prompts]
    nodraft = [ask(q, a.max_tokens, draft=False) for q in prompts]
    conc = [None] * len(prompts)
    errors = []

    def worker(i):
        try:
            conc[i] = ask(prompts[i], a.max_tokens)
        except client.ApiError as exc:
            errors.append(str(exc))
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise client.ApiError(f"a concurrent request failed: {errors[0]}")

    def tok(r):
        return r.token_sha or "-"
    print(f"{'#':>2}  {'serial':<12} {'nodraft':<12} {'conc':<12}  token sha serial/nodraft/conc       accepted  finish")
    for i, (s, n, c) in enumerate(zip(serial, nodraft, conc)):
        acc = s.speculative.get("accepted")
        print(f"{i:>2}  {text_sha(s):<12} {text_sha(n):<12} {text_sha(c):<12}  {tok(s)}/{tok(n)}/{tok(c)}  "
              f"{acc if acc is not None else '-':>8}  {s.finish_reason}"
              + ("" if same(s, n) else "  NODRAFT DIFFERS") + ("" if same(s, c) else "  CONC DIFFERS"))
    for i, (s, n, c) in enumerate(zip(serial, nodraft, conc)):
        if not (same(s, n) and same(s, c)):
            print(f"  #{i} serial : {s.content[:160]!r}\n  #{i} nodraft: {n.content[:160]!r}\n  #{i} conc   : {c.content[:160]!r}")

    code = client.PASS
    drafted = [s.tensorfold.get("drafts") for s in serial]
    undrafted = [n.tensorfold.get("drafts") for n in nodraft]
    accepted = sum(s.speculative.get("accepted") or 0 for s in serial)
    draft_ok = all(same(s, n) for s, n in zip(serial, nodraft))
    if any(u is True for u in undrafted):
        print('FAIL draft: the server reports drafts on for "draft": false requests')
        code = client.FAIL
    elif not draft_ok:
        print("FAIL draft: a greedy reply with drafts differs from the same request with \"draft\": false")
        code = client.FAIL
    elif not all(d is True for d in drafted):
        why = "the server reports it served without drafts" if any(d is False for d in drafted) else \
            "the server does not report whether drafts ran"
        print(f"UNCHECKED draft: replies equal, but {why}, so drafted vs serial was not compared")
        code = client.UNCHECKED
    else:
        print(f"PASS draft: {len(prompts)} drafted replies ({accepted} drafts accepted) equal the \"draft\": false ones")
        if accepted == 0:
            print("  note: no draft was accepted, so the drafted path barely ran")
    if all(same(s, c) for s, c in zip(serial, conc)):
        note = f" (server max_batch_size {lanes}: the requests queued, batching not exercised)" if lanes == 1 else \
            (f" (server max_batch_size {lanes})" if lanes is not None else "")
        print(f"PASS conc: {len(prompts)} concurrent replies equal the one-at-a-time ones{note}")
    else:
        print("FAIL conc: a reply sent concurrently differs from the same request sent alone")
        code = client.FAIL
    print("exact:", {client.PASS: "PASS", client.FAIL: "FAIL", client.UNCHECKED: "UNCHECKED"}[code], flush=True)
    return code


if __name__ == "__main__":
    client.run(main)
