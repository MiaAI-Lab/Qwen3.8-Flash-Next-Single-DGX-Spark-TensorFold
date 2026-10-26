#!/usr/bin/env python3
"""Tool calling: an array parameter comes back as a JSON array (whole and streamed), and a tool-result turn works.

Usage: tools/toolcheck.py      (API_URL / PORT / MODEL as in client.py; thinking off, greedy)
Checks, each printed PASS or FAIL:
  1. call:     a reply calls add_tags with doc_id the integer 42 and tags a JSON array holding urgent, finance, q3.
  2. stream:   the same request streamed; the argument deltas joined give the same typed arguments.
  3. result:   the conversation goes on with the tool's result (a role "tool" message); the next reply is text, not
               another call, and quotes the confirmation code that only the tool result held.
Exit 0 all pass, 1 any fails, 2 could not run.
"""
import json
import sys

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
import client  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "add_tags", "description": "Attach tags to a document.",
    "parameters": {"type": "object", "required": ["doc_id", "tags"], "properties": {
        "doc_id": {"type": "integer", "description": "The document id."},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "The tags to attach."}}}}}]
ASK = [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool. "
                                   "Then tell me the confirmation code the tool returns."}]
WANT_TAGS = {"urgent", "finance", "q3"}
CODE = "TAGGED-58213-KESTREL"


def check_call(r: client.Reply, how: str):
    """(ok, why, call) for the first add_tags call of a reply."""
    if not r.tool_calls:
        return False, f"{how}: no tool call (finish {r.finish_reason}, content {r.content.strip()[:120]!r})", None
    for c in r.tool_calls:
        print(f"  {how} tool call {c['name']}({c['arguments']})  id={c['id']}")
    c = r.tool_calls[0]
    if c["name"] != "add_tags":
        return False, f"{how}: called {c['name']!r}, not add_tags", c
    try:
        args = json.loads(c["arguments"]) if isinstance(c["arguments"], str) else c["arguments"]
    except ValueError:
        return False, f"{how}: arguments are not JSON: {c['arguments']!r}", c
    if not isinstance(args, dict):
        return False, f"{how}: arguments are not a JSON object: {args!r}", c
    tags, doc = args.get("tags"), args.get("doc_id")
    if not isinstance(tags, list):
        return False, f"{how}: tags is {type(tags).__name__} {tags!r}, not a JSON array", c
    if not all(isinstance(t, str) for t in tags):
        return False, f"{how}: tags holds non-strings: {tags!r}", c
    if not WANT_TAGS <= {t.strip().lower() for t in tags}:
        return False, f"{how}: tags {tags!r} miss {sorted(WANT_TAGS - {t.strip().lower() for t in tags})}", c
    if isinstance(doc, bool) or not isinstance(doc, int) or doc != 42:
        return False, f"{how}: doc_id is {doc!r} ({type(doc).__name__}), not the integer 42", c
    if r.finish_reason != "tool_calls":
        return False, f"{how}: typed arguments, but finish_reason is {r.finish_reason!r}, not 'tool_calls'", c
    return True, f"{how}: tags is a JSON array {tags}, doc_id the integer 42, finish tool_calls", c


def main() -> int:
    results = []
    r1 = client.chat(ASK, max_tokens=512, tools=TOOLS)
    ok, why, call = check_call(r1, "call")
    results.append((ok, why))
    r2 = client.chat(ASK, max_tokens=512, tools=TOOLS, stream=True)
    results.append(check_call(r2, "stream")[:2])

    if ok:
        used, args = "the server's own call", call["arguments"]
        call_id = call["id"] or "call_0"
    else:
        used, args, call_id = "a written call (check 1 failed)", json.dumps({"doc_id": 42, "tags": sorted(WANT_TAGS)}), "call_0"
    turn = ASK + [
        {"role": "assistant", "content": r1.content if ok else "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "add_tags", "arguments": args}}]},
        {"role": "tool", "tool_call_id": call_id, "name": "add_tags",
         "content": json.dumps({"ok": True, "doc_id": 42, "tags_added": 3, "confirmation": CODE})}]
    r3 = client.chat(turn, max_tokens=256, tools=TOOLS)
    text = r3.content.strip()
    if r3.tool_calls:
        results.append((False, f"result: answered the tool result with another call {r3.tool_calls[0]['name']}"
                               f"({r3.tool_calls[0]['arguments']}) instead of text"))
    elif CODE.lower() not in text.lower():
        results.append((False, f"result: the reply after the tool result lacks the code {CODE}: {text[:160]!r}"))
    else:
        results.append((True, f"result: after {used} and its result, the reply quotes {CODE}: {text[:100]!r}"))

    for ok_i, why in results:
        print(("PASS " if ok_i else "FAIL ") + why)
    passed = sum(ok_i for ok_i, _ in results)
    print(f"toolcheck: {passed}/{len(results)} passed", flush=True)
    return client.PASS if passed == len(results) else client.FAIL


if __name__ == "__main__":
    client.run(main)
