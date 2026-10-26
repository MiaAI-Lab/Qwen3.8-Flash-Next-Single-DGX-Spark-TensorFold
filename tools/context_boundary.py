#!/usr/bin/env python3
"""Check the advertised prompt/reply window without filling it with generated text.

Usage: tools/context_boundary.py --context 262144
API_URL / PORT / MODEL / API_KEY as in client.py. Uses a short, greedy, thinking-off
reply with natural EOS. Tests whole and streamed replies, with drafts on and off.
The context must match the server's --context; no service configuration is changed.
"""
import argparse
import sys

sys.dont_write_bytecode = True
import client  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context", type=int, required=True)
    args = parser.parse_args()
    messages = [{"role": "user", "content": "Reply with exactly OK and nothing else."}]
    template = {"enable_thinking": False}
    tokenized = client.post_json("/tokenize", {
        "model": client.model_id(), "messages": messages,
        "chat_template_kwargs": template, "add_generation_prompt": True,
    })
    count = tokenized.get("count")
    if type(count) is not int or args.context - count <= 16:
        raise client.ApiError("need a token count and a context with more than 16 reply tokens")
    room = args.context - count
    checked = 0
    for stream in (False, True):
        for draft in (False, True):
            for reserve in (16, 1, 0):
                try:
                    reply = client.chat(messages, max_tokens=room - reserve,
                                        stream=stream, draft=draft, timeout=120)
                except client.ApiError as exc:
                    print(f"FAIL: stream={stream} draft={draft} reserve={reserve}: {exc}")
                    return client.FAIL
                if (reply.prompt_tokens != count or reply.finish_reason != "stop"
                        or not reply.content.strip()):
                    print(f"FAIL: incomplete reply or prompt count mismatch: stream={stream} "
                          f"draft={draft} reserve={reserve}")
                    return client.FAIL
                checked += 1
            try:
                client.chat(messages, max_tokens=room + 1, stream=stream, draft=draft, timeout=120)
            except client.ApiError as exc:
                if "HTTP 400" not in str(exc) or "context" not in str(exc).lower():
                    print(f"FAIL: expected a context HTTP 400 before any stream opens: {exc}")
                    return client.FAIL
            else:
                print("FAIL: the server accepted prompt + reply > context")
                return client.FAIL
            checked += 1
    print(f"context_boundary: PASS: {checked} checks; prompt={count}, context={args.context}")
    return client.PASS


if __name__ == "__main__":
    client.run(main)
