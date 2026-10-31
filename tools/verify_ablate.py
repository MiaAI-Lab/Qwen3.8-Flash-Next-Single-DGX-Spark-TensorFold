"""Per-request abliteration (patches 0018, 0019): every model id given (default: the stock SERVED_NAME and SERVED_NAME-abliterated)
is sent one long prompt twice; the second pass resumes that id's own kept state (the prompt cache line says so) and its
greedy reply matches the cold one. Ids never resume each other's states: each cold pass starts from zero.
Usage: tools/verify_ablate.py [model id ...]   (API_URL, default http://127.0.0.1:$PORT with PORT 8888; CONTAINER for the logs)"""
import json, os, sys, time, urllib.request, subprocess

API_URL = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")).rstrip("/").removesuffix("/v1")
CONTAINER = os.environ.get("CONTAINER", "qwen38-flash-next-tf")
SERVED = os.environ.get("SERVED_NAME", "Qwen3.8-Flash-Next")
MODELS = sys.argv[1:] or [SERVED + "-abliterated", SERVED]
filler = " ".join(f"Record {i}: the valve on line {i % 97} was inspected at {i % 24:02d}:00 and logged as nominal." for i in range(450))
msgs = [
    {"role": "system", "content": "You are a terse maintenance assistant. Answer in one short sentence."},
    {"role": "user", "content": "Here is the inspection log.\n" + filler},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "Which line number appears in record 133? Reply with the number only."},
]

def ask(model):
    body = {"model": model, "messages": msgs, "max_tokens": 24, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    req = urllib.request.Request(API_URL + "/v1/chat/completions", json.dumps(body).encode(), headers)
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], out.get("usage", {}), time.time() - t

def cache_line():
    logs = subprocess.run(["docker", "logs", "--tail", "40", CONTAINER], capture_output=True, text=True)
    lines = [l for l in (logs.stdout + logs.stderr).splitlines() if "prompt cache:" in l]
    return lines[-1] if lines else "(no prompt cache line)"

ok = True
for model in MODELS:
    cold, usage, dt = ask(model)
    print(f"{model:20} cold  {dt:6.2f}s prompt={usage.get('prompt_tokens')} reply={cold!r}\n                     {cache_line()[:170]}")
    again, usage, dt = ask(model)
    print(f"{model:20} again {dt:6.2f}s prompt={usage.get('prompt_tokens')} reply={again!r}\n                     {cache_line()[:170]}")
    ok &= cold == again
print("SAME_GREEDY_REPLIES" if ok else "REPLIES_DIFFER")
sys.exit(0 if ok else 1)
