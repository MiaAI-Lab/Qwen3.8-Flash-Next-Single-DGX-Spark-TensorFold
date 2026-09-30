# Cooperative Flash Next prefill

A newly admitted long prompt used to monopolize the CUDA scheduler worker until
prefill finished. The worker now runs **one round of existing decodes after each
fully committed nonfinal prefill chunk**. Completion and error replies use the
normal scheduler path. This is cooperative execution on the same worker/CUDA
stream, not concurrent CUDA execution or another admission thread.

The pending request's slot is already reserved but is not yet live. The callback
never admits another request or invokes background-lane eviction: prefill scratch
must not be reentered. Recurrent/KV state, MTP absorption and PLE history are
committed before yielding. The final chunk, first sample and initial draft setup
remain atomic. Decoders without `prefill_yield` retain their existing contract;
serial prefill has no callback.

## Offline checks

The recipe remains pinned to TensorFold **v0.3.6.3**. Patch
`0010-flash-next-cooperative-prefill.patch` follows 0001–0009 and applies with
`patch -p0` against site-packages. No configuration/default changes are required;
the existing patch hash automatically gives the image a new identity. Existing
prebuilt images do not contain this change; normal preparation builds locally when
no image with the new hash is available. No image was built or published here.

With an existing local TensorFold git checkout containing v0.3.6.3:

```sh
# Requires git, patch, Python and NumPy; no torch/Docker/server/GPU/download:
python3 tools/check_cooperative_prefill.py /path/to/TensorFold
python3 tests/test_prefill_fairness_probe.py
```

The checker resolves the tag to `191188075bca56a7c71074a79375eb4c1cb22e1c`,
archives it into temporary storage, applies all patches with zero fuzz, and runs
both source suites. It separately checks the default and optional language patch
chains in the basename order used by the Docker build. The language patch's
existing `0010-...draft-languages` name does not collide with this new filename.
For an already patched source tree, set `TENSORFOLD_SOURCE` to its `tensorfold`
package directory and run each `tests/test_cooperative_prefill*.py` file directly.

The stdlib lifecycle suite executes actual scheduler, Stream, admission, prefill,
round and finish declarations, extracting them via AST to avoid CUDA imports.
Tensor/kernel execution and the worker thread are explicit test doubles. It covers
chunk boundaries, MTP resume order, solo prefill, reserved slots, cache eviction,
completion/cancellation/error delivery, and bounded queued work. The NumPy suite
executes real host `commit` and keyed `ReadAhead` declarations with a shift-kernel
stand-in. These are **not native CUDA or device-fault-recovery tests**.

The new patch changes only scheduler.py, Flash Next decode.py and multi.py.
It preserves the existing vision, copy-draft and read-ahead recipe changes. The
resulting three source files match the reviewed frozen GPU candidate byte for byte.
The separate current-upstream adaptation is not part of this recipe patch.
The current recipe default is five slots; the archived run below used four.
**The default five-slot startup budget and optional language GPU path were not
revalidated by these offline checks.**

## Archived same-model observation

A separate, earlier single-GB10 run used TensorFold **0.3.6.3**, recipe patches
0001–0009, and the reviewed three-file callback fix. Settings: Vontra
`Qwen3.8-Flash-Next-MLX-4bit-MTP`, four slots, 262144 context, int8 KV, SSD PLE,
2048-row chunks, six MTP drafts/confidence 0.60; requests used temperature 0,
top_p 1, seed 7319 and thinking off. Control and candidate used the same checkpoint
and selected runtime settings; only the three scheduling source files differed.

One paired lifecycle trial injected a **154063-token uncached** synthetic prompt
while a 48-token answer was already streaming:

| Observation | Control | Cooperative |
|---|---:|---:|
| Active reply completed after injection (s) | 77.720 | 14.726 |
| Maximum active content-event gap (s) | 76.853 | 0.939 |
| Long request time to first content (s) | 76.814 | 77.751 |
| Long request wall time (s) | 77.124 | 78.005 |
| Active reply completed before long first content | No | Yes |

Both full active answers and both long markers matched byte for byte. Exact
sanitized numbers and source-record/output SHA-256 values are in
[`benchmarks/cooperative-prefill-observation.json`](benchmarks/cooperative-prefill-observation.json).
These are one paired observation, not a statistical throughput claim or current-main
GPU result. The raw deployment record hashes bind the extraction; raw deployment
logs are not included. No cross-engine speed or quality ranking is needed for this fix.

## Reproduce the scheduling experiment

Use an explicitly authorized, isolated server; the probe sends substantial GPU work.
Keep model revision, runtime, KV format, chunk size, speculative settings and slot
count fixed between control and candidate. Use at least four slots for three active
requests plus one long prompt. Warm a short request first. Start a fresh server/cache
for each arm, and use the same seed in each pair. Keep other traffic off the server.

```sh
python3 tools/bench_prefill_fairness.py http://127.0.0.1:8888 Qwen3.8-Flash-Next \
  --active 3 --tokens 2048 --words 120000 --seed 7319 > control.json
# Run the same command against the candidate server, saving candidate.json.
# Repeat fresh paired runs with seeds 7320 and 7321; alternate arm order.
```

This compact public probe uses new, entirely synthetic prompts. It reproduces the
**protocol**, not the historical prompts or exact numbers above. `--words` counts
words, not tokenizer tokens: inspect `long.usage.prompt_tokens` and tune it to fit
context with room for output/speculation. It refuses missing/nonzero cache counters,
absent active overlap, missing usage/finish/DONE, unexpected reasoning, or a wrong
long marker. A failed probe can leave server work running: client disconnect is
not prompt cancellation. Do not retry into a busy server without checking it.

For the short-active completion case, use `--active 1 --tokens 48` (increase tokens
if the active reply finishes before injection). Compare full `active[*].text` and
`long.text` for each matched control/candidate pair; deterministic output is a
measured check, not an all-prompt guarantee. Reports contain relative content-event
times, usage, output and overlap metrics, not the endpoint or local paths. SSE can
bundle tokens: maximum content-event gap is **not per-token ITL**. Long TTFT includes
scheduling work; fairness can increase the long request's latency.

## Limits

This is not prompt cancellation, queue bounding, overload admission control, a
slow-reader backpressure fix, or comprehensive JSON/tool/vision/format validation.
The frozen implementation's pre-output cancellation could still occupy work for
roughly 74–93 seconds. Waiting and client token queues remain unbounded. Python
error-path tests do not demonstrate recovery from a poisoned CUDA context. Other
formats and current-main GPU performance need their own validation. No weights,
allocations, startup budgets, serving defaults or sampling policy change here.
