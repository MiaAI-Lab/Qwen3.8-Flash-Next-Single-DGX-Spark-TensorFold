<h1 align="center">Qwen3.8 Flash Next on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

> **NVIDIA's quantization-aware-distilled (QAD) FP4 checkpoint, close to bf16.** The default model is
> [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): NVIDIA trained the
> NVFP4 and MXFP8 weights against the bf16 model as teacher, so the 4-bit weights are tuned to match it. On NVIDIA's own
> numbers it scores 91.5% on GPQA Diamond and 88.7% on ToolEvalBench against 92.2% and 89.4% for bf16 (model card, not
> measured here). It is what `./start.sh` serves; the [performance](#performance) numbers below are for it.

Serve **Qwen3.8 Flash Next** from a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API. It runs
[TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, `tensorfold-native` (branch
[`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext)), in NVIDIA's PyTorch container. One GPU,
with image and video input (`VISION=1`). `VISION=0` serves text only.

The default checkpoint is NVIDIA's FP4 QAD export: routed experts in NVFP4 (group size 16, CSF-compressed scale factors),
attention and shared-expert linears in MXFP8, the MTP experts and the vision tower quantized too (MXFP8 attention, W4A16
NVFP4 MLP), an FP8 n-gram table, and all 512 routed experts with 10 per token (the INT4 cut keeps 5). The optional INT4-AutoRound checkpoint (`CHECKPOINT=int4`) uses 5 experts per token (4.8B active
parameters) and scores about 10% lower general capability on its authors' own harness (46.6-47.6 against 51.8). Tool use
is unchanged.

- Checkpoint (default): [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
  at revision `b9ce18a6887e18c089fdea8509a33e63429a3300` (~96 GiB; loading takes about 81 GiB of memory with vision on, measured).
- Checkpoint (optional, `CHECKPOINT=int4`): [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
  at revision `1464274120d36a4d8fcaa934552334a7d83ce0fd` (~122 GiB). Routed experts are GPTQ int4 (group size 128), the
  n-gram table is FP8 and lives in `ple-table/`, and it is loaded with the weights. There is no SSD n-gram reader and
  no int8 KV cache. KV is FP8 by default (lossy, ~98.8% top-1 agreement with bf16); `KV_DTYPE=bf16` is exact. FP8 works with image and video input.
- API model id: `Qwen3.8-Flash-Next`
- Context: **262,144 tokens** a request (the native window). Up to 5 requests at once (`PARALLEL`). A full window's
  fp8 cache is 4.37 GiB (bf16 is 7.52 GiB). The engine grows each request's cache as it goes and refuses one that would not fit, keeping
  10 GiB free, and another 2 GiB for the vision helper.
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## Performance

Numbers for the default checkpoint, **`nvidia/Qwen3.8-Flash-Next-FP4-Local`** (NVIDIA's QAD FP4 export), as `./start.sh` serves it: 262,144-token
window, 5 requests at once, fp8 KV, vision on, MTP drafts. One DGX Spark (GB10), measured with
[sparkDash](https://github.com/MiaAI-Lab/sparkDash)'s Decode and Prefill benchmarks, running on the same Spark as the server (no network between
them), 400 tokens a request. Aggregate is the sum of the per-request decode speeds; TTFT is the time to the first token.

**Decode, prose**

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 46–55 tok/s (varies) | 46–55 tok/s | 195 ms |
| 2 | 73.5 tok/s | 38.1 tok/s | 204 ms |
| 3 | 90.3 tok/s | 31.2 tok/s | 218 ms |
| 4 | 106.4 tok/s | 27.9 tok/s | 243 ms |
| 5 | 132.7 tok/s | 27.7 tok/s | 266 ms |

**Decode, code**

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 67.9 tok/s | 67.9 tok/s | 181 ms |
| 2 | 100.1 tok/s | 51.3 tok/s | 330 ms |
| 3 | 117.1 tok/s | 41.1 tok/s | 315 ms |
| 4 | 131.2 tok/s | 34.8 tok/s | 323 ms |
| 5 | 149.7 tok/s | 31.9 tok/s | 324 ms |

**Prefill** (cold prompts, no prompt-cache hit)

| Prompt tokens | Prefill | TTFT |
| ---: | ---: | ---: |
| 4,136 | 2,665 tok/s | 1.73 s |
| 8,230 | 2,583 tok/s | 3.36 s |
| 16,421 | 2,501 tok/s | 6.74 s |
| 32,806 | 2,430 tok/s | 13.67 s |
| 65,573 | 2,331 tok/s | 28.30 s |
| 131,110 | 2,197 tok/s | 59.86 s |

Prefill is 13 to 15% faster than before patch 0014 (4k: 2,325 to 2,665 tok/s, 16k: 2,191 to 2,501, 128k: 1,947 to 2,197): the routed experts of
prompt chunks now run through a shared-memory tile kernel (`TF_FLASHNEXT_PROMPT_EXPERT_TILE`, on by default, bit-identical results). Paired,
alternating runs of the same build with the tile off and on (3 server starts each, median of 3 prompts per start, `tools/bench_streams.py`):
2,090 to 2,397 tok/s at 4k (+14.7%), 2,174 to 2,474 at 16k (+13.8%), 2,161 to 2,450 at 32k (+13.4%). Decode is unchanged (the tile only runs for
prompt chunks of 1,024 rows or more).

Single-stream prose varies from run to run, 46 to 55 tok/s with the same settings (46.2, 46.9, 48.5 and 55.0 tok/s in four runs, about 15%); the
other points agreed within 2 tok/s. Run sparkDash on the Spark itself: from another machine over a slow link it adds the link's round trip to every
TTFT (about 1.1 s over a 387 ms Tailscale path in our test) and staggers the requests, which lowers the multi-stream numbers.

Against our earlier Python-engine build of the same checkpoint (measured the same way on another Spark: prose 54.4, 81.3, 88.5, 101.6 and 123.1 tok/s aggregate
at 1 to 5 requests; code 84.9, 128.3, 142.3, 159.5 and 181.9; prefill 2,315 tok/s at 4k down to 1,889 at 128k), prefill is now ahead at every size.
Decode is behind it at one and two streams, level at three and ahead at four and five on prose, and 15 to 21% behind on code, because the CUDA path of
this engine drafts with the MTP head only (no copy drafter for code).

### Measured with `tools/bench_streams.py`

Measured on a DGX Spark (GB10, 121 GiB, 115 GiB available) with the NVIDIA FP4 checkpoint, the Zig engine built from `patches/`,
`./start.sh` defaults (262,144-token window, 5 requests, fp8 KV, vision on, MTP drafts), greedy, thinking off, 256 tokens a
request, median of 3. The "Total" column is the sum of the per-request decode speeds the server reports. The Python recipe
numbers are our earlier Python build of the same checkpoint (patches 0003-0006), not the INT4-AutoRound recipe, measured on the same machine.

**Decode, prose** (Zig measured on the guest Spark with `tools/bench_streams.py`; the Python column is our earlier Python build of the same checkpoint, measured on the same machine with the same tool)

| Concurrent requests | Zig total | Zig per request | Zig TTFT | Python total | Python per request |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 49.8 tok/s | 49.8 | 68 ms | 49.5 | 49.5 |
| 2 | 93.2 tok/s | 46.6 | 101 ms | 93.3 | 46.7 |
| 3 | 131.7 tok/s | 43.9 | 130 ms | 123.5 | 41.2 |
| 4 | 166.8 tok/s | 41.7 | 163 ms | 155.2 | 38.8 |
| 5 | 198.2 tok/s | 39.6 | 210 ms | 180.0 | 36.0 |

**Decode, code**

| Concurrent requests | Zig total | Zig per request | Zig TTFT | Python total | Python per request |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50.9 tok/s | 50.9 | 78 ms | 57.1 | 57.1 |
| 2 | 93.5 tok/s | 46.7 | 100 ms | 108.4 | 54.2 |
| 3 | 135.5 tok/s | 45.2 | 135 ms | 145.0 | 48.3 |
| 4 | 173.1 tok/s | 43.3 | 164 ms | 184.4 | 46.1 |
| 5 | 205.5 tok/s | 41.1 | 196 ms | 215.0 | 43.0 |

Two full runs of the decode tables on a quiet machine agreed to within 1 tok/s except one prose point (5 streams: 186.8 and 187.2).
(Thinking is on in these runs, so the "code" prompt, "Write a Python quicksort with docstring and tests.", is mostly
reasoning prose at 256 tokens, which is why the Zig code and prose speeds are close.) The Zig engine matches the Python build
on prose at one and two streams and is ahead from three streams on (+7% at 3, +4% at 5); on code it is behind at every count
(50.9 against 57.1 tok/s at one stream, 205.5 against 215.0 at five). The Python build has copy drafting (draft tokens copied
from earlier text), which helps most on code; the CUDA path of the Zig engine has no such drafter, only the MTP head.
The Zig TTFT column is a repeat of the same short prompt (the benchmark's own): the server keeps a state one token short of a
prompt it has served before, so a repeat prefills one row (about 50 ms of GPU time, the cost of any decode round). That repeat-prompt
cache is why the TTFT is so low; a cold prompt takes longer: a first-time 63-token prompt costs about 170 ms (150 ms in the
prompt pass: with 64 rows nearly every routed expert is read), and a long one costs its prefill (table below).

**Prefill** (one request, fresh prose each run so no prompt-cache hit, median of 3, tools/bench_streams.py)

| Prompt tokens | Zig prefill | Zig TTFT | Python prefill | Python TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 4.0k | 2,400 tok/s | 1.67 s | 2,315 (4k) | 1.96 s |
| 7.9k | 2,428 tok/s | 3.28 s | 2,290 (8k) | 3.77 s |
| 15.8k | 2,468 tok/s | 6.42 s | 2,273 (16k) | 7.40 s |
| 31.5k | 2,450 tok/s | 12.91 s | 2,199 (32k) | 15.09 s |
| 62.9k | 2,400 tok/s | 26.31 s | 2,049 (64k) | 32.18 s |
| 125.8k | 2,264 tok/s | 55.78 s | 1,889 (128k) | 69.59 s |

(`tools/bench_streams.py` sizes its prompts in generated words, about 0.8 tokens each, so its default `--sizes 5000,...,160000` give these rows.
Prefill is ahead of the Python build at every size (before patch 0014 it was slightly behind up to 16k).
Needle in a haystack passes at 32k (prefill 16 s) and at 194,723 tokens (prefill 99.4 s, 1,959 tok/s). A single decode stream varies by
a few tok/s from run to run, and a run taken while another job shared the machine read 34.8 tok/s.

**Memory and mixed load** (default `./start.sh`, no `MEM_CHECK` override, guest Spark with about 116 GiB available before the start). The
memory check passed (the window needed about 108 GiB at start). The engine loaded in 61 s and reported 24.36 GiB of sequence memory (the
guest runs other containers, so this is less than the 30 GiB above). System memory in use: 85 GiB idle, 89 GiB after the 195k prompt, and a peak of
98.6 GiB with five 100k-token prompts and six short ones in flight together. All five long prompts were answered correctly (each finds
its own passphrase) and all short ones were right, with no crash and no hang. By default the prefill is serial: five 100k prompts took about 250 s in all (265 s before patch 0014), and a short
request sent while they ran waited for the prefills to finish (185 to 260 s) before it was answered. `TF_FLASHNEXT_FILL_FAIR=1` changes that (next paragraph).

**Mixed load: short requests behind long prompts.** Opt in with `TF_FLASHNEXT_FILL_FAIR=1` (patch 0015, one rank). The engine then fills prompts in
slices of two layers, hands the prompt buffers on at chunk boundaries, shortest prompt first (a long one ages by 1,024 rows per boundary it loses), and
decoders keep running between slices. Five 100k-token prompts, one running decoder and a short request every 20 s (`--parallel 8`): a short request waited
1.8 s at the median and 3.9 s at the worst (187 s and 256 s without it), and the running decoder's longest gap between tokens fell from 12 to 15 s per
long prompt (stalls of up to 218 s in the stress) to 0.27 s. The price: the long prompts finish later, the first at 111 s instead of 53 s, and a mixed load
takes longer in total (about 22% in our run, 280 s to 342 s, before the tile; the same load measured with the tile took 311 s). With nothing else running,
prefill speed is unchanged. Replies are bit-identical to the default (7-prompt test, drafted equal to serial). With the default `PARALLEL` of 5, five long
prompts and a decoder fill all five slots, so a short request still waits for a slot. Off by default; two ranks are not supported by it.

**Quality.** Teacher forcing the Python build's greedy output of seven prompts through the Zig engine, its next token equals
the Python token at 2,270 of 2,295 positions (98.9%); the disagreements are near-ties between two plausible words
(`tools/teacher_force.py` is not part of this recipe; see the lab notes). `tools/toolcheck.py` and `tools/visioncheck.py` pass.
Greedy free-running output differs from the Python build after 85 to 350 characters in six of seven prompts for that reason.

### Optional INT4-AutoRound checkpoint (`CHECKPOINT=int4`)

These are the numbers the recipe published for the INT4-AutoRound checkpoint (azampatti), measured with the Zig engine before the NVIDIA FP4
support (patches 0011 to 0013) was added. They are for `CHECKPOINT=int4`, not for the default checkpoint above. The "Python recipe" they
mention is the older recipe on the MLX 4-bit checkpoint.

Measured with sparkDash on spark4, 2026-10-08, engine `20e709a` (image `tensorfold-qwen38:zig-db28187`, patches `d84d7cc22655`): FP8 KV, vision on, `PARALLEL` 8, a 262,144-token window. Greedy, thinking off. Aggregate is every stream's tokens over the time from the first token of the earliest stream to the last token of the latest. The FP8 n-gram table is on the GPU, and MTP drafts are checked against the model's own sample.

**Decode, prose** (sparkDash)

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 64.4 tok/s | 64.4 tok/s | 98 ms |
| 2 | 89.0 tok/s | 46.4 tok/s | 180 ms |
| 4 | 140.7 tok/s | 37.2 tok/s | 216 ms |
| 8 | 200.9 tok/s | 26.7 tok/s | 405 ms |

Against the Python recipe's prose table (62.4 / 90.5 / 106.7 / 119.3 at 1 / 2 / 4 / 5), one request is 64.4 against 62.4 and four are 140.7 against 106.7. Two requests are 89.0 against 90.5. Eight requests are 200.9; the Python recipe's five-request row was 119.3.

**Prefill** (sparkDash, one request)

| Prompt | Tokens | Prefill speed | Time to first token |
| ---: | ---: | ---: | ---: |
| 4k | 4,134 | 2,526 tok/s | 1.64 s |
| 8k | 8,232 | 2,606 tok/s | 3.16 s |
| 16k | 16,422 | 2,643 tok/s | 6.21 s |
| 32k | 32,804 | 2,630 tok/s | 12.47 s |
| 64k | 65,578 | 2,564 tok/s | 25.57 s |
| 128k | 131,111 | 2,415 tok/s | 54.30 s |

Against the Python recipe's prefill (2,503 / 2,520 / 2,499 / 2,414 / 2,200 at 8k / 16k / 32k / 64k / 128k), this boot is ahead at each of those sizes.

A 199,730-token needle on this boot was found (prefill 100.2 s). One 320×240 image took 0.22 s to the first token (99 prompt tokens), ten of them 0.65 s (836 tokens), and a 30-second 320×240 video 2.02 s (2,678 tokens).

**Decode, code** (lab bench on the same boot: a short Python function, 256 tokens, end-of-sequence ignored)

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 57.5 tok/s | 57.5 tok/s | 113 ms |
| 2 | 92.7 tok/s | 73.9 tok/s | 166 ms |
| 4 | 130.8 tok/s | 49.6 tok/s | 194 ms |
| 8 | 189.3 tok/s | 40.1 tok/s | 379 ms |

The tables below are the earlier bf16, text-only boot.
The previous Python recipe on the MLX 4-bit checkpoint (int8 KV, n-gram tables read from SSD) measured prose decode at
62.4 / 90.5 / 106.7 / 119.3 tok/s for 1 / 2 / 4 / 5 requests, and prefill at 2,503 / 2,520 / 2,499 / 2,414 / 2,200
tok/s for 8k / 16k / 32k / 64k / 128k.

**Decode, prose** (sparkDash, 2026-10-07, spark4, engine `6eb39c1`: greedy, thinking off, the hash-map prompt). That boot reported 33.28 GiB of sequence memory.

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 63.3 tok/s | 63.3 tok/s | 101 ms |
| 2 | 95.4 tok/s | 49.0 tok/s | 162 ms |
| 3 | 118.5 tok/s | 43.2 tok/s | 227 ms |
| 4 | 148.6 tok/s | 39.7 tok/s | 275 ms |

Against the Python recipe's prose table, one request is 63.3 against 62.4, two are 95.4 against 90.5, and four are 148.6 against 106.7. The Python recipe's five-request row was 119.3 tok/s; this run stopped at four, the number of full windows the cache budget holds.

**Prefill** (same boot, one request, greedy, thinking off)

| Prompt | Tokens | Prefill speed | Time to first token |
| ---: | ---: | ---: | ---: |
| ~8k | 6,574 | 768 tok/s | 8.57 s |
| ~16k | 13,107 | 1,300 tok/s | 10.09 s |
| ~32k | 26,209 | 1,678 tok/s | 15.62 s |
| ~64k | 52,386 | 1,971 tok/s | 26.57 s |
| ~128k | 104,775 | 1,991 tok/s | 52.63 s |

The 6,574-token row is the first long prefill after load. The later rows are 1,300 to 1,991 tok/s, under the Python recipe's 2,200-2,520. A 199,730-token needle on the same boot prefilled in 123.1 s (1,623 tok/s) and the passphrase was found.

## Requirements

- A DGX Spark (or another GB10 system with 128 GB unified memory) with nothing else large on the GPU. The default
  window needs `MemAvailable` of about 108 GiB at start with the NVIDIA checkpoint (loading takes ~81 GiB) and 93 GiB with
  `CHECKPOINT=int4` (weights ~66 GiB): weights, one full cache, scratch slack and a floor.
  `start.sh` stops before launching below that (`MEM_NEED_GIB`).
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~160 GB free disk on a fresh machine: ~96 GiB (NVIDIA FP4) or ~122 GiB (INT4-AutoRound) for the checkpoint under `~/.cache/huggingface` and the image under
  Docker's root. `scripts/prepare.sh` checks both. Keep at least 100 GB free.
- Optional: the `hf` CLI on the host and a Hugging Face token in `~/.cache/huggingface/token` or `HF_TOKEN`. The
  checkpoint is public.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
./start.sh
```

The first run builds the engine (TensorFold at the pinned commit, plus `patches/`, then the TP=1 kernel set) and
downloads the checkpoint if it is not already in the cache. Later starts load the weights. `start.sh` shows each
step, runs a smoke test, prints `Qwen3.8-Flash-Next is now LIVE! on port 8888`, and returns you to the shell.

```bash
curl -s http://<spark-address>:8888/v1/models

curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

Any OpenAI client works with `base_url = "http://<spark-address>:8888/v1"` and the model `Qwen3.8-Flash-Next`.
Streaming, tool calls and reasoning content are supported. The model thinks before it answers, so give replies
enough `max_tokens`. `"draft": false` on a request serves one token at a time; a drafted reply matches that.

```bash
./start.sh restart
./stop.sh
docker logs -f qwen38-flash-next-tf
curl -s http://<spark-address>:8888/health
```

## Choosing the checkpoint (FP4 or INT4)

`./start.sh` serves NVIDIA's FP4 QAD checkpoint by default. To run the optional INT4-AutoRound checkpoint
([`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound))
instead, set `CHECKPOINT=int4`:

```bash
CHECKPOINT=int4 ./start.sh restart       # stops the running server, then starts the INT4 one
```

The first time, `start.sh` runs `scripts/prepare.sh`, which downloads the INT4 checkpoint (~122 GiB) and checks it; later starts just load it.
Nothing else needs to change: the image and the patches are the same for both checkpoints, and the engine reads the format from the checkpoint.

To keep INT4 as your default, put it in `.env` next to `start.sh` (the environment still wins over it):

```bash
echo 'CHECKPOINT=int4' >> .env
./start.sh restart
```

To go back to FP4, remove that line (or run `CHECKPOINT=nvfp4 ./start.sh restart`). Both checkpoints can sit in `~/.cache/huggingface` side by side
(~96 GiB + ~122 GiB); switching only restarts the server, with no new download once both are cached.

| | FP4 (`nvfp4`, default) | INT4 (`int4`) |
| --- | --- | --- |
| Model | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | [`azampatti/…-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound) |
| Experts per token | 10 of 512 (all experts kept) | 5 (top-5 cut, 4.8B active parameters) |
| Quality | quantization-aware distillation, close to bf16 (see the top of this page) | about 10% lower general capability on its authors' harness |
| Download / weights in memory | ~96 GiB / ~81 GiB | ~122 GiB / ~66 GiB |
| `MemAvailable` needed at start | ~108 GiB | ~93 GiB |
| Default requests at once (`PARALLEL`) | 5 | 5 (the INT4 numbers below were measured with 8: `PARALLEL=8 CHECKPOINT=int4 ./start.sh restart`) |
| Speed | the tables above | the [INT4 tables](#optional-int4-autoround-checkpoint-checkpointint4) below (faster on this hardware) |

Check which checkpoint is running with `docker inspect -f '{{.Config.Cmd}}' qwen38-flash-next-tf`: the path after `serve` names it. `DRY_RUN=1 CHECKPOINT=int4 ./start.sh`
prints what a start would do (the model, the memory it needs and the docker command) without stopping or starting anything. To serve a different model directory of the same
format, set `MODEL_ID` (and `MODEL_REVISION`, or leave it empty for the latest) instead of `CHECKPOINT`. A `MODEL_ID` set by hand takes no preset revision pin.

## Images and video

The model's own vision tower (27 layers, 0.84 GiB, from the same checkpoint) turns images and video frames into
tokens. A Python helper in the image decodes them (Pillow, PyAV) and runs the tower. Send them as OpenAI-style
content parts in a user message:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 2000
}'
```

A video is a `video_url` part (`{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`).

| | Images | Videos |
| --- | --- | --- |
| Formats | JPEG, PNG, WebP | MP4, WebM, MOV, MKV (anything FFmpeg decodes) |
| Per request | up to 50 (all of a chat's turns count), 10 MB each, 64 MB in all | up to 4 (`MAX_VIDEOS`), 64 MB each, 96 MB in all, up to an hour of footage |
| Tokens | up to 16,384 for all images, at most 4,096 an image (`"detail": "low"`: 256 an image) | 2 frames a second (at most 256 frames), up to 16,384 tokens a request (`TENSORFOLD_VIDEO_TOKENS`) |

By default only data URLs are accepted. `VISION_URLS=1` also lets the server fetch public `https://` URLs. Image prompts are kept for prefix reuse
(each image matched by a hash of its bytes; `TF_FLASHNEXT_MEDIA_REUSE=0` turns that off), so a chat's next turn after an image
prefills only its new rows, but the vision tower still runs for every image each turn. Video prompts are not kept. A request body
can be up to 96 MiB. `VISION=0 ./start.sh restart` serves text only and gives the 2 GiB helper workspace back to the
KV pool.

Measured on the Zig engine at TP=1 with `--vision` (warm, three reps): one image (1,217 tokens) 1.0 s to first token,
10 images (3,035 tokens) 2.3 s, a 30-second video (6,875 tokens) 5.2–5.6 s. The first use of a new size is slower (a
5,107-token image took 35 s). The helper stays at 0.84 GiB for the tower and peaked around 1–1.7 GiB. Text replies
with vision on matched the text-only engine. Those timings are from that engine build; this recipe's own vision boot
has not been taken while the text server is in use.

## What `start.sh` and `scripts/prepare.sh` do

1. **Setup:** `scripts/prepare.sh` when the image or the checkpoint is not the one these settings name.
2. **Checks:** the arguments (tensorfold-native's own parser, in a throwaway container), the port, and free memory.
   A typo leaves a running server alone. `./start.sh restart` stops only after those checks pass.
3. **Launch:** `tensorfold-native serve` with the settings from `scripts/config.sh`.
4. **Loading:** the log, and a stop if `MemAvailable` falls below `MEM_FLOOR_GIB` (10 GiB).
5. **Smoke test**, then the LIVE message.

`FOREGROUND=1 ./start.sh` stays attached and exits with the container's exit code. Extra arguments go to
`tensorfold-native serve` after the defaults (`./start.sh restart --context 131072`). `DRY_RUN=1 ./start.sh` prints
the docker command and changes nothing.

**`scripts/prepare.sh`:**

1. Preflight: Docker, disk.
2. The image `tensorfold-qwen38:zig-db28187`: TensorFold at `TF_REF` with every `patches/*.patch` applied, Zig 0.17.0,
   on `nvcr.io/nvidia/pytorch:26.07-py3`, and the TP=1 Triton kernel set (cubins pinned with `KERNEL_SOURCE_MTIME`).
   It tries the matching prebuilt image from GitHub Container Registry first (`PULL=0` builds locally).
3. The checkpoint, at the pinned revision.
4. A check that it is a `qwen4_exp` ModelOpt NVFP4 checkpoint (or the GPTQ int4 one with its `ple-table/` n-gram files) and
   every shard present.

```bash
scripts/prepare.sh
scripts/prepare.sh --rebuild
PREPARE=1 ./start.sh restart
```

## KV and memory

KV is FP8 by default (`KV_DTYPE=fp8`). It is lossy: about 98.8% top-1 agreement with a bf16 cache, so a free-running
reply can differ from the bf16 one. `KV_DTYPE=bf16` is the exact cache. FP8 works together with image and video input.
The format is adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8`.

One token's cache at TP=1, 13 layers (12 attention plus the MTP head):

| | bytes a token | one 262,144-token sequence |
| --- | ---: | ---: |
| bf16 | 30,784 | 7.52 GiB |
| fp8 | 17,888 | 4.37 GiB |

FP8 stores keys with both scales and values as codes. The indexer key (256 bytes) and pooled blocks (64 bytes) stay
bf16. A 1,048,576-token fp8 sequence is 17.46 GiB; the same window in bf16 is 30.06 GiB.

**Total available KV cache with the default checkpoint (NVIDIA FP4): 24.36 GiB, about 1.46 million fp8 tokens.** That is the shared sequence
pool the engine reported on a Spark with about 116 GiB of memory available at start, vision on, `PARALLEL=5` (the server log line `sequence memory`).
Loading the weights takes about 81 GiB of that memory (the 26.8 GiB NVFP4 n-gram table is read from disk by the host, in the page cache, not held on the
GPU). The engine's sequence budget is `MemAvailable` after the weights load, minus `TENSORFOLD_MEMORY_RESERVE_GIB` (10 GiB), and with `VISION=1` minus 2 GiB kept for
the helper (`TENSORFOLD_VISION_WORKSPACE_MIB`; the tower peaks around 0.9–1.7 GiB). A full fp8 window is 4.37 GiB, so **five** streams can sit at 262,144
tokens at once (21.9 GiB) and a sixth (26.2 GiB) does not fit; shorter requests share the pool. The pool is shared and streams grow as they go. A
request is admitted when its prompt plus its reply budget (`max_tokens`) fits beside the others; one that does not fit is refused, not queued. The pool
follows the free memory at start (other containers on the machine shrink it), and each GiB less reserve adds a GiB to it. `VISION=0` returns the 2 GiB helper
workspace to the pool. Under a mixed load of five 100k-token prompts and six short ones, system memory in use peaked at 98.6 GiB.

With `CHECKPOINT=int4` the weights, including the FP8 n-gram table on the GPU, measured 65.7 GiB, and the engine reported **30.22 GiB** of sequence memory
on a Spark with more memory free (six full fp8 windows; a seventh did not fit). A pool probe with the helper not resident held 36.22 GiB and fit eight full windows.

A 1,048,576-token window (`CONTEXT=1048576`, YaRN factor 4) is one 17.46 GiB fp8 cache on top of the weights. A prompt
that fills that window also needs prefill scratch, which the engine's budget does not reserve. It is not the default.
`start.sh` allows it when `MemAvailable` covers weights, one cache, a small scratch slack and the floor.

`prepare.sh` and `start.sh` refuse to run when `MemAvailable` is under 10 GiB (`MEM_FLOOR_GIB`). Keep at least 10 GiB
free under load. On this machine, exhausting unified memory freezes it.

### Kernel `NV_ERR_NO_MEMORY` messages under load (issue #33)

Under a long mixed load, stock 1.0.0 can log `NV_ERR_NO_MEMORY ... _memdescAllocInternal` in the kernel log while
`MemAvailable` stays above 30 GiB; no request failed in the runs below. On a GB10 (driver 580.173.02, kernel
6.17.0-1029-nvidia), with 8 clients for 45 minutes, a ~62k-token prompt every fifth request and images every
seventh, stock logged 103 and 90 in two runs (139 in one run on 6.17.0-1031-nvidia). A trace put 108 of 115 inside
the `cuMemCreate` calls that grow a kept ~62k-token state (103 of them 28 MiB, 5 of 4 or 6 MiB).

A stock driver probe on 1031 recorded 12 failed system-memory allocations requesting 64 KiB pages; each 64 KiB block
must be physically contiguous. NVIDIA's 580.173.02 source logs the failure (`memdescAlloc`) before retrying eligible
default-page allocations with 4 KiB pages (`sysmemConstruct`). The aggregate counts fit that retry, but the probe
does not pair each buffer with its retry. All 105 load requests succeeded. This does not identify the patched runs'
remaining warnings or the Linux allocator's reason for each failure.

`patches/0010-system-memory-cache.patch` is an opt-in mitigation. Its allocation changes are off unless set (the
patch also fixes partial-growth accounting and partial ring cleanup, which apply either way):

- `TF_FLASHNEXT_SYSTEM_CACHE=1` backs the K/V caches, fixed sequence state and snapshots with locked, ordinary
  system memory registered with CUDA (`cuMemHostRegister`), instead of native CUDA backing allocations for those
  buffers. It needs Linux, one GPU (TP=1), VMM and an integrated GPU with pageable memory access, and refuses to start
  otherwise. `start.sh` already runs the container with `--ulimit memlock=-1`. CUDA registration still needs driver
  metadata and mappings. Weights, scratch and ring-cache misses still use native allocations.
- `TF_FLASHNEXT_RING_CACHE=1` (with the above) also reuses whole retired indexer-ring regions, capped at 64 MiB
  physical, 32 entries, 1 GiB of addresses and 512 mappings of idle retained regions per engine.

From a checkout containing patch 0010, build locally and restart the container (a published image with this patch
is not established by these tests). Remove `TF_FLASHNEXT_GRAPH_LOG` from the environment and `.env` entirely: setting
it to 0 turns diagnostics on.

```bash
./scripts/prepare.sh --rebuild
TF_FLASHNEXT_SYSTEM_CACHE=1 TF_FLASHNEXT_RING_CACHE=1 ./start.sh restart
```

Check the running container's image and both flags, and that `TF_FLASHNEXT_GRAPH_LOG` is absent. To turn the
allocation options off, restart with both flags set to 0; the accounting and cleanup fixes remain. A full rollback
needs the previous recipe and image.

Same reproduction load on kernel 1029, two runs per build with 45 minutes of admission each. Request counts are
totals across both runs; throughput includes drain time.

| build | messages | total requests / errors | tok/s | p50 / p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| stock | 103, 90 | 580 / 0 | 62.3, 62.9 | 80.7 / 99.4, 80.7 / 97.9 |
| `SYSTEM_CACHE=1` | 0, 10 | 578 / 0 | 62.6, 62.4 | 81.6 / 95.3, 81.1 / 100.0 |
| `SYSTEM_CACHE=1 RING_CACHE=1` * | 0, 2 | 580 / 0 | 62.5, 62.3 | 82.9 / 91.7, 81.1 / 94.6 |

\* These two runs also had `TF_FLASHNEXT_GRAPH_LOG=0` set, which turns graph diagnostics *on* (the engine checks
that the variable exists) and adds a synchronization, so they don't isolate the effect of ring reuse on warnings or
speed. On 6.17.0-1031-nvidia, without it, one run each: stock 139 messages, 62.8 tok/s, 80.7 / 96.9 s; with both
settings 0 messages, 61.8 tok/s, 82.5 / 100.4 s.

The prompt-cache budget policy and the observed peak (8,370 MiB, eight states) are unchanged; the budget itself is
computed from free memory at startup and varied slightly between runs. Greedy replies matched stock on seven
cases, including two ~62k passphrases, prompt-cache repeat and extension, eviction and images. Two runs per build
show no large speed change, not equality. The allocation behind the few messages left with the patch was not traced.
Untested: other GPUs, other context sizes, long production soaks; TP=2 is refused. The test load completed fewer
requests than in the issue (290 vs 986) and had no prompt reuse. `--prompt-cache-gib 2` without the patch gave 38
in one run.

These results cover the 1.0.0 recipe: TensorFold db281878 plus patches 0001–0009 and the allocator patch. The
tested image replaced the native binary, built with Zig 0.17.0, and reused the control GPU kernels; the normal recipe
build and a 1.0.7 rebase have not been validated by these tests. The patched residual warnings remain unidentified.
Longer soaks and the reporter's workload with prompt reuse remain open.

Without the patch, the driver setting `NVreg_RegistryDwords="RmEnableLargePageSizeSysmemDefault=0"` (nvidia module
option, reboot) makes default system-memory allocations start at 4 KiB. In one 45-minute run on 1031 it gave 0
messages, but 58.6 instead of 62.8 tok/s (p50 86.4 vs 80.7 s).

## Configuration

An experimental Dutch MTP draft vocabulary (5,176 additional token IDs) is provided in
[docs/nl.md](docs/nl.md). Its Zig runtime loading depends on [PR #79](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/pull/79); the default configuration is unchanged.

Every setting lives in [`scripts/config.sh`](scripts/config.sh). Override it from the environment
(`PARALLEL=2 ./start.sh`) or from `.env` next to `start.sh` (`KEY=value`; the environment wins).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `5` | requests decoded together. One that would not fit (prompt plus `max_tokens`) is refused |
| `CONTEXT` | `262144` | prompt + reply window. `1048576` turns on YaRN factor 4 |
| `DRAFTS` | `1` | MTP drafts; `0` passes `--no-drafts` |
| `TF_FLASHNEXT_DEPTH` | `6` | drafts a round. The stop rule is the engine's hybrid (see `TF_FLASHNEXT_PRODUCT_STREAMS`) |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | default sampling; a request's own values win |
| `THINKING` | `1` | think before answering; `0` passes `--no-thinking` |
| `MAX_TOKENS` | `32768` | the reply budget when a request sets none |
| `VISION` / `VISION_URLS` | `1` / `0` | image and video (`--vision`); `VISION_URLS=1` also accepts public `https://` URLs |
| `TENSORFOLD_MAX_IMAGES` / `TENSORFOLD_IMAGE_TOKENS` | `50` / `16384` | images a request may carry, and the tokens they share (4,096 at most an image) |
| `MAX_VIDEOS` | `4` | videos a request (`--vision-max-videos`) |
| `TENSORFOLD_VIDEO_TOKENS` | `16384` | a request's video token budget |
| `TENSORFOLD_VISION_WORKSPACE_MIB` | `2048` | kept out of the KV budget for the helper |
| `SERVED_NAME` | `Qwen3.8-Flash-Next` | the model id in `/v1/models` |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `KV_DTYPE` | `fp8` | lossy (~98.8% top-1 agreement with bf16, about 1.8x the pool). `bf16` is exact. Works with `VISION=1`. Adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `10` | kept free when the engine sizes the pool. `prepare.sh` and `start.sh` refuse a Spark below this |
| `TF_FLASHNEXT_PRODUCT_STREAMS` | `5` | running-product draft stop at this many streams or fewer; confidence 0.5 above it |
| `TF_FLASHNEXT_REPEAT_CACHE` | `1` | a prompt the server has served before keeps a state one token short of its end, so a repeat prefills one row (63-token prompt: TTFT 170 ms to 70 ms). `0` turns it off |
| `TF_FLASHNEXT_ADMIT_WAIT_MS` | `3` | an idle server waits this long after a first request for others sent with it, so a burst prefills in one pass and decodes in step |
| `TF_FLASHNEXT_PREFILL_TAIL` | `512` | a short last prompt chunk joins the previous one |
| `TF_FLASHNEXT_PROMPT_EXPERT_TILE` | `2` | routed experts of prompt chunks of 1,024 rows or more run through a shared-memory tile kernel (patch 0014): `2` gate/up and down, `1` gate/up only, `0` the older kernels. Same output bits, cold prefill +13 to 15%. One rank only (two ranks keep the older kernels unless it is set) |
| `TF_FLASHNEXT_FILL_FAIR` | `0` | `1` fills prompts in slices, shortest first, so short requests and running decoders are not stalled behind long prompts (patch 0015): a short request behind five 100k prompts waits about 2 s instead of 3 to 4 minutes; a mixed load takes about 20% longer in total. `TF_FLASHNEXT_FILL_LAYERS` (slice, default 2) and `TF_FLASHNEXT_FILL_AGING` (rows, default 1024) tune it. One rank only |
| `TF_FLASHNEXT_ADMIT_RECLAIM` | `1` | a prompt that does not fit frees kept prompt states first (a superseded one, else the least recently used; never the state the prompt resumes), and only when that makes it fit; `0` refuses it with `NoRoom` as before (patch 0029). One rank only |
| `TF_FLASHNEXT_SUPERSEDE` | `1` | a conversation's next turn frees the kept state it resumed once it keeps a longer one, so the cache holds about one state a conversation instead of one a turn (patch 0029). `0` turns it off. One rank only |
| `TF_FLASHNEXT_MEDIA_REUSE` | `1` | image prompts keep and resume prompt states like text ones, each image told apart by a hash of its bytes, so a conversation's next turn after a screenshot prefills only its new rows (patch 0030). Videos are not kept. `0` restores the old rule (an image prompt prefills from its start). One rank only |
| `CHECKPOINT` | `nvfp4` | `nvfp4` serves `nvidia/Qwen3.8-Flash-Next-FP4-Local`; `int4` serves azampatti's INT4-AutoRound (sets its model id and revision). `MODEL_ID` / `MODEL_REVISION` override either |
| `MODEL_REVISION` | `b9ce18a6…` | the checkpoint commit this recipe serves (`14642741…` with `CHECKPOINT=int4`) |

The Python recipe's int8 KV is gone: KV is fp8 (or bf16). The NVIDIA checkpoint's NVFP4 n-gram table (26.8 GiB) is read from disk by the host
(`TF_NGRAM_READ=pread` by default, with read-ahead of the next prompt chunk; `TF_FLASHNEXT_WARM_AHEAD=0` turns that off). The INT4 checkpoint's FP8 table is on the GPU.

### Thinking and sampling

By default the model thinks first. Per request, `temperature`, `top_p`, `top_k` and `seed` override the defaults
(`temperature: 0` is greedy). `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking, and
`"chat_template_kwargs": {"reasoning_effort": "low"}` (or `"medium"`, `"xhigh"`) sets reasoning effort. The reasoning
comes back in `reasoning_content`, the answer in `content`. A small `max_tokens` can return empty `content` because
the model was still thinking.

- **A reply that spends its whole `max_tokens` thinking:** `thinking_budget` (a request field; `--thinking-budget N` as a server default, for example
  `./start.sh restart --thinking-budget 4096`) closes the think block after that many reply tokens and lets the answer start
  (`zig/src/core/lanes/stream.zig`, `think_budget`; the closing is done in the engine's lanes, so drafted and serial replies cut at the same token).
  `reasoning_effort: low` and `enable_thinking: false` are the other levers.
- **Without a `seed`, the same prompt gives the same sampled reply.** The engine derives the seed from the prompt's token ids (`sampling.seedFor`), so a
  repeated request at temperature 1.0 is repeated, which also makes `"draft": false` comparisons reproducible. Clients that want a different reply per call
  (best-of-n, "regenerate", retries) send their own random `seed`, or start the server with `TENSORFOLD_RANDOM_SEED=1 ./start.sh restart` (`patches/0031-server-random-seed.patch`): a sampled
  request without a `seed` then draws a fresh one, as OpenAI's API does (an explicit `seed` still wins; off by default so the `exact.py` and `"draft": false`
  comparisons stay reproducible). `TENSORFOLD_SEED_SALT=<integer>` changes every derived seed at once.
- **`presence_penalty`, `frequency_penalty` and `repetition_penalty` are accepted and have no effect** (the engine does not implement them), and out-of-range
  sampling values such as `temperature: -1` are not rejected. Use `top_p`, `top_k`, `min_p` and `temperature` to steer repetition.
- **A control token in the reasoning ends the turn.** The engine stops on the checkpoint's end tokens (`<|endoftext|>` 248044 and `<|im_end|>` 248046 in its
  `eos_token_id`; the report saw `<|im_start|>` 248045 stop a reply too). If the model writes one while it is still thinking, for example when it quotes it in its analysis, the reply stops right there
  with `finish_reason: "stop"`, empty `content` and a cut `reasoning_content` ([#76](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/issues/76)).
  A client can treat a `stop` with empty `content` as a failed turn and retry with `enable_thinking: false`.
- **`tool_choice: "required"` with thinking off** gives the model no room to reason before the call: the engine replaces the first answer token that
  does not open a call with the call opener (`zig/src/core/lanes/call_gate.zig`), and the function is then picked from that forced start. With many
  tools, the pick can be less relevant than under `auto`. Keep thinking on (the gate waits until the think block has closed), use `auto` with an
  instruction to call a tool, or name the function in `tool_choice`.

### API notes

- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (Anthropic) and
  `/v1/messages/count_tokens`, `/v1/models`, `/tokenize` and `/detokenize` (also under `/v1/`), `/health`, `/stats`
  and Prometheus `/metrics`.
- Refused with HTTP 400: `logprobs: true` / `top_logprobs` (no token probabilities) and `n` other than 1.
- `"draft": false` serves the request without drafts, the serial reference.
- Tool calls work, including `tool_choice: "required"` and a named function (Anthropic `tool_choice: {"type": "any"}`
  too): the reply then always opens a tool call.
- A conversation's next turn resumes from the kept state of its earlier turns, with images too (patch 0030: each
  image is matched by a hash of its bytes, so a changed image resumes only up to its first row). Video prompts are not
  kept.
- Structured outputs (`response_format`, the `guided_*` fields) are refused with HTTP 400: this engine does not
  enforce them.
- When the KV pool is full, a new request is refused. Admission reserves the prompt plus `max_tokens`; the request is
  not queued. Kept prompt states share that pool: admission frees them first (patch 0029), so a request is refused
  only when it would not fit with every freeable state gone. A request whose prompt plus `max_tokens` equals `--context` is admitted: the draft rows sit past that
  window. That admission change was authored by [321sssrt-bit](https://github.com/321sssrt-bit)
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)).

## What the patches are

`patches/*.patch` is the Zig Flash Next CUDA engine against TensorFold `db281878` (the `zig-flashnext` pin), applied
with `git apply` in the checkout root. They are the engine, not speed patches on a Python package. Drafts are still
checked: a drafted reply matches `"draft": false`. `0011` to `0015`, `0021`, `0029`, `0030` and `0031` are this recipe's own: `0011` lets the vision helper load the NVIDIA
checkpoint's quantized tower, `0012` makes the Zig engine serve the NVIDIA FP4 checkpoint (config, CSF-compressed expert scales, MXFP8 linears, the
NVFP4 n-gram table with read-ahead) and `0013` tunes decoding (draft depth 6, the repeat-prompt cache, burst admission). `0014` adds a shared-memory expert tile for prompt chunks (on by default, bit-identical, cold prefill +13 to 15%) and `0015` an opt-in fair prefill for mixed loads (`TF_FLASHNEXT_FILL_FAIR=1`). `0029` frees a sequence only after
the GPU work queued on it is done (a request refused for room mid-prompt could unmap memory under queued kernels and
kill the CUDA context with `CUDA_ERROR_ILLEGAL_ADDRESS`), and lets admission free kept prompt states before it refuses
a prompt (`TF_FLASHNEXT_ADMIT_RECLAIM`, `TF_FLASHNEXT_SUPERSEDE`, both on).
`0030` lets image prompts keep and resume prompt states (`TF_FLASHNEXT_MEDIA_REUSE`, on), so the next turn after a screenshot prefills only its new rows. `0021` makes `TF_FLASHNEXT_FILL=1` fill prompts that arrive together one after another instead of failing them, and `0031` adds the opt-in `TENSORFOLD_RANDOM_SEED=1` (a fresh seed for a sampled request without one).

## Checks

`tools/` talks to the running server (`API_URL`, default `http://127.0.0.1:8888`, or `PORT`):

| Script | What it does |
| --- | --- |
| `tools/exact.py` | greedy replies: drafted, `"draft": false`, and concurrent, must match |
| `tools/toolcheck.py` | a tool call whose array argument comes back as a JSON array |
| `tools/needle.py` | a passphrase in a ~200,000-token prompt |
| `tools/prompt_reuse.py` | a resumed prompt matches a fresh one |
| `tools/bench.py` | prefill at 8k-128k and a short decode |
| `tools/bench_streams.py` | decode at 1 to 5 concurrent requests (prose and code, with TTFT) and prefill 4k to 128k, median of 3 |
| `tools/context_boundary.py` | a prompt plus reply that fills `--context` succeeds; one token over returns HTTP 400. Authored by [321sssrt-bit](https://github.com/321sssrt-bit) |

## Repository layout

```
start.sh      set up (first run) and start the server
stop.sh       stop it
scripts/      prepare.sh, config.sh, publish-image.sh, banner.sh
patches/      the Zig engine, applied onto TensorFold at TF_REF
tools/        checks
.github/      issue and pull request templates, GitHub Sponsors
CREDITS.md    who and what this builds on
CHANGELOG.md  upgrade notes
NOTICE        third-party notices
```

## License

MIT, see [`LICENSE`](LICENSE). The model weights, downloaded from Hugging Face and not part of this repository, are
under the checkpoint's own license: the default NVIDIA FP4 checkpoint under the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) (its base model, Qwen3.8 Flash Next, is under the
Qwen Community License 1.0), and the optional INT4 checkpoint under the Qwen Community License 1.0 per its model card.

**Third-party software in the image.** The image `scripts/prepare.sh` builds is based on NVIDIA's PyTorch container
`nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA software in it is
governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start. By pulling or running the image you accept them. The MIT license above
covers this repository's scripts only.

## Credits

- **Engine:** [TensorFold](https://github.com/ashhart/TensorFold) and its Zig engine (`tensorfold-native`, branch
  [`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext)) by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors). The Flash Next CUDA engine in this recipe is ported from
  TensorFold's Python Flash Next engine, written by Ash Hart and the TensorFold contributors.
- **Zig CUDA serving path and CUDA family registry:** Jürgen Schmied ([jschmied](https://github.com/jschmied)), in
  [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
  (commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac)).
- **NVIDIA FP4 support in the Zig engine** (patches 0011 to 0015): MiaAI-Lab, built on TensorFold's Python implementation of the same formats by the
  TensorFold Contributors.
- **Free after drain and admission reclaim** (patch 0029): [tempeste](https://github.com/tempeste), on TensorFold's prompt cache and lanes engine by
  Ash Hart and the TensorFold contributors.
- **Image prompts in the prompt cache** (patch 0030): [tempeste](https://github.com/tempeste), on TensorFold's prompt cache, lane host and vision
  input by Ash Hart and the TensorFold contributors.
- **Model:** [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team (Alibaba).
- **Default checkpoint:** NVIDIA's [FP4 QAD export](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (Model Optimizer).
- **Optional INT4 checkpoint** (`CHECKPOINT=int4`): [azampatti](https://huggingface.co/azampatti)'s
  [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
  (the top-5 expert cut, the shared-expert healing and the checkpoint), with its AutoRound int4 quantization by
  [Intel](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) and its hybrid checkpoint and FP8 n-gram table by
  [Saren-Arterius](https://github.com/Saren-Arterius) ([`Saren/Qwen3.8-Flash-Next-ple-table-fp8`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8)).
- **Previous checkpoint** (before the Zig engine): [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP)
  by [Vontra](https://huggingface.co/Vontra).
- **Image and video input:** adapted from MiaAI-Lab's single-Spark patches 0008 and 0009 and from TensorFold's vision code by Ash Hart and the
  TensorFold contributors. The helper uses Hugging Face [transformers](https://github.com/huggingface/transformers) (Apache 2.0) for the tower modules and the
  image processor, [PyAV](https://github.com/PyAV-Org/PyAV) and [FFmpeg](https://ffmpeg.org/) for video decoding, and
  [Pillow](https://python-pillow.org/) for image decoding.
- **FP8 KV format** (`KV_DTYPE=fp8`): adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` in
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
- **Full-window admission fix and `tools/context_boundary.py`:** [321sssrt-bit](https://github.com/321sssrt-bit)
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)): a request whose prompt plus `max_tokens` equals the window is
  admitted, because the engine already keeps the draft rows beyond that window.
- **Prompt-chunk change** on the earlier Python engine: [MovieMaker93](https://github.com/MovieMaker93)
  ([TensorFold #40](https://github.com/ashhart/TensorFold/pull/40)).
- **Language draft vocabularies** of the earlier Python recipe: Javier ([jvr0x](https://github.com/jvr0x))
  ([#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)); that draft-language image is not built anymore.
- **RoCE all-gather:** the one-shot all-gather in the engine implements the RoCEnante protocol of [b12x](https://github.com/local-inference-lab/b12x) by
  local-inference-lab; this recipe leaves it off.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and the Zig contributors; the image is based on NVIDIA's
  [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
- **Benchmarks:** the performance numbers were measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) by MiaAI-Lab.

The full list is in [`CREDITS.md`](CREDITS.md).
