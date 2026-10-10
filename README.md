<h1 align="center">Qwen3.8 Flash Next on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve **Qwen3.8 Flash Next** from a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API. It runs
[TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, `tensorfold-native` (branch
[`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext)), in NVIDIA's PyTorch container. One GPU,
with image and video input (`VISION=1`). `VISION=0` serves text only.

**Credits.** [TensorFold](https://github.com/ashhart/TensorFold) and its Zig engine are by Ash Hart
([ashhart](https://github.com/ashhart)) and the [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors),
branch [`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext). The Flash Next CUDA engine in this
recipe is ported from TensorFold's Python Flash Next engine, written by Ash Hart and the TensorFold contributors.
The Zig CUDA serving path and the CUDA family registry this recipe runs on were authored by Jürgen Schmied
([jschmied](https://github.com/jschmied)) in [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
(commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac)).
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is by the Qwen team (Alibaba). The checkpoint is
by azampatti ([azampatti](https://huggingface.co/azampatti)), who authored its top-5 expert cut, the shared-expert
healing and the checkpoint,
[`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound);
its AutoRound int4 quantization is by Intel
([`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)), and
its hybrid checkpoint and FP8 n-gram table are by Saren-Arterius ([Saren-Arterius](https://github.com/Saren-Arterius),
[`Saren/Qwen3.8-Flash-Next-ple-table-fp8`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8)).
[Zig](https://ziglang.org) is by the Zig Software Foundation and the Zig contributors. The image is based on
NVIDIA's [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). The one-shot RoCE
all-gather in the engine implements the RoCEnante protocol of [b12x](https://github.com/local-inference-lab/b12x) by
local-inference-lab (Apache-2.0); that implementation is new code, and this one-Spark recipe leaves it off.
Image and video support is adapted from MiaAI-Lab's single-Spark patches 0008 and 0009 and from TensorFold's vision
code, written by Ash Hart and the [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors).
The FP8 KV format (`KV_DTYPE=fp8`) is adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` in
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
The helper uses Hugging Face [transformers](https://github.com/huggingface/transformers) (Apache 2.0) for the tower
modules and the image processor, [PyAV](https://github.com/PyAV-Org/PyAV) and [FFmpeg](https://ffmpeg.org/) for video
decoding, and [Pillow](https://python-pillow.org/) for image decoding.
[MovieMaker93](https://github.com/MovieMaker93) authored the prompt-chunk change this recipe used on TensorFold's
Python engine ([TensorFold #40](https://github.com/ashhart/TensorFold/pull/40)). Javier
([jvr0x](https://github.com/jvr0x)) authored the language draft vocabularies that recipe added for MTP
([#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)). The checkpoint this recipe served
before the Zig engine was by [Vontra](https://huggingface.co/Vontra),
[`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP).
[321sssrt-bit](https://github.com/321sssrt-bit) authored the full-window admission fix and
`tools/context_boundary.py` in the two-Spark recipe
([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
[PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)): a request whose prompt plus
`max_tokens` equals the window is admitted, because the engine already keeps the draft rows beyond that window.
More in [Credits](#credits) and
[`CREDITS.md`](CREDITS.md).

INT4-AR uses 5 experts per token (4.8B active parameters) and scores about 10% lower general capability on its
authors' own harness (46.6-47.6 against 51.8). Tool use is unchanged.

- Checkpoint: [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
  at revision `1464274120d36a4d8fcaa934552334a7d83ce0fd` (~122 GiB). Routed experts are GPTQ int4 (group size 128), the
  n-gram table is FP8 and lives in `ple-table/`, and it is loaded with the weights. There is no SSD n-gram reader and
  no int8 KV cache. KV is FP8 by default (lossy, ~98.8% top-1 agreement with bf16); `KV_DTYPE=bf16` is exact. FP8 works with image and video input.
- API model id: `Qwen3.8-Flash-Next`
- Context: **262,144 tokens** a request (the native window). Up to 8 requests at once. A full window's
  fp8 cache is 4.37 GiB (bf16 is 7.52 GiB). The engine grows each request's cache as it goes and refuses one that would not fit, keeping
  10 GiB free, and another 2 GiB for the vision helper.
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## Performance

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
  window needs about 92 GiB `MemAvailable` at start (weights ~66 GiB, one full cache, scratch slack and a floor).
  `start.sh` stops before launching below that (`MEM_NEED_GIB`).
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~160 GB free disk on a fresh machine: ~122 GiB for the checkpoint under `~/.cache/huggingface` and the image under
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

By default only data URLs are accepted. `VISION_URLS=1` also lets the server fetch public `https://` URLs. Image and
video prompts are not kept for prefix reuse, so each turn of a chat with images processes them again. A request body
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
4. A check that it is GPTQ int4 `qwen4_exp` with its `ple-table/` n-gram files and every shard present.

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

Weights, including the FP8 n-gram table, measured 65.7 GiB on this engine at TP=1. After they load, the engine's
sequence budget is `MemAvailable` minus 10 GiB (`TENSORFOLD_MEMORY_RESERVE_GIB`). With `VISION=1` it also keeps 2 GiB
for the helper (`TENSORFOLD_VISION_WORKSPACE_MIB`; the tower peaks around 0.9–1.7 GiB). The pool is shared and streams
grow as they go. A request is admitted when its prompt plus its reply budget (`max_tokens`) fits beside the others;
one that does not fit is refused, not queued. On the spark4 boot above, with vision on, the engine reported
**30.22 GiB** of sequence memory. A full fp8 window is 4.37 GiB, so **six** streams can sit at 262,144 tokens at once
(26.2 GiB) and a seventh (30.6 GiB) does not fit. `PARALLEL` 8 is the cap: shorter requests share the pool, and a
full window past the sixth is refused. A pool probe with the helper not resident held 36.22 GiB and fit eight full
windows. `VISION=0` returns the 2 GiB helper workspace to the pool.

A 1,048,576-token window (`CONTEXT=1048576`, YaRN factor 4) is one 17.46 GiB fp8 cache on top of the weights. A prompt
that fills that window also needs prefill scratch, which the engine's budget does not reserve. It is not the default.
`start.sh` allows it when `MemAvailable` covers weights, one cache, a small scratch slack and the floor.

`prepare.sh` and `start.sh` refuse to run when `MemAvailable` is under 10 GiB (`MEM_FLOOR_GIB`). Keep at least 10 GiB
free under load. On this machine, exhausting unified memory freezes it.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh). Override it from the environment
(`PARALLEL=2 ./start.sh`) or from `.env` next to `start.sh` (`KEY=value`; the environment wins).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `8` | requests decoded together. One that would not fit (prompt plus `max_tokens`) is refused |
| `CONTEXT` | `262144` | prompt + reply window. `1048576` turns on YaRN factor 4 |
| `DRAFTS` | `1` | MTP drafts; `0` passes `--no-drafts` |
| `TF_FLASHNEXT_DEPTH` | `15` | drafts a round. The stop rule is the engine's hybrid (see `TF_FLASHNEXT_PRODUCT_STREAMS`) |
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
| `TF_FLASHNEXT_PRODUCT_STREAMS` | `2` | running-product draft stop at this many streams or fewer; confidence 0.5 above it |
| `TF_FLASHNEXT_PREFILL_TAIL` | `512` | a short last prompt chunk joins the previous one |
| `TF_FLASHNEXT_FP8_FUSED_SPLIT` / `TF_FLASHNEXT_HEAD_ROW_TILES` | `4` / `2` | block-FP8 split-K shapes of 2048+ columns (`TF_FLASHNEXT_FP8_FUSED_SPLIT_N`) below 256 rows on the fused kernel at pipeline depth 4, and the int4 lm_head in 32-row items (`patches/0018`). `0` / `1`: the cluster kernel and 16-row items |
| `MODEL_REVISION` | `14642741…` | the checkpoint commit this recipe serves |

The Python recipe's int8 KV and SSD n-gram reader are gone. KV is fp8 (or bf16), and the n-gram table is on the GPU.

### Thinking and sampling

By default the model thinks first. Per request, `temperature`, `top_p`, `top_k` and `seed` override the defaults
(`temperature: 0` is greedy). `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking, and
`"chat_template_kwargs": {"reasoning_effort": "low"}` (or `"medium"`, `"xhigh"`) sets reasoning effort. The reasoning
comes back in `reasoning_content`, the answer in `content`. A small `max_tokens` can return empty `content` because
the model was still thinking.

### API notes

- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (Anthropic) and
  `/v1/messages/count_tokens`, `/v1/models`, `/tokenize` and `/detokenize` (also under `/v1/`), `/health`, `/stats`
  and Prometheus `/metrics`.
- Refused with HTTP 400: `logprobs: true` / `top_logprobs` (no token probabilities) and `n` other than 1.
- `"draft": false` serves the request without drafts, the serial reference.
- Tool calls work, including `tool_choice: "required"` and a named function (Anthropic `tool_choice: {"type": "any"}`
  too): the reply then always opens a tool call.
- A conversation's next turn resumes from the kept state of its earlier turns. Image and video prompts are not kept.
- Structured outputs (`response_format`, the `guided_*` fields) are refused with HTTP 400: this engine does not
  enforce them.
- When the KV pool is full, a new request is refused. Admission reserves the prompt plus `max_tokens`; the request is
  not queued. A request whose prompt plus `max_tokens` equals `--context` is admitted: the draft rows sit past that
  window. That admission change was authored by [321sssrt-bit](https://github.com/321sssrt-bit)
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)).

## What the patches are

`patches/0001`-`0009` are the Zig Flash Next CUDA engine against TensorFold `db281878` (the `zig-flashnext` pin),
applied with `git apply` in the checkout root. They are the engine, not speed patches on a Python package. Drafts are
still checked: a drafted reply matches `"draft": false`.

Speed patches on top of the engine (each keeps every reply byte-identical):

| Patch | What it changes |
| --- | --- |
| `0018-flash-next-fp8-fused-split-head-rows` | block-FP8 split-K projections of 2048+ columns below 256 rows on the fused form (one block a column tile adds its K slices in slice order, the cluster kernel's sum; `TF_FLASHNEXT_FP8_FUSED_SPLIT`), and the int4 lm_head in 32-row items (`TF_FLASHNEXT_HEAD_ROW_TILES=2`) |

## Checks

`tools/` talks to the running server (`API_URL`, default `http://127.0.0.1:8888`, or `PORT`):

| Script | What it does |
| --- | --- |
| `tools/exact.py` | greedy replies: drafted, `"draft": false`, and concurrent, must match |
| `tools/toolcheck.py` | a tool call whose array argument comes back as a JSON array |
| `tools/needle.py` | a passphrase in a ~200,000-token prompt |
| `tools/prompt_reuse.py` | a resumed prompt matches a fresh one |
| `tools/bench.py` | prefill at 8k-128k and a short decode |
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
under the Qwen Community License 1.0, and the INT4-AR checkpoint under the license on its model card.

**Third-party software in the image.** The image `scripts/prepare.sh` builds is based on NVIDIA's PyTorch container
`nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA software in it is
governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start. By pulling or running the image you accept them. The MIT license above
covers this repository's scripts only.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)) and
the TensorFold contributors, the Zig CUDA serving path by Jürgen Schmied
([jschmied](https://github.com/jschmied), [PR #443](https://github.com/ashhart/TensorFold/pull/443)),
[Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen, and the INT4-AR checkpoint by
[azampatti](https://huggingface.co/azampatti), with its AutoRound int4 quantization by
[Intel](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) and its hybrid checkpoint and FP8 n-gram
table by [Saren-Arterius](https://github.com/Saren-Arterius). Image and video support is adapted from MiaAI-Lab's
patches 0008 and 0009 and from TensorFold's vision code by Ash Hart and the TensorFold contributors. The helper uses
Hugging Face [transformers](https://github.com/huggingface/transformers), [PyAV](https://github.com/PyAV-Org/PyAV),
FFmpeg and [Pillow](https://python-pillow.org/). The FP8 KV format is adapted from MiaAI-Lab's GLM recipe patch
`0038-glm-kv-fp8`. The one-shot RoCE all-gather implements the RoCEnante protocol of
[b12x](https://github.com/local-inference-lab/b12x) by local-inference-lab; this recipe leaves it off.
The prompt-chunk change on the earlier Python engine was by [MovieMaker93](https://github.com/MovieMaker93)
([TensorFold #40](https://github.com/ashhart/TensorFold/pull/40)). Javier
([jvr0x](https://github.com/jvr0x)) authored the language draft vocabularies of the earlier Python recipe
([#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)); that draft-language image is not
built anymore. [Zig](https://ziglang.org) is by the Zig Software Foundation and the Zig contributors. The previous
checkpoint was by [Vontra](https://huggingface.co/Vontra).
[321sssrt-bit](https://github.com/321sssrt-bit) authored the full-window admission fix and `tools/context_boundary.py`
([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
[PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)).
The full list is in [`CREDITS.md`](CREDITS.md).
