<h1 align="center">Qwen3.8 Flash Next on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
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
[TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, `tensorfold-native` (release
[v1.1.4](https://github.com/ashhart/TensorFold/releases/tag/v1.1.4), commit `eab85828`, with this recipe's patches on top), in NVIDIA's PyTorch container. One GPU,
with image and video input (`VISION=1`). `VISION=0` serves text only.

The default checkpoint is NVIDIA's FP4 QAD export: routed experts in NVFP4 (group size 16, CSF-compressed scale factors),
attention and shared-expert linears in MXFP8, the MTP experts and the vision tower quantized too (MXFP8 attention, W4A16
NVFP4 MLP), an FP8 n-gram table, and all 512 routed experts with 10 per token (the INT4 cut keeps 5). The INT4-AutoRound checkpoint (served by the db281878 recipe on `main`, not by this v1.1.4 branch) uses 5 experts per token (4.8B active
parameters) and scores about 10% lower general capability on its authors' own harness (46.6-47.6 against 51.8). Tool use
is unchanged.

- Checkpoint (default): [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
  at revision `b9ce18a6887e18c089fdea8509a33e63429a3300` (~96 GiB; loading takes about 81 GiB of memory with vision on, measured).
- Checkpoint (not supported on this branch; the db281878 recipe on `main` serves it): [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
  at revision `1464274120d36a4d8fcaa934552334a7d83ce0fd` (~122 GiB). Routed experts are GPTQ int4 (group size 128), the
  n-gram table is FP8 and lives in `ple-table/`, and it is loaded with the weights. There is no SSD n-gram reader and
  no int8 KV cache. KV is FP8 by default (lossy, ~98.8% top-1 agreement with bf16); `KV_DTYPE=bf16` is exact. FP8 works with image and video input.
- API model id: `Qwen3.8-Flash-Next`
- Context: **262,144 tokens** a request (the native window). Up to 5 requests at once (`PARALLEL`). A full
  262,144-token window takes 4.78 GiB a stream with the fp8 cache, and a `PARALLEL` whose windows do not all fit is refused when the server starts
  (see [KV and memory](#kv-and-memory)). The engine keeps 9 GiB free, and another 2 GiB for the vision helper.
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## Performance

Numbers for the default checkpoint, **`nvidia/Qwen3.8-Flash-Next-FP4-Local`** (NVIDIA's QAD FP4 export), as `./start.sh` serves it: 262,144-token
window, 5 requests at once, fp8 KV, vision on, MTP drafts, TensorFold v1.1.4 with patches `0001` to `0017`. One DGX Spark (GB10), measured with
[sparkDash](https://github.com/MiaAI-Lab/sparkDash)'s Decode and Prefill benchmarks, running on the same Spark as the server (no network between
them). Prose: temperature 0.7, thinking on. Code: temperature 0, thinking off. Aggregate is the sum of the per-request decode speeds; TTFT is the
time to the first token.

How a decode figure is taken (sparkDash's `DecodeBench`): each request streams, and its decode speed is `(completion_tokens - 1) / (time of the last streamed token - time of
the first)`, so TTFT is not in it. The prose prompt asks for a step-by-step explanation of how a hash map works. The code prompts (`DECODE_CODE_TASKS` in sparkDash's
`src/shared/llmPrompts.js`) each name a small Python function or class and end "Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this
needs. Keep writing code.", and every concurrent stream gets a different task. MTP drafts pay only when they are accepted, so decode speed depends on the text: a code request that asks for docstrings and
comments, or a different prose prompt, decodes at another speed, and a figure from another harness is comparable only with the same prompts, sampling and thinking setting. `tools/bench_streams.py`
uses its own, shorter prompts and does not turn thinking off for code, so its code rows are not comparable with the table's ([#85](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/issues/85),
[#86](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/issues/86)).

**Decode, prose**

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 55.6 tok/s | 55.6 tok/s | 136 ms |
| 2 | 95.4 tok/s | 49.4 tok/s | 234 ms |
| 4 | 145.1 tok/s | 38.1 tok/s | 300 ms |
| 5 | 156.1 tok/s | 35.0 tok/s | 370 ms |

**Decode, code**

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 110.5 tok/s | 110.5 tok/s | 164 ms |
| 2 | 147.2 tok/s | 78.7 tok/s | 212 ms |
| 4 | 192.9 tok/s | 52.7 tok/s | 374 ms |
| 5 | 217.6 tok/s | 48.0 tok/s | 403 ms |

**Prefill** (cold prompts, no prompt-cache hit)

| Prompt tokens | Prefill | TTFT |
| ---: | ---: | ---: |
| 4,134 | 3,050 tok/s | 1.52 s |
| 8,231 | 2,956 tok/s | 2.95 s |
| 16,421 | 2,839 tok/s | 5.95 s |
| 32,807 | 2,735 tok/s | 12.16 s |
| 65,574 | 2,230 tok/s | 29.57 s |
| 131,110 | 2,269 tok/s | 57.94 s |

Three concurrent requests were not part of this run. Against the previous sparkDash run of the db281878 build with patches `0001` to `0015` (same Spark, same
tool; the CHANGELOG has every row): prose aggregate is +10% at one request, +30% at two, +36% at four and +18% at five; code is +63%, +47%, +47% and +45%;
cold prefill is +14.5% at 4k, +14.4% at 8k, +13.5% at 16k, +12.6% at 32k, -4.3% at 64k and +3.3% at 128k. Time to first token under load is higher at four
and five requests (+23% and +39% on prose, +16% and +24% on code) and lower at one (-30% prose, -9% code). The 64k prefill point reads below the 32k and 128k points and
below the 2,331 tok/s of the earlier run; we have not repeated it, so treat it as one noisy reading until we do.

Single-stream prose varied from run to run in the earlier measurements, 46 to 55 tok/s with the same settings (46.2, 46.9, 48.5 and 55.0 tok/s in four runs, about
15%). The one new reading, 55.6 tok/s, is at the top of that range, so the +10% at one request is within the old spread. Run sparkDash on the Spark itself: from
another machine over a slow link it adds the link's round trip to every TTFT (about 1.1 s over a 387 ms Tailscale path in our test) and staggers the requests,
which lowers the multi-stream numbers.

Against our earlier Python-engine build of the same checkpoint (measured the same way on another Spark: prose 54.4, 81.3, 88.5, 101.6 and 123.1 tok/s aggregate
at 1 to 5 requests; code 84.9, 128.3, 142.3, 159.5 and 181.9; prefill 2,315 tok/s at 4k down to 1,889 at 128k), this build is ahead at every point measured:
prose +2% at one request, +17% at two, +43% at four and +27% at five; code +30%, +15%, +21% and +20%; prefill +32% at 4k and +20% at 128k.
The Python numbers come from an earlier run with its own settings, so read these percentages as indicative.

### Checks on the current build

Run on one DGX Spark with the defaults (5 requests, 262,144-token window, fp8 KV, vision on), all passing. The sampled-exactness check and the 200k needle ran on the
integration build with patches 0001 to 0016 (patch 0017 changes only how many drafts are proposed); the others also ran on the live container with 0017:

- `tools/exact.py --pad 3000`: six drafted greedy replies (538 drafts accepted) equal the `"draft": false` ones, and six concurrent replies equal the
  one-at-a-time ones.
- A sampled-exactness check (six sampled replies, drafted against `"draft": false`, equal across runs) with the engine's prompt-keyed seeds (`--prompt-seed`).
- `tools/toolcheck.py` 3/3, `tools/visioncheck.py` (the model sees the red circle and the blue square), `tools/prompt_reuse.py 30000` (every follow-up
  resumed at least 90% of its prompt, worst 99.3%).
- A 199,730-token needle in a haystack is found, with a prefill of 100.7 s.
- `tool_choice: "required"`, a named function, Anthropic `{"type":"any"}` and a `response_format` JSON schema each return a valid call or object (live requests).
- `tools/context_boundary.py --context 262144`: a prompt plus reply that fills the window is admitted, with drafts on and off; one token over is refused.

## Requirements

- A DGX Spark (or another GB10 system with 128 GB unified memory) with nothing else large on the GPU. The default
  window needs `MemAvailable` of about 107 GiB at start (loading takes ~81 GiB): weights, one full cache, scratch slack and a floor.
  `start.sh` stops before launching below that (`MEM_NEED_GIB`).
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~160 GB free disk on a fresh machine: ~96 GiB for the checkpoint under `~/.cache/huggingface` and the image under
  Docker's root. `scripts/prepare.sh` checks both. Keep at least 100 GB free.
- Optional: the `hf` CLI on the host and a Hugging Face token in `~/.cache/huggingface/token` or `HF_TOKEN`. The
  checkpoint is public. `ABLIT=1` (below) needs a token, because its overlay is gated.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
./start.sh
```

The first run builds the engine (TensorFold v1.1.4 plus `patches/`; the kernels are embedded in the binary) and
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

## Choosing the checkpoint

`./start.sh` serves NVIDIA's FP4 QAD checkpoint, `nvidia/Qwen3.8-Flash-Next-FP4-Local` at the pinned revision (`CHECKPOINT=nvfp4`, the default). The
INT4-AutoRound checkpoint (`CHECKPOINT=int4`) is **not supported on this branch**: TensorFold v1.1.4's Flash Next engine has no INT4 path, and none of the numbers
or patches here were tried with it. The db281878 recipe on `main` still serves it.

Check which checkpoint is running with `docker inspect -f '{{.Config.Cmd}}' qwen38-flash-next-tf`: the path after `serve` names it. `DRY_RUN=1 ./start.sh`
prints what a start would do (the model, the memory it needs and the docker command) without stopping or starting anything. To serve a different model directory of the same
format, set `MODEL_ID` (and `MODEL_REVISION`, or leave it empty for the latest) instead of `CHECKPOINT`. A `MODEL_ID` set by hand takes no preset revision pin.

## ABLIT (opt-in)

`ABLIT=1` serves NVIDIA's FP4 checkpoint with [drowzeys's Keys "house-qsa" abliteration overlay](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47)
grafted in. The overlay replaces 24 tensors: `self_attn.o_proj` (weight and scale, MXFP8) of the 12 attention (QSA) layers (L3, 7, 11, ... 47), edited by a
rank-1 refusal-direction projection and requantized. DeltaNet `out_proj`, every expert, the MTP head, the n-gram table, the vision tower and the chat template stay NVIDIA's
(see the overlay's own `ABLIT_META.json`). The model has had its safety refusals removed; read the overlay's Responsible Use Agreement, which you accept to download it.

The overlay is gated, free and approved automatically. Accept its terms at its Hugging Face page while logged in, then `export HF_TOKEN=hf_...` or `hf auth login`
with that account, before:

```bash
ABLIT=1 ./start.sh restart
```

Set `ABLIT=0` (the default) to return to the stock checkpoint. The first `ABLIT=1` start downloads the overlay's ~190 MB sidecar; the checkpoint itself is not downloaded
again and is not modified. The replaced tensors have the stock tensors' dtypes and shapes, so the memory a start needs is the same; speed has not been measured with the overlay.

How it is applied: `scripts/prepare.sh` runs [`scripts/ablit_view.py`](scripts/ablit_view.py), which builds a folder `$HF_CACHE/tensorfold-qwen38/ablit-b9ce18a6-c44f5b92` holding
symlinks to every file of the stock snapshot, the sidecar as `tensors/house-qsa-oproj-mxfp8-L3-47.safetensors`, and a `model.safetensors.index.json` that points the 24 tensor
names at it. TensorFold's engine reads the shards its index names, in path order, and takes a tensor from the first shard that holds it; the sidecar's path sorts before the
stock `tensors/model-*` shards. The script refuses to build the folder if the sidecar does not hold exactly those 24 names, if a stock tensor has another dtype or shape, or if the
sidecar would not sort first. It only works for `CHECKPOINT=nvfp4` at the pinned revision (`ABLIT=1` with another `MODEL_ID`, `MODEL_REVISION` or `CHECKPOINT=int4` stops with a message). Check it
is active with `docker inspect -f '{{.Config.Cmd}}' qwen38-flash-next-tf`: the path after `serve` ends in `ablit-b9ce18a6-c44f5b92`.

**What has been checked.** The folder builder against the real sidecar and the real stock index and shard headers (24 of 24 names, same dtypes and shapes), and offline tests
(`tools/test_ablit_view.py`). A server has booted with `ABLIT=1` on this build and answered `/health`; `tools/exact.py`, `tools/needle.py` and a refusal check on it have **not** been run yet. The overlay author's own
numbers (refusal32 32/32 on a thinking-off run of their TensorFold recipe) are theirs, not measured here. The edit changes the weights, so replies differ from the stock checkpoint's: the byte-identical
checks in this repository compare a build against itself, not `ABLIT=1` against `ABLIT=0`.

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
(each image matched by a hash of its bytes, built into TensorFold v1.1.4), so a chat's next turn after an image
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
4. **Loading:** the log, and a stop if `MemAvailable` falls below `MEM_FLOOR_GIB` (9 GiB, the memory reserve).
5. **Smoke test**, then the LIVE message.

`FOREGROUND=1 ./start.sh` stays attached and exits with the container's exit code. Extra arguments go to
`tensorfold-native serve` after the defaults (`./start.sh restart --context 131072`). `DRY_RUN=1 ./start.sh` prints
the docker command and changes nothing.

**`scripts/prepare.sh`:**

1. Preflight: Docker, disk.
2. The image `tensorfold-qwen38:v1.1.4-eab8582`: TensorFold v1.1.4 (`TF_REF`) with every `patches/*.patch` applied, Zig 0.17.0, libjpeg-turbo and libwebp's decoder
   (built from checksummed sources), on `nvcr.io/nvidia/pytorch:26.07-py3`; the build runs `fatbins native` only. `libtfgrammar.so` is copied out of TensorFold's v1.1.4 release archive
   (SHA-256 pinned in `config.sh`). It tries a matching prebuilt image from GitHub
   Container Registry first, by digest (`PULL=0` builds locally; the published image of the current patch set is `v1.1.4-eab8582-4baf16137ecd`, also `:latest`).
3. The checkpoint, at the pinned revision.
4. A check that it is a `qwen4_exp` ModelOpt NVFP4 checkpoint and every shard present.
5. With `ABLIT=1` only: the overlay's sidecar from Hugging Face, and the checkpoint folder with it grafted in ([ABLIT](#ablit-opt-in)).

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

**Total available KV cache with the default checkpoint (NVIDIA FP4): about 27 to 29 GiB, roughly 1.4 to 1.6 million fp8 tokens.** That is the pool the engine
reports at start (the server log line "N streams at once, 4.78 GiB each ... of 29.0 GiB left after a 9.0 GiB reserve"), measured on a Spark with vision on and
`PARALLEL=5`: 28.0, 29.0 and 27.3 GiB on three starts. The engine counts 4.78 GiB for each stream at a 262,144-token window. Loading the weights takes about
81 GiB of memory (the 26.8 GiB NVFP4 n-gram table is read from disk by the host through the page cache, not held on the GPU). The pool is `MemAvailable` after
the weights load, minus `TENSORFOLD_MEMORY_RESERVE_GIB` (9 GiB), and with `VISION=1` minus 2 GiB kept for the helper (`TENSORFOLD_VISION_WORKSPACE_MIB`; the tower
peaks around 0.9-1.7 GiB). It follows the free memory at start (other containers on the machine shrink it), and each GiB less reserve adds a GiB to it.
`VISION=0` returns the 2 GiB helper workspace to the pool.

**Every stream gets a full window when the server starts.** Five windows take 23.9 GiB, and what is left of the pool is the prompt cache (3.4 to 5.1 GiB
on the starts above). A `PARALLEL` whose windows do not all fit is refused at start: `PARALLEL=8` at 262,144 tokens needs 38.27 GiB and the server answered
"the CUDA memory budget leaves 27.22 GiB". Within the pool, a request is admitted when its prompt plus its reply budget (`max_tokens`) fits beside the others;
one that does not fit is refused, not queued.

**The prompt cache is a cap, not a reservation.** On CUDA the prompt cache takes at most what the weights and the streams leave, so `--prompt-cache-gib` (pass it after the defaults:
`./start.sh restart --prompt-cache-gib 12`) can only lower the 3.4 to 5.1 GiB above, never raise it: at 5 streams a larger value changes nothing, and it binds only when the leftover is bigger
(a smaller `PARALLEL` or `--context`). `--prompt-cache-over-cap` is refused on CUDA (it lifts a Mac's 70% cap). A larger cache for long multi-turn prompts costs streams or
window length: `PARALLEL=4` leaves one window's 4.78 GiB more for the cache than five do ([#81](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/issues/81)).

`PARALLEL=6` at 262,144 tokens (28.7 GiB) fits only with a smaller reserve. We tried `TENSORFOLD_MEMORY_RESERVE_GIB=7`: it started ("28.9 GiB left after a 7.0 GiB
reserve"), `tools/exact.py`, `toolcheck.py` and `visioncheck.py` passed and decode speed was unchanged, but the prompt cache fell to 0.2 GiB and
`tools/prompt_reuse.py 30000` failed (a 32k-token follow-up resumed 4.7% of its prompt and prefilled again, 13.8 s). Multi-turn use then re-reads its history on
every turn, so we left the default at 5. Each GiB the reserve gives up is a GiB of page cache the n-gram table loses, so it is not free. For more streams, lower `CONTEXT` instead: eight streams at 163,840 tokens need about 24 GiB (calculated from the sizes above, not run).

A 1,048,576-token window (`CONTEXT=1048576`, YaRN factor 4) is one 17.46 GiB fp8 cache on top of the weights. A prompt
that fills that window also needs prefill scratch, which the engine's budget does not reserve. It is not the default.
`start.sh` allows it when `MemAvailable` covers weights, one cache, a small scratch slack and the floor.

`prepare.sh` and `start.sh` refuse to run when `MemAvailable` is under 9 GiB (`MEM_FLOOR_GIB`). Keep at least 9 GiB
free under load. On this machine, exhausting unified memory freezes it.

### Kernel `NV_ERR_NO_MEMORY` messages under load (issue #33)

Under a long mixed load the stock 1.0.0 engine could log `NV_ERR_NO_MEMORY ... _memdescAllocInternal` in the kernel log while `MemAvailable` stayed above 30 GiB, with no
failed request. The cause traced to the `cuMemCreate` calls that grow a kept prompt state, and to 64 KiB system-memory pages that must be physically contiguous
(NVIDIA's driver retries eligible allocations with 4 KiB pages). The opt-in mitigation patch of the 1.0.0 recipe (`TF_FLASHNEXT_SYSTEM_CACHE`, `TF_FLASHNEXT_RING_CACHE`) is not
part of the v1.1.4 patch series, and those settings do not exist in this build. We have not re-run the reproduction load on v1.1.4, so we do not know whether the messages still appear.

The driver-side workaround is independent of the engine: the nvidia module option `NVreg_RegistryDwords="RmEnableLargePageSizeSysmemDefault=0"` (reboot) makes default
system-memory allocations start at 4 KiB. In one 45-minute run on kernel 6.17.0-1031-nvidia with the 1.0.0 engine it gave 0 messages against 139, at 58.6 instead of 62.8 tok/s.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh). Override it from the environment
(`PARALLEL=2 ./start.sh`; the memory a `PARALLEL` needs is explained under [KV and memory](#kv-and-memory)) or from `.env` next to `start.sh` (`KEY=value`; the environment wins).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `5` | requests decoded together. One that would not fit (prompt plus `max_tokens`) is refused |
| `CONTEXT` | `262144` | prompt + reply window. `1048576` turns on YaRN factor 4 |
| `DRAFTS` | `1` | MTP drafts; `0` passes `--no-drafts` |
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
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `9` | kept free when the engine sizes the pool (9 since patch 0014: the fused prompt path needs about 0.55 GiB more scratch, and five full windows need 23.9 GiB). `prepare.sh` and `start.sh` refuse a Spark below this |
| `TF_FLASHNEXT_DRAFTS` | `6` | drafts a round at most |
| `TF_FLASHNEXT_CONFIDENCE` / `TF_FLASHNEXT_PRODUCT_STREAMS` | `-0.4` / `5` | the draft chain keeps draft j while the running product of the draft probabilities is at least 0.4, for up to 5 streams drafting together (0.5 per draft above that) (patch 0017). Only the number of drafts changes, never a reply. `TF_FLASHNEXT_CONFIDENCE=0.6` restores v1.1.4's per-draft rule. Greedy code +4% at 5 streams, +10% at 3, +14% at 1 |
| `TF_FLASHNEXT_SAMPLED_HEADS` / `TF_FLASHNEXT_SAMPLE_CAND` | `1` / `1` | sampled requests join the batched MTP head pass, and their top-k candidates are found on the GPU (patches 0007, 0008) |
| `TF_FLASHNEXT_HOST_DRAW_THREADS` | `7` | worker threads for the host-side sampled draws (patch 0011); bit-identical to one thread |
| `TF_FLASHNEXT_KEEP_LATE` | `1` | a partly rejected draft window does not replay the DeltaNet state: the kept rows wait in a log and the next verify pass applies them (patch 0012) |
| `TF_FLASHNEXT_GATHER_EARLY` / `TF_FLASHNEXT_GATHER_READ` | `1` / `pread` | the n-gram rows of all lanes are gathered in one pass, early, and read with a `pread` thread pool instead of an `mmap` that page-faults (patches 0010, 0013). `mmap` restores the stock reader |
| `TF_FLASHNEXT_DRAFT_AHEAD` / `TF_FLASHNEXT_WARM_AHEAD` | `1` / `1` | read ahead the n-gram rows of the draft tokens during the head step, and of the next prompt chunk during the current one |
| `TF_FLASHNEXT_PREFILL_TAIL` | `512` | the last two prompt chunks are evened out when the last would be under this many rows |
| `TF_FLASHNEXT_PROMPT_HC2` / `TF_FLASHNEXT_PROMPT_SHARED_DIRECT` | `1` / `1` | prompt passes run the hyper-connection read-out as two fused kernels instead of five, and the shared expert's rows go straight into the output slot (patch 0014). Bit-identical |
| `TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS` / `TF_FLASHNEXT_HEAD_PIECE` | `256` / `1536` | the routed experts of prompt chunks use the shared-memory row tiles from this many rows; the MTP head takes prompt pieces of this many rows (patch 0014) |
| `TF_FLASHNEXT_PROMPT_GEMM2` | `1` | the MX128 prompt tile runs with two pipeline stages and two blocks an SM (patch 0015). Same bits, 15-25% faster per call |
| `TF_FLASHNEXT_SHARED_SIDE` / `TF_FLASHNEXT_GDN_LANES` | `1` / `1` | the shared expert runs on a second stream in decode rounds, and a shared round takes every lane's DeltaNet rows in three launches a layer (needs `KEEP_LATE=1`) (patch 0016). Bit-identical |
| `CHECKPOINT` | `nvfp4` | `nvfp4` serves `nvidia/Qwen3.8-Flash-Next-FP4-Local`. `int4` is not supported on this branch. `MODEL_ID` / `MODEL_REVISION` override the model |
| `MODEL_REVISION` | `b9ce18a6…` | the checkpoint commit this recipe serves |
| `ABLIT` | `0` | `1` serves the checkpoint with drowzeys's Keys house-qsa abliteration overlay ([above](#ablit-opt-in)); gated, needs `HF_TOKEN`. Only with `CHECKPOINT=nvfp4` at the pinned revision |

KV is fp8 (or bf16). The NVIDIA checkpoint's NVFP4 n-gram table (26.8 GiB) is read from disk by the host
(`TF_FLASHNEXT_GATHER_READ=pread` by default, with read-ahead of the next prompt chunk; `TF_FLASHNEXT_WARM_AHEAD=0` turns that off).

### Thinking and sampling

By default the model thinks first. Per request, `temperature`, `top_p`, `top_k` and `seed` override the defaults
(`temperature: 0` is greedy). `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking, and
`"chat_template_kwargs": {"reasoning_effort": "low"}` (or `"medium"`, `"xhigh"`) sets reasoning effort. The reasoning
comes back in `reasoning_content`, the answer in `content`. A small `max_tokens` can return empty `content` because
the model was still thinking.

- **A reply that spends its whole `max_tokens` thinking:** `thinking_budget` (a request field; `--thinking-budget N` as a server default, for example
  `./start.sh restart --thinking-budget 4096`) closes the think block after that many reply tokens and lets the answer start
  (`think_budget` in `zig/src/core/lanes/stream.zig`; the closing is done in the engine's lanes, so drafted and serial replies cut at the same token).
  `reasoning_effort: low` and `enable_thinking: false` are the other levers.
- **Without a `seed`, a sampled request draws a fresh one** (TensorFold v1.1.4's default, as OpenAI's API does); an explicit `seed` still wins. For reproducible
  replies send a `seed`, or start the server with `--prompt-seed` (the engine then derives the seed from the prompt's token ids, so the same prompt gives the same
  sampled reply, and `"draft": false` comparisons are reproducible; `tools/exact.py` and the sampled-exactness checks run that way). `TENSORFOLD_SEED_SALT=<integer>`
  changes every derived seed at once.
- **`presence_penalty`, `frequency_penalty` and `repetition_penalty` are accepted and have no effect** (the engine does not implement them), and out-of-range
  sampling values such as `temperature: -1` are not rejected. Use `top_p`, `top_k`, `min_p` and `temperature` to steer repetition.
- **A control token in the reasoning ends the turn.** The engine stops on the checkpoint's end tokens (`<|endoftext|>` 248044 and `<|im_end|>` 248046 in its
  `eos_token_id`; the report saw `<|im_start|>` 248045 stop a reply too). If the model writes one while it is still thinking, for example when it quotes it in its analysis, the reply stops right there
  with `finish_reason: "stop"`, empty `content` and a cut `reasoning_content` ([#76](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/issues/76)).
  A client can treat a `stop` with empty `content` as a failed turn and retry with `enable_thinking: false`.
- **`tool_choice: "required"`, a named function and structured output** are enforced by the engine's grammar library, `libtfgrammar.so` (xgrammar), which the image takes
  prebuilt from TensorFold's v1.1.4 release archive. With thinking off, `required` still gives the model no room to reason before the call: the grammar forces a call from the
  first token, and the function is then picked from that start. With many tools the pick can be less relevant than under `auto`. Keep thinking on, use `auto` with an
  instruction to call a tool, or name the function in `tool_choice`.

### API notes

- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (Anthropic) and
  `/v1/messages/count_tokens`, `/v1/models`, `/tokenize` and `/detokenize` (also under `/v1/`), `/health`
  and Prometheus `/metrics` (stage timers, time per output token, prefix-cache and speculative-decoding counters).
- Refused with HTTP 400: `logprobs: true` / `top_logprobs` (no token probabilities) and `n` other than 1.
- `"draft": false` serves the request without drafts, the serial reference.
- Tool calls work, including `tool_choice: "required"`, a named function and Anthropic's `{"type": "any"}`: the reply then always opens a tool call.
- A conversation's next turn resumes from the kept state of its earlier turns, with images too (built into TensorFold v1.1.4: a changed
  image resumes only up to its first row). Video prompts are not
  kept.
- Structured outputs (`response_format` with a JSON schema, the `guided_*` fields) are enforced with `libtfgrammar.so`; a reply follows the schema. Without the library (a hand-run
  `tensorfold-native` that lacks it) the server answers HTTP 400.
- When the KV pool is full, a new request is refused. Admission reserves the prompt plus `max_tokens`; the request is
  not queued. Kept prompt states share that pool: admission frees them first (built into TensorFold v1.1.4), so a request is refused
  only when it would not fit with every freeable state gone. A request whose prompt plus `max_tokens` equals `--context` is admitted: the draft rows sit past that
  window (checked with `tools/context_boundary.py`; a streamed request over the limit is refused with an error inside the stream). That admission change was authored by [321sssrt-bit](https://github.com/321sssrt-bit)
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)).

## What the patches are

`patches/*.patch` are this recipe's changes to TensorFold v1.1.4 (commit `eab85828`, which brought its own Flash Next CUDA engine), applied with `git apply`
in the checkout root by `scripts/prepare.sh`. The earlier series for TensorFold `db281878` (patches `0001` to `0031`) is retired, since v1.1.4 replaced the engine it
targeted. Drafted replies are still checked: a drafted reply matches `"draft": false`.

| Patch | What it does |
| --- | --- |
| `0001`, `0002` | the vision helper loads NVIDIA's quantized vision tower (MXFP8 attention, NVFP4 MLP and merger decoded to bf16 at load); image budget 16,384 tokens |
| `0003` | `--kv-cache fp8` for Flash Next: five full 262,144-token windows fit (stock v1.1.4 fits three) |
| `0004`, `0006` | prompt expert tile and prefill work: no drain between prompt chunks, vectorised hyper-connection kernels, a 128x128 MXFP8 prompt tile, 3,072-row chunks |
| `0005` | the fp8 attention reader keeps the next tile in registers; the MTP head's cache stays bf16 |
| `0007` to `0011` | sampled requests join the batched draft pass, GPU top-k candidates, a round-cost model (off), n-gram rows of all lanes gathered at once, host draws on worker threads |
| `0012`, `0013` | a partly rejected draft window no longer replays the DeltaNet state; the n-gram table is read with `pread` and read ahead instead of through an `mmap` |
| `0014`, `0015` | fused hyper-connection kernels, expert row tiles and MTP head pieces for prompt passes; a two-stage MX128 prompt tile |
| `0016` | the shared expert runs on a second stream in decode rounds; a shared round takes every lane's DeltaNet rows in three launches a layer |
| `0017` | the draft chain stops on the running product of its draft probabilities (the rule of the earlier build) |

Every knob is an environment variable with its default in `scripts/config.sh`; the CHANGELOG lists them with the measured effect of each round.

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
| `tools/context_boundary.py --context N` | a prompt plus reply that fills `--context` succeeds; one token over is refused (HTTP 400, or an error in the stream). Authored by [321sssrt-bit](https://github.com/321sssrt-bit) |

## Repository layout

```
start.sh      set up (first run) and start the server
stop.sh       stop it
scripts/      prepare.sh, config.sh, publish-image.sh, banner.sh, ablit_view.py
patches/      this recipe's changes, applied onto TensorFold v1.1.4 at TF_REF
tools/        checks
.github/      issue and pull request templates, GitHub Sponsors
CREDITS.md    who and what this builds on
CHANGELOG.md  upgrade notes
NOTICE        third-party notices
```

## License

MIT, see [`LICENSE`](LICENSE). The model weights, downloaded from Hugging Face and not part of this repository, are
under the checkpoint's own license (the `ABLIT=1` overlay inherits NVIDIA's and the Qwen license, and has its own Responsible Use Agreement): the default NVIDIA FP4 checkpoint under the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) (its base model, Qwen3.8 Flash Next, is under the
Qwen Community License 1.0), and the INT4 checkpoint (db281878 recipe only) under the Qwen Community License 1.0 per its model card.

**Third-party software in the image.** The image `scripts/prepare.sh` builds is based on NVIDIA's PyTorch container
`nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA software in it is
governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start. By pulling or running the image you accept them. The MIT license above
covers this repository's scripts only.

## Credits

- **Engine:** [TensorFold](https://github.com/ashhart/TensorFold) and its Zig engine (`tensorfold-native`, release
  [v1.1.4](https://github.com/ashhart/TensorFold/releases/tag/v1.1.4)) by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors). The Flash Next CUDA engine this recipe builds on is TensorFold's own, written by Ash Hart and the TensorFold contributors.
- **Zig CUDA serving path and CUDA family registry:** Jürgen Schmied ([jschmied](https://github.com/jschmied)), in
  [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
  (commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac)).
- **The 17 patches in `patches/` (0001 to 0017):** written by MiaAI-Lab (Mia's AI Lab), MIT License. They modify
  TensorFold's own Flash Next CUDA engine (code by Ash Hart and the TensorFold contributors, under TensorFold's licenses). The list, patch by patch, with where each
  idea comes from, is in [`CREDITS.md`](CREDITS.md): the vision tower (0001, 0002), fp8 KV (0003, 0005; the format adapted from MiaAI-Lab's GLM recipe patch 0038), prompt
  expert tile and prefill (0004, 0006, 0014, 0015), sampled streams (0007 to 0011; the GPU top-k idea from MiaAI-Lab's DeepSeek recipe), rollback, `pread` n-gram reader and
  decode (0012, 0013, 0016, 0017; 0013, 0016 and 0017 carry forward parts of the earlier db281878 series, whose NVIDIA FP4 port is by MiaAI-Lab from
  the TensorFold Contributors' Python implementation of the same formats).
- **Earlier db281878 series** (on `main`): free after drain and admission reclaim (patch 0029) and image prompts in the prompt cache (patch 0030) by [tempeste](https://github.com/tempeste), now built
  into TensorFold v1.1.4; sliced fills one at a time (patch 0021) by [BadAd84](https://github.com/BadAd84) (PR #53).
- **Model:** [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team (Alibaba).
- **Default checkpoint:** NVIDIA's [FP4 QAD export](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (Model Optimizer).
- **INT4 checkpoint** (served by the db281878 recipe on `main`, not by this branch): [azampatti](https://huggingface.co/azampatti)'s
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
- **Grammar runtime:** [xgrammar](https://github.com/mlc-ai/xgrammar) by the MLC-AI team (Apache-2.0), shipped as `libtfgrammar.so` in TensorFold's release archive.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and the Zig contributors; the image is based on NVIDIA's
  [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
- **Benchmarks:** the performance numbers were measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) by MiaAI-Lab.

The full list is in [`CREDITS.md`](CREDITS.md).
