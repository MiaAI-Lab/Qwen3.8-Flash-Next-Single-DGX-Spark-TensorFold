<h1 align="center">Qwen3.8 Flash Next on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve **Qwen3.8 Flash Next** from a single NVIDIA DGX Spark (GB10, 128 GB) through an OpenAI-compatible API, with
**5 concurrent requests at the full 262,144-token context** and **image and video input**. It runs
[TensorFold](https://github.com/ashhart/TensorFold) v0.6.6 (`cb2ebf0`) in NVIDIA's PyTorch container, plus
`patches/0002-flash-next-v066.patch` (copy drafts, SSD read-ahead, first token before the next draft, no unused MTP
logits while prompts absorb, 96 MiB request bodies).

**Authored by Mia ([Mia's AI Lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab))**: this
recipe, its scripts and patches, and the switch to the NVFP4 checkpoint. The checkpoint is
[NVIDIA's own NVFP4 export](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), quantized with NVIDIA's
[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer); the engine is
[Ash Hart's TensorFold](https://github.com/ashhart/TensorFold). Full credits in
[`CREDITS.md`](CREDITS.md).

- Checkpoint: [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
  (NVIDIA ModelOpt NVFP4: FP4 routed experts, an FP8 n-gram table and block-FP8 MTP experts, with the MTP draft
  head; ~133 GB, revision `fc694b54`). It needs `patches/0003-nvidia-vision.patch` for its vision tower's
  `model_type`, which upstream's check does not list yet. The MLX 4-bit checkpoint
  `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` stays available as a `MODEL_ID` fallback, the only format two ranks
  and `--ple-on-ssd` read; `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` is the other NVFP4 export (smaller, its
  n-gram table in the main shards).

## Performance

One DGX Spark, the recipe's defaults on TensorFold v0.6.6 (`nvidia/Qwen3.8-Flash-Next-NVFP4`, 5 streams
x 262,144 tokens, int8 KV cache, image input on, MTP 6/0.60, copy drafts on), measured through the OpenAI API on
2026-10-06: `temperature 0`, thinking off, a fresh prompt each sample so nothing is reused from the prefix cache.
The prompt sets and table shape are the ones this README has always used, so the numbers are comparable with the
MLX-era ones below.

**Decode, prose**

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 35.2 tok/s | 35.2 tok/s | 29 ms |
| 2 | 64.8 tok/s | 32.8 tok/s | 28 ms |
| 4 | 121.2 tok/s | 30.7 tok/s | 16 ms |

**Decode, code**

| Concurrent requests | Aggregate | Per request | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 52.1 tok/s | 52.1 tok/s | 29 ms |
| 2 | 94.8 tok/s | 49.0 tok/s | 77 ms |
| 4 | 166.2 tok/s | 43.3 tok/s | 50 ms |

**Prefill**

| Prompt | Tokens | Prefill speed | Time to first token |
| ---: | ---: | ---: | ---: |
| 8k | 6,307 | 1,558 tok/s | 4.05 s |
| 16k | 12,600 | 1,616 tok/s | 7.81 s |
| 32k | 25,163 | 1,567 tok/s | 16.07 s |
| 64k | 50,308 | 1,503 tok/s | 33.47 s |
| 128k | 100,557 | 1,479 tok/s | 68.01 s |

Single-request decode is 52.1 tok/s on code and 35.2 on prose; the concurrent aggregate scales to 166.2 tok/s at four
streams while each keeps 43.3 tok/s. Prefill holds a ~1.5k tok/s plateau from 8k to 64k and eases to 1.48k at 128k.

Two things move these decode numbers, so compare like with like: **generation length matters** (a short reply pays
the per-request ramp), and the tables above are 256 tokens with thinking off.

Startup: 88.93 GiB estimate within a 107.58 GiB budget, 264 s to loaded (the one-time GB10 kernel compile included),
**27.7 GiB free for stream caches** against 4.47 GiB for one full 262,144-token window, so the whole pool fits. The
n-gram tables are the FP8 PLE table in `model-fp8-mtp-ple.safetensors` (47.7 GiB); the engine reports them read in
9.6 s and re-read after warm-up, and warns they do not fit resident beside the weights and caches, so prompt lookups
page them from disk. That is the main cost of NVIDIA's export against local-inference-lab's, whose table is smaller and
stays memory-mapped.

Against the MLX 4-bit tables this replaces (63.6 prose / 96.9 code single-request, 114.5 / 166.4 at four streams,
2,180-2,468 tok/s prefill): decode and prefill are both lower here at one stream, while the four-stream aggregate is
level (121.2 against 114.5 prose, 166.2 against 166.4 code). The MLX tables do not record their generation length,
thinking mode or prompt revision, so treat the two sets as the same shape rather than a controlled A/B.

Against local-inference-lab's NVFP4 export, which this recipe also supports via `MODEL_ID`: NVIDIA's is slower here
(35.2 / 52.1 against 42.3 / 61.1 single-request; 1,479-1,616 against 1,680-1,952 prefill), and the difference tracks
its FP8 n-gram table being 47.7 GiB and paging from disk, where that export's is smaller and stays mapped. NVIDIA's is
the vendor's own export with published accuracy tables and the one this recipe defaults to; the other is a drop-in
`MODEL_ID=local-inference-lab/Qwen3.8-Flash-Next-NVFP4` if prefill matters more than provenance.

Prefill is the consistent weak spot: 1,479-1,616 tok/s here against 2,180-2,468 for the MLX checkpoint, a wider gap
than the ~1.4-1.5x to vLLM that [TF #447](https://github.com/ashhart/TensorFold/issues/447) reports, for the reason
above (the paged FP8 table).
That is where TensorFold's CUDA path is behind, and it is the reason to keep the MLX checkpoint (`MODEL_ID=…`) if
prompts, not decoding, dominate a workload. Upstream's Zig CUDA server is where it is meant to close.

Verified on the same boot: drafted replies equal `"draft": false` (identical `token_sha`), 5 truly concurrent streams
each completed independently, a passphrase at 26.7k tokens was retrieved, a drawn image was named correctly
(`tools/visioncheck.py` OK), and tool-call arrays come back as JSON arrays (`tools/toolcheck.py` OK). `docs/v061.md`
has the v0.6.0-to-v0.6.1 comparison and the MLX-era tables for reference.

## Requirements

- A DGX Spark (or another GB10 system with 128 GB unified memory) with nothing else large on the GPU: the default
  setting admits 88.93 GiB when the server starts and grows to ~111 GiB with all five streams at their full window
  (see [KV pool and memory](#kv-pool-and-memory)). Measured boots ran with 110-114 GiB of `MemAvailable`.
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~175 GB free disk on a fresh machine: ~133 GB for the checkpoint download under `$HF_CACHE`
  (~133 GB) and ~35 GB for the image under Docker's root (~24 GB); `scripts/prepare.sh` checks both.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
./start.sh
```

That is all. The first run sets everything up (see below): it pulls the prebuilt image (~11 GB) and downloads the
~133 GiB checkpoint, then compiles the CUDA kernels for the GB10 (a few minutes, once). Later starts take ~2.5 minutes to load the
weights. `start.sh` shows each step, the server's log and the loading progress, runs a smoke test, prints
`Qwen3.8-Flash-Next is now LIVE! on port 8888` with the endpoint, and returns you to the shell.

```bash
curl -s http://<spark-address>:8888/v1/models

curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

Any OpenAI client works with `base_url = "http://<spark-address>:8888/v1"` and the model `Qwen3.8-Flash-Next`.
Streaming, tool calls (typed parameters, e.g. arrays come back as JSON arrays), reasoning content, images and
videos are supported. The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`;
a request without one gets `MAX_TOKENS` (32,768).

```bash
./start.sh restart                            # restart it, e.g. after changing a setting
./stop.sh                                     # stop the server and free the GPU memory
docker logs -f qwen38-flash-next-tf           # server log
curl -s http://<spark-address>:8888/health    # busy flag and live token totals
```

## Images and video

The model's own vision tower (27 layers, 0.84 GiB, from the same checkpoint) turns images and video frames into
tokens, placed with Qwen's 3-D rotary positions, the way the reference implementation does. Send them as OpenAI-style
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

A video is a `video_url` part (`{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}`); with
OpenAI's Python client, pass the same dicts in `messages`.

| | Images | Videos |
| --- | --- | --- |
| Formats | JPEG, PNG, WebP | MP4, WebM, MOV, MKV (anything FFmpeg decodes) |
| Per request | up to 50 (all of a chat's turns count), 10 MB each, 64 MB in all | up to 2, 64 MB each, 96 MB in all, up to an hour of footage |
| Tokens | up to 16,384 for all images, at most 4,096 an image (50 images: ~320 each; `"detail": "low"`: 256 an image) | 2 frames a second (at most 256 frames, spread over the whole video), each pair of frames one timestamped block; up to 16,384 tokens a request (`TENSORFOLD_VIDEO_TOKENS`) |

By default only data URLs are accepted; `VISION_URLS=1` also lets the server fetch public `https://` URLs. Image
and video prompts are not kept for prefix reuse, so each turn of a chat with images processes them again. A request
body can be up to 96 MiB (base64 makes data URLs a third larger than the files). Text requests are unaffected:
their replies stay byte-identical with vision on. `VISION=0 ./start.sh restart` serves text only.

## Other languages

**Recommended only for replies mostly in Chinese or Japanese; leave it off otherwise.**

MTP drafts may only propose tokens from a list, and TensorFold's list (79,591 tokens) is English and code: it holds
50 Chinese characters and 433 Cyrillic tokens. Replies in other languages still come out right (every token is
checked against the full vocabulary), but fewer drafts are accepted, so they decode slower. A second image adds a
language's tokens to that list (patch `patches/languages/0010`). It is opt-in: put the language in a `.env` file next
to `start.sh` and restart.

```bash
echo 'DRAFT_LANGUAGE=zh' >> .env     # or ja; several: zh,ja
./start.sh restart                   # switches to the language image (pulled or built the first time)
```

Remove the line (or leave it empty) and `./start.sh restart` to go back to the default image. The output is
byte-identical with either image; only speed changes. Measured on one Spark (one stream, recipe sampling, seed 1234,
one boot per arm, 2026-09-29):

| Replies in | Default image | Language image | Change |
| --- | --- | --- | --- |
| Chinese, thinking off / on | 35.5 / 38.4 tok/s | 45.9 / 50.6 tok/s | **+29% / +32%** |
| Japanese, thinking off / on | 39.5 / 46.0 tok/s | 46.9 / 49.2 tok/s | **+19% / +7%** |

The larger list makes every draft step a little slower, which is why it does not pay off for English or code.
`DRAFT_LANGUAGE` also accepts `ru`, `de`, `fr` and `pt`, but those have not been measured to help, so they are not
recommended. The language token lists come from the vLLM recipe's language draft vocabularies; see
[`CREDITS.md`](CREDITS.md).

## What `start.sh` and `scripts/prepare.sh` do

**`./start.sh`** works in five steps, each shown as it runs:

1. **Setup:** runs `scripts/prepare.sh` whenever the setup is not ready: on the first run, after the patches change,
   or with another model or image. It compares what `prepare.sh` last left ready with the current settings, so later
   starts skip it instantly.
2. **Checks:** the arguments (with TensorFold's own parser, in a throwaway container), the previous server, the
   port and the free memory.
3. **Launch:** `tensorfold serve` with the settings from `scripts/config.sh`.
4. **Loading:** the server's log as it comes, and every 15 s the elapsed time and how much of the startup estimate is
   on the GPU. If the server stops, the last log lines and the reason are shown.
5. **Smoke test:** one chat completion, then the LIVE message and the endpoint.

If the server is already running, `./start.sh` says so and leaves it alone; `./start.sh restart` stops it and
starts it again. It stops the server only after the setup and the argument check pass, so a typo leaves the running
server alone and the server is down only while it restarts. Stopping cuts off requests still running (`stop.sh` warns
when there are any). Extra arguments go to `tensorfold serve` after the defaults, so they win
(`./start.sh restart --context 131072`); `./start.sh --help` lists the options. `FOREGROUND=1 ./start.sh` stays
attached to the server's log and exits with its exit code (for a systemd unit).

**`scripts/prepare.sh`** does the one-time setup, and is safe to re-run (each step skips work already done):

1. Preflight: Docker, the NVIDIA runtime, disk space.
2. The image `tensorfold-qwen38:v0.6.6`: TensorFold v0.6.6 with every `patches/*.patch` applied, plus
   `transformers` (the vision tower) and PyAV (video decoding), on NVIDIA's PyTorch container
   (`nvcr.io/nvidia/pytorch:26.07-py3`). With `DRAFT_LANGUAGE` set it is `tensorfold-qwen38:v0.6.6-languages`
   instead, which also applies `patches/languages/*.patch`.
   It first tries the matching prebuilt image from GitHub
   Container Registry (`ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:v0.6.6-<patches hash>`,
   ~11 GB; `:latest` is the default image, `:languages` the language image); if that tag is not there (e.g. after
   you change `patches/`), or with `PULL=0`, it builds the image locally instead (a few minutes).
3. Downloads the checkpoint into `$HF_CACHE` (resumable).
4. Verifies the checkpoint with `tensorfold info`.

Run it yourself to download ahead of time or to rebuild the image from scratch:

```bash
scripts/prepare.sh             # set up without starting the server
scripts/prepare.sh --rebuild   # rebuild the image from scratch
PREPARE=1 ./start.sh restart   # force prepare.sh, then restart; PREPARE=0 skips the check
```

After changing `patches/`, `scripts/publish-image.sh` pushes the new image to GitHub Container Registry
(`latest` and `v0.6.6-<patches hash>`), and `DRAFT_LANGUAGE=zh scripts/publish-image.sh` the language image
(`languages` and its own `v0.6.6-<patches hash>`).

## KV pool and memory

TensorFold gives every stream its own cache for a full window, so the KV pool is streams x window:

| | Default |
| --- | ---: |
| Streams (`PARALLEL`) | 5 |
| Window per stream (`CONTEXT`, the model's native maximum) | 262,144 tokens |
| **KV pool** | **1,310,720 tokens** (4 streams: 1,048,576) |
| KV precision (`KV_DTYPE`) | int8 (an fp16 scale per 32 values) |
| Memory a stream, allocated (server log, NVFP4 checkpoint) | 4.47 GiB: the KV cache, the sparse-attention index and the stream's own buffers |
| **Memory for the pool, allocated** | **~22.4 GiB** (5 x 4.47 GiB) |

The server reports these at every start. With NVIDIA's checkpoint (measured 2026-10-07): `CUDA rank 0 startup
estimate 88.93 GiB within 107.58 GiB; native 262144, allocated prompt/reply window 262144, cache slots 262151` and
`up to 5 streams, each growing to 262144 prompt/reply tokens while memory lasts (27.7 GiB free for their caches,
4.47 GiB for one at the full window)`. The 27.7 GiB of headroom against 5 x 4.47 GiB means the whole pool fits. Its
47.7 GiB FP8 n-gram table does not fit resident beside the weights and caches, so the engine reports it pages from
disk at each lookup. The MLX-era numbers were `startup estimate 102.50 GiB within 103.26 GiB`.

Where the memory goes with NVIDIA's checkpoint (TensorFold's own estimate, measured): the weights and fixed buffers
take 88.93 GiB of the 107.58 GiB budget. The MLX recipe for comparison: 75.2 GiB weights + 0.84 GiB vision tower +
22.5 GiB stream caches + 4.0 GiB fixed buffers = a 102.5 GiB estimate.

The vision tower's scratch (~0.8 GiB at most, measured on a 4,096-token image and a 256-frame video) is taken only
while an image or video encodes and is handed back right after; startup reserves none for it
(`TENSORFOLD_VISION_WORKSPACE_MIB`). With `VISION=0` the tower is not loaded.

TensorFold's budget is the free memory at start (`MemAvailable`) minus its reserve. This recipe sets the reserve to
its 2 GiB floor (`TENSORFOLD_MEMORY_RESERVE_GIB=2`); unset, TensorFold would hold back a tenth of RAM (12.2 GiB, at
least 4 GiB). On the Spark's unified memory, running out tends to freeze the machine rather
than fail an allocation: raise the reserve (e.g. `TENSORFOLD_MEMORY_RESERVE_GIB=6`) if other workloads share the
box. Streams grow their caches as their context grows, while memory lasts; a stream that cannot grow waits.

Other settings, each verified on the NVFP4 checkpoint (its own startup estimate, measured by constructing the engine
with that setting; the stream caches are not pre-allocated, they grow as a stream's context grows, so the startup
estimate barely moves between rows and the headroom is what decides whether the full pool fits):

| Setting | KV pool | Startup estimate | Headroom | Note |
| --- | ---: | ---: | ---: | --- |
| `PARALLEL=4` (int8) | 1,048,576 | 81.88 GiB | 29.5 GiB | more headroom |
| `PARALLEL=5` (int8, default) | 1,310,720 | 82.23 GiB | 29.2 GiB | the default |
| `PARALLEL=6 CONTEXT=220000` (int8) | 1,320,000 | 81.86 GiB | 29.6 GiB | shorter windows, one more stream |
| `PARALLEL=6 KV_DTYPE=int4` | 1,572,864 | 81.09 GiB | 30.4 GiB | int4 changes outputs slightly; quality not measured here |
| `PARALLEL=8 KV_DTYPE=int4 CONTEXT=250000` | 2,000,000 | 81.53 GiB | 30.0 GiB | the largest pool of the set |
| `PARALLEL=3 KV_DTYPE=bf16` | 786,432 | 84.57 GiB | 27.0 GiB | full-precision KV |

Each row leaves 27-30 GiB for stream caches, well above the 4.47 GiB one full 262,144-token window takes at int8
(int4 is smaller, bf16 larger), so every one of them admits the whole pool. The MLX checkpoint's equivalents were
tighter (its own estimates ran 97.6-103 GiB against a 103 GiB budget), which is the practical gain from moving its
29.8 GiB of n-gram tables off the resident budget.

A setting that does not fit is refused at startup, before any weights load, with a message naming a window that
fits.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh) and can be overridden from the environment
(`PARALLEL=4 ./start.sh`), in a `.env` file next to `start.sh` (`KEY=value` lines, e.g. `PARALLEL=4`; the environment
wins over it), or with `tensorfold serve` flags (`./start.sh --context 131072`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `5` | requests decoded together (streams) |
| `CONTEXT` | `262144` | prompt + reply window per stream |
| `KV_DTYPE` | `int8` | `bf16`, `int8` or `int4` KV cache |
| `PLE_ON_SSD` | `0` | MLX checkpoint only (`MODEL_ID` override): read its 29.8 GiB n-gram tables from SSD instead of RAM. The NVFP4 checkpoint refuses the flag: its tables ship inside the checkpoint and load with the weights |
| `VISION` | `1` | image and video input (`--vision`); `0` serves text only |
| `VISION_URLS` | `0` | `1` also accepts public `https://` image and video URLs (default: data URLs only) |
| `DRAFT_LANGUAGE` | empty | `zh` or `ja`: serve the language image, for replies mostly in that language ([Other languages](#other-languages)) |
| `MTP_DRAFTS` / `MTP_CONFIDENCE` | `6` / `0.60` | at most 6 MTP drafts a round; a chain stops before a draft under 60% |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `1.0` / `0.95` / `20` | default sampling (Qwen's thinking-mode values); a request's own values win |
| `THINKING` | `1` | open a think block by default; `0` answers directly unless a request asks to think |
| `MAX_TOKENS` | `32768` | reply length for a request without `max_tokens` (TensorFold's own default, 4,096, can end a thinking reply before it answers); clamped to the stream's window |
| `SERVED_NAME` | `Qwen3.8-Flash-Next` | the model id in `/v1/models` and in replies |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `VISION_MAX_IMAGES` | `50` | images a request may carry, all of a chat's turns counted (`--vision-max-images`) |
| `TENSORFOLD_MTP_COPY` | `1` | prompt-lookup drafts for text that repeats the prompt (needs `PARALLEL` >= 2); `0` turns them off |
| `TENSORFOLD_IMAGE_TOKENS` | `16384` | the tokens a request's images share (`--vision-image-tokens`), each at most 4,096 |
| `TENSORFOLD_VIDEO_TOKENS` | `16384` | a request's video token budget |
| `TENSORFOLD_VISION_WORKSPACE_MIB` | `0` | what startup reserves for the vision tower's scratch |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `2` | GiB left out of MemAvailable at admission (floor since v0.6.0; unset would take about a tenth of RAM) |
| `PREPARE` | `auto` | `start.sh` runs `scripts/prepare.sh` when needed; `1` always, `0` never |
| `PULL` | `1` | `prepare.sh` tries the prebuilt image first; `0` always builds locally |
| `STOP_TIMEOUT` | `30` | seconds `stop.sh` gives the server to shut down before removing it |

Any `TENSORFOLD_*` variable in the environment is passed into the container (`TENSORFOLD_NO_UPDATE_CHECK=1`, the
default, stops TensorFold asking GitHub for a newer release at each start). Less common settings are described in
`scripts/config.sh`: `MODEL_ID`, `TF_VERSION`, `TF_REPO`, `BASE_IMAGE` (the patches are made for TensorFold v0.6.6;
after changing any of these run `scripts/prepare.sh --rebuild`), `IMAGE`, `CONTAINER_NAME`, `GHCR_IMAGE`, `HF_CACHE`
(default `$HF_HOME` or `~/.cache/huggingface`), `KERNEL_CACHE`, `MIN_FREE_GB`, `IMAGE_FREE_GB`. `start.sh` also takes
`FOREGROUND=1`, `WAIT_TIMEOUT` (seconds, default 1800) and `HF_HUB_OFFLINE=0` (let the server reach Hugging Face; by
default it serves from the local cache only).

### Thinking and sampling

By default the model thinks before it answers, with Qwen's recommended thinking-mode sampling: temperature 1.0,
top_p 0.95, top_k 20. TensorFold v0.6.6 honours `min_p` and a top-level `reasoning_effort` as well as the template
kwargs; presence and repetition penalties are still unused (same as presence_penalty 0.0 and repetition_penalty 1.0).
It also takes `--prefill-fp8` (FP8 prompt kernels for this checkpoint's MXFP8 linears; default bf16 prompts) and
`--api-key` if you need them; the recipe sets neither.
Per request:

- `temperature`, `top_p`, `top_k`, `min_p` and `seed` override the defaults (`temperature: 0` decodes greedily).
- `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking, and
  `"chat_template_kwargs": {"reasoning_effort": "low"}` (or `"xhigh"`) sets Qwen's reasoning effort; a top-level
  OpenAI-style `reasoning_effort` is honoured too. Without either, the template default applies (`None` on this
  CUDA CLI, not a forced medium).
- The reasoning comes back in `reasoning_content`, the answer in `content`.

## What the patches change

`scripts/prepare.sh` bakes every `patches/*.patch` into the image (unified diffs against TensorFold's site-packages,
applied with `patch -p0`), and `start.sh` rebuilds or re-pulls the image by itself when the patches change.

| Patch | Change |
| --- | --- |
| `0002-flash-next-v066` | Flash Next on v0.6.6: copy drafts with compact MTP rows (`TENSORFOLD_MTP_COPY`), SSD read-ahead around the n-gram table (`--ple-on-ssd`, MLX checkpoint), the first token out before the next draft chain, no unused MTP draft-vocabulary logits while prompts absorb, 96 MiB request bodies, and 64 MiB video / 64 MiB total image byte budgets to match. The `TENSORFOLD_PREFILL_ROWS` override and the video input moved upstream (v0.6.3, #238 and #240). Percentage effects in older notes were measured on v0.5.0, not on this pin. |
| `languages/0010-flash-next-draft-languages` | Opt-in language image only (`DRAFT_LANGUAGE`): language draft vocabularies for the MTP head, rebased onto v0.6.6 (hunks moved, the token lists are byte-identical). |

v0.6.6 brings, upstream since this recipe's v0.6.1 pin: the `TENSORFOLD_PREFILL_ROWS` override and `TENSORFOLD_MEMORY_RESERVE_GIB` (#238, #165), Flash Next video input (#240), NVFP4 and FP8 prompt kernels with the same bits (#242), `--vision-image-tokens` (#239), `--vision-offload` (#187), the Anthropic Messages API (#223), API keys, `tensorfold plan`, and `--name-priority ID=background` (#445). NVFP4 checkpoints like this recipe's run on one GPU: `--tp 2` and `--ple-on-ssd` refuse them at startup.

The checkpoint switch itself answers [#317](https://github.com/ashhart/TensorFold/issues/317) (the MLX "24% slower" report, retracted once the real causes — a cold n-gram page cache and the lone-stream graph slot being recaptured on short replies — were found; fixed in [#337](https://github.com/ashhart/TensorFold/pull/337), which stays out of 0.6.x because the Python engine is frozen at #286), and the prefill gap it leaves is [#447](https://github.com/ashhart/TensorFold/issues/447), measured on this very checkpoint and left to upstream's Zig CUDA server.

**Outputs are unchanged.** Every speed patch changes speed only: drafts are verified against the model's own keyed
samples, and the prefill changes read the same bytes and select the same attention blocks. This was checked by
comparing reply hashes (sampled and greedy, prompts up to 149k tokens) against unpatched TensorFold, with vision on
and off, and with a ~195k-token needle-in-a-haystack test. Text rows take exactly the rotary path they always did;
on image and video prompts, drafted replies equal the serial reference too. Any request can also be sent with
`"draft": false` to get TensorFold's serial, one-token-at-a-time reference.

## Checks

The scripts in `tools/` talk to the running server (`API_URL`, default `http://127.0.0.1:8888`; or just `PORT`),
from this machine or another one (`API_URL=http://<spark-address>:8888 tools/bench.py`).
`tools/test_copy_draft_rows.py` is the exception: it exercises patched TensorFold source on the CPU.

| Script | What it does |
| --- | --- |
| `tools/bench.py [label]` | smoke: prefill at ~0.85k / 3.2k / 12.6k / 50k tokens (fresh random prompts) and a short decode check. `--suite --seed 1 --jsonl run.jsonl --clients 1,2,4,5` is the concurrent, fixed-seed driver |
| `tools/test_copy_draft_rows.py` | mixed copy/MTP proposal-row ownership (needs `TF_SRC` pointing at patched TensorFold `src`; no model) |
| `tools/test_astra_patches.py` | first-token-before-draft, MTP absorb-without-head, PLE concat (`TF_SRC`) |
| `tools/needle.py` | hides a passphrase in a ~195k-token prompt and checks the model returns it |
| `tools/toolcheck.py` | makes a tool call with an array parameter and checks it comes back as a JSON array |
| `tools/visioncheck.py` | sends a drawn image (a red circle and a blue square) and checks the model names both |

## Repository layout

```
start.sh      set up (first run) and start the server
stop.sh       stop it
scripts/      prepare.sh (image + checkpoint), config.sh (all settings), publish-image.sh (push the image to GHCR),
              banner.sh (start.sh's banner)
patches/      patches baked into the image; patches/languages/ only into the language image (DRAFT_LANGUAGE)
tools/        benchmark and checks
.github/      issue and pull request templates, GitHub Sponsors
CREDITS.md    who and what this builds on
```

## License

MIT, see [`LICENSE`](LICENSE), which also carries TensorFold's MIT notice for the patches. The model weights, downloaded from Hugging Face and not
part of this repository, are under the Qwen Community License 1.0.

**Third-party software in the image.** The prebuilt image (and the one `scripts/prepare.sh` builds) is based on
NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA
software in it is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start (it shows in `start.sh`'s output); by pulling or running the image you
accept them. The image also contains Hugging Face `transformers` (Apache 2.0) and PyAV (BSD) with its FFmpeg
libraries (LGPL). The MIT license above covers this repository's scripts and patches only.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)), [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
by Qwen, and [NVIDIA's NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (quantized with
NVIDIA's [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer); local-inference-lab's NVFP4 export and
Vontra's MLX 4-bit checkpoint, which this recipe served until v0.6.6, are alternatives), with a prompt-chunk change by
[MovieMaker93](https://github.com/MovieMaker93) ([TensorFold #40](https://github.com/ashhart/TensorFold/pull/40)). The full list,
including the runtime stack and licenses, is in [`CREDITS.md`](CREDITS.md).
