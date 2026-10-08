# Changelog

Newest first. This recipe serves one DGX Spark. The image is `tensorfold-qwen38:zig-db28187`, built by
`scripts/prepare.sh` from TensorFold `db281878` plus `patches/`.

## Unreleased

- `TF_FLASHNEXT_CONFIDENCE` defaults to `-0.15` (was unset: the engine's `-0.4`, tuned at TP=2). The running-product
  draft stop at up to two live streams lets chains run deeper on one Spark: decode +10% (prose) and +3% (code) at one
  client, +7% and +1% at two, replies byte-identical (it changes only where a draft chain stops).

## [1.0.0] - 2026-10-08

TensorFold's Zig engine (`tensorfold-native`, TP=1). Engine `37763df`, merged into `zig-single` as `453439d`
(owner build `20e709a` plus the graph-failure fallback and the full-window admission fix). `patches/0001`–`0009`
are that tree against TensorFold `db281878`. `zig/src/cuda/graph.zig` ships in
`patches/0001-build-registry-server.patch`. The kernel set is unchanged (`KERNEL_SOURCE_MTIME=1791318675`; the
kernel-set check passed, 320 kernels).

Image `ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:zig-db28187-4144696f21e3` and `:latest`
(`sha256:cb98098439d95d2bff27bc771ba180e5857613ecc41f6b6e81e494782df9433a`). Label `tf.patches` is `4144696f21e3`.

### Upgrade from the Python recipe

- **Checkpoint.** Was [Vontra's MLX 4-bit conversion](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP).
  Now [azampatti's INT4-AutoRound](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound) at
  `1464274120d36a4d8fcaa934552334a7d83ce0fd` (top-5 experts, 4.8B active).
- **KV.** Was int8. Now FP8 by default (`KV_DTYPE=fp8`): lossy, about 98.8% top-1 agreement with bf16, about 1.8x the
  pool. `KV_DTYPE=bf16` is the exact cache. One 262,144-token sequence is 4.37 GiB in fp8 and 7.52 GiB in bf16. The
  FP8 format is adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8`.
- **N-gram table.** The SSD reader is gone. The FP8 table in the checkpoint's `ple-table/` loads with the weights.
- **Draft-language image.** Gone. `DRAFT_LANGUAGE` and `patches/languages/0010` are not applied. Javier
  ([jvr0x](https://github.com/jvr0x)) authored those lists
  ([#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)).
- **Image and video.** `VISION=1` by default, `VISION_URLS=0`, 50 images, 4 videos, 16,384 image tokens a request
  (at most 4,096 an image), 16,384 video tokens. Adapted from MiaAI-Lab's patches 0008 and 0009 and from TensorFold's
  vision code by Ash Hart and the TensorFold contributors. The helper is TensorFold's Python tree with transformers
  5.17.0, PyAV 19.0.1 and Pillow. FP8 KV and `--vision` run together.
- **Memory floor.** `TENSORFOLD_MEMORY_RESERVE_GIB=10`. `prepare.sh` and `start.sh` refuse a Spark with less than
  10 GiB `MemAvailable`. With vision on, the engine also keeps 2 GiB for the helper
  (`TENSORFOLD_VISION_WORKSPACE_MIB=2048`; the tower peaks around 0.9–1.7 GiB). A request that does not fit the pool
  (prompt plus `max_tokens`) is refused, not queued. Structured outputs (`response_format`, `guided_*`) are refused.
- **Streams.** `PARALLEL` is 8 (a cap, 1 to 16). On spark4 with FP8 KV and vision on, sequence memory was 30.22 GiB,
  which holds six full 262,144-token windows. A request that would not fit is refused. `TF_FLASHNEXT_PRODUCT_STREAMS`
  is 2.

### Fixed
- **A server exit under long load.** A CUDA graph capture, instantiate or upload that fails now leaves the round's
  result from its eager run in place (same bits), pauses graph captures for a while (64 rounds, doubling on repeated
  failures, up to 4096), and logs a warning. The server keeps serving. `TF_FLASHNEXT_GRAPH_LOG=1` logs the graph
  counts and a context-wide check before each instantiate. It is off by default. The two-Spark stack hit
  `cuGraphInstantiateWithFlags: CUDA_ERROR_NOT_PERMITTED` after about 15 minutes of heavy load; this server runs the
  same engine.
- **Requests that fill the window exactly were refused**
  ([#1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [#2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2), reported, diagnosed and fixed by
  [321sssrt-bit](https://github.com/321sssrt-bit)): admission counted the draft window against `--context`, so a
  prompt plus `max_tokens` equal to the window got `PromptTooLong`. The engine already keeps those rows beyond the
  window; admission now checks prompt + reply against `--context`.

### Added
- `tools/context_boundary.py` (by [321sssrt-bit](https://github.com/321sssrt-bit), from
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)): full prompt/reply budgets
  succeed with drafts on and off, one token over the window returns HTTP 400.

### Earlier Zig boot

Engine `6eb39c1`, bf16 KV, text only, spark4: prose 63.3 / 95.4 / 118.5 / 148.6 tok/s at 1 / 2 / 3 / 4 requests.
Sequence memory on that boot was 33.28 GiB. The README's current tables are the later FP8 and vision boot.

## Python recipe, through 0.6.0

## [0.6.0] - 2026-10-02

Commit `22a3010` (#13). Images unchanged from 0.5.0.

### Changed

- A request without `max_tokens` now gets 32,768 tokens (`MAX_TOKENS`), not TensorFold's 4,096. A thinking reply
  could use all 4,096 before it answered and end with no content or tool call. The engine clamps the value to the
  room left in the stream's window, so the startup estimate is unchanged; a request's own `max_tokens` wins. Ported
  from #11 by [MovieMaker93](https://github.com/MovieMaker93).
- README: decode (prose and code) and prefill tables measured on TensorFold v0.6.1.

## [0.5.0] - 2026-10-02

Commit `78a14eb` (#12). TensorFold **v0.6.1** (`17c73e1`). Images `v0.6.1-797af1d9df4d` (`:latest`) and
`v0.6.1-f8a0b4cb702b` (`:languages`).

### Changed

- TensorFold v0.3.6.3 to v0.6.1. The nine patches become one, `0002-flash-next-v061`. Image input is now
  TensorFold's own Flash Next vision (upstream #146, adapted from this recipe's patch). The patch adds video, many
  images, copy drafts, SSD read-ahead, first token before the next draft, no MTP logits while prompts absorb,
  and 96 MiB request bodies.
- The image limit is set with `VISION_MAX_IMAGES` (`--vision-max-images`, default 50) instead of
  `TENSORFOLD_MAX_IMAGES`.
- `TENSORFOLD_PREFILL_ROWS` overrides TensorFold's own choice of prompt piece rows. The recipe sets 2,048 when the
  n-gram tables are on SSD, where TensorFold's 4,096-row pieces measured 10-30% slower from 5k to 16k tokens.
- Language image: the draft-vocabulary patch (`patches/languages/0010`) is rebased onto v0.6.1.

### Added

- From TensorFold v0.6.1:
  - A new conversation on a shared system prompt reuses it: a 31k-token one answered in 0.18 s instead of 14 s.
  - Forks resume from their shared prefix, and short prompts are admitted while a long one fills.
  - `/metrics` in Prometheus format, with vLLM's metric names.
  - `logprobs`, and tool-call arguments streamed as they are generated.
  - Tool-call values written in Python's spelling (`True`, `None`) decode to their schema type.
- `docs/v061.md`: the port and its measurements.

### Fixed

- README: the reserve section names `TENSORFOLD_MEMORY_RESERVE_GIB`, the setting TensorFold reads, instead of
  `TENSORFOLD_HOST_RESERVE_MIB`.

Measured against v0.6.0: identical tokens in 34 greedy replies, and prefill and decode within noise. Five concurrent
~236k-token prompts completed. The v0.5.0 and v0.6.0 ports that led here were never released.

## [0.4.1] - 2026-10-01

Commit `d88f37c`. Documentation only.

### Changed

- Credits link [MovieMaker93](https://github.com/MovieMaker93) to their GitHub profile.

## [0.4.0] - 2026-09-29

Commit `a3aa898` (#4). TensorFold v0.3.6.3. Images `v0.3.6.3-c1f5d72f8d16` and `v0.3.6.3-0e8d365c1178`
(`:languages`).

### Added

- Up to 50 images a request (`TENSORFOLD_MAX_IMAGES`; a chat's turns all count), sharing 16,384 tokens
  (`TENSORFOLD_IMAGE_TOKENS`), each at most 4,096. The tower encodes runs of whole images within one full-size
  image's scratch. Request bodies may be up to 96 MiB.
- Language draft vocabularies for MTP (de, fr, ja, pt, ru, zh), from
  [jvr0x](https://github.com/jvr0x)'s lists for the vLLM recipe. Opt-in as a second image,
  `tensorfold-qwen38:v0.3.6.3-languages`, which `start.sh` serves when `DRAFT_LANGUAGE` is set. Output is
  byte-identical. Measured: Chinese +29% / +32% (thinking off / on), Japanese +19% / +7%.
- `scripts/config.sh` reads `./.env` (`KEY=value` lines, parsed, never executed; the environment wins).

## [0.3.0] - 2026-09-29

Commit `856bb6b`. TensorFold **v0.3.6.3**. Image `v0.3.6.3-5f313914582d`.

### Added

- Image and video input (`--vision`, on by default): the checkpoint's own vision tower, interleaved 3-D rotary
  positions, and video as timestamped frame groups. Prompt chunks drop to 2,048 rows to make room for the tower,
  whose scratch is borrowed only while encoding, so the KV pool stays 5 x 262,144.
- `tools/visioncheck.py`.

### Changed

- TensorFold v0.3.6.2 to v0.3.6.3. Typed tool parameters are upstream (TensorFold #75), so that patch is gone and
  the speed patches are renumbered 0001-0007. Replies unchanged.

## [0.2.0] - 2026-09-29

Commit `4cd9956`. TensorFold v0.3.6.2. Image `v0.3.6.2-82e893ed2bcc`.

### Changed

- Default pool 5 streams x 262,144 tokens (1,310,720-token KV pool), up from 4. Measured: 119.3 tok/s aggregate
  prose decode with 5 concurrent requests, 27.0 per request.
- Qwen's thinking-mode sampling is pinned: temperature 1.0, top_p 0.95, top_k 20.
- `start.sh` does everything: it runs `scripts/prepare.sh` when needed (pulls the prebuilt image, else builds;
  downloads and checks the checkpoint), shows progress and the server's log, and runs a smoke test.
  `./start.sh restart` checks new arguments before it stops the running server.

### Added

- `CREDITS.md`, and `LICENSE` with TensorFold's notice for the patches.

### Fixed

- No Hugging Face token in the container or on command lines; no orphaned log follower.
- `HOST` is honoured by the readiness checks, and the disk check counts the image.
- Correct exit codes with `FOREGROUND=1`, and a start lock.
- The native SSD reader patch reads `errno` before locking.
- The tools run on Python 3.11 and take `API_URL`.

## [0.1.0] - 2026-09-29

Commit `7893602`. TensorFold **v0.3.6.2**. Image `v0.3.6.2-b3fd6cff9b72`.

### Added

- `start.sh` serves `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` on one DGX Spark: 4 x 262,144 context,
  OpenAI-compatible API on port 8888. `stop.sh` stops it.
- Eight patches over TensorFold: typed tool parameters, live token counters, SSD read-ahead and a native SSD reader
  for the n-gram tables, tiled sparse-attention select, stream draft stats, configurable prefill rows (a port of
  TensorFold #40 by [MovieMaker93](https://github.com/MovieMaker93)), and copy drafts. Prefill ~1.7x, decode +4%,
  byte-identical output.
- `scripts/prepare.sh` builds the image and downloads the checkpoint; `scripts/publish-image.sh` pushes the image
  to GitHub Container Registry; `scripts/config.sh` holds every setting.
- `tools/bench.py`, `tools/needle.py`, `tools/toolcheck.py`.
- `.github`: Sponsors, issue and PR templates.

[1.0.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/releases/tag/v0.1.0
