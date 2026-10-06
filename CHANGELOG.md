# Changelog

All notable changes to this recipe. Recipe versions are this repository's own; the TensorFold version each one
runs is named in its entry. Versions before 0.6.0 were numbered afterwards, from the git history: the commit each
release ends at is given, and the prebuilt images are
`ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:<tag>`.

## [0.7.0] - 2026-10-06

TensorFold **v0.6.6** (`cb2ebf0`). Checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (`7c4f1bc1`).
Images `v0.6.6-35a2e85de8ad` (`:latest`) and `v0.6.6-7c0f3be0bb7e` (`:languages`); the prebuilt tags are not pushed
to GHCR yet, so `scripts/prepare.sh` builds the image locally on the first run.

### Changed

- The checkpoint is now **NVIDIA ModelOpt NVFP4**
  ([`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)):
  FP4 routed experts and n-gram rows, MXFP8 elsewhere, the MTP head kept. About 106 GB on disk; the ~78 GiB of GPU
  weights replace the MLX checkpoint's 75.2 GiB resident + 29.8 GiB SSD tables, whose memory goes to the KV pool
  instead. NVIDIA's own [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
  shows the format holds quality against FP8 (GPQA 91.5 against 92.0 and friends). The MLX 4-bit checkpoint stays
  available as `MODEL_ID=TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` — still the only format two ranks and
  `--ple-on-ssd` read, and the reason `PLE_ON_SSD` now defaults to 0.

  Why NVFP4 rather than staying on MLX: the "MLX 4-bit decodes slower" reports traced back to
  [#317](https://github.com/ashhart/TensorFold/issues/317), whose author retracted the regression (it was a cold
  n-gram page cache plus the lone-stream graph slot being recaptured on short replies, found by @Arminova and fixed
  by @grearjake-star in [#337](https://github.com/ashhart/TensorFold/pull/337)). Upstream's answer at the end of that
  thread is decisive: the Python engine is frozen (#286), so #337 ships only in the Zig engine, where Flash Next on
  CUDA is the next family after Nemotron. NVFP4 is where upstream's work is, and
  [#447](https://github.com/ashhart/TensorFold/issues/447) measures this exact checkpoint.
- TensorFold v0.6.1 to v0.6.6. Upstream absorbed the `TENSORFOLD_PREFILL_ROWS` override (#238) and the Flash Next
  video input (#240), so `patches/0002-flash-next-v066.patch` carries only copy drafts, the SSD read-ahead, the
  first-token-before-draft order, the absorb-logits skip and the 96 MiB request bodies. New upstream knobs the
  recipe passes through: `--vision-image-tokens` (was the `TENSORFOLD_IMAGE_TOKENS` env), `--prefill-fp8` (FP8
  prompt kernels for this checkpoint's MXFP8 linears), `--api-key`, `tensorfold plan`, and `--name-priority`.
  NVFP4 checkpoints run on one GPU: `--tp 2` and `--ple-on-ssd` refuse them at startup.
- The language image's patch (`patches/languages/0010`) is rebased onto v0.6.6; the token lists are byte-identical.
  `DRAFT_LANGUAGE` serves it as before.

### Fixed

- The `TENSORFOLD_IMAGE_TOKENS` budget is passed as `--vision-image-tokens` (the old env name stopped being read
  upstream); the video token budget env is unchanged.

Measured 2026-10-06 on the new defaults, one DGX Spark, through the OpenAI API: single-stream decode 54.3 tok/s code
(greedy), 49.3 tok/s chat sampled; concurrent aggregate 31.6 / 51.1 / 73.6 tok/s at 1 / 2 / 4 streams with thinking
on (acceptance 0.62-0.67); prefill 1,215-1,835 tok/s from 0.85k to 50k tokens, i.e. the same ~1.8k plateau the MLX
recipe had at the mid sizes but a lower 0.85k number, and still the ~1.4-1.5x behind vLLM that
[#447](https://github.com/ashhart/TensorFold/issues/447) reports.
Startup estimate 83.07 GiB within a 111.71 GiB budget, leaving 34.8 GiB for stream caches against 5 x 4.47 GiB — the
full 5 x 262,144 pool fits with room to spare (the MLX checkpoint needed 102.5 GiB and its 29.8 GiB of n-gram tables
on SSD). First load 254 s, of which ~171 s is the one-time GB10 kernel compile; later starts reuse it.

Verified on the same boot: drafted replies equal `"draft": false` (identical `token_sha`), 5 concurrent streams each
completed, a 26.7k-token needle was retrieved, `tools/visioncheck.py` and `tools/toolcheck.py` both pass.

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

[Unreleased]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/releases/tag/v0.1.0
