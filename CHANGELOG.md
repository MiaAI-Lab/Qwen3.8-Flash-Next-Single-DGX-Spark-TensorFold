# Changelog

All notable changes to this recipe. Recipe versions are this repository's own; the TensorFold version each one
runs is named in its entry. Versions before 0.6.0 were numbered afterwards, from the git history: the commit each
release ends at is given, and the prebuilt images are
`ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:<tag>`.

## [Unreleased]

## [0.6.0] - 2026-10-02

Commit `5840c51` (#13). Images unchanged from 0.5.0.

### Changed

- A request without `max_tokens` now gets 32,768 tokens (`MAX_TOKENS`), not TensorFold's 4,096. A thinking reply
  could use all 4,096 before it answered and end with no content or tool call. The engine clamps the value to the
  room left in the stream's window, so the startup estimate is unchanged; a request's own `max_tokens` wins. Ported
  from #11 by [MovieMaker93](https://github.com/MovieMaker93).
- README: decode (prose and code) and prefill tables measured on TensorFold v0.6.1.

## [0.5.0] - 2026-10-02

Commit `92f1e23` (#12). TensorFold **v0.6.1** (`17c73e1`). Images `v0.6.1-797af1d9df4d` (`:latest`) and
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

Commit `c506d4c`. Documentation only.

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

[Unreleased]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/releases/tag/v0.1.0
