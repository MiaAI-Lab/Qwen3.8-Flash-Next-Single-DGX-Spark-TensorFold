# Changelog

Newest first. This recipe serves one DGX Spark. The image is `tensorfold-qwen38:zig-db28187`, built by
`scripts/prepare.sh` from TensorFold `db281878` plus `patches/`.

## Unreleased

**Triage of the open pull requests and issues (2026-10-25): two ports, one new setting and README notes.**

- `patches/0021-flash-next-sliced-fill-one-at-a-time.patch`, from [BadAd84](https://github.com/BadAd84)'s PR #53 and hand-rebased onto the FP4 engine tree: with
  `TF_FLASHNEXT_FILL=1` (without `FILL_FAIR`), prompts that arrive together are filled one after another; before, a second fill returned
  `PromptBuffersBusy` and every filling prompt ended with that error, and a resume's MTP absorb could overwrite another prompt's open chunk. The resume's
  absorb now runs when its own chunk opens. It does not fix #71 (one filling prompt, `CudaFailed`), which has a different trigger.
- `patches/0031-server-random-seed.patch`, new, for #75 and #68: `TENSORFOLD_RANDOM_SEED=1` makes a sampled request without a `seed` draw a fresh one, as
  OpenAI's API does. Off by default (the engine keys an omitted seed to the prompt, which keeps the `exact.py` and `"draft": false` comparisons
  reproducible); an explicit `seed` always wins. Two server tests; they fail without the change.
- Patch 0030 (below) was merged with review fixes: its image key moved to `server/media_match.zig` with length-prefixed fields, a wider identity
  (up to 248 bits) and no key for runs under 4 rows, with four tests, and `zig build test` now also runs the core `engine_api` tests (lane host, prompt
  cache), which were not built before. Both 0029 and 0030 were credited to [tempeste](https://github.com/tempeste).
- README: the NV_ERR_NO_MEMORY section corrected and extended by [hecisaza](https://github.com/hecisaza) (PR #64); a "Thinking and sampling" list on
  `thinking_budget`, the prompt-keyed default seed, the ignored penalty fields, control tokens in the reasoning and `tool_choice: "required"` without
  thinking (#7, #68, #74, #75, #76).
- Not merged: structured output (#69, #66; a new xgrammar build dependency and compile limits, left for a decision) and the stacked engine patches
  0010 to 0028 of BadAd84 (#35 to #63, written against the INT4 build: they need a rebase onto patches 0011 to 0015, a new number and a re-measure on the
  NVIDIA FP4 checkpoint). `zig build test` on the pinned TensorFold `db281878` with patches 0001 to 0031 applied in order: 235 of 235 tests (clean cache; no GPU, so the kernels and
  the byte-identical claims of 0021, 0029 and 0030 rest on their authors' and reviewers' runs).

**Image prompts resume the prompt cache: the next turn after a screenshot prefills only its new rows (#5).**

- `patches/0030-flash-next-image-prompt-reuse.patch`: a prompt with an image was neither kept nor resumed, so every turn of a conversation with an
  image prefilled from its first token, the long system prompt included. Image placeholder rows were all the same token, so a token match could not
  tell two images apart. Each image's rows now carry match ids from a SHA-256 over its data URL bytes, its detail and its rotary grid (an https
  image's hash also takes its feature rows, as the URL's content may change), so the cache resumes up to the first row where two prompts differ, image
  or text. On resume the rotary table and the features of the kept rows are attached before the prompt continues; each prompt segment gets only the
  images before its cut. Videos are not kept.
  - A segment whose images are refused for memory at a kept mark ends that request alone; its fill is freed once.
  - A deferred fair-fill begin (`TF_FLASHNEXT_FILL_FAIR=1`) resumes only the kept state it saw (its generation); a state evicted meanwhile means a
    prefill from 0. Works with `TF_FLASHNEXT_FILL_FAIR` off and on.
  - `TF_FLASHNEXT_MEDIA_REUSE=0` restores the old rule. One rank only.
- Before and after, same server settings (defaults, NVIDIA FP4 checkpoint, 5 x 262,144 fp8), a 12.2k-token system prompt, temperature 0:

  | request | prompt | 0015: cached, prefill | 0030: cached, prefill |
  |---|---:|---:|---:|
  | text only, cold | 12,226 | 0, 6.56 s | 0, 6.58 s |
  | image A, first turn | 12,528 | 0, 6.24 s | 11,687, 0.93 s |
  | image A, next turn | 12,614 | 0, 6.29 s | 12,521, 0.35 s |
  | image B in place of A | 12,614 | 0, 6.28 s | 12,199, 0.72 s |
  | text only, next turn | 12,260 | 12,219, 0.25 s | 12,219, 0.28 s |
  | image A turn repeated | 12,614 | 0, 6.30 s | 12,607, 0.19 s |

  Every reply has the same token hash on both builds (image B is read as its own text, not image A's). With `TF_FLASHNEXT_FILL_FAIR=1` the hashes and
  cached counts are the same; with `TF_FLASHNEXT_MEDIA_REUSE=0` every image request is cached 0, as on 0015.
- Checks: `tools/exact.py` passes with the same token hashes as 0015, `tools/toolcheck.py` 3/3, `tools/visioncheck.py` (also with fair fill and with
  reuse off), `tools/prompt_reuse.py 30000` (worst follow-up resumed 99.3%). `zig build test`: 203 of 203 (202 on 0015). Prefill and decode are
  unchanged: 100k prompt 1,969 tok/s (1,974 on 0015), decode code 49.3 tok/s, chat 42.1 (the same).

**Free after drain and admission reclaim: no more `CUDA_ERROR_ILLEGAL_ADDRESS` after a refused prompt, and kept prompt states give way to new requests.**

- `patches/0029-flash-next-free-after-drain-admission-reclaim.patch`, three parts:
  - Free after drain (always on): `Region.deinit` unmapped a sequence's memory without waiting for the GPU, so a sequence
    freed while its prompt kernels were still queued (a prompt refused for room mid-fill, a kept state evicted during its fill
    copy) could kill the CUDA context with `CUDA_ERROR_ILLEGAL_ADDRESS`. Freeing a sequence or a kept state now drains the context
    first (`Engine.drain`, `freeSeqDrained`). The toy engine models queued work, and two new unit tests fail without the drains.
  - Admission reclaim (`TF_FLASHNEXT_ADMIT_RECLAIM`, on): kept prompt states share the sequence pool with live requests, and a
    prompt that did not fit was refused with `NoRoom` even when idle kept states held the room. Admission now frees kept states
    (superseded first, then least recently used, never the state the prompt resumes or one a lane holds), and only when that
    makes the prompt fit. `0` refuses as before.
  - Supersede (`TF_FLASHNEXT_SUPERSEDE`, on): a conversation's next turn frees the kept state it resumed once it keeps a longer
    one, so the cache holds about one state a conversation instead of one a turn. `0` turns it off.
  - The startup log says `prompt cache: supersede on|off, admission reclaim on|off`. One rank only.
  - Kept states are told apart by an entry id and fill copies by a generation, so a deferred fair fill (`TF_FLASHNEXT_FILL_FAIR=1`)
    never begins from a state that was freed and reused.
- NoRoom repro: three long conversations (30k, 100k and 100k tokens, two turns each) fill the prompt cache, then 5 concurrent ~2,000-token
  prompts with `max_tokens` up to the window. `./start.sh --prompt-cache-gib 12`, otherwise defaults: 0015 admitted 3 of 5 and refused 2 with
  HTTP 500 `NoRoom`, in each of two runs (8 states, 11,854 MiB kept). 0029 admitted 5 of 5 in both runs, freeing 1 and then 4 kept states, with 0 `NoRoom`
  and 0 CUDA errors. At the default cache budget (a quarter of the pool) five full windows plus the cache still fit, so neither build refused.
- Checks (defaults, NVIDIA FP4 checkpoint, 5 x 262,144 fp8): `tools/exact.py` passes with the same token hashes as 0015 (6 drafted replies, 495 drafts
  accepted, equal to `"draft": false`; concurrent equal to serial), before and after the other checks. `tools/toolcheck.py` 3/3, `tools/visioncheck.py`,
  and `tools/prompt_reuse.py 100000 4 --other` pass (worst follow-up resumed 99.8%). `zig build test`: 206 of 206 (202 on 0015).
  Prefill and decode are unchanged: 100k prompt 1,973 tok/s (1,974 on 0015), 50k 2,062 (2,057), decode code 49.2 tok/s (49.3), chat 42.1 (42.1).
- On a DGX Spark serving several long-context coding agents, 19 hours with an earlier form of the same change: 0 `NoRoom`, 0 CUDA errors, 66 kept
  states freed by admission and 6,166 superseded.
- Open pull requests already claim patch numbers 0016 to 0028, so this one is 0029.

**Faster prompt prefill on the NVIDIA FP4 checkpoint (+13 to 15%), and an opt-in fair prefill for mixed loads.**

- `patches/0014-engine-prompt-expert-tile.patch`: the routed experts of prompt chunks of 1,024 rows or more run through a new shared-memory kernel
  (`fn_experts_tile.cu`: 64 token-expert pairs a CTA, activations staged once in shared memory, weights by `cp.async`; same arithmetic and rounding as the
  older kernels, so the output bits are identical). On by default: `TF_FLASHNEXT_PROMPT_EXPERT_TILE=2` (gate/up and down; `1` gate/up only; `0` the older
  kernels); `scripts/config.sh` exports it. Two ranks keep the older kernels unless it is set. Decode and chunks under 1,024 rows are untouched.
  Paired alternating runs, tile off and on, 3 server starts each, median of 3 prompts: 4k 2,090 to 2,397 tok/s (+14.7%), 16k 2,174 to 2,474 (+13.8%),
  32k 2,161 to 2,450 (+13.4%). sparkDash: 4k 2,325 to 2,665 tok/s, 16k 2,191 to 2,501, 128k 1,947 to 2,197. Five 100k-token prompts in 250 s (265 s).
- `patches/0015-engine-fair-prefill.patch`, off by default (`TF_FLASHNEXT_FILL_FAIR=1`; `TF_FLASHNEXT_FILL_LAYERS`, `TF_FLASHNEXT_FILL_AGING`): prompts are
  filled in two-layer slices, one owner of the prompt buffers handed on at chunk boundaries, shortest prompt first with aging. Five 100k prompts, a running
  decoder and a short request every 20 s: shorts wait 1.8 s median and 3.9 s worst (187 s and 256 s before), the decoder's longest gap 0.27 s (12 to 15 s per
  long prompt, up to 218 s). Costs about 20% of the total time of such a mixed load; idle prefill is unchanged. One rank only. The unit tests compile again
  (the round-log commit had broken `zig build test`).
- Checks on the default build (tile on): the 7-prompt greedy test has the same token hashes as before (and as with the tile off), drafted equal to
  `"draft": false` and concurrent equal to serial (`tools/exact.py`), teacher-forced 2,270 of 2,295 (98.91%), `tools/toolcheck.py` 3/3, `tools/visioncheck.py`,
  a 194,723-token needle (prefill 99.4 s), five 100k prompts plus six shorts without a crash or hang, all answers right. With `TF_FLASHNEXT_FILL_FAIR=1`:
  the same 7-prompt hashes, exact.py, toolcheck, visioncheck and needle pass, no `PromptBuffersBusy`.
- README Performance tables re-measured with sparkDash on the final build; decode is unchanged (prose 1 stream 48.5 tok/s, 5 streams 132.7; code 67.9 and 149.7).

**The default checkpoint is now NVIDIA's QAD FP4 export; INT4-AutoRound is optional.**

- `./start.sh` serves `nvidia/Qwen3.8-Flash-Next-FP4-Local` (NVIDIA's quantization-aware-distilled FP4 checkpoint, close to bf16: 91.5% GPQA Diamond and
  88.7% ToolEvalBench against 92.2% and 89.4% for bf16, per its model card). `CHECKPOINT=int4` (or `CHECKPOINT=int4` in `.env`) serves azampatti's
  INT4-AutoRound as before. New README section "Choosing the checkpoint (FP4 or INT4)" explains the switch, the downloads, the memory each needs and how to
  check which one is running. `scripts/config.sh` has a per-checkpoint pin, download size, `WEIGHTS_GIB` (81 GiB measured for the NVIDIA one, 66 for INT4)
  and label; `prepare.sh` verifies either format; `start.sh`'s memory check and banner follow the checkpoint (about 108 GiB `MemAvailable` needed for FP4,
  93 for INT4). `PARALLEL` is 5 by default (was 8, for INT4): five full 262,144-token windows fit the 24.36 GiB pool the FP4 checkpoint leaves.
- `patches/0012-engine-nvidia-fp4.patch`: the Zig engine loads and runs the NVIDIA checkpoint: the config and `quantized_layers` it declares, the
  CSF-compressed NVFP4 routed-expert scales (new `cuda_csf.zig`), MXFP8 attention, DeltaNet, indexer and shared-expert linears (qmmf mode 2 for
  decode rows, the prompt GEMM for prompt rows, byte-equal to the Python implementation on real layers), fp32 hyper-connection and n-gram norms, NVFP4 MTP
  experts, and the 26.8 GiB NVFP4 n-gram table (new `cuda_ssd_table.zig`: a pread pool with dedupe, rows bit-equal to the Python table) with read-ahead of the
  next prompt chunk's rows (`TF_FLASHNEXT_WARM_AHEAD=0` off; +3.5% prefill). The safetensors index read limit is raised.
- `patches/0011-vision-nvidia-fp4.patch`: the vision helper reads this checkpoint's quantized tower (MXFP8 attention, NVFP4 MLP).
- `patches/0013-engine-tune.patch` (and `scripts/config.sh`): `TF_FLASHNEXT_DEPTH` 6 and `TF_FLASHNEXT_PRODUCT_STREAMS` 5 (the engine's 15 drafts a round cost
  20 to 28% at 3 to 5 streams); a prompt the server has served before keeps a state one token short of its end, so a repeat prefills one row
  (`TF_FLASHNEXT_REPEAT_CACHE=0` off); an idle server waits 3 ms for requests sent together (`TF_FLASHNEXT_ADMIT_WAIT_MS`), so a burst prefills in one pass and
  decodes in step instead of staggered; `TF_FLASHNEXT_ROUND_LOG=1` logs a round's time by part. Greedy output is unchanged by these.
- Checks on the NVIDIA checkpoint: the 7-prompt greedy test is identical across drafted, serial and repeated runs; teacher-forced against our Python build the Zig
  engine picks the same next token at 98.9% of 2,295 positions (the disagreements are near-ties); tool calls, image input and a 194,723-token needle pass; five
  100k-token prompts plus six short ones at once ran without a crash or hang (short requests wait for the long prefills, since prefill is serial).
- Measured on the Spark itself with [sparkDash](https://github.com/MiaAI-Lab/sparkDash): prose decode 46.2 / 72.8 / 88.1 / 106.5 / 132.1 tok/s and code decode
  66.7 / 101.9 / 117.8 / 133.7 / 150.1 tok/s at 1 to 5 requests; prefill 2,325 tok/s at 4k to 1,947 tok/s at 128k (TTFT 1.95 s to 67.5 s). The README's
  Performance section leads with these; the INT4 numbers follow below them, and `tools/bench_streams.py` (new: decode at 1 to 5 requests, prose and code, and
  prefill 4k to 128k, median of 3) gives a second set. The KV and memory section now states the FP4 pool: 24.36 GiB, about 1.46 million fp8 tokens.
- README, `CREDITS.md` and `NOTICE` describe the NVIDIA checkpoint (NVIDIA Open Model Agreement; the base model's Qwen Community License 1.0 applies too) and keep
  INT4-AutoRound as the optional checkpoint. Credits for the NVIDIA FP4 support in the Zig engine go to the TensorFold Contributors and MiaAI-Lab.
- `scripts/prepare.sh` takes `DOCKER_BUILD_ARGS` (empty by default; e.g. `--network host` on a machine where Docker builds cannot resolve DNS).


Script and tool fixes:

- `start.sh` reads `MemAvailable` from `/proc/meminfo`. It used `free -g | awk '/^Mem:/'`, which came back empty on
  non-English systems (`Speicher:`, `内存：`) and refused to start. Reported by
  [shuaixn](https://github.com/shuaixn) (#37) and [Siphi1239](https://github.com/Siphi1239) (#39). Test:
  `tools/test_start_meminfo.py`.
- `tools/bench.py` reads the 1.0.0 stream stats (`usage` arrives in its own chunk; `prefill_seconds`,
  `tokens_per_second`, `usage.prompt_tokens_details.cached_tokens`) and keeps the 0.6.x names as a fallback. Its
  prompt-cache guard defaulted to 0 and no longer passed for the right reason. Reported by
  [naeemakhtar110](https://github.com/naeemakhtar110) (#38). Test: `tools/test_bench_stats.py`.
- The ready summary of `start.sh` names the KV cache type in use (it always said `bf16 KV`). Authored by
  [mcclanahanaman](https://github.com/mcclanahanaman) (#51).

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
