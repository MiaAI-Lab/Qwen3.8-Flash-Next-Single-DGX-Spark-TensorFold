**Decode prose and prefill: investigation and action plan**

**Status, 2026-10-02.** The recipe now runs TensorFold v0.6.1 with `patches/0002-flash-next-v061.patch`; see
`docs/v061.md` for the port and its measurements. The rest of this note describes the v0.6.0 state.

**Status, 2026-10-01.** The running default is TensorFold v0.6.0 (`c464617`), image `tensorfold-qwen38:v0.6.0`, patch `patches/0002-flash-next-v060.patch`. Admission: estimate 85.18 GiB within 108.20 GiB, vision on, up to 5 streams that grow toward 262144 (26.7 GiB left for caches at startup, 4.47 GiB quoted for one full window), eager, 0 graphs, `TENSORFOLD_PREFILL_ROWS=2048`, `TENSORFOLD_MTP_COPY=1`, `TENSORFOLD_MEMORY_RESERVE_GIB=2`. `/health` ok. Vision check and tool check passed. The language patch is not rebased, so `DRAFT_LANGUAGE` builds are not usable. Native SSD workers and cross-stream PLE batching from the v0.5.0 series are not in this patch. JSONL: `/tmp/astra-overnight/v060-decode.jsonl`, `/tmp/astra-overnight/v060-prefill.jsonl`.

One paired sample against the v0.5.0 baseline in `docs/astra-overnight.md`, same seed and prompts, thinking on. Not three boots.

| | v0.5.0 | v0.6.0 |
| --- | ---: | ---: |
| C=1 per-request decode | 54.7 tok/s | 49.4 tok/s |
| C=1 TTFT p50/p95 | 217 / 411 ms | 244 / 404 ms |
| C=1 accept / yield | 0.643 / 2.05 | 0.641 / 2.07 |
| Prefill 7,730 tok | 2,153 tok/s | 2,024 tok/s |
| Prefill 30,748 tok | 2,302 tok/s | 2,089 tok/s |
| Prefill 61,373 tok | 2,233 tok/s | 2,188 tok/s |

The C=5 arm on v0.6.0 reported median `cached` 90 of 91 prompt tokens. v0.6.0 keeps the prompt snapshot one token early, so the same prompt run at C=1 was reused. That C=5 figure (aggregate 90.1 tok/s) is a warm-prefix run, not an uncached comparison to the v0.5.0 C=5 aggregate of 90.9 tok/s.

One mixed run: a 256-token decode and, 0.6 s later, an uncached 15,422-token prefill (7.78 s). Decode gaps during that window: 9 gaps, maximum 1.84 s. The decode did not stop for the whole prefill. One sample, not a pause budget.

Items 6 and 7 below are not done. Item 5's interleaving is the upstream v0.6.0 behavior measured once above, not a local scheduler. The v0.5.0 sweep (no promotion of 4096 rows, MTP 6/0.60, copy on) is in `docs/astra-overnight.md`.

---

Investigated on 2026-09-30 against recipe commit `a3aa898`. The best next steps are to establish a reproducible benchmark, fix the mixed copy/MTP draft-row mapping, measure the single-stream graph path, and remove avoidable work from prompt ingestion. The largest architectural opportunities are graph replay for concurrent decoding and scheduling prompt chunks between decode rounds. Increasing prompt chunks alone is a small, already-known optimization; it will not resolve long pauses during mixed traffic.

This is a source investigation with CPU checks, not a new model-throughput benchmark. No speedup below is presented as newly measured on the checkpoint. The recipe's configuration and patches were left unchanged.

**Upstream follow-up: migrate deliberately to v0.5.0 before substantial new runtime work.** Checked on 2026-09-30: the latest published release is [TensorFold v0.5.0](https://github.com/ashhart/TensorFold/releases/tag/v0.5.0), released September 29. Its peeled tag and upstream `main` both resolved to `9cd52ab4daba68ddd09be89be8f23ad43175e821` during this check. The recipe remains pinned to v0.3.6.3; this investigation does not change the deployed version. There is enough useful maintenance, API and loading work to justify a migration, but no evidence that a version bump alone materially improves short-context prose over this already-patched recipe. Do the migration before developing large graph/scheduler changes against the older interfaces, and keep the current image as the comparison and rollback baseline.

I compared the tagged source, not just release descriptions. The applicability to this recipe is:

| Upstream addition | Value for this Flash Next recipe | Qualification |
| --- | --- | --- |
| Faster CUDA loading, commits `8ae247f` and `da8a5df` | Direct/aligned reads, read-ahead during layer packing, GPU expert packing, fewer allocator flushes, and release of pinned loader staging can shorten startup | The Flash Next loader is changed. Measure load time and peak host memory; this is not an established steady-state decode/prefill speedup |
| QSA tiled selection, commit `a2dba9c` / PR #93 | Maintained replacement for local patch 0004 | The recipe already solves the large-register-row problem. Upstream additionally chooses 4,096-block/8-warp tiles for 64+ rows and 8,192-block/16-warp tiles for smaller windows, and exits the listing pass once all selected blocks are placed; benchmark it against the local implementation |
| Health and draft counters, commits `51b098d` and `c4bf25f` | Retire local patches 0001 and 0005; richer request, cache, decode, draft and stream telemetry | Accounting has changed; retain upstream semantics and update benchmark/monitor expectations |
| Responses API, grammar-constrained output, request parsing/sampling updates | Better client compatibility and structured-output support; Flash Next's decoder now handles grammar constraints and per-request `ignore_eos` | These improve functionality, not proven prose token/s. Requests using previously ignored fields may now produce different output |
| CUDA background priority | Background generations can yield a slot to waiting foreground work and later replay | Flash Next still completes an already-started prompt's prefill before yielding; this does not implement the proposed chunk scheduler |
| Dense 27B long-context attention acceleration | Useful upstream design examples | The large Spark decode/prefill gains advertised for Qwen3.8-27B use a different attention/runtime path. They are not Flash Next measurements |
| v0.4.0 Flash Next speedups, mixed quantization, and larger serving pools on Macs | Potential design references | Those changes target the MLX/Metal backend, not this CUDA MLX-4-bit-checkpoint runtime. A checkpoint named “MLX” does not make its CUDA execution use Metal optimizations |

The useful source changes can be inspected in the [loading commit](https://github.com/ashhart/TensorFold/commit/8ae247f), [loader cleanup](https://github.com/ashhart/TensorFold/commit/da8a5df), [QSA implementation](https://github.com/ashhart/TensorFold/commit/a2dba9c), [health implementation](https://github.com/ashhart/TensorFold/blob/9cd52ab4daba68ddd09be89be8f23ad43175e821/src/tensorfold/cuda/health.py), and [Flash Next decoder changes](https://github.com/ashhart/TensorFold/compare/v0.3.6.3...v0.5.0). In particular, the new `cuda/direct_read.py` loader uses `O_DIRECT` where supported; Flash Next's inference-time `ssd_table.py` still uses buffered reads. Do not confuse faster checkpoint loading with replacement of local patches 0002/0003 or removal of inference-time page-cache effects. The new host-table prefetch overlap is also conditional on **not** using `--ple-on-ssd`, so that part does not benefit the recipe default.

The earlier optimization findings largely survive this upgrade. In v0.5.0, Flash Next's concurrent decoder still calls eager `stage`/`compute`; its `graphs.py`, `mtp.py`, and `forward.py` are unchanged from upstream v0.3.6.3. The special “one active stream uses graphs under parallel serving” change belongs to `qwen3_5_moe`, not `qwen4_exp`. Flash Next still builds its initial draft before emitting the first token and still computes unused MTP logits while absorbing prompts. It still lacks local copy drafting, SSD prefill read-ahead/native gathering, configurable Flash Next prefill rows, and Flash Next vision/video support. The [v0.5.0 API documentation](https://github.com/ashhart/TensorFold/blob/9cd52ab4daba68ddd09be89be8f23ad43175e821/docs/api.md) explicitly distinguishes Flash Next's blocking prefill from the incremental background-prefill support in other families.

**Upgrade patch inventory and actual compatibility checks.** A plain `TF_VERSION=v0.5.0 ./start.sh` is not a working migration. `prepare.sh` applies every default patch and stops on failure; patch 0001 already fails against v0.5.0. In a scratch source tree, after omitting the upstreamed patches, I successfully applied **0002 → 0003 → 0006**, then reproduced a failed hunk in 0007. Some successful hunks used offsets/fuzz, so even those patches should be regenerated against the new base. Separate pristine-tree dry runs also found conflicts in 0008/0009 and the language patch; those depend on earlier changes and are indicators for manual integration, not a completed dependency-aware rebase.

| Local patch | v0.5.0 migration action |
| --- | --- |
| 0001 live counters | Remove; use upstream `cuda/health.py` instead of merging duplicate counters |
| 0002 SSD read-ahead | Retain and regenerate; applied in the scratch sequence |
| 0003 native SSD reader | Retain and regenerate after 0002; applied in the scratch sequence |
| 0004 QSA tiled selection | Remove; qualify upstream's selector against the current recipe and its exact block-list tests |
| 0005 stream draft stats | Remove; upstream counts drafted rows in `counted()` and accepted rows in `take()`. Reapplying local counting risks incompatible signatures or double accounting |
| 0006 prefill rows | Retain and regenerate; applied in the scratch sequence |
| 0007 copy drafts | Manually merge with grammar/EOS changes; include the proposal-row fix documented below, rather than reintroducing the defect |
| 0008 vision/video | Manually merge with new `constraint`, `background`, `stop_eos`, and sampling paths; plain upstream still rejects Flash Next `--vision` |
| 0009 many images/body limit | Rebase after 0008. HTTP handling moved to `cuda/http.py`; port the 96 MiB request-body limit there, where upstream currently uses 32 MiB |
| languages/0010 | Rebase after the new default patch stack; qualify language-image selection and preserve the empty/default English setting |

There is an extra interaction to guard when combining the new background scheduler with the local vision patch: the scheduler checks `s.vision` before permitting replay, while local admission clears that payload after encoding. Preserve an explicit image-request marker or otherwise prevent image streams from being treated as text-only replay candidates. Replaying image placeholder IDs without their image data is not a valid continuation. This is an integration risk inferred from the two implementations, not a failure observed in a completed v0.5.0 port. Test image plus background priority, grammar plus vision, and mixed copy/non-copy constrained requests during the rebase.

A concrete migration sequence is:

1. Keep the v0.3.6.3 image and patch hash intact. Create a separate v0.5.0 patch set/image pinned to the inspected tag/commit, retain the existing base container and checkpoint, and regenerate the patches in the order above. Build with `PULL=0` during development; record the resulting image ID and patch hash. Do not use a floating upstream branch as a release pin.
2. Merge current upstream functionality rather than overwriting it with old patched functions. Especially retain grammar masking/advancement, per-request EOS handling, priority/replay behavior, and updated sampling fields. Finish the default image first, then the optional language image.
3. Add the fixed-input benchmark and use existing `tensorfold.token_sha` and `return_token_ids:true` for parity. Both are already present in **v0.3.6.3** as well as v0.5.0; no new token-hash hook is needed. Use full returned IDs when diagnosing a mismatch because `token_sha` is truncated to 12 hex characters.
4. Make prompt rendering and sampling explicit in A/B requests. The CUDA CLI's default `--reasoning-effort` changes from `medium` to the template's default (`None`), and upstream now handles `min_p`, thinking budgets and top-level effort fields. Compare identical rendered token IDs for performance attribution. Separately test the new intended API behavior; do not call a changed request interpretation a numerical regression.
5. Run the CPU suite, then compiled-reader/kernel tests, the full-model parity matrix, long-context QSA boundary tests, five-stream memory stress, and image/video checks from this report. Include cold/warm startup and peak host memory for the new loader. Test the upstream QSA implementation against the old local selector, not merely against an unpatched old release.
6. Compare the complete migrated recipe against the complete current recipe. Isolate the copy-row correction in an additional arm, or apply it to both comparison arms, so an improvement is not incorrectly attributed to the upstream release. Promote v0.5.0 only once functionality, memory headroom, outputs and performance pass; update configuration, image tags, README API caveats and credits together.

For the unmodified v0.5.0 checkout I ran the original four CPU test files plus `test_cuda_server_health.py`, `test_cuda_priority.py`, `test_responses_api.py`, and `test_direct_read.py`: **106 passed, 28 skipped**. PyTorch-dependent tests, including the direct reader, were skipped; these results do not validate loader CUDA behavior or a migrated recipe. No v0.5.0 serving image was built and no model benchmark was run. This follow-up establishes a useful upgrade target and its rebase work, not a finished upgrade.

**Evidence and limits.** I inspected the launcher, configuration, benchmark tools, all default patches, and the relevant patched TensorFold implementation. I fetched TensorFold tag `v0.3.6.3`, commit `191188075bca56a7c71074a79375eb4c1cb22e1c`, into a temporary checkout and successfully applied patches `0001`–`0009` in order. References to runtime functions below mean that revision plus these patches, not current upstream `main`.

The host is an aarch64 DGX Spark with GB10 and driver `580.159.03`. At inspection it had approximately 108 GiB `MemAvailable` and 158 GiB disk available. The recipe container/image and checkpoint were absent from the inspected Docker inventory and default Hugging Face cache. The documented default needs roughly 115 GiB available at startup. Downloading the approximately 106 GiB checkpoint and image would consume most of the remaining disk, and the full default was not ready to run with this memory availability. Existing services were left running. Host Python had NumPy and pytest, but not PyTorch or Triton; compiled CUDA/C++ behavior and end-to-end output parity remain untested here.

Validation completed:

- Applied all nine default patches to the exact upstream tag without failed hunks.
- Ran `python3 -m pytest -q tests/test_ngram_read_ahead.py tests/test_ple_ssd.py tests/test_cuda_stream_slots.py tests/test_cuda_geometry.py` in that checkout: **57 passed, 21 skipped**. These are CPU checks; the skipped checks do not establish CUDA correctness. The SSD suite primarily exercises the Python reader.
- Exercised the patched native gather's descriptor construction with a Python `pread` callback: empty input, duplicated/interleaved rows, and a reversed full synthetic table all matched the reference bytes. This checks descriptor/layout logic, not the C++ thread pool.
- Executed the actual `_draft_all` method extracted from the patched source with controlled MTP outputs. Three copy-selection arrangements preserved row ownership; the fourth reproduced the misalignment described below. A reproducible script is included at the end.
- Microbenchmarked the existing `CopyIndex` on synthetic token-ID sequences on this host, separately measuring initial lookup time and Python allocation peak. Results below concern CPU indexing only.
- Evaluated the patched capacity geometry with the checkpoint's text configuration to calculate memory deltas. These are admission-model calculations, not peak-RSS or GPU-allocation measurements.

**What the existing numbers establish.** The [README](../README.md#performance) reports 62.4 tok/s for one prose request and 119.3 tok/s aggregate for five. Those are different objectives: five requests improve aggregate output by about 1.91×, while reported per-request speed falls from 62.4 to 27.0 tok/s and TTFT rises from 152 to 528 ms. Aggregate throughput need not equal concurrency times the reported per-request statistic when request durations and measurement intervals differ. The benchmark must expose those intervals before comparing results.

The published prefill table reports roughly 2,500 tok/s at 8k–32k, 2,414 at 64k, and 2,200 at 128k. It used four streams and 4,096-row chunks. The current default is five streams, vision enabled, and 2,048-row chunks. The README reports a 4–5% prefill penalty for the vision/default-chunk configuration on its tested long prompts. Do not treat the historical table as a fresh five-stream baseline. Likewise, the historical comparison against unpatched v0.3.6.2 does not isolate each patch's benefit on v0.3.6.3.

Already implemented: prompt-chunk SSD read-ahead, native threaded reads, register-spill avoidance in long-context QSA selection, configurable prefill rows, copy drafts, and draft acceptance counters. The [configuration](../scripts/config.sh) also documents a recent MTP sweep: `6 / 0.60` already beat `6 / 0.30` by about 3% on prose. Repeating these as new recommendations would overstate the remaining easy gains.

**First repair the measurement.** [tools/bench.py](../tools/bench.py) is useful as a smoke benchmark, but cannot reproduce the performance table or reliably rank small optimizations:

| Current behavior | Consequence | Required change |
| --- | --- | --- |
| Requests run sequentially | No aggregate concurrency curve or mixed prefill/decode measurement | Barrier-start concurrent clients at 1, 2, 4, and 5 active requests; also test queued load above capacity |
| Prompt seed comes from wall time; sampled decode has no request seed | Different text and acceptance behavior between runs | Persist a prompt manifest and fixed request seeds, reuse the same inputs across configurations |
| Synthetic text draws from a small word list | PLE locality and expert routing may differ from natural prose | Keep this synthetic case, add held-out natural prose and varied RAG/document prompts |
| Assertion checks only `cached == 0` | It rules out reported prefix reuse, not cached SSD pages or repeated n-grams | Record engine prefix hits separately from OS page-cache conditions and PLE row reuse |
| Decode generates at most 256 tokens with thinking enabled | The measured output may be primarily reasoning, not visible prose | Label thinking mode; record first reasoning and first visible answer separately; use sustained 512–1,024-token outputs |
| Reports medians and discards most telemetry | No tail latency, uncertainty, acceptance distribution, or reproducibility artifact | Save each request as JSONL, including all server stats and streamed text/hash |
| Uses `time.time()` and substitutes completion time if no token appears | Wall-clock adjustment or an empty/error response can distort TTFT | Use `perf_counter()`; record empty output and SSE/API errors as failures |

Identical repeated decode prompts do warm storage/kernel paths. However, concurrent `_slot_for` reuses only a **strictly shorter** matching cached prompt, so an identical prompt is not automatically a prefix-cache hit. Test repeated identical prompts and genuinely extended conversations separately.

Add queue/admission start, prompt-compute end, first-token enqueue, draft start/end, target-forward time, sampling/copy time, and per-round accepted-token counts to optional runtime instrumentation. The current `MultiDecoder.admit()` starts its timer before vision encoding/prefill and stops it **after building the initial draft chain**; its `prefill_s` is therefore not pure prompt compute. `decode_s` can include time the scheduler spends prefilling other requests. Patch 0001's health `prefill_seconds_total` measures time to first callback, which can include scheduler waiting. Neither field should be used unqualified as GPU prefill time.

Report these distinct quantities:

- Client TTFT to first nonempty reasoning/content delta, plus time to first visible answer.
- End-to-end request latency and inter-emission-gap p50/p95/p99. Speculation emits several tokens together; an SSE chunk is not one token, so token-level ITL needs server timestamps.
- Aggregate generated tokens divided by a precisely defined common wall interval; per-request generated tokens divided by that request's decode interval, and a separate end-to-end rate.
- Prompt tokens actually computed divided by prompt-compute wall time; report cached tokens separately.
- Draft acceptance `accepted / drafted`, committed yield `1 + accepted / rounds` for the normal decode rounds, draft-depth distribution, and total wall time per committed token. Handle zero denominators and the initial prefill token explicitly.

Optimize wall time per committed token, not acceptance percentage alone. A shorter chain can increase acceptance percentage while producing fewer tokens per expensive target forward.

**A confirmed proposal-routing defect should be fixed before further MTP tuning.** In patched `families/qwen4_exp/cuda/multi.py::_draft_all`, `mtp_compute()` returns one logit row per entry in `todo`. Patch 0007 removes copy-covered streams from `active`, but does not remove their rows from `logits`. `_picks()` reads rows sequentially, so remaining streams can use the wrong proposal distribution.

For streams `[A, B]`, with distinct MTP rows `[A_logits, B_logits]`, suppose A finds a prompt-copy continuation and B does not. `active` becomes `[B]`, while logits remain `[A_logits, B_logits]`. B receives a proposal from `A_logits`, keyed using B's sampling position/settings. With controlled row tokens 11 and 22, the reproduction produced A's copy token 99 and **B's token 11 instead of 22**. The defect also applies when a middle stream is removed. Removing only the final stream happens to leave the remaining prefix aligned.

The `row` stored in `active` is an index into residual-stream buffers, not generally the compact MTP logit-row index: an absorbed segment can contain multiple accepted tokens. Fix this with an explicit mapping from the original `todo` ordinal to each surviving logit row, while retaining the existing residual-stream offsets for the next `mtp_stage`. Gather/compact the corresponding logits before `_picks`, or pass explicit logit indices into it. Do not index logits with `a1 - 1`.

Target verification still compares every draft against the target model's keyed sample. The demonstrated failure is wrong proposals, not demonstrated wrong final answers. Its likely performance effects are wasted verification rows and lower acceptance in mixed copy/prose traffic; the magnitude is unknown. Test two and three streams, every removed position, variable accepted lengths, mixed sampling settings, all-copy and no-copy controls, and sampled/greedy token parity against `draft:false`. Keep copy enabled after fixing it if quoting/editing workloads still benefit. Turning it off globally is only a diagnostic ablation for ordinary prose. Sources: [patch 0007](../patches/0007-flash-next-copy-drafts.patch), [upstream MTP computation][mtp], and [concurrent decoder][multi].

Copy lookup has another measurable cost: the first `CopyIndex.propose()` indexes almost the entire prompt into Python eight-token tuples, even if no continuation matches. Extracting the actual class from `families/qwen3_5/cuda/decode.py` and calling a fresh `CopyIndex(8).propose(list(range(N)), 8)` took median **2.76 / 49.82 / 85.88 ms** for N = 8,192 / 131,110 / 195,000 respectively, over five runs on this host. Separate `tracemalloc` runs measured peak index allocations of **1.82 / 32.85 / 52.17 MiB**, excluding the already-created context list. These unique-ID synthetic inputs returned no copy proposal; they are not natural-prose or model-throughput measurements. The indexing occurs in the decode worker, so measure the first post-TTFT gap with copy on/off at long context. Consider incremental construction during scheduled prefill, a bounded index, or a workload-sensitive enablement policy. Preserve collision verification and exact target verification if replacing tuple keys with compact hashes; compare the editing benefit before limiting the index.

**Decode latency has an unexploited graph path.** `FlashNextEngine` selects `MultiDecoder` whenever configured `streams > 1`. That path calls eager `stage`/`compute`; `_slot` explicitly sets `graphs = None`. This remains true with just one active request on the default five-slot server. `PARALLEL=1` instead constructs the graph-capable `Engine`, whose graph cache keys include row count, context bucket, and recurrent-state parity. The published “one concurrent request” row does not tell us whether its server was configured with one or several slots. Sources: [engine][engine], [graphs][graphs], [multi decoder][multi].

First compare **one active client** at configured `PARALLEL=1` versus `PARALLEL=5`, holding context, chunks, sampling, vision, and prompts constant. This measures the complete configuration difference, including memory/cache effects; an internal eager-versus-graph toggle on the same single-stream engine is needed to attribute the result specifically to replay. `PARALLEL=1` also removes this recipe's concurrent copy-draft path. Do not publish it as a five-client serving improvement.

If CPU launch gaps are material, implement bounded graph replay for concurrent decoding. Start with one active slot, then common two/four/five-stream layouts. Current Python loops close over stream state and cache pointers, so a cache keyed only by total row count is insufficient. Account for slot identity, each stream's row span/context bucket, recurrent parity, and vision rotary mode, or refactor those values into stable device metadata. Bound graph count and memory; retain an eager fallback. Capture GPU computation only: SSD reads and host sampling remain outside capture. Measure cold capture latency at newly reached context buckets as well as warm performance.

The current concurrent warmup admits one synthetic stream with a two-token output limit and executes at most one round. It does not exercise every multi-stream verification shape or long-context selector path. Warm the actual measurement shapes explicitly, and report first-use compilation/capture latency separately from steady state.

Two additional decode opportunities should be profiled with graphs and draft tuning:

- `forward.py` uses fused hyper-connection readout only for non-prefill windows with `R <= 16`. Five streams with six drafts can verify up to 35 rows, crossing into the less-fused path. Record the actual row-count histogram and profile 15/16/17/24/35-row shapes. Extending fusion or using an occupancy-aware draft cap may help; simply raising the constant can increase register pressure and must be benchmarked and checked for parity.
- Target `sample_streams()` copies candidate IDs and values to the CPU; `_picks()` also performs top-k, maximum, log-sum-exp, and a CPU readback at each MTP depth. Fuse/reduce transfers before considering a GPU sampler. Any GPU sampler must preserve the position-keyed RNG, tie ordering, top-k/top-p rules, and fallback behavior. Reducing synchronization should be motivated by an Nsight timeline, not assumed to beat weight/attention work.

**Speculation should depend on the workload.** Sweep `MTP_DRAFTS` in `{0, 1, 2, 3, 4, 6, 8}` and, for promising nonzero depths, confidence in `{0.30, 0.50, 0.60, 0.75, 0.90}`. Use a staged sweep rather than the full Cartesian product everywhere. Start at one active request and five active requests, then validate finalists at intermediate concurrency and long contexts. The implementation always keeps the first draft even below its confidence threshold; setting a very high threshold is not equivalent to disabling drafts.

Track target time, drafter time, yield, and verification-window size. A useful round model is `time per committed token ≈ (target + draft + staging + sampling time) / committed tokens`, measured over many rounds. Fixed `6 / 0.60` is a reasonable baseline, not a universal optimum for both lightly loaded and saturated serving. Only add an adaptive depth/confidence controller after fixed settings have established a stable improvement and an explicit fallback.

The current default draft vocabulary is already biased toward English/code. Keep `DRAFT_LANGUAGE` empty for English prose. A smaller, coverage-tested English vocabulary is an optional later experiment, not an automatic win: a cheaper draft head can lose more through missed proposals. Language-patch environment settings should not be assumed to exist in the default image. Avoid claiming a throughput improvement by changing temperature, disabling thinking, shortening output, or switching KV precision; those are different output/quality workloads.

**First-token delivery currently waits for work belonging to the next decode round.** `MultiDecoder.admit()` samples the first token during prefill, then constructs its initial MTP chain before `s.take([first])`. Emit that token once its state and request registration are valid, before drafting the next round. Preserve cancellation, EOS handling, bookkeeping, and the already-sampled token. This can lower observed TTFT by the initial drafting cost without improving prompt computation or sustained throughput. Instrument the stages first and report those distinctions. In particular, do not remove required MTP cache absorption just to move the timestamp.

**Prefill can avoid an unused MTP vocabulary projection.** `_prefill_chunks()` calls `mtp_forward()` to absorb known prompt tokens and discards its returned logits. `mtp_compute()` nevertheless finishes with a projection through the draft vocabulary head, once per nonempty absorption; resumed-prefix absorption has the same pattern. Add a separate cache-absorption mode that skips the unused vocabulary projection. The existing `last_only=False` is not that mode: it computes more logits through the full head. Preserve MTP attention/index state, position advancement, and the main-model tail used for the first draft. Consider skipping additional final mixing only after establishing that no later consumer needs it. Sources: [patched prefill loop][decode], [patch 0008](../patches/0008-flash-next-vision.patch), [MTP forward][mtp].

The main-model head already runs only on the final prompt chunk. Preserve that optimization. This MTP change is a bounded first implementation, but its throughput benefit is unknown and may be small: it removes a one-row projection per chunk, not the whole MTP pass.

**SSD staging has specific remaining costs.** Linux `SSDTable._no_cache()` uses `POSIX_FADV_RANDOM`; it does not use `O_DIRECT` or bypass the page cache. “SSD mode” avoids retaining the entire table as an admitted resident allocation, but repeated reads can still hit cached pages. The existing synthetic prompt generator cannot guarantee untouched n-gram rows. Sources: [SSD table][ssd], [native reader patch](../patches/0003-flash-next-ssd-native-reader.patch).

The [checkpoint configuration][config] has one PLE layer, bigram/trigram hashing with eight heads each, and 160 values per table row. At group-32 affine 4-bit precision that is 80 bytes of codes plus 10 bytes each of scales and biases: 100 bytes/row, or **1,600 logical table bytes per processed token before deduplication**. At 2,500 prompt tok/s that is only 4 MB/s of useful payload. This is not an SSD-bandwidth requirement: tiny random reads, page fetches, metadata preparation, scatter copies, and read amplification can cost far more. In the uncoalesced case there are up to 48 component reads per token. Measure physical I/O and request counts before blaming sequential SSD bandwidth.

In the current concurrent `stage()`, each stream independently hashes, gathers, and transfers its PLE rows. For decode, concatenate IDs across streams for a given PLE table, gather once, and scatter to the original stream segments. This can reduce native-pool wakeups, duplicate reads across streams, NumPy work, and small transfers. Preserve each stream's `ple_history`/`ple_last`, original row order, and EOS resets. The native reader already deduplicates and coalesces within a gather; the new opportunity is batching **across** gathers. Compare combined versus separate reads at one, two, and five streams with identical staged bytes.

Tune the native worker pool after adding telemetry. It currently requests 64 workers per batch. Once grown, the pool sets `busy_` to all existing helper threads and wakes them all, even for a later small batch. An adaptive policy therefore requires changing worker participation, not just passing a smaller `threads` argument. Compare small decode batches with 1/4/8/16 workers and larger prefill batches with 16/32/64; record read latency, CPU time, queue depth, context switches, and NVMe utilization. These are proposed experiment values, not supported environment variables today.

Prompt read-ahead already overlaps next-chunk reads with GPU compute. The wrapper can retain two futures, but the current prefill loop normally submits only the next chunk. Instrument wait time in `gather()` and time spent hashing/copying separately. If the GPU still waits for reads, try bounded two-chunk lookahead and precomputed IDs; if reads are already hidden, deeper queues merely add memory/CPU pressure. A two-buffer pinned staging design can overlap transfers, but requires CUDA events to prevent reuse while copies or compute still consume a buffer. Keep all new buffers in the capacity calculation.

Only consider an explicit bounded PLE row/page cache after measuring reuse on natural prose and accounting for the existing OS cache. A 256 MiB–1 GiB cache is an experiment requiring real headroom, including index overhead. Do not allocate it from an assumed unused reserve. Packing codes/scales/biases into a verified interleaved on-disk row format could reduce tiny reads, but adds conversion/storage and checkpoint-validation work; leave it behind batching and worker tuning in priority.

**Chunk size is a capacity and latency tradeoff.** The existing knob accepts 256–16,384 rows; image prompts also require divisibility by the index compression ratio, four for this checkpoint. Start with 1,024/2,048/4,096/8,192, not the maximum. Larger chunks can improve standalone throughput but consume scratch and increase the length of any non-preemptible scheduling quantum.

Calculated changes below use patched `indexed_stream_geometry`, int8 KV, six drafts, eight kept entries, and the fetched model configuration. Weight memory is held constant. They are deltas from five streams, 262,144 context, 2,048 rows; positive means more estimated memory. They do not establish admission on this host.

| Candidate | Geometry delta | Practical interpretation |
| --- | ---: | --- |
| 5 streams, 262,144 context, 4,096 rows | +0.939 GiB | Tight with vision on; do not enable blindly |
| 5 streams, 262,144 context, 8,192 rows | +2.817 GiB | Needs explicit headroom; no measured speedup here |
| 4 streams, 262,144 context, 4,096 rows | −3.883 GiB | Useful vision-enabled benchmark profile, sacrifices one slot |
| 5 streams, 131,072 context, 4,096 rows | −10.273 GiB | Room for experiments if half-sized windows meet the workload |
| 6 streams, 131,072 context, 4,096 rows | −7.686 GiB | Test aggregate throughput; more slots do not guarantee it |

For a text-only service, `VISION=0` removes approximately 0.84 GiB of tower weights and selects 4,096 rows; the net estimate is nearly unchanged. This is the recipe's already-documented low-effort prefill profile, with roughly 2–5% historical benefit. For vision-enabled serving with five full-context slots, first recover or explicitly budget scratch before raising rows.

Do not set `PLE_ON_SSD=0` as a general speed recommendation. Adding the approximately 29.8 GiB tables to the default approximately 102.5 GiB estimate would put it near 132.3 GiB, already beyond physical usable memory before host reserve. Reducing KV precision or adopting a smaller checkpoint is a separate quality/capacity decision. On unified memory, CPU page cache, pinned staging, graphs, model allocations, and other services compete for the same RAM. NVIDIA specifies up to 273 GB/s memory bandwidth for this system; that is a hardware ceiling, not a measured available bandwidth or a direct token/s predictor. [NVIDIA system overview](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html).

**Mixed traffic needs resumable prefill.** The scheduler's worker calls `_admit()` before `round()`. `_admit()` calls `MultiDecoder.admit()` synchronously, which processes the entire prompt. It may admit multiple waiting requests before the next decode round. The chunks in `_prefill_chunks()` bound scratch; they do **not** yield to active decoders. With a free slot, a new long prompt can therefore pause existing generations for its entire prefill. At full occupancy, the new prompt waits for a slot instead. This is source-established serialization, not a measured stall duration here. Sources: [scheduler][scheduler], [concurrent admission][multi].

Introduce a pending-prefill state and an incremental `prefill_step` that returns after a bounded chunk. Schedule ready decode rounds between prompt chunks; rotate fairly among prompt requests; cap admission work per iteration. Preserve DeltaNet recurrent/conv state, PLE history, MTP length/tail, QSA pooling, and vision positions at every boundary. Shared prompt scratch must not overwrite decode scratch or retained tails; cancel/disconnect must release the partially admitted slot and pending read-ahead safely. Existing prefix snapshots must still mean the same computation state.

At a rough standalone 2,500 tok/s, 2,048 and 4,096 tokens represent about 0.82 and 1.64 seconds of work respectively. These are illustrative arithmetic estimates, not a scheduling guarantee, but show why merely yielding once per large chunk may still produce poor interactive latency. Evaluate smaller chunks while decoders are active and larger chunks while idle, using preallocated maximum scratch. A target near 100 ms would imply roughly 250 tokens at that rate, close to the current 256-row lower bound; measure the throughput penalty and actual p95 before promising such a latency target. Keep compute chunking and scheduler fairness as independently measured changes.

**The GPU work after I/O needs profiling, especially at long context.** There are three concrete places to investigate rather than replacing this hybrid architecture with generic dense-attention advice:

| Area | Source observation | Experiment and constraint |
| --- | --- | --- |
| Sparse index scoring/selection | Every new query scores compressed history; selected KV budget is bounded, but scoring grows with context. Patch 0004 uses tiled radix selection above 32,768 blocks | Time `_scores`, `_select`, `_select_tiles`, attention and merge separately at 32k/64k/128k/195k/near-limit contexts; test threshold/tile/warp choices and prompt attention row blocks. Preserve exact selected IDs, lower-ID tie handling, tail blocks and causal limits |
| Gated DeltaNet prefill | 36 of 48 layers use recurrent attention; `gdn_prefill.cu` loops sequentially over prompt tokens within each head/value-row block, loading 32-step tiles | Profile register pressure, occupancy, dependency stalls, and memory. Test launch tiling/staging that preserves recurrence arithmetic before considering a parallel chunk/WY algorithm |
| Affine matmuls and MoE | There are already CUDA tensor-core prefill paths and grouped expert routing, with separate prefill/decode arithmetic | Tune actual GB10 shapes, expert occupancy, small/large expert tiles, and HC fusion. Do not propose “enable tensor cores” as if they were absent; do not substitute prefill kernels into verification without exact-output checks |

At compression ratio four, 32,768 complete blocks correspond to about 131k tokens; inspect both sides of the dispatch boundary, including graph context buckets. The model selects 512 complete blocks for a 2,048-token attention budget plus the partial tail. Changing the selection algorithm's launch geometry can be exact; shrinking that budget changes model behavior. Likewise, a mathematically equivalent parallel DeltaNet recurrence can change floating-point rounding and is not automatically byte-identical. Sources: [QSA patch](../patches/0004-flash-next-qsa-tiled-select.patch), [attention][attention], [forward][forward], [GDN prefill][gdn], [expert prefill][experts].

Use NVTX ranges around SSD stage/wait, target forward, MTP absorption/draft, sampling, and each prompt chunk. First obtain a short Nsight Systems trace covering representative steady-state requests; only then profile the dominant kernels with Nsight Compute. Measure an unprofiled run separately, since profiling changes timing. For a component consuming fraction `f` of wall time, even a 2× improvement gives only `1 / (1 - f + f/2)` overall speedup. Use this bound to decide whether a kernel rewrite is worth pursuing.

**Prefix reuse is valuable but is a separate workload.** Concurrent reuse holds prompt-end snapshots tied to idle slots, requires a strict prefix extension, and is disabled for image prompts. It is not general shared-prefix paged KV caching, and increasing a generic cache flag does not necessarily enlarge this fixed CUDA path. Stabilize reusable system/document prefixes and measure real conversation extensions first. Later investigate additional prompt-boundary snapshots or shared read-only KV prefixes, with explicit recurrent-state and memory ownership. Do not reuse decoded states as prompt states casually: the runtime deliberately distinguishes prefill and decode arithmetic. Publish warm-prefix TTFT alongside, not as a substitute for, uncached prefill throughput.

**Implementation order and acceptance gates.** Effort ranges are planning estimates for implementation plus local validation, excluding checkpoint provisioning and lengthy benchmark sweeps. Gains are unknown until the matching hardware experiment passes.

| Order | Concrete deliverable | Estimated effort | Promotion gate |
| --- | --- | --- | --- |
| 0 | Extend `tools/bench.py` or add a benchmark driver with fixed manifests, concurrent streaming, JSONL, output hashes, and the timing distinctions above | 1–2 days | Reproduce comparable current-default baselines across at least three independent server boots; make cache state and prompt/output counts auditable |
| 1 | New patch correcting copy/MTP logit ownership, with regression tests | 0.5–1 day | Controlled routing cases pass; sampled and greedy final token sequences match reference; no pure-prose slowdown |
| 2 | Run configuration sweep: graph-capable one-slot diagnostic, chunk sizes under safe memory profiles, then MTP depth/confidence finalists | 1–2 days of experiments | Select profiles separately for one-user latency, five-user aggregate throughput, and prefill; retain only repeatable gains |
| 3 | Separate initial-token delivery from drafting; add MTP prompt absorption without an unused vocabulary projection | 1–2 days | Timing attribution proves the intended work disappeared; TTFT/output/EOS/cancellation tests pass; label TTFT-only gains accurately |
| 4 | Batch PLE gathers across streams and implement adaptive native worker participation | 2–4 days | Identical staged bytes, clean concurrency/error tests, lower measured staging cost and better end-to-end rate |
| 5 | Add resumable, fairly scheduled prefill | 3–6 days | Mixed 128k/195k prompt arrivals substantially reduce active-stream p95 pauses; agreed throughput loss stays bounded; no state/cancellation leaks |
| 6 | Add bounded concurrent CUDA graphs and optimize measured fusion/sampling gaps | 3–7 days | Sustained end-to-end gain at common concurrency; capture latency and graph memory bounded; parity across slot churn, context buckets and vision modes |
| 7 | Tune the dominant prefill/QSA/GDN/MoE kernels from profiles | 3–10 days per focused change | Kernel improvement survives full-model testing; exact-output path remains exact |
| Later | New numerical paths: parallel GDN reassociation, FP8 activation prefill, NVFP4/EXL3 checkpoint alternatives, int4 KV | Separate project | Explicit quality evaluation, memory accounting, and separate published baselines; no byte-identity claim |

For ordinary speed changes, use a predeclared provisional gate of at least 5% improvement in the targeted end-to-end metric, a confidence interval excluding zero, and no material regression elsewhere (for example more than 3% sustained-throughput loss). Smaller low-complexity wins can still be worthwhile if repeatable; the routing bug should be fixed for correct proposal ownership regardless of measured gain. Scheduler work has a different objective: agree on a p95 pause/TTFT budget and an acceptable throughput tradeoff before implementing it. These are decision thresholds, not forecasts.

**Benchmark matrix and execution.** First provision enough memory and disk for a dedicated benchmark window. Keep the same image, checkpoint revision, patch hash, kernel cache state, driver, and sampling across paired runs. Record temperatures/clocks, competing load, `MemAvailable`, process memory, swap activity, and SSD I/O. Randomize A/B order and repeat finalists across at least three boots with at least ten paired requests per primary case. Bootstrap paired differences or otherwise report uncertainty. Do not call sequential runs with different random prompts a controlled 3% improvement.

| Workload | Inputs | Main measurements |
| --- | --- | --- |
| Short-context prose | Several held-out natural prompts, 512–1,024 output tokens, thinking on/off labeled, sampled fixed seeds and greedy | Decode yield/rate, acceptance, TTFT, answer-start time; C=1/2/4/5 |
| Long-context decode | Same output task following 8k/32k/128k/195k prompt tokens | Context-dependent target/MTP cost and QSA boundary behavior |
| Uncached prefill | Actual tokenized lengths around 1k/8k/16k/32k/64k/128k/195k/240k, 16–32 output tokens | Prompt compute, server/client TTFT, SSD waits, memory; ensure room for output and speculative reserve |
| Warm reuse | Identical prompt, strict extension, shared-prefix but different suffix; each labeled | Reported cached tokens, storage warming, incremental TTFT |
| Mixed load | Keep 1–4 prose decoders active while admitting an uncached 128k/195k prompt; repeat at full occupancy with queued requests | Per-stream maximum/p95 emission gap, queued TTFT, aggregate throughput, fairness |
| Copy/mixed regression | Quoting/editing plus unrelated prose, varied stream order and seeds | Correct draft ownership, source-specific acceptance, final token/text hashes |
| Capacity/vision regression | Five long text streams and representative image/video arrivals under the selected profile | Peak host availability, allocation growth, successful completion, vision correctness |

Store all response text including `reasoning_content`, finish reason, input/output token counts, seed, sampling settings, and `tensorfold` statistics. For exactness, use the existing `tensorfold.token_sha` and request `return_token_ids:true`; this interface exists in both inspected versions. Comparing full returned token IDs is stronger than a decoded-text or truncated-hash comparison. Compare serial versus speculative generation under each candidate configuration and compare the candidate to the patched baseline with the same seeded inputs. Include partial chunks, EOS, cancellation, heterogeneous sampling, QSA ties, context-bucket transitions, cached-prefix extension, and simultaneous copy/non-copy streams.

For storage tests, report warmed and initially uncached file-page conditions separately. A restarted process does not make the page cache cold. Prefer controlled distinct working sets or file-scoped eviction on an isolated benchmark host; do not globally drop this shared host's caches. Never equate `cached: 0` with cold physical SSD reads.

The following are **existing, valid configuration arms** for the eventual benchmark session, not commands executed during this investigation. Each restart replaces that recipe's running server and should be run only in the dedicated benchmark window. Preserve the original `.env` and explicit environment before the sweep; restore them afterward. Read the startup admission result for every arm.

```bash
# Current default, made explicit.
PARALLEL=5 CONTEXT=262144 KV_DTYPE=int8 PLE_ON_SSD=1 VISION=1 \
  TENSORFOLD_PREFILL_ROWS=2048 MTP_DRAFTS=6 MTP_CONFIDENCE=0.60 \
  DRAFT_LANGUAGE= ./start.sh restart

# One-client graph-path diagnostic; keep chunks equal to the baseline.
PARALLEL=1 CONTEXT=262144 KV_DTYPE=int8 PLE_ON_SSD=1 VISION=1 \
  TENSORFOLD_PREFILL_ROWS=2048 MTP_DRAFTS=6 MTP_CONFIDENCE=0.60 \
  DRAFT_LANGUAGE= ./start.sh restart

# Text-only, full-capacity profile already supported by the recipe.
PARALLEL=5 CONTEXT=262144 KV_DTYPE=int8 PLE_ON_SSD=1 VISION=0 \
  TENSORFOLD_PREFILL_ROWS=4096 MTP_DRAFTS=6 MTP_CONFIDENCE=0.60 \
  DRAFT_LANGUAGE= ./start.sh restart

# Vision retained, one slot exchanged for chunk/headroom experiments.
PARALLEL=4 CONTEXT=262144 KV_DTYPE=int8 PLE_ON_SSD=1 VISION=1 \
  TENSORFOLD_PREFILL_ROWS=4096 MTP_DRAFTS=6 MTP_CONFIDENCE=0.60 \
  DRAFT_LANGUAGE= ./start.sh restart
```

Use `MTP_DRAFTS=0` as a no-MTP server arm, and per-request `"draft": false` as the serial verification reference. They are not identical resource/cache conditions: disabling MTP at startup changes allocations, whereas a per-request reference uses the configured server. Check actual startup arguments and image identity instead of relying on shell defaults. The existing `tools/bench.py` can smoke-check an arm, but the extended driver from step 0 is needed for the claimed comparisons. Finish each promoted change with `tools/needle.py`, `tools/toolcheck.py`, and `tools/visioncheck.py` when vision is enabled, plus the targeted parity and mixed-load suite.

**Reproducing the proposal-row finding without loading the model.** Run the following against a scratch checkout, never over an installed server. Obtain upstream with `git clone --depth 1 --branch v0.3.6.3 https://github.com/ashhart/TensorFold.git /tmp/astra-tensorfold-v0363`; verify its commit matches the one above. From its `src/` directory apply this recipe's default `patches/*.patch` in sorted order using `patch --batch -p0`. Set `TF_SRC` to the resulting `src` directory. The script extracts and executes the real method, with fake model operations and distinct compact MTP logit rows. It proves control-flow row ownership, not model accuracy or throughput.

```bash
TF_SRC=/tmp/astra-tensorfold-v0363/src python3 - <<'PY'
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

path = Path(os.environ['TF_SRC']) / 'tensorfold/families/qwen4_exp/cuda/multi.py'
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == 'MultiDecoder')
fn = next(n for n in cls.body
          if isinstance(n, ast.FunctionDef) and n.name == '_draft_all')

class State:
    mtp_drafted = 0
    mtp_len = 0
    pos = 10
    def set_mtp_len(self, n):
        self.mtp_len = n

def stage(w, b, windows):
    segs, at = [], 0
    for st, keep, _ in windows:
        segs.append((st, at, at + len(keep)))
        at += len(keep)
    return segs

scope = dict(COPY_MATCH=8, mtp_stage=stage,
             mtp_compute=lambda w, s, b: [11, 22][:len(s)])
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)

for copy_a, copy_b in [(False, False), (False, True), (True, False), (True, True)]:
    def stream(sid, copied):
        return NS(sid=sid, st=State(), out=[1], count=100, sampling=None,
                  context=[1], copies=NS(propose=lambda context, n:
                                        [99] if copied else []))
    a, b = stream(1, copy_a), stream(2, copy_b)
    decoder = NS(depth=1, confidence=0.6, w=None,
                 mbuf=NS(streams=[0, 0]), buf=NS(streams=[0, 0]),
                 _picks=lambda logits, positions, samplings:
                 [(t, 1.0) for t in logits[:len(positions)]])
    scope['_draft_all'](decoder, [(a, 0, [1]), (b, 1, [1])], {1: [2], 2: [2]})
    want = ([99] if copy_a else [11], [99] if copy_b else [22])
    got = (a.drafts, b.drafts)
    print((copy_a, copy_b), 'expected', want, 'actual', got, 'aligned', got == want)
PY
```

Observed output:

```text
(False, False) expected ([11], [22]) actual ([11], [22]) aligned True
(False, True) expected ([11], [99]) actual ([11], [99]) aligned True
(True, False) expected ([99], [22]) actual ([99], [11]) aligned False
(True, True) expected ([99], [99]) actual ([99], [99]) aligned True
```

**Source identities.** The checkpoint metadata was read at Hugging Face revision `dadefa8066e3be900a0d148d0f5a2f4eb1cf6534`; weights were not downloaded. Upstream links below are pinned to the inspected TensorFold commit. Apply the local patches when reconciling functions changed by the recipe. Local patch links and function names are the authority for those differences.

[multi]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/families/qwen4_exp/cuda/multi.py
[mtp]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/families/qwen4_exp/cuda/mtp.py
[engine]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/families/qwen4_exp/cuda/engine.py
[graphs]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/families/qwen4_exp/cuda/graphs.py
[decode]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/families/qwen4_exp/cuda/decode.py
[ssd]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/families/qwen4_exp/ssd_table.py
[scheduler]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/cuda/scheduler.py
[attention]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/families/qwen4_exp/cuda/attention.py
[forward]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/families/qwen4_exp/cuda/forward.py
[gdn]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/cuda/kernels/gdn_prefill.cu
[experts]: https://github.com/ashhart/TensorFold/blob/191188075bca56a7c71074a79375eb4c1cb22e1c/src/tensorfold/cuda/experts_prefill.cu
[config]: https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP/blob/dadefa8066e3be900a0d148d0f5a2f4eb1cf6534/config.json
