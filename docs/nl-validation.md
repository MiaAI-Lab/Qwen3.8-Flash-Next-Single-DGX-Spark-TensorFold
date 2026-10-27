# Dutch source and runtime performance validation

## Source provenance and compatibility with Zig

The pinned Dutch extension was generated from 20,000 distinct Dutch Wikipedia articles (17,994 selection; 2,006 holdout), using the tokenizer recorded in [data/nl/report.json](../data/nl/report.json). The source report includes revisions, SHA-256 hashes and article-split metadata. The tokenizer vocabulary has 248,077 IDs.

The patched **Zig** recipe's `patches/0003-config-weights.patch` embeds 79,591 default IDs with SHA-256 `88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e`, identical to the original Python recipe baseline. The committed 5,176 Dutch IDs have SHA-256 `098fab3f73098ed48b3486e9e586f560be46368a718f96103e5bec32e9600498`. The lists do not intersect and form 84,767 unique IDs.

Corpus token coverage on held-out Wikipedia articles: **86.8968% default** versus **99.4759% with Dutch IDs**. This is token occurrence coverage on encyclopedic text, not draft acceptance or an engine performance estimate.

## Preliminary Zig runtime test on ASUS GX10 (2026-10-27)

Tested PR #20 at `b138a743cd871cf62b8307c2e2fe4a7d1892a065` plus the separate [PR #79](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/pull/79) patch at `46da52ff3661774569ec27095d4a2228a1a10b19`. The image was built from TensorFold `db281878ddb836fd0df510d8771ecb7e0fe47d26`, patches 0001–0032, and the pinned NVIDIA FP4 checkpoint `b9ce18a6887e18c089fdea8509a33e63429a3300` on GB10, driver 580.173.02. The server logged 84,767 IDs for the Dutch file. CPU vocabulary validation passed 5 tests.

The same image was restarted in A1/B1/A2/B2 order: A used the embedded default, B the Dutch union; one server and one request at a time, context 32,768, parallel 1, thinking off, greedy temperature 0, seed 4321, maximum 512 output tokens. Each block ran six Dutch, two English and two code prompts, plus four serial `draft=false` controls.

| Pair | Dutch prompts | Default median | Dutch median | Median of paired gains |
| --- | ---: | ---: | ---: | ---: |
| A1 / B1 | 6 | 37.01 tok/s | 40.22 tok/s | +6.79% |
| A2 / B2 | 6 | 36.70 tok/s | 40.25 tok/s | +6.34% |

All 20 drafted A/B comparisons matched text SHA-256, server token hash, length and finish reason; all 16 no-draft controls matched the corresponding drafted result and the server reported the expected draft setting. In each Dutch block five of six replies reached the 512-token cap; one stopped naturally. English paired median changes were +0.41% and -1.08%; code -0.36% and -1.40%, with only two prompts per category per block.

These are preliminary C=1, thinking-off, greedy measurements, not a general speed guarantee. Complete long answers, thinking on, concurrency and broader English/code coverage were not evaluated. The list was generated with the pinned Vontra tokenizer; complete identity with the FP4 checkpoint tokenizer has not been independently established. No separate runtime memory or startup cost comparison was recorded. Production was restored and its health, original image and a short completion checked afterward. Per-request records and the benchmark driver are retained on the test machine.

## Historical GB10 measurements — Python v0.6.1 only

On 2026-10-04, an ASUS GX10 (GB10, driver 580.173.02) ran the older TensorFold Python v0.6.1 language-loader stack and the pinned `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` checkpoint. One active model server, thinking off, one stream, temperature 0 and seed 42:

| Suite | Pairs | Default median | Dutch median | Paired median gain |
| --- | ---: | ---: | ---: | ---: |
| Long complete Dutch answers | 8 | 50.67 tok/s | 54.13 tok/s | +7.56% |
| Short Dutch (512-token cap) | 8 | 52.01 tok/s | 56.88 tok/s | +8.29% |

Long replies had 1,466–1,864 tokens, finished naturally, and matched token IDs/hashes and finish reasons across both arms and their `draft=false` controls. Draft acceptance rose 50.48% → 55.74% in the long suite. Original raw per-request evidence, comparator scripts, startup details and restoration notes remain available in the [original source commit](https://github.com/Timminater/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/56cb80a0000c93384a1b925f422fbe9afffec789) (before the Zig port).

**Limitations:** fixed order, one block per suite, C=1 and thinking off. English/code controls are small. Python results cannot be extrapolated to the new Zig engine, NVIDIA FP4 checkpoint, thinking-on generation or many simultaneous requests. The separate Zig experiment above does not extend these historical Python results to other workloads. PR #79 supplies the file-loading mechanism and reports separate Spanish results.

The updated CI validates ID counts, hashes, disjointness, corruption handling and that the generated runtime file preserves the complete Zig default. It does **not** build Zig or run a DGX Spark workload.
