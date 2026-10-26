# Dutch source and historical performance validation

## Source provenance and compatibility with Zig

The pinned Dutch extension was generated from 20,000 distinct Dutch Wikipedia articles (17,994 selection; 2,006 holdout), using the tokenizer recorded in [data/nl/report.json](../data/nl/report.json). The source report includes revisions, SHA-256 hashes and article-split metadata. The tokenizer vocabulary has 248,077 IDs.

The patched **Zig** recipe's `patches/0003-config-weights.patch` embeds 79,591 default IDs with SHA-256 `88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e`, identical to the original Python recipe baseline. The committed 5,176 Dutch IDs have SHA-256 `098fab3f73098ed48b3486e9e586f560be46368a718f96103e5bec32e9600498`. The lists do not intersect and form 84,767 unique IDs.

Corpus token coverage on held-out Wikipedia articles: **86.8968% default** versus **99.4759% with Dutch IDs**. This is token occurrence coverage on encyclopedic text, not draft acceptance or an engine performance estimate.

## Historical GB10 measurements — Python v0.6.1 only

On 2026-10-04, an ASUS GX10 (GB10, driver 580.173.02) ran the older TensorFold Python v0.6.1 language-loader stack and the pinned `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` checkpoint. One active model server, thinking off, one stream, temperature 0 and seed 42:

| Suite | Pairs | Default median | Dutch median | Paired median gain |
| --- | ---: | ---: | ---: | ---: |
| Long complete Dutch answers | 8 | 50.67 tok/s | 54.13 tok/s | +7.56% |
| Short Dutch (512-token cap) | 8 | 52.01 tok/s | 56.88 tok/s | +8.29% |

Long replies had 1,466–1,864 tokens, finished naturally, and matched token IDs/hashes and finish reasons across both arms and their `draft=false` controls. Draft acceptance rose 50.48% → 55.74% in the long suite. Original raw per-request evidence, comparator scripts, startup details and restoration notes remain available in the [original source commit](https://github.com/Timminater/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/56cb80a0000c93384a1b925f422fbe9afffec789) (before the Zig port).

**Limitations:** fixed order, one block per suite, C=1 and thinking off. English/code controls are small. Python results cannot be extrapolated to the new Zig engine, NVIDIA FP4 checkpoint, thinking-on generation or many simultaneous requests. Zig runtime throughput and output exactness for Dutch are **not yet measured**. PR #79 supplies the candidate file-loading mechanism and reports separate Spanish results.

The updated CI validates ID counts, hashes, disjointness, corruption handling and that the generated runtime file preserves the complete Zig default. It does **not** build Zig or run a DGX Spark workload.
