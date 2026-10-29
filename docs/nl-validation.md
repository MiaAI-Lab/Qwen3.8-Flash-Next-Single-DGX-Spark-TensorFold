# Dutch vocabulary validation

The extension was selected from 17,994 Dutch Wikipedia articles and evaluated on 2,006 held-out articles. Token occurrence coverage was 86.8968% for the default and 99.4759% for the union. This is not a throughput measurement. See [source provenance](../data/nl/report.json).

## TensorFold 1.1.4 on ASUS GX10 — 2026-10-29

TensorFold `eab8582819688325e371907e4706e1c602b8a273`, recipe main `fd3f43e11238553685ebb68b6d8ab2b112818e3c`, local patch 0018, NVIDIA GB10 and driver 580.173.02. The NVIDIA FP4 checkpoint `b9ce18a6887e18c089fdea8509a33e63429a3300` tokenizer.json SHA-256 is `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`, identical to the tokenizer used to select the vocabulary. The 1.1.4 embedded default list exactly matches the original 79,591 baseline IDs. The CUDA loader logged 84,767 IDs.

One image, A/B/A/B restarts, one model server, 32,768 context, parallel 1, FP8 KV, thinking off, greedy temperature 0, seed 4321, 512-token output limit. Each block had six Dutch, two English and two code prompts; four of those had serial `draft=false` controls.

| Pair | Dutch prompts | Default median | Dutch median | Median paired gain |
| --- | ---: | ---: | ---: | ---: |
| A1/B1 | 6 | 38.93 tok/s | 40.21 tok/s | +3.83% |
| A2/B2 | 6 | 38.58 tok/s | 39.74 tok/s | +4.00% |

All 20 A/B output-text SHA-256 and full server token SHA-256 hashes, output lengths and finish reasons matched. All 16 `draft=false` controls matched their drafted counterpart; draft state was verified in the server response. Five of six Dutch answers in each block reached the cap; one stopped naturally. English median paired changes were -0.64% / -0.77%, code -0.16% / -0.46%, with only two prompts per category. Prompt caching was active, including in no-draft controls; decode rates are server-reported and prefill/TTFT are not compared as uncached measurements. The machine also used swap during testing.

These are preliminary C=1 greedy results; longer complete answers, thinking on, concurrency and wider language/code coverage remain to be qualified. [Per-request records](../data/nl/gx10-tf114-20261029.jsonl) omit generated text but retain text and token hashes, timings and cache statistics. The [benchmark driver](../tools/bench_nl_114.py) contains the prompts and request settings.

## Earlier results

The old Zig `db281878` plus closed PR #79 measured paired medians +6.79% and +6.34%, and the prior Python v0.6.1 experiments were different again. Those results do not establish performance for TensorFold 1.1.4. See the [earlier PR comment](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/pull/20#issuecomment-6438571378).
