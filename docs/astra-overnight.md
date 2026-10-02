# Overnight measurements, 2026-09-30

Image `tensorfold-qwen38:v0.5.0`, container `qwen38-flash-next-tf`, API `http://127.0.0.1:8888`. No git or `scripts/config.sh` change. The server ended on the starting default.

JSONL is under `/tmp/astra-overnight/`. Startup lines are in `/tmp/astra-overnight/arms/`.

## Baseline (no restart)

Admission before any restart: 5 streams × 262144, 4799 MiB a stream, estimate 102.50 GiB within 109.41 GiB, eager, 0 decode graphs. `MemAvailable` after load was about 5.1 GiB. Cmd: `--parallel 5 --context 262144 --kv-dtype int8 --mtp-drafts 6 --mtp-confidence 0.60 --ple-on-ssd --vision --thinking`. Env: `TENSORFOLD_PREFILL_ROWS=2048`, `TENSORFOLD_MTP_COPY=1`, `TENSORFOLD_HOST_RESERVE_MIB=0`.

Decode suite, seed 1, thinking on, max 512 (`baseline.jsonl`). One pass.

| Clients | Per-request tok/s | Aggregate tok/s | TTFT p50/p95 | Accept | Yield |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.7 | 50.5 | 217 / 411 ms | 0.643 | 2.047 |
| 2 | 37.5 | 69.5 | 313 / 400 ms | 0.649 | 2.008 |
| 4 | 22.0 | 82.1 | 534 / 804 ms | 0.633 | 2.023 |
| 5 | 20.5 | 90.9 | 714 / 1094 ms | 0.643 | 2.047 |

Several C=1 prompts never reached a visible answer inside 512 tokens. Where an answer started, answer TTFT was about 2.2–2.6 s. That is thinking time, not prefill.

Greedy, temperature 0, C=1 (`greedy.jsonl`): 56.0 tok/s per request, yield 2.234. `draft:false` on the same prompts (`draft-false.jsonl`): 35.4 tok/s, yield 1.000. Full token ids matched on all five prompts.

Uncached prefill, 24 output tokens, thinking on, engine `cached=0` (`prefill-baseline.jsonl`). `answer_ttft` was null: the 24 tokens were reasoning. Page cache was not dropped.

| Prompt tokens | Prefill tok/s | Server prefill_s | Client TTFT | First post-TTFT gap |
| ---: | ---: | ---: | ---: | ---: |
| 1,031 | 1,524 | 0.68 s | 0.70 s | 68 ms |
| 7,730 | 2,153 | 3.59 s | 3.60 s | 53 ms |
| 15,392 | 2,336 | 6.59 s | 6.61 s | 48 ms |
| 30,748 | 2,302 | 13.36 s | 13.40 s | 180 ms |
| 61,373 | 2,233 | 27.48 s | 27.58 s | 86 ms |
| 122,685 | 2,023 | 60.65 s | 60.84 s | 231 ms |
| 186,864 | 1,914 | 97.64 s | 98.02 s | 1,161 ms |

`tools/visioncheck.py` and `tools/toolcheck.py` passed on this server.

## Arms

Held constant unless named: seed 1, thinking on, temperature default 1.0 / top_p 0.95 / top_k 20, int8 KV, context 262144, ple-on-ssd, copy on, MTP 6/0.60, vision on, 2048 rows. Each restart recorded admission, `MemAvailable`, and argv. One sample unless noted. Fresh boots of the 6/0.60 default sat near 93–95 aggregate tok/s at C=5; the warm baseline was 90.9. C=5 differences of a few tok/s against that warm baseline are not treated as a config win.

| Arm | What changed | Decode | Prefill | Result |
| --- | --- | --- | --- | --- |
| `PARALLEL=1` + vision | one slot | refused at startup | — | patch 0008 requires `--parallel` 2 or more |
| rows 1024 | chunk size | C=1 55.3, C=5 agg 94.2 | 16k–64k about 6% below baseline | not kept |
| rows 4096 | chunk size | C=1 54.9, C=5 agg 92.8 | screening: 16k 2,400, 32k 2,413, 64k 2,368 | see repeats |
| rows 8192 | chunk size | C=1 55.0, C=5 agg 93.0 | 8k 1,984, 32k 2,205, 64k 2,242 | admitted 105.31 within 109.91 GiB; 2.2 GiB left after load; slower than 2048 |
| MTP depth 0 | startup drafts off | C=1 35.5, C=5 agg 76.4, yield 1.00 | 32k 2,387 | serial reference, not a win |
| MTP depth 2 / 0.60 | depth | C=1 51.1, yield 1.86, accept 0.670 | 32k 2,316 | slower C=1 despite higher acceptance |
| MTP depth 4 / 0.60 | depth | C=1 54.1, yield 2.01 | 32k 2,314 | tied with 6 on C=1 |
| MTP depth 8 / 0.60 | depth | C=1 55.3, yield 2.06, accept 0.630 | 32k 2,317 | tied with 6 on C=1 |
| copy off | `TENSORFOLD_MTP_COPY=0` | C=1 55.3, yield 2.05 | prefill unchanged | first gap only; see below |
| SSD threads 1 | fixed pool | C=1 49.7 | 32k 510 | regression |
| SSD threads 4 | fixed pool | C=1 53.5 | 32k 1,596 | regression |
| SSD threads 8 | fixed pool | C=1 55.0 | 32k 2,188 | slightly under adaptive |
| SSD threads 16 | fixed pool | C=1 52.7 | 32k 2,259 | slightly under adaptive |
| vision off, rows 2048 | vision only | C=1 53.7, C=5 agg 95.1 | 32k 2,340, 64k 2,276 | under 5% vs vision-on baseline |
| vision off, rows 4096 | documented text profile (two knobs) | C=1 51.3, C=5 agg 95.3 | 8k 2,131, 32k 2,371, 64k 2,314 | not a 5% prefill gain vs vision-on 2048 |
| vision off, `PARALLEL=1`, rows 2048 | graph engine, text only | C=1 56.4 vs 53.7 on the matching 5-stream text server | 8k 2,335, 32k 2,361 | 21 graphs captured; one sample; cannot serve vision |

Copy off, same long prompts, first post-TTFT gap: 180 → 63 ms at 31k, 231 → 73 ms at 123k, 1,161 → 54 ms at 187k. Short-prompt decode tok/s did not change. Labeled as a first-gap result. Copy stays on.

No depth beat 6/0.60 on C=1, so confidence was not swept.

## 4096 vs 2048 repeats

Three fresh boots each, order 4096, 2048, 2048, 4096, 4096, 2048. Same suite, C=1 decode, plus 30,748- and 61,373-token prefills.

C=1 decode stayed 54.8–55.1 tok/s on every boot.

| | 2048 rows | 4096 rows | Difference |
| --- | ---: | ---: | ---: |
| 30,748 tok, mean of 3 | 2,319 | 2,372 | +2.3% |
| 61,373 tok, mean of 3 | 2,279 | 2,352 | +3.2% |

Every 4096 sample was above every 2048 sample at both sizes. The difference is repeatable and below the 5% gate. The earlier single 4096 pass against the warm baseline (+4.8% at 32k, +6.0% at 64k) was larger than these paired fresh boots. 4096 was not promoted. Decode did not regress.

## Not done in the v0.5.0 sweep

- Nothing in the v0.5.0 sweep cleared the 5% gate. 4096 rows was not promoted.
- No concurrent CUDA graphs on the 5-stream vision server, no QSA/GDN/MoE kernel rewrite. `PARALLEL=1` with vision cannot start.
- `needle.py` was not run. Confidence levels other than 0.60 were not run.

## v0.6.0 default, after the sweep

The recipe pin was moved to v0.6.0 (`c464617`) after the sweep. Image `tensorfold-qwen38:v0.6.0`. The v0.5.0 patches 0002–0013 do not apply; `patches/0002-flash-next-v060.patch` is the replacement. `TENSORFOLD_MEMORY_RESERVE_GIB=2` replaces host-reserve 0. Native SSD workers, cross-stream PLE batching, and the language patch were not ported.

One sample, same seed and prompts as the v0.5.0 baseline (`v060-decode.jsonl`, `v060-prefill.jsonl`). C=1 decode 49.4 tok/s versus 54.7. Prefill 2,024 / 2,089 / 2,188 tok/s at 7,730 / 30,748 / 61,373 tokens versus 2,153 / 2,302 / 2,233. C=5 reused the prefix (`cached` 90) because the snapshot is kept one token early; it is not an uncached C=5 comparison. Vision check and tool check passed. One mixed run: a 15,422-token prefill (7.78 s) overlapped a 256-token decode; the largest decode gap in that window was 1.84 s.

## Defaults left running

v0.6.0, 5 streams growing toward 262144, vision, int8, ple-on-ssd, MTP 6/0.60, 2048 rows, copy on, `TENSORFOLD_MEMORY_RESERVE_GIB=2`. Last admission: 85.18 GiB within 108.20 GiB. `/health` ok.

The v0.5.0 sweep did not justify a different MTP depth, chunk size, or copy setting. The one v0.6.0 sample is slower on uncached C=1 decode and on 8k–32k prefill than that baseline. It is one sample.
