# Dutch implementation validation

Validated locally on 2026-10-03 against recipe base `4c0dea8` and TensorFold
`17c73e189f5e6a5304cda7ea37f086f9c49b4788` (v0.6.1).

- Complete default + language + Dutch patch stack applied to a clean source tree with `patch --fuzz=0`.
  No rejected hunks or offsets were reported.
- All 20 tests in `tools/test_nl_draft.py` passed. They exercise the real patched vocabulary loader,
  multilingual unions, base preservation, generated artifact hashes, malformed/missing settings,
  `.env` precedence, image hashes, selection budgets and A/B/serial comparison failure handling.
- Bash syntax checks passed for config, preparation and startup scripts.
- Python compilation and `git diff --check` passed.
- Regeneration was run twice; extension and patch SHA-256 hashes were identical. The second run
  additionally checked train/holdout article-ID and exact-text overlap: zero.
- Independent held-out Wikipedia coverage: 86.8968% default, 99.4759% default + Dutch.
  5,176 IDs added; 84,767 union IDs. Full inputs and provenance are in `data/nl/report.json`.

The inherited copy-row and Astra harnesses fail on the original patched recipe too, because they refer to
older engine internals. These pre-existing failures are documented in `docs/nl.md` and excluded from the
new Dutch CI job. They were not changed or represented as passing checks.

## Preliminary GB10 runtime evidence (2026-10-04)

Tested on an ASUS GX10 with NVIDIA GB10, driver 580.173.02 (CUDA 13.0 reported by `nvidia-smi`).
The implementation commit was `626eb2e76524620532c0fbac9e876273de09a671`, with the pinned
TensorFold v0.6.1 and `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` snapshot
`2b170fa6309d5d1ee380b35636075fac7945f286`. Tokenizer SHA-256:
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
Subsequent PR preparation changes documentation and evidence only.

A used the existing default recipe image, verified by its `tf.patches=797af1d9df4d` label.
B was built on that image by applying the two language patches in a separate patch directory;
it reported `tf.patches=bf7f3a46a62c` and 84,767 draft IDs with `nl`. This exercised a local
language image build and inference, but was not a clean rebuild from the pinned NGC base.

| Image | Docker content ID |
|---|---|
| A, default | `sha256:667693ccd7eea95d365ee5a17eed97067e13deb131a6e61fbbbbe6f69d0ebc69` |
| B, Dutch | `sha256:39d85f05479d9166afcb1c7c255a00acdd927801e7d5a6608181ad8a8b0cd650` |

Both arms used context 262,144, int8 KV, parallel capacity 5, MTP depth 6/confidence 0.60,
PLE on SSD, copy drafting on, vision enabled, prefill rows 2,048 and a 2 GiB memory reserve.
Requests used one client, temperature 0, seed 42 and thinking off. The server's sampling defaults
were temperature 1.0/top-p 0.95/top-k 20; request temperature overrode the default.
Only one model server ran at a time. Each block ran A, serial A (`draft=false`), B, serial B;
warmups used separate prompt markers and were excluded. Every measured request had zero cached tokens.

| Suite/group | Pairs | A median tok/s | B median tok/s | Median paired B/A gain | Acceptance A / B |
|---|---:|---:|---:|---:|---:|
| Long Dutch, complete answers | 8 | 50.67 | 54.13 | +7.56% | 50.48% / 55.74% |
| Short Dutch, 512-token cap | 8 | 52.01 | 56.88 | +8.29% | 49.36% / 54.73% |
| Short English controls | 2 | 57.37 | 57.92 | +0.97% | 57.59% / 57.59% |
| Short code controls | 2 | 70.83 | 71.39 | +0.78% | 69.62% / 69.62% |

The long suite is [tools/nl_benchmark_long.json](../tools/nl_benchmark_long.json). User prompts
contained 271-315 tokens; answers contained 1,466-1,864 tokens (785-1,006 words), below the
2,048-token cap. All 32 long-suite responses across the four arms ended with `stop`.
Median TTFT was 463 ms for A and 458 ms for B. Token ID arrays, output hashes, completion counts
and finish reasons matched for every prompt across A, B and both serial controls. The comparator
accepted all eight pairs without issues. The earlier short suite's responses all ended with `length`.

Startup was excluded. A's first long-suite startup compiled CUDA extensions (113.5 s); B's startup
used a warm kernel cache (32.4 s). No claim about startup speed follows from these numbers.
Each suite contains only one paired block at C=1 in a fixed order. The long suite has no English/code
controls; the short suite has only two prompts per control group. Clock/thermal drift, order effects,
thinking-on behavior and concurrent throughput have not been qualified. These are preliminary
measurements, not a general Dutch speed guarantee or universal output-equality proof. Follow the
alternating five-block procedure in [nl.md](nl.md) for broader qualification.

### Reviewable evidence

[data/nl/benchmarks/2026-10-04-gb10](../data/nl/benchmarks/2026-10-04-gb10) contains the long-suite
per-request telemetry and exact token IDs, with output SHA-256 values computed from the retained raw
responses. Response prose, emission timing arrays and host-specific operational logs are omitted.
The long and short comparator reports are included. Recompute the long report without a GPU:

```bash
E=data/nl/benchmarks/2026-10-04-gb10
python tools/compare_nl_bench.py --a "$E/long01-A.jsonl" --b "$E/long01-B.jsonl" \
  --serial-a "$E/long01-serial-A.jsonl" --serial-b "$E/long01-serial-B.jsonl"
```

The original long evidence archive (manifest, full responses, server/startup logs and restore checks)
was retained with SHA-256 `76b4b3313ce098ffc7a9a7822a59c8dc178957b63e252829cbd53b52542b5c83`;
the short archive SHA-256 is `ef649325061f6440371c0afe3f53367080b4401b4f96353c66ff6d191170dd96`.
These archives are not shipped in the repository. Production was restored after each suite with the
same original container/image/labels/serve-script hash; health, model listing and inference smoke
checks passed, and test containers were stopped.
