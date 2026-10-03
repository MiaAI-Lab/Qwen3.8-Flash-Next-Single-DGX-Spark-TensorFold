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

No Docker/CUDA image build, model inference, physical DGX Spark run, MTP acceptance measurement,
speed measurement, or runtime exact-output comparison was performed locally. The benchmark driver,
manifest and strict A/B/serial comparator are ready for that qualification. This remains experimental.
