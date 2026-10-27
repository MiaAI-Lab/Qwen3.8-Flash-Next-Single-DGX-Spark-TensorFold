# Dutch MTP draft vocabulary for the Zig engine

This is an **opt-in, experimental Dutch vocabulary extension** for Qwen3.8 Flash Next on one DGX Spark. It preserves the Zig engine's 79,591 default draft token IDs and contributes 5,176 distinct Dutch IDs (84,767 total). The target model's complete vocabulary and its normal sampling behavior are not modified.

**Dependency:** [PR #79](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/pull/79) implements `TENSORFOLD_DRAFT_VOCAB` in the Zig CUDA engine. This PR deliberately does not duplicate PR #79's engine patch. Until PR #79 is integrated or applied locally, loading the generated file is **not supported** by the current main branch.

## Prepare a complete runtime file (after PR #79 is available)

Run from the repository checkout on the Spark. The helper extracts the canonical default IDs directly from the pinned `patches/0003-config-weights.patch` and validates both lists against the recorded SHA-256 hashes. This needs only Python's standard library; it does not download a model or rebuild the engine.

```bash
source ./scripts/config.sh
python3 tools/nl_draft_vocab.py --check
python3 tools/nl_draft_vocab.py --output "$HF_CACHE/draft_vocab_nl.txt"
```

The host's `HF_CACHE` is already mounted read-only inside the serving container at `/root/.cache/huggingface`. Do **not** pass a host filesystem path as the runtime setting. With PR #79's Zig patch present in `patches/` and built into the image, use:

```bash
TENSORFOLD_DRAFT_VOCAB=/root/.cache/huggingface/draft_vocab_nl.txt PULL=0 ./start.sh restart
```

Alternatively put `TENSORFOLD_DRAFT_VOCAB=/root/.cache/huggingface/draft_vocab_nl.txt` in `.env` and restart. `start.sh` passes `TENSORFOLD_*` variables into the container. Expect a startup log indicating 84,767 draft IDs. Use `TENSORFOLD_DRAFT_VOCAB=default ./start.sh restart` to revert without changing or deleting the host file. Rebuilding (`PULL=0`) is needed when applying the still-unmerged PR #79; **this Dutch PR itself changes no engine patch or image hash**.

The helper refuses an absent/modified default hunk, changed source hashes, duplicate/unsorted IDs or wrong vocabulary sizes. No extra ID file is selected by default.

## Regenerate the Dutch IDs and provenance

The existing reproducible selection logic uses a pinned Dutch Wikipedia shard and the pinned Vontra tokenizer, with separate article-level train/holdout sets. Regeneration is **not necessary** to use the shipped 5,176 IDs.

```bash
python3 -m venv /tmp/nl-venv
source /tmp/nl-venv/bin/activate
pip install -r tools/requirements-nl.txt
python tools/prepare_nl_corpus.py --out /tmp/nl-corpus
python tools/build_nl_draft_vocab.py \
  --tokenizer /tmp/nl-corpus/tokenizer.json \
  --train /tmp/nl-corpus/train.jsonl --holdout /tmp/nl-corpus/holdout.jsonl \
  --provenance /tmp/nl-corpus/provenance.json
python tools/nl_draft_vocab.py --check
```

`tools/build_nl_draft_vocab.py` now reads the same pinned Zig default IDs. It **no longer writes the obsolete Python v0.6.1 language-loader patch**. It writes the extension and source/coverage report only. The tokenizer and Wikipedia shard are pinned by file hashes; the large source texts are not committed.

## Evidence and limits

- Source-only checks: 79,591 default + 5,176 added = 84,767 unique IDs, with zero overlap. Default and extension hashes equal the original [source report](../data/nl/report.json).
- Held-out Wikipedia token coverage was 86.8968% with the default and 99.4759% with the extension. Corpus token coverage does **not** predict runtime acceptance.
- Historical ASUS GX10 measurements on **TensorFold Python v0.6.1 with an MLX checkpoint** indicated +7.56% median long-Dutch and +8.29% median short-Dutch decode. See [validation](nl-validation.md). These historical figures do **not** establish Zig performance.
- A preliminary Dutch Zig A/B test on the GX10 with PR #79's patch measured paired median gains of +6.79% and +6.34% in two blocks of six prompts. Outputs matched the tested controls. See [validation and limits](nl-validation.md); long complete answers, thinking on, concurrent workloads, memory/startup costs and broader English/code controls remain unqualified.

CPU checks: `python3 tools/nl_draft_vocab.py --check && python3 -m unittest discover -s tools -p 'test_nl_draft_vocab.py' -v`.
