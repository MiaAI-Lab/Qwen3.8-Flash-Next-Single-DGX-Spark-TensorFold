# Dutch draft vocabulary for TensorFold 1.1.4 CUDA

This opt-in extension adds 5,176 Dutch IDs to the 79,591 default draft IDs, giving 84,767 IDs. The CUDA runtime patch `0018-draft-vocab-file.patch` reads `TENSORFOLD_DRAFT_VOCAB` at model startup. Without it, or with `default`, the built-in list is used. The target model's full vocabulary is unchanged. This patch is independent of the closed PR #79 and is for the TensorFold 1.1.4 CUDA engine, not the Metal pack builder.

The compressed baseline and Dutch extension are committed in `data/nl/`; the helper builds their union and checks sorted, unique, in-range IDs and the pinned baseline and extension hashes. The source tokenizer and corpus provenance are recorded in `data/nl/report.json`.

```bash
source ./scripts/config.sh
python3 tools/nl_draft_vocab.py --check
python3 tools/nl_draft_vocab.py --output "$HF_CACHE/draft_vocab_nl.txt"
PULL=0 TENSORFOLD_DRAFT_VOCAB=/root/.cache/huggingface/draft_vocab_nl.txt ./start.sh restart
```

The loader expects a container-visible path; the HF cache is mounted read-only at `/root/.cache/huggingface`. Startup reports `draft vocabulary: 84767 ids from TENSORFOLD_DRAFT_VOCAB`. Set the variable to `default` and restart to select the original list. The modified patch stack changes the image hash and needs an image containing patch 0018; `PULL=0` builds locally. A missing, empty, malformed or out-of-range file stops startup.

Regenerate using the pinned shard and tokenizer with `tools/prepare_nl_corpus.py` and `tools/build_nl_draft_vocab.py`; the latter writes both the extension and complete union. The baseline hash is checked against the original 1.1.4 embedded table. See [validation and limitations](nl-validation.md).

Run CPU checks with `python3 tools/nl_draft_vocab.py --check` and `python3 -m unittest discover -s tools -p 'test_nl_draft_vocab.py' -v`.
