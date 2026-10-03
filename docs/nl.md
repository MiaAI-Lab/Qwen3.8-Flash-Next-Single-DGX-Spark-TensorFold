# Dutch draft vocabulary (experimental)

This fork adds `DRAFT_LANGUAGE=nl` and combinations such as `nl,de`. It changes only the MTP draft vocabulary,
not the main model's scored vocabulary. The original repository is not modified. Speed and exact runtime output
equality must still be qualified on a DGX Spark; the CPU checks below do not exercise CUDA or load model weights.

## Enable on a DGX Spark

```bash
git clone --branch feat/dutch-draft-language https://github.com/Timminater/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
DRAFT_LANGUAGE=nl PULL=0 scripts/prepare.sh
DRAFT_LANGUAGE=nl ./start.sh restart
```

To persist it, set `DRAFT_LANGUAGE=nl` in `.env`. Process environment settings win over `.env`.
The new patches have a new content hash; the original language image does not contain Dutch. `PULL=0` builds
locally. Startup should report `drafts over 84,767 token ids (nl)`; verify this in the server log. Several languages
produce a larger sorted union, with duplicate IDs removed. Restore the baseline with
`DRAFT_LANGUAGE= ./start.sh restart` (the explicitly empty environment value overrides `.env`).

No image was published and no live server was restarted as part of this implementation.

## Generated artifact and limits

TensorFold is pinned to v0.6.1, commit `17c73e189f5e6a5304cda7ea37f086f9c49b4788`.
The recipe base was `4c0dea8`. Its default draft vocab contains **79,591 IDs**, rather than the 47k mentioned in
the earlier discussion of the vLLM recipe. The full tokenizer has 248,077 named IDs (the model's padded output
dimension is separate). We preserve every default ID and pin added/byte tokens before ranking missing IDs.

The seed uses the first 20,000 unique nonempty articles from one pinned Dutch Wikipedia shard. A deterministic
article-ID hash reserves approximately 10% as holdout, before selection. It is an encyclopedic seed, not a random
sample of all Dutch Wikipedia or the model's response distribution. Token coverage cannot predict MTP acceptance.

| Measurement | Selection articles | Held-out articles |
|---|---:|---:|
| Articles | 17,994 | 2,006 |
| Token occurrences | 37,575,880 | 4,141,468 |
| Default coverage | 86.894% | 86.897% |
| Default + Dutch coverage | 99.500% | 99.476% |

There are **5,176 extension IDs**, making the union **84,767 IDs** (6.50% more rows than default).
The generation budget is a maximum union of 100,000, stopping at 99.5% training coverage; holdout does not
influence selection. The wider head can slow English/code. No Dutch speedup has been measured here.

- `data/nl/draft_vocab_nl.txt`: sorted extension only, no default IDs repeated.
- `data/nl/report.json`: exact revisions, input/output SHA-256 hashes, counts and coverage.
- `patches/languages/0010`: registers `nl` in the existing language loader.
- `patches/languages/0011-flash-next-draft-nl.patch`: installs the generated file into TensorFold.
- `scripts/config.sh`: accepts `nl` before Docker/build/download actions, rejects malformed combinations.

The existing image build copies every selected patch and hashes their contents. Consequently the Dutch artifact
is included in image identity and in the stale-image rebuild check. Changing the loose data file alone does not
change an image: regenerate the patch as well. The tests require the data and patched runtime file to agree.

## Reproduce generation

No full checkpoint or GPU is needed. Use Python 3.12 in a virtual environment:

```bash
python3 -m venv /tmp/nl-venv
source /tmp/nl-venv/bin/activate
pip install -r tools/requirements-nl.txt
python tools/prepare_nl_corpus.py --out /tmp/nl-corpus
git clone --branch v0.6.1 https://github.com/ashhart/TensorFold.git /tmp/tf-nl
python tools/build_nl_draft_vocab.py \
  --tokenizer /tmp/nl-corpus/tokenizer.json \
  --base /tmp/tf-nl/src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt \
  --train /tmp/nl-corpus/train.jsonl --holdout /tmp/nl-corpus/holdout.jsonl \
  --provenance /tmp/nl-corpus/provenance.json
```

The downloader pins both Hugging Face commits and retains article URLs locally. It downloads one ~506 MB
Parquet shard and tokenizer.json; texts and tokenizer are not committed to the fork. Wikipedia source texts are
CC BY-SA 4.0 / GFDL; retain attribution if redistributing text. The generated token-ID frequency selection contains
no article prose. The generator uses tokenizer.json directly without executing remote model code.

Local reuse flags `--local-shard` and `--local-tokenizer` are available; use only the pinned files. For tuning on
actual model replies, use separate UTF-8 JSONL train/holdout files with a `text` field, preserve provenance,
and include Dutch reasoning text if the deployment uses thinking. Use a fresh final holdout after any tuning.

## CPU validation

```bash
pip install numpy
# /tmp/tf-nl must be the unpatched pinned tree before this loop.
(cd /tmp/tf-nl/src; for p in "$OLDPWD"/patches/*.patch "$OLDPWD"/patches/languages/*.patch; do
  patch --batch --fuzz=0 -p0 < "$p" || exit 1
done)
bash -n scripts/config.sh scripts/prepare.sh start.sh
TF_SRC=/tmp/tf-nl/src python tools/test_nl_draft.py
```

`.github/workflows/nl-draft.yml` repeats these checks against the exact TensorFold commit on Linux. The tests execute
the actual patched token loader with NumPy, without importing torch. They check default preservation, sorted IDs,
multilingual unions, unknown/missing vocab failures, config/.env precedence, image hashes, selection budgets,
and benchmark rejection of failures, cache hits, mismatched prompts or missing output-equality evidence.

The inherited `test_copy_draft_rows.py` and `test_astra_patches.py` target older v0.5.0 internals. They fail
against the original v0.6.1 recipe as well as this fork (`_mtp` mock missing, changed admission control,
removed `NATIVE_WORKERS`). They are not included in the Dutch CI job; updating those unrelated harnesses
is outside this change. Passing the Dutch tests does not qualify those older engine behaviors.

## A/B qualification on a DGX Spark

Run A = default image/default draft vocab, B = the locally built language image with `nl`. Keep the same checkpoint,
driver, CUDA base, TensorFold commit, KV precision, context, concurrency limit, SSD setting, vision setting,
prefill rows, MTP depth/confidence, copy drafting and all sampling parameters. Build both images before timing.
The image difference is the language patch stack; record both image hashes. Do not time preparation or startup.

Use an otherwise idle Spark. Capture the recipe commit, tokenizer/model snapshot hashes, `.env` values (redact
credentials), `nvidia-smi`, CPU/GPU clock/power/temperature, SSD state, container image IDs, health and startup logs.
Confirm each arm's logged vocabulary and configuration. Wait for `/health` and a successful warmup; do not
change clocks, temperature limits or cache policy between arms. The driver cannot verify startup settings itself.

The manifest has eight independently authored Dutch tasks and four English/code controls. It was not used for
vocab generation. Every request uses temperature 0, seed 42, thinking off and max_tokens 1024. Test a separate
thinking-on suite if that is your workload, keeping both arms identical and including reasoning in equality checks.

Use at least five paired blocks, alternating restart order: A/B, B/A, A/B, B/A, A/B. Each restart has fresh warmups
which are excluded from records. The script sends all 12 prompts at C=1,2,4 with three repetitions. Markers vary
by block/concurrency/repetition/prompt, and are identical within each A/B pair. Outputs record prompt/manifest
hashes, token IDs, token digest, output digest, TTFT, elapsed time, decode stats and acceptance where available.
The comparer rejects cached prompts instead of reporting a cache benefit as a language benefit.

Example for block 01 (use the reverse order for block 02):

```bash
mkdir -p benchmark-results
DRAFT_LANGUAGE= PULL=0 ./start.sh restart
docker logs qwen38-flash-next-tf > benchmark-results/01-A-startup.log 2>&1
python tools/bench_nl.py --arm A --block 01 --jsonl benchmark-results/01-A.jsonl
python tools/bench_nl.py --arm serial --block 01 --jsonl benchmark-results/01-serial-A.jsonl
DRAFT_LANGUAGE=nl PULL=0 ./start.sh restart
docker logs qwen38-flash-next-tf > benchmark-results/01-B-startup.log 2>&1
python tools/bench_nl.py --arm B --block 01 --jsonl benchmark-results/01-B.jsonl
python tools/bench_nl.py --arm serial --block 01 --jsonl benchmark-results/01-serial-B.jsonl
python tools/compare_nl_bench.py --a benchmark-results/01-A.jsonl \
  --b benchmark-results/01-B.jsonl --serial-a benchmark-results/01-serial-A.jsonl \
  --serial-b benchmark-results/01-serial-B.jsonl --out benchmark-results/01-report.json
```

After all blocks, pass all A files to `--a` and B files to `--b`. It reports median paired speed ratios per workload
and concurrency, per-request decode speed, TTFT, elapsed time, batch wall time and aggregate draft acceptance.
Inspect raw records and block-to-block variation; report each block as well as pooled results. Do not interpret
many repeated requests as many independent hardware runs. Check full-batch generated tokens / batch wall time
for end-to-end aggregate throughput; do not sum per-request decode rates. Inspect p95 TTFT and inter-emission
gaps from raw records for interactive latency. Missing draft statistics are reported, never inferred.

The `--arm serial` runs use the same block/manifest/clients and set `draft=false` per request. Pass their files
to `--serial-a` and `--serial-b` to compare token IDs/digests, output digest, length and finish reason against each
drafted arm, using `(block, clients, repeat, name)` as pairing key. If any differ, stop qualification and investigate
before claiming preserved output. The A/B comparer requires exact token AND text equality for every A/B pair;
missing IDs/digests are insufficient evidence. Equal visible prose alone does not establish identical reasoning.

A result is usable only when all requests succeeded, pairing/configuration/output checks pass, and Dutch gains
repeat across restarts without unacceptable English/code or latency regressions. Report early EOS/length truncation
counts and all failures. Do not claim improved quality, universal output equality or Dutch speed from Wikipedia
coverage. No GPU benchmark results are included with this fork.
