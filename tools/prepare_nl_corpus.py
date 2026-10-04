#!/usr/bin/env python3
"""Download pinned tokenizer/Wikipedia seed data and make disjoint article splits.

pip install -r tools/requirements-nl.txt
python tools/prepare_nl_corpus.py --out /tmp/nl-corpus
Only tokenizer/config files and one corpus shard are downloaded, no model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
MODEL_REV = "2b170fa6309d5d1ee380b35636075fac7945f286"
DATASET = "wikimedia/wikipedia"
DATASET_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SHARD = "20231101.nl/train-00000-of-00006.parquet"
TOKENIZER_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
SHARD_SHA = "83d8a4c39a03e025071aeb64cf5b395870b74abcb4bb14c1c5d7ce44fd05c5d7"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--documents", type=int, default=20000)
    ap.add_argument("--local-shard", type=Path, help="reuse an already downloaded pinned shard")
    ap.add_argument("--local-tokenizer", type=Path, help="reuse an already downloaded pinned tokenizer.json")
    args = ap.parse_args()
    if args.documents < 100:
        ap.error("use at least 100 articles")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    args.out.mkdir(parents=True, exist_ok=True)
    cache = str(args.out / ".hf-cache")
    tokenizer = args.local_tokenizer or Path(hf_hub_download(MODEL, "tokenizer.json", revision=MODEL_REV, cache_dir=cache))
    shard = args.local_shard or Path(hf_hub_download(DATASET, SHARD, repo_type="dataset", revision=DATASET_REV, cache_dir=cache))
    if sha(tokenizer) != TOKENIZER_SHA or sha(shard) != SHARD_SHA:
        raise ValueError("tokenizer/shard does not match the pinned input hashes")
    (args.out / "tokenizer.json").write_bytes(tokenizer.read_bytes())
    counts = {"train": 0, "holdout": 0}
    seen = set()
    handles = {k: (args.out / f"{k}.jsonl").open("w", encoding="utf-8", newline="\n") for k in counts}
    try:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["id", "url", "title", "text"]):
            for row in batch.to_pylist():
                if len(seen) >= args.documents:
                    break
                article = str(row["id"])
                if article in seen or not row["text"].strip():
                    continue
                seen.add(article)
                # Split on article identity, never on chunks of the same article.
                arm = "holdout" if int(hashlib.sha256(article.encode()).hexdigest(), 16) % 10 == 0 else "train"
                handles[arm].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[arm] += 1
            if len(seen) >= args.documents:
                break
    finally:
        for handle in handles.values():
            handle.close()
    if not all(counts.values()):
        raise ValueError("empty split")
    provenance = {
        "model": MODEL, "model_revision": MODEL_REV, "dataset": DATASET,
        "dataset_revision": DATASET_REV, "shard": SHARD,
        "shard_sha256": SHARD_SHA,
        "requested_documents": args.documents, "documents": counts,
        "split": "first N unique nonempty articles; sha256(article_id) % 10 == 0 is holdout",
        "source_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "license": "Wikipedia CC BY-SA 4.0 / GFDL; article URLs retained in local corpus",
        "scope": "one shard, encyclopedic seed data; not representative model replies",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
