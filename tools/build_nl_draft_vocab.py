#!/usr/bin/env python3
"""Build a deterministic Dutch extension, preserving TensorFold's default vocabulary.

Uses tokenizer.json directly: no remote code, model weights, torch or GPU needed.
Training and holdout inputs are UTF-8 JSONL with a text field. Outputs token IDs,
a patch baked into the language image, and a provenance/coverage report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ids(path, vocab_size):
    values = [int(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)) or any(i < 0 or i >= vocab_size for i in values):
        raise ValueError("base vocabulary must contain unique, in-range token IDs")
    return set(values)


def count(path, tokenizer):
    counts, documents, identities = Counter(), 0, set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row["text"]
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            if text.strip():
                identities.add(("text", hashlib.sha256(text.encode()).hexdigest()))
                if "id" in row:
                    identities.add(("id", str(row["id"])))
                counts.update(tokenizer.encode(text, add_special_tokens=False).ids)
                documents += 1
    if not counts:
        raise ValueError(f"empty token corpus: {path}")
    return counts, documents, identities


def select(base, counts, pinned, max_size, target):
    keep = set(base) | set(pinned)
    if not 0 < target <= 1 or max_size < len(keep):
        raise ValueError("invalid coverage target or budget smaller than base/pinned IDs")
    total = sum(counts.values())
    if not total:
        raise ValueError("empty training counts")
    covered = sum(counts[i] for i in keep)
    for tid, frequency in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        if covered / total >= target or len(keep) >= max_size:
            break
        if tid not in keep:
            keep.add(tid)
            covered += frequency
    return sorted(keep - base), covered / total


def coverage(counts, keep):
    return sum(n for i, n in counts.items() if i in keep) / sum(counts.values())


def extension_patch(ids):
    if not ids:
        raise ValueError("no extension IDs selected")
    path = "tensorfold/families/qwen4_exp/cuda/draft_vocab_nl.txt"
    return f"--- /dev/null\n+++ {path}\n@@ -0,0 +1,{len(ids)} @@\n" + "".join(f"+{i}\n" for i in ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--max-size", type=int, default=100000, help="total base + extension budget (v0.6.1 base is 79,591)")
    ap.add_argument("--coverage", type=float, default=.995, help="training selection target, not a speed claim")
    ap.add_argument("--output-dir", type=Path, default=Path("data/nl"))
    ap.add_argument("--patch", type=Path, default=Path("patches/languages/0011-flash-next-draft-nl.patch"))
    args = ap.parse_args()
    if args.train.resolve() == args.holdout.resolve() or sha(args.train) == sha(args.holdout):
        ap.error("training and holdout must be different files")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    spec = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    base = read_ids(args.base, tok.get_vocab_size())
    pinned = {t["id"] for t in spec.get("added_tokens", [])}
    pinned |= {i for piece, i in tok.get_vocab().items() if len(piece) == 1 or
               (piece.startswith("<0x") and piece.endswith(">"))}
    train, train_docs, train_ids = count(args.train, tok)
    ids, train_coverage = select(base, train, pinned, args.max_size, args.coverage)
    holdout, holdout_docs, holdout_ids = count(args.holdout, tok)
    if train_ids & holdout_ids:
        raise ValueError("train/holdout share article IDs or identical texts")
    keep = base | set(ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab = args.output_dir / "draft_vocab_nl.txt"
    vocab.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8", newline="\n")
    args.patch.parent.mkdir(parents=True, exist_ok=True)
    args.patch.write_text(extension_patch(ids), encoding="utf-8", newline="\n")
    report = {
        "schema": 1, "provenance": json.loads(args.provenance.read_text(encoding="utf-8")),
        "sha256": {"tokenizer": sha(args.tokenizer),
                   "base_ids_lf": hashlib.sha256("".join(f"{i}\n" for i in sorted(base)).encode()).hexdigest(),
                   "train": sha(args.train), "holdout": sha(args.holdout), "extension": sha(vocab),
                   "patch": sha(args.patch)},
        "tokenizer_vocab_size": tok.get_vocab_size(), "base_size": len(base),
        "extension_size": len(ids), "union_size": len(keep), "max_size": args.max_size,
        "selection_target": args.coverage, "training_target_reached": train_coverage >= args.coverage,
        "train": {"documents": train_docs, "tokens": sum(train.values()),
                  "base_coverage": coverage(train, base), "nl_coverage": train_coverage},
        "holdout": {"documents": holdout_docs, "tokens": sum(holdout.values()),
                    "base_coverage": coverage(holdout, base), "nl_coverage": coverage(holdout, keep)},
        "split_overlap": 0,
        "status": "Wikipedia seed vocabulary; model-output coverage and DGX Spark speed unmeasured",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
