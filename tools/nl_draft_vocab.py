#!/usr/bin/env python3
"""Validate and emit the complete Dutch CUDA draft vocabulary."""
from __future__ import annotations
import argparse, gzip, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT / "data/nl/draft_vocab_default.txt.gz"
EXTENSION=ROOT / "data/nl/draft_vocab_nl.txt"
REPORT=ROOT / "data/nl/report.json"

def parse_ids(lines, label):
    if not lines or any(re.fullmatch(r"[0-9]+", line) is None for line in lines):
        raise ValueError(f"{label}: expected nonempty decimal IDs")
    ids=[int(line) for line in lines]
    if any(a >= b for a,b in zip(ids,ids[1:])): raise ValueError(f"{label}: IDs must be sorted and unique")
    return ids

def ids_sha256(ids):
    return hashlib.sha256("".join(f"{i}\n" for i in ids).encode("ascii")).hexdigest()

def validated_ids(default=DEFAULT, extension=EXTENSION, report_file=REPORT):
    report=json.loads(report_file.read_text(encoding="utf-8"))
    base=parse_ids(gzip.decompress(default.read_bytes()).decode("ascii").splitlines(),"default")
    extra=parse_ids(extension.read_text(encoding="utf-8").splitlines(),"Dutch extension")
    if len(extra)!=report["extension_size"] or ids_sha256(extra)!=report["sha256"]["extension"]: raise ValueError("Dutch extension changed")
    if not set(base).isdisjoint(extra): raise ValueError("Dutch extension overlaps default")
    union=sorted([*base,*extra])
    if len(base)!=report["base_size"] or ids_sha256(base)!=report["sha256"]["base_ids_lf"]: raise ValueError("default vocabulary changed")
    if len(union)!=report["union_size"] or union[-1]>=report["tokenizer_vocab_size"]: raise ValueError("invalid complete vocabulary")
    return base, extra, union

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--check",action="store_true"); ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if not args.check and args.output is None: ap.error("--output required unless --check")
    base,extra,union=validated_ids(); print(f"{len(base)} default + {len(extra)} Dutch = {len(union)} IDs")
    if args.output and not args.check:
        dest=args.output.resolve()
        if dest in {DEFAULT.resolve(), EXTENSION.resolve(), REPORT.resolve()}: ap.error("refusing to overwrite input")
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text("".join(f"{i}\n" for i in union),encoding="ascii",newline="\n")
        print(f"Wrote {dest} (SHA-256 {ids_sha256(union)})")
if __name__ == "__main__": main()
