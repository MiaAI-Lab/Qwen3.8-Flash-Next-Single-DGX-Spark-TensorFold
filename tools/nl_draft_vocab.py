#!/usr/bin/env python3
"""Build the complete Dutch MTP draft-ID list for PR #79's Zig runtime file loader.

The Zig recipe's pinned default list is introduced by patches/0003-config-weights.patch.
Only the 5,176 extra Dutch IDs live in data/nl/draft_vocab_nl.txt. No Zig engine
patch is generated: PR #79 supplies TENSORFOLD_DRAFT_VOCAB support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATCH = ROOT / "patches/0003-config-weights.patch"
DUTCH_EXTENSION = ROOT / "data/nl/draft_vocab_nl.txt"
REPORT = ROOT / "data/nl/report.json"
ZIG_VOCAB_PATH = "zig/src/families/flashnext/cuda_draft_vocab.txt"
PATCH_MARKER = f"diff --git a/{ZIG_VOCAB_PATH} b/{ZIG_VOCAB_PATH}"


def parse_ids(lines: list[str], label: str) -> list[int]:
    """Require the exact ascending, nonduplicated decimal-ID format."""
    if not lines or any(re.fullmatch(r"[0-9]+", line) is None for line in lines):
        raise ValueError(f"{label}: expected nonempty lines of decimal token IDs")
    ids = [int(line) for line in lines]
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError(f"{label}: IDs must be unique and sorted ascending")
    return ids


def default_ids_from_patch(patch: Path) -> list[int]:
    """Extract the new-file hunk exactly as the pinned recipe's git apply does."""
    text = patch.read_text(encoding="utf-8")
    if text.count(PATCH_MARKER) != 1:
        raise ValueError(f"{patch}: missing or ambiguous Zig draft vocabulary hunk")
    section = text.split(PATCH_MARKER, 1)[1].split("\ndiff --git ", 1)[0]
    header = re.search(r"(?m)^@@ -0,0 \+1,([0-9]+) @@(?:.*)$", section)
    if header is None:
        raise ValueError(f"{patch}: Zig vocabulary is not a complete new-file hunk")
    lines = section[header.end():].splitlines()
    if any(not line.startswith("+") or line.startswith("+++") for line in lines if line):
        raise ValueError(f"{patch}: unexpected non-addition in Zig vocabulary hunk")
    added = [line[1:] for line in lines if line]
    if len(added) != int(header.group(1)):
        raise ValueError(f"{patch}: incomplete Zig draft vocabulary hunk")
    return parse_ids(added, "Zig default")


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256("".join(f"{i}\n" for i in ids).encode("ascii")).hexdigest()


def validated_ids(
    patch: Path = DEFAULT_PATCH,
    extension: Path = DUTCH_EXTENSION,
    report_file: Path = REPORT,
) -> tuple[list[int], list[int], list[int]]:
    report = json.loads(report_file.read_text(encoding="utf-8"))
    base = default_ids_from_patch(patch)
    dutch = parse_ids(extension.read_text(encoding="utf-8").splitlines(), "Dutch extension")

    if len(base) != report["base_size"] or ids_sha256(base) != report["sha256"]["base_ids_lf"]:
        raise ValueError("Zig default draft vocabulary changed; re-evaluate Dutch IDs before use")
    if len(dutch) != report["extension_size"] or ids_sha256(dutch) != report["sha256"]["extension"]:
        raise ValueError("Dutch extension differs from the reproducibly generated, pinned list")
    if base[-1] >= report["tokenizer_vocab_size"] or dutch[-1] >= report["tokenizer_vocab_size"]:
        raise ValueError("draft ID outside the tokenizer's vocabulary")
    overlap = set(base).intersection(dutch)
    if overlap:
        raise ValueError(f"Dutch extension duplicates {len(overlap)} default IDs")
    union = sorted([*base, *dutch])
    if len(union) != report["union_size"]:
        raise ValueError("unexpected union size")
    return base, dutch, union


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, help="write complete ID file here (on the host, in HF_CACHE)")
    ap.add_argument("--check", action="store_true", help="validate without writing")
    ap.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    ap.add_argument("--extension", type=Path, default=DUTCH_EXTENSION)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()
    if not args.check and args.output is None:
        ap.error("--output is required without --check")
    base, dutch, union = validated_ids(args.patch, args.extension, args.report)
    print(f"Dutch Zig draft vocabulary: {len(base)} default + {len(dutch)} Dutch = {len(union)} IDs")
    if args.output is not None and not args.check:
        dest = args.output.resolve()
        if dest in {args.patch.resolve(), args.extension.resolve(), args.report.resolve()}:
            ap.error("refusing to overwrite an input file")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(f"{i}\n" for i in union), encoding="ascii", newline="\n")
        print(f"Wrote {dest} (SHA-256: {ids_sha256(union)})")


if __name__ == "__main__":
    main()
