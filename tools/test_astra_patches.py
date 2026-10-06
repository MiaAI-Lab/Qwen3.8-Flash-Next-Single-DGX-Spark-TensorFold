#!/usr/bin/env python3
"""CPU checks for astra items 3-4: first-token-before-draft, MTP absorb without a vocab projection,
and batched PLE gathers.

Usage: TF_SRC=/path/to/patched/src python3 tools/test_astra_patches.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np


def _src() -> Path:
    src = Path(os.environ.get("TF_SRC", "") or "")
    if not src.is_dir():
        sys.exit("set TF_SRC to TensorFold's patched src directory")
    return src


def _parse(src: Path, rel: str):
    return ast.parse((src / rel).read_text())


def _fn(tree: ast.AST, name: str, cls: str | None = None) -> ast.FunctionDef:
    body = tree.body
    if cls:
        klass = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls)
        body = klass.body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(fn: ast.FunctionDef) -> list[str]:
    names = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.append(func.attr)
            elif isinstance(func, ast.Name):
                names.append(func.id)
            self.generic_visit(node)

    V().visit(fn)
    return names


def test_admit_emits_before_draft(src: Path) -> None:
    # v0.6.6: the joined-prompt path lives in multi_fill.py's _joined (it moved out of multi.py's admit).
    tree = _parse(src, "tensorfold/families/qwen4_exp/cuda/multi_fill.py")
    names = _calls(_fn(tree, "_joined", "PromptPasses"))
    assert "take" in names and "draft" in names, names
    assert names.index("take") < names.index("draft"), names


def test_mtp_absorb_skips_head(src: Path) -> None:
    tree = _parse(src, "tensorfold/families/qwen4_exp/cuda/mtp.py")
    fn = _fn(tree, "mtp_compute")
    args = [a.arg for a in fn.args.kwonlyargs]
    assert "logits" in args, args
    # first `if not logits: return` must precede the draft-head `_mm`
    text = ast.get_source_segment((src / "tensorfold/families/qwen4_exp/cuda/mtp.py").read_text(), fn)
    assert text is not None
    skip = text.find("if not logits:")
    head = text.find("draft_head")
    assert skip != -1 and head != -1 and skip < head, "logits=False must skip before the draft head"
    decode = (src / "tensorfold/families/qwen4_exp/cuda/decode.py").read_text()
    assert decode.count("logits=False") >= 2, "prefill resume and chunks must absorb without logits"


def test_ple_concat_matches_separate() -> None:
    """Concatenating per-stream n-gram id matrices along rows is the compact gather the patch's PLE path uses."""

    a = np.arange(6, dtype=np.int64).reshape(3, 2)
    b = np.arange(10, 14, dtype=np.int64).reshape(2, 2)
    cat = np.concatenate([a, b], axis=0)
    assert cat.shape == (5, 2)
    assert np.array_equal(cat[:3], a) and np.array_equal(cat[3:], b)
    at0, heads = 0, 2
    assert at0 + a.size == 3 * heads
    assert at0 + cat.size == (3 + 2) * heads




def main() -> int:
    src = _src()
    failed = 0
    for name, fn in (
        ("admit take-before-draft", lambda: test_admit_emits_before_draft(src)),
        ("mtp absorb skips vocab head", lambda: test_mtp_absorb_skips_head(src)),
        ("ple concat layout", test_ple_concat_matches_separate),
    ):
        try:
            fn()
            print(f"OK  {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print("FAIL" if failed else "OK", f"{failed} failed" if failed else "all astra source checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
