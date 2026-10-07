#!/usr/bin/env python3
"""Copy/MTP proposal-row ownership in MultiDecoder._draft_all, without loading the model.

Copy drafts drop copy-covered streams from `active`, but their compact MTP rows stay in `logits`.
`_picks` reads those rows sequentially, so a remaining stream can sample another stream's proposal.
The patch compacts logits to the surviving todo ordinals. Residual `row` stays a buffer offset.

Usage:
  TF_SRC=/path/to/patched/src python3 tools/test_copy_draft_rows.py

TF_SRC is TensorFold's `src` directory with this recipe's patches already applied (0002 + 0010 on v0.6.6).
Exit code 1 if any case misaligns. This checks control-flow row ownership, not model accuracy.
"""
from __future__ import annotations

import ast
import os
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS

COPY_TOKEN = 99


class Rows:
    """Minimal compact-row container: production uses a torch tensor; `logits[keep]` must work."""

    def __init__(self, xs):
        self.xs = [int(x) for x in xs]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return Rows(self.xs[i] for i in idx)
        if isinstance(idx, slice):
            return Rows(self.xs[idx])
        return self.xs[idx]

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        return iter(self.xs)


class State:
    def __init__(self, logit_id: int):
        self.mtp_drafted = 0
        self.mtp_len = 0
        self.pos = 10
        self.logit_id = logit_id

    def set_mtp_len(self, n):
        self.mtp_len = n


def extract_draft_all(src: Path):
    path = src / "tensorfold/families/qwen4_exp/cuda/multi.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiDecoder")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_draft_all")
    staged = []

    def stage(w, b, windows):
        segs, at = [], 0
        for st, keep, _ in windows:
            segs.append((st, at, at + len(keep)))
            at += len(keep)
        staged.append([(st.logit_id, len(keep), at0, at1) for st, at0, at1 in segs])
        return segs

    def compute(w, segs, b):
        return Rows(st.logit_id for st, _, _ in segs)

    scope = dict(COPY_MATCH=8, mtp_stage=stage, mtp_compute=compute)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    return scope["_draft_all"], staged


def stream(sid: int, copied: bool, logit_id: int, keep: list[int], sampling=None):
    s = NS(sid=sid, st=State(logit_id), out=[1], count=100, sampling=sampling,
           context=[1], copies=NS(propose=lambda context, n: [COPY_TOKEN] * n if copied else []),
           drafts=None)
    return s, sid - 1, keep   # a0 is unused by the fake stage; keep length sets residual a1-1


def run_case(draft_all, copies, keeps, depth=1, samplings=None):
    n = len(copies)
    ids = [10 * (i + 1) + 1 for i in range(n)]          # 11, 21, 31, ...
    samplings = samplings or [NS(name=f"s{i}") for i in range(n)]
    todo = [stream(i + 1, copies[i], ids[i], keeps[i], samplings[i]) for i in range(n)]
    streams = [t[0] for t in todo]
    decoder = NS(depth=depth, confidence=0.6, w=None,
                 mbuf=NS(streams=list(range(64))), buf=NS(streams=list(range(64))),
                 _mtp=lambda segs: Rows(st.logit_id for st, _, _ in segs),
                 _picks=lambda logits, positions, smp: [(int(logits[i]), 1.0) for i in range(len(positions))])
    new = {s.sid: [2] for s in streams}
    draft_all(decoder, todo, new)
    want, got = [], []
    for s, copied, logit_id in zip(streams, copies, ids):
        expected = [COPY_TOKEN] * min(depth, 99) if copied else [logit_id] * min(depth, 99)
        # room = min(depth, count - len(out) - len(keep)); depth=1 and room>=1 → one draft
        room = min(depth, s.count - len(s.out) - len(next(k for t, a0, k in todo if t is s)))
        expected = expected[:room]
        want.append(expected)
        got.append(list(s.drafts))
    return want, got, streams


def main() -> int:
    src = Path(os.environ.get("TF_SRC", "") or "")
    if not src.is_dir():
        sys.exit("set TF_SRC to TensorFold's patched src directory (this recipe's patches 0001-0010 applied)")
    draft_all, _staged = extract_draft_all(src)
    failed = 0

    # Two-stream matrix from docs/astra.md, plus both-copy / neither-copy.
    print("two streams, keep length 1:")
    for copy_a, copy_b in product((False, True), repeat=2):
        want, got, _ = run_case(draft_all, [copy_a, copy_b], [[1], [1]])
        ok = got == want
        failed += not ok
        print(f"  copy={copy_a, copy_b} expected {want} actual {got} aligned {ok}")

    # Middle, first, last, and two-of-three removals; residual rows diverge from logit rows.
    print("three streams, mixed keep lengths 2/1/3 (residual a1-1 is not the compact logit index):")
    keeps = [[1, 1], [1], [1, 1, 1]]
    for copies in product((False, True), repeat=3):
        want, got, streams = run_case(draft_all, list(copies), keeps)
        ok = got == want
        failed += not ok
        if not ok or (any(copies) and not all(copies)):
            print(f"  copy={copies} expected {want} actual {got} aligned {ok}")

    print("three streams, identical keep lengths, every removed position:")
    for copies in (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ):
        want, got, _ = run_case(draft_all, list(copies), [[1], [1], [1]])
        ok = got == want
        failed += not ok
        print(f"  copy={copies} expected {want} actual {got} aligned {ok}")

    print("mixed sampling objects stay paired with surviving streams:")
    want, got, streams = run_case(
        draft_all, [True, False, True], [[2], [1], [3]],
        samplings=[NS(name="A"), NS(name="B"), NS(name="C")])
    ok = got == want and streams[1].sampling.name == "B"
    failed += not ok
    print(f"  expected {want} actual {got} sampling {streams[1].sampling.name} aligned {ok}")

    print("all-copy returns without MTP picks; no-copy keeps source rows:")
    want, got, _ = run_case(draft_all, [True, True], [[1], [1]])
    ok = got == want
    failed += not ok
    print(f"  all-copy expected {want} actual {got} aligned {ok}")
    want, got, _ = run_case(draft_all, [False, False, False], [[2], [1], [1]])
    ok = got == want
    failed += not ok
    print(f"  no-copy expected {want} actual {got} aligned {ok}")

    print("FAIL" if failed else "OK", f"{failed} misaligned" if failed else "all cases aligned")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
