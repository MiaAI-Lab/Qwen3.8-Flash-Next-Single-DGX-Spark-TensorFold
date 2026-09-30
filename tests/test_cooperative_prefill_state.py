"""Actual commit/read-ahead source; NumPy state, CUDA shift kernel stand-in.
No GPU required. NumPy is the only non-stdlib dependency; not native GPU proof.
"""

import ast
import os
import pathlib
import types
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = pathlib.Path(
    os.environ.get("TENSORFOLD_SOURCE", pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorfold")
)


def extract(rel, names, ns):
    tree = ast.parse((ROOT / rel).read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    for n in nodes:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(  # noqa: S102 — execute trusted local source declarations, without CUDA imports
        compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])), str(ROOT / rel), "exec"),
        ns,
    )
    return ns


class Seams(unittest.TestCase):
    def test_actual_commit_preserves_other_slot_history_and_tail(self):
        # shift_windows CUDA implementation is replaced, but commit's offsets,
        # PLE history, ple_last lifecycle, cur handling, and pos are real source.
        def shift(old, new, keep, channels):
            old[:] = np.concatenate([old, new[:, :, :channels]], axis=1)[:, keep : keep + old.shape[1], :]

        ns = extract("families/qwen4_exp/cuda/forward.py", {"commit"}, {"np": np, "shift_windows": shift})

        def state(seed):
            s = types.SimpleNamespace(
                pos=2,
                cur=[],
                ple_history=np.array([seed, seed + 1]),
                ple_last=None,
                ple_tail=np.array([[seed], [seed + 1]]),
            )
            s.set_pos = lambda n: setattr(s, "pos", n)
            return s

        pending, active = state(10), state(90)
        w = types.SimpleNamespace(cfg=types.SimpleNamespace(ngram_size=3, conv_dim=1))
        pb = types.SimpleNamespace(prefill=True, ple_nrow=np.array([[12], [13], [14]]))
        db = types.SimpleNamespace(prefill=False, ple_nrow=np.array([[0], [92], [93], [94]]))
        pending.ple_last = (pending.ple_history, np.array([12, 13, 14]))
        ns["commit"](w, pending, pb, 3, 3)
        self.assertEqual(pending.ple_history.tolist(), [13, 14])
        self.assertEqual(pending.ple_tail.tolist(), [[13], [14]])
        snap = (pending.pos, pending.ple_history.copy(), pending.ple_tail.copy())
        active.ple_last = (active.ple_history, np.array([92, 93, 94]))
        ns["commit"](w, active, db, 3, 2, at=1)
        self.assertEqual(active.ple_history.tolist(), [92, 93])
        self.assertEqual(active.ple_tail.tolist(), [[92], [93]])
        self.assertEqual(active.pos, 4)
        self.assertIsNone(active.ple_last)
        self.assertEqual(pending.pos, snap[0])
        np.testing.assert_array_equal(pending.ple_history, snap[1])
        np.testing.assert_array_equal(pending.ple_tail, snap[2])
        pending.ple_last = (pending.ple_history, np.array([15]))
        pb.ple_nrow = np.array([[15]])
        ns["commit"](w, pending, pb, 1, 1)
        self.assertEqual(pending.ple_history.tolist(), [14, 15])
        self.assertEqual(pending.pos, 6)
        with self.assertRaises(ValueError):
            ns["commit"](w, pending, pb, 2, 1)

    def test_actual_read_ahead_interleaved_decode_cannot_return_wrong_rows(self):
        class Table:
            def gather(self, ids):
                return np.asarray(ids).copy() * 17

        ns = extract(
            "families/qwen4_exp/host_table.py",
            {"ReadAhead", "_key"},
            {"np": np, "ThreadPoolExecutor": ThreadPoolExecutor},
        )
        a = ns["ReadAhead"](Table())
        try:
            prompt = np.array([[1, 2], [3, 4]])
            decode = np.array([[101, 102]])
            a.read_ahead(prompt)
            np.testing.assert_array_equal(a.gather(decode), decode * 17)
            self.assertEqual(len(a._ahead), 1)
            np.testing.assert_array_equal(a.gather(prompt), prompt * 17)
            self.assertEqual(len(a._ahead), 0)
            for i in range(5):
                a.read_ahead(np.array([i]))
            self.assertEqual(len(a._ahead), a.depth)
            for i in range(5):
                np.testing.assert_array_equal(a.gather(np.array([i])), np.array([i]) * 17)
        finally:
            a._pool.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
