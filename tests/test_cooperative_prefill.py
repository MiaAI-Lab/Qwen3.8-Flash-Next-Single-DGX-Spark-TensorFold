"""Cooperative prefill lifecycle tests, runnable with stdlib unittest and no GPU.

Extract real declarations without importing torch/CUDA entry points. Only tensor,
kernel and worker-thread execution are doubles; scheduling, admission, slot/cache
ownership, Stream.take and completion/error delivery run the production code.
Set TENSORFOLD_SOURCE to a tensorfold package directory to test a different tree.
"""

import ast
import dataclasses
import itertools
import os
import pathlib
import queue
import time
import types
import unittest

ROOT = pathlib.Path(
    os.environ.get("TENSORFOLD_SOURCE", pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorfold")
)


class ThreadDouble:
    def __init__(self, **kw):
        self.kw = kw

    def start(self):
        pass


class Tensor(list):
    def __getitem__(self, k):
        v = super().__getitem__(k)
        return Tensor(v) if isinstance(k, slice) else v

    def clone(self):
        return Tensor(self)


class State:
    def __init__(self, *args):
        self.reset(None)

    def reset(self, w):
        self.pos = 0
        self.ple_history = None
        self.mtp_len = 0
        self.mtp_drafted = 0

    def snapshot(self):
        return {"pos": self.pos}

    def restore(self, s):
        self.pos = s["pos"]

    def set_mtp_len(self, n):
        self.mtp_len = n


class Engine:
    def reset(self):
        self.st.reset(self.w)

    def sample(self, *a):
        return [7]


def extract(rel, names, ns):
    tree = ast.parse((ROOT / rel).read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(  # noqa: S102 — execute trusted local source declarations, without CUDA imports
        compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])), str(ROOT / rel), "exec"),
        ns,
    )
    return ns


class Integration(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.fault = None
        self.rounds = 0
        torch = types.SimpleNamespace(no_grad=lambda: lambda f: f, cuda=types.SimpleNamespace(empty_cache=lambda: None))
        ns = {
            "__name__": __name__,
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "time": time,
            "itertools": itertools,
            "queue": queue,
            "threading": types.SimpleNamespace(Thread=ThreadDouble),
            "torch": torch,
            "Engine": Engine,
            "DEPTH": 0,
            "CONFIDENCE": 0,
            "PREFILL_ROWS": 2,
        }
        extract("cuda/streams.py", {"Stream", "accept"}, ns)
        self.Stream = ns["Stream"]

        def stage(w, b, windows):
            seg = []
            at = 0
            for st, toks in windows:
                seg.append((st, at, at + len(toks)))
                at += len(toks)
                self.events.append(("stage", b.name, id(st), st.pos, list(toks)))
            b.streams = Tensor(range(at))
            return seg

        def compute(w, segs, b, **kw):
            if self.fault and self.fault(b, segs):
                raise RuntimeError("injected kernel boundary")
            if b.name == "decode":
                self.rounds += 1
            return Tensor([7] * segs[-1][2])

        def commit(w, st, b, r, k, at=0):
            st.pos += k
            self.events.append(("commit", b.name, id(st), st.pos))

        ns.update(
            stage=stage,
            compute=compute,
            commit=commit,
            forward=lambda w, st, b, chunk, **kw: compute(w, stage(w, b, [(st, chunk)]), b, **kw),
            sample_streams=lambda logits, starts, positions, samplings: [[7] * len(p) for p in positions],
        )
        extract("families/qwen4_exp/cuda/decode.py", {"prefill", "_prefill_chunks"}, ns)
        extract("families/qwen4_exp/cuda/multi.py", {"_slot", "MultiDecoder"}, ns)
        extract("cuda/scheduler.py", {"Waiting", "Scheduler"}, ns)
        self.ns = ns
        self.d = object.__new__(ns["MultiDecoder"])
        d = self.d
        d.w = types.SimpleNamespace(mtp=None)
        d.depth = 0
        d.capacity = 100
        d.confidence = 0
        d.copy = False
        d.vision = None
        d.eos = ()
        d.keep = 2
        d.kept = []
        d.streams = {}
        d.next_id = 0
        d.free = [State() for _ in range(3)]
        d.prefill_yield = None
        d.buf = types.SimpleNamespace(name="decode", rows=3, streams=Tensor())
        d.mbuf = None
        d.pbuf = types.SimpleNamespace(name="prefill", rows=2, streams=Tensor(), rope_rows=None)
        self.allslots = {id(x) for x in d.free}
        self.sch = ns["Scheduler"](d, max_streams=3)

    def add(self, prompt, count, draft=False):
        s = self.Stream(prompt, count, draft=draft)
        q = queue.Queue()
        s.emit = lambda new: (q.put(("tokens", new)), False)[1]
        self.sch.waiting.put((s, q))
        return s, q

    def drain(self):
        for _ in range(50):
            done = self.sch._admit()
            done += self.d.round()
            self.d.finish(done)
            for s in done:
                self.sch._reply(s, "error" if s.error else "done", s.error or s.stats())
            if not self.d.live() and self.sch.waiting.empty():
                break
        self.assertFalse(self.d.live())
        self.assertTrue(self.sch.waiting.empty())
        self.assertFalse(self.sch.boxes)

    def messages(self, q):
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    def slots_ok(self):
        self.assertEqual(len(self.d.free), len({id(x) for x in self.d.free}))
        owners = (
            {id(x) for x in self.d.free} | {id(x[1]) for x in self.d.kept} | {id(s.st) for s in self.d.streams.values()}
        )
        self.assertEqual(owners, self.allslots)

    def test_actual_global_callback_wiring_and_legacy_optout(self):
        self.ns.update(State=State, Buffers=lambda *a, **kw: object(), _tensors=lambda st: [], os=os)
        w = types.SimpleNamespace(comm=None, cfg=types.SimpleNamespace(eos=[]), mtp=None, draft_ids=None)
        decoder = self.ns["MultiDecoder"](w, slots=2, capacity=100)
        self.assertIsNone(decoder.prefill_yield)
        scheduler = self.ns["Scheduler"](decoder)
        self.assertEqual(decoder.prefill_yield, scheduler._prefill_round)
        self.assertEqual(self.d.prefill_yield, self.sch._prefill_round)
        old = types.SimpleNamespace()
        self.ns["Scheduler"](old)
        self.assertFalse(hasattr(old, "prefill_yield"))

    def test_completion_during_admit_and_reserved_slot_not_reused(self):
        a, qa = self.add([1], 2)
        self.sch._admit()
        b, _qb = self.add(list(range(9)), 3)
        c, _qc = self.add([8, 9], 1)
        since = len(self.events)
        done = self.sch._admit()
        self.assertTrue(a.done)
        self.assertNotEqual(id(a.st), id(b.st))
        self.assertEqual(sum(k == "done" for k, v in self.messages(qa)), 1)
        # c's admission must only begin after b's entire prefill, not inside callback.
        bi = [i for i, e in enumerate(self.events) if len(e) > 2 and e[2] == id(b.st) and e[1] == "prefill"]
        ci = [
            i for i, e in enumerate(self.events) if i >= since and len(e) > 2 and e[2] == id(c.st) and e[1] == "prefill"
        ]
        self.assertLess(max(bi), min(ci))
        self.d.finish(done)
        for s in done:
            self.sch._reply(s, "done", s.stats())
        self.drain()
        self.slots_ok()

    def test_active_cancel_via_real_take_and_finish(self):
        a, qa = self.add([1], 20, draft=True)
        self.sch._admit()
        a.emit = lambda new: True
        _b, _qb = self.add(list(range(9)), 2)
        self.sch._admit()
        self.assertTrue(a.done)
        self.assertNotIn(a.sid, self.d.streams)
        self.assertEqual(len(a.out), 2)
        self.assertEqual(sum(k == "done" for k, v in self.messages(qa)), 1)
        self.drain()
        self.slots_ok()

    def test_callback_error_drops_live_and_aborts_prefill_recovery(self):
        _a, qa = self.add([1], 20, draft=True)
        self.sch._admit()
        _b, qb = self.add(list(range(9)), 2)
        self.fault = lambda buf, segs: buf.name == "decode"
        self.sch._admit()
        self.assertEqual(self.d.live(), 0)
        self.assertEqual(sum(k == "error" for k, v in self.messages(qa)), 1)
        self.assertEqual(sum(k == "error" for k, v in self.messages(qb)), 1)
        prompt_commits = [e for e in self.events if e[0] == "commit" and e[1] == "prefill"]
        self.assertEqual([e[3] for e in prompt_commits], [1, 2])
        self.slots_ok()
        self.fault = None
        self.add([9, 8, 7], 3)
        self.drain()
        self.slots_ok()

    def test_prefill_fault_after_completed_callback_keeps_reply_once(self):
        _a, qa = self.add([1], 2)
        self.sch._admit()
        _b, qb = self.add(list(range(9)), 2)
        self.fault = lambda buf, segs: buf.name == "prefill" and segs[0][0].pos == 2
        self.sch._admit()
        self.assertEqual(sum(k == "done" for k, v in self.messages(qa)), 1)
        self.assertEqual(sum(k == "error" for k, v in self.messages(qb)), 1)
        self.slots_ok()
        self.fault = None
        self.add([2, 3, 4], 3)
        self.drain()
        self.slots_ok()

    def test_cached_prompt_extension_and_eviction_after_yield(self):
        a, _qa = self.add([1, 2, 3], 2, draft=True)
        self.drain()
        self.slots_ok()
        b, _qb = self.add([1, 2, 3, 4, 5], 2, draft=True)
        self.drain()
        self.assertEqual(b.cached, 3)
        self.assertIs(a.st, b.st)
        for j in range(6):
            self.add([j + 20, j + 21], 2, draft=True)
            self.drain()
            self.slots_ok()

    def test_queue_exceeds_slots_without_alias_or_duplicate_terminal(self):
        pairs = [self.add([j + 20] * 5, 3, draft=bool(j % 2)) for j in range(12)]
        self.drain()
        self.slots_ok()
        for s, q in pairs:
            self.assertEqual(len(s.out), 3)
            self.assertEqual(sum(k == "done" for k, v in self.messages(q)), 1)

    def test_immediate_completed_admissions_not_freed_twice(self):
        pairs = [self.add([j] * 5, 1) for j in range(8)]
        self.drain()
        self.slots_ok()
        for s, q in pairs:
            self.assertEqual(sum(k == "done" for k, v in self.messages(q)), 1)

    def test_no_yield_final_or_empty_pending_and_chunk_grid(self):
        for n in range(1, 15):
            calls = []
            st = State()
            e = self.ns["_slot"](self.d.w, st, self.d.buf, None, self.d.pbuf, 100)
            e.prefill_yield = lambda calls=calls, st=st: calls.append(st.pos)
            self.ns["prefill"](e, list(range(n)), None, mtp=False)
            self.assertEqual(calls, list(range(2, n, 2)))
            self.assertEqual(st.pos, n)
            self.assertIsNone(e.pbuf.rope_rows)

    def test_solo_prefill_without_callback(self):
        e = self.ns["_slot"](self.d.w, State(), self.d.buf, None, self.d.pbuf, 100)
        self.assertEqual(self.ns["prefill"](e, [1, 2, 3], None, mtp=False), 7)
        self.assertEqual(e.st.pos, 3)

    def test_resumed_mtp_absorbs_before_yield_and_final_sample(self):
        e = self.ns["_slot"](types.SimpleNamespace(mtp=object()), State(), self.d.buf, object(), self.d.pbuf, 100)
        events = []
        self.ns["mtp_forward"] = lambda w, st, pb, ids, rows: events.append(("mtp", list(ids)))
        e.prefill_yield = lambda: events.append(("yield", e.st.pos, e.st.mtp_len))
        e.sample = lambda *a: events.append(("sample", e.st.pos)) or [7]
        self.ns["prefill"](e, [1, 2, 3, 4, 5], None, resume={"state": {"pos": 2}, "tail": Tensor([1])})
        self.assertEqual(events, [("mtp", [3]), ("mtp", [4, 5]), ("yield", 4, 3), ("sample", 5)])
        self.assertEqual(e.first, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
