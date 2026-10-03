#!/usr/bin/env python3
"""CPU tests: real patched loader, generator budget, config and benchmark validity.

TF_SRC=/path/to/patched/TensorFold/src python tools/test_nl_draft.py
No torch/GPU/model weights. Loader functions are extracted from actual patched
source and executed with real NumPy, as in this recipe's existing CPU tests.
"""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from build_nl_draft_vocab import select, read_ids, extension_patch, count
from compare_nl_bench import compare, serial_equality
from bench_nl import jobs

ROOT = Path(__file__).resolve().parents[1]


class GeneratorTests(unittest.TestCase):
    def test_preserves_base_and_pins(self):
        ids, cov = select({0, 1}, Counter({2: 90, 3: 10}), {4}, 4, .9)
        self.assertEqual(ids, [2, 4])
        self.assertEqual(cov, .9)

    def test_ties_sort_by_id(self):
        ids, _ = select({0}, Counter({3: 2, 2: 2}), set(), 2, 1)
        self.assertEqual(ids, [2])

    def test_budget_and_unmet_target(self):
        ids, cov = select({0}, Counter({1: 2, 2: 1}), set(), 2, 1)
        self.assertEqual(len(ids), 1)
        self.assertLess(cov, 1)

    def test_invalid_budget_target_empty(self):
        for counts, budget, target in ((Counter({2: 1}), 1, .99), (Counter({2: 1}), 4, 0), (Counter(), 4, 1)):
            with self.assertRaises(ValueError):
                select({0, 1}, counts, set(), budget, target)

    def test_base_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ids"
            for data in ("", "1\n1\n", "-1\n", "10\n", "x\n"):
                path.write_text(data)
                with self.assertRaises(ValueError):
                    read_ids(path, 10)

    def test_patch_has_correct_length(self):
        self.assertIn("@@ -0,0 +1,2 @@\n+3\n+4\n", extension_patch([3, 4]))

    def test_corpus_identities_and_empty_input(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                from types import SimpleNamespace
                return SimpleNamespace(ids=[1, 2])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            path.write_text('{"id":"nl-1","text":"Hallo"}\n', encoding="utf-8")
            counts, docs, identities = count(path, FakeTokenizer())
            self.assertEqual(docs, 1)
            self.assertEqual(sum(counts.values()), 2)
            self.assertIn(("id", "nl-1"), identities)
            path.write_text('{"text":""}\n')
            with self.assertRaises(ValueError):
                count(path, FakeTokenizer())


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = os.environ.get("TF_SRC")
        if not src:
            raise RuntimeError("set TF_SRC to the fully patched TensorFold src directory")
        cls.path = Path(src) / "tensorfold/families/qwen4_exp/cuda/weight_types.py"
        tree = ast.parse(cls.path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if (isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "DRAFT_LANGUAGES" for t in n.targets)) or
                 (isinstance(n, ast.FunctionDef) and n.name in ("draft_vocab_setting", "draft_token_ids"))]
        scope = {"__file__": str(cls.path), "np": np, "Path": Path, "os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(cls.path), "exec"), scope)
        cls.load = staticmethod(scope["draft_token_ids"])
        cls.setting = staticmethod(scope["draft_vocab_setting"])
        cls.codes = scope["DRAFT_LANGUAGES"]

    def test_nl_union_and_no_base_loss(self):
        base = self.load("default")
        nl = self.load("nl")
        ext = np.loadtxt(self.path.with_name("draft_vocab_nl.txt"), dtype=np.int64).reshape(-1)
        np.testing.assert_array_equal(nl, np.union1d(base, ext))
        self.assertTrue(set(base) <= set(nl))
        self.assertEqual(len(ext), len(set(ext)))
        self.assertEqual(sorted(ext.tolist()), ext.tolist())
        self.assertFalse(set(ext) & set(base))
        report = json.loads((ROOT / "data/nl/report.json").read_text())
        self.assertEqual(len(nl), report["union_size"])
        self.assertTrue(all(0 <= i < report["tokenizer_vocab_size"] for i in nl))
        shipped = ROOT / "data/nl/draft_vocab_nl.txt"
        self.assertEqual(shipped.read_bytes(), self.path.with_name("draft_vocab_nl.txt").read_bytes())
        self.assertEqual(hashlib.sha256(shipped.read_bytes()).hexdigest(), report["sha256"]["extension"])
        patch = ROOT / "patches/languages/0011-flash-next-draft-nl.patch"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), report["sha256"]["patch"])
        base_text = "".join(f"{int(i)}\n" for i in base)
        self.assertEqual(hashlib.sha256(base_text.encode()).hexdigest(), report["sha256"]["base_ids_lf"])

    def test_combinations_and_duplicate_codes(self):
        for code in self.codes:
            with self.subTest(code=code):
                np.testing.assert_array_equal(self.load("nl,"+code), np.union1d(self.load("nl"), self.load(code)))
        np.testing.assert_array_equal(self.load("nl,nl"), self.load("nl"))

    def test_full_default_env_and_custom_file(self):
        self.assertIsNone(self.load("full"))
        self.assertIsNone(self.load(None))
        np.testing.assert_array_equal(self.load(5), np.arange(5))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "custom.txt"
            path.write_text("3\n1\n3\n")
            np.testing.assert_array_equal(self.load(str(path)), [1, 3])
        original = os.environ.pop("TENSORFOLD_DRAFT_VOCAB", None)
        try:
            self.assertEqual(self.setting(), "default")
            os.environ["TENSORFOLD_DRAFT_VOCAB"] = " nl,de "
            self.assertEqual(self.setting(), "nl,de")
        finally:
            os.environ.pop("TENSORFOLD_DRAFT_VOCAB", None)
            if original is not None:
                os.environ["TENSORFOLD_DRAFT_VOCAB"] = original

    def test_unknown_language_fails(self):
        for value in ("xx", "nl,xx", ",", "missing-file.txt"):
            with self.assertRaises(ValueError):
                self.load(value)

    def test_missing_extension_fails(self):
        # Use the actual function in a directory without bundled data.
        original = self.load.__globals__["__file__"]
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "weight_types.py"
                path.with_name("draft_vocab.txt").write_text("1\n2\n")
                self.load.__globals__["__file__"] = str(path)
                with self.assertRaises(FileNotFoundError):
                    self.load("nl")
        finally:
            self.load.__globals__["__file__"] = original


class ConfigTests(unittest.TestCase):
    def shell(self, code, cwd=ROOT, value=None):
        env = dict(os.environ)
        for name in ("DRAFT_LANGUAGE", "IMAGE", "TENSORFOLD_DRAFT_VOCAB"):
            env.pop(name, None)
        if value is not None:
            env["DRAFT_LANGUAGE"] = value
        bash = os.environ.get("BASH_EXECUTABLE") or shutil.which("bash")
        if not bash:
            self.fail("Bash is required for config validation")
        # Git Bash accepts forward slashes in Windows paths.
        command = f'source "{(ROOT / "scripts/config.sh").as_posix()}"; check_draft_language; ' + code
        return subprocess.run([bash, "-c", command], cwd=cwd, env=env, capture_output=True, text=True)

    def test_valid_and_invalid_codes(self):
        for value in ("", "nl", "nl,de", "de,fr,ja,nl,pt,ru,zh"):
            self.assertEqual(self.shell("true", value=value).returncode, 0, value)
        for value in ("NL", "xx", "nl,xx", "nl,", ",nl", "nl de", "nl;true"):
            self.assertNotEqual(self.shell("true", value=value).returncode, 0, value)

    def test_nl_image_environment_and_hash(self):
        result = self.shell('printf "%s|%s\\n" "$IMAGE" "$TENSORFOLD_DRAFT_VOCAB"; patches_hash', value="nl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("v0.6.1-languages|nl", result.stdout)
        default = self.shell("patches_hash", value="")
        german = self.shell("patches_hash", value="de")
        self.assertEqual(result.stdout.splitlines()[-1], german.stdout.strip())
        self.assertNotEqual(default.stdout.strip(), german.stdout.strip())

    def test_dotenv_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / ".env").write_text('DRAFT_LANGUAGE="nl,de"\n')
            self.assertEqual(self.shell('printf "%s" "$TENSORFOLD_DRAFT_VOCAB"', cwd=temp).stdout, "nl,de")
            self.assertEqual(self.shell('printf "%s" "$TENSORFOLD_DRAFT_VOCAB"', cwd=temp, value="nl").stdout, "nl")


class BenchmarkTests(unittest.TestCase):
    def record(self, arm):
        return dict(arm=arm, block="01", clients=1, actual_clients=1, repeat=0, name="nl-1",
                    manifest_sha256="m", request_sha256="p", group="nl", temperature=0,
                    thinking=False, seed=42, max_tokens=1024, ok=True, error=None,
                    finish_reason="length", cached=0, decode_s=2 if arm == "A" else 1,
                    completion_tokens=100, ttft=.1, elapsed=3, output_sha="o", token_ids=[1, 2],
                    drafted=90, accepted=60, batch_wall_s=3)

    def test_valid_pair(self):
        report = compare({1: self.record("A")}, {1: self.record("B")})
        self.assertTrue(report["valid"])
        self.assertEqual(report["groups"][0]["median_paired_speed_ratio"], 2)

    def test_rejects_failures_cache_drift_and_missing_equality(self):
        for field, value in (("ok", False), ("cached", 10), ("cached", None),
                             ("token_ids", None), ("token_ids", [2, 3]), ("output_sha", "bad"),
                             ("request_sha256", "different"), ("decode_s", 0), ("finish_reason", None),
                             ("completion_tokens", 99), ("manifest_sha256", None)):
            row = self.record("B")
            row[field] = value
            self.assertFalse(compare({1: self.record("A")}, {1: row})["valid"], field)

    def test_pair_set_mismatch(self):
        with self.assertRaises(ValueError):
            compare({1: self.record("A")}, {})

    def test_serial_equality(self):
        row = self.record("A")
        serial = {**row, "arm": "serial", "draft": False}
        self.assertEqual(serial_equality({1: row}, {1: serial}), [])
        self.assertTrue(serial_equality({1: row}, {1: {**serial, "token_ids": [9]}}))
        self.assertTrue(serial_equality({1: row}, {}))

    def test_fresh_prompts_and_serial_control(self):
        requests = [{"id": "nl", "prompt": "Leg iets uit."}]
        a = jobs(requests, "01", 1, 0)
        self.assertEqual(a, jobs(requests, "01", 1, 0))
        self.assertNotEqual(a[0]["prompt"], jobs(requests, "01", 1, 1)[0]["prompt"])
        self.assertNotEqual(a[0]["prompt"], jobs(requests, "01", 1, 0, warmup=True)[0]["prompt"])
        self.assertFalse(jobs(requests, "01", 1, 0, serial=True)[0]["draft"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
