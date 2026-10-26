#!/usr/bin/env python3
"""Dependency-free checks of the Dutch extension against the Zig recipe default."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nl_draft_vocab import (
    PATCH_MARKER,
    default_ids_from_patch,
    ids_sha256,
    parse_ids,
    validated_ids,
)


class DutchDraftVocabularyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default, cls.extra, cls.union = validated_ids()

    def test_real_recipe_and_exact_dutch_extension(self):
        self.assertEqual((len(self.default), len(self.extra), len(self.union)), (79591, 5176, 84767))
        self.assertEqual(len(set(self.union)), len(self.union))
        self.assertEqual(self.union, sorted(self.union))
        self.assertTrue(set(self.default).issubset(self.union))

    def test_original_hashes_still_match(self):
        self.assertEqual(ids_sha256(self.default), "88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e")
        self.assertEqual(ids_sha256(self.extra), "098fab3f73098ed48b3486e9e586f560be46368a718f96103e5bec32e9600498")

    def test_reject_unsorted_duplicate_or_nonnumeric_ids(self):
        for text in (["1", "1"], ["2", "1"], ["-1"], ["2 # note"], []):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_ids(text, "synthetic")

    def test_extract_new_file_hunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            patch = Path(tmp) / "patch.diff"
            patch.write_text(f"{PATCH_MARKER}\nnew file mode 100644\n--- /dev/null\n+++ b/id.txt\n@@ -0,0 +1,3 @@\n+0\n+2\n+4\ndiff --git a/next b/next\n", encoding="utf-8")
            self.assertEqual(default_ids_from_patch(patch), [0, 2, 4])
            patch.write_text(patch.read_text().replace("+4\n", ""), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                default_ids_from_patch(patch)

    def test_reject_overlap_and_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patch, ext, report = (root / p for p in ("base.patch", "nl.txt", "report.json"))
            patch.write_text(f"{PATCH_MARKER}\n--- /dev/null\n+++ b/id.txt\n@@ -0,0 +1,3 @@\n+0\n+2\n+4\n", encoding="utf-8")
            ext.write_text("1\n3\n", encoding="utf-8")
            r = {"base_size": 3, "extension_size": 2, "union_size": 5, "tokenizer_vocab_size": 10,
                 "sha256": {"base_ids_lf": ids_sha256([0, 2, 4]), "extension": ids_sha256([1, 3])}}
            report.write_text(json.dumps(r), encoding="utf-8")
            self.assertEqual(validated_ids(patch, ext, report)[2], [0, 1, 2, 3, 4])
            # A default-ID duplicate is rejected even when the extension's hash is updated.
            ext.write_text("2\n3\n", encoding="utf-8")
            r["sha256"]["extension"] = ids_sha256([2, 3])
            report.write_text(json.dumps(r), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicates"):
                validated_ids(patch, ext, report)
            # Updating the patch without updating provenance must stop the build.
            patch.write_text(patch.read_text().replace("+4\n", "+5\n"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                validated_ids(patch, ext, report)


if __name__ == "__main__":
    unittest.main()
