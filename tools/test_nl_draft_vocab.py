import gzip, json, tempfile, unittest
from pathlib import Path
from nl_draft_vocab import DEFAULT, EXTENSION, REPORT, ids_sha256, parse_ids, validated_ids

class VocabularyTest(unittest.TestCase):
    def test_actual_union(self):
        base,extra,union=validated_ids()
        self.assertEqual((len(base),len(extra),len(union)),(79591,5176,84767))
        self.assertEqual(ids_sha256(union),"f4d4b6fc44589004fd310951548ae21570508c0a42527a7bf8ebe5946dd03b15")
        self.assertTrue(set(base).isdisjoint(extra))
    def test_invalid(self):
        for v in ([],["1","1"],["2","1"],["-1"],["a"]):
            with self.subTest(v=v), self.assertRaises(ValueError): parse_ids(v,"test")
    def test_modified_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); default=p/"default"; extra=p/"extra"; report=p/"report"
            default.write_bytes(DEFAULT.read_bytes()); extra.write_text(EXTENSION.read_text()); report.write_text(REPORT.read_text())
            default.write_bytes(gzip.compress(gzip.decompress(default.read_bytes()).replace(b"0\n",b"",1),mtime=0))
            with self.assertRaises(ValueError): validated_ids(default,extra,report)
            default.write_bytes(DEFAULT.read_bytes()); extra.write_text(extra.read_text().replace("\n","\n0\n",1))
            with self.assertRaises(ValueError): validated_ids(default,extra,report)
if __name__ == "__main__": unittest.main()
