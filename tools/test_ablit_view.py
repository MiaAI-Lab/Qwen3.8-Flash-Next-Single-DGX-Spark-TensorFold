#!/usr/bin/env python3
"""scripts/ablit_view.py builds the ABLIT=1 checkpoint folder from a stock snapshot and a sidecar.

Checks on tiny fake safetensors files: the 24 house-qsa names come from the sidecar and sort first, every other
file of the snapshot is linked, nothing in the snapshot is touched, a finished view is reused, and a wrong sidecar
(a missing name, another shape) or a stock tensor that would sort before the sidecar is refused.

Usage: tools/test_ablit_view.py        (standard library only; no GPU, no Docker, no download)
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ablit_view", ROOT / "scripts" / "ablit_view.py")
av = importlib.util.module_from_spec(spec)
spec.loader.exec_module(av)


def safetensors(path: Path, tensors: dict) -> None:
    """tensors: name -> (dtype, shape); the data is zeros."""
    head, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        head[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(head).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(off))


def make_snapshot(snap: Path, shard_dir="tensors") -> dict:
    other = "model.language_model.layers.0.mlp.gate.weight"
    stock = {n: ("F8_E4M3" if n.endswith("weight") else "U8", (4, 8)) for n in av.WANT}
    safetensors(snap / shard_dir / "model-00001-of-00002.safetensors", {**stock, other: ("BF16", (2, 2))})
    safetensors(snap / shard_dir / "model-00002-of-00002.safetensors", {"lm_head.weight": ("BF16", (2, 2))})
    wm = {n: f"{shard_dir}/model-00001-of-00002.safetensors" for n in [*av.WANT, other]}
    wm["lm_head.weight"] = f"{shard_dir}/model-00002-of-00002.safetensors"
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": wm}))
    (snap / "config.json").write_text("{}")
    return stock


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "ablit_view.py"), *map(str, args)],
                          capture_output=True, text=True)


def main() -> int:
    fails = []

    def check(ok, what):
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snap, view = tmp / "hub" / "snap", tmp / "tensorfold-qwen38" / "view"
        stock = make_snapshot(snap)
        side = tmp / "hub" / "overlay" / "fp4-local" / "house-qsa-oproj-mxfp8-L3-47.safetensors"
        safetensors(side, stock)
        before = sorted(str(p.relative_to(snap)) for p in snap.rglob("*"))
        index_before = (snap / "model.safetensors.index.json").read_text()

        r = run(snap, side, view)
        check(r.returncode == 0, f"builds the view ({r.stderr.strip() or r.stdout.strip()})")
        wm = json.loads((view / "model.safetensors.index.json").read_text())["weight_map"]
        sidecar_rel = "tensors/house-qsa-oproj-mxfp8-L3-47.safetensors"
        check(all(wm[k] == sidecar_rel for k in av.WANT), "all 24 names map to the sidecar")
        check(wm["lm_head.weight"] == "tensors/model-00002-of-00002.safetensors", "other names keep their stock shard")
        check(len(av.WANT) == 24, "24 names")
        check(sorted(set(wm.values()))[0] == sidecar_rel, "the sidecar sorts first")
        check(os.path.realpath(view / sidecar_rel) == os.path.realpath(side), "the sidecar link reaches the overlay file")
        check((view / "config.json").is_file() and (view / "tensors" / "model-00002-of-00002.safetensors").is_file(),
              "stock files are linked and resolve")
        check(all(p.is_symlink() for p in view.rglob("*") if p.is_file() and p.name != "model.safetensors.index.json"),
              "every file but the index is a symlink")
        check(all(not os.path.isabs(os.readlink(p)) for p in view.rglob("*") if p.is_symlink()), "every link is relative")
        check(json.loads((view / "model.safetensors.index.json").read_text()).get("metadata") == {"total_size": 1},
              "the index keeps its metadata")
        check(sorted(str(p.relative_to(snap)) for p in snap.rglob("*")) == before
              and (snap / "model.safetensors.index.json").read_text() == index_before, "the snapshot is untouched")
        check(not [p for p in view.parent.iterdir() if p.name.startswith(".ablit-")], "no temporary folder is left")

        marker = view / "marker"
        marker.write_text("kept")
        r = run(snap, side, view)
        check(r.returncode == 0 and "already built" in r.stdout and marker.read_text() == "kept", "a finished view is reused as it is")

        (view / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        r = run(snap, side, view)
        check(r.returncode != 0 and marker.exists(), "a broken view is refused, not rewritten")

        # a sidecar that lacks one name
        short = {k: v for k, v in stock.items() if k != av.WANT[0]}
        side2 = tmp / "short.safetensors"
        safetensors(side2, short)
        r = run(snap, side2, tmp / "v2")
        check(r.returncode != 0 and not (tmp / "v2").exists(), "a sidecar with a name missing is refused")

        # another shape than the stock tensor
        odd = dict(stock)
        odd[av.WANT[0]] = ("F8_E4M3", (4, 16))
        side3 = tmp / "odd.safetensors"
        safetensors(side3, odd)
        r = run(snap, side3, tmp / "v3")
        check(r.returncode != 0 and not (tmp / "v3").exists(), "a sidecar with another shape is refused")

        # a stock shard that would sort before the sidecar (the first file holding a name wins in the engine)
        snap4 = tmp / "hub" / "snap4"
        stock4 = make_snapshot(snap4, shard_dir="abc")
        side4 = tmp / "hub" / "overlay4" / "house-qsa-oproj-mxfp8-L3-47.safetensors"
        safetensors(side4, stock4)
        r = run(snap4, side4, tmp / "v4")
        check(r.returncode != 0 and "sort before" in r.stderr and not (tmp / "v4").exists(),
              "a stock shard that sorts before the sidecar is refused")

    print(f"{len(fails)} failed" if fails else "all passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
