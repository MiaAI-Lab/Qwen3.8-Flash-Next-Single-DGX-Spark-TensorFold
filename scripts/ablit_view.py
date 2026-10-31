#!/usr/bin/env python3
"""Build the checkpoint folder ABLIT=1 serves: NVIDIA's FP4-Local snapshot with the Keys house-qsa sidecar grafted in.

Usage: ablit_view.py <stock snapshot> <sidecar .safetensors> <view dir>

The sidecar holds `o_proj.weight` and `o_proj.weight_scale` of the 12 QSA layers (MXFP8 g32), requantized after the house
projection. tensorfold-native lists the shards named by model.safetensors.index.json, in path order, and a tensor
name is taken from the first of them that holds it. So the view links every file of the stock snapshot, adds the sidecar
as `tensors/house-qsa-...` (which sorts before the stock `tensors/model-...` shards) and writes an index that points the
24 names at it. Every file is a relative symlink, so the view works under the container's mount of HF_CACHE; nothing in
the snapshot is touched or copied. A view that already exists and checks out is reused, never rewritten.
Standard library only.
"""
import json
import os
import shutil
import struct
import sys
import tempfile

LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
PREFIX = "model.language_model.layers."
WANT = [f"{PREFIX}{layer}.self_attn.o_proj.{part}" for layer in LAYERS for part in ("weight", "weight_scale")]
INDEX = "model.safetensors.index.json"


def header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def check_view(view, side_rel):
    """The index names every sidecar tensor from the sidecar, and the sidecar resolves."""
    with open(os.path.join(view, INDEX)) as f:
        wm = json.load(f)["weight_map"]
    wrong = [k for k in WANT if wm.get(k) != side_rel]
    if wrong:
        sys.exit(f"{view}: the index does not map {wrong[0]} to {side_rel} ({len(wrong)} of {len(WANT)} wrong)")
    if not os.path.isfile(os.path.join(view, side_rel)):
        sys.exit(f"{view}: {side_rel} does not resolve")


def build(snap, side, view):
    side_rel = "tensors/" + os.path.basename(side)
    if os.path.lexists(view):
        check_view(view, side_rel)
        print(f"view {view}: already built, {len(WANT)} tensors from {side_rel}")
        return

    sh = header(side)
    have = sorted(k for k in sh if k != "__metadata__")
    if have != sorted(WANT):
        sys.exit(f"{side}: holds {len(have)} tensors, not the {len(WANT)} house-qsa o_proj weights and scales"
                 f" (first difference: {next((k for k in have + WANT if (k in have) != (k in WANT)), '?')})")
    with open(os.path.join(snap, INDEX)) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    stock_headers = {}
    for k in WANT:
        if k not in wm:
            sys.exit(f"{snap}: the index has no {k}: this is not the FP4-Local snapshot the sidecar was made for")
        shard = wm[k]
        stock = stock_headers.setdefault(shard, header(os.path.join(snap, shard)))[k]
        if (stock["dtype"], stock["shape"]) != (sh[k]["dtype"], sh[k]["shape"]):
            sys.exit(f"{k}: the stock snapshot has {stock['dtype']} {stock['shape']}, the sidecar {sh[k]['dtype']} {sh[k]['shape']}")
    if not all(side_rel < shard for shard in set(wm.values())):
        sys.exit(f"{side_rel} would not sort before every stock shard, so a stock tensor could win")

    os.makedirs(os.path.dirname(view), exist_ok=True)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(view), prefix=".ablit-")
    try:
        os.chmod(tmp, 0o755)
        os.mkdir(os.path.join(tmp, "tensors"))

        def link(target, at):
            os.symlink(os.path.relpath(target, os.path.dirname(at)), at)

        for name in sorted(os.listdir(snap)):
            src = os.path.join(snap, name)
            if name == INDEX:
                continue
            if name == "tensors" and os.path.isdir(src):
                for shard in sorted(os.listdir(src)):
                    link(os.path.join(src, shard), os.path.join(tmp, "tensors", shard))
            else:
                link(src, os.path.join(tmp, name))
        link(side, os.path.join(tmp, side_rel))
        for k in WANT:
            wm[k] = side_rel
        with open(os.path.join(tmp, INDEX), "w") as f:
            json.dump(idx, f)
            f.write("\n")
        check_view(tmp, side_rel)
        os.rename(tmp, view)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"view {view}: stock snapshot linked, {len(WANT)} tensors from {side_rel}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    build(*(os.path.abspath(a) for a in sys.argv[1:]))
