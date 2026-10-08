#!/usr/bin/env python3
"""The MTP layer's bf16 linears on block FP8, for TF_FLASHNEXT_MTP_FP8 (patches/0011).

Usage: tools/mtp_fp8.py SNAPSHOT OUTDIR      (scripts/prepare.sh runs it in the image, which has torch and safetensors)

INT4-AutoRound keeps the MTP layer in bf16: its attention q/k/v/o, both hyper-connections' down/up and the mixer's,
~139 MB that every draft step reads. This writes them as e4m3 codes (`.weight`) and fp32 scales
(`.weight_scale_inv`, one per 128x128 block: scale = block amax / 448, weight ~= codes * scale), the format of the
checkpoint's own block-FP8 tensors, into OUTDIR/mtp-fp8.safetensors. The engine loads them in place of the bf16 ones.
The MTP head only drafts and every draft is verified, so replies stay byte-identical; only the drafts can change.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NAMES = [f"mtp.layers.0.self_attn.{p}_proj" for p in "qkvo"]
NAMES += [f"mtp.layers.0.{b}_hyper_connection.input_mix_weight_{d}" for b in ("attn", "mlp") for d in ("down", "up")]
NAMES += [f"mtp.hyper_connection_mixer.input_mix_weight_{d}" for d in ("down", "up")]


def block_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = w.shape
    rb, cb = (rows + 127) // 128, (cols + 127) // 128
    pad = torch.zeros(rb * 128, cb * 128, dtype=torch.float32)
    pad[:rows, :cols] = w.float()
    blocks = pad.view(rb, 128, cb, 128)
    amax = blocks.abs().amax(dim=(1, 3))
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    codes = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes.view(rb * 128, cb * 128)[:rows, :cols].contiguous(), scale.contiguous()


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    snap, outdir = sys.argv[1], sys.argv[2]
    weight_map = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for name in NAMES:
        key = name + ".weight"
        if key not in weight_map:
            print(f"{key}: not in the checkpoint", file=sys.stderr)
            return 1
        with safe_open(os.path.join(snap, weight_map[key]), "pt") as f:
            w = f.get_tensor(key)
        if w.dtype != torch.bfloat16 or w.dim() != 2:
            print(f"{key}: {w.dtype} {tuple(w.shape)}, expected a bf16 matrix", file=sys.stderr)
            return 1
        out[key], out[name + ".weight_scale_inv"] = block_fp8(w)
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, "mtp-fp8.safetensors.tmp")
    save_file(out, tmp)
    os.replace(tmp, os.path.join(outdir, "mtp-fp8.safetensors"))
    print(f"{len(NAMES)} MTP linears on block FP8 in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
