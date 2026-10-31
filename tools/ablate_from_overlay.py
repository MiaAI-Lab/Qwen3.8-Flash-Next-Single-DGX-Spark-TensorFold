#!/usr/bin/env python3
"""A per-layer identity file (patch 0020) from a baked abliteration overlay: for each edited o_proj, the overlay holds
W' = D (W - lam v v^T W) (D a per-row rescale, e.g. a row-norm restore), MXFP8 like the stock FP4-Local checkpoint.
Given the stock weights and each layer's lam, this recovers v (the left singular vector of W - W'/D) and D, and writes
`layers`, `directions`, `lam`, `row_scale` for TENSORFOLD_IDENTITIES. It prints each layer's reconstruction error, which
should match the overlay's own requantisation error.

Usage: tools/ablate_from_overlay.py STOCK_SNAPSHOT OVERLAY.safetensors LAMBDAS.json OUT.safetensors
  LAMBDAS.json: {"layers": [{"L": 3, "lambda": 6.95}, ...]} (the house-qsa sidecar's HOUSE_QSA_MXFP8.json has this form)
Needs torch and safetensors (the image's base, nvcr.io/nvidia/pytorch, has both); runs on the CPU."""
import json, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "model.language_model.layers."


def dequant(w, s):
    """MXFP8 g32: F8_E4M3 weights, one E8M0 scale (U8) per 32 inputs."""
    w = w.view(torch.float8_e4m3fn).float()
    return (w.view(w.shape[0], -1, 32) * torch.pow(2.0, s.float() - 127.0).unsqueeze(-1)).view(w.shape)


def load(path, name):
    with safe_open(path, "pt") as f:
        return dequant(f.get_tensor(name + ".weight"), f.get_tensor(name + ".weight_scale"))


def fit(W, H, lam, rounds=6):
    D = torch.ones(W.shape[0])
    for _ in range(rounds):
        v = torch.linalg.svd(W - H / D[:, None], full_matrices=False)[0][:, 0]
        D = W.norm(dim=1) / (W - lam * torch.outer(v, v @ W)).norm(dim=1)
    recon = D[:, None] * (W - lam * torch.outer(v, v @ W))
    return v, D, ((recon - H).norm() / H.norm()).item()


def main():
    stock, overlay, lambdas, out = sys.argv[1:5]
    wmap = json.load(open(f"{stock}/model.safetensors.index.json"))["weight_map"]
    layers, dirs, lams, scales = [], [], [], []
    for entry in sorted(json.load(open(lambdas))["layers"], key=lambda e: e["L"]):
        L, lam = int(entry["L"]), float(entry["lambda"])
        name = f"{PREFIX}{L}.self_attn.o_proj"
        v, D, rel = fit(load(f"{stock}/{wmap[name + '.weight']}", name), load(overlay, name), lam)
        print(f"layer {L:2d}: lam {lam:.2f}, reconstruction error {rel:.4f}, row scale [{D.min():.3f}, {D.max():.3f}]", flush=True)
        layers.append(L); dirs.append(v); lams.append(lam); scales.append(D)
    save_file({"layers": torch.tensor(layers, dtype=torch.int64), "directions": torch.stack(dirs).contiguous(),
               "lam": torch.tensor(lams, dtype=torch.float32), "row_scale": torch.stack(scales).contiguous()}, out)
    print(f"wrote {out}: {len(layers)} layers")


if __name__ == "__main__":
    main()
