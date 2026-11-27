#!/usr/bin/env python3
"""Build a checkpoint view with the distilled Qwen3.8-Flash-Next MTP draft head overlaid.

Usage: mtp_view.py <base snapshot or ablit view> <mtp-body.pt> <mtp-w3exp.pt> <sidecar .safetensors> <view dir>

Converts prestonlogan/Qwen3.8-Flash-Next-MTP-Draft-Head (BF16 body + Hibana W3 3-bit experts) into a
single safetensors sidecar matching NVIDIA's FP4-Local MTP layout (26 BF16 body tensors + 512 NVFP4
routed experts: 4,634 tensors total; the 3 stock QSA indexer tensors stay untouched), then creates a
relative-symlink checkpoint view under HF_CACHE without modifying the base snapshot.

Works cleanly on top of either the stock NVFP4 snapshot or an ABLIT=1 view: the sidecar is linked as
`tensors/00-mtp-distilled.safetensors` so it sorts before both `house-qsa-*` and `model-*` shards.
"""
import json
import os
import shutil
import struct
import sys
import tempfile

INDEX = "model.safetensors.index.json"
PREFIX = "mtp"
EXPERTS = 512
INTER = 640
HIDDEN = 2560
W3_CODEBOOK = (-3.25, -2.0, -1.125, -0.375, 0.375, 1.125, 2.0, 3.25)

BODY_MAP = {
    f"{PREFIX}.fc_embedding.weight": "fc_e",
    f"{PREFIX}.fc_hidden.weight": "fc_h",
    f"{PREFIX}.pre_fc_norm_embedding.weight": "norm_e",
    f"{PREFIX}.pre_fc_norm_hidden.weight": "norm_h",
    f"{PREFIX}.layers.0.attn_hyper_connection.hc_norm.weight": "attn_hc.norm",
    f"{PREFIX}.layers.0.attn_hyper_connection.input_mix_weight_down.weight": "attn_hc.down",
    f"{PREFIX}.layers.0.attn_hyper_connection.input_mix_weight_up.weight": "attn_hc.up",
    f"{PREFIX}.layers.0.attn_hyper_connection.block_inject_weight.weight": "attn_hc.inject",
    f"{PREFIX}.layers.0.mlp_hyper_connection.hc_norm.weight": "mlp_hc.norm",
    f"{PREFIX}.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": "mlp_hc.down",
    f"{PREFIX}.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": "mlp_hc.up",
    f"{PREFIX}.layers.0.mlp_hyper_connection.block_inject_weight.weight": "mlp_hc.inject",
    f"{PREFIX}.hyper_connection_mixer.hc_norm.weight": "mixer.norm",
    f"{PREFIX}.hyper_connection_mixer.input_mix_weight_down.weight": "mixer.down",
    f"{PREFIX}.hyper_connection_mixer.input_mix_weight_up.weight": "mixer.up",
    f"{PREFIX}.layers.0.self_attn.q_proj.weight": "q_proj",
    f"{PREFIX}.layers.0.self_attn.k_proj.weight": "k_proj",
    f"{PREFIX}.layers.0.self_attn.v_proj.weight": "v_proj",
    f"{PREFIX}.layers.0.self_attn.o_proj.weight": "o_proj",
    f"{PREFIX}.layers.0.self_attn.q_norm.weight": "q_norm",
    f"{PREFIX}.layers.0.self_attn.k_norm.weight": "k_norm",
    f"{PREFIX}.layers.0.mlp.gate.weight": "router",
    f"{PREFIX}.layers.0.mlp.shared_expert_gate.weight": "shared_gate",
    f"{PREFIX}.layers.0.mlp.shared_expert.gate_proj.weight": "shared_gate_proj",
    f"{PREFIX}.layers.0.mlp.shared_expert.up_proj.weight": "shared_up_proj",
    f"{PREFIX}.layers.0.mlp.shared_expert.down_proj.weight": "shared_down_proj",
}


def expected_keys():
    keys = list(BODY_MAP.keys())
    for e in range(EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"{PREFIX}.layers.0.mlp.experts.{e}.{proj}"
            keys.extend((f"{base}.weight", f"{base}.weight_scale", f"{base}.weight_scale_2"))
    return sorted(keys)


def header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def convert_sidecar(body_pt, w3exp_pt, sidecar_path):
    import torch
    from safetensors.torch import save_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"converting distilled MTP head to NVFP4 sidecar on {device} -> {sidecar_path}", flush=True)

    fp4_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device)
    fp4_mids = (fp4_pos[1:] + fp4_pos[:-1]) / 2.0  # [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    lv = torch.tensor(W3_CODEBOOK, dtype=torch.float32, device=device)
    s_idx = torch.arange(8, device=device)
    rot = 16 * (s_idx // 4) + (s_idx % 4)
    j_idx = torch.arange(4, device=device)
    b_pos = [(4 * j_idx[None, :] + b + rot[:, None]) % 32 for b in range(3)]

    def fc1_inverse(inter):
        b = torch.arange(inter // 16, device=device)[:, None]
        j = torch.arange(16, device=device)[None, :]
        order = torch.cat([16 * b + j, inter + 16 * b + j], dim=1).reshape(-1)
        return torch.argsort(order)

    def dequant_w3_batch(planes, scales, row_scale):
        # planes: [B, N/32, K/128, 3, 32, 4] uint32
        B, nb, kc, bits = planes.shape[0], planes.shape[1], planes.shape[2], planes.shape[3]
        n, k = nb * 32, kc * 128
        words = planes.to(torch.int64).permute(0, 1, 2, 4, 3, 5).reshape(B, nb, kc, 32, 4, bits)
        words = words.reshape(B, nb, kc, 8, 4, 4, bits).permute(0, 1, 5, 3, 2, 4, 6)  # [B, nb, nt, g, kc, t, b]
        q = torch.zeros(B, nb, 4, 8, kc, 8, 4, 4, dtype=torch.int64, device=device)
        for b in range(bits):
            pos = b_pos[b]
            q |= ((words[..., b][:, :, :, :, :, None, :, None] >> pos[None, None, None, None, None, :, None, :]) & 1) << b
        q = q.reshape(B, n, k)
        sc = scales.reshape(B, nb, kc, 4, 4, 4, 2).permute(0, 1, 4, 3, 6, 2, 5).reshape(B, n, k // 32)
        sc_eff = sc.contiguous().view(torch.float8_e4m3fn).float() * row_scale[:, :, None]
        return lv[q] * sc_eff.repeat_interleave(32, dim=2)

    def quant_nvfp4_batch(w):
        # w: [B, N, K] float32 -> codes uint8 [B, N, K/2], scales float8_e4m3fn [B, N, K/16], scale2 float32 [B]
        B, n, k = w.shape
        # Global scale per matrix: maps max(|w|) to 6.0 * 448.0 = 2688.0 (ModelOpt NVFP4 convention)
        global_amax = w.abs().amax(dim=(1, 2)).clamp_min(1e-12)
        scale2 = global_amax / (6.0 * 448.0)  # [B] fp32
        g = w.reshape(B, n, k // 16, 16)
        group_amax = g.abs().amax(dim=-1)  # [B, N, K/16]
        block_s = (group_amax / (6.0 * scale2[:, None, None])).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
        eff_s = (block_s.float() * scale2[:, None, None]).clamp_min(1e-30)
        norm = g / eff_s[:, :, :, None]
        sign = (norm < 0).to(torch.uint8) << 3
        mag = torch.bucketize(norm.abs(), fp4_mids).to(torch.uint8)  # 0..7
        nib = (sign | mag).reshape(B, n, k // 2, 2)
        packed = (nib[..., 0] | (nib[..., 1] << 4)).contiguous()
        return packed.cpu(), block_s.contiguous().cpu(), scale2.contiguous().cpu()

    out = {}
    body = torch.load(body_pt, map_location="cpu", weights_only=True)
    for ckpt_k, sd_k in BODY_MAP.items():
        if sd_k not in body:
            sys.exit(f"{body_pt}: missing tensor {sd_k}")
        out[ckpt_k] = body[sd_k].to(torch.bfloat16).contiguous()
    del body

    w3e = torch.load(w3exp_pt, map_location="cpu", weights_only=True)
    inv = fc1_inverse(INTER)
    chunk = 32
    for lo in range(0, EXPERTS, chunk):
        hi = min(EXPERTS, lo + chunk)
        w13 = dequant_w3_batch(
            w3e["planes13"][lo:hi].to(device),
            w3e["scales13"][lo:hi].to(device),
            w3e["row13"][lo:hi].to(device),
        )[:, inv, :]
        up_c, up_s, up_s2 = quant_nvfp4_batch(w13[:, :INTER, :])
        gt_c, gt_s, gt_s2 = quant_nvfp4_batch(w13[:, INTER:, :])
        del w13
        w2 = dequant_w3_batch(
            w3e["planes2"][lo:hi].to(device),
            w3e["scales2"][lo:hi].to(device),
            w3e["row2"][lo:hi].to(device),
        )
        dn_c, dn_s, dn_s2 = quant_nvfp4_batch(w2)
        del w2
        for i in range(hi - lo):
            e = lo + i
            p = f"{PREFIX}.layers.0.mlp.experts.{e}"
            out[f"{p}.gate_proj.weight"] = gt_c[i]
            out[f"{p}.gate_proj.weight_scale"] = gt_s[i]
            out[f"{p}.gate_proj.weight_scale_2"] = gt_s2[i].reshape(())
            out[f"{p}.up_proj.weight"] = up_c[i]
            out[f"{p}.up_proj.weight_scale"] = up_s[i]
            out[f"{p}.up_proj.weight_scale_2"] = up_s2[i].reshape(())
            out[f"{p}.down_proj.weight"] = dn_c[i]
            out[f"{p}.down_proj.weight_scale"] = dn_s[i]
            out[f"{p}.down_proj.weight_scale_2"] = dn_s2[i].reshape(())
    del w3e

    os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
    tmp = sidecar_path + ".tmp"
    save_file(out, tmp)
    os.replace(tmp, sidecar_path)
    print(f"wrote {sidecar_path} ({len(out)} tensors, {os.path.getsize(sidecar_path) / (1 << 20):.1f} MiB)", flush=True)


def check_view(view, side_rel, want):
    with open(os.path.join(view, INDEX)) as f:
        wm = json.load(f)["weight_map"]
    wrong = [k for k in want if wm.get(k) != side_rel]
    if wrong:
        sys.exit(f"{view}: index does not map {wrong[0]} to {side_rel} ({len(wrong)} of {len(want)} wrong)")
    if not os.path.isfile(os.path.join(view, side_rel)):
        sys.exit(f"{view}: {side_rel} does not resolve")


def build(base_dir, body_pt, w3exp_pt, sidecar_path, view):
    want = expected_keys()
    if not os.path.isfile(sidecar_path):
        convert_sidecar(body_pt, w3exp_pt, sidecar_path)
    else:
        sh = header(sidecar_path)
        have = sorted(k for k in sh if k != "__metadata__")
        if have != want:
            convert_sidecar(body_pt, w3exp_pt, sidecar_path)

    side_rel = "tensors/00-" + os.path.basename(sidecar_path)
    if os.path.lexists(view):
        check_view(view, side_rel, want)
        print(f"view {view}: already built, {len(want)} MTP tensors from {side_rel}")
        return

    sh = header(sidecar_path)
    have = sorted(k for k in sh if k != "__metadata__")
    if have != want:
        sys.exit(f"{sidecar_path}: holds {len(have)} tensors, expected {len(want)}")

    with open(os.path.join(base_dir, INDEX)) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    stock_headers = {}
    for k in want:
        if k not in wm:
            sys.exit(f"{base_dir}: index has no {k}: not an NVIDIA FP4-Local snapshot")
        shard = wm[k]
        stock = stock_headers.setdefault(shard, header(os.path.join(base_dir, shard)))[k]
        if (stock["dtype"], stock["shape"]) != (sh[k]["dtype"], sh[k]["shape"]):
            sys.exit(f"{k}: base has {stock['dtype']} {stock['shape']}, sidecar has {sh[k]['dtype']} {sh[k]['shape']}")
    if not all(side_rel < shard for shard in set(wm.values())):
        sys.exit(f"{side_rel} would not sort before every base shard")

    os.makedirs(os.path.dirname(view), exist_ok=True)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(view), prefix=".mtp-")
    try:
        os.chmod(tmp, 0o755)
        os.mkdir(os.path.join(tmp, "tensors"))

        def link(target, at):
            os.symlink(os.path.relpath(os.path.realpath(target), os.path.dirname(at)), at)

        for name in sorted(os.listdir(base_dir)):
            src = os.path.join(base_dir, name)
            if name == INDEX:
                continue
            if name == "tensors" and os.path.isdir(src):
                for shard in sorted(os.listdir(src)):
                    link(os.path.join(src, shard), os.path.join(tmp, "tensors", shard))
            else:
                link(src, os.path.join(tmp, name))
        link(sidecar_path, os.path.join(tmp, side_rel))
        for k in want:
            wm[k] = side_rel
        with open(os.path.join(tmp, INDEX), "w") as f:
            json.dump(idx, f)
            f.write("\n")
        check_view(tmp, side_rel, want)
        os.rename(tmp, view)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"view {view}: base linked, {len(want)} distilled MTP tensors from {side_rel}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    build(*(os.path.abspath(a) for a in sys.argv[1:]))
