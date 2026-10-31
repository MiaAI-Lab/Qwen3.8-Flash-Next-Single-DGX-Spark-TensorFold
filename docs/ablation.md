# Per-request abliteration: one checkpoint, stock and abliterated

Patches 0018 to 0020 let the server apply an abliteration at run time, per request. The served checkpoint is the stock
one; a request that names an identity in `TENSORFOLD_IDENTITIES` gets that abliterated model's reply from the same
weights, in the same decode round as everyone else. Requests that name no identity are unchanged byte for byte.

## How it works

An abliterated build edits the residual-stream writers: `W' = W - R diag(lam) R^T W`, with `R` the refusal directions
(orthonormal rows) and `lam` the strengths, so the writer's output is `y - R diag(lam) (R^T y)`. The engine subtracts
that term from the writers' outputs, for the rows of requests that named an identity: the token mixers' output
projections (`linear_attn.out_proj`, `self_attn.o_proj`) with `lam_attn` and the shared expert's down projection with
`lam_shared`, from the identity's `first_layer` up. The MTP draft head is untouched; the main forward verifies its drafts.

A baked checkpoint requantises the edited weights, so it is close to, not bit-equal with, the run-time correction.

## Settings

| Setting | Meaning |
|---|---|
| `TENSORFOLD_IDENTITIES` | `id=path` entries separated by commas (at most 32): the model id a request names and its directions file, either a path inside the container or `hf:owner/repo/file[@revision]` on Hugging Face. `/v1/models` lists every id |
| `TENSORFOLD_ABLATE`, `TENSORFOLD_ABLATE_MODEL` | one identity the short way (the file, the model id); read only while `TENSORFOLD_IDENTITIES` is unset |
| `TENSORFOLD_ABLATE_SCALE` | multiplies every identity's strengths (default 1), for tuning runs |

`start.sh` passes every `TENSORFOLD_*` variable into the container and mounts the Hugging Face cache at
`/root/.cache/huggingface`, so a file under `~/.cache/huggingface/ablate/` needs no extra mount:

```bash
TENSORFOLD_IDENTITIES="Qwen3.8-Flash-Next-abliterated=/root/.cache/huggingface/ablate/abliterated.safetensors" ./start.sh
```

A file on Hugging Face is downloaded by `scripts/prepare.sh` into the same cache, like the checkpoint, and `start.sh`
gives the engine its path. A gated repo needs its terms accepted by the account of `HF_TOKEN` (or `hf auth login`).
drowzeys's house-qsa abliteration (the overlay `ABLIT=1` grafts) as a directions file, for example:

```bash
TENSORFOLD_IDENTITIES="Qwen3.8-Flash-Next-abliterated=hf:Loke-60000/Qwen3.8-Flash-Next-FP4-Local-house-qsa-directions/house-qsa.safetensors" ./start.sh
```

The engine logs one `identity i id: ...` line each at load; a file that fails its checks stops the load with the reason.
`ABLIT=2` sets this up for drowzeys's house-qsa abliteration in one switch: the stock model as `SERVED_NAME` and the
abliterated one as `SERVED_NAME-abliterated`, with the directions file downloaded at a pinned revision (`ABLIT_DIRECTIONS`).
A `TENSORFOLD_IDENTITIES` of your own takes precedence. Don't list identities with `ABLIT=1`: then every request, the
stock id included, already runs on the overlay's abliterated weights.

## Directions files

A global file applies the same directions at every layer from `first_layer` up:

| Tensor | Meaning |
|---|---|
| `directions` [k, d] fp32 | orthonormal rows, k <= 4 |
| `lam_attn` [k], `lam_shared` [k] fp32 | the strengths for the token mixers and the shared expert |
| `first_layer` [1] int64 | the first layer edited |

A per-layer file (patch 0020) edits only the layers it lists, each with a direction of its own, the way
`W' = D (W - lam v v^T W)` does (`D` a per-row rescale, as after a row-norm restore). At a listed layer the token
mixer's output becomes `D (y - lam (v . y) v)`; other layers and the shared expert are left alone:

| Tensor | Meaning |
|---|---|
| `layers` [n] int64 | the layers it edits, increasing |
| `directions` [n, d] fp32 | one unit direction a listed layer |
| `lam` [n] fp32 | each layer's strength |
| `row_scale` [n, d] fp32 | optional: each layer's `D` (ones when absent) |

`tools/ablate_from_overlay.py` writes a per-layer file from a baked overlay of `self_attn.o_proj` weights (MXFP8, like the
stock checkpoint) and each layer's `lam`, for example drowzeys's house-qsa sidecar (the one `ABLIT=1` downloads) and its `HOUSE_QSA_MXFP8.json`.
It prints each layer's reconstruction error, which should match the overlay's own requantisation error. It needs torch,
which the image's base has:

```bash
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface -v "$PWD":/w nvcr.io/nvidia/pytorch:26.07-py3 \
  python3 /w/tools/ablate_from_overlay.py /root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/<rev> \
  /w/house-qsa-oproj-mxfp8-L3-47.safetensors /w/HOUSE_QSA_MXFP8.json /root/.cache/huggingface/ablate/house-qsa.safetensors
```

## Behaviour

- Each identity keeps prompt states of its own: a conversation's next turn resumes its own identity's state, never the
  stock model's or another identity's. `zig/src/core/lane_host_reuse_test.zig` covers it.
- Two ranks (TP=2) are refused at load.
- The server refuses to start when the list is malformed, names an id twice, or has more than 32 entries.

## Checks

`tools/verify_ablate.py` sends one long prompt twice to each model id it is given (default: the stock `SERVED_NAME`
and `SERVED_NAME-abliterated`): the second pass resumes that id's own kept state and its greedy reply matches the cold one.
