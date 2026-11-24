# Per-request abliteration: one checkpoint, stock and abliterated

Patches 0006 to 0008 let the server apply an abliteration at run time, per request. The served checkpoint is the stock
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
| `TENSORFOLD_IDENTITIES` | `id=path` entries separated by commas (at most 32): the model id a request names and its directions file, either a path inside the container or `hf:owner/repo/file[@revision][:sha256:HEX]` on Hugging Face. `/v1/models` lists every id |
| `TENSORFOLD_ABLATE`, `TENSORFOLD_ABLATE_MODEL` | one identity the short way (the file, the model id); read only while `TENSORFOLD_IDENTITIES` is unset |
| `TENSORFOLD_ABLATE_SCALE` | multiplies every identity's strengths (default 1), for tuning runs |
| `ABLIT=2`, `ABLIT_DIRECTIONS_*` | the house-qsa identity in one switch (below) |

**An identity needs an API key or a loopback host.** An identity is a model whose refusals have been removed, so
`start.sh` refuses to start with any identity (`TENSORFOLD_IDENTITIES`, `TENSORFOLD_ABLATE`, `ABLIT=2`) when the server
would listen on a non-loopback address with no API key: set `TENSORFOLD_API_KEY` (or pass `--api-key` / `--api-key-file`),
or `HOST=127.0.0.1`. The key then guards every model id, the stock one included; only `/health` stays open.

`start.sh` passes every `TENSORFOLD_*` variable into the container and mounts the Hugging Face cache at
`/root/.cache/huggingface`, so a file under `~/.cache/huggingface/ablate/` needs no extra mount:

```bash
export TENSORFOLD_API_KEY=$(openssl rand -hex 24)
TENSORFOLD_IDENTITIES="Qwen3.8-Flash-Next-abliterated=/root/.cache/huggingface/ablate/abliterated.safetensors" ./start.sh
```

A file on Hugging Face is downloaded by `scripts/prepare.sh` into the same cache, like the checkpoint, and `start.sh`
gives the engine its path. A gated repo needs its terms accepted by the account of `HF_TOKEN` (or `hf auth login`).
Pin the file with `@<commit>` and `:sha256:<its SHA-256>`: `prepare.sh` checks the download against it and `start.sh`
checks it again at every start, and either stops on a mismatch. An entry without a pin is served, and `prepare.sh` prints
the file's SHA-256 so you can add one.

The engine logs one `identity i id: ...` line each at load; a file that fails its checks stops the load with the reason.

## ABLIT=2

`ABLIT=2` sets this up for drowzeys's house-qsa abliteration (the overlay `ABLIT=1` grafts): the stock model as
`SERVED_NAME`, unchanged, and the abliterated one as `SERVED_NAME-abliterated`, both from the stock weights.

```bash
TENSORFOLD_API_KEY=... ABLIT=2 ./start.sh restart
```

The directions file is recorded in `scripts/config.sh` like the other pinned downloads: `ABLIT_DIRECTIONS_FILE`
`house-qsa.safetensors` in `ABLIT_DIRECTIONS_ID` `Loke-60000/Qwen3.8-Flash-Next-FP4-Local-house-qsa-directions`, at
`ABLIT_DIRECTIONS_REVISION` `5983c8cc`, 246,192 bytes, `ABLIT_DIRECTIONS_SHA256`
`f5941a3b085e340c4e3bb94319b7a76367fe0130e24f30a2013d9767a75978dc`; `prepared_state` records all four. It is the per-layer file
`tools/ablate_from_overlay.py` recovers from the overlay's sidecar (`fp4-local/house-qsa-oproj-mxfp8-L3-47.safetensors`
at `c44f5b92`) and its `HOUSE_QSA_MXFP8.json`; you can write your own the same way (below) and list it in
`TENSORFOLD_IDENTITIES` yourself, which takes precedence. Like the overlay, it fits only
`nvidia/Qwen3.8-Flash-Next-FP4-Local` at `b9ce18a6`. Don't list identities with `ABLIT=1`: then every request, the stock
id included, already runs on the overlay's abliterated weights.

## Directions files

A global file applies the same directions at every layer from `first_layer` up:

| Tensor | Meaning |
|---|---|
| `directions` [k, d] fp32 | orthonormal rows, k <= 4 |
| `lam_attn` [k], `lam_shared` [k] fp32 | the strengths for the token mixers and the shared expert |
| `first_layer` [1] int64 | the first layer edited |

A per-layer file (patch 0008) edits only the layers it lists, each with a direction of its own, the way
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
  stock model's or another identity's. The server marks the request with the identity its `model` names
  (`Request.named`, kept in `Request.identity`), which also changes its prompt-cache key.
  `zig/src/core/lane_host_identity_test.zig` covers it.
- A pass whose rows name no identity launches nothing extra, and an identity's kernels run only at the layers it edits.
- Two ranks (TP=2) are refused at load.
- The server refuses to start when the list is malformed, names an id twice, has more than 32 entries, or uses the served
  name or an alias as an identity id, and the load stops on a `first_layer` past the last layer (47).

## Checks

`tools/verify_ablate.py` sends one long prompt twice to each model id it is given (default: the stock `SERVED_NAME`
and `SERVED_NAME-abliterated`): the second pass resumes that id's own kept state and its greedy reply matches the cold one.
Set `API_KEY` for a keyed server. `start.sh` runs it after its smoke test whenever identities are listed (and the window
holds its ~12k-token prompt), on the stock id and every identity, and fails the start if it fails. `tools/exact.py` and `tools/toolcheck.py` take `MODEL=SERVED_NAME-abliterated` to run
on the identity. Offline, `tools/test_identities_hf.py` covers the `hf:` entries and their SHA-256 pin, and
`tools/test_identities_gate.py` the API-key gate; the engine side is `zig/src/core/lane_host_identity_test.zig`.
What has been run on a Spark is listed in the README, under [ABLIT=2](../README.md#ablit2-stock-and-abliterated-from-one-server).
