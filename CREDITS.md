# Credits

This repository is a thin layer of scripts. Almost everything that makes it work was built by others. Its own work is
licensed under the MIT License ([`LICENSE`](LICENSE)). The image `scripts/prepare.sh` builds also carries TensorFold's
license files from the pinned checkout.

## Model

- **[Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** by the
  [Qwen team](https://huggingface.co/Qwen) (Alibaba): the model's design, training and evaluations, its chat template
  and its MTP head. Its license, the
  [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), comes with the
  weights.
- **NVIDIA** ([nvidia](https://huggingface.co/nvidia)): the checkpoint this recipe serves by default,
  [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local): Qwen3.8 Flash Next quantized with
  quantization-aware distillation (QAD) against the bf16 model as teacher, using NVIDIA Model Optimizer: NVFP4 routed experts with CSF-compressed
  scale factors, MXFP8 attention, DeltaNet and shared-expert linears, an NVFP4 n-gram table and quantized MTP and vision towers. On NVIDIA's own
  numbers it scores 91.5% on GPQA Diamond and 88.7% on ToolEvalBench against 92.2% and 89.4% for bf16 (its model card, not measured here). Used under the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/); the base model's
  Qwen Community License 1.0 applies as well.
- **drowzeys** ([drowzeys](https://huggingface.co/drowzeys)): the Keys house-qsa abliteration overlay for the NVIDIA checkpoint,
  [`drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47`](https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-FP4-QAD-house-qsa-L3-47): the requantized
  `self_attn.o_proj` weights of 12 layers that `ABLIT=1` grafts onto the checkpoint. Gated; its license is NVIDIA's and the Qwen license. Not part of this repository:
  `scripts/prepare.sh` downloads the sidecar when `ABLIT=1`. `scripts/ablit_view.py` is MiaAI-Lab's own and uses none of the overlay's scripts.
- **azampatti** ([azampatti](https://huggingface.co/azampatti)): the optional INT4 checkpoint (`CHECKPOINT=int4`),
  [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound):
  azampatti authored its top-5 expert cut, the shared-expert healing, and the checkpoint. Under the license on
  its model card (the Qwen license).
- **Intel** ([Intel](https://huggingface.co/Intel)): the AutoRound int4 quantization the INT4-AR checkpoint is built on,
  [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound).
- **Saren-Arterius** ([Saren-Arterius](https://github.com/Saren-Arterius)): the hybrid checkpoint the INT4-AR
  checkpoint is built from and its FP8 n-gram table,
  [`Saren/Qwen3.8-Flash-Next-ple-table-fp8`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8) (in its
  `ple-table/`). The recipe uses their weights only, no code from their repositories.
- **[Vontra](https://huggingface.co/Vontra)**: the checkpoint this recipe served before the Zig engine,
  [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP): the MLX
  4-bit conversion, the preserved native MTP draft head, validation and packaging.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** and its Zig engine (`tensorfold-native`, branch
  [`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext)) by Ash Hart
  ([ashhart](https://github.com/ashhart)) and the TensorFold contributors (Apache 2.0 from v0.6.0; code written before
  v0.6.0 keeps its MIT notice): the engine that serves the model, its CUDA runtime and kernels, MTP drafting with
  exact verification and the OpenAI-compatible server. `scripts/prepare.sh` builds it from TensorFold's source at the
  commit pinned in `scripts/config.sh`.
  The Flash Next CUDA engine in this recipe is ported from TensorFold's Python Flash Next CUDA engine, written by
  Ash Hart and the TensorFold contributors ([contributors](https://github.com/ashhart/TensorFold/graphs/contributors));
  their authorship is recorded in TensorFold's history.
- **The 4 patches in `patches/` (0001 to 0004) on TensorFold v1.2.7 are written by MiaAI-Lab** (Mia's AI Lab,
  [MiaAI-Lab](https://github.com/MiaAI-Lab)), under this
  repository's MIT License. They modify TensorFold's own Flash Next CUDA engine, which is by Ash Hart and the TensorFold
  contributors; that code keeps TensorFold's licenses (see `NOTICE`). Patch by patch:
  - `0001-vision-nvidia-fp4-tower`: the CUDA vision tower loads NVIDIA's MXFP8 and NVFP4 vision linears, decoded to bf16 at load. MiaAI-Lab.
  - `0002-vision-image-tokens-16384`: image-token budget 16,384. MiaAI-Lab.
  - `0003-kv-cache-fp8`: `--kv-cache fp8` for Flash Next, so five full 262,144-token windows fit; the fp8 reader keeps the next tile in registers and
    the MTP head's cache stays bf16 while it fits (the v1.1.4 series' 0003 and 0005, rebased onto v1.2.7). MiaAI-Lab; the format is adapted from
    MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` in
    [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
  - `0004-prompt-mxfp8-tile`: prompt passes run the wide MXFP8 projections on a 128 x 128 tile (64 x 64 for short calls) with the block scales folded
    into the weights (the v1.1.4 series' 0006 and 0015, ported to v1.2.7's qmmf lane; the 64 x 64 form is new). MiaAI-Lab, built on TensorFold's qmmf
    kernels and fragment helpers by Ash Hart and the TensorFold contributors.
- **The v1.1.4 series** (patches 0001 to 0017 on TensorFold v1.1.4, written by MiaAI-Lab, retired with the move to v1.2.7): v1.2.7 carries most of
  what 0004 and 0006 to 0017 did in TensorFold's own code, written by Ash Hart and the TensorFold contributors; TensorFold's K32 prompt expert staging
  is adapted from patch 0014 (credited in TensorFold's `THIRD_PARTY_NOTICES.md`). The series as it was: the prompt expert tile (0004), prefill kernels
  (0006, 0014, 0015), sampled streams in the batched head and GPU top-k candidates (0007, 0008; the approach is Track B of MiaAI-Lab's DeepSeek recipe
  [DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold), patch 0021),
  a round-cost model, n-gram rows of all lanes at once and host draws on threads (0009 to 0011), pending DeltaNet rows (0012), the `pread` n-gram reader
  (0013, carrying forward the db281878 series' port of the Python engine's reader by the TensorFold Contributors), the shared expert on a side stream and
  lane-batched DeltaNet (0016) and the running-product stop rule (0017, the db281878 series' rule).
- **Earlier series on TensorFold db281878** (patches 0011 to 0015, 0021 and 0029 to 0031, on `main`, retired on this branch):
  - NVIDIA FP4 support in the Zig engine (`0011` to `0015`): by MiaAI-Lab, as a port of TensorFold's Python implementation of the same formats (CSF scale
    decoding, MXFP8 linears, the NVFP4 n-gram table and its read-ahead), written by the TensorFold Contributors, to the Zig engine and its CUDA kernels.
    The engine's own new files say "Written for TensorFold by TensorFold Contributors".
  - Free after drain and admission reclaim (`0029`), image prompts in the prompt cache (`0030`): by [tempeste](https://github.com/tempeste), on
    TensorFold's prompt cache, lane host and vision input by Ash Hart and the TensorFold contributors. Both are built into TensorFold v1.1.4.
  - Sliced fills one at a time (`0021`): by [BadAd84](https://github.com/BadAd84) (PR #53), rebased onto the FP4 engine by MiaAI-Lab.
  - `TENSORFOLD_RANDOM_SEED` (`0031`): MiaAI-Lab; v1.1.4 has its own random-seed handling.
- **The Zig CUDA serving path and the CUDA family registry** were authored by Jürgen Schmied
  ([jschmied](https://github.com/jschmied)) in [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
  (commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac)):
  `tensorfold-native` serving CUDA families through one registry, which this recipe's engine runs on.
- TensorFold itself builds on, and credits in its
  [third-party notices](https://github.com/ashhart/TensorFold/blob/zig-flashnext/THIRD_PARTY_NOTICES.md), the
  projects whose code it adapts. Those notices name
  [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple, MIT),
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (Prince Canuma, MIT),
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT), and the Qwen Flash Next modeling code in
  Hugging Face [transformers](https://github.com/huggingface/transformers) (the Qwen Team and the Hugging Face team,
  Apache 2.0).
- **[b12x](https://github.com/local-inference-lab/b12x)** by local-inference-lab (Apache-2.0): the one-shot RoCE
  all-gather in the engine implements the RoCEnante protocol of b12x by local-inference-lab; that implementation is
  new code written for this engine. This recipe serves one Spark, so it does not turn that path on.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and the Zig contributors (MIT): the language and
  compiler TensorFold's Zig engine is written in and built with (0.17.0).

## Earlier patches on the Python engine

This recipe no longer applies those patches (the serving path is the Zig engine). The people who authored them:

- `0001-cuda-live-token-counters`: by MiaAI-Lab, submitted upstream as
  [TensorFold #79](https://github.com/ashhart/TensorFold/pull/79). The recipe's earlier typed-tool-parameters patch
  ([#75](https://github.com/ashhart/TensorFold/pull/75)) is part of TensorFold v0.3.6.3.
- `0006-flash-next-prefill-rows`: a port of [TensorFold #40](https://github.com/ashhart/TensorFold/pull/40) by
  **[MovieMaker93](https://github.com/MovieMaker93)**, rebased onto v0.3.6.3.
- `0007-flash-next-copy-drafts`: uses TensorFold's own `CopyIndex` prompt-lookup index from its Qwen3.5 27B engine.
- `0008-flash-next-vision`: builds on TensorFold's Qwen3.5/3.8 dense vision support, runs the vision tower from
  Hugging Face transformers, and follows transformers' Qwen3.5 rotary index and Qwen3-VL's video processing.
  Image and video input on the Zig engine is adapted from this patch and from patch 0009 (many images), both by
  MiaAI-Lab, and from TensorFold's vision code by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors.
- `0010-flash-next-draft-languages` (the optional language image, not built anymore): the language token lists come from
  **Javier ([jvr0x](https://github.com/jvr0x))**'s language draft vocabularies for this model's vLLM recipe
  ([MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)),
  built from each language's Wikipedia by token frequency.
- `0002`-`0005`, `0007`-`0009`: by MiaAI-Lab.
- FP8 KV (`KV_DTYPE=fp8`) is adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` in
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).

## Recipe

- **[321sssrt-bit](https://github.com/321sssrt-bit)** authored the full-window admission fix and
  `tools/context_boundary.py` in
  [Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold)
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)): admission counted the draft
  window against `--context`, so a prompt plus `max_tokens` equal to the window was refused. The engine already keeps
  those rows beyond the window. The admission change in the engine patches and `tools/context_boundary.py` are theirs.
- **[sparkDash](https://github.com/MiaAI-Lab/sparkDash)** by MiaAI-Lab: the Decode and Prefill benchmarks the README's performance numbers were measured with.
- The scripts (`start.sh`, `stop.sh`, `scripts/`) and the checks in `tools/` are MiaAI-Lab's. `client.py`,
  `needle.py`, `toolcheck.py` and `prompt_reuse.py` are adapted from MiaAI-Lab's two-Spark recipe
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold);
  `exact.py` is from MiaAI-Lab's Flash Next two-Spark TensorFold recipe.

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the base of the image, with NVIDIA's CUDA (nvcc builds the engine's kernels),
  NCCL and related libraries. Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for
  NVIDIA AI Products.
- **[Triton](https://github.com/triton-lang/triton)** (MIT): the language the engine's Triton kernels are written in;
  the engine replays their compiled cubins.
- **[Hugging Face transformers](https://github.com/huggingface/transformers)** (Apache 2.0): the Qwen vision tower's
  modules and the image processor.
- **[PyAV](https://github.com/PyAV-Org/PyAV)** (BSD-3-Clause) and **[FFmpeg](https://ffmpeg.org/)** (LGPL): video
  decoding. **[Pillow](https://python-pillow.org/)** (MIT-CMU): image decoding.
- **[Hugging Face Hub](https://huggingface.co/)**: model hosting, the `hf` CLI and `huggingface_hub` (Apache 2.0), and
  the [safetensors](https://github.com/huggingface/safetensors) format (Apache 2.0) the checkpoint ships in.
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache 2.0): running the server
  on the GPU in a container.
- **[GitHub Container Registry](https://ghcr.io)**: hosting the prebuilt image, when one is published.

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell,
  128 GB unified memory): every number in the README was measured on one.

## README

- Badges by [Shields.io](https://shields.io/).

If you believe something here is missing or credited wrongly, please open an issue.
