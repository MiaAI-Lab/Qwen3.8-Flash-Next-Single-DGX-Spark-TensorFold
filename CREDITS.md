# Credits

This repository is a thin layer of scripts and patches. Almost everything that makes it work was built by others.

## Model

- **[Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** by [Qwen](https://qwen.ai/): the model's
  design, training and evaluations. Released under the **Qwen Community License 1.0**, which governs any use of the
  weights (read it before commercial use, in particular its terms for Model-as-a-Service businesses). The weights are
  not part of this repository; `scripts/prepare.sh` downloads them from Hugging Face.
- **[Vontra](https://huggingface.co/Vontra)**: the checkpoint served here,
  [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP): the MLX
  4-bit conversion, the preserved native MTP draft head, validation and packaging.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the TensorFold contributors.
  v0.6.0 and later are Apache-2.0 (releases through v0.5.0 stay MIT). The engine serves the model, including the CUDA engine
  for Qwen3.8 Flash Next, MTP drafting with exact verification, the quantized KV cache, the OpenAI-compatible server
  and Flash Next image input on CUDA (TensorFold #146, adapted from this recipe's v0.6.0 vision patch), which
  `patches/0002-flash-next-v061.patch` extends with video and many images.
  Every file in `patches/` is a modification of TensorFold v0.6.1 (`17c73e1`).
- TensorFold itself builds on, and credits in its
  [third-party notices](https://github.com/ashhart/TensorFold/blob/v0.6.1/THIRD_PARTY_NOTICES.md):
  [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple, MIT),
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (Prince Canuma, MIT),
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT), whose cache quantization scheme the int8 and int4
  KV caches used here follow, and the Qwen Flash Next modeling code in
  Hugging Face [transformers](https://github.com/huggingface/transformers) (the Qwen Team and the Hugging Face team,
  Apache 2.0), from which its n-gram helpers are translated.

## Patches

- Former `0001-cuda-live-token-counters`, `0004-flash-next-qsa-tiled-select` and `0005-cuda-stream-draft-stats`
  are in TensorFold v0.5.0 (`cuda/health.py`, tiled QSA, stream draft accounting). The recipe's earlier
  typed-tool-parameters patch ([#75](https://github.com/ashhart/TensorFold/pull/75)) was already in v0.3.6.3.
- `0002-flash-next-v061.patch` folds the v0.5.0 series into one diff; image input itself is now TensorFold's own.
  The prefill rows override is a port of [TensorFold #40](https://github.com/ashhart/TensorFold/pull/40) by
  **[MovieMaker93](https://github.com/MovieMaker93)**. Copy drafts use TensorFold's `CopyIndex` from the Qwen3.5 27B
  engine. Vision builds on the Qwen3.5/3.8 dense tower, Hugging Face transformers, and Qwen3-VL video processing.
  The v0.5.0 series and the v0.6.0 and v0.6.1 rebases are by MiaAI-Lab, developed with
  [Claude Code](https://claude.com/claude-code) and Cursor.
- `languages/0010-flash-next-draft-languages`: the language token lists come from
  **Javier ([jvr0x](https://github.com/jvr0x))**'s language draft vocabularies for this model's vLLM recipe
  ([MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)),
  built from each language's Wikipedia by token frequency.

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the base of the image, with NVIDIA's CUDA, cuDNN, cuBLAS, NCCL and related
  libraries. Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for NVIDIA AI Products;
  see the README's License section.
- **[PyTorch](https://pytorch.org/)** (BSD-3-Clause): tensors, CUDA streams and the C++ extension builder that compiles
  the native n-gram reader in patch 0003.
- **[Triton](https://github.com/triton-lang/triton)** (MIT): the language most of TensorFold's Flash Next CUDA kernels,
  and TensorFold's tiled attention-block select, are written in.
- **[NumPy](https://numpy.org/)** (BSD-3-Clause): the host-side n-gram lookups and read planning, and video patches.
- **[Hugging Face transformers](https://github.com/huggingface/transformers)** (Apache 2.0): the Qwen vision tower's
  modules and the image processor.
- **[PyAV](https://github.com/PyAV-Org/PyAV)** (BSD-3-Clause) and **[FFmpeg](https://ffmpeg.org/)** (LGPL): video
  decoding. **[Pillow](https://python-pillow.org/)** (MIT-CMU): image decoding.
- **[Hugging Face Hub](https://huggingface.co/)**: model hosting, the `hf` CLI and `huggingface_hub` (Apache 2.0), and
  the [safetensors](https://github.com/huggingface/safetensors) format (Apache 2.0) the checkpoint ships in.
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache 2.0): running the server
  on the GPU in a container.
- **[GitHub Container Registry](https://ghcr.io)**: hosting the prebuilt image.

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell,
  128 GB unified memory): every number in the README was measured on one.

## README

- Badges by [Shields.io](https://shields.io/).

If you believe something here is missing or credited wrongly, please open an issue.
