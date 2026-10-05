# Qwen3.8 Flash Next on DGX Spark (TensorFold Docker Compose)

This deployment containerizes [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) with full GPU passthrough, unified memory tuning, and integration into the shared Docker network (`llm-network`).

## Architecture & Features

- **Engine:** TensorFold v0.6.1 with GB10 sm_121 optimizations, copy drafts, and SSD read-ahead.
- **Model:** `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (~105 GiB, 4-bit group size 32 with MTP draft head).
- **Concurrency & Context:** Configured for single stream (`PARALLEL=1`) at full **262,144** token context window with int8 KV cache.
- **N-Gram Table:** Streamed directly from NVMe SSD (`PLE_ON_SSD=1`), saving ~30 GB of unified memory.
- **Network:** Connected to the external `llm-network` Docker network. Other containers reach the API at `http://qwen-tensorfold:8888/v1` or `http://qwen38-flash-next-tensorfold:8888/v1`.
- **Port:** Exposed on host port `8888` by default.

## Quick Start

### 1. Build and Start Container
```bash
docker compose build
docker compose up -d
```

### 2. View Logs
```bash
docker compose logs -f
```

### 3. Test Endpoint
```bash
curl http://127.0.0.1:8888/v1/models

curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in 20 words."}
    ],
    "max_tokens": 100
  }'
```

## Client Integrations

See [CLIENTS.md](./CLIENTS.md) for detailed configuration guides for:
- **VS Code**: `chatLanguageModels.json` custom model configuration
- **Continue.dev**: `~/.continue/config.json`
- **OnlyOffice**: AI Plugin setup with CORS support
- **Open WebUI & LibreChat**: Docker-to-Docker network setup
- **OpenAI Python / TypeScript SDK**: Standard client examples

## Hardware and Memory Note

On NVIDIA DGX Spark (128 GB unified memory / 121.7 GiB visible):
- TensorFold with `PLE_ON_SSD=1` requires ~85 GiB memory.
