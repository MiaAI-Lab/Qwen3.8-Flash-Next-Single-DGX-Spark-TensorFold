# Client Setup & Integration Guide

This guide explains how to connect client applications and IDEs to the **Qwen3.8-Flash-Next** endpoint powered by TensorFold.

## Server Parameters Summary

| Parameter | Value |
|---|---|
| **API Format** | OpenAI-compatible (`/v1`) |
| **Base URL (Local)** | `http://localhost:8888/v1` or `http://127.0.0.1:8888/v1` |
| **Base URL (Docker Network)** | `http://qwen-tensorfold:8888/v1` (on `llm-network`) |
| **Base URL (LAN / Remote)** | `http://<spark-ip>:8888/v1` |
| **Model ID** | `Qwen3.8-Flash-Next` |
| **API Key** | `1234` (any non-empty string is accepted) |
| **Context Window** | `262144` tokens |
| **Max Output Tokens** | `32768` |
| **Capabilities** | Tool calling, Vision (Images/Videos), Thinking / Reasoning |

---

## 1. VS Code (Custom Language Models)

Add the following block to your VS Code user configuration file (`chatLanguageModels.json`):

- **Linux**: `~/.config/Code/User/chatLanguageModels.json`
- **macOS**: `~/Library/Application Support/Code/User/chatLanguageModels.json`
- **Windows**: `%APPDATA%\Code\User\chatLanguageModels.json`

```json
{
    "name": "TensorFold",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
        {
            "id": "Qwen3.8-Flash-Next",
            "name": "Qwen3.8-Flash-Next",
            "url": "http://localhost:8888/v1",
            "toolCalling": true,
            "vision": true,
            "contextWindow": 262144,
            "maxOutputTokens": 32768,
            "thinking": true,
            "reasoningEffortFormat": "chat-completions",
            "supportsReasoningEffort": [
                "none",
                "low",
                "medium",
                "xhigh"
            ]
        }
    ],
    "settings": {
        "Qwen3.8-Flash-Next": {
            "reasoningEffort": "xhigh"
        }
    }
}
```

---

## 2. Continue.dev (VS Code & JetBrains Extension)

Add this provider block to your `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Qwen3.8 Flash Next (TensorFold)",
      "provider": "openai",
      "model": "Qwen3.8-Flash-Next",
      "apiBase": "http://localhost:8888/v1",
      "apiKey": "1234",
      "contextLength": 262144,
      "completionOptions": {
        "maxTokens": 32768,
        "temperature": 1.0,
        "topP": 0.95
      }
    }
  ]
}
```

---

## 3. OnlyOffice (AI Plugin)

In OnlyOffice Document Server or Desktop Editors:
1. Open the **Plugins** tab and click **ChatGPT** / **AI Helper**.
2. Open plugin settings and enter:
   - **API URL**: `http://localhost:8888/v1` (or `http://<spark-ip>:8888/v1`)
   - **API Key**: `1234`
   - **Model**: `Qwen3.8-Flash-Next`
3. Save settings. (Full CORS preflight support is enabled for browser editors).

---

## 4. Open WebUI / LibreChat (Docker Network)

When running alongside other services on the `llm-network` Docker network:

```env
OPENAI_API_BASE_URL=http://qwen-tensorfold:8888/v1
OPENAI_API_KEY=1234
```

---

## 5. Python (Official OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8888/v1",
    api_key="1234",
)

response = client.chat.completions.create(
    model="Qwen3.8-Flash-Next",
    messages=[
        {"role": "user", "content": "Write a python function to compute fibonacci numbers."}
    ],
    max_tokens=1000,
)

print(response.choices[0].message.content)
```
