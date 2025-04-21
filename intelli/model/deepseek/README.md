# DeepSeek Model Loader (Offline Inference)

This module provides a lightweight wrapper for loading and running DeepSeek models directly from HuggingFace without relying on `transformers`

## Features

- Offline model loading via `safetensors` index
- Supports both full precision and `float16` quantized models
- Configurable via native `config.json` from HuggingFace
- Memory-efficient execution using `torch`-level tools

## Requirements

```bash
pip install torch safetensors huggingface_hub
```

## Authentication

Some DeepSeek models are gated and require Hugging Face authentication. To access them, log in via the CLI:

```bash
huggingface-cli login
```

This will prompt you to enter your Hugging Face access token. You can create one at https://huggingface.co/settings/tokens.
