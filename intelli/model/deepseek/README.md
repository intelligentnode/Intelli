# DeepSeek Model Implementation

This module provides a lightweight implementation for loading and running DeepSeek models directly from HuggingFace without relying on the `transformers` library.

## Features

- Efficient model loading with support for both local and remote models
- Byte-level BPE tokenization for accurate text encoding
- Support for split safetensors files for large models
- Memory optimization with quantization options
- Cross-platform compatibility (Windows, Linux, macOS)
- Robust error handling and fallback mechanisms

## Components

### DeepSeekLoader

The `DeepSeekLoader` class handles loading model weights and configurations:

- Automatically detects and downloads model files if needed
- Supports loading from split safetensors files
- Handles quantization for memory efficiency
- Provides fallback mechanisms when model loading fails

### DeepSeekTokenizer

The `DeepSeekTokenizer` class handles text tokenization:

- Implements Byte-level BPE tokenization
- Handles different tokenizer formats
- Provides robust error handling for encoding/decoding
- Generates meaningful fallback code when tokenization fails

### DeepSeekWrapper

The `DeepSeekWrapper` class provides a high-level interface:

- Follows Intelli's wrapper pattern for consistency
- Provides chat interface for easy integration
- Handles text generation with configurable parameters
- Implements robust error handling throughout the pipeline

## Usage

```python
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

# Initialize with a model ID
wrapper = DeepSeekWrapper(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Chat interface
response = wrapper.chat({"prompt": "Write a Python function to calculate fibonacci numbers."})
print(response["choices"][0]["text"])

# Direct tokenization and inference
tokens = wrapper.tokenize("Hello, world!")
logits = wrapper.infer(torch.tensor([tokens]))
```

## Requirements

- PyTorch
- safetensors
- huggingface_hub

## Authentication

Some DeepSeek models are gated and require Hugging Face authentication. To access them, set the `HUGGINGFACE_TOKEN` environment variable or use the `huggingface-cli login` command.
