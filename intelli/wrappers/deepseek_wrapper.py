import os
import json
import torch
from torch import nn
from typing import List
from intelli.model.deepseek.helpers import (
    load_model_weights,
    download_model,
    get_device,
    download_config,
    download_model_index,
    load_bpe_tokenizer,
    bpe_tokenize,
)
from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError
from huggingface_hub import HfApi, hf_hub_download


class DeepSeekWrapper:
    def __init__(
        self,
        repo_id: str,
        config_path: str = None,
        quantized: bool = False,
    ):
        """
        Initializes the DeepSeek model wrapper

        :param model_path: Path to the safetensors model weights
        :param config_path: Path to the model configuration JSON
        :param quantized: Whether to load the model in quantized format for memory efficiency
        """
        self.repo_id = repo_id
        self.device = get_device()
        self.quantized = quantized

        self.config_path = config_path or download_config(repo_id)
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.vocab, self.merges = load_bpe_tokenizer(repo_id)

        self.model = self._build_model()
        self.model.to(self.device, memory_format=torch.channels_last)

        api = HfApi()
        try:
            files = api.list_repo_files(repo_id)
        except HfHubHTTPError as e:
            raise RuntimeError(f"Could not list files in {repo_id}") from e

        load_model_weights(self.model, repo_id, files)

    def tokenize(self, text):
        """
        Byte-level BPE tokenization (UTF-8 → subword IDs)
        using the exact merges the model was trained with
        """
        return bpe_tokenize(text, self.vocab, self.merges)

    def decode(self, token_ids: List[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        toks = []
        print(f"Decoding tokens: {token_ids}")

        for i in token_ids:
            t = inv_vocab.get(i, "")
            if not t or t == "<unk>":
                continue
            toks.append(t)

        print(f"Raw tokens: {toks}")

        result = ""
        for t in toks:
            if t == "Ġ":
                result += " "
            else:
                if t.startswith("Ġ"):
                    result += " " + t[1:]
                else:
                    if t == "Ä":
                        result += " "
                    else:
                        result += t
        return result.strip()

    def _build_model(self):
        """Constructs a transformer-based model based on the config"""
        hidden_size = self.config.get("hidden_size", 4096)
        intermediate_size = self.config.get("intermediate_size", 16384)
        num_heads = self.config.get("num_attention_heads", 32)
        num_layers = self.config.get("num_hidden_layers", 32)
        vocab_size = self.config.get("vocab_size", 50257)

        class TransformerBlock(nn.Module):
            def __init__(
                self, hidden_size, num_heads, intermediate_size, quantized=False
            ):
                super().__init__()
                self.quantized = quantized
                self.attention = nn.MultiheadAttention(
                    hidden_size, num_heads, batch_first=True
                )
                self.ffn = nn.Sequential(
                    nn.Linear(
                        hidden_size,
                        intermediate_size,
                        dtype=torch.float16 if self.quantized else torch.float32,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        intermediate_size,
                        hidden_size,
                        dtype=torch.float16 if self.quantized else torch.float32,
                    ),
                )
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)

            def forward(self, x):
                attn_output, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_output)
                ffn_output = self.ffn(x)
                return self.norm2(x + ffn_output)

        class DeepSeekModel(nn.Module):
            def __init__(self, quantized=False):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList(
                    [
                        TransformerBlock(
                            hidden_size, num_heads, intermediate_size, quantized
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.output_layer = nn.Linear(hidden_size, vocab_size)

            def forward(self, x):
                x = self.embedding(x)
                for layer in self.layers:
                    x = layer(x)
                return self.output_layer(x)

        model = DeepSeekModel(quantized=self.quantized)
        if self.quantized:
            model = model.to(dtype=torch.float16)
        return model

    def infer(self, input_tensor):
        """
        Performs inference on the given input tensor.
        :param input_tensor: Input tensor for inference.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(input_tensor.to(self.device))
