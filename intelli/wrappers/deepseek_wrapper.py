# deepseek_wrapper.py
import os
import json
import glob
import re
import subprocess
import sys


class DeepSeekTokenizer:
    def __init__(self, eos_token="<|endoftext|>"):
        self.eos_token = eos_token
        # Reserve token id 0 as EOS.
        self.eos_id = 0
        # For demonstration, we build a simple vocabulary.
        self.vocab = {}

    def _get_token_id(self, token):
        if token not in self.vocab:
            # A simple hash-based id (in production load a fixed vocabulary).
            self.vocab[token] = (hash(token) % 50000) + 1
        return self.vocab[token]

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        token_ids = [self._get_token_id(token) for token in tokens]
        return token_ids

    def detokenize(self, token_ids):
        # For demonstration, simply join token ids (in production, map back to tokens)
        return " ".join(f"<{tid}>" for tid in token_ids if tid != self.eos_id)


class DeepSeekWrapper:
    def __init__(
        self, model_path, config_path=None, temperature=0.2, max_new_tokens=200
    ):
        """
        Offline wrapper for DeepSeek.
        :param model_path: Path to the model checkpoint (HF weights or converted).
        :param config_path: Path to the DeepSeek config file.
        :param temperature: Sampling temperature.
        :param max_new_tokens: Maximum tokens to generate.
        """
        import torch

        self.torch = torch

        self.model_path = model_path
        self.config_path = config_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = DeepSeekTokenizer()
        self.eos_id = self.tokenizer.eos_id

        # Check if conversion is needed (if the weights are still in HF checkpoint format)
        self._maybe_convert_model()
        self._load_model()

    def _maybe_convert_model(self):
        """
        Check if converted weights (files matching "model*-mp*.safetensors")
        exist. If not, assume that the model_path contains original HF checkpoint shards
        and run the conversion script.
        """
        converted_files = glob.glob(
            os.path.join(self.model_path, "model*-mp*.safetensors")
        )
        hf_shard_files = glob.glob(
            os.path.join(self.model_path, "model-000*-of-*.safetensors")
        )
        if not converted_files and hf_shard_files:
            print("Converted model weights not found. Running conversion...")
            # Create a subfolder for the converted weights
            converted_dir = os.path.join(self.model_path, "converted")
            os.makedirs(converted_dir, exist_ok=True)
            # Determine n_experts from the config if available; otherwise use default 256.
            n_experts = 256
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    hf_config = json.load(f)
                n_experts = hf_config.get("n_routed_experts", 256)
            mp = 1  # For single-GPU inference.
            # Assume the conversion script is located at: <model_root>/deepseek/convert.py
            conversion_script = os.path.join(
                os.path.dirname(__file__), "deepseek", "convert.py"
            )
            cmd = [
                sys.executable,
                conversion_script,
                "--hf-ckpt-path",
                self.model_path,
                "--save-path",
                converted_dir,
                "--n-experts",
                str(n_experts),
                "--model-parallel",
                str(mp),
            ]
            print("Running conversion command:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            # Update the model_path to point to the converted directory.
            self.model_path = converted_dir
        elif not converted_files and not hf_shard_files:
            raise ValueError("No valid weight files found in the model path.")

    def _load_model(self):
        """
        Load and initialize the DeepSeek model using the converted weights.
        """
        # Import your model definitions from your local package.
        from intelli.model.deepseek.model import ModelArgs, Transformer

        # Use fallback config if none provided.
        if not self.config_path:
            self.config_path = os.path.join(
                self.model_path, "configs", "config_671B.json"
            )
        with open(self.config_path, "r") as f:
            hf_config = json.load(f)

        # Map HF config keys to those expected by ModelArgs.
        mapping = {
            "vocab_size": "vocab_size",
            "dim": "hidden_size",  # HF: hidden_size → our: dim
            "inter_dim": "intermediate_size",  # HF: intermediate_size → our: inter_dim
            "moe_inter_dim": "moe_intermediate_size",  # HF: moe_intermediate_size → our: moe_inter_dim
            "n_layers": "num_hidden_layers",  # HF: num_hidden_layers → our: n_layers
            "n_dense_layers": "n_dense_layers",  # if not present, we will default to 1
            "n_heads": "num_attention_heads",  # HF: num_attention_heads → our: n_heads
            "n_routed_experts": "n_routed_experts",
            "n_shared_experts": "n_shared_experts",
            "n_activated_experts": "num_experts_per_tok",  # HF: num_experts_per_tok → our: n_activated_experts
            "route_scale": "routed_scaling_factor",  # HF: routed_scaling_factor → our: route_scale
            "q_lora_rank": "q_lora_rank",
            "kv_lora_rank": "kv_lora_rank",
            "qk_nope_head_dim": "qk_nope_head_dim",
            "qk_rope_head_dim": "qk_rope_head_dim",
            "v_head_dim": "v_head_dim",
            "mscale": lambda cfg: cfg.get("rope_scaling", {}).get("mscale", 1.0),
        }

        mapped_config = {}
        for our_key, hf_key in mapping.items():
            if isinstance(hf_key, str):
                if hf_key in hf_config:
                    mapped_config[our_key] = hf_config[hf_key]
            else:
                mapped_config[our_key] = hf_key(hf_config)
        if "n_dense_layers" not in mapped_config:
            mapped_config["n_dense_layers"] = 1

        # Set default dtype based on config (you can improve this by checking "torch_dtype" or similar).
        if mapped_config.get("dtype", "bf16") == "fp8":
            self.torch.set_default_dtype(self.torch.float32)
        else:
            self.torch.set_default_dtype(self.torch.bfloat16)
        self.torch.set_num_threads(8)

        self.args = ModelArgs(**mapped_config)
        self.model = Transformer(self.args).cuda()

        # Load converted weight shards.
        shard_files = sorted(
            glob.glob(os.path.join(self.model_path, "model*-mp*.safetensors"))
        )
        if not shard_files:
            raise ValueError("No converted weight shard files found in the model path.")
        for shard_file in shard_files:
            print(f"Loading weights from {shard_file}")
            from safetensors.torch import load_model

            load_model(self.model, shard_file)
        self.model.eval()

    def _sample(self, logits):
        """
        Sample one token id from logits using temperature-based sampling.
        """
        if self.temperature <= 0.0:
            return self.torch.argmax(logits, dim=-1, keepdim=True)
        scaled_logits = logits / max(self.temperature, 1e-5)
        probs = self.torch.softmax(scaled_logits, dim=-1)
        return self.torch.multinomial(probs, num_samples=1)

    def generate(self, prompt):
        """
        Generate text autoregressively from a given prompt.
        """
        tokens = self.tokenizer.tokenize(prompt)
        tokens_tensor = self.torch.tensor(
            [tokens], dtype=self.torch.long, device="cuda"
        )
        with self.torch.no_grad():
            for _ in range(self.max_new_tokens):
                # The model expects the full sequence (note: for efficiency you might want to cache previous activations)
                logits = self.model(
                    tokens_tensor, start_pos=0
                )  # (batch_size, vocab_size)
                last_logits = logits[0]
                next_token = self._sample(last_logits)
                tokens_tensor = self.torch.cat(
                    [tokens_tensor, next_token.unsqueeze(0)], dim=1
                )
                if next_token.item() == self.eos_id:
                    break
        generated_tokens = tokens_tensor[0].tolist()[len(tokens) :]
        output_text = self.tokenizer.detokenize(generated_tokens)
        return output_text
