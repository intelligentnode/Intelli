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
        self.vocab = {}

    def _get_token_id(self, token):
        if token not in self.vocab:
            self.vocab[token] = (hash(token) % 50000) + 1
        return self.vocab[token]

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        token_ids = [self._get_token_id(token) for token in tokens]
        return token_ids

    def detokenize(self, token_ids):
        return " ".join(f"<{tid}>" for tid in token_ids if tid != self.eos_id)

class DeepSeekWrapper:
    def __init__(self, model_path, config_path=None, temperature=0.2, max_new_tokens=200):
        import torch
        self.torch = torch

        self.model_path = model_path
        self.config_path = config_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = DeepSeekTokenizer()
        self.eos_id = self.tokenizer.eos_id

        self._maybe_convert_model()
        self._load_model()

    def _maybe_convert_model(self):
        """
        Check if converted weights (files matching "model*-mp*.safetensors") exist.
        If not—and if original HF checkpoint shards exist—run the conversion script.
        """
        converted_files = glob.glob(os.path.join(self.model_path, "model*-mp*.safetensors"))
        hf_shard_files = glob.glob(os.path.join(self.model_path, "model-000*-of-*.safetensors"))
        if not converted_files and hf_shard_files:
            print("Converted model weights not found. Running conversion...")
            converted_dir = os.path.join(self.model_path, "converted")
            os.makedirs(converted_dir, exist_ok=True)
            n_experts = 256
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    hf_config = json.load(f)
                n_experts = hf_config.get("n_routed_experts", 256)
            mp = 1  # For single-GPU inference.
            # Compute the conversion script path relative to the project layout.
            # Assuming conversion script is at <project_root>/model/deepseek/convert.py
            conversion_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "deepseek", "convert.py")
            cmd = [
                sys.executable,
                conversion_script,
                "--hf-ckpt-path", self.model_path,
                "--save-path", converted_dir,
                "--n-experts", str(n_experts),
                "--model-parallel", str(mp)
            ]
            print("Running conversion command:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            self.model_path = converted_dir
        elif not converted_files and not hf_shard_files:
            raise ValueError("No valid weight files found in the model path.")

    def _load_model(self):
        from intelli.model.deepseek.model import ModelArgs, Transformer

        # Clear any GPU cached memory before loading.
        self.torch.cuda.empty_cache()

        # Use the provided config file or fall back to a default.
        if not self.config_path:
            self.config_path = os.path.join(self.model_path, "configs", "config_671B.json")
        with open(self.config_path, "r") as f:
            hf_config = json.load(f)

        mapping = {
            "vocab_size": "vocab_size",
            "dim": "hidden_size",
            "inter_dim": "intermediate_size",
            "moe_inter_dim": "moe_intermediate_size",
            "n_layers": "num_hidden_layers",
            "n_dense_layers": "n_dense_layers",
            "n_heads": "num_attention_heads",
            "n_routed_experts": "n_routed_experts",
            "n_shared_experts": "n_shared_experts",
            "n_activated_experts": "num_experts_per_tok",
            "route_scale": "routed_scaling_factor",
            "q_lora_rank": "q_lora_rank",
            "kv_lora_rank": "kv_lora_rank",
            "qk_nope_head_dim": "qk_nope_head_dim",
            "qk_rope_head_dim": "qk_rope_head_dim",
            "v_head_dim": "v_head_dim",
            "mscale": lambda cfg: cfg.get("rope_scaling", {}).get("mscale", 1.0)
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

        if mapped_config.get("dtype", "bf16") == "fp8":
            self.torch.set_default_dtype(self.torch.float32)
        else:
            self.torch.set_default_dtype(self.torch.bfloat16)
        self.torch.set_num_threads(8)

        self.args = ModelArgs(**mapped_config)
        # Instantiate the model on CPU first.
        self.model = Transformer(self.args)
        # Move the model to GPU.
        self.model = self.model.cuda()

        shard_files = sorted(glob.glob(os.path.join(self.model_path, "model*-mp*.safetensors")))
        if not shard_files:
            raise ValueError("No converted weight shard files found in the model path.")
        for shard_file in shard_files:
            print(f"Loading weights from {shard_file}")
            from safetensors.torch import load_model
            load_model(self.model, shard_file)
        self.model.eval()

    def _sample(self, logits):
        if self.temperature <= 0.0:
            return self.torch.argmax(logits, dim=-1, keepdim=True)
        scaled_logits = logits / max(self.temperature, 1e-5)
        probs = self.torch.softmax(scaled_logits, dim=-1)
        return self.torch.multinomial(probs, num_samples=1)

    def generate(self, prompt):
        tokens = self.tokenizer.tokenize(prompt)
        tokens_tensor = self.torch.tensor([tokens], dtype=self.torch.long, device="cuda")
        with self.torch.no_grad():
            for _ in range(self.max_new_tokens):
                logits = self.model(tokens_tensor, start_pos=0)
                last_logits = logits[0]
                next_token = self._sample(last_logits)
                tokens_tensor = self.torch.cat([tokens_tensor, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.eos_id:
                    break
        generated_tokens = tokens_tensor[0].tolist()[len(tokens):]
        output_text = self.tokenizer.detokenize(generated_tokens)
        return output_text
