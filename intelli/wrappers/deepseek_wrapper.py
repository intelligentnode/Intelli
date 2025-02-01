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
            # A simple hash-based id (in production, load a fixed vocabulary)
            self.vocab[token] = (hash(token) % 50000) + 1
        return self.vocab[token]

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        token_ids = [self._get_token_id(token) for token in tokens]
        return token_ids

    def detokenize(self, token_ids):
        # For demonstration, join token ids as strings.
        return " ".join(f"<{tid}>" for tid in token_ids if tid != self.eos_id)


class DeepSeekWrapper:
    def __init__(
        self,
        model_path,
        config_path=None,
        temperature=0.2,
        max_new_tokens=200,
        model_parallel=1,
        device="cuda",
        enable_dp_attention=False,
        use_fp8=False,
    ):
        """
        Offline wrapper for DeepSeek.

        :param model_path: Path to the model checkpoint (HF weights or converted).
        :param config_path: Path to the DeepSeek config file.
        :param temperature: Sampling temperature.
        :param max_new_tokens: Maximum tokens to generate.
        :param model_parallel: Number of GPUs over which to split the model.
                               (For 8B models, a value >1 is recommended on multi-GPU systems.)
        :param device: Device to load the model onto ("cuda" or "cpu").
        :param enable_dp_attention: If True, enable data-parallel attention mode.
        :param use_fp8: If True, try to use FP8 quantization for inference.
        """
        import torch

        self.torch = torch

        self.model_path = model_path
        self.config_path = config_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model_parallel = model_parallel
        self.device = device
        self.enable_dp_attention = enable_dp_attention
        self.use_fp8 = use_fp8

        self.model = None
        self.tokenizer = DeepSeekTokenizer()
        self.eos_id = self.tokenizer.eos_id

        # Optionally set FP8 as default dtype (if supported)
        if self.use_fp8:
            try:
                self.torch.set_default_dtype(self.torch.float8_e4m3fn)
                print("Using FP8 quantization.")
            except (AttributeError, TypeError):
                print("FP8 dtype not available; falling back to BF16.")
                self.torch.set_default_dtype(self.torch.bfloat16)
        else:
            self.torch.set_default_dtype(self.torch.bfloat16)

        self._maybe_convert_model()
        self._load_model()

    def _maybe_convert_model(self):
        """
        Check if converted weights (files matching "model*-mp*.safetensors") exist.
        If not—and if original HF checkpoint shards exist—run the conversion script.
        """
        converted_files = glob.glob(
            os.path.join(self.model_path, "model*-mp*.safetensors")
        )
        hf_shard_files = glob.glob(
            os.path.join(self.model_path, "model-000*-of-*.safetensors")
        )
        if not converted_files and hf_shard_files:
            print("Converted model weights not found. Running conversion...")
            converted_dir = os.path.join(self.model_path, "converted")
            os.makedirs(converted_dir, exist_ok=True)
            n_experts = 256
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    hf_config = json.load(f)
                n_experts = hf_config.get("n_routed_experts", 256)
            mp = self.model_parallel  # use the provided model parallelism factor

            # Compute the conversion script path relative to the project layout.
            # Here we assume the conversion script is located at:
            # <project_root>/model/deepseek/convert.py
            conversion_script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "model",
                "deepseek",
                "convert.py",
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
            # Update model_path to the converted directory.
            self.model_path = converted_dir
        elif not converted_files and not hf_shard_files:
            raise ValueError("No valid weight files found in the model path.")

    def _load_model(self):
        """
        Load and initialize the DeepSeek model using the converted weights.
        """
        from intelli.model.deepseek.model import ModelArgs, Transformer

        # Clear any cached GPU memory.
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

        # Use provided config or fall back to default.
        if not self.config_path:
            self.config_path = os.path.join(
                self.model_path, "configs", "config_671B.json"
            )
        with open(self.config_path, "r") as f:
            hf_config = json.load(f)

        mapping = {
            "vocab_size": "vocab_size",
            "dim": "hidden_size",  # HF: hidden_size → our: dim
            "inter_dim": "intermediate_size",  # HF: intermediate_size → our: inter_dim
            "moe_inter_dim": "moe_intermediate_size",  # HF: moe_intermediate_size → our: moe_inter_dim
            "n_layers": "num_hidden_layers",  # HF: num_hidden_layers → our: n_layers
            "n_dense_layers": "n_dense_layers",  # Default to 1 if not present
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

        # Instantiate model arguments.
        self.args = ModelArgs(**mapped_config)

        # Instantiate the model on CPU first.
        self.model = Transformer(self.args)

        # If DP attention is enabled, set a flag on the model (your attention modules must check this flag).
        if self.enable_dp_attention:
            print("Enabling data-parallel attention mode.")
            self.model.enable_dp_attention = True

        # Move model to the chosen device.
        self.model = self.model.to(self.device)

        # Optionally compile the model with torch.compile (requires PyTorch 2.0+)
        try:
            self.model = self.torch.compile(self.model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print("torch.compile() failed or not supported; continuing without it.", e)

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
        if self.temperature <= 0.0:
            return self.torch.argmax(logits, dim=-1, keepdim=True)
        scaled_logits = logits / max(self.temperature, 1e-5)
        probs = self.torch.softmax(scaled_logits, dim=-1)
        return self.torch.multinomial(probs, num_samples=1)

    def generate(self, prompt):
        tokens = self.tokenizer.tokenize(prompt)
        tokens_tensor = self.torch.tensor(
            [tokens], dtype=self.torch.long, device=self.device
        )
        with self.torch.no_grad():
            for _ in range(self.max_new_tokens):
                logits = self.model(tokens_tensor, start_pos=0)
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
