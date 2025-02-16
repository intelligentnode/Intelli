import os
import json
import glob
import re
import subprocess
import sys


# Use BPE or SentencePiece-based tokenizer for production.
class DeepSeekTokenizer:
    def __init__(self, eos_token="<|endoftext|>"):
        self.eos_token = eos_token
        # We'll reserve token id 0 as the EOS token.
        self.eos_id = 0
        # Minimal "vocab" as a dictionary, purely for demonstration.
        self.vocab = {}

    def _get_token_id(self, token):
        # This is a naive approach. In production, load a real vocab file or use an actual tokenizer.
        if token not in self.vocab:
            self.vocab[token] = (hash(token) % 50000) + 1
        return self.vocab[token]

    def tokenize(self, text):
        # Lowercase and split by words/punctuation as a demonstration.
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        token_ids = [self._get_token_id(tok) for tok in tokens]
        return token_ids

    def detokenize(self, token_ids):
        # For demonstration, join as <id> except skip the eos token (0).
        return " ".join(f"<{tid}>" for tid in token_ids if tid != self.eos_id)


class DeepSeekWrapper:
    """
    Offline inference wrapper for DeepSeek (R1 or variants).
    """

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
        :param model_path: Path to the checkpoint directory.
            e.g. the directory containing 'model.safetensors' or multiple shards.
        :param config_path: Path to a config.json (or similar). If None, tries a default config.
        :param temperature: Sampling temperature. Set to <=0.0 for greedy.
        :param max_new_tokens: Max number of tokens to generate.
        :param model_parallel: Model parallel factor (if the model or code supports it).
        :param device: "cuda" or "cpu".
        :param enable_dp_attention: Some advanced 'data-parallel attention' modes if your code supports it.
        :param use_fp8: Attempt to use FP8. Fallback to BF16 if not available.
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

        # Prepare a minimal tokenizer placeholder.
        self.tokenizer = DeepSeekTokenizer()
        self.eos_id = self.tokenizer.eos_id

        # Attempt FP8 if requested, else BF16 (fallback if your GPU doesn't support it).
        # If your system doesn't have torch.float8_e4m3fn, the except block will fallback.
        if self.use_fp8:
            try:
                self.torch.set_default_dtype(self.torch.float8_e4m3fn)
                print("[DeepSeek] Using FP8 quantization (float8_e4m3fn).")
            except (AttributeError, TypeError):
                print("[DeepSeek] FP8 not available; falling back to BF16.")
                self.torch.set_default_dtype(self.torch.bfloat16)
        else:
            self.torch.set_default_dtype(self.torch.bfloat16)

        self.model = None
        self._maybe_convert_model()
        self._load_model()

    def _maybe_convert_model(self):
        """
        If needed, run the local `convert.py` script to create unified safetensor shards.
        This is typically done if you have multiple HF shards (model-00001-of-000xx.safetensors).
        If you already have a single 'model.safetensors' or 'model0-mp1.safetensors' file, no conversion is done.
        """
        # 1) If there's exactly one 'model.safetensors' file, assume no conversion needed.
        single_ckpt = glob.glob(os.path.join(self.model_path, "model.safetensors"))
        if single_ckpt:
            print("[DeepSeek] Single checkpoint file found; skipping conversion.")
            return

        # 2) If we have a "converted" directory with 'model*-mp*.safetensors', skip.
        converted_files = glob.glob(os.path.join(self.model_path, "model*-mp*.safetensors"))

        # 3) If we have multiple HF shards "model-000x-of-xxxx.safetensors" but no converted shard, do conversion.
        hf_shard_files = glob.glob(os.path.join(self.model_path, "model-000*-of-*.safetensors"))
        if not converted_files and hf_shard_files:
            print("[DeepSeek] Multiple HF shards found, no converted weights. Converting now...")
            converted_dir = os.path.join(self.model_path, "converted")
            os.makedirs(converted_dir, exist_ok=True)

            # Read n_experts from config if present:
            n_experts = 256
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config_data = json.load(f)
                # Some DeepSeek configs might have "n_routed_experts" or similar.
                n_experts = config_data.get("n_routed_experts", 256)

            mp = self.model_parallel
            # prepare convert path
            conversion_script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "model",
                "deepseek",
                "convert.py",
            )
            if not os.path.isfile(conversion_script):
                raise FileNotFoundError(f"Conversion script not found at: {conversion_script}")

            cmd = [
                sys.executable,
                conversion_script,
                "--hf-ckpt-path", self.model_path,
                "--save-path", converted_dir,
                "--n-experts", str(n_experts),
                "--model-parallel", str(mp),
            ]
            print("[DeepSeek] Running conversion command:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            self.model_path = converted_dir
        elif not converted_files and not hf_shard_files:
            # If there's no single file, no "converted" shards, no HF shards, error out.
            raise ValueError("[DeepSeek] No valid checkpoint files found.")

    def _load_model(self):
        """
        Load the final local DeepSeek model from safetensors shards.
        The model code is in `intelli.model.deepseek`.
        """
        print("[DeepSeek] Loading model config...")

        # If user didn't specify a config, try a default:
        if not self.config_path:
            default_cfg = os.path.join(self.model_path, "config.json")
            if os.path.isfile(default_cfg):
                self.config_path = default_cfg
            else:
                # Or fallback to a known config name. Adjust as needed (e.g. config_671B.json).
                fallback_cfg = os.path.join(self.model_path, "configs", "config_671B.json")
                if os.path.isfile(fallback_cfg):
                    self.config_path = fallback_cfg

        if not self.config_path or not os.path.exists(self.config_path):
            raise FileNotFoundError(f"[DeepSeek] Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_data = json.load(f)

        # import local code.
        from intelli.model.deepseek.model import ModelArgs, Transformer

        # Create ModelArgs from config_data.
        self.args = ModelArgs(**config_data)

        # Optionally, if the user wants a special property for dp_attention:
        if self.enable_dp_attention:
            print("[DeepSeek] Data-parallel attention is requested. (Custom usage)")

        # Clear CUDA cache if device is "cuda":
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

        # Instantiate the model on CPU first or directly on GPU:
        self.model = Transformer(self.args).to(self.device)
        self.model.eval()

        # Gather safetensor shards to load.
        # 1) Single-file case "model.safetensors"
        single_ckpt = glob.glob(os.path.join(self.model_path, "model.safetensors"))
        if single_ckpt:
            shard_files = single_ckpt
        else:
            # 2) Already-converted shards e.g. "model0-mp1.safetensors", "model1-mp1.safetensors", etc.
            shard_files = sorted(glob.glob(os.path.join(self.model_path, "model*-mp*.safetensors")))
            if not shard_files:
                raise ValueError("[DeepSeek] No final shards (model*-mp*.safetensors) found.")

        # Load weights:
        from safetensors.torch import load_model
        for sf in shard_files:
            print(f"[DeepSeek] Loading weight shard: {sf}")
            load_model(self.model, sf)

        # Attempt torch.compile for optimization:
        try:
            self.model = self.torch.compile(self.model)
            print("[DeepSeek] Model compiled with torch.compile()")
        except Exception as e:
            print("[DeepSeek] torch.compile() not used or failed. Reason:", e)

        print("[DeepSeek] Model loaded successfully.")

    def _sample(self, logits):
        """
        Sample next token index from the final logits, factoring temperature.
        """
        if self.temperature <= 0.0:  # Greedy
            return self.torch.argmax(logits, dim=-1, keepdim=True)
        scaled_logits = logits / max(self.temperature, 1e-9)
        probs = self.torch.softmax(scaled_logits, dim=-1)
        return self.torch.multinomial(probs, num_samples=1)

    def generate(self, prompt):
        """
        Minimal auto-regressive generation loop.
        If your local model uses: model(tokens_tensor, start_pos=some_index),
        adapt accordingly.  The snippet below is naive, calling forward
        for each step. For large models, you want a kv-cache approach.
        """
        tokens = self.tokenizer.tokenize(prompt)
        tokens_tensor = self.torch.tensor([tokens], dtype=self.torch.long, device=self.device)

        new_tokens = []
        with self.torch.no_grad():
            for _ in range(self.max_new_tokens):
                # pass the entire seq. Not memory efficient for large context.
                logits = self.model(tokens_tensor, start_pos=0)
                # logits shape might be: (batch_size=1, seq_len, vocab_size)
                # If so, we want the last token's logits.
                last_token_logits = logits[0, -1, :]

                next_token_id = self._sample(last_token_logits.unsqueeze(0)).squeeze(0)
                # next_token_id shape: (1,)
                token_int = next_token_id.item()
                tokens_tensor = self.torch.cat([tokens_tensor, next_token_id.view(1,1)], dim=1)

                if token_int == self.eos_id:
                    break
                new_tokens.append(token_int)

        # Convert the newly generated tokens to text:
        output_text = self.tokenizer.detokenize(new_tokens)
        return output_text
