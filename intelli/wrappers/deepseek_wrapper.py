import os
import json
import glob
import re

class DeepSeekTokenizer:
    def __init__(self, eos_token="<|endoftext|>"):
        self.eos_token = eos_token
        # Reserve token id 0 as EOS.
        self.eos_id = 0
        # For demonstration, we build a simple vocabulary.
        # In production, load a full vocabulary from file.
        self.vocab = {}

    def _get_token_id(self, token):
        # If token is already in vocab, return it;
        # otherwise, assign a new id (starting from 1, since 0 is reserved for EOS).
        if token not in self.vocab:
            # Use a simple hash-based scheme for demonstration.
            # In a real tokenizer, you would use a fixed vocabulary.
            self.vocab[token] = (hash(token) % 50000) + 1
        return self.vocab[token]

    def tokenize(self, text):
        """
        An improved tokenizer that lowercases the text and uses regex to split
        into words and punctuation.
        """
        text = text.lower()
        # This regex finds words and any punctuation as separate tokens.
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        token_ids = [self._get_token_id(token) for token in tokens]
        return token_ids

    def detokenize(self, token_ids):
        """
        For demonstration, convert token ids to a string by joining them.
        (In production, you should reverse the exact tokenization process.)
        Here we simply return a space‐joined string of the token ids.
        """
        # Since we don't store a reverse mapping here, we simply display the ids.
        # In production, you would map ids back to their corresponding tokens.
        return " ".join(f"<{tid}>" for tid in token_ids if tid != self.eos_id)


class DeepSeekWrapper:
    def __init__(self, model_path, config_path=None, temperature=0.2, max_new_tokens=200):
        """
        A simple offline wrapper for DeepSeek.
        :param model_path: Path to the converted DeepSeek weights 
                           (e.g., '/path/to/DeepSeek-V3-Demo')
        :param config_path: Path to DeepSeek config file (e.g., 'configs/config_671B.json').
        :param temperature: Temperature for token sampling.
        :param max_new_tokens: Maximum new tokens to generate.
        """
        # Import torch here and store as an instance variable.
        import torch
        self.torch = torch

        self.model_path = model_path
        self.config_path = config_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = DeepSeekTokenizer()
        self.eos_id = self.tokenizer.eos_id  # reserved EOS token id
        self._load_model()

    def _load_model(self):
        """
        Load or initialize the DeepSeek model in single-GPU mode
        using the code from model.py, generate.py, etc.
        """
        from intelli.model.deepseek.model import ModelArgs, Transformer

        # Use fallback if no config_path was provided.
        if not self.config_path:
            self.config_path = os.path.join(self.model_path, "configs", "config_671B.json")

        with open(self.config_path, "r") as f:
            hf_config = json.load(f)

        # Map HF config keys to those expected by ModelArgs.
        mapping = {
            "vocab_size": "vocab_size",               # same key
            "dim": "hidden_size",                     # HF: hidden_size → our: dim
            "inter_dim": "intermediate_size",         # HF: intermediate_size → our: inter_dim
            "moe_inter_dim": "moe_intermediate_size", # HF: moe_intermediate_size → our: moe_inter_dim
            "n_layers": "num_hidden_layers",          # HF: num_hidden_layers → our: n_layers
            "n_dense_layers": "n_dense_layers",       # if not present, add default later
            "n_heads": "num_attention_heads",         # HF: num_attention_heads → our: n_heads
            "n_routed_experts": "n_routed_experts",     # same key
            "n_shared_experts": "n_shared_experts",     # same key
            "n_activated_experts": "num_experts_per_tok",  # HF: num_experts_per_tok → our: n_activated_experts
            "route_scale": "routed_scaling_factor",     # HF: routed_scaling_factor → our: route_scale
            "q_lora_rank": "q_lora_rank",              # same key
            "kv_lora_rank": "kv_lora_rank",            # same key
            "qk_nope_head_dim": "qk_nope_head_dim",    # same key
            "qk_rope_head_dim": "qk_rope_head_dim",    # same key
            "v_head_dim": "v_head_dim",                # same key
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

        # Set default dtype.
        if mapped_config.get("dtype", "bf16") == "fp8":
            self.torch.set_default_dtype(self.torch.float32)
        else:
            self.torch.set_default_dtype(self.torch.bfloat16)
        self.torch.set_num_threads(8)

        # Initialize the model.
        self.args = ModelArgs(**mapped_config)
        self.model = Transformer(self.args).cuda()

        # Load all weight shards.
        from safetensors.torch import load_model
        shard_files = sorted(glob.glob(os.path.join(self.model_path, "model-000*-of-*.safetensors")))
        if not shard_files:
            raise ValueError("No weight shard files found.")
        for shard_file in shard_files:
            print(f"Loading weights from {shard_file}")
            load_model(self.model, shard_file)
        self.model.eval()

    def _sample(self, logits):
        """
        Sample one token id from logits using temperature-based sampling.
        :param logits: 1D tensor of logits.
        :return: A tensor containing the sampled token id.
        """
        if self.temperature <= 0.0:
            return self.torch.argmax(logits, dim=-1, keepdim=True)
        scaled_logits = logits / max(self.temperature, 1e-5)
        probs = self.torch.softmax(scaled_logits, dim=-1)
        return self.torch.multinomial(probs, num_samples=1)

    def generate(self, prompt):
        """
        Generate text autoregressively from a given prompt.
        :param prompt: The input prompt (string).
        :return: The generated text (string).
        """
        tokens = self.tokenizer.tokenize(prompt)
        tokens_tensor = self.torch.tensor([tokens], dtype=self.torch.long, device="cuda")
        with self.torch.no_grad():
            for _ in range(self.max_new_tokens):
                logits = self.model(tokens_tensor, start_pos=0)  # (batch_size, vocab_size)
                last_logits = logits[0]
                next_token = self._sample(last_logits)
                tokens_tensor = self.torch.cat([tokens_tensor, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.eos_id:
                    break
        generated_tokens = tokens_tensor[0].tolist()[len(tokens):]
        output_text = self.tokenizer.detokenize(generated_tokens)
        return output_text
