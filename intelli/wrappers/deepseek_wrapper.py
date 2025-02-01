import os
import json
import torch

# A very basic tokenizer for demonstration.
# In production you should use a robust tokenizer.
class DeepSeekTokenizer:
    def __init__(self, eos_token="<|endoftext|>"):
        self.eos_token = eos_token
        # Reserve token id 0 as EOS.
        self.eos_id = 0

    def tokenize(self, text):
        """
        A basic whitespace tokenizer.
        Each word is converted to a token id by taking hash(word) mod 50000 plus 1.
        (Token id 0 is reserved for EOS.)
        """
        tokens = [hash(word) % 50000 + 1 for word in text.split()]
        return tokens

    def detokenize(self, token_ids):
        """
        For demonstration, convert token ids to a string by joining them.
        (In production this should be a reverse of the actual tokenization.)
        """
        # Skip the EOS token (id == 0)
        words = [f"<{tid}>" for tid in token_ids if tid != self.eos_id]
        return " ".join(words)


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
        import torch
        from intelli.model.deepseek.model import ModelArgs, Transformer
        import json

        if not self.config_path:
            # fallback if user doesn't pass config
            self.config_path = os.path.join(self.model_path, "configs", "config_671B.json")

        with open(self.config_path, "r") as f:
            hf_config = json.load(f)

        # Map HF config keys to the ones expected by ModelArgs.
        # (expects keys like "dim" but HF config uses "hidden_size".)
        mapping = {
            "vocab_size": "vocab_size",               # same key
            "dim": "hidden_size",                     # HF: hidden_size → our: dim
            "inter_dim": "intermediate_size",         # HF: intermediate_size → our: inter_dim
            "moe_inter_dim": "moe_intermediate_size", # HF: moe_intermediate_size → our: moe_inter_dim
            "n_layers": "num_hidden_layers",          # HF: num_hidden_layers → our: n_layers
            "n_dense_layers": "n_dense_layers",       # If not present, we’ll add a default below.
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
            # For mscale, we need to extract it from inside the "rope_scaling" dictionary.
            "mscale": lambda cfg: cfg.get("rope_scaling", {}).get("mscale", 1.0)
        }

        mapped_config = {}
        for our_key, hf_key in mapping.items():
            if isinstance(hf_key, str):
                if hf_key in hf_config:
                    mapped_config[our_key] = hf_config[hf_key]
            else:
                # if hf_key is a callable (for example, for mscale)
                mapped_config[our_key] = hf_key(hf_config)

        # If n_dense_layers is not in the config, add a default value (e.g., 1).
        if "n_dense_layers" not in mapped_config:
            mapped_config["n_dense_layers"] = 1

        # print or log the mapped configuration for debugging:
        # print("Mapped config for ModelArgs:", mapped_config)

        # Set default dtype based on our configuration.
        if mapped_config.get("dtype", "bf16") == "fp8":
            torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(8)

        # Initialize the model using the mapped configuration.
        self.args = ModelArgs(**mapped_config)
        self.model = Transformer(self.args).cuda()

        # For single GPU usage, we assume a single weight file.
        from safetensors.torch import load_model
        # Here we assume the weight file is "model-00001-of-000163.safetensors"
        model_file = os.path.join(self.model_path, "model-00001-of-000163.safetensors")
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found: {model_file}")
        load_model(self.model, model_file)
        self.model.eval()


    def _sample(self, logits):
        """
        Sample one token id from logits using temperature-based sampling.
        :param logits: 1D tensor of logits.
        :return: A tensor containing the sampled token id.
        """
        if self.temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        scaled_logits = logits / max(self.temperature, 1e-5)
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate(self, prompt):
        """
        Generate text autoregressively from a given prompt.
        :param prompt: The input prompt (string).
        :return: The generated text (string).
        """
        # Tokenize the prompt.
        tokens = self.tokenizer.tokenize(prompt)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device="cuda")
        # Set the maximum total length (prompt + new tokens).
        max_length = tokens_tensor.shape[1] + self.max_new_tokens

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                # Forward pass: assume the model returns logits for the last token.
                logits = self.model(tokens_tensor, start_pos=0)  # shape: (batch_size, vocab_size)
                # Use the logits for the last token in the batch.
                last_logits = logits[0]  # shape: (vocab_size,)
                next_token = self._sample(last_logits)
                tokens_tensor = torch.cat([tokens_tensor, next_token.unsqueeze(0)], dim=1)
                # If the EOS token is generated, stop.
                if next_token.item() == self.eos_id:
                    break

        # Exclude the prompt tokens from the generated part.
        generated_tokens = tokens_tensor[0].tolist()[len(tokens):]
        output_text = self.tokenizer.detokenize(generated_tokens)
        return output_text
