import os

class DeepSeekWrapper:
    def __init__(self, model_path, config_path=None, temperature=0.2, max_new_tokens=200):
        """
        A simple offline wrapper for DeepSeek. 
        :param model_path: Path to the converted DeepSeek weights 
                           (e.g., '/path/to/DeepSeek-V3-Demo')
        :param config_path: Path to DeepSeek config file (e.g., 'configs/config_671B.json').
        :param temperature: Temperature for token sampling.
        :param max_new_tokens: Max new tokens to generate in one pass.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = None

        # Lazy import so users who don't install [offline] won't break
        try:
            import torch
            import safetensors  # needed if the user uses safetensors
            # from transformers import ...  # if needed, but you want to avoid it
        except ImportError as e:
            raise ImportError(
                "DeepSeek offline requires optional packages not installed. "
                "Please install via: pip install intelli[offline] "
                f"Missing: {e.name}"
            ) from e

        self._load_model()

    def _load_model(self):
        """
        Load or initialize the DeepSeek model in single-GPU mode
        using the code from model.py, generate.py, etc.
        """
        import torch
        from intelli.model.deepseek.model import ModelArgs, Transformer
        from intelli.model.deepseek.generate import main as generate_main
        # Optionally read config JSON (e.g., config_671B.json)
        import json

        if not self.config_path:
            # fallback if user doesn't pass config
            self.config_path = os.path.join(self.model_path, "config_671B.json")

        with open(self.config_path, "r") as f:
            config_data = json.load(f)

        self.args = ModelArgs(**config_data)
        # Single GPU usage, not distributed:
        torch.set_default_dtype(torch.bfloat16 if self.args.dtype != "fp8" else torch.float32)
        torch.set_num_threads(8)

        # Initialize model
        self.model = Transformer(self.args).cuda()
        
        # "load_model" from generate.py or safetensors:
        from safetensors.torch import load_model
        # E.g. load model0-mp1.safetensors if you only have 1 shard
        # Adjust if you have multiple shards
        model_file = os.path.join(self.model_path, "model0-mp1.safetensors")
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found: {model_file}")
        load_model(self.model, model_file)

        self.model.eval()

    def generate(self, prompt):
        """
        Minimal generate function using the model's forward pass.
        For advanced usage, see 'generate.py' in DeepSeek repo.
        """
        import torch
        # We assume 'prompt' is a string. Tokenize if needed or just do an integer mock:
        # If you have your own tokenizer from deepseek: 
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained(...)
        # But you mentioned you don't want HF Transformers downloads, 
        # so here's a placeholder:
        tokens = self._basic_tokenize(prompt)

        # Single step or multi-step generation
        # Similar to the logic in generate.py
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device="cuda")
        max_len = len(tokens) + self.max_new_tokens

        # We'll do an auto-regressive loop for demonstration:
        for _ in range(self.max_new_tokens):
            logits = self.model(tokens_tensor)
            # logits shape: (batch_size, vocab_size)
            # We pick the last token's distribution
            next_token_logits = logits[0, :]
            # Temperature-based sample:
            next_token_id = self._sample(next_token_logits, temperature=self.temperature)
            # If next_token_id is some EOS, break
            tokens_tensor = torch.cat([tokens_tensor, next_token_id.view(1,1)], dim=1)
        # Convert back to string
        generated_tokens = tokens_tensor[0].tolist()
        output_text = self._basic_detokenize(generated_tokens)
        return output_text

    def _sample(self, logits, temperature=1.0):
        import torch
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        # softmax with temperature
        probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _basic_tokenize(self, text):
        """
        You can implement your own tokenization logic or use a 
        local custom BPE from DeepSeek if it doesn't rely on HF.
        For now, we treat each whitespace as a token for DEMO ONLY.
        """
        return [hash(t) % 50000 for t in text.split()]  # obviously not correct, just a dummy

    def _basic_detokenize(self, token_ids):
        """
        The reverse of _basic_tokenize. 
        Dummy approach for demonstration. 
        """
        return " ".join(f"<{tid}>" for tid in token_ids)
