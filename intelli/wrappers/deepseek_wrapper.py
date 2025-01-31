import os
import json
import torch
from safetensors.torch import load_model

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
        Loads the DeepSeek model using local DeepSeek inference code.
        """
        # Use a default config path if none was provided.
        if not self.config_path:
            self.config_path = os.path.join(self.model_path, "configs", "config_671B.json")
        if not os.path.exists(self.config_path):
            raise ValueError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_data = json.load(f)

        # Import model definitions from your local deepseek module.
        from intelli.model.deepseek.model import ModelArgs, Transformer

        self.args = ModelArgs(**config_data)
        # Set default dtype according to the config.
        if self.args.dtype == "fp8":
            torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(8)

        # Initialize the model on GPU.
        self.model = Transformer(self.args).cuda()

        # Assume single shard (model0-mp1.safetensors) in the model directory.
        model_file = os.path.join(self.model_path, "model0-mp1.safetensors")
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
