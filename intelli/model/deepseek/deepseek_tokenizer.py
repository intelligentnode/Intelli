import os
import json
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from huggingface_hub import hf_hub_download

class DeepSeekTokenizer:
    def __init__(self, model_path: Optional[str] = None, model_id: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        self.model_id = model_id

        # Handle path conversion
        if model_path:
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
            self.model_path = Path(model_path)
        else:
            self.model_path = None

        self.vocab = None
        self.eos_token_id = 2  # Default EOS token ID
        self.load_vocab()

    def load_vocab(self):
        if not self.model_path:
            cache_dir = Path.home() / ".cache" / "intelli" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)

            vocab_file = hf_hub_download(
                repo_id=self.model_id,
                filename="tokenizer.json",
                cache_dir=cache_dir
            )
            self.model_path = Path(vocab_file).parent

        # Find tokenizer file
        tokenizer_file = None
        for root, _, files in os.walk(self.model_path):
            for file in files:
                if 'tokenizer' in file and file.endswith('.json'):
                    tokenizer_file = os.path.join(root, file)
                    break
            if tokenizer_file:
                break

        if not tokenizer_file:
            raise FileNotFoundError(f"Tokenizer file not found in {self.model_path}")

        print(f"Loading tokenizer from: {tokenizer_file}")
        with open(tokenizer_file, 'r') as f:
            self.vocab = json.load(f)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)

            # Simple tokenization by splitting on whitespace
            # In a real implementation, this would use a more sophisticated tokenizer
            tokens = text.split()

            # Convert tokens to IDs using vocabulary
            # Default to unknown token ID (0) if token not in vocabulary
            return [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
        except Exception as e:
            print(f"Error in encoding: {str(e)}")
            # Return a single token ID as fallback
            return [0]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text
        """
        try:
            # Ensure token_ids is a list of integers
            if not isinstance(token_ids, list):
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                else:
                    token_ids = [0]  # Default to a single unknown token

            # Create reverse vocabulary mapping (ID -> token)
            rev_vocab = {v: k for k, v in self.vocab.items() if isinstance(v, int)}

            # Convert IDs to tokens and join with spaces
            return ' '.join(rev_vocab.get(id, '<unk>') for id in token_ids)
        except Exception as e:
            print(f"Error in decoding: {str(e)}")
            # Return a fallback message
            return "Error decoding tokens"