import os
import json
from pathlib import Path
from typing import List, Optional
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
        tokens = text.split()
        return [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        rev_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join(rev_vocab.get(id, '<unk>') for id in token_ids)