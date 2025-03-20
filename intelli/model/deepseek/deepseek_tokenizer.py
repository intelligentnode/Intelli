import json
from pathlib import Path
from typing import List, Optional
from huggingface_hub import hf_hub_download

class DeepSeekTokenizer:
    def __init__(self, model_path: Optional[str] = None, model_id: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        self.model_id = model_id
        self.model_path = Path(model_path) if model_path else None
        self.vocab = None
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
            
        with open(self.model_path / "tokenizer.json", 'r') as f:
            self.vocab = json.load(f)
            
    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
    def decode(self, token_ids: List[int]) -> str:
        rev_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join(rev_vocab.get(id, '<unk>') for id in token_ids) 