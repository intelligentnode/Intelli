from typing import List, Optional
from deepseek_tokenizer import Tokenizer

class DeepSeekTokenizer:
    """Lightweight wrapper for DeepSeek tokenizer."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the DeepSeek tokenizer.
        
        Args:
            model_path: Optional path to tokenizer model directory (not needed for default tokenizer)
        """
        self.tokenizer = Tokenizer        
        self.bos_token_id = 151646  # Match model config
        self.eos_token_id = 151643  # Match model config
        self.pad_token_id = 151643  # Match model config
        
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token ids.
        
        Args:
            text: Input text to encode
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token ids
        """
        if not text:
            return []
            
        tokens = self.tokenizer.encode(text)
        
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
            
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text.
        
        Args:
            token_ids: List of token ids to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not token_ids:
            return ""
            
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in {
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id
            }]
            
        return self.tokenizer.decode(token_ids)
    
    def num_tokens(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.vocab_size 