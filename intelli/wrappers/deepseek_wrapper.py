import os
from typing import Optional, List, Union
import torch

from ..model.deepseek import DeepSeekModelLoader, DeepSeekTokenizer

class IntelliDeepSeekWrapper:
    """Wrapper for DeepSeek model with memory optimizations."""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        quantize: bool = False,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Args:
            model_name_or_path: Either:
                - Name of the model (e.g. 'deepseek-ai/deepseek-v3-7b')
                - Path to local model directory containing config.json and model.safetensors
            device: Device to load model on ('cpu', 'cuda', 'mps')
            quantize: Whether to use quantization
            max_batch_size: Maximum batch size for inference
            max_seq_len: Maximum sequence length
            cache_dir: Directory to cache models (only used with model names)
            hf_token: HuggingFace access token (only used with model names)
        """
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        torch.set_default_dtype(torch.float32)
        
        if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
            model_path = model_name_or_path
            self.model = DeepSeekModelLoader(
                model_path=model_path,
                device=device,
                quantize=quantize,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len
            )
        else:
            self.model = DeepSeekModelLoader.from_pretrained(
                model_name=model_name_or_path,
                device=device,
                quantize=quantize,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                cache_dir=cache_dir,
                hf_token=hf_token
            )
            model_path = os.path.join(
                cache_dir or os.path.expanduser("~/.cache/intelli/models"),
                model_name_or_path.split('/')[-1]
            )
            
        self.tokenizer = DeepSeekTokenizer(model_path)
        
    def generate(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling parameter
            stop_sequences: List of sequences that stop generation
            
        Returns:
            Generated text
        """
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        
        output_ids = output_ids.cpu()
        
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in output_text:
                    output_text = output_text[:output_text.index(stop_seq)]
                    
        return output_text.strip()
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        **kwargs
    ) -> Union[str, List[str]]:
        """Convenience method for generation.
        
        Args:
            prompt: Input text prompt or list of prompts
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            Generated text or list of generated texts
        """
        if isinstance(prompt, list):
            return [self.generate(p, **kwargs) for p in prompt]
        return self.generate(prompt, **kwargs) 