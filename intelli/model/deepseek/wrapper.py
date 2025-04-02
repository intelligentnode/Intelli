import os
import torch
import safetensors
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from typing import Optional, Dict, List
import json
from llama_cpp import Llama
from intelli.model.deepseek.helpers.quantize import is_quantized_model_available

class DeepSeekWrapper:
    """
    A wrapper class for running inference on DeepSeek GGUF models.
    
    Example usage:
    ```python
    model = DeepSeekWrapper(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        filename="model.q4_gguf",
        model_repo_id="bartowski/quantized-models"
    )
    response = model.generate({"prompt": "2+2=", "temperature": 0.1})
    ```
    """
    def __init__(
        self,
        repo_id: str,  # Main repo for config
        filename: str,
        quantized: bool = False,
        n_gpu_layers: int = 0,
        flash_attention: bool = True,
        cache_dir: str = "models",
        model_repo_id: Optional[str] = None  # Optional separate model repo
    ):
        """
        Load DeepSeek models from HuggingFace Hub with memory optimizations
        
        Args:
            repo_id: HuggingFace repository ID (e.g. "deepseek-ai/deepseek-v2")
            filename: Model filename (e.g. "model.safetensors")
            quantized: Load 4/8-bit quantized version if available
            n_gpu_layers: Number of layers to offload to GPU (0=CPU-only)
            flash_attention: Use Flash Attention optimization
            cache_dir: Local directory to cache models
            model_repo_id: Optional separate model repository ID
        """
        self.repo_id = repo_id
        self.model_repo_id = model_repo_id or repo_id  # Use main repo if not specified
        self.filename = filename
        self.quantized = quantized
        self.n_gpu_layers = n_gpu_layers
        self.cache_dir = cache_dir
        
        self._load_config()
        self._init_mmap() 

        if self.quantized and not self.filename.endswith(".gguf"):
            raise ValueError("GGUF model required for quantized inference")

    def _load_config(self):
        """Load model configuration from original model repo"""
        try:
            config_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="config.json",
                cache_dir=self.cache_dir
            )
        except EntryNotFoundError:
            config_path = hf_hub_download(
                repo_id=self.model_repo_id,
                filename="config.json",
                cache_dir=self.cache_dir
            )
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        if "num_hidden_layers" not in self.config:
            raise ValueError("Invalid DeepSeek model config")
        if "hidden_size" not in self.config:
            raise ValueError("Missing hidden_size in config")
        if "num_attention_heads" not in self.config:
            raise ValueError("Missing num_attention_heads in config")

    def _init_mmap(self):
        """Initialize memory mapping for GGUF file"""
        if not self.filename.endswith(".gguf"):
            raise ValueError("GGUF model required for this wrapper")

        self.llm = Llama(
            model_path=hf_hub_download(
                repo_id=self.model_repo_id,
                filename=self.filename,
                cache_dir=self.cache_dir
            ),
            n_gpu_layers=self.n_gpu_layers,
            verbose=False
        )

    def generate(self, inputs: Dict, max_length: int = 128) -> str:
        if not inputs.get("prompt"):
            raise ValueError("Prompt cannot be empty")
        
        if not isinstance(inputs.get("max_tokens", 0), int):
            raise TypeError("max_tokens must be an integer")
        
        """Generate text using GGUF model with proper formatting"""
        formatted_prompt = (
            f"<|begin_of_sentence|>User: {inputs['prompt']}<|end_of_sentence|>\n"
            "Assistant:"
        )
        
        output = self.llm.create_completion(
            prompt=formatted_prompt,
            temperature=inputs.get("temperature", 0.7),
            max_tokens=inputs.get("max_tokens", 100),
            stop=["<|end_of_sentence|>", "<|im_end|>"]
        )
        
        full_response = output['choices'][0]['text'].strip()
        return full_response.split("<|end_of_sentence|>")[0].strip()

    def batch_generate(self, inputs_list: List[Dict]) -> List[str]:
        """Process multiple prompts in a single call"""
        return [self.generate(inputs) for inputs in inputs_list]

# Helper functions in model/deepseek/helpers/
# - quantize.py: 4/8-bit quantization utils
# - memory_map.py: Memory mapping optimizations
