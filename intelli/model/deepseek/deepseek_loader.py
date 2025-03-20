import os
import json
import torch
import safetensors
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download, list_repo_files

class DeepSeekLoader:
    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1"
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 model_id: str = DEFAULT_MODEL_ID,
                 device: str = "cuda", 
                 quantize: bool = False):
        self.model_id = model_id
        self.device = device
        self.quantize = quantize
        self.model = None
        self.config = None
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = self._get_default_cache_path()
            
        print(f"Using model ID: {self.model_id}")
        print(f"Using device: {self.device}")
        
        self._ensure_model_files()
        
    def _get_default_cache_path(self) -> Path:
        cache_dir = Path.home() / ".cache" / "intelli" / "models" / self.model_id.split("/")[-1]
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
        
    def _ensure_model_files(self):
        try:
            # List all files in the repository to find model weights
            download_kwargs = {
                "repo_id": self.model_id,
            }
            
            if self.hf_token:
                download_kwargs["token"] = self.hf_token
                
            files = list_repo_files(**download_kwargs)
            
            # Find model weight files
            model_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt', '.ckpt'))]
            config_files = [f for f in files if f.endswith('config.json')]
            tokenizer_files = [f for f in files if 'tokenizer' in f and f.endswith('.json')]
            
            if not model_files:
                raise ValueError(f"No model weight files found in {self.model_id}")
                
            if not config_files:
                raise ValueError(f"No config.json found in {self.model_id}")
                
            if not tokenizer_files:
                raise ValueError(f"No tokenizer files found in {self.model_id}")
                
            # Download files
            for filename in [model_files[0], config_files[0], tokenizer_files[0]]:
                file_path = self.model_path / Path(filename).name
                if not file_path.exists():
                    hf_hub_download(
                        repo_id=self.model_id,
                        filename=filename,
                        cache_dir=self.model_path,
                        token=self.hf_token
                    )
                    
            # Store filenames for later use
            self.model_file = Path(model_files[0]).name
            self.config_file = Path(config_files[0]).name
            self.tokenizer_file = Path(tokenizer_files[0]).name
                
        except Exception as e:
            print(f"Error downloading model files: {str(e)}")
            raise
            
    def load_model(self):
        config_path = self.model_path / self.config_file
        model_path = self.model_path / self.model_file
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Handle different model file formats
        if model_path.suffix == '.safetensors':
            tensors = self._load_safetensors(model_path)
        elif model_path.suffix in ['.bin', '.pt', '.ckpt']:
            tensors = self._load_torch(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
                
        return tensors
        
    def _load_safetensors(self, model_path):
        tensors = {}
        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if self.quantize and tensor.dtype == torch.float32:
                    tensor = self._quantize_tensor(tensor)
                tensors[key] = tensor
                
        if self.device != "cpu":
            for key in tensors:
                tensors[key] = tensors[key].to(self.device)
                
        return tensors
        
    def _load_torch(self, model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        
        if self.quantize:
            for key, tensor in state_dict.items():
                if tensor.dtype == torch.float32:
                    state_dict[key] = self._quantize_tensor(tensor)
                    
        if self.device != "cpu":
            for key in state_dict:
                state_dict[key] = state_dict[key].to(self.device)
                
        return state_dict
        
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.float32:
            return tensor
            
        scale = tensor.abs().max() / 127.0
        tensor_int8 = (tensor / scale).round().clip(-127, 127).to(torch.int8)
        return tensor_int8, scale

class DeepSeekModel(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        
        self.embeddings = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = torch.nn.ModuleList([
            DeepSeekLayer(config) for _ in range(self.num_layers)
        ])
        self.norm = torch.nn.LayerNorm(self.hidden_size)
        self.head = torch.nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

class DeepSeekLayer(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.attention = DeepSeekAttention(config)
        self.mlp = DeepSeekMLP(config)
        self.input_norm = torch.nn.LayerNorm(self.hidden_size)
        self.post_norm = torch.nn.LayerNorm(self.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x))
        x = x + self.mlp(self.post_norm(x))
        return x

class DeepSeekAttention(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        out = torch.matmul(attention_weights, v)
        out = out.reshape(batch_size, seq_length, self.hidden_size)
        return self.o_proj(out)

class DeepSeekMLP(torch.nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# Call NVIDIA Deepseek. 