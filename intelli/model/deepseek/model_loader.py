import os
import json
import torch
import torch.nn as nn
import safetensors.torch
from typing import Optional, Dict, Any, Union, Tuple
from einops import rearrange
import numpy as np
import requests
from tqdm import tqdm
import math

class DeepSeekModelLoader:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        quantize: bool = False,
        max_batch_size: int = 32,
        max_seq_len: int = 4096
    ):
        """Initialize the DeepSeek model loader with memory optimizations.
        
        Args:
            model_path: Path to model directory
            device: Device to load model on ('cpu', 'cuda', 'mps')
            quantize: Whether to use quantization
            max_batch_size: Maximum batch size for inference
            max_seq_len: Maximum sequence length
        """
        self.model_path = model_path
        self.device = device
        self.quantize = quantize
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # Enable torch optimizations
        torch.backends.cuda.matmul.allow_tf32 = True  
        torch.backends.cudnn.benchmark = True  
        
        # Set default dtype
        self.dtype = torch.float16 if quantize else torch.float32
        torch.set_default_dtype(self.dtype)
        
        self.config = self._load_config()
        
        self.model = None
        
        self._load_model()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_model(self):
        """Load model weights with memory optimizations."""
        # Initialize empty model structure
        self.model = DeepSeekModel(self.config)
        
        self.model = self.model.to(dtype=self.dtype)
        
        weights_path = os.path.join(self.model_path, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        with safetensors.torch.safe_open(weights_path, framework="pt", device="cpu") as f:
            tensor_names = f.keys()
            
            for tensor_name in [n for n in tensor_names if "weight" not in n]:
                tensor = f.get_tensor(tensor_name)
                tensor = tensor.to(dtype=self.dtype)
                self._set_tensor(self.model, tensor_name, tensor)
            
            for tensor_name in [n for n in tensor_names if "weight" in n]:
                tensor = f.get_tensor(tensor_name)
                
                if self.quantize and tensor.ndim == 2 and tensor.shape[0] > 1024:
                    tensor = self._quantize_to_int8(tensor)
                else:
                    tensor = tensor.to(dtype=self.dtype)
                
                self._set_tensor(self.model, tensor_name, tensor)
        
        self.model = self.model.to(device=self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None
            param.requires_grad_(False)
    
    def _quantize_to_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to int8 with dynamic scaling."""
        tensor = tensor.to(torch.float32)
        
        scale = tensor.abs().max(dim=1, keepdim=True)[0] / 127.0        
        tensor_int8 = (tensor / scale).round().to(torch.int8)
        return (tensor_int8.to(torch.float32) * scale).to(self.dtype)
    
    def _set_tensor(self, model: nn.Module, name: str, tensor: torch.Tensor):
        """Set tensor in model with memory-efficient approach."""
        if name.startswith('model.'):
            name = name[6:] 
            
        name_parts = name.split('.')
        target = model
        
        for part in name_parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"Model has no attribute {part} in {name}")
            target = getattr(target, part)
            
        if hasattr(target, name_parts[-1]):
            param = getattr(target, name_parts[-1])
            param.data = tensor
            param.requires_grad_(False)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate text with memory-efficient approach and numerical stability."""
        with torch.no_grad():
            input_ids = input_ids.to(device=self.device)
            
            batch_size = input_ids.shape[0]
            current_length = input_ids.shape[1]
            
            while current_length < max_length:
                chunk_start = max(0, current_length - self.max_seq_len)
                chunk = input_ids[:, chunk_start:current_length]
                
                outputs = self.model(chunk)
                next_token_logits = outputs[:, -1, :]
                
                if temperature > 0:
                    next_token_logits = next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0]
                    next_token_logits = next_token_logits / (temperature + 1e-8)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_tokens], dim=1)
                    current_length += 1
                    continue
                
                next_token_logits = next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0]
                probs = torch.softmax(next_token_logits, dim=-1)
                
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                current_length += 1
                
                if (next_tokens == self.model.eos_token_id).any():
                    break
                    
            return input_ids
            
    @staticmethod
    def from_pretrained(
        model_name: str,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        **kwargs
    ) -> 'DeepSeekModelLoader':
        """Load a pretrained model from HuggingFace."""
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/intelli/models")
            
        model_path = os.path.join(cache_dir, model_name.split('/')[-1])
        os.makedirs(model_path, exist_ok=True)
        
        files = ['config.json', 'model.safetensors']
        base_url = f"https://huggingface.co/{model_name}/resolve/main"
        
        headers = {}
        if hf_token:
            headers['Authorization'] = f'Bearer {hf_token}'
        
        for filename in files:
            filepath = os.path.join(model_path, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                url = f"{base_url}/{filename}"
                
                try:
                    response = requests.get(url, stream=True, headers=headers)
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 401:
                        raise ValueError(
                            f"Access to {model_name} requires authentication. Please:\n"
                            "1. Login to HuggingFace: https://huggingface.co/login\n"
                            "2. Accept the model's terms of use\n"
                            "3. Get your access token from: https://huggingface.co/settings/tokens\n"
                            "4. Pass the token as hf_token parameter"
                        ) from e
                    raise
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024 
                
                with open(filepath, 'wb') as f, tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(block_size):
                        size = f.write(data)
                        pbar.update(size)
                        
        return DeepSeekModelLoader(model_path, **kwargs)


class DeepSeekModel(nn.Module):
    """Memory-optimized implementation of DeepSeek model architecture."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        self.hidden_size = config["hidden_size"]  # 1536
        self.num_attention_heads = config["num_attention_heads"]  # 12
        self.num_key_value_heads = config["num_key_value_heads"]  # 2
        self.num_hidden_layers = config["num_hidden_layers"]  # 28
        self.vocab_size = config["vocab_size"]  # 151936
        self.intermediate_size = config["intermediate_size"]  # 8960
        
        self.bos_token_id = config.get("bos_token_id", 151646)
        self.eos_token_id = config.get("eos_token_id", 151643)
        self.pad_token_id = config.get("pad_token_id", 151643)
        
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(self.num_hidden_layers)
        ])
        
        self.norm = nn.LayerNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))
        
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            attention_mask = attention_mask & (input_ids != self.pad_token_id)
            attention_mask = attention_mask & (input_ids != self.pad_token_id)
        
        attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        
        return logits


class TransformerLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)
        
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SelfAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.get("attention_dropout", 0.0)
        self.scale = self.head_dim ** -0.5
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_heads} must be divisible by "
                f"num_key_value_heads {self.num_key_value_heads}"
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_key_value_heads, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Input hidden size {hidden_size} doesn't match configured hidden size {self.hidden_size}"
            )
        
        query_states = self.q_proj(hidden_states)  # [batch_size, seq_length, hidden_size]
        key_states = self.k_proj(hidden_states)    # [batch_size, seq_length, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)  # [batch_size, seq_length, num_kv_heads * head_dim]
        
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=2)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=2)
        
        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        expected_shape = (batch_size, self.num_heads, seq_length, self.head_dim)
        if attn_output.size() != expected_shape:
            raise ValueError(
                f"Unexpected attention output shape: got {attn_output.size()}, "
                f"expected {expected_shape}"
            )
        
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        return attn_output


class MLP(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"] 
        self.dropout = config.get("hidden_dropout", 0.0)
        
        if self.intermediate_size <= self.hidden_size:
            raise ValueError(
                f"intermediate_size {self.intermediate_size} must be larger than "
                f"hidden_size {self.hidden_size}"
            )
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # matches hidden_act: 'silu' in config
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Input hidden size {hidden_size} doesn't match configured hidden size {self.hidden_size}"
            )
        
        gate_states = self.gate_proj(hidden_states)  # [batch_size, seq_length, intermediate_size]
        up_states = self.up_proj(hidden_states)      # [batch_size, seq_length, intermediate_size]
        
        if gate_states.size() != (batch_size, seq_length, self.intermediate_size):
            raise ValueError(
                f"Unexpected gate projection shape: got {gate_states.size()}, "
                f"expected {(batch_size, seq_length, self.intermediate_size)}"
            )
        
        gate_states = self.act_fn(gate_states)
        hidden_states = gate_states * up_states
        
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states 