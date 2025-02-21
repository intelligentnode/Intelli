import torch
from typing import Union

def quantize_weights(weights: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Quantize weights using bitsandbytes with safety checks"""
    try:
        from bitsandbytes import functional as bnb
        return bnb.quantize_blockwise(weights, blocksize=64, quant_type=f"nf{bits}")
    except ImportError:
        raise RuntimeError("bitsandbytes required for quantization")

def is_quantized_model_available(repo_id: str) -> bool:
    """Check if quantized version exists on HuggingFace Hub"""
    from huggingface_hub import model_info
    info = model_info(repo_id)
    return any(f.rfilename.endswith(".gguf") for f in info.siblings) 