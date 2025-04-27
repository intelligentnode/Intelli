import json
from safetensors.torch import safe_open
import os
from huggingface_hub import hf_hub_download
import torch


def get_device():
    """Returns the best available device (CUDA if available, otherwise CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def download_model(repo_id: str, filename: str, cache_dir: str = "~/.cache/deepseek"):
    """Downloads a model file from Hugging Face if not found locally"""
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, filename)

    if not os.path.exists(model_path):
        print(f"Downloading {filename} from {repo_id}...")
        model_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir
        )
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Using cached model: {model_path}")

    return model_path


def download_model_index(repo_id: str, cache_dir: str = "~/.cache/deepseek"):
    """Downloads the model index JSON that references all split safetensors"""
    filename = "model.safetensors.index.json"
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    model_index_path = os.path.join(cache_dir, filename)

    if not os.path.exists(model_index_path):
        print(f"Downloading {filename} from {repo_id}...")
        model_index_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir
        )
        print(f"Index downloaded to {model_index_path}")
    else:
        print(f"Using cached index: {model_index_path}")

    return model_index_path


def load_bpe_tokenizer(repo_id: str, cache_dir: str = "~/.cache/deepseek"):
    """
    Downloads tokenizer.json, and returns:
      - vocab: dict[token:str -> id:int]
      - merges: list of merge rules [(a:str, b:str), ...]
    """
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    tok_path = os.path.join(cache_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = hf_hub_download(
            repo_id=repo_id, filename="tokenizer.json", cache_dir=cache_dir
        )
    data = json.load(open(tok_path, "r", encoding="utf-8"))
    vocab = data["model"]["vocab"]
    merges = [tuple(pair.split()) for pair in data["model"]["merges"]]
    return vocab, merges


def bpe_tokenize(text: str, vocab: dict, merges: list):
    """
    Tokenize text using Byte-level BPE algorithm.

    Args:
        text: Text to tokenize
        vocab: Dictionary mapping tokens to IDs
        merges: List of merge rules as tuples

    Returns:
        List of token IDs
    """
    # Start with byte-level representation
    try:
        # Convert text to UTF-8 bytes
        bytes_encoded = text.encode("utf-8")

        # Convert bytes to individual characters
        tokens = [chr(b) for b in bytes_encoded]

        # Apply merges in order
        i = 0
        while i < len(merges) and len(tokens) > 1:
            pair = merges[i]
            a, b = pair

            # Find all occurrences of the pair and merge them
            j = 0
            while j < len(tokens) - 1:
                if tokens[j] == a and tokens[j+1] == b:
                    tokens[j:j+2] = [a+b]  # Merge the pair
                else:
                    j += 1

            i += 1

        # Start with BOS token
        token_ids = [1]  # <s> token

        # Convert tokens to IDs
        for token in tokens:
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                # If token not in vocab, try to encode it byte by byte
                found = False
                for char in token:
                    if char in vocab:
                        token_ids.append(vocab[char])
                        found = True
                    else:
                        # Try different UNK token formats
                        if "[UNK]" in vocab:
                            token_ids.append(vocab["[UNK]"])
                        elif "<unk>" in vocab:
                            token_ids.append(vocab["<unk>"])
                        else:
                            # Default to 0 if no UNK token found
                            token_ids.append(0)

                # If we couldn't encode any part of the token, add UNK
                if not found:
                    # Try different UNK token formats
                    if "[UNK]" in vocab:
                        token_ids.append(vocab["[UNK]"])
                    elif "<unk>" in vocab:
                        token_ids.append(vocab["<unk>"])
                    else:
                        # Default to 0 if no UNK token found
                        token_ids.append(0)

                    # Handle common whitespace characters
                    if token in [' ', '\t', '\n', '\r']:
                        # Try to find the token in the vocabulary again
                        if token in vocab:
                            token_ids.append(vocab[token])
                        else:
                            # If still not found, use UNK but don't print a warning
                            if "[UNK]" in vocab:
                                token_ids.append(vocab["[UNK]"])
                            elif "<unk>" in vocab:
                                token_ids.append(vocab["<unk>"])
                            else:
                                token_ids.append(0)
                    else:
                        # For non-whitespace tokens, print a warning
                        print(f"Unknown token: {repr(token)}")

        return token_ids
    except Exception as e:
        print(f"Error in BPE tokenization: {str(e)}")
        # Return a minimal token sequence as fallback
        return [1]  # Just return start token


def load_safetensors_weights(
    model, model_path: str, repo_id: str = None, cache_dir: str = "~/.cache/deepseek"
):
    """Loads model weights from split safetensors using the index file"""
    if not repo_id:
        raise ValueError("repo_id is required to load weights using index file")

    index_path = download_model_index(repo_id, cache_dir)
    with open(index_path, "r") as f:
        index_data = json.load(f)

    shard_files = set(index_data["weight_map"].values())

    shard_paths = [download_model(repo_id, shard, cache_dir) for shard in shard_files]

    state_dict = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device=get_device()) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully from split safetensors.")


def quantize_model(model, dtype=torch.float16):
    """Converts the model to a lower precision for memory optimization"""
    model = model.half() if dtype == torch.float16 else model
    return model


def download_config(repo_id: str, cache_dir: str = "~/.cache/deepseek"):
    """Downloads config.json from Hugging Face if not found locally"""
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    config_path = os.path.join(cache_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"Downloading config.json from {repo_id}...")
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", cache_dir=cache_dir
        )
        print(f"Config downloaded to {config_path}")
    else:
        print(f"Using cached config: {config_path}")

    return config_path
