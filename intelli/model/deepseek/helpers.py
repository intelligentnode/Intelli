import json
import os
import re
import torch
from safetensors.torch import safe_open, load_file
from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError
from huggingface_hub import HfApi, hf_hub_download


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
    with open(tok_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = data["model"]["vocab"]
    merges = data["model"]["merges"]
    merge_ranks = {tuple(merge.split()): idx for idx, merge in enumerate(merges)}
    return vocab, merge_ranks


def bpe_tokenize(text: str, vocab: dict, merge_ranks: dict):
    text = text.replace(" ", "Ġ")
    tokens = [chr(b) for b in text.encode("utf-8")]

    def get_pairs(tokens):
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    while True:
        pairs = get_pairs(tokens)
        if not pairs:
            break

        best, best_rank = None, float("inf")
        for p in pairs:
            r = merge_ranks.get(p)
            if r is not None and r < best_rank:
                best, best_rank = p, r

        if best is None:
            break

        new_tokens, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return [vocab.get(t, vocab.get("<unk>", 0)) for t in tokens]


def load_safetensors_weights(
    model, model_path: str, repo_id: str = None, cache_dir: str = "~/.cache/deepseek"
):
    """Loads model weights from split safetensors using the index file"""
    if model_path.endswith(".index.json"):
        if repo_id is None:
            raise ValueError("repo_id is required to load weights using index file")

        index_path = download_model_index(repo_id, cache_dir)
        with open(index_path, "r") as f:
            index_data = json.load(f)

        shard_files = set(index_data["weight_map"].values())
        shard_paths = [
            download_model(repo_id, shard, cache_dir) for shard in shard_files
        ]

        state_dict = {}
        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device=get_device()) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

            model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully from split safetensors.")

    else:
        weights = load_file(model_path)
        model.load_state_dict(weights, strict=False)
        print("Loaded single safetensors file.")


def load_model_weights(
    model, repo_id: str = None, files: list = None, cache_dir: str = "~/.cache/deepseek"
):
    """
    Loads model weights into the provided model using any supported format:
    - Split safetensors (index)
    - Single safetensors file
    - PyTorch .bin file
    """
    if files is None:
        api = HfApi()
        try:
            files = api.list_repo_files(repo_id)
        except Exception as e:
            raise RuntimeError(f"Could not list files for {repo_id}") from e

    if "model.safetensors.index.json" in files:
        index_path = download_model_index(repo_id, cache_dir)
        load_safetensors_weights(model, index_path, repo_id=repo_id)
        print("Loaded model from split safetensors.")
    elif "model.safetensors" in files:
        model_path = download_model(repo_id, "model.safetensors", cache_dir)
        load_safetensors_weights(model, model_path, repo_id=None)
        print("Loaded model from single safetensors file.")
    elif "pytorch_model.bin" in files:
        model_path = download_model(repo_id, "pytorch_model.bin", cache_dir)
        print(f"Loading PyTorch .bin model: {model_path}")
        state_dict = torch.load(model_path, map_location=get_device())
        model.load_state_dict(state_dict, strict=False)
        print("Loaded model from PyTorch .bin file.")
    else:
        raise FileNotFoundError(
            f"None of the supported model formats found in repo {repo_id}."
        )


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
