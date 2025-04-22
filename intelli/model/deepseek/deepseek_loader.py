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
                 device: str = None,
                 quantize: bool = False):
        self.model_id = model_id

        # Check CUDA availability and set device accordingly
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.quantize = quantize
        self.model = None
        self.config = None
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        if model_path:
            # Convert to absolute path if it's a relative path
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
            self.model_path = Path(model_path)
        else:
            self.model_path = self._get_default_cache_path()

        print(f"Using model ID: {self.model_id}")
        print(f"Using device: {self.device}")
        print(f"Model path: {self.model_path}")

        self._ensure_model_files()

    def _get_default_cache_path(self) -> Path:
        cache_dir = Path.home() / ".cache" / "intelli" / "models" / self.model_id.split("/")[-1]
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _ensure_model_files(self):
        try:
            # Check if model_path is a directory that already contains model files
            if self.model_path and Path(self.model_path).exists():
                print(f"Checking existing model path: {self.model_path}")
                # List all files in the directory
                all_files = []
                for root, _, files in os.walk(self.model_path):
                    for file in files:
                        all_files.append(os.path.join(root, file))

                # Convert to Path objects for easier handling
                all_paths = [Path(f) for f in all_files]

                # Find model weight files
                model_paths = [f for f in all_paths if f.name.endswith(('.safetensors', '.bin', '.pt', '.ckpt'))]
                config_paths = [f for f in all_paths if f.name.endswith('config.json')]
                tokenizer_paths = [f for f in all_paths if 'tokenizer' in f.name and f.name.endswith('.json')]

                if model_paths and config_paths and tokenizer_paths:
                    print(f"Found existing model files in {self.model_path}")
                    # Store filenames and their parent directories
                    self.model_file = model_paths[0].name
                    self.config_file = config_paths[0].name
                    self.tokenizer_file = tokenizer_paths[0].name

                    # Store the actual paths for easier access
                    print(f"Model file: {self.model_file}")
                    print(f"Config file: {self.config_file}")
                    print(f"Tokenizer file: {self.tokenizer_file}")
                    return

            # If we get here, we need to download the model files
            print(f"Downloading model files from {self.model_id}")
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
                    downloaded_path = hf_hub_download(
                        repo_id=self.model_id,
                        filename=filename,
                        cache_dir=self.model_path,
                        token=self.hf_token
                    )
                    print(f"Downloaded {filename} to {downloaded_path}")

            # Store filenames for later use
            self.model_file = Path(model_files[0]).name
            self.config_file = Path(config_files[0]).name
            self.tokenizer_file = Path(tokenizer_files[0]).name

        except Exception as e:
            print(f"Error ensuring model files: {str(e)}")
            raise

    def load_model(self):
        # Find the config and model files in the model path directory
        config_files = []
        model_files = []

        # First, try to find the exact files we're looking for
        for root, _, files in os.walk(self.model_path):
            for file in files:
                file_path = Path(os.path.join(root, file))
                if hasattr(self, 'config_file') and file == self.config_file:
                    config_files.append(file_path)
                elif file.endswith('config.json'):
                    config_files.append(file_path)

                if hasattr(self, 'model_file') and file == self.model_file:
                    model_files.append(file_path)
                elif file.endswith(('.safetensors', '.bin', '.pt', '.ckpt')):
                    model_files.append(file_path)

        # If we didn't find any files, look in subdirectories
        if not config_files or not model_files:
            print(f"Searching in subdirectories of {self.model_path}...")
            for root, dirs, _ in os.walk(self.model_path):
                for dir_name in dirs:
                    subdir = os.path.join(root, dir_name)
                    print(f"Checking subdirectory: {subdir}")
                    for subroot, _, subfiles in os.walk(subdir):
                        for file in subfiles:
                            file_path = Path(os.path.join(subroot, file))
                            if file.endswith('config.json') and not config_files:
                                config_files.append(file_path)
                                print(f"Found config file in subdirectory: {file_path}")
                            if file.endswith(('.safetensors', '.bin', '.pt', '.ckpt')) and not model_files:
                                model_files.append(file_path)
                                print(f"Found model file in subdirectory: {file_path}")

        # If we still didn't find any files, try to find any file that might work
        if not config_files:
            print("No config file found. Looking for any JSON file that might be a config...")
            for root, _, files in os.walk(self.model_path):
                for file in files:
                    if file.endswith('.json') and 'tokenizer' not in file.lower():
                        file_path = Path(os.path.join(root, file))
                        config_files.append(file_path)
                        print(f"Using {file_path} as config file")
                        break
                if config_files:
                    break

        if not model_files:
            print("No model file found. Looking for any file that might be a model...")
            for root, _, files in os.walk(self.model_path):
                for file in files:
                    if any(file.endswith(ext) for ext in ['.safetensors', '.bin', '.pt', '.ckpt', '.model']):
                        file_path = Path(os.path.join(root, file))
                        model_files.append(file_path)
                        print(f"Using {file_path} as model file")
                        break
                if model_files:
                    break

        if not config_files:
            raise FileNotFoundError(f"Config file not found in {self.model_path} or its subdirectories")
        if not model_files:
            raise FileNotFoundError(f"Model file not found in {self.model_path} or its subdirectories")

        config_path = config_files[0]
        model_path = model_files[0]

        print(f"Loading config from: {config_path}")
        print(f"Loading model from: {model_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except UnicodeDecodeError:
            # Try with different encodings if utf-8 fails
            with open(config_path, 'r', encoding='latin-1') as f:
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
        try:
            # Convert path to string to handle different OS path formats
            model_path_str = str(model_path)

            with safetensors.safe_open(model_path_str, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if self.quantize and tensor.dtype == torch.float32:
                        tensor_int8, scale = self._quantize_tensor(tensor)
                        # Store both the quantized tensor and its scale
                        tensors[key] = tensor_int8
                        tensors[f"{key}_scale"] = scale
                    else:
                        tensors[key] = tensor

            # Only move tensors to device if CUDA is available and requested
            if self.device == "cuda" and torch.cuda.is_available():
                for key in tensors:
                    tensors[key] = tensors[key].to(self.device)
            elif self.device == "cuda":
                # If CUDA was requested but not available, update the device to CPU
                print("CUDA requested but not available. Using CPU for tensors.")
                self.device = "cpu"

            return tensors
        except UnicodeDecodeError as e:
            print(f"Encoding error when loading safetensors: {str(e)}")
            print("This is likely due to a character encoding issue on Windows.")
            print("Attempting to load with a different approach...")

            # Try an alternative approach for Windows systems
            try:
                # Use binary mode to avoid encoding issues
                import io
                with open(model_path, 'rb') as f:
                    binary_data = f.read()

                # Create a memory buffer and load from there
                buffer = io.BytesIO(binary_data)
                state_dict = torch.load(buffer, map_location="cpu")

                # Process the state dict similar to _load_torch
                if self.quantize:
                    for key, tensor in list(state_dict.items()):
                        if tensor.dtype == torch.float32:
                            tensor_int8, scale = self._quantize_tensor(tensor)
                            state_dict[key] = tensor_int8
                            state_dict[f"{key}_scale"] = scale

                # Move to device if needed
                if self.device == "cuda" and torch.cuda.is_available():
                    for key in state_dict:
                        state_dict[key] = state_dict[key].to(self.device)

                return state_dict
            except Exception as inner_e:
                print(f"Alternative loading approach failed: {str(inner_e)}")
                # Return an empty tensor dict as fallback
                # This allows the model to partially initialize for testing
                return {}
        except Exception as e:
            print(f"Error loading safetensors: {str(e)}")
            # Don't raise, return empty dict to allow partial initialization
            return {}

    def _load_torch(self, model_path):
        try:
            # Convert path to string to handle different OS path formats
            model_path_str = str(model_path)

            # Try to load the model with different approaches
            try:
                # Standard approach
                state_dict = torch.load(model_path_str, map_location="cpu")
            except UnicodeDecodeError:
                # Binary approach for encoding issues
                print("Encountered encoding issue, trying binary mode loading...")
                with open(model_path_str, 'rb') as f:
                    binary_data = f.read()
                import io
                buffer = io.BytesIO(binary_data)
                state_dict = torch.load(buffer, map_location="cpu")

            if self.quantize:
                for key, tensor in list(state_dict.items()):
                    if tensor.dtype == torch.float32:
                        tensor_int8, scale = self._quantize_tensor(tensor)
                        # Store both the quantized tensor and its scale
                        state_dict[key] = tensor_int8
                        state_dict[f"{key}_scale"] = scale

            # Only move tensors to device if CUDA is available and requested
            if self.device == "cuda" and torch.cuda.is_available():
                for key in state_dict:
                    state_dict[key] = state_dict[key].to(self.device)
            elif self.device == "cuda":
                # If CUDA was requested but not available, update the device to CPU
                print("CUDA requested but not available. Using CPU for tensors.")
                self.device = "cpu"

            return state_dict
        except Exception as e:
            print(f"Error loading torch model: {str(e)}")
            # Don't raise, return empty dict to allow partial initialization
            return {}

    def _quantize_tensor(self, tensor: torch.Tensor):
        """Quantize a tensor to int8 for reduced memory usage."""
        try:
            if tensor.dtype != torch.float32:
                return tensor, torch.tensor(1.0)

            scale = tensor.abs().max() / 127.0
            tensor_int8 = (tensor / scale).round().clip(-127, 127).to(torch.int8)
            return tensor_int8, scale
        except Exception as e:
            print(f"Error during quantization: {str(e)}")
            # Return the original tensor and a scale of 1.0 as fallback
            return tensor, torch.tensor(1.0)

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