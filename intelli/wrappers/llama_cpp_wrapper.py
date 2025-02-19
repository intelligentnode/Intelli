import os
import logging
import requests
import importlib
import numpy as np
from typing import Optional, Dict, Union, List

# Initial attempt to import
try:
    import llama_cpp
except ImportError:
    llama_cpp = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


class IntelliLlamaCPPWrapper:
    """
    IntelliLlamaCPPWrapper integrates llama.cpp for local inference or server-based usage.

    Basic usage:
        1. Offline mode:
            wrapper = IntelliLlamaCPPWrapper(
                model_path="path/to/model.gguf",
                model_params={"n_ctx": 512, "n_gpu_layers": 20, ...}
            )
            result = wrapper.generate_text({"prompt": "Hello world"})
            print(result["choices"][0]["text"])

        2. Server mode:
            wrapper = IntelliLlamaCPPWrapper(server_url="http://localhost:8080")
            result = wrapper.generate_text({"prompt": "Hello from server"})
            print(result["choices"][0]["text"])

        3. Embeddings:
            # Ensure you load the model with embedding=True in model_params for offline mode.
            wrapper = IntelliLlamaCPPWrapper(
                model_path="model.gguf",
                model_params={"embedding": True, "n_ctx": 512}
            )
            emb = wrapper.get_embeddings({"input": "Hello Llama"})
            print(emb)    # returns a dict with key "embedding" (a flat list of floats)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        server_url: Optional[str] = None,
        model_params: Optional[Dict] = None,
    ):
        """
        If server_url is set, we'll use a running llama.cpp server. Otherwise, if model_path is provided,
        we load that model locally. If neither is set, you can still call 'load_local_model()' later.

        Args:
            model_path: Path to a local GGUF model file or None if using server.
            server_url: URL to a llama.cpp server.
            model_params: Dictionary with model parameters (e.g., n_ctx, n_gpu_layers, embedding, etc.)
        """
        # Declare globals at the top of __init__
        global llama_cpp
        global hf_hub_download

        self.logger = logging.getLogger(__name__)
        self.model = None
        self.server_url = server_url
        self.model_params = model_params or {}

        # If llama_cpp is still None, try dynamic import
        if llama_cpp is None:
            try:
                llama_cpp = importlib.import_module("llama_cpp")
                self.logger.info(
                    "Successfully imported llama_cpp after dynamic import."
                )
            except ImportError:
                self.logger.error(
                    "Could not import llama_cpp. Please install it via "
                    "`pip install intelli[llamacpp]` or `pip install llama-cpp-python`."
                )

        # If huggingface_hub is not available, try dynamic import
        if hf_hub_download is None:
            try:
                hf_hub = importlib.import_module("huggingface_hub")
                hf_hub_download = hf_hub.hf_hub_download
                self.logger.info(
                    "Successfully imported huggingface_hub after dynamic import."
                )
            except ImportError:
                self.logger.error(
                    "Could not import huggingface_hub. Please install it via "
                    "`pip install intelli[llamacpp]` or `pip install huggingface_hub`."
                )

        if self.server_url:
            self.logger.info(f"Using server mode at {self.server_url}")
        else:
            if model_path is not None:
                self.load_local_model(model_path, self.model_params)

    def download_model(
        self, repo_id: str, filename: str, local_dir: str = "models"
    ) -> str:
        """
        Download a GGUF model file from Hugging Face Hub into local_dir.

        Args:
            repo_id: The Hugging Face repository ID (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF").
            filename: The model filename to download (e.g., "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf").
            local_dir: Directory to save the model.
        Returns:
            The local file path to the downloaded model.
        """
        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is not installed; use 'pip install intelli[llamacpp]'"
            )

        self.logger.info(f"Downloading {filename} from {repo_id} to {local_dir}")
        file_path = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=local_dir
        )
        self.logger.info(f"Model downloaded to: {file_path}")
        return file_path

    def load_local_model(self, model_path: str, model_params: Dict):
        """
        Load a llama.cpp model file for offline usage using llama-cpp-python.

        Args:
            model_path: The path to your GGUF model file.
            model_params: Additional parameters for llama-cpp (n_ctx, n_gpu_layers, embedding, etc.)
        """
        if llama_cpp is None:
            raise ImportError(
                "llama-cpp-python not installed; use 'pip install intelli[llamacpp]'"
            )

        self.logger.info(f"Loading local llama.cpp model from: {model_path}")

        n_ctx = model_params.get("n_ctx", 2048)
        n_gpu_layers = model_params.get("n_gpu_layers", 0)
        n_threads = model_params.get("n_threads", 1)
        embedding_mode = model_params.get("embedding", False)
        seed = model_params.get("seed", -1)
        n_batch = model_params.get("n_batch", 512)
        verbose = model_params.get("verbose", False)  # Set verbose to False by default

        self.model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            embedding=embedding_mode,
            seed=seed,
            n_batch=n_batch,
            verbose=verbose,
        )
        self.logger.info("Llama model loaded offline successfully.")

    def generate_text(self, params: Dict) -> Dict:
        """
        Generate text via local inference or server.

        Args:
            params: A dict including:
                - prompt (str): The prompt text.
                - max_tokens (int): Maximum tokens to generate.
                - temperature (float): Sampling temperature.
                - top_p (float): Top-p sampling.
                - (Optional) stop (list): List of stop sequences.
                - (Optional) echo (bool): Whether to echo the prompt.

        Returns:
            A dict with structure: {"choices": [{"text": "..."}]}.
        """
        prompt = params.get("prompt", "")
        if not prompt:
            raise ValueError("generate_text requires 'prompt' in params")

        max_tokens = params.get("max_tokens", 128)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.95)

        if self.server_url:
            return self._generate_text_server(
                prompt, max_tokens, temperature, top_p, params
            )
        else:
            return self._generate_text_local(
                prompt, max_tokens, temperature, top_p, params
            )

    def _generate_text_local(self, prompt, max_tokens, temperature, top_p, params):
        if not self.model:
            raise RuntimeError("Local model not loaded. Call load_local_model() first.")

        generate_kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": params.get("stop", []),
            "echo": params.get("echo", False),
        }
        result = self.model(**generate_kwargs)
        # Newer llama-cpp-python returns dict; older may return plain text.
        if isinstance(result, str):
            text_out = result
        else:
            text_out = result["choices"][0]["text"]

        return {"choices": [{"text": text_out}]}

    def _generate_text_server(self, prompt, max_tokens, temperature, top_p, params):
        url = f"{self.server_url}/v1/completions"
        payload = {
            "model": params.get("model", "default"),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()  # e.g. {"choices": [{"text": "..."}]}
        except requests.RequestException as e:
            raise RuntimeError(f"Llama server request failed: {e}")

    def generate_text_stream(self, params: Dict):
        """
        Stream text from offline or server. Yields partial text chunks as str.

        Args:
            params: Same as generate_text, with an additional 'stream': True.
        """
        if self.server_url:
            yield from self._generate_text_stream_server(params)
        else:
            yield from self._generate_text_stream_local(params)

    def _generate_text_stream_server(self, params: Dict):
        url = f"{self.server_url}/v1/completions"
        payload = {
            "model": params.get("model", "default"),
            "prompt": params.get("prompt", ""),
            "max_tokens": params.get("max_tokens", 128),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.95),
            "stream": True,
        }
        try:
            with requests.post(url, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if line and line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        yield line[len("data: ") :].strip()
        except requests.RequestException as e:
            raise RuntimeError(f"Llama server streaming error: {e}")

    def _generate_text_stream_local(self, params: Dict):
        if not self.model:
            raise RuntimeError("No local model loaded for streaming.")

        prompt = params.get("prompt", "")
        max_tokens = params.get("max_tokens", 128)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.95)

        res = self._generate_text_local(prompt, max_tokens, temperature, top_p, params)
        full_text = res["choices"][0]["text"]
        chunk_size = 16
        for i in range(0, len(full_text), chunk_size):
            yield full_text[i : i + chunk_size]

    def get_embeddings(self, params: Dict) -> Union[Dict, List[Dict]]:
        """
        Retrieve embeddings from the local model or server.

        Args:
            params: A dict with key "input": either a str or a list of str,
                    and optionally "model" for server mode.

        For local mode, returns a dict with key "embedding" (a flat list of floats)
        if a single string is provided; if a list is provided, returns a list of such dicts.

        Returns:
            The embedding(s) in a simplified format.
        """
        if self.server_url:
            return self._get_embeddings_server(params)
        else:
            if not self.model:
                raise RuntimeError(
                    "Local model not loaded; cannot retrieve embeddings."
                )

            input_data = params.get("input", None)
            if not input_data:
                raise ValueError("No 'input' provided for embeddings.")

            if isinstance(input_data, str):
                raw_emb = self.model.create_embedding(input_data)
                return self._process_embedding_output(raw_emb)
            elif isinstance(input_data, list):
                results = []
                for text_item in input_data:
                    raw_emb = self.model.create_embedding(text_item)
                    results.append(self._process_embedding_output(raw_emb))
                return results
            else:
                raise TypeError("'input' must be a str or list of str")

    def _process_embedding_output(self, raw_emb: Dict) -> Dict:
        """
        Process the raw output from llama_cpp.Llama.create_embedding() into a simple dict.

        Expected raw_emb structure (for a single input):
        {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [[...token1...], [...token2...], ...], "index": 0}
            ],
            "model": "path/to/model",
            "usage": {"prompt_tokens": ..., "total_tokens": ...}
        }

        This function averages the token-level embeddings into a single flat vector.

        Args:
            raw_emb: Raw embedding output dictionary.
        Returns:
            A dict with key "embedding" containing a flat list of floats.
        """
        if (
            isinstance(raw_emb, dict)
            and raw_emb.get("object") == "list"
            and "data" in raw_emb
        ):
            data = raw_emb["data"]
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if "embedding" in first_item:
                    emb = first_item["embedding"]
                    # If emb is a list of lists (token embeddings), average them.
                    if (
                        isinstance(emb, list)
                        and len(emb) > 0
                        and isinstance(emb[0], list)
                    ):
                        emb_array = np.array(emb)
                        avg_emb = emb_array.mean(axis=0).tolist()
                        return {"embedding": avg_emb}
                    else:
                        return {"embedding": emb}
        # Fallback: return the raw output
        return raw_emb

    def _get_embeddings_server(self, params: Dict):
        """
        Retrieve embeddings via a llama.cpp server that supports /v1/embeddings.

        Args:
            params: A dict with keys "input" and optionally "model".
        Returns:
            The server's JSON response.
        """
        url = f"{self.server_url}/v1/embeddings"
        payload = {
            "model": params.get("model", "default"),
            "input": params.get("input", ""),
        }
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Embedding server request failed: {e}")
