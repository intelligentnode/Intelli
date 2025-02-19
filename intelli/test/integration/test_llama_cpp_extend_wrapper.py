import unittest
import os
import shutil
import sys
import importlib
import contextlib

from intelli.wrappers.llama_cpp_wrapper import IntelliLlamaCPPWrapper

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


@contextlib.contextmanager
def suppress_stderr():
    """
    Temporarily redirect stderr to /dev/null
    to reduce llama.cpp's metal/C++ logs.
    """
    null = None
    old_stderr = None
    try:
        null = open(os.devnull, 'w')
        old_stderr = sys.stderr
        sys.stderr = null
        yield
    finally:
        if null:
            null.close()
        if old_stderr:
            sys.stderr = old_stderr


class TestIntelliLlamaCPPWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create a temp directory and download multiple sub-7B GGUF models
        for llama.cpp usage. We'll test each one if it downloads successfully.

        If a download fails, we explicitly raise an exception so
        the test fails (instead of skipping).
        """
        cls.temp_dir = os.path.join("..", "temp", "llama_cpp_wrapper_extend_noskip")
        os.makedirs(cls.temp_dir, exist_ok=True)

        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is not installed. Use 'pip install intelli[llamacpp]'."
            )

        cls.models_info = {
            "tiny_llama_1.1b": {
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            },
            "deepseek_qwen_1.5b": {
                "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
                "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf",
            },
            "gemma_2b_it": {
                "repo_id": "bartowski/gemma-2-2b-it-GGUF",
                "filename": "gemma-2-2b-it-Q4_K_M.gguf",
            },
        }

        cls.downloaded_paths = {}
        for key, info in cls.models_info.items():
            repo_id = info["repo_id"]
            filename = info["filename"]
            print(f"\nAttempting download of {filename} from {repo_id} to {cls.temp_dir}")
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=cls.temp_dir
                )
                cls.downloaded_paths[key] = model_path
                print(f"Model [{key}] downloaded to: {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed downloading {repo_id}/{filename}: {e}"
                )

    @classmethod
    def tearDownClass(cls):
        """
        Optionally remove the temp directory after tests.
        """
        if os.path.exists(cls.temp_dir):
            try:
                shutil.rmtree(cls.temp_dir)
            except OSError as err:
                print(f"Error removing temp dir {cls.temp_dir}: {err}")

    def test_deepseek_qwen_generation(self):
        """
        Test conversation with DeepSeek-R1-Distill-Qwen-1.5B.
        """
        key = "deepseek_qwen_1.5b"
        model_path = self.downloaded_paths[key]

        with suppress_stderr():
            # Hide logs while loading/running
            wrapper = IntelliLlamaCPPWrapper(
                model_path=model_path,
                model_params={
                    "n_threads": 2,
                    "n_ctx": 512,
                    "n_batch": 128,
                    "verbose": False  # disable python-level logs
                },
            )

        # The recommended prompt format from the repo:
        # <|begin_of_sentence|> for system, <|User|>, <|Assistant|>
        prompt = (
            "<|begin_of_sentence|>System: You are a helpful assistant.\n"
            "<|User|>What's the capital of France?\n"
            "<|Assistant|>"
        )
        params = {
            "prompt": prompt,
            "max_tokens": 40,
            "temperature": 0.7,
        }
        with suppress_stderr():
            # Hide logs during generation
            result = wrapper.generate_text(params)
        text_out = result["choices"][0]["text"]
        print(f"\n[DeepSeek-Qwen-1.5B] Chat output:\n{text_out}\n")
        self.assertIn("Paris", text_out, "Expected mention of 'Paris' in the answer.")

    def test_gemma_2b_generation(self):
        """
        Text generation with gemma-2-2b-it (Q4_K_M).
        """
        key = "gemma_2b_it"
        model_path = self.downloaded_paths[key]

        with suppress_stderr():
            wrapper = IntelliLlamaCPPWrapper(
                model_path=model_path,
                model_params={
                    "n_threads": 2,
                    "n_ctx": 512,
                    "n_batch": 128,
                    "verbose": False,
                }
            )
        # The recommended prompt format from the repo might have <bos>, etc.
        prompt = "<bos><start_of_turn>user Hello gemma-2-2b, how do you greet in Italian?\n<start_of_turn>model"
        params = {
            "prompt": prompt,
            "max_tokens": 40,
            "temperature": 0.7,
        }
        with suppress_stderr():
            result = wrapper.generate_text(params)
        text_out = result["choices"][0]["text"]
        print(f"\n[gemma-2-2b-it] Chat output:\n{text_out}\n")
        self.assertTrue(any(w in text_out.lower() for w in ["ciao", "salve"]),
                        "Expected some Italian greeting in the output.")

    def test_tiny_llama_generation(self):
        """
        Basic text generation with TinyLlama 1.1B chat model.
        """
        key = "tiny_llama_1.1b"
        model_path = self.downloaded_paths[key]

        with suppress_stderr():
            wrapper = IntelliLlamaCPPWrapper(
                model_path=model_path,
                model_params={
                    "n_threads": 2,
                    "n_ctx": 512,
                    "n_batch": 256,
                    "verbose": False,
                },
            )
        params = {
            "prompt": "User: Hi TinyLlama, what's 2 + 2?\nAssistant:",
            "max_tokens": 32,
            "temperature": 0.8,
            "top_p": 0.9,
        }
        with suppress_stderr():
            result = wrapper.generate_text(params)
        text_out = result["choices"][0]["text"]
        print(f"\n[TinyLlama-1.1B] Output:\n{text_out}\n")
        self.assertIn("4", text_out, "Expected the answer '4' in the output.")

    def test_offline_embeddings_tiny(self):
        """
        Single test for embedding extraction with TinyLlama.
        """
        key = "tiny_llama_1.1b"
        model_path = self.downloaded_paths[key]

        with suppress_stderr():
            wrapper = IntelliLlamaCPPWrapper()
            model_params = {
                "embedding": True,
                "n_threads": 2,
                "n_ctx": 256,
                "n_batch": 128,
                "verbose": False,
            }
            wrapper.load_local_model(model_path, model_params)

        text = "Ciao mondo! (embedding test)"
        with suppress_stderr():
            emb_result = wrapper.get_embeddings({"input": text})
        emb = emb_result["embedding"]
        print(f"\n[TinyLlama-1.1b embedding] vector sample (first 5 dims): {emb[:5]}")
        self.assertGreater(len(emb), 10, "Should have more than 10 dims.")


if __name__ == "__main__":
    unittest.main()
