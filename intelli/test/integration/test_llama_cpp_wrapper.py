import unittest
import os
import shutil

from intelli.wrappers.llama_cpp_wrapper import IntelliLlamaCPPWrapper

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


class TestIntelliLlamaCPPWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create a temp directory and download a small LLaMA-based GGUF model
        for offline tests.
        """
        cls.temp_dir = os.path.join("..", "temp", "tinyllama_tests")
        os.makedirs(cls.temp_dir, exist_ok=True)

        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is not installed. Use 'pip install intelli[llamacpp]'."
            )

        # Use TheBloke's TinyLlama model for testing
        cls.repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        cls.filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

        try:
            print(f"Downloading {cls.filename} from {cls.repo_id} to {cls.temp_dir}")
            cls.model_path = hf_hub_download(
                repo_id=cls.repo_id, filename=cls.filename, local_dir=cls.temp_dir
            )
            print(f"Model downloaded: {cls.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed downloading the TinyLlama model: {e}")

    @classmethod
    def tearDownClass(cls):
        """
        Remove the temp directory to clean up after tests.
        """
        if os.path.exists(cls.temp_dir):
            try:
                shutil.rmtree(cls.temp_dir)
            except OSError as err:
                print(f"Error removing temp dir {cls.temp_dir}: {err}")

    def test_offline_model_generation_basic(self):
        """
        Test generating text with a local llama.cpp model using IntelliLlamaCPPWrapper.
        """
        wrapper = IntelliLlamaCPPWrapper(
            model_path=self.model_path,
            model_params={"n_threads": 2, "n_ctx": 512, "n_batch": 256},
        )
        params = {
            "prompt": "User: Hello Llama, how are you?\nAssistant:",
            "max_tokens": 32,
            "temperature": 0.8,
            "top_p": 0.9,
        }
        result = wrapper.generate_text(params)
        self.assertIn("choices", result, "Result should have 'choices' key.")
        self.assertGreater(
            len(result["choices"]), 0, "Should have at least one choice."
        )
        text_out = result["choices"][0]["text"]
        self.assertIsInstance(text_out, str, "The generated text should be a string.")
        self.assertGreater(len(text_out), 0, "Output text should not be empty.")
        print(f"Offline generation output:\n{text_out}\n")

    def test_offline_model_generation_second(self):
        """
        Another chat test to ensure multiple prompts work fine.
        """
        wrapper = IntelliLlamaCPPWrapper(
            model_path=self.model_path,
            model_params={"n_threads": 2, "n_ctx": 512, "n_batch": 256},
        )
        # Update prompt to follow a chat format for consistent output.
        params = {
            "prompt": "User: What is 2+2?\nAssistant:",
            "max_tokens": 30,
            "temperature": 0.6,
            "top_p": 0.95,
        }
        result = wrapper.generate_text(params)
        self.assertIn("choices", result)
        self.assertTrue(len(result["choices"]) > 0)
        text_out = result["choices"][0]["text"]
        self.assertTrue(isinstance(text_out, str))
        self.assertTrue(len(text_out) > 0, "Output text should not be empty.")
        print(f"Second offline generation output:\n{text_out}\n")

    def test_offline_embeddings(self):
        """
        A single test for embedding extraction, ensuring we retrieve
        a valid embedding dict from llama-cpp-python for a single input.
        """
        wrapper = IntelliLlamaCPPWrapper()
        # "embedding": True is required for offline embedding mode.
        model_params = {"embedding": True, "n_threads": 2, "n_ctx": 256, "n_batch": 128}
        wrapper.load_local_model(self.model_path, model_params)

        text = "Hello from TinyLlama"
        emb_result = wrapper.get_embeddings({"input": text})

        # Expecting a simplified dict with key "embedding" (a flat list of floats).
        self.assertIsInstance(
            emb_result, dict, "Should be a dict for single input embedding."
        )
        self.assertIn("embedding", emb_result, "Result must have 'embedding' key.")
        emb = emb_result["embedding"]
        self.assertIsInstance(emb, list, "'embedding' should be a list of floats.")
        self.assertGreater(
            len(emb), 10, "Embedding vector should have more than 10 dimensions."
        )
        print(f"Embedding vector sample (first 5 dims): {emb[:5]} ...\n")

    def test_server_mode_generation(self):
        """
        Optional test: if a llama.cpp server is running on localhost:8080,
        generate text via server mode.
        """
        server_url = "http://localhost:8080"
        wrapper = IntelliLlamaCPPWrapper(server_url=server_url)
        params = {
            "prompt": "User: Hello from server mode!\nAssistant:",
            "max_tokens": 20,
            "temperature": 0.7,
        }
        try:
            result = wrapper.generate_text(params)
            self.assertIn("choices", result)
            self.assertGreater(len(result["choices"]), 0)
            text_out = result["choices"][0]["text"]
            self.assertIsInstance(text_out, str)
            self.assertGreater(len(text_out), 0)
            print(f"Server generation output:\n{text_out}\n")
        except Exception as e:
            self.skipTest(
                f"Skipping server mode test. Server not available or failed: {e}"
            )


if __name__ == "__main__":
    unittest.main()
