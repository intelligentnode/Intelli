import unittest
import os
import json
from dotenv import load_dotenv
from intelli.wrappers.vllm_wrapper import VLLMWrapper

load_dotenv()


class TestVLLMWrapperIntegration(unittest.TestCase):
    """Integration tests for VLLMWrapper."""

    def setUp(self):
        """Set up test environment."""
        self.vllm_embed_url = os.getenv("VLLM_EMBED_URL")
        self.deepseek_url = os.getenv("DEEPSEEK_VLLM_URL")
        self.llama_url = os.getenv("LLAMA_VLLM_URL")

    def test_vllm_embedding(self):
        """Test embedding functionality."""
        if not self.vllm_embed_url:
            self.skipTest("VLLM_EMBED_URL environment variable not set")

        wrapper = VLLMWrapper(self.vllm_embed_url)

        # Fix: Pass a dictionary with a "texts" key instead of a direct list
        response = wrapper.get_embeddings({"texts": ["hello world"]})
        print("VLLM Embeddings sample:", response["embeddings"][0][:3])

        self.assertIn("embeddings", response)
        self.assertTrue(len(response["embeddings"]) > 0)
        self.assertTrue(len(response["embeddings"][0]) > 0)

    def test_deepseek_completion(self):
        """Test completion with DeepSeek model."""
        if not self.deepseek_url:
            self.skipTest("DEEPSEEK_VLLM_URL environment variable not set")

        wrapper = VLLMWrapper(self.deepseek_url)

        params = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "prompt": "What is machine learning?",
            "max_tokens": 100,
            "temperature": 0.7
        }

        response = wrapper.generate_text(params)
        print("Deepseek Completion:", response["choices"][0]["text"])

        self.assertIn("choices", response)
        self.assertTrue(len(response["choices"]) > 0)
        self.assertIn("text", response["choices"][0])
        self.assertTrue(len(response["choices"][0]["text"]) > 0)

    def test_deepseek_streaming(self):
        """Test the streaming functionality with DeepSeek model."""
        if not self.deepseek_url:
            self.skipTest("DEEPSEEK_VLLM_URL environment variable not set")

        try:
            # Create wrapper with debugging enabled
            wrapper = VLLMWrapper(self.deepseek_url)
            wrapper.is_log = True

            # Set up test parameters
            params = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "prompt": "Hello there, how are you?",
                "max_tokens": 50,
                "temperature": 0.2,
                "stream": True
            }

            print("\n\nTesting DeepSeek Streaming:")

            # First test direct API to verify it works
            import requests
            headers = {"Content-Type": "application/json"}
            direct_response = requests.post(
                f"{self.deepseek_url}/v1/completions",
                json=params,
                headers=headers,
                stream=True
            )

            print("Direct API response status:", direct_response.status_code)

            # Now test through our wrapper
            print("Testing through wrapper:")
            full_text = ""

            # Collect output from stream
            for chunk in wrapper.generate_text_stream(params):
                print(f"Received chunk: '{chunk}'")
                full_text += chunk

            print(f"Complete text: '{full_text}'")
            self.assertTrue(len(full_text) > 0, "DeepSeek streaming response should not be empty")

        except Exception as e:
            import traceback
            print("\nDetailed error trace:")
            traceback.print_exc()
            self.fail(f"DeepSeek streaming test failed: {str(e)}")

    def test_llama_completion(self):
        """Test completion with Llama model."""
        if not self.llama_url:
            self.skipTest("LLAMA_VLLM_URL environment variable not set")

        wrapper = VLLMWrapper(self.llama_url)

        params = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "prompt": "What is machine learning?",
            "max_tokens": 100,
            "temperature": 0.7
        }

        response = wrapper.generate_text(params)
        print("Llama Completion:", response["choices"][0]["text"])

        self.assertIn("choices", response)
        self.assertTrue(len(response["choices"]) > 0)
        self.assertIn("text", response["choices"][0])
        self.assertTrue(len(response["choices"][0]["text"]) > 0)


if __name__ == "__main__":
    unittest.main()