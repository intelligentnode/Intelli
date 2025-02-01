import unittest
import os
from dotenv import load_dotenv
from intelli.wrappers.nvidia_wrapper import NvidiaWrapper

load_dotenv()


class TestNvidiaWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("NVIDIA_API_KEY")
        assert cls.api_key, "NVIDIA_API_KEY is not set."
        cls.wrapper = NvidiaWrapper(cls.api_key)

    def test_generate_text_llama(self):
        params = {
            "model": "meta/llama-3.3-70b-instruct",
            "messages": [
                {"role": "user", "content": "Write a limerick about GPU computing."}
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.7,
            "stream": False,
        }
        response = self.wrapper.generate_text(params)
        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        message = response["choices"][0]["message"]["content"]
        self.assertTrue(isinstance(message, str) and len(message) > 0)

    def test_generate_text_deepseek(self):
        params = {
            "model": "deepseek-ai/deepseek-r1",
            "messages": [
                {"role": "user", "content": "Which number is larger, 9.11 or 9.8?"}
            ],
            "max_tokens": 4096,
            "temperature": 0.6,
            "top_p": 0.7,
            "stream": False,
        }
        response = self.wrapper.generate_text(params)
        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        message = response["choices"][0]["message"]["content"]
        self.assertTrue(isinstance(message, str) and len(message) > 0)

    def test_get_embeddings(self):
        params = {
            "input": ["What is the capital of France?"],
            "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "NONE",
        }
        response = self.wrapper.get_embeddings(params)
        self.assertIn("data", response)
        self.assertGreater(len(response["data"]), 0)
        self.assertIn("embedding", response["data"][0])
        embedding = response["data"][0]["embedding"]
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)


if __name__ == "__main__":
    unittest.main()
