# test_nvidia_wrapper_nim.py

import unittest
import os
from dotenv import load_dotenv
from intelli.wrappers.nvidia_wrapper import NvidiaWrapper

# Load environment variables from .env file (if available)
load_dotenv()

class TestNvidiaWrapperNim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # get the API key and the local NIM URL
        cls.api_key = os.getenv("NVIDIA_API_KEY")
        cls.nim_base_url = os.getenv("NVIDIA_NIM_BASE_URL", "http://localhost:8000")
        if not cls.api_key:
            raise ValueError("NVIDIA_API_KEY must be set in your environment.")
        # create the wrapper using the local base URL.
        cls.wrapper = NvidiaWrapper(cls.api_key, base_url=cls.nim_base_url)

    def test_chat_completion(self):
        """Test chat completion using NVIDIA NIM."""
        params = {
            "model": "google/gemma-2-9b-it",
            "messages": [
                {"role": "user", "content": "Write a limerick about GPU computing."}
            ],
            "max_tokens": 64,
            "temperature": 0.5,
            "top_p": 1,
            "stream": False,
        }
        response = self.wrapper.generate_text(params)
        self.assertIn("choices", response, "Response should contain 'choices'.")
        self.assertGreater(len(response["choices"]), 0, "There should be at least one choice.")
        # verify non-empty string.
        message = response["choices"][0]["message"]["content"]
        self.assertIsInstance(message, str, "Message content should be a string.")
        self.assertGreater(len(message), 0, "Message content should not be empty.")

if __name__ == "__main__":
    unittest.main()
