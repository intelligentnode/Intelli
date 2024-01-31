import unittest
import requests_mock
import os
from dotenv import load_dotenv
from wrappers.mistral_ai_wrapper import MistralAIWrapper

load_dotenv()

class TestMistralAIWrapper(unittest.TestCase):
    @requests_mock.Mocker()
    def test_generate_text(self, m):
        api_key = os.getenv("MISTRAL_API_KEY")
        mistral = MistralAIWrapper(api_key)
        
        mock_response = {
            "choices": [
                {"message": {"content": "Vincent van Gogh"}}
            ]
        }
        m.post("https://api.mistral.ai/v1/chat/completions", json=mock_response)
        
        params = {
            "model": "mistral-tiny",
            "messages": [{"role": "user", "content": "Who is the most renowned French painter?"}]
        }
        result = mistral.generate_text(params)
        self.assertIn("Vincent van Gogh", result["choices"][0]["message"]["content"], "Mistral Generate Model failed to return expected content.")
        
    @requests_mock.Mocker()
    def test_get_embeddings(self, m):
        api_key = os.getenv("MISTRAL_API_KEY")
        mistral = MistralAIWrapper(api_key)
        
        mock_response = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]}
            ]
        }
        m.post("https://api.mistral.ai/v1/embeddings", json=mock_response)
        
        params = {
            "model": "mistral-embed",
            "input": ["Embed this sentence.", "As well as this one."]
        }
        result = mistral.get_embeddings(params)
        self.assertGreater(len(result["data"]), 0, "Mistral Embeddings response length should be greater than 0")
        self.assertIn(0.1, result["data"][0]["embedding"], "Mistral Embeddings failed to return the expected embedding.")

if __name__ == "__main__":
    unittest.main()