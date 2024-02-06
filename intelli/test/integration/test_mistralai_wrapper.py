import unittest
import os
from intelli.wrappers.mistralai_wrapper import MistralAIWrapper
from dotenv import load_dotenv
load_dotenv()

class TestMistralAIWrapperIntegration(unittest.TestCase):
    def setUp(self):
        """Set up for the test case."""
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.assertIsNotNone(self.api_key, "MISTRAL_API_KEY must not be None.")
        self.mistral = MistralAIWrapper(self.api_key)

    def test_generate_text_integration(self):
        """Integration test for generate_text method."""
        
        params = {
            "model": "mistral-tiny",
            "messages": [{"role": "user", "content": "Who is the most renowned French painter?"}]
        }

        # Call the model
        result = self.mistral.generate_text(params)
        print('generate text result: ', result['choices'][0]['message']['content'])
        self.assertIn('message', result['choices'][0], "The API response doesn't match the expected format.")

    def test_get_embeddings_integration(self):
        """Integration test for get_embeddings method."""
        
        params = {
            "model": "mistral-embed",
            "input": ["Embed this sentence.", "As well as this one."]
        }

        # Call the model
        result = self.mistral.get_embeddings(params)
        print('embedding sample result: ', result['data'][0]['embedding'][:3])
        self.assertTrue('data' in result and len(result['data']) > 0, "The API response should contain embeddings data.")

if __name__ == "__main__":
    unittest.main()
