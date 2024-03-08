
import unittest
from intelli.controller.remote_embed_model import RemoteEmbedModel
from intelli.model.input.embed_input import EmbedInput
import os
from dotenv import load_dotenv
load_dotenv()

class TestRemoteEmbedModel(unittest.TestCase):
    
    # Set up API keys for different providers
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    def test_openai_embeddings(self):
        """Test retrieving embeddings from OpenAI."""
        if self.OPENAI_API_KEY is None:
            self.skipTest("OPENAI_API_KEY environment variable is not set.")
            
        provider = 'openai'
        model = RemoteEmbedModel(self.OPENAI_API_KEY, provider)
        embed_input = EmbedInput(["This is a test sentence for embeddings."])
        embed_input.set_default_values(provider)
        
        result = model.get_embeddings(embed_input)
        self.assertIn('data', result, "OpenAI response should contain 'data' field")

    def test_gemini_embeddings(self):
        """Test retrieving embeddings from Gemini."""
        if self.GEMINI_API_KEY is None:
            self.skipTest("GEMINI_API_KEY environment variable is not set.")
            
        provider = 'gemini'
        model = RemoteEmbedModel( self.GEMINI_API_KEY, provider)
        embed_input = EmbedInput(["Explore Gemini's API for embeddings."], "models/embedding-001")
        
        result = model.get_embeddings(embed_input)
        self.assertIsInstance(result['values'], list, "Gemini response should be a list of embeddings")
    
    def test_mistral_embeddings(self):
        """Test retrieving embeddings from Mistral."""
        if self.MISTRAL_API_KEY is None:
            self.skipTest("MISTRAL_API_KEY environment variable is not set.")
            
        provider = 'mistral'
        model = RemoteEmbedModel(self.MISTRAL_API_KEY, provider)
        embed_input = EmbedInput(["Mistral provides interesting insights."])
        embed_input.set_default_values(provider)
        
        result = model.get_embeddings(embed_input)
        # Assuming a similar response format for simplicity; adjust according to actual API
        self.assertIn('data', result, "Mistral response should contain 'data' field")

if __name__ == '__main__':
    unittest.main()