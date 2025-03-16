import unittest
from intelli.controller.remote_embed_model import RemoteEmbedModel
from intelli.model.input.embed_input import EmbedInput
import os
from dotenv import load_dotenv

load_dotenv()


class TestRemoteEmbedModel(unittest.TestCase):

    # Set up API keys for different providers
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    def test_openai_embeddings(self):
        """Test retrieving embeddings from OpenAI."""
        if self.OPENAI_API_KEY is None:
            self.skipTest("OPENAI_API_KEY environment variable is not set.")

        provider = "openai"
        model = RemoteEmbedModel(self.OPENAI_API_KEY, provider)
        embed_input = EmbedInput(["This is a test sentence for embeddings."])
        embed_input.set_default_values(provider)

        result = model.get_embeddings(embed_input)
        self.assertIn("data", result, "OpenAI response should contain 'data' field")

    def test_mistral_embeddings(self):
        """Test retrieving embeddings from Mistral."""
        if self.MISTRAL_API_KEY is None:
            self.skipTest("MISTRAL_API_KEY environment variable is not set.")

        provider = "mistral"
        model = RemoteEmbedModel(self.MISTRAL_API_KEY, provider)
        embed_input = EmbedInput(["Mistral provides interesting insights."])
        embed_input.set_default_values(provider)

        result = model.get_embeddings(embed_input)
        # Assuming a similar response format for simplicity; adjust according to actual API
        self.assertIn("data", result, "Mistral response should contain 'data' field")

    def test_gemini_embeddings(self):
        """Test retrieving embeddings from Gemini."""
        if self.GEMINI_API_KEY is None:
            self.skipTest("GEMINI_API_KEY environment variable is not set.")

        provider = "gemini"
        model = RemoteEmbedModel(self.GEMINI_API_KEY, provider)
        embed_input = EmbedInput(
            ["Explore Gemini's API for embeddings."], "models/embedding-001"
        )

        result = model.get_embeddings(embed_input)
        self.assertIsInstance(
            result["values"], list, "Gemini response should be a list of embeddings"
        )

    def test_vllm_embeddings(self):
        """Test retrieving embeddings from vLLM."""
        vllm_embed_url = os.getenv("VLLM_EMBED_URL")
        if not vllm_embed_url:
            self.skipTest("VLLM_EMBED_URL environment variable is not set.")

        provider = "vllm"
        model = RemoteEmbedModel(
            api_key=None, provider_name=provider, options={"baseUrl": vllm_embed_url}
        )

        # Add debug prints to understand what's being passed
        test_sentence = "This is a test sentence for vLLM embeddings."
        embed_input = EmbedInput(
            [test_sentence],
            model="BAAI/bge-small-en-v1.5",
        )

        print(f"Testing vLLM embeddings with URL: {vllm_embed_url}")
        print(f"Using model: {embed_input.model}")
        print(f"Input text: {embed_input.texts}")

        # Print the actual request that will be sent
        vllm_request = embed_input.get_vllm_inputs()
        print(f"vLLM request params: {vllm_request}")

        # Get embeddings
        result = model.get_embeddings(embed_input)
        print(f"Embedding result structure: {list(result.keys())}")

        # Print sample of embeddings
        if "embeddings" in result and len(result["embeddings"]) > 0:
            print(f"First few dimensions of embedding: {result['embeddings'][0][:5]}")
            print(f"Embedding dimensions: {len(result['embeddings'][0])}")

        self.assertIn(
            "embeddings", result, "vLLM response should contain 'embeddings' field"
        )
        self.assertTrue(
            len(result["embeddings"]) > 0, "Should return at least one embedding"
        )
        self.assertTrue(
            len(result["embeddings"][0]) > 0, "Embedding should have dimensions"
        )


if __name__ == "__main__":
    unittest.main()
