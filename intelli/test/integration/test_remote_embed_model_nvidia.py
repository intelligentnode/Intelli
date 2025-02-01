import unittest
import os
from dotenv import load_dotenv
from intelli.model.input.embed_input import EmbedInput
from intelli.controller.remote_embed_model import RemoteEmbedModel

load_dotenv()

class TestRemoteEmbedModelNvidia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("NVIDIA_API_KEY")
        assert cls.api_key, "NVIDIA_API_KEY is not set."
        cls.embed_model = RemoteEmbedModel(cls.api_key, "nvidia")

    def test_get_embeddings(self):
        text = "What is the capital of France?"
        embed_input = EmbedInput([text], model="nvidia/llama-3.2-nv-embedqa-1b-v2")
        result = self.embed_model.get_embeddings(embed_input)
        self.assertIn("data", result)
        self.assertGreater(len(result["data"]), 0)
        self.assertIn("embedding", result["data"][0])
        embedding = result["data"][0]["embedding"]
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        print("Nvidia embedding sample:", embedding[:5])

if __name__ == "__main__":
    unittest.main()
