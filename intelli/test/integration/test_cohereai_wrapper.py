import unittest
import os
import json
from intelli.wrappers.cohereai_wrapper import CohereAIWrapper
from intelli.utils.cohere_stream_parser import CohereStreamParser
from dotenv import load_dotenv


load_dotenv()

class TestCohereAIWrapperIntegration(unittest.TestCase):
    def setUp(self):
        """Set up for the test case."""
        self.api_key = os.getenv("COHERE_API_KEY")
        self.assertIsNotNone(self.api_key, "COHERE_API_KEY must not be None.")
        self.cohere = CohereAIWrapper(self.api_key)

    def test_cohere_generate_model(self):
        try:
            params = {
                'model': 'command',
                'prompt': 'Write a blog outline for a blog titled "The Art of Effective Communication"',
                'temperature': 0.7,
                'max_tokens': 200,
            }

            result = self.cohere.generate_text(params)
            print('Cohere Language Model Result:', result['generations'][0]['text'])
        except Exception as error:
            print('Cohere Language Model Error:', error)

    async def test_cohere_web_chat(self):
        try:
            params = {
                'model': 'command-nightly',
                'message': 'what is the command to install intellinode npm module ?',
                'temperature': 0.3,
                'chat_history': [],
                'prompt_truncation': 'auto',
                'stream': False,
                'citation_quality': 'accurate',
                'connectors': [{'id': 'web-search'}],
            }
            result = self.cohere.generate_chat_text(params)

            print('Cohere Chat Result:', json.dumps(result, indent=2))
        except Exception as error:
            print('Cohere Chat Error:', error)

    def test_cohere_embeddings(self):
        try:
            params = {
                'texts': ['Hello from Cohere!', 'Hallo von Cohere!', '您好，来自 Cohere！'],
                'model': 'embed-multilingual-v2.0',
                'truncate': 'END',
            }

            result = self.cohere.get_embeddings(params)
            embeddings = result['embeddings']
            print('Cohere Embeddings Result Sample:', embeddings[0][:50])
            self.assertTrue(embeddings, 0, 'test_cohere_embeddings response length should be greater than 0')
        except Exception as error:
            print('Cohere Embeddings Error:', error)

    def test_cohere_chat_stream(self):
        try:
            params = {
                'model': 'command',
                'message': 'how to use intellinode npm module ?',
                'stream': True,
                'chat_history': [],
                'prompt_truncation': 'auto',
                'citation_quality': 'accurate',
                'temperature': 0.3
            }

            response_chunks = ''
            stream_parser = CohereStreamParser()

            for chunk in self.cohere.generate_chat_text(params):
                chunk_text = chunk.decode('utf-8')
                for content_text in stream_parser.feed(chunk_text):
                    print('Result Chunk:', content_text)
                    response_chunks += content_text

            print('Concatenated Text:', response_chunks)
            self.assertTrue(response_chunks > 0,
                       'test_cohere_chat_stream response length should be greater than 0')
        except Exception as error:
            print('Cohere Chat Error:', error)

if __name__ == "__main__":
    unittest.main()
