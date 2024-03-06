import unittest
import os
from intelli.wrappers.anthropic_wrapper import AnthropicWrapper
from dotenv import load_dotenv

load_dotenv()


class TestAnthropicWrapperIntegration(unittest.TestCase):
    def setUp(self):
        """Set up for the test case."""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.assertIsNotNone(self.api_key, "ANTHROPIC_API_KEY must not be None.")
        self.anthropic = AnthropicWrapper(self.api_key)

    def test_generate_text_integration(self):
        """Integration test for generate_text method."""
        params = {
            "model": "claude-3-sonnet-20240229",
            "messages": [
                {
                    "role": "user",
                    "content": "Who is the most renowned French painter? Provide a single direct short answer."
                }
            ],
            "max_tokens": 256
        }

        # Call the model
        result = self.anthropic.generate_text(params)
        print(f"generate text result: {result['content'][0]['text']}")
        self.assertTrue('content' in result and isinstance(result['content'], list) and len(result['content']) > 0,
                        "The API response should include 'content' and it should be a non-empty list.")
        self.assertIn('text', result['content'][0], "The API response content should have a 'text' field.")

    def test_stream_text_integration(self):
        """Integration test for stream_text method."""
        params = {
            "model": "claude-3-sonnet-20240229",
            "messages": [
                {
                    "role": "user",
                    "content": "Who is the American mathematician know as the father of \"information theory\"? "
                               "Provide a single direct short answer."
                }
            ],
            "max_tokens": 256
        }

        event_count = 0
        try:
            for line in self.anthropic.stream_text(params):
                print(f"Received line: {line}")
                event_count += 1
                if event_count > 10:  # Break after receiving a few events to avoid infinite loop
                    break
            self.assertGreater(event_count, 0, "Should have received at least one streaming event.")
        except Exception as error:
            self.fail(f"Streaming failed with exception: {str(error)}")


if __name__ == "__main__":
    unittest.main()
