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
            "model": "claude-opus-5",
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
        self.assertTrue('content' in result and isinstance(result['content'], list) and len(result['content']) > 0,
                        "The API response should include 'content' and it should be a non-empty list.")
        # Claude 5 models can return a thinking block before the answer, so pick
        # the first text block instead of assuming content[0] is text.
        text_blocks = [block for block in result['content'] if block.get('type') == 'text']
        self.assertTrue(text_blocks, "The API response content should include a text block.")
        print(f"generate text result: {text_blocks[0]['text']}")

    def test_stream_text_integration(self):
        """Integration test for stream_text method."""
        params = {
            "model": "claude-sonnet-5",
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
