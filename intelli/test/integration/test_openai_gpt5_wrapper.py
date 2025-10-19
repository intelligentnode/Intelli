import unittest
from intelli.wrappers.openai_wrapper import OpenAIWrapper
import os
from dotenv import load_dotenv

load_dotenv()


class TestOpenAIGPT5Wrapper(unittest.TestCase):

    def setUp(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.openai = OpenAIWrapper(self.api_key)
    
    def test_generate_gpt5_response_minimal(self):
        """Test GPT-5 with minimal reasoning effort"""
        params = {
            "model": "gpt-5",
            "input": "Write a haiku about code.",
            "reasoning": {"effort": "minimal"}
        }
        
        result = self.openai.generate_gpt5_response(params)
        print('GPT-5 Response (minimal):\n', result, '\n')
        
        # GPT-5 returns output array
        self.assertTrue("output" in result)
        self.assertTrue(isinstance(result['output'], list))
        self.assertTrue(len(result['output']) > 0)
        
        # Check for message type in output
        has_message = any(item.get('type') == 'message' for item in result['output'])
        self.assertTrue(has_message, "Response should contain at least one message type")
    
    def test_generate_gpt5_response_low(self):
        """Test GPT-5 with low reasoning effort"""
        params = {
            "model": "gpt-5",
            "input": "Explain what is machine learning in one sentence.",
            "reasoning": {"effort": "low"}
        }
        
        result = self.openai.generate_gpt5_response(params)
        print('GPT-5 Response (low):\n', result, '\n')
        
        self.assertTrue("output" in result)
        self.assertTrue(isinstance(result['output'], list))
        
    def test_generate_gpt5_response_detailed(self):
        """Test GPT-5 with different reasoning effort"""
        params = {
            "model": "gpt-5",
            "input": "What are the benefits of using Python?",
            "reasoning": {"effort": "low"}
        }
        
        result = self.openai.generate_gpt5_response(params)
        print('GPT-5 Response (low effort):\n', result, '\n')
        
        self.assertTrue("output" in result)
        
        # Extract and verify content
        message_content = None
        for item in result['output']:
            if item.get('type') == 'message':
                message_content = item.get('content')
                break
        
        self.assertIsNotNone(message_content, "Should have message content in response")
        print('Extracted message content:', message_content[:100] if message_content else 'None')


if __name__ == "__main__":
    unittest.main()

