import unittest
import os
import sys
from dotenv import load_dotenv
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput

load_dotenv()


class TestChatbotVLLM(unittest.TestCase):
    """Tests for the VLLM provider in Chatbot."""

    def setUp(self):
        """Set up the test environment."""

        self.deepseek_url = os.getenv("DEEPSEEK_VLLM_URL")

        if not self.deepseek_url:
            self.skipTest("DEEPSEEK_VLLM_URL environment variable not set")

        # Initialize the chatbot with VLLM provider
        self.chatbot = Chatbot(
            api_key=None,
            provider=ChatProvider.VLLM,
            options={
                "baseUrl": self.deepseek_url,
                "debug": True
            }
        )

    def test_vllm_chat(self):
        print("\nTesting VLLM regular chat completion:")

        # Create chat input
        chat_input = ChatModelInput(
            system="You are a helpful assistant.",
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            max_tokens=150,
            temperature=0.7
        )

        chat_input.add_user_message("What is machine learning?")
        response = self.chatbot.chat(chat_input)

        # Handle response format (could be dict with "result" or directly the text)
        if isinstance(response, dict) and "result" in response:
            chat_output = response["result"]
        else:
            chat_output = response

        # Print and verify output
        print(f"VLLM chat output: {chat_output}")
        self.assertTrue(len(chat_output) > 0, "VLLM chat response should not be empty")
        self.assertTrue(isinstance(chat_output, list), "VLLM chat response should be a list")
        self.assertTrue(len(chat_output[0]) > 0, "VLLM chat response content should not be empty")

    def test_vllm_text_chat(self):

        # Create chat input without system message
        chat_input = ChatModelInput(
            system="",  # Empty system message to force text completion
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            max_tokens=150,
            temperature=0.7
        )
        chat_input.add_user_message("What is machine learning?")
        response = self.chatbot.chat(chat_input)

        # Handle response format
        if isinstance(response, dict) and "result" in response:
            chat_output = response["result"]
        else:
            chat_output = response

        # Print and verify output
        print(f"VLLM text completion output: {chat_output}")
        self.assertTrue(len(chat_output) > 0, "VLLM text completion response should not be empty")
        self.assertTrue(isinstance(chat_output, list), "VLLM text completion response should be a list")
        self.assertTrue(len(chat_output[0]) > 0, "VLLM text completion content should not be empty")

    def test_vllm_stream(self):
        """Test streaming with VLLM."""
        print("\nTesting VLLM streaming:")

        # Create chat input
        stream_input = ChatModelInput(
            system="You are a helpful assistant.",
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            max_tokens=150,
            temperature=0.7
        )
        stream_input.add_user_message("What is machine learning?")

        # Collect streaming output
        stream_output = ""
        chunks_received = 0

        for chunk in self.chatbot.stream(stream_input):
            chunks_received += 1
            if chunks_received <= 5:  # Print first 5 chunks for debugging
                print(f"Stream chunk {chunks_received}: {chunk}")
            stream_output += chunk

        # Print summary and verify output
        print(f"Received {chunks_received} chunks total")
        print(f"VLLM streaming first 100 chars: {stream_output[:100]}...")
        print(f"Total streaming output length: {len(stream_output)}")

        self.assertTrue(chunks_received > 0, "Should receive at least one chunk")
        self.assertTrue(len(stream_output) > 0, "VLLM streaming response should not be empty")


if __name__ == "__main__":
    unittest.main()