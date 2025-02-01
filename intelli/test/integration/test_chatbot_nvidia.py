import unittest
import os
import asyncio
from dotenv import load_dotenv
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput

load_dotenv()

class TestChatbotNvidiaChatAndStream(unittest.TestCase):
    def setUp(self):
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        assert self.nvidia_api_key, "NVIDIA_API_KEY is not set."
        self.chatbot = Chatbot(self.nvidia_api_key, ChatProvider.NVIDIA.value)

    def test_nvidia_chat_and_stream(self):

        # Test normal chat
        print("Testing Nvidia chat")
        normal_input = ChatModelInput("You are a helpful assistant.", model="deepseek-ai/deepseek-r1", max_tokens=1024, temperature=0.6)
        normal_input.add_user_message("What is the capital city of france?")
        response = self.chatbot.chat(normal_input)
        if isinstance(response, dict) and "result" in response:
            normal_output = response["result"]
        else:
            normal_output = response
        self.assertTrue(len(normal_output) > 0, "Nvidia normal chat response should not be empty")
        print("Nvidia normal chat output:", normal_output)

        # Test streaming chat
        print("Testing Nvidia stream")
        stream_input = ChatModelInput("You are a helpful assistant.", model="deepseek-ai/deepseek-r1", max_tokens=1024, temperature=0.6)
        stream_input.add_user_message("What is the capital city of france?")
        stream_output = asyncio.run(self.get_stream_output(stream_input))
        self.assertTrue(len(stream_output) > 0, "Nvidia stream response should not be empty")
        print("Nvidia stream output:", stream_output)

    async def get_stream_output(self, chat_input):
        output = ""
        for chunk in self.chatbot.stream(chat_input):
            output += chunk
        return output

if __name__ == "__main__":
    unittest.main()
