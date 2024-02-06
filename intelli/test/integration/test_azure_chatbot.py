import unittest
import os
import asyncio
from intelli.function.chatbot import Chatbot
from intelli.utils.proxy_helper import ProxyHelper
from intelli.model.input.chatbot_input import ChatModelInput
from dotenv import load_dotenv
load_dotenv()

class TestChatbot(unittest.TestCase):
    def setUp(self):
        # Get azure keys
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_resource = os.getenv("AZURE_RESOURCE")
        # Initiate the proxy
        proxy_helper = ProxyHelper()
        proxy_helper.set_azure_openai(azure_resource)
        # Wrapp the proxy as parameter
        options = {
            'proxy_helper': proxy_helper
        }
        
        # Creating Chatbot instances
        self.openai_bot = Chatbot(azure_api_key, "openai", options=options)

    def test_openai_chat(self):
        print('---- start openai ----')
        input = ChatModelInput("You are a helpful assistant.", "gpt_basic")
        input.add_user_message("What is the capital of France?")
        
        response = self.openai_bot.chat(input)

        print('openai response: ', response)
        
        self.assertTrue(len(response) > 0, "OpenAI chat response should not be empty")
    
if __name__ == '__main__':
    unittest.main()