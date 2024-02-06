import unittest
import os
import asyncio
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from dotenv import load_dotenv
load_dotenv()

class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
        # Creating Chatbot instances for each AI model
        self.openai_bot = Chatbot(self.openai_api_key, "openai")
        self.gemini_bot = Chatbot(self.gemini_api_key, "gemini")
        self.mistral_bot = Chatbot(self.mistral_api_key, "mistral")
    
    def test_openai_chat(self):
        print('---- start openai ----')
        input = ChatModelInput("You are a helpful assistant.", "gpt-3.5-turbo")
        input.add_user_message("What is the capital of France?")
        
        response = self.openai_bot.chat(input)

        print('openai response: ', response)
        
        self.assertTrue(len(response) > 0, "OpenAI chat response should not be empty")

    def test_gemini_chat(self):
        print('---- start gemini ----')
        input = ChatModelInput("You are a helpful assistant.", "gemini-model")
        input.add_user_message("Describe a starry night.")
        
        response = self.gemini_bot.chat(input)
        
        print('gemini response: ', response)

        self.assertTrue(len(response) > 0, "Gemini chat response should not be empty")


    def test_mistral_chat(self):
        print('---- start mistral ----')
        input = ChatModelInput("You are a helpful assistant.", "mistral-tiny")
        input.add_user_message("Who is Leonardo da Vinci?")
        
        response = self.mistral_bot.chat(input)

        print('mistral response: ', response)
        
        self.assertTrue(len(response) > 0, "Mistral chat response should not be empty")
    
    def test_openai_stream(self):
        print('---- start openai stream ----')
        input = ChatModelInput("You are a helpful assistant.", "gpt-3.5-turbo")
        input.add_user_message("Tell me a story about a lion in the savanna.")

        # Use asyncio.run() to get the result of the coroutine
        full_text = asyncio.run(self._get_openai_stream(input))

        print('openai stream response: ', full_text)

        self.assertTrue(len(full_text) > 0, "OpenAI stream response should not be empty")

    async def _get_openai_stream(self, chat_input):
        
        full_text = ''
        
        for content in self.openai_bot.stream(chat_input):
            full_text += content
            print('content item: ', content) 
        
        return full_text
    
if __name__ == '__main__':
    unittest.main()