import unittest
import os
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from dotenv import load_dotenv
load_dotenv()

class TestChatbotWithData(unittest.TestCase):
    def setUp(self):
        # Loading API keys from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        one_key = os.getenv("INTELLI_ONE_KEY")
        api_base = os.getenv("INTELLI_API_BASE")
        
        # Creating Chatbot instances for each AI model with attach_reference set to True
        self.openai_bot = Chatbot(self.openai_api_key, "openai", 
                                  {"one_key": one_key, "api_base": api_base})
        self.gemini_bot = Chatbot(self.gemini_api_key, "gemini", 
                                  {"one_key": one_key, "api_base": api_base})
        self.mistral_bot = Chatbot(self.mistral_api_key, "mistral", 
                                   {"one_key": one_key, "api_base": api_base})

    def test_openai_chat_with_data(self):
        print('---- start openai with data ----')
        input = ChatModelInput("You are a helpful assistant.", "gpt-3.5-turbo")
        input.attach_reference = True  # Explicitly attaching references
        input.add_user_message("Why is Mars called the Red Planet?")

        response = self.openai_bot.chat(input)

        print('openai response with data: ', response)
        
        # Checking the presence of response and references
        self.assertTrue('result' in response and len(response['result']) > 0, "OpenAI chat response should not be empty")
        self.assertTrue('references' in response, "References should be included in the response")

    def test_gemini_chat_with_data(self):
        print('---- start gemini with data ----')
        input = ChatModelInput("You are a helpful assistant.", "gemini-model")
        input.attach_reference = True
        input.add_user_message("Why is Mars called the Red Planet?")

        response = self.gemini_bot.chat(input)
        
        print('gemini response with data: ', response)
        
        # Gemini might not support reference attachment like OpenAI, so modify this test accordingly if needed
        self.assertTrue(len(response) > 0, "Gemini chat response should not be empty")

    def test_mistral_chat_with_data(self):
        print('---- start mistral with data ----')
        input = ChatModelInput("You are a helpful assistant.", "mistral-tiny")
        input.attach_reference = True
        input.add_user_message("Why is Mars called the Red Planet?")
        
        response = self.mistral_bot.chat(input)

        print('mistral response with data: ', response)
        
        # Like Gemini, adjust expectations based on Mistral's capabilities
        self.assertTrue(len(response) > 0, "Mistral chat response should not be empty")

if __name__ == '__main__':
    unittest.main()