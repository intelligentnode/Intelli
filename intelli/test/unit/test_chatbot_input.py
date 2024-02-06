import unittest
from intelli.model.input.chatbot_input import ChatModelInput

class TestChatModelInput(unittest.TestCase):
    def setUp(self):
        self.system_message = "Let's start a conversation."
        

    def test_add_and_delete_messages(self):
        chat_model_input = ChatModelInput(system=self.system_message, model="test-model")
        chat_model_input.add_user_message("Hello, World!")
        chat_model_input.add_assistant_message("Hi, Universe!")
        self.assertEqual(len(chat_model_input.messages), 2)

        chat_model_input.delete_last_message(chat_model_input.messages[0])
        self.assertEqual(len(chat_model_input.messages), 1)

        chat_model_input.clean_messages()
        self.assertEqual(len(chat_model_input.messages), 0)

    def test_get_openai_input(self):
        chat_model_input = ChatModelInput(system=self.system_message, model="test-model")
        chat_model_input.add_system_message("System message for OpenAI example")
        params = chat_model_input.get_openai_input()
        self.assertIn('model', params)
        self.assertEqual(params['model'], "test-model")
        self.assertTrue('messages' in params)

    def test_get_mistral_input(self):
        chat_model_input = ChatModelInput(system=self.system_message, model="test-model")
        chat_model_input.add_user_message("User message for Mistral example")
        params = chat_model_input.get_mistral_input()
        self.assertIn('model', params)
        self.assertEqual(params['model'], "test-model")
        self.assertTrue('messages' in params)

    def test_get_gemini_input(self):
        chat_model_input = ChatModelInput(system=self.system_message, model="test-model")
        chat_model_input.add_assistant_message("Assistant message for Gemini example")
        params = chat_model_input.get_gemini_input()
        self.assertTrue('contents' in params)
        self.assertTrue('generationConfig' in params)

if __name__ == '__main__':
    unittest.main()