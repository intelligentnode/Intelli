import unittest
from intelli.config import config
from intelli.model.input.chatbot_input import ChatModelInput

class TestChatModelInput(unittest.TestCase):
    def setUp(self):
        self.system_message = "Let's start a conversation."
        

    def test_add_and_delete_messages(self):
        chat_model_input = ChatModelInput(system=self.system_message, model="test-model")
        # The constructor auto-adds the system message, so this starts at 1.
        chat_model_input.add_user_message("Hello, World!")
        chat_model_input.add_assistant_message("Hi, Universe!")
        self.assertEqual(len(chat_model_input.messages), 3)  # system + user + assistant

        chat_model_input.delete_last_message(chat_model_input.messages[0])
        self.assertEqual(len(chat_model_input.messages), 2)

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

    def test_mistral_default_model_from_config(self):
        chat_model_input = ChatModelInput(system=self.system_message)
        chat_model_input.add_user_message("hi")
        params = chat_model_input.get_mistral_input()
        self.assertEqual(params['model'], config['url']['mistral']['models']['chat'])

    def test_anthropic_default_model_from_config(self):
        chat_model_input = ChatModelInput(system=self.system_message)
        chat_model_input.add_user_message("hi")
        params = chat_model_input.get_anthropic_input()
        self.assertEqual(params['model'], config['url']['anthropic']['models']['chat'])

    def test_anthropic_omits_sampling_params_for_models_that_reject_them(self):
        # Claude 5 family and Opus 4.7+ return HTTP 400 when temperature is sent.
        for model in ["claude-sonnet-5", "claude-opus-5", "claude-fable-5-1",
                      "claude-opus-4-7", "claude-opus-4-8"]:
            chat_model_input = ChatModelInput(system=self.system_message, model=model,
                                              temperature=0.5, top_p=0.9)
            chat_model_input.add_user_message("hi")
            params = chat_model_input.get_anthropic_input()
            self.assertNotIn('temperature', params, model)
            self.assertNotIn('top_p', params, model)

    def test_anthropic_keeps_temperature_for_older_models(self):
        # Sonnet 4.x, Opus <= 4.6, Haiku 4.5 (also dated ids) still accept temperature.
        for model in ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5",
                      "claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929",
                      "claude-haiku-4-5-20251001", "claude-3-7-sonnet-20250219"]:
            chat_model_input = ChatModelInput(system=self.system_message, model=model,
                                              temperature=0.5)
            chat_model_input.add_user_message("hi")
            params = chat_model_input.get_anthropic_input()
            self.assertEqual(params.get('temperature'), 0.5, model)

    def test_gpt5_reasoning_effort_defaults(self):
        # Plain GPT-5 models default to 'low'; the *-pro models only accept
        # medium/high/xhigh, so they default to 'medium'; explicit values win.
        cases = [("gpt-5.5", None, 'low'), ("gpt-5.5-pro", None, 'medium'),
                 ("gpt-5.5-pro", 'high', 'high'), ("gpt-5.4", 'none', 'none')]
        for model, effort, expected in cases:
            chat_model_input = ChatModelInput(system=self.system_message, model=model,
                                              reasoning_effort=effort)
            chat_model_input.add_user_message("hi")
            params = chat_model_input.get_openai_input()
            self.assertEqual(params['reasoning']['effort'], expected, model)

if __name__ == '__main__':
    unittest.main()