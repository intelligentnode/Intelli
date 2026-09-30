import json
import unittest
from unittest.mock import patch

from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput


class FakeOpenAIWrapper:
    """Stands in for OpenAIWrapper and records which endpoint each call hit."""

    def __init__(self, api_key, proxy_helper=None, timeout=180):
        self.chat_requests = []
        self.responses_requests = []

    def generate_chat_text(self, params, functions=None, function_call=None):
        self.chat_requests.append(dict(params))
        if params.get("stream"):
            chunks = [{"choices": [{"delta": {"content": part}}]} for part in ("Hel", "lo")]
            return iter([f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"])
        return {"choices": [{"message": {"content": "chat completions reply"}}]}

    def generate_gpt5_response(self, params):
        self.responses_requests.append(dict(params))
        return {"output": [{"type": "message",
                            "content": [{"type": "output_text", "text": "responses reply"}]}]}


class TestChatbotOpenAIRouting(unittest.TestCase):
    def setUp(self):
        patcher = patch("intelli.function.chatbot.OpenAIWrapper", FakeOpenAIWrapper)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chatbot = Chatbot("fake-key", "openai")
        self.wrapper = self.chatbot.wrapper

    def _input(self, model):
        chat_input = ChatModelInput("You are a helpful assistant.", model=model)
        chat_input.add_user_message("hi")
        return chat_input

    def test_gpt5_uses_responses_api(self):
        result = self.chatbot.chat(self._input("gpt-5.5"))

        self.assertEqual(result, ["responses reply"])
        self.assertEqual(self.wrapper.chat_requests, [])
        params = self.wrapper.responses_requests[0]
        self.assertEqual(params["model"], "gpt-5.5")
        self.assertIn("input", params)
        self.assertNotIn("messages", params)

    def test_chat_suffix_uses_chat_completions(self):
        for model in ("gpt-5.5:chat", "gpt-5.5#chat", "gpt-5.5|chat"):
            with self.subTest(model=model):
                self.wrapper.chat_requests.clear()
                result = self.chatbot.chat(self._input(model))

                self.assertEqual(result, ["chat completions reply"])
                self.assertEqual(self.wrapper.responses_requests, [])
                params = self.wrapper.chat_requests[0]
                # The override is stripped before the model id reaches the API.
                self.assertEqual(params["model"], "gpt-5.5")
                self.assertIn("messages", params)
                self.assertNotIn("input", params)

    def test_chat_suffix_stream_uses_chat_completions(self):
        chunks = list(self.chatbot.stream(self._input("gpt-5.5:chat")))

        self.assertEqual("".join(chunks), "Hello")
        self.assertEqual(self.wrapper.responses_requests, [])
        params = self.wrapper.chat_requests[0]
        self.assertEqual(params["model"], "gpt-5.5")
        self.assertTrue(params["stream"])
        self.assertIn("messages", params)

    def test_pre_gpt5_model_uses_chat_completions(self):
        result = self.chatbot.chat(self._input("gpt-4.1"))

        self.assertEqual(result, ["chat completions reply"])
        self.assertEqual(self.wrapper.responses_requests, [])
        self.assertEqual(self.wrapper.chat_requests[0]["model"], "gpt-4.1")

    def test_direct_call_without_route_keeps_model_based_routing(self):
        # Backward compatibility for callers that pass built params directly.
        self.chatbot._chat_openai({"model": "gpt-5.5", "input": "hi"})
        self.chatbot._chat_openai({"model": "gpt-4.1", "messages": []})

        self.assertEqual(self.wrapper.responses_requests[0]["model"], "gpt-5.5")
        self.assertEqual(self.wrapper.chat_requests[0]["model"], "gpt-4.1")

    def test_gpt5_stream_is_not_supported(self):
        with self.assertRaises(NotImplementedError):
            list(self.chatbot.stream(self._input("gpt-5.5")))
        self.assertEqual(self.wrapper.chat_requests, [])
        self.assertEqual(self.wrapper.responses_requests, [])


if __name__ == "__main__":
    unittest.main()
