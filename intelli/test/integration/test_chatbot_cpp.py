import unittest
import os
import shutil
import asyncio
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput
import contextlib
import sys
import os


try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


@contextlib.contextmanager
def suppress_stderr():
    """
    A context manager that temporarily redirects stderr to /dev/null.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(sys.stderr.fileno())
    sys.stderr.flush()
    os.dup2(devnull, sys.stderr.fileno())
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(old_stderr)


class TestChatbotLlamaCPP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up a temporary directory and download a small LLaMA GGUF model for testing.
        """
        cls.temp_dir = os.path.join("temp", "llamacpp_tests")
        os.makedirs(cls.temp_dir, exist_ok=True)

        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is not installed. Use 'pip install intelli[llamacpp]'."
            )

        # Use TinyLlama model for testing (or another small model)
        cls.repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        cls.filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        try:
            print(f"Downloading {cls.filename} from {cls.repo_id} to {cls.temp_dir}")
            cls.model_path = hf_hub_download(
                repo_id=cls.repo_id, filename=cls.filename, local_dir=cls.temp_dir
            )
            print(f"Model downloaded: {cls.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed downloading TinyLlama model: {e}")

    @classmethod
    def tearDownClass(cls):
        """
        Remove the temporary directory after tests.
        """
        if os.path.exists(cls.temp_dir):
            try:
                shutil.rmtree(cls.temp_dir)
            except OSError as err:
                print(f"Error removing temp dir {cls.temp_dir}: {err}")

    def setUp(self):
        """
        Create a chatbot instance using the llama.cpp provider with minimal logs.
        """
        options = {
            "model_path": self.model_path,
            "model_params": {
                "n_ctx": 512,
                "embedding": False,
                "verbose": False,
            },  # For chat, embedding not needed.
        }

        # Suppress stderr to hide noisy llama.cpp logs.
        with suppress_stderr():
            self.chatbot = Chatbot(
                api_key=None, provider=ChatProvider.LLAMACPP.value, options=options
            )

    def test_chat_normal(self):
        """
        Test a normal chat interaction using llama.cpp.
        """
        # Create a ChatModelInput instance (assuming its constructor takes an initial system prompt).
        chat_input = ChatModelInput(
            "You are a helpful assistant.",
            model="llamacpp",
            max_tokens=64,
            temperature=0.7,
        )
        chat_input.add_user_message("What is the capital of France?")
        response = self.chatbot.chat(chat_input)
        # Response can be a dict with key 'result'
        if isinstance(response, dict) and "result" in response:
            output = response["result"]
        else:
            output = response

        self.assertTrue(
            isinstance(output, list) and len(output) > 0,
            "Chat response should be a non-empty list.",
        )
        self.assertTrue(
            isinstance(output[0], str) and len(output[0]) > 0,
            "Response text should be non-empty.",
        )
        print("Chat response:", output[0])
        print("---")

    def test_chat_streaming(self):
        """
        Test streaming chat using llama.cpp.
        """
        chat_input = ChatModelInput(
            "You are a helpful assistant.",
            model="llamacpp",
            max_tokens=64,
            temperature=0.7,
        )
        chat_input.add_user_message("Tell me a joke.")
        stream_output = asyncio.run(self._get_stream_output(chat_input))
        self.assertTrue(
            isinstance(stream_output, str) and len(stream_output) > 0,
            "Streaming response should be non-empty.",
        )
        print("Streaming chat response:", stream_output)
        print("---")

    async def _get_stream_output(self, chat_input):
        output = ""
        for chunk in self.chatbot.stream(chat_input):
            output += chunk
        return output


if __name__ == "__main__":
    unittest.main()
