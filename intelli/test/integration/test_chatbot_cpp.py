import unittest
import os
import shutil
import asyncio
import sys
import contextlib

from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


@contextlib.contextmanager
def suppress_stderr():
    """
    A context manager that temporarily redirects stderr to /dev/null
    (sync usage only).
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
        os.close(devnull)


class TestChatbotLlamaCPP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Download two models (TinyLlama & DeepSeek-Qwen) for llama.cpp usage.
        If a download fails, the entire test run fails (no skipping).
        """
        cls.temp_dir = os.path.join("temp", "llamacpp_tests_2models")
        os.makedirs(cls.temp_dir, exist_ok=True)

        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is not installed. Use 'pip install intelli[llamacpp]'."
            )

        # We'll test 2 models
        cls.models = {
            "tiny_llama": {
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            },
            "deepseek_qwen": {
                "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
                "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf",
            },
        }
        cls.model_paths = {}

        for key, info in cls.models.items():
            repo_id = info["repo_id"]
            fn = info["filename"]
            print(f"Downloading {fn} from {repo_id} to {cls.temp_dir}")
            try:
                path = hf_hub_download(
                    repo_id=repo_id, filename=fn, local_dir=cls.temp_dir
                )
                cls.model_paths[key] = path
                print(f"[{key}] Model downloaded: {path}")
            except Exception as e:
                raise RuntimeError(f"Failed downloading {repo_id}/{fn}: {e}")

    @classmethod
    def tearDownClass(cls):
        """Remove temp directory."""
        if os.path.exists(cls.temp_dir):
            try:
                shutil.rmtree(cls.temp_dir)
            except OSError as err:
                print(f"Error removing temp dir {cls.temp_dir}: {err}")

    def _build_chatbot(self, model_path: str):
        """
        Helper: create a Chatbot with minimal logs (stderr suppressed).
        """
        options = {
            "model_path": model_path,
            "model_params": {"n_ctx": 512, "embedding": False, "verbose": False},
        }
        with suppress_stderr():
            chatbot = Chatbot(
                api_key=None, provider=ChatProvider.LLAMACPP.value, options=options
            )
        return chatbot

    # -----------
    # TinyLlama
    # -----------
    def test_tinyllama_chat_normal(self):
        model_path = self.model_paths["tiny_llama"]
        chatbot = self._build_chatbot(model_path)

        chat_input = ChatModelInput(
            "You are a helpful assistant.",
            model="llamacpp",
            max_tokens=64,
            temperature=0.7,
        )
        chat_input.add_user_message("What is the capital of France?")

        with suppress_stderr():
            response = chatbot.chat(chat_input)

        output = response["result"] if isinstance(response, dict) else response
        self.assertTrue(isinstance(output, list) and len(output) > 0)
        self.assertTrue(isinstance(output[0], str) and len(output[0]) > 0)
        print("\n[TinyLlama] Normal chat output:", output[0])

    def test_tinyllama_chat_streaming(self):
        model_path = self.model_paths["tiny_llama"]
        chatbot = self._build_chatbot(model_path)

        chat_input = ChatModelInput(
            "You are a helpful assistant.",
            model="llamacpp",
            max_tokens=64,
            temperature=0.7,
        )
        chat_input.add_user_message("Tell me a short joke.")

        # We'll capture the streaming output in an async function
        # but we do the stderr suppression outside the async call
        with suppress_stderr():
            stream_output = asyncio.run(self._gather_stream_output(chatbot, chat_input))

        self.assertTrue(isinstance(stream_output, str) and len(stream_output) > 0)
        print("\n[TinyLlama] Streaming chat output:", stream_output)

    # -----------
    # DeepSeek-Qwen
    # -----------
    def test_deepseek_chat_normal(self):
        model_path = self.model_paths["deepseek_qwen"]
        chatbot = self._build_chatbot(model_path)

        chat_input = ChatModelInput(
            "You are a helpful assistant.",
            model="llamacpp",
            max_tokens=64,
            temperature=0.7,
        )
        chat_input.add_user_message(
            "Explain the difference between lists and tuples in Python."
        )

        with suppress_stderr():
            response = chatbot.chat(chat_input)

        output = response["result"] if isinstance(response, dict) else response
        self.assertTrue(isinstance(output, list) and len(output) > 0)
        self.assertTrue(isinstance(output[0], str) and len(output[0]) > 0)
        print("\n[DeepSeek-Qwen] Normal chat output:", output[0])

    def test_deepseek_chat_streaming(self):
        model_path = self.model_paths["deepseek_qwen"]
        chatbot = self._build_chatbot(model_path)

        chat_input = ChatModelInput(
            "You are a helpful assistant.",
            model="llamacpp",
            max_tokens=64,
            temperature=0.7,
        )
        chat_input.add_user_message("Give me a creative greeting in Spanish.")

        # Again, do the stderr suppression around the call
        with suppress_stderr():
            stream_output = asyncio.run(self._gather_stream_output(chatbot, chat_input))

        self.assertTrue(isinstance(stream_output, str) and len(stream_output) > 0)
        print("\n[DeepSeek-Qwen] Streaming chat output:", stream_output)

    # -----------
    # Utilities
    # -----------
    async def _gather_stream_output(
        self, chatbot: Chatbot, chat_input: ChatModelInput
    ) -> str:
        """
        Collect the streaming output in a single string.
        No 'async with' for the sync context manager, so we do it outside.
        """
        output = ""
        for chunk in chatbot.stream(chat_input):
            output += chunk
        return output


if __name__ == "__main__":
    unittest.main()
