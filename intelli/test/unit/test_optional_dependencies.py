import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def run_without_numpy(code):
    """Run the code in a new process where numpy can not be imported."""
    script = "import sys\nsys.modules['numpy'] = None\n" + code
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )


class TestOptionalDependencies(unittest.TestCase):
    def test_basic_use_without_numpy(self):
        result = run_without_numpy(
            "from intelli.function.chatbot import Chatbot, ChatProvider\n"
            "from intelli.model.input.chatbot_input import ChatModelInput\n"
            "from intelli.flow import Agent, Task, SequenceFlow, TextTaskInput, TextProcessor\n"
            "Chatbot('key', ChatProvider.OPENAI)\n"
            "Agent(agent_type='text', provider='openai', mission='test', model_params={'key': 'key'})\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_audio_without_numpy_shows_install_command(self):
        result = run_without_numpy(
            "from intelli.model.input.text_recognition_input import SpeechRecognitionInput\n"
            "SpeechRecognitionInput(audio_data=b'audio').get_audio_data()\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Warning: numpy is required", result.stdout)
        self.assertIn("pip install intelli[offline]", result.stderr)

    def test_llamacpp_embeddings_without_numpy_shows_install_command(self):
        result = run_without_numpy(
            "from intelli.wrappers.llama_cpp_wrapper import IntelliLlamaCPPWrapper\n"
            "wrapper = object.__new__(IntelliLlamaCPPWrapper)\n"
            "raw = {'object': 'list', 'data': [{'embedding': [[1.0, 2.0], [3.0, 4.0]]}]}\n"
            "wrapper._process_embedding_output(raw)\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Warning: numpy is required", result.stdout)
        self.assertIn("pip install intelli[llamacpp]", result.stderr)

    def test_llamacpp_embeddings_average_with_numpy(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed")
        from intelli.wrappers.llama_cpp_wrapper import IntelliLlamaCPPWrapper

        wrapper = object.__new__(IntelliLlamaCPPWrapper)
        raw = {"object": "list", "data": [{"embedding": [[1.0, 2.0], [3.0, 4.0]]}]}
        self.assertEqual(wrapper._process_embedding_output(raw), {"embedding": [2.0, 3.0]})


if __name__ == "__main__":
    unittest.main()
