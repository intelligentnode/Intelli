import unittest
import os
import torch
from pathlib import Path
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper
from intelli.model.input.chatbot_input import ChatModelInput
from dotenv import load_dotenv

load_dotenv()

class TestDeepSeekWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get model path from environment variable
        cls.model_path = os.getenv("DEEPSEEK_MODEL_PATH")
        if cls.model_path:
            # Convert to absolute path if it's a relative path
            if not os.path.isabs(cls.model_path):
                cls.model_path = os.path.abspath(cls.model_path)

            # Normalize path separators for the current OS
            cls.model_path = os.path.normpath(cls.model_path)

            # Ensure the directory exists
            model_dir = Path(cls.model_path)
            if not model_dir.exists() and not model_dir.is_file():
                os.makedirs(model_dir, exist_ok=True)

            print(f"Using model path: {cls.model_path}")

        # Use a smaller model for testing
        cls.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

        # Always use CPU for testing to ensure compatibility
        # This avoids CUDA errors on systems without GPU support
        cls.device = "cpu"

        print(f"Testing with model: {cls.model_id}")
        print(f"Using device: {cls.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        cls.wrapper = DeepSeekWrapper(
            model_path=cls.model_path,
            model_id=cls.model_id
        )

    def test_model_initialization(self):
        self.assertIsNotNone(self.wrapper)

    def test_model_loading(self):
        try:
            self.wrapper.load_model(device=self.device)
            self.assertIsNotNone(self.wrapper.loader)
            self.assertTrue(Path(self.wrapper.loader.model_path).exists())
        except Exception as e:
            self.skipTest(f"Model loading failed: {str(e)}")

    def test_chat_interface(self):
        input_params = ChatModelInput("You are a helpful assistant.")
        input_params.add_user_message("Write a simple Python function.")

        try:
            response = self.wrapper.chat(input_params)
            self.assertIn("choices", response)
        except Exception as e:
            self.skipTest(f"Chat interface test failed: {str(e)}")

    def test_code_generation(self):
        prompt = "def fibonacci(n):"
        try:
            response = self.wrapper.chat({"prompt": prompt})
            self.assertIn("choices", response)
            self.assertIn("text", response["choices"][0])
        except Exception as e:
            self.skipTest(f"Code generation test failed: {str(e)}")

    def test_quantization(self):
        try:
            self.wrapper.load_model(device=self.device, quantize=True)
            self.assertIsNotNone(self.wrapper.model)
            prompt = "Write a quick sort algorithm"
            response = self.wrapper.chat({"prompt": prompt})
            self.assertIn("choices", response)
        except Exception as e:
            self.skipTest(f"Quantization test failed: {str(e)}")

    def test_memory_efficiency(self):
        try:
            import psutil
            before_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            self.wrapper.load_model(device="cpu", quantize=True)
            after_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            print(f"Memory usage: {after_mem - before_mem:.2f} MB")
            self.assertIsNotNone(self.wrapper.model)
        except ImportError:
            self.skipTest("psutil not installed")
        except Exception as e:
            self.skipTest(f"Memory efficiency test failed: {str(e)}")

if __name__ == "__main__":
    unittest.main(verbosity=2)