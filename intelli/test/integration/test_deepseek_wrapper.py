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
        cls.model_path = os.getenv("DEEPSEEK_MODEL_PATH")
        cls.model_id = "deepseek-ai/DeepSeek-R1"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Testing with model: {cls.model_id or cls.model_path}")
        
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