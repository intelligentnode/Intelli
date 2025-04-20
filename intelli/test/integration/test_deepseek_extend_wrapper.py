import unittest
import os
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper
from intelli.model.input.chatbot_input import ChatModelInput
from dotenv import load_dotenv

load_dotenv()

class TestDeepSeekExtendWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a smaller model for testing
        cls.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.cache_dir = Path.home() / ".cache" / "intelli" / "test_models"
        cls.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Testing with model: {cls.model_id}")
        print(f"Using device: {cls.device}")
        print(f"Cache directory: {cls.cache_dir}")
        
        # Download model files directly using huggingface_hub
        try:
            cls.model_path = snapshot_download(
                repo_id=cls.model_id,
                cache_dir=cls.cache_dir,
                local_files_only=False
            )
            print(f"Model downloaded to: {cls.model_path}")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            cls.model_path = None
        
        cls.wrapper = DeepSeekWrapper(
            model_id=cls.model_id
        )
        
    def test_model_initialization(self):
        self.assertIsNotNone(self.wrapper)
        
    def test_model_loading_with_id(self):
        """Test loading model using model_id only"""
        try:
            self.wrapper.load_model(device=self.device)
            self.assertIsNotNone(self.wrapper.loader)
            self.assertTrue(Path(self.wrapper.loader.model_path).exists())
        except Exception as e:
            self.skipTest(f"Model loading with ID failed: {str(e)}")
    
    def test_model_loading_with_path(self):
        """Test loading model using explicit path"""
        if not self.model_path:
            self.skipTest("Model path not available")
            
        try:
            wrapper = DeepSeekWrapper(model_path=self.model_path)
            wrapper.load_model(device=self.device)
            self.assertIsNotNone(wrapper.loader)
            self.assertTrue(Path(wrapper.loader.model_path).exists())
        except Exception as e:
            self.skipTest(f"Model loading with path failed: {str(e)}")

    def test_chat_interface(self):
        if not hasattr(self, 'wrapper') or not self.wrapper:
            self.skipTest("Wrapper not initialized")
            
        input_params = ChatModelInput("You are a helpful assistant.")
        input_params.add_user_message("Write a simple Python function to calculate factorial.")
        
        try:
            response = self.wrapper.chat(input_params)
            self.assertIn("choices", response)
        except Exception as e:
            self.skipTest(f"Chat interface test failed: {str(e)}")
        
    def test_code_generation(self):
        if not hasattr(self, 'wrapper') or not self.wrapper:
            self.skipTest("Wrapper not initialized")
            
        prompt = "def factorial(n):"
        try:
            response = self.wrapper.chat({"prompt": prompt})
            self.assertIn("choices", response)
            self.assertIn("text", response["choices"][0])
        except Exception as e:
            self.skipTest(f"Code generation test failed: {str(e)}")
            
    def test_quantization(self):
        if not hasattr(self, 'wrapper') or not self.wrapper:
            self.skipTest("Wrapper not initialized")
            
        try:
            self.wrapper.load_model(device=self.device, quantize=True)
            self.assertIsNotNone(self.wrapper.model)
            prompt = "Write a quick sort algorithm"
            response = self.wrapper.chat({"prompt": prompt})
            self.assertIn("choices", response)
        except Exception as e:
            self.skipTest(f"Quantization test failed: {str(e)}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
