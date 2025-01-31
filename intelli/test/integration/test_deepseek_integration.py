import os
import unittest
from huggingface_hub import hf_hub_download
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

class TestDeepSeekIntegration(unittest.TestCase):
    def setUp(self):
        # Define a local directory to hold the downloaded DeepSeek files.
        self.model_dir = "./deepseek_model"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Ensure the config subdirectory exists.
        config_dir = os.path.join(self.model_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Define local file paths.
        self.config_path = os.path.join(config_dir, "config_671B.json")
        self.model_file = os.path.join(self.model_dir, "model0-mp1.safetensors")
        
        # Download the config file from Hugging Face if not present.
        if not os.path.exists(self.config_path):
            print("Downloading config file from Hugging Face...")
            hf_hub_download(
                repo_id="deepseek-ai/DeepSeek-R1", 
                filename="configs/config_671B.json", 
                local_dir=config_dir, 
                local_dir_use_symlinks=False
            )
        
        # Download the model file if not present.
        if not os.path.exists(self.model_file):
            print("Downloading model file from Hugging Face (this may take a while)...")
            hf_hub_download(
                repo_id="deepseek-ai/DeepSeek-R1", 
                filename="model0-mp1.safetensors", 
                local_dir=self.model_dir, 
                local_dir_use_symlinks=False
            )
        
        # Initialize the DeepSeek wrapper.
        self.wrapper = DeepSeekWrapper(
            model_path=self.model_dir,
            config_path=self.config_path,
            temperature=0.7,
            max_new_tokens=20
        )

    def test_basic_generate(self):
        prompt = "Hello from Intelli. How are you?"
        output = self.wrapper.generate(prompt)
        print("DeepSeek output:", output)
        self.assertTrue(len(output) > 0, "Generated output should not be empty.")

if __name__ == "__main__":
    unittest.main()
