import os
import unittest
from huggingface_hub import hf_hub_download
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

class TestDeepSeekIntegration(unittest.TestCase):
    def setUp(self):
        # Install the DeepSeek files into "./temp/deepseek_model"
        self.model_dir = os.path.join(os.getcwd(), "temp", "deepseek_model")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Download the config file.
        # The official DeepSeek-R1 repo does not have "config_16B.json", so we use "config.json".
        self.config_filename = "config.json"
        self.config_path = os.path.join(self.model_dir, self.config_filename)
        if not os.path.exists(self.config_path):
            print("Downloading config.json from Hugging Face...")
            hf_hub_download(
                repo_id="deepseek-ai/DeepSeek-R1",
                filename=self.config_filename,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
            )
        
        # Download one model weight file.
        # For single-GPU inference we choose one shard – here "model-00001-of-000163.safetensors".
        self.model_filename = "model-00001-of-000163.safetensors"
        self.model_file_path = os.path.join(self.model_dir, self.model_filename)
        if not os.path.exists(self.model_file_path):
            print("Downloading model weight file from Hugging Face (this may take a while)...")
            hf_hub_download(
                repo_id="deepseek-ai/DeepSeek-R1",
                filename=self.model_filename,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
            )
        
        # Initialize the DeepSeek wrapper with the downloaded files.
        # (Your wrapper's _load_model() should translate the downloaded config.json keys
        # into the keys expected by ModelArgs.)
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
