import os
import unittest
from huggingface_hub import hf_hub_download
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

class TestDeepSeekIntegration8B(unittest.TestCase):
    def setUp(self):
        # Install the DeepSeek 8B model files into "./temp/deepseek_model_8b"
        self.model_dir = os.path.join(os.getcwd(), "temp", "deepseek_model_8b")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Download the config file from the DeepSeek-R1-Distill-Llama-8B repo.
        self.config_filename = "config.json"
        self.config_path = os.path.join(self.model_dir, self.config_filename)
        if not os.path.exists(self.config_path):
            print("Downloading config.json from Hugging Face for 8B model...")
            hf_hub_download(
                repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                filename=self.config_filename,
                local_dir=self.model_dir
            )
        
        # Download one weight file shard.
        # For single-GPU inference we choose one shard – here "model-00001-of-000002.safetensors".
        self.model_filename = "model-00001-of-000002.safetensors"
        self.model_file_path = os.path.join(self.model_dir, self.model_filename)
        if not os.path.exists(self.model_file_path):
            print("Downloading model weight file from Hugging Face for 8B model (this may take a while)...")
            hf_hub_download(
                repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                filename=self.model_filename,
                local_dir=self.model_dir
            )
        
        # Initialize the DeepSeek wrapper with the downloaded files.
        # (Your wrapper will load the config and then look for the weight file using the expected name.)
        self.wrapper = DeepSeekWrapper(
            model_path=self.model_dir,
            config_path=self.config_path,
            temperature=0.7,
            max_new_tokens=20
        )
    
    def test_basic_generate(self):
        prompt = "Hello from Intelli with the 8B model. How are you?"
        output = self.wrapper.generate(prompt)
        print("DeepSeek 8B output:", output)
        self.assertTrue(len(output) > 0, "Generated output should not be empty.")

if __name__ == "__main__":
    unittest.main()
