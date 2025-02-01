import os
import unittest
from huggingface_hub import hf_hub_download
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

class TestDeepSeekIntegration8B(unittest.TestCase):
    def setUp(self):
        # Directory for DeepSeek-R1-Distill-Llama-8B files.
        self.model_dir = os.path.join(os.getcwd(), "temp", "deepseek_model_8b")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Download the config file from the DeepSeek-R1-Distill-Llama-8B repository.
        self.config_filename = "config.json"
        self.config_path = os.path.join(self.model_dir, self.config_filename)
        if not os.path.exists(self.config_path):
            print("Downloading config.json from Hugging Face for 8B model...")
            hf_hub_download(
                repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                filename=self.config_filename,
                local_dir=self.model_dir
            )
        
        # Download all weight shard files.
        shard_filenames = [
            "model-00001-of-000002.safetensors",
            "model-00002-of-000002.safetensors"
        ]
        for filename in shard_filenames:
            local_path = os.path.join(self.model_dir, filename)
            if not os.path.exists(local_path):
                print(f"Downloading {filename} from Hugging Face for 8B model...")
                hf_hub_download(
                    repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    filename=filename,
                    local_dir=self.model_dir
                )
        
        # Initialize the DeepSeek wrapper.
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
