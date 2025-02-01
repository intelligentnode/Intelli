import os
import unittest
from huggingface_hub import hf_hub_download, list_repo_files
from intelli.wrappers.universal_wrapper import DeepSeekWrapper

class TestDeepSeekQwenIntegration(unittest.TestCase):
    def setUp(self):
        # Create a local directory for DeepSeek-R1-Distill-Qwen-1.5B files.
        self.model_dir = os.path.join(os.getcwd(), "temp", "deepseek_qwen_1.5b")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Specify the repository ID for DeepSeek-R1-Distill-Qwen-1.5B.
        repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        
        # List all files in the repository.
        files = list_repo_files(repo_id)
        
        # Download all required files.
        # We expect to see: config.json, generation_config.json, model.safetensors,
        # tokenizer.json, and tokenizer_config.json.
        for filename in files:
            if (filename.endswith(".safetensors") or 
                filename in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]):
                print(f"Downloading {filename} from Hugging Face for DeepSeek-R1-Distill-Qwen-1.5B...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False  # this parameter is deprecated; it's fine to include it
                )
        
        # Instantiate the universal wrapper.
        # Since this is a 1.5B model, it should comfortably run on a single GPU.
        self.wrapper = UniversalWrapper(
            model_path=self.model_dir,
            config_path=os.path.join(self.model_dir, "config.json"),
            temperature=0.7,
            max_new_tokens=20,
            model_parallel=1,           # single-file checkpoint; no need for multi-GPU splitting
            device="cuda",              # set to "cpu" if GPU is not available
            enable_dp_attention=False,  # data-parallel attention not needed for this small model
            use_fp8=False               # use BF16 for stability on a 1.5B model
        )
    
    def test_basic_generate(self):
        prompt = "Solve 2+2 and tell me the result."
        output = self.wrapper.generate(prompt)
        print("DeepSeek Qwen 1.5B output:", output)
        self.assertTrue(len(output) > 0, "Generated output should not be empty.")

if __name__ == "__main__":
    unittest.main()
