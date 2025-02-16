import os
import unittest
from huggingface_hub import hf_hub_download, list_repo_files
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

class TestDeepSeekIntegration(unittest.TestCase):
    def setUp(self):
        # Directory for downloaded checkpoint files
        self.model_dir = os.path.join(os.getcwd(), "temp", "deepseek_model")
        os.makedirs(self.model_dir, exist_ok=True)

        # We'll test with DeepSeek-R1. You can change to another repo if needed.
        repo_id = "deepseek-ai/DeepSeek-R1"

        # List all files in the repository
        files = list_repo_files(repo_id)

        # Download .safetensors and config.json
        for filename in files:
            if filename.endswith(".safetensors") or filename == "config.json":
                print(f"Downloading {filename} from Hugging Face for DeepSeek-R1...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                )

        # Initialize the DeepSeek wrapper
        self.wrapper = DeepSeekWrapper(
            model_path=self.model_dir,
            config_path=os.path.join(self.model_dir, "config.json"),
            temperature=0.7,
            max_new_tokens=20,
            device="cuda",
            enable_dp_attention=False,  # or True, if your code is set up for dp attn
            use_fp8=False,  # set to True if you want to try FP8, but ensure GPU support
        )

    def test_basic_generate(self):
        prompt = "Hello from Intelli. How are you?"
        output = self.wrapper.generate(prompt)
        print("DeepSeek output:", output)
        self.assertTrue(len(output) > 0, "Expected non-empty generation output.")

if __name__ == "__main__":
    unittest.main()
