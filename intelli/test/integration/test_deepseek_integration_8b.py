import os
import unittest
from huggingface_hub import hf_hub_download, list_repo_files
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

class TestDeepSeekIntegration8B(unittest.TestCase):
    def setUp(self):
        # Directory for downloaded checkpoint files
        self.model_dir = os.path.join(os.getcwd(), "temp", "deepseek_model_8b")
        os.makedirs(self.model_dir, exist_ok=True)

        repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

        # List files in the repo
        files = list_repo_files(repo_id)

        # Download relevant files
        for filename in files:
            # Typically we want safetensors, config.json, possibly tokenizer.json
            if filename.endswith(".safetensors") or filename == "config.json":
                print(f"Downloading {filename} from HF for the 8B model...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                )

        # Init wrapper
        self.wrapper = DeepSeekWrapper(
            model_path=self.model_dir,
            config_path=os.path.join(self.model_dir, "config.json"),
            temperature=0.7,
            max_new_tokens=20,
            model_parallel=1,  # adjust if you want multi-GPU
            device="cuda",
            enable_dp_attention=False,
            use_fp8=False,  # set True if your GPU supports it
        )

    def test_basic_generate(self):
        prompt = "Hello from Intelli with the 8B model. How are you?"
        output = self.wrapper.generate(prompt)
        print("DeepSeek 8B output:", output)
        self.assertTrue(len(output) > 0, "Expected non-empty generation output.")

if __name__ == "__main__":
    unittest.main()
