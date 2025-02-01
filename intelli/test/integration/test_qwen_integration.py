import os
import unittest
from huggingface_hub import hf_hub_download, list_repo_files
from intelli.wrappers.universal_wrapper import UniversalWrapper


class TestQwenIntegration(unittest.TestCase):
    def setUp(self):
        # Create a local directory for Qwen2.5-Math-1.5B files.
        self.model_dir = os.path.join(os.getcwd(), "temp", "qwen2_1.5b")
        os.makedirs(self.model_dir, exist_ok=True)

        # Specify the repository ID (update with the correct repo ID if needed)
        repo_id = "Qwen/Qwen2.5-Math-1.5B"

        # List all files in the repository.
        files = list_repo_files(repo_id)

        # Download all required files: config.json, model.safetensors, tokenizer.json, tokenizer_config.json, vocab.json, etc.
        for filename in files:
            if filename.endswith(".safetensors") or filename in [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
            ]:
                print(
                    f"Downloading {filename} from Hugging Face for Qwen2.5-Math-1.5B..."
                )
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,  # deprecated parameter; harmless here
                )

        # Instantiate the universal wrapper.
        # For Qwen2.5-Math-1.5B, we do not need model parallelism, so we set model_parallel=1.
        self.wrapper = UniversalWrapper(
            model_path=self.model_dir,
            config_path=os.path.join(self.model_dir, "config.json"),
            temperature=0.7,
            max_new_tokens=20,
            model_parallel=1,  # 1 for single-GPU usage
            device="cuda",  # set "cpu" if GPU is not available
            enable_dp_attention=False,  # DP attention not needed for this small model
            use_fp8=False,  # For stability, use BF16 on smaller models unless FP8 is proven available
        )

    def test_basic_generate(self):
        prompt = "Solve 2+2 and tell me the result."
        output = self.wrapper.generate(prompt)
        print("Qwen2.5-Math-1.5B output:", output)
        self.assertTrue(len(output) > 0, "Generated output should not be empty.")


if __name__ == "__main__":
    unittest.main()
