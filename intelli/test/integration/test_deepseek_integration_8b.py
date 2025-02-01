import os
import unittest
from huggingface_hub import hf_hub_download, list_repo_files
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper


class TestDeepSeekIntegration8B(unittest.TestCase):
    def setUp(self):
        # Directory for DeepSeek-R1-Distill-Llama-8B files.
        self.model_dir = os.path.join(os.getcwd(), "temp", "deepseek_model_8b")
        os.makedirs(self.model_dir, exist_ok=True)

        repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

        # List all files in the repository.
        files = list_repo_files(repo_id)

        # Download all relevant files (safetensors and config files).
        for filename in files:
            if filename.endswith(".safetensors") or filename in [
                "config.json",
                "tokenizer.json",
            ]:
                print(f"Downloading {filename} from Hugging Face for 8B model...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                )

        # Initialize the DeepSeek wrapper.
        self.wrapper = DeepSeekWrapper(
                    model_path=self.model_dir,
                    config_path=os.path.join(self.model_dir, "config.json"),
                    temperature=0.7,
                    max_new_tokens=20,
                    model_parallel=1,
                    device="cuda",
                    enable_dp_attention=True,
                    use_fp8=True)

    def test_basic_generate(self):
        prompt = "Hello from Intelli with the 8B model. How are you?"
        output = self.wrapper.generate(prompt)
        print("DeepSeek 8B output:", output)
        self.assertTrue(len(output) > 0, "Generated output should not be empty.")


if __name__ == "__main__":
    unittest.main()
