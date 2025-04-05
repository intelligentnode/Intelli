import unittest
import torch
import os
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper


class TestDeepSeekWrapper(unittest.TestCase):

    def setUp(self):
        self.repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        self.model_filename = "model.safetensors.index.json"
        self.quantized = True

    def test_load_and_infer(self):
        model = DeepSeekWrapper(
            repo_id=self.repo_id,
            model_filename=self.model_filename,
            config_path=None,
            quantized=self.quantized,
        )

        dummy_input = torch.randint(0, 100, (1, 16))
        output = model.infer(dummy_input)

        self.assertEqual(output.shape[1], dummy_input.shape[1])
        print("Inference successful, output shape:", output.shape)


if __name__ == "__main__":
    unittest.main()
