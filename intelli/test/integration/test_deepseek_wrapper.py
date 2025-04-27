import unittest
import torch
import os
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

class TestDeepSeekWrapper(unittest.TestCase):

    def setUp(self):
        # using a small distill model for CI speed
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

    def test_bpe_tokenization(self):
        model = DeepSeekWrapper(
            repo_id=self.repo_id,
            model_filename=self.model_filename,
            config_path=None,
            quantized=self.quantized,
        )

        word = "tokenization"
        token_ids = model.tokenize(word)

        self.assertIsInstance(token_ids, list)
        self.assertGreater(
            len(token_ids),
            1,
            msg=f"BPE did not split '{word}' into subwords: got {token_ids!r}",
        )

        y = model.infer(torch.tensor([token_ids], dtype=torch.long))
        self.assertEqual(y.shape[1], len(token_ids))

    def test_tokenize_and_infer_from_text(self):
        model = DeepSeekWrapper(
            repo_id=self.repo_id,
            model_filename=self.model_filename,
            config_path=None,
            quantized=self.quantized,
        )

        self.assertIsInstance(model.vocab, dict)
        self.assertGreater(len(model.vocab), 0)

        text = "This is a tokenization test."
        token_ids = model.tokenize(text)
        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)
        print("Token IDs:", token_ids)

        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        output = model.infer(input_tensor)

        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], len(token_ids))
        self.assertEqual(output.shape[2], model.config["vocab_size"])
        print("Inference successful on text input, output shape:", output.shape)


if __name__ == "__main__":
    unittest.main()
