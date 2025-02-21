import unittest
from intelli.model.deepseek.wrapper import DeepSeekWrapper

class TestDeepSeekWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DeepSeekWrapper(
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            filename="DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf",
            quantized=True,
            n_gpu_layers=0,
            model_repo_id="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
        )

    def test_generation(self):
        inputs = {
            "prompt": "What is 2+2? Respond only with the number.",
            "temperature": 0.01,
            "max_tokens": 10
        }
        result = self.model.generate(inputs)
        self.assertIsInstance(result, str)
        
        self.assertTrue(
            any(valid in result for valid in ["4", "four", "Four"]),
            f"Unexpected response: {result}"
        )

    def test_edge_cases(self):
        test_cases = [
            {"prompt": "5+3=", "allowed": ["8", "eight"]},
            {"prompt": "Capital of France?", "allowed": ["Paris"]},
            {"prompt": "1+1=", "allowed": ["2", "two"]}
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                result = self.model.generate({
                    "prompt": case["prompt"],
                    "temperature": 0.01,
                    "max_tokens": 25
                })
                self.assertTrue(
                    any(valid.lower() in result.lower() for valid in case["allowed"]),
                    f"Unexpected response: {result}"
                )

if __name__ == "__main__":
    unittest.main() 