import unittest
import os
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

class TestDeepSeekIntegration(unittest.TestCase):
    def setUp(self):
        # You can parametrize or set the path from environment
        self.model_path = "/path/to/DeepSeek-V3-Demo"
        self.config_path = "/path/to/DeepSeek-V3-Demo/configs/config_671B.json"

        self.wrapper = DeepSeekWrapper(model_path=self.model_path, config_path=self.config_path)

    def test_basic_generate(self):
        prompt = "Hello from Intelli. How are you?"
        output = self.wrapper.generate(prompt)
        print("DeepSeek output:", output)
        self.assertTrue(len(output) > 0)

if __name__ == "__main__":
    unittest.main()
