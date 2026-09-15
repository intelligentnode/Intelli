import os
import shutil
import tempfile
import unittest

from dotenv import load_dotenv

from intelli.function.coding_agent import CodingAgent

load_dotenv()


class TestCodingAgentIntegration(unittest.TestCase):
    """Live end-to-end run: the agent fixes a real bug until tests pass.

    Uses Gemini by default (set CODING_AGENT_PROVIDER/CODING_AGENT_MODEL to
    switch, e.g. anthropic + claude-sonnet-4-6 or openai + gpt-5.5).
    """

    def setUp(self):
        self.provider = os.getenv("CODING_AGENT_PROVIDER", "gemini")
        key_var = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                   "anthropic": "ANTHROPIC_API_KEY", "mistral": "MISTRAL_API_KEY"}
        self.api_key = os.getenv(key_var.get(self.provider, "GEMINI_API_KEY"))
        if not self.api_key:
            self.skipTest(f"No API key for provider {self.provider}")
        self.model = os.getenv("CODING_AGENT_MODEL", "gemini-2.5-flash" if self.provider == "gemini" else None)

        self.ws = tempfile.mkdtemp(prefix="intelli_coder_")
        with open(os.path.join(self.ws, "calc.py"), "w") as f:
            f.write("def add(a, b):\n    return a - b\n")
        with open(os.path.join(self.ws, "test_calc.py"), "w") as f:
            f.write("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_fixes_bug_until_tests_green(self):
        print(f"---- live coding agent ({self.provider}) ----")
        agent = CodingAgent(
            api_key=self.api_key,
            provider=self.provider,
            model=self.model,
            workspace=self.ws,
            max_iterations=12,
            log=True,
        )
        result = agent.run(
            "The tests in test_calc.py fail. Find and fix the bug in calc.py.",
            test_command="python3 -m pytest -q test_calc.py",
        )

        print("result:", result["success"], "| iterations:", result["iterations"])
        self.assertTrue(result["success"], f"Agent did not get tests green: {result}")
        with open(os.path.join(self.ws, "calc.py")) as f:
            self.assertIn("a + b", f.read())


if __name__ == "__main__":
    unittest.main()
