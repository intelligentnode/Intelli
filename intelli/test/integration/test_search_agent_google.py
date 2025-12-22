import os
import unittest

from dotenv import load_dotenv

from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.tasks.task import Task
from intelli.flow.types import AgentTypes


load_dotenv()


class TestSearchAgentGoogle(unittest.TestCase):
    def setUp(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        if not self.google_api_key or not self.google_cse_id:
            self.skipTest("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")

    def test_google_web_search_agent(self):
        search_agent = Agent(
            agent_type=AgentTypes.SEARCH.value,
            provider="google",
            mission="web search",
            model_params={
                "google_api_key": self.google_api_key,
                "google_cse_id": self.google_cse_id,
                "k": 3,
                "safe": "active",
                "timeout": 20.0,
                "as_text": True,
            },
        )

        task = Task(
            TextTaskInput("Google responses API overview"),
            search_agent,
            log=True,
        )

        flow = SequenceFlow([task], log=True)
        result = flow.start()

        self.assertIn("task1", result)
        self.assertIsInstance(result["task1"], str)
        self.assertTrue(len(result["task1"].strip()) > 0)


if __name__ == "__main__":
    unittest.main()


