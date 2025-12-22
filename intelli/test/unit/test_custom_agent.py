import unittest

from intelli.flow import SequenceFlow, Task, TextTaskInput
from intelli.flow.agents.custom_agent import CustomAgent
from intelli.flow.types import AgentTypes


class UppercaseAgent(CustomAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentTypes.TEXT.value,
            provider="custom",
            mission="uppercase",
        )

    def execute(self, agent_input, new_params=None):
        # agent_input.desc contains the templated prompt; make the behavior deterministic.
        text = agent_input.desc or ""
        return text.upper()


class TestCustomAgent(unittest.TestCase):
    def test_custom_agent_runs_in_sequence_flow(self):
        agent = UppercaseAgent()
        task = Task(TextTaskInput("hello"), agent, log=False)

        flow = SequenceFlow([task], log=False)
        out = flow.start()

        self.assertIn("task1", out)
        self.assertIsInstance(out["task1"], str)
        self.assertTrue("HELLO" in out["task1"])


if __name__ == "__main__":
    unittest.main()


