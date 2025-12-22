import unittest
import asyncio
from intelli.flow.flow import Flow
from intelli.flow.tasks.task import Task
from intelli.flow.tasks.loop_task import LoopTask
from intelli.flow.input.agent_input import TextAgentInput
from intelli.flow.agents.agent import Agent
from intelli.flow.store.memory import Memory

class MockErrorAgent:
    def __init__(self, agent_type="text", provider="mock"):
        self.type = agent_type
        self.provider = provider
        self.mission = "mock mission"
        self.model_params = {}

    def execute(self, agent_input, new_params=None):
        return "Error executing agent: Mocked error occurred"

class TestFlowEdgeCases(unittest.TestCase):

    def setUp(self):
        self.memory = Memory()

    def test_loop_task_stop_on_error(self):
        # Create a task that always fails
        fail_agent = MockErrorAgent()
        fail_task = Task(TextAgentInput("fail"), fail_agent)
        
        # LoopTask with stop_on_error=True
        loop_task = LoopTask(
            desc="loop until error",
            steps=[fail_task],
            max_loops=5,
            stop_on_error=True,
            log=True
        )
        
        # Execute loop
        result = loop_task.execute(memory=self.memory)
        
        # Verify it stopped after 1 iteration instead of 5
        self.assertTrue(result.startswith("Error"))
        # We can't easily check internal history without a custom memory key
        # but the result should be the error from the first iteration.

    def test_flow_catch_soft_errors(self):
        # Create a task that returns an error string
        fail_agent = MockErrorAgent()
        fail_task = Task(TextAgentInput("fail"), fail_agent)
        
        flow = Flow(
            tasks={"task1": fail_task},
            map_paths={},
            log=True
        )
        
        async def run_flow():
            return await flow.start()
            
        results = asyncio.run(run_flow())
        
        # Verify the error was caught in flow.errors
        self.assertIn("task1", flow.errors)
        self.assertTrue(flow.errors["task1"].startswith("Error executing agent"))

if __name__ == "__main__":
    unittest.main()

