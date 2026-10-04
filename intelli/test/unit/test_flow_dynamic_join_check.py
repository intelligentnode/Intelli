import asyncio
import unittest

from intelli.flow import DynamicConnector, Flow, Task, TextTaskInput
from intelli.flow.agents.custom_agent import CustomAgent
from intelli.flow.types import AgentTypes


class StepAgent(CustomAgent):
    """Plain Python step that returns a fixed reply."""

    def __init__(self, reply):
        super().__init__(agent_type=AgentTypes.TEXT.value, provider="custom", mission="step")
        self.reply = reply

    def execute(self, agent_input, new_params=None):
        return self.reply


def build_tasks(names):
    return {name: Task(TextTaskInput(f"run {name}"), StepAgent(f"output of {name}")) for name in names}


def route_to(key, destinations):
    return DynamicConnector(decision_fn=lambda output, output_type: key, destinations=destinations)


class TestTaskWaitingForTwoDestinations(unittest.TestCase):
    def test_shared_task_after_two_destinations_of_one_connector_is_rejected(self):
        tasks = build_tasks(["decision", "tool_execution", "direct_response", "formatter"])

        with self.assertRaisesRegex(ValueError, "'formatter' would never run"):
            Flow(
                tasks=tasks,
                map_paths={"tool_execution": ["formatter"], "direct_response": ["formatter"]},
                dynamic_connectors={
                    "decision": route_to("tool", {"tool": "tool_execution", "direct": "direct_response"})
                },
            )

    def test_each_destination_with_its_own_next_task_runs_the_chosen_path(self):
        for chosen, path in (("tool", ["tool_execution", "tool_formatter"]),
                             ("direct", ["direct_response", "direct_formatter"])):
            tasks = build_tasks(
                ["decision", "tool_execution", "direct_response", "tool_formatter", "direct_formatter"]
            )
            flow = Flow(
                tasks=tasks,
                map_paths={"tool_execution": ["tool_formatter"], "direct_response": ["direct_formatter"]},
                dynamic_connectors={
                    "decision": route_to(chosen, {"tool": "tool_execution", "direct": "direct_response"})
                },
            )
            out = asyncio.run(flow.start())

            self.assertEqual(flow.errors, {})
            self.assertEqual(sorted(out), sorted(["decision"] + path))

    def test_shared_task_is_allowed_when_another_connector_can_start_one_parent(self):
        # "y" can also be started by the connector on "b", so "x" and "y" can both run
        tasks = build_tasks(["a", "b", "x", "y", "z", "join"])
        flow = Flow(
            tasks=tasks,
            map_paths={"x": ["join"], "y": ["join"]},
            dynamic_connectors={
                "a": route_to("x", {"x": "x", "y": "y"}),
                "b": route_to("y", {"y": "y", "z": "z"}),
            },
        )
        out = asyncio.run(flow.start())

        self.assertEqual(flow.errors, {})
        self.assertEqual(sorted(out), ["a", "b", "join", "x", "y"])


if __name__ == "__main__":
    unittest.main()
