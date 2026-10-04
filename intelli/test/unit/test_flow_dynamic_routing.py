import asyncio
import unittest

from intelli.flow import DynamicConnector, Flow, Task, TextTaskInput
from intelli.flow.agents.custom_agent import CustomAgent
from intelli.flow.types import AgentTypes


class StepAgent(CustomAgent):
    """Plain Python step that returns a fixed reply and keeps the prompts it receives."""

    def __init__(self, reply):
        super().__init__(agent_type=AgentTypes.TEXT.value, provider="custom", mission="step")
        self.reply = reply
        self.prompts = []

    def execute(self, agent_input, new_params=None):
        self.prompts.append(agent_input.desc)
        return self.reply


def run_flow(names, map_paths, dynamic_connectors):
    tasks = {name: Task(TextTaskInput(f"run {name}"), StepAgent(f"output of {name}")) for name in names}
    flow = Flow(tasks=tasks, map_paths=map_paths, dynamic_connectors=dynamic_connectors)
    out = asyncio.run(flow.start(initial_input="start"))
    return flow, tasks, out


def route_to(key):
    return DynamicConnector(decision_fn=lambda output, output_type: key, destinations={"x": "x", "y": "y"})


class TestDynamicRoutingWithStaticParents(unittest.TestCase):
    def test_only_the_chosen_destination_runs(self):
        for chosen, other in (("x", "y"), ("y", "x")):
            flow, tasks, out = run_flow(["a", "b", "x", "y"], {"b": ["x", "y"]}, {"a": route_to(chosen)})

            self.assertEqual(flow.errors, {})
            self.assertEqual(sorted(out), sorted(["a", "b", chosen]))
            self.assertEqual(tasks[other].agent.prompts, [])

    def test_chosen_destination_gets_the_static_parent_output(self):
        flow, tasks, out = run_flow(["a", "b", "x", "y"], {"b": ["x", "y"]}, {"a": route_to("x")})

        prompt = tasks["x"].agent.prompts[0]
        self.assertIn("output of a", prompt)
        self.assertIn("output of b", prompt)

    def test_chosen_destination_waits_for_a_later_static_parent(self):
        flow, tasks, out = run_flow(
            ["a", "b", "c", "x", "y"], {"b": ["c"], "c": ["x", "y"]}, {"a": route_to("x")}
        )

        self.assertEqual(sorted(out), ["a", "b", "c", "x"])
        self.assertIn("output of c", tasks["x"].agent.prompts[0])

    def test_static_children_that_are_not_destinations_still_run(self):
        flow, tasks, out = run_flow(
            ["a", "b", "x", "y", "side"], {"b": ["x", "y", "side"]}, {"a": route_to("x")}
        )

        self.assertEqual(sorted(out), ["a", "b", "side", "x"])

    def test_connector_on_a_task_that_joins_parents_keeps_working(self):
        flow, tasks, out = run_flow(
            ["a", "b", "join", "x", "y"], {"a": ["join"], "b": ["join"]}, {"join": route_to("x")}
        )

        self.assertEqual(flow.errors, {})
        self.assertEqual(sorted(out), ["a", "b", "join", "x"])


if __name__ == "__main__":
    unittest.main()
