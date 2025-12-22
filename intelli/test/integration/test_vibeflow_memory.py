import asyncio
import json
import os
import tempfile
import unittest

from intelli.flow.agents.custom_agent import CustomAgent
from intelli.flow.types import AgentTypes, InputTypes
from intelli.flow.vibe import VibeFlow, AgentSpec


class LocalReturnAgent(CustomAgent):
    def __init__(self, text: str):
        super().__init__(agent_type=AgentTypes.TEXT.value, provider="local", mission="return")
        self._text = text

    def execute(self, agent_input, new_params=None):
        return self._text


class TestVibeFlowMemory(unittest.TestCase):
    def test_build_from_spec_with_memory_and_save_load(self):
        # Spec: task1 writes output to memory via output_memory_map; task2 reads it via memory_key.
        spec = {
            "version": "1",
            "tasks": [
                {
                    "name": "t1",
                    "desc": "produce greeting",
                    "agent": {
                        "agent_type": "text",
                        "provider": "local",
                        "mission": "return hello",
                        "model_params": {"return": "hello"},
                    },
                },
                {
                    "name": "t2",
                    "desc": "consume greeting",
                    "memory_key": "greeting",
                    "agent": {
                        "agent_type": "text",
                        "provider": "local",
                        "mission": "return ok",
                        "model_params": {"return": "ok"},
                    },
                },
            ],
            "map_paths": {"t1": ["t2"]},
            "output_memory_map": {"t1": "greeting"},
            "log": False,
        }

        def local_factory(agent_spec: AgentSpec):
            return LocalReturnAgent(agent_spec.model_params.get("return", ""))

        vf = VibeFlow(planner_provider="openai", planner_fn=lambda s, u: spec)
        flow = vf.build_from_spec(
            spec,
            agent_factories={(AgentTypes.TEXT.value, "local"): local_factory},
        )

        # Execute flow and ensure memory is populated.
        results = asyncio.run(flow.start(initial_input="ignored", initial_input_type=InputTypes.TEXT.value))
        self.assertIn("t1", results)
        self.assertIn("t2", results)
        self.assertEqual(results["t1"]["output"], "hello")
        self.assertEqual(results["t2"]["output"], "ok")
        self.assertTrue(flow.memory.has_key("greeting"))
        self.assertEqual(flow.memory.retrieve("greeting"), "hello")

        # Save/load spec bundle (graph rendering off for test).
        with tempfile.TemporaryDirectory() as d:
            meta = vf.save_bundle(d, spec, flow, render_graph=False)
            self.assertTrue(os.path.exists(meta["spec"]))
            loaded = vf.load_spec(meta["spec"])
            self.assertEqual(loaded["version"], "1")
            self.assertIn("tasks", loaded)


if __name__ == "__main__":
    unittest.main()


