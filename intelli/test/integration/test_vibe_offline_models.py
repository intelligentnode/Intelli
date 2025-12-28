import os
import unittest

from dotenv import load_dotenv

from intelli.flow import VibeAgent


load_dotenv()


class TestVibeOfflineModels(unittest.TestCase):
    def setUp(self):
        # Provide safe dummy values so ${ENV:...} placeholders can be resolved during build.
        os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8000")
        os.environ.setdefault("LLAMACPP_MODEL_PATH", "/tmp/fake-model.gguf")
        os.environ.setdefault("OPENAI_API_KEY", "test_key")

    def test_build_from_spec_allows_vllm_text_agent_with_base_url(self):
        vf = VibeAgent(planner_provider="openai", planner_api_key="test_key", planner_fn=lambda *_: {})

        spec = {
            "version": "1",
            "tasks": [
                {
                    "name": "research",
                    "desc": "Find key facts about quantum computing.",
                    "input_type": "text",
                    "agent": {
                        "agent_type": "text",
                        "provider": "openai",
                        "mission": "Researcher",
                        "model_params": {"key": "${ENV:OPENAI_API_KEY}", "model": "gpt-4o-mini"},
                        "options": {},
                    },
                },
                {
                    "name": "summarize_local",
                    "desc": "Summarize the facts in 5 bullet points.",
                    "input_type": "text",
                    "agent": {
                        "agent_type": "text",
                        "provider": "vllm",
                        "mission": "Analyst",
                        "model_params": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"},
                        "options": {"baseUrl": "${ENV:VLLM_BASE_URL}"},
                    },
                },
            ],
            "map_paths": {"research": ["summarize_local"]},
            "dynamic_connectors": [],
        }

        flow = vf.build_from_spec(spec)
        self.assertIn("summarize_local", flow.tasks)
        self.assertEqual(flow.tasks["summarize_local"].agent.provider, "vllm")
        self.assertEqual(flow.tasks["summarize_local"].agent.options.get("baseUrl"), os.environ["VLLM_BASE_URL"])

    def test_build_from_spec_rejects_vllm_text_agent_without_base_url(self):
        vf = VibeAgent(planner_provider="openai", planner_api_key="test_key", planner_fn=lambda *_: {})

        bad_spec = {
            "version": "1",
            "tasks": [
                {
                    "name": "summarize_local",
                    "desc": "Summarize locally.",
                    "input_type": "text",
                    "agent": {
                        "agent_type": "text",
                        "provider": "vllm",
                        "mission": "Analyst",
                        "model_params": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"},
                        "options": {},  # missing baseUrl
                    },
                }
            ],
            "map_paths": {},
            "dynamic_connectors": [],
        }

        with self.assertRaises(ValueError):
            vf.build_from_spec(bad_spec)

    def test_build_from_spec_allows_llamacpp_text_agent_with_model_path(self):
        vf = VibeAgent(planner_provider="openai", planner_api_key="test_key", planner_fn=lambda *_: {})

        spec = {
            "version": "1",
            "tasks": [
                {
                    "name": "summarize_local",
                    "desc": "Summarize locally.",
                    "input_type": "text",
                    "agent": {
                        "agent_type": "text",
                        "provider": "llamacpp",
                        "mission": "Analyst",
                        "model_params": {"model": "llama.cpp"},
                        "options": {
                            "model_path": "${ENV:LLAMACPP_MODEL_PATH}",
                            "model_params": {"n_ctx": 512},
                        },
                    },
                }
            ],
            "map_paths": {},
            "dynamic_connectors": [],
        }

        flow = vf.build_from_spec(spec)
        self.assertIn("summarize_local", flow.tasks)
        self.assertEqual(flow.tasks["summarize_local"].agent.provider, "llamacpp")
        self.assertEqual(flow.tasks["summarize_local"].agent.options.get("model_path"), os.environ["LLAMACPP_MODEL_PATH"])

    def test_build_from_spec_rejects_llamacpp_text_agent_without_model_path(self):
        vf = VibeAgent(planner_provider="openai", planner_api_key="test_key", planner_fn=lambda *_: {})

        bad_spec = {
            "version": "1",
            "tasks": [
                {
                    "name": "summarize_local",
                    "desc": "Summarize locally.",
                    "input_type": "text",
                    "agent": {
                        "agent_type": "text",
                        "provider": "llamacpp",
                        "mission": "Analyst",
                        "model_params": {"model": "llama.cpp"},
                        "options": {},  # missing model_path
                    },
                }
            ],
            "map_paths": {},
            "dynamic_connectors": [],
        }

        with self.assertRaises(ValueError):
            vf.build_from_spec(bad_spec)


if __name__ == "__main__":
    unittest.main()


