import base64
import unittest

from intelli.function.computer_agent import (
    ComputerAgent,
    ComputerEnvironment,
    _anthropic_tool_spec,
)

FAKE_PNG = b"\x89PNG_fake_image_bytes"


class FakeEnvironment(ComputerEnvironment):
    """Records every action so tests can assert the executed sequence."""

    display_width = 800
    display_height = 600

    def __init__(self):
        self.calls = []

    def screenshot(self):
        self.calls.append(("screenshot",))
        return FAKE_PNG

    def click(self, x, y, button="left", modifiers=None):
        self.calls.append(("click", x, y, button))

    def double_click(self, x, y):
        self.calls.append(("double_click", x, y))

    def move(self, x, y):
        self.calls.append(("move", x, y))

    def mouse_down(self, x, y):
        self.calls.append(("mouse_down", x, y))

    def mouse_up(self, x, y):
        self.calls.append(("mouse_up", x, y))

    def go_back(self):
        self.calls.append(("go_back",))

    def go_forward(self):
        self.calls.append(("go_forward",))

    def drag(self, path):
        self.calls.append(("drag", tuple(path)))

    def type_text(self, text):
        self.calls.append(("type", text))

    def key(self, combo):
        self.calls.append(("key", combo))

    def hold_key(self, combo, duration):
        self.calls.append(("hold_key", combo, duration))

    def scroll(self, x, y, scroll_x=0, scroll_y=0, modifiers=None):
        self.calls.append(("scroll", x, y, scroll_x, scroll_y, tuple(modifiers or ())))

    def wait(self, seconds):
        self.calls.append(("wait", seconds))


class FakeAnthropicWrapper:
    """Plays back scripted Messages-API responses and records requests."""

    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def generate_text(self, params, extra_headers=None):
        # Deep-copy: the agent mutates its messages list across iterations,
        # while a real wrapper serializes the payload at send time.
        import copy
        self.requests.append({"params": copy.deepcopy(params), "headers": extra_headers})
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


class FakeOpenAIWrapper:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def generate_gpt5_response(self, params):
        self.requests.append(params)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]


class TestAnthropicComputerLoop(unittest.TestCase):
    def _make_agent(self, responses, **kwargs):
        env = FakeEnvironment()
        agent = ComputerAgent(api_key="k", provider="anthropic",
                              model="claude-sonnet-4-6", environment=env, **kwargs)
        agent._wrapper = FakeAnthropicWrapper(responses)
        return agent, env

    def test_tool_spec_selection(self):
        self.assertEqual(_anthropic_tool_spec("claude-sonnet-4-6"),
                         ("computer_20251124", "computer-use-2025-11-24"))
        self.assertEqual(_anthropic_tool_spec("claude-opus-4-8"),
                         ("computer_20251124", "computer-use-2025-11-24"))
        self.assertEqual(_anthropic_tool_spec("claude-sonnet-4-5"),
                         ("computer_20250124", "computer-use-2025-01-24"))

    def test_screenshot_click_then_done(self):
        responses = [
            {"stop_reason": "tool_use", "content": [
                {"type": "text", "text": "Taking a look."},
                {"type": "tool_use", "id": "tu_1", "name": "computer",
                 "input": {"action": "screenshot"}},
            ]},
            {"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "tu_2", "name": "computer",
                 "input": {"action": "left_click", "coordinate": [10, 20]}},
            ]},
            {"stop_reason": "end_turn", "content": [
                {"type": "text", "text": "Done: clicked the button."},
            ]},
        ]
        agent, env = self._make_agent(responses)
        result = agent.run("click the button")

        self.assertTrue(result["success"])
        self.assertEqual(result["output"], "Done: clicked the button.")
        self.assertIn(("screenshot",), env.calls)
        self.assertIn(("click", 10, 20, "left"), env.calls)

        wrapper = agent._wrapper
        # Beta header + tool definition on every request.
        self.assertEqual(wrapper.requests[0]["headers"]["anthropic-beta"], "computer-use-2025-11-24")
        tool = wrapper.requests[0]["params"]["tools"][0]
        self.assertEqual(tool["type"], "computer_20251124")
        self.assertEqual(tool["display_width_px"], 800)

        # Screenshot tool_result must carry the base64 image back.
        followup_messages = wrapper.requests[1]["params"]["messages"]
        tool_result = followup_messages[-1]["content"][0]
        self.assertEqual(tool_result["tool_use_id"], "tu_1")
        image = tool_result["content"][0]
        self.assertEqual(image["source"]["data"], base64.standard_b64encode(FAKE_PNG).decode())
        # Assistant content must be replayed verbatim.
        self.assertEqual(followup_messages[1]["content"], responses[0]["content"])

    def test_on_action_blocks_execution(self):
        responses = [
            {"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "computer",
                 "input": {"action": "left_click", "coordinate": [5, 5]}},
            ]},
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]},
        ]
        agent, env = self._make_agent(responses, on_action=lambda a: False)
        result = agent.run("blocked task")

        self.assertTrue(result["success"])
        self.assertNotIn(("click", 5, 5, "left"), env.calls)
        blocked = agent._wrapper.requests[1]["params"]["messages"][-1]["content"][0]
        self.assertTrue(blocked.get("is_error"))

    def test_max_iterations(self):
        loop_forever = {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "tu", "name": "computer", "input": {"action": "screenshot"}},
        ]}
        agent, _ = self._make_agent([loop_forever], max_iterations=3)
        result = agent.run("never ends")
        self.assertFalse(result["success"])
        self.assertEqual(result["iterations"], 3)

    def test_action_mapping_hold_scroll_mouse(self):
        # hold_key honors duration; scroll carries the modifier; mouse up/down map.
        responses = [
            {"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "1", "name": "computer",
                 "input": {"action": "hold_key", "text": "shift", "duration": 2}},
                {"type": "tool_use", "id": "2", "name": "computer",
                 "input": {"action": "scroll", "coordinate": [10, 20],
                           "scroll_direction": "down", "scroll_amount": 3, "text": "shift"}},
                {"type": "tool_use", "id": "3", "name": "computer",
                 "input": {"action": "left_mouse_down", "coordinate": [7, 8]}},
                {"type": "tool_use", "id": "4", "name": "computer",
                 "input": {"action": "left_mouse_up", "coordinate": [7, 8]}},
            ]},
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]},
        ]
        agent, env = self._make_agent(responses)
        agent.run("do actions")
        self.assertIn(("hold_key", "shift", 2.0), env.calls)
        self.assertIn(("scroll", 10, 20, 0, 300, ("shift",)), env.calls)
        self.assertIn(("mouse_down", 7, 8), env.calls)
        self.assertIn(("mouse_up", 7, 8), env.calls)

    def test_on_action_hook_fails_closed(self):
        # A hook that raises must block the action, not silently allow it.
        env = FakeEnvironment()
        agent = ComputerAgent(api_key="k", provider="anthropic",
                              model="claude-sonnet-4-6", environment=env,
                              on_action=lambda a: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertTrue(agent._blocked({"action": "left_click"}))


class TestOpenAIComputerLoop(unittest.TestCase):
    def _make_agent(self, responses, **kwargs):
        env = FakeEnvironment()
        agent = ComputerAgent(api_key="k", provider="openai", environment=env, **kwargs)
        agent._wrapper = FakeOpenAIWrapper(responses)
        return agent, env

    def test_batched_actions_then_final_message(self):
        responses = [
            {"id": "resp_1", "output": [
                {"type": "computer_call", "call_id": "call_1",
                 "actions": [
                     {"type": "click", "button": "left", "x": 100, "y": 50},
                     {"type": "type", "text": "hello"},
                 ],
                 "pending_safety_checks": []},
            ]},
            {"id": "resp_2", "output": [
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Task complete."},
                ]},
            ]},
        ]
        agent, env = self._make_agent(responses)
        result = agent.run("fill the form")

        self.assertTrue(result["success"])
        self.assertEqual(result["output"], "Task complete.")
        self.assertIn(("click", 100, 50, "left"), env.calls)
        self.assertIn(("type", "hello"), env.calls)

        wrapper = agent._wrapper
        self.assertEqual(wrapper.requests[0]["tools"], [{"type": "computer"}])
        self.assertEqual(wrapper.requests[0]["model"], "gpt-5.5")
        # Follow-up chains via previous_response_id + computer_call_output.
        follow = wrapper.requests[1]
        self.assertEqual(follow["previous_response_id"], "resp_1")
        out = follow["input"][0]
        self.assertEqual(out["type"], "computer_call_output")
        self.assertEqual(out["call_id"], "call_1")
        self.assertTrue(out["output"]["image_url"].startswith("data:image/png;base64,"))

    def test_safety_checks_deny_by_default(self):
        # SECURITY: with no on_safety_check hook, a flagged risk must NOT be
        # auto-acknowledged. The loop stops, surfaces the checks, and executes nothing.
        checks = [{"id": "cu_sc_1", "code": "sensitive_domain", "message": "careful"}]
        responses = [
            {"id": "resp_1", "output": [
                {"type": "computer_call", "call_id": "call_1",
                 "action": {"type": "keypress", "keys": ["CTRL", "A"]},
                 "pending_safety_checks": checks},
            ]},
        ]
        agent, env = self._make_agent(responses)
        result = agent.run("select all")

        self.assertFalse(result["success"])
        self.assertEqual(result["pending_safety_checks"], checks)
        self.assertNotIn(("key", "CTRL+A"), env.calls)   # action not executed
        self.assertEqual(len(agent._wrapper.requests), 1)  # loop did not continue

    def test_safety_checks_executed_when_approved(self):
        checks = [{"id": "cu_sc_1", "code": "sensitive_domain", "message": "careful"}]
        responses = [
            {"id": "resp_1", "output": [
                {"type": "computer_call", "call_id": "call_1",
                 "action": {"type": "keypress", "keys": ["CTRL", "A"]},
                 "pending_safety_checks": checks},
            ]},
            {"id": "resp_2", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
            ]},
        ]
        agent, env = self._make_agent(responses, on_safety_check=lambda call, c: True)
        result = agent.run("select all")

        self.assertTrue(result["success"])
        self.assertIn(("key", "CTRL+A"), env.calls)
        follow = agent._wrapper.requests[1]
        self.assertEqual(follow["input"][0]["acknowledged_safety_checks"], checks)

    def test_back_forward_and_wheel_routing(self):
        responses = [
            {"id": "resp_1", "output": [
                {"type": "computer_call", "call_id": "c1", "actions": [
                    {"type": "click", "button": "back", "x": 0, "y": 0},
                    {"type": "click", "button": "forward", "x": 0, "y": 0},
                    {"type": "click", "button": "wheel", "x": 5, "y": 6},
                ], "pending_safety_checks": []},
            ]},
            {"id": "resp_2", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
            ]},
        ]
        agent, env = self._make_agent(responses)
        agent.run("navigate")
        self.assertIn(("go_back",), env.calls)
        self.assertIn(("go_forward",), env.calls)
        # 'wheel' becomes a middle click, never a silent left click.
        self.assertIn(("click", 5, 6, "middle"), env.calls)
        self.assertNotIn(("click", 0, 0, "left"), env.calls)


class TestComputerFlowIntegration(unittest.TestCase):
    def test_agent_type_registration(self):
        from intelli.flow.types import AgentTypes, Matcher

        self.assertEqual(AgentTypes.COMPUTER.value, "computer")
        self.assertEqual(Matcher.input["computer"], "text")
        self.assertEqual(Matcher.output["computer"], "text")

    def test_flow_task_runs_computer_handler_with_custom_env(self):
        from intelli.flow.agents.agent import Agent
        from intelli.flow.tasks.task import Task
        from intelli.flow.input.task_input import TextTaskInput
        import intelli.function.computer_agent as ca

        env = FakeEnvironment()
        responses = [
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "browsed ok"}]},
        ]

        # Patch the wrapper factory so no network is needed.
        original = ca.ComputerAgent._get_wrapper
        ca.ComputerAgent._get_wrapper = lambda self: FakeAnthropicWrapper(responses)
        try:
            agent = Agent(
                agent_type="computer",
                provider="anthropic",
                mission="operate the browser",
                model_params={"key": "k", "model": "claude-sonnet-4-6"},
                options={"environment": env},
            )
            task = Task(TextTaskInput("open the docs page"), agent, log=False)
            task.execute()
            self.assertIn("browsed ok", task.output)
        finally:
            ca.ComputerAgent._get_wrapper = original

    def test_browser_env_requires_playwright_cleanly(self):
        # Whether or not playwright is installed, importing the module must work
        # and construction must either succeed or raise a helpful ImportError.
        from intelli.function.browser_env import PlaywrightBrowserEnvironment, _normalize_key

        self.assertEqual(_normalize_key("ctrl+s"), "Control+s")
        self.assertEqual(_normalize_key("Return"), "Enter")
        self.assertEqual(_normalize_key("CTRL+A"), "Control+A")
        self.assertEqual(_normalize_key("Page_Down"), "PageDown")

        try:
            import playwright  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError) as ctx:
                PlaywrightBrowserEnvironment()
            self.assertIn("intelli[computer]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
