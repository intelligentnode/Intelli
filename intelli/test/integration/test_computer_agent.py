import concurrent.futures
import os
import unittest

from dotenv import load_dotenv

load_dotenv()

# A self-contained page with a button that mutates the DOM when clicked.
PAGE = (
    "data:text/html,<html><body>"
    "<h1 id='status'>ready</h1>"
    "<button style='position:absolute;left:100px;top:150px'"
    " onclick=\"document.getElementById('status').textContent='CLICKED'\">Go</button>"
    "</body></html>"
)


def _playwright_available():
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


class ScriptedAnthropicWrapper:
    """Stubs only the LLM: returns a screenshot action, then a click, then done.

    This lets the REAL browser environment and the REAL agent loop run end-to-end
    without a funded provider key (provider inference is covered by unit tests).
    """

    def __init__(self, click_coord):
        self.n = 0
        self.click_coord = click_coord

    def generate_text(self, params, extra_headers=None):
        self.n += 1
        if self.n == 1:
            return {"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "t1", "name": "computer",
                 "input": {"action": "screenshot"}}]}
        if self.n == 2:
            return {"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "t2", "name": "computer",
                 "input": {"action": "left_click", "coordinate": list(self.click_coord)}}]}
        return {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "clicked the Go button"}]}


@unittest.skipUnless(_playwright_available(),
                     "playwright not installed (pip install intelli[computer])")
class TestComputerAgentBrowserIntegration(unittest.TestCase):
    """Drives a real headless Chromium through the full ComputerAgent loop."""

    def _run_in_thread(self, fn):
        # Sync Playwright must not run inside an asyncio loop; the flow layer runs
        # tasks in an executor thread, so mirror that here.
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(fn).result()

    def test_agent_loop_drives_real_browser(self):
        from intelli.function.browser_env import PlaywrightBrowserEnvironment
        from intelli.function.computer_agent import ComputerAgent

        def task():
            env = PlaywrightBrowserEnvironment(start_url=PAGE, headless=True, width=800, height=600)
            agent = ComputerAgent(api_key="x", provider="anthropic",
                                  model="claude-sonnet-4-6", environment=env)
            agent._wrapper = ScriptedAnthropicWrapper(click_coord=(120, 160))
            try:
                result = agent.run("click the Go button")
                title = env.page.text_content("#status")
                return result, title
            finally:
                env.close()

        result, title = self._run_in_thread(task)
        self.assertTrue(result["success"])
        self.assertEqual(result["output"], "clicked the Go button")
        self.assertEqual(result["actions"], 2)          # screenshot + click executed
        self.assertEqual(title, "CLICKED")               # the real page actually changed

    def test_browser_env_primitive_actions(self):
        from intelli.function.browser_env import PlaywrightBrowserEnvironment

        def task():
            env = PlaywrightBrowserEnvironment(start_url="data:text/html,<h1>A</h1>", headless=True)
            env.page.goto("data:text/html,<h1>B</h1>")
            env.go_back()
            back = env.page.text_content("h1")
            env.go_forward()
            fwd = env.page.text_content("h1")
            png = env.screenshot()
            # exercise the newer primitives added during the security review
            env.scroll(100, 100, scroll_y=120, modifiers=["shift"])
            env.hold_key("shift", 0.02)
            env.mouse_down(10, 10)
            env.mouse_up(20, 20)
            env.close()
            return back, fwd, png[:4]

        back, fwd, header = self._run_in_thread(task)
        self.assertEqual(back, "A")
        self.assertEqual(fwd, "B")
        self.assertEqual(header, b"\x89PNG")

    def test_safety_checks_not_auto_acknowledged(self):
        # SECURITY regression: OpenAI pending_safety_checks must not be auto-approved.
        from intelli.function.browser_env import PlaywrightBrowserEnvironment
        from intelli.function.computer_agent import ComputerAgent

        class SafetyWrapper:
            def generate_gpt5_response(self, params):
                return {"id": "resp_1", "output": [
                    {"type": "computer_call", "call_id": "c1",
                     "actions": [{"type": "click", "button": "left", "x": 1, "y": 1}],
                     "pending_safety_checks": [{"id": "sc1", "code": "sensitive_domain",
                                               "message": "flagged"}]}]}

        def task():
            env = PlaywrightBrowserEnvironment(start_url=PAGE, headless=True)
            agent = ComputerAgent(api_key="x", provider="openai", environment=env)
            agent._wrapper = SafetyWrapper()
            try:
                return agent.run("do something risky")
            finally:
                env.close()

        result = self._run_in_thread(task)
        self.assertFalse(result["success"])
        self.assertEqual(result["pending_safety_checks"][0]["id"], "sc1")


class TestComputerAgentLiveProvider(unittest.TestCase):
    """Optional real-provider run. Off by default (needs a funded key + a browser).

    Enable with:  RUN_LIVE_COMPUTER_USE=1 COMPUTER_USE_PROVIDER=anthropic \
                  COMPUTER_USE_MODEL=claude-sonnet-4-6 python -m unittest ...
    """

    def test_live_provider_navigation(self):
        if os.getenv("RUN_LIVE_COMPUTER_USE") != "1":
            self.skipTest("set RUN_LIVE_COMPUTER_USE=1 to run a real provider computer-use loop")
        if not _playwright_available():
            self.skipTest("playwright not installed")

        provider = os.getenv("COMPUTER_USE_PROVIDER", "anthropic")
        key = os.getenv("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
        if not key:
            self.skipTest(f"no API key for provider {provider}")
        model = os.getenv("COMPUTER_USE_MODEL",
                          "claude-sonnet-4-6" if provider == "anthropic" else "gpt-5.5")

        from intelli.function.browser_env import PlaywrightBrowserEnvironment
        from intelli.function.computer_agent import ComputerAgent

        def task():
            env = PlaywrightBrowserEnvironment(start_url="https://example.com", headless=True)
            agent = ComputerAgent(api_key=key, provider=provider, model=model,
                                  environment=env, max_iterations=8, log=True)
            try:
                return agent.run("Report the main heading text shown on this page.")
            finally:
                env.close()

        result = concurrent.futures.ThreadPoolExecutor().submit(task).result()
        print("live computer-use output:", result.get("output"))
        self.assertIn("output", result)


if __name__ == "__main__":
    unittest.main()
