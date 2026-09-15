import json
import os
import shutil
import tempfile
import unittest

from intelli.function.coding_agent import CodingAgent


def scripted_chat(replies):
    """Return a chat_fn that plays back canned replies in order."""
    state = {"i": 0}

    def chat(system, history):
        reply = replies[min(state["i"], len(replies) - 1)]
        state["i"] += 1
        return reply

    return chat


class TestCodingAgent(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp()
        with open(os.path.join(self.ws, "calc.py"), "w") as f:
            f.write("def add(a, b):\n    return a - b\n")
        with open(os.path.join(self.ws, "test_calc.py"), "w") as f:
            f.write("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_loop_edits_until_tests_green(self):
        replies = [
            '{"thought": "inspect", "tool": "read_file", "args": {"path": "calc.py"}}',
            '{"thought": "fix", "tool": "edit_file", "args": {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}}',
            '{"thought": "done", "tool": "finish", "args": {"summary": "fixed"}}',
        ]
        agent = CodingAgent(workspace=self.ws, chat_fn=scripted_chat(replies))
        result = agent.run("fix add", test_command="python3 -m pytest -q test_calc.py")

        self.assertTrue(result["success"])
        self.assertEqual(result["iterations"], 3)
        self.assertIn("[exit code 0]", result["test_output"])

    def test_finish_rejected_while_tests_fail(self):
        # The agent tries to finish without fixing; the loop must push back,
        # then accept finish only after the real fix.
        replies = [
            '{"tool": "finish", "args": {"summary": "premature"}}',
            '{"tool": "edit_file", "args": {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}}',
            '{"tool": "finish", "args": {"summary": "actually fixed"}}',
        ]
        agent = CodingAgent(workspace=self.ws, chat_fn=scripted_chat(replies))
        result = agent.run("fix add", test_command="python3 -m pytest -q test_calc.py")

        self.assertTrue(result["success"])
        self.assertEqual(result["summary"], "actually fixed")

    def test_invalid_json_gets_corrective_message(self):
        replies = [
            "I will now fix the bug (no json)",
            '{"tool": "finish", "args": {"summary": "ok"}}',
        ]
        agent = CodingAgent(workspace=self.ws, chat_fn=scripted_chat(replies))
        result = agent.run("say done")  # no test_command -> finish is accepted
        self.assertTrue(result["success"])
        self.assertEqual(result["iterations"], 2)

    def test_max_iterations_cap(self):
        # A model that never finishes must stop at the cap and report failure.
        replies = ['{"tool": "list_files", "args": {}}']
        agent = CodingAgent(workspace=self.ws, chat_fn=scripted_chat(replies), max_iterations=3)
        result = agent.run("loop forever", test_command="python3 -m pytest -q test_calc.py")
        self.assertFalse(result["success"])
        self.assertEqual(result["iterations"], 3)

    def test_action_extraction_variants(self):
        extract = CodingAgent._extract_action
        self.assertEqual(extract('{"tool": "bash", "args": {}}')["tool"], "bash")
        self.assertEqual(
            extract('prose ```json\n{"tool": "search", "args": {"pattern": "x"}}\n``` more')["tool"],
            "search",
        )
        # Multiple objects: pick the one that looks like a tool call.
        self.assertEqual(
            extract('{"note": 1} and {"tool": "read_file", "args": {"path": "a"}}')["tool"],
            "read_file",
        )
        # Literal newlines inside strings must not break parsing.
        self.assertEqual(
            extract('{"tool": "write_file", "args": {"path": "a.py", "content": "l1\nl2"}}')["tool"],
            "write_file",
        )
        # Unbalanced braces INSIDE file content must not corrupt extraction
        # (string-aware decoder). A lone '{' in content:
        act = extract('{"tool": "write_file", "args": {"path": "a.js", "content": "function f() {\n"}}')
        self.assertEqual(act["tool"], "write_file")
        self.assertEqual(act["args"]["content"], "function f() {\n")
        # A lone '}' in content:
        act2 = extract('{"tool": "write_file", "args": {"path": "a.js", "content": "}"}}')
        self.assertEqual(act2["args"]["content"], "}")
        # When a prose example precedes the real action, the LAST tool object wins.
        act3 = extract('Example: {"tool": "bash", "args": {"command": "ls"}}. '
                       'Now: {"tool": "read_file", "args": {"path": "x"}}')
        self.assertEqual(act3["tool"], "read_file")
        self.assertIsNone(extract("no json at all"))


class TestCoderFlowIntegration(unittest.TestCase):
    """The 'coder' agent type must resolve through the flow layer."""

    def test_agent_type_registration(self):
        from intelli.flow.types import AgentTypes, Matcher

        self.assertEqual(AgentTypes.CODER.value, "coder")
        self.assertEqual(Matcher.input["coder"], "text")
        self.assertEqual(Matcher.output["coder"], "text")

    def test_flow_task_runs_coder_handler(self):
        from intelli.flow.agents.agent import Agent
        from intelli.flow.tasks.task import Task
        from intelli.flow.input.task_input import TextTaskInput

        ws = tempfile.mkdtemp()
        try:
            with open(os.path.join(ws, "m.py"), "w") as f:
                f.write("x = 1\n")

            agent = Agent(
                agent_type="coder",
                provider="openai",
                mission="apply the requested change",
                model_params={"workspace": ws, "max_iterations": 3},
            )
            # Inject a scripted chat_fn through the handler by monkeypatching
            # CodingAgent's default builder: simplest is to patch the class method.
            import intelli.function.coding_agent as ca

            original = ca.CodingAgent._build_default_chat_fn
            ca.CodingAgent._build_default_chat_fn = lambda self, *a: scripted_chat(
                ['{"tool": "write_file", "args": {"path": "m.py", "content": "x = 2\\n"}}',
                 '{"tool": "finish", "args": {"summary": "updated m.py"}}']
            )
            try:
                task = Task(TextTaskInput("set x to 2 in m.py"), agent, log=False)
                task.execute()
                output = task.output
            finally:
                ca.CodingAgent._build_default_chat_fn = original

            self.assertIn("succeeded", output)
            with open(os.path.join(ws, "m.py")) as f:
                self.assertEqual(f.read(), "x = 2\n")
        finally:
            shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
