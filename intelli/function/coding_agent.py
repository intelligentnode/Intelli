"""
CodingAgent: an autonomous coding loop over a workspace (SWE-agent pattern).

The agent talks to any intelli chat provider through a JSON action protocol:
each model reply is one JSON object choosing a tool (read_file, write_file,
edit_file, list_files, search, bash, finish). Tool observations are fed back
as user messages until the task is done. When a test_command is provided the
agent keeps iterating until the tests pass (exit code 0) or max_iterations.

Usage:
    agent = CodingAgent(api_key=KEY, provider="gemini", workspace="./repo")
    result = agent.run("Fix the divide() bug", test_command="python -m pytest -q")
"""

import json

from intelli.function.workspace_toolkit import WorkspaceToolkit

SYSTEM_PROMPT = """You are an autonomous coding agent working inside a repository workspace.

You interact ONLY by replying with a single JSON object per turn (no prose outside it):
{"thought": "<brief reasoning>", "tool": "<tool_name>", "args": { ... }}

Tools:
- read_file   {"path": "relative/path"}
- write_file  {"path": "relative/path", "content": "full file content"}
- edit_file   {"path": "relative/path", "old_text": "unique snippet", "new_text": "replacement"}
- list_files  {}
- search      {"pattern": "regex"}
- bash        {"command": "shell command"}
- finish      {"summary": "what you changed and why"}

Rules:
- One tool call per reply. Read code before editing it. Prefer minimal edits.
- old_text for edit_file must match the file content exactly and be unique.
- Call finish only when the task is complete."""


class CodingAgent:
    """Provider-agnostic coding agent that edits a workspace until done/green."""

    def __init__(self, api_key=None, provider="openai", workspace=".", model=None,
                 options=None, allow_bash=True, bash_timeout=120,
                 max_iterations=20, chat_fn=None, log=False):
        """
        Args:
            api_key/provider/model/options: forwarded to the intelli Chatbot.
            workspace: directory the agent operates in (all paths confined to it).
            allow_bash/bash_timeout: shell execution policy.
            max_iterations: hard cap on agent turns.
            chat_fn: optional override callable(system, history)->str used for
                testing or custom model backends; history is [(role, text), ...].
            log: print each action when True.
        """
        self.toolkit = WorkspaceToolkit(workspace, allow_bash=allow_bash, bash_timeout=bash_timeout)
        self.max_iterations = max_iterations
        self.log = log
        self._chat_fn = chat_fn or self._build_default_chat_fn(api_key, provider, model, options)

    def _build_default_chat_fn(self, api_key, provider, model, options):
        # Local import keeps this module importable without provider deps.
        from intelli.function.chatbot import Chatbot
        from intelli.model.input.chatbot_input import ChatModelInput

        chatbot = Chatbot(api_key, provider, options or {})

        def chat(system, history):
            chat_input = ChatModelInput(system, model=model, temperature=0.2, max_tokens=4096)
            for role, text in history:
                if role == "user":
                    chat_input.add_user_message(text)
                else:
                    chat_input.add_assistant_message(text)
            response = chatbot.chat(chat_input)
            first = response[0] if isinstance(response, list) and response else response
            return first if isinstance(first, str) else json.dumps(first)

        return chat

    # strict=False tolerates literal newlines/tabs inside JSON strings, which
    # models routinely emit for multi-line file content.
    _decoder = json.JSONDecoder(strict=False)

    @classmethod
    def _extract_action(cls, text):
        """Extract the intended tool-call JSON object from a model reply.

        Scans from each '{' with a real JSON decoder, which is string- and
        brace-aware: braces that appear inside file content (a very common case
        for write_file) no longer corrupt parsing. When several JSON objects are
        present (e.g. a prose example plus the real action), the LAST tool-bearing
        object wins, since models emit their chosen action after any reasoning.
        """
        if not text:
            return None
        found = None
        i, n = 0, len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            try:
                obj, end = cls._decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if isinstance(obj, dict) and "tool" in obj:
                found = obj  # keep scanning; last match wins
            i = max(end, i + 1)  # skip past the parsed object
        return found

    def run(self, task, test_command=None):
        """
        Execute the coding task. Returns:
            {"success": bool, "summary": str, "iterations": int, "test_output": str|None}
        """
        history = [("user", f"Task: {task}\n\nWorkspace files:\n{self.toolkit.list_files()}")]
        test_output = None

        for iteration in range(1, self.max_iterations + 1):
            reply = self._chat_fn(SYSTEM_PROMPT, history)
            history.append(("assistant", reply))

            action = self._extract_action(reply)
            if action is None:
                history.append(("user", "Reply with a single valid JSON object using the documented format."))
                continue

            tool = action.get("tool")
            args = action.get("args") or {}
            if self.log:
                print(f"[coder iter {iteration}] {tool} {args if tool != 'write_file' else {'path': args.get('path')}}")

            if tool == "finish":
                summary = args.get("summary") or action.get("summary") or "Task finished."
                if test_command:
                    test_output = self.toolkit.run_bash(test_command)
                    if test_output.startswith("[exit code 0]"):
                        return {"success": True, "summary": summary, "iterations": iteration, "test_output": test_output}
                    # Tests still failing: push the output back and keep iterating.
                    history.append(("user", f"Tests are still failing. Fix them before finishing.\n{test_output}"))
                    continue
                return {"success": True, "summary": summary, "iterations": iteration, "test_output": None}

            observation = self.toolkit.execute(tool, args)
            history.append(("user", f"[{tool} result]\n{observation}"))

        # Iteration budget exhausted; report the final test state if applicable.
        if test_command and test_output is None:
            test_output = self.toolkit.run_bash(test_command)
        success = bool(test_command and test_output and test_output.startswith("[exit code 0]"))
        return {"success": success, "summary": "Stopped: max iterations reached.",
                "iterations": self.max_iterations, "test_output": test_output}
