"""
WorkspaceToolkit: local file and shell tools for coding agents.

All paths are confined to the workspace directory - the toolkit refuses to
read or write anything outside it. Used by CodingAgent and the flow 'coder'
agent type, but usable standalone as well.
"""

import os
import re
import subprocess


class WorkspaceToolkit:
    """File/search/shell tools scoped to a single workspace directory."""

    def __init__(self, workspace_dir, allow_bash=True, bash_timeout=120,
                 max_read_chars=60000, max_output_chars=20000):
        self.workspace = os.path.realpath(workspace_dir)
        if not os.path.isdir(self.workspace):
            raise ValueError(f"Workspace directory does not exist: {workspace_dir}")
        self.allow_bash = allow_bash
        self.bash_timeout = bash_timeout
        self.max_read_chars = max_read_chars
        self.max_output_chars = max_output_chars
        # Directories skipped when listing/searching.
        self._skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea"}

    # ---- internals ----
    def _resolve(self, path):
        """Resolve a relative path inside the workspace; reject escapes."""
        candidate = os.path.realpath(os.path.join(self.workspace, path or ""))
        if candidate != self.workspace and not candidate.startswith(self.workspace + os.sep):
            raise ValueError(f"Path escapes the workspace: {path}")
        return candidate

    def _truncate(self, text, limit):
        if len(text) > limit:
            return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"
        return text

    def _iter_files(self):
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in self._skip_dirs]
            for f in files:
                yield os.path.join(root, f)

    # ---- tools ----
    def read_file(self, path):
        full = self._resolve(path)
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return self._truncate(f.read(), self.max_read_chars)

    def write_file(self, path, content):
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full) or self.workspace, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"

    def edit_file(self, path, old_text, new_text):
        """Exact, unique string replacement (safer than regex for code edits)."""
        full = self._resolve(path)
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        count = content.count(old_text)
        if count == 0:
            return f"Error: old_text not found in {path}"
        if count > 1:
            return f"Error: old_text appears {count} times in {path}; provide a unique snippet"
        with open(full, "w", encoding="utf-8") as f:
            f.write(content.replace(old_text, new_text))
        return f"Edited {path}"

    def list_files(self, max_files=200):
        rels = []
        for full in self._iter_files():
            rels.append(os.path.relpath(full, self.workspace))
            if len(rels) >= max_files:
                rels.append("... [more files omitted]")
                break
        return "\n".join(sorted(rels)) if rels else "(empty workspace)"

    def search(self, pattern, max_results=50):
        """Regex search across workspace files; returns file:line: text matches."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        results = []
        for full in self._iter_files():
            rel = os.path.relpath(full, self.workspace)
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                            if len(results) >= max_results:
                                return "\n".join(results + ["... [more results omitted]"])
            except (OSError, UnicodeDecodeError):
                continue
        return "\n".join(results) if results else "No matches found."

    def run_bash(self, command):
        """Run a shell command in the workspace; returns exit code + output."""
        if not self.allow_bash:
            return "Error: bash execution is disabled for this agent"
        try:
            # Disable .pyc caching: rapid edit->test cycles within the same
            # mtime second would otherwise run stale bytecode and report
            # failures for code that is already fixed.
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.run(
                command, shell=True, cwd=self.workspace, env=env,
                capture_output=True, text=True, timeout=self.bash_timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return f"[exit code {proc.returncode}]\n{self._truncate(output, self.max_output_chars)}"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {self.bash_timeout}s"

    # ---- dispatch used by the agent loop ----
    def execute(self, tool, args):
        """Execute a named tool with dict args; always returns a string observation."""
        args = args or {}
        try:
            if tool == "read_file":
                return self.read_file(args["path"])
            if tool == "write_file":
                return self.write_file(args["path"], args.get("content", ""))
            if tool == "edit_file":
                return self.edit_file(args["path"], args.get("old_text", ""), args.get("new_text", ""))
            if tool == "list_files":
                return self.list_files()
            if tool == "search":
                return self.search(args.get("pattern", ""))
            if tool == "bash":
                return self.run_bash(args.get("command", ""))
            return f"Error: unknown tool '{tool}'"
        except KeyError as e:
            return f"Error: missing required argument {e} for tool '{tool}'"
        except Exception as e:
            return f"Error: {e}"
