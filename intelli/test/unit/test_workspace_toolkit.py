import os
import shutil
import tempfile
import unittest

from intelli.function.workspace_toolkit import WorkspaceToolkit


class TestWorkspaceToolkit(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp()
        with open(os.path.join(self.ws, "hello.py"), "w") as f:
            f.write("def hello():\n    return 'world'\n")
        self.toolkit = WorkspaceToolkit(self.ws)

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def test_read_write_edit(self):
        self.assertIn("world", self.toolkit.read_file("hello.py"))

        self.toolkit.write_file("pkg/new.py", "x = 1\n")
        self.assertEqual(self.toolkit.read_file("pkg/new.py"), "x = 1\n")

        result = self.toolkit.edit_file("hello.py", "return 'world'", "return 'earth'")
        self.assertNotIn("Error", result)
        self.assertIn("earth", self.toolkit.read_file("hello.py"))

    def test_edit_requires_unique_match(self):
        self.toolkit.write_file("dup.py", "a = 1\na = 1\n")
        self.assertIn("Error", self.toolkit.edit_file("dup.py", "a = 1", "a = 2"))
        self.assertIn("Error", self.toolkit.edit_file("dup.py", "not-present", "x"))

    def test_path_confinement(self):
        # Escaping the workspace must be rejected for every path-taking tool.
        with self.assertRaises(ValueError):
            self.toolkit.read_file("../outside.txt")
        with self.assertRaises(ValueError):
            self.toolkit.write_file("/etc/intelli_test.txt", "x")
        # And via the dispatch facade it surfaces as an error string, not a crash.
        self.assertIn("Error", self.toolkit.execute("read_file", {"path": "../../etc/passwd"}))

    def test_list_and_search(self):
        listing = self.toolkit.list_files()
        self.assertIn("hello.py", listing)
        matches = self.toolkit.search(r"def hello")
        self.assertIn("hello.py:1", matches)
        self.assertEqual(self.toolkit.search(r"no_such_symbol_xyz"), "No matches found.")

    def test_bash_execution_and_disable(self):
        out = self.toolkit.run_bash("echo intelli_ok")
        self.assertIn("[exit code 0]", out)
        self.assertIn("intelli_ok", out)

        locked = WorkspaceToolkit(self.ws, allow_bash=False)
        self.assertIn("Error", locked.run_bash("echo nope"))

    def test_dispatch_unknown_tool(self):
        self.assertIn("unknown tool", self.toolkit.execute("teleport", {}))


if __name__ == "__main__":
    unittest.main()
