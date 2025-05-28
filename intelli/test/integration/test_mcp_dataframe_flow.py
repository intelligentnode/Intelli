"""
Integration Test for MCP DataFrame Flow.

This test suite verifies the functionality of the MCP DataFrame server by
launching it as a subprocess and interacting with it using an MCP Agent.
It tests common DataFrame operations like get_head, get_schema, get_shape,
select_columns, and filter_rows.

The test relies on the `mcp_dataframe_server.py` script and the
`sample_data.csv` file located in the `test/integration/data` directory.
"""
import unittest
import os
import sys
import json
import subprocess
import time
from typing import Dict, Any, List, Optional

# Adjust path to import from the parent directory (Intelli root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add 'Intelli' to sys.path
intelli_root_path = parent_dir
sys.path.insert(0, intelli_root_path)

from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from intelli.mcp.dataframe_utils import PANDAS_AVAILABLE, POLARS_AVAILABLE


# Skip tests if neither pandas nor polars is installed
SKIP_TESTS = not (PANDAS_AVAILABLE or POLARS_AVAILABLE)
SKIP_REASON = "Neither pandas nor polars is available. Skipping MCP DataFrame tests."

@unittest.skipIf(SKIP_TESTS, SKIP_REASON)
class TestMCPDataFrameFlow(unittest.TestCase):
    server_process: Optional[subprocess.Popen] = None
    server_path: str
    output_dir: str = os.path.join(current_dir, "temp", "mcp_dataframe")

    @classmethod
    def setUpClass(cls):
        """Set up for the test class. Starts the MCP DataFrame server."""
        cls.server_path = os.path.join(current_dir, "mcp_dataframe_server.py")
        
        if not os.path.exists(cls.server_path):
            raise FileNotFoundError(f"MCP DataFrame server script not found at {cls.server_path}")

        # Ensure data directory and sample_data.csv exist
        data_dir = os.path.join(current_dir, "data")
        sample_csv = os.path.join(data_dir, "sample_data.csv")
        if not os.path.exists(sample_csv):
            raise FileNotFoundError(
                f"Sample CSV file not found at {sample_csv}. "
                "This test requires 'sample_data.csv' in the 'test/integration/data' directory."
            )
        
        os.makedirs(cls.output_dir, exist_ok=True)
        
        try:
            print(f"Starting MCP DataFrame server: {sys.executable} {cls.server_path}")
            cls.server_process = subprocess.Popen(
                [sys.executable, cls.server_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.PIPE, text=True
            )
            # Wait for the server to initialize
            # You might need to adjust this based on server startup time and output
            time.sleep(3) # Give it a few seconds to start and print its info
            
            # Check if server started successfully
            if cls.server_process.poll() is not None:
                stdout, stderr = cls.server_process.communicate()
                print("Server Process STDOUT:", stdout)
                print("Server Process STDERR:", stderr)
                raise RuntimeError(f"MCP DataFrame server failed to start. Exit code: {cls.server_process.poll()}")
            print("MCP DataFrame server started successfully.")

        except Exception as e:
            print(f"Error starting MCP DataFrame server: {e}")
            if cls.server_process:
                cls.server_process.terminate()
                cls.server_process.wait()
            raise

    @classmethod
    def tearDownClass(cls):
        """Tear down for the test class. Stops the MCP DataFrame server."""
        if cls.server_process:
            print("Terminating MCP DataFrame server...")
            cls.server_process.terminate()
            try:
                stdout, stderr = cls.server_process.communicate(timeout=5)
                print("Server STDOUT on termination:", stdout)
                print("Server STDERR on termination:", stderr)
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, killing...")
                cls.server_process.kill()
                stdout, stderr = cls.server_process.communicate()
                print("Server STDOUT on kill:", stdout)
                print("Server STDERR on kill:", stderr)
            print("MCP DataFrame server terminated.")

    def _create_mcp_agent(self) -> Agent:
        """Helper to create an MCP agent configured for the test server."""
        return Agent(
            agent_type=AgentTypes.MCP.value,
            provider="mcp",
            mission="Interact with DataFrame MCP server",
            model_params={
                "command": sys.executable, # This tells MCP to use stdio
                "args": [self.server_path], # Path to the server script
                "tool": "get_head" # Default tool, will be overridden in tasks
            }
        )

    def _run_mcp_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Helper function to run a specific MCP tool and get its output."""
        mcp_agent = self._create_mcp_agent()
        
        # Update agent's model_params for the specific tool call
        updated_params = mcp_agent.model_params.copy()
        updated_params["tool"] = tool_name
        for key, value in tool_args.items():
            updated_params[f"arg_{key}"] = value
        
        # Set the updated parameters in the agent
        mcp_agent.model_params = updated_params
        
        # Execute the agent's tool call
        print(f"Executing tool '{tool_name}' with args: {tool_args}")
        # TextTaskInput is not really used by MCP agents but required by the agent.execute interface
        result = mcp_agent.execute(TextTaskInput("Calling MCP tool"))
        print(f"Raw result from tool '{tool_name}': {result}")
        
        # MCP tools in our DataFrame server return JSON strings or dicts for schema/shape
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                # If it's not JSON, it might be an error message string
                return result 
        return result # For schema/shape which return dicts

    def test_01_get_head(self):
        """Test the get_head tool."""
        print("\n--- Testing get_head ---")
        result = self._run_mcp_tool("get_head", {"n": 3})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIn("ID", result[0])
        self.assertIn("Name", result[0])
        self.assertEqual(result[0]["ID"], 1)
        self.assertEqual(result[0]["Name"], "Alice")

    def test_02_get_schema(self):
        """Test the get_schema tool."""
        print("\n--- Testing get_schema ---")
        result = self._run_mcp_tool("get_schema", {})
        self.assertIsInstance(result, dict)
        self.assertIn("ID", result)
        self.assertIn("Name", result)
        self.assertIn("Age", result)
        self.assertIn("City", result)
        self.assertIn("Salary", result)
        # Exact type strings can vary (e.g. 'int64' vs 'Int64' vs 'integer')
        # So we check for presence and general type categories
        self.assertTrue("int" in result["ID"].lower() or "Int" in result["ID"])
        self.assertTrue("object" in result["Name"].lower() or "str" in result["Name"].lower() or "Utf8" in result["Name"])
        self.assertTrue("int" in result["Salary"].lower() or "Int" in result["Salary"])

    def test_03_get_shape(self):
        """Test the get_shape tool."""
        print("\n--- Testing get_shape ---")
        result = self._run_mcp_tool("get_shape", {})
        self.assertIsInstance(result, dict)
        self.assertIn("rows", result)
        self.assertIn("columns", result)
        self.assertEqual(result["rows"], 15) # Based on sample_data.csv
        self.assertEqual(result["columns"], 5) # Based on sample_data.csv

    def test_04_select_columns(self):
        """Test the select_columns tool."""
        print("\n--- Testing select_columns ---")
        selected_cols = ["Name", "Salary"]
        result = self._run_mcp_tool("select_columns", {"columns": selected_cols})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 15) # All rows
        self.assertIn("Name", result[0])
        self.assertIn("Salary", result[0])
        self.assertNotIn("ID", result[0])
        self.assertNotIn("Age", result[0])
        self.assertNotIn("City", result[0])
        self.assertEqual(result[0]["Name"], "Alice")

    def test_05_filter_rows_equal_string(self):
        """Test filter_rows with string equality."""
        print("\n--- Testing filter_rows (City == New York) ---")
        result = self._run_mcp_tool("filter_rows", {"column": "City", "operator": "==", "value": "New York"})
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        for row in result:
            self.assertEqual(row["City"], "New York")
        self.assertEqual(len(result), 5) # Expected count for New York

    def test_06_filter_rows_greater_than_int(self):
        """Test filter_rows with integer greater than."""
        print("\n--- Testing filter_rows (Age > 35) ---")
        result = self._run_mcp_tool("filter_rows", {"column": "Age", "operator": ">", "value": 35})
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        for row in result:
            self.assertGreater(row["Age"], 35)
        self.assertEqual(len(result), 4) # Eve (40), Hank (45), Karen (38), Nick (42)

    def test_07_filter_rows_contains_string(self):
        """Test filter_rows with string contains."""
        print("\n--- Testing filter_rows (Name contains 'li') ---")
        result = self._run_mcp_tool("filter_rows", {"column": "Name", "operator": "contains", "value": "li"})
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        for row in result:
            self.assertIn("li", row["Name"].lower())
        # Alice, Charlie, Olivia
        self.assertEqual(len(result), 3)
        
    def test_08_filter_rows_in_list_int(self):
        """Test filter_rows with 'in' list of integers."""
        print("\n--- Testing filter_rows (ID in [1, 5, 10]) ---")
        id_list = [1, 5, 10]
        result = self._run_mcp_tool("filter_rows", {"column": "ID", "operator": "in", "value": id_list})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for row in result:
            self.assertIn(row["ID"], id_list)

    def test_09_filter_rows_no_match(self):
        """Test filter_rows with a condition that yields no results."""
        print("\n--- Testing filter_rows (Salary < 10000) ---")
        result = self._run_mcp_tool("filter_rows", {"column": "Salary", "operator": "<", "value": 10000})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_10_select_non_existent_column(self):
        """Test select_columns with a non-existent column."""
        print("\n--- Testing select_columns (NonExistentColumn) ---")
        result = self._run_mcp_tool("select_columns", {"columns": ["Name", "NonExistentColumn"]})
        self.assertIsInstance(result, str) # Expecting an error message string
        self.assertIn("Error: The following columns were not found", result)
        self.assertIn("NonExistentColumn", result)

    def test_11_filter_non_existent_column(self):
        """Test filter_rows with a non-existent column."""
        print("\n--- Testing filter_rows (NonExistentColumn > 10) ---")
        result = self._run_mcp_tool("filter_rows", {"column": "NonExistentColumn", "operator": ">", "value": 10})
        self.assertIsInstance(result, str)
        self.assertIn("Error: Column 'NonExistentColumn' not found", result)

    def test_12_filter_invalid_operator(self):
        """Test filter_rows with an invalid operator."""
        print("\n--- Testing filter_rows (Age ** 2) ---") # Using an unsupported operator
        result = self._run_mcp_tool("filter_rows", {"column": "Age", "operator": "**", "value": 2})
        self.assertIsInstance(result, str)
        self.assertIn("Error: Unsupported operator '**'", result)


if __name__ == "__main__":
    if SKIP_TESTS:
        print(SKIP_REASON)
    else:
        print("Running MCP DataFrame Flow tests...")
        unittest.main() 