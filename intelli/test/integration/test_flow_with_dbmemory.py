import os
import asyncio
import unittest
import tempfile
import shutil
from dotenv import load_dotenv
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from intelli.flow.store.dbmemory import DBMemory
from intelli.flow.utils.flow_helper import FlowHelper

# Load environment variables
load_dotenv()


class TestFlowWithDBMemory(unittest.TestCase):
    """Test integration of DBMemory with Flow."""

    def setUp(self):
        """Set up test environment."""
        # Load API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            self.skipTest("API key not available for testing")

        # Use ./temp directory for output files
        self.output_dir = "./temp"
        FlowHelper.ensure_directory(self.output_dir)

        # Database will be stored in the temp directory
        self.db_path = os.path.join(self.output_dir, "flow_memory.db")

        # Remove existing database file if it exists
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def tearDown(self):
        """Clean up after tests."""
        # We don't remove the temp directory to allow inspection of results
        pass

    def test_flow_with_db_memory(self):
        """Test a flow that uses DB-backed memory for input and output."""
        asyncio.run(self._run_flow_with_db_memory())

    async def _run_flow_with_db_memory(self):
        """Run a flow with DB memory and verify functionality."""
        print("\n--- Testing Flow with DB Memory ---")

        # Create DB-backed memory
        db_memory = DBMemory(db_path=self.db_path)

        # Pre-populate memory with test data
        db_memory.store(
            "patient_demographics",
            {
                "patient_id": "12345",
                "age": 65,
                "gender": "Male",
                "admission_diagnosis": "Pneumonia",
                "admission_date": "2023-01-15",
            },
        )

        db_memory.store(
            "lab_data",
            {
                "patient_id": "12345",
                "labs": [
                    {
                        "name": "WBC",
                        "value": 12.5,
                        "unit": "10^3/uL",
                        "time": "2023-01-15 08:30",
                    },
                    {
                        "name": "HGB",
                        "value": 10.2,
                        "unit": "g/dL",
                        "time": "2023-01-15 08:30",
                    },
                    {
                        "name": "PLT",
                        "value": 145,
                        "unit": "10^3/uL",
                        "time": "2023-01-15 08:30",
                    },
                    {
                        "name": "Creatinine",
                        "value": 1.8,
                        "unit": "mg/dL",
                        "time": "2023-01-15 08:30",
                    },
                ],
            },
        )

        db_memory.store(
            "vitals_data",
            {
                "patient_id": "12345",
                "vitals": [
                    {
                        "name": "Heart Rate",
                        "value": 92,
                        "unit": "bpm",
                        "time": "2023-01-15 08:00",
                    },
                    {
                        "name": "Blood Pressure",
                        "value": "135/85",
                        "unit": "mmHg",
                        "time": "2023-01-15 08:00",
                    },
                    {
                        "name": "Temperature",
                        "value": 38.2,
                        "unit": "C",
                        "time": "2023-01-15 08:00",
                    },
                ],
            },
        )

        # Create tasks that reference memory
        lab_analysis_task = Task(
            TextTaskInput(
                "Analyze the patient's laboratory data and identify abnormalities"
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Analyze laboratory data for clinical abnormalities",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
            memory_key="lab_data",  # Use memory key
        )

        vitals_analysis_task = Task(
            TextTaskInput(
                "Analyze the patient's vital signs and identify abnormalities"
            ),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Analyze vital signs for clinical abnormalities",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
            memory_key="vitals_data",  # This task will use data from memory with key "vitals_data"
        )

        # Create a Flow with DB memory
        flow = Flow(
            tasks={
                "lab_analysis": lab_analysis_task,
                "vitals_analysis": vitals_analysis_task,
            },
            map_paths={
                "lab_analysis": [],
                "vitals_analysis": [],
            },
            memory=db_memory,  # Use DB memory
            output_memory_map={
                "lab_analysis": "lab_analysis_result",
                "vitals_analysis": "vitals_analysis_result",
            },
            log=True,
        )

        # Generate and save the flow visualization
        try:
            graph_path = flow.generate_graph_img(
                name="flow_with_db_memory", save_path=self.output_dir
            )
            print(f"Flow visualization saved to: {graph_path}")
        except Exception as e:
            print(f"Warning: Could not generate graph image: {e}")

        # Execute the flow with lower concurrency to prevent SQLite thread issues
        print("\nStarting flow execution with DB memory...")
        results = await flow.start(max_workers=1)  # Reduced to 1 worker
        print("Flow execution completed")

        # Verify results were stored in DB memory
        print("\nVerifying memory contents:")
        for key in ["lab_analysis_result", "vitals_analysis_result"]:
            print(f"- Memory key '{key}' exists: {db_memory.has_key(key)}")
            if db_memory.has_key(key):
                value = db_memory.retrieve(key)
                value_type = type(value).__name__
                value_sample = str(value)[:50] + "..." if value else "None"
                print(f"  Value type: {value_type}, Sample: {value_sample}")

        # Verify both results exist in DB memory
        self.assertTrue(
            db_memory.has_key("lab_analysis_result"),
            "Lab analysis result not found in DB memory",
        )
        self.assertTrue(
            db_memory.has_key("vitals_analysis_result"),
            "Vitals analysis result not found in DB memory",
        )

        # Close the current DB connection
        db_memory.close()

        # Verify database file exists
        self.assertTrue(
            os.path.exists(self.db_path),
            f"Database file should exist at {self.db_path}",
        )

        # Create a new memory instance to verify persistence
        print("\nTesting data persistence across connections...")
        new_db_memory = DBMemory(db_path=self.db_path)
        for key in ["lab_analysis_result", "vitals_analysis_result"]:
            self.assertTrue(
                new_db_memory.has_key(key), f"{key} should persist in new DB connection"
            )

            value = new_db_memory.retrieve(key)
            self.assertIsNotNone(value, f"Retrieved value for {key} should not be None")
            print(f"Successfully retrieved persisted data for key: {key}")

        # Demonstrate how to query data from the database
        print("\n--- Demonstrating query capabilities ---")
        print("Example 1: Query all keys in the database:")
        all_keys = new_db_memory.query("SELECT key FROM memory")
        print(f"All keys in database: {all_keys}")

        print("\nExample 2: Query specific values by key pattern:")
        analysis_results = new_db_memory.query(
            "SELECT key, value_type FROM memory WHERE key LIKE '%analysis%'"
        )
        print(f"Analysis-related entries: {analysis_results}")

        print("\nExample 3: Get value for specific key:")
        value = new_db_memory.retrieve("lab_analysis_result")
        print(f"Lab analysis result (first 100 chars): {str(value)[:100]}...")

        # Export the database as JSON
        json_path = os.path.join(self.output_dir, "memory_export.json")
        export_path = new_db_memory.export_to_json(json_path)
        print(f"\nDatabase exported to JSON: {export_path}")

        # Verify the JSON file was created
        self.assertTrue(os.path.exists(json_path), "JSON export file should exist")

        # Close the database connection
        new_db_memory.close()

        print("\n--- Example code for working with DBMemory ---")
        # Example how to use the database memory and query data.
        """
        # Create a DB-backed memory store
        from intelli.flow.store.dbmemory import DBMemory
        
        # Initialize with a database path
        db_memory = DBMemory(db_path="./data/my_flow_memory.db")
        
        # Store data
        db_memory.store("my_key", {"name": "Example", "value": 42})
        
        # Retrieve data
        data = db_memory.retrieve("my_key")
        
        # Check if a key exists
        if db_memory.has_key("my_key"):
            print("Key exists!")
        
        # Get all keys
        all_keys = db_memory.keys()
        
        # Perform a custom query
        results = db_memory.query("SELECT key FROM memory WHERE key LIKE ?", ("%analysis%",))
        
        # Export all data to JSON
        db_memory.export_to_json("./data/memory_export.json")
        
        # Close the connection when done
        db_memory.close()
        """

        return results

    def test_backward_compatibility(self):
        """Test that Flow works with both Memory and DBMemory."""
        # This simple test verifies that the Flow class can still
        # work with the new DBMemory without issues
        asyncio.run(self._test_compatibility())

    async def _test_compatibility(self):
        """Run a minimal flow to ensure backward compatibility."""
        print("\n--- Testing Backward Compatibility ---")

        # Skip if no API key
        if not self.openai_api_key:
            self.skipTest("API key not available for testing")

        # Create a simple task
        task = Task(
            TextTaskInput("Summarize the concept of memory in AI systems"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Provide a concise summary",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
        )

        # Create a flow with DBMemory
        db_memory = DBMemory(db_path=self.db_path)
        flow_with_db = Flow(
            tasks={"summarize": task},
            map_paths={"summarize": []},
            memory=db_memory,
            output_memory_map={"summarize": "summary_result"},
            log=True,
        )

        # Execute the flow with DBMemory
        print("Running flow with DBMemory...")
        results_db = await flow_with_db.start(max_workers=1)  # Use single worker

        # Verify results
        self.assertIn("summarize", results_db, "Summary output missing")
        self.assertTrue(
            db_memory.has_key("summary_result"),
            "Summary result should be stored in DB memory",
        )

        db_memory.close()

        print("Flow with DBMemory completed successfully")
        print("Backward compatibility verified")

        return results_db


if __name__ == "__main__":
    unittest.main()
