import os
import asyncio
import unittest
from dotenv import load_dotenv
import json

from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.store.memory import Memory
from intelli.flow.types import AgentTypes
from intelli.flow.utils.flow_helper import FlowHelper

# Load environment variables
load_dotenv()


class TestFlowMemory(unittest.TestCase):
    def setUp(self):
        # Load API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            self.skipTest("API key not available for testing")

        # Create output directory
        self.output_dir = "./temp"
        FlowHelper.ensure_directory(self.output_dir)

        # Create a memory instance
        self.memory = Memory()

        # Populate memory
        self.memory.store("patient_demographics", {
            "patient_id": "12345",
            "age": 65,
            "gender": "Male",
            "admission_diagnosis": "Pneumonia",
            "admission_date": "2023-01-15"
        })

        self.memory.store("lab_data", {
            "patient_id": "12345",
            "labs": [
                {"name": "WBC", "value": 12.5, "unit": "10^3/uL", "time": "2023-01-15 08:30"},
                {"name": "HGB", "value": 10.2, "unit": "g/dL", "time": "2023-01-15 08:30"},
                {"name": "PLT", "value": 145, "unit": "10^3/uL", "time": "2023-01-15 08:30"},
                {"name": "Creatinine", "value": 1.8, "unit": "mg/dL", "time": "2023-01-15 08:30"}
            ]
        })

        self.memory.store("vitals_data", {
            "patient_id": "12345",
            "vitals": [
                {"name": "Heart Rate", "value": 92, "unit": "bpm", "time": "2023-01-15 08:00"},
                {"name": "Blood Pressure", "value": "135/85", "unit": "mmHg", "time": "2023-01-15 08:00"},
                {"name": "Temperature", "value": 38.2, "unit": "C", "time": "2023-01-15 08:00"},
                {"name": "Respiratory Rate", "value": 22, "unit": "bpm", "time": "2023-01-15 08:00"},
                {"name": "SpO2", "value": 94, "unit": "%", "time": "2023-01-15 08:00"}
            ]
        })

        self.memory.store("medication_data", {
            "patient_id": "12345",
            "medications": [
                {"name": "Ceftriaxone", "dose": "1g", "route": "IV", "frequency": "q12h"},
                {"name": "Azithromycin", "dose": "500mg", "route": "IV", "frequency": "q24h"},
                {"name": "Albuterol", "dose": "2.5mg", "route": "Nebulizer", "frequency": "q6h prn"}
            ]
        })

    def test_flow_with_memory(self):
        """Test a flow that uses memory for input and output"""
        asyncio.run(self._run_flow_with_memory())

    async def _run_flow_with_memory(self):
        print("\n--- Testing Flow with Memory ---")

        # Create tasks that reference memory - using single memory key
        lab_analysis_task = Task(
            TextTaskInput("Analyze the patient's laboratory data and identify abnormalities"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Analyze laboratory data for clinically significant abnormalities",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
            memory_key="lab_data"  # Single memory key
        )

        vitals_analysis_task = Task(
            TextTaskInput("Analyze the patient's vital signs and identify abnormalities"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Analyze vital signs for clinically significant abnormalities",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
            memory_key="vitals_data"  # This task will use data from memory with key "vitals_data"
        )

        integration_task = Task(
            TextTaskInput("Integrate the laboratory and vital sign analyses to provide a clinical assessment"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Integrate clinical data analyses into a comprehensive assessment",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True
            # No memory_key, this task will receive output from previous tasks
        )

        # This task uses multiple memory keys
        prediction_task = Task(
            TextTaskInput(
                "Based on the integrated analysis, predict the patient's risk level and expected length of stay"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Predict patient outcomes based on clinical data",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True,
            memory_key=["patient_demographics", "medication_data"]  # Multiple memory keys
        )

        # Create a Flow with memory and output memory mapping
        flow = Flow(
            tasks={
                "lab_analysis": lab_analysis_task,
                "vitals_analysis": vitals_analysis_task,
                "integration": integration_task,
                "prediction": prediction_task,
            },
            map_paths={
                "lab_analysis": ["integration"],
                "vitals_analysis": ["integration"],
                "integration": ["prediction"],
                "prediction": [],
            },
            memory=self.memory, 
            output_memory_map={
                "lab_analysis": "lab_analysis_result",
                "vitals_analysis": "vitals_analysis_result",
                "integration": "integrated_assessment",
                "prediction": "patient_prediction",
            },  
            log=True 
        )

        # Generate and save the flow visualization
        try:
            graph_path = flow.generate_graph_img(
                name="flow_with_memory", save_path=self.output_dir
            )
            print(f"Flow visualization saved to: {graph_path}")
        except Exception as e:
            print(f"Warning: Could not generate graph image: {e}")

        # Execute the flow
        print("\nStarting flow execution...")
        results = await flow.start(max_workers=2)
        print("Flow execution completed")

        # Debug outputs
        print("\nDebug - Memory keys after flow execution:", self.memory.keys())
        print("Debug - Task outputs:", list(results.keys()))

        # Track all memory operations immediately after flow execution
        print("\nVerifying memory contents:")
        for key in ["lab_analysis_result", "vitals_analysis_result", "integrated_assessment", "patient_prediction"]:
            print(f"- Memory key '{key}' exists: {self.memory.has_key(key)}")
            if self.memory.has_key(key):
                value = self.memory.retrieve(key)
                value_type = type(value).__name__
                value_sample = str(value)[:50] + "..." if value else "None"
                print(f"  Value type: {value_type}, Sample: {value_sample}")

        # Add debugging for the prediction task specifically
        print("\nChecking prediction task output:")
        if "prediction" in results:
            prediction_output = results["prediction"]["output"]
            print(f"- Prediction output exists in results: {prediction_output is not None}")
            print(f"- Type: {type(prediction_output).__name__}")
            print(f"- Sample: {str(prediction_output)[:100]}...")
        else:
            print("- Prediction task output not found in results!")

        # Simplified test validation - focus on the most important assertions first
        print("\nValidating results...")
        # Check if tasks executed successfully
        for task_name in ["lab_analysis", "vitals_analysis", "integration", "prediction"]:
            self.assertIn(task_name, results, f"{task_name} output missing from results")

        # Test memory storage - relaxed assertion for debugging
        for key in ["lab_analysis_result", "vitals_analysis_result", "integrated_assessment"]:
            self.assertTrue(self.memory.has_key(key), f"{key} not found in memory")

        # Final test for prediction - this is where it was failing
        self.assertTrue(self.memory.has_key("patient_prediction"), "Patient prediction not stored in memory")

        # Save memory contents to file for inspection
        memory_file_path = os.path.join(self.output_dir, "memory_contents.json")
        with open(memory_file_path, 'w') as f:
            # Only include text-based outputs that can be serialized to JSON
            memory_content = {}
            for k in ["lab_analysis_result", "vitals_analysis_result",
                      "integrated_assessment", "patient_prediction"]:
                if self.memory.has_key(k):
                    try:
                        # Try to make it JSON serializable
                        memory_content[k] = self.memory.retrieve(k)
                        json.dumps(memory_content[k])
                    except (TypeError, OverflowError):
                        print("Warnning: issue with JSON serializable")
                        # If not serializable, store string representation
                        memory_content[k] = str(self.memory.retrieve(k))

            json.dump(memory_content, f, indent=2)

        print(f"\nMemory contents saved to: {memory_file_path}")

        return results

    def test_backward_compatibility(self):
        """Test that backward compatibility is maintained"""
        # Skip if no API key
        if not self.openai_api_key:
            self.skipTest("API key not available for testing")

        # Create tasks without memory references (old-style)
        task1 = Task(
            TextTaskInput("Analyze patient demographics"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Summarize patient demographics",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True
        )

        task2 = Task(
            TextTaskInput("Follow up on demographic analysis"),
            Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Provide clinical context based on demographics",
                model_params={"key": self.openai_api_key, "model": "gpt-4o"},
            ),
            log=True
        )

        # Create a Flow without memory (old-style)
        flow = Flow(
            tasks={
                "demographics": task1,
                "context": task2,
            },
            map_paths={
                "demographics": ["context"],
                "context": [],
            },
            log=True
        )

        asyncio.run(self._run_backward_compatible_flow(flow))

    async def _run_backward_compatible_flow(self, flow):
        print("\n--- Testing Backward Compatibility ---")

        # Execute the flow
        results = await flow.start(max_workers=2)

        # Validate results
        self.assertIn("demographics", results, "Demographics output missing")
        self.assertIn("context", results, "Context output missing")

        print("\nBackward compatibility maintained, flow executed successfully")
        return results


if __name__ == "__main__":
    unittest.main()
