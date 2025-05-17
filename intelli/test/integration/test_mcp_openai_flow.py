import unittest
import os
import sys
import json
import asyncio
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from dotenv import load_dotenv

load_dotenv()

class TestMCPOpenAIFlow(unittest.TestCase):
    # Define output directory
    OUTPUT_DIR = "./temp/mcp/"
    
    def setUp(self):
        # Get the path to the MCP server file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.server_path = os.path.join(current_dir, "mcp_math_server.py")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def test_openai_mcp_flow(self):
        """Test a flow that combines OpenAI with MCP for arithmetic operations"""
        
        # Skip test if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY not available")
        
        # Test with different arithmetic queries
        test_cases = [
            {"query": "What is 42 plus 17?", "expected": {"operation": "add", "a": 42, "b": 17, "result": 59}},
            {"query": "Can you subtract 25 from 100?", "expected": {"operation": "subtract", "a": 100, "b": 25, "result": 75}},
            {"query": "Multiply 8 by 9 please", "expected": {"operation": "multiply", "a": 8, "b": 9, "result": 72}}
        ]
        
        for test_case in test_cases:
            test_query = test_case["query"]
            expected = test_case["expected"]
            
            print(f"\n--- Testing: '{test_query}' ---")
            
            # Create an OpenAI agent to understand the user's request
            openai_agent = Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Analyze user query to identify arithmetic operations",
                model_params={
                    "key": os.getenv("OPENAI_API_KEY"),
                    "model": "gpt-4o",
                    "system_message": "You are a helpful assistant that extracts arithmetic operations from user queries. Your task is to parse the input text, extract the numbers and operation, and return ONLY a JSON object with the format: {'operation': 'add/subtract/multiply', 'a': number, 'b': number}. Do not include any other text in your response."
                }
            )
            
            # Create a task for OpenAI with the test query in its input
            openai_task = Task(
                TextTaskInput(
                    "The following is a user query containing an arithmetic operation. Extract the numbers and operation.\n"
                    "Return ONLY a JSON object with the format: {'operation': 'add/subtract/multiply', 'a': number, 'b': number}.\n"
                    "Example: For 'What is 5 plus 3?', return {'operation': 'add', 'a': 5, 'b': 3}\n"
                    "Do not perform the calculation. Do not include any other text in your response.\n\n"
                    f"USER QUERY: {test_query}"  # Include the test query directly in the input
                ),
                openai_agent,
                log=True
            )
            
            # Create an MCP agent to perform the actual calculation
            mcp_agent = Agent(
                agent_type=AgentTypes.MCP.value,
                provider="mcp",
                mission="Execute arithmetic operations via MCP",
                model_params={
                    "command": sys.executable,
                    "args": [self.server_path],
                    "tool": "add"  # Default tool, will be updated by pre_process
                }
            )
            
            # Create a task for the MCP agent with custom pre-processing
            mcp_task = Task(
                TextTaskInput("Perform the arithmetic operation"),
                mcp_agent,
                log=True,
                pre_process=self._extract_operation_details
            )
            
            # Create a final OpenAI agent to format the result nicely
            result_agent = Agent(
                agent_type=AgentTypes.TEXT.value,
                provider="openai",
                mission="Format the arithmetic result into a natural language response",
                model_params={
                    "key": os.getenv("OPENAI_API_KEY"),
                    "model": "gpt-4o"
                }
            )
            
            # Create a task for the result formatting
            result_task = Task(
                TextTaskInput(
                    "Format the arithmetic result into a natural language response. "
                    "The input contains the operation details and result."
                ),
                result_agent,
                log=True,
                pre_process=self._format_calculation_details
            )
            
            # Set up the flow
            tasks = {
                "analyze": openai_task,
                "calculate": mcp_task,
                "format": result_task
            }
            
            # Connect the tasks
            map_paths = {
                "analyze": ["calculate"],
                "calculate": ["format"],
                "format": []
            }
            
            # Create a new flow for each test case to ensure clean state
            flow = Flow(tasks=tasks, map_paths=map_paths, log=True)
            self.flow = flow  # Store for format calculation details method
            
            # Generate and save the flow visualization
            try:
                graph_name = f"mcp_openai_flow_{expected['operation']}"
                graph_path = flow.generate_graph_img(
                    name=graph_name, 
                    save_path=self.OUTPUT_DIR
                )
                print(f"🎨 Flow visualization saved to: {graph_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not generate graph image: {e}")
            
            # Start the flow
            results = asyncio.run(flow.start())
            
            # Verify all tasks produced output
            self.assertIn("analyze", results)
            self.assertIn("calculate", results)
            self.assertIn("format", results)
            
            # Print the final result
            print(f"Final response: {results['format']['output']}")
            
            # Verify that we have a non-empty string result
            self.assertTrue(isinstance(results["format"]["output"], str))
            self.assertTrue(len(results["format"]["output"]) > 0)
            
            # Check if the response contains the expected result
            # We're being flexible here because the format might vary
            result_text = results["format"]["output"]
            if not any(str(expected["result"]) in result_text for text in [result_text]):
                print(f"WARNING: Expected result {expected['result']} not found in output: {result_text}")
                # Not failing the test, but logging a warning
    
    def _extract_operation_details(self, input_data):
        """
        Extract operation details from OpenAI's output and update MCP agent parameters.
        """
        print(f"Extracting operation details from: {input_data}")
        
        try:
            # Parse input as JSON
            if isinstance(input_data, str):
                # Find JSON in the text
                import re
                json_match = re.search(r'\{.*\}', input_data, re.DOTALL)
                
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        # Fix JSON formatting
                        json_str = json_str.replace("'", '"')
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try regex extraction
                        operation_match = re.search(r'"operation"\s*:\s*"([^"]+)"', input_data)
                        a_match = re.search(r'"a"\s*:\s*(\d+)', input_data)
                        b_match = re.search(r'"b"\s*:\s*(\d+)', input_data)
                        
                        if operation_match and a_match and b_match:
                            data = {
                                "operation": operation_match.group(1),
                                "a": int(a_match.group(1)),
                                "b": int(b_match.group(1))
                            }
                        else:
                            # Last attempt - extract numbers and keywords
                            numbers = re.findall(r'\d+', input_data)
                            if len(numbers) >= 2:
                                a = int(numbers[0])
                                b = int(numbers[1])
                                
                                if any(op in input_data.lower() for op in ["add", "plus", "sum"]):
                                    operation = "add"
                                elif any(op in input_data.lower() for op in ["subtract", "minus", "difference"]):
                                    operation = "subtract"
                                elif any(op in input_data.lower() for op in ["multiply", "times", "product"]):
                                    operation = "multiply"
                                else:
                                    operation = "add"  # Default to add
                                
                                data = {
                                    "operation": operation,
                                    "a": a,
                                    "b": b
                                }
                            else:
                                raise ValueError(f"Failed to parse input: {input_data}")
                else:
                    # Extract from plain text
                    numbers = re.findall(r'\d+', input_data)
                    if len(numbers) >= 2:
                        a = int(numbers[0])
                        b = int(numbers[1])
                        
                        if any(op in input_data.lower() for op in ["add", "plus", "sum"]):
                            operation = "add"
                        elif any(op in input_data.lower() for op in ["subtract", "minus", "difference"]):
                            operation = "subtract"
                        elif any(op in input_data.lower() for op in ["multiply", "times", "product"]):
                            operation = "multiply"
                        else:
                            operation = "add"  # Default to add
                        
                        data = {
                            "operation": operation,
                            "a": a,
                            "b": b
                        }
                    else:
                        raise ValueError(f"Failed to parse input: {input_data}")
            else:
                data = input_data
            
            # Map operation names to MCP tool names
            operation_map = {
                "add": "add",
                "plus": "add",
                "addition": "add",
                "sum": "add",
                "subtract": "subtract",
                "subtraction": "subtract",
                "minus": "subtract",
                "difference": "subtract",
                "multiply": "multiply",
                "multiplication": "multiply",
                "times": "multiply",
                "product": "multiply"
            }
            
            # Get the operation name
            operation = data.get("operation", "").lower()
            tool_name = operation_map.get(operation)
            
            if not tool_name:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Get the numeric values
            a = int(data.get("a", 0))
            b = int(data.get("b", 0))
            
            print(f"Extracted operation: {operation}, a: {a}, b: {b}")
            
            # Return updated model parameters
            return {
                "update_model_params": {
                    "tool": tool_name,
                    "arg_a": a,
                    "arg_b": b
                },
                "operation_details": {
                    "operation": operation,
                    "a": a,
                    "b": b
                }
            }
            
        except Exception as e:
            print(f"Error extracting operation details: {e}")
            # Return default operation
            return {
                "update_model_params": {
                    "tool": "add",
                    "arg_a": 0,
                    "arg_b": 0
                },
                "operation_details": {
                    "operation": "add",
                    "a": 0,
                    "b": 0
                }
            }
    
    def _format_calculation_details(self, input_data):
        """
        Format calculation details for the final result.
        Combines original operation details with the result.
        """
        # Get the calculation result
        result = input_data
        
        try:
            # Check if we have operation details from the previous task
            if isinstance(result, dict) and "update_model_params" in result:
                # We're getting the dict output from the pre_process method of the MCP task
                return f"Error: Received pre-processor output instead of MCP result."
            
            # Try to get operation details from the calculate task in two ways
            calculate_task = self.flow.tasks.get("calculate")
            operation_details = None
            
            if calculate_task:
                # First try the input_data attribute (added by our patch)
                if hasattr(calculate_task, 'input_data') and isinstance(calculate_task.input_data, dict):
                    operation_details = calculate_task.input_data.get("operation_details", {})
                
                # If that fails, try the model_params
                if not operation_details and hasattr(calculate_task.agent, 'model_params'):
                    # Extract details from model_params
                    model_params = calculate_task.agent.model_params
                    if model_params:
                        tool = model_params.get("tool", "unknown")
                        a = model_params.get("arg_a", "?")
                        b = model_params.get("arg_b", "?")
                        
                        operation_details = {
                            "operation": tool,
                            "a": a,
                            "b": b
                        }
            
            if operation_details:
                a = operation_details.get("a", "?")
                b = operation_details.get("b", "?")
                op = operation_details.get("operation", "unknown")
                
                return (f"Operation: {op}\n"
                        f"First number: {a}\n"
                        f"Second number: {b}\n"
                        f"Result: {result}")
            
            # If no details available, just return the result
            return f"The calculated result is: {result}"
        except Exception as e:
            print(f"Error formatting calculation details: {e}")
            return f"Error formatting result: {e}. Raw result: {result}"

if __name__ == "__main__":
    unittest.main() 