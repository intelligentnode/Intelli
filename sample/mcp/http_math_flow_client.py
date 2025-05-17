import os
import asyncio
import json
import re
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

async def run_math_flow(query="What is 7 plus 8?"):
    # Create LLM agent to parse the query
    llm_agent = Agent(
        agent_type=AgentTypes.TEXT.value,
        provider="openai",
        mission="Parse math query",
        model_params={
            "key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-3.5-turbo",
            "system_message": """
            You are a specialized parser for arithmetic operations. 
            Extract numbers and operations from user queries.
            Return ONLY a valid JSON object with this format:
            {"operation": "add", "a": number, "b": number}
            """
        }
    )
    
    # Create a task for the LLM
    llm_task = Task(
        TextTaskInput(query),
        llm_agent,
        log=True
    )
    
    # MCP server base URL - include the /mcp path
    MCP_SERVER_URL = "http://localhost:8000/mcp"
    
    # Function to extract operation details from LLM output
    def extract_operation(input_data):
        try:
            print(f"Raw LLM output: {input_data}")
            
            # Clean the input to find JSON
            cleaned_input = input_data.strip()
            
            # Try direct JSON parsing first
            try:
                data = json.loads(cleaned_input)
                print("Parsed JSON directly")
            except json.JSONDecodeError:
                # Try to extract JSON using regex
                json_match = re.search(r'\{.*\}', cleaned_input, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).replace("'", '"')
                    data = json.loads(json_str)
                    print("Extracted JSON with regex")
                else:
                    # Last resort: parse natural language
                    print("Extracting from natural language")
                    words = cleaned_input.lower().split()
                    numbers = [int(word) for word in words if word.isdigit()]
                    
                    if any(op in words for op in ["add", "plus", "sum", "+"]):
                        operation = "add"
                    elif any(op in words for op in ["subtract", "minus", "difference", "-"]):
                        operation = "subtract"
                    elif any(op in words for op in ["multiply", "times", "product", "*", "x"]):
                        operation = "multiply"
                    else:
                        operation = "add"  # Default
                    
                    if len(numbers) >= 2:
                        data = {"operation": operation, "a": numbers[0], "b": numbers[1]}
                    else:
                        print("Could not extract numbers, using defaults")
                        return {"update_model_params": {
                            "url": MCP_SERVER_URL,
                            "tool": "add",
                            "arg_a": 0,  # Default value
                            "arg_b": 0   # Default value
                        }}
            
            # Map operation name to tool
            op_map = {"add": "add", "plus": "add", "subtract": "subtract", "multiply": "multiply"}
            operation = data.get("operation", "").lower()
            tool = op_map.get(operation, "add")
            
            a = int(data.get("a", 0))
            b = int(data.get("b", 0))
            
            print(f"Extracted operation: {tool}, a: {a}, b: {b}")
            
            # Use parameter names matching the server tool function signatures
            # Use arg_ prefix for parameters to work with MCPAgentHandler
            return {
                "update_model_params": {
                    "url": MCP_SERVER_URL,
                    "tool": tool,
                    "arg_a": a,  # MCP server parameter
                    "arg_b": b   # MCP server parameter
                }
            }
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            return {"update_model_params": {
                "url": MCP_SERVER_URL,
                "tool": "add",
                "arg_a": 0,  # Updated to match server parameter name
                "arg_b": 0   # Updated to match server parameter name
            }}
    
    # Create MCP agent with proper URL parameter for HTTP transport
    mcp_agent = Agent(
        agent_type=AgentTypes.MCP.value,
        provider="mcp",
        mission="Calculate arithmetic result using HTTP API",
        model_params={
            "url": MCP_SERVER_URL,
            "tool": "add",
            "arg_a": 0,  # Default value
            "arg_b": 0   # Default value
        }
    )
    
    # Create a task for the calculation
    mcp_task = Task(
        TextTaskInput("Calculate"),
        mcp_agent,
        log=True,
        pre_process=extract_operation
    )
    
    # Create the flow
    tasks = {"parse": llm_task, "calculate": mcp_task}
    map_paths = {"parse": ["calculate"], "calculate": []}
    
    flow = Flow(tasks=tasks, map_paths=map_paths, log=True)
    
    print("\n=== Flow Created ===")
    print(f"Tasks: {list(tasks.keys())}")
    print("=================\n")
    
    # Run the flow
    results = await flow.start()
    
    # Format and print results
    print("\n=== Flow Results ===")
    print(f"Query: {query}")
    print(f"LLM parsing: {results['parse']['output']}")
    print(f"Calculation result: {results['calculate']['output']}")
    print("====================\n")
    
    return results

if __name__ == "__main__":
    # Example query to test
    user_query = "What is 42 plus 28?"
    
    print(f"Processing query: {user_query}")
    print("Using MCP Calculator Server at http://localhost:8000/mcp")
    
    # Run the flow
    asyncio.run(run_math_flow(user_query))
