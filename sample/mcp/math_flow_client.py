# math_flow_client.py
import os
import asyncio
import sys
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from intelli.flow.utils import create_mcp_preprocessor
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

async def run_math_flow(query="What is 7 plus 8?"):
    # Get the path to the MCP server file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "mcp_math_server.py")
    
    # Create LLM agent with stronger prompt
    llm_agent = Agent(
        agent_type=AgentTypes.TEXT.value,
        provider="openai",
        mission="Parse math query",
        model_params={
            "key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-3.5-turbo",
            "system_message": """
            You are a specialized parser for arithmetic operations. 
            
            Your ONLY job is to extract numbers and operations from user queries.
            
            You MUST return ONLY a valid JSON object with this exact format:
            {"operation": "add", "a": number, "b": number}
            
            Supported operations:
            - "add" (for addition, plus, sum)
            - "subtract" (for subtraction, minus, difference)
            - "multiply" (for multiplication, times, product)
            
            DO NOT include explanations, confirmations or any other text.
            Your entire response must be ONLY the JSON object.
            """
        }
    )
    
    # Create a task for the LLM
    llm_task = Task(
        TextTaskInput(query),
        llm_agent,
        log=True
    )
    
    # Create a preprocessor for the subprocess-based MCP server with complete operation mapping
    extract_operation = create_mcp_preprocessor(
        server_path=server_path,
        default_tool="add",
        operations_map={
            # Addition operations
            "add": "add",
            "plus": "add",
            "sum": "add",
            "+": "add",
            
            # Subtraction operations
            "subtract": "subtract",
            "minus": "subtract",
            "difference": "subtract",
            "-": "subtract",
            
            # Multiplication operations
            "multiply": "multiply",
            "times": "multiply",
            "product": "multiply",
            "*": "multiply",
            "x": "multiply"
        },
        param_names=["a", "b"]
    )
    
    # Create proper MCP agent using subprocess approach
    mcp_agent = Agent(
        agent_type=AgentTypes.MCP.value,
        provider="mcp",
        mission="Calculate arithmetic result",
        model_params={
            "command": sys.executable,
            "args": [server_path],
            "tool": "add",
            "arg_a": 0,
            "arg_b": 0
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
    print("Using MCP server with subprocess transport")
    
    # Run the flow
    asyncio.run(run_math_flow(user_query))