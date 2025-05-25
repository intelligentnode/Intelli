import os
import asyncio
import re
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from intelli.mcp import create_mcp_preprocessor, MCPJSONExtractor
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Enhanced preprocessor to handle calculation expressions
def create_enhanced_preprocessor(server_url):
    # Get the standard preprocessor
    standard_preprocessor = create_mcp_preprocessor(
        server_url=server_url,
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
    
    # Wrapper function that adds expression extraction capability
    def enhanced_preprocessor(input_data):
        # First try the standard processor
        result = standard_preprocessor(input_data)
        
        # If no parameters were extracted, try to parse expressions
        if not any(key.startswith('arg_') for key in result.get('update_model_params', {})):
            print("Standard extraction failed, attempting to parse expressions...")
            
            try:
                # Look for patterns like "X + Y = Z" or "X + Y"
                expression_pattern = r'(\d+)\s*([+\-*x×])\s*(\d+)'
                match = re.search(expression_pattern, input_data)
                
                if match:
                    num1 = int(match.group(1))
                    operator = match.group(2)
                    num2 = int(match.group(3))
                    
                    # Map operator to tool
                    tool_map = {'+': 'add', '-': 'subtract', '*': 'multiply', 'x': 'multiply', '×': 'multiply'}
                    tool = tool_map.get(operator, 'add')
                    
                    print(f"Extracted from expression: {tool}, a={num1}, b={num2}")
                    
                    return {
                        "update_model_params": {
                            "url": server_url,
                            "tool": tool,
                            "arg_a": num1,
                            "arg_b": num2
                        }
                    }
            except Exception as e:
                print(f"Expression parsing failed: {e}")
        
        return result
    
    return enhanced_preprocessor

# Function to generate and save the flow graph
def save_flow_graph(flow, name="http_math_flow"):
    os.makedirs("./temp/mcp", exist_ok=True)
    try:
        graph_path = flow.generate_graph_img(
            name=name,
            save_path="./temp/mcp"
        )
        print(f"\n🎨 Flow visualization saved to: {graph_path}")
    except Exception as e:
        print(f"⚠️ Could not generate graph: {e}")

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
            
            You MUST extract numbers and operations from user queries and format them as JSON.
            
            ONLY return a valid JSON object with this exact format:
            {"operation": "add", "a": number, "b": number}
            
            For example:
            - For "What is 5 plus 3?" return {"operation": "add", "a": 5, "b": 3}
            - For "Subtract 10 from 20" return {"operation": "subtract", "a": 20, "b": 10}
            
            DO NOT perform the calculation.
            DO NOT include any explanations or text outside the JSON object.
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
    
    # MCP server base URL - include the /mcp path
    MCP_SERVER_URL = "http://localhost:8000/mcp"
    
    # Create enhanced preprocessor that can handle both JSON and expressions
    extract_operation = create_enhanced_preprocessor(MCP_SERVER_URL)
    
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
    
    # Generate flow visualization
    save_flow_graph(flow)
    
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
