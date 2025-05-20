"""
HTTP MCP DataFrame Flow Client Example.

Demonstrates using Intelli Flow to query a DataFrame server via HTTP.
Query operations include schema, shape, head, column selection and filtering.

Run steps:
1. Start server: python http_mcp_dataframe_server.py
2. Check server is running at http://localhost:8000/mcp
3. Run client: python http_dataframe_flow_client.py
"""
import os
import asyncio
import json

from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow
from intelli.flow.types import AgentTypes
from intelli.flow.utils import create_mcp_preprocessor
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()

# Server connection settings
MCP_DATAFRAME_SERVER_URL = "http://localhost:8000/mcp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "mcp_dataframe_client")

# Generate and save flow visualization
def save_flow_graph(flow: Flow, name: str = "http_dataframe_flow"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        graph_path = flow.generate_graph_img(
            name=name,
            save_path=OUTPUT_DIR
        )
        print(f"\n🎨 Flow visualization saved to: {graph_path}")
    except Exception as e:
        print(f"⚠️ Could not generate graph: {e}")

# Main flow definition and execution function
async def run_dataframe_query_flow():
    print(f"Targeting MCP DataFrame Server at: {MCP_DATAFRAME_SERVER_URL}")

    # Task 1: Get DataFrame Schema
    schema_agent = Agent(
        agent_type=AgentTypes.MCP.value,
        provider="mcp",
        mission="Get DataFrame schema",
        model_params={
            "url": MCP_DATAFRAME_SERVER_URL,
            "tool": "get_schema"
        }
    )
    schema_task = Task(
        TextTaskInput("Requesting schema"), 
        schema_agent,
        log=True
    )

    # Task 2: Get DataFrame Shape
    shape_agent = Agent(
        agent_type=AgentTypes.MCP.value,
        provider="mcp",
        mission="Get DataFrame shape",
        model_params={
            "url": MCP_DATAFRAME_SERVER_URL,
            "tool": "get_shape",
        }
    )
    shape_task = Task(TextTaskInput("Requesting shape"), shape_agent, log=True)

    # Task 3: Get DataFrame Head
    head_agent = Agent(
        agent_type=AgentTypes.MCP.value,
        provider="mcp",
        mission="Get DataFrame head",
        model_params={
            "url": MCP_DATAFRAME_SERVER_URL,
            "tool": "get_head",
            "arg_n": 3 # Get first 3 rows
        }
    )
    head_task = Task(TextTaskInput("Requesting head"), head_agent, log=True)

    # Task 4: Select Specific Columns
    select_agent = Agent(
        agent_type=AgentTypes.MCP.value,
        provider="mcp",
        mission="Select specific columns",
        model_params={
            "url": MCP_DATAFRAME_SERVER_URL,
            "tool": "select_columns",
            "arg_columns": ["Name", "Salary"]
        }
    )
    select_task = Task(
        TextTaskInput(json.dumps({"columns_to_select": ["Name", "Salary"]})),
        select_agent,
        log=True
    )

    # Task 5: Filter Rows
    filter_agent = Agent(
        agent_type=AgentTypes.MCP.value,
        provider="mcp",
        mission="Filter rows from DataFrame",
        model_params={
            "url": MCP_DATAFRAME_SERVER_URL,
            "tool": "filter_rows",
            "arg_column": "City",
            "arg_operator": "==",
            "arg_value": "New York"
        }
    )
    filter_task = Task(TextTaskInput("Requesting filtered data"), filter_agent, log=True)

    # Define flow tasks and connections
    tasks_dict = {
        "get_schema": schema_task,
        "get_shape": shape_task,
        "get_head": head_task,
        "select_cols": select_task,
        "filter_data": filter_task
    }
    
    # Sequential flow: schema -> shape -> head -> select -> filter
    map_paths_dict = {
        "get_schema": ["get_shape"],
        "get_shape": ["get_head"],
        "get_head": ["select_cols"],
        "select_cols": ["filter_data"],
        "filter_data": []
    }

    data_flow = Flow(tasks=tasks_dict, map_paths=map_paths_dict, log=True)
    save_flow_graph(data_flow, name="http_dataframe_query_flow")

    print("\n=== DataFrame Query Flow Created ===")
    print(f"Tasks: {list(tasks_dict.keys())}")
    print("==================================\n")

    # Run the flow
    try:
        flow_results = await data_flow.start()
    except Exception as e:
        print(f"🛑 Error running flow: {e}")
        print(f"Make sure server is running at {MCP_DATAFRAME_SERVER_URL}")
        return

    # Display results
    print("\n=== DataFrame Query Flow Results ===")
    for task_name, result_data in flow_results.items():
        output = result_data.get('output')
        # Format output as pretty JSON when possible
        if isinstance(output, str):
            try:
                output = json.dumps(json.loads(output), indent=2)
            except json.JSONDecodeError:
                pass
        elif isinstance(output, dict) or isinstance(output, list):
            output = json.dumps(output, indent=2)
            
        print(f"\n--- Task: {task_name} ---")
        print(f"Output:\n{output}")
    print("==================================\n")

    return flow_results

if __name__ == "__main__":
    print("🚀 Starting HTTP MCP DataFrame Flow Client Example...")
    print("Ensure the `http_mcp_dataframe_server.py` is running separately.")
    
    asyncio.run(run_dataframe_query_flow())
    
    print("🏁 DataFrame Flow Client Example Finished.") 