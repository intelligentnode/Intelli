---
sidebar_position: 8
---
# Dynamic Tool Routing

Dynamic Tool Routing enables flows to automatically route based on LLM tool/function call decisions. Instead of hardcoded routing logic, the flow examines LLM output and routes to different tasks based on whether tools were invoked.

## Installation

```bash
pip install "intelli[mcp]"
```

## Overview

Dynamic Tool Routing allows LLMs to decide execution paths by:
- Calling tools/functions when needed (APIs, MCP servers, databases).
- Returning direct responses when tools aren't required.
- Selecting between multiple available tools.

The `ToolDynamicConnector` analyzes LLM output and routes to appropriate tasks automatically.

## Core Components

| Component | Function |
|-----------|----------|
| **ToolDynamicConnector** | Analyzes LLM output and routes based on tool usage. |
| **LLM Agent with Tools** | Agent configured with available tools/functions. |
| **Tool Processors** | Tasks executed when tools are called. |
| **Direct Response** | Tasks executed for direct LLM responses. |

## Key Parameters

### ToolDynamicConnector

```python
ToolDynamicConnector(
    destinations={
        "tool_called": "task_name",    # Required: Route when tools are called
        "no_tool": "task_name"         # Required: Route for direct responses
    },
    decision_fn=custom_function,       # Optional: Custom routing logic
    name="router_name"                 # Optional: Connector identifier
)
```

### LLM Agent with Tools

```python
Agent(
    agent_type="text",
    provider="openai",                 # "openai", "anthropic", etc.
    mission="Tool usage instructions", # Guide LLM when to use tools
    model_params={
        "key": "api_key",
        "model": "gpt-4o",             # Model supporting function calling
        "tools": tool_definitions      # Required: Available tools/functions
    }
)
```

### MCP Agent

```python
# Local MCP Server
Agent(
    agent_type="mcp",
    provider="mcp",
    model_params={
        "command": "python",           # Command to run server
        "args": ["server_file.py"]     # Server file path
    }
)

# Remote MCP Server  
Agent(
    agent_type="mcp",
    provider="mcp", 
    model_params={
        "url": "http://server:8000/mcp", # Remote server URL
        "tool": "tool_name",             # Specific tool to call
        "input_arg": "parameter_name"    # Input parameter mapping
    }
)
```

### Flow with Dynamic Routing

```python
Flow(
    tasks={"task_id": task_object},
    map_paths={},                      # Static routing
    dynamic_connectors={               # Dynamic routing rules
        "source_task": connector_obj
    }
)
```

## Routing Logic

```
User Input → LLM Agent → ToolDynamicConnector → Route Decision
```

## Basic Implementation

### Math Calculator Example

This example demonstrates how to create a flow where the LLM decides whether to use math tools or provide a direct response.

#### 1. Define Tools

First, define the function schemas that the LLM can call. These match the actual tools available in the MCP math server.

```python
import asyncio
from intelli.flow import Flow, Task, Agent, ToolDynamicConnector
from intelli.flow.input.task_input import TextTaskInput

# Define function schemas for LLM - matching actual MCP server tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "subtract", 
            "description": "Subtract second number from first number",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Number to subtract"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers together", 
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]
```

#### MCP Math Server Setup

For these tools to work, you'll need the MCP math server file:

**Download**: [mcp_math_server.py](https://github.com/intelligentnode/Intelli/blob/main/intelli/test/integration/mcp_math_server.py)

#### 2. Create Agents

Create agents for LLM decisions, MCP math processing, and direct responses.

```python
# LLM agent with math tool definitions
llm_agent = Agent(
    agent_type="text",
    provider="openai",
    mission="Math assistant. Use add, subtract, or multiply tools for calculations.",
    model_params={
        "key": "your-openai-api-key",
        "model": "gpt-4o",
        "tools": tools
    }
)

# MCP agent for math operations - will handle any of the math tools
mcp_agent = Agent(
    agent_type="mcp",
    provider="mcp",
    mission="Execute mathematical operations through MCP server",
    model_params={
        "command": "python",
        "args": ["mcp_math_server.py"]
        # Note: tool and input_arg will be set dynamically based on LLM choice
    }
)

# Direct response agent for non-math queries
direct_agent = Agent(
    agent_type="text", 
    provider="openai",
    mission="Provide direct responses for non-mathematical queries",
    model_params={
        "key": "your-openai-api-key",
        "model": "gpt-3.5-turbo"
    }
)
```

#### 3. Create Tasks

Wrap each agent in a task. Tasks define the input and execution context for each agent.

```python
# Create tasks
llm_task = Task(TextTaskInput("Process user input"), llm_agent)
math_task = Task(TextTaskInput("Execute math operation"), mcp_agent)
direct_task = Task(TextTaskInput("Provide direct response"), direct_agent)
```

#### 4. Configure Dynamic Routing

Set up the `ToolDynamicConnector` to route based on whether the LLM calls tools or responds directly.

```python
# Configure dynamic routing
tool_connector = ToolDynamicConnector(
    destinations={
        "tool_called": "math_processor",
        "no_tool": "direct_response"
    },
    name="math_router"
)
```

#### 5. Build Flow

Create the flow with tasks, routing paths, and dynamic connectors.

```python
# Build flow
flow = Flow(
    tasks={
        "llm_decision": llm_task,
        "math_processor": math_task,
        "direct_response": direct_task
    },
    dynamic_connectors={
        "llm_decision": tool_connector
    }
)
```

#### 6. Execute and Test

Run the flow with different types of queries to see the dynamic routing in action.

```python
async def run_test():
    # Test 1: Addition query (triggers add tool)
    llm_task.task_input = TextTaskInput("What is 15 + 23?")
    result = await flow.start()
    print(f"Addition result: {result}")
    
    # Test 2: Multiplication query (triggers multiply tool)
    llm_task.task_input = TextTaskInput("Calculate 7 times 8")
    result = await flow.start()
    print(f"Multiplication result: {result}")
    
    # Test 3: Subtraction query (triggers subtract tool)
    llm_task.task_input = TextTaskInput("What is 100 minus 37?")
    result = await flow.start()
    print(f"Subtraction result: {result}")
    
    # Test 4: General query (no tool)
    llm_task.task_input = TextTaskInput("Explain machine learning")
    result = await flow.start()
    print(f"Direct result: {result}")

if __name__ == "__main__":
    asyncio.run(run_test())
```

### Expected Output

```
Addition result: {'math_processor': {'output': '38', 'type': 'text'}}
Multiplication result: {'math_processor': {'output': '56', 'type': 'text'}}
Subtraction result: {'math_processor': {'output': '63', 'type': 'text'}}
Direct result: {'direct_response': {'output': 'Machine learning is...', 'type': 'text'}}
```

## Anthropic Integration

Anthropic (Claude) models use a different tool schema format than OpenAI:

```python
# Anthropic tool schema
anthropic_tools = [
    {
        "name": "analyze_data",
        "description": "Analyze CSV data files and generate insights",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string", 
                    "description": "Path to the CSV file to analyze"
                }
            },
            "required": ["file_path"]
        }
    }
]

# Claude agent with tool support
claude_agent = Agent(
    agent_type="text",
    provider="anthropic",
    mission="Data analysis assistant. Use analyze_data tool for CSV files.",
    model_params={
        "key": "your-anthropic-api-key",
        "model": "claude-3-7-sonnet-20250219",
        "tools": anthropic_tools
    }
)
```

The key difference is that Anthropic uses `input_schema` instead of `parameters`, and doesn't wrap tools in a `function` object.

## Advanced Features

### Custom Decision Functions

Create custom routing logic beyond simple tool detection:

```python
def custom_routing_logic(output, output_type):
    """Simple custom decision function"""
    
    # Check for tool responses
    if isinstance(output, dict):
        if output.get("type") in ["tool_response", "function_response"]:
            return "tool_processor"
    
    # Check for errors in text
    if isinstance(output, str) and "error" in output.lower():
        return "error_handler"
    
    return "direct_response"

# Simple custom connector
advanced_connector = ToolDynamicConnector(
    decision_fn=custom_routing_logic,
    destinations={
        "tool_processor": "process_task",
        "error_handler": "error_task",
        "direct_response": "response_task",
        # Required fallbacks
        "tool_called": "process_task",
        "no_tool": "response_task"
    }
)
```

### Dynamic to Static Chaining

Combine dynamic routing with static processing chains:

```python
# Flow with mixed routing
flow = Flow(
    tasks={
        "decision": decision_task,
        "tool_execution": tool_task,      # Dynamic destination
        "direct_response": direct_task,   # Dynamic destination  
        "formatter": formatter_task,      # Static destination
        "validator": validator_task       # Static destination
    },
    map_paths={
        # Static routing: both dynamic destinations → formatter → validator
        "tool_execution": ["formatter"],
        "direct_response": ["formatter"], 
        "formatter": ["validator"]
    },
    dynamic_connectors={
        # Dynamic routing from decision task
        "decision": tool_connector
    }
)
```

This creates a flow where:
1. LLM makes dynamic routing decision
2. Either tool or direct path executes  
3. Both paths converge to static formatting and validation

### Tool Information Extraction

Extract tool details from LLM responses:

```python
connector = ToolDynamicConnector(
    destinations={"tool_called": "processor", "no_tool": "responder"}
)

# Extract tool details
tool_info = connector.get_tool_info(llm_output)
if tool_info:
    print(f"Tool: {tool_info['name']}")
    print(f"Arguments: {tool_info['arguments']}")
    print(f"Call ID: {tool_info['id']}")
```

**Learn More**: For MCP documentation: [Get Started with MCP](https://docs.intellinode.ai/docs/python/mcp/get-started).
