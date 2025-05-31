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

**Learn More**: For MCP documentation, see [Get Started with MCP](https://docs.intellinode.ai/docs/python/mcp/get-started)

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

## Advanced Multi-Tool Routing

For more complex scenarios, you can create specialized routing based on which specific tool was called:

```python
def math_tool_router(output, output_type):
    """Route based on specific math tool called"""
    if isinstance(output, dict) and output.get("type") == "tool_response":
        tool_calls = output.get("tool_calls", [])
        if tool_calls:
            tool_name = tool_calls[0]["function"]["name"]
            if tool_name in ["add", "subtract"]:
                return "basic_math"
            elif tool_name == "multiply":
                return "multiplication"
    return "direct_response"

# Custom routing connector
custom_connector = ToolDynamicConnector(
    decision_fn=math_tool_router,
    destinations={
        "basic_math": "simple_calculator",
        "multiplication": "advanced_calculator", 
        "direct_response": "text_responder",
        "tool_called": "simple_calculator",  # Required fallback
        "no_tool": "text_responder"          # Required fallback
    }
)
```

## Multi-Tool Configuration

Configure multiple tools for LLM selection:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Query internal database",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Database table"},
                    "filter": {"type": "string", "description": "Query filter"}
                },
                "required": ["table"]
            }
        }
    }
]
```

## Anthropic Integration

Anthropic tools use different schema format:

```python
anthropic_tools = [
    {
        "name": "analyze_data",
        "description": "Analyze CSV data files",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to CSV file"},
                "analysis_type": {"type": "string", "description": "Type of analysis"}
            },
            "required": ["file_path"]
        }
    }
]

anthropic_agent = Agent(
    agent_type="text",
    provider="anthropic",
    mission="Data analysis assistant with CSV processing capabilities",
    model_params={
        "key": "your-anthropic-api-key",
        "model": "claude-3-sonnet-20240229",
        "tools": anthropic_tools
    }
)
```

## Custom Decision Functions

Implement custom routing logic beyond tool detection:

```python
def custom_routing_logic(output, output_type):
    """Custom decision function for specialized routing"""
    # Check for tool responses
    if isinstance(output, dict):
        if output.get("type") in ["tool_response", "function_response"]:
            return "execute_tool"
    
    # Keyword-based routing
    if isinstance(output, str):
        if any(keyword in output.lower() for keyword in ["search", "query", "analyze"]):
            return "needs_processing"
        else:
            return "direct_answer"
    
    return "direct_answer"

# Apply custom logic
custom_connector = ToolDynamicConnector(
    decision_fn=custom_routing_logic,
    destinations={
        "execute_tool": "tool_processor",
        "needs_processing": "data_processor",
        "direct_answer": "response_formatter",
        "tool_called": "tool_processor",  # Required
        "no_tool": "response_formatter"   # Required
    }
)
```

## Tool Information Extraction

Extract tool metadata from LLM responses:

```python
connector = ToolDynamicConnector(
    destinations={"tool_called": "processor", "no_tool": "responder"}
)

# Extract tool details
tool_info = connector.get_tool_info(llm_output)
if tool_info:
    print(f"Tool: {tool_info['name']}")           # e.g., "add"
    print(f"Arguments: {tool_info['arguments']}")  # e.g., {"a": 15, "b": 23}
    print(f"Call ID: {tool_info['id']}")
```

## MCP Server Integration

Route to MCP servers based on tool decisions:

```python
# MCP agent configuration - for remote HTTP MCP servers
mcp_agent = Agent(
    agent_type="mcp",
    provider="mcp",
    mission="Process data through MCP server",
    model_params={
        "url": "http://localhost:8000/mcp",  # HTTP MCP server
        "tool": "process_data",
        "input_arg": "data_path"
    }
)

# Route to MCP only when tools are called
tool_connector = ToolDynamicConnector(
    destinations={
        "tool_called": "mcp_processor",
        "no_tool": "simple_response"
    }
)
```

## Implementation Guidelines

### Tool Schema Design

Define clear, specific tool descriptions:

```python
{
    "name": "calculate_tax",
    "description": "Calculate tax amount for given income and rate. Use only for tax calculations.",
    "parameters": {
        "type": "object",
        "properties": {
            "income": {
                "type": "number",
                "description": "Annual income in USD"
            },
            "tax_rate": {
                "type": "number",
                "description": "Tax rate as decimal (0.25 = 25%)"
            }
        },
        "required": ["income", "tax_rate"]
    }
}
```

### Error Handling

Ensure both routing paths are defined:

```python
ToolDynamicConnector(
    destinations={
        "tool_called": "tool_processor",
        "no_tool": "fallback_responder"  # Always provide fallback
    }
)
```

### Task Chaining

Chain tool results through processing tasks:

```python
flow = Flow(
    tasks={
        "llm_decision": decision_task,
        "tool_execution": tool_task,
        "result_formatter": format_task,
        "direct_response": response_task
    },
    map_paths={
        "tool_execution": ["result_formatter"]  # Chain tool output to formatter
        # Tasks without static routing don't need to be listed
    },
    dynamic_connectors={
        "llm_decision": tool_connector
    }
)
```

## Troubleshooting

### Tool Call Issues
- Verify tool schema matches provider requirements (OpenAI vs Anthropic)
- Check model supports function calling (GPT-4, Claude-3+)
- Test with explicit tool invocation requests

### Routing Problems
- Confirm `ToolDynamicConnector` has required `"tool_called"` and `"no_tool"` keys
- Verify destination task names match flow task IDs
- Check decision function return values match destination keys

### MCP Integration Issues
- Validate MCP server accessibility and tool availability
- Test MCP agent configuration independently
- Verify tool parameter mapping between LLM and MCP

## Response Format Detection

The connector detects these LLM response formats:

### OpenAI Tool Response
```python
{
    "type": "tool_response",
    "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "add",
            "arguments": '{"a": 15, "b": 23}'
        }
    }]
}
```

### Anthropic Tool Response
```python
{
    "type": "tool_response",
    "tool_calls": [{
        "id": "call_456",
        "type": "function",
        "function": {
            "name": "multiply",
            "arguments": '{"a": 7, "b": 8}'
        }
    }]
}
```

### Direct Text Response
```python
"Machine learning is a method of data analysis..."
``` 