---
sidebar_position: 4
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
| **ToolDynamicConnector** | Analyzes LLM output and routes based on tool usage |
| **LLM Agent with Tools** | Agent configured with available tools/functions |
| **Tool Processors** | Tasks executed when tools are called |
| **Direct Response** | Tasks executed for direct LLM responses |

## Routing Logic

```
User Input → LLM Agent → ToolDynamicConnector → Route Decision
```

## Basic Implementation

### Math Calculator Example

This example demonstrates how to create a flow where the LLM decides whether to use a calculator tool or provide a direct response.

#### 1. Define Tools

First, define the function schema that the LLM can call. This tells the LLM what tools are available and how to use them.

```python
# math_calculator_flow.py
import asyncio
from intelli.flow import Flow, Task, Agent, ToolDynamicConnector
from intelli.flow.input.task_input import TextTaskInput

# Define function schema for LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]
```

#### 2. Create Agents

Create three agents: one LLM agent with tool capabilities, one MCP agent for processing calculations, and one for direct responses.

```python
# LLM agent with tool definitions
llm_agent = Agent(
    agent_type="text",
    provider="openai",
    mission="Assistant with calculation capabilities. Use calculate tool for math operations.",
    model_params={
        "key": "your-openai-api-key",
        "model": "gpt-4o",
        "tools": tools
    }
)

# MCP agent for calculation processing
mcp_agent = Agent(
    agent_type="mcp",
    provider="mcp",
    mission="Execute mathematical calculations",
    model_params={
        "command": "python",
        "args": ["mcp_math_server.py"],
        "tool": "calculate",
        "input_arg": "expression"
    }
)

# Direct response agent
direct_agent = Agent(
    agent_type="text",
    provider="openai",
    mission="Provide direct responses for non-calculation queries",
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
calc_task = Task(TextTaskInput("Execute calculation"), mcp_agent)
direct_task = Task(TextTaskInput("Provide direct response"), direct_agent)
```

#### 4. Configure Dynamic Routing

Set up the `ToolDynamicConnector` to route based on whether the LLM calls tools or responds directly.

```python
# Configure dynamic routing
tool_connector = ToolDynamicConnector(
    destinations={
        "tool_called": "calculator",
        "no_tool": "direct_response"
    },
    name="math_router"
)
```

#### 5. Build Flow

Create the flow with tasks, routing paths, and dynamic connectors. The `map_paths` define static connections while `dynamic_connectors` handle conditional routing.

```python
# Build flow
flow = Flow(
    tasks={
        "llm_decision": llm_task,
        "calculator": calc_task,
        "direct_response": direct_task
    },
    map_paths={
        "llm_decision": [],
        "calculator": [],
        "direct_response": []
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
    # Test 1: Mathematical query (triggers tool)
    llm_task.task_input = TextTaskInput("Calculate 15 * 23 + 47")
    result = await flow.start()
    print(f"Math result: {result}")
    
    # Test 2: General query (no tool)
    llm_task.task_input = TextTaskInput("Explain machine learning")
    result = await flow.start()
    print(f"Direct result: {result}")

if __name__ == "__main__":
    asyncio.run(run_test())
```

### Expected Output

```
Math result: {'calculator': {'output': '392', 'type': 'text'}}
Direct result: {'direct_response': {'output': 'Machine learning is...', 'type': 'text'}}
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
    print(f"Tool: {tool_info['name']}")
    print(f"Arguments: {tool_info['arguments']}")
    print(f"Call ID: {tool_info['id']}")
```

## MCP Server Integration

Route to MCP servers based on tool decisions:

```python
# MCP agent configuration
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
        "tool_execution": ["result_formatter"]  # Chain tool output
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
            "name": "calculate",
            "arguments": '{"expression": "15*23"}'
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
            "name": "analyze_data",
            "arguments": '{"file": "data.csv"}'
        }
    }]
}
```

### Direct Text Response
```python
"Machine learning is a method of data analysis..."
```