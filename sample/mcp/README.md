# MCP Calculator Demo

## What's Included
- **Calculator Server**: HTTP-based MCP server with math operations.
- **Flow Client**: Client that combines AI understanding with MCP functions.

## Requirements
- Python 3.10+
- OpenAI API key (set as OPENAI_API_KEY environment variable)
- Packages: `mcp`, `intelli`, `httpx`, `openai`

## Quick Start

1. **Start the server**
   ```bash
   python http_mcp_calculator_server.py
   ```
   This runs an MCP calculator server at http://localhost:8000/mcp

2. **Run the client**
   ```bash
   python http_math_flow_client.py
   ```
   Try changing the query in the script to test different operations!

## How It Works

**Server (http_mcp_calculator_server.py)**
- Creates an MCP server with math tools
- Tools: add, subtract, multiply
- Uses streamable HTTP transport at "/mcp" endpoint

**Client (http_math_flow_client.py)**
- Creates a two-step flow:
  * OpenAI parses natural language into math operations
  * MCP agent sends operations to the calculator server
- Shows the progression from text to calculation to result

## Try These Examples

Modify the `user_query` variable in the client to try:
```python
user_query = "What is 25 multiplied by 4?"
user_query = "Can you subtract 15 from 100?"
user_query = "Add 123 and 456 please"
```

The flow will extract the operation and numbers automatically.

## How Components Connect

```
User Query → OpenAI Parser → MCP Client → Calculator Server → Result
```

The client handles all parameter conversion and error handling automatically!

