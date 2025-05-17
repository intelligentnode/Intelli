# mcp_calculator_server.py
import os
from mcp.server.fastmcp import FastMCP

# Create a calculator MCP server with stateless HTTP support
mcp = FastMCP("Calculator", stateless_http=True)

@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers"""
    print(f"Adding {a} + {b}")
    result = a + b
    print(f"Result: {result}")
    return str(result)

@mcp.tool()
def subtract(a: int, b: int) -> str:
    """Subtract second number from first number"""
    print(f"Subtracting {b} from {a}")
    result = a - b
    print(f"Result: {result}")
    return str(result)

@mcp.tool()
def multiply(a: int, b: int) -> str:
    """Multiply two numbers"""
    print(f"Multiplying {a} * {b}")
    result = a * b
    print(f"Result: {result}")
    return str(result)

if __name__ == "__main__":
    print("Starting Calculator MCP Server...")
    
    # MCP server endpoint (will be available at this path)
    mcp_path = "/mcp"
    
    print(f"Server will run at http://0.0.0.0:8000")
    print("Registered tools: add, subtract, multiply")
    print(f"MCP endpoint URL to use in client: http://localhost:8000{mcp_path}")
    print("NOTE: Client must use arg_a and arg_b parameter names")
    
    # Use the FastMCP built-in run method for HTTP
    # With the correct transport type for HTTP
    mcp.run(
        transport="streamable-http",  # Standard HTTP streaming transport
        mount_path=mcp_path           # Server endpoint path
    )
