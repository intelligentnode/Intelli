# http_mcp_calculator_server.py
import os
from intelli.flow.utils import MCPServerBuilder

# Create a server with HTTP support
server = MCPServerBuilder("Calculator", stateless_http=True)

# Add tools with decorators
@server.add_tool
def add(a: int, b: int) -> str:
    """Add two numbers"""
    print(f"Adding {a} + {b}")
    result = a + b
    print(f"Result: {result}")
    return str(result)

@server.add_tool
def subtract(a: int, b: int) -> str:
    """Subtract second number from first number"""
    print(f"Subtracting {b} from {a}")
    result = a - b
    print(f"Result: {result}")
    return str(result)

@server.add_tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers"""
    print(f"Multiplying {a} * {b}")
    result = a * b
    print(f"Result: {result}")
    return str(result)

if __name__ == "__main__":
    print("Starting Calculator MCP Server...")
    
    # Configure HTTP server with streamable-http transport
    mcp_path = "/mcp"
    host = "0.0.0.0"    # Used for info display only
    port = 8000         # Used for info display only
    
    # Run the server with HTTP transport
    # Note: FastMCP internally handles host/port configuration
    server.run(
        transport="streamable-http",
        mount_path=mcp_path,
        host=host,
        port=port
    )
