# mcp_math_server.py
from intelli.flow.utils import MCPServerBuilder

# Create a server using the builder
server = MCPServerBuilder("MathTools")

# Add tools with decorators
@server.add_tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@server.add_tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

@server.add_tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    # Run the server with stdio transport
    server.run(transport="stdio")
