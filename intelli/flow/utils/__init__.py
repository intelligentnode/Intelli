"""
Intelli Flow utilities

This package provides helper utilities for working with Intelli Flows.
"""

# Import utility modules and expose main components
try:
    from intelli.mcp import (
        MCPServerBuilder,
        MCPJSONExtractor,
        create_mcp_preprocessor
    )
except ImportError:
    # MCP utilities may not be available if MCP isn't installed
    pass
