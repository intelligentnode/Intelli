"""
MCP (Model Context Protocol) Package for Intelli

This package provides utilities for working with MCP (Model Context Protocol),
making it easier to create MCP servers and integrate them with Intelli.
"""

from .utils import MCPServerBuilder, MCPJSONExtractor, create_mcp_preprocessor

# Import DataFrame utilities with graceful fallback
try:
    from .dataframe_utils import (
        BaseDataFrameMCPServerBuilder,
        PandasMCPServerBuilder, 
        PolarsMCPServerBuilder,
        PANDAS_AVAILABLE,
        POLARS_AVAILABLE
    )
    _DATAFRAME_UTILS_AVAILABLE = True
except ImportError:
    _DATAFRAME_UTILS_AVAILABLE = False

__all__ = [
    'MCPServerBuilder',
    'MCPJSONExtractor',
    'create_mcp_preprocessor'
]

# Add DataFrame utilities to __all__ if available
if _DATAFRAME_UTILS_AVAILABLE:
    __all__.extend([
        'BaseDataFrameMCPServerBuilder',
        'PandasMCPServerBuilder',
        'PolarsMCPServerBuilder',
        'PANDAS_AVAILABLE',
        'POLARS_AVAILABLE'
    ]) 