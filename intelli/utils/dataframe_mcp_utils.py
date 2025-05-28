"""
DataFrame MCP Utilities for Intelli - Backward Compatibility Module

This module maintains backward compatibility by re-exporting DataFrame utilities
from their new location in the mcp package.

DEPRECATED: Please import from intelli.mcp.dataframe_utils instead.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from intelli.utils.dataframe_mcp_utils is deprecated. "
    "Please import from intelli.mcp instead: "
    "from intelli.mcp import PandasMCPServerBuilder, PolarsMCPServerBuilder",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location for backward compatibility
try:
    from intelli.mcp.dataframe_utils import (
        BaseDataFrameMCPServerBuilder,
        PandasMCPServerBuilder,
        PolarsMCPServerBuilder,
        PANDAS_AVAILABLE,
        POLARS_AVAILABLE
    )
except ImportError:
    # If the new location isn't available, provide error message
    def _create_unavailable_class(name):
        class UnavailableClass:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"{name} is no longer available from this location. "
                    "Please install the MCP package with 'pip install intelli[mcp]' "
                    "and import from intelli.mcp instead."
                )
        return UnavailableClass
    
    BaseDataFrameMCPServerBuilder = _create_unavailable_class("BaseDataFrameMCPServerBuilder")
    PandasMCPServerBuilder = _create_unavailable_class("PandasMCPServerBuilder")
    PolarsMCPServerBuilder = _create_unavailable_class("PolarsMCPServerBuilder")
    PANDAS_AVAILABLE = False
    POLARS_AVAILABLE = False

# Maintain the same __all__ for compatibility
__all__ = [
    'BaseDataFrameMCPServerBuilder',
    'PandasMCPServerBuilder',
    'PolarsMCPServerBuilder',
    'PANDAS_AVAILABLE',
    'POLARS_AVAILABLE'
] 