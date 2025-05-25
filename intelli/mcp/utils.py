"""
MCP Utilities for Intelli

This module provides utilities for working with MCP (Model Context Protocol),
making it easier to create MCP servers and integrate them with Intelli.
"""
import os
import sys
import re
import json
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar, Tuple

# Import when needed    
T = TypeVar('T')

class MCPServerBuilder:
    """Helper class for building and running MCP servers with various transports"""
    
    def __init__(self, name: str, stateless_http: bool = False):
        """
        Initialize a new MCP server builder
        
        Args:
            name: Name of the MCP server
            stateless_http: Whether to enable stateless HTTP support
        """
        # Lazy import FastMCP only when needed
        try:
            from mcp.server.fastmcp import FastMCP
            self.mcp = FastMCP(name, stateless_http=stateless_http)
            self.tools: List[str] = []
        except ImportError:
            raise ImportError(
                "MCP server utilities require the 'mcp' module. "
                "Install it using 'pip install intelli[mcp]'."
            )
        
    def add_tool(self, func: Callable) -> Callable:
        """
        Add a tool to the MCP server
        
        Args:
            func: Function to use as a tool
            
        Returns:
            The original function for chaining
        """
        self.tools.append(func.__name__)
        return self.mcp.tool()(func)
        
    def run(self, 
            transport: str = "stdio", 
            mount_path: str = "/mcp", 
            host: str = "0.0.0.0", 
            port: int = 8000,
            print_info: bool = True) -> None:
        """
        Run the MCP server with the specified transport
        
        Args:
            transport: Transport to use ("stdio", "http", "streamable-http", etc.)
            mount_path: Path for HTTP server endpoint
            host: Host address for HTTP server (used for info display only)
            port: Port for HTTP server (used for info display only)
            print_info: Whether to print server info
        """
        if print_info:
            self._print_server_info(transport, mount_path, host, port)
        
        # FastMCP.run() only accepts transport and mount_path
        if transport in ["http", "streamable-http"]:
            self.mcp.run(
                transport=transport,
                mount_path=mount_path
            )
        else:
            self.mcp.run(transport=transport)
    
    def _print_server_info(self, transport, mount_path, host, port):
        """Print information about the server"""
        tools_list = ", ".join(self.tools)
        
        print(f"Starting MCP Server '{self.mcp.name}'...")
        print(f"Registered tools: {tools_list}")
        
        if transport in ["http", "streamable-http"]:
            print(f"Server will run at http://{host}:{port}")
            print(f"MCP endpoint URL: http://localhost:{port}{mount_path}")
            print("NOTE: Client must use arg_* parameter prefix (e.g., arg_a, arg_b)")
        else:
            print(f"Server running with {transport} transport")


class MCPJSONExtractor:
    """Utility for extracting JSON data from LLM responses for MCP tools"""
    
    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON from text that might contain explanations or formatting
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON as a dictionary
        """
        text = text.strip()
        
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Try to extract JSON using regex
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0).replace("'", '"')
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Attempt key-value extraction if JSON parsing fails
        result = {}
        
        # Look for "key": value or "key" : value patterns
        kv_pattern = r'"([^"]+)"\s*:\s*(?:"([^"]+)"|(\d+))'
        matches = re.findall(kv_pattern, text)
        
        for match in matches:
            key = match[0]
            # Get the non-empty value (either string or number)
            value = match[1] if match[1] else match[2]
            
            # Convert to number if needed
            if value and value.isdigit():
                value = int(value)
                
            result[key] = value
                
        return result


def create_mcp_preprocessor(
    server_path: str = None,
    server_url: str = None,
    default_tool: str = None,
    operations_map: Dict[str, str] = None,
    param_names: List[str] = None
) -> Callable:
    """
    Create a pre-processor function for MCP tasks that extracts operation details
    
    Args:
        server_path: Path to MCP server script for subprocess transport
        server_url: URL for MCP server for HTTP/WebSocket transport
        default_tool: Default tool to use if extraction fails
        operations_map: Mapping from operation names to tool names
        param_names: List of parameter names to extract from the LLM output
        
    Returns:
        Pre-processor function for MCP task
    """
    # Set defaults
    if default_tool is None:
        default_tool = "help"
    
    if param_names is None:
        param_names = ["a", "b"]  # Default parameters to extract
    
    def preprocess_mcp_input(input_data):
        """Extract operation details from input data"""
        try:
            # Extract JSON from LLM output
            data = MCPJSONExtractor.extract_json(input_data)
            
            if not data:
                print("No data could be extracted, using defaults")
                return _create_model_params_update(server_path, server_url, default_tool, {})
            
            # Get operation and map to tool name if mapping provided
            operation = data.get("operation", "").lower()
            
            if operations_map and operation in operations_map:
                tool = operations_map.get(operation)
            else:
                # If no mapping or operation not found in mapping, use operation as tool name
                # or fall back to default tool
                tool = operation or default_tool
            
            # Extract parameters based on provided param_names
            params = {}
            for param in param_names:
                # Try different possible variations of the parameter name
                for key in [param, f"{param}1", f"{param}2", f"arg_{param}", param.lower(), param.upper()]:
                    if key in data:
                        # Try to convert to int if possible, otherwise keep as is
                        try:
                            params[f"arg_{param}"] = int(data[key])
                        except (ValueError, TypeError):
                            params[f"arg_{param}"] = data[key]
                        break
            
            print(f"Extracted operation: {tool}, params: {params}")
            
            return _create_model_params_update(server_path, server_url, tool, params)
            
        except Exception as e:
            print(f"Error in MCP pre-processor: {e}")
            return _create_model_params_update(server_path, server_url, default_tool, {})
    
    def _create_model_params_update(server_path, server_url, tool, params):
        """Create the model params update dictionary with extracted parameters"""
        # Create base parameters
        model_params = {
            "tool": tool
        }
        
        # Add extracted parameters
        model_params.update(params)
        
        # Add transport-specific parameters
        if server_url:
            # HTTP/WebSocket configuration
            model_params["url"] = server_url
        elif server_path:
            # Subprocess configuration
            model_params["command"] = sys.executable
            model_params["args"] = [server_path]
        else:
            raise ValueError("Either server_path or server_url must be provided")
        
        return {"update_model_params": model_params}
    
    return preprocess_mcp_input 