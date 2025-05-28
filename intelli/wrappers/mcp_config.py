"""
MCP configuration helper functions to simplify agent creation.
"""

def local_server_config(script_path, python_executable=None, env=None):
    """
    Create configuration for a local MCP server running from a Python script.
    
    Args:
        script_path: Path to the MCP server script
        python_executable: Python executable to use (default: sys.executable)
        env: Environment variables for the subprocess
        
    Returns:
        Dict with MCP agent configuration
    """
    import sys
    
    executable = python_executable or sys.executable
    
    return {
        "command": executable,
        "args": [script_path],
        "env": env
    }

def websocket_server_config(url):
    """
    Create configuration for a WebSocket MCP server.
    
    Args:
        url: WebSocket URL (ws:// or wss://)
        
    Returns:
        Dict with MCP agent configuration
    """
    return {"url": url}

def http_server_config(url):
    """
    Create configuration for an HTTP MCP server.
    
    Args:
        url: HTTP URL (http:// or https://)
        
    Returns:
        Dict with MCP agent configuration
    """
    return {"url": url}

def create_mcp_agent(server_config, tool_name, **tool_args):
    """
    Create an MCP agent with the given configuration.
    
    Args:
        server_config: Server configuration created by one of the helper functions
        tool_name: Name of the tool to execute
        **tool_args: Arguments to pass to the tool
        
    Returns:
        Dict with model_params for creating an MCP agent
    """
    # Prepare model parameters
    model_params = {
        "tool": tool_name,
        **server_config
    }
    
    # Add tool arguments with arg_ prefix
    for arg_name, arg_value in tool_args.items():
        model_params[f"arg_{arg_name}"] = arg_value
    
    return model_params 