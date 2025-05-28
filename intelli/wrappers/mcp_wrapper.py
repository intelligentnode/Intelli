import asyncio
import json
import threading
import functools
import urllib.parse

# ---------------------------------------------------------------------------
# Optional MCP SDK imports
# ---------------------------------------------------------------------------

# Basic availability flag – only requires the core SDK (no transport extras)
try:
    from mcp import ClientSession, StdioServerParameters  # type: ignore
    MCP_AVAILABLE = True
except ImportError as _core_e:  # pragma: no cover
    MCP_AVAILABLE = False
    _MCP_CORE_IMPORT_ERROR = _core_e  # Preserve for error messages

# ----------------------------------------------------
# Optional transport-specific clients.  These live in
# separate extras (e.g.  `mcp[ws]`, `mcp[cli]`).
# If they fail to import we merely disable remote
# capability but still allow stdio usage.
# ----------------------------------------------------

REMOTE_AVAILABLE = False
if MCP_AVAILABLE:
    try:
        from mcp.client.stdio import stdio_client  # type: ignore
        from mcp.client.websocket import websocket_client  # type: ignore
        # Try importing the streamable HTTP client first, as it's preferred
        try:
            from mcp.client.streamable_http import streamablehttp_client as mcp_http_client # Alias
            print("Intelli MCPWrapper: Using streamablehttp_client for HTTP.")
        except ImportError:
            # Fallback to the older http_client if streamable_http is not found
            from mcp.client.http import http_client as mcp_http_client # Alias
            print("Intelli MCPWrapper: Using legacy http_client for HTTP (fallback).")
        REMOTE_AVAILABLE = True
    except ImportError as _transport_e:  # pragma: no cover
        # Remote transports require additional deps (websockets, httpx-sse …)
        _MCP_TRANSPORT_IMPORT_ERROR = _transport_e
        print(f"Intelli MCPWrapper: Failed to import remote transport clients. Error: {_MCP_TRANSPORT_IMPORT_ERROR}")

# Dummy fallbacks to satisfy type checkers when MCP isn't installed
if not MCP_AVAILABLE:
    class ClientSession:  # type: ignore
        pass

    class StdioServerParameters:  # type: ignore
        def __init__(self, command=None, args=None, env=None):
            self.command = command
            self.args = args or []
            self.env = env


class MCPWrapper:
    """
    Wrapper for MCP (Model Context Protocol) servers.
    This wrapper provides methods to interact with MCP servers using the MCP SDK.
    """
    
    def __init__(self, server_config=None):
        """
        Initialize the MCP wrapper.
        
        Args:
            server_config: Either:
                - A string URL for remote MCP server (ws:// or http://)
                - A dict with {'command': cmd, 'args': [], 'env': {}} for local subprocess
                - A dict with {'url': 'ws://...'} for websocket
                - A dict with {'url': 'http://...'} for HTTP
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK is not installed. Install it with 'pip install intelli[mcp]'. "
                f"Original error: {_MCP_CORE_IMPORT_ERROR}"
            )
        
        self.server_params = None
        self.connection_type = None
        self.remote_url = None
        
        # Configure based on input type
        if server_config is None:
            raise ValueError("Server configuration is required")
        
        # String URL - assume websocket if available
        if isinstance(server_config, str):
            if not REMOTE_AVAILABLE:
                raise ImportError(
                    "Remote MCP connection requires additional dependencies. "
                    "Install the MCP SDK with WebSocket/HTTP extras: 'pip install mcp[ws]' "
                    f"Original error: {_MCP_TRANSPORT_IMPORT_ERROR}"
                )
                
            self.remote_url = server_config
            if server_config.startswith('http'):
                self.connection_type = 'http'
            else:
                self.connection_type = 'websocket'
        
        # Dict configuration
        elif isinstance(server_config, dict):
            if 'url' in server_config:
                if not REMOTE_AVAILABLE:
                    raise ImportError(
                        "Remote MCP connection requires additional dependencies. "
                        "Install the MCP SDK with WebSocket/HTTP extras: 'pip install mcp[ws]' "
                        f"Original error: {_MCP_TRANSPORT_IMPORT_ERROR}"
                    )
                    
                self.remote_url = server_config['url']
                if self.remote_url.startswith('http'):
                    self.connection_type = 'http'
                else:
                    self.connection_type = 'websocket'
            
            elif 'command' in server_config:
                self.connection_type = 'stdio'
                self.server_params = StdioServerParameters(
                    command=server_config['command'],
                    args=server_config.get('args', []),
                    env=server_config.get('env')
                )
            else:
                raise ValueError("Invalid server configuration. Provide 'url' or 'command'")
        else:
            raise ValueError("Server configuration must be a URL string or dictionary")
    
    # --------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------
    def _get_http_base_url(self, url):
        """
        Extract the base URL for HTTP connections.
        Full URL with path works best for this client.
        """
        parsed = urllib.parse.urlparse(url)
        
        # For streamable_http client, use the full URL including path
        print(f"Intelli MCPWrapper: Using full URL: {url} for HTTP connection")
        return url

    async def _open(self):
        """
        Return (session, session_ctx, client_ctx) to ensure proper cleanup
        """
        try:
            if self.connection_type == 'stdio':
                client_ctx = stdio_client(self.server_params)
                aenter_result = await client_ctx.__aenter__()
                
                # Handle different return types
                if isinstance(aenter_result, tuple):
                    if len(aenter_result) == 3:
                        read, write, _proc = aenter_result
                    elif len(aenter_result) == 2:
                        read, write = aenter_result
                    else:
                        raise ValueError(f"Unexpected result from stdio_client.__aenter__(): {aenter_result}")
                else:
                    raise ValueError(f"Unexpected result type from stdio_client.__aenter__(): {type(aenter_result)}")
            
            elif self.connection_type == 'websocket':
                client_ctx = websocket_client(self.remote_url)
                read, write = await client_ctx.__aenter__()
                
            elif self.connection_type == 'http':
                # Normalize HTTP URL for the SDK
                normalized_url = self._get_http_base_url(self.remote_url)
                
                print(f"Intelli MCPWrapper: Connecting to HTTP MCP server at {normalized_url}")
                
                # Use the aliased mcp_http_client
                client_ctx = mcp_http_client(normalized_url)
                aenter_result = await client_ctx.__aenter__()
                # streamablehttp_client returns 3 items, older http_client might return 2
                if isinstance(aenter_result, tuple) and len(aenter_result) == 3:
                    read, write, _ = aenter_result 
                elif isinstance(aenter_result, tuple) and len(aenter_result) == 2:
                    read, write = aenter_result 
                else:
                    raise ValueError(f"Unexpected result type or length from HTTP client context manager: {aenter_result}")
            
            session_ctx = ClientSession(read, write)
            session = await session_ctx.__aenter__()
            await session.initialize()
            return session, session_ctx, client_ctx
            
        except Exception as e:
            # Clean up client context if error occurs
            if 'client_ctx' in locals():
                try:
                    await client_ctx.__aexit__(None, None, None)
                except:
                    pass
            raise e
    
    async def _close(self, session_ctx, client_ctx):
        """Close both context managers properly"""
        await session_ctx.__aexit__(None, None, None)
        await client_ctx.__aexit__(None, None, None)
    
    # --------------------------------------------------------------
    # Async implementations
    # --------------------------------------------------------------
    async def _list_tools_async(self):
        """List all available tools from the MCP server."""
        session, session_ctx, client_ctx = await self._open()
        try:
            tools = await session.list_tools()
            return tools
        finally:
            await self._close(session_ctx, client_ctx)
    
    async def _call_tool_async(self, name, arguments):
        """Call a tool on the MCP server."""
        session, session_ctx, client_ctx = await self._open()
        try:
            result = await session.call_tool(name, arguments)
            return result
        finally:
            await self._close(session_ctx, client_ctx)
    
    async def _read_resource_async(self, resource_uri):
        """Read a resource from the MCP server."""
        session, session_ctx, client_ctx = await self._open()
        try:
            result = await session.read_resource(resource_uri)
            return result
        finally:
            await self._close(session_ctx, client_ctx)
    
    # --------------------------------------------------------------
    # Public synchronous facade
    # --------------------------------------------------------------
    def execute_tool(self, name, arguments):
        """
        Synchronous wrapper for calling a tool.
        Runs in the caller's event loop if one exists.
        
        Args:
            name (str): The name of the tool to call.
            arguments (dict): The arguments to pass to the tool.
            
        Returns:
            The result of the tool call.
        """
        try:
            print(f"Executing MCP tool '{name}' with arguments: {arguments}")
            
            # Remove None values and 'input' parameter 
            filtered_args = {k: v for k, v in arguments.items() if v is not None and k != 'input'}
            
            # Process parameters based on naming convention
            # Can handle both normal and arg_* prefixed parameters
            converted_args = {}
            
            for k, v in filtered_args.items():
                if k.startswith('arg_'):
                    # Remove the arg_ prefix
                    param_name = k[4:]
                    converted_args[param_name] = v
                    print(f"Converting parameter '{k}' to '{param_name}', value: {v}")
                else:
                    # Use parameter as is
                    converted_args[k] = v

            print(f"Filtered arguments for tool call: {converted_args}")
            
            coro = self._call_tool_async(name, converted_args)
            try:
                loop = asyncio.get_running_loop()  # Check if in event loop
                result = loop.run_until_complete(coro)
            except RuntimeError:
                result = asyncio.run(coro)  # Create new loop
                
            print(f"MCP tool execution result: {result}")
            return result
        except Exception as e:
            print(f"Error executing tool {name}: {e}")
            return f"Error executing tool {name}: {str(e)}"
    
    def get_resource(self, resource_uri):
        """
        Synchronous wrapper for reading a resource.
        Runs in the caller's event loop if one exists.
        
        Args:
            resource_uri (str): The URI of the resource to read.
            
        Returns:
            tuple: (content, mime_type) of the resource.
        """
        coro = self._read_resource_async(resource_uri)
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
    
    def get_tools(self):
        """
        Synchronous wrapper for listing tools.
        Runs in the caller's event loop if one exists.
        
        Returns:
            List of available tools.
        """
        coro = self._list_tools_async()
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro) 