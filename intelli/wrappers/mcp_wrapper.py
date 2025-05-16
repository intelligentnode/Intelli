import asyncio
import json
import threading
import functools

# Try to import MCP, but make it optional
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Define dummy classes for type hints
    class ClientSession:
        pass
    class StdioServerParameters:
        def __init__(self, command=None, args=None, env=None):
            self.command = command
            self.args = args or []
            self.env = env


class MCPWrapper:
    """
    Wrapper for MCP (Model Context Protocol) servers.
    This wrapper provides methods to interact with MCP servers using the MCP SDK.
    """
    
    def __init__(self, server_command, server_args=None, server_env=None):
        """
        Initialize the MCP wrapper.
        
        Args:
            server_command (str): The command to run the MCP server.
            server_args (list, optional): List of arguments to pass to the server command.
            server_env (dict, optional): Environment variables for the server process.
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK is not installed. Please install it using 'pip install mcp-python-client'"
            )
            
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args or [],
            env=server_env
        )
    
    # --------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------
    async def _open(self):
        """
        Return (session, session_ctx, client_ctx) to ensure proper cleanup
        """
        client_ctx = stdio_client(self.server_params)
        try:
            # Try to handle both old and new MCP API versions
            aenter_result = await client_ctx.__aenter__()
            
            # Check if the result is a tuple with 2 or 3 elements
            if isinstance(aenter_result, tuple):
                if len(aenter_result) == 3:
                    read, write, _proc = aenter_result
                elif len(aenter_result) == 2:
                    read, write = aenter_result
                else:
                    raise ValueError(f"Unexpected result from stdio_client.__aenter__(): {aenter_result}")
            else:
                # If it's not a tuple, assume it's some other object with the streams
                raise ValueError(f"Unexpected result type from stdio_client.__aenter__(): {type(aenter_result)}")
                
            session_ctx = ClientSession(read, write)
            session = await session_ctx.__aenter__()
            await session.initialize()
            return session, session_ctx, client_ctx
        except Exception as e:
            # If anything goes wrong, ensure we clean up the client context
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
            
            print(f"Filtered arguments for tool call: {filtered_args}")
            
            coro = self._call_tool_async(name, filtered_args)
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