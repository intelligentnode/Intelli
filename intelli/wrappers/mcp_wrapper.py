import asyncio
import datetime
import json
import logging
import threading
import functools
import urllib.parse
import contextlib

logger = logging.getLogger(__name__)

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
# Optional transport-specific clients. Each transport is imported
# independently and guarded so that a missing optional dependency disables
# only that transport instead of all remote capability. The legacy
# `mcp.client.http` module was removed from the SDK; streamable-http is the
# modern HTTP client and SSE is provided separately.
# ----------------------------------------------------

stdio_client = None
websocket_client = None
streamablehttp_client = None
sse_client = None
_MCP_TRANSPORT_IMPORT_ERROR = None

if MCP_AVAILABLE:
    try:
        from mcp.client.stdio import stdio_client  # type: ignore
    except ImportError as _e:  # pragma: no cover
        _MCP_TRANSPORT_IMPORT_ERROR = _e
    try:
        from mcp.client.streamable_http import streamablehttp_client  # type: ignore
    except ImportError as _e:  # pragma: no cover
        _MCP_TRANSPORT_IMPORT_ERROR = _e
    try:
        from mcp.client.sse import sse_client  # type: ignore
    except ImportError as _e:  # pragma: no cover
        _MCP_TRANSPORT_IMPORT_ERROR = _e
    try:
        from mcp.client.websocket import websocket_client  # type: ignore
    except ImportError as _e:  # pragma: no cover
        # WebSocket transport needs the `websockets` extra.
        _MCP_TRANSPORT_IMPORT_ERROR = _e

# Backward-compatible alias for the older internal name + remote capability flag.
mcp_http_client = streamablehttp_client
REMOTE_AVAILABLE = any(
    client is not None for client in (streamablehttp_client, sse_client, websocket_client)
)

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
    
    def __init__(self, server_config=None, timeout=None):
        """
        Initialize the MCP wrapper.

        Args:
            server_config: Either:
                - A string URL for remote MCP server (ws:// / http:// / https://)
                - A dict with {'command': cmd, 'args': [], 'env': {}} for local subprocess
                - A dict with {'url': '...'} for a remote server. Optional dict keys:
                    'headers': {..} auth/custom headers for http/sse transports
                    'transport': 'http' | 'streamable_http' | 'sse' | 'websocket'
                    'timeout': per-operation read timeout in seconds
            timeout: optional per-operation read timeout (seconds); overridden by a
                'timeout' key inside a dict server_config when both are given.
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK is not installed. Install it with 'pip install intelli[mcp]'. "
                f"Original error: {_MCP_CORE_IMPORT_ERROR}"
            )
        
        self.server_params = None
        self.connection_type = None
        self.remote_url = None
        # Optional auth/custom headers for HTTP (streamable) and SSE transports.
        self.headers = None
        # Optional per-operation read timeout (seconds).
        self.timeout = timeout
        # Optional persistent connection (session, session_ctx, client_ctx)
        self._persistent = None

        # Configure based on input type
        if server_config is None:
            raise ValueError("Server configuration is required")

        # String URL - infer transport from the scheme/path.
        if isinstance(server_config, str):
            self.remote_url = server_config
            self.connection_type = self._resolve_connection_type(server_config)
            self._require_transport(self.connection_type)

        # Dict configuration
        elif isinstance(server_config, dict):
            # Optional headers (auth tokens etc.) for remote HTTP/SSE transports.
            self.headers = server_config.get('headers')
            # Optional per-operation timeout override.
            if server_config.get('timeout') is not None:
                self.timeout = server_config.get('timeout')

            if 'url' in server_config:
                self.remote_url = server_config['url']
                self.connection_type = self._resolve_connection_type(
                    self.remote_url, server_config.get('transport')
                )
                self._require_transport(self.connection_type)
                if self.headers and self.connection_type == 'websocket':
                    logger.warning(
                        "MCP websocket transport does not support custom headers in this "
                        "SDK version; ignoring the provided 'headers'."
                    )

            elif 'command' in server_config:
                if stdio_client is None:
                    raise ImportError(
                        "MCP stdio transport is unavailable. Install the MCP SDK: "
                        f"'pip install intelli[mcp]'. Original error: {_MCP_TRANSPORT_IMPORT_ERROR}"
                    )
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
    @staticmethod
    def _resolve_connection_type(url, explicit_transport=None):
        """
        Determine the transport from an explicit override or the URL.
        Accepted transport overrides: http/streamable_http, sse, websocket/ws.
        """
        if explicit_transport:
            mapping = {
                'http': 'http',
                'streamable_http': 'http',
                'streamable-http': 'http',
                'streamablehttp': 'http',
                'sse': 'sse',
                'websocket': 'websocket',
                'ws': 'websocket',
            }
            key = str(explicit_transport).lower()
            if key not in mapping:
                raise ValueError(
                    f"Unsupported MCP transport '{explicit_transport}'. "
                    "Use one of: http, streamable_http, sse, websocket."
                )
            return mapping[key]

        if url.startswith('ws://') or url.startswith('wss://'):
            return 'websocket'
        # Servers commonly expose SSE on a '/sse' path; default http(s) to streamable-http.
        if url.rstrip('/').endswith('/sse'):
            return 'sse'
        return 'http'

    @staticmethod
    def _require_transport(connection_type):
        """Raise a clear ImportError if the chosen transport's client is unavailable."""
        clients = {
            'http': streamablehttp_client,
            'sse': sse_client,
            'websocket': websocket_client,
            'stdio': stdio_client,
        }
        if clients.get(connection_type) is None:
            hint = "'pip install mcp[ws]'" if connection_type == 'websocket' else "'pip install intelli[mcp]'"
            raise ImportError(
                f"MCP {connection_type} transport is unavailable. Install the required "
                f"dependencies ({hint}). Original error: {_MCP_TRANSPORT_IMPORT_ERROR}"
            )
    def _get_http_base_url(self, url):
        """
        Extract the base URL for HTTP connections.
        Full URL with path works best for this client.
        """
        # For streamable_http client, use the full URL including path
        logger.debug("Intelli MCPWrapper: using full URL %s for HTTP connection", url)
        return url

    def _build_remote_client(self, client_fn, url=None):
        """Instantiate an HTTP/SSE transport client, passing auth headers when set."""
        target = url or self.remote_url
        if self.headers:
            return client_fn(target, headers=self.headers)
        return client_fn(target)

    @staticmethod
    def _unpack_streams(aenter_result, label):
        """Return (read, write) from a transport client's __aenter__ result.

        streamable-http yields (read, write, get_session_id) while SSE/websocket
        yield (read, write); take the first two either way.
        """
        if isinstance(aenter_result, tuple) and len(aenter_result) >= 2:
            return aenter_result[0], aenter_result[1]
        raise ValueError(f"Unexpected result from {label} MCP client: {aenter_result!r}")

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
                # websocket_client does not accept custom headers in the SDK.
                client_ctx = websocket_client(self.remote_url)
                read, write = await client_ctx.__aenter__()

            elif self.connection_type == 'sse':
                logger.debug("Intelli MCPWrapper: connecting to SSE MCP server at %s", self.remote_url)
                client_ctx = self._build_remote_client(sse_client)
                aenter_result = await client_ctx.__aenter__()
                read, write = self._unpack_streams(aenter_result, 'SSE')

            elif self.connection_type == 'http':
                # Normalize HTTP URL for the SDK (streamable-http uses the full URL).
                normalized_url = self._get_http_base_url(self.remote_url)
                logger.debug("Intelli MCPWrapper: connecting to HTTP MCP server at %s", normalized_url)
                client_ctx = self._build_remote_client(streamablehttp_client, url=normalized_url)
                aenter_result = await client_ctx.__aenter__()
                # streamable_http returns (read, write, get_session_id); SSE returns (read, write)
                read, write = self._unpack_streams(aenter_result, 'HTTP')

            else:
                raise ValueError(f"Unsupported MCP connection type: {self.connection_type}")

            # Apply an optional per-operation read timeout when supported by the SDK.
            if self.timeout is not None:
                try:
                    session_ctx = ClientSession(
                        read, write,
                        read_timeout_seconds=datetime.timedelta(seconds=self.timeout),
                    )
                except TypeError:
                    session_ctx = ClientSession(read, write)
            else:
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

    async def _acquire(self):
        """
        Acquire an MCP session.
        If inside a persistent context, reuse that session.
        Returns: (session, session_ctx, client_ctx, should_close)
        """
        if self._persistent is not None:
            session, session_ctx, client_ctx = self._persistent
            return session, session_ctx, client_ctx, False
        session, session_ctx, client_ctx = await self._open()
        return session, session_ctx, client_ctx, True

    async def _release(self, session_ctx, client_ctx, should_close: bool):
        """Release an MCP session based on ownership."""
        if should_close:
            await self._close(session_ctx, client_ctx)
    
    # --------------------------------------------------------------
    # Async implementations
    # --------------------------------------------------------------
    async def _list_tools_async(self):
        """List all available tools from the MCP server."""
        session, session_ctx, client_ctx, should_close = await self._acquire()
        try:
            tools = await session.list_tools()
            return tools
        finally:
            await self._release(session_ctx, client_ctx, should_close)
    
    async def _call_tool_async(self, name, arguments):
        """Call a tool on the MCP server."""
        session, session_ctx, client_ctx, should_close = await self._acquire()
        try:
            result = await session.call_tool(name, arguments)
            return result
        finally:
            await self._release(session_ctx, client_ctx, should_close)
    
    async def _read_resource_async(self, resource_uri):
        """Read a resource from the MCP server."""
        session, session_ctx, client_ctx, should_close = await self._acquire()
        try:
            result = await session.read_resource(resource_uri)
            return result
        finally:
            await self._release(session_ctx, client_ctx, should_close)

    async def _list_resources_async(self):
        """List resources exposed by the MCP server."""
        session, session_ctx, client_ctx, should_close = await self._acquire()
        try:
            return await session.list_resources()
        finally:
            await self._release(session_ctx, client_ctx, should_close)

    async def _list_prompts_async(self):
        """List prompts exposed by the MCP server."""
        session, session_ctx, client_ctx, should_close = await self._acquire()
        try:
            return await session.list_prompts()
        finally:
            await self._release(session_ctx, client_ctx, should_close)

    async def _get_prompt_async(self, name, arguments=None):
        """Fetch a prompt (with optional arguments) from the MCP server."""
        session, session_ctx, client_ctx, should_close = await self._acquire()
        try:
            return await session.get_prompt(name, arguments or {})
        finally:
            await self._release(session_ctx, client_ctx, should_close)

    # --------------------------------------------------------------
    # Persistent connection (optional) – additive, backward compatible
    # --------------------------------------------------------------
    @contextlib.asynccontextmanager
    async def aconnect(self):
        """
        Keep a single MCP connection open across multiple calls.

        Usage:
            async with wrapper.aconnect():
                await wrapper.execute_tool_async(...)
        """
        if self._persistent is not None:
            # Nested usage: reuse the existing persistent connection.
            yield self
            return

        session, session_ctx, client_ctx = await self._open()
        self._persistent = (session, session_ctx, client_ctx)
        try:
            yield self
        finally:
            try:
                await self._close(session_ctx, client_ctx)
            finally:
                self._persistent = None

    @contextlib.contextmanager
    def connect(self):
        """
        Synchronous persistent connection context manager.
        Useful for batching tool calls without reconnect overhead.

        Usage:
            with wrapper.connect():
                wrapper.execute_tool(...)
        """
        # Open connection using a coroutine in a safe way.
        if self._persistent is not None:
            yield self
            return

        session, session_ctx, client_ctx = self._run_coro_sync(self._open())
        self._persistent = (session, session_ctx, client_ctx)
        try:
            yield self
        finally:
            try:
                self._run_coro_sync(self._close(session_ctx, client_ctx))
            finally:
                self._persistent = None

    # --------------------------------------------------------------
    # Async public APIs (additive)
    # --------------------------------------------------------------
    async def execute_tool_async(self, name, arguments):
        """Async: Call a tool on the MCP server."""
        filtered_args = {k: v for k, v in (arguments or {}).items() if v is not None and k != 'input'}
        converted_args = {}
        for k, v in filtered_args.items():
            if k.startswith('arg_'):
                converted_args[k[4:]] = v
            else:
                converted_args[k] = v
        return await self._call_tool_async(name, converted_args)

    async def get_tools_async(self):
        """Async: List all available tools from the MCP server."""
        return await self._list_tools_async()

    async def get_resource_async(self, resource_uri):
        """Async: Read a resource from the MCP server."""
        return await self._read_resource_async(resource_uri)

    async def list_resources_async(self):
        """Async: List resources exposed by the MCP server."""
        return await self._list_resources_async()

    async def get_prompts_async(self):
        """Async: List prompts exposed by the MCP server."""
        return await self._list_prompts_async()

    async def get_prompt_async(self, name, arguments=None):
        """Async: Fetch a prompt (with optional arguments) from the MCP server."""
        return await self._get_prompt_async(name, arguments)

    # --------------------------------------------------------------
    # Coroutine runner for sync facade (safe inside running loops)
    # --------------------------------------------------------------
    def _run_coro_sync(self, coro):
        """
        Run a coroutine from synchronous code.
        - If no event loop is running: asyncio.run(coro)
        - If a loop is running (common in async apps): run in a separate thread
          so we don't call run_until_complete on a running loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result_container = {}
        error_container = {}

        def _runner():
            try:
                result_container["result"] = asyncio.run(coro)
            except Exception as e:
                error_container["error"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()

        if "error" in error_container:
            raise error_container["error"]
        return result_container.get("result")
    
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
            logger.debug("Executing MCP tool '%s'", name)

            # Remove None values and the literal 'input' parameter (guard None args).
            filtered_args = {k: v for k, v in (arguments or {}).items() if v is not None and k != 'input'}

            # Process parameters based on naming convention
            # Can handle both normal and arg_* prefixed parameters
            converted_args = {}

            for k, v in filtered_args.items():
                if k.startswith('arg_'):
                    # Remove the arg_ prefix
                    param_name = k[4:]
                    converted_args[param_name] = v
                else:
                    # Use parameter as is
                    converted_args[k] = v

            logger.debug("Filtered arguments for tool '%s': %s", name, converted_args)

            result = self._run_coro_sync(self._call_tool_async(name, converted_args))

            logger.debug("MCP tool '%s' execution complete", name)
            return result
        except Exception as e:
            logger.error("Error executing tool %s: %s", name, e)
            return f"Error executing tool {name}: {str(e)}"
    
    def get_resource(self, resource_uri):
        """
        Synchronous wrapper for reading a resource.
        Runs in the caller's event loop if one exists.

        Args:
            resource_uri (str): The URI of the resource to read.

        Returns:
            The MCP read_resource result (ReadResourceResult with .contents).
        """
        return self._run_coro_sync(self._read_resource_async(resource_uri))

    # --------------------------------------------------------------
    # Result normalization (additive) — turn a CallToolResult into a
    # plain dict so callers can detect tool errors and structured output.
    # --------------------------------------------------------------
    @staticmethod
    def _convert_args(arguments):
        """Drop None values and the literal 'input' key; strip the 'arg_' prefix."""
        converted = {}
        for k, v in (arguments or {}).items():
            if v is None or k == 'input':
                continue
            converted[k[4:] if k.startswith('arg_') else k] = v
        return converted

    @staticmethod
    def normalize_tool_result(result):
        """
        Normalize an MCP CallToolResult into:
            {'is_error': bool, 'text': str, 'structured': Any, 'content': list}
        Gracefully handles non-result values (e.g. an error string from execute_tool).
        """
        if isinstance(result, str):
            return {'is_error': True, 'text': result, 'structured': None, 'content': []}
        is_error = bool(getattr(result, 'isError', False))
        structured = getattr(result, 'structuredContent', None)
        content = getattr(result, 'content', None) or []
        texts = [getattr(item, 'text') for item in content if getattr(item, 'text', None) is not None]
        if texts:
            text = "\n".join(texts)
        elif structured is not None:
            text = json.dumps(structured)
        else:
            text = ""
        return {'is_error': is_error, 'text': text, 'structured': structured, 'content': content}

    def execute_tool_normalized(self, name, arguments=None):
        """Sync: call a tool and return a normalized {is_error,text,structured,content} dict."""
        result = self._run_coro_sync(self._call_tool_async(name, self._convert_args(arguments)))
        return self.normalize_tool_result(result)

    async def execute_tool_normalized_async(self, name, arguments=None):
        """Async: call a tool and return a normalized {is_error,text,structured,content} dict."""
        result = await self._call_tool_async(name, self._convert_args(arguments))
        return self.normalize_tool_result(result)
    
    def get_tools(self):
        """
        Synchronous wrapper for listing tools.
        Runs in the caller's event loop if one exists.

        Returns:
            List of available tools.
        """
        return self._run_coro_sync(self._list_tools_async())

    def list_resources(self):
        """Synchronous wrapper for listing resources exposed by the MCP server."""
        return self._run_coro_sync(self._list_resources_async())

    def get_prompts(self):
        """Synchronous wrapper for listing prompts exposed by the MCP server."""
        return self._run_coro_sync(self._list_prompts_async())

    def get_prompt(self, name, arguments=None):
        """Synchronous wrapper for fetching a prompt (with optional arguments)."""
        return self._run_coro_sync(self._get_prompt_async(name, arguments))