"""
Tool-aware dynamic connector for routing based on LLM tool/function calls.

This connector examines the output from LLM agents and routes to different
tasks based on whether tools were invoked or not.
"""

from intelli.flow.dynamic_connector import DynamicConnector, ConnectorMode
from typing import Any, Optional, Dict, Callable


class ToolDynamicConnector(DynamicConnector):
    """
    Dynamic connector that routes based on tool/function calls in LLM output.
    
    This connector enables flows to dynamically decide whether to execute
    MCP or other tool-based tasks based on LLM decisions.
    """
    
    def __init__(
        self,
        decision_fn: Optional[Callable[[Any, str], str]] = None,
        destinations: Dict[str, str] = None,
        name: str = "tool_aware_connector",
        description: str = "Routes based on tool usage in LLM output",
        mode: ConnectorMode = ConnectorMode.CUSTOM,
    ):
        """
        Initialize the tool-aware connector.
        
        Args:
            decision_fn: Optional custom decision function (defaults to tool detection)
            destinations: Must include "tool_called" and "no_tool" keys
            name: Connector name
            description: Connector description
            mode: Connector mode
        """
        # Use custom decision function or default tool detection
        if decision_fn is None:
            decision_fn = self._default_tool_decision
            
        super().__init__(decision_fn, destinations, name, description, mode)
        
        # Validate required destinations
        if not destinations or "tool_called" not in destinations or "no_tool" not in destinations:
            raise ValueError(
                "ToolDynamicConnector requires destinations with 'tool_called' and 'no_tool' keys"
            )
    
    def _default_tool_decision(self, output: Any, output_type: str) -> str:
        """
        Default decision function that detects tool usage.
        
        Returns:
            "tool_called" if tools were invoked
            "no_tool" if direct response
            None if cannot determine
        """
        # Check for tool response structure
        if isinstance(output, dict):
            # Check for standard tool response format
            if output.get("type") in ["tool_response", "function_response"]:
                if output.get("tool_calls") or output.get("function_call"):
                    return "tool_called"
        
        # Check for text content (direct response)
        if isinstance(output, str) and output.strip():
            return "no_tool"
        elif isinstance(output, dict) and output.get("content") and not output.get("tool_calls"):
            return "no_tool"
        
        # Cannot determine
        return None
    
    def get_tool_info(self, output: Any) -> Optional[Dict[str, Any]]:
        """
        Extract tool information from the output.
        
        Returns:
            Dict with tool name and arguments, or None if no tools
        """
        if not isinstance(output, dict):
            return None
            
        # Handle new format (tool_calls)
        if output.get("type") == "tool_response" and output.get("tool_calls"):
            first_tool = output["tool_calls"][0]
            return {
                "name": first_tool["function"]["name"],
                "arguments": first_tool["function"].get("arguments", "{}"),
                "id": first_tool.get("id")
            }
        # Handle legacy format (function_call)
        elif output.get("type") == "function_response" and output.get("function_call"):
            return {
                "name": output["function_call"]["name"],
                "arguments": output["function_call"].get("arguments", "{}"),
                "id": None
            }
        
        return None 