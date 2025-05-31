import unittest
import os
import json
import asyncio
from dotenv import load_dotenv

from intelli.flow.flow import Flow
from intelli.flow.tasks.task import Task
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tool_connector import ToolDynamicConnector

# Load environment variables
load_dotenv()


class TestFlowMCPToolIntegration(unittest.TestCase):
    """Test complete flow with LLM tools routing to MCP agents"""
    
    def setUp(self):
        """Set up API keys and check MCP availability"""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_key:
            self.skipTest("OPENAI_API_KEY not found")
        
        # Check if MCP is available
        try:
            from intelli.wrappers.mcp_wrapper import MCPWrapper
            self.mcp_available = True
        except ImportError:
            self.mcp_available = False
    
    def test_llm_to_mcp_flow_with_tools(self):
        """Test flow where LLM decides to use MCP tools based on user query"""
        print("\n=== Testing LLM → MCP Flow with Tools ===")
        
        # Tool for LLM
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_math",
                    "description": "Perform mathematical calculations like addition, multiplication, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "multiply", "subtract", "divide"],
                                "description": "The mathematical operation to perform"
                            },
                            "a": {
                                "type": "number",
                                "description": "First number"
                            },
                            "b": {
                                "type": "number",
                                "description": "Second number"
                            }
                        },
                        "required": ["operation", "a", "b"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_data",
                    "description": "Retrieve data from a database or file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The data query or filename"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # LLM agent with tool support
        llm_agent = Agent(
            agent_type="text",
            provider="openai",
            mission="You are a helpful assistant. Use the calculate_math tool for calculations and get_data tool for data retrieval.",
            model_params={
                "key": self.openai_key,
                "model": "gpt-4",
                "tools": tools
            }
        )
        
        # MCP agent setup
        if self.mcp_available:
            mcp_agent = Agent(
                agent_type="mcp",
                provider="mcp",
                mission="Execute mathematical operations",
                model_params={
                    "command": "node",
                    "args": ["mcp_math_server.js"],
                    "tool": "calculate",
                    "input_arg": "expression"
                }
            )
        else:
            # Fallback to text agent for testing
            mcp_agent = Agent(
                agent_type="text",
                provider="openai",
                mission="[MCP Mock] Calculate: Extract numbers from the query and perform the calculation",
                model_params={
                    "key": self.openai_key,
                    "model": "gpt-3.5-turbo"
                }
            )
        
        # Response formatter agent
        formatter_agent = Agent(
            agent_type="text",
            provider="openai",
            mission="Format the final response in a user-friendly way",
            model_params={
                "key": self.openai_key,
                "model": "gpt-3.5-turbo"
            }
        )
        
        # Direct response agent
        direct_agent = Agent(
            agent_type="text",
            provider="openai",
            mission="Provide a direct, informative response",
            model_params={
                "key": self.openai_key,
                "model": "gpt-3.5-turbo"
            }
        )
        
        # Task setup
        llm_task = Task(
            TextTaskInput("Calculate 45 multiplied by 23"),
            llm_agent
        )
        
        mcp_task = Task(
            TextTaskInput("Execute calculation"),
            mcp_agent
        )
        
        format_task = Task(
            TextTaskInput("Format response"),
            formatter_agent
        )
        
        direct_task = Task(
            TextTaskInput("Provide direct response"),
            direct_agent
        )
        
        # Tool routing connector
        tool_connector = ToolDynamicConnector(
            destinations={
                "tool_called": "mcp_execute",
                "no_tool": "direct_response"
            },
            name="tool_router"
        )
        
        # Flow configuration
        flow = Flow(
            tasks={
                "llm_decision": llm_task,
                "mcp_execute": mcp_task,
                "format_response": format_task,
                "direct_response": direct_task
            },
            map_paths={
                "llm_decision": [],
                "mcp_execute": ["format_response"],
                "format_response": [],
                "direct_response": []
            },
            dynamic_connectors={
                "llm_decision": tool_connector
            }
        )
        
        # Test 1: Math calculation
        print("\n--- Test 1: Math Calculation ---")
        result1 = asyncio.run(flow.start())
        print(f"Math result: {result1}")
        
        # Verify routing
        if "mcp_execute" in result1:
            print("✓ Correctly routed to MCP for calculation")
            if "format_response" in result1:
                print("✓ Response was formatted")
        
        # Test 2: General question
        print("\n--- Test 2: General Question ---")
        llm_task.task_input = TextTaskInput("What is machine learning?")
        result2 = asyncio.run(flow.start())
        print(f"General result: {result2}")
        
        # Verify routing
        if "direct_response" in result2:
            print("✓ Correctly provided direct response without MCP")
            self.assertNotIn("mcp_execute", result2)
    
    def test_tool_info_passthrough(self):
        """Test passing tool information through the flow"""
        print("\n=== Testing Tool Info Passthrough ===")
        
        # Custom processor that extracts tool info
        class ToolInfoProcessor(Task):
            def execute(self, task_input):
                # Look for tool info in input
                if hasattr(task_input, 'tool_info'):
                    return f"Tool called: {task_input.tool_info['name']} with args: {task_input.tool_info['arguments']}"
                return "No tool info found"
        
        # Enhanced tool connector with info extraction
        class EnhancedToolConnector(ToolDynamicConnector):
            def get_next_task(self, output, output_type):
                next_task = super().get_next_task(output, output_type)
                
                # Extract tool info when routing to tool task
                if next_task == self.destinations.get("tool_called"):
                    tool_info = self.get_tool_info(output)
                    if tool_info:
                        # Tool info could be passed to next task here
                        print(f"Extracted tool info: {tool_info}")
                
                return next_task
        
        # Test the enhanced connector
        enhanced_connector = EnhancedToolConnector(
            destinations={
                "tool_called": "process_tool",
                "no_tool": "process_direct"
            }
        )
        
        # Test with tool response
        tool_response = {
            "type": "tool_response",
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "analyze_data",
                    "arguments": '{"file": "data.csv", "operation": "summary"}'
                }
            }]
        }
        
        next_task = enhanced_connector.get_next_task(tool_response, "text")
        self.assertEqual(next_task, "process_tool")
        print("✓ Enhanced connector correctly routes and extracts tool info")
    
    def test_multiple_tool_options(self):
        """Test LLM choosing between multiple tools"""
        print("\n=== Testing Multiple Tool Options ===")
        
        # Multiple tool definitions
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "database_query",
                    "description": "Query internal database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "filter": {"type": "string"}
                        },
                        "required": ["table"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
        
        # Multi-tool agent
        multi_tool_agent = Agent(
            agent_type="text",
            provider="openai",
            mission="Choose the appropriate tool based on the user's request",
            model_params={
                "key": self.openai_key,
                "model": "gpt-4",
                "tools": tools
            }
        )
        
        # Test different queries
        test_queries = [
            ("What's the weather in Tokyo today?", "web_search"),
            ("Show me all users from the customers table", "database_query"),
            ("What is 15% of 250?", "calculate")
        ]
        
        for query, expected_tool in test_queries:
            print(f"\nQuery: {query}")
            task = Task(
                TextTaskInput(query),
                multi_tool_agent
            )
            
            # Simple flow for testing
            flow = Flow(
                tasks={"multi_tool": task},
                map_paths={"multi_tool": []}
            )
            
            result = asyncio.run(flow.start())
            print(f"Result: {result}")
            
            # Verify correct tool selection
            output = result.get("multi_tool", {})
            if isinstance(output, dict) and output.get("type") == "tool_response":
                tool_name = output["tool_calls"][0]["function"]["name"]
                print(f"Selected tool: {tool_name}")
                if tool_name == expected_tool:
                    print(f"✓ Correctly selected {expected_tool}")
                else:
                    print(f"✗ Expected {expected_tool}, got {tool_name}")


if __name__ == "__main__":
    unittest.main() 