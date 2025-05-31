import unittest
import os
import json
from dotenv import load_dotenv
import asyncio

from intelli.flow.flow import Flow
from intelli.flow.tasks.task import Task
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tool_connector import ToolDynamicConnector
from intelli.model.input.chatbot_input import ChatModelInput

# Load environment variables
load_dotenv()


class TestFlowToolRouting(unittest.TestCase):
    """Test flow with dynamic tool routing based on LLM decisions"""
    
    def setUp(self):
        """Set up API keys"""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Skip tests if API keys not available
        if not self.openai_key:
            self.skipTest("OPENAI_API_KEY not found")
    
    def test_openai_tool_routing_flow(self):
        """Test flow that routes based on OpenAI tool decisions"""
        print("\n=== Testing OpenAI Tool Routing Flow ===")
        
        # Tools available to the LLM
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_data",
                    "description": "Search for data in a database or knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
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
            mission="You are a helpful assistant. Use the search_data tool when users ask about specific data or facts you don't know.",
            model_params={
                "key": self.openai_key,
                "model": "gpt-4",
                "tools": tools
            }
        )
        
        # Mock MCP agent for testing
        mcp_agent = Agent(
            agent_type="text",  # Using text agent as mock
            provider="openai",
            mission="[MCP Mock] Return mock search results",
            model_params={
                "key": self.openai_key,
                "model": "gpt-3.5-turbo"
            }
        )
        
        # Direct response agent
        direct_agent = Agent(
            agent_type="text",
            provider="openai",
            mission="[Direct Response] Format the response nicely",
            model_params={
                "key": self.openai_key,
                "model": "gpt-3.5-turbo"
            }
        )
        
        # Task definitions
        llm_task = Task(
            TextTaskInput("Process user query and decide if tools are needed"),
            llm_agent
        )
        
        mcp_task = Task(
            TextTaskInput("Execute MCP search"),
            mcp_agent
        )
        
        direct_task = Task(
            TextTaskInput("Provide direct response"),
            direct_agent
        )
        
        # Tool routing connector
        tool_connector = ToolDynamicConnector(
            destinations={
                "tool_called": "mcp_search",
                "no_tool": "direct_response"
            },
            name="tool_router",
            description="Routes to MCP if tools are called, otherwise direct response"
        )
        
        # Flow setup
        flow = Flow(
            tasks={
                "llm_decision": llm_task,
                "mcp_search": mcp_task,
                "direct_response": direct_task
            },
            map_paths={
                "llm_decision": [],  # Dynamic routing will handle connections
                "mcp_search": [],
                "direct_response": []
            },
            dynamic_connectors={
                "llm_decision": tool_connector
            }
        )
        
        # Test 1: Query that might trigger tool use
        print("\n--- Test 1: Query requiring tool ---")
        input1 = TextTaskInput(
            desc="What's the latest data on climate change in 2024?"
        )
        result1 = asyncio.run(flow.start())
        print(f"Result 1: {result1}")
        
        # Verify routing behavior
        if "mcp_search" in result1:
            print("✓ Correctly routed to MCP task")
        else:
            print("! Routed to direct response (LLM decided not to use tool)")
        
        # Test 2: Query that shouldn't trigger tool use  
        print("\n--- Test 2: Query not requiring tool ---")
        # Update task input for second test
        llm_task.task_input = TextTaskInput("What is Python programming language?")
        result2 = asyncio.run(flow.start())
        print(f"Result 2: {result2}")
        
        # Verify routing behavior
        if "direct_response" in result2:
            print("✓ Correctly routed to direct response")
        else:
            print("! Unexpectedly routed to MCP")
    
    def test_anthropic_tool_routing_flow(self):
        """Test flow that routes based on Anthropic tool decisions"""
        if not self.anthropic_key:
            self.skipTest("ANTHROPIC_API_KEY not found")
            
        print("\n=== Testing Anthropic Tool Routing Flow ===")
        
        # Tools for Anthropic
        tools = [
            {
                "name": "analyze_data",
                "description": "Analyze data from a CSV file or database",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data_source": {
                            "type": "string",
                            "description": "The data source to analyze"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis to perform"
                        }
                    },
                    "required": ["data_source"]
                }
            }
        ]
        
        # Anthropic LLM agent with tool support
        llm_agent = Agent(
            agent_type="text",
            provider="anthropic",
            mission="You are a helpful assistant. Use the analyze_data tool when users ask to analyze CSV files or data.",
            model_params={
                "key": self.anthropic_key,
                "model": "claude-3-7-sonnet-20250219",
                "tools": tools
            }
        )
        
        # Data processing agent
        process_agent = Agent(
            agent_type="text",
            provider="anthropic",
            mission="[Data Processor] Simulate data analysis results",
            model_params={
                "key": self.anthropic_key,
                "model": "claude-3-haiku-20240307"
            }
        )
        
        # Simple response agent
        simple_agent = Agent(
            agent_type="text",
            provider="anthropic",
            mission="Provide a simple formatted response",
            model_params={
                "key": self.anthropic_key,
                "model": "claude-3-haiku-20240307"
            }
        )
        
        # Task definitions
        llm_task = Task(
            TextTaskInput("Analyze my sales_data.csv file and show me the trends"),
            llm_agent
        )
        
        process_task = Task(
            TextTaskInput("Process data analysis request"),
            process_agent
        )
        
        simple_task = Task(
            TextTaskInput("Provide simple response"),
            simple_agent
        )
        
        # Tool connector
        tool_connector = ToolDynamicConnector(
            destinations={
                "tool_called": "data_processor",
                "no_tool": "simple_response"
            }
        )
        
        # Flow setup
        flow = Flow(
            tasks={
                "llm_decision": llm_task,
                "data_processor": process_task,
                "simple_response": simple_task
            },
            map_paths={
                "llm_decision": [],
                "data_processor": [],
                "simple_response": []
            },
            dynamic_connectors={
                "llm_decision": tool_connector
            }
        )
        
        # Test with data analysis request
        print("\n--- Test: Data analysis request ---")
        result = asyncio.run(flow.start())
        print(f"Result: {result}")
    
    def test_custom_tool_decision_logic(self):
        """Test ToolDynamicConnector with custom decision logic"""
        print("\n=== Testing Custom Tool Decision Logic ===")
        
        # Custom decision function with keyword checking
        def custom_decision(output, output_type):
            # Standard tool response detection
            if isinstance(output, dict):
                if output.get("type") in ["tool_response", "function_response"]:
                    return "use_tool"
            
            # Keyword-based routing for text responses
            if isinstance(output, str):
                if any(keyword in output.lower() for keyword in ["search", "analyze", "calculate"]):
                    return "maybe_tool"
                else:
                    return "no_tool"
            
            return "no_tool"
        
        # Connector with custom logic
        custom_connector = ToolDynamicConnector(
            decision_fn=custom_decision,
            destinations={
                "use_tool": "tool_task",
                "maybe_tool": "review_task",
                "no_tool": "direct_task",
                "tool_called": "tool_task",
            },
            name="custom_tool_router"
        )
        
        # Test the connector
        # Test 1: Tool response
        tool_output = {
            "type": "tool_response",
            "tool_calls": [{"function": {"name": "test"}}]
        }
        decision1 = custom_connector.get_next_task(tool_output, "text")
        self.assertEqual(decision1, "tool_task")
        print(f"Tool response → {decision1}")
        
        # Test 2: Text with keywords
        text_output = "Let me search for that information"
        decision2 = custom_connector.get_next_task(text_output, "text")
        self.assertEqual(decision2, "review_task")
        print(f"Text with keywords → {decision2}")
        
        # Test 3: Regular text
        regular_output = "Python is a programming language"
        decision3 = custom_connector.get_next_task(regular_output, "text")
        self.assertEqual(decision3, "direct_task")
        print(f"Regular text → {decision3}")
    
    def test_tool_info_extraction(self):
        """Test extracting tool information from outputs"""
        print("\n=== Testing Tool Info Extraction ===")
        
        connector = ToolDynamicConnector(
            destinations={"tool_called": "next", "no_tool": "end"}
        )
        
        # Test OpenAI format
        openai_output = {
            "type": "tool_response",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris", "unit": "celsius"}'
                }
            }]
        }
        
        tool_info = connector.get_tool_info(openai_output)
        self.assertIsNotNone(tool_info)
        self.assertEqual(tool_info["name"], "get_weather")
        self.assertEqual(tool_info["id"], "call_123")
        print(f"OpenAI tool info: {tool_info}")
        
        # Test legacy function format
        legacy_output = {
            "type": "function_response",
            "function_call": {
                "name": "calculate",
                "arguments": '{"a": 5, "b": 10}'
            }
        }
        
        legacy_info = connector.get_tool_info(legacy_output)
        self.assertIsNotNone(legacy_info)
        self.assertEqual(legacy_info["name"], "calculate")
        self.assertIsNone(legacy_info["id"])
        print(f"Legacy function info: {legacy_info}")


if __name__ == "__main__":
    unittest.main() 