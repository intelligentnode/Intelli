import unittest
import os
import json
from dotenv import load_dotenv

from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput

# Load environment variables
load_dotenv()


class TestChatbotTools(unittest.TestCase):
    """Test chatbot with tools/functions functionality"""
    
    def setUp(self):
        """Set up API keys and chatbot instances"""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Skip tests if API keys not available
        if not self.openai_key:
            self.skipTest("OPENAI_API_KEY not found")
        if not self.anthropic_key:
            self.skipTest("ANTHROPIC_API_KEY not found")
            
        # Initialize chatbots
        self.openai_bot = Chatbot(api_key=self.openai_key, provider="openai")
        self.anthropic_bot = Chatbot(api_key=self.anthropic_key, provider="anthropic")
    
    def test_openai_with_functions(self):
        """Test OpenAI with custom functions (legacy format)"""
        print("\n=== Testing OpenAI with Functions ===")
        
        # Define a weather function
        functions = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
        
        # Create chat input with functions
        chat_input = ChatModelInput(
            system="You are a helpful assistant that can check the weather.",
            model="gpt-4",
            functions=functions
        )
        chat_input.add_user_message("What's the weather like in Paris?")
        
        # Get response
        result = self.openai_bot.chat(chat_input)[0]
        print(f"OpenAI response: {result}")
        
        # Check if function was called
        if isinstance(result, dict) and result.get("type") == "function_response":
            self.assertIn("function_call", result)
            self.assertEqual(result["function_call"]["name"], "get_weather")
            
            # Extract function arguments
            args = json.loads(result["function_call"]["arguments"])
            self.assertIn("location", args)
            print(f"Function called with args: {args}")
        else:
            # Model may choose to respond directly
            print("Model responded directly without calling function")
    
    def test_openai_with_tools(self):
        """Test OpenAI with tools (new format)"""
        print("\n=== Testing OpenAI with Tools ===")
        
        # Calculator tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_sum",
                    "description": "Calculate the sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "First number"
                            },
                            "b": {
                                "type": "number",
                                "description": "Second number"
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]
        
        # Create chat input with tools
        chat_input = ChatModelInput(
            system="You are a helpful assistant that can perform calculations.",
            model="gpt-4",
            tools=tools
        )
        chat_input.add_user_message("What is 15 plus 27?")
        
        # Get response
        result = self.openai_bot.chat(chat_input)[0]
        print(f"OpenAI tools response: {result}")
        
        # Check if tool was called
        if isinstance(result, dict) and result.get("type") == "tool_response":
            self.assertIn("tool_calls", result)
            self.assertEqual(result["tool_calls"][0]["function"]["name"], "calculate_sum")
            
            # Extract tool arguments
            args = json.loads(result["tool_calls"][0]["function"]["arguments"])
            self.assertIn("a", args)
            self.assertIn("b", args)
            print(f"Tool called with args: {args}")
    
    def test_anthropic_with_tools(self):
        """Test Anthropic with tools"""
        print("\n=== Testing Anthropic with Tools ===")
        
        # Tool for Anthropic
        tools = [
            {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock symbol, e.g. AAPL"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        ]
        
        # Create chat input with tools
        chat_input = ChatModelInput(
            system="You are a helpful assistant that can check stock prices.",
            model="claude-3-7-sonnet-20250219",
            tools=tools
        )
        chat_input.add_user_message("What's the current price of Apple stock?")
        
        # Get response
        result = self.anthropic_bot.chat(chat_input)[0]
        print(f"Anthropic response: {result}")
        
        # Verify tool usage
        if isinstance(result, dict) and result.get("type") == "tool_response":
            self.assertIn("tool_calls", result)
            self.assertEqual(result["tool_calls"][0]["function"]["name"], "get_stock_price")
            
            # Extract tool arguments
            args = json.loads(result["tool_calls"][0]["function"]["arguments"])
            self.assertIn("symbol", args)
            print(f"Tool called with args: {args}")
        else:
            print("Model responded directly without calling tool")
    
    def test_no_tool_usage(self):
        """Test that models respond directly when no tools are needed"""
        print("\n=== Testing Direct Response (No Tools) ===")
        
        # Create chat input without tools
        chat_input = ChatModelInput(
            system="You are a helpful assistant.",
            model="gpt-4"
        )
        chat_input.add_user_message("What is Python?")
        
        # Test OpenAI
        result = self.openai_bot.chat(chat_input)[0]
        print(f"OpenAI direct response: {result[:100]}...")
        self.assertIsInstance(result, str)
        self.assertIn("Python", result)
        
        # Test Anthropic
        chat_input.model = "claude-3-7-sonnet-20250219"
        result = self.anthropic_bot.chat(chat_input)[0]
        print(f"Anthropic direct response: {result[:100]}...")
        self.assertIsInstance(result, str)
        self.assertIn("Python", result)


if __name__ == "__main__":
    unittest.main() 