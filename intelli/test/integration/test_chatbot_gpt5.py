import unittest
import os
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.agents.agent import Agent
from intelli.flow.types import AgentTypes
from dotenv import load_dotenv

load_dotenv()


class TestChatbotGPT5(unittest.TestCase):
    
    def setUp(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            self.skipTest("OPENAI_API_KEY not found")
        self.openai_bot = Chatbot(self.openai_api_key, ChatProvider.OPENAI)
    
    def test_gpt5_chat_minimal_effort(self):
        """Test GPT-5 with minimal reasoning effort"""
        print('---- start GPT-5 minimal ----')
        chat_input = ChatModelInput(
            "You are a helpful assistant.", 
            "gpt-5",
            reasoning_effort="minimal"
        )
        chat_input.add_user_message("Write a short poem about AI.")
        
        response = self.openai_bot.chat(chat_input)
        
        print('GPT-5 response (minimal): ', response)
        
        self.assertTrue(len(response) > 0, "GPT-5 chat response should not be empty")
        self.assertTrue(len(response[0]) > 0, "Response content should not be empty")
    
    def test_gpt5_chat_low_effort(self):
        """Test GPT-5 with low reasoning effort"""
        print('---- start GPT-5 low effort ----')
        chat_input = ChatModelInput(
            "You are a helpful assistant.", 
            "gpt-5",
            reasoning_effort="low"
        )
        chat_input.add_user_message("What is the capital of France?")
        
        response = self.openai_bot.chat(chat_input)
        
        print('GPT-5 response (low): ', response)
        
        self.assertTrue(len(response) > 0, "GPT-5 chat response should not be empty")
    
    def test_gpt5_chat_with_verbosity(self):
        """Test GPT-5 with verbosity parameter"""
        print('---- start GPT-5 with verbosity ----')
        chat_input = ChatModelInput(
            "You are a helpful assistant.", 
            "gpt-5",
            reasoning_effort="minimal",
            verbosity="high"
        )
        chat_input.add_user_message("Explain quantum computing.")
        
        response = self.openai_bot.chat(chat_input)
        
        print('GPT-5 response (with verbosity): ', response[0][:200])
        
        self.assertTrue(len(response) > 0, "GPT-5 chat response should not be empty")
    
    def test_gpt5_no_temperature_maxtoken(self):
        """Test that temperature and max_tokens are excluded for GPT-5"""
        print('---- start GPT-5 no temp/max_tokens ----')
        # Even if we set these, they should be excluded for GPT-5
        chat_input = ChatModelInput(
            "You are a helpful assistant.", 
            "gpt-5",
            temperature=0.7,  # This should be ignored
            max_tokens=100,   # This should be ignored
            reasoning_effort="minimal"
        )
        chat_input.add_user_message("Say hello")
        
        # Get the params that would be sent
        params = chat_input.get_openai_input()
        
        # Verify temperature and max_tokens are not in params
        self.assertNotIn('temperature', params, "Temperature should not be in GPT-5 params")
        self.assertNotIn('max_tokens', params, "max_tokens should not be in GPT-5 params")
        
        # Verify reasoning is included
        self.assertIn('reasoning', params, "Reasoning should be in GPT-5 params")
        
        print('GPT-5 params structure:', params.keys())

    def test_gpt5_tools_and_tool_choice_params(self):
        """
        Test GPT-5 params include tools (new feature) and can pass through tool_choice via options.

        This validates our new Responses/tools support in ChatModelInput without depending on
        the model actually choosing to call a tool.
        """
        print('---- start GPT-5 tools/tool_choice params ----')

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_sum",
                    "description": "Calculate the sum of 2 numbers a and b",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        tool_choice = {
            "type": "function",
            "function": {"name": "calculate_sum"},
        }

        chat_input = ChatModelInput(
            "You are a helpful assistant that can use tools.",
            "gpt-5",
            reasoning_effort="minimal",
            tools=tools,
            tool_choice=tool_choice,  # Passed through **options
        )
        chat_input.add_user_message("What is 15 plus 27? Use the tool.")

        params = chat_input.get_openai_input()

        # Basic debug prints (no network call) to confirm payload shape.
        print("GPT-5 params keys:", sorted(params.keys()))
        print("GPT-5 input preview:", params.get("input", "")[:120])
        print("GPT-5 tools count:", len(params.get("tools", [])))
        print("GPT-5 tool_choice:", params.get("tool_choice"))

        # GPT-5 should use Responses-style params
        self.assertIn("input", params, "GPT-5 params should include 'input'")
        self.assertNotIn("messages", params, "GPT-5 params should not include 'messages'")

        # New feature: tools are included for GPT-5
        self.assertIn("tools", params, "GPT-5 params should include tools when provided")
        self.assertEqual(params["tools"], tools)

        # tool_choice should pass through via options
        self.assertIn("tool_choice", params, "GPT-5 params should include tool_choice when provided via options")
        self.assertEqual(params["tool_choice"], tool_choice)

        # Still exclude these for GPT-5
        self.assertNotIn("temperature", params, "Temperature should not be in GPT-5 params")
        self.assertNotIn("max_tokens", params, "max_tokens should not be in GPT-5 params")
    
    def test_gpt5_stream_not_supported(self):
        """Test that streaming raises NotImplementedError for GPT-5"""
        print('---- start GPT-5 stream test ----')
        chat_input = ChatModelInput(
            "You are a helpful assistant.", 
            "gpt-5",
            reasoning_effort="minimal"
        )
        chat_input.add_user_message("Tell me a story")
        
        # Streaming should raise NotImplementedError
        with self.assertRaises(NotImplementedError) as context:
            for _ in self.openai_bot.stream(chat_input):
                pass
        
        self.assertIn("GPT-5", str(context.exception))
        print('Correctly raised NotImplementedError for GPT-5 streaming')
    
    def test_gpt5_simple_flow(self):
        """Test GPT-5 with a simple flow"""
        print('---- start GPT-5 simple flow ----')
        
        # Create a text agent with GPT-5
        agent = Agent(
            agent_type=AgentTypes.TEXT.value,
            provider="openai",
            mission="You are a creative writer",
            model_params={
                "key": self.openai_api_key,
                "model": "gpt-5",
                "reasoning_effort": "minimal"
            }
        )
        
        # Create task input
        task_input = TextTaskInput("Write a creative tagline for a tech company")
        
        # Create a task
        task = Task(
            task_input=task_input,
            agent=agent,
            log=True
        )
        
        # Create and execute flow
        flow = SequenceFlow([task], log='gpt5_flow')
        
        final_result = flow.start(task_input)
        
        print('GPT-5 flow result:', final_result)
        
        self.assertIsNotNone(final_result, "Flow should return a result")
        self.assertTrue(len(str(final_result)) > 0, "Result should not be empty")
    
    def test_backward_compatibility_gpt4(self):
        """Test that existing GPT-4 code still works (backward compatibility)"""
        print('---- start backward compatibility test ----')
        
        # Old code should still work
        chat_input = ChatModelInput(
            "You are a helpful assistant.", 
            "gpt-4o",
            temperature=0.7,
            max_tokens=100
        )
        chat_input.add_user_message("Say hello")
        
        # Get params and verify they still have temperature/max_tokens
        params = chat_input.get_openai_input()
        
        self.assertIn('temperature', params, "Temperature should be in GPT-4 params")
        self.assertIn('max_tokens', params, "max_tokens should be in GPT-4 params")
        self.assertNotIn('reasoning', params, "Reasoning should not be in GPT-4 params")
        
        print('Backward compatibility verified - GPT-4 params:', params.keys())


if __name__ == '__main__':
    unittest.main()

