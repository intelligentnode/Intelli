"""
Sample script to test GPT-5 features in Intelli library
Demonstrates:
- Basic GPT-5 chat functionality
- Different reasoning efforts (minimal, low, medium, high)
- Verbosity settings
- Flow integration with GPT-5
"""

import os
import asyncio
from dotenv import load_dotenv
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.flow import Agent, Task, Flow, TextTaskInput, AgentTypes

# Load environment variables
load_dotenv()

# Access API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def test_basic_gpt5_chat():
    """Test basic GPT-5 chat functionality with default settings"""
    print("\n=== Testing Basic GPT-5 Chat ===")
    
    # Initialize chatbot with OpenAI provider (GPT-5 is default)
    chatbot = Chatbot(api_key=OPENAI_KEY, provider=ChatProvider.OPENAI)
    
    # Create chat input - GPT-5 is the default model
    chat_input = ChatModelInput("You are a helpful assistant.")
    chat_input.add_user_message("Explain quantum computing in simple terms.")
    
    # Get response
    response = chatbot.chat(chat_input)
    print(f"GPT-5 Response: {response[0][:200]}...")
    
def test_gpt5_reasoning_efforts():
    """Test GPT-5 with different reasoning efforts"""
    print("\n=== Testing GPT-5 Reasoning Efforts ===")
    
    chatbot = Chatbot(api_key=OPENAI_KEY, provider=ChatProvider.OPENAI)
    
    reasoning_efforts = ["minimal", "low", "medium", "high"]
    
    for effort in reasoning_efforts:
        print(f"\n--- Testing with {effort} reasoning effort ---")
        
        # Create chat input with specific reasoning effort
        chat_input = ChatModelInput(
            "You are a helpful math tutor.",
            "gpt-5",
            reasoning_effort=effort
        )
        chat_input.add_user_message("What is 25 * 37? Show your work.")
        
        # Get response
        response = chatbot.chat(chat_input)
        print(f"Response ({effort}): {response[0][:150]}...")

def test_gpt5_with_verbosity():
    """Test GPT-5 with different verbosity levels"""
    print("\n=== Testing GPT-5 with Verbosity ===")
    
    chatbot = Chatbot(api_key=OPENAI_KEY, provider=ChatProvider.OPENAI)
    
    # Test with different verbosity levels
    verbosity_levels = ["low", "medium", "high"]
    
    for verbosity in verbosity_levels:
        print(f"\n--- Testing with {verbosity} verbosity ---")
        chat_input = ChatModelInput(
            "You are a creative writing assistant.",
            "gpt-5",
            reasoning_effort="low",
            verbosity=verbosity  # Use string values: 'low', 'medium', 'high'
        )
        chat_input.add_user_message("Write a haiku about artificial intelligence.")
        
        response = chatbot.chat(chat_input)
        print(f"Response: {response[0][:150]}...")

async def test_gpt5_in_flow():
    """Test GPT-5 integration in a Flow"""
    print("\n=== Testing GPT-5 in Flow ===")
    
    # Create multiple agents with different reasoning efforts
    analyst_agent = Agent(
        agent_type=AgentTypes.TEXT.value,
        provider="openai",
        mission="You are a data analyst who breaks down complex problems",
        model_params={
            "key": OPENAI_KEY,
            "model": "gpt-5",
            "reasoning_effort": "high"  # High effort for analysis
        }
    )
    
    writer_agent = Agent(
        agent_type=AgentTypes.TEXT.value,
        provider="openai", 
        mission="You are a technical writer who creates clear documentation",
        model_params={
            "key": OPENAI_KEY,
            "model": "gpt-5",
            "reasoning_effort": "medium"  # Medium effort for writing
        }
    )
    
    reviewer_agent = Agent(
        agent_type=AgentTypes.TEXT.value,
        provider="openai",
        mission="You are a reviewer who provides concise feedback",
        model_params={
            "key": OPENAI_KEY,
            "model": "gpt-5", 
            "reasoning_effort": "minimal"  # Minimal effort for quick review
        }
    )
    
    # Create tasks
    analysis_task = Task(
        TextTaskInput("Analyze the pros and cons of using microservices architecture"),
        analyst_agent,
        log=True
    )
    
    documentation_task = Task(
        TextTaskInput("Create a technical documentation based on the analysis"),
        writer_agent,
        log=True
    )
    
    review_task = Task(
        TextTaskInput("Review the documentation and provide feedback"),
        reviewer_agent,
        log=True
    )
    
    # Create flow
    flow = Flow(
        tasks={
            "analysis": analysis_task,
            "documentation": documentation_task,
            "review": review_task
        },
        map_paths={
            "analysis": ["documentation"],
            "documentation": ["review"],
            "review": []
        },
        log=True
    )
    
    # Generate visualization
    flow.generate_graph_img(
        name="gpt5_reasoning_flow",
        save_path="../temp"
    )
    
    # Execute flow
    results = await flow.start(max_workers=2)
    
    print("\n--- Flow Results ---")
    for task_name, result in results.items():
        print(f"\n{task_name.upper()}:")
        if isinstance(result, dict) and "output" in result:
            print(f"{result['output'][:200]}...")
        else:
            print(f"{str(result)[:200]}...")

def test_gpt5_vs_gpt4_compatibility():
    """Test backward compatibility - comparing GPT-5 with GPT-4"""
    print("\n=== Testing GPT-5 vs GPT-4 Compatibility ===")
    
    chatbot = Chatbot(api_key=OPENAI_KEY, provider=ChatProvider.OPENAI)
    
    # Test with GPT-4 (should work as before)
    print("\n--- Testing with GPT-4 ---")
    chat_input_gpt4 = ChatModelInput(
        "You are a helpful assistant.",
        "gpt-4",
        temperature=0.7,
        max_tokens=150
    )
    chat_input_gpt4.add_user_message("What is the capital of France?")
    
    try:
        response = chatbot.chat(chat_input_gpt4)
        print(f"GPT-4 Response: {response[0]}")
    except Exception as e:
        print(f"GPT-4 Error: {e}")
    
    # Test with GPT-5 (temperature and max_tokens should be ignored)
    print("\n--- Testing with GPT-5 ---")
    chat_input_gpt5 = ChatModelInput(
        "You are a helpful assistant.",
        "gpt-5",
        temperature=0.7,  # This will be ignored
        max_tokens=150,   # This will be ignored
        reasoning_effort="minimal"
    )
    chat_input_gpt5.add_user_message("What is the capital of France?")
    
    try:
        response = chatbot.chat(chat_input_gpt5)
        print(f"GPT-5 Response: {response[0]}")
    except Exception as e:
        print(f"GPT-5 Error: {e}")

def test_gpt5_streaming_not_supported():
    """Test that streaming is not supported for GPT-5"""
    print("\n=== Testing GPT-5 Streaming (Should Fail) ===")
    
    chatbot = Chatbot(api_key=OPENAI_KEY, provider=ChatProvider.OPENAI)
    
    chat_input = ChatModelInput(
        "You are a helpful assistant.",
        "gpt-5",
        reasoning_effort="minimal"
    )
    chat_input.add_user_message("Tell me a story about robots.")
    
    try:
        # This should raise NotImplementedError
        for chunk in chatbot.stream(chat_input):
            print(chunk, end="")
    except NotImplementedError as e:
        print(f"✓ Expected error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def main():
    """Run all GPT-5 tests"""
    print("🚀 Starting GPT-5 Feature Tests")
    
    if not OPENAI_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Run synchronous tests
    test_basic_gpt5_chat()
    test_gpt5_reasoning_efforts()
    test_gpt5_with_verbosity()
    test_gpt5_vs_gpt4_compatibility()
    test_gpt5_streaming_not_supported()
    
    # Run async flow test
    print("\n" + "="*50)
    asyncio.run(test_gpt5_in_flow())
    
    print("\n✅ All GPT-5 tests completed!")

if __name__ == "__main__":
    main()
