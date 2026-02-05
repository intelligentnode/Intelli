"""
Sample script demonstrating Azure Agent wrapper usage.
This script creates an agent, runs a thread, and prints the conversation.
"""

import sys
import os
from dotenv import load_dotenv

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelli.wrappers.azure_agent_wrapper import AzureAgentWrapper

# Load environment variables
load_dotenv()


def main():
    print("🚀 Azure Agent Wrapper Sample\n")

    connection_string = os.getenv("AZURE_PROJECT_CONNECTION_STRING")
    if not connection_string:
        print("❌ Error: Please set AZURE_PROJECT_CONNECTION_STRING in your .env file")
        return

    print("📦 Initializing Azure Agent Wrapper...")
    wrapper = AzureAgentWrapper(connection_string=connection_string)
    print("✅ Wrapper initialized successfully\n")

    print("--- Initializing Azure AI Agent in Foundry ---")

    model_name = os.getenv("AZURE_AGENT_MODEL", "gpt-5.2")
    agent = wrapper.create_agent(
        model=model_name,
        name="sample-foundry-agent",
        instructions="You are a helpful AI assistant running in Azure AI Foundry.",
    )
    print(f"Agent created. ID: {agent.name}:{agent.version}")

    conversation = wrapper.create_conversation(
        items=[
            {
                "type": "message",
                "role": "user",
                "content": "Hello, can you explain what Azure AI Foundry is?",
            }
        ]
    )
    print(f"Conversation created. ID: {conversation.id}")

    response = wrapper.create_response(
        conversation_id=conversation.id,
        agent=agent,
        input_text="Please respond in a short paragraph.",
    )
    print(f"Response status: {getattr(response, 'status', None)}")
    output_items = getattr(response, "output", None)
    if output_items is None and hasattr(response, "get"):
        output_items = response.get("output")

    if output_items:
        print("\n--- Conversation Output ---")
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "message":
                print(f"{item.get('role', 'assistant')}: {item.get('content')}")

    # Cleanup (Optional)
    # wrapper.delete_agent(agent.id)


if __name__ == "__main__":
    main()

