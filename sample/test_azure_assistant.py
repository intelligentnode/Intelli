"""
Sample script demonstrating Azure Assistant wrapper usage
This script shows how to create, manage, and interact with Azure OpenAI Assistants
"""

import sys
import os
from dotenv import load_dotenv

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelli.wrappers.azure_assistant_wrapper import AzureAssistantWrapper

# Load environment variables
load_dotenv()


def main():
    print("🚀 Azure Assistant Wrapper Sample\n")
    
    # Get Azure credentials
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not endpoint:
        print("❌ Error: Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in your .env file")
        return
    
    # Initialize the wrapper
    print("📦 Initializing Azure Assistant Wrapper...")
    wrapper = AzureAssistantWrapper(
        api_key=api_key,
        base_url=endpoint
    )
    print("✅ Wrapper initialized successfully\n")
    
    # Example 1: Create a GPT-4 assistant
    print("=" * 60)
    print("Example 1: Create GPT-4 Assistant")
    print("=" * 60)
    
    gpt4_assistant = wrapper.create_assistant(
        name="Sample GPT-4 Assistant",
        model="gpt-4",
        instructions="You are a helpful assistant that provides clear and concise answers.",
        description="A sample assistant using GPT-4",
        temperature=0.4
    )
    
    print(f"✅ Created GPT-4 Assistant:")
    print(f"   ID: {gpt4_assistant.id}")
    print(f"   Name: {gpt4_assistant.name}")
    print(f"   Model: {gpt4_assistant.model}\n")
    
    # Example 2: Create a GPT-5-mini assistant
    print("=" * 60)
    print("Example 2: Create GPT-5-mini Assistant")
    print("=" * 60)
    
    gpt5_assistant = wrapper.create_assistant(
        name="Sample GPT-5-mini Assistant",
        model="gpt-5-mini",
        instructions="You are an advanced assistant with reasoning capabilities.",
        description="A sample assistant using GPT-5-mini",
        reasoning_effort="low"
    )
    
    print(f"✅ Created GPT-5-mini Assistant:")
    print(f"   ID: {gpt5_assistant.id}")
    print(f"   Name: {gpt5_assistant.name}")
    print(f"   Model: {gpt5_assistant.model}\n")
    
    # Example 3: Create assistant with tools
    print("=" * 60)
    print("Example 3: Create Assistant with Tools")
    print("=" * 60)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform basic arithmetic operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate, e.g. '2 + 2'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    tools_assistant = wrapper.create_assistant(
        name="Sample Assistant with Tools",
        model="gpt-4",
        instructions="You are a helpful assistant with access to weather and calculator tools.",
        description="An assistant with custom tools",
        tools=tools,
        temperature=0.3
    )
    
    print(f"✅ Created Assistant with Tools:")
    print(f"   ID: {tools_assistant.id}")
    print(f"   Tools: {len(tools_assistant.tools)} tool(s)\n")
    
    # Example 4: Create assistant with metadata
    print("=" * 60)
    print("Example 4: Create Assistant with Metadata")
    print("=" * 60)
    
    metadata = {
        "environment": "development",
        "version": "1.0.0",
        "created_by": "sample_script",
        "category": "general"
    }
    
    metadata_assistant = wrapper.create_assistant(
        name="Sample Assistant with Metadata",
        model="gpt-4",
        instructions="You are a sample assistant.",
        metadata=metadata
    )
    
    print(f"✅ Created Assistant with Metadata:")
    print(f"   ID: {metadata_assistant.id}")
    print(f"   Metadata: {metadata_assistant.metadata}\n")
    
    # Example 5: Retrieve an assistant
    print("=" * 60)
    print("Example 5: Retrieve Assistant")
    print("=" * 60)
    
    retrieved = wrapper.retrieve_assistant(gpt4_assistant.id)
    print(f"✅ Retrieved Assistant:")
    print(f"   ID: {retrieved.id}")
    print(f"   Name: {retrieved.name}")
    print(f"   Model: {retrieved.model}")
    print(f"   Created: {retrieved.created_at}\n")
    
    # Example 6: Update an assistant
    print("=" * 60)
    print("Example 6: Update Assistant")
    print("=" * 60)
    
    updated = wrapper.update_assistant(
        assistant_id=gpt4_assistant.id,
        name="Updated GPT-4 Assistant",
        instructions="Updated instructions: You are an expert assistant.",
        description="An updated assistant description"
    )
    
    print(f"✅ Updated Assistant:")
    print(f"   ID: {updated.id}")
    print(f"   Name: {updated.name}")
    print(f"   Instructions: {updated.instructions[:50]}...\n")
    
    # Example 7: Update assistant model (switch from GPT-4 to GPT-5-mini)
    print("=" * 60)
    print("Example 7: Update Assistant Model (GPT-4 to GPT-5-mini)")
    print("=" * 60)
    
    model_updated = wrapper.update_assistant(
        assistant_id=gpt4_assistant.id,
        model="gpt-5-mini",
        reasoning_effort="medium"
    )
    
    print(f"✅ Model Updated:")
    print(f"   ID: {model_updated.id}")
    print(f"   Old Model: gpt-4")
    print(f"   New Model: {model_updated.model}\n")
    
    # Example 8: List assistants
    print("=" * 60)
    print("Example 8: List Assistants")
    print("=" * 60)
    
    assistants = wrapper.list_assistants(limit=10)
    print(f"✅ Found {len(assistants.data)} assistant(s):")
    for assistant in assistants.data[:5]:  # Show first 5
        print(f"   - {assistant.name} (ID: {assistant.id}, Model: {assistant.model})")
    print()
    
    # Example 9: Delete assistants (cleanup)
    print("=" * 60)
    print("Example 9: Cleanup - Delete Assistants")
    print("=" * 60)
    
    assistants_to_delete = [
        gpt4_assistant.id,
        gpt5_assistant.id,
        tools_assistant.id,
        metadata_assistant.id
    ]
    
    for assistant_id in assistants_to_delete:
        try:
            deletion_status = wrapper.delete_assistant(assistant_id)
            if deletion_status.deleted:
                print(f"✅ Deleted assistant: {assistant_id}")
        except Exception as e:
            print(f"⚠️  Could not delete assistant {assistant_id}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

