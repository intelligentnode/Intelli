import unittest
import os
from dotenv import load_dotenv
from intelli.wrappers.azure_assistant_wrapper import AzureAssistantWrapper

load_dotenv()


class TestAzureAssistantWrapper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key or not self.endpoint:
            self.skipTest("Azure OpenAI credentials not found in environment variables")
        
        self.wrapper = AzureAssistantWrapper(
            api_key=self.api_key,
            base_url=self.endpoint
        )
        
        # Store created assistant IDs for cleanup
        self.created_assistant_ids = []

    def tearDown(self):
        """Clean up created assistants"""
        for assistant_id in self.created_assistant_ids:
            try:
                self.wrapper.delete_assistant(assistant_id)
                print(f"Cleaned up assistant: {assistant_id}")
            except Exception as e:
                print(f"Warning: Failed to cleanup assistant {assistant_id}: {e}")

    def test_create_assistant_gpt4(self):
        """Test creating an assistant with GPT-4 model"""
        print("\n---- Test: Create GPT-4 Assistant ----")
        
        assistant = self.wrapper.create_assistant(
            name="Test GPT-4 Assistant",
            model="gpt-4",
            instructions="You are a helpful assistant for testing purposes.",
            description="Test assistant with GPT-4 model",
            temperature=0.5
        )
        
        self.assertIsNotNone(assistant)
        self.assertIsNotNone(assistant.id)
        self.assertEqual(assistant.name, "Test GPT-4 Assistant")
        self.assertEqual(assistant.model, "gpt-4")
        
        self.created_assistant_ids.append(assistant.id)
        print(f"Created assistant with ID: {assistant.id}")

    def test_create_assistant_gpt5(self):
        """Test creating an assistant with GPT-5-mini model"""
        print("\n---- Test: Create GPT-5-mini Assistant ----")
        
        assistant = self.wrapper.create_assistant(
            name="Test GPT-5-mini Assistant",
            model="gpt-5-mini",
            instructions="You are a helpful assistant for testing purposes.",
            description="Test assistant with GPT-5-mini model",
            reasoning_effort="low"
        )
        
        self.assertIsNotNone(assistant)
        self.assertIsNotNone(assistant.id)
        self.assertEqual(assistant.name, "Test GPT-5-mini Assistant")
        self.assertEqual(assistant.model, "gpt-5-mini")
        
        self.created_assistant_ids.append(assistant.id)
        print(f"Created assistant with ID: {assistant.id}")

    def test_retrieve_assistant(self):
        """Test retrieving an assistant by ID"""
        print("\n---- Test: Retrieve Assistant ----")
        
        # First create an assistant
        assistant = self.wrapper.create_assistant(
            name="Retrieve Test Assistant",
            model="gpt-4",
            instructions="You are a test assistant."
        )
        self.created_assistant_ids.append(assistant.id)
        
        # Now retrieve it
        retrieved = self.wrapper.retrieve_assistant(assistant.id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, assistant.id)
        self.assertEqual(retrieved.name, "Retrieve Test Assistant")
        print(f"Retrieved assistant: {retrieved.id}")

    def test_update_assistant(self):
        """Test updating an assistant"""
        print("\n---- Test: Update Assistant ----")
        
        # Create an assistant
        assistant = self.wrapper.create_assistant(
            name="Update Test Assistant",
            model="gpt-4",
            instructions="Original instructions."
        )
        self.created_assistant_ids.append(assistant.id)
        
        # Update the assistant
        updated = self.wrapper.update_assistant(
            assistant_id=assistant.id,
            name="Updated Test Assistant",
            instructions="Updated instructions.",
            description="Updated description"
        )
        
        self.assertIsNotNone(updated)
        self.assertEqual(updated.id, assistant.id)
        self.assertEqual(updated.name, "Updated Test Assistant")
        self.assertEqual(updated.instructions, "Updated instructions.")
        print(f"Updated assistant: {updated.id}")

    def test_update_assistant_model_switch(self):
        """Test updating assistant model (GPT-4 to GPT-5-mini)"""
        print("\n---- Test: Update Assistant Model Switch ----")
        
        # Create GPT-4 assistant
        assistant = self.wrapper.create_assistant(
            name="Model Switch Test",
            model="gpt-4",
            instructions="Test instructions."
        )
        self.created_assistant_ids.append(assistant.id)
        
        # Switch to GPT-5-mini
        updated = self.wrapper.update_assistant(
            assistant_id=assistant.id,
            model="gpt-5-mini",
            reasoning_effort="medium"
        )
        
        self.assertIsNotNone(updated)
        self.assertEqual(updated.model, "gpt-5-mini")
        print(f"Switched assistant model to: {updated.model}")

    def test_list_assistants(self):
        """Test listing assistants"""
        print("\n---- Test: List Assistants ----")
        
        # Create a test assistant
        assistant = self.wrapper.create_assistant(
            name="List Test Assistant",
            model="gpt-4",
            instructions="Test assistant for listing."
        )
        self.created_assistant_ids.append(assistant.id)
        
        # List assistants
        assistants = self.wrapper.list_assistants(limit=10)
        
        self.assertIsNotNone(assistants)
        # Check if our assistant is in the list
        assistant_ids = [a.id for a in assistants.data]
        self.assertIn(assistant.id, assistant_ids)
        print(f"Found {len(assistants.data)} assistants")

    def test_delete_assistant(self):
        """Test deleting an assistant"""
        print("\n---- Test: Delete Assistant ----")
        
        # Create an assistant
        assistant = self.wrapper.create_assistant(
            name="Delete Test Assistant",
            model="gpt-4",
            instructions="Test assistant to be deleted."
        )
        assistant_id = assistant.id
        
        # Delete it
        deletion_status = self.wrapper.delete_assistant(assistant_id)
        
        self.assertIsNotNone(deletion_status)
        self.assertTrue(deletion_status.deleted)
        self.assertEqual(deletion_status.id, assistant_id)
        print(f"Deleted assistant: {assistant_id}")
        
        # Verify it's deleted by trying to retrieve (should fail)
        try:
            self.wrapper.retrieve_assistant(assistant_id)
            self.fail("Assistant should have been deleted")
        except Exception:
            # Expected to fail
            pass

    def test_create_assistant_with_tools(self):
        """Test creating an assistant with tools"""
        print("\n---- Test: Create Assistant with Tools ----")
        
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
            }
        ]
        
        assistant = self.wrapper.create_assistant(
            name="Tools Test Assistant",
            model="gpt-4",
            instructions="You are a helpful assistant with access to weather tools.",
            tools=tools
        )
        
        self.assertIsNotNone(assistant)
        self.assertIsNotNone(assistant.tools)
        self.assertEqual(len(assistant.tools), 1)
        
        self.created_assistant_ids.append(assistant.id)
        print(f"Created assistant with tools: {assistant.id}")

    def test_create_assistant_with_metadata(self):
        """Test creating an assistant with metadata"""
        print("\n---- Test: Create Assistant with Metadata ----")
        
        metadata = {
            "environment": "test",
            "version": "1.0",
            "created_by": "test_script"
        }
        
        assistant = self.wrapper.create_assistant(
            name="Metadata Test Assistant",
            model="gpt-4",
            instructions="Test assistant with metadata.",
            metadata=metadata
        )
        
        self.assertIsNotNone(assistant)
        self.assertIsNotNone(assistant.metadata)
        self.assertEqual(assistant.metadata.get("environment"), "test")
        
        self.created_assistant_ids.append(assistant.id)
        print(f"Created assistant with metadata: {assistant.id}")


if __name__ == "__main__":
    unittest.main()

