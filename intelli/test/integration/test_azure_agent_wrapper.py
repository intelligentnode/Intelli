import unittest
import os
import time
import uuid
import logging
from contextlib import contextmanager
from dotenv import load_dotenv
from intelli.wrappers.azure_agent_wrapper import AzureAgentWrapper

load_dotenv()


class TestAzureAgentWrapper(unittest.TestCase):
    def setUp(self):
        self.connection_string = os.getenv("AZURE_PROJECT_CONNECTION_STRING")
        if not self.connection_string:
            self.skipTest("AZURE_PROJECT_CONNECTION_STRING not found in environment variables")

        self.model = os.getenv("AZURE_AGENT_MODEL", "gpt-5.2")
        self.verbose = os.getenv("AZURE_AGENT_TEST_VERBOSE", "1") not in ("0", "false", "False")
        self.run_id = uuid.uuid4().hex[:8]
        self.agent_name = f"test-foundry-agent-{self.run_id}"
        self.wrapper = AzureAgentWrapper(
            connection_string=self.connection_string,
            retry_attempts=4,
            retry_delay_seconds=0.75,
            timeout=30,
        )
        self.created_agent_ids = []
        self.summary = {}

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _retry(self, label: str, func, attempts: int = 4, delay_seconds: float = 0.5):
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                self._log(f"{label} failed (attempt {attempt}/{attempts}): {exc}")
                if attempt < attempts:
                    time.sleep(delay_seconds)
        raise last_error

    @contextmanager
    def _suppress_wrapper_logs(self):
        logger = logging.getLogger("intelli.wrappers.azure_agent_wrapper")
        previous_level = logger.level
        previous_propagate = logger.propagate
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False
        try:
            yield
        finally:
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate

    def _create_agent(
        self,
        name: str,
        instructions: str,
        description: str = None,
        tools: list = None,
    ):
        agent = self.wrapper.create_agent(
            model=self.model,
            name=name,
            instructions=instructions,
            description=description,
            tools=tools,
        )
        agent_id = f"{agent.name}:{agent.version}"
        self.created_agent_ids.append(agent_id)
        return agent, agent_id

    def tearDown(self):
        """
        Runs after each test to clean up any agents we created.
        """
        for agent_id in self.created_agent_ids:
            try:
                self.wrapper.delete_agent(agent_id)
                print(f"Cleaned up agent: {agent_id}")
            except Exception as e:
                error_text = str(e).lower()
                if "not_found" in error_text or "not found" in error_text:
                    print(f"Warning: Agent already deleted: {agent_id}")
                    continue
                print(f"Error: Failed to cleanup agent {agent_id}: {e}")
                raise

    def test_agent_crud_and_run(self):
        print("\n---- Test: Azure Agent CRUD + Run ----")
        print(f"Run ID: {self.run_id}")

        self._log("Creating agent (version 1)...")
        agent = self.wrapper.create_agent(
            model=self.model,
            name=self.agent_name,
            instructions="You are a helpful assistant for testing purposes.",
            description="Test agent for CRUD and run flow",
        )
        self.assertIsNotNone(agent)
        agent_id = f"{agent.name}:{agent.version}"
        self._log(f"Created agent: name={agent.name}, version={agent.version}, id={agent_id}")
        self.created_agent_ids.append(agent_id)
        self.summary["created_agent_id"] = agent_id

        self._log("Fetching agent version 1...")
        fetched_agent = self.wrapper.get_agent(agent_id)
        self.assertEqual(fetched_agent.name, agent.name)
        self.assertEqual(fetched_agent.version, agent.version)
        self._log(f"Fetched agent: name={fetched_agent.name}, version={fetched_agent.version}")

        self._log(f"Listing agent versions for name={agent.name}...")
        agents = self.wrapper.list_agents(agent.name)
        if hasattr(agents, "data"):
            agent_items = agents.data
        else:
            agent_items = list(agents)
        if self.verbose:
            print("Agent versions:")
            for item in agent_items:
                item_name = getattr(item, "name", None)
                item_version = getattr(item, "version", None)
                item_id = getattr(item, "id", None)
                print(f"  - name={item_name} version={item_version} id={item_id}")
        self.assertTrue(
            any(
                getattr(item, "id", None) == agent_id
                or (getattr(item, "name", None) == agent.name and getattr(item, "version", None) == agent.version)
                for item in agent_items
            )
        )

        self._log("Updating agent (creates new version)...")
        updated_instructions = "Updated instructions for test."
        updated = self.wrapper.update_agent(
            agent_id=agent_id,
            instructions=updated_instructions,
        )
        self.assertEqual(updated.name, agent.name)
        self.assertNotEqual(updated.version, agent.version)
        updated_id = f"{updated.name}:{updated.version}"
        self.created_agent_ids.append(updated_id)
        self._log(f"Updated agent: name={updated.name}, version={updated.version}, id={updated.id}")
        self.summary["updated_agent_id"] = updated_id
        if hasattr(updated, "definition") and updated.definition:
            updated_def = updated.definition
        elif hasattr(updated, "get"):
            updated_def = updated.get("definition")
        else:
            updated_def = None
        if updated_def:
            self.assertEqual(updated_def.get("instructions"), updated_instructions)
            self._log(f"Updated instructions: {updated_def.get('instructions')}")

        self._log("Creating conversation...")
        conversation = self.wrapper.create_conversation(
            items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "Explain Azure AI Foundry in one short paragraph.",
                }
            ]
        )
        self._log(f"Conversation created: id={conversation.id}")
        response = self.wrapper.create_response(
            conversation_id=conversation.id,
            agent=updated,
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "Please answer directly.",
                }
            ],
        )
        response_status = getattr(response, "status", None)
        if response_status is None and hasattr(response, "get"):
            response_status = response.get("status")
        self.assertEqual(response_status, "completed")

        output_items = getattr(response, "output", None)
        if output_items is None and hasattr(response, "get"):
            output_items = response.get("output")
        self.assertTrue(output_items)
        self._log(f"Response completed: status={response_status}")
        if output_items:
            preview = output_items[0]
            if isinstance(preview, dict):
                self._log(f"Response preview: type={preview.get('type')} role={preview.get('role')}")
                content = preview.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict):
                        text = first.get("text")
                        if isinstance(text, str):
                            self._log(f"Response text: {text[:300]}")

        self._log("Deleting updated agent version...")
        self.wrapper.delete_agent(updated_id)
        self.created_agent_ids.remove(updated_id)
        self._log("Verifying delete via list_versions...")
        remaining_versions = self._retry(
            "List versions after delete",
            lambda: self.wrapper.list_agents(agent.name),
            attempts=5,
            delay_seconds=0.75,
        )
        if hasattr(remaining_versions, "data"):
            remaining_items = remaining_versions.data
        else:
            remaining_items = list(remaining_versions)
        if self.verbose:
            print("Remaining versions:")
            for item in remaining_items:
                item_name = getattr(item, "name", None)
                item_version = getattr(item, "version", None)
                item_id = getattr(item, "id", None)
                print(f"  - name={item_name} version={item_version} id={item_id}")
        self.assertFalse(
            any(getattr(item, "version", None) == updated.version for item in remaining_items)
        )
        self._log("Delete verified.")

        print("Summary:")
        print(f"  agent_name: {self.agent_name}")
        print(f"  created_agent_id: {self.summary.get('created_agent_id')}")
        print(f"  updated_agent_id: {self.summary.get('updated_agent_id')}")

    def test_create_agent_same_name_multiple_versions(self):
        print("\n---- Test: Create same agent name twice ----")
        self._log(f"Base agent name: {self.agent_name}")

        agent_v1, agent_id_v1 = self._create_agent(
            name=self.agent_name,
            instructions="Version one instructions.",
            description="First version for duplicate name test.",
        )
        self._log(f"Created v1: {agent_id_v1}")

        agent_v2, agent_id_v2 = self._create_agent(
            name=self.agent_name,
            instructions="Version two instructions.",
            description="Second version for duplicate name test.",
        )
        self._log(f"Created v2: {agent_id_v2}")

        self.assertEqual(agent_v1.name, agent_v2.name)
        self.assertNotEqual(agent_v1.version, agent_v2.version)

        versions = self.wrapper.list_agents(self.agent_name, as_list=True)
        self.assertTrue(isinstance(versions, list))
        version_ids = {
            getattr(item, "id", None)
            or f"{getattr(item, 'name', None)}:{getattr(item, 'version', None)}"
            for item in versions
        }
        self.assertIn(agent_id_v1, version_ids)
        self.assertIn(agent_id_v2, version_ids)

    def test_validation_errors(self):
        print("\n---- Test: Validation error paths ----")
        with self._suppress_wrapper_logs():
            with self.assertRaises(ValueError):
                self.wrapper.get_agent("missing_version")
            with self.assertRaises(ValueError):
                self.wrapper.update_agent(agent_id="missing_version")
            with self.assertRaises(ValueError):
                self.wrapper.delete_agent("missing_version")
            with self.assertRaises(ValueError):
                self.wrapper.create_response(
                    conversation_id="conv",
                    agent="agent-no-version",
                    input_text="Hello",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_response(
                    conversation_id="conv",
                    agent="agent:1",
                    input_text=None,
                    input_items=None,
                )

    def test_create_conversation_and_response_with_metadata(self):
        print("\n---- Test: Conversation metadata + input normalization ----")
        agent, agent_id = self._create_agent(
            name=f"{self.agent_name}-meta",
            instructions="Metadata test instructions.",
            tools=[{"type": "code_interpreter"}],
        )
        self._log(f"Created agent: {agent_id}")

        conversation = self.wrapper.create_conversation(
            items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "Say hello and mention metadata.",
                }
            ],
            metadata={"run_id": self.run_id, "purpose": "edge-case-test"},
        )
        self.assertIsNotNone(conversation)
        self.assertTrue(getattr(conversation, "id", None))

        response = self.wrapper.create_response(
            conversation_id=conversation.id,
            agent=agent,
            input_items=[
                {
                    "role": "user",
                    "content": "Respond succinctly.",
                }
            ],
        )
        response_status = getattr(response, "status", None)
        if response_status is None and hasattr(response, "get"):
            response_status = response.get("status")
        self.assertEqual(response_status, "completed")


if __name__ == "__main__":
    unittest.main()

