import unittest
import os
import time
import uuid
import logging
from contextlib import contextmanager
from unittest.mock import MagicMock
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

    def _definition_to_dict(self, definition):
        if definition is None:
            return None
        if isinstance(definition, dict):
            return definition
        for attr_name in ("as_dict", "to_dict", "model_dump", "dict"):
            converter = getattr(definition, attr_name, None)
            if callable(converter):
                converted = converter()
                if isinstance(converted, dict):
                    return converted
        if hasattr(definition, "__dict__") and isinstance(definition.__dict__, dict):
            return definition.__dict__
        return None

    def tearDown(self):
        """
        Runs after each test to clean up any agents we created.
        """
        cleanup_errors = []
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
                cleanup_errors.append(e)
        if cleanup_errors:
            raise cleanup_errors[0]

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
        self._log(
            f"Updated agent: name={updated.name}, version={updated.version}, id={updated_id}"
        )
        self.summary["updated_agent_id"] = updated_id
        if hasattr(updated, "definition") and updated.definition:
            updated_def = updated.definition
        elif hasattr(updated, "get"):
            updated_def = updated.get("definition")
        else:
            updated_def = None
        updated_def_dict = self._definition_to_dict(updated_def)
        if updated_def_dict:
            self.assertEqual(updated_def_dict.get("instructions"), updated_instructions)
            self._log(f"Updated instructions: {updated_def_dict.get('instructions')}")

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
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-bad-meta",
                    instructions="bad",
                    metadata="not-a-dict",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-bad-temp",
                    instructions="bad",
                    temperature="hot",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-bad-top",
                    instructions="bad",
                    top_p="high",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-bad-format",
                    instructions="bad",
                    response_format="xml",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-bad-format-dict",
                    instructions="bad",
                    response_format={"type": "xml"},
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-bad-tools",
                    instructions="bad",
                    tool_resources="not-a-dict",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-tools-not-list",
                    instructions="bad",
                    tools="not-a-list",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-tools-not-dict",
                    instructions="bad",
                    tools=["not-a-dict"],
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_response(
                    conversation_id="conv",
                    agent="agent:1",
                    input_items="not-a-list",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_response(
                    conversation_id="conv",
                    agent="agent:1",
                    input_items=["not-a-dict"],
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_response(
                    conversation_id="conv",
                    agent="agent:1",
                    input_items=[{"role": "user"}],
                )
            with self.assertRaises(ValueError):
                self.wrapper.update_conversation_metadata(
                    conversation_id="conv",
                    metadata="not-a-dict",
                )
            with self.assertRaises(ValueError):
                self.wrapper.submit_tool_outputs(
                    response_id="resp",
                    tool_outputs="not-a-list",
                )
            with self.assertRaises(ValueError):
                self.wrapper.submit_tool_outputs(
                    response_id="resp",
                    tool_outputs=["not-a-dict"],
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_agent(
                    model=self.model,
                    name=f"{self.agent_name}-bad-si-create",
                    instructions="bad",
                    structured_inputs="not-a-dict",
                )
            with self.assertRaises(ValueError):
                self.wrapper.update_agent(
                    agent_id="agent:1",
                    structured_inputs="not-a-dict",
                )
            with self.assertRaises(ValueError):
                self.wrapper.create_response(
                    conversation_id="conv",
                    agent="agent:1",
                    input_text="Hello",
                    structured_inputs="not-a-dict",
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

        updated = self.wrapper.update_conversation_metadata(
            conversation_id=conversation.id,
            metadata={"run_id": self.run_id, "purpose": "metadata-update"},
        )
        updated_meta = getattr(updated, "metadata", None)
        if updated_meta is None and hasattr(updated, "get"):
            updated_meta = updated.get("metadata")
        if isinstance(updated_meta, dict):
            self.assertEqual(updated_meta.get("purpose"), "metadata-update")

        self.wrapper.delete_conversation(conversation.id)

    def test_create_update_with_generation_settings(self):
        print("\n---- Test: Create/update with metadata + sampling settings ----")
        agent, agent_id = self._create_agent(
            name=f"{self.agent_name}-settings",
            instructions="Settings test instructions.",
            description="Settings test",
            tools=[{"type": "code_interpreter"}],
        )
        self._log(f"Created agent: {agent_id}")

        updated = self.wrapper.update_agent(
            agent_id=agent_id,
            instructions="Settings test instructions v2.",
            metadata={"run_id": self.run_id, "purpose": "settings-update"},
            temperature=0.2,
            top_p=0.9,
            response_format="text",
        )
        updated_id = f"{updated.name}:{updated.version}"
        self.created_agent_ids.append(updated_id)
        self._log(f"Updated agent: {updated_id}")

        updated_def = None
        if hasattr(updated, "definition") and updated.definition:
            updated_def = updated.definition
        elif hasattr(updated, "get"):
            updated_def = updated.get("definition")
        updated_def_dict = self._definition_to_dict(updated_def)
        if updated_def_dict:
            self.assertEqual(updated_def_dict.get("instructions"), "Settings test instructions v2.")
            self.assertEqual(updated_def_dict.get("temperature"), 0.2)
            self.assertEqual(updated_def_dict.get("top_p"), 0.9)
            self.assertEqual(updated_def_dict.get("response_format"), {"type": "text"})

    def test_resolve_agent_reference_accepts_zero_version(self):
        print("\n---- Test: Resolve agent reference accepts version 0 ----")
        resolved = self.wrapper._resolve_agent_reference(
            {"name": f"{self.agent_name}-zero-version", "version": 0}
        )
        self.assertEqual(resolved, {"name": f"{self.agent_name}-zero-version", "version": "0"})

    def test_create_agent_with_empty_tools_list(self):
        print("\n---- Test: Create agent with empty tools list ----")
        agent, agent_id = self._create_agent(
            name=f"{self.agent_name}-empty-tools",
            instructions="Empty tools list test.",
            tools=[],
        )
        self._log(f"Created agent: {agent_id}")
        agent_def = getattr(agent, "definition", None)
        if agent_def is None and hasattr(agent, "get"):
            agent_def = agent.get("definition")
        agent_def_dict = self._definition_to_dict(agent_def)
        if agent_def_dict is not None and "tools" in agent_def_dict:
            self.assertEqual(agent_def_dict.get("tools"), [])

    def _response_status(self, response):
        status = getattr(response, "status", None)
        if status is None and hasattr(response, "get"):
            status = response.get("status")
        return status

    def test_create_agent_with_structured_inputs(self):
        """
        create_agent stores structured_inputs in the agent definition so
        Foundry can wire placeholders (e.g. vector_store_id) into the
        agent's tools at response time.
        """
        print("\n---- Test: create_agent with structured_inputs ----")
        structured_inputs = {
            "vector_store_id": {
                "type": "string",
                "description": "Vector store id for file_search.",
            }
        }
        agent = self.wrapper.create_agent(
            model=self.model,
            name=f"{self.agent_name}-si-create",
            instructions="Structured inputs creation test.",
            structured_inputs=structured_inputs,
        )
        agent_id = f"{agent.name}:{agent.version}"
        self.created_agent_ids.append(agent_id)
        self._log(f"Created agent: {agent_id}")

        agent_def_dict = self._definition_to_dict(getattr(agent, "definition", None))
        self.assertIsNotNone(agent_def_dict)
        self.assertEqual(
            agent_def_dict.get("structured_inputs"),
            structured_inputs,
        )

    def test_update_agent_with_structured_inputs(self):
        """
        update_agent forwards structured_inputs into the new agent version's
        definition, and structured_inputs alone is enough to satisfy the
        has_updates guard.
        """
        print("\n---- Test: update_agent with structured_inputs ----")
        agent, agent_id = self._create_agent(
            name=f"{self.agent_name}-si-update",
            instructions="Structured inputs update test.",
        )
        self._log(f"Created agent: {agent_id}")

        structured_inputs = {
            "vector_store_id": {
                "type": "string",
                "description": "Vector store id for file_search.",
            }
        }
        updated = self.wrapper.update_agent(
            agent_id=agent_id,
            structured_inputs=structured_inputs,
        )
        updated_id = f"{updated.name}:{updated.version}"
        self.created_agent_ids.append(updated_id)
        self._log(f"Updated agent: {updated_id}")

        updated_def_dict = self._definition_to_dict(getattr(updated, "definition", None))
        self.assertIsNotNone(updated_def_dict)
        self.assertEqual(
            updated_def_dict.get("structured_inputs"),
            structured_inputs,
        )

    def test_create_response_stream_returns_iterable(self):
        """
        stream=True forwards stream=True to the Responses API and returns
        the raw stream object so the caller can iterate events. The wrapper
        must skip its polling loop in this case.
        """
        print("\n---- Test: create_response stream=True ----")
        agent, agent_id = self._create_agent(
            name=f"{self.agent_name}-stream",
            instructions="Streaming test.",
        )
        self._log(f"Created agent: {agent_id}")

        conversation = self.wrapper.create_conversation(
            items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "Stream a one-sentence reply.",
                }
            ]
        )
        self._log(f"Conversation created: id={conversation.id}")

        stream = self.wrapper.create_response(
            conversation_id=conversation.id,
            agent=agent,
            input_items=[
                {
                    "role": "user",
                    "content": "Reply briefly.",
                }
            ],
            stream=True,
        )

        self.assertTrue(
            hasattr(stream, "__iter__") or hasattr(stream, "__next__"),
            f"stream=True must return an iterable, got {type(stream)!r}",
        )
        event_count = 0
        for _ in stream:
            event_count += 1
            if event_count >= 200:
                break
        self.assertGreater(event_count, 0, "expected at least one streamed event")
        self._log(f"Consumed {event_count} streamed event(s).")

        self.wrapper.delete_conversation(conversation.id)


class TestAzureAgentWrapperPayloadContract(unittest.TestCase):
    """
    Payload-level contract tests for create_response. These assertions
    require inspecting what the wrapper hands to client.responses.create,
    which the live Foundry path does not expose, so we mock the OpenAI
    client. No Azure credentials are required, which is intentional: the
    agent_reference + tools 400 error this guards against is a wrapper
    contract regression, not an Azure outage.
    """

    def setUp(self):
        self.wrapper = AzureAgentWrapper.__new__(AzureAgentWrapper)
        self.wrapper._retry_attempts = 1
        self.wrapper._retry_delay_seconds = 0
        self.wrapper._timeout = None
        self.wrapper.client = MagicMock()

        self.fake_client = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp_test_123"
        mock_response.status = "completed"
        self.fake_client.responses.create.return_value = mock_response
        self.wrapper._openai_client = self.fake_client

        self.wrapper.wait_for_response = MagicMock(return_value=mock_response)

        self.agent_ref = {"name": "agent-x", "version": "1"}
        self.conversation_id = "conv_abc"

    def _create_kwargs(self):
        self.assertTrue(self.fake_client.responses.create.called)
        _, call_kwargs = self.fake_client.responses.create.call_args
        return call_kwargs

    def test_create_response_drops_per_call_tools_with_agent_reference(self):
        """
        With agent_reference set (the only path create_response uses today),
        per-call `tools` must NOT make it into the outbound payload, and a
        warning must be logged so callers can see why their override was
        ignored.
        """
        with self.assertLogs(
            "intelli.wrappers.azure_agent_wrapper", level="WARNING"
        ) as captured:
            self.wrapper.create_response(
                conversation_id=self.conversation_id,
                agent=self.agent_ref,
                input_text="hello",
                wait_for_completion=False,
                tools=[{"type": "code_interpreter"}],
            )

        call_kwargs = self._create_kwargs()
        self.assertNotIn("tools", call_kwargs)
        self.assertIn("agent_reference", call_kwargs["extra_body"])
        self.assertTrue(
            any("Ignoring per-call tools" in msg for msg in captured.output),
            f"expected an 'Ignoring per-call tools' warning, got {captured.output}",
        )

    def test_create_response_drops_tool_resources_with_agent_reference(self):
        """
        Same drop-with-warning behavior when the override comes in via
        tool_resources={'file_search': {...}} rather than tools=[...].
        """
        with self.assertLogs(
            "intelli.wrappers.azure_agent_wrapper", level="WARNING"
        ):
            self.wrapper.create_response(
                conversation_id=self.conversation_id,
                agent=self.agent_ref,
                input_text="hello",
                wait_for_completion=False,
                tool_resources={"file_search": {"vector_store_ids": ["vs_1"]}},
            )

        call_kwargs = self._create_kwargs()
        self.assertNotIn("tools", call_kwargs)

    def test_create_response_includes_structured_inputs_in_extra_body(self):
        """
        structured_inputs is forwarded inside extra_body alongside
        agent_reference (NOT as a top-level kwarg) so the OpenAI Python
        SDK doesn't reject it as an unknown parameter. Pins the contract
        that broke against live Foundry with a TypeError.
        """
        structured_inputs = {"vector_store_id": "vs_per_call_xyz"}
        self.wrapper.create_response(
            conversation_id=self.conversation_id,
            agent=self.agent_ref,
            input_text="hello",
            wait_for_completion=False,
            structured_inputs=structured_inputs,
        )

        call_kwargs = self._create_kwargs()
        self.assertNotIn("structured_inputs", call_kwargs)
        self.assertEqual(
            call_kwargs["extra_body"]["structured_inputs"],
            structured_inputs,
        )
        self.assertIn("agent_reference", call_kwargs["extra_body"])
        self.assertNotIn("tools", call_kwargs)

    def test_create_response_default_payload_unchanged(self):
        """
        Backward-compat guard: when none of the new kwargs are supplied,
        the outbound payload contains exactly the legacy keys.
        """
        self.wrapper.create_response(
            conversation_id=self.conversation_id,
            agent=self.agent_ref,
            input_text="hello",
        )

        call_kwargs = self._create_kwargs()
        self.assertEqual(
            set(call_kwargs.keys()),
            {"conversation", "input", "extra_body"},
        )

    def test_create_response_rejects_non_dict_structured_inputs(self):
        """structured_inputs must be a dict — caller error surfaces as ValueError."""
        wrapper_logger = logging.getLogger("intelli.wrappers.azure_agent_wrapper")
        previous_level = wrapper_logger.level
        previous_propagate = wrapper_logger.propagate
        wrapper_logger.setLevel(logging.CRITICAL + 1)
        wrapper_logger.propagate = False
        try:
            with self.assertRaises(ValueError):
                self.wrapper.create_response(
                    conversation_id=self.conversation_id,
                    agent=self.agent_ref,
                    input_text="hello",
                    structured_inputs="not-a-dict",
                )
        finally:
            wrapper_logger.setLevel(previous_level)
            wrapper_logger.propagate = previous_propagate


if __name__ == "__main__":
    unittest.main()

