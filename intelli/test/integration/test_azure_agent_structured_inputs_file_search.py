"""
End-to-end test for AzureAgentWrapper.create_response with the
structured_inputs + file_search workflow described in:
https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/structured-inputs

Two complementary tests live here:

* TestAzureAgentStructuredInputsFileSearch (real-Azure)
    Creates a temporary agent with a ``{{vector_store_id}}`` placeholder
    in its file_search tool, runs a response with the actual vector store
    id supplied via ``structured_inputs``, and asserts the assistant's
    reply contains a configurable expected snippet that only the vector
    store's contents could provide. Skipped cleanly when any required
    env var is missing.

* TestAzureAgentStructuredInputsContract (no Azure required)
    Mock-only test that pins the wrapper's outbound-payload contract:
    ``structured_inputs`` must travel inside ``extra_body`` next to
    ``agent_reference``, never as a top-level OpenAI SDK kwarg (which
    fails against live Foundry with
    ``TypeError: Responses.create() got an unexpected keyword argument
    'structured_inputs'``).

Run:
    python -m unittest \
        intelli.test.integration.test_azure_agent_structured_inputs_file_search -v
"""

import os
import unittest
import uuid
from unittest.mock import MagicMock

from dotenv import load_dotenv

from intelli.wrappers.azure_agent_wrapper import AzureAgentWrapper

load_dotenv()


REQUIRED_ENV_VARS = (
    "AZURE_AGENT_SWEDEN_PAYG_CONNECTION_STRING",
    "AZURE_AGENT_SWEDEN_PAYG_CLIENT_ID",
    "AZURE_AGENT_SWEDEN_PAYG_CLIENT_SECRET",
    "AZURE_AGENT_SWEDEN_PAYG_TENANT_ID",
    "AZURE_AGENT_TEST_VECTOR_STORE_ID",
    "AZURE_AGENT_TEST_EXPECTED_TEXT",
    "AZURE_AGENT_TEST_MODEL",
)


def _extract_response_text(response):
    """
    Best-effort extraction of the assistant's textual reply, tolerant of
    both the SDK's typed objects and dict-like payloads.
    """
    text = getattr(response, "output_text", None)
    if text:
        return text
    if hasattr(response, "get"):
        text = response.get("output_text")
        if text:
            return text

    output = getattr(response, "output", None)
    if output is None and hasattr(response, "get"):
        output = response.get("output")
    if not output:
        return ""

    chunks = []
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type is None and hasattr(item, "get"):
            item_type = item.get("type")
        if item_type and item_type != "message":
            continue
        content = getattr(item, "content", None)
        if content is None and hasattr(item, "get"):
            content = item.get("content")
        if not content:
            continue
        for piece in content:
            piece_text = getattr(piece, "text", None)
            if piece_text is None and hasattr(piece, "get"):
                piece_text = piece.get("text")
            if isinstance(piece_text, str):
                chunks.append(piece_text)
            elif isinstance(piece_text, dict):
                value = piece_text.get("value")
                if isinstance(value, str):
                    chunks.append(value)
    return "\n".join(chunks)


class TestAzureAgentStructuredInputsFileSearch(unittest.TestCase):
    """
    Real-Foundry end-to-end test: a temporary agent declares a
    ``{{vector_store_id}}`` template on its file_search tool, the wrapper
    supplies the actual id through ``structured_inputs`` per response,
    and the assistant's reply must contain a configurable snippet that
    can only come from the vector store contents.
    """

    @classmethod
    def setUpClass(cls):
        missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
        if missing:
            raise unittest.SkipTest(
                "Skipping structured_inputs file_search end-to-end test; "
                f"missing required env vars: {', '.join(missing)}"
            )
        try:
            import azure.ai.projects  # noqa: F401
            import azure.identity  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(
                "Skipping structured_inputs file_search end-to-end test; "
                f"Azure SDK not installed ({exc}). Install with: "
                "pip install azure-ai-projects azure-identity"
            )
        cls.vector_store_id = os.environ["AZURE_AGENT_TEST_VECTOR_STORE_ID"]
        cls.expected_text = os.environ["AZURE_AGENT_TEST_EXPECTED_TEXT"]
        cls.model = os.environ["AZURE_AGENT_TEST_MODEL"]

    def setUp(self):
        self.run_id = uuid.uuid4().hex[:8]
        self.created_agent_id = None
        self.created_conversation_id = None

        self._env_overrides = {
            "AZURE_PROJECT_CONNECTION_STRING": os.environ[
                "AZURE_AGENT_SWEDEN_PAYG_CONNECTION_STRING"
            ],
            "AZURE_CLIENT_ID": os.environ["AZURE_AGENT_SWEDEN_PAYG_CLIENT_ID"],
            "AZURE_CLIENT_SECRET": os.environ[
                "AZURE_AGENT_SWEDEN_PAYG_CLIENT_SECRET"
            ],
            "AZURE_TENANT_ID": os.environ["AZURE_AGENT_SWEDEN_PAYG_TENANT_ID"],
        }
        self._env_saved = {key: os.environ.get(key) for key in self._env_overrides}
        for key, value in self._env_overrides.items():
            os.environ[key] = value

        try:
            self.wrapper = AzureAgentWrapper(
                retry_attempts=3,
                retry_delay_seconds=0.5,
                timeout=60,
            )
        except Exception:
            self._restore_env()
            raise

    def _restore_env(self):
        for key, original in self._env_saved.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    def tearDown(self):
        try:
            if self.created_conversation_id:
                try:
                    self.wrapper.delete_conversation(self.created_conversation_id)
                    print(f"Cleaned up conversation: {self.created_conversation_id}")
                except Exception as exc:
                    print(
                        f"Warning: failed to delete conversation "
                        f"{self.created_conversation_id}: {exc}"
                    )
            if self.created_agent_id:
                try:
                    self.wrapper.delete_agent(self.created_agent_id)
                    print(f"Cleaned up agent: {self.created_agent_id}")
                except Exception as exc:
                    print(
                        f"Warning: failed to delete agent "
                        f"{self.created_agent_id}: {exc}"
                    )
        finally:
            self._restore_env()

    def test_create_response_with_structured_inputs_file_search(self):
        print("\n---- Test: structured_inputs + file_search end-to-end ----")
        agent_name = f"structured-input-file-search-test-{self.run_id}"
        agent = self.wrapper.create_agent(
            model=self.model,
            name=agent_name,
            instructions=(
                "You are a test agent. Use file_search to answer questions "
                "from the provided vector store. Answer concisely."
            ),
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": ["{{vector_store_id}}"],
                }
            ],
            structured_inputs={
                "vector_store_id": {
                    "description": "Per-call vector store id for file search",
                    "required": True,
                    "schema": {"type": "string"},
                }
            },
        )
        agent_id = f"{agent.name}:{agent.version}"
        self.created_agent_id = agent_id
        print(f"Created agent: {agent_id}")

        question = (
            "What doctor names are mentioned in the reference file? "
            "Answer with only the names."
        )
        conversation = self.wrapper.create_conversation(
            items=[{"type": "message", "role": "user", "content": question}]
        )
        self.created_conversation_id = conversation.id
        print(f"Created conversation: {conversation.id}")

        response = self.wrapper.create_response(
            conversation_id=conversation.id,
            agent=agent_id,
            input_text=question,
            structured_inputs={"vector_store_id": self.vector_store_id},
            wait_for_completion=True,
            timeout_seconds=120,
        )

        status = getattr(response, "status", None)
        if status is None and hasattr(response, "get"):
            status = response.get("status")
        self.assertEqual(
            status,
            "completed",
            f"expected response.status='completed', got {status!r}",
        )

        response_text = _extract_response_text(response)
        self.assertTrue(
            response_text and response_text.strip(),
            "expected a non-empty assistant reply",
        )
        print(f"Response text: {response_text}")

        self.assertIn(
            self.expected_text.lower(),
            response_text.lower(),
            "expected the response to reference the vector store contents "
            f"(snippet={self.expected_text!r}); got: {response_text!r}",
        )


class TestAzureAgentStructuredInputsContract(unittest.TestCase):
    """
    Mock-only contract test, runs without Azure credentials. Pins the
    `structured_inputs` envelope so we never regress back to passing it
    as a top-level OpenAI SDK kwarg, which fails against live Foundry
    with ``TypeError: Responses.create() got an unexpected keyword
    argument 'structured_inputs'``.
    """

    def test_structured_inputs_ride_inside_extra_body(self):
        wrapper = AzureAgentWrapper.__new__(AzureAgentWrapper)
        wrapper._retry_attempts = 1
        wrapper._retry_delay_seconds = 0
        wrapper._timeout = None
        wrapper.client = MagicMock()

        fake_client = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp_test_123"
        mock_response.status = "completed"
        fake_client.responses.create.return_value = mock_response
        wrapper._openai_client = fake_client

        wrapper.wait_for_response = MagicMock(return_value=mock_response)

        wrapper.create_response(
            conversation_id="conv_abc",
            agent={"name": "agent-x", "version": "1"},
            input_text="hello",
            structured_inputs={"vector_store_id": "vs_test"},
            wait_for_completion=False,
        )

        fake_client.responses.create.assert_called_once()
        _, call_kwargs = fake_client.responses.create.call_args
        self.assertNotIn(
            "structured_inputs",
            call_kwargs,
            "structured_inputs must NOT be a top-level kwarg on "
            "client.responses.create — the OpenAI SDK rejects unknown "
            "parameters with TypeError.",
        )
        self.assertEqual(
            call_kwargs["extra_body"]["structured_inputs"],
            {"vector_store_id": "vs_test"},
        )
        self.assertIn("agent_reference", call_kwargs["extra_body"])


if __name__ == "__main__":
    unittest.main()
