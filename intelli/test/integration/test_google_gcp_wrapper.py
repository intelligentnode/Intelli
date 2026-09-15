import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

from intelli.wrappers.google_gcp_wrapper import (
    GoogleGCPWrapper,
    _check_adk_imports,
    _check_rag_imports,
    _to_plain,
)

load_dotenv()

WRAPPER_MODULE = "intelli.wrappers.google_gcp_wrapper"


# ----------------------------------------------------------------------
# Fakes shared by the contract tests
# ----------------------------------------------------------------------


class FakePart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call

    @classmethod
    def from_text(cls, text):
        return cls(text=text)


class FakeContent:
    def __init__(self, role=None, parts=None):
        self.role = role
        self.parts = list(parts or [])


class FakeTypes:
    Content = FakeContent
    Part = FakePart


class FakeEvent:
    def __init__(self, text=None, partial=False, final=True, author="agent",
                 function_calls=None):
        self.content = (
            FakeContent(role="model", parts=[FakePart(text=text)])
            if text is not None
            else None
        )
        self.partial = partial
        self.author = author
        self._final = final
        self._function_calls = function_calls or []

    def is_final_response(self):
        return self._final

    def get_function_calls(self):
        return self._function_calls


class FakeSessionService:
    def __init__(self):
        self.sessions = {}
        self.created = []
        self.deleted = []

    async def create_session(self, app_name, user_id, session_id=None, state=None):
        resolved = session_id or "session-%d" % (len(self.created) + 1)
        self.created.append(
            {"app_name": app_name, "user_id": user_id, "session_id": resolved, "state": state}
        )
        session = SimpleNamespace(id=resolved, state=state or {})
        self.sessions[resolved] = session
        return session

    async def get_session(self, app_name, user_id, session_id):
        return self.sessions.get(session_id)

    async def delete_session(self, app_name, user_id, session_id):
        self.deleted.append(session_id)
        self.sessions.pop(session_id, None)


class FakeRunner:
    def __init__(self, events=None):
        self.session_service = FakeSessionService()
        self.events = events or []
        self.calls = []

    async def run_async(self, **kwargs):
        self.calls.append(kwargs)
        for event in self.events:
            yield event


def build_wrapper(**overrides):
    """Construct a wrapper without running __init__, so no credentials are needed."""
    wrapper = GoogleGCPWrapper.__new__(GoogleGCPWrapper)
    wrapper.project_id = "test-project"
    wrapper.location = "us-central1"
    wrapper.credentials = object()
    wrapper.default_model = "gemini-2.5-flash"
    wrapper.app_name = "intelli-test"
    wrapper._timeout = None
    wrapper._retry_attempts = 1
    wrapper._retry_delay_seconds = 0.0
    wrapper._runners = {}
    wrapper._vertex_initialized = True
    for key, value in overrides.items():
        setattr(wrapper, key, value)
    return wrapper


class TestGoogleGCPWrapperContract(unittest.TestCase):
    """
    Contract tests that need no Google Cloud credentials and no google SDKs.

    These cover the parts of the wrapper that live calls cannot observe: the
    credential resolution order, the environment variables ADK depends on, input
    validation, message normalization, and the normalized result shapes. GCP
    credentials are rarely available in CI, so this class is the main regression
    guard for the wrapper.
    """

    def setUp(self):
        self._saved_env = {
            key: os.environ.get(key)
            for key in (
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_GENAI_USE_ENTERPRISE",
            )
        }
        for key in self._saved_env:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # -- auth resolution -------------------------------------------------

    def _patch_auth(self, default_result=None, default_error=None, from_file=None):
        """Patch _check_auth_imports with fake google.auth / service_account modules."""
        auth_module = MagicMock()
        if default_error is not None:
            auth_module.default.side_effect = default_error
        else:
            auth_module.default.return_value = default_result or (
                SimpleNamespace(kind="adc"),
                "adc-project",
            )
        service_account = MagicMock()
        service_account.Credentials.from_service_account_file.return_value = (
            from_file or SimpleNamespace(kind="sa", project_id="sa-project")
        )
        patcher = patch(
            f"{WRAPPER_MODULE}._check_auth_imports",
            return_value=(auth_module, service_account),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return auth_module, service_account

    def test_explicit_credentials_take_priority_over_adc(self):
        auth_module, _ = self._patch_auth()
        explicit = SimpleNamespace(kind="explicit", project_id="explicit-project")

        wrapper = GoogleGCPWrapper(credentials=explicit)

        self.assertIs(wrapper.credentials, explicit)
        self.assertEqual(wrapper.project_id, "explicit-project")
        auth_module.default.assert_not_called()

    def test_credentials_path_takes_priority_over_adc(self):
        auth_module, service_account = self._patch_auth()
        with patch("os.path.isfile", return_value=True):
            wrapper = GoogleGCPWrapper(credentials_path="/tmp/key.json")

        service_account.Credentials.from_service_account_file.assert_called_once()
        self.assertEqual(wrapper.project_id, "sa-project")
        auth_module.default.assert_not_called()

    def test_missing_key_file_raises_value_error(self):
        self._patch_auth()
        with patch("os.path.isfile", return_value=False):
            with self.assertRaises(ValueError) as ctx:
                GoogleGCPWrapper(credentials_path="/tmp/nope.json")
        self.assertIn("not found", str(ctx.exception))

    def test_falls_back_to_application_default_credentials(self):
        auth_module, _ = self._patch_auth()

        wrapper = GoogleGCPWrapper()

        auth_module.default.assert_called_once()
        self.assertEqual(wrapper.project_id, "adc-project")

    def test_adc_failure_raises_actionable_value_error(self):
        self._patch_auth(default_error=Exception("no credentials found"))

        with self.assertRaises(ValueError) as ctx:
            GoogleGCPWrapper()

        message = str(ctx.exception)
        self.assertIn("gcloud auth application-default login", message)
        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS", message)

    def test_missing_project_raises_value_error(self):
        self._patch_auth(default_result=(SimpleNamespace(kind="adc"), None))

        with self.assertRaises(ValueError) as ctx:
            GoogleGCPWrapper()

        self.assertIn("project", str(ctx.exception))

    def test_explicit_project_wins_over_environment(self):
        self._patch_auth()
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"

        wrapper = GoogleGCPWrapper(project_id="arg-project")

        self.assertEqual(wrapper.project_id, "arg-project")
        # ADK reads the environment, so the explicit project must be published there.
        self.assertEqual(os.environ["GOOGLE_CLOUD_PROJECT"], "arg-project")

    def test_backend_env_sets_both_vertex_flags(self):
        self._patch_auth()

        wrapper = GoogleGCPWrapper(project_id="p", location="europe-west4")

        self.assertEqual(os.environ["GOOGLE_CLOUD_LOCATION"], "europe-west4")
        # ADK 2.x renamed the flag, so both names must be set.
        self.assertEqual(os.environ["GOOGLE_GENAI_USE_VERTEXAI"], "1")
        self.assertEqual(os.environ["GOOGLE_GENAI_USE_ENTERPRISE"], "1")
        self.assertEqual(wrapper.location, "europe-west4")

    def test_location_and_model_default_from_config(self):
        self._patch_auth()

        wrapper = GoogleGCPWrapper(project_id="p")

        self.assertEqual(wrapper.location, "us-central1")
        self.assertEqual(wrapper.default_model, "gemini-2.5-flash")

    def test_existing_vertex_flag_is_not_overwritten(self):
        self._patch_auth()
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"

        GoogleGCPWrapper(project_id="p")

        self.assertEqual(os.environ["GOOGLE_GENAI_USE_VERTEXAI"], "0")

    # -- resource name helpers -------------------------------------------

    def test_qualify_corpus_name(self):
        wrapper = build_wrapper()
        self.assertEqual(
            wrapper.qualify_corpus_name("1234"),
            "projects/test-project/locations/us-central1/ragCorpora/1234",
        )
        full = "projects/other/locations/us-east4/ragCorpora/9"
        self.assertEqual(wrapper.qualify_corpus_name(full), full)
        with self.assertRaises(ValueError):
            wrapper.qualify_corpus_name("")

    def test_qualify_data_store_name(self):
        wrapper = build_wrapper()
        self.assertEqual(
            wrapper.qualify_data_store_name("my-store"),
            "projects/test-project/locations/us-central1/collections/"
            "default_collection/dataStores/my-store",
        )

    # -- agent and tool construction -------------------------------------

    def _patch_adk(self, agent_cls=None):
        agent_cls = agent_cls or MagicMock(name="LlmAgent")
        patcher = patch(
            f"{WRAPPER_MODULE}._check_adk_imports",
            return_value=(agent_cls, MagicMock(name="InMemoryRunner"), FakeTypes),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return agent_cls

    def test_create_agent_maps_instructions_to_instruction(self):
        agent_cls = self._patch_adk()
        wrapper = build_wrapper()

        wrapper.create_agent("helper", "Be concise.", output_key="answer")

        _, kwargs = agent_cls.call_args
        self.assertEqual(kwargs["name"], "helper")
        self.assertEqual(kwargs["instruction"], "Be concise.")
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(kwargs["output_key"], "answer")

    def test_create_agent_validates_required_fields(self):
        self._patch_adk()
        wrapper = build_wrapper()

        with self.assertRaises(ValueError):
            wrapper.create_agent("", "instructions")
        with self.assertRaises(ValueError):
            wrapper.create_agent("name", "")

    def test_create_agent_requires_a_model(self):
        self._patch_adk()
        wrapper = build_wrapper(default_model=None)

        with self.assertRaises(ValueError) as ctx:
            wrapper.create_agent("helper", "Be concise.")

        self.assertIn("model is required", str(ctx.exception))

    def test_create_agent_wraps_bare_callables_as_tools(self):
        agent_cls = self._patch_adk()
        wrapper = build_wrapper()
        sentinel = object()

        def lookup_order(order_id: str) -> str:
            """Look up an order."""
            return order_id

        with patch.object(wrapper, "create_function_tool", return_value=sentinel) as wrap:
            wrapper.create_agent("helper", "Be concise.", tools=[lookup_order])

        wrap.assert_called_once_with(lookup_order)
        _, kwargs = agent_cls.call_args
        self.assertEqual(kwargs["tools"], [sentinel])

    def test_create_agent_passes_prebuilt_tools_through(self):
        agent_cls = self._patch_adk()
        wrapper = build_wrapper()
        tool = SimpleNamespace(name="already_a_tool")

        wrapper.create_agent("helper", "Be concise.", tools=tool)

        _, kwargs = agent_cls.call_args
        self.assertEqual(kwargs["tools"], [tool])

    def test_vertex_search_tool_requires_exactly_one_id(self):
        wrapper = build_wrapper()
        with self.assertRaises(ValueError):
            wrapper.create_vertex_search_tool()
        with self.assertRaises(ValueError):
            wrapper.create_vertex_search_tool(data_store_id="a", search_engine_id="b")

    def test_vertex_search_tool_qualifies_data_store(self):
        wrapper = build_wrapper()
        tool_cls = MagicMock(name="VertexAiSearchTool")
        with patch(f"{WRAPPER_MODULE}._resolve_symbol", return_value=tool_cls):
            wrapper.create_vertex_search_tool(data_store_id="store-1")

        _, kwargs = tool_cls.call_args
        self.assertTrue(kwargs["data_store_id"].startswith("projects/test-project/"))

    def test_rag_retrieval_tool_qualifies_corpora(self):
        wrapper = build_wrapper()
        tool_cls = MagicMock(name="VertexAiRagRetrieval")
        with patch(f"{WRAPPER_MODULE}._resolve_symbol", return_value=tool_cls):
            wrapper.create_rag_retrieval_tool(rag_corpora="corpus-7", similarity_top_k=3)

        _, kwargs = tool_cls.call_args
        self.assertEqual(
            kwargs["rag_corpora"],
            ["projects/test-project/locations/us-central1/ragCorpora/corpus-7"],
        )
        self.assertEqual(kwargs["similarity_top_k"], 3)

    def test_rag_retrieval_tool_requires_a_source(self):
        wrapper = build_wrapper()
        with self.assertRaises(ValueError):
            wrapper.create_rag_retrieval_tool()

    def test_unsupported_kwargs_are_dropped_with_a_warning(self):
        """SDK drift must degrade to a warning, not a TypeError."""
        wrapper = build_wrapper()

        def narrow_tool(name, description):
            return {"name": name, "description": description}

        with self.assertLogs(WRAPPER_MODULE, level="WARNING") as captured:
            result = wrapper._call_filtered(
                "narrow_tool", narrow_tool, name="n", description="d", extra_field=1
            )

        self.assertEqual(result, {"name": "n", "description": "d"})
        self.assertTrue(any("extra_field" in line for line in captured.output))

    # -- message normalization -------------------------------------------

    def test_message_normalization(self):
        self._patch_adk()
        wrapper = build_wrapper()

        from_str = wrapper._to_content("hello")
        self.assertEqual(from_str.role, "user")
        self.assertEqual(from_str.parts[0].text, "hello")

        from_dict = wrapper._to_content({"role": "model", "text": "hi"})
        self.assertEqual(from_dict.role, "model")

        prebuilt = FakeContent(role="user", parts=[FakePart(text="kept")])
        self.assertIs(wrapper._to_content(prebuilt), prebuilt)

        for bad in ("", "   ", {"role": "user"}, 42):
            with self.assertRaises(ValueError):
                wrapper._to_content(bad)

    # -- execution -------------------------------------------------------

    def _wrapper_with_runner(self, events):
        wrapper = build_wrapper()
        runner = FakeRunner(events)
        patcher = patch.object(
            wrapper, "_get_runner", return_value=(runner, "intelli-test")
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self._patch_adk()
        return wrapper, runner

    def test_run_async_normalizes_the_result(self):
        call = SimpleNamespace(name="lookup_order", args={"order_id": "7"})
        events = [
            FakeEvent(text="thinking", final=False, author="agent",
                      function_calls=[call]),
            FakeEvent(text="final answer", final=True, author="agent"),
        ]
        wrapper, runner = self._wrapper_with_runner(events)

        result = asyncio.run(wrapper.run_async(SimpleNamespace(), "hello"))

        self.assertEqual(result["text"], "final answer")
        self.assertEqual(result["author"], "agent")
        self.assertEqual(result["session_id"], "session-1")
        self.assertEqual(
            result["function_calls"], [{"name": "lookup_order", "args": {"order_id": "7"}}]
        )
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(runner.calls[0]["user_id"], GoogleGCPWrapper.DEFAULT_USER_ID)

    def test_run_async_creates_a_session_when_none_is_given(self):
        wrapper, runner = self._wrapper_with_runner([FakeEvent(text="ok")])

        asyncio.run(wrapper.run_async(SimpleNamespace(), "hello"))

        self.assertEqual(len(runner.session_service.created), 1)

    def test_run_async_reuses_a_supplied_session(self):
        wrapper, runner = self._wrapper_with_runner([FakeEvent(text="ok")])

        result = asyncio.run(
            wrapper.run_async(SimpleNamespace(), "hello", session_id="session-abc")
        )

        self.assertEqual(result["session_id"], "session-abc")
        self.assertEqual(runner.session_service.created, [])
        self.assertEqual(runner.calls[0]["session_id"], "session-abc")

    def test_run_async_skips_partial_events(self):
        events = [
            FakeEvent(text="par", partial=True, final=False),
            FakeEvent(text="whole", final=True),
        ]
        wrapper, _ = self._wrapper_with_runner(events)

        result = asyncio.run(wrapper.run_async(SimpleNamespace(), "hello"))

        self.assertEqual(result["text"], "whole")

    def test_run_async_times_out(self):
        wrapper = build_wrapper(_timeout=0.05)
        self._patch_adk()

        class SlowRunner(FakeRunner):
            async def run_async(self, **kwargs):
                await asyncio.sleep(5)
                yield FakeEvent(text="too late")

        runner = SlowRunner()
        with patch.object(wrapper, "_get_runner", return_value=(runner, "app")):
            with self.assertRaises(TimeoutError):
                asyncio.run(wrapper.run_async(SimpleNamespace(), "hello"))

    def test_stream_async_yields_partial_chunks(self):
        events = [
            FakeEvent(text="Hel", partial=True, final=False),
            FakeEvent(text="lo!", partial=True, final=False),
            FakeEvent(text="Hello!", final=True),
        ]
        wrapper, _ = self._wrapper_with_runner(events)

        async def collect():
            return [chunk async for chunk in wrapper.stream_async(SimpleNamespace(), "hi")]

        chunks = asyncio.run(collect())

        # The aggregated final event must not duplicate the streamed chunks.
        self.assertEqual(chunks, ["Hel", "lo!"])

    def test_stream_async_falls_back_to_the_final_text(self):
        wrapper, _ = self._wrapper_with_runner([FakeEvent(text="one shot", final=True)])

        async def collect():
            return [chunk async for chunk in wrapper.stream_async(SimpleNamespace(), "hi")]

        self.assertEqual(asyncio.run(collect()), ["one shot"])

    def test_session_helpers(self):
        wrapper, runner = self._wrapper_with_runner([])
        agent = SimpleNamespace()

        created = asyncio.run(
            wrapper.create_session(agent, "user-1", state={"seed": 1})
        )
        self.assertEqual(created["user_id"], "user-1")
        self.assertEqual(created["state"], {"seed": 1})

        fetched = asyncio.run(
            wrapper.get_session(agent, "user-1", created["session_id"])
        )
        self.assertEqual(fetched["session_id"], created["session_id"])

        deleted = asyncio.run(
            wrapper.delete_session(agent, "user-1", created["session_id"])
        )
        self.assertTrue(deleted["deleted"])
        self.assertIsNone(
            asyncio.run(wrapper.get_session(agent, "user-1", created["session_id"]))
        )

    def test_session_helpers_validate_arguments(self):
        wrapper, _ = self._wrapper_with_runner([])
        with self.assertRaises(ValueError):
            asyncio.run(wrapper.create_session(SimpleNamespace(), ""))
        with self.assertRaises(ValueError):
            asyncio.run(wrapper.get_session(SimpleNamespace(), "u", ""))

    def test_get_runner_caches_per_agent(self):
        wrapper = build_wrapper()
        runner_cls = MagicMock(side_effect=lambda **kwargs: FakeRunner())
        with patch(
            f"{WRAPPER_MODULE}._check_adk_imports",
            return_value=(MagicMock(), runner_cls, FakeTypes),
        ):
            agent_a, agent_b = SimpleNamespace(), SimpleNamespace()
            first, _ = wrapper._get_runner(agent_a)
            second, _ = wrapper._get_runner(agent_a)
            third, _ = wrapper._get_runner(agent_b)

        self.assertIs(first, second, "the same agent must reuse one runner")
        self.assertIsNot(first, third, "distinct agents must get distinct runners")
        self.assertEqual(runner_cls.call_count, 2)

    def test_get_runner_requires_an_agent(self):
        wrapper = build_wrapper()
        with self.assertRaises(ValueError):
            wrapper._get_runner(None)

    # -- sync bridging ---------------------------------------------------

    def test_sync_run_returns_the_result(self):
        wrapper, _ = self._wrapper_with_runner([FakeEvent(text="sync ok")])

        self.assertEqual(wrapper.run(SimpleNamespace(), "hello")["text"], "sync ok")

    def test_sync_stream_returns_chunks(self):
        events = [
            FakeEvent(text="a", partial=True, final=False),
            FakeEvent(text="b", partial=True, final=False),
        ]
        wrapper, _ = self._wrapper_with_runner(events)

        self.assertEqual(list(wrapper.stream(SimpleNamespace(), "hello")), ["a", "b"])

    def test_sync_helpers_reject_a_running_loop(self):
        wrapper, _ = self._wrapper_with_runner([FakeEvent(text="x")])

        async def call_from_loop():
            with self.assertRaises(RuntimeError) as ctx:
                wrapper.run(SimpleNamespace(), "hello")
            self.assertIn("await the async variant", str(ctx.exception).lower())

            with self.assertRaises(RuntimeError):
                next(iter(wrapper.stream(SimpleNamespace(), "hello")))

        asyncio.run(call_from_loop())

    # -- RAG lifecycle ---------------------------------------------------

    def _fake_rag(self, **attrs):
        rag = MagicMock(name="rag")
        rag.create_corpus.return_value = SimpleNamespace(
            name="projects/p/locations/us-central1/ragCorpora/42",
            display_name="docs",
        )
        rag.list_corpora.return_value = [SimpleNamespace(name="c1"), SimpleNamespace(name="c2")]
        rag.get_corpus.return_value = SimpleNamespace(name="c1")
        rag.list_files.return_value = [SimpleNamespace(name="f1")]
        rag.import_files.return_value = SimpleNamespace(
            imported_rag_files_count=3, failed_rag_files_count=0
        )
        for key, value in attrs.items():
            setattr(rag, key, value)
        return rag

    def test_create_rag_corpus_returns_a_plain_dict(self):
        wrapper = build_wrapper()
        rag = self._fake_rag()
        with patch.object(wrapper, "_init_vertex", return_value=rag):
            result = wrapper.create_rag_corpus("docs", description="my docs")

        _, kwargs = rag.create_corpus.call_args
        self.assertEqual(kwargs["display_name"], "docs")
        self.assertEqual(kwargs["description"], "my docs")
        self.assertIsInstance(result, dict)
        self.assertTrue(result["name"].endswith("/42"))

    def test_create_rag_corpus_validates_display_name(self):
        wrapper = build_wrapper()
        with self.assertRaises(ValueError):
            wrapper.create_rag_corpus("")

    def test_create_rag_corpus_nests_embedding_config_in_backend_config(self):
        """Newer vertexai.rag takes backend_config rather than a bare embedding config."""
        wrapper = build_wrapper()

        def create_corpus(display_name=None, description=None, backend_config=None):
            return SimpleNamespace(name="c", backend_config=backend_config)

        rag = self._fake_rag(create_corpus=MagicMock(side_effect=create_corpus))
        rag.RagVectorDbConfig = lambda rag_embedding_model_config: SimpleNamespace(
            embedding=rag_embedding_model_config
        )
        rag.RagEmbeddingModelConfig = lambda vertex_prediction_endpoint: SimpleNamespace(
            endpoint=vertex_prediction_endpoint
        )
        rag.VertexPredictionEndpoint = lambda publisher_model: publisher_model

        with patch.object(wrapper, "_init_vertex", return_value=rag):
            wrapper.create_rag_corpus("docs", embedding_model="text-embedding-005")

        _, kwargs = rag.create_corpus.call_args
        self.assertEqual(
            kwargs["backend_config"].embedding.endpoint,
            "publishers/google/models/text-embedding-005",
        )

    def test_import_rag_files_uses_a_transformation_config(self):
        wrapper = build_wrapper()
        rag = self._fake_rag()
        rag.TransformationConfig = lambda chunking_config: SimpleNamespace(
            chunking=chunking_config
        )
        rag.ChunkingConfig = lambda chunk_size, chunk_overlap: SimpleNamespace(
            size=chunk_size, overlap=chunk_overlap
        )

        with patch.object(wrapper, "_init_vertex", return_value=rag):
            result = wrapper.import_rag_files("42", "gs://bucket/docs/*", chunk_size=256)

        _, kwargs = rag.import_files.call_args
        self.assertEqual(kwargs["paths"], ["gs://bucket/docs/*"])
        self.assertTrue(kwargs["corpus_name"].endswith("/ragCorpora/42"))
        self.assertEqual(kwargs["transformation_config"].chunking.size, 256)
        self.assertEqual(result["imported_rag_files_count"], 3)

    def test_import_rag_files_falls_back_to_chunk_arguments(self):
        """Older vertexai.rag has no TransformationConfig, so chunking is flat."""
        wrapper = build_wrapper()

        def import_files(corpus_name=None, paths=None, chunk_size=None,
                         chunk_overlap=None, max_embedding_requests_per_min=None):
            return SimpleNamespace(imported_rag_files_count=1, failed_rag_files_count=0)

        rag = self._fake_rag(import_files=MagicMock(side_effect=import_files))
        rag.TransformationConfig = None
        rag.ChunkingConfig = None

        with patch.object(wrapper, "_init_vertex", return_value=rag):
            wrapper.import_rag_files("42", ["gs://bucket/a.pdf"], chunk_size=128)

        _, kwargs = rag.import_files.call_args
        self.assertEqual(kwargs["chunk_size"], 128)
        self.assertNotIn("transformation_config", kwargs)

    def test_import_rag_files_requires_paths(self):
        wrapper = build_wrapper()
        with patch.object(wrapper, "_init_vertex", return_value=self._fake_rag()):
            with self.assertRaises(ValueError):
                wrapper.import_rag_files("42", [])

    def test_list_and_delete_rag_corpora(self):
        wrapper = build_wrapper()
        rag = self._fake_rag()
        with patch.object(wrapper, "_init_vertex", return_value=rag):
            self.assertEqual(len(wrapper.list_rag_corpora()), 2)
            self.assertEqual(len(wrapper.list_rag_files("42")), 1)
            deleted = wrapper.delete_rag_corpus("42")

        self.assertTrue(deleted["deleted"])
        self.assertTrue(deleted["name"].endswith("/ragCorpora/42"))

    def test_retrieval_query_flattens_contexts(self):
        wrapper = build_wrapper()
        contexts = SimpleNamespace(
            contexts=[
                SimpleNamespace(
                    text="passage one", source_uri="gs://b/a.pdf",
                    source_display_name="a.pdf", distance=0.12,
                )
            ]
        )
        rag = self._fake_rag()
        rag.retrieval_query.return_value = SimpleNamespace(contexts=contexts)
        rag.RagResource = lambda rag_corpus: SimpleNamespace(rag_corpus=rag_corpus)
        rag.RagRetrievalConfig = lambda top_k, filter=None: SimpleNamespace(
            top_k=top_k, filter=filter
        )
        rag.Filter = lambda vector_distance_threshold: vector_distance_threshold

        with patch.object(wrapper, "_init_vertex", return_value=rag):
            result = wrapper.retrieval_query("what is rag", rag_corpora="42")

        _, kwargs = rag.retrieval_query.call_args
        self.assertEqual(kwargs["text"], "what is rag")
        self.assertTrue(kwargs["rag_resources"][0].rag_corpus.endswith("/ragCorpora/42"))
        self.assertEqual(kwargs["rag_retrieval_config"].top_k, 5)
        self.assertEqual(result["contexts"][0]["text"], "passage one")
        self.assertEqual(result["contexts"][0]["distance"], 0.12)

    def test_retrieval_query_validates_arguments(self):
        wrapper = build_wrapper()
        with patch.object(wrapper, "_init_vertex", return_value=self._fake_rag()):
            with self.assertRaises(ValueError):
                wrapper.retrieval_query("")
            with self.assertRaises(ValueError):
                wrapper.retrieval_query("q")

    def test_retry_skips_value_errors_and_retries_transient_ones(self):
        wrapper = build_wrapper(_retry_attempts=3)
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("transient")
            return "ok"

        self.assertEqual(wrapper._with_retry("flaky", flaky), "ok")
        self.assertEqual(attempts["count"], 3)

        def invalid():
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            wrapper._with_retry("invalid", invalid)

    def test_to_plain_handles_nested_sdk_objects(self):
        payload = SimpleNamespace(
            name="corpus",
            nested=[SimpleNamespace(value=1)],
            mapping={"key": SimpleNamespace(value=2)},
        )
        self.assertEqual(
            _to_plain(payload),
            {"name": "corpus", "nested": [{"value": 1}], "mapping": {"key": {"value": 2}}},
        )


class TestGoogleGCPWrapperLive(unittest.TestCase):
    """
    Live tests against Google Cloud. Skipped unless google-adk is installed and
    credentials plus a project are available.

    Setup:
        pip install intelli[gcp]
        gcloud auth application-default login
        export GOOGLE_CLOUD_PROJECT=your-project

    Optional:
        GCP_AGENT_MODEL, GCP_TEST_RAG_CORPUS, GCP_TEST_DATA_STORE_ID
    """

    def setUp(self):
        try:
            _check_adk_imports()
        except ImportError as exc:
            self.skipTest(str(exc))

        if not os.getenv("GOOGLE_CLOUD_PROJECT") and not os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ):
            self.skipTest(
                "set GOOGLE_CLOUD_PROJECT (with ADC) or GOOGLE_APPLICATION_CREDENTIALS"
            )

        try:
            self.wrapper = GoogleGCPWrapper(
                default_model=os.getenv("GCP_AGENT_MODEL"),
                timeout=120.0,
            )
        except (ValueError, ImportError) as exc:
            self.skipTest(f"GCP credentials unavailable: {exc}")

        self.created_corpora = []

    def tearDown(self):
        for corpus_name in getattr(self, "created_corpora", []):
            try:
                self.wrapper.delete_rag_corpus(corpus_name)
            except Exception as exc:
                print(f"failed to clean up corpus {corpus_name}: {exc}")

    def _agent(self, name="intelli_test_agent", instructions=None, tools=None):
        return self.wrapper.create_agent(
            name=name,
            instructions=instructions or "You are concise. Answer in one short sentence.",
            tools=tools,
        )

    def test_create_agent(self):
        agent = self._agent()
        self.assertEqual(agent.name, "intelli_test_agent")
        print(f"created agent with model {getattr(agent, 'model', None)}")

    def test_run_single_turn(self):
        result = self.wrapper.run(self._agent(), "Reply with the single word: ready")

        print(f"agent replied: {result['text']!r}")
        self.assertTrue(result["text"].strip(), "expected non-empty response text")
        self.assertTrue(result["session_id"])

    def test_multi_turn_session_keeps_context(self):
        agent = self._agent(
            instructions="Remember what the user tells you and answer briefly."
        )
        first = self.wrapper.run(agent, "My favorite color is teal. Acknowledge briefly.")
        self.assertTrue(first["text"].strip())

        second = self.wrapper.run(
            agent, "What is my favorite color?", session_id=first["session_id"]
        )

        print(f"recall answer: {second['text']!r}")
        self.assertEqual(second["session_id"], first["session_id"])
        self.assertIn("teal", second["text"].lower())

    def test_stream(self):
        chunks = list(
            self.wrapper.stream(self._agent(), "Count from one to five in words.")
        )

        print(f"received {len(chunks)} chunk(s)")
        self.assertGreater(len(chunks), 0, "expected at least one streamed chunk")
        self.assertTrue("".join(chunks).strip())

    def test_function_tool_round_trip(self):
        calls = []

        def get_stock_level(sku: str) -> dict:
            """Return the warehouse stock level for a SKU.

            Args:
                sku: The stock keeping unit to look up.
            """
            calls.append(sku)
            return {"sku": sku, "units": 17}

        agent = self._agent(
            name="intelli_tool_agent",
            instructions=(
                "You look up stock levels. Always call the get_stock_level tool "
                "before answering, then state the number of units."
            ),
            tools=[get_stock_level],
        )

        result = self.wrapper.run(agent, "How many units of SKU ABC-123 are in stock?")

        print(f"tool calls: {result['function_calls']}, answer: {result['text']!r}")
        self.assertTrue(calls, "expected the model to invoke the function tool")
        self.assertIn("17", result["text"])

    def test_rag_corpus_lifecycle(self):
        try:
            _check_rag_imports()
        except ImportError as exc:
            self.skipTest(str(exc))

        corpus = self.wrapper.create_rag_corpus(
            display_name="intelli-integration-test", description="temporary test corpus"
        )
        self.assertIn("name", corpus)
        self.created_corpora.append(corpus["name"])
        print(f"created corpus {corpus['name']}")

        names = [item.get("name") for item in self.wrapper.list_rag_corpora()]
        self.assertIn(corpus["name"], names)

        fetched = self.wrapper.get_rag_corpus(corpus["name"])
        self.assertEqual(fetched["name"], corpus["name"])

        self.wrapper.delete_rag_corpus(corpus["name"])
        self.created_corpora.remove(corpus["name"])

    def test_rag_retrieval_query(self):
        corpus = os.getenv("GCP_TEST_RAG_CORPUS")
        if not corpus:
            self.skipTest("set GCP_TEST_RAG_CORPUS to a populated RAG corpus")
        try:
            _check_rag_imports()
        except ImportError as exc:
            self.skipTest(str(exc))

        result = self.wrapper.retrieval_query(
            "summarize the main topic", rag_corpora=corpus, similarity_top_k=3
        )

        print(f"retrieved {len(result['contexts'])} context(s)")
        self.assertIsInstance(result["contexts"], list)

    def test_rag_grounded_agent(self):
        corpus = os.getenv("GCP_TEST_RAG_CORPUS")
        if not corpus:
            self.skipTest("set GCP_TEST_RAG_CORPUS to a populated RAG corpus")

        tool = self.wrapper.create_rag_retrieval_tool(
            rag_corpora=corpus,
            description="Retrieve passages from the project knowledge base.",
        )
        agent = self._agent(
            name="intelli_rag_agent",
            instructions=(
                "Answer only from the retrieved passages. Use the retrieval tool first."
            ),
            tools=[tool],
        )

        result = self.wrapper.run(agent, "What is this knowledge base about?")

        print(f"grounded answer: {result['text']!r}")
        self.assertTrue(result["text"].strip())

    def test_vertex_search_grounded_agent(self):
        data_store = os.getenv("GCP_TEST_DATA_STORE_ID")
        if not data_store:
            self.skipTest("set GCP_TEST_DATA_STORE_ID to a Vertex AI Search data store")

        tool = self.wrapper.create_vertex_search_tool(data_store_id=data_store)
        agent = self._agent(
            name="intelli_search_agent",
            instructions="Answer using the search tool results only.",
            tools=[tool],
        )

        result = self.wrapper.run(agent, "What information is available here?")

        print(f"search grounded answer: {result['text']!r}")
        self.assertTrue(result["text"].strip())


if __name__ == "__main__":
    unittest.main()
