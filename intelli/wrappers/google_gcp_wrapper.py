import asyncio
import importlib
import inspect
import logging
import os
import random
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

from intelli.config import config

logger = logging.getLogger(__name__)

_INSTALL_HINT = "pip install intelli[gcp]"

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# ADK relocates tool classes between releases, so every symbol is resolved
# through _resolve_symbol against an ordered list of known locations.
_LLM_AGENT_CANDIDATES = (
    ("google.adk.agents", "LlmAgent"),
    ("google.adk", "Agent"),
)
_APP_CANDIDATES = (
    ("google.adk.apps", "App"),
    ("google.adk", "App"),
)
_FUNCTION_TOOL_CANDIDATES = (
    ("google.adk.tools.function_tool", "FunctionTool"),
    ("google.adk.tools", "FunctionTool"),
)
_VERTEX_SEARCH_TOOL_CANDIDATES = (
    ("google.adk.tools.vertex_ai_search_tool", "VertexAiSearchTool"),
    ("google.adk.tools", "VertexAiSearchTool"),
)
_RAG_RETRIEVAL_CANDIDATES = (
    ("google.adk.tools.retrieval.vertex_ai_rag_retrieval", "VertexAiRagRetrieval"),
    ("google.adk.tools.retrieval", "VertexAiRagRetrieval"),
)
_RUN_CONFIG_CANDIDATES = (
    ("google.adk.agents.run_config", "RunConfig"),
    ("google.adk.agents", "RunConfig"),
)
_STREAMING_MODE_CANDIDATES = (
    ("google.adk.agents.run_config", "StreamingMode"),
    ("google.adk.agents._streaming_mode", "StreamingMode"),
    ("google.adk.agents", "StreamingMode"),
)


def _check_adk_imports():
    """Return (LlmAgent, InMemoryRunner, genai types) or raise with an install hint."""
    try:
        from google.adk.runners import InMemoryRunner
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-adk is required for GCP agent support. "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc
    return _resolve_symbol(_LLM_AGENT_CANDIDATES, "LlmAgent"), InMemoryRunner, types


def _check_rag_imports():
    """Return (vertexai, vertexai.rag) or raise with an install hint."""
    try:
        import vertexai
        from vertexai import rag
    except ImportError as exc:
        raise ImportError(
            "google-cloud-aiplatform (1.x) is required for Vertex RAG corpus support. "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc
    return vertexai, rag


def _check_auth_imports():
    """Return (google.auth, service_account module) or raise with an install hint."""
    try:
        import google.auth
        from google.oauth2 import service_account
    except ImportError as exc:
        raise ImportError(
            "google-auth is required to resolve Google Cloud credentials. "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc
    return google.auth, service_account


def _resolve_symbol(candidates, label: str):
    """Import the first available (module, attribute) pair from candidates."""
    attempts = []
    for module_path, attr in candidates:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            attempts.append(f"{module_path} ({exc})")
            continue
        symbol = getattr(module, attr, None)
        if symbol is not None:
            return symbol
        attempts.append(f"{module_path} (no attribute {attr})")
    raise ImportError(
        f"unable to locate {label} in the installed google-adk. Tried: "
        + "; ".join(attempts)
        + f". Install or upgrade with: {_INSTALL_HINT}"
    )


def _to_plain(value: Any, _depth: int = 0) -> Any:
    """Convert SDK/proto objects into plain dicts, lists and scalars."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if _depth > 12:
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_plain(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item, _depth + 1) for item in value]
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _to_plain(method(), _depth + 1)
            except Exception:
                continue
    if hasattr(value, "__dict__"):
        return {
            key: _to_plain(item, _depth + 1)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


class GoogleGCPWrapper:
    """
    Wrapper for Google ADK agents backed by Vertex AI, plus Vertex AI Search and
    RAG Engine grounding.

    Authenticates with Application Default Credentials by default, and accepts a
    service account key file or a prebuilt credentials object instead.
    """

    DEFAULT_LOCATION = "us-central1"
    DEFAULT_USER_ID = "intelli-user"

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials_path: Optional[str] = None,
        credentials: Optional[Any] = None,
        default_model: Optional[str] = None,
        app_name: str = "intelli",
        timeout: Optional[float] = None,
        retry_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        """
        Initialize the GCP wrapper.

        Args:
            project_id: Google Cloud project. Defaults to GOOGLE_CLOUD_PROJECT, then
                        the project reported by Application Default Credentials.
            location: Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION, then
                      the configured default, then us-central1.
            credentials_path: Service account JSON key file. Defaults to
                              GOOGLE_APPLICATION_CREDENTIALS.
            credentials: Prebuilt google.auth credentials, used as-is when provided.
            default_model: Model applied when create_agent receives no model.
            app_name: ADK application name used for sessions.
            timeout: Seconds to allow per agent turn and per RAG call.
            retry_attempts: Attempts for retryable RAG calls.
            retry_delay_seconds: Base backoff between retries.
        """
        gcp_config = config.get("url", {}).get("gcp", {})

        self.location = (
            location
            or os.getenv("GOOGLE_CLOUD_LOCATION")
            or gcp_config.get("default_location")
            or self.DEFAULT_LOCATION
        )
        self.default_model = default_model or gcp_config.get("models", {}).get("text")
        self.app_name = app_name
        self._timeout = timeout
        self._retry_attempts = max(1, retry_attempts)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)

        self.credentials, discovered_project = self._resolve_credentials(
            credentials, credentials_path
        )
        self.project_id = (
            project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or discovered_project
        )
        if not self.project_id:
            raise ValueError(
                "unable to determine the Google Cloud project. Pass project_id, set "
                "GOOGLE_CLOUD_PROJECT, or use credentials that carry a project."
            )

        self._runners: Dict[Any, Any] = {}
        self._vertex_initialized = False
        self._apply_backend_env()

        logger.debug(
            "GoogleGCPWrapper initialized for project %s in %s",
            self.project_id,
            self.location,
        )

    # ------------------------------------------------------------------
    # Auth and environment
    # ------------------------------------------------------------------

    def _resolve_credentials(self, credentials: Optional[Any], credentials_path: Optional[str]):
        """Resolve credentials from an explicit object, a key file, then ADC."""
        if credentials is not None:
            return credentials, getattr(credentials, "project_id", None)

        auth_module, service_account = _check_auth_imports()

        key_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path:
            if not os.path.isfile(key_path):
                raise ValueError(f"service account key file not found: {key_path}")
            resolved = service_account.Credentials.from_service_account_file(
                key_path, scopes=[_CLOUD_PLATFORM_SCOPE]
            )
            return resolved, getattr(resolved, "project_id", None)

        try:
            return auth_module.default(scopes=[_CLOUD_PLATFORM_SCOPE])
        except Exception as exc:
            raise ValueError(
                "unable to resolve Google Cloud credentials. Run "
                "'gcloud auth application-default login', or set "
                "GOOGLE_APPLICATION_CREDENTIALS to a service account key file, or pass "
                f"credentials/credentials_path explicitly. Underlying error: {exc}"
            ) from exc

    def _apply_backend_env(self) -> None:
        """
        Publish the resolved project and region to the environment.

        ADK reads these variables when it builds the model client, so they must be
        present in os.environ rather than only on this instance. ADK 2.x renamed the
        Vertex switch from GOOGLE_GENAI_USE_VERTEXAI to GOOGLE_GENAI_USE_ENTERPRISE,
        so both are set and either version routes to Vertex.
        """
        os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = self.location
        for flag in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"):
            os.environ.setdefault(flag, "1")

    def _init_vertex(self):
        """Initialize vertexai once, on the first RAG lifecycle call."""
        vertexai, rag = _check_rag_imports()
        if not self._vertex_initialized:
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=self.credentials,
            )
            self._vertex_initialized = True
        return rag

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _with_retry(self, label: str, func: Callable[[], Any]) -> Any:
        """Call func, retrying transient failures with jittered exponential backoff."""
        last_error = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return func()
            except (ValueError, TypeError):
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self._retry_attempts:
                    backoff = self._retry_delay_seconds * (2 ** (attempt - 1))
                    time.sleep(backoff + random.uniform(0, backoff * 0.2))
        logger.exception("%s failed after %s attempts", label, self._retry_attempts)
        raise last_error

    @staticmethod
    def _supported_kwargs(target: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop kwargs the target does not accept, tolerating SDK drift."""
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return kwargs
        params = signature.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in params}

    def _call_filtered(self, label: str, target: Callable, **kwargs) -> Any:
        """Invoke target with only the kwargs it accepts, warning about the rest."""
        supported = self._supported_kwargs(target, kwargs)
        dropped = sorted(set(kwargs) - set(supported))
        if dropped:
            logger.warning(
                "%s: installed SDK does not accept %s, ignoring", label, ", ".join(dropped)
            )
        return target(**supported)

    def _with_timeout(self, target: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Add the configured timeout when the target accepts one."""
        if self._timeout is None or "timeout" in kwargs:
            return kwargs
        if "timeout" in self._supported_kwargs(target, {"timeout": self._timeout}):
            return {**kwargs, "timeout": self._timeout}
        return kwargs

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def qualify_corpus_name(self, corpus: str) -> str:
        """Expand a bare corpus id into a full Vertex RAG resource name."""
        if not corpus or not isinstance(corpus, str):
            raise ValueError("corpus must be a non-empty string")
        if corpus.startswith("projects/"):
            return corpus
        return (
            f"projects/{self.project_id}/locations/{self.location}/ragCorpora/{corpus}"
        )

    def qualify_data_store_name(self, data_store_id: str) -> str:
        """Expand a bare data store id into a full Discovery Engine resource name."""
        if not data_store_id or not isinstance(data_store_id, str):
            raise ValueError("data_store_id must be a non-empty string")
        if data_store_id.startswith("projects/"):
            return data_store_id
        return (
            f"projects/{self.project_id}/locations/{self.location}/collections/"
            f"default_collection/dataStores/{data_store_id}"
        )

    # ------------------------------------------------------------------
    # Agent and tool construction
    # ------------------------------------------------------------------

    def create_agent(
        self,
        name: str,
        instructions: str,
        model: Optional[str] = None,
        description: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        sub_agents: Optional[List[Any]] = None,
        output_key: Optional[str] = None,
        generate_content_config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Build an ADK LlmAgent.

        Args:
            name: Agent name. ADK requires a valid Python identifier.
            instructions: System instruction for the agent.
            model: Vertex model id. Falls back to the wrapper default_model.
            tools: ADK tool objects or plain Python callables, which are wrapped
                   as FunctionTool automatically.
            sub_agents: Child agents for delegation.
            output_key: Session state key to store the agent's final text under.
            generate_content_config: genai GenerateContentConfig for sampling.
            **kwargs: Passed through to LlmAgent for less common fields.

        Returns:
            An ADK LlmAgent instance.
        """
        if not name:
            raise ValueError("name is required")
        if not instructions:
            raise ValueError("instructions is required")

        resolved_model = model or self.default_model
        if not resolved_model:
            raise ValueError(
                "model is required. Pass model=, set default_model on the wrapper, or "
                "configure url.gcp.models.text."
            )

        LlmAgent, _, _ = _check_adk_imports()

        agent_kwargs: Dict[str, Any] = {
            "name": name,
            "model": resolved_model,
            "instruction": instructions,
        }
        if description is not None:
            agent_kwargs["description"] = description
        if tools:
            agent_kwargs["tools"] = [self._coerce_tool(tool) for tool in self._as_list(tools)]
        if sub_agents:
            agent_kwargs["sub_agents"] = self._as_list(sub_agents)
        if output_key is not None:
            agent_kwargs["output_key"] = output_key
        if generate_content_config is not None:
            agent_kwargs["generate_content_config"] = generate_content_config
        agent_kwargs.update(kwargs)

        try:
            return LlmAgent(**agent_kwargs)
        except Exception:
            logger.exception("failed to create ADK agent %s", name)
            raise

    def _coerce_tool(self, tool: Any) -> Any:
        """Wrap bare callables as FunctionTool, pass ADK tools through."""
        if inspect.isfunction(tool) or inspect.ismethod(tool):
            return self.create_function_tool(tool)
        return tool

    def create_function_tool(self, func: Callable, require_confirmation: bool = False) -> Any:
        """
        Wrap a Python callable as an ADK FunctionTool.

        The function signature and docstring become the tool schema, so both should
        be descriptive.
        """
        if not callable(func):
            raise ValueError("func must be callable")

        FunctionTool = _resolve_symbol(_FUNCTION_TOOL_CANDIDATES, "FunctionTool")
        tool_kwargs: Dict[str, Any] = {"func": func}
        if require_confirmation:
            tool_kwargs["require_confirmation"] = True
        try:
            return self._call_filtered("create_function_tool", FunctionTool, **tool_kwargs)
        except Exception:
            logger.exception("failed to wrap %s as a FunctionTool", getattr(func, "__name__", func))
            raise

    def create_vertex_search_tool(
        self,
        data_store_id: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Any:
        """
        Build an ADK VertexAiSearchTool over a Vertex AI Search data store or engine.

        Exactly one of data_store_id or search_engine_id must be provided. ADK does
        not let this tool be renamed, so name and description are ignored (with a
        warning) unless a future ADK release accepts them.
        """
        if bool(data_store_id) == bool(search_engine_id):
            raise ValueError(
                "provide exactly one of data_store_id or search_engine_id"
            )

        VertexAiSearchTool = _resolve_symbol(
            _VERTEX_SEARCH_TOOL_CANDIDATES, "VertexAiSearchTool"
        )
        tool_kwargs: Dict[str, Any] = {}
        if data_store_id:
            tool_kwargs["data_store_id"] = self.qualify_data_store_name(data_store_id)
        else:
            tool_kwargs["search_engine_id"] = search_engine_id
        if name is not None:
            tool_kwargs["name"] = name
        if description is not None:
            tool_kwargs["description"] = description

        try:
            return self._call_filtered(
                "create_vertex_search_tool", VertexAiSearchTool, **tool_kwargs
            )
        except Exception:
            logger.exception("failed to create VertexAiSearchTool")
            raise

    def create_rag_retrieval_tool(
        self,
        rag_corpora: Optional[Any] = None,
        rag_resources: Optional[List[Any]] = None,
        similarity_top_k: int = 5,
        vector_distance_threshold: float = 0.5,
        name: str = "retrieve_docs",
        description: Optional[str] = None,
    ) -> Any:
        """
        Build an ADK VertexAiRagRetrieval tool over one or more RAG corpora.

        Args:
            rag_corpora: Corpus id or full resource name, or a list of them.
            rag_resources: Prebuilt rag.RagResource objects, used instead of rag_corpora.
            similarity_top_k: Number of contexts to retrieve.
            vector_distance_threshold: Maximum embedding distance to accept.
        """
        if not rag_corpora and not rag_resources:
            raise ValueError("provide rag_corpora or rag_resources")

        VertexAiRagRetrieval = _resolve_symbol(
            _RAG_RETRIEVAL_CANDIDATES, "VertexAiRagRetrieval"
        )
        tool_kwargs: Dict[str, Any] = {
            "name": name,
            "description": description
            or "Retrieve grounding passages from the Vertex AI RAG corpus.",
            "similarity_top_k": similarity_top_k,
            "vector_distance_threshold": vector_distance_threshold,
        }
        if rag_resources:
            tool_kwargs["rag_resources"] = self._as_list(rag_resources)
        else:
            tool_kwargs["rag_corpora"] = [
                self.qualify_corpus_name(corpus) for corpus in self._as_list(rag_corpora)
            ]

        try:
            return self._call_filtered(
                "create_rag_retrieval_tool", VertexAiRagRetrieval, **tool_kwargs
            )
        except Exception:
            logger.exception("failed to create VertexAiRagRetrieval tool")
            raise

    # ------------------------------------------------------------------
    # Vertex RAG corpus lifecycle
    # ------------------------------------------------------------------

    def create_rag_corpus(
        self,
        display_name: str,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a Vertex AI RAG corpus.

        Args:
            display_name: Human readable corpus name.
            description: Optional corpus description.
            embedding_model: Publisher model id such as "text-embedding-005", or a
                             full endpoint path.

        Returns:
            The corpus as a plain dict, including its "name" resource path.
        """
        if not display_name:
            raise ValueError("display_name is required")

        rag = self._init_vertex()
        corpus_kwargs = self._build_corpus_kwargs(rag, display_name, description, embedding_model)
        corpus = self._with_retry(
            "create_rag_corpus", lambda: rag.create_corpus(**corpus_kwargs)
        )
        logger.info("created RAG corpus %s", getattr(corpus, "name", display_name))
        return _to_plain(corpus)

    def _build_corpus_kwargs(
        self,
        rag: Any,
        display_name: str,
        description: Optional[str],
        embedding_model: Optional[str],
    ) -> Dict[str, Any]:
        """Assemble create_corpus kwargs, adapting to the installed rag module."""
        corpus_kwargs: Dict[str, Any] = {"display_name": display_name}
        if description:
            corpus_kwargs["description"] = description
        if not embedding_model:
            return corpus_kwargs

        embedding_config = self._build_embedding_config(rag, embedding_model)
        accepted = self._supported_kwargs(
            rag.create_corpus,
            {"backend_config": None, "rag_embedding_model_config": None},
        )
        vector_db_cls = getattr(rag, "RagVectorDbConfig", None)
        if "backend_config" in accepted and vector_db_cls is not None:
            corpus_kwargs["backend_config"] = vector_db_cls(
                rag_embedding_model_config=embedding_config
            )
        elif "rag_embedding_model_config" in accepted:
            corpus_kwargs["rag_embedding_model_config"] = embedding_config
        else:
            raise ValueError(
                "the installed vertexai.rag does not support a custom embedding model; "
                "omit embedding_model to use the default"
            )
        return corpus_kwargs

    @staticmethod
    def _build_embedding_config(rag: Any, embedding_model: str) -> Any:
        """Build a RagEmbeddingModelConfig for a publisher model or endpoint path."""
        config_cls = getattr(rag, "RagEmbeddingModelConfig", None)
        if config_cls is None:
            raise ValueError(
                "the installed vertexai.rag does not expose RagEmbeddingModelConfig"
            )
        endpoint = (
            embedding_model
            if "/" in embedding_model
            else f"publishers/google/models/{embedding_model}"
        )
        endpoint_cls = getattr(rag, "VertexPredictionEndpoint", None)
        if endpoint_cls is not None:
            return config_cls(
                vertex_prediction_endpoint=endpoint_cls(publisher_model=endpoint)
            )
        return config_cls(endpoint=endpoint)

    def list_rag_corpora(self) -> List[Dict[str, Any]]:
        """List the RAG corpora in the configured project and region."""
        rag = self._init_vertex()
        corpora = self._with_retry("list_rag_corpora", rag.list_corpora)
        return [_to_plain(corpus) for corpus in corpora]

    def get_rag_corpus(self, corpus_name: str) -> Dict[str, Any]:
        """Fetch a single RAG corpus by id or full resource name."""
        name = self.qualify_corpus_name(corpus_name)
        rag = self._init_vertex()
        corpus = self._with_retry("get_rag_corpus", lambda: rag.get_corpus(name=name))
        return _to_plain(corpus)

    def delete_rag_corpus(self, corpus_name: str) -> Dict[str, Any]:
        """Delete a RAG corpus by id or full resource name."""
        name = self.qualify_corpus_name(corpus_name)
        rag = self._init_vertex()
        self._with_retry("delete_rag_corpus", lambda: rag.delete_corpus(name=name))
        logger.info("deleted RAG corpus %s", name)
        return {"name": name, "deleted": True}

    def import_rag_files(
        self,
        corpus_name: str,
        paths: Any,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        max_embedding_requests_per_min: int = 900,
    ) -> Dict[str, Any]:
        """
        Import files into a RAG corpus.

        Args:
            corpus_name: Corpus id or full resource name.
            paths: One or more Cloud Storage URIs ("gs://bucket/dir/*") or Google
                   Drive file/folder links.
            chunk_size: Tokens per chunk.
            chunk_overlap: Overlap between chunks.
            max_embedding_requests_per_min: Embedding rate limit for the import.

        Returns:
            Dict with imported and failed file counts.
        """
        name = self.qualify_corpus_name(corpus_name)
        path_list = [str(path) for path in self._as_list(paths) if path]
        if not path_list:
            raise ValueError("paths is required")

        rag = self._init_vertex()
        import_kwargs: Dict[str, Any] = {
            "corpus_name": name,
            "paths": path_list,
            "max_embedding_requests_per_min": max_embedding_requests_per_min,
        }
        transformation = self._build_transformation_config(rag, chunk_size, chunk_overlap)
        if transformation is not None:
            import_kwargs["transformation_config"] = transformation
        else:
            import_kwargs["chunk_size"] = chunk_size
            import_kwargs["chunk_overlap"] = chunk_overlap
        import_kwargs = self._with_timeout(rag.import_files, import_kwargs)

        response = self._with_retry(
            "import_rag_files",
            lambda: self._call_filtered("import_rag_files", rag.import_files, **import_kwargs),
        )
        result = {
            "corpus_name": name,
            "imported_rag_files_count": getattr(response, "imported_rag_files_count", None),
            "failed_rag_files_count": getattr(response, "failed_rag_files_count", None),
            "raw": _to_plain(response),
        }
        logger.info(
            "imported %s files into %s", result["imported_rag_files_count"], name
        )
        return result

    @staticmethod
    def _build_transformation_config(rag: Any, chunk_size: int, chunk_overlap: int) -> Optional[Any]:
        """Build a TransformationConfig when the installed rag module has one."""
        transformation_cls = getattr(rag, "TransformationConfig", None)
        chunking_cls = getattr(rag, "ChunkingConfig", None)
        if transformation_cls is None or chunking_cls is None:
            return None
        return transformation_cls(
            chunking_config=chunking_cls(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        )

    def list_rag_files(self, corpus_name: str) -> List[Dict[str, Any]]:
        """List the files indexed in a RAG corpus."""
        name = self.qualify_corpus_name(corpus_name)
        rag = self._init_vertex()
        files = self._with_retry(
            "list_rag_files", lambda: rag.list_files(corpus_name=name)
        )
        return [_to_plain(item) for item in files]

    def retrieval_query(
        self,
        query: str,
        rag_corpora: Optional[Any] = None,
        rag_resources: Optional[List[Any]] = None,
        similarity_top_k: int = 5,
        vector_distance_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Retrieve grounding contexts directly, without invoking a model.

        Returns:
            Dict with a "contexts" list of {text, source_uri, distance} entries and
            the raw response under "raw".
        """
        if not query:
            raise ValueError("query is required")
        if not rag_corpora and not rag_resources:
            raise ValueError("provide rag_corpora or rag_resources")

        rag = self._init_vertex()
        resources = self._as_list(rag_resources)
        if not resources:
            resource_cls = getattr(rag, "RagResource", None)
            if resource_cls is None:
                raise ValueError(
                    "the installed vertexai.rag does not expose RagResource"
                )
            resources = [
                resource_cls(rag_corpus=self.qualify_corpus_name(corpus))
                for corpus in self._as_list(rag_corpora)
            ]

        query_kwargs: Dict[str, Any] = {"text": query, "rag_resources": resources}
        retrieval_config = self._build_retrieval_config(
            rag, similarity_top_k, vector_distance_threshold
        )
        if retrieval_config is not None:
            query_kwargs["rag_retrieval_config"] = retrieval_config
        else:
            query_kwargs["similarity_top_k"] = similarity_top_k
            query_kwargs["vector_distance_threshold"] = vector_distance_threshold

        response = self._with_retry(
            "retrieval_query",
            lambda: self._call_filtered(
                "retrieval_query", rag.retrieval_query, **query_kwargs
            ),
        )
        return {"contexts": self._extract_contexts(response), "raw": _to_plain(response)}

    @staticmethod
    def _build_retrieval_config(
        rag: Any, similarity_top_k: int, vector_distance_threshold: float
    ) -> Optional[Any]:
        """Build a RagRetrievalConfig when the installed rag module has one."""
        config_cls = getattr(rag, "RagRetrievalConfig", None)
        if config_cls is None:
            return None
        filter_cls = getattr(config_cls, "Filter", None) or getattr(rag, "Filter", None)
        if filter_cls is not None:
            return config_cls(
                top_k=similarity_top_k,
                filter=filter_cls(vector_distance_threshold=vector_distance_threshold),
            )
        return config_cls(top_k=similarity_top_k)

    @staticmethod
    def _extract_contexts(response: Any) -> List[Dict[str, Any]]:
        """Flatten a retrieval response into simple context dicts."""
        container = getattr(response, "contexts", None)
        entries = getattr(container, "contexts", None) if container is not None else None
        if entries is None:
            entries = container if isinstance(container, (list, tuple)) else []
        contexts = []
        for entry in entries or []:
            contexts.append(
                {
                    "text": getattr(entry, "text", None),
                    "source_uri": getattr(entry, "source_uri", None),
                    "source_display_name": getattr(entry, "source_display_name", None),
                    "distance": getattr(entry, "distance", None),
                }
            )
        return contexts

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _get_runner(self, agent: Any, app_name: Optional[str] = None):
        """
        Return a cached (runner, app_name) pair for an agent.

        The cache keeps one InMemoryRunner, and therefore one session service, per
        agent so that session state survives across turns. Keying on id(agent) is
        safe because the cached runner holds a reference to the agent.
        """
        if agent is None:
            raise ValueError("agent is required")
        resolved_app = app_name or self.app_name
        key = (resolved_app, id(agent))
        runner = self._runners.get(key)
        if runner is None:
            _, InMemoryRunner, _ = _check_adk_imports()
            runner = self._build_runner(InMemoryRunner, agent, resolved_app)
            self._runners[key] = runner
        return runner, resolved_app

    @staticmethod
    def _build_runner(InMemoryRunner: Any, agent: Any, app_name: str) -> Any:
        """Construct an InMemoryRunner, falling back to the App wrapper form."""
        try:
            return InMemoryRunner(agent=agent, app_name=app_name)
        except TypeError:
            App = _resolve_symbol(_APP_CANDIDATES, "App")
            return InMemoryRunner(app=App(name=app_name, root_agent=agent))

    def _to_content(self, message: Any) -> Any:
        """Normalize a string or {role, text} dict into a genai Content."""
        _, _, types = _check_adk_imports()
        if isinstance(message, types.Content):
            return message
        if isinstance(message, str):
            if not message.strip():
                raise ValueError("message cannot be empty")
            return types.Content(role="user", parts=[types.Part.from_text(text=message)])
        if isinstance(message, dict):
            text = message.get("text") or message.get("content")
            if not text:
                raise ValueError("message dict must include 'text' or 'content'")
            return types.Content(
                role=message.get("role", "user"),
                parts=[types.Part.from_text(text=text)],
            )
        raise ValueError(
            "message must be a string, a {'role', 'text'} dict, or a genai types.Content"
        )

    @staticmethod
    def _event_text(event: Any) -> str:
        """Concatenate the text parts of an event."""
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            return ""
        return "".join(part.text for part in parts if getattr(part, "text", None))

    @staticmethod
    def _event_function_calls(event: Any) -> List[Dict[str, Any]]:
        """Extract function calls from an event, across ADK versions."""
        getter = getattr(event, "get_function_calls", None)
        calls = []
        if callable(getter):
            try:
                calls = getter() or []
            except Exception:
                calls = []
        if not calls:
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            calls = [
                part.function_call
                for part in (parts or [])
                if getattr(part, "function_call", None)
            ]
        return [
            {
                "name": getattr(call, "name", None),
                "args": _to_plain(getattr(call, "args", None)) or {},
            }
            for call in calls
        ]

    @staticmethod
    def _is_final_response(event: Any) -> bool:
        """Detect a final response event, falling back to the partial flag."""
        checker = getattr(event, "is_final_response", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                pass
        return not getattr(event, "partial", False)

    def _default_stream_config(self) -> Optional[Any]:
        """Build a RunConfig that enables partial (SSE) events, when available."""
        try:
            RunConfig = _resolve_symbol(_RUN_CONFIG_CANDIDATES, "RunConfig")
            StreamingMode = _resolve_symbol(_STREAMING_MODE_CANDIDATES, "StreamingMode")
        except ImportError:
            logger.warning("streaming RunConfig unavailable, falling back to one chunk")
            return None
        return RunConfig(streaming_mode=StreamingMode.SSE)

    async def create_session(
        self,
        agent: Any,
        user_id: str,
        session_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        app_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an ADK session for an agent and return its identifiers."""
        if not user_id:
            raise ValueError("user_id is required")
        runner, resolved_app = self._get_runner(agent, app_name)
        session = await runner.session_service.create_session(
            app_name=resolved_app,
            user_id=user_id,
            session_id=session_id,
            state=state,
        )
        return {
            "session_id": getattr(session, "id", session_id),
            "user_id": user_id,
            "app_name": resolved_app,
            "state": _to_plain(getattr(session, "state", None)),
        }

    async def get_session(
        self,
        agent: Any,
        user_id: str,
        session_id: str,
        app_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a session, returning None when it does not exist."""
        if not user_id or not session_id:
            raise ValueError("user_id and session_id are required")
        runner, resolved_app = self._get_runner(agent, app_name)
        session = await runner.session_service.get_session(
            app_name=resolved_app, user_id=user_id, session_id=session_id
        )
        if session is None:
            return None
        return {
            "session_id": getattr(session, "id", session_id),
            "user_id": user_id,
            "app_name": resolved_app,
            "state": _to_plain(getattr(session, "state", None)),
        }

    async def delete_session(
        self,
        agent: Any,
        user_id: str,
        session_id: str,
        app_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a session."""
        if not user_id or not session_id:
            raise ValueError("user_id and session_id are required")
        runner, resolved_app = self._get_runner(agent, app_name)
        await runner.session_service.delete_session(
            app_name=resolved_app, user_id=user_id, session_id=session_id
        )
        return {"session_id": session_id, "deleted": True}

    async def _ensure_session(self, runner: Any, resolved_app: str, user_id: str,
                              session_id: Optional[str]) -> str:
        """Return the given session id, creating a session when none was supplied."""
        if session_id:
            return session_id
        session = await runner.session_service.create_session(
            app_name=resolved_app, user_id=user_id
        )
        return getattr(session, "id", None)

    async def run_async(
        self,
        agent: Any,
        message: Any,
        user_id: str = DEFAULT_USER_ID,
        session_id: Optional[str] = None,
        app_name: Optional[str] = None,
        run_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run one agent turn and return the normalized result.

        Args:
            agent: An agent from create_agent.
            message: A string, a {role, text} dict, or a genai Content.
            user_id: Caller identity used to scope sessions.
            session_id: Reuse an existing session to continue a conversation.
                        A new session is created when omitted.
            run_config: Optional ADK RunConfig.

        Returns:
            Dict with "text", "session_id", "author", "function_calls" and the raw
            ADK "events" list.
        """
        runner, resolved_app = self._get_runner(agent, app_name)
        content = self._to_content(message)
        resolved_session = await self._ensure_session(
            runner, resolved_app, user_id, session_id
        )

        run_kwargs: Dict[str, Any] = {
            "user_id": user_id,
            "session_id": resolved_session,
            "new_message": content,
        }
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        events: List[Any] = []
        texts: List[str] = []
        function_calls: List[Dict[str, Any]] = []
        author = None

        async def consume():
            nonlocal author
            async for event in runner.run_async(**run_kwargs):
                events.append(event)
                if getattr(event, "partial", False):
                    continue
                function_calls.extend(self._event_function_calls(event))
                text = self._event_text(event)
                if text and self._is_final_response(event):
                    texts.append(text)
                    author = getattr(event, "author", None) or author

        try:
            if self._timeout is not None:
                await asyncio.wait_for(consume(), timeout=self._timeout)
            else:
                await consume()
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"agent turn exceeded the {self._timeout}s timeout"
            ) from exc
        except Exception:
            logger.exception("ADK agent run failed")
            raise

        return {
            "text": "".join(texts),
            "session_id": resolved_session,
            "author": author,
            "function_calls": function_calls,
            "events": events,
        }

    async def stream_async(
        self,
        agent: Any,
        message: Any,
        user_id: str = DEFAULT_USER_ID,
        session_id: Optional[str] = None,
        app_name: Optional[str] = None,
        run_config: Optional[Any] = None,
    ) -> AsyncIterator[str]:
        """
        Stream an agent turn, yielding text chunks as they arrive.

        Enables ADK's SSE streaming mode unless a run_config is supplied. When the
        installed ADK cannot stream, the full final text is yielded as one chunk.
        """
        runner, resolved_app = self._get_runner(agent, app_name)
        content = self._to_content(message)
        resolved_session = await self._ensure_session(
            runner, resolved_app, user_id, session_id
        )

        run_kwargs: Dict[str, Any] = {
            "user_id": user_id,
            "session_id": resolved_session,
            "new_message": content,
        }
        effective_config = run_config or self._default_stream_config()
        if effective_config is not None:
            run_kwargs["run_config"] = effective_config

        streamed = False
        try:
            async for event in runner.run_async(**run_kwargs):
                text = self._event_text(event)
                if not text:
                    continue
                if getattr(event, "partial", False):
                    streamed = True
                    yield text
                elif not streamed and self._is_final_response(event):
                    yield text
        except Exception:
            logger.exception("ADK agent stream failed")
            raise

    # ------------------------------------------------------------------
    # Sync convenience wrappers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_running_loop() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "the sync helpers cannot run inside an active event loop. Await the "
            "async variant (run_async / stream_async) instead."
        )

    def _run_sync(self, coro) -> Any:
        """Drive a coroutine to completion on a fresh event loop."""
        try:
            self._reject_running_loop()
        except RuntimeError:
            coro.close()
            raise
        return asyncio.run(coro)

    def _iter_sync(self, agen) -> Iterator[Any]:
        """Drive an async generator from sync code, yielding as chunks arrive."""
        self._reject_running_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            try:
                loop.run_until_complete(agen.aclose())
            except Exception:
                logger.debug("failed to close the async generator cleanly", exc_info=True)
            asyncio.set_event_loop(None)
            loop.close()

    def run(
        self,
        agent: Any,
        message: Any,
        user_id: str = DEFAULT_USER_ID,
        session_id: Optional[str] = None,
        app_name: Optional[str] = None,
        run_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Synchronous run_async. Raises if called from inside an event loop."""
        return self._run_sync(
            self.run_async(
                agent,
                message,
                user_id=user_id,
                session_id=session_id,
                app_name=app_name,
                run_config=run_config,
            )
        )

    def stream(
        self,
        agent: Any,
        message: Any,
        user_id: str = DEFAULT_USER_ID,
        session_id: Optional[str] = None,
        app_name: Optional[str] = None,
        run_config: Optional[Any] = None,
    ) -> Iterator[str]:
        """Synchronous stream_async. Raises if called from inside an event loop."""
        return self._iter_sync(
            self.stream_async(
                agent,
                message,
                user_id=user_id,
                session_id=session_id,
                app_name=app_name,
                run_config=run_config,
            )
        )

    def create_session_sync(
        self,
        agent: Any,
        user_id: str,
        session_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        app_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous create_session."""
        return self._run_sync(
            self.create_session(
                agent, user_id, session_id=session_id, state=state, app_name=app_name
            )
        )

    def delete_session_sync(
        self,
        agent: Any,
        user_id: str,
        session_id: str,
        app_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous delete_session."""
        return self._run_sync(
            self.delete_session(agent, user_id, session_id, app_name=app_name)
        )
