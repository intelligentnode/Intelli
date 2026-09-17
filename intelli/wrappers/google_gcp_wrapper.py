import asyncio
import importlib
import inspect
import logging
import os
import random
import tempfile
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union

from intelli.config import config

logger = logging.getLogger(__name__)

_INSTALL_HINT = "pip install intelli[agent]"

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


def _check_speech_imports():
    """Return Speech-to-Text V2 client pieces or raise with an install hint."""
    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
    except ImportError as exc:
        raise ImportError(
            "google-cloud-speech is required for Chirp 3 transcription. "
            f"Install it with: {_INSTALL_HINT}"
        ) from exc
    return ClientOptions, SpeechClient, cloud_speech


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


def _is_not_found(exc: Exception) -> bool:
    """Best-effort 404 detection across google-api-core and gRPC errors."""
    if exc.__class__.__name__ == "NotFound":
        return True
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            return getattr(code(), "name", "") == "NOT_FOUND"
        except Exception:
            return False
    if getattr(exc, "status_code", None) == 404:
        return True
    return "404" in str(exc) or "not found" in str(exc).lower()


class GoogleGCPWrapper:
    """
    Wrapper for Google ADK agents backed by Vertex AI, plus Vertex AI Search and
    RAG Engine grounding.

    Authenticates with Application Default Credentials by default, and accepts a
    service account key file or a prebuilt credentials object instead.
    """

    DEFAULT_LOCATION = "us-central1"
    DEFAULT_CHIRP_LOCATION = "us"
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
        cache_runners: bool = True,
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
            cache_runners: Reuse runners so sessions survive across turns. Set to
                           False for per-request agents to avoid retaining them.
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
        self._cache_runners = cache_runners

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

    def _resolve_transcription_audio(
        self,
        audio_file: Optional[Union[str, bytes, bytearray]],
        audio_uri: Optional[str],
    ) -> Tuple[Optional[str], Optional[bytes]]:
        """Return (gcs_uri, inline_bytes) from the supported inputs."""
        resolved_uri = audio_uri
        audio_bytes = None

        if isinstance(audio_file, (bytes, bytearray)):
            if not audio_file:
                raise ValueError("audio_file bytes cannot be empty")
            audio_bytes = bytes(audio_file)
        elif isinstance(audio_file, str) and audio_file.startswith("gs://"):
            if resolved_uri and resolved_uri != audio_file:
                raise ValueError("pass only one of audio_file or audio_uri for a gs:// path")
            resolved_uri = audio_file
        elif isinstance(audio_file, str):
            if not os.path.isfile(audio_file):
                raise ValueError(f"audio file does not exist: {audio_file}")
            with open(audio_file, "rb") as handle:
                audio_bytes = handle.read()
            if not audio_bytes:
                raise ValueError(f"audio file is empty: {audio_file}")
        elif audio_file is not None:
            raise ValueError("audio_file must be a file path, gs:// URI, or bytes")

        if resolved_uri and audio_bytes is not None:
            raise ValueError("pass either local audio or audio_uri, not both")
        if not resolved_uri and audio_bytes is None:
            raise ValueError("audio_file or audio_uri is required")
        if resolved_uri and not resolved_uri.startswith("gs://"):
            raise ValueError("audio_uri must be a gs:// Cloud Storage URI")

        return resolved_uri, audio_bytes

    # ------------------------------------------------------------------
    # Speech transcription (Chirp 3 on Speech-to-Text V2)
    # ------------------------------------------------------------------

    def transcribe_chirp3(
        self,
        audio_file: Optional[Union[str, bytes, bytearray]] = None,
        audio_uri: Optional[str] = None,
        language: Optional[Union[str, List[str]]] = None,
        model: Optional[str] = None,
        location: Optional[str] = None,
        diarization: bool = False,
        custom_vocabulary: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio with Chirp 3 on Cloud Speech-to-Text V2.

        Local files/bytes use synchronous ``Recognize`` (audio shorter than
        about one minute). Existing ``gs://`` URIs use ``BatchRecognize``,
        which supports recordings up to about one hour. This method does not
        upload or store files.

        Args:
            audio_file: Local path or raw audio bytes. A ``gs://`` string is
                        treated as ``audio_uri``.
            audio_uri: Existing Cloud Storage URI. Intelli does not write this.
            language: BCP-47 code (``en-US``), a 2-letter alias (``en``),
                      ``auto``, or a list of codes.
            model: Speech-to-Text model id. Defaults to ``chirp_3``.
            location: Speech-to-Text region. Defaults to ``us`` (Chirp 3 GA).
            diarization: Label speakers. Honored on batch recognition.
            custom_vocabulary: Phrase hints for names and domain terms.
            timeout: Seconds to wait for a batch job. Defaults to the wrapper
                     timeout, then 600 seconds.

        Returns:
            ``{"text": str, "model": str, "method": str, "response": dict}``
        """
        resolved_uri, audio_bytes = self._resolve_transcription_audio(
            audio_file, audio_uri
        )
        gcp_config = config.get("url", {}).get("gcp", {})
        resolved_model = (
            model or gcp_config.get("models", {}).get("chirp") or "chirp_3"
        )
        resolved_location = (
            location
            or gcp_config.get("chirp_location")
            or self.DEFAULT_CHIRP_LOCATION
        )
        language_codes = self._chirp_language_codes(language)
        ClientOptions, SpeechClient, cloud_speech = _check_speech_imports()
        client = self._chirp_client(
            ClientOptions, SpeechClient, resolved_location
        )
        recognition_config = self._chirp_recognition_config(
            cloud_speech,
            model=resolved_model,
            language_codes=language_codes,
            diarization=diarization,
            custom_vocabulary=custom_vocabulary,
        )
        recognizer = (
            f"projects/{self.project_id}/locations/{resolved_location}/recognizers/_"
        )

        if resolved_uri:
            method = "batch_recognize"
            response = self._chirp_batch_recognize(
                client,
                cloud_speech,
                recognizer=recognizer,
                recognition_config=recognition_config,
                audio_uri=resolved_uri,
                timeout=timeout,
            )
            text = self._chirp_text_from_batch(response, resolved_uri)
        else:
            method = "recognize"
            response = self._with_retry(
                "transcribe_chirp3",
                lambda: self._chirp_sync_recognize(
                    client,
                    cloud_speech,
                    recognizer=recognizer,
                    recognition_config=recognition_config,
                    audio_bytes=audio_bytes,
                ),
            )
            text = self._chirp_text_from_recognize(response)

        logger.debug(
            "transcribed audio with %s via %s in %s (%s chars)",
            resolved_model,
            method,
            resolved_location,
            len(text),
        )
        return {
            "text": text,
            "model": resolved_model,
            "method": method,
            "response": _to_plain(response),
        }

    def _chirp_client(self, ClientOptions: Any, SpeechClient: Any, location: str) -> Any:
        """Build a Speech-to-Text V2 client for the Chirp regional endpoint."""
        options = self._call_filtered(
            "ClientOptions",
            ClientOptions,
            api_endpoint=f"{location}-speech.googleapis.com",
            quota_project_id=self.project_id,
        )
        return self._call_filtered(
            "SpeechClient",
            SpeechClient,
            credentials=self.credentials,
            client_options=options,
        )

    def _chirp_language_codes(
        self, language: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """Normalize Whisper-style codes to Chirp BCP-47 values."""
        aliases = {
            "en": "en-US",
            "es": "es-US",
            "fr": "fr-FR",
            "de": "de-DE",
            "it": "it-IT",
            "pt": "pt-BR",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "zh": "cmn-Hans-CN",
            "hi": "hi-IN",
        }
        codes = self._as_list(language) if language else ["auto"]
        mapped = []
        for code in codes:
            if not isinstance(code, str) or not code.strip():
                continue
            resolved = code.strip()
            mapped.append(aliases.get(resolved, resolved))
        return mapped or ["auto"]

    def _chirp_recognition_config(
        self,
        cloud_speech: Any,
        model: str,
        language_codes: List[str],
        diarization: bool = False,
        custom_vocabulary: Optional[List[str]] = None,
    ) -> Any:
        """Build a V2 RecognitionConfig for Chirp 3."""
        config_kwargs: Dict[str, Any] = {
            "auto_decoding_config": cloud_speech.AutoDetectDecodingConfig(),
            "language_codes": language_codes,
            "model": model,
        }
        if diarization:
            diarization_config = self._call_filtered(
                "SpeakerDiarizationConfig",
                cloud_speech.SpeakerDiarizationConfig,
            )
            config_kwargs["features"] = self._call_filtered(
                "RecognitionFeatures",
                cloud_speech.RecognitionFeatures,
                diarization_config=diarization_config,
            )
        phrases = [term for term in (custom_vocabulary or []) if term]
        adaptation_cls = getattr(cloud_speech, "SpeechAdaptation", None)
        phrase_set_cls = getattr(cloud_speech, "PhraseSet", None)
        phrase_set_wrap_cls = getattr(adaptation_cls, "AdaptationPhraseSet", None) if adaptation_cls else None
        if phrases and phrase_set_cls is not None and phrase_set_wrap_cls is not None:
            inline_set = self._call_filtered(
                "PhraseSet",
                phrase_set_cls,
                phrases=[{"value": term} for term in phrases],
            )
            phrase_set_wrap = self._call_filtered(
                "AdaptationPhraseSet",
                phrase_set_wrap_cls,
                inline_phrase_set=inline_set,
            )
            config_kwargs["adaptation"] = self._call_filtered(
                "SpeechAdaptation",
                adaptation_cls,
                phrase_sets=[phrase_set_wrap],
            )
        elif phrases:
            logger.warning(
                "installed speech SDK does not support Chirp phrase adaptation, ignoring"
            )
        return self._call_filtered(
            "RecognitionConfig", cloud_speech.RecognitionConfig, **config_kwargs
        )

    def _chirp_sync_recognize(
        self,
        client: Any,
        cloud_speech: Any,
        recognizer: str,
        recognition_config: Any,
        audio_bytes: Optional[bytes],
    ) -> Any:
        """Run synchronous Recognize for short inline audio."""
        request = self._call_filtered(
            "RecognizeRequest",
            cloud_speech.RecognizeRequest,
            recognizer=recognizer,
            config=recognition_config,
            content=audio_bytes,
        )
        return client.recognize(request=request)

    def _chirp_batch_recognize(
        self,
        client: Any,
        cloud_speech: Any,
        recognizer: str,
        recognition_config: Any,
        audio_uri: str,
        timeout: Optional[float] = None,
    ) -> Any:
        """Run BatchRecognize for an existing GCS URI. Does not upload."""
        wait_timeout = timeout if timeout is not None else (self._timeout or 600)
        file_metadata = self._call_filtered(
            "BatchRecognizeFileMetadata",
            cloud_speech.BatchRecognizeFileMetadata,
            uri=audio_uri,
        )
        output_config = self._call_filtered(
            "RecognitionOutputConfig",
            cloud_speech.RecognitionOutputConfig,
            inline_response_config=self._call_filtered(
                "InlineOutputConfig",
                cloud_speech.InlineOutputConfig,
            ),
        )
        request = self._call_filtered(
            "BatchRecognizeRequest",
            cloud_speech.BatchRecognizeRequest,
            recognizer=recognizer,
            config=recognition_config,
            files=[file_metadata],
            recognition_output_config=output_config,
        )
        operation = client.batch_recognize(request=request)
        result_fn = getattr(operation, "result", None)
        if not callable(result_fn):
            return operation
        try:
            return result_fn(timeout=wait_timeout)
        except TypeError:
            return result_fn()

    @staticmethod
    def _chirp_text_from_recognize(response: Any) -> str:
        """Join alternative transcripts from a RecognizeResponse."""
        chunks: List[str] = []
        for result in getattr(response, "results", None) or []:
            alternatives = getattr(result, "alternatives", None) or []
            if not alternatives:
                continue
            transcript = getattr(alternatives[0], "transcript", "") or ""
            if transcript.strip():
                chunks.append(transcript.strip())
        return " ".join(chunks).strip()

    def _chirp_text_from_batch(self, response: Any, audio_uri: str) -> str:
        """Extract transcript text from a BatchRecognize response."""
        results_map = getattr(response, "results", None)
        file_result = None
        if results_map is None:
            return self._chirp_text_from_recognize(response)
        try:
            file_result = results_map[audio_uri]
        except Exception:
            getter = getattr(results_map, "get", None)
            if callable(getter):
                file_result = getter(audio_uri)
        if file_result is None:
            return ""
        transcript = getattr(file_result, "transcript", file_result)
        return self._chirp_text_from_recognize(transcript)

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

    def list_rag_corpora(
        self, display_name_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List RAG corpora, optionally filtered by display-name prefix."""
        rag = self._init_vertex()
        corpora = self._with_retry("list_rag_corpora", rag.list_corpora)
        results = [_to_plain(corpus) for corpus in corpora]
        if not display_name_prefix:
            return results
        return [
            corpus
            for corpus in results
            if str(corpus.get("display_name") or "").startswith(display_name_prefix)
        ]

    def get_rag_corpus(self, corpus_name: str) -> Dict[str, Any]:
        """Fetch a single RAG corpus by id or full resource name."""
        name = self.qualify_corpus_name(corpus_name)
        rag = self._init_vertex()
        corpus = self._with_retry("get_rag_corpus", lambda: rag.get_corpus(name=name))
        return _to_plain(corpus)

    def delete_rag_corpus(
        self, corpus_name: str, missing_ok: bool = True
    ) -> Dict[str, Any]:
        """Delete a RAG corpus, optionally tolerating one that is already gone."""
        name = self.qualify_corpus_name(corpus_name)
        rag = self._init_vertex()
        try:
            self._with_retry("delete_rag_corpus", lambda: rag.delete_corpus(name=name))
        except Exception as exc:
            if missing_ok and _is_not_found(exc):
                logger.info("RAG corpus %s already gone", name)
                return {"name": name, "deleted": False, "missing": True}
            raise
        logger.info("deleted RAG corpus %s", name)
        return {"name": name, "deleted": True, "missing": False}

    def rag_corpus_exists(self, corpus_name: str) -> bool:
        """Return whether a corpus is still reachable."""
        try:
            self.get_rag_corpus(corpus_name)
            return True
        except Exception as exc:
            if _is_not_found(exc):
                return False
            raise

    def update_rag_corpus(
        self,
        corpus_name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rename or re-describe a RAG corpus."""
        if not display_name and not description:
            raise ValueError("provide display_name or description")
        name = self.qualify_corpus_name(corpus_name)
        rag = self._init_vertex()
        update_kwargs: Dict[str, Any] = {"corpus_name": name}
        if display_name:
            update_kwargs["display_name"] = display_name
        if description:
            update_kwargs["description"] = description
        update_kwargs = self._with_timeout(rag.update_corpus, update_kwargs)
        corpus = self._with_retry(
            "update_rag_corpus",
            lambda: self._call_filtered(
                "update_rag_corpus", rag.update_corpus, **update_kwargs
            ),
        )
        logger.info("updated RAG corpus %s", name)
        return _to_plain(corpus)

    def upload_rag_file(
        self,
        corpus_name: str,
        file: Any,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a local file, in-memory bytes tuple, or readable file object."""
        name = self.qualify_corpus_name(corpus_name)
        rag = self._init_vertex()
        path, resolved_name, cleanup = self._materialize(file)
        upload_kwargs: Dict[str, Any] = {
            "corpus_name": name,
            "path": path,
            "display_name": display_name or resolved_name,
        }
        if description:
            upload_kwargs["description"] = description
        upload_kwargs = self._with_timeout(rag.upload_file, upload_kwargs)

        try:
            rag_file = self._with_retry(
                "upload_rag_file",
                lambda: self._call_filtered(
                    "upload_rag_file", rag.upload_file, **upload_kwargs
                ),
            )
        finally:
            cleanup()

        logger.info(
            "uploaded %s into %s", getattr(rag_file, "name", resolved_name), name
        )
        return _to_plain(rag_file)

    @staticmethod
    def _materialize(file: Any) -> Tuple[str, str, Callable[[], None]]:
        """Return (path, display name, cleanup) for any supported file form."""
        if isinstance(file, (tuple, list)):
            if len(file) < 2:
                raise ValueError(
                    "file tuple must be (filename, content[, content_type])"
                )
            filename = str(file[0] or "document")
            content = file[1] or b""
        elif hasattr(file, "read"):
            filename = str(getattr(file, "name", "") or "document")
            content = file.read()
            if hasattr(file, "seek"):
                file.seek(0)
        elif isinstance(file, str):
            if not os.path.isfile(file):
                raise ValueError(f"file not found: {file}")
            return file, os.path.basename(file), lambda: None
        else:
            raise ValueError(
                "file must be a local path, a (filename, content) tuple, or a "
                "readable file object"
            )

        if isinstance(content, str):
            content = content.encode("utf-8")
        display_name = os.path.basename(filename) or "document"
        suffix = os.path.splitext(display_name)[1] or ".txt"
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            handle.write(content)
            handle.flush()
        finally:
            handle.close()

        def cleanup() -> None:
            try:
                os.unlink(handle.name)
            except OSError:
                logger.debug("could not remove temp file %s", handle.name)

        return handle.name, display_name, cleanup

    def delete_rag_file(
        self,
        file_name: str,
        corpus_name: Optional[str] = None,
        missing_ok: bool = True,
    ) -> Dict[str, Any]:
        """Delete one file from a RAG corpus."""
        if not file_name or not isinstance(file_name, str):
            raise ValueError("file_name must be a non-empty string")
        rag = self._init_vertex()
        delete_kwargs: Dict[str, Any] = {"name": file_name}
        if corpus_name:
            delete_kwargs["corpus_name"] = self.qualify_corpus_name(corpus_name)
        try:
            self._with_retry(
                "delete_rag_file",
                lambda: self._call_filtered(
                    "delete_rag_file", rag.delete_file, **delete_kwargs
                ),
            )
        except Exception as exc:
            if missing_ok and _is_not_found(exc):
                logger.info("RAG file %s already gone", file_name)
                return {"name": file_name, "deleted": False, "missing": True}
            raise
        return {"name": file_name, "deleted": True, "missing": False}

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
        Return a (runner, app_name) pair for an agent.

        Runners are cached by default so session state survives across turns. Set
        cache_runners=False at construction time for short-lived, per-request
        agents; those runners are closed after generation and never retained.
        """
        if agent is None:
            raise ValueError("agent is required")
        resolved_app = app_name or self.app_name
        _, InMemoryRunner, _ = _check_adk_imports()
        if not getattr(self, "_cache_runners", True):
            return self._build_runner(InMemoryRunner, agent, resolved_app), resolved_app

        key = (resolved_app, id(agent))
        runner = self._runners.get(key)
        if runner is None:
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

    @staticmethod
    async def _dispose(runner: Any) -> None:
        """Release a runner's plugins and services; never raise."""
        close = getattr(runner, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("runner close failed", exc_info=True)

    async def _dispose_if_uncached(self, runner: Any) -> None:
        if not getattr(self, "_cache_runners", True):
            await self._dispose(runner)

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
    def _event_usage(event: Any) -> Dict[str, Optional[int]]:
        """Pull prompt and candidate token counts from an event when present."""
        usage = getattr(event, "usage_metadata", None)
        if usage is None:
            return {}
        return {
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }

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

    async def _iter_events_with_timeout(
        self, events: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        """Iterate events while enforcing the configured whole-turn timeout."""
        if self._timeout is None:
            async for event in events:
                yield event
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        iterator = events.__aiter__()
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"agent stream exceeded the {self._timeout}s timeout"
                )
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"agent stream exceeded the {self._timeout}s timeout"
                ) from exc
            yield event

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
        user_id: Optional[str] = None,
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
            Dict with text, session details, function calls, token usage, and the
            raw ADK events.
        """
        runner, resolved_app = self._get_runner(agent, app_name)
        content = self._to_content(message)
        resolved_user = user_id or self.DEFAULT_USER_ID
        resolved_session = await self._ensure_session(
            runner, resolved_app, resolved_user, session_id
        )

        run_kwargs: Dict[str, Any] = {
            "user_id": resolved_user,
            "session_id": resolved_session,
            "new_message": content,
        }
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        events: List[Any] = []
        texts: List[str] = []
        function_calls: List[Dict[str, Any]] = []
        usage: Dict[str, Optional[int]] = {}
        author = None

        async def consume():
            nonlocal author, usage
            async for event in runner.run_async(**run_kwargs):
                events.append(event)
                usage = self._event_usage(event) or usage
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
        finally:
            await self._dispose_if_uncached(runner)

        return {
            "text": "".join(texts),
            "session_id": resolved_session,
            "author": author,
            "function_calls": function_calls,
            "usage": usage,
            "events": events,
        }

    async def stream_async(
        self,
        agent: Any,
        message: Any,
        user_id: Optional[str] = None,
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
        resolved_user = user_id or self.DEFAULT_USER_ID
        resolved_session = await self._ensure_session(
            runner, resolved_app, resolved_user, session_id
        )

        run_kwargs: Dict[str, Any] = {
            "user_id": resolved_user,
            "session_id": resolved_session,
            "new_message": content,
        }
        effective_config = run_config or self._default_stream_config()
        if effective_config is not None:
            run_kwargs["run_config"] = effective_config

        streamed = False
        try:
            async for event in self._iter_events_with_timeout(
                runner.run_async(**run_kwargs)
            ):
                text = self._event_text(event)
                if not text:
                    continue
                if getattr(event, "partial", False):
                    streamed = True
                    yield text
                elif not streamed and self._is_final_response(event):
                    yield text
        except TimeoutError:
            raise
        except Exception:
            logger.exception("ADK agent stream failed")
            raise
        finally:
            await self._dispose_if_uncached(runner)

    async def stream_events_async(
        self,
        agent: Any,
        message: Any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        app_name: Optional[str] = None,
        run_config: Optional[Any] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Yield ("delta", text) events followed by one ("complete", result)."""
        runner, resolved_app = self._get_runner(agent, app_name)
        content = self._to_content(message)
        resolved_user = user_id or self.DEFAULT_USER_ID
        resolved_session = await self._ensure_session(
            runner, resolved_app, resolved_user, session_id
        )

        run_kwargs: Dict[str, Any] = {
            "user_id": resolved_user,
            "session_id": resolved_session,
            "new_message": content,
        }
        effective_config = run_config or self._default_stream_config()
        if effective_config is not None:
            run_kwargs["run_config"] = effective_config

        chunks: List[str] = []
        usage: Dict[str, Optional[int]] = {}
        streamed = False
        try:
            async for event in self._iter_events_with_timeout(
                runner.run_async(**run_kwargs)
            ):
                usage = self._event_usage(event) or usage
                text = self._event_text(event)
                if not text:
                    continue
                if getattr(event, "partial", False):
                    streamed = True
                    chunks.append(text)
                    yield "delta", text
                elif not streamed and self._is_final_response(event):
                    chunks.append(text)
                    yield "delta", text
        finally:
            await self._dispose_if_uncached(runner)

        yield "complete", {
            "text": "".join(chunks),
            "session_id": resolved_session,
            "usage": usage,
        }

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

    def stream_events(
        self,
        agent: Any,
        message: Any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        app_name: Optional[str] = None,
        run_config: Optional[Any] = None,
    ) -> Iterator[Tuple[str, Any]]:
        """Synchronous stream_events_async. Raises inside an event loop."""
        return self._iter_sync(
            self.stream_events_async(
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
