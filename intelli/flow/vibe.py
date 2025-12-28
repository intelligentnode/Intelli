from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from intelli.flow.agents.agent import Agent
from intelli.flow.dynamic_connector import ConnectorMode, DynamicConnector
from intelli.flow.flow import Flow
from intelli.flow.input.task_input import TextTaskInput, ImageTaskInput
from intelli.flow.store.memory import Memory
from intelli.flow.tasks.task import Task
from intelli.flow.tool_connector import ToolDynamicConnector
from intelli.flow.types import AgentTypes, InputTypes
from intelli.flow.utils.dynamic_utils import (
    data_exists_router,
    error_router,
    sentiment_router,
    text_content_router,
    text_length_router,
    type_router,
)
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput


# Planner (Architect) providers supported by Chatbot.
# - openai/anthropic/gemini: require planner_api_key
# - vllm: requires planner_options.baseUrl (or vllmBaseUrl); api key optional
# - llamacpp: requires planner_options.model_path; api key not used
ALLOWED_PLANNER_PROVIDERS = {"openai", "anthropic", "gemini", "vllm", "llamacpp"}
SUPPORTED_CONNECTOR_KINDS = {
    "length",
    "content",
    "sentiment",
    "error",
    "type",
    "data_exists",
    "tool",
}


@dataclass
class AgentSpec:
    agent_type: str
    provider: str
    mission: str = ""
    model_params: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    name: str
    desc: str
    agent: AgentSpec
    exclude: bool = False
    memory_key: Optional[Any] = None  # string or list[str]
    model_params: Dict[str, Any] = field(default_factory=dict)  # overrides per task
    input_type: str = InputTypes.TEXT.value  # text|image|audio... (v1 uses text/image)
    img: Optional[str] = None  # base64 string for image tasks
    post_process: Optional[str] = None  # name of the processor from the registry


@dataclass
class DynamicConnectorSpec:
    """
    Supported connector kinds:
      - length: thresholds + keys
      - content: keywords dict
      - sentiment
      - error
      - type
      - data_exists
      - tool (ToolDynamicConnector)
    """

    source: str
    kind: str
    destinations: Dict[str, str]
    name: str = "dynamic_connector"
    description: str = ""
    # kind-specific configs
    thresholds: Optional[List[int]] = None
    keys: Optional[List[str]] = None
    keywords: Optional[Dict[str, List[str]]] = None
    error_dest: Optional[str] = None
    success_dest: Optional[str] = None
    type_destinations: Optional[Dict[str, str]] = None
    default_dest: Optional[str] = None


@dataclass
class FlowSpec:
    version: str
    tasks: List[TaskSpec]
    map_paths: Dict[str, List[str]] = field(default_factory=dict)
    dynamic_connectors: List[DynamicConnectorSpec] = field(default_factory=list)
    output_memory_map: Dict[str, str] = field(default_factory=dict)
    # execution defaults (optional)
    max_workers: int = 10
    log: bool = False
    auto_save_outputs: bool = False
    output_dir: str = "./outputs"


class VibeFlow:
    """
    VibeFlow: generate / load / edit a Flow from a natural language description.

    - Planner LLM providers are restricted to: openai, anthropic, gemini
    - For testability, you can inject a `planner_fn` that returns a FlowSpec dict.
    """

    def __init__(
        self,
        *,
        planner_provider: str = "openai",
        planner_api_key: Optional[str] = None,
        planner_model: Optional[str] = "gpt-5.2",
        planner_options: Optional[Dict[str, Any]] = None,
        context_files: Optional[List[str]] = None,
        max_context_chars: int = 120_000,
        planner_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        # Preferred models/providers
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
        speech_model: Optional[str] = None,
        recognition_model: Optional[str] = None,
        processors: Optional[Dict[str, Callable]] = None,
    ):
        planner_provider = (planner_provider or "").lower()
        if planner_provider not in ALLOWED_PLANNER_PROVIDERS:
            raise ValueError(
                f"VibeFlow planner_provider must be one of {sorted(ALLOWED_PLANNER_PROVIDERS)}"
            )

        self.planner_provider = planner_provider
        self.planner_api_key = planner_api_key
        self.planner_model = planner_model
        self.planner_options = planner_options or {}
        self.context_files = context_files or self.default_context_files()
        self.max_context_chars = max_context_chars
        self._planner_fn = planner_fn  # for tests / offline usage
        self.processors = processors or {}

        # Preferences
        self.preferences = {
            "text": text_model,
            "image": image_model,
            "speech": speech_model,
            "recognition": recognition_model,
        }

        self.last_spec: Optional[Dict[str, Any]] = None
        self.last_flow: Optional[Flow] = None

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    async def build(
        self,
        description: str,
        *,
        save_dir: Optional[str] = None,
        graph_name: str = "vibeflow_graph",
        render_graph: bool = True,
        agent_factories: Optional[Dict[Tuple[str, str], Callable[[AgentSpec], Any]]] = None,
    ) -> Flow:
        spec_dict = self._plan(description, existing_spec=None)
        flow = self.build_from_spec(spec_dict, agent_factories=agent_factories)

        self.last_spec = spec_dict
        self.last_flow = flow

        if save_dir:
            self.save_bundle(
                save_dir,
                spec_dict,
                flow,
                graph_name=graph_name,
                render_graph=render_graph,
            )

        return flow

    async def edit(
        self,
        spec_path: str,
        instruction: str,
        *,
        save_dir: Optional[str] = None,
        graph_name: str = "vibeflow_graph",
        render_graph: bool = True,
        agent_factories: Optional[Dict[Tuple[str, str], Callable[[AgentSpec], Any]]] = None,
    ) -> Flow:
        existing = self.load_spec(spec_path)
        prompt = f"EDIT INSTRUCTION:\n{instruction}\n"
        spec_dict = self._plan(prompt, existing_spec=existing)
        flow = self.build_from_spec(spec_dict, agent_factories=agent_factories)

        self.last_spec = spec_dict
        self.last_flow = flow

        if save_dir:
            self.save_bundle(
                save_dir,
                spec_dict,
                flow,
                graph_name=graph_name,
                render_graph=render_graph,
            )
        return flow

    def build_from_spec(
        self,
        spec: Dict[str, Any],
        *,
        agent_factories: Optional[Dict[Tuple[str, str], Callable[[AgentSpec], Any]]] = None,
        memory: Optional[Memory] = None,
    ) -> Flow:
        self._validate_spec(spec)
        flow_spec = self._parse_spec(spec)
        return self._build_flow(flow_spec, agent_factories=agent_factories, memory=memory)

    def save_bundle(
        self,
        save_dir: str,
        spec: Dict[str, Any],
        flow: Optional[Flow] = None,
        *,
        graph_name: str = "vibeflow_graph",
        render_graph: bool = True,
    ) -> Dict[str, str]:
        os.makedirs(save_dir, exist_ok=True)

        redacted = self._redact_spec(spec)
        spec_path = os.path.join(save_dir, "flow_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(redacted, f, indent=2, ensure_ascii=False)

        graph_path = ""
        if render_graph and flow is not None:
            try:
                graph_path = flow.generate_graph_img(name=graph_name, save_path=save_dir)
            except Exception:
                graph_path = ""

        meta = {"spec": spec_path}
        if graph_path:
            meta["graph"] = graph_path
        meta_path = os.path.join(save_dir, "vibeflow_bundle.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

    def load_spec(self, spec_path: str) -> Dict[str, Any]:
        with open(spec_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_bundle(self, bundle_path: str) -> Dict[str, Any]:
        with open(bundle_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        spec_path = meta.get("spec")
        if not spec_path:
            raise ValueError("Invalid bundle: missing 'spec' path")
        return self.load_spec(spec_path)

    # ------------------------------------------------------------------
    # Planner prompt + parsing
    # ------------------------------------------------------------------
    def _plan(self, description: str, existing_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # Redact existing spec before sending it to the LLM for security
        safe_existing_spec = self._redact_spec(existing_spec) if existing_spec else None

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(description, safe_existing_spec)

        if self._planner_fn is not None:
            return self._planner_fn(system_prompt, user_prompt)

        # Only OpenAI/Gemini/Anthropic require an API key.
        if self.planner_provider in {"openai", "anthropic", "gemini"} and not self.planner_api_key:
            raise ValueError("planner_api_key is required when planner_fn is not provided")

        # Local planner providers require connection details in planner_options.
        if self.planner_provider == "vllm":
            base_url = self.planner_options.get("vllmBaseUrl") or self.planner_options.get("baseUrl")
            if not base_url:
                raise ValueError("vllm planner_provider requires planner_options.baseUrl (or vllmBaseUrl)")
        if self.planner_provider == "llamacpp":
            model_path = self.planner_options.get("model_path")
            if not model_path:
                raise ValueError("llamacpp planner_provider requires planner_options.model_path")

        chatbot = Chatbot(self.planner_api_key, self.planner_provider, options=self.planner_options)
        chat_input = ChatModelInput(system=system_prompt, model=self.planner_model)
        chat_input.add_user_message(user_prompt)
        
        try:
            raw = chatbot.chat(chat_input)[0]
            # Chatbot returns a string for normal responses; may return dict for tool calls.
            if isinstance(raw, dict):
                raise ValueError(f"Planner returned non-text response: {raw}")
            return self._extract_json_object(raw)
        except Exception as e:
            # Single retry for self-correction if JSON extraction fails
            chat_input.add_assistant_message(str(raw) if 'raw' in locals() else "Error")
            chat_input.add_user_message(
                f"Your previous response was not a valid JSON object. Error: {str(e)}. "
                "Please provide the corrected FlowSpec JSON now. Output only the JSON object."
            )
            raw_retry = chatbot.chat(chat_input)[0]
            if isinstance(raw_retry, dict):
                raise ValueError(f"Planner retry returned non-text response: {raw_retry}")
            return self._extract_json_object(raw_retry)

    def _build_system_prompt(self) -> str:
        context = self._load_context_text()
        
        # Build preference instructions
        pref_instr = ""
        if self.preferences["text"]:
            pref_instr += f"- For text tasks, use these model/provider details: {self.preferences['text']}\n"
        else:
            pref_instr += "- Prefer OpenAI, Gemini, and Anthropic providers for text tasks unless the user explicitly mentions another model.\n"
            
        if self.preferences["image"]:
            pref_instr += f"- For image tasks, use these model/provider details: {self.preferences['image']}\n"
        else:
            pref_instr += "- For image tasks, use 'openai' or 'gemini' as provider. Always include 'width': 1024, 'height': 1024, and 'response_format': 'b64_json' in model_params for image tasks.\n"
            
        if self.preferences["speech"]:
            pref_instr += f"- For speech generation, use these model/provider details: {self.preferences['speech']}\n"
        else:
            pref_instr += "- Use 'tts-1' for OpenAI speech generation.\n"
            
        if self.preferences["recognition"]:
            pref_instr += f"- For recognition tasks, use these model/provider details: {self.preferences['recognition']}\n"
        else:
            pref_instr += "- Use 'whisper-1' for OpenAI recognition tasks.\n"

        # Multi-modal chaining instructions
        proc_list = ", ".join(self.processors.keys()) if self.processors else "None"
        chain_instr = (
            "\nMulti-Modal Chaining Rules:\n"
            "- To transcribe and process audio: Task A (recognition) -> Task B (text).\n"
            "- To process text and speak it: Task A (text) -> Task B (speech).\n"
            "- To describe an image and translate: Task A (vision) -> Task B (text).\n"
            f"- Available custom processors to use in 'post_process' field: {proc_list}.\n"
        )

        local_instr = (
            "\nLocal / Offline Rules (opt-in only):\n"
            "- Only use provider 'vllm' or 'llamacpp' for a text agent when the user explicitly asks for local/offline inference AND provides the required connection details.\n"
            "- For vLLM: set agent.provider='vllm' and agent.options.baseUrl to a user-provided URL or env var (e.g. '${ENV:VLLM_BASE_URL}' or '${ENV:DEEPSEEK_VLLM_URL}'). Include agent.model_params.model (e.g. 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'). API key is optional.\n"
            "- For llama.cpp: set agent.provider='llamacpp' and agent.options.model_path to a user-provided local file path or env var (e.g. '${ENV:LLAMACPP_MODEL_PATH}'). Optionally include agent.options.model_params (e.g. {'n_ctx': 2048}). API key is not required.\n"
            "- If the user does not provide the required URL/path, do not select local providers—use OpenAI/Gemini/Anthropic instead.\n"
        )

        return (
            "You are VibeFlow Planner for the Intelli Python library.\n"
            "Your job is to generate a valid FlowSpec JSON for building an Intelli Flow.\n\n"
            "Rules:\n"
            "- Output MUST be a single JSON object (no markdown, no code fences).\n"
            "- Keep tasks small and clear. Prefer a linear flow and consolidate redundant steps to avoid overly complex graphs.\n"
            "- Do not include API keys in plaintext. Use placeholders like '${ENV:OPENAI_API_KEY}'.\n"
            "- For web search, use 'google' as provider and include 'google_api_key': '${ENV:GOOGLE_API_KEY}' and 'google_cse_id': '${ENV:GOOGLE_CSE_ID}' in model_params. Avoid 'intellicloud' search. Search agents MUST use 'google' as provider.\n"
            + pref_instr +
            chain_instr +
            local_instr +
            "- When generating speech, only provide the raw text to be spoken in the task description. Do not include meta-instructions like 'Generate audio for...'.\n"
            "- Set 'stream': false for speech generation tasks if the output is used as input for a subsequent task.\n"
            "- Only use standard model parameters (e.g., 'model', 'key', 'temperature', 'max_tokens'). Avoid inventing new parameter names like 'target_language'.\n"
            "- The Flow graph MUST be a DAG (no cycles). Use dynamic_connectors for conditional routing.\n"
            "- Tasks run when their dependencies are completed; each task consumes predecessor outputs or memory.\n\n"
            "FlowSpec JSON schema (high-level):\n"
            "{\n"
            '  "version": "1",\n'
            '  "tasks": [\n'
            "    {\n"
            '      "name": "task_name",\n'
            '      "desc": "what this task does",\n'
            '      "exclude": false,\n'
            '      "memory_key": null,\n'
            '      "post_process": "optional_processor_name",\n'
            '      "model_params": {},\n'
            '      "input_type": "text",\n'
            '      "agent": {\n'
            '        "agent_type": "text",\n'
            '        "provider": "openai|anthropic|gemini",\n'
            '        "mission": "system mission",\n'
            '        "model_params": {"key": "${ENV:OPENAI_API_KEY}", "model": "gpt-5.2"},\n'
            '        "options": {}\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "map_paths": {"task1": ["task2"]},\n'
            '  "dynamic_connectors": [\n'
            "    {\n"
            '      "source": "task1",\n'
            '      "kind": "length|content|sentiment|error|type|data_exists|tool",\n'
            '      "destinations": {"short": "task2", "long": "task3"},\n'
            '      "thresholds": [100, 200],\n'
            '      "keys": ["short","medium","long"],\n'
            '      "keywords": {"a":["x"],"b":["y"]}\n'
            "    }\n"
            "  ],\n"
            '  "output_memory_map": {"task2": "final_result"},\n'
            '  "max_workers": 10,\n'
            '  "log": false\n'
            "}\n\n"
            "Library context (read-only reference):\n"
            + context
        )

    def _build_user_prompt(self, description: str, existing_spec: Optional[Dict[str, Any]]) -> str:
        if existing_spec is None:
            return f"USER REQUEST:\n{description}\n"
        return (
            "You will modify an existing FlowSpec.\n"
            "Return the full updated FlowSpec JSON.\n\n"
            f"EXISTING_FLOW_SPEC:\n{json.dumps(existing_spec, indent=2)}\n\n"
            f"REQUEST:\n{description}\n"
        )

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        # best-effort: locate first JSON object
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            raise ValueError("Planner output did not contain a JSON object")
        return json.loads(m.group(0))

    # ------------------------------------------------------------------
    # Spec parsing + validation
    # ------------------------------------------------------------------
    def _validate_spec(self, spec: Dict[str, Any]) -> None:
        if not isinstance(spec, dict):
            raise ValueError("FlowSpec must be a JSON object")
        if "tasks" not in spec or not isinstance(spec["tasks"], list) or not spec["tasks"]:
            raise ValueError("FlowSpec.tasks must be a non-empty list")
        if "version" not in spec:
            raise ValueError("FlowSpec.version is required")

        # basic name uniqueness
        names = [t.get("name") for t in spec["tasks"] if isinstance(t, dict)]
        if len(names) != len(set(names)):
            raise ValueError("Task names must be unique")

        task_names = set(names)

        # map_paths must reference valid tasks
        map_paths = spec.get("map_paths", {}) or {}
        if not isinstance(map_paths, dict):
            raise ValueError("FlowSpec.map_paths must be an object")
        for src, dsts in map_paths.items():
            if src not in task_names:
                raise ValueError(f"map_paths references unknown source task '{src}'")
            if not isinstance(dsts, list):
                raise ValueError(f"map_paths['{src}'] must be a list of task names")
            for d in dsts:
                if d not in task_names:
                    raise ValueError(f"map_paths references unknown destination task '{d}'")

        # enforce planner-friendly providers for text agents
        for t in spec["tasks"]:
            agent = (t or {}).get("agent") or {}
            if agent.get("agent_type") == AgentTypes.TEXT.value:
                p = (agent.get("provider") or "").lower()
                if p and p not in ALLOWED_PLANNER_PROVIDERS and p not in {"local", "vllm", "llamacpp"}:
                    raise ValueError(
                        f"Unsupported text agent provider '{p}'. Allowed: {sorted(ALLOWED_PLANNER_PROVIDERS)}"
                    )

                if p == "vllm":
                    opts = agent.get("options") or {}
                    if not isinstance(opts, dict):
                        raise ValueError("vllm text agent requires agent.options to be an object")
                    base_url = opts.get("baseUrl") or opts.get("vllmBaseUrl")
                    if not base_url or not isinstance(base_url, str):
                        raise ValueError("vllm text agent requires agent.options.baseUrl")
                    if not (base_url.startswith("${ENV:") or base_url.startswith("http://") or base_url.startswith("https://")):
                        raise ValueError("vllm baseUrl must be an http(s) URL or an ${ENV:...} placeholder")

                if p == "llamacpp":
                    opts = agent.get("options") or {}
                    if not isinstance(opts, dict):
                        raise ValueError("llamacpp text agent requires agent.options to be an object")
                    model_path = opts.get("model_path") or opts.get("modelPath")
                    if not model_path or not isinstance(model_path, str):
                        raise ValueError("llamacpp text agent requires agent.options.model_path")
                    if not (model_path.startswith("${ENV:") or model_path.strip()):
                        raise ValueError("llamacpp model_path must be a non-empty path or an ${ENV:...} placeholder")
            
            # Sanity check for OpenAI image generation
            if agent.get("agent_type") == AgentTypes.IMAGE.value and agent.get("provider") == "openai":
                m_params = agent.get("model_params", {})
                if "response_format" not in m_params:
                    # Injecting default if missing to avoid corrupted outputs
                    m_params["response_format"] = "b64_json"
                    if "size" not in m_params:
                        m_params["size"] = "1024x1024"

        # validate dynamic connectors
        dyn = spec.get("dynamic_connectors", []) or []
        if not isinstance(dyn, list):
            raise ValueError("FlowSpec.dynamic_connectors must be a list")
        for c in dyn:
            if not isinstance(c, dict):
                raise ValueError("Each dynamic connector must be an object")
            source = c.get("source")
            if source not in task_names:
                raise ValueError(f"dynamic_connectors references unknown source task '{source}'")

            kind = (c.get("kind") or "").lower()
            if kind not in SUPPORTED_CONNECTOR_KINDS:
                raise ValueError(f"Unsupported dynamic connector kind: {kind}")

            destinations = c.get("destinations") or {}
            if not isinstance(destinations, dict) or not destinations:
                raise ValueError(f"dynamic connector '{source}' must include non-empty destinations")
            for _, dest_task in destinations.items():
                if dest_task not in task_names:
                    raise ValueError(f"dynamic connector '{source}' points to unknown task '{dest_task}'")

            # kind-specific rules
            if kind == "tool":
                if "tool_called" not in destinations or "no_tool" not in destinations:
                    raise ValueError("tool connector requires destinations: 'tool_called' and 'no_tool'")

            if kind == "length":
                thresholds = c.get("thresholds")
                keys = c.get("keys")
                if not isinstance(thresholds, list) or not thresholds:
                    raise ValueError("length connector requires non-empty 'thresholds' list")
                if keys is not None and (not isinstance(keys, list) or not keys):
                    raise ValueError("length connector 'keys' must be a non-empty list when provided")
                if keys is not None and len(keys) != len(thresholds) + 1:
                    raise ValueError("length connector requires len(keys) == len(thresholds) + 1")

            if kind == "content":
                keywords = c.get("keywords")
                if not isinstance(keywords, dict) or not keywords:
                    raise ValueError("content connector requires non-empty 'keywords' object")

            if kind == "type":
                type_destinations = c.get("type_destinations")
                default_dest = c.get("default_dest")
                if type_destinations is not None and not isinstance(type_destinations, dict):
                    raise ValueError("type connector 'type_destinations' must be an object when provided")
                if default_dest is not None and not isinstance(default_dest, str):
                    raise ValueError("type connector 'default_dest' must be a string when provided")

    def _parse_spec(self, spec: Dict[str, Any]) -> FlowSpec:
        tasks: List[TaskSpec] = []
        for t in spec["tasks"]:
            agent_d = t.get("agent") or {}
            agent = AgentSpec(
                agent_type=agent_d.get("agent_type", AgentTypes.TEXT.value),
                provider=agent_d.get("provider", "openai"),
                mission=agent_d.get("mission", ""),
                model_params=agent_d.get("model_params", {}) or {},
                options=agent_d.get("options", {}) or {},
            )
            tasks.append(
                TaskSpec(
                    name=t["name"],
                    desc=t.get("desc", ""),
                    agent=agent,
                    exclude=bool(t.get("exclude", False)),
                    memory_key=t.get("memory_key"),
                    model_params=t.get("model_params", {}) or {},
                    input_type=t.get("input_type", InputTypes.TEXT.value) or InputTypes.TEXT.value,
                    img=t.get("img"),
                    post_process=t.get("post_process"),
                )
            )

        connectors: List[DynamicConnectorSpec] = []
        for c in spec.get("dynamic_connectors", []) or []:
            connectors.append(
                DynamicConnectorSpec(
                    source=c["source"],
                    kind=c.get("kind", "custom"),
                    destinations=c.get("destinations", {}) or {},
                    name=c.get("name", "dynamic_connector"),
                    description=c.get("description", ""),
                    thresholds=c.get("thresholds"),
                    keys=c.get("keys"),
                    keywords=c.get("keywords"),
                    error_dest=c.get("error_dest"),
                    success_dest=c.get("success_dest"),
                    type_destinations=c.get("type_destinations"),
                    default_dest=c.get("default_dest"),
                )
            )

        return FlowSpec(
            version=str(spec.get("version")),
            tasks=tasks,
            map_paths=spec.get("map_paths", {}) or {},
            dynamic_connectors=connectors,
            output_memory_map=spec.get("output_memory_map", {}) or {},
            max_workers=int(spec.get("max_workers", 10) or 10),
            log=bool(spec.get("log", False)),
            auto_save_outputs=bool(spec.get("auto_save_outputs", False)),
            output_dir=str(spec.get("output_dir", "./outputs")),
        )

    def _resolve_placeholders(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolves ${ENV:VAR_NAME} placeholders in a dictionary or list.
        """
        if isinstance(params, dict):
            return {k: self._resolve_placeholders(v) for k, v in params.items()}
        elif isinstance(params, list):
            return [self._resolve_placeholders(v) for v in params]
        elif isinstance(params, str) and params.startswith("${ENV:") and params.endswith("}"):
            env_var = params[6:-1]
            return os.getenv(env_var, params)
        return params

    def _build_flow(
        self,
        spec: FlowSpec,
        *,
        agent_factories: Optional[Dict[Tuple[str, str], Callable[[AgentSpec], Any]]] = None,
        memory: Optional[Memory] = None,
    ) -> Flow:
        tasks: Dict[str, Any] = {}
        for t in spec.tasks:
            # Resolve placeholders in agent model_params
            t.agent.model_params = self._resolve_placeholders(t.agent.model_params)
            # Resolve placeholders in agent options (e.g. vLLM baseUrl, llama.cpp model_path)
            t.agent.options = self._resolve_placeholders(t.agent.options)
            # Resolve placeholders in task model_params
            t.model_params = self._resolve_placeholders(t.model_params)

            agent_obj = self._create_agent(t.agent, agent_factories=agent_factories)

            if t.input_type == InputTypes.IMAGE.value:
                task_input = ImageTaskInput(t.desc, t.img)
            else:
                task_input = TextTaskInput(t.desc)

            # Map post_process function if specified
            post_process_fn = None
            if t.post_process and t.post_process in self.processors:
                post_process_fn = self.processors[t.post_process]

            tasks[t.name] = Task(
                task_input=task_input,
                agent=agent_obj,
                exclude=t.exclude,
                model_params=t.model_params,
                memory_key=t.memory_key,
                post_process=post_process_fn,
                log=spec.log,
            )

        dynamic_connectors = self._create_dynamic_connectors(spec.dynamic_connectors)

        return Flow(
            tasks=tasks,
            map_paths=spec.map_paths,
            dynamic_connectors=dynamic_connectors,
            log=spec.log,
            memory=memory,
            output_memory_map=spec.output_memory_map,
            auto_save_outputs=spec.auto_save_outputs,
            output_dir=spec.output_dir,
        )

    def _create_agent(
        self,
        agent_spec: AgentSpec,
        *,
        agent_factories: Optional[Dict[Tuple[str, str], Callable[[AgentSpec], Any]]] = None,
    ):
        p = (agent_spec.provider or "").lower()
        key = (agent_spec.agent_type, p)
        if agent_factories and key in agent_factories:
            return agent_factories[key](agent_spec)

        return Agent(
            agent_type=agent_spec.agent_type,
            provider=agent_spec.provider,
            mission=agent_spec.mission,
            model_params=agent_spec.model_params,
            options=agent_spec.options,
        )

    def _create_dynamic_connectors(
        self, connectors: List[DynamicConnectorSpec]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for c in connectors:
            kind = (c.kind or "custom").lower()
            if kind == "tool":
                out[c.source] = ToolDynamicConnector(
                    destinations=c.destinations,
                    name=c.name,
                    description=c.description or "Routes based on tool usage",
                )
                continue

            if kind == "length":
                thresholds = c.thresholds or [100, 200]
                keys = c.keys or list(c.destinations.keys())

                def _fn(output, output_type, _t=thresholds, _k=keys):
                    return text_length_router(output, output_type, _t, _k)

                out[c.source] = DynamicConnector(
                    decision_fn=_fn,
                    destinations=c.destinations,
                    name=c.name,
                    description=c.description or "Routes based on text length",
                    mode=ConnectorMode.LENGTH_BASED,
                )
                continue

            if kind == "content":
                kw = c.keywords or {}

                def _fn(output, output_type, _kw=kw):
                    return text_content_router(output, output_type, _kw)

                out[c.source] = DynamicConnector(
                    decision_fn=_fn,
                    destinations=c.destinations,
                    name=c.name,
                    description=c.description or "Routes based on content keywords",
                    mode=ConnectorMode.CONTENT_BASED,
                )
                continue

            if kind == "sentiment":

                def _fn(output, output_type):
                    return sentiment_router(output, output_type, "positive", "neutral", "negative")

                out[c.source] = DynamicConnector(
                    decision_fn=_fn,
                    destinations=c.destinations,
                    name=c.name,
                    description=c.description or "Routes based on sentiment",
                    mode=ConnectorMode.CONTENT_BASED,
                )
                continue

            if kind == "error":
                err = c.error_dest or "error"
                ok = c.success_dest or "success"

                def _fn(output, output_type, _e=err, _s=ok):
                    return error_router(output, output_type, _e, _s)

                out[c.source] = DynamicConnector(
                    decision_fn=_fn,
                    destinations=c.destinations,
                    name=c.name,
                    description=c.description or "Routes based on error detection",
                    mode=ConnectorMode.ERROR_BASED,
                )
                continue

            if kind == "type":
                td = c.type_destinations or {}
                d = c.default_dest or next(iter(c.destinations.keys()), "")

                def _fn(output, output_type, _td=td, _d=d):
                    return type_router(output, output_type, _td, _d)

                out[c.source] = DynamicConnector(
                    decision_fn=_fn,
                    destinations=c.destinations,
                    name=c.name,
                    description=c.description or "Routes based on output type",
                    mode=ConnectorMode.TYPE_BASED,
                )
                continue

            if kind == "data_exists":
                exists = next(iter(c.destinations.keys()), "exists")
                missing = list(c.destinations.keys())[1] if len(c.destinations) > 1 else "missing"

                def _fn(output, output_type, _e=exists, _m=missing):
                    return data_exists_router(output, output_type, _e, _m)

                out[c.source] = DynamicConnector(
                    decision_fn=_fn,
                    destinations=c.destinations,
                    name=c.name,
                    description=c.description or "Routes based on data existence",
                    mode=ConnectorMode.CUSTOM,
                )
                continue

            raise ValueError(f"Unsupported dynamic connector kind: {kind}")
        return out

    # ------------------------------------------------------------------
    # Context loading (for planner prompt)
    # ------------------------------------------------------------------
    @staticmethod
    def default_context_files() -> List[str]:
        # Keep it small: the planner mostly needs the conceptual API.
        return [
            "flow/flow.py",
            "flow/tasks/task.py",
            "flow/agents/agent.py",
            "flow/types.py",
            "flow/dynamic_connector.py",
            "flow/tool_connector.py",
            "flow/utils/dynamic_utils.py",
            "flow/store/memory.py",
            "flow/store/dbmemory.py",
            "flow/input/task_input.py",
            "flow/input/agent_input.py",
        ]

    def _load_context_text(self) -> str:
        chunks: List[str] = []
        remaining = self.max_context_chars
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        for rel in self.context_files:
            path = os.path.join(base, rel)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
            except Exception:
                # Silently skip files that cannot be read
                continue

            if remaining <= 0:
                break
            snippet = txt[: max(0, remaining)]
            remaining -= len(snippet)
            chunks.append(f"\n--- FILE: {rel} ---\n{snippet}\n")
        return "\n".join(chunks)

    # ------------------------------------------------------------------
    # Redaction for saving
    # ------------------------------------------------------------------
    @staticmethod
    def _redact_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
        secret_keys = {
            "key",
            "api_key",
            "one_key",
            "google_api_key",
            "key_value",
            "anthropic_key",
            "openai_key",
        }

        def _walk(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if k in secret_keys and isinstance(v, str) and not v.startswith("${ENV:"):
                        out[k] = "<REDACTED>"
                    else:
                        out[k] = _walk(v)
                return out
            if isinstance(obj, list):
                return [_walk(x) for x in obj]
            return obj

        return _walk(spec)


class VibeAgent(VibeFlow):
    """
    VibeAgent is an alias for VibeFlow to allow the user to call it by either name.
    """
    pass
