from __future__ import annotations

import inspect
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
from intelli.store.factory import HISTORY_TYPES, STORE_TYPES


# Planner (Architect) providers supported by Chatbot.
# - openai/anthropic/gemini: require planner_api_key
# - vllm: requires planner_options.baseUrl (or vllmBaseUrl); api key optional
# - llamacpp: requires planner_options.model_path; api key not used
ALLOWED_PLANNER_PROVIDERS = {"openai", "anthropic", "gemini", "vllm", "llamacpp"}
# Providers of language steps. The assistant also takes 'vertex' (Gemini on Vertex AI) and 'ollama' (vllm on
# http://localhost:11434); 'local' is accepted on text agents of older specs.
TEXT_AGENT_PROVIDERS = {"openai", "anthropic", "gemini", "mistral", "nvidia", "aws", "vllm", "llamacpp", "local"}
ASSISTANT_PROVIDERS = (TEXT_AGENT_PROVIDERS - {"local"}) | {"vertex", "ollama"}
COMPUTER_PROVIDERS = {"anthropic", "openai"}
# Planner replies checked against the spec rules; a failing reply is sent back with the error.
PLANNER_ATTEMPTS = 3
SUPPORTED_CONNECTOR_KINDS = {
    "length",
    "content",
    "sentiment",
    "error",
    "type",
    "data_exists",
    "tool",
}

# The planner instructions: a self-contained reference of the step types and rules. {preferences} and
# {registered} are filled per VibeFlow.
PLANNER_PROMPT = """You are the VibeFlow planner of Intelli, a Python library that runs AI work as a flow: a graph of small tasks, each on its own model.
Turn the user's request into one FlowSpec JSON object.

## Output
Reply with the FlowSpec JSON object only: no markdown, no code fences, no comments.

## Design the graph
- One clear job per task, named in snake_case.
- Tasks that do not depend on each other run in parallel. When the user wants one result from parallel work, add a task that joins it.
- Keep the graph as small as the request allows: no extra review, formatting or "final output" tasks unless the user asks for them.
- The graph must be a DAG. map_paths maps each task to the tasks that receive its output.

## Run input
- The app runs the flow with `await flow.start(initial_input=...)`. Every task without predecessors receives that input: a ticket, a question, commit messages, a blog post.
- Write each desc as an instruction about its input, e.g. "Rate the urgency of the ticket." Put content in desc only when it is the same for every run (a topic or a city the user named). Never invent sample data.
- A task receives only its predecessors' outputs, joined into one text. Make sure every task gets what it needs.

## Step types (agent.agent_type)
- "assistant": every language step: write, rewrite, classify, extract, summarize, translate, plan, review, write code, answer questions, chat. It can also answer from the user's documents, keep conversation history, remember a person across conversations, call registered tools and search the web. See "Assistant steps".
- "image": generate an image. provider "openai" (model "gpt-image-2"), "gemini" or "stability"; model_params "width": 1024, "height": 1024. Its input text (or its desc when it has no predecessor) is the image description.
- "vision": look at the image made by the previous task and answer about it in text. provider "openai" or "gemini"; model_params must include "model" ("gpt-5.5" or "gemini-2.5-flash") and "extension": "png".
- "speech": text to speech. provider "openai" (model "tts-1", "voice": "alloy") or "elevenlabs" ("voice": a voice id, "model": "eleven_multilingual_v2"); set "stream": false and leave the mission empty (it would be spoken too). It speaks its input word for word, so its predecessor must output only the words to say.
- "recognition": speech to text from the audio of the previous task. provider "openai" (model "whisper-1").
- "computer": operate a website through screenshots like a person: check user journeys on a site, read portals that have no API. See "Computer steps".
- "search": raw web results from Google Custom Search: provider "google" with model_params "google_api_key": "${ENV:GOOGLE_API_KEY}" and "google_cse_id": "${ENV:GOOGLE_CSE_ID}". For answers that need current web facts, prefer an assistant on gemini with "google_search": true.
- "mcp": call one tool of an MCP server: model_params "command" and "args" (a local server) or "url" (a remote one), "tool", plus "arg_<name>" values or "input_arg" (the argument that receives the task input).
- "coder": a coding agent that edits files in model_params "workspace" and runs "test_command". provider "anthropic" or "openai".

## Assistant steps
- agent.mission: the assistant's role and standing rules. desc: this step's instruction. The step's input is the message it answers.
- model_params: "key", "model", and optional "temperature" and "max_tokens".
- Documents (retrieval): when answers must come from the user's documents (a handbook, policies, an FAQ, a catalog, notes, contracts, files), set:
  - options.knowledge: a registered store name, or a store config. Use {"type": "memory", "path": "./knowledge.json"} (a local file, no server) unless the user names a database, e.g. {"type": "qdrant", "url": "${ENV:QDRANT_URL}", "collection": "docs"}; other types: chroma, weaviate, milvus, elasticsearch, pinecone, pgvector, mongodb_atlas, firestore, vertex_rag, vertex_vector_search.
  - options.files: paths of text files the user named, or options.documents: [{"id": "refunds", "text": "...", "metadata": {"title": "Refund policy"}}] for text given in the request. They are embedded the first time the step runs. A registered store may already hold the documents.
  - model_params "show_sources": true lists the cited sources under the answer; "top_k" sets how many chunks are retrieved (default 4).
- Conversation history: when one conversation goes on over several runs (a chat, follow-up questions), set options.history {"type": "file", "dir": "./conversations"} and model_params "conversation_id" (e.g. "default"; the app can change it per conversation). Without it every run starts fresh.
- Long-term memory: when the assistant must remember a person's preferences or facts across conversations, set options.memory {"type": "memory", "path": "./memory.json"} and model_params "user_id" (e.g. "default").
- Stores are embedded with the assistant's own provider on openai, gemini, mistral and aws. On other providers add an "embedder" to every store config, e.g. {"provider": "openai", "api_key": "${ENV:OPENAI_API_KEY}"}.
- Tools: options.tools lists registered tool names. The assistant calls them when needed and answers with their results.
- Current web facts: provider "gemini" with model_params "google_search": true; add "show_sources": true to list the web sources.
- Add documents, history, memory and tools only when the request needs them.

## Computer steps
- provider "anthropic" (model "claude-sonnet-5") or "openai" (model "gpt-5.5"); model_params "key", "model", "start_url" (required: the first page), "max_iterations" (10 to 30) and "headless": true.
- desc: one journey in plain words and what to report, e.g. "Search for a laptop stand and report how many results appear." Give each journey its own task so they run in parallel, then join them in an assistant task that writes the report.
- model_params "on_action": a registered guard name. Set it whenever a guard is registered and the step should not change anything (smoke checks, reading portals).
- Never put passwords in desc.

## Routing (dynamic_connectors)
- Use a connector when only one of several next tasks should run.
- kind "content": routes on words in the source task's output. Make the source's mission end its answer with a label line such as "ROUTE: high", and use the exact label lines as keywords: {"normal": ["route: normal"], "high": ["route: high"]}. When nothing matches, the first key is chosen, so put the safe default first.
- A keyword key without a destination stops the flow there. To run a task only for one label, map just that key: keywords {"normal": ["route: normal"], "alert": ["route: alert"]} with destinations {"alert": "write_alert"}; never add a task that does nothing.
- kind "sentiment": keys "positive", "neutral" and "negative". kind "length": "thresholds" (in characters) and len(thresholds)+1 "keys". kind "error": "error_dest" and "success_dest" keys.
- "destinations" maps each key to a task. A destination runs only when it is chosen: do not list it in the source's map_paths, and do not join two branches of one connector in a later task (it would wait for a branch that never runs).

## Providers, models and keys
{preferences}
- Keys are placeholders, never real values: "${ENV:OPENAI_API_KEY}", "${ENV:ANTHROPIC_API_KEY}", "${ENV:GEMINI_API_KEY}", "${ENV:MISTRAL_API_KEY}", "${ENV:STABILITY_API_KEY}", "${ENV:ELEVENLABS_API_KEY}". The key goes in model_params "key".
- Other language providers, when the user names them: "mistral" ("mistral-large-latest"), "aws" (Amazon Bedrock; options {"region": "us-east-1"}; no key with AWS credentials) and "vertex" (Gemini on Vertex AI; options {"project_id": "${ENV:VERTEX_PROJECT_ID}", "location": "us-central1"}).
- Use only the model_params and options described here.

## Local models (only when the user asks for local or offline models)
- Ollama: provider "ollama", model_params "model" (e.g. "qwen2.5:0.5b"); options {"baseUrl": "http://localhost:11434"} unless the user gives another URL.
- vLLM: provider "vllm", options {"baseUrl": the URL the user gave or an ${ENV:...} placeholder} and model_params "model".
- llama.cpp: provider "llamacpp", options {"model_path": the path the user gave or an ${ENV:...} placeholder}.
- They need no key. Stores of vllm and llamacpp assistants need an "embedder"; Ollama assistants embed with "nomic-embed-text".

## Registered by the app (use only these names)
{registered}

## FlowSpec
{
  "version": "1",
  "tasks": [
    {
      "name": "answer",
      "desc": "Answer the customer's question.",
      "agent": {
        "agent_type": "assistant",
        "provider": "openai",
        "mission": "You are Acme support. Reply in at most three sentences.",
        "model_params": {"key": "${ENV:OPENAI_API_KEY}", "model": "gpt-5.5"},
        "options": {}
      },
      "model_params": {},
      "exclude": false,
      "post_process": null
    }
  ],
  "map_paths": {"answer": []},
  "dynamic_connectors": [],
  "auto_save_outputs": false,
  "output_dir": "./outputs",
  "log": false
}
- "exclude": true leaves a task's output out of the result. Use it only for helper text nobody reads, such as an image prompt; keep the other steps in the result so people can check them.
- Task "model_params" override the agent's model_params for that task.
- "post_process": a registered processor name; it turns the task's output into another value.
- Set "auto_save_outputs": true when the flow makes images or audio, so they are saved in "output_dir".
"""


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

    - Planner LLM providers are restricted to: openai, anthropic, gemini, vllm, llamacpp
    - For testability, you can inject a `planner_fn` that returns a FlowSpec dict.
    - Specs are JSON, so the app registers its Python objects by name: `processors` (a task's post_process),
      `tools` (functions an assistant step can call), `stores` (vector stores or chat histories for assistant
      steps) and `guards` (on_action / on_safety_check hooks of computer steps).
    """

    def __init__(
        self,
        *,
        planner_provider: str = "openai",
        planner_api_key: Optional[str] = None,
        planner_model: Optional[str] = "gpt-5.5",
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
        tools: Optional[Dict[str, Callable]] = None,
        stores: Optional[Dict[str, Any]] = None,
        guards: Optional[Dict[str, Callable]] = None,
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
        self.tools = tools or {}
        self.stores = stores or {}
        self.guards = guards or {}

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

        # The planner fixes its own spec: a reply that is not JSON or breaks a rule goes back with the error.
        for attempt in range(1, PLANNER_ATTEMPTS + 1):
            raw = chatbot.chat(chat_input)[0]
            # Chatbot returns a string for normal responses; may return dict for tool calls.
            if isinstance(raw, dict):
                raise ValueError(f"Planner returned non-text response: {raw}")
            try:
                spec = self._extract_json_object(raw)
                self._validate_spec(spec, planning=True)
                return spec
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                # malformed shapes (a list where an object belongs, ...) are planner mistakes too
                if attempt == PLANNER_ATTEMPTS:
                    raise
                chat_input.add_assistant_message(str(raw))
                chat_input.add_user_message(
                    f"The FlowSpec is not valid: {e}\n"
                    "Fix it and reply with the complete corrected FlowSpec JSON object only."
                )

    def _build_system_prompt(self) -> str:
        prompt = PLANNER_PROMPT.replace("{preferences}", self._preferences_text())
        prompt = prompt.replace("{registered}", self._registered_text())
        context = self._load_context_text()
        if context:
            prompt += "\nReference files from the app (read-only):\n" + context
        return prompt

    def _preferences_text(self) -> str:
        p = self.preferences
        lines = [
            f"- Language steps: {p['text']}." if p["text"] else
            '- Language steps: provider "openai" (model "gpt-5.5"), "anthropic" ("claude-sonnet-5") or "gemini" '
            '("gemini-2.5-flash"), unless the user names another provider or model for a step.'
        ]
        for kind, label in (("image", "Image"), ("speech", "Speech"), ("recognition", "Recognition")):
            if p[kind]:
                lines.append(f"- {label} steps: {p[kind]}.")
        return "\n".join(lines)

    def _registered_text(self) -> str:
        def describe(registry, kind=None, none="none"):
            if not registry:
                return none
            items = []
            for name, value in registry.items():
                if kind == "store":
                    summary = self._store_kind(value)
                else:
                    summary = (inspect.getdoc(value) or "").strip().split("\n")[0]
                items.append(f"{name} ({summary})" if summary else name)
            return "; ".join(items)

        return "\n".join([
            f"- tools (assistant options.tools): {describe(self.tools, none='none, so no tools')}",
            "- stores (assistant options.knowledge, options.memory or options.history): "
            + describe(self.stores, "store", none="none, so use store configs"),
            f"- guards (computer model_params.on_action): {describe(self.guards, none='none')}",
            f"- processors (task post_process): {describe(self.processors, none='none, so no post_process')}",
        ])

    @staticmethod
    def _store_kind(value) -> str:
        """'vector store (...)' or 'chat history (...)' for a registered store object or config."""
        if isinstance(value, dict):
            kind = str(value.get("type", "")).lower()
            return f"chat history ({kind})" if kind in HISTORY_TYPES and kind not in STORE_TYPES else f"vector store ({kind})"
        if hasattr(value, "get_messages"):
            return f"chat history ({type(value).__name__})"
        return f"vector store ({type(value).__name__})"

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
    def _validate_spec(self, spec: Dict[str, Any], planning: bool = False) -> None:
        """Check a FlowSpec. planning=True (a new plan) also rejects post_process names that are not registered."""
        if not isinstance(spec, dict):
            raise ValueError("FlowSpec must be a JSON object")
        if "tasks" not in spec or not isinstance(spec["tasks"], list) or not spec["tasks"]:
            raise ValueError("FlowSpec.tasks must be a non-empty list")
        if "version" not in spec:
            raise ValueError("FlowSpec.version is required")
        if not all(isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"] for t in spec["tasks"]):
            raise ValueError("Each task must be an object with a name (a string)")

        # basic name uniqueness
        names = [t.get("name") for t in spec["tasks"]]
        if len(names) != len(set(names)):
            raise ValueError("Task names must be unique")

        task_names = set(names)
        agent_types = {t["name"]: (t.get("agent") or {}).get("agent_type", AgentTypes.TEXT.value)
                       for t in spec["tasks"]}

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
                if not isinstance(d, str) or d not in task_names:
                    raise ValueError(f"map_paths references unknown destination task '{d}'")

        for t in spec["tasks"]:
            name = t["name"]
            agent = t.get("agent") or {}
            agent_type = agent_types[name]
            if planning and not (isinstance(t.get("agent"), dict) and agent.get("agent_type")):
                raise ValueError(f"Task '{name}' needs an agent object with agent_type and provider")
            if planning and agent_type in (AgentTypes.TEXT.value, AgentTypes.ASSISTANT.value) \
                    and not agent.get("provider"):
                raise ValueError(f"Task '{name}' needs agent.provider")
            if agent_type not in AgentTypes._value2member_map_:
                raise ValueError(f"Task '{name}' has an unknown agent_type '{agent_type}'. "
                                 f"Use one of: {sorted(AgentTypes._value2member_map_)}")
            p = (agent.get("provider") or "").lower()

            # providers of language steps, and the connection details of local ones
            if agent_type in (AgentTypes.TEXT.value, AgentTypes.ASSISTANT.value):
                allowed = ASSISTANT_PROVIDERS if agent_type == AgentTypes.ASSISTANT.value else TEXT_AGENT_PROVIDERS
                if p and p not in allowed:
                    raise ValueError(f"Unsupported {agent_type} agent provider '{p}'. Allowed: {sorted(allowed)}")

                if p == "vllm":
                    opts = agent.get("options") or {}
                    if not isinstance(opts, dict):
                        raise ValueError(f"vllm {agent_type} agent requires agent.options to be an object")
                    base_url = opts.get("baseUrl") or opts.get("vllmBaseUrl")
                    if not base_url or not isinstance(base_url, str):
                        raise ValueError(f"vllm {agent_type} agent requires agent.options.baseUrl")
                    if not (base_url.startswith("${ENV:") or base_url.startswith("http://") or base_url.startswith("https://")):
                        raise ValueError("vllm baseUrl must be an http(s) URL or an ${ENV:...} placeholder")

                if p == "llamacpp":
                    opts = agent.get("options") or {}
                    if not isinstance(opts, dict):
                        raise ValueError(f"llamacpp {agent_type} agent requires agent.options to be an object")
                    model_path = opts.get("model_path") or opts.get("modelPath")
                    if not model_path or not isinstance(model_path, str):
                        raise ValueError(f"llamacpp {agent_type} agent requires agent.options.model_path")
                    if not (model_path.startswith("${ENV:") or model_path.strip()):
                        raise ValueError("llamacpp model_path must be a non-empty path or an ${ENV:...} placeholder")

            if agent_type == AgentTypes.ASSISTANT.value:
                self._validate_assistant(name, agent, p, planning)
            if agent_type == AgentTypes.COMPUTER.value:
                self._validate_computer(name, agent, p)

            # Sanity check for OpenAI image generation
            if agent_type == AgentTypes.IMAGE.value and agent.get("provider") == "openai":
                m_params = agent.get("model_params", {})
                if "response_format" not in m_params:
                    # Injecting default if missing to avoid corrupted outputs
                    m_params["response_format"] = "b64_json"
                    if "size" not in m_params:
                        m_params["size"] = "1024x1024"

            post_process = t.get("post_process")
            if planning and post_process and post_process not in self.processors:
                raise ValueError(f"Task '{name}' uses post_process '{post_process}', which is not registered. "
                                 f"Registered processors: {sorted(self.processors) or 'none'}")

        # validate dynamic connectors
        dyn = spec.get("dynamic_connectors", []) or []
        if not isinstance(dyn, list):
            raise ValueError("FlowSpec.dynamic_connectors must be a list")
        for c in dyn:
            if not isinstance(c, dict):
                raise ValueError("Each dynamic connector must be an object")
            source = c.get("source")
            if not isinstance(source, str) or source not in task_names:
                raise ValueError(f"dynamic_connectors references unknown source task '{source}'")

            kind = (c.get("kind") or "").lower()
            if kind not in SUPPORTED_CONNECTOR_KINDS:
                raise ValueError(f"Unsupported dynamic connector kind: {kind}")

            destinations = c.get("destinations") or {}
            if not isinstance(destinations, dict) or not destinations:
                raise ValueError(f"dynamic connector '{source}' must include non-empty destinations")
            for _, dest_task in destinations.items():
                if not isinstance(dest_task, str) or dest_task not in task_names:
                    raise ValueError(f"dynamic connector '{source}' points to unknown task '{dest_task}'")

            # a later task that waits for two branches of one connector never runs
            branches = set(destinations.values())
            for task_name in task_names:
                joined = sorted(src for src, dsts in map_paths.items() if task_name in dsts and src in branches)
                if len(joined) > 1:
                    raise ValueError(
                        f"Task '{task_name}' waits for {joined}, but the connector on '{source}' runs only one of "
                        "them. Give each branch its own next task, or route after the join instead.")

            # kind-specific rules
            if kind == "tool":
                if "tool_called" not in destinations or "no_tool" not in destinations:
                    raise ValueError("tool connector requires destinations: 'tool_called' and 'no_tool'")
                if agent_types.get(source) == AgentTypes.ASSISTANT.value:
                    raise ValueError(
                        f"A 'tool' connector routes on a model's tool call, but the assistant '{source}' runs its "
                        "tools itself. Give the assistant options.tools, or route on its text with a 'content' "
                        "connector.")

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
                # a keyword key without a destination stops there; a destination no keyword chooses never runs
                unreachable = [key for key in destinations if key not in keywords]
                if unreachable:
                    raise ValueError(f"content connector on '{source}': destinations {unreachable} can never be "
                                     "chosen, because keywords has no such key")

            if kind == "type":
                type_destinations = c.get("type_destinations")
                default_dest = c.get("default_dest")
                if type_destinations is not None and not isinstance(type_destinations, dict):
                    raise ValueError("type connector 'type_destinations' must be an object when provided")
                if default_dest is not None and not isinstance(default_dest, str):
                    raise ValueError("type connector 'default_dest' must be a string when provided")

        # checked last, when every name is known to be a task
        self._check_acyclic(map_paths, dyn)

    @staticmethod
    def _check_acyclic(map_paths: Dict[str, Any], connectors: List[Any]) -> None:
        """The graph of map_paths and connector routes must have no cycle."""
        edges: Dict[str, List[str]] = {}
        for src, dsts in map_paths.items():
            edges.setdefault(src, []).extend(dsts if isinstance(dsts, list) else [])
        for c in connectors:
            if isinstance(c, dict) and isinstance(c.get("destinations"), dict):
                edges.setdefault(c.get("source"), []).extend(c["destinations"].values())
        state: Dict[str, int] = {}  # 1 visiting, 2 done

        def visit(node, path):
            state[node] = 1
            for nxt in edges.get(node, []):
                if state.get(nxt) == 1:
                    cycle = path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt]
                    raise ValueError(f"The graph has a cycle: {' -> '.join(cycle)}. A flow must be a DAG.")
                if state.get(nxt) is None:
                    visit(nxt, path + [nxt])
            state[node] = 2

        for start in list(edges):
            if state.get(start) is None:
                visit(start, [start])

    def _validate_assistant(self, name: str, agent: Dict[str, Any], provider: str, planning: bool = False) -> None:
        """Registered names and store configs of an assistant step (they may sit in options or model_params)."""
        from intelli.flow.agents.handlers import AssistantAgentHandler

        settings = {**(agent.get("model_params") or {}), **(agent.get("options") or {})}
        tools = settings.get("tools")
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError(f"Task '{name}': options.tools must be a list of registered tool names")
            unknown = [tool for tool in tools if isinstance(tool, str) and tool not in self.tools]
            if unknown:
                raise ValueError(f"Task '{name}' uses tools {unknown}, which are not registered. "
                                 f"Registered tools: {sorted(self.tools) or 'none'}")
        for field_name in ("knowledge", "memory", "history"):
            self._validate_store(name, field_name, settings.get(field_name))
        if (settings.get("documents") or settings.get("files")) and not settings.get("knowledge"):
            raise ValueError(f"Task '{name}': documents and files need options.knowledge (a vector store)")
        # a new plan that keeps history or memory says whose: without the ids every run starts over
        if planning and settings.get("history") and not settings.get("conversation_id"):
            raise ValueError(f"Task '{name}': options.history continues a conversation only with "
                             'model_params "conversation_id" (e.g. "default")')
        if planning and settings.get("memory") and not settings.get("user_id"):
            raise ValueError(f"Task '{name}': options.memory remembers a person only with "
                             'model_params "user_id" (e.g. "default"; the app sets it per person)')
        if settings.get("google_search") and provider not in ("gemini", "vertex"):
            raise ValueError(f"Task '{name}': google_search needs provider 'gemini' or 'vertex', not '{provider}'")
        if provider not in AssistantAgentHandler.EMBEDDING_PROVIDERS:
            for field_name in ("knowledge", "memory"):
                config = settings.get(field_name)
                if isinstance(config, dict) and not config.get("embedder") and config.get("type") != "vertex_rag":
                    raise ValueError(
                        f"Task '{name}': a {provider} assistant has no embedding model for options.{field_name}; "
                        "add an embedder to the store config, e.g. "
                        '"embedder": {"provider": "openai", "api_key": "${ENV:OPENAI_API_KEY}"}')

    def _validate_store(self, name: str, field_name: str, value: Any) -> None:
        if value is None:
            return
        history = field_name == "history"
        types = HISTORY_TYPES if history else STORE_TYPES
        if isinstance(value, str):
            if value in self.stores:
                registered = self.stores[value]
                kind = self._store_kind(registered)
                if history != kind.startswith("chat history"):
                    raise ValueError(f"Task '{name}': options.{field_name} '{value}' is a {kind}")
                return
            if value.lower() in types:
                return
            raise ValueError(f"Task '{name}': options.{field_name} '{value}' is not a registered store "
                             f"(registered: {sorted(self.stores) or 'none'}). Use a config such as "
                             f'{{"type": "{"file" if history else "memory"}", ...}}.')
        if isinstance(value, dict):
            kind = str(value.get("type", "memory" if history else "")).lower()
            if kind not in types:
                raise ValueError(f"Task '{name}': options.{field_name} has the unknown type '{kind}'. "
                                 f"Use one of: {sorted(types)}")
            return
        if not isinstance(value, str) and (hasattr(value, "get_messages") or hasattr(value, "query")):
            return  # an object passed in a spec built in Python
        raise ValueError(f"Task '{name}': options.{field_name} must be a store name or a config object")

    def _validate_computer(self, name: str, agent: Dict[str, Any], provider: str) -> None:
        if provider not in COMPUTER_PROVIDERS:
            raise ValueError(f"Task '{name}': computer steps need provider 'anthropic' or 'openai', not '{provider}'")
        params = agent.get("model_params") or {}
        options = agent.get("options") or {}
        start_url = params.get("start_url")
        if not options.get("environment") and not (
                isinstance(start_url, str) and start_url.startswith(("http://", "https://", "file://", "${ENV:"))):
            raise ValueError(f"Task '{name}': computer steps need model_params.start_url "
                             "(an http(s) URL or an ${ENV:...} placeholder)")
        for hook in ("on_action", "on_safety_check"):
            value = params.get(hook, options.get(hook))
            if value is not None and not callable(value) and value not in self.guards:
                raise ValueError(f"Task '{name}': {hook} '{value}' is not a registered guard. "
                                 f"Registered guards: {sorted(self.guards) or 'none'}")

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
            # Registered names become the app's objects (tools, stores, guards)
            self._resolve_references(t.agent)

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

    def _resolve_references(self, agent: AgentSpec) -> None:
        """Replace registered names in an agent's options and model_params with the app's objects."""
        for settings in (agent.options, agent.model_params):
            if agent.agent_type == AgentTypes.ASSISTANT.value:
                if isinstance(settings.get("tools"), list):
                    settings["tools"] = [self.tools[tool] if isinstance(tool, str) else tool
                                         for tool in settings["tools"]]
                for field_name in ("knowledge", "memory", "history"):
                    value = settings.get(field_name)
                    if isinstance(value, str):
                        # a registered store, or a bare store type such as "memory"
                        settings[field_name] = self.stores[value] if value in self.stores else {"type": value}
            if agent.agent_type == AgentTypes.COMPUTER.value:
                for hook in ("on_action", "on_safety_check"):
                    if isinstance(settings.get(hook), str):
                        settings[hook] = self.guards[settings[hook]]

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
        # The planner prompt is a self-contained reference; context_files can add the app's own documents.
        return []

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
            "access_token",
            "vertex_api_key",
            "secret_access_key",
            "aws_secret_access_key",
            "session_token",
            "aws_session_token",
            "token",
            "password",
            "connection_string",
        }
        # user:password@ inside URLs and connection strings (MongoDB, PostgreSQL, ...)
        url_credentials = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@")

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
            if isinstance(obj, str):
                return url_credentials.sub(r"\1<REDACTED>@", obj)
            return obj

        return _walk(spec)


class VibeAgent(VibeFlow):
    """
    VibeAgent is an alias for VibeFlow to allow the user to call it by either name.
    """
    pass
