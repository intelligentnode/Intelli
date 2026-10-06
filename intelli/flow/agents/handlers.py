from abc import ABC, abstractmethod
from intelli.flow.types import AgentTypes


class AgentHandler(ABC):
    """Base class for agent type-specific handlers"""

    def __init__(self, provider, mission, model_params, options):
        self.provider = provider
        self.mission = mission
        self.model_params = model_params
        self.options = options

    @abstractmethod
    def execute(self, agent_input, custom_params):
        """Execute the agent functionality"""
        pass


class TextAgentHandler(AgentHandler):
    """Handler for text-based agents"""

    def execute(self, agent_input, custom_params):
        from intelli.function.chatbot import Chatbot
        from intelli.model.input.chatbot_input import ChatModelInput

        # Backwards-compatible improvement:
        # Preserve existing behavior for standard params, but also allow passing
        # newer provider-specific fields via ChatModelInput(**options).
        # Exclude the API key so it never becomes part of the model request payload.
        f_params = {k: v for k, v in custom_params.items() if k != "key"}

        chat_input = ChatModelInput(self.mission, **f_params)
        api_key = custom_params.get("key")
        provider_lower = (self.provider or "").lower()
        # AWS can authenticate with IAM credentials from options or the AWS credential chain.
        if not api_key and provider_lower not in {"vllm", "llamacpp", "keras", "aws"}:
            raise ValueError(f"API key is required for {self.provider} text generation")

        chatbot = Chatbot(api_key, self.provider, self.options)
        chat_input.add_user_message(agent_input.desc)
        result = chatbot.chat(chat_input)[0]
        return result


class ImageAgentHandler(AgentHandler):
    """Handler for image generation agents"""

    def execute(self, agent_input, custom_params):
        from intelli.controller.remote_image_model import RemoteImageModel
        from intelli.model.input.image_input import ImageModelInput

        f_params = {
            key: value
            for key, value in custom_params.items()
            if hasattr(ImageModelInput("test"), key)
        }

        image_input = ImageModelInput(
            prompt=self.mission + ": " + agent_input.desc, **f_params
        )

        image_model = RemoteImageModel(custom_params.get("key"), self.provider, options=self.options)
        result = image_model.generate_images(image_input)[0]
        return result


class VisionAgentHandler(AgentHandler):
    """Handler for vision-based agents"""

    def execute(self, agent_input, custom_params):
        from intelli.controller.remote_vision_model import RemoteVisionModel
        from intelli.model.input.vision_input import VisionModelInput

        vision_input = VisionModelInput(
            content=self.mission + ": " + agent_input.desc,
            image_data=agent_input.img,
            extension=custom_params.get("extension", "png"),
            model=custom_params["model"],
            max_tokens=custom_params.get("max_tokens"),
        )

        vision_model = RemoteVisionModel(custom_params.get("key"), self.provider, options=self.options)
        result = vision_model.image_to_text(vision_input)
        return result


# OpenAI speech reads at most 4096 characters per request
OPENAI_SPEECH_LIMIT = 4096


def split_for_speech(text, limit=OPENAI_SPEECH_LIMIT):
    """Split text into pieces of at most `limit` characters, at sentence ends where possible."""
    import re

    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sentence) > limit:
            # a sentence longer than the limit: cut it at a space
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece]


class SpeechAgentHandler(AgentHandler):
    """Handler for speech synthesis agents"""

    def execute(self, agent_input, custom_params):
        from intelli.controller.remote_speech_model import RemoteSpeechModel
        from intelli.model.input.text_speech_input import Text2SpeechInput

        # Get text content
        text_content = agent_input.desc
        if self.mission and not text_content.startswith(self.mission):
            text_content = f"{self.mission}: {text_content}"

        # Create the text-to-speech input
        speech_input = Text2SpeechInput(
            text=text_content,
            language=custom_params.get("language", "en"),
            gender=custom_params.get("gender", "FEMALE"),
            voice=custom_params.get("voice", "alloy"),
            model=custom_params.get("model", "tts-1"),
            stream=custom_params.get("stream", True),
        )

        # Provider-specific parameters
        if self.provider.lower() == "openai":
            if "voice" in custom_params:
                speech_input.voice = custom_params["voice"]
            if "model" in custom_params:
                speech_input.model = custom_params["model"]
        elif self.provider.lower() == "elevenlabs":
            # Voice ID handling for ElevenLabs
            if "voice" in custom_params:
                speech_input.voice_id = custom_params["voice"]
            if "model" in custom_params:
                speech_input.model_id = custom_params["model"]

        # Create speech model
        api_key = custom_params.get("key")
        if not api_key and self.provider.lower() != "aws":
            raise ValueError(f"API key is required for {self.provider} speech synthesis")

        speech_model = RemoteSpeechModel(key_value=api_key, provider=self.provider.lower(), options=self.options)

        # OpenAI reads at most 4096 characters: longer text is spoken in pieces and the audio joined
        if self.provider.lower() == "openai" and not speech_input.stream and len(text_content) > OPENAI_SPEECH_LIMIT:
            audio = b""
            for piece in split_for_speech(text_content):
                speech_input.text = piece
                audio += speech_model.generate_speech(speech_input)
            return audio

        # Generate speech
        result = speech_model.generate_speech(speech_input)
        return result


class RecognitionAgentHandler(AgentHandler):
    """Handler for speech recognition agents"""

    def execute(self, agent_input, custom_params):
        from intelli.controller.remote_recognition_model import RemoteRecognitionModel, SupportedRecognitionModels
        from intelli.model.input.text_recognition_input import SpeechRecognitionInput
        import os

        # Determine audio source
        file_path = None
        audio_data = None

        # Handle different input types
        if hasattr(agent_input, "audio") and agent_input.audio:
            audio_data = agent_input.audio
            print(
                f"Found audio data in agent_input.audio: {type(audio_data)}, size: {len(audio_data) if isinstance(audio_data, (bytes, bytearray)) else 'unknown'}")
        elif isinstance(agent_input, (bytes, bytearray)):
            audio_data = agent_input
            print(f"Received direct bytes data for recognition, size: {len(audio_data)}")
        elif isinstance(agent_input, str):
            if agent_input.startswith("file:"):
                file_path = agent_input[5:]  # Remove 'file:' prefix
            elif os.path.exists(agent_input):
                file_path = agent_input
            print(f"Using file path for recognition: {file_path}")
        else:
            print(f"Warning: Unrecognized agent_input type for recognition: {type(agent_input)}")

        # Create recognition input with available data
        recognition_input = SpeechRecognitionInput(
            audio_file_path=file_path,
            audio_data=audio_data,
            language=custom_params.get("language", "en"),
            model=custom_params.get("model", "whisper-1")
        )

        # Add provider-specific parameters
        if self.provider.lower() == "keras":
            recognition_input.user_prompt = custom_params.get("user_prompt", "")
            recognition_input.condition_on_previous_text = custom_params.get("condition_on_previous_text", True)
            recognition_input.max_steps = custom_params.get("max_steps", 80)
            recognition_input.max_chunk_sec = custom_params.get("max_chunk_sec", 30)
        elif self.provider.lower() == "elevenlabs" and "model" in custom_params:
            recognition_input.model_id = custom_params["model"]

        # Determine provider
        provider_enum = None
        provider_lower = self.provider.lower()
        if provider_lower == "openai":
            provider_enum = SupportedRecognitionModels["OPENAI"]
        elif provider_lower == "keras":
            provider_enum = SupportedRecognitionModels["KERAS"]
        elif provider_lower == "elevenlabs":
            provider_enum = SupportedRecognitionModels["ELEVENLABS"]
        else:
            provider_enum = self.provider

        # Create recognition model
        if provider_lower == "keras":
            # Keras doesn't need an API key
            model_name = custom_params.get("model_name", "whisper_tiny_en")
            print(f"Creating Keras recognition model with model_name: {model_name}")
            recognition_model = RemoteRecognitionModel(
                provider=provider_enum,
                model_name=model_name,
                model_params=custom_params,
            )
        else:
            # Remote services need API key
            print(f"Creating {self.provider} recognition model with model: {custom_params.get('model', 'default')}")
            recognition_model = RemoteRecognitionModel(
                key_value=custom_params["key"],
                provider=provider_enum,
                model_name=custom_params.get("model"),
            )

        # Recognize speech
        try:
            result = recognition_model.recognize_speech(recognition_input)
            print(f"Recognition successful, result: '{result[:50]}...' (truncated)")
            return result
        except Exception as e:
            print(f"Error in recognition: {e}")
            import traceback
            traceback.print_exc()
            return f"Error during speech recognition: {str(e)}"


class EmbedAgentHandler(AgentHandler):
    """Handler for embedding agents"""

    def execute(self, agent_input, custom_params):
        from intelli.controller.remote_embed_model import RemoteEmbedModel
        from intelli.model.input.embed_input import EmbedInput

        text_input = agent_input.desc
        if self.mission and not text_input.startswith(self.mission):
            text_input = f"{self.mission}: {text_input}"

        embed_input = EmbedInput(
            texts=[text_input],
            model=custom_params.get("model"),
        )

        # Try to set default values for the provider
        try:
            embed_input.set_default_values(self.provider)
        except ValueError:
            # If no default is available for this provider, continue
            pass

        # Create embed model
        embed_model = RemoteEmbedModel(
            api_key=custom_params.get("key"),
            provider_name=self.provider,
            options=self.options,
        )

        result = embed_model.get_embeddings(embed_input)
        return result


class SearchAgentHandler(AgentHandler):
    """Handler for search agents"""

    def execute(self, agent_input, custom_params):
        # ------------------------------------------------------------
        # Provider 1 (existing): Intellicloud semantic search
        # ------------------------------------------------------------
        if "one_key" in custom_params:
            from intelli.wrappers.intellicloud_wrapper import IntellicloudWrapper

            wrapper = IntellicloudWrapper(
                api_key=custom_params["one_key"], api_base=custom_params.get("api_base")
            )

            filters = {}
            if "document_name" in custom_params:
                filters["document_name"] = custom_params["document_name"]

            k = custom_params.get("k", 3)
            return wrapper.semantic_search(
                query_text=agent_input.desc, k=k, filters=filters
            )

        # ------------------------------------------------------------
        # Provider 2 (new): Google Custom Search JSON API (web search)
        # ------------------------------------------------------------
        if custom_params.get("google_api_key") and custom_params.get("google_cse_id"):
            from intelli.wrappers.google_search_wrapper import GoogleCustomSearchWrapper

            wrapper = GoogleCustomSearchWrapper(
                api_key=custom_params["google_api_key"],
                cse_id=custom_params["google_cse_id"],
            )

            k = custom_params.get("k", 5)
            safe = custom_params.get("safe", "active")
            timeout = float(custom_params.get("timeout", 20.0))
            as_text = bool(custom_params.get("as_text", True))

            results = wrapper.search(
                agent_input.desc, num=k, safe=safe, timeout=timeout
            )
            return (
                GoogleCustomSearchWrapper.to_text(results)
                if as_text
                else results
            )

        # ------------------------------------------------------------
        # Provider 3: Amazon Bedrock Knowledge Base retrieval
        # ------------------------------------------------------------
        if custom_params.get("knowledge_base_id"):
            from intelli.wrappers.aws_wrapper import AWSWrapper

            # Knowledge Bases need IAM credentials (options or the AWS credential chain).
            wrapper = AWSWrapper.from_options(custom_params.get("key"), self.options)
            response = wrapper.retrieve(
                custom_params["knowledge_base_id"], agent_input.desc,
                number_of_results=custom_params.get("k", 5),
            )
            as_text = bool(custom_params.get("as_text", True))
            return AWSWrapper.retrieval_to_text(response) if as_text else response

        raise ValueError(
            "SearchAgent missing credentials. Provide either:\n"
            "- 'one_key' (Intellicloud semantic search)\n"
            "- OR ('google_api_key' and 'google_cse_id') for Google web search\n"
            "- OR 'knowledge_base_id' for an Amazon Bedrock Knowledge Base"
        )


class MCPAgentHandler(AgentHandler):
    """Handler for MCP-based agents"""
    
    def execute(self, agent_input, custom_params):
        try:
            from intelli.wrappers.mcp_wrapper import MCPWrapper
        except ImportError as e:
            return (
                "Error: MCP agent requires the 'mcp' module. "
                "Install it using 'pip install intelli[mcp]'. "
                f"Original error: {e}"
            )
        
        try:
            # Create server configuration from parameters
            server_config = self._create_server_config(custom_params)
            
            # Create wrapper with server details
            wrapper = MCPWrapper(server_config)
            
            # Get tool name and arguments
            tool_name, arguments = self._prepare_tool_arguments(agent_input, custom_params)
            
            # Debug info
            print(f"MCP Agent executing tool '{tool_name}' with arguments: {arguments}")
            
            # Execute the tool and normalize the result (captures isError + structured content)
            result = wrapper.execute_tool(tool_name, arguments)
            normalized = wrapper.normalize_tool_result(result)

            if normalized.get("is_error"):
                return f"Error from MCP tool '{tool_name}': {normalized.get('text') or result}"
            if normalized.get("text"):
                return normalized["text"]
            if normalized.get("structured") is not None:
                return normalized["structured"]

            return str(result)
        except Exception as e:
            return f"Error executing MCP agent: {str(e)}"

    def _create_server_config(self, params):
        """Create server configuration from parameters"""
        # Check for URL-based configuration (remote http/sse/websocket).
        if "url" in params:
            cfg = {"url": params["url"]}
            # Forward optional remote options (auth headers, transport, timeout).
            for key in ("headers", "transport", "timeout"):
                if params.get(key) is not None:
                    cfg[key] = params[key]
            return cfg

        # Check for subprocess-based configuration
        if "command" in params:
            return {
                "command": params["command"],
                "args": params.get("args", []),
                "env": params.get("env")
            }

        raise ValueError("MCPAgent requires either 'url' or 'command' in model_params")
    
    def _prepare_tool_arguments(self, agent_input, params):
        """Extract tool name and prepare arguments dictionary"""
        # Get tool name
        tool_name = params.get("tool", "")
        if not tool_name:
            raise ValueError("MCPAgent requires 'tool' name in model_params")
        
        # Build arguments dictionary
        arguments = {}
        
        # Look for arg_* prefixed parameters first
        for k, v in params.items():
            if k.startswith("arg_"):
                arg_name = k[4:]  # Strip prefix
                arguments[arg_name] = v
        
        # Fall back to input_arg if specified and no arguments found
        if not arguments and "input_arg" in params:
            input_arg = params["input_arg"]
            arguments[input_arg] = agent_input.desc
        
        return tool_name, arguments


class CoderAgentHandler(AgentHandler):
    """Handler for autonomous coding agents (agent_type='coder').

    model_params: key, model, workspace (required), test_command, max_iterations,
    allow_bash, bash_timeout. The task input text is the coding task.
    """

    def execute(self, agent_input, custom_params):
        from intelli.function.coding_agent import CodingAgent

        workspace = custom_params.get("workspace")
        if not workspace:
            raise ValueError("CoderAgent requires 'workspace' in model_params")

        task = agent_input.desc
        if self.mission and self.mission not in task:
            task = f"{self.mission}: {task}"

        agent = CodingAgent(
            api_key=custom_params.get("key"),
            provider=self.provider,
            model=custom_params.get("model"),
            workspace=workspace,
            options=self.options,
            allow_bash=custom_params.get("allow_bash", True),
            bash_timeout=custom_params.get("bash_timeout", 120),
            max_iterations=custom_params.get("max_iterations", 20),
            log=custom_params.get("log", False),
        )
        result = agent.run(task, test_command=custom_params.get("test_command"))

        # Return a text summary so downstream flow tasks can consume it.
        status = "succeeded" if result.get("success") else "did not fully succeed"
        return f"Coding task {status} after {result.get('iterations')} iteration(s). {result.get('summary', '')}"


class ComputerAgentHandler(AgentHandler):
    """Handler for computer-use agents (agent_type='computer').

    model_params: key, model, max_iterations, start_url (browser env) or an
    'environment' instance passed via options. The task input text is the goal.
    """

    def execute(self, agent_input, custom_params):
        from intelli.function.computer_agent import ComputerAgent

        task = agent_input.desc
        if self.mission and self.mission not in task:
            task = f"{self.mission}: {task}"

        # Environment: explicit instance wins; otherwise a Playwright browser.
        environment = (self.options or {}).get("environment") or custom_params.get("environment")
        owns_environment = False
        if environment is None:
            from intelli.function.browser_env import PlaywrightBrowserEnvironment
            environment = PlaywrightBrowserEnvironment(
                start_url=custom_params.get("start_url", "about:blank"),
                headless=custom_params.get("headless", True),
            )
            owns_environment = True

        opts = self.options or {}
        agent = ComputerAgent(
            api_key=custom_params.get("key"),
            provider=self.provider,
            model=custom_params.get("model"),
            environment=environment,
            max_iterations=custom_params.get("max_iterations", 25),
            # Forward the human-in-the-loop hooks; defaults stay safe (no
            # auto-acknowledgement of provider safety checks) when unset.
            on_action=custom_params.get("on_action") or opts.get("on_action"),
            on_safety_check=custom_params.get("on_safety_check") or opts.get("on_safety_check"),
            log=custom_params.get("log", False),
        )
        try:
            result = agent.run(task)
        finally:
            if owns_environment:
                environment.close()
        return result.get("output", "")


class AssistantAgentHandler(AgentHandler):
    """Handler for assistant agents (agent_type='assistant'): a text agent that also answers from documents (RAG
    over a vector store), keeps conversation history and long-term memory, and runs its own tools
    (intelli.function.assistant.Assistant).

    model_params: key, model, temperature, max_tokens, top_k, memory_top_k, min_score, max_history,
    max_tool_steps, google_search, auto_title, show_sources (append the cited sources to the answer), and per
    turn: conversation_id, user_id, filter, attachments. Other params go to the model request, as on text agents.
    options: the provider settings (baseUrl, region, project_id, ...) plus
        knowledge / memory: a VectorStore, or a store config ({'type': 'qdrant', 'url': ...}, see
            intelli.store.factory); memory makes the assistant remember earlier conversations of a user_id.
        history: a ChatHistory or a config ({'type': 'file', 'dir': './conversations'}).
        documents: texts or {'id', 'text', 'metadata'} for the knowledge store; files: text file paths. Both are
            added the first time the step runs, unless the store already holds records.
        tools: functions the model can call (callables or tool dicts).
        embedder: the embedder of config stores (default: the agent's provider and key, when it has embeddings).

    The task input (the run's input or the previous steps' output) is the user message, used for retrieval,
    history and memory; the task description joins the mission as the instruction. Without a conversation_id each
    run is a new conversation, removed from the default in-memory history after the answer.
    """

    BUILD_PARAMS = ("model", "temperature", "max_tokens", "top_k", "memory_top_k", "min_score", "max_history",
                    "max_tool_steps", "google_search", "auto_title")
    TURN_PARAMS = ("conversation_id", "user_id", "filter", "attachments", "show_sources")
    SETTINGS = ("knowledge", "memory", "history", "documents", "files", "tools", "embedder", "chunk_size",
                "chunk_overlap")
    EMBEDDING_PROVIDERS = {"openai", "gemini", "vertex", "mistral", "nvidia", "aws", "ollama"}

    def __init__(self, provider, mission, model_params, options):
        super().__init__(provider, mission, model_params, options)
        import threading
        self._lock = threading.Lock()
        self._assistants = {}
        self._resources = None
        self.last_reply = None

    def execute(self, agent_input, custom_params):
        from intelli.function.assistant import DEFAULT_SYSTEM

        params = dict(custom_params or {})
        assistant = self._assistant(params)
        mission = self.mission or DEFAULT_SYSTEM
        message = getattr(agent_input, "message", None)
        instruction = getattr(agent_input, "instruction", None)
        if message is None:
            message, system_message = agent_input.desc, mission
        else:
            system_message = f"{mission}\n\nCurrent task: {instruction}" if instruction else mission

        conversation_id = params.get("conversation_id")
        reply = assistant.chat(message, conversation_id=conversation_id, user_id=params.get("user_id"),
                               attachments=params.get("attachments"), filter=params.get("filter"),
                               system_message=system_message)
        self.last_reply = reply
        if not conversation_id and self._resources["temporary_history"]:
            assistant.history.delete_conversation(reply["conversation_id"])

        text = reply["text"]
        cited = [reference for reference in reply.get("references") or [] if reference.get("cited")]
        if params.get("show_sources") and (cited or reply.get("citations")):
            lines = [f"[{reference['index']}] {assistant._source_label(reference)}" for reference in cited]
            lines += [f"- {item.get('title') or item.get('uri')} ({item.get('uri')})"
                      for item in reply.get("citations") or [] if isinstance(item, dict)]
            text = f"{text}\n\nSources:\n" + "\n".join(lines)
        return text

    def _settings(self, params):
        """Store settings from options, or from model_params when a spec put them there."""
        options = self.options or {}
        merged = {key: params[key] for key in self.SETTINGS if key in params}
        merged.update({key: options[key] for key in self.SETTINGS if key in options})
        return merged

    def _assistant(self, params):
        """One Assistant per set of build params (tasks may override them), sharing the stores and history."""
        import json
        from intelli.function.assistant import Assistant, DEFAULT_SYSTEM

        with self._lock:
            if self._resources is None:
                self._resources = self._create_resources(params)
            input_options = self._input_options(params)
            key = json.dumps({"key": params.get("key"), "input": input_options,
                              **{k: params.get(k) for k in self.BUILD_PARAMS}}, sort_keys=True, default=str)
            if key not in self._assistants:
                resources = self._resources
                assistant = Assistant(
                    provider=self.provider, api_key=params.get("key"),
                    options={k: v for k, v in (self.options or {}).items() if k not in self.SETTINGS},
                    system_message=self.mission or DEFAULT_SYSTEM, history=resources["history"],
                    knowledge=resources["knowledge"], memory=resources["memory"], tools=resources["tools"],
                    input_options=input_options,
                    **{k: params[k] for k in self.BUILD_PARAMS if params.get(k) is not None})
                self._load_documents(assistant, resources)
                self._assistants[key] = assistant
            return self._assistants[key]

    def _input_options(self, params):
        skip = {"key", *self.BUILD_PARAMS, *self.TURN_PARAMS, *self.SETTINGS, "stream"}
        return {k: v for k, v in params.items() if k not in skip}

    def _create_resources(self, params):
        from intelli.store.factory import create_vector_store, create_chat_history

        settings = self._settings(params)
        embedder = settings.get("embedder") or self._default_embedder(params)
        history = create_chat_history(settings.get("history"))
        tools = settings.get("tools")
        names = [tool for tool in tools or [] if isinstance(tool, str)] if isinstance(tools, list) else []
        if names:
            raise ValueError(f"Assistant tools must be functions or tool dicts, not names: {names}. "
                             "VibeFlow resolves names from its tools registry.")
        return {
            "knowledge": self._store(settings.get("knowledge"), embedder, create_vector_store),
            "memory": self._store(settings.get("memory"), embedder, create_vector_store),
            "history": history,
            "temporary_history": history is None,
            "tools": tools,
            "settings": settings,
            "loaded": False,
        }

    def _store(self, config, embedder, create_vector_store):
        if config is None:
            return None
        if isinstance(config, dict) and not config.get("embedder") and embedder is None \
                and str(config.get("type", "")).lower() != "vertex_rag":
            raise ValueError(
                f"The {self.provider} assistant has no embedding model for its vector store: set 'embedder' in "
                "the store config, e.g. {'provider': 'openai', 'api_key': '${ENV:OPENAI_API_KEY}'}.")
        return create_vector_store(config, default_embedder=embedder)

    def _default_embedder(self, params):
        """The agent's own provider embeds config stores when it has an embedding model."""
        provider = (self.provider or "").lower()
        options = {k: v for k, v in (self.options or {}).items() if k not in self.SETTINGS}
        key = params.get("key")
        if provider == "gemini" and options.get("vertex"):
            provider = "vertex"
        if provider == "aws":
            return {"provider": "aws", "api_key": key, "options": options}
        if provider == "vertex":
            google = ("project_id", "location", "access_token", "credentials")
            return {"provider": "vertex", "api_key": key, "options": {k: options[k] for k in google if k in options}}
        if provider == "ollama":
            base_url = options.get("baseUrl")
            return {"provider": "ollama", "options": {"base_url": base_url.rstrip("/") + "/v1"} if base_url else {}}
        if provider in self.EMBEDDING_PROVIDERS and key:
            return {"provider": provider, "api_key": key}
        return None

    def _load_documents(self, assistant, resources):
        """Add the configured documents and files once, unless the knowledge store already holds records."""
        settings = resources["settings"]
        documents, files = settings.get("documents"), settings.get("files")
        if resources["loaded"] or not (documents or files):
            return
        if not assistant.knowledge:
            raise ValueError("Assistant documents and files need a knowledge store (options.knowledge).")
        count = getattr(assistant.knowledge, "count", None)
        if not (callable(count) and count() > 0):
            sizes = {k: settings[k] for k in ("chunk_size", "chunk_overlap") if settings.get(k)}
            if documents:
                assistant.add_documents([documents] if isinstance(documents, (str, dict)) else documents, **sizes)
            if files:
                assistant.add_files(files, **sizes)
        # marked only after success, so a failed load (a network error) is tried again on the next run
        resources["loaded"] = True


# Factory to get the appropriate handler
def get_agent_handler(agent_type, provider, mission, model_params, options):
    """Factory function to get the appropriate agent handler"""
    handlers = {
        AgentTypes.TEXT.value: TextAgentHandler,
        AgentTypes.IMAGE.value: ImageAgentHandler,
        AgentTypes.VISION.value: VisionAgentHandler,
        AgentTypes.SPEECH.value: SpeechAgentHandler,
        AgentTypes.RECOGNITION.value: RecognitionAgentHandler,
        AgentTypes.EMBED.value: EmbedAgentHandler,
        AgentTypes.SEARCH.value: SearchAgentHandler,
        AgentTypes.MCP.value: MCPAgentHandler,
        AgentTypes.CODER.value: CoderAgentHandler,
        AgentTypes.COMPUTER.value: ComputerAgentHandler,
        AgentTypes.ASSISTANT.value: AssistantAgentHandler,
    }

    if agent_type not in handlers:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    return handlers[agent_type](provider, mission, model_params, options)
