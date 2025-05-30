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

        f_params = {
            key: value
            for key, value in custom_params.items()
            if hasattr(ChatModelInput("test", None), key)
        }

        chat_input = ChatModelInput(self.mission, **f_params)
        chatbot = Chatbot(custom_params["key"], self.provider, self.options)
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

        image_model = RemoteImageModel(custom_params["key"], self.provider)
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
        )

        vision_model = RemoteVisionModel(custom_params["key"], self.provider)
        result = vision_model.image_to_text(vision_input)
        return result


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
        if not api_key:
            raise ValueError(f"API key is required for {self.provider} speech synthesis")

        speech_model = RemoteSpeechModel(key_value=api_key, provider=self.provider.lower())

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
            api_key=custom_params["key"],
            provider_name=self.provider,
            options=self.options,
        )

        result = embed_model.get_embeddings(embed_input)
        return result


class SearchAgentHandler(AgentHandler):
    """Handler for search agents"""

    def execute(self, agent_input, custom_params):
        from intelli.wrappers.intellicloud_wrapper import IntellicloudWrapper

        if "one_key" not in custom_params:
            raise ValueError("SearchAgent requires 'one_key' in model_params")

        # Create IntelliCloud wrapper
        wrapper = IntellicloudWrapper(
            api_key=custom_params["one_key"], api_base=custom_params.get("api_base")
        )

        # Prepare filters
        filters = {}
        if "document_name" in custom_params:
            filters["document_name"] = custom_params["document_name"]

        # Perform semantic search
        k = custom_params.get("k", 3)
        result = wrapper.semantic_search(
            query_text=agent_input.desc, k=k, filters=filters
        )

        return result


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
            
            # Execute the tool
            result = wrapper.execute_tool(tool_name, arguments)
            
            # Handle result conversion
            if hasattr(result, "content") and isinstance(result.content, list):
                text_content = next((item.text for item in result.content if hasattr(item, 'text')), None)
                if text_content:
                    return text_content
            
            return str(result)
        except Exception as e:
            return f"Error executing MCP agent: {str(e)}"
            
    def _create_server_config(self, params):
        """Create server configuration from parameters"""
        # Check for URL-based configuration
        if "url" in params:
            return {"url": params["url"]}
            
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
    }

    if agent_type not in handlers:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    return handlers[agent_type](provider, mission, model_params, options)
