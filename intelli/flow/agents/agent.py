from abc import ABC, abstractmethod

from intelli.controller.remote_embed_model import RemoteEmbedModel
from intelli.controller.remote_image_model import RemoteImageModel
from intelli.controller.remote_vision_model import RemoteVisionModel
from intelli.controller.remote_speech_model import (
    RemoteSpeechModel,
    SupportedSpeechModels,
)
from intelli.controller.remote_recognition_model import (
    RemoteRecognitionModel,
    SupportedRecognitionModels,
)
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput
from intelli.flow.types import AgentTypes
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.model.input.image_input import ImageModelInput
from intelli.model.input.vision_input import VisionModelInput
from intelli.model.input.text_speech_input import Text2SpeechInput
from intelli.model.input.text_recognition_input import SpeechRecognitionInput
from intelli.model.input.embed_input import EmbedInput
from intelli.wrappers.intellicloud_wrapper import IntellicloudWrapper


class BasicAgent(ABC):

    @abstractmethod
    def execute(self, agent_input):
        pass


class Agent(BasicAgent):
    def __init__(self, agent_type, provider, mission, model_params, options=None):

        if agent_type not in AgentTypes._value2member_map_:
            raise ValueError("Incorrect agent type. Accepted types in AgentTypes.")

        self.type = agent_type
        self.provider = provider
        self.mission = mission
        self.model_params = model_params
        self.options = options

    def execute(self, agent_input: AgentInput, new_params={}):

        custom_params = dict(self.model_params)
        if (
            new_params is not None
            and isinstance(new_params, dict)
            and new_params
            and self.model_params is not None
        ):
            custom_params.update(new_params)

        # Check the agent type and call the appropriate function
        if self.type == AgentTypes.TEXT.value:
            return self._execute_text_agent(agent_input, custom_params)
        elif self.type == AgentTypes.IMAGE.value:
            return self._execute_image_agent(agent_input, custom_params)
        elif self.type == AgentTypes.VISION.value:
            return self._execute_vision_agent(agent_input, custom_params)
        elif self.type == AgentTypes.SPEECH.value:
            return self._execute_speech_agent(agent_input, custom_params)
        elif self.type == AgentTypes.RECOGNITION.value:
            return self._execute_recognition_agent(agent_input, custom_params)
        elif self.type == AgentTypes.EMBED.value:
            return self._execute_embed_agent(agent_input, custom_params)
        elif self.type == AgentTypes.SEARCH.value:
            return self._execute_search_agent(agent_input, custom_params)
        else:
            raise ValueError(f"Unsupported agent type: {self.type}.")

    def _execute_text_agent(self, agent_input, custom_params):
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

    def _execute_image_agent(self, agent_input, custom_params):
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

    def _execute_vision_agent(self, agent_input, custom_params):
        vision_input = VisionModelInput(
            content=self.mission + ": " + agent_input.desc,
            image_data=agent_input.img,
            extension=custom_params.get("extension", "png"),
            model=custom_params["model"],
        )

        vision_model = RemoteVisionModel(custom_params["key"], self.provider)
        result = vision_model.image_to_text(vision_input)
        return result

    def _execute_speech_agent(self, agent_input, custom_params):
        """
        Execute the speech agent to convert text to speech.

        Handles OpenAI, Google, and ElevenLabs speech services with proper
        parameter handling for each provider.
        """
        # Get text content
        text_content = agent_input.desc
        if self.mission and not text_content.startswith(self.mission):
            text_content = f"{self.mission}: {text_content}"

        # Create the text-to-speech input
        speech_input = Text2SpeechInput(
            text=text_content,
            language=custom_params.get('language', 'en'),
            gender=custom_params.get('gender', 'FEMALE')
        )

        # Get the API key from custom params
        api_key = custom_params.get('key')
        if not api_key:
            raise ValueError(f"API key is required for {self.provider} speech synthesis")

        # Handle provider-specific parameters
        if self.provider.lower() == 'openai':
            # For OpenAI, set voice and model
            if 'voice' in custom_params:
                speech_input.voice = custom_params['voice']
            if 'model' in custom_params:
                speech_input.model = custom_params['model']

            # Create speech model
            speech_model = RemoteSpeechModel(key_value=api_key, provider='openai')

        elif self.provider.lower() == 'google':
            # For Google, just need language which is already set
            speech_model = RemoteSpeechModel(key_value=api_key, provider='google')

        elif self.provider.lower() == 'elevenlabs':
            # For ElevenLabs, we need to handle the voice_id

            # Create speech model
            speech_model = RemoteSpeechModel(key_value=api_key, provider='elevenlabs')

            # If 'voice' param is provided, use it to look up the voice_id
            if 'voice' in custom_params:
                try:
                    # List available voices
                    voices_result = speech_model.list_voices()

                    if 'voices' in voices_result and len(voices_result['voices']) > 0:
                        voice_param = custom_params['voice']

                        # First, check if voice param is already a valid voice_id
                        voice_id = None
                        for voice in voices_result['voices']:
                            if voice.get('voice_id') == voice_param:
                                voice_id = voice_param
                                break

                        # If not found as ID, try to match by name
                        if not voice_id:
                            for voice in voices_result['voices']:
                                if voice.get('name', '').lower() == voice_param.lower():
                                    voice_id = voice.get('voice_id')
                                    break

                        # If still not found, use the first available voice
                        if not voice_id:
                            voice_id = voices_result['voices'][0]['voice_id']
                            print(
                                f"Voice '{voice_param}' not found, using '{voices_result['voices'][0]['name']}' instead")

                        # Set the voice_id on the input params
                        speech_input.voice_id = voice_id

                except Exception as e:
                    print(f"Warning: Error getting ElevenLabs voices: {e}")

            # Set model_id if provided
            if 'model' in custom_params:
                speech_input.model_id = custom_params['model']

        else:
            # For any other provider, just pass the provider as-is
            speech_model = RemoteSpeechModel(key_value=api_key, provider=self.provider)

        # Generate speech
        result = speech_model.generate_speech(speech_input)
        return result

    def _execute_recognition_agent(self, agent_input, custom_params):
        """
        Execute the recognition agent to convert speech to text.
        """
        # Determine file path vs audio data
        file_path = None
        audio_data = None

        if hasattr(agent_input, "audio") and agent_input.audio:
            audio_data = agent_input.audio
        elif isinstance(agent_input, str) and agent_input.startswith("file:"):
            file_path = agent_input[5:]  # Remove 'file:' prefix

        # Create recognition input
        recognition_input = SpeechRecognitionInput(
            audio_file_path=file_path,
            audio_data=audio_data,
            language=custom_params.get("language", "en"),
        )

        # Add model if provided
        if "model" in custom_params:
            recognition_input.model = custom_params["model"]

        # Map provider to supported models enum if possible
        provider_enum = None
        if self.provider.lower() == "openai":
            provider_enum = SupportedRecognitionModels["OPENAI"]
        elif self.provider.lower() == "keras":
            provider_enum = SupportedRecognitionModels["KERAS"]
        elif self.provider.lower() == "elevenlabs":
            provider_enum = SupportedRecognitionModels["ELEVENLABS"]
            # Add model_id for ElevenLabs
            if "model" in custom_params:
                recognition_input.model_id = custom_params["model"]
        else:
            provider_enum = self.provider  # Use as-is if not mapped

        # Create recognition model with correct parameters
        if self.provider.lower() == "keras":
            # Keras doesn't need an API key
            recognition_model = RemoteRecognitionModel(
                provider=provider_enum,
                model_name=custom_params.get("model_name", "whisper_tiny_en"),
                model_params=custom_params,
            )
        else:
            # Remote services need API key
            recognition_model = RemoteRecognitionModel(
                key_value=custom_params["key"],
                provider=provider_enum,
                model_name=custom_params.get("model"),
            )

        # Recognize speech
        result = recognition_model.recognize_speech(recognition_input)
        return result

    def _execute_embed_agent(self, agent_input, custom_params):
        """
        Execute the embedding agent to generate embeddings from text.
        """
        # Get text input
        text_input = agent_input.desc
        if self.mission and not text_input.startswith(self.mission):
            text_input = f"{self.mission}: {text_input}"

        # Create embedding input - uses 'texts' parameter, not 'inputs'
        embed_input = EmbedInput(
            texts=[text_input],  # Wrapped in list as EmbedInput expects a list of texts
            model=custom_params.get("model"),
        )

        # Try to set default values for the provider
        try:
            embed_input.set_default_values(self.provider)
        except ValueError:
            # If no default is available for this provider, continue with what we have
            pass

        # Create embed model
        embed_model = RemoteEmbedModel(
            api_key=custom_params["key"],
            provider_name=self.provider,
            options=self.options,
        )

        result = embed_model.get_embeddings(embed_input)
        return result

    def _execute_search_agent(self, agent_input, custom_params):
        """
        Execute the search agent to perform semantic search.
        """
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
