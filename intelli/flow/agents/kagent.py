from intelli.flow.agents.agent import BasicAgent
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput
from intelli.wrappers.keras_wrapper import KerasWrapper
from intelli.flow.types import AgentTypes

class KerasAgent(BasicAgent):
    def __init__(self, agent_type, provider="", mission="", model_params={}, options=None, log=False, external=False):
        super().__init__()
        self.type = agent_type
        self.provider = provider
        self.mission = mission
        self.model_params = model_params
        self.options = options if options is not None else {}
        self.log = log
        self.external = external

        if not external:
            self.wrapper = KerasWrapper(self.model_params.get("model_name"), self.model_params)
        else:
            self.wrapper = None

    def set_keras_model(self, model, model_params):
        if not self.external:
            raise Exception("Initiate the agent with external flag to set the model.")
        if not self.wrapper:
            self.wrapper = KerasWrapper()
        self.wrapper.set_model(model, model_params)

    def update_model_params(self, model_params):
        self.model_params = model_params
        if self.wrapper:
            self.wrapper.update_model_params(model_params)

    def execute(self, agent_input: AgentInput, new_params={}):
        """
        Execute the Keras agent based on the agent type.
        Handles both text generation and speech recognition (whisper) models.
        """
        custom_params = dict(self.model_params)
        if new_params and isinstance(new_params, dict):
            custom_params.update(new_params)

        if not self.wrapper:
            raise ValueError("Model wrapper is not set.")

        # Handle different agent types
        if self.type == AgentTypes.TEXT.value:
            return self._execute_text_generation(agent_input, custom_params)
        elif self.type == AgentTypes.RECOGNITION.value:
            return self._execute_speech_recognition(agent_input, custom_params)
        else:
            raise ValueError(f"Unsupported agent type for Keras: {self.type}")

    def _execute_text_generation(self, agent_input, custom_params):
        """Handle text generation using Keras models"""
        if not isinstance(agent_input, TextAgentInput):
            raise ValueError("Text generation requires TextAgentInput")

        max_length = custom_params.get("max_length", 180)
        model_input = agent_input.desc if not self.mission else self.mission + ": " + agent_input.desc

        if self.log:
            print("Call the model generate with input: ", model_input)

        generated_output = self.wrapper.generate(model_input, max_length=max_length)

        # Clean up output if needed
        if isinstance(generated_output, str) and generated_output.startswith(model_input):
            generated_output = generated_output.replace(model_input, "", 1).strip()

        return generated_output

    def _execute_speech_recognition(self, agent_input, custom_params):
        """Handle speech recognition using Whisper models"""
        # Extract audio data
        audio_data = None
        sample_rate = custom_params.get("sample_rate", 16000)

        if hasattr(agent_input, "audio") and agent_input.audio:
            # Audio directly from agent input
            audio_data = agent_input.audio
        elif isinstance(agent_input, bytes):
            # Raw audio bytes
            audio_data = agent_input
        else:
            raise ValueError("Speech recognition requires audio data")

        # Prepare transcript parameters
        language = custom_params.get("language", "<|en|>")  # Whisper language prompt
        user_prompt = custom_params.get("user_prompt", "")
        if self.mission and not user_prompt:
            user_prompt = self.mission

        condition_on_previous_text = custom_params.get("condition_on_previous_text", True)
        max_steps = custom_params.get("max_steps", None)
        max_chunk_sec = custom_params.get("max_chunk_sec", None)

        if self.log:
            print(f"Transcribing audio with language: {language}")

        # Call the transcript method
        result = self.wrapper.transcript(
            audio_data=audio_data,
            sample_rate=sample_rate,
            language=language,
            user_prompt=user_prompt,
            condition_on_previous_text=condition_on_previous_text,
            max_steps=max_steps,
            max_chunk_sec=max_chunk_sec
        )

        return result

    def fine_tune_model_with_lora(self, fine_tuning_config, enable_lora=True, custom_loss=None, custom_metrics=None):
        """Fine-tune the model using LoRA technique"""
        if not self.wrapper:
            raise ValueError("Model wrapper is not set.")

        self.wrapper.fine_tune(
            dataset=fine_tuning_config.get('dataset'),
            fine_tuning_config=fine_tuning_config,
            enable_lora=enable_lora,
            custom_loss=custom_loss,
            custom_metrics=custom_metrics
        )
