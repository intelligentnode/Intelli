from intelli.model.input.vision_input import VisionModelInput
from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper
from intelli.wrappers.openai_wrapper import OpenAIWrapper


class RemoteVisionModel:
    supported_vision_models = {
        "openai": OpenAIWrapper,
        "gemini": GeminiAIWrapper,
    }

    def __init__(self, api_key, provider="openai"):

        self.api_key = api_key

        if provider in self.supported_vision_models:
            self.provider = provider
            self.provider_wrapper = self.supported_vision_models[provider](api_key)
        else:
            supported_models = ", ".join(self.supported_vision_models.keys())
            raise ValueError(f"The provided provider {provider} not supported. Supported providers: {supported_models}")

    def image_to_text(self, vision_input):

        if isinstance(vision_input, dict):
            inputs = vision_input
        elif isinstance(vision_input, VisionModelInput):
            inputs = vision_input.get_provider_inputs(self.provider)
        else:
            raise ValueError("vision_input must be an instance of VisionModelInput or a dictionary.")

        if self.provider == "openai":
            return self.call_openai_vision(inputs)
        elif self.provider == "gemini":
            return self.call_gemini_vision(inputs)

    def call_openai_vision(self, inputs):
        data = self.provider_wrapper.image_to_text(inputs)
        return " ".join(choice['message']['content'] for choice in data['choices'])

    def call_gemini_vision(self, inputs):
        data = self.provider_wrapper.image_to_text_params(inputs)
        return " ".join(part['text'] for part in data['candidates'][0]['content']['parts'])
