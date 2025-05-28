from intelli.model.input.image_input import ImageModelInput
from intelli.wrappers.openai_wrapper import OpenAIWrapper
from intelli.wrappers.stability_wrapper import StabilityAIWrapper


class RemoteImageModel:
    supported_image_models = {
        "openai": OpenAIWrapper,
        "stability": StabilityAIWrapper,
    }

    def __init__(self, api_key, provider="openai"):
        if provider in self.supported_image_models:
            self.provider = self.supported_image_models[provider](api_key)
        else:
            supported_models = ", ".join(self.supported_image_models.keys())
            raise ValueError(f"The received provider {provider} not supported. Supported providers: {supported_models}")

    def generate_images(self, image_input):
        if isinstance(image_input, dict):
            inputs = image_input
        elif isinstance(image_input, ImageModelInput):
            inputs = image_input.get_openai_inputs() if isinstance(self.provider,
                                                                   OpenAIWrapper) else image_input.get_stability_inputs()
        else:
            raise ValueError("image_input must be an instance of ImageModelInput or a dictionary.")

        results = self.provider.generate_images(inputs)

        if isinstance(self.provider, OpenAIWrapper):
            return [data['url'] if 'url' in data else data['b64_json'] for data in results['data']]
        else:
            return [image_obj['base64'] for image_obj in results['artifacts']]
