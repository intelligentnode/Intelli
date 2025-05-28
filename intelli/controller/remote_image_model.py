from intelli.model.input.image_input import ImageModelInput
from intelli.wrappers.openai_wrapper import OpenAIWrapper
from intelli.wrappers.stability_wrapper import StabilityAIWrapper
from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper


class RemoteImageModel:
    supported_image_models = {
        "openai": OpenAIWrapper,
        "stability": StabilityAIWrapper,
        "gemini": GeminiAIWrapper,
    }

    def __init__(self, api_key, provider="openai"):
        if provider in self.supported_image_models:
            self.provider_name = provider
            self.provider = self.supported_image_models[provider](api_key)
        else:
            supported_models = ", ".join(self.supported_image_models.keys())
            raise ValueError(f"The received provider {provider} not supported. Supported providers: {supported_models}")

    def generate_images(self, image_input):
        if isinstance(image_input, dict):
            inputs = image_input
        elif isinstance(image_input, ImageModelInput):
            if self.provider_name == "gemini":
                inputs = image_input.get_gemini_inputs()
            elif self.provider_name == "openai":
                inputs = image_input.get_openai_inputs()
            else:  # stability
                inputs = image_input.get_stability_inputs()
        else:
            raise ValueError("image_input must be an instance of ImageModelInput or a dictionary.")

        if self.provider_name == "gemini":
            # Extract model override if provided
            model_override = inputs.get("model")
            results = self.provider.generate_image(
                inputs.get("prompt", ""), 
                inputs.get("config_params"),
                model_override=model_override
            )
            # Extract image data from Gemini response
            images = []
            if 'candidates' in results:
                for candidate in results['candidates']:
                    if 'content' in candidate and 'parts' in candidate['content']:
                        for part in candidate['content']['parts']:
                            if 'inline_data' in part and part['inline_data'].get('mime_type', '').startswith('image/'):
                                images.append(part['inline_data']['data'])
            return images
        elif self.provider_name == "openai":
            results = self.provider.generate_images(inputs)
            return [data['url'] if 'url' in data else data['b64_json'] for data in results['data']]
        else:  # stability
            results = self.provider.generate_images(inputs)
            return [image_obj['base64'] for image_obj in results['artifacts']]
