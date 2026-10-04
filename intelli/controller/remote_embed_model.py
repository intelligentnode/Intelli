from intelli.model.input.embed_input import EmbedInput
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper
from intelli.wrappers.mistralai_wrapper import MistralAIWrapper
from intelli.wrappers.openai_wrapper import OpenAIWrapper
from intelli.wrappers.nvidia_wrapper import NvidiaWrapper
from intelli.wrappers.vllm_wrapper import VLLMWrapper
from intelli.wrappers.aws_wrapper import AWSWrapper


class RemoteEmbedModel:
    def __init__(self, api_key, provider_name, options=None):
        self.provider_name = provider_name.lower()
        self.options = options or {}
        self.timeout = self.options.get("timeout", 180)
        providers = {
            'openai': OpenAIWrapper,
            'mistral': MistralAIWrapper,
            'gemini': GoogleAIWrapper,
            'nvidia': NvidiaWrapper,
            'vllm': VLLMWrapper,
            'aws': AWSWrapper
        }

        if self.provider_name == 'vllm':
            base_url = self.options.get("baseUrl")
            if not base_url:
                raise ValueError("VLLM provider requires baseUrl in options")
            self.provider = providers[self.provider_name](base_url, api_key, timeout=self.timeout)
        elif self.provider_name == 'gemini':
            # Vertex AI options (vertex, project_id, location) are read from options.
            self.provider = GoogleAIWrapper.from_options(api_key, self.options, timeout=self.timeout)
        elif self.provider_name == 'aws':
            # Region and IAM credentials (when no Bedrock API key is used) are read from options.
            self.provider = AWSWrapper.from_options(api_key, self.options, timeout=self.timeout)
        elif self.provider_name in providers:
            self.provider = providers[self.provider_name](api_key, timeout=self.timeout)
        else:
            if api_key in providers:
                raise Exception(f"Send the provider name as second parameter (api_key, provider_name).")
            else:    
                raise Exception(f"Provider {provider_name} not supported.")

    def get_embeddings(self, embed_input):
        if not isinstance(embed_input, EmbedInput):
            raise Exception("embed_input must be an instance of EmbedInput.")

        if self.provider_name == 'openai':
            params = embed_input.get_openai_inputs()
        elif self.provider_name == 'mistral':
            params = embed_input.get_mistral_inputs()
        elif self.provider_name == 'gemini':
            params = embed_input.get_gemini_inputs()
        elif self.provider_name == 'nvidia':
            params = embed_input.get_nvidia_inputs()
        elif self.provider_name == 'vllm':
            params = embed_input.get_vllm_inputs()
        elif self.provider_name == 'aws':
            params = embed_input.get_aws_inputs()
        else:
            raise Exception("Invalid provider name.")

        return self.provider.get_embeddings(params)
