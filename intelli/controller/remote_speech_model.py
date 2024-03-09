from intelli.model.input.text_speech_input import Text2SpeechInput
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper
from intelli.wrappers.openai_wrapper import OpenAIWrapper

SupportedSpeechModels = {
    'GOOGLE': 'google',
    'OPENAI': 'openai',
}


class RemoteSpeechModel:

    def __init__(self, key_value, provider=None):
        if not provider:
            provider = SupportedSpeechModels['GOOGLE']

        supported_models = self.get_supported_models()

        if provider in supported_models:
            self.initiate(key_value, provider)
        else:
            models = " - ".join(supported_models)
            raise ValueError(f"The received key value is not supported. Send any model from: {models}")

    def initiate(self, key_value, key_type):
        self.key_type = key_type
        if key_type == SupportedSpeechModels['GOOGLE']:
            self.google_wrapper = GoogleAIWrapper(key_value)
        elif key_type == SupportedSpeechModels['OPENAI']:
            self.openai_wrapper = OpenAIWrapper(key_value)
        else:
            raise ValueError('Invalid provider name')

    def get_supported_models(self):
        return list(SupportedSpeechModels.values())

    def generate_speech(self, input_params):
        if not isinstance(input_params, Text2SpeechInput):
            raise ValueError('Invalid input: Must be an instance of Text2SpeechInput')

        if self.key_type == SupportedSpeechModels['GOOGLE']:
            params = input_params.get_google_input()
            response = self.google_wrapper.generate_speech(params)
            return response

        elif self.key_type == SupportedSpeechModels['OPENAI']:
            params = input_params.get_openai_input()
            response = self.openai_wrapper.text_to_speech(params)
            return response
        else:
            raise ValueError('The keyType is not supported')
