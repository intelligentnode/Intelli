from intelli.model.input.text_speech_input import Text2SpeechInput
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper
from intelli.wrappers.openai_wrapper import OpenAIWrapper
from intelli.wrappers.elevenlabs_wrapper import ElevenLabsWrapper

SupportedSpeechModels = {
    'GOOGLE': 'google',
    'OPENAI': 'openai',
    'ELEVENLABS': 'elevenlabs',
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
        elif key_type == SupportedSpeechModels['ELEVENLABS']:
            self.elevenlabs_wrapper = ElevenLabsWrapper(key_value)
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

        elif self.key_type == SupportedSpeechModels['ELEVENLABS']:
            params = input_params.get_elevenlabs_input()
            response = self.elevenlabs_wrapper.text_to_speech(
                text=params['text'],
                voice_id=params['voice_id'],
                model_id=params.get('model_id'),
                output_format=params.get('output_format', 'mp3_44100_128')
            )
            return response
        else:
            raise ValueError('The keyType is not supported')

    def list_voices(self):
        """Get available voices for the current provider"""
        if self.key_type == SupportedSpeechModels['ELEVENLABS']:
            return self.elevenlabs_wrapper.list_voices()
        else:
            raise ValueError(f"Voice listing not supported for provider: {self.key_type}")

    def stream_speech(self, input_params):
        """Stream speech for providers that support it"""
        if not isinstance(input_params, Text2SpeechInput):
            raise ValueError('Invalid input: Must be an instance of Text2SpeechInput')

        if self.key_type == SupportedSpeechModels['ELEVENLABS']:
            params = input_params.get_elevenlabs_input()
            response = self.elevenlabs_wrapper.stream_text_to_speech(
                text=params['text'],
                voice_id=params['voice_id'],
                model_id=params.get('model_id'),
                output_format=params.get('output_format', 'mp3_44100_128')
            )
            return response
        else:
            raise ValueError(f"Streaming not supported for provider: {self.key_type}")
