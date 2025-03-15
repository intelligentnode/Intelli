from intelli.model.input.text_recognition_input import SpeechRecognitionInput
from intelli.wrappers.keras_wrapper import KerasWrapper
import requests
import os

SupportedRecognitionModels = {
    'OPENAI': 'openai',
    'KERAS': 'keras',
}


class RemoteRecognitionModel:
    """
    Remote model for speech recognition using either OpenAI's API or
    Keras offline models with Whisper.
    """

    def __init__(self, key_value=None, provider=None, model_name=None, model_params=None):
        if not provider:
            provider = SupportedRecognitionModels['OPENAI']

        supported_models = self.get_supported_models()

        if provider in supported_models:
            self.initiate(key_value, provider, model_name, model_params)
        else:
            models = " - ".join(supported_models)
            raise ValueError(f"The received provider is not supported. Send any provider from: {models}")

    def initiate(self, key_value, key_type, model_name=None, model_params=None):
        self.key_type = key_type
        self.api_key = key_value  # Store the API key for direct use if needed

        if key_type == SupportedRecognitionModels['OPENAI']:
            if not key_value:
                raise ValueError("API key is required for OpenAI")
            from intelli.wrappers.openai_wrapper import OpenAIWrapper
            self.openai_wrapper = OpenAIWrapper(key_value)
        elif key_type == SupportedRecognitionModels['KERAS']:
            if not model_name:
                model_name = "whisper_tiny_en"  # Default model
            self.keras_wrapper = KerasWrapper(model_name=model_name, model_params=model_params)
        else:
            raise ValueError('Invalid provider name')

    def get_supported_models(self):
        return list(SupportedRecognitionModels.values())

    def _direct_openai_stt_request(self, file_path, model="whisper-1", language=None):
        """
        Make a direct request to OpenAI's speech-to-text API using multipart/form-data.

        This bypasses the wrapper to ensure correct formatting of the request.
        """
        url = "https://api.openai.com/v1/audio/transcriptions"

        # Prepare the multipart/form-data payload
        files = {
            'file': open(file_path, 'rb')
        }

        data = {
            'model': model
        }

        if language:
            data['language'] = language

        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }

        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json().get('text', '')
        except Exception as e:
            raise Exception(f"OpenAI API request failed: {str(e)}")
        finally:
            # Clean up by closing the file
            if 'file' in files and hasattr(files['file'], 'close'):
                files['file'].close()

    def recognize_speech(self, input_params):
        """
        Recognize speech from audio input, using either OpenAI or Keras.

        Args:
            input_params: SpeechRecognitionInput object containing audio data and parameters

        Returns:
            Transcribed text as string
        """
        if not isinstance(input_params, SpeechRecognitionInput):
            raise ValueError('Invalid input: Must be an instance of SpeechRecognitionInput')

        if self.key_type == SupportedRecognitionModels['OPENAI']:
            params = input_params.get_openai_input()

            # Use our direct OpenAI implementation to ensure correct formatting
            return self._direct_openai_stt_request(
                file_path=params['file_path'],
                model=params['model'],
                language=params['language']
            )

        elif self.key_type == SupportedRecognitionModels['KERAS']:
            keras_params = input_params.get_keras_input()

            response = self.keras_wrapper.transcript(
                audio_data=keras_params['audio_data'],
                sample_rate=keras_params['sample_rate'],
                language=keras_params['language'],
                user_prompt=keras_params['user_prompt'],
                condition_on_previous_text=keras_params['condition_on_previous_text'],
                max_steps=keras_params['max_steps'],
                max_chunk_sec=keras_params['max_chunk_sec']
            )
            return response
        else:
            raise ValueError('The keyType is not supported')
