from intelli.model.input.text_recognition_input import SpeechRecognitionInput
from intelli.wrappers.keras_wrapper import KerasWrapper
from intelli.wrappers.elevenlabs_wrapper import ElevenLabsWrapper
import requests
import os

SupportedRecognitionModels = {
    'OPENAI': 'openai',
    'KERAS': 'keras',
    'ELEVENLABS': 'elevenlabs',
}


class RemoteRecognitionModel:
    """
    Remote model for speech recognition using OpenAI's API,
    Keras offline models with Whisper, or Eleven Labs.
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
        elif key_type == SupportedRecognitionModels['ELEVENLABS']:
            if not key_value:
                raise ValueError("API key is required for Eleven Labs")
            self.elevenlabs_wrapper = ElevenLabsWrapper(key_value)
        else:
            raise ValueError('Invalid provider name')

    def get_supported_models(self):
        return list(SupportedRecognitionModels.values())

    def _direct_openai_stt_request(self, file_path, model="whisper-1", language=None):
        """
        Make a direct request to OpenAI's speech-to-text API using multipart/form-data.
        Enhanced with better error handling and validation.
        """
        url = "https://api.openai.com/v1/audio/transcriptions"

        # Validate file path
        if not file_path:
            raise ValueError("File path is required for OpenAI speech-to-text")

        if not os.path.exists(file_path):
            raise ValueError(f"File path does not exist: {file_path}")

        print(f"Sending audio file to OpenAI for transcription: {file_path}")

        # Prepare the multipart/form-data payload
        files = None
        try:
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

            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json().get('text', '')
            print(f"OpenAI transcription successful, length: {len(result)}")
            return result
        except requests.RequestException as e:
            error_msg = f"OpenAI API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_msg += f" - {error_json.get('error', {}).get('message', '')}"
                except:
                    error_msg += f" - Status code: {e.response.status_code}"
            print(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"OpenAI API request failed: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
        finally:
            # Clean up by closing the file
            if files and 'file' in files and hasattr(files['file'], 'close'):
                files['file'].close()

    def recognize_speech(self, input_params):
        """
        Recognize speech from audio input, using either OpenAI, Keras, or Eleven Labs.

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

        elif self.key_type == SupportedRecognitionModels['ELEVENLABS']:
            params = input_params.get_elevenlabs_input()

            # Determine the input type (file_path or audio_data)
            if 'file_path' in params and params['file_path']:
                result = self.elevenlabs_wrapper.speech_to_text(
                    audio_file=params['file_path'],
                    model_id=params.get('model_id', 'scribe_v1'),
                    language_code=params.get('language')
                )
            elif 'audio_data' in params and params['audio_data']:
                result = self.elevenlabs_wrapper.speech_to_text(
                    audio_file=params['audio_data'],
                    model_id=params.get('model_id', 'scribe_v1'),
                    language_code=params.get('language')
                )
            else:
                raise ValueError("Either file_path or audio_data must be provided")

            # Return just the transcribed text
            return result.get('text', '')

        else:
            raise ValueError('The keyType is not supported')
