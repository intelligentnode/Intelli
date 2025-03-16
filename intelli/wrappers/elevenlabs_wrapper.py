import requests
import json
import os
from typing import Union, Optional, BinaryIO, Dict, Any

from intelli.config import config


class ElevenLabsWrapper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = config['url']['elevenlabs']['base']
        self.session = requests.Session()
        self.session.headers.update({
            'xi-api-key': self.api_key,
            'Content-Type': 'application/json'
        })

    def text_to_speech(self, text: str, voice_id: str, model_id: Optional[str] = None,
                       output_format: str = "mp3_44100_128") -> bytes:
        """Convert text to speech using the specified voice"""
        url = f"{self.base_url}/text-to-speech/{voice_id}"

        params = {
            "output_format": output_format
        }

        payload = {
            "text": text
        }

        if model_id:
            payload["model_id"] = model_id

        try:
            response = self.session.post(url, json=payload, params=params)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as error:
            raise Exception(f"ElevenLabs API error: {error}")

    def stream_text_to_speech(self, text: str, voice_id: str, model_id: Optional[str] = None,
                              output_format: str = "mp3_44100_128") -> requests.Response:
        """Stream text to speech using the specified voice"""
        url = f"{self.base_url}/text-to-speech/{voice_id}/stream"

        params = {
            "output_format": output_format
        }

        payload = {
            "text": text
        }

        if model_id:
            payload["model_id"] = model_id

        try:
            response = self.session.post(url, json=payload, params=params, stream=True)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as error:
            raise Exception(f"ElevenLabs API error: {error}")

    def speech_to_text(self, audio_file: Union[str, BinaryIO, bytes], model_id: str = "scribe_v1",
                       language_code: Optional[str] = None) -> Dict[str, Any]:
        """Convert speech to text"""
        url = f"{self.base_url}/speech-to-text"

        # Set up multipart form data
        files = {}
        close_file = False

        # Handle different input types
        if isinstance(audio_file, str):
            files['file'] = open(audio_file, 'rb')
            close_file = True
        elif hasattr(audio_file, 'read') and callable(getattr(audio_file, 'read')):
            files['file'] = audio_file
        elif isinstance(audio_file, bytes):
            import io
            files['file'] = ('audio.mp3', io.BytesIO(audio_file), 'audio/mpeg')
        else:
            raise ValueError("audio_file must be a file path, file-like object, or bytes")

        data = {
            'model_id': model_id
        }

        if language_code:
            data['language_code'] = language_code

        # Need to use a different header for multipart form
        headers = {
            'xi-api-key': self.api_key
        }

        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            raise Exception(f"ElevenLabs API error: {error}")
        finally:
            # Close file if we opened it
            if close_file and 'file' in files and hasattr(files['file'], 'close'):
                files['file'].close()

    def speech_to_speech(self, audio_file: Union[str, BinaryIO, bytes], voice_id: str,
                         model_id: Optional[str] = None, output_format: str = "mp3_44100_128",
                         remove_background_noise: bool = False) -> bytes:
        """Transform audio from one voice to another"""
        url = f"{self.base_url}/speech-to-speech/{voice_id}"

        params = {
            "output_format": output_format
        }

        # Set up multipart form data
        files = {}
        close_file = False

        # Handle different input types
        if isinstance(audio_file, str):
            files['audio'] = open(audio_file, 'rb')
            close_file = True
        elif hasattr(audio_file, 'read') and callable(getattr(audio_file, 'read')):
            files['audio'] = audio_file
        elif isinstance(audio_file, bytes):
            import io
            files['audio'] = ('audio.mp3', io.BytesIO(audio_file), 'audio/mpeg')
        else:
            raise ValueError("audio_file must be a file path, file-like object, or bytes")

        data = {}

        if model_id:
            data['model_id'] = model_id

        if remove_background_noise:
            data['remove_background_noise'] = 'true'

        # Need to use a different header for multipart form
        headers = {
            'xi-api-key': self.api_key
        }

        try:
            response = requests.post(url, headers=headers, files=files, data=data, params=params)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as error:
            raise Exception(f"ElevenLabs API error: {error}")
        finally:
            # Close file if we opened it
            if close_file and 'audio' in files and hasattr(files['audio'], 'close'):
                files['audio'].close()

    def list_voices(self, show_legacy: bool = False) -> Dict[str, Any]:
        """Get a list of all available voices"""
        url = f"{self.base_url}/voices"

        params = {}
        if show_legacy:
            params["show_legacy"] = "true"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            raise Exception(f"ElevenLabs API error: {error}")
