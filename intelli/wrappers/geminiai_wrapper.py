import warnings

import requests

from intelli.config import config
from intelli.wrappers.googleai_wrapper import GoogleAIWrapper


class GeminiAIWrapper:
    """
    DEPRECATED: use GoogleAIWrapper (intelli.wrappers.googleai_wrapper) directly.

    GeminiAIWrapper is kept for backward compatibility. Every method forwards to
    GoogleAIWrapper and keeps its old signature, defaults and return shape.
    GoogleAIWrapper adds Vertex AI (vertex=True, project_id=...), streaming chat,
    Imagen, Veo, Lyria, the Live API, Agent Engine and more.

    Migration:
        GeminiAIWrapper(key).generate_content(params)  ->  GoogleAIWrapper(key).generate_content(params)
        GeminiAIWrapper(key).generate_speech(text)     ->  GoogleAIWrapper(key).generate_gemini_speech(text)
        (GoogleAIWrapper.generate_speech is Google Cloud Text-to-Speech.)
    """

    def __init__(self, api_key, timeout=180, **google_options):
        warnings.warn(
            "GeminiAIWrapper is deprecated; use GoogleAIWrapper from intelli.wrappers.googleai_wrapper instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json'
        })
        # Gemini Developer API unless Vertex options (vertex=True, project_id=...) are passed.
        google_options.setdefault('vertex', False)
        self.google = GoogleAIWrapper(api_key, timeout=timeout, session=session, **google_options)
        self.VERTEX_BASE_URL = config['url']['gemini']['vertex_base']
        self.models = config['url']['gemini']['models']
        self.endpoints = config['url']['gemini']['endpoints']

    # The old wrapper read these attributes on every call; they stay live by proxying to self.google.
    @property
    def session(self):
        return self.google.session

    @session.setter
    def session(self, value):
        self.google.session = value

    @property
    def timeout(self):
        return self.google.timeout

    @timeout.setter
    def timeout(self, value):
        self.google.timeout = value

    @property
    def API_KEY(self):
        return self.google.api_key

    @API_KEY.setter
    def API_KEY(self, value):
        self.google.api_key = value
        self.google.headers['X-Goog-Api-Key'] = value

    @property
    def API_BASE_URL(self):
        return self.google._dev_models_base

    @API_BASE_URL.setter
    def API_BASE_URL(self, value):
        self.google._dev_models_base = value
        self.google._dev_api_base = value[:-len('/models')] if value.endswith('/models') else value

    @property
    def UPLOAD_BASE_URL(self):
        return self.google._dev_upload_base

    @UPLOAD_BASE_URL.setter
    def UPLOAD_BASE_URL(self, value):
        self.google._dev_upload_base = value

    @property
    def FILES_BASE_URL(self):
        return self.google._dev_files_base

    @FILES_BASE_URL.setter
    def FILES_BASE_URL(self, value):
        self.google._dev_files_base = value

    _KEY_MAP = GoogleAIWrapper._KEY_MAP
    _REVERSE_KEY_MAP = GoogleAIWrapper._REVERSE_KEY_MAP

    def _camelize(self, obj):
        """Convert known snake_case keys to camelCase recursively."""
        return self.google._camelize(obj)

    def _snake_alias(self, obj):
        """Add snake_case aliases for known camelCase keys recursively."""
        return self.google._snake_alias(obj)

    def _model(self, kind, model_override=None):
        """The requested model, else this wrapper's default (Vertex uses Vertex defaults)."""
        if model_override:
            return model_override
        return None if self.google.vertex else self.models[kind]

    def generate_content(self, params, vision=False, model_override=None):
        """Generate content using Gemini models"""
        return self.google.generate_content(params, vision, self._model('vision' if vision else 'text', model_override))

    def generate_content_with_system_instructions(self, content_parts, system_instruction=None, model_override=None):
        """Generate content with system instructions support"""
        return self.google.generate_content_with_system_instructions(
            content_parts, system_instruction, self._model('text', model_override))

    def generate_structured_content(self, content_parts, response_schema, system_instruction=None,
                                    model_override=None, response_mime_type="application/json",
                                    generation_config=None, tools=None, tool_config=None):
        """Generate structured JSON outputs using Gemini structured outputs."""
        return self.google.generate_structured_content(
            content_parts, response_schema, system_instruction, self._model('text', model_override),
            response_mime_type, generation_config, tools, tool_config)

    def stream_generate_content(self, params, vision=False, model_override=None):
        """Stream content from Gemini (yields decoded lines)."""
        return self.google.stream_generate_content(
            params, vision, self._model('vision' if vision else 'text', model_override), raw=True)

    def image_to_text(self, user_input, image_data, extension):
        """Convert image to text using vision model"""
        return self.google.image_to_text(user_input, image_data, extension, self._model('vision'))

    def image_to_text_params(self, params, model_override=None):
        """Process image to text with custom parameters"""
        return self.google.image_to_text_params(params, self._model('vision', model_override))

    def image_to_text_with_file_uri(self, user_input, file_uri, mime_type):
        """Convert image to text using uploaded file URI"""
        return self.google.image_to_text_with_file_uri(user_input, file_uri, mime_type, self._model('vision'))

    def multiple_images_to_text(self, user_input, images_data):
        """Process multiple images with text prompt"""
        return self.google.multiple_images_to_text(user_input, images_data, self._model('vision'))

    def get_bounding_boxes(self, user_input, image_data, extension):
        """Get bounding box coordinates for objects in image"""
        return self.google.get_bounding_boxes(user_input, image_data, extension, self._model('vision'))

    def get_image_segmentation(self, user_input, image_data, extension):
        """Get image segmentation masks"""
        return self.google.get_image_segmentation(user_input, image_data, extension, self._model('vision'))

    def generate_image(self, prompt, config_params=None, model_override=None):
        """Generate images using Gemini image generation models"""
        return self.google.generate_image(prompt, config_params, self._model('image_generation', model_override))

    def generate_video(self, prompt, config_params=None, project_id=None):
        """Generate videos using Veo on Vertex AI (needs a project)"""
        if not project_id and not self.google.project_id:
            raise ValueError("Project ID is required for video generation")
        params = {
            "aspectRatio": "16:9",
            "personGeneration": "dont_allow"
        }
        if config_params:
            params.update(config_params)
        return self.google.generate_video(prompt, params, project_id=project_id,
                                          model=self._model('video_generation'))

    def check_video_generation_status(self, operation_name, project_id=None):
        """Check the status of video generation operation"""
        if not project_id and not self.google.project_id:
            raise ValueError("Project ID is required to check video generation status")
        return self.google.check_video_generation_status(operation_name, project_id)

    def wait_for_video_completion(self, operation_name, project_id, max_wait_time=300, poll_interval=5):
        """Wait for video generation to complete"""
        if not project_id and not self.google.project_id:
            raise ValueError("Project ID is required to check video generation status")
        return self.google.wait_for_video_completion(operation_name, project_id, max_wait_time, poll_interval)

    def generate_speech(self, text, voice_config=None, model_override=None):
        """Generate speech using Gemini TTS"""
        return self.google.generate_gemini_speech(text, voice_config, self._model('tts', model_override))

    def generate_multi_speaker_speech(self, text, speaker_configs):
        """Generate multi-speaker speech"""
        return self.google.generate_multi_speaker_speech(text, speaker_configs, self._model('tts'))

    def upload_file(self, file_path, display_name=None):
        """Upload a file using the Files API"""
        return self.google.upload_file(file_path, display_name)

    def list_files(self):
        """List uploaded files"""
        return self.google.list_files()

    def delete_file(self, file_name):
        """Delete an uploaded file"""
        return self.google.delete_file(file_name)

    def get_embeddings(self, params):
        """Get embeddings for text"""
        if isinstance(params, dict) and not params.get('model') and not self.google.vertex:
            params = {**params, 'model': self.models['embedding']}
        return self.google.get_embeddings(params)

    def get_batch_embeddings(self, params):
        """Get batch embeddings for multiple texts"""
        return self.google.get_batch_embeddings(params, model=self._model('embedding'))

    def _get_mime_type(self, file_path):
        """Get MIME type for file"""
        return self.google._get_mime_type(file_path)
