import json
import requests
import base64
import contextlib
import copy
import io
import mimetypes
import os
import re
import time
import wave
from typing import List, Dict, Any, Optional, Union

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class GoogleAIError(Exception):
    """
    Error raised by the Gemini / Vertex AI (Agent Platform) methods of GoogleAIWrapper.

    It is a plain Exception subclass, so existing `except Exception` handlers keep working.
    API keys and access tokens are removed from the message and details.
    """

    def __init__(self, message, status_code=None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class GoogleAIWrapper:
    """
    One wrapper for Google AI services.

    Google Cloud APIs (unchanged): Text-to-Speech, Speech-to-Text, Vision, Natural Language
    and Translation, authenticated with a Google Cloud API key.

    Gemini models and media generation, on one of two backends:
    - Gemini Developer API (generativelanguage.googleapis.com, AI Studio key). This is the default.
    - Vertex AI / Gemini Enterprise Agent Platform (aiplatform.googleapis.com), selected with
      vertex=True, or automatically when project_id, credentials or access_token is given.
      Auth is an Agent Platform API key (express mode, or project-scoped when project_id is
      set) or Application Default Credentials / an OAuth access token.

    Examples:
        GoogleAIWrapper(gemini_key)                                         # Gemini Developer API
        GoogleAIWrapper(vertex_key, vertex=True)                            # Vertex express mode
        GoogleAIWrapper(vertex_key, vertex=True, project_id="my-project")   # project-scoped (Veo, Live)
        GoogleAIWrapper(project_id="my-project", location="global")         # ADC (gcloud auth application-default login)

    See instructions/how to docs/GOOGLE_AI_WRAPPER.md for the full guide.
    """

    def __init__(self, api_key=None, timeout=180, *, vertex=None, project_id=None, location=None,
                 credentials=None, access_token=None, api_version=None, base_url=None,
                 quota_project_id=None, session=None):
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-Goog-Api-Key': self.api_key,
        }
        # Base URLs for different services based on config
        self.api_speech_url = config['url']['google']['base'].format(config['url']['google']['speech']['prefix'])
        self.api_vision_url = config['url']['google']['base'].format(config['url']['google']['vision']['prefix'])
        self.api_language_url = config['url']['google']['base'].format(config['url']['google']['language']['prefix'])
        self.api_translation_url = config['url']['google']['base'].format(
            config['url']['google']['translation']['prefix'])
        self.api_speech_to_text_url = config['url']['google']['base'].format(
            config['url']['google']['speechtotext']['prefix'])

        self._init_genai(vertex, project_id, location, credentials, access_token, api_version,
                         base_url, quota_project_id, session)

    # Text-to-Speech methods
    def generate_speech(self, params):
        """Generate speech using Google Text-to-Speech API"""
        url = self.api_speech_url + config['url']['google']['speech']['synthesize']['postfix']
        param = self.get_synthesize_input(params)

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(param), timeout=self.timeout)
            response.raise_for_status()
            return response.json()['audioContent']
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def get_synthesize_input(self, params):
        """Format input parameters for speech synthesis"""
        return {
            'input': {
                'text': params['text'],
            },
            'voice': {
                'languageCode': params['languageCode'],
                'name': params['name'],
                'ssmlGender': params['ssmlGender'],
            },
            'audioConfig': {
                'audioEncoding': params.get('audioEncoding', 'MP3'),
                'speakingRate': params.get('speakingRate', 1.0),
                'pitch': params.get('pitch', 0.0),
                'volumeGainDb': params.get('volumeGainDb', 0.0),
            },
        }

    def generate_speech_with_ssml(self, ssml_text, voice_params):
        """Generate speech using SSML input"""
        url = self.api_speech_url + config['url']['google']['speech']['synthesize']['postfix']
        
        param = {
            'input': {
                'ssml': ssml_text,
            },
            'voice': voice_params,
            'audioConfig': {
                'audioEncoding': 'MP3',
            },
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(param), timeout=self.timeout)
            response.raise_for_status()
            return response.json()['audioContent']
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    # Speech-to-Text methods
    def transcribe_audio(self, audio_content, config_params=None):
        """Transcribe audio to text using Google Speech API"""
        url = self.api_speech_to_text_url + config['url']['google']['speechtotext']['recognize']['postfix']

        if config_params is None:
            config_params = {
                'languageCode': 'en-US',
                'enableAutomaticPunctuation': True
            }

        # Prepare the request payload
        payload = {
            'config': config_params,
            'audio': {
                'content': base64.b64encode(audio_content).decode('utf-8')
            }
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def transcribe_audio_long_running(self, audio_uri, config_params=None):
        """Transcribe long audio files using long-running operation"""
        url = self.api_speech_to_text_url + "/speech:longrunningrecognize"

        if config_params is None:
            config_params = {
                'languageCode': 'en-US',
                'enableAutomaticPunctuation': True,
                'enableWordTimeOffsets': True
            }

        payload = {
            'config': config_params,
            'audio': {
                'uri': audio_uri
            }
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    # Vision API methods
    def analyze_image(self, image_content, features=None):
        """Analyze images using Google Vision API"""
        url = self.api_vision_url + config['url']['google']['vision']['annotate']['postfix']

        if features is None:
            features = [
                {'type': 'LABEL_DETECTION', 'maxResults': 10},
                {'type': 'TEXT_DETECTION'},
                {'type': 'FACE_DETECTION'},
                {'type': 'LANDMARK_DETECTION'}
            ]

        # Prepare the request payload
        payload = {
            'requests': [{
                'image': {
                    'content': base64.b64encode(image_content).decode('utf-8')
                },
                'features': features
            }]
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def analyze_image_from_uri(self, image_uri, features=None):
        """Analyze images from URI using Google Vision API"""
        url = self.api_vision_url + config['url']['google']['vision']['annotate']['postfix']

        if features is None:
            features = [
                {'type': 'LABEL_DETECTION', 'maxResults': 10},
                {'type': 'TEXT_DETECTION'},
                {'type': 'FACE_DETECTION'},
                {'type': 'LANDMARK_DETECTION'}
            ]

        payload = {
            'requests': [{
                'image': {
                    'source': {
                        'imageUri': image_uri
                    }
                },
                'features': features
            }]
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def detect_objects_with_localization(self, image_content):
        """Detect and localize objects in images"""
        features = [
            {'type': 'OBJECT_LOCALIZATION', 'maxResults': 50}
        ]
        return self.analyze_image(image_content, features)

    def detect_text_with_handwriting(self, image_content):
        """Detect text including handwriting in images"""
        features = [
            {'type': 'DOCUMENT_TEXT_DETECTION'},
            {'type': 'TEXT_DETECTION'}
        ]
        return self.analyze_image(image_content, features)

    def detect_faces_with_emotions(self, image_content):
        """Detect faces with emotion analysis"""
        features = [
            {'type': 'FACE_DETECTION', 'maxResults': 20}
        ]
        return self.analyze_image(image_content, features)

    def detect_logos_and_brands(self, image_content):
        """Detect logos and brand marks in images"""
        features = [
            {'type': 'LOGO_DETECTION', 'maxResults': 10}
        ]
        return self.analyze_image(image_content, features)

    def get_image_properties(self, image_content):
        """Get detailed image properties including colors"""
        features = [
            {'type': 'IMAGE_PROPERTIES'}
        ]
        return self.analyze_image(image_content, features)

    def detect_safe_search(self, image_content):
        """Detect inappropriate content in images"""
        features = [
            {'type': 'SAFE_SEARCH_DETECTION'}
        ]
        return self.analyze_image(image_content, features)

    def crop_hints(self, image_content, aspect_ratios=None):
        """Get crop hints for images"""
        features = [
            {'type': 'CROP_HINTS'}
        ]
        
        if aspect_ratios:
            features[0]['cropHintsParams'] = {
                'aspectRatios': aspect_ratios
            }
            
        return self.analyze_image(image_content, features)

    # Natural Language API methods
    def analyze_text(self, text, features=None):
        """Analyze text using Google Natural Language API"""
        url = self.api_language_url + config['url']['google']['language']['analyze']['postfix']

        if features is None:
            features = {
                'extractSyntax': True,
                'extractEntities': True,
                'extractDocumentSentiment': True,
                'classifyText': True
            }

        # Prepare the request payload
        payload = {
            'document': {
                'type': 'PLAIN_TEXT',
                'content': text
            },
            'features': features
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def analyze_entity_sentiment(self, text):
        """Analyze entity sentiment in text"""
        url = self.api_language_url + "/documents:analyzeEntitySentiment"

        payload = {
            'document': {
                'type': 'PLAIN_TEXT',
                'content': text
            }
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def classify_text(self, text):
        """Classify text into categories"""
        url = self.api_language_url + "/documents:classifyText"

        payload = {
            'document': {
                'type': 'PLAIN_TEXT',
                'content': text
            }
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    # Translation API methods
    def translate_text(self, text, target_language, source_language=None):
        """Translate text using Google Translation API"""
        url = self.api_translation_url + config['url']['google']['translation']['translate']['postfix']

        # Prepare the request payload
        payload = {
            'q': text,
            'target': target_language
        }

        if source_language:
            payload['source'] = source_language

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def detect_language(self, text):
        """Detect the language of text"""
        url = self.api_translation_url + "/detect"

        payload = {
            'q': text
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def get_supported_languages(self, target_language='en'):
        """Get list of supported languages"""
        url = self.api_translation_url + "/languages"

        params = {
            'target': target_language
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def describe_image(self, image_content):
        """Generate a comprehensive description of an image using multiple Vision API features."""

        # Combine multiple detection types
        features = [
            {'type': 'LABEL_DETECTION', 'maxResults': 10},
            {'type': 'OBJECT_LOCALIZATION', 'maxResults': 10},
            {'type': 'WEB_DETECTION', 'maxResults': 5},
            {'type': 'IMAGE_PROPERTIES', 'maxResults': 3},
            {'type': 'LANDMARK_DETECTION', 'maxResults': 3},
            {'type': 'LOGO_DETECTION', 'maxResults': 3},
            {'type': 'TEXT_DETECTION', 'maxResults': 1},
            {'type': 'FACE_DETECTION', 'maxResults': 5}
        ]

        # Get comprehensive analysis
        result = self.analyze_image(image_content, features)

        # Extract relevant information from each detection type
        description = {}
        response = result.get('responses', [{}])[0]

        if 'labelAnnotations' in response:
            description['labels'] = [label.get('description') for label in response.get('labelAnnotations', [])]

        if 'localizedObjectAnnotations' in response:
            description['objects'] = [obj.get('name') for obj in response.get('localizedObjectAnnotations', [])]

        if 'webDetection' in response:
            web_detection = response.get('webDetection', {})
            description['web_entities'] = [entity.get('description') for entity in web_detection.get('webEntities', [])]
            description['web_labels'] = [label.get('label') for label in web_detection.get('bestLabelAnnotations', [])]

        if 'imagePropertiesAnnotation' in response:
            colors = response.get('imagePropertiesAnnotation', {}).get('dominantColors', {}).get('colors', [])
            description['colors'] = []
            for color in colors:
                rgb = color.get('color', {})
                hex_color = f"#{rgb.get('red', 0):02x}{rgb.get('green', 0):02x}{rgb.get('blue', 0):02x}"
                description['colors'].append({
                    'hex': hex_color,
                    'score': color.get('score')
                })

        if 'landmarkAnnotations' in response:
            description['landmarks'] = [landmark.get('description') for landmark in
                                        response.get('landmarkAnnotations', [])]

        if 'logoAnnotations' in response:
            description['logos'] = [logo.get('description') for logo in response.get('logoAnnotations', [])]

        if 'faceAnnotations' in response:
            description['faces'] = []
            for face in response.get('faceAnnotations', []):
                face_info = {
                    'joy': face.get('joyLikelihood', 'UNKNOWN'),
                    'sorrow': face.get('sorrowLikelihood', 'UNKNOWN'),
                    'anger': face.get('angerLikelihood', 'UNKNOWN'),
                    'surprise': face.get('surpriseLikelihood', 'UNKNOWN'),
                    'confidence': face.get('detectionConfidence', 0)
                }
                description['faces'].append(face_info)

        # Extract text
        if 'textAnnotations' in response and len(response.get('textAnnotations', [])) > 0:
            description['text'] = response.get('textAnnotations', [{}])[0].get('description')

        # Generate a natural language description
        summary = self._generate_image_summary(description)

        return {
            'detailed_analysis': description,
            'summary': summary
        }

    def _generate_image_summary(self, description):
        """Generate a natural language summary from the detailed image analysis."""
        parts = []

        # Add information about objects
        if description.get('objects'):
            objects_str = ', '.join(description.get('objects')[:5])
            parts.append(f"This image contains {objects_str}.")

        # Add information about labels if objects aren't available
        elif description.get('labels'):
            labels_str = ', '.join(description.get('labels')[:5])
            parts.append(f"This image shows {labels_str}.")

        # Add landmark information
        if description.get('landmarks'):
            landmark = description.get('landmarks')[0]
            parts.append(f"The landmark in this image appears to be {landmark}.")

        # Add logo information
        if description.get('logos'):
            logos_str = ', '.join(description.get('logos'))
            parts.append(f"The image contains the following logos: {logos_str}.")

        # Add face information
        if description.get('faces'):
            face_count = len(description.get('faces'))
            if face_count == 1:
                parts.append("There is 1 person visible in the image.")
            else:
                parts.append(f"There are {face_count} people visible in the image.")

        # Add text information
        if description.get('text'):
            parts.append(f"The image contains text: \"{description.get('text')[:100]}\".")

        # Add color information
        if description.get('colors') and len(description.get('colors')) > 0:
            color_hex = description.get('colors')[0].get('hex')
            parts.append(f"The dominant color in the image is {color_hex}.")

        # If no description could be generated
        if not parts:
            return "The image analysis did not yield a clear description."

        return ' '.join(parts)

    def batch_analyze_images(self, image_requests):
        """Analyze multiple images in a single request"""
        url = self.api_vision_url + config['url']['google']['vision']['annotate']['postfix']

        payload = {
            'requests': image_requests
        }

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def extract_document_text(self, image_content):
        """Extract text from documents with layout information"""
        features = [
            {'type': 'DOCUMENT_TEXT_DETECTION'}
        ]
        
        result = self.analyze_image(image_content, features)
        
        # Process the response to extract structured text information
        response = result.get('responses', [{}])[0]
        
        if 'fullTextAnnotation' in response:
            full_text = response['fullTextAnnotation']
            return {
                'text': full_text.get('text', ''),
                'pages': full_text.get('pages', []),
                'confidence': self._calculate_average_confidence(full_text)
            }
        
        return {'text': '', 'pages': [], 'confidence': 0}

    def _calculate_average_confidence(self, full_text_annotation):
        """Calculate average confidence from text detection"""
        confidences = []
        
        for page in full_text_annotation.get('pages', []):
            for block in page.get('blocks', []):
                if 'confidence' in block:
                    confidences.append(block['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0

    # ==================================================================
    # Gemini and Vertex AI (Gemini Enterprise Agent Platform) methods
    # ==================================================================

    _CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

    # snake_case -> camelCase keys normalized in request bodies. Kept identical to the
    # GeminiAIWrapper map so request bodies stay the same for existing callers.
    # Google APIs accept snake_case field names too, so other keys are sent as given.
    _KEY_MAP = {
        # top-level / common
        "system_instruction": "systemInstruction",
        "generation_config": "generationConfig",
        "safety_settings": "safetySettings",
        "tool_config": "toolConfig",
        "response_modalities": "responseModalities",
        "speech_config": "speechConfig",
        "voice_config": "voiceConfig",
        "multi_speaker_voice_config": "multiSpeakerVoiceConfig",
        "speaker_voice_configs": "speakerVoiceConfigs",
        "prebuilt_voice_config": "prebuiltVoiceConfig",
        "voice_name": "voiceName",
        "response_mime_type": "responseMimeType",
        "response_schema": "responseSchema",
        # content parts
        "inline_data": "inlineData",
        "file_data": "fileData",
        "mime_type": "mimeType",
        "file_uri": "fileUri",
    }

    _REVERSE_KEY_MAP = {v: k for k, v in _KEY_MAP.items()}

    _MIME_TYPES = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.heic': 'image/heic',
        '.heif': 'image/heif',
        '.gif': 'image/gif',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.webm': 'video/webm',
        '.mpeg': 'video/mpeg',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.html': 'text/html',
        '.csv': 'text/csv',
        '.json': 'application/json',
    }

    def _init_genai(self, vertex, project_id, location, credentials, access_token, api_version,
                    base_url, quota_project_id, session):
        gemini_cfg = config['url']['gemini']
        vertex_cfg = gemini_cfg.get('vertex', {})

        env_vertex = (os.getenv('GOOGLE_GENAI_USE_VERTEXAI') or os.getenv('GOOGLE_GENAI_USE_ENTERPRISE')
                      or '').strip().lower() in ('1', 'true', 'yes')
        if vertex is None:
            vertex = bool(project_id or credentials is not None or access_token or env_vertex)
        self.vertex = bool(vertex)

        # With ADC (no API key) the project can come from the environment.
        if self.vertex and not project_id and not self.api_key:
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or None
        self.project_id = project_id

        if location is None and self.vertex and os.getenv('GOOGLE_CLOUD_LOCATION'):
            location = os.getenv('GOOGLE_CLOUD_LOCATION')
        self._location_explicit = location is not None
        if location is None and self.vertex and project_id:
            location = vertex_cfg.get('default_location', 'global')
        self.location = location

        self._credentials = credentials
        self._access_token = access_token
        self.quota_project_id = quota_project_id
        self.base_url = base_url.rstrip('/') if base_url else None
        self.session = session if session is not None else requests.Session()

        # Gemini Developer API endpoints (same config as GeminiAIWrapper)
        self._dev_models_base = gemini_cfg['base']
        self._dev_api_base = (self._dev_models_base[:-len('/models')]
                              if self._dev_models_base.endswith('/models') else self._dev_models_base)
        self._dev_upload_base = gemini_cfg['upload_base']
        self._dev_files_base = gemini_cfg['files_base']
        self._vertex_api_version = vertex_cfg.get('api_version', 'v1beta1')

        if self.vertex:
            self.api_version = api_version or self._vertex_api_version
            self._vertex_api_version = self.api_version
            self.models = {**gemini_cfg['models'], **vertex_cfg.get('models', {})}
        else:
            if api_version:
                self._dev_api_base = re.sub(r'/v[^/]+$', '/' + api_version, self._dev_api_base)
                self._dev_models_base = self._dev_api_base + '/models'
            self.api_version = self._dev_api_base.rsplit('/', 1)[-1]
            self.models = dict(gemini_cfg['models'])
        self._capability_locations = dict(vertex_cfg.get('locations', {}))

    @classmethod
    def from_options(cls, api_key=None, options=None, timeout=None):
        """
        Build a wrapper from a Chatbot / Agent / controller options dict.

        Recognized keys: vertex, project_id (or vertex_project), location (or vertex_location),
        credentials, access_token, api_version, quota_project_id, timeout.
        Without these keys the result is GoogleAIWrapper(api_key, timeout=...), the Gemini Developer API.
        """
        options = options or {}
        kwargs = {}
        for key in ('vertex', 'project_id', 'location', 'credentials', 'access_token',
                    'api_version', 'quota_project_id'):
            if options.get(key) is not None:
                kwargs[key] = options[key]
        if options.get('vertex_project') and 'project_id' not in kwargs:
            kwargs['project_id'] = options['vertex_project']
        if options.get('vertex_location') and 'location' not in kwargs:
            kwargs['location'] = options['vertex_location']
        if timeout is None:
            timeout = options.get('timeout', 180)
        return cls(api_key, timeout=timeout, **kwargs)

    # ------------------------------------------------------------------
    # Request / response normalization
    # ------------------------------------------------------------------
    def _camelize(self, obj: Any) -> Any:
        """Convert known snake_case keys to camelCase recursively."""
        if isinstance(obj, list):
            return [self._camelize(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            new_k = self._KEY_MAP.get(k, k)
            out[new_k] = self._camelize(v)
        return out

    def _snake_alias(self, obj: Any) -> Any:
        """
        Add snake_case aliases for known camelCase keys recursively.
        Does not remove the original camelCase keys.
        """
        if isinstance(obj, list):
            return [self._snake_alias(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            aliased_v = self._snake_alias(v)
            out[k] = aliased_v
            snake_k = self._REVERSE_KEY_MAP.get(k)
            if snake_k and snake_k not in out:
                out[snake_k] = aliased_v
        return out

    def _unalias(self, obj: Any) -> Any:
        """Drop the snake_case aliases added by _snake_alias, so a response part can be sent back."""
        if isinstance(obj, list):
            return [self._unalias(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        return {k: self._unalias(v) for k, v in obj.items()
                if not (k in self._KEY_MAP and self._KEY_MAP[k] in obj)}

    def _content(self, content):
        """Normalize one content item. Vertex AI rejects contents without a role."""
        if isinstance(content, str):
            return {'role': 'user', 'parts': [{'text': content}]}
        if isinstance(content, dict) and self.vertex and not content.get('role'):
            parts = content.get('parts') or []
            is_model = any(isinstance(p, dict) and ('functionCall' in p or 'function_call' in p) for p in parts)
            content = {**content, 'role': 'model' if is_model else 'user'}
        return content

    def _prepare_body(self, params):
        """Body for generateContent-style calls: camelize known keys, add roles on Vertex."""
        if isinstance(params, str):
            params = {'contents': [{'role': 'user', 'parts': [{'text': params}]}]}
        body = self._camelize(params)
        if self.vertex:
            # Vertex reads 'model' in the body as a resource name; the model is in the URL.
            body.pop('model', None)
        contents = body.get('contents')
        if isinstance(contents, (str, dict)):
            contents = [contents]
        if isinstance(contents, list):
            body['contents'] = [self._content(c) for c in contents]
        system_instruction = body.get('systemInstruction')
        if isinstance(system_instruction, str):
            body['systemInstruction'] = {'parts': [{'text': system_instruction}]}
        return body

    def _default_model(self, kind):
        model = self.models.get(kind) or config['url']['gemini']['models'].get(kind)
        if not model:
            raise ValueError(f"No default model is configured for '{kind}'. Pass the model explicitly.")
        return model

    # ------------------------------------------------------------------
    # URLs
    # ------------------------------------------------------------------
    def _vertex_host(self, location=None):
        vertex_cfg = config['url']['gemini'].get('vertex', {})
        if not location or location == 'global':
            return vertex_cfg.get('global_host', 'https://aiplatform.googleapis.com')
        if location in ('us', 'eu'):
            return vertex_cfg.get('multi_regional_host',
                                  'https://aiplatform.{location}.rep.googleapis.com').format(location=location)
        return vertex_cfg.get('regional_host', 'https://{location}-aiplatform.googleapis.com').format(location=location)

    def _location_for(self, capability=None, location=None):
        """Explicit location, else the capability default (e.g. Veo and Live in us-central1), else the wrapper location."""
        if location:
            return location
        if capability and not self._location_explicit and capability in self._capability_locations:
            return self._capability_locations[capability]
        return self.location

    @staticmethod
    def _vertex_model_path(model):
        model = (model or '').strip()
        if not model:
            raise ValueError("A model name is required")
        if model.startswith(('projects/', 'publishers/')):
            return model
        if model.startswith('models/'):
            return 'publishers/google/' + model
        if '/' in model:
            publisher, model_id = model.split('/', 1)
            return f'publishers/{publisher}/models/{model_id}'
        return f'publishers/google/models/{model}'

    def _model_path(self, model):
        if self.vertex:
            return self._vertex_model_path(model)
        model = (model or '').strip()
        if not model:
            raise ValueError("A model name is required")
        return model if model.startswith(('models/', 'tunedModels/')) else f'models/{model}'

    def _resource_url(self, path, location=None):
        """Full URL for a resource path such as 'publishers/google/models/x:generateContent'."""
        path = path.lstrip('/')
        if not self.vertex:
            root = self.base_url or self._dev_api_base
            return f"{root}/{path}"
        match = re.match(r'projects/[^/]+/locations/([^/]+)/', path)
        if match:
            host_location = match.group(1)
        else:
            if not self.api_key and not self.project_id:
                self._get_access_token()  # ADC can provide the project
            if self.project_id:
                location = location or self.location or 'global'
                path = f"projects/{self.project_id}/locations/{location}/{path}"
            host_location = location
        root = self.base_url or f"{self._vertex_host(host_location)}/{self.api_version}"
        return f"{root}/{path}"

    def _model_url(self, model, method, location=None):
        return self._resource_url(f"{self._model_path(model)}:{method}", location)

    def _vertex_project_url(self, project, location, path):
        root = self.base_url if (self.base_url and self.vertex) else f"{self._vertex_host(location)}/{self._vertex_api_version}"
        return f"{root}/projects/{project}/locations/{location}/{path}"

    def _name_url(self, name):
        """URL for a full Vertex resource name (projects/.../locations/<loc>/...)."""
        match = re.match(r'projects/[^/]+/locations/([^/]+)/', name)
        location = match.group(1) if match else None
        root = self.base_url if (self.base_url and self.vertex) else f"{self._vertex_host(location)}/{self._vertex_api_version}"
        return f"{root}/{name}"

    def _require_project(self, feature):
        if not self.vertex:
            raise GoogleAIError(f"{feature} needs Vertex AI. Create the wrapper with vertex=True and project_id.")
        if not self.project_id and not self.api_key:
            self._get_access_token()  # ADC can provide the project
        if not self.project_id:
            raise GoogleAIError(f"{feature} needs a Google Cloud project. Create the wrapper with project_id=...")
        return self.project_id

    # ------------------------------------------------------------------
    # Auth, requests and errors
    # ------------------------------------------------------------------
    def _auth_headers(self):
        if self.api_key:
            return {'x-goog-api-key': self.api_key}
        headers = {'Authorization': f'Bearer {self._get_access_token()}'}
        quota_project = self.quota_project_id or getattr(self._credentials, 'quota_project_id', None)
        if quota_project:
            headers['x-goog-user-project'] = quota_project
        return headers

    def _get_access_token(self):
        token = self._access_token() if callable(self._access_token) else self._access_token
        if token:
            return token
        if not self.vertex:
            raise GoogleAIError("The Gemini Developer API needs an API key. Pass api_key, or use "
                                "vertex=True with Application Default Credentials.")
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError:
            raise GoogleAIError("Vertex AI without an API key uses Application Default Credentials. Install "
                                "google-auth (pip install google-auth) and run: "
                                "gcloud auth application-default login") from None
        credentials = self._credentials
        if credentials is None:
            try:
                credentials, default_project = google.auth.default(scopes=[self._CLOUD_SCOPE])
            except Exception as error:
                raise GoogleAIError(f"Could not load Application Default Credentials ({type(error).__name__}). "
                                    "Run: gcloud auth application-default login, or pass api_key or "
                                    "access_token.") from None
            self._credentials = credentials
            if not self.project_id and default_project:
                self.project_id = default_project
                if not self.location:
                    self.location = config['url']['gemini'].get('vertex', {}).get('default_location', 'global')
        if not getattr(credentials, 'valid', False) or not getattr(credentials, 'token', None):
            try:
                credentials.refresh(Request())
            except Exception as error:
                raise GoogleAIError(self._redact(f"Could not refresh Google credentials: {error}")) from None
        return credentials.token

    def _secrets(self):
        values = [self.api_key]
        if isinstance(self._access_token, str):
            values.append(self._access_token)
        token = getattr(self._credentials, 'token', None)
        if isinstance(token, str):
            values.append(token)
        return [v for v in values if isinstance(v, str) and len(v) >= 6]

    def _redact(self, text):
        text = str(text)
        for secret in self._secrets():
            text = text.replace(secret, '<redacted>')
        return re.sub(r'([?&]key=)[^&\s"\']+', r'\1<redacted>', text)

    def _redact_obj(self, obj):
        if obj is None or isinstance(obj, str):
            return self._redact(obj) if obj is not None else None
        try:
            return json.loads(self._redact(json.dumps(obj)))
        except (TypeError, ValueError):
            return self._redact(obj)

    def _api_error(self, prefix, error):
        response = getattr(error, 'response', None)
        status = getattr(response, 'status_code', None) if response is not None else None
        details = None
        if response is not None:
            try:
                details = response.json()
            except Exception:
                try:
                    details = (response.text or '')[:2000] or None
                except Exception:
                    details = None
        message = f"{prefix}: {error}"
        if details:
            message += f" - Details: {details if isinstance(details, str) else json.dumps(details)}"
        if not self.vertex and 'API_KEY_SERVICE_BLOCKED' in message:
            message += (" (hint: this key is not enabled for the Gemini Developer API. If it is a Vertex AI / "
                        "Agent Platform key, create the wrapper with vertex=True.)")
        if self.vertex and 'API keys are not supported by this API' in message:
            message += (" (hint: this Vertex AI endpoint needs OAuth. Use Application Default Credentials "
                        "(gcloud auth application-default login) or pass access_token.)")
        if self.vertex and 'RESOURCE_PROJECT_INVALID' in message:
            message += " (hint: this endpoint needs a project. Create the wrapper with project_id=...)"
        return GoogleAIError(self._redact(message), status_code=status, details=self._redact_obj(details))

    def _request(self, http_method, url, body=None, *, params=None, stream=False,
                 error_prefix='Gemini API error', headers=None, auth=True):
        request_headers = {'Content-Type': 'application/json'}
        if auth:
            request_headers.update(self._auth_headers())
        if headers:
            request_headers.update(headers)
        senders = {'GET': self.session.get, 'POST': self.session.post,
                   'DELETE': self.session.delete, 'PATCH': self.session.patch}
        kwargs = {'headers': request_headers, 'timeout': self.timeout}
        if params:
            kwargs['params'] = params
        if body is not None:
            kwargs['json'] = body
        if stream:
            kwargs['stream'] = True
        try:
            response = senders[http_method](url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as error:
            raise self._api_error(error_prefix, error) from None

    def _request_json(self, http_method, url, body=None, *, alias=True, error_prefix='Gemini API error', **kwargs):
        response = self._request(http_method, url, body, error_prefix=error_prefix, **kwargs)
        try:
            data = response.json()
        except ValueError:
            if not (getattr(response, 'text', '') or '').strip():
                return {}
            raise GoogleAIError(f"{error_prefix}: the API returned a response that is not JSON") from None
        return self._snake_alias(data) if alias else data

    @staticmethod
    def _iter_sse(response):
        """Parse a server-sent events stream into JSON objects."""
        buffer = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.decode('utf-8') if isinstance(raw_line, bytes) else raw_line
            if line.startswith('data:'):
                buffer.append(line[5:].lstrip())
            elif not line.strip() and buffer:
                yield json.loads('\n'.join(buffer))
                buffer = []
        if buffer:
            yield json.loads('\n'.join(buffer))

    # ------------------------------------------------------------------
    # Parts and media helpers
    # ------------------------------------------------------------------
    def _get_mime_type(self, file_path):
        """Get MIME type for a file path or URI."""
        extension = os.path.splitext(file_path.split('?')[0])[1].lower()
        if extension in self._MIME_TYPES:
            return self._MIME_TYPES[extension]
        guessed = mimetypes.guess_type(file_path)[0]
        return guessed or 'application/octet-stream'

    def media_part(self, source=None, mime_type=None, *, data=None, path=None, uri=None, video_metadata=None):
        """
        Build one content part for an image, audio, video or document.

        source can be: a part dict (returned as is), bytes, a (bytes, mime_type) tuple,
        a local file path, or a gs://, https:// or YouTube URI.
        """
        if isinstance(source, dict):
            return source
        if isinstance(source, tuple):
            source, mime_type = source[0], (source[1] if len(source) > 1 else mime_type)
        if isinstance(source, (bytes, bytearray)):
            data = source
        elif isinstance(source, str):
            if source.startswith(('gs://', 'http://', 'https://')):
                uri = source
            elif os.path.exists(source):
                path = source
            else:
                raise ValueError(f"Not a file path or URI: {source[:80]}")
        if path:
            with open(path, 'rb') as file:
                data = file.read()
            mime_type = mime_type or self._get_mime_type(path)
        if uri:
            if not mime_type:
                is_youtube = 'youtube.com' in uri or 'youtu.be' in uri
                mime_type = 'video/mp4' if is_youtube else self._get_mime_type(uri)
            part = {'fileData': {'mimeType': mime_type, 'fileUri': uri}}
        elif data is not None:
            if isinstance(data, (bytes, bytearray)):
                data = base64.b64encode(bytes(data)).decode('utf-8')
            if not mime_type:
                raise ValueError("mime_type is required for raw data")
            part = {'inlineData': {'mimeType': mime_type, 'data': data}}
        else:
            raise ValueError("Provide data, a file path or a URI")
        if video_metadata:
            part['videoMetadata'] = video_metadata
        return part

    def _to_parts(self, prompt=None, media=None):
        parts = []
        if isinstance(prompt, (str, dict)):
            prompt = [prompt]
        for item in prompt or []:
            parts.append({'text': item} if isinstance(item, str) else item)
        if media is not None and not isinstance(media, list):
            media = [media]
        for item in media or []:
            parts.append(self.media_part(item))
        return parts

    def _build_request(self, prompt=None, system_instruction=None, media=None, generation_config=None,
                       tools=None, tool_config=None, safety_settings=None, history=None):
        contents = [self._content(c) for c in (history or [])]
        if prompt is not None or media:
            contents.append({'role': 'user', 'parts': self._to_parts(prompt, media)})
        body = {'contents': contents}
        if system_instruction:
            body['systemInstruction'] = ({'parts': [{'text': system_instruction}]}
                                         if isinstance(system_instruction, str) else system_instruction)
        if generation_config:
            body['generationConfig'] = dict(generation_config)
        if tools:
            body['tools'] = tools
        if tool_config:
            body['toolConfig'] = tool_config
        if safety_settings:
            body['safetySettings'] = safety_settings
        return body

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_parts(response):
        for candidate in (response or {}).get('candidates') or []:
            for part in ((candidate or {}).get('content') or {}).get('parts') or []:
                if isinstance(part, dict):
                    yield part

    @staticmethod
    def extract_text(response, include_thoughts=False):
        """Join the text parts of the first candidate (thought summaries are skipped by default)."""
        candidates = (response or {}).get('candidates') or []
        if not candidates:
            return ""
        parts = ((candidates[0] or {}).get('content') or {}).get('parts') or []
        return "".join(p.get('text', '') for p in parts
                       if isinstance(p, dict) and (include_thoughts or not p.get('thought')))

    @staticmethod
    def extract_function_calls(response):
        """Return [{'name', 'args', 'id'?}] for every functionCall part."""
        calls = []
        for part in GoogleAIWrapper._iter_parts(response):
            call = part.get('functionCall') or part.get('function_call')
            if call:
                calls.append(dict(call))
        return calls

    @staticmethod
    def extract_grounding(response):
        """groundingMetadata of the first candidate (web sources, search queries), or {}."""
        candidates = (response or {}).get('candidates') or []
        return (candidates[0] or {}).get('groundingMetadata', {}) if candidates else {}

    @staticmethod
    def _extract_media(response, prefix):
        items = []
        for part in GoogleAIWrapper._iter_parts(response):
            inline = part.get('inlineData') or part.get('inline_data')
            if inline:
                mime = inline.get('mimeType') or inline.get('mime_type') or ''
                if mime.startswith(prefix):
                    items.append({'mime_type': mime, 'data': inline.get('data')})
        for prediction in (response or {}).get('predictions') or []:
            if isinstance(prediction, dict) and prediction.get('bytesBase64Encoded'):
                mime = prediction.get('mimeType') or ('audio/wav' if prefix == 'audio/' else 'image/png')
                if mime.startswith(prefix):
                    items.append({'mime_type': mime, 'data': prediction['bytesBase64Encoded']})
        return items

    @staticmethod
    def extract_images(response):
        """Images from a Gemini response or an Imagen prediction: [{'mime_type', 'data' (base64)}]."""
        return GoogleAIWrapper._extract_media(response, 'image/')

    @staticmethod
    def extract_audio(response):
        """Audio from a Gemini TTS / music response or a Lyria prediction: [{'mime_type', 'data' (base64)}]."""
        return GoogleAIWrapper._extract_media(response, 'audio/')

    @staticmethod
    def extract_videos(operation):
        """Videos from a finished Veo operation: [{'mime_type', 'data' (base64) or None, 'uri' or None}]."""
        response = (operation or {}).get('response') or {}
        videos = []
        for video in response.get('videos') or []:  # Vertex AI
            videos.append({'mime_type': video.get('mimeType', 'video/mp4'),
                           'data': video.get('bytesBase64Encoded'), 'uri': video.get('gcsUri')})
        samples = (response.get('generateVideoResponse') or {}).get('generatedSamples') or []
        for sample in samples:  # Gemini Developer API
            video = sample.get('video') or {}
            videos.append({'mime_type': video.get('mimeType', 'video/mp4'),
                           'data': video.get('encodedVideo') or video.get('bytesBase64Encoded'),
                           'uri': video.get('uri')})
        return videos

    @staticmethod
    def pcm_to_wav(pcm, sample_rate=24000, channels=1, sample_width=2):
        """Wrap raw 16-bit PCM (what Gemini TTS and the Live API return) in a WAV container."""
        if isinstance(pcm, str):
            pcm = base64.b64decode(pcm)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()

    @staticmethod
    def audio_to_wav(audio):
        """Return WAV bytes for an item from extract_audio (raw L16 PCM is wrapped, WAV is returned as is)."""
        raw = base64.b64decode(audio['data']) if isinstance(audio.get('data'), str) else audio.get('data')
        mime = (audio.get('mime_type') or '').lower()
        if raw[:4] == b'RIFF' or 'wav' in mime:
            return raw
        if 'l16' in mime or 'pcm' in mime:
            match = re.search(r'rate=(\d+)', mime)
            return GoogleAIWrapper.pcm_to_wav(raw, int(match.group(1)) if match else 24000)
        return raw

    # ------------------------------------------------------------------
    # Text, chat and multimodal generation
    # ------------------------------------------------------------------
    def generate_content(self, params, vision=False, model_override=None, *, model=None):
        """
        Call :generateContent. params is a Gemini request body (snake_case or camelCase),
        or a plain prompt string. Returns the JSON response with snake_case aliases added.
        """
        model = model or model_override or self._default_model('vision' if vision else 'text')
        return self._request_json('POST', self._model_url(model, 'generateContent'), self._prepare_body(params),
                                  error_prefix='Gemini API error')

    def stream_generate_content(self, params, vision=False, model_override=None, *, model=None, raw=False):
        """
        Call :streamGenerateContent and yield one response chunk (dict) at a time (server-sent events).
        raw=True keeps the old behavior: the endpoint without alt=sse, yielding raw decoded lines.
        """
        model = model or model_override or self._default_model('vision' if vision else 'text')
        url = self._model_url(model, 'streamGenerateContent')
        body = self._prepare_body(params)
        response = self._request('POST', url, body, params=None if raw else {'alt': 'sse'}, stream=True,
                                 error_prefix='Gemini stream error')
        try:
            if raw:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        yield line
            else:
                for chunk in self._iter_sse(response):
                    yield self._snake_alias(chunk)
        except requests.exceptions.RequestException as error:
            raise self._api_error('Gemini stream error', error) from None
        finally:
            response.close()

    def generate_text(self, prompt, model=None, *, system_instruction=None, media=None, generation_config=None,
                      tools=None, tool_config=None, safety_settings=None, history=None):
        """Generate and return text. media takes paths, bytes, (bytes, mime) tuples, gs:// / https URIs or parts."""
        body = self._build_request(prompt, system_instruction, media, generation_config, tools, tool_config,
                                   safety_settings, history)
        return self.extract_text(self.generate_content(body, model=model or self._default_model('text')))

    def stream_text(self, prompt, model=None, *, system_instruction=None, media=None, generation_config=None,
                    tools=None, tool_config=None, safety_settings=None, history=None):
        """Yield text chunks as the model writes them."""
        body = self._build_request(prompt, system_instruction, media, generation_config, tools, tool_config,
                                   safety_settings, history)
        for chunk in self.stream_generate_content(body, model=model or self._default_model('text')):
            text = self.extract_text(chunk)
            if text:
                yield text

    def start_chat(self, model=None, *, system_instruction=None, generation_config=None, tools=None,
                   tool_config=None, safety_settings=None, history=None):
        """Start a multi-turn chat session that keeps the history (including thought signatures)."""
        return GoogleAIChatSession(self, model=model, system_instruction=system_instruction,
                                   generation_config=generation_config, tools=tools, tool_config=tool_config,
                                   safety_settings=safety_settings, history=history)

    def generate_content_with_system_instructions(self, content_parts, system_instruction=None, model_override=None):
        """Generate content with system instructions support"""
        params = {
            "contents": [{
                "parts": content_parts
            }]
        }

        if system_instruction:
            params["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }

        return self.generate_content(params, model_override=model_override)

    def generate_structured_content(
        self,
        content_parts: List[Dict[str, Any]],
        response_schema: Dict[str, Any],
        system_instruction: Optional[str] = None,
        model_override: Optional[str] = None,
        response_mime_type: str = "application/json",
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
    ):
        """Generate structured JSON output that follows response_schema."""
        gen_cfg = generation_config.copy() if generation_config else {}
        gen_cfg.setdefault("response_mime_type", response_mime_type)
        gen_cfg.setdefault("response_schema", response_schema)

        params: Dict[str, Any] = {
            "contents": [{"parts": content_parts}],
            "generation_config": gen_cfg,
        }
        if system_instruction:
            params["system_instruction"] = {"parts": [{"text": system_instruction}]}
        if tools is not None:
            params["tools"] = tools
        if tool_config is not None:
            params["tool_config"] = tool_config

        return self.generate_content(params, model_override=model_override)

    def count_tokens(self, params, model=None):
        """Count the tokens of a request body or a prompt string."""
        model = model or self._default_model('text')
        body = self._prepare_body(params)
        if not self.vertex:
            # The Developer API takes extra request fields only inside generateContentRequest.
            extra = {k: body.pop(k) for k in ('systemInstruction', 'tools', 'generationConfig', 'toolConfig',
                                              'safetySettings', 'cachedContent') if k in body}
            if extra:
                body = {'generateContentRequest': {'model': self._model_path(model), **body, **extra}}
        return self._request_json('POST', self._model_url(model, 'countTokens'), body, alias=False,
                                  error_prefix='Gemini countTokens error')

    def compute_tokens(self, params, model=None):
        """Return the token ids and pieces of a request (Vertex AI only)."""
        if not self.vertex:
            raise GoogleAIError("compute_tokens is only available on Vertex AI. Create the wrapper with vertex=True.")
        body = self._prepare_body(params)
        body = {'contents': body.get('contents', [])}
        return self._request_json('POST', self._model_url(model or self._default_model('text'), 'computeTokens'),
                                  body, alias=False, error_prefix='Gemini computeTokens error')

    # ------------------------------------------------------------------
    # Image, audio, video and document understanding
    # ------------------------------------------------------------------
    def image_to_text(self, user_input, image_data, extension, model_override=None):
        """Convert image to text using vision model"""
        params = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{user_input}"},
                        {
                            "inline_data": {
                                "mime_type": f"image/{extension}",
                                "data": image_data
                            }
                        }
                    ]
                }
            ]
        }

        return self.image_to_text_params(params=params, model_override=model_override)

    def image_to_text_params(self, params, model_override=None):
        """Process image to text with custom parameters"""
        return self.generate_content(params, True, model_override=model_override)

    def image_to_text_with_file_uri(self, user_input, file_uri, mime_type, model_override=None):
        """Convert image to text using a file URI (Files API URI, gs:// URI or https URL)."""
        params = {
            "contents": [
                {
                    "parts": [
                        {"text": user_input},
                        {
                            "file_data": {
                                "mime_type": mime_type,
                                "file_uri": file_uri
                            }
                        }
                    ]
                }
            ]
        }
        return self.generate_content(params, True, model_override=model_override)

    def multiple_images_to_text(self, user_input, images_data, model_override=None):
        """Process multiple images with text prompt"""
        parts = [{"text": user_input}]

        for img_data in images_data:
            if 'file_uri' in img_data:
                parts.append({
                    "file_data": {
                        "mime_type": img_data['mime_type'],
                        "file_uri": img_data['file_uri']
                    }
                })
            else:
                parts.append({
                    "inline_data": {
                        "mime_type": img_data['mime_type'],
                        "data": img_data['data']
                    }
                })

        params = {
            "contents": [{
                "parts": parts
            }]
        }

        return self.generate_content(params, True, model_override=model_override)

    def get_bounding_boxes(self, user_input, image_data, extension, model_override=None):
        """Get bounding box coordinates for objects in image"""
        prompt = f"{user_input}. Return bounding boxes in [ymin, xmin, ymax, xmax] format normalized to 0-1000."
        return self.image_to_text(prompt, image_data, extension, model_override=model_override)

    def get_image_segmentation(self, user_input, image_data, extension, model_override=None):
        """Get image segmentation masks"""
        prompt = f"""
        {user_input}
        Output a JSON list of segmentation masks where each entry contains the 2D
        bounding box in the key "box_2d", the segmentation mask in key "mask", and
        the text label in the key "label". Use descriptive labels.
        """
        return self.image_to_text(prompt, image_data, extension, model_override=model_override)

    def media_to_text(self, prompt, media, model=None, **kwargs):
        """Ask about images, audio, video or documents and return text (see media_part for media formats)."""
        return self.generate_text(prompt, model=model or self._default_model('vision'), media=media, **kwargs)

    def audio_to_text(self, audio, prompt="Transcribe this audio.", mime_type=None, model=None, **kwargs):
        """Transcribe or describe audio with a Gemini model. audio is bytes, a path or a gs:// / https URI."""
        return self.media_to_text(prompt, [self.media_part(audio, mime_type)], model=model, **kwargs)

    def video_to_text(self, video, prompt="Summarize this video.", mime_type=None, model=None,
                      video_metadata=None, **kwargs):
        """Summarize or ask about a video. video is bytes, a path, a gs:// URI or a YouTube URL."""
        part = self.media_part(video, mime_type, video_metadata=video_metadata)
        return self.media_to_text(prompt, [part], model=model, **kwargs)

    # ------------------------------------------------------------------
    # Image generation: Gemini native images and Imagen
    # ------------------------------------------------------------------
    def generate_image(self, prompt, config_params=None, model_override=None, *, images=None):
        """
        Generate images with a Gemini image model. Pass images (paths, bytes, URIs or parts)
        to edit them or combine them. config_params is merged into generationConfig
        (e.g. {"imageConfig": {"aspectRatio": "16:9"}}).
        """
        model = model_override or self._default_model('image_generation')

        default_config = {
            "responseModalities": ["TEXT", "IMAGE"]
        }

        if config_params:
            default_config.update(config_params)

        params = {
            "contents": [{
                "parts": [{"text": prompt}] + [self.media_part(image) for image in (images or [])]
            }],
            "generationConfig": default_config
        }
        return self._request_json('POST', self._model_url(model, 'generateContent'), self._prepare_body(params),
                                  error_prefix='Gemini Image Generation error')

    def edit_image(self, prompt, images, config_params=None, model_override=None):
        """Edit one or more images with a Gemini image model and a text instruction."""
        if not isinstance(images, list):
            images = [images]
        return self.generate_image(prompt, config_params, model_override, images=images)

    def _imagen_image(self, image, mime_type=None):
        if isinstance(image, dict):
            return image
        if isinstance(image, (bytes, bytearray)):
            result = {'bytesBase64Encoded': base64.b64encode(bytes(image)).decode('utf-8')}
        elif isinstance(image, str) and image.startswith('gs://'):
            result = {'gcsUri': image}
        elif isinstance(image, str) and os.path.exists(image):
            with open(image, 'rb') as file:
                result = {'bytesBase64Encoded': base64.b64encode(file.read()).decode('utf-8')}
            mime_type = mime_type or self._get_mime_type(image)
        elif isinstance(image, str):
            result = {'bytesBase64Encoded': image}  # already base64
        else:
            raise ValueError("image must be bytes, a path, a gs:// URI, base64 text or a dict")
        if mime_type:
            result['mimeType'] = mime_type
        return result

    def imagen_generate_images(self, prompt, number_of_images=1, model=None, *, aspect_ratio=None,
                               negative_prompt=None, parameters=None):
        """Generate images with Imagen (:predict). Google lists Imagen as deprecated in favor of Gemini image models."""
        request_parameters = {'sampleCount': number_of_images}
        if aspect_ratio:
            request_parameters['aspectRatio'] = aspect_ratio
        if negative_prompt:
            request_parameters['negativePrompt'] = negative_prompt
        if parameters:
            request_parameters.update(parameters)
        body = {'instances': [{'prompt': prompt}], 'parameters': request_parameters}
        url = self._model_url(model or self._default_model('imagen'), 'predict', self._location_for('imagen'))
        return self._request_json('POST', url, body, alias=False, error_prefix='Imagen error')

    def imagen_edit_image(self, prompt, image, mask=None, *, edit_mode=None, mask_mode=None, model=None,
                          parameters=None):
        """
        Edit an image with Imagen (imagen-3.0-capability-001). Give a mask image, or a mask_mode such as
        "MASK_MODE_BACKGROUND" / "MASK_MODE_FOREGROUND" for an automatic mask.
        """
        references = [{'referenceType': 'REFERENCE_TYPE_RAW', 'referenceId': 1,
                       'referenceImage': self._imagen_image(image)}]
        if mask is not None:
            references.append({'referenceType': 'REFERENCE_TYPE_MASK', 'referenceId': 2,
                               'referenceImage': self._imagen_image(mask),
                               'maskImageConfig': {'maskMode': 'MASK_MODE_USER_PROVIDED', 'dilation': 0.01}})
        elif mask_mode:
            references.append({'referenceType': 'REFERENCE_TYPE_MASK', 'referenceId': 2,
                               'maskImageConfig': {'maskMode': mask_mode, 'dilation': 0.01}})
        request_parameters = {'sampleCount': 1,
                              'editMode': edit_mode or ('EDIT_MODE_INPAINT_INSERTION' if (mask is not None or mask_mode)
                                                        else 'EDIT_MODE_DEFAULT')}
        if parameters:
            request_parameters.update(parameters)
        body = {'instances': [{'prompt': prompt, 'referenceImages': references}], 'parameters': request_parameters}
        url = self._model_url(model or self._default_model('imagen_edit'), 'predict', self._location_for('imagen'))
        return self._request_json('POST', url, body, alias=False, error_prefix='Imagen edit error')

    def imagen_upscale_image(self, image, upscale_factor='x2', model=None, *, parameters=None):
        """Upscale an image with Imagen (upscale_factor 'x2', 'x3' or 'x4')."""
        request_parameters = {'mode': 'upscale', 'sampleCount': 1, 'upscaleConfig': {'upscaleFactor': upscale_factor}}
        if parameters:
            request_parameters.update(parameters)
        body = {'instances': [{'prompt': 'Upscale the image', 'image': self._imagen_image(image)}],
                'parameters': request_parameters}
        url = self._model_url(model or self._default_model('imagen_upscale'), 'predict', self._location_for('imagen'))
        return self._request_json('POST', url, body, alias=False, error_prefix='Imagen upscale error')

    # ------------------------------------------------------------------
    # Video generation (Veo)
    # ------------------------------------------------------------------
    def generate_video(self, prompt, config_params=None, project_id=None, *, model=None, image=None,
                       last_frame=None, location=None):
        """
        Start a Veo video generation and return the long-running operation (use its 'name' to poll).

        Vertex AI needs a project: pass project_id here or create the wrapper with project_id.
        Passing project_id also selects Vertex AI on a Developer API wrapper (the old GeminiAIWrapper
        behavior). config_params are Veo parameters, e.g. {"durationSeconds": 4, "aspectRatio": "16:9",
        "resolution": "720p", "generateAudio": False, "storageUri": "gs://bucket/path/"}.
        """
        instance = {'prompt': prompt}
        if image is not None:
            instance['image'] = self._imagen_image(image, None if isinstance(image, dict) else 'image/png')
        if last_frame is not None:
            instance['lastFrame'] = self._imagen_image(last_frame, None if isinstance(last_frame, dict) else 'image/png')
        parameters = {'aspectRatio': '16:9'}
        if config_params:
            parameters.update(config_params)
        body = {'instances': [instance], 'parameters': parameters}

        if self.vertex or project_id:
            project = project_id or self.project_id
            if not project and not self.api_key:
                project = self._require_project('Veo video generation')
            if not project:
                raise ValueError("Project ID is required for video generation on Vertex AI "
                                 "(Veo does not run in express mode). Pass project_id.")
            location = self._location_for('video_generation', location) or 'us-central1'
            model = model or (self._default_model('video_generation') if self.vertex
                              else config['url']['gemini']['models']['video_generation'])
            url = self._vertex_project_url(project, location, f"{self._vertex_model_path(model)}:predictLongRunning")
        else:
            model = model or self._default_model('video_generation')
            url = self._model_url(model, 'predictLongRunning')
        return self._request_json('POST', url, body, alias=False, error_prefix='Veo Video Generation error')

    def get_video_operation(self, operation_name, project_id=None):
        """Poll a Veo operation (Vertex: :fetchPredictOperation, Developer API: GET operations/...)."""
        if isinstance(operation_name, dict):
            operation_name = operation_name.get('name')
        if not operation_name:
            raise ValueError("operation_name is required")
        if operation_name.startswith('projects/'):
            resource = operation_name.rpartition('/operations/')[0]
            return self._request_json('POST', f"{self._name_url(resource)}:fetchPredictOperation",
                                      {'operationName': operation_name}, alias=False,
                                      error_prefix='Video status check error')
        if self.vertex:
            raise ValueError("Vertex AI operation names start with 'projects/'.")
        return self._request_json('GET', f"{self._dev_api_base}/{operation_name}", alias=False,
                                  error_prefix='Video status check error')

    def check_video_generation_status(self, operation_name, project_id=None):
        """Check the status of video generation operation"""
        return self.get_video_operation(operation_name, project_id)

    def wait_for_video_completion(self, operation_name, project_id=None, max_wait_time=300, poll_interval=5):
        """Wait for video generation to complete and return the finished operation."""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status = self.check_video_generation_status(operation_name, project_id)

            if status.get('done', False):
                return status

            time.sleep(poll_interval)

        raise TimeoutError(f"Video generation did not complete within {max_wait_time} seconds")

    def download_media(self, uri):
        """Download a generated file from an https URI (e.g. a Developer API Veo video) with the wrapper auth."""
        response = self._request('GET', uri, params={'alt': 'media'} if 'alt=' not in uri else None,
                                 error_prefix='Download error')
        return response.content

    # ------------------------------------------------------------------
    # Music (Lyria) and speech (Gemini TTS)
    # ------------------------------------------------------------------
    def generate_music(self, prompt, model=None, *, negative_prompt=None, seed=None, sample_count=None,
                       generation_config=None, location=None):
        """
        Generate music. Lyria 2 models (lyria-002, Vertex AI) use :predict and return
        predictions[].bytesBase64Encoded WAV. Lyria 3 models (e.g. lyria-3.5) use generateContent
        and return audio parts. Read the audio with extract_audio(response).
        """
        model = model or self._default_model('music')
        if '/' not in model and re.match(r'lyria-0\d\d', model):
            if not self.vertex:
                raise GoogleAIError(f"{model} runs on Vertex AI. Create the wrapper with vertex=True, "
                                    "or use a Lyria 3 model on the Developer API.")
            instance = {'prompt': prompt}
            if negative_prompt:
                instance['negative_prompt'] = negative_prompt
            if seed is not None:
                instance['seed'] = seed
            body = {'instances': [instance]}
            if sample_count:
                body['parameters'] = {'sample_count': sample_count}
            url = self._model_url(model, 'predict', self._location_for('music', location))
            return self._request_json('POST', url, body, alias=False, error_prefix='Lyria music error')
        text = prompt if not negative_prompt else f"{prompt}\nAvoid: {negative_prompt}"
        config_values = {'responseModalities': ['AUDIO', 'TEXT']}
        if generation_config:
            config_values.update(generation_config)
        if seed is not None:
            config_values.setdefault('seed', seed)
        body = self._prepare_body({'contents': [{'role': 'user', 'parts': [{'text': text}]}],
                                   'generationConfig': config_values})
        return self._request_json('POST', self._model_url(model, 'generateContent', location), body,
                                  error_prefix='Lyria music error')

    def generate_gemini_speech(self, text, voice_config=None, model_override=None, *, voice=None):
        """
        Generate speech with a Gemini TTS model. Returns the response; read the audio with
        extract_audio(response) and audio_to_wav(item). (GoogleAIWrapper.generate_speech is Cloud TTS.)
        """
        model = model_override or self._default_model('tts')

        default_voice_config = {
            "prebuilt_voice_config": {
                "voice_name": voice or "Kore"
            }
        }

        if voice_config:
            default_voice_config.update(voice_config)

        params = {
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": default_voice_config
                }
            }
        }
        return self._request_json('POST', self._model_url(model, 'generateContent'), self._prepare_body(params),
                                  error_prefix='Gemini TTS error')

    def generate_multi_speaker_speech(self, text, speaker_configs, model_override=None):
        """Generate multi-speaker speech"""
        model = model_override or self._default_model('tts')

        params = {
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "multi_speaker_voice_config": {
                        "speaker_voice_configs": speaker_configs
                    }
                }
            }
        }
        return self._request_json('POST', self._model_url(model, 'generateContent'), self._prepare_body(params),
                                  error_prefix='Gemini Multi-Speaker TTS error')

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    @staticmethod
    def _content_text(content):
        if isinstance(content, str):
            return content
        parts = (content or {}).get('parts') or []
        return "\n".join(p.get('text', '') for p in parts if isinstance(p, dict) and p.get('text'))

    def _vertex_embed(self, model_id, texts, task_type=None, title=None, output_dimensionality=None):
        """Vertex AI embeddings through :predict. gemini-embedding-001 takes one input per request."""
        vectors = []
        batch_size = 1 if model_id.startswith('gemini-embedding') else 250
        for start in range(0, len(texts), batch_size):
            instances = []
            for text in texts[start:start + batch_size]:
                instance = {'content': text}
                if task_type:
                    instance['task_type'] = task_type
                if title:
                    instance['title'] = title
                instances.append(instance)
            body = {'instances': instances}
            if output_dimensionality:
                body['parameters'] = {'outputDimensionality': output_dimensionality}
            data = self._request_json('POST', self._model_url(model_id, 'predict'), body, alias=False,
                                      error_prefix='Gemini API error')
            for prediction in data.get('predictions') or []:
                vectors.append(((prediction or {}).get('embeddings') or {}).get('values', []))
        return vectors

    def get_embeddings(self, params):
        """
        Get one embedding: params {'model'?, 'content': {'parts': [{'text': ...}]}, 'taskType'?,
        'outputDimensionality'?}. Returns {'embedding': {'values': [...]}} on both backends.
        """
        params = dict(params) if isinstance(params, dict) else params
        # Honor a per-call model (accept 'gemini-embedding-001' or 'models/...');
        # fall back to the configured default when not provided.
        requested = params.get('model') if isinstance(params, dict) else None
        model_id = (requested or self._default_model('embedding')).split('/')[-1]
        if self.vertex:
            vectors = self._vertex_embed(
                model_id, [self._content_text(params.get('content'))],
                task_type=params.get('taskType') or params.get('task_type'), title=params.get('title'),
                output_dimensionality=params.get('outputDimensionality') or params.get('output_dimensionality'))
            return {'embedding': {'values': vectors[0] if vectors else []}}
        # embedContent expects the body 'model' in 'models/<id>' form.
        if isinstance(params, dict):
            params['model'] = f"models/{model_id}"
        return self._request_json('POST', self._model_url(model_id, 'embedContent'), self._camelize(params),
                                  error_prefix='Gemini API error')

    def get_batch_embeddings(self, params):
        """Get batch embeddings for multiple texts. Returns [{'values': [...]}, ...]."""
        model = self._default_model('embedding')

        if self.vertex:
            requests_list = params.get('requests', []) if isinstance(params, dict) else []
            texts = [self._content_text(req.get('content')) for req in requests_list]
            return [{'values': values} for values in self._vertex_embed(model.split('/')[-1], texts)]

        # Format according to the documentation
        if "requests" in params:
            batch_params = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": req.get("content", {})
                    } for req in params["requests"]
                ]
            }
        else:
            batch_params = params

        data = self._request_json('POST', self._model_url(model, 'batchEmbedContents'),
                                  self._camelize(batch_params), error_prefix='Gemini API error')
        return data.get("embeddings", [])

    def embed_texts(self, texts, model=None, *, task_type=None, title=None, output_dimensionality=None):
        """Embed a list of texts and return a list of vectors (lists of floats)."""
        if isinstance(texts, str):
            texts = [texts]
        model_id = (model or self._default_model('embedding')).split('/')[-1]
        if self.vertex:
            return self._vertex_embed(model_id, list(texts), task_type, title, output_dimensionality)
        requests_list = []
        for text in texts:
            request = {'model': f'models/{model_id}', 'content': {'parts': [{'text': text}]}}
            if task_type:
                request['taskType'] = task_type
            if title:
                request['title'] = title
            if output_dimensionality:
                request['outputDimensionality'] = output_dimensionality
            requests_list.append(request)
        data = self._request_json('POST', self._model_url(model_id, 'batchEmbedContents'),
                                  {'requests': requests_list}, alias=False, error_prefix='Gemini API error')
        return [item.get('values', []) for item in data.get('embeddings', [])]

    # ------------------------------------------------------------------
    # Files API (Gemini Developer API only)
    # ------------------------------------------------------------------
    def _developer_api_only(self, feature):
        if self.vertex:
            raise NotImplementedError(
                f"{feature} is only available on the Gemini Developer API. On Vertex AI send files inline "
                "(media_part with bytes or a path) or as gs:// / https URIs.")

    def upload_file(self, file_path, display_name=None):
        """Upload a file using the Files API"""
        self._developer_api_only("The Files API")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        mime_type = self._get_mime_type(file_path)
        display_name = display_name or os.path.basename(file_path)

        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        }
        metadata = {"file": {"display_name": display_name}}
        response = self._request('POST', self._dev_upload_base, metadata, headers=headers,
                                 error_prefix='File upload error')

        upload_url = response.headers.get('x-goog-upload-url')
        if not upload_url:
            raise GoogleAIError("File upload error: upload URL not found in response headers")

        with open(file_path, 'rb') as f:
            file_content = f.read()

        upload_headers = {
            "Content-Length": str(file_size),
            "Content-Type": mime_type,
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize"
        }
        try:
            upload_response = self.session.post(upload_url, headers=upload_headers, data=file_content,
                                                timeout=self.timeout)
            upload_response.raise_for_status()
        except requests.exceptions.RequestException as error:
            raise self._api_error('File upload error', error) from None
        return upload_response.json()

    def list_files(self):
        """List uploaded files"""
        self._developer_api_only("The Files API")
        return self._request_json('GET', self._dev_files_base, alias=False, error_prefix='List files error')

    def get_file(self, file_name):
        """Get the metadata (state, uri) of an uploaded file ('files/<id>' or '<id>')."""
        self._developer_api_only("The Files API")
        return self._request_json('GET', self._file_url(file_name), alias=False, error_prefix='Get file error')

    def delete_file(self, file_name):
        """Delete an uploaded file ('files/<id>' or '<id>')."""
        self._developer_api_only("The Files API")
        response = self._request('DELETE', self._file_url(file_name), error_prefix='Delete file error')
        return response.json() if response.content else {"status": "deleted"}

    def _file_url(self, file_name):
        if file_name.startswith('files/'):
            return f"{self._dev_api_base}/{file_name}"
        return f"{self._dev_files_base}/{file_name}"

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def list_models(self, page_size=None, page_token=None):
        """
        List models. Developer API: works with an API key. Vertex AI: lists Google publisher
        models and needs OAuth (ADC); API keys are rejected there, use model_catalog() instead.
        """
        params = {}
        if page_size:
            params['pageSize'] = page_size
        if page_token:
            params['pageToken'] = page_token
        if self.vertex:
            url = f"{self.base_url or self._vertex_host(None) + '/' + self.api_version}/publishers/google/models"
        else:
            url = f"{self.base_url or self._dev_api_base}/models"
        return self._request_json('GET', url, params=params or None, alias=False, error_prefix='List models error')

    @staticmethod
    def model_catalog():
        """Known Vertex AI model ids by capability (checked against the Agent Platform docs)."""
        return copy.deepcopy(config['url']['gemini'].get('vertex', {}).get('catalog', {}))

    # ------------------------------------------------------------------
    # Context caching
    # ------------------------------------------------------------------
    def _cache_url(self, name=None, location=None):
        if self.vertex:
            if name and name.startswith('projects/'):
                return self._name_url(name)
            project = self._require_project('Context caching')
            location = location or self.location or 'global'
            path = f"cachedContents/{name.split('/')[-1]}" if name else 'cachedContents'
            return self._vertex_project_url(project, location, path)
        if name:
            return f"{self._dev_api_base}/{name if name.startswith('cachedContents/') else 'cachedContents/' + name}"
        return f"{self._dev_api_base}/cachedContents"

    def create_cached_content(self, model, contents, *, system_instruction=None, ttl='3600s', display_name=None,
                              tools=None, tool_config=None, location=None):
        """
        Cache a large prompt prefix and reuse it with generate_content({'cachedContent': name, ...}).
        Vertex AI needs a project and OAuth (API keys are rejected); the Developer API works with a key.
        """
        body = self._prepare_body({'contents': contents})
        if self.vertex:
            project = self._require_project('Context caching')
            location = location or self.location or 'global'
            model_path = self._vertex_model_path(model)
            body['model'] = (model_path if model_path.startswith('projects/')
                             else f"projects/{project}/locations/{location}/{model_path}")
        else:
            body['model'] = self._model_path(model)
        if system_instruction:
            body['systemInstruction'] = ({'parts': [{'text': system_instruction}]}
                                         if isinstance(system_instruction, str) else system_instruction)
        if ttl:
            body['ttl'] = ttl
        if display_name:
            body['displayName'] = display_name
        if tools:
            body['tools'] = tools
        if tool_config:
            body['toolConfig'] = tool_config
        return self._request_json('POST', self._cache_url(location=location), body, alias=False,
                                  error_prefix='Context cache error')

    def get_cached_content(self, name):
        return self._request_json('GET', self._cache_url(name), alias=False, error_prefix='Context cache error')

    def list_cached_contents(self, page_size=None, page_token=None, location=None):
        params = {}
        if page_size:
            params['pageSize'] = page_size
        if page_token:
            params['pageToken'] = page_token
        return self._request_json('GET', self._cache_url(location=location), params=params or None, alias=False,
                                  error_prefix='Context cache error')

    def delete_cached_content(self, name):
        response = self._request('DELETE', self._cache_url(name), error_prefix='Context cache error')
        return response.json() if response.content else {"status": "deleted"}

    # ------------------------------------------------------------------
    # Agent Engine (deployed agents, Vertex AI)
    # ------------------------------------------------------------------
    def _agent_engine_name(self, name, location=None):
        if name.startswith('projects/'):
            return name
        project = self._require_project('Agent Engine')
        location = self._location_for('agent_engine', location) or 'us-central1'
        return f"projects/{project}/locations/{location}/reasoningEngines/{name}"

    def list_agent_engines(self, location=None, page_size=None, page_token=None, filter=None):
        """List the agents deployed to Agent Engine in the project."""
        project = self._require_project('Agent Engine')
        location = self._location_for('agent_engine', location) or 'us-central1'
        params = {}
        if page_size:
            params['pageSize'] = page_size
        if page_token:
            params['pageToken'] = page_token
        if filter:
            params['filter'] = filter
        return self._request_json('GET', self._vertex_project_url(project, location, 'reasoningEngines'),
                                  params=params or None, alias=False, error_prefix='Agent Engine error')

    def get_agent_engine(self, name, location=None):
        """Get one deployed agent (id or full resource name)."""
        return self._request_json('GET', self._name_url(self._agent_engine_name(name, location)), alias=False,
                                  error_prefix='Agent Engine error')

    def query_agent_engine(self, name, input=None, class_method=None, location=None):
        """Call a deployed agent's query method: returns {'output': ...}."""
        body = {'input': input or {}}
        if class_method:
            body['classMethod'] = class_method
        url = f"{self._name_url(self._agent_engine_name(name, location))}:query"
        return self._request_json('POST', url, body, alias=False, error_prefix='Agent Engine error')

    def stream_query_agent_engine(self, name, input=None, class_method='stream_query', location=None):
        """Call a deployed agent's streaming method and yield each event (dict, or text when not JSON)."""
        body = {'input': input or {}}
        if class_method:
            body['classMethod'] = class_method
        url = f"{self._name_url(self._agent_engine_name(name, location))}:streamQuery"
        response = self._request('POST', url, body, params={'alt': 'sse'}, stream=True,
                                 error_prefix='Agent Engine error')
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith('data:'):
                    line = line[5:].strip()
                try:
                    yield json.loads(line)
                except ValueError:
                    yield line
        finally:
            response.close()

    # ------------------------------------------------------------------
    # Live API (bidirectional streaming over a websocket)
    # ------------------------------------------------------------------
    def _live_endpoint(self, model=None, location=None):
        model = model or self._default_model('live')
        if self.vertex:
            project = self._require_project('The Live API')
            location = self._location_for('live', location) or 'us-central1'
            host = self._vertex_host(location).replace('https://', 'wss://', 1)
            url = f"{host}/ws/google.cloud.aiplatform.{self.api_version}.LlmBidiService/BidiGenerateContent"
            model_path = self._vertex_model_path(model)
            model_name = (model_path if model_path.startswith('projects/')
                          else f"projects/{project}/locations/{location}/{model_path}")
        else:
            host = re.match(r'https://[^/]+', self._dev_api_base).group(0).replace('https://', 'wss://', 1)
            url = f"{host}/ws/google.ai.generativelanguage.{self.api_version}.GenerativeService.BidiGenerateContent"
            model_name = self._model_path(model)
        return url, self._auth_headers(), model_name

    @contextlib.asynccontextmanager
    async def live_connect(self, model=None, config=None, location=None):
        """
        Open a Live API session (async context manager) and yield a GoogleAILiveSession.

        config is the setup message without 'model', e.g. {"generationConfig": {"responseModalities": ["AUDIO"]},
        "systemInstruction": {...}, "outputAudioTranscription": {}}. Vertex AI needs project_id; the
        Live models run in us-central1.
        """
        try:
            import websockets
        except ImportError:
            raise GoogleAIError("The Live API needs the websockets package: pip install websockets") from None
        url, headers, model_name = self._live_endpoint(model, location)
        setup = {'model': model_name}
        setup.update(self._camelize(config or {}))
        generation_config = setup.setdefault('generationConfig', {})
        generation_config.setdefault('responseModalities', ['AUDIO'])

        async def _open():
            try:
                return await websockets.connect(url, additional_headers=headers, max_size=None,
                                                open_timeout=self.timeout)
            except TypeError:
                return await websockets.connect(url, extra_headers=headers, max_size=None,
                                                open_timeout=self.timeout)

        try:
            websocket = await _open()
        except Exception as error:
            raise GoogleAIError(self._redact(f"Live API connection error: {error}")) from None
        try:
            try:
                await websocket.send(json.dumps({'setup': setup}))
                first = json.loads(await websocket.recv())
            except Exception as error:
                raise GoogleAIError(self._redact(f"Live API setup error: {error}")) from None
            if 'setupComplete' not in first:
                raise GoogleAIError(self._redact(f"Live API setup failed: {first}"))
            yield GoogleAILiveSession(websocket, first)
        finally:
            await websocket.close()

    async def live_generate_async(self, text, model=None, config=None, location=None):
        """Send one text turn over the Live API and return {'text', 'transcription', 'audio', ...}."""
        setup_config = self._camelize(config or {})
        modalities = (setup_config.get('generationConfig') or {}).get('responseModalities', ['AUDIO'])
        if 'AUDIO' in modalities:
            setup_config.setdefault('outputAudioTranscription', {})
        async with self.live_connect(model, setup_config, location) as session:
            await session.send_text(text)
            return await session.receive_turn()

    def live_generate(self, text, model=None, config=None, location=None):
        """Blocking version of live_generate_async (do not call it from inside a running event loop)."""
        import asyncio
        return asyncio.run(self.live_generate_async(text, model, config, location))


class GoogleAIChatSession:
    """
    Multi-turn chat for GoogleAIWrapper. The history keeps the model turns exactly as returned,
    including thought signatures, which Gemini 3 needs for multi-turn tool use and image editing.
    """

    def __init__(self, wrapper, model=None, system_instruction=None, generation_config=None, tools=None,
                 tool_config=None, safety_settings=None, history=None):
        self.wrapper = wrapper
        self.model = model or wrapper._default_model('text')
        self.system_instruction = system_instruction
        self.generation_config = generation_config
        self.tools = tools
        self.tool_config = tool_config
        self.safety_settings = safety_settings
        self.history = [wrapper._content(c) for c in (history or [])]
        self.last_response = None

    def _body(self, user_content):
        return self.wrapper._build_request(None, self.system_instruction, None, self.generation_config, self.tools,
                                           self.tool_config, self.safety_settings,
                                           history=self.history + [user_content])

    def _user_content(self, message=None, media=None, parts=None):
        return {'role': 'user', 'parts': parts if parts is not None else self.wrapper._to_parts(message, media)}

    def send(self, message=None, media=None, *, parts=None):
        """Send a user turn and return the full response. The turn and the reply are added to history."""
        user_content = self._user_content(message, media, parts)
        response = self.wrapper.generate_content(self._body(user_content), model=self.model)
        self.last_response = response
        self.history.append(user_content)
        candidates = response.get('candidates') or []
        model_content = (candidates[0] or {}).get('content') if candidates else None
        if model_content and model_content.get('parts'):
            model_content = self.wrapper._unalias(model_content)
            model_content.setdefault('role', 'model')
            self.history.append(model_content)
        return response

    def send_text(self, message=None, media=None):
        """Send a user turn and return only the reply text."""
        return GoogleAIWrapper.extract_text(self.send(message, media))

    def send_function_response(self, name, response, call_id=None):
        """Return a tool result to the model after it asked for a function call."""
        function_response = {'name': name, 'response': response}
        if call_id:
            function_response['id'] = call_id
        return self.send(parts=[{'functionResponse': function_response}])

    def stream(self, message=None, media=None):
        """Yield reply text chunks; the turn is added to history when the stream ends."""
        user_content = self._user_content(message, media)
        parts = []
        for chunk in self.wrapper.stream_generate_content(self._body(user_content), model=self.model):
            for part in GoogleAIWrapper._iter_parts(chunk):
                parts.append(self.wrapper._unalias(part))
            text = GoogleAIWrapper.extract_text(chunk)
            if text:
                yield text
        self.history.append(user_content)
        if parts:
            self.history.append({'role': 'model', 'parts': self._merge_text_parts(parts)})

    @staticmethod
    def _merge_text_parts(parts):
        merged = []
        for part in parts:
            plain_text = set(part.keys()) == {'text'}
            if plain_text and merged and set(merged[-1].keys()) == {'text'}:
                merged[-1] = {'text': merged[-1]['text'] + part['text']}
            else:
                merged.append(dict(part))
        return merged

    def reset(self):
        self.history = []


class GoogleAILiveSession:
    """An open Live API session. Use it inside `async with wrapper.live_connect(...) as session`."""

    def __init__(self, websocket, setup_response=None):
        self._ws = websocket
        self.setup_response = setup_response or {}

    async def send(self, message):
        """Send a raw client message (setup is already done)."""
        await self._ws.send(json.dumps(message))

    async def send_text(self, text, turn_complete=True):
        await self.send({'clientContent': {'turns': [{'role': 'user', 'parts': [{'text': text}]}],
                                           'turnComplete': turn_complete}})

    async def send_audio(self, data, mime_type='audio/pcm;rate=16000'):
        """Stream a chunk of microphone audio (16-bit PCM)."""
        if isinstance(data, (bytes, bytearray)):
            data = base64.b64encode(bytes(data)).decode('utf-8')
        await self.send({'realtimeInput': {'audio': {'data': data, 'mimeType': mime_type}}})

    async def send_audio_stream_end(self):
        await self.send({'realtimeInput': {'audioStreamEnd': True}})

    async def send_tool_response(self, function_responses):
        await self.send({'toolResponse': {'functionResponses': function_responses}})

    async def receive(self):
        """Yield every server message as a dict."""
        async for raw in self._ws:
            yield json.loads(raw)

    async def receive_turn(self):
        """Collect messages until the model finishes its turn (or asks for a tool call)."""
        result = {'text': '', 'transcription': '', 'input_transcription': '', 'audio': b'',
                  'audio_mime_type': None, 'tool_calls': [], 'usage': None, 'messages': 0}
        async for message in self.receive():
            result['messages'] += 1
            if message.get('usageMetadata'):
                result['usage'] = message['usageMetadata']
            if message.get('toolCall'):
                result['tool_calls'].extend(message['toolCall'].get('functionCalls') or [])
                break
            content = message.get('serverContent') or {}
            for part in (content.get('modelTurn') or {}).get('parts') or []:
                if part.get('text') and not part.get('thought'):
                    result['text'] += part['text']
                inline = part.get('inlineData')
                if inline and inline.get('data'):
                    result['audio'] += base64.b64decode(inline['data'])
                    result['audio_mime_type'] = inline.get('mimeType')
            if (content.get('outputTranscription') or {}).get('text'):
                result['transcription'] += content['outputTranscription']['text']
            if (content.get('inputTranscription') or {}).get('text'):
                result['input_transcription'] += content['inputTranscription']['text']
            if content.get('turnComplete'):
                break
        return result

    async def close(self):
        await self._ws.close()
