import json
import requests
import base64
from typing import List, Dict, Any, Optional, Union

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class GoogleAIWrapper:

    def __init__(self, api_key):
        self.api_key = api_key
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

    # Text-to-Speech methods
    def generate_speech(self, params):
        """Generate speech using Google Text-to-Speech API"""
        url = self.api_speech_url + config['url']['google']['speech']['synthesize']['postfix']
        param = self.get_synthesize_input(params)

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(param))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(param))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
            response = requests.get(url, headers=self.headers, params=params)
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
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
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
