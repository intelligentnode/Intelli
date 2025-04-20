import json
import requests
import base64

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
        url = self.api_speech_url + config['url']['google']['speech']['synthesize']['postfix']
        param = self.get_synthesize_input(params)

        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(param))
            response.raise_for_status()
            return response.json()['audioContent']
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def get_synthesize_input(self, params):
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
                'audioEncoding': 'MP3',
            },
        }

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
            {'type': 'TEXT_DETECTION', 'maxResults': 1}
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
