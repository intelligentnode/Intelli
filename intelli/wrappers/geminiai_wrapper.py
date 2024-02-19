import base64
import requests

from intelli.config import config


class GeminiAIWrapper:

    def __init__(self, api_key):
        self.API_BASE_URL = config['url']['gemini']['base']
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        self.API_KEY = api_key

    def generate_content(self, params, vision=False):
        endpoint = config['url']['gemini']['visionEndpoint'] if vision else config['url']['gemini']['contentEndpoint']
        url = f"{self.API_BASE_URL}{endpoint}"

        try:
            response = self.session.post(url, json=params, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            raise Exception(str(error))
        except Exception as error:
            raise Exception(str(error))

    def image_to_text(self, user_input, image_data, extension):

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

        return self.image_to_text_params(params=params)

    def image_to_text_params(self, params):

        return self.generate_content(params, True)

    def get_embeddings(self, params):
        url = f"{self.API_BASE_URL}{config['url']['gemini']['embeddingEndpoint']}"

        try:
            response = self.session.post(url, json=params, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json().get('embedding', [])
        except requests.exceptions.RequestException as error:
            raise Exception(str(error))
        except Exception as error:
            raise Exception(str(error))

    def get_batch_embeddings(self, params):
        url = f"{self.API_BASE_URL}{config['url']['gemini']['batchEmbeddingEndpoint']}"

        try:
            response = self.session.post(url, json=params, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json().get('embeddings', [])
        except requests.exceptions.RequestException as error:
            raise Exception(str(error))
        except Exception as error:
            raise Exception(str(error))
