import base64
import requests
import json

from intelli.config import config


class GeminiAIWrapper:

    def __init__(self, api_key):
        self.API_BASE_URL = config['url']['gemini']['base']
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        self.API_KEY = api_key
        self.models = config['url']['gemini']['models']
        self.endpoints = config['url']['gemini']['endpoints']

    def generate_content(self, params, vision=False):
        model = self.models['vision'] if vision else self.models['text']
        url = f"{self.API_BASE_URL}/{model}{self.endpoints['generateContent']}"

        try:
            response = self.session.post(url, json=params, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            # Add better error handling
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini API error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(f"Gemini API error: {error}")
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
        model = self.models['embedding']
        url = f"{self.API_BASE_URL}/{model}:embedContent"

        try:
            response = self.session.post(url, json=params, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini API error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(str(error))

    def get_batch_embeddings(self, params):
        model = self.models['embedding']
        url = f"{self.API_BASE_URL}/{model}:batchEmbedContents"

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

        try:
            response = self.session.post(url, json=batch_params, params={'key': self.API_KEY})
            response.raise_for_status()
            return response.json().get("embeddings", [])
        except requests.exceptions.RequestException as error:
            if hasattr(error, 'response') and error.response:
                try:
                    error_detail = error.response.json()
                    raise Exception(f"Gemini API error: {error} - Details: {json.dumps(error_detail)}")
                except:
                    pass
            raise Exception(str(error))
