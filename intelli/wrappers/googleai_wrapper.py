import json
import requests

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class GoogleAIWrapper:

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-Goog-Api-Key': self.api_key,
        }
        self.api_speech_url = config['url']['google']['base'].format(config['url']['google']['speech']['prefix'])

    def generate_speech(self, params):
        url = self.api_speech_url + config['url']['google']['speech']['synthesize']['postfix']
        param = self.get_synthesize_input(params)

        response = requests.post(url, headers=self.headers, data=json.dumps(param))
        response.raise_for_status()
        return response.json()['audioContent']

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
