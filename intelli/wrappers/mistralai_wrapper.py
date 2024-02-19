import requests

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class MistralAIWrapper:

    def __init__(self, api_key):
        self.API_BASE_URL = config['url']['mistral']['base']
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        })

    def generate_text(self, params):
        url = f"{self.API_BASE_URL}{config['url']['mistral']['completions']}"

        try:
            response = self.session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def get_embeddings(self, params):
        url = f"{self.API_BASE_URL}{config['url']['mistral']['embed']}"

        try:
            response = self.session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))
