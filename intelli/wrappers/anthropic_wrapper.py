import requests

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class AnthropicWrapper:
    def __init__(self, api_key):
        self.API_BASE_URL = config['url']['anthropic']['base']
        self.API_VERSION = config['url']['anthropic']['version']
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': self.API_VERSION
        })

    def generate_text(self, params):
        url = f"{self.API_BASE_URL}{config['url']['anthropic']['messages']}"
        response = self.session.post(url, json=params)
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            response.close()
            self.session.close()

    def stream_text(self, params):
        """Yields text from streaming API."""
        url = f"{self.API_BASE_URL}{config['url']['anthropic']['messages']}"
        headers = self.session.headers.copy()
        params['stream'] = True
        try:
            with requests.post(url, headers=headers, json=params, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        yield decoded_line
        except requests.exceptions.RequestException as error:
            raise Exception(f"Stream request failed: {error}")
