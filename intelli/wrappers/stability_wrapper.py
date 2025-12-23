import os
import requests

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class StabilityAIWrapper:
    def __init__(self, api_key, timeout=180):
        self.api_base_url = config['url']['stability']['base']
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
        })

    def generate_images(self, params, engine='stable-diffusion-xl-1024-v1-0'):
        endpoint = config['url']['stability']['text_to_image'].format(engine)
        url = f"{self.api_base_url}{endpoint}"
        try:
            response = self.session.post(url, json=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))
