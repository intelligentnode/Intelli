import requests
import os
from config import config
from utils.conn_helper import ConnHelper

class StabilityAIWrapper:
    def __init__(self, api_key):
        self.api_base_url = config['stability']['base']
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
        })

    def generate_text_to_image(self, params, engine='stable-diffusion-xl-beta-v2-2-2'):
        endpoint = config['stability']['text_to_image'].format(engine)
        url = f"{self.api_base_url}{endpoint}"
        try:
            response = self.session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

