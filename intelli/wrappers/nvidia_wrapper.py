import requests
from intelli.config import config


class NvidiaWrapper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = config["url"]["nvidia"]["base"]
        self.chat_endpoint = config["url"]["nvidia"]["chat"]
        self.embeddings_endpoint = config["url"]["nvidia"]["embeddings"]
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def generate_text(self, params: dict) -> dict:
        if "stream" not in params:
            params["stream"] = False
        url = self.base_url + self.chat_endpoint
        response = requests.post(url, json=params, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def generate_text_stream(self, params: dict):
        params["stream"] = True
        url = self.base_url + self.chat_endpoint
        response = requests.post(url, json=params, headers=self.headers, stream=True)
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                yield line

    def get_embeddings(self, params: dict) -> dict:
        url = self.base_url + self.embeddings_endpoint
        response = requests.post(url, json=params, headers=self.headers)
        response.raise_for_status()
        return response.json()
