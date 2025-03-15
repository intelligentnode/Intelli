import requests
import json

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class VLLMWrapper:

    def __init__(self, api_base_url, api_key=None):
        """
        Initialize the VLLM wrapper.

        Args:
            api_base_url (str): Base URL for the VLLM API.
            api_key (str, optional): API key for authentication. Defaults to None.
        """
        self.api_base_url = api_base_url
        self.session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)
        self.is_log = False

    def generate_text(self, params):
        """
        Generate text completions using VLLM API.

        Args:
            params (dict): Parameters for the text generation request.
                Example: {
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "prompt": "What is machine learning?",
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "stream": False
                }

        Returns:
            dict or generator: If stream=False, returns the JSON response from the API.
                              If stream=True, returns a generator yielding text chunks.
        """
        endpoint = f"{self.api_base_url}{config['url']['vllm']['completions']}"

        try:
            if params.get("stream", False):
                response = self.session.post(endpoint, json=params, stream=True)
                response.raise_for_status()
                return response.iter_lines(decode_unicode=True)
            else:
                response = self.session.post(endpoint, json=params)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def generate_text_stream(self, params):
        """
        Generate text completions using VLLM API with streaming support.

        Args:
            params (dict): Parameters for the text generation request.

        Yields:
            str: Text chunks as they are generated.
        """
        if not params.get("stream", False):
            params["stream"] = True

        for line in self.generate_text(params):
            if not line:
                continue

            # Process the line with "data:" prefix
            if line.startswith("data:"):
                # Skip the [DONE] message
                if "DONE" in line:
                    continue

                # Extract the JSON part
                json_str = line[5:].strip()  # Remove "data: " prefix

                try:
                    data = json.loads(json_str)

                    # Extract text from choices
                    if "choices" in data and len(data["choices"]) > 0:
                        text = data["choices"][0].get("text", "")
                        if text:
                            if self.is_log:
                                print(f"Extracted text: {text}")
                            yield text
                except json.JSONDecodeError as e:
                    if self.is_log:
                        print(f"JSON decode error: {e}")

    def generate_chat_text(self, params):
        """
        Generate chat completions using VLLM API.

        Args:
            params (dict): Parameters for the chat completion request.
                Example: {
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is machine learning?"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "stream": False
                }

        Returns:
            dict or generator: If stream=False, returns the JSON response from the API.
                              If stream=True, returns a generator yielding text chunks.
        """
        endpoint = f"{self.api_base_url}{config['url']['vllm']['chat']}"

        try:
            if params.get("stream", False):
                response = self.session.post(endpoint, json=params, stream=True)
                response.raise_for_status()
                return response.iter_lines(decode_unicode=True)
            else:
                response = self.session.post(endpoint, json=params)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))

    def generate_chat_text_stream(self, params):
        """
        Generate chat completions using VLLM API with streaming support.

        Args:
            params (dict): Parameters for the chat completion request.

        Yields:
            str: Text chunks as they are generated.
        """
        if not params.get("stream", False):
            params["stream"] = True

        for line in self.generate_chat_text(params):
            if not line:
                continue

            # Process the line with "data:" prefix
            if line.startswith("data:"):

                if "DONE" in line:
                    continue

                # Extract the JSON part
                json_str = line[5:].strip()  # Remove "data: " prefix

                try:
                    data = json.loads(json_str)

                    # Extract content from choices
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]

                        # Delta format for chat
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content:
                                if self.is_log:
                                    print(f"Extracted content: {content}")
                                yield content
                except json.JSONDecodeError as e:
                    if self.is_log:
                        print(f"JSON decode error: {e}")

    def get_embeddings(self, params):
        """
        Get embeddings for a list of texts using VLLM API.

        Args:
            params (dict): Parameters for the embedding request.
                Should contain a "texts" key with a list of strings to embed.
                May optionally contain a "model" key to specify the embedding model.

        Returns:
            dict: JSON response from the API containing the embeddings.
        """
        endpoint = f"{self.api_base_url}{config['url']['vllm']['embed']}"

        try:
            response = self.session.post(endpoint, json=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))
