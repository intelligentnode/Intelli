import os
import requests
from urllib.parse import urljoin

from intelli.utils.conn_helper import ConnHelper
from intelli.utils.proxy_helper import ProxyHelper


class OpenAIWrapper:

    def __init__(self, api_key, proxy_helper=None):
        self.api_key = api_key
        self.proxy_helper = proxy_helper or ProxyHelper.get_instance()
        # set the headers
        if self.proxy_helper.get_openai_type() == 'azure':
            print('Set OpenAI Azure settings')
            if not self.proxy_helper.get_openai_resource_name():
                raise ValueError('Set your Azure resource name')
            headers = {
                'Content-Type': 'application/json',
                'api-key': f'{self.api_key}',
            }
        else:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            }
            # check if organization exists for non-Azure
            organization = self.proxy_helper.get_openai_organization()
            if organization:
                headers['OpenAI-Organization'] = organization

        # define the connection session
        self.session = BaseURLSession(base_url=self.proxy_helper.get_openai_url())
        self.session.headers.update(headers)

    def generate_chat_text(self, params, functions=None, function_call=None):
        url = self.proxy_helper.get_openai_chat_url(params['model'])
        payload = params.copy()
        if functions:
            payload['functions'] = functions
        if function_call:
            payload['function_call'] = function_call

        try:
            response = self.session.post(url, json=payload, stream=params.get('stream', False))
            response.raise_for_status()
            if params.get('stream', False):
                return response.iter_lines(decode_unicode=True)
            else:
                return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            self.session.close()

    def generate_images(self, params):
        url = self.proxy_helper.get_openai_image_url()
        try:
            response = self.session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            self.session.close()

    def upload_file(self, file_path, purpose):

        url = urljoin(self.proxy_helper.openai_url, self.proxy_helper.get_openai_files_url())

        with open(file_path, 'rb') as file:
            files = {
                'file': (os.path.basename(file_path), file, 'application/jsonl')
            }
            data = {'purpose': purpose}
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }

            # make direct post request due to conflicts in common content type
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json()

    def store_fine_tuning_data(self, params):
        url = self.proxy_helper.get_openai_finetuning_job_url()
        try:
            response = self.session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))

    def list_fine_tuning_data(self):
        url = self.proxy_helper.get_openai_finetuning_job_url()
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))

    def get_embeddings(self, params):
        url = self.proxy_helper.get_openai_embed_url(params['model'])
        try:
            response = self.session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))

    def speech_to_text(self, params, headers=None):
        url = self.proxy_helper.get_openai_audio_transcriptions_url(params.get('model', ''))
        try:
            custom_headers = self.session.headers.copy()
            if headers:
                custom_headers.update(headers)
            response = self.session.post(url, data=params, headers=custom_headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))

    def text_to_speech(self, params, headers=None):
        url = self.proxy_helper.get_openai_audio_speech_url(params['model'])
        try:
            custom_headers = self.session.headers.copy()
            if headers:
                custom_headers.update(headers)
            response = self.session.post(url, json=params, headers=custom_headers, stream=True)
            response.raise_for_status()
            return response.iter_content(chunk_size=8192) if params.get('stream', False) else response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))

    def image_to_text(self, params, headers=None):
        url = self.proxy_helper.get_openai_chat_url(params['model'])

        if headers:
            combined_headers = {**self.session.headers, **headers}
        else:
            combined_headers = self.session.headers

        try:
            response = self.session.post(url, json=params, headers=combined_headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))


class BaseURLSession(requests.Session):
    def __init__(self, base_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = base_url

    def request(self, method, url, *args, **kwargs):
        # Prepend the base URL if it's defined and the URL is relative
        if self.base_url and not url.startswith(("http://", "https://")):
            url = f"{self.base_url}{url}"
        return super().request(method, url, *args, **kwargs)
