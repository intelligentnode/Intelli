import os
import requests

from intelli.utils.conn_helper import ConnHelper
from intelli.utils.proxy_helper import ProxyHelper


class OpenAIWrapper:

    def __init__(self, api_key, proxy_helper=None):
        self.api_key = api_key
        self.proxy_helper = proxy_helper or ProxyHelper.get_instance()
        # Build default headers once; sessions will be created per request.
        if self.proxy_helper.get_openai_type() == 'azure':
            print('Set OpenAI Azure settings')
            if not self.proxy_helper.get_openai_resource_name():
                raise ValueError('Set your Azure resource name')
            self._headers = {
                'Content-Type': 'application/json',
                'api-key': f'{self.api_key}',
            }
        else:
            self._headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            }
            # check if organization exists for non-Azure
            organization = self.proxy_helper.get_openai_organization()
            if organization:
                self._headers['OpenAI-Organization'] = organization

        self._base_url = self.proxy_helper.get_openai_url()

    def _new_session(self):
        """
        Create a fresh session per request.
        This avoids invalidating the wrapper instance (the old code closed a shared session).
        """
        session = BaseURLSession(base_url=self._base_url)
        session.headers.update(self._headers)
        return session

    def generate_chat_text(self, params, functions=None, function_call=None, tools=None, tool_choice=None):
        """
        Backwards compatible chat API:
        - Legacy: supports `functions` / `function_call` (function_call responses).
        - Modern: supports `tools` / `tool_choice` (tool_calls responses).
        Preference:
          - If caller supplies tools/tool_choice (either in params or as explicit args), we do NOT
            inject legacy functions/function_call fields.
        """
        url = self.proxy_helper.get_openai_chat_url(params['model'])
        payload = params.copy()

        # Prefer explicit args if provided.
        if tools is not None:
            payload['tools'] = tools
        if tool_choice is not None:
            payload['tool_choice'] = tool_choice

        # Legacy tool calling: only set if caller didn't opt into tools.
        if functions and 'tools' not in payload:
            payload['functions'] = functions
        if function_call is not None and 'tool_choice' not in payload and 'tools' not in payload:
            payload['function_call'] = function_call

        session = self._new_session()

        try:
            response = session.post(url, json=payload, stream=params.get('stream', False))
            response.raise_for_status()
            if params.get('stream', False):
                return response.iter_lines(decode_unicode=True)
            else:
                return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()

    def generate_images(self, params):
        url = self.proxy_helper.get_openai_image_url()
        session = self._new_session()
        try:
            response = session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()

    def upload_file(self, file_path, purpose):
        # Use relative URL so BaseURLSession preserves base path (important for Azure '/openai').
        url = self.proxy_helper.get_openai_files_url()

        with open(file_path, 'rb') as file:
            files = {
                'file': (os.path.basename(file_path), file, 'application/jsonl')
            }
            data = {'purpose': purpose}
            # Remove JSON content-type for multipart upload.
            headers = dict(self._headers)
            headers.pop('Content-Type', None)

            session = self._new_session()
            try:
                # Use session to keep consistent base URL handling and headers.
                response = session.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                return response.json()
            finally:
                session.close()

    def store_fine_tuning_data(self, params):
        url = self.proxy_helper.get_openai_finetuning_job_url()
        session = self._new_session()
        try:
            response = session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()

    def list_fine_tuning_data(self):
        url = self.proxy_helper.get_openai_finetuning_job_url()
        session = self._new_session()
        try:
            response = session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()

    def get_embeddings(self, params):
        url = self.proxy_helper.get_openai_embed_url(params['model'])
        session = self._new_session()
        try:
            response = session.post(url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()

    def speech_to_text(self, params, headers=None, files=None):
        """
        Convert speech to text using OpenAI's API.

        Args:
            params: Dictionary with parameters like 'model', 'language', etc.
            headers: Optional additional headers
            files: Dictionary containing file objects for multipart/form-data

        Returns:
            JSON response from OpenAI
        """
        url = self.proxy_helper.get_openai_audio_transcriptions_url(params.get('model', ''))
        session = self._new_session()
        try:
            custom_headers = session.headers.copy()
            if headers:
                custom_headers.update(headers)

            # For speech to text, we need to use files parameter for multipart/form-data
            if files:
                response = session.post(url, data=params, files=files, headers=custom_headers)
            else:
                response = session.post(url, data=params, headers=custom_headers)

            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()

    def text_to_speech(self, params, headers=None):
        url = self.proxy_helper.get_openai_audio_speech_url(params['model'])
        session = self._new_session()
        try:
            custom_headers = session.headers.copy()
            if headers:
                custom_headers.update(headers)
            response = session.post(url, json=params, headers=custom_headers, stream=True)
            response.raise_for_status()
            return response.iter_content(chunk_size=8192) if params.get('stream', False) else response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()

    def image_to_text(self, params, headers=None):
        url = self.proxy_helper.get_openai_chat_url(params['model'])
        session = self._new_session()

        if headers:
            combined_headers = {**session.headers, **headers}
        else:
            combined_headers = session.headers

        try:
            response = session.post(url, json=params, headers=combined_headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()

    def generate_gpt5_response(self, params, tools=None, tool_choice=None):
        """
        Generate responses using GPT-5 API (uses /v1/responses endpoint).
        
        Args:
            params: Dictionary with parameters like 'model', 'input', 'reasoning', etc.
            
        Returns:
            JSON response from OpenAI GPT-5
        """
        url = self.proxy_helper.get_openai_responses_url(params.get('model', ''))
        payload = params.copy()

        # New-ish API: adding new parameters is safe and backwards compatible.
        if tools is not None and 'tools' not in payload:
            payload['tools'] = tools
        if tool_choice is not None and 'tool_choice' not in payload:
            payload['tool_choice'] = tool_choice

        session = self._new_session()
        
        try:
            response = session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))
        finally:
            session.close()


class BaseURLSession(requests.Session):
    def __init__(self, base_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = base_url

    def request(self, method, url, *args, **kwargs):
        # Prepend the base URL if it's defined and the URL is relative
        if self.base_url and not url.startswith(("http://", "https://")):
            url = f"{self.base_url}{url}"
        return super().request(method, url, *args, **kwargs)
