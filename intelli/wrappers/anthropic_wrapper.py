import requests

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class AnthropicWrapper:
    def __init__(self, api_key, timeout=180):
        self.API_BASE_URL = config['url']['anthropic']['base']
        self.API_VERSION = config['url']['anthropic']['version']
        self.timeout = timeout
        self._headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': self.API_VERSION,
        }
        # Kept for backward compatibility; reused across calls and NOT closed
        # per-request (closing a shared Session broke the second call).
        self.session = requests.Session()
        self.session.headers.update(self._headers)

    def _build_headers(self, extra_headers=None):
        headers = dict(self._headers)
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def generate_text(self, params, extra_headers=None):
        """
        Call the Messages API.

        Args:
            params: Messages API request body.
            extra_headers: Optional dict of additional headers (e.g.
                {'anthropic-beta': 'context-1m-2025-08-07'}) to enable beta
                features per request. Additive and backward compatible.
        """
        url = f"{self.API_BASE_URL}{config['url']['anthropic']['messages']}"
        try:
            response = requests.post(
                url, headers=self._build_headers(extra_headers), json=params, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            raise Exception(ConnHelper.get_error_message(error))

    def stream_text(self, params, extra_headers=None):
        """Yields raw SSE lines from the streaming Messages API."""
        url = f"{self.API_BASE_URL}{config['url']['anthropic']['messages']}"
        params['stream'] = True
        try:
            with requests.post(
                url, headers=self._build_headers(extra_headers), json=params,
                stream=True, timeout=self.timeout
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        yield line.decode('utf-8')
        except requests.exceptions.RequestException as error:
            raise Exception(f"Stream request failed: {error}")
