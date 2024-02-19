import requests

from intelli.config import config
from intelli.utils.conn_helper import ConnHelper


class IntellicloudWrapper:
    def __init__(self, api_key, api_base=None):
        self.ONE_KEY = api_key
        self.API_BASE_URL = api_base if api_base else config['url']['intellicloud']['base']

    def semantic_search(self, query_text, k=3, filters=None):
        if filters is None:
            filters = {}

        url = f"{self.API_BASE_URL}{config['url']['intellicloud']['semantic_search']}"
        # set the data
        data = {'one_key': self.ONE_KEY, 'query_text': query_text, 'k': k}
        if filters and 'document_name' in filters:
            data['document_name'] = filters['document_name']

        # call the document search
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            return response.json()['data']
        except requests.RequestException as e:
            raise Exception(ConnHelper.get_error_message(e))
