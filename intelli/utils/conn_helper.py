import json
import requests


class ConnHelper:

    @staticmethod
    def convert_map_to_json(params):
        return json.dumps(params)

    @staticmethod
    def get_error_message(error):
        if isinstance(error, requests.exceptions.RequestException):
            if error.response is not None:
                try:
                    return f'Unexpected HTTP response: {error.response.status_code} Error details: {error.response.json()}'
                except json.JSONDecodeError:
                    return f'Unexpected HTTP response: {error.response.status_code} Error details: {error.response.text}'

        return str(error)
