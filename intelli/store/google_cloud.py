import os
import re
import threading
import time
from datetime import datetime
from urllib.parse import quote, unquote

from intelli.store.vector_store import StoreError, HttpClient

CLOUD_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'


class GoogleCloudService:
    """
    Shared plumbing for the Google Cloud stores (Firestore, RAG Engine, Vector Search): OAuth headers and one HTTP
    client. These APIs reject API keys, so credentials come from an access token (a string or a function), a
    google.auth credentials object, or Application Default Credentials (`gcloud auth application-default login`,
    needs `pip install google-auth`).
    """

    def __init__(self, project_id=None, access_token=None, credentials=None, quota_project_id=None, timeout=120,
                 retries=0, session=None, label='Google Cloud'):
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT') or None
        self.label = label
        self.client = HttpClient('', {'Content-Type': 'application/json'}, timeout=timeout, retries=retries,
                                 session=session, label=label)
        self._access_token = access_token
        self._credentials = credentials
        self._quota_project_id = quota_project_id
        self._last_token = None
        self._lock = threading.Lock()

    def headers(self):
        headers = {'Authorization': f'Bearer {self._token()}'}
        quota_project = self._quota_project_id or getattr(self._credentials, 'quota_project_id', None)
        if quota_project:
            headers['x-goog-user-project'] = quota_project
        return headers

    def _token(self):
        if self._access_token:
            token = self._access_token() if callable(self._access_token) else self._access_token
            return str(token).strip()
        with self._lock:
            if self._credentials is None:
                self._credentials = self._default_credentials()
            credentials = self._credentials
            if not getattr(credentials, 'valid', False) or not getattr(credentials, 'token', None):
                try:
                    from google.auth.transport.requests import Request
                    credentials.refresh(Request())
                except Exception as error:
                    raise StoreError(f'{self.label}: could not refresh Google credentials ({error}).') from None
            self._last_token = str(credentials.token).strip()
            return self._last_token

    def _default_credentials(self):
        try:
            import google.auth
        except ImportError:
            raise StoreError(f'{self.label} uses OAuth: pass access_token=, or install google-auth '
                             '(pip install google-auth) and run gcloud auth application-default login.') from None
        try:
            credentials, project = google.auth.default(scopes=[CLOUD_SCOPE])
        except Exception as error:
            raise StoreError(f'{self.label}: could not load Application Default Credentials '
                             f'({type(error).__name__}). Run gcloud auth application-default login, or pass '
                             'access_token=.') from None
        if not self.project_id and project:
            self.project_id = project
        return credentials

    def project(self):
        if self.project_id:
            return self.project_id
        if not self._access_token:
            with self._lock:
                if self._credentials is None:
                    self._credentials = self._default_credentials()
            self.project_id = self.project_id or getattr(self._credentials, 'project_id', None)
        if not self.project_id:
            raise StoreError(f'{self.label} needs a project_id (or GOOGLE_CLOUD_PROJECT).')
        return self.project_id

    def request(self, method, url, body=None, **kwargs):
        headers = {**self.headers(), **(kwargs.pop('headers', None) or {})}
        return self.client.request(method, url, body, headers=headers, **kwargs)

    def wait_for_operation(self, operation, operation_url, max_wait=600, poll=3):
        """Poll a long-running operation until it is done; returns its response (or raises its error)."""
        current = operation or {}
        started = time.time()
        while not current.get('done'):
            if time.time() - started > max_wait:
                raise StoreError(f"{self.label}: operation {current.get('name')} did not finish in time.")
            time.sleep(poll)
            current = self.request('GET', operation_url(current['name']))
        if current.get('error'):
            raise StoreError(f"{self.label} operation failed: {current['error']}", details=current['error'])
        return current.get('response') or current


class FirestoreVector:
    """A vector value for Firestore (stored in the map form the Firestore SDKs write)."""

    def __init__(self, values):
        self.values = list(values)


class Firestore:
    """Firestore REST values <-> Python values."""

    @staticmethod
    def to_value(value):
        if value is None:
            return {'nullValue': None}
        if isinstance(value, FirestoreVector):
            return {'mapValue': {'fields': {
                '__type__': {'stringValue': '__vector__'},
                'value': {'arrayValue': {'values': [{'doubleValue': float(v)} for v in value.values]}},
            }}}
        if isinstance(value, bool):
            return {'booleanValue': value}
        if isinstance(value, int):
            return {'integerValue': str(value)}
        if isinstance(value, float):
            return {'doubleValue': value}
        if isinstance(value, str):
            return {'stringValue': value}
        if isinstance(value, datetime):
            return {'timestampValue': value.isoformat()}
        if isinstance(value, (list, tuple)):
            return {'arrayValue': {'values': [Firestore.to_value(item) for item in value]}}
        if isinstance(value, dict):
            return {'mapValue': {'fields': Firestore.to_fields(value)}}
        return {'stringValue': str(value)}

    @staticmethod
    def to_fields(data):
        return {key: Firestore.to_value(value) for key, value in (data or {}).items()}

    @staticmethod
    def from_value(value):
        if not isinstance(value, dict):
            return None
        if 'nullValue' in value:
            return None
        if 'booleanValue' in value:
            return value['booleanValue']
        if 'integerValue' in value:
            return int(value['integerValue'])
        if 'doubleValue' in value:
            return float(value['doubleValue'])
        if 'stringValue' in value:
            return value['stringValue']
        if 'timestampValue' in value:
            return value['timestampValue']
        if 'arrayValue' in value:
            return [Firestore.from_value(item) for item in (value['arrayValue'] or {}).get('values') or []]
        if 'mapValue' in value:
            fields = (value['mapValue'] or {}).get('fields') or {}
            if (fields.get('__type__') or {}).get('stringValue') == '__vector__':
                return Firestore.from_value(fields.get('value'))
            return Firestore.from_fields(fields)
        for key in ('referenceValue', 'geoPointValue', 'bytesValue'):
            if key in value:
                return value[key]
        return None

    @staticmethod
    def from_fields(fields):
        return {key: Firestore.from_value(value) for key, value in (fields or {}).items()}

    @staticmethod
    def doc_id(record_id):
        """A document id from any record id: '/' is not allowed in Firestore ids (same encoding as IntelliNode)."""
        encoded = quote(str(record_id), safe="-_!~*'()").replace('.', '%2E')
        if not encoded or re.match(r'^__.*__$', encoded):
            raise StoreError(f"Invalid Firestore document id '{record_id}'.")
        return encoded

    @staticmethod
    def decode_id(name):
        return unquote(str(name).split('/')[-1])

    @staticmethod
    def where(filter, prefix=''):
        """Field filter(s) for metadata equality: {key: value} -> a where clause on <prefix><key>."""
        filters = []
        for key, expected in (filter or {}).items():
            many = isinstance(expected, (list, tuple))
            filters.append({'fieldFilter': {
                'field': {'fieldPath': f'{prefix}{Firestore.field_path(key)}'},
                'op': 'IN' if many else 'EQUAL',
                'value': {'arrayValue': {'values': [Firestore.to_value(item) for item in expected]}} if many
                else Firestore.to_value(expected),
            }})
        if not filters:
            return None
        return filters[0] if len(filters) == 1 else {'compositeFilter': {'op': 'AND', 'filters': filters}}

    @staticmethod
    def field_path(name):
        """A field path segment: simple names as is, others in backticks."""
        if re.match(r'^[A-Za-z_][A-Za-z_0-9]*$', name):
            return name
        return '`' + re.sub(r'([`\\])', r'\\\1', str(name)) + '`'
