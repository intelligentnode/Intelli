import json
import math
import time
import uuid

import requests


def new_id():
    """A random id for records and messages added without one."""
    return uuid.uuid4().hex


class StoreError(Exception):
    """
    Error raised by the vector stores and chat histories. It is a plain Exception subclass, so existing
    `except Exception` handlers keep working. status_code is the HTTP status when the service answered.
    """

    def __init__(self, message, status_code=None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class HttpClient:
    """
    The small JSON-over-HTTP client the REST stores share: a base URL, default headers, a timeout and optional
    retries on network errors and 408 / 425 / 429 / 5xx answers. Errors become StoreError('<label> error: ...').
    """

    RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, base_url='', headers=None, timeout=120, retries=0, retry_delay=0.5, session=None,
                 label='HTTP'):
        self.base_url = (base_url or '').rstrip('/')
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.retries = max(0, retries or 0)
        self.retry_delay = retry_delay
        self.session = session if session is not None else requests.Session()
        self.label = label

    def request(self, method, path, body=None, *, data=None, params=None, headers=None, files=None, retries=None,
                raw=False):
        """Send a request and return the decoded JSON ({} for an empty body), or the response when raw=True."""
        url = path if path.startswith('http') else self.base_url + path
        request_headers = dict(self.headers)
        request_headers.update(headers or {})
        if files is not None:
            request_headers.pop('Content-Type', None)
        kwargs = {'headers': request_headers, 'timeout': self.timeout}
        if params:
            kwargs['params'] = params
        if files is not None:
            kwargs['files'] = files
        elif data is not None:
            kwargs['data'] = data
        elif body is not None:
            kwargs['data'] = json.dumps(body)
        attempts = self.retries if retries is None else max(0, retries)
        for attempt in range(attempts + 1):
            try:
                response = self.session.request(method, url, **kwargs)
            except requests.exceptions.RequestException as error:
                if attempt < attempts:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise StoreError(f'{self.label} error: {error}') from None
            status = response.status_code
            if status >= 400:
                if attempt < attempts and status in self.RETRY_STATUSES:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                text = (response.text or '')[:2000]
                try:
                    details = response.json()
                except ValueError:
                    details = text or None
                raise StoreError(f'{self.label} error: HTTP {status}: {text}', status_code=status, details=details)
            if raw:
                return response
            content = response.content or b''
            if not content.strip():
                return {}
            try:
                return response.json()
            except ValueError:
                return response.text


class VectorStore:
    """
    Base class of every vector store: Memory, Pinecone, Qdrant, Chroma, Weaviate, Milvus, Elasticsearch, pgvector,
    MongoDB Atlas, Firestore and Vertex AI share this interface, so an Assistant or a RAG step can swap them.

    Records are {'id', 'vector', 'text'?, 'metadata'?}. Query results are [{'id', 'score', 'text', 'metadata'}]
    sorted by score, where a higher score is more similar (each store converts its own distance to a similarity).

    Subclasses implement upsert(records), query(vector=..., top_k=..., filter=...) and delete(ids); the base class
    adds text embedding (add_documents, search) through the optional embedder.
    """

    def __init__(self, embedder=None):
        """
        Args:
            embedder: an Embedder, a function (texts, kind) -> vectors, or settings
                {'provider', 'api_key', 'model', 'dimensions', 'options'} to build one (see intelli.store.embedder).
        """
        from intelli.store.embedder import to_embedder
        self.embedder = to_embedder(embedder) if embedder else None

    def embed(self, texts, kind='document'):
        """Embed texts with the store's embedder. kind is 'document' or 'query' (Gemini and Cohere differ)."""
        if not self.embedder:
            raise StoreError(f'{type(self).__name__} has no embedder: pass embedder= to the constructor, '
                             'or give vectors.')
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        vectors = self.embedder.embed(items, kind=kind)
        if not isinstance(vectors, list) or len(vectors) != len(items):
            count = len(vectors) if isinstance(vectors, list) else 'no'
            raise StoreError(f'The embedder returned {count} vectors for {len(items)} texts.')
        return vectors

    def add_documents(self, documents):
        """
        Add documents, embedding the ones without a vector. Returns the ids.

        Args:
            documents: [{'id'?, 'text', 'metadata'?, 'vector'?}] or plain strings.
        """
        records = [{'text': item} if isinstance(item, str) else dict(item) for item in (documents or [])]
        missing = [record for record in records if not isinstance(record.get('vector'), (list, tuple))]
        if missing:
            vectors = self.embed([str(record.get('text') or '') for record in missing], 'document')
            for record, vector in zip(missing, vectors):
                record['vector'] = vector
        for record in records:
            if record.get('id') in (None, ''):
                record['id'] = new_id()
            record['id'] = str(record['id'])
            record['metadata'] = record.get('metadata') or {}
        self.upsert(records)
        return [record['id'] for record in records]

    def search(self, text, top_k=5, filter=None):
        """Embed the query text and return the top_k closest records: [{'id', 'score', 'text', 'metadata'}]."""
        vector = self.embed([str(text)], 'query')[0]
        return self.query(vector=vector, top_k=top_k, filter=filter)

    def upsert(self, records):
        """Upsert [{'id', 'vector', 'text'?, 'metadata'?}]. Returns the ids."""
        raise NotImplementedError(f'{type(self).__name__}.upsert is not implemented.')

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        """
        Nearest records to a vector (or to a text when an embedder is set).

        filter is metadata equality ({'key': value}, all must match; a list value means one of). native_filter
        is the store's own filter syntax, for anything else.
        """
        raise NotImplementedError(f'{type(self).__name__}.query is not implemented.')

    def delete(self, ids):
        """Delete records by id."""
        raise NotImplementedError(f'{type(self).__name__}.delete is not implemented.')

    def _query_vector(self, vector=None, text=None):
        """The vector of a query: vector, or the embedded text."""
        if isinstance(vector, (list, tuple)):
            return list(vector)
        if text is not None:
            return self.embed([str(text)], 'query')[0]
        raise StoreError('query needs a vector, or a text and an embedder.')


def matches_filter(metadata, filter):
    """True when every key of filter equals the same key of metadata (a list in the filter means one of)."""
    if not filter:
        return True
    data = metadata or {}
    for key, expected in filter.items():
        actual = data.get(key)
        if isinstance(expected, (list, tuple)):
            if not any(value == actual for value in expected):
                return False
        elif actual != expected:
            return False
    return True


def cosine_similarity(a, b):
    if len(a) != len(b):
        raise StoreError(f'Vector size mismatch: {len(a)} and {len(b)}.')
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def to_items(records):
    """Validate records and normalise them to {'id', 'vector', 'text', 'metadata'}."""
    items = []
    for record in records or []:
        vector = record.get('vector')
        if not isinstance(vector, (list, tuple)):
            raise StoreError(f"Record '{record.get('id')}' has no vector.")
        items.append({'id': str(record.get('id')), 'vector': list(vector), 'text': record.get('text'),
                      'metadata': record.get('metadata') or {}})
    return items


def sorted_matches(matches):
    """Matches sorted by score, highest first (a missing score sorts last)."""
    return sorted(matches, key=lambda match: match['score'] if isinstance(match.get('score'), (int, float))
                  else float('-inf'), reverse=True)
