import json
import math
import threading

from intelli.store.vector_store import VectorStore, StoreError, HttpClient, to_items, sorted_matches


def _literal(value):
    """Milvus string literals are double quoted with backslash escapes, which json.dumps produces."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise StoreError(f'Milvus filters cannot take the number {value}.')
        return str(value)
    return json.dumps(str(value))


def to_milvus_filter(filter):
    conditions = []
    for key, value in (filter or {}).items():
        field = f'metadata[{json.dumps(key)}]'
        if isinstance(value, (list, tuple)):
            conditions.append(f"{field} in [{', '.join(_literal(item) for item in value)}]")
        else:
            conditions.append(f'{field} == {_literal(value)}')
    return ' and '.join(conditions) if conditions else None


def _parse_metadata(value):
    if not value:
        return {}
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return {}


class MilvusVectorStore(VectorStore):
    """
    Milvus or Zilliz Cloud through the RESTful API v2.

        store = MilvusVectorStore(url='http://localhost:19530', token='root:Milvus', collection='docs',
                                  embedder=embedder)

    A missing collection is created on the first upsert with a VarChar primary key `id`, a FloatVector `vector`
    (AUTOINDEX, COSINE), a VarChar `text` and a JSON `metadata` field; it is loaded right away. With COSINE (and IP)
    Milvus returns the similarity itself in `distance`; L2 distances become 1 / (1 + distance).
    filter becomes a boolean expression on the JSON field (metadata["key"] == "value", metadata["key"] in [...]);
    native_filter is a Milvus expression string.
    """

    # API: https://milvus.io/api-reference/restful/v2.6.x/v2/Vector%20(v2)/Search.md
    def __init__(self, url='http://localhost:19530', token=None, collection=None, db_name=None, dimension=None,
                 metric_type='COSINE', create_collection=True, max_text_length=65535, consistency_level=None,
                 batch_size=500, embedder=None, timeout=120, retries=0, session=None):
        """
        Args:
            token: 'user:password' or a Zilliz Cloud API key.
            consistency_level: e.g. 'Strong' to read your own writes at once.
        """
        super().__init__(embedder)
        if not collection:
            raise StoreError('MilvusVectorStore needs a collection name.')
        self.collection = collection
        self.db_name = db_name
        self.dimension = dimension
        self.metric_type = metric_type or 'COSINE'
        self.create_collection = create_collection is not False
        self.max_text_length = max_text_length or 65535
        self.consistency_level = consistency_level
        self.batch_size = batch_size or 500
        self._ready = False
        self._lock = threading.Lock()
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        self.client = HttpClient(url or 'http://localhost:19530', headers, timeout=timeout, retries=retries,
                                 session=session, label='Milvus')

    def ensure_collection(self, dimension=None):
        """Create the collection when it does not exist. Returns True when it was created."""
        dimension = dimension or self.dimension
        if self._has():
            return False
        if not dimension:
            raise StoreError(f"Milvus collection '{self.collection}' does not exist and no dimension is known "
                             'to create it.')
        try:
            self._request('/v2/vectordb/collections/create', {
                'schema': {
                    'autoId': False,
                    'fields': [
                        {'fieldName': 'id', 'dataType': 'VarChar', 'isPrimary': True,
                         'elementTypeParams': {'max_length': 512}},
                        {'fieldName': 'vector', 'dataType': 'FloatVector', 'elementTypeParams': {'dim': str(dimension)}},
                        {'fieldName': 'text', 'dataType': 'VarChar',
                         'elementTypeParams': {'max_length': self.max_text_length}},
                        {'fieldName': 'metadata', 'dataType': 'JSON'},
                    ],
                },
                'indexParams': [{'fieldName': 'vector', 'indexName': 'vector', 'metricType': self.metric_type,
                                 'indexType': 'AUTOINDEX'}],
            })
        except StoreError:
            try:
                if self._has():
                    return False
            except StoreError:
                pass
            raise
        return True

    def upsert(self, records):
        # Milvus rejects a batch that repeats a primary key; the last record wins, like separate upserts.
        items = list({item['id']: item for item in to_items(records)}.values())
        if not items:
            return []
        if self.create_collection:
            self._prepare(len(items[0]['vector']))
        rows = [{'id': item['id'], 'vector': item['vector'], 'text': item['text'] or '', 'metadata': item['metadata']}
                for item in items]
        for start in range(0, len(rows), self.batch_size):
            self._request('/v2/vectordb/entities/upsert', {'data': rows[start:start + self.batch_size]})
        return [str(record.get('id')) for record in records or []]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        body = {
            'data': [query_vector],
            'annsField': 'vector',
            'limit': top_k or 5,
            'outputFields': ['text', 'metadata'],
            'searchParams': {'metricType': self.metric_type},
        }
        condition = native_filter or to_milvus_filter(filter)
        if condition:
            body['filter'] = condition
        if self.consistency_level:
            body['consistencyLevel'] = self.consistency_level
        hits = self._request('/v2/vectordb/entities/search', body)
        flat = []
        for hit in hits if isinstance(hits, list) else []:
            flat.extend(hit if isinstance(hit, list) else [hit])
        matches = [{
            'id': str(hit.get('id')),
            'score': 1 / (1 + hit['distance']) if self.metric_type == 'L2' else hit.get('distance'),
            'text': hit.get('text') or None,
            'metadata': _parse_metadata(hit.get('metadata')),
        } for hit in flat]
        return sorted_matches(matches)

    def delete(self, ids):
        items = [str(record_id) for record_id in ids or []]
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            self._request('/v2/vectordb/entities/delete',
                          {'filter': f"id in [{', '.join(_literal(item) for item in batch)}]"})

    def _has(self):
        data = self._request('/v2/vectordb/collections/has', {})
        return bool((data or {}).get('has'))

    def _prepare(self, dimension):
        with self._lock:
            if not self._ready:
                self.ensure_collection(self.dimension or dimension)
                self._ready = True

    def _request(self, path, body):
        """Every v2 call is a POST that answers HTTP 200 with {code, message} on failure, so check code too."""
        payload = {'collectionName': self.collection}
        if self.db_name:
            payload['dbName'] = self.db_name
        payload.update(body)
        response = self.client.request('POST', path, payload)
        if isinstance(response, dict) and response.get('code') not in (None, 0, 200):
            raise StoreError(f"Milvus error: {response.get('message') or 'request failed'} "
                             f"(code {response.get('code')})", details=response)
        return response.get('data') if isinstance(response, dict) else None
