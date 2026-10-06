import re
import threading
import uuid
from urllib.parse import quote

from intelli.store.vector_store import VectorStore, StoreError, HttpClient, to_items, sorted_matches

UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
DISTANCE_METRICS = {'Euclid', 'Manhattan'}


def stable_uuid(record_id):
    """
    The UUID a store that only accepts UUID ids (Qdrant, Weaviate) keeps for a record id: the id itself when it
    already is a UUID, otherwise uuid5(NAMESPACE_DNS, id), the same mapping as IntelliNode and Weaviate's
    generate_uuid5(id), so upsert, query and delete always map an id to the same point.
    """
    text = str(record_id)
    if UUID_PATTERN.match(text):
        return text.lower()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


def to_qdrant_filter(filter):
    conditions = [{'key': f'metadata.{key}',
                   'match': {'any': list(value)} if isinstance(value, (list, tuple)) else {'value': value}}
                  for key, value in (filter or {}).items()]
    return {'must': conditions} if conditions else None


class QdrantVectorStore(VectorStore):
    """
    Qdrant (self-hosted or Qdrant Cloud) through its REST API.

        store = QdrantVectorStore(url='http://localhost:6333', collection='docs', embedder=embedder)

    Qdrant point ids must be unsigned integers or UUIDs, so every record id is stored as its UUID v5 (see
    stable_uuid) and the original id is kept in the payload: {text, metadata, id}. Query results return the original
    id. filter matches payload.metadata keys with match.value or match.any for lists; native_filter is a Qdrant
    filter object ({'must', 'should', 'must_not'}).
    """

    # API: https://api.qdrant.tech/api-reference/points/upsert-points
    def __init__(self, url='http://localhost:6333', api_key=None, collection=None, distance='Cosine',
                 create_collection=True, dimension=None, vector_name=None, text_key='text', batch_size=256,
                 embedder=None, timeout=120, retries=0, session=None):
        """
        Args:
            distance: Cosine, Dot, Euclid or Manhattan (for a new collection).
            create_collection: create the collection on the first upsert when it does not exist.
            vector_name: a named vector of the collection.
        """
        super().__init__(embedder)
        if not collection:
            raise StoreError('QdrantVectorStore needs a collection name.')
        self.collection = collection
        self.distance = distance or 'Cosine'
        self.create_collection = create_collection is not False
        self.dimension = dimension
        self.vector_name = vector_name
        self.text_key = text_key or 'text'
        self.batch_size = batch_size or 256
        self._ready = False
        self._lock = threading.Lock()
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['api-key'] = api_key
        self.client = HttpClient(url or 'http://localhost:6333', headers, timeout=timeout, retries=retries,
                                 session=session, label='Qdrant')

    def ensure_collection(self, dimension=None):
        """Create the collection when it does not exist. Returns True when it was created."""
        dimension = dimension or self.dimension
        if self._exists():
            return False
        if not dimension:
            raise StoreError(f"Qdrant collection '{self.collection}' does not exist and no dimension is known "
                             'to create it.')
        vectors = {'size': dimension, 'distance': self.distance}
        try:
            # a conflict means another call created it: no retries, then check again
            self.client.request('PUT', self._path(),
                                {'vectors': {self.vector_name: vectors} if self.vector_name else vectors}, retries=0)
        except StoreError:
            try:
                if self._exists():
                    return False
            except StoreError:
                pass
            raise
        return True

    def upsert(self, records):
        items = to_items(records)
        if not items:
            return []
        if self.create_collection:
            self._prepare(len(items[0]['vector']))
        points = [{
            'id': stable_uuid(item['id']),
            'vector': {self.vector_name: item['vector']} if self.vector_name else item['vector'],
            'payload': {self.text_key: item['text'], 'metadata': item['metadata'], 'id': item['id']},
        } for item in items]
        for start in range(0, len(points), self.batch_size):
            self.client.request('PUT', f'{self._path()}/points?wait=true',
                                {'points': points[start:start + self.batch_size]})
        return [item['id'] for item in items]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        body = {'query': query_vector, 'limit': top_k or 5, 'with_payload': True}
        if self.vector_name:
            body['using'] = self.vector_name
        condition = native_filter or to_qdrant_filter(filter)
        if condition:
            body['filter'] = condition
        data = self.client.request('POST', f'{self._path()}/points/query', body)
        points = ((data or {}).get('result') or {}).get('points') or []
        matches = []
        for point in points:
            payload = point.get('payload') or {}
            # Cosine and Dot are similarities; Euclid and Manhattan are distances
            score = 1 / (1 + point['score']) if self.distance in DISTANCE_METRICS else point.get('score')
            matches.append({
                'id': str(payload['id']) if payload.get('id') is not None else str(point.get('id')),
                'score': score,
                'text': payload.get(self.text_key),
                'metadata': payload.get('metadata') or {},
            })
        return sorted_matches(matches)

    def delete(self, ids):
        points = [stable_uuid(record_id) for record_id in ids or []]
        for start in range(0, len(points), 1000):
            self.client.request('POST', f'{self._path()}/points/delete?wait=true',
                                {'points': points[start:start + 1000]})

    def _path(self):
        return f"/collections/{quote(self.collection, safe='')}"

    def _exists(self):
        data = self.client.request('GET', f'{self._path()}/exists')
        return bool(((data or {}).get('result') or {}).get('exists'))

    def _prepare(self, dimension):
        with self._lock:
            if not self._ready:
                self.ensure_collection(self.dimension or dimension)
                self._ready = True
