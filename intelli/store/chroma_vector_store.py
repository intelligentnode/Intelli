import threading
from urllib.parse import quote

from intelli.store.vector_store import VectorStore, StoreError, HttpClient, to_items, sorted_matches


def to_chroma_where(filter):
    """Chroma rejects a where clause with more than one key unless the conditions are wrapped in $and."""
    clauses = [{key: {'$in': list(value)} if isinstance(value, (list, tuple)) else {'$eq': value}}
               for key, value in (filter or {}).items()]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {'$and': clauses}


def collection_space(collection):
    """The distance space of a collection; Chroma's default is l2."""
    config = collection.get('configuration_json') or {}
    return (((config.get('hnsw') or {}).get('space'))
            or ((config.get('spann') or {}).get('space'))
            or ((collection.get('metadata') or {}).get('hnsw:space'))
            or 'l2')


class ChromaVectorStore(VectorStore):
    """
    A Chroma server (self-hosted or Chroma Cloud) through the v2 REST API.

        store = ChromaVectorStore(url='http://localhost:8000', collection='docs', embedder=embedder)

    The collection is fetched or created by name (with the cosine space) on first use. Scores are converted from
    the collection's distance: cosine and ip give 1 - distance, l2 gives 1 / (1 + distance).
    filter becomes a where clause ($eq / $in, joined by $and); native_filter is a Chroma where clause.
    Chroma metadata values must be strings, numbers, booleans or lists of them (no nested objects).
    """

    # API: https://docs.trychroma.com/reference/chroma-api/record/query-collection
    def __init__(self, url='http://localhost:8000', collection=None, tenant='default_tenant',
                 database='default_database', api_key=None, space='cosine', batch_size=1000, embedder=None,
                 timeout=120, retries=0, session=None):
        """
        Args:
            api_key: Chroma Cloud key, sent as x-chroma-token.
            space: distance space for a new collection (cosine, l2 or ip).
        """
        super().__init__(embedder)
        if not collection:
            raise StoreError('ChromaVectorStore needs a collection name.')
        self.collection = collection
        self.tenant = tenant or 'default_tenant'
        self.database = database or 'default_database'
        self.space = space or 'cosine'
        self.batch_size = batch_size or 1000
        self.collection_id = None
        self._lock = threading.Lock()
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['x-chroma-token'] = api_key
        self.client = HttpClient(url or 'http://localhost:8000', headers, timeout=timeout, retries=retries,
                                 session=session, label='Chroma')

    def get_collection(self):
        """Get or create the collection by name; returns its id."""
        with self._lock:
            if self.collection_id is None:
                collection = self.client.request('POST', f'{self._database_path()}/collections', {
                    'name': self.collection,
                    'metadata': {'hnsw:space': self.space},
                    'get_or_create': True,
                })
                self.space = collection_space(collection)
                self.collection_id = collection['id']
            return self.collection_id

    def upsert(self, records):
        items = to_items(records)
        if not items:
            return []
        path = self._collection_path()
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            self.client.request('POST', f'{path}/upsert', {
                'ids': [item['id'] for item in batch],
                'embeddings': [item['vector'] for item in batch],
                'documents': [item['text'] for item in batch],
                # older Chroma servers reject an empty metadata object
                'metadatas': [item['metadata'] or None for item in batch],
            })
        return [item['id'] for item in items]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        path = self._collection_path()
        body = {'query_embeddings': [query_vector], 'n_results': top_k or 5,
                'include': ['documents', 'metadatas', 'distances']}
        where = native_filter or to_chroma_where(filter)
        if where:
            body['where'] = where
        data = self.client.request('POST', f'{path}/query', body) or {}

        def first(values):
            return values[0] if isinstance(values, list) and values and isinstance(values[0], list) else []

        documents, metadatas, distances = first(data.get('documents')), first(data.get('metadatas')), first(
            data.get('distances'))
        matches = []
        for index, record_id in enumerate(first(data.get('ids'))):
            matches.append({
                'id': record_id,
                'score': self._score(distances[index]),
                'text': documents[index] if index < len(documents) else None,
                'metadata': (metadatas[index] if index < len(metadatas) else None) or {},
            })
        return sorted_matches(matches)

    def delete(self, ids):
        items = [str(record_id) for record_id in ids or []]
        if not items:
            return
        path = self._collection_path()
        for start in range(0, len(items), self.batch_size):
            self.client.request('POST', f'{path}/delete', {'ids': items[start:start + self.batch_size]})

    def _score(self, distance):
        # cosine distance is 1 - cos and ip distance is 1 - dot; l2 is the squared euclidean distance
        return 1 / (1 + distance) if self.space == 'l2' else 1 - distance

    def _database_path(self):
        return f"/api/v2/tenants/{quote(self.tenant, safe='')}/databases/{quote(self.database, safe='')}"

    def _collection_path(self):
        return f"{self._database_path()}/collections/{quote(str(self.get_collection()), safe='')}"
