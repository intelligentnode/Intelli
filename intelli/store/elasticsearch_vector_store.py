import base64
import json
import threading
from urllib.parse import quote

from intelli.store.vector_store import VectorStore, StoreError, HttpClient, to_items, sorted_matches


def to_elastic_filter(filter):
    clauses = [{'terms': {f'metadata.{key}': list(value)}} if isinstance(value, (list, tuple))
               else {'term': {f'metadata.{key}': value}} for key, value in (filter or {}).items()]
    return {'bool': {'filter': clauses}} if clauses else None


class ElasticsearchVectorStore(VectorStore):
    """
    An Elasticsearch index (self-managed, Elastic Cloud or Serverless) with a dense_vector field, through the REST
    API.

        store = ElasticsearchVectorStore(url='https://my-deployment.es.io', api_key=key, index='docs',
                                         embedder=embedder)

    A missing index is created on the first upsert: the vector field (dense_vector, index: true, similarity cosine),
    `text`, and `metadata` whose string values are mapped as keyword so filters match exact values. Upserts and
    deletes go through _bulk; queries use the top-level knn search. Elasticsearch scores cosine and dot_product as
    (1 + cos) / 2, which is converted back to the cosine similarity; l2_norm keeps 1 / (1 + distance^2).
    filter becomes term / terms clauses on metadata.<key>; native_filter is a query DSL filter for knn.filter.
    """

    # API: https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector
    def __init__(self, url='http://localhost:9200', api_key=None, username=None, password=None, index=None,
                 dimension=None, similarity='cosine', create_index=True, vector_field='embedding', refresh='wait_for',
                 num_candidates=None, batch_size=500, embedder=None, timeout=120, retries=0, session=None):
        """
        Args:
            api_key: the encoded API key, sent as Authorization: ApiKey.
            username / password: basic auth instead of an API key.
            refresh: the _bulk refresh value ('wait_for'), or False to skip waiting.
            num_candidates: knn candidates (default 10 x top_k, at least 100).
        """
        super().__init__(embedder)
        if not index:
            raise StoreError('ElasticsearchVectorStore needs an index name.')
        self.index = index
        self.dimension = dimension
        self.similarity = similarity or 'cosine'
        self.create_index = create_index is not False
        self.vector_field = vector_field or 'embedding'
        self.refresh = refresh
        self.num_candidates = num_candidates
        self.batch_size = batch_size or 500
        self._ready = False
        self._lock = threading.Lock()
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'ApiKey {api_key}'
        elif username:
            token = base64.b64encode(f'{username}:{password or ""}'.encode('utf-8')).decode('ascii')
            headers['Authorization'] = f'Basic {token}'
        self.client = HttpClient(url or 'http://localhost:9200', headers, timeout=timeout, retries=retries,
                                 session=session, label='Elasticsearch')

    def ensure_index(self, dimension=None):
        """Create the index with the vector mapping when it does not exist. Returns True when it was created."""
        dimension = dimension or self.dimension
        if self._exists():
            return False
        if not dimension:
            raise StoreError(f"Elasticsearch index '{self.index}' does not exist and no dimension is known to "
                             'create it.')
        try:
            self.client.request('PUT', self._path(), {
                'mappings': {
                    'dynamic_templates': [{
                        'metadata_strings': {'path_match': 'metadata.*', 'match_mapping_type': 'string',
                                             'mapping': {'type': 'keyword'}},
                    }],
                    'properties': {
                        self.vector_field: {'type': 'dense_vector', 'dims': dimension, 'index': True,
                                            'similarity': self.similarity},
                        'text': {'type': 'text'},
                        'metadata': {'type': 'object'},
                    },
                },
            })
        except StoreError:
            # resource_already_exists_exception: created meanwhile by another call
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
        if self.create_index:
            self._prepare(len(items[0]['vector']))
        for start in range(0, len(items), self.batch_size):
            lines = []
            for item in items[start:start + self.batch_size]:
                lines.append(json.dumps({'index': {'_index': self.index, '_id': item['id']}}))
                lines.append(json.dumps({'text': item['text'], 'metadata': item['metadata'],
                                         self.vector_field: item['vector']}))
            self._bulk(lines)
        return [item['id'] for item in items]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        top_k = top_k or 5
        knn = {
            'field': self.vector_field,
            'query_vector': query_vector,
            'k': top_k,
            'num_candidates': min(10000, max(top_k, self.num_candidates or max(100, top_k * 10))),
        }
        condition = native_filter or to_elastic_filter(filter)
        if condition:
            knn['filter'] = condition
        data = self.client.request('POST', f'{self._path()}/_search',
                                   {'knn': knn, 'size': top_k, '_source': {'excludes': [self.vector_field]}})
        matches = []
        for hit in ((data or {}).get('hits') or {}).get('hits') or []:
            source = hit.get('_source') or {}
            matches.append({'id': hit.get('_id'), 'score': self._score(hit.get('_score')), 'text': source.get('text'),
                            'metadata': source.get('metadata') or {}})
        return sorted_matches(matches)

    def delete(self, ids):
        items = [str(record_id) for record_id in ids or []]
        for start in range(0, len(items), self.batch_size):
            self._bulk([json.dumps({'delete': {'_index': self.index, '_id': record_id}})
                        for record_id in items[start:start + self.batch_size]])

    def _score(self, score):
        if not isinstance(score, (int, float)):
            return None
        return 2 * score - 1 if self.similarity in ('cosine', 'dot_product') else score

    def _path(self):
        return f"/{quote(self.index, safe='')}"

    def _exists(self):
        try:
            self.client.request('HEAD', self._path(), raw=True)
            return True
        except StoreError as error:
            if error.status_code == 404:
                return False
            raise

    def _prepare(self, dimension):
        with self._lock:
            if not self._ready:
                self.ensure_index(self.dimension or dimension)
                self._ready = True

    def _bulk(self, lines):
        """_bulk answers 200 and reports failures per item; a delete of a missing id is not a failure."""
        path = f"/_bulk?refresh={quote(str(self.refresh), safe='')}" if self.refresh else '/_bulk'
        data = self.client.request('POST', path, data=('\n'.join(lines) + '\n').encode('utf-8'),
                                   headers={'Content-Type': 'application/x-ndjson'})
        if not isinstance(data, dict) or not data.get('errors'):
            return data
        for item in data.get('items') or []:
            action, result = next(iter(item.items()), (None, None))
            if not result or not result.get('error') or (action == 'delete' and result.get('status') == 404):
                continue
            error = result['error']
            reason = error.get('reason') or error.get('type') or json.dumps(error)
            raise StoreError(f"Elasticsearch error: {action} {result.get('_id')}: {reason}",
                             status_code=result.get('status'), details=error)
        return data
