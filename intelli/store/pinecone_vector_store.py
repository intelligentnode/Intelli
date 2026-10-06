import json
import re

from intelli.store.vector_store import VectorStore, StoreError, HttpClient, to_items, sorted_matches

# Pinecone rejects upsert requests over 2MB; stay under it with room for the JSON envelope
MAX_BATCH_BYTES = 1500000


def to_pinecone_filter(filter):
    if not filter:
        return None
    return {key: {'$in': list(value)} if isinstance(value, (list, tuple)) else {'$eq': value}
            for key, value in filter.items()}


def clean_metadata(metadata):
    """Pinecone metadata takes strings, numbers, booleans and string lists, and rejects nulls."""
    return {key: value for key, value in (metadata or {}).items() if value is not None}


class PineconeVectorStore(VectorStore):
    """
    A Pinecone index through the data plane REST API.

        store = PineconeVectorStore(api_key=key, index_host='docs-abc123.svc.aped-4627-b74a.pinecone.io',
                                    embedder=embedder)

    Create the index first (console or control plane) with the embedder's dimension; with the cosine metric the
    scores are cosine similarities. The record text is kept in the metadata under text_key. filter becomes
    {key: {'$eq'}} or {key: {'$in'}}; native_filter is a Pinecone metadata filter.
    """

    # API: https://docs.pinecone.io/reference/api/2026-07/data-plane/upsert
    def __init__(self, api_key=None, index_host=None, namespace=None, api_version='2026-07', metric='cosine',
                 text_key='text', batch_size=100, embedder=None, timeout=120, retries=0, session=None):
        """
        Args:
            api_key: Pinecone API key.
            index_host: the index host, with or without https://.
            metric: the index metric: cosine, dotproduct or euclidean.
        """
        super().__init__(embedder)
        if not api_key:
            raise StoreError('PineconeVectorStore needs an api_key.')
        if not index_host:
            raise StoreError('PineconeVectorStore needs the index_host of the index.')
        host = str(index_host).rstrip('/')
        self.namespace = namespace
        self.metric = metric or 'cosine'
        self.text_key = text_key or 'text'
        self.batch_size = min(batch_size or 100, 1000)
        self.client = HttpClient(
            host if re.match(r'^https?://', host, re.I) else f'https://{host}',
            {'Content-Type': 'application/json', 'Api-Key': api_key, 'X-Pinecone-Api-Version': api_version},
            timeout=timeout, retries=retries, session=session, label='Pinecone')

    def upsert(self, records):
        items = to_items(records)
        vectors = []
        for item in items:
            metadata = clean_metadata(item['metadata'])
            if item['text'] is not None:
                metadata[self.text_key] = item['text']
            vector = {'id': item['id'], 'values': item['vector']}
            if metadata:
                vector['metadata'] = metadata
            vectors.append(vector)
        batch, size = [], 0
        for vector in vectors:
            vector_size = len(json.dumps(vector))
            if batch and (len(batch) >= self.batch_size or size + vector_size > MAX_BATCH_BYTES):
                self.client.request('POST', '/vectors/upsert', self._with_namespace({'vectors': batch}))
                batch, size = [], 0
            batch.append(vector)
            size += vector_size
        if batch:
            self.client.request('POST', '/vectors/upsert', self._with_namespace({'vectors': batch}))
        return [item['id'] for item in items]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        body = self._with_namespace({'vector': query_vector, 'topK': top_k or 5, 'includeMetadata': True,
                                     'includeValues': False})
        condition = native_filter or to_pinecone_filter(filter)
        if condition:
            body['filter'] = condition
        data = self.client.request('POST', '/query', body)
        matches = []
        for match in (data or {}).get('matches') or []:
            metadata = dict(match.get('metadata') or {})
            text_value = metadata.pop(self.text_key, None)
            # cosine and dotproduct scores are similarities; euclidean is a distance
            score = 1 / (1 + match['score']) if self.metric == 'euclidean' else match.get('score')
            matches.append({'id': match.get('id'), 'score': score, 'text': text_value, 'metadata': metadata})
        return sorted_matches(matches)

    def delete(self, ids):
        items = [str(record_id) for record_id in ids or []]
        for start in range(0, len(items), 1000):
            self.client.request('POST', '/vectors/delete', self._with_namespace({'ids': items[start:start + 1000]}))

    def _with_namespace(self, body):
        return {**body, 'namespace': self.namespace} if self.namespace else body
