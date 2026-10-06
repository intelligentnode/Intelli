# API: Vector Search 2.0 (vectorsearch.googleapis.com v1: collections, dataObjects:batchCreate / search) and
# Vector Search 1.0 (aiplatform v1: indexes:upsertDatapoints / removeDatapoints, indexEndpoints:findNeighbors)
import re
from urllib.parse import quote

from intelli.store.vector_store import VectorStore, StoreError, to_items, sorted_matches
from intelli.store.google_cloud import GoogleCloudService

VECTOR_SEARCH_BASE = 'https://vectorsearch.googleapis.com/v1'


def _imul(a, b):
    return (a * b) & 0xFFFFFFFF


def data_object_id(record_id):
    """
    A valid data object id (1-63 lowercase letters, digits or hyphens, starting with a letter) for any record id:
    the id itself when it is valid, else a slug and a stable hash, the same mapping as IntelliNode.
    """
    text = str(record_id)
    if re.match(r'^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$', text):
        return text
    # FNV-1a, two rounds, over UTF-16 code units like JavaScript's charCodeAt
    h1, h2 = 0x811c9dc5, 0x01000193
    encoded = text.encode('utf-16-le')
    for index in range(0, len(encoded), 2):
        code = encoded[index] | (encoded[index + 1] << 8)
        h1 = _imul(h1 ^ code, 0x01000193)
        h2 = _imul(h2 ^ code, 0x811c9dc5)
    slug = re.sub(r'^-+|-+$', '', re.sub(r'[^a-z0-9]+', '-', text.lower()))[:40]
    value = f"d-{slug + '-' if slug else ''}{h1:x}{h2:x}"[:63]
    return value.rstrip('-')


class VertexVectorSearchStore(VectorStore):
    """
    Vertex AI Vector Search 2.0 collections: managed vector search that stores the data objects (text and metadata)
    with their vectors. Create the collection once with create_collection(dimensions).
    Metadata keys are stored as top-level data fields, so filters work on them ({'genre': 'sci-fi'}).
    Credentials: OAuth (access_token, a credentials object, or `gcloud auth application-default login`).
    """

    def __init__(self, project_id=None, location='us-central1', collection=None, vector_field='embedding',
                 distance_metric='COSINE_DISTANCE', access_token=None, credentials=None, quota_project_id=None,
                 embedder=None, timeout=120, retries=0, session=None):
        """
        Args:
            collection: the collection id or full name.
            distance_metric: COSINE_DISTANCE or DOT_PRODUCT.
        """
        super().__init__(embedder)
        if not collection:
            raise StoreError('VertexVectorSearchStore needs a collection id.')
        self.service = GoogleCloudService(project_id=project_id, access_token=access_token, credentials=credentials,
                                          quota_project_id=quota_project_id, timeout=timeout, retries=retries,
                                          session=session, label='Vertex AI Vector Search')
        self.location = location or 'us-central1'
        self.collection = collection
        self.vector_field = vector_field or 'embedding'
        self.distance_metric = distance_metric or 'COSINE_DISTANCE'

    def _collection_name(self):
        if self.collection.startswith('projects/'):
            return self.collection
        return f'projects/{self.service.project()}/locations/{self.location}/collections/{self.collection}'

    def create_collection(self, dimensions=None, display_name=None, description=None):
        """Create the collection (once). Waits for the operation when the API returns one."""
        if not dimensions:
            raise StoreError('create_collection needs the vector dimensions.')
        parent = f'projects/{self.service.project()}/locations/{self.location}'
        body = {
            'displayName': display_name or self.collection,
            'vectorSchema': {self.vector_field: {'denseVector': {'dimensions': dimensions}}},
            'dataSchema': {'type': 'object', 'properties': {'text': {'type': 'string'},
                                                            'sourceId': {'type': 'string'}}},
        }
        if description:
            body['description'] = description
        result = self.service.request(
            'POST', f"{VECTOR_SEARCH_BASE}/{parent}/collections?collectionId={quote(self.collection, safe='')}", body)
        if isinstance(result, dict) and result.get('name') and 'done' in result:
            return self.service.wait_for_operation(result, lambda name: f'{VECTOR_SEARCH_BASE}/{name}')
        return result

    def upsert(self, records):
        items = to_items(records)
        collection = self._collection_name()
        # create fails on an existing id, so replaced records are deleted first
        self.delete([item['id'] for item in items], ignore_missing=True)
        for start in range(0, len(items), 1000):
            requests_list = [{
                'dataObjectId': data_object_id(item['id']),
                'dataObject': {
                    'data': {**item['metadata'], 'text': item['text'], 'sourceId': item['id']},
                    'vectors': {self.vector_field: {'dense': {'values': item['vector']}}},
                },
            } for item in items[start:start + 1000]]
            self.service.request('POST', f'{VECTOR_SEARCH_BASE}/{collection}/dataObjects:batchCreate',
                                 {'requests': requests_list})
        return [item['id'] for item in items]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        condition = native_filter or self._filter(filter)
        search = {'searchField': self.vector_field, 'vector': {'values': query_vector}, 'topK': top_k or 5,
                  'distanceMetric': self.distance_metric}
        if condition:
            search['filter'] = condition
        result = self.service.request('POST', f'{VECTOR_SEARCH_BASE}/{self._collection_name()}/dataObjects:search',
                                      {'vectorSearch': search}) or {}
        matches = []
        for item in result.get('results') or []:
            data_object = item.get('dataObject') or {}
            data = dict(data_object.get('data') or {})
            text_value = data.pop('text', None)
            source_id = data.pop('sourceId', None)
            matches.append({
                'id': source_id or data_object.get('dataObjectId') or str(data_object.get('name') or '').split('/')[-1],
                'score': self._score(item.get('distance')),
                'text': text_value,
                'metadata': data,
            })
        return sorted_matches(matches)

    def _score(self, distance):
        if not isinstance(distance, (int, float)):
            return 0
        return 1 - distance if self.distance_metric == 'COSINE_DISTANCE' else distance

    @staticmethod
    def _filter(filter):
        if not filter:
            return None
        clauses = [{key: {'$in': list(value)} if isinstance(value, (list, tuple)) else {'$eq': value}}
                   for key, value in filter.items()]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {'$and': clauses}

    def delete(self, ids, ignore_missing=False):
        collection = self._collection_name()
        for record_id in ids or []:
            try:
                self.service.request('DELETE', f'{VECTOR_SEARCH_BASE}/{collection}/dataObjects/{data_object_id(record_id)}')
            except StoreError as error:
                if not (ignore_missing and error.status_code == 404):
                    raise


class VertexVectorSearchIndexStore(VectorStore):
    """
    Vertex AI Vector Search 1.0: an index with stream updates deployed to a public index endpoint. Text and metadata
    travel in embeddingMetadata (up to 2 KB per datapoint); metadata equality filters use restricts.
    """

    def __init__(self, project_id=None, location='us-central1', index=None, index_endpoint=None,
                 deployed_index_id=None, public_endpoint_domain=None, distance_measure='DOT_PRODUCT_DISTANCE',
                 restrict_keys=None, access_token=None, credentials=None, quota_project_id=None, embedder=None,
                 timeout=120, retries=0, session=None):
        """
        Args:
            index / index_endpoint: ids or full names.
            public_endpoint_domain: e.g. 123.us-central1-456.vdb.vertexai.goog.
            distance_measure: DOT_PRODUCT_DISTANCE, COSINE_DISTANCE or SQUARED_L2_DISTANCE.
            restrict_keys: metadata keys sent as restricts (so they can be filtered).
        """
        super().__init__(embedder)
        for name, value in (('index', index), ('index_endpoint', index_endpoint),
                            ('deployed_index_id', deployed_index_id), ('public_endpoint_domain', public_endpoint_domain)):
            if not value:
                raise StoreError(f'VertexVectorSearchIndexStore needs {name}.')
        self.service = GoogleCloudService(project_id=project_id, access_token=access_token, credentials=credentials,
                                          quota_project_id=quota_project_id, timeout=timeout, retries=retries,
                                          session=session, label='Vertex AI Vector Search')
        self.location = location or 'us-central1'
        self.index = index
        self.index_endpoint = index_endpoint
        self.deployed_index_id = deployed_index_id
        self.public_endpoint_domain = re.sub(r'/+$', '', re.sub(r'^https?://', '', str(public_endpoint_domain)))
        self.distance_measure = distance_measure or 'DOT_PRODUCT_DISTANCE'
        self.restrict_keys = list(restrict_keys or [])

    def _name(self, kind, value):
        if str(value).startswith('projects/'):
            return value
        return f'projects/{self.service.project()}/locations/{self.location}/{kind}/{value}'

    def upsert(self, records):
        index = self._name('indexes', self.index)
        datapoints = []
        for item in to_items(records):
            metadata = item['metadata']
            restricts = [{'namespace': key, 'allowList': [str(value) for value in (
                metadata[key] if isinstance(metadata[key], list) else [metadata[key]])]}
                for key in self.restrict_keys if metadata.get(key) is not None]
            datapoint = {'datapointId': item['id'], 'featureVector': item['vector'],
                         'embeddingMetadata': {'text': item['text'], 'metadata': metadata}}
            if restricts:
                datapoint['restricts'] = restricts
            datapoints.append(datapoint)
        url = f'https://{self.location}-aiplatform.googleapis.com/v1/{index}:upsertDatapoints'
        for start in range(0, len(datapoints), 1000):
            self.service.request('POST', url, {'datapoints': datapoints[start:start + 1000]})
        return [datapoint['datapointId'] for datapoint in datapoints]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        restricts = [{'namespace': key, 'allowList': [str(item) for item in (value if isinstance(value, (list, tuple))
                                                                             else [value])]}
                     for key, value in (filter or {}).items()]
        datapoint = {'datapointId': 'query', 'featureVector': query_vector}
        if restricts:
            datapoint['restricts'] = restricts
        datapoint.update(native_filter or {})
        endpoint = self._name('indexEndpoints', self.index_endpoint)
        body = {'deployedIndexId': self.deployed_index_id, 'returnFullDatapoint': True,
                'queries': [{'datapoint': datapoint, 'neighborCount': top_k or 5}]}
        result = self.service.request('POST', f'https://{self.public_endpoint_domain}/v1/{endpoint}:findNeighbors',
                                      body) or {}
        nearest = result.get('nearestNeighbors') or []
        neighbors = (nearest[0] or {}).get('neighbors') or [] if nearest else []
        matches = []
        for neighbor in neighbors:
            point = neighbor.get('datapoint') or {}
            payload = point.get('embeddingMetadata') or {}
            matches.append({'id': point.get('datapointId'), 'score': self._score(neighbor.get('distance')),
                            'text': payload.get('text'), 'metadata': payload.get('metadata') or {}})
        return sorted_matches(matches)

    def _score(self, distance):
        """DOT_PRODUCT_DISTANCE comes back as the dot product (higher is closer); the others are distances."""
        if not isinstance(distance, (int, float)):
            return 0
        if self.distance_measure == 'COSINE_DISTANCE':
            return 1 - distance
        if self.distance_measure == 'SQUARED_L2_DISTANCE':
            return 1 / (1 + distance)
        return distance

    def delete(self, ids):
        index = self._name('indexes', self.index)
        self.service.request('POST', f'https://{self.location}-aiplatform.googleapis.com/v1/{index}:removeDatapoints',
                             {'datapointIds': [str(record_id) for record_id in ids or []]})
