import json
import math
import re
import threading

from intelli.store.vector_store import VectorStore, StoreError, HttpClient, to_items, sorted_matches
from intelli.store.qdrant_vector_store import stable_uuid

ID_PROPERTY = 'recordId'
METADATA_PROPERTY = 'metadataJson'
# metadata keys that can also be stored as their own (filterable) property
PROPERTY_NAME = re.compile(r'^[_a-z][_0-9A-Za-z]*$')
RESERVED = {'id', '_id', '_additional', ID_PROPERTY, METADATA_PROPERTY}


def _is_scalar(value):
    if isinstance(value, bool) or isinstance(value, str):
        return True
    return isinstance(value, (int, float)) and math.isfinite(value)


def _value_key(value):
    if isinstance(value, bool):
        return 'valueBoolean'
    if isinstance(value, (int, float)):
        return 'valueNumber'
    return 'valueText'


def to_weaviate_where(filter):
    operands = []
    for key, value in (filter or {}).items():
        if not isinstance(value, (list, tuple)):
            operands.append({'path': [key], 'operator': 'Equal', _value_key(value): value})
            continue
        options = [{'path': [key], 'operator': 'Equal', _value_key(item): item} for item in value]
        operands.append(options[0] if len(options) == 1 else {'operator': 'Or', 'operands': options})
    if not operands:
        return None
    return operands[0] if len(operands) == 1 else {'operator': 'And', 'operands': operands}


def gql(value, key=None):
    """GraphQL input literal: object keys are bare names and `operator` values are enums."""
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(gql(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{name}: {gql(item, name)}' for name, item in value.items()) + '}'
    if key == 'operator' and re.match(r'^[A-Za-z]+$', str(value)):
        return str(value)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise StoreError(f'Weaviate cannot take the number {value}.')
        return repr(value) if isinstance(value, float) else str(value)
    return json.dumps(value)


def _parse_metadata(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


class WeaviateVectorStore(VectorStore):
    """
    A Weaviate collection (class) through the REST and GraphQL APIs, with vectors you provide.

        store = WeaviateVectorStore(url='http://localhost:8080', class_name='Docs', embedder=embedder)

    Objects need UUIDs, so each record id is stored as its UUID v5 (the same as the Python client's
    generate_uuid5(id)) and the original id goes in the recordId property. Metadata is kept whole as JSON in
    metadataJson, and its scalar keys (lowercase names) are also written as their own properties so `filter` can
    match them (Equal; a text property follows its tokenization). native_filter is a Weaviate where object, e.g.
    {'path': ['year'], 'operator': 'GreaterThan', 'valueInt': 2020}.

    A new class is created with a self-provided named vector (vector_name='default'); an existing class keeps its
    own vector setup (named or legacy). Scores: cosine 1 - distance, dot -distance, other metrics 1 / (1 + distance).
    """

    # API: https://docs.weaviate.io/weaviate/api/graphql/search-operators (GraphQL), /v1/batch/objects, /v1/schema
    def __init__(self, url='http://localhost:8080', api_key=None, class_name=None, text_key='text',
                 vector_name='default', distance='cosine', create_class=True, headers=None, batch_size=100,
                 embedder=None, timeout=120, retries=0, session=None):
        """
        Args:
            class_name: the collection; the first letter is capitalized.
            vector_name: the named vector (None for the legacy unnamed vector).
            headers: extra headers, e.g. module API keys.
        """
        super().__init__(embedder)
        if not class_name:
            raise StoreError('WeaviateVectorStore needs a class_name.')
        name = str(class_name)
        self.class_name = name[:1].upper() + name[1:]
        if not re.match(r'^[A-Z][_0-9A-Za-z]*$', self.class_name):
            raise StoreError(f"Invalid Weaviate class name '{class_name}'.")
        self.text_key = text_key or 'text'
        self.vector_name = vector_name
        self.distance = distance or 'cosine'
        self.create_class = create_class is not False
        self.batch_size = batch_size or 100
        self._class_ready = False
        self._lock = threading.Lock()
        request_headers = {'Content-Type': 'application/json', **(headers or {})}
        if api_key:
            request_headers['Authorization'] = f'Bearer {api_key}'
        self.client = HttpClient(url or 'http://localhost:8080', request_headers, timeout=timeout, retries=retries,
                                 session=session, label='Weaviate')

    def upsert(self, records):
        items = to_items(records)
        if not items:
            return []
        self._prepare(True)
        objects = []
        for item in items:
            properties = {self.text_key: item['text'], ID_PROPERTY: item['id'],
                          METADATA_PROPERTY: json.dumps(item['metadata'])}
            for key, value in item['metadata'].items():
                if key == self.text_key or key in RESERVED or not PROPERTY_NAME.match(key):
                    continue
                if _is_scalar(value) or (isinstance(value, list) and value and all(_is_scalar(v) for v in value)):
                    properties[key] = value
            obj = {'class': self.class_name, 'id': stable_uuid(item['id']), 'properties': properties}
            if self.vector_name:
                obj['vectors'] = {self.vector_name: item['vector']}
            else:
                obj['vector'] = item['vector']
            objects.append(obj)
        for start in range(0, len(objects), self.batch_size):
            results = self.client.request('POST', '/v1/batch/objects',
                                          {'objects': objects[start:start + self.batch_size]})
            # the batch answers 200 and reports failures per object
            for result in results if isinstance(results, list) else []:
                errors = (result.get('result') or {}).get('errors')
                if errors:
                    messages = '; '.join(error.get('message', '') for error in errors.get('error') or [])
                    raise StoreError(f"Weaviate error: object {result.get('id')}: "
                                     f"{messages or json.dumps(errors)}")
        return [item['id'] for item in items]

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        query_vector = self._query_vector(vector, text)
        self._prepare(False)
        near = {'vector': query_vector}
        if self.vector_name:
            near['targetVectors'] = [self.vector_name]
        args = [f'nearVector: {gql(near)}', f'limit: {int(top_k or 5)}']
        where = native_filter or to_weaviate_where(filter)
        if where:
            args.append(f'where: {gql(where)}')
        fields = f'{self.text_key} {ID_PROPERTY} {METADATA_PROPERTY} _additional {{ id distance }}'
        query = f"{{ Get {{ {self.class_name}({', '.join(args)}) {{ {fields} }} }} }}"
        data = self.client.request('POST', '/v1/graphql', {'query': query}) or {}
        if data.get('errors'):
            raise StoreError('Weaviate error: ' + '; '.join(error.get('message', '') for error in data['errors']))
        objects = ((data.get('data') or {}).get('Get') or {}).get(self.class_name) or []
        matches = []
        for obj in objects:
            additional = obj.get('_additional') or {}
            matches.append({
                'id': obj.get(ID_PROPERTY) if obj.get(ID_PROPERTY) is not None else additional.get('id'),
                'score': self._score(additional.get('distance')),
                'text': obj.get(self.text_key),
                'metadata': _parse_metadata(obj.get(METADATA_PROPERTY)),
            })
        return sorted_matches(matches)

    def delete(self, ids):
        uuids = [stable_uuid(record_id) for record_id in ids or []]
        for start in range(0, len(uuids), 1000):
            self.client.request('DELETE', '/v1/batch/objects', {
                'match': {'class': self.class_name,
                          'where': {'path': ['id'], 'operator': 'ContainsAny',
                                    'valueTextArray': uuids[start:start + 1000]}},
                'output': 'minimal',
            })

    def _score(self, distance):
        if not isinstance(distance, (int, float)):
            return None
        if self.distance == 'cosine':
            return 1 - distance
        if self.distance == 'dot':
            return -distance
        return 1 / (1 + distance)

    def _prepare(self, create):
        """Read the class once: an existing class decides the vector setup; a missing one is created on upsert."""
        with self._lock:
            if self._class_ready:
                return
            if self._load_class():
                self._class_ready = True
            elif create and self.create_class:
                self._create_class()
                self._class_ready = True

    def _load_class(self):
        try:
            schema = self.client.request('GET', f'/v1/schema/{self.class_name}')
        except StoreError as error:
            if error.status_code == 404:
                return False
            raise
        schema = schema or {}
        named = list((schema.get('vectorConfig') or {}).keys())
        if not named:
            self.vector_name = None
        elif self.vector_name not in named:
            self.vector_name = named[0]
        if self.vector_name:
            index_config = (schema['vectorConfig'][self.vector_name] or {}).get('vectorIndexConfig')
        else:
            index_config = schema.get('vectorIndexConfig')
        if index_config and index_config.get('distance'):
            self.distance = index_config['distance']
        return True

    def _create_class(self):
        vector_index_config = {'distance': self.distance}
        definition = {
            'class': self.class_name,
            'properties': [
                {'name': self.text_key, 'dataType': ['text']},
                {'name': ID_PROPERTY, 'dataType': ['text'], 'tokenization': 'field'},
                {'name': METADATA_PROPERTY, 'dataType': ['text'], 'indexFilterable': False,
                 'indexSearchable': False},
            ],
        }
        if self.vector_name:
            definition['vectorConfig'] = {self.vector_name: {'vectorizer': {'none': {}}, 'vectorIndexType': 'hnsw',
                                                             'vectorIndexConfig': vector_index_config}}
        else:
            definition.update({'vectorizer': 'none', 'vectorIndexType': 'hnsw',
                               'vectorIndexConfig': vector_index_config})
        try:
            self.client.request('POST', '/v1/schema', definition, retries=0)
        except StoreError:
            # created meanwhile by another call
            try:
                if self._load_class():
                    return
            except StoreError:
                pass
            raise
