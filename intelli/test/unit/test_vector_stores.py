"""
Offline tests for the vector stores: each store's requests (URL, body, headers) and how its answers become
[{'id', 'score', 'text', 'metadata'}]. The HTTP session, the DB-API connection and the MongoDB collection are fakes.
Live checks against real servers are in intelli/test/integration/test_vector_stores_live.py.
"""
import json
import os
import sys
import tempfile
import types
import unittest
import uuid

import requests

from intelli.store import (MemoryVectorStore, PineconeVectorStore, QdrantVectorStore, ChromaVectorStore,
                           WeaviateVectorStore, MilvusVectorStore, ElasticsearchVectorStore, PgVectorStore,
                           MongoDBAtlasVectorStore, FirestoreVectorStore, FirestoreChatHistory, VertexRAGStore,
                           VertexVectorSearchStore, VertexVectorSearchIndexStore, StoreError, matches_filter)
from intelli.store.google_cloud import Firestore, FirestoreVector
from intelli.store.qdrant_vector_store import stable_uuid
from intelli.store.vertex_vector_search_store import data_object_id
from intelli.store.vector_store import HttpClient


class FakeResponse:
    def __init__(self, data=None, status=200, text=None):
        self.status_code = status
        if text is None:
            text = '' if data is None else json.dumps(data)
        self.text = text
        self.content = text.encode()
        self._data = data

    def json(self):
        if self._data is None:
            raise ValueError('no json')
        return json.loads(json.dumps(self._data))


class FakeSession:
    """Answers requests from a script (a list, or a function of the call) and records them."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def request(self, method, url, **kwargs):
        body = kwargs.get('data')
        if isinstance(body, (bytes, str)) and kwargs.get('headers', {}).get('Content-Type') == 'application/json':
            body = json.loads(body)
        call = {'method': method, 'url': url, 'body': body, 'headers': kwargs.get('headers') or {},
                'files': kwargs.get('files'), 'params': kwargs.get('params')}
        self.calls.append(call)
        reply = self.replies(call) if callable(self.replies) else self.replies[len(self.calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, FakeResponse) else FakeResponse(reply)


def records():
    return [{'id': 'a', 'vector': [1.0, 0.0], 'text': 'alpha', 'metadata': {'lang': 'en'}},
            {'id': 'b', 'vector': [0.0, 1.0], 'text': 'beta', 'metadata': {'lang': 'fr'}}]


class TestBaseAndMemory(unittest.TestCase):
    def test_filters_and_memory_store(self):
        self.assertTrue(matches_filter({'lang': 'en', 'n': 1}, {'lang': ['en', 'de'], 'n': 1}))
        self.assertFalse(matches_filter({'lang': 'fr'}, {'lang': 'en'}))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'store.json')
            store = MemoryVectorStore(path=path)
            store.upsert(records())
            hits = store.query(vector=[1.0, 0.1], top_k=2)
            self.assertEqual([hit['id'] for hit in hits], ['a', 'b'])
            self.assertEqual(store.query(vector=[1.0, 0.1], filter={'lang': 'fr'})[0]['id'], 'b')
            self.assertEqual(store.query(vector=[1.0, 0.0], native_filter=lambda m, r: r['id'] == 'b')[0]['id'], 'b')
            reloaded = MemoryVectorStore(path=path)
            self.assertEqual(reloaded.count(), 2, 'records persist in the JSON file')
            reloaded.clear({'lang': 'en'})
            self.assertEqual([r['id'] for r in reloaded.get(['a', 'b'])], ['b'])
            reloaded.delete(['b'])
            self.assertEqual(MemoryVectorStore(path=path).count(), 0)
        with self.assertRaises(StoreError):
            MemoryVectorStore().search('needs an embedder')
        with self.assertRaises(StoreError):
            MemoryVectorStore().upsert([{'id': 'x'}])

    def test_http_client_retries_and_errors(self):
        session = FakeSession([FakeResponse({'error': 'busy'}, 503), FakeResponse({'ok': True})])
        client = HttpClient('http://x', retries=1, retry_delay=0, session=session, label='Test')
        self.assertEqual(client.request('GET', '/a'), {'ok': True})
        session = FakeSession([FakeResponse({'error': 'nope'}, 400)])
        with self.assertRaises(StoreError) as raised:
            HttpClient('http://x', retries=3, session=session, label='Test').request('GET', '/a')
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn('Test error: HTTP 400', str(raised.exception))
        self.assertEqual(len(session.calls), 1, 'a 400 is not retried')
        session = FakeSession([requests.exceptions.ConnectionError('down')])
        with self.assertRaises(StoreError):
            HttpClient('http://x', session=session, label='Test').request('GET', '/a')


class TestRestStores(unittest.TestCase):
    def test_pinecone(self):
        session = FakeSession([{}, {'matches': [
            {'id': 'b', 'score': 0.4, 'metadata': {'text': 'beta', 'lang': 'fr'}},
            {'id': 'a', 'score': 0.9, 'metadata': {'text': 'alpha', 'lang': 'en'}}]}, {}])
        store = PineconeVectorStore(api_key='pk', index_host='idx.svc.pinecone.io', namespace='ns', session=session)
        store.upsert([{**records()[0], 'metadata': {'lang': 'en', 'empty': None}}])
        call = session.calls[0]
        self.assertEqual(call['url'], 'https://idx.svc.pinecone.io/vectors/upsert')
        self.assertEqual(call['headers']['Api-Key'], 'pk')
        self.assertEqual(call['body'], {'vectors': [{'id': 'a', 'values': [1.0, 0.0],
                                                     'metadata': {'lang': 'en', 'text': 'alpha'}}], 'namespace': 'ns'})
        hits = store.query(vector=[1, 0], top_k=2, filter={'lang': ['en', 'fr'], 'n': 1})
        self.assertEqual([h['id'] for h in hits], ['a', 'b'])
        self.assertEqual(hits[0], {'id': 'a', 'score': 0.9, 'text': 'alpha', 'metadata': {'lang': 'en'}})
        self.assertEqual(session.calls[1]['body']['filter'], {'lang': {'$in': ['en', 'fr']}, 'n': {'$eq': 1}})
        store.delete(['a'])
        self.assertEqual(session.calls[2]['body'], {'ids': ['a'], 'namespace': 'ns'})

    def test_qdrant(self):
        session = FakeSession([{'result': {'exists': False}}, {'result': True}, {'result': {}}, {'result': {'points': [
            {'id': stable_uuid('a'), 'score': 0.8, 'payload': {'text': 'alpha', 'metadata': {'lang': 'en'}, 'id': 'a'}}]}},
            {'result': {}}])
        store = QdrantVectorStore(url='http://q:6333', api_key='qk', collection='docs', session=session)
        store.upsert(records())
        self.assertEqual(session.calls[1]['method'], 'PUT')
        self.assertEqual(session.calls[1]['body'], {'vectors': {'size': 2, 'distance': 'Cosine'}})
        points = session.calls[2]['body']['points']
        self.assertEqual(points[0]['id'], str(uuid.uuid5(uuid.NAMESPACE_DNS, 'a')))
        self.assertEqual(points[0]['payload'], {'text': 'alpha', 'metadata': {'lang': 'en'}, 'id': 'a'})
        self.assertEqual(session.calls[2]['headers']['api-key'], 'qk')
        hits = store.query(vector=[1, 0], filter={'lang': 'en', 'tag': ['x', 'y']})
        self.assertEqual(hits, [{'id': 'a', 'score': 0.8, 'text': 'alpha', 'metadata': {'lang': 'en'}}])
        self.assertEqual(session.calls[3]['body']['filter'], {'must': [
            {'key': 'metadata.lang', 'match': {'value': 'en'}}, {'key': 'metadata.tag', 'match': {'any': ['x', 'y']}}]})
        store.delete(['a'])
        self.assertEqual(session.calls[4]['body'], {'points': [stable_uuid('a')]})
        self.assertEqual(stable_uuid('6F9619FF-8B86-D011-B42D-00C04FC964FF'), '6f9619ff-8b86-d011-b42d-00c04fc964ff')

    def test_chroma(self):
        session = FakeSession([{'id': 'col-1', 'configuration_json': {'hnsw': {'space': 'cosine'}}}, {}, {
            'ids': [['b', 'a']], 'documents': [['beta', 'alpha']], 'metadatas': [[{'lang': 'fr'}, None]],
            'distances': [[0.6, 0.1]]}, {}])
        store = ChromaVectorStore(url='http://c:8000', collection='docs', api_key='ck', session=session)
        store.upsert(records())
        self.assertEqual(session.calls[0]['body'], {'name': 'docs', 'metadata': {'hnsw:space': 'cosine'},
                                                    'get_or_create': True})
        self.assertEqual(session.calls[1]['url'],
                         'http://c:8000/api/v2/tenants/default_tenant/databases/default_database/collections/col-1/upsert')
        self.assertEqual(session.calls[1]['headers']['x-chroma-token'], 'ck')
        hits = store.query(vector=[1, 0], filter={'lang': 'en', 'n': [1, 2]})
        self.assertEqual([(h['id'], round(h['score'], 2)) for h in hits], [('a', 0.9), ('b', 0.4)])
        self.assertEqual(hits[0]['metadata'], {})
        self.assertEqual(session.calls[2]['body']['where'], {'$and': [{'lang': {'$eq': 'en'}}, {'n': {'$in': [1, 2]}}]})
        store.delete(['a'])
        self.assertTrue(session.calls[3]['url'].endswith('/collections/col-1/delete'))

    def test_weaviate(self):
        def reply(call):
            if call['url'].endswith('/v1/schema/Docs') and call['method'] == 'GET':
                return FakeResponse({'error': 'not found'}, 404)
            if call['url'].endswith('/v1/graphql'):
                return {'data': {'Get': {'Docs': [{'text': 'alpha', 'recordId': 'a', 'metadataJson': '{"lang": "en"}',
                                                   '_additional': {'id': stable_uuid('a'), 'distance': 0.2}}]}}}
            return [] if call['url'].endswith('/v1/batch/objects') else {}

        session = FakeSession(reply)
        store = WeaviateVectorStore(url='http://w:8080', class_name='docs', api_key='wk', session=session)
        store.upsert([{**records()[0], 'metadata': {'lang': 'en', 'Upper': 'skip', 'nested': {'x': 1}}}])
        schema = session.calls[1]['body']
        self.assertEqual(schema['class'], 'Docs')
        self.assertEqual(schema['vectorConfig']['default']['vectorizer'], {'none': {}})
        obj = session.calls[2]['body']['objects'][0]
        self.assertEqual(obj['id'], stable_uuid('a'))
        self.assertEqual(obj['properties'], {'text': 'alpha', 'recordId': 'a',
                                             'metadataJson': json.dumps({'lang': 'en', 'Upper': 'skip', 'nested': {'x': 1}}),
                                             'lang': 'en'})
        self.assertEqual(obj['vectors'], {'default': [1.0, 0.0]})
        hits = store.query(vector=[1, 0], top_k=3, filter={'lang': 'en', 'year': [2024, 2025]})
        self.assertEqual(hits, [{'id': 'a', 'score': 0.8, 'text': 'alpha', 'metadata': {'lang': 'en'}}])
        query = session.calls[3]['body']['query']
        self.assertIn('nearVector: {vector: [1, 0], targetVectors: ["default"]}', query)
        self.assertIn('operator: And', query)
        self.assertIn('{path: ["year"], operator: Equal, valueNumber: 2024}', query)
        self.assertEqual(session.calls[3]['headers']['Authorization'], 'Bearer wk')
        store.delete(['a'])
        self.assertEqual(session.calls[4]['body']['match']['where']['valueTextArray'], [stable_uuid('a')])

    def test_weaviate_batch_errors(self):
        session = FakeSession(lambda call: {'class': 'Docs', 'vectorConfig': {'v': {'vectorIndexConfig': {'distance': 'dot'}}}}
                              if call['method'] == 'GET' else [{'id': 'x', 'result': {'errors': {'error': [{'message': 'bad'}]}}}])
        store = WeaviateVectorStore(class_name='Docs', session=session)
        with self.assertRaises(StoreError) as raised:
            store.upsert(records())
        self.assertIn('bad', str(raised.exception))
        self.assertEqual((store.vector_name, store.distance), ('v', 'dot'), 'an existing class keeps its vector setup')

    def test_milvus(self):
        session = FakeSession([{'code': 0, 'data': {'has': False}}, {'code': 0, 'data': {}}, {'code': 0, 'data': {}},
                               {'code': 0, 'data': [{'id': 'a', 'distance': 0.95, 'text': 'alpha', 'metadata': '{"lang": "en"}'}]},
                               {'code': 0, 'data': {}}, {'code': 1100, 'message': 'collection not found'}])
        store = MilvusVectorStore(url='http://m:19530', token='root:Milvus', collection='docs', db_name='app',
                                  session=session)
        store.upsert(records() + [{**records()[0], 'text': 'alpha 2'}])
        self.assertEqual(session.calls[1]['body']['schema']['fields'][1]['elementTypeParams'], {'dim': '2'})
        self.assertEqual(session.calls[2]['body']['dbName'], 'app')
        self.assertEqual([row['id'] for row in session.calls[2]['body']['data']], ['a', 'b'], 'the last record wins')
        self.assertEqual(session.calls[2]['body']['data'][0]['text'], 'alpha 2')
        self.assertEqual(session.calls[2]['headers']['Authorization'], 'Bearer root:Milvus')
        hits = store.query(vector=[1, 0], filter={'lang': 'en', 'n': [1, 2]})
        self.assertEqual(hits, [{'id': 'a', 'score': 0.95, 'text': 'alpha', 'metadata': {'lang': 'en'}}])
        self.assertEqual(session.calls[3]['body']['filter'], 'metadata["lang"] == "en" and metadata["n"] in [1, 2]')
        store.delete(['a'])
        self.assertEqual(session.calls[4]['body']['filter'], 'id in ["a"]')
        with self.assertRaises(StoreError) as raised:
            store.delete(['b'])
        self.assertIn('collection not found (code 1100)', str(raised.exception))

    def test_elasticsearch(self):
        session = FakeSession([FakeResponse(status=404, text=''), {'acknowledged': True}, {'errors': False},
                               {'hits': {'hits': [{'_id': 'a', '_score': 0.95, '_source': {'text': 'alpha',
                                                                                           'metadata': {'lang': 'en'}}}]}},
                               {'errors': True, 'items': [{'delete': {'_id': 'x', 'status': 404, 'error': {'type': 'missing'}}}]}])
        store = ElasticsearchVectorStore(url='http://e:9200', username='elastic', password='pw', index='docs',
                                         session=session)
        store.upsert(records())
        self.assertEqual(session.calls[0]['method'], 'HEAD')
        mapping = session.calls[1]['body']['mappings']
        self.assertEqual(mapping['properties']['embedding'], {'type': 'dense_vector', 'dims': 2, 'index': True,
                                                              'similarity': 'cosine'})
        bulk = session.calls[2]
        self.assertEqual(bulk['url'], 'http://e:9200/_bulk?refresh=wait_for')
        self.assertEqual(bulk['headers']['Content-Type'], 'application/x-ndjson')
        lines = bulk['body'].decode().strip().split('\n')
        self.assertEqual(json.loads(lines[0]), {'index': {'_index': 'docs', '_id': 'a'}})
        self.assertTrue(bulk['headers']['Authorization'].startswith('Basic '))
        hits = store.query(vector=[1, 0], top_k=2, filter={'lang': ['en']})
        self.assertEqual([(h['id'], round(h['score'], 2)) for h in hits], [('a', 0.9)])
        knn = session.calls[3]['body']['knn']
        self.assertEqual((knn['k'], knn['num_candidates']), (2, 100))
        self.assertEqual(knn['filter'], {'bool': {'filter': [{'terms': {'metadata.lang': ['en']}}]}})
        store.delete(['x'])  # a delete of a missing id is not an error


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None

    def execute(self, sql, params=None):
        self.connection.statements.append((' '.join(sql.split()), params))
        if sql.lstrip().startswith('SELECT'):
            self.description = [('id',), ('text',), ('metadata',), ('score',)]

    def fetchall(self):
        return [('a', 'alpha', '{"lang": "en"}', 0.9), ('b', 'beta', {'lang': 'fr'}, 0.2)]

    def close(self):
        pass


class FakeConnection:
    def __init__(self):
        self.statements = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class TestDriverStores(unittest.TestCase):
    def test_pgvector(self):
        connection = FakeConnection()
        store = PgVectorStore(connection=connection, table='app.docs')
        store.upsert(records())
        sql = [statement for statement, _ in connection.statements]
        self.assertEqual(sql[0], 'CREATE EXTENSION IF NOT EXISTS vector')
        self.assertIn('embedding vector(2) NOT NULL', sql[1])
        self.assertIn('CREATE INDEX IF NOT EXISTS docs_embedding_idx ON app.docs USING hnsw', sql[2])
        self.assertIn('ON CONFLICT (id) DO UPDATE', sql[3])
        self.assertEqual(connection.statements[3][1], ['a', 'alpha', '{"lang": "en"}', '[1,0]',
                                                       'b', 'beta', '{"lang": "fr"}', '[0,1]'])
        hits = store.query(vector=[1, 0], top_k=2, filter={'lang': 'en', 'tag': ['x']})
        self.assertEqual(hits[0], {'id': 'a', 'score': 0.9, 'text': 'alpha', 'metadata': {'lang': 'en'}})
        self.assertEqual(hits[1]['metadata'], {'lang': 'fr'})
        statement, params = connection.statements[4]
        self.assertIn('WHERE %s::jsonb @> (metadata -> %s) AND metadata @> %s::jsonb', statement)
        self.assertEqual(params, ['[1,0]', '["x"]', 'tag', '{"lang": "en"}', '[1,0]', 2])
        store.query(vector=[1, 0], native_filter={'sql': "metadata->>'lang' = %s", 'params': ['en']})
        self.assertIn("WHERE (metadata->>'lang' = %s)", connection.statements[5][0])
        store.delete(['a', 'b'])
        self.assertEqual(connection.statements[6], ('DELETE FROM app.docs WHERE id = ANY(%s)', [['a', 'b']]))
        with self.assertRaises(StoreError):
            PgVectorStore(connection=FakeConnection(), table='docs; DROP TABLE x')

    def test_mongodb_atlas(self):
        class ReplaceOne:
            def __init__(self, filter, replacement, upsert=False):
                self.filter, self.replacement, self.upsert = filter, replacement, upsert

        class Collection:
            def __init__(self):
                self.writes, self.pipelines, self.deleted, self.indexes = [], [], [], []

            def bulk_write(self, operations, ordered=True):
                self.writes.extend(operations)

            def aggregate(self, pipeline):
                self.pipelines.append(pipeline)
                return iter([{'_id': 'a', 'text': 'alpha', 'metadata': {'lang': 'en'}, 'score': 0.95}])

            def delete_many(self, query):
                self.deleted.append(query)

            def create_search_index(self, model):
                self.indexes.append(model)

        sys.modules['pymongo'] = types.SimpleNamespace(ReplaceOne=ReplaceOne)
        try:
            collection = Collection()
            store = MongoDBAtlasVectorStore(collection=collection, path='embedding.values')
            store.upsert(records())
        finally:
            del sys.modules['pymongo']
        self.assertEqual(collection.writes[0].filter, {'_id': 'a'})
        self.assertEqual(collection.writes[0].replacement, {'text': 'alpha', 'metadata': {'lang': 'en'},
                                                            'embedding': {'values': [1.0, 0.0]}})
        self.assertTrue(collection.writes[0].upsert)
        hits = store.query(vector=[1, 0], top_k=3, filter={'lang': 'en', 'n': [1]})
        self.assertEqual([(h['id'], round(h['score'], 2)) for h in hits], [('a', 0.9)])
        stage = collection.pipelines[0][0]['$vectorSearch']
        self.assertEqual((stage['numCandidates'], stage['limit']), (60, 3))
        self.assertEqual(stage['filter'], {'$and': [{'metadata.lang': {'$eq': 'en'}}, {'metadata.n': {'$in': [1]}}]})
        store.delete(['a'])
        self.assertEqual(collection.deleted, [{'_id': {'$in': ['a']}}])
        store.create_index(dimension=2, filter_fields=['lang'])
        self.assertEqual(collection.indexes[0]['definition']['fields'][1], {'type': 'filter', 'path': 'metadata.lang'})


ROOT = 'projects/p1/databases/(default)/documents'


class TestGoogleCloudStores(unittest.TestCase):
    def test_firestore_values(self):
        self.assertEqual(Firestore.to_value(FirestoreVector([0.5])), {'mapValue': {'fields': {
            '__type__': {'stringValue': '__vector__'}, 'value': {'arrayValue': {'values': [{'doubleValue': 0.5}]}}}}})
        data = {'a': 1, 'b': 1.5, 'c': 'x', 'd': [True, None], 'e': {'f': 'g'}}
        self.assertEqual(Firestore.from_fields(Firestore.to_fields(data)), data)
        self.assertEqual(Firestore.doc_id('docs/a.md#0'), 'docs%2Fa%2Emd%230')
        self.assertEqual(Firestore.where({'my-key': 'v'})['fieldFilter']['field']['fieldPath'], '`my-key`')

    def test_firestore_vector_store(self):
        def document(record_id, distance):
            return {'document': {'name': f'{ROOT}/vectors/{record_id}', 'fields': Firestore.to_fields(
                {'id': record_id, 'text': f'text {record_id}', 'metadata': {'lang': 'en'}, '_distance': distance})}}

        session = FakeSession([{}, [document('a', 0.1), document('b', 0.4)], {}])
        store = FirestoreVectorStore(project_id='p1', collection='vectors', access_token='tok', session=session)
        store.upsert([records()[0]])
        self.assertEqual(session.calls[0]['url'], f'https://firestore.googleapis.com/v1/{ROOT}:commit')
        self.assertEqual(session.calls[0]['headers']['Authorization'], 'Bearer tok')
        write = session.calls[0]['body']['writes'][0]['update']
        self.assertEqual(write['name'], f'{ROOT}/vectors/a')
        self.assertEqual(write['fields']['embedding']['mapValue']['fields']['__type__']['stringValue'], '__vector__')
        hits = store.query(vector=[1, 0], top_k=2, filter={'lang': 'en'})
        self.assertEqual([(h['id'], round(h['score'], 2)) for h in hits], [('a', 0.9), ('b', 0.6)])
        query = session.calls[1]['body']['structuredQuery']
        self.assertEqual(query['where'], {'fieldFilter': {'field': {'fieldPath': 'metadata.lang'}, 'op': 'EQUAL',
                                                          'value': {'stringValue': 'en'}}})
        self.assertEqual(query['findNearest']['distanceResultField'], '_distance')
        store.delete(['a'])
        self.assertEqual(session.calls[2]['body']['writes'], [{'delete': f'{ROOT}/vectors/a'}])
        self.assertIn('"dimension":"768"', store.index_command(768))

    def test_firestore_chat_history(self):
        name = f'{ROOT}/conversations/c1'

        def message(record_id, role, content, seq):
            return {'document': {'name': f'{name}/messages/{record_id}', 'fields': Firestore.to_fields(
                {'id': record_id, 'role': role, 'content': content, 'createdAt': '2026-10-05T00:00:00.000Z', 'seq': seq})}}

        latest = [message('m2', 'assistant', 'hello', 2), message('m1', 'user', 'hi', 1)]
        session = FakeSession([FakeResponse({'error': {}}, 404), {}, {}, latest,
                               [{'document': {'name': name, 'fields': Firestore.to_fields(
                                   {'userId': 'u1', 'updatedAt': '2026-10-05T01:00:00.000Z'})}}], latest, {}])
        history = FirestoreChatHistory(project_id='p1', access_token='tok', session=session)
        history.save_conversation({'id': 'c1', 'user_id': 'u1'})
        saved = session.calls[1]['body']['writes'][0]['update']
        self.assertEqual(saved['name'], name)
        self.assertEqual(saved['fields']['userId'], {'stringValue': 'u1'}, 'the IntelliNode field names are kept')
        history.add_messages('c1', [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}])
        writes = session.calls[2]['body']['writes']
        self.assertTrue(writes[0]['update']['name'].startswith(f'{name}/messages/'))
        self.assertIn('createdAt', writes[0]['update']['fields'])
        self.assertGreater(int(writes[1]['update']['fields']['seq']['integerValue']),
                           int(writes[0]['update']['fields']['seq']['integerValue']))
        self.assertEqual(writes[2]['updateMask'], {'fieldPaths': ['updatedAt']})
        messages = history.get_messages('c1', limit=2)
        self.assertEqual([m['content'] for m in messages], ['hi', 'hello'], 'oldest first')
        self.assertEqual(messages[0]['created_at'], '2026-10-05T00:00:00.000Z')
        self.assertEqual(session.calls[3]['body']['structuredQuery']['orderBy'],
                         [{'field': {'fieldPath': 'seq'}, 'direction': 'DESCENDING'}])
        conversations = history.list_conversations(user_id='u1')
        self.assertEqual([(c['id'], c['user_id']) for c in conversations], [('c1', 'u1')])
        history.delete_conversation('c1')
        self.assertEqual([w['delete'] for w in session.calls[6]['body']['writes']],
                         [f'{name}/messages/m2', f'{name}/messages/m1', name])

    def test_vertex_rag_store(self):
        corpus = 'projects/p1/locations/europe-west4/ragCorpora/123'
        session = FakeSession([
            {'contexts': {'contexts': [{'sourceUri': 'policy.pdf', 'sourceDisplayName': 'Policy',
                                        'text': 'Refunds in 30 days', 'score': 0.25,
                                        'chunk': {'chunkId': 'c9', 'pageSpan': {'firstPage': 2, 'lastPage': 2}}}]}},
            {'ragFile': {'name': f'{corpus}/ragFiles/f1'}},
            {'name': 'projects/p1/locations/europe-west4/operations/o1', 'done': True,
             'response': {'importedRagFilesCount': '2'}},
        ])
        rag = VertexRAGStore(project_id='p1', location='europe-west4', corpus='123', access_token='tok',
                             session=session)
        hits = rag.query(text='refunds', top_k=3)
        self.assertEqual(hits, [{'id': 'c9', 'score': 0.75, 'text': 'Refunds in 30 days', 'metadata': {
            'source': 'policy.pdf', 'title': 'Policy', 'pages': {'firstPage': 2, 'lastPage': 2}}}])
        self.assertEqual(session.calls[0]['url'], 'https://europe-west4-aiplatform.googleapis.com/v1/projects/p1/'
                                                  'locations/europe-west4:retrieveContexts')
        self.assertEqual(session.calls[0]['body'], {'vertexRagStore': {'ragResources': [{'ragCorpus': corpus}]},
                                                    'query': {'text': 'refunds', 'ragRetrievalConfig': {'topK': 3}}})
        names = rag.add_documents([{'id': 'faq', 'text': 'Q and A'}])
        self.assertEqual(names, [f'{corpus}/ragFiles/f1'])
        upload = session.calls[1]
        self.assertEqual(upload['url'], f'https://europe-west4-aiplatform.googleapis.com/upload/v1/{corpus}/'
                                        'ragFiles:upload')
        self.assertEqual(upload['headers']['X-Goog-Upload-Protocol'], 'multipart')
        self.assertNotIn('Content-Type', upload['headers'], 'requests sets the multipart boundary')
        metadata = json.loads(upload['files']['metadata'][1])
        self.assertEqual(metadata['ragFile'], {'displayName': 'faq.txt'})
        self.assertEqual(upload['files']['file'][1], b'Q and A')
        result = rag.import_files(['gs://bucket/docs/', 'https://drive.google.com/drive/folders/FOLDER1'])
        self.assertEqual(result, {'importedRagFilesCount': '2'})
        config = session.calls[2]['body']['importRagFilesConfig']
        self.assertEqual(config['gcsSource'], {'uris': ['gs://bucket/docs/']})
        self.assertEqual(config['googleDriveSource']['resourceIds'],
                         [{'resourceType': 'RESOURCE_TYPE_FOLDER', 'resourceId': 'FOLDER1'}])
        self.assertEqual(rag.tool(top_k=2), {'retrieval': {'vertexRagStore': {
            'ragResources': [{'ragCorpus': corpus}], 'ragRetrievalConfig': {'topK': 2}}}})
        with self.assertRaises(StoreError):
            rag.upsert([])
        with self.assertRaises(StoreError):
            rag.query(vector=[1])

    def test_vertex_vector_search(self):
        self.assertEqual(data_object_id('doc-1'), 'doc-1')
        mapped = data_object_id('Docs/Policy.md#0')
        self.assertRegex(mapped, r'^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$')
        self.assertEqual(mapped, data_object_id('Docs/Policy.md#0'))
        self.assertNotEqual(mapped, data_object_id('docs/policy.md#0'))

        collection = 'projects/p1/locations/us-central1/collections/kb'
        session = FakeSession([FakeResponse({'error': {}}, 404), {}, {'results': [
            {'dataObject': {'dataObjectId': 'a', 'data': {'text': 'hello', 'sourceId': 'A', 'lang': 'en'}},
             'distance': 0.2}]}])
        store = VertexVectorSearchStore(project_id='p1', collection='kb', access_token='tok', session=session)
        store.upsert([{'id': 'A', 'vector': [0.1], 'text': 'hello', 'metadata': {'lang': 'en'}}])
        self.assertEqual(session.calls[0]['method'], 'DELETE')
        self.assertEqual(session.calls[1]['url'], f'https://vectorsearch.googleapis.com/v1/{collection}/dataObjects:batchCreate')
        self.assertEqual(session.calls[1]['body']['requests'][0]['dataObject'], {
            'data': {'lang': 'en', 'text': 'hello', 'sourceId': 'A'}, 'vectors': {'embedding': {'dense': {'values': [0.1]}}}})
        hits = store.query(vector=[0.1], top_k=1, filter={'lang': 'en'})
        self.assertEqual([(h['id'], round(h['score'], 2), h['text'], h['metadata']) for h in hits],
                         [('A', 0.8, 'hello', {'lang': 'en'})])
        self.assertEqual(session.calls[2]['body']['vectorSearch']['filter'], {'lang': {'$eq': 'en'}})

        session = FakeSession([{}, {'nearestNeighbors': [{'neighbors': [
            {'datapoint': {'datapointId': 'a', 'embeddingMetadata': {'text': 't', 'metadata': {'lang': 'en'}}},
             'distance': 0.9}]}]}])
        index_store = VertexVectorSearchIndexStore(project_id='p1', index='i1', index_endpoint='e1',
                                                   deployed_index_id='d1',
                                                   public_endpoint_domain='1.us-central1-2.vdb.vertexai.goog',
                                                   restrict_keys=['lang'], access_token='tok', session=session)
        index_store.upsert([{'id': 'a', 'vector': [1], 'text': 't', 'metadata': {'lang': 'en'}}])
        self.assertEqual(session.calls[0]['url'], 'https://us-central1-aiplatform.googleapis.com/v1/projects/p1/'
                                                  'locations/us-central1/indexes/i1:upsertDatapoints')
        self.assertEqual(session.calls[0]['body']['datapoints'][0]['restricts'], [{'namespace': 'lang', 'allowList': ['en']}])
        hits = index_store.query(vector=[1], top_k=1, filter={'lang': 'en'})
        self.assertEqual(hits, [{'id': 'a', 'score': 0.9, 'text': 't', 'metadata': {'lang': 'en'}}])
        self.assertEqual(session.calls[1]['url'], 'https://1.us-central1-2.vdb.vertexai.goog/v1/projects/p1/'
                                                  'locations/us-central1/indexEndpoints/e1:findNeighbors')


if __name__ == '__main__':
    unittest.main()
