"""
Live checks for the vector store adapters against real servers. Each store runs only when its settings are present,
otherwise it is skipped. No embedding key is needed: a fake embedder puts one keyword per axis.

    QDRANT_URL [QDRANT_API_KEY]
    CHROMA_URL [CHROMA_API_KEY CHROMA_TENANT CHROMA_DATABASE]
    PINECONE_API_KEY PINECONE_INDEX_HOST   (an existing cosine index of VECTOR_TEST_DIMENSION, default 8)
    WEAVIATE_URL [WEAVIATE_API_KEY]
    MILVUS_URL [MILVUS_TOKEN]
    ELASTICSEARCH_URL [ELASTICSEARCH_API_KEY | ELASTICSEARCH_USERNAME ELASTICSEARCH_PASSWORD]
    PG_CONNECTION_STRING                   (needs psycopg or psycopg2, and the pgvector extension)
    MONGODB_URI MONGODB_DB MONGODB_COLLECTION [MONGODB_INDEX] [MONGODB_CREATE_INDEX=1]
                                           (needs pymongo and an Atlas Vector Search index on `embedding` with
                                            filter fields metadata.topic and metadata.lang; MONGODB_CREATE_INDEX=1
                                            creates it, e.g. on a local mongodb/mongodb-atlas-local container)

Local servers without API keys (Docker):

    docker run -d -p 6333:6333 qdrant/qdrant
    docker run -d -p 8000:8000 chromadb/chroma
    docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true cr.weaviate.io/semitechnologies/weaviate
    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg17
    docker run -d -p 9200:9200 -e discovery.type=single-node -e xpack.security.enabled=false elasticsearch:8.19.22
    docker run -d -p 27017:27017 mongodb/mongodb-atlas-local:8.0     (MONGODB_URI=mongodb://localhost:27017/?directConnection=true)
    Milvus: the standalone_embed.sh script from milvus.io, or milvusdb/milvus with `milvus run standalone`

Collections, classes, indexes and tables are created with a unique name and dropped at the end.

Run:
    QDRANT_URL=http://localhost:6333 python3 -m pytest intelli/test/integration/test_vector_stores_live.py -q -s
"""
import os
import time
import unittest
import uuid

from intelli.store import (QdrantVectorStore, ChromaVectorStore, PineconeVectorStore, WeaviateVectorStore,
                           MilvusVectorStore, ElasticsearchVectorStore, PgVectorStore, MongoDBAtlasVectorStore,
                           StoreError)

ENV = os.environ
DIMENSION = int(ENV.get('VECTOR_TEST_DIMENSION') or 8)
KEYWORDS = ['cat', 'dog', 'car', 'bus', 'sofa', 'park', 'tires', 'noon']
RUN = uuid.uuid4().hex[:10]


def embedder(texts):
    return [[1 if index < len(KEYWORDS) and KEYWORDS[index] in str(text).lower() else 0.01
             for index in range(DIMENSION)] for text in texts]


DOCUMENTS = [
    {'id': f'{RUN}-cat', 'text': 'The cat sleeps on the sofa', 'metadata': {'topic': 'animals', 'lang': 'en'}},
    {'id': f'{RUN}-dog', 'text': 'A dog runs in the park', 'metadata': {'topic': 'animals', 'lang': 'fr'}},
    {'id': f'{RUN}-car', 'text': 'The car needs new tires', 'metadata': {'topic': 'vehicles', 'lang': 'en'}},
    {'id': f'{RUN}-bus', 'text': 'The bus leaves at noon', 'metadata': {'topic': 'vehicles', 'lang': 'de'}},
]


def eventually(read, check, timeout=60):
    """Some stores (Pinecone, Atlas, Elasticsearch without refresh) are eventually consistent."""
    deadline = time.time() + timeout
    while True:
        value = read()
        try:
            check(value)
            return value
        except AssertionError:
            if time.time() > deadline:
                raise
            time.sleep(1)


class StoreContract:
    """The same checks for every store."""

    eventual = False

    def make_store(self):
        raise NotImplementedError

    def cleanup(self, store):
        pass

    def test_contract(self):
        store = self.make_store()
        try:
            ids = store.add_documents(DOCUMENTS)
            self.assertEqual(ids, [document['id'] for document in DOCUMENTS])
            first = lambda: store.search('Where does the cat sleep?', 2)

            def check_first(hits):
                self.assertTrue(hits, 'no hits')
                self.assertEqual(hits[0]['id'], f'{RUN}-cat')
                self.assertEqual(hits[0]['text'], 'The cat sleeps on the sofa')
                self.assertEqual(hits[0]['metadata'].get('topic'), 'animals')
                # the query shares one of the document's two keywords: cosine ~0.71 on every store
                self.assertAlmostEqual(hits[0]['score'], 0.714, delta=0.01)

            eventually(first, check_first, 60 if self.eventual else 5)

            filtered = eventually(lambda: store.search('cat dog car bus', 4, {'topic': 'vehicles'}),
                                  lambda hits: self.assertEqual(sorted(h['id'] for h in hits),
                                                                sorted([f'{RUN}-car', f'{RUN}-bus'])),
                                  60 if self.eventual else 5)
            print(type(store).__name__, 'filter ok:', [h['id'] for h in filtered])
            eventually(lambda: store.search('cat dog car bus', 4, {'lang': ['en', 'de']}),
                       lambda hits: self.assertEqual(sorted(h['id'] for h in hits),
                                                     sorted([f'{RUN}-cat', f'{RUN}-car', f'{RUN}-bus'])),
                       60 if self.eventual else 5)
            eventually(lambda: store.search('cat dog car bus', 4, {'topic': 'animals', 'lang': 'fr'}),
                       lambda hits: self.assertEqual([h['id'] for h in hits], [f'{RUN}-dog']),
                       60 if self.eventual else 5)

            # upsert replaces a record
            store.add_documents([{'id': f'{RUN}-cat', 'text': 'The cat now sleeps in the park',
                                  'metadata': {'topic': 'animals', 'lang': 'en', 'version': 2}}])
            eventually(lambda: store.query(text='cat park', top_k=1),
                       lambda hits: self.assertEqual(hits[0]['metadata'].get('version'), 2),
                       60 if self.eventual else 5)

            store.delete([f'{RUN}-cat', f'{RUN}-dog'])
            eventually(lambda: store.search('cat dog car bus', 4),
                       lambda hits: self.assertEqual(sorted(h['id'] for h in hits),
                                                     sorted([f'{RUN}-car', f'{RUN}-bus'])),
                       60 if self.eventual else 5)
        finally:
            self.cleanup(store)


@unittest.skipUnless(ENV.get('QDRANT_URL'), 'Set QDRANT_URL')
class TestQdrantLive(StoreContract, unittest.TestCase):
    def make_store(self):
        return QdrantVectorStore(url=ENV['QDRANT_URL'], api_key=ENV.get('QDRANT_API_KEY'),
                                 collection=f'intelli_{RUN}', embedder=embedder)

    def cleanup(self, store):
        store.client.request('DELETE', store._path())


@unittest.skipUnless(ENV.get('CHROMA_URL'), 'Set CHROMA_URL')
class TestChromaLive(StoreContract, unittest.TestCase):
    def make_store(self):
        return ChromaVectorStore(url=ENV['CHROMA_URL'], api_key=ENV.get('CHROMA_API_KEY'),
                                 tenant=ENV.get('CHROMA_TENANT') or 'default_tenant',
                                 database=ENV.get('CHROMA_DATABASE') or 'default_database',
                                 collection=f'intelli_{RUN}', embedder=embedder)

    def cleanup(self, store):
        store.client.request('DELETE', f'{store._database_path()}/collections/{store.collection}')


@unittest.skipUnless(ENV.get('PINECONE_API_KEY') and ENV.get('PINECONE_INDEX_HOST'),
                     'Set PINECONE_API_KEY and PINECONE_INDEX_HOST')
class TestPineconeLive(StoreContract, unittest.TestCase):
    eventual = True

    def make_store(self):
        return PineconeVectorStore(api_key=ENV['PINECONE_API_KEY'], index_host=ENV['PINECONE_INDEX_HOST'],
                                   namespace=f'intelli-{RUN}', embedder=embedder)

    def cleanup(self, store):
        try:
            store.client.request('POST', '/vectors/delete', {'deleteAll': True, 'namespace': store.namespace})
        except StoreError:
            pass


@unittest.skipUnless(ENV.get('WEAVIATE_URL'), 'Set WEAVIATE_URL')
class TestWeaviateLive(StoreContract, unittest.TestCase):
    def make_store(self):
        return WeaviateVectorStore(url=ENV['WEAVIATE_URL'], api_key=ENV.get('WEAVIATE_API_KEY'),
                                   class_name=f'Intelli{RUN}', embedder=embedder)

    def cleanup(self, store):
        store.client.request('DELETE', f'/v1/schema/{store.class_name}')


@unittest.skipUnless(ENV.get('MILVUS_URL'), 'Set MILVUS_URL')
class TestMilvusLive(StoreContract, unittest.TestCase):
    def make_store(self):
        return MilvusVectorStore(url=ENV['MILVUS_URL'], token=ENV.get('MILVUS_TOKEN'), collection=f'intelli_{RUN}',
                                 consistency_level='Strong', embedder=embedder)

    def cleanup(self, store):
        store._request('/v2/vectordb/collections/drop', {})


@unittest.skipUnless(ENV.get('ELASTICSEARCH_URL'), 'Set ELASTICSEARCH_URL')
class TestElasticsearchLive(StoreContract, unittest.TestCase):
    def make_store(self):
        return ElasticsearchVectorStore(url=ENV['ELASTICSEARCH_URL'], api_key=ENV.get('ELASTICSEARCH_API_KEY'),
                                        username=ENV.get('ELASTICSEARCH_USERNAME'),
                                        password=ENV.get('ELASTICSEARCH_PASSWORD'), index=f'intelli-{RUN}',
                                        embedder=embedder)

    def cleanup(self, store):
        store.client.request('DELETE', store._path())


def _pg_connect(url):
    try:
        import psycopg
        return psycopg.connect(url)
    except ImportError:
        import psycopg2
        return psycopg2.connect(url)


@unittest.skipUnless(ENV.get('PG_CONNECTION_STRING'), 'Set PG_CONNECTION_STRING')
class TestPgVectorLive(StoreContract, unittest.TestCase):
    def make_store(self):
        self.connection = _pg_connect(ENV['PG_CONNECTION_STRING'])
        return PgVectorStore(connection=self.connection, table=f'intelli_{RUN}', embedder=embedder)

    def cleanup(self, store):
        store._execute(f'DROP TABLE IF EXISTS {store.table}')
        self.connection.close()


@unittest.skipUnless(ENV.get('MONGODB_URI') and ENV.get('MONGODB_DB') and ENV.get('MONGODB_COLLECTION'),
                     'Set MONGODB_URI, MONGODB_DB and MONGODB_COLLECTION')
class TestMongoDBAtlasLive(StoreContract, unittest.TestCase):
    eventual = True

    def make_store(self):
        from pymongo import MongoClient
        self.client = MongoClient(ENV['MONGODB_URI'])
        collection = self.client[ENV['MONGODB_DB']][ENV['MONGODB_COLLECTION']]
        store = MongoDBAtlasVectorStore(collection=collection, index_name=ENV.get('MONGODB_INDEX') or 'vector_index',
                                        embedder=embedder)
        if ENV.get('MONGODB_CREATE_INDEX'):
            # e.g. a local mongodb/mongodb-atlas-local container: create the index and wait until it is queryable
            store.create_index(dimension=DIMENSION, filter_fields=['topic', 'lang'])
            eventually(lambda: list(collection.list_search_indexes(store.index_name)),
                       lambda indexes: self.assertTrue(indexes and indexes[0].get('queryable')), 120)
        return store

    def cleanup(self, store):
        store.collection.delete_many({'_id': {'$regex': f'^{RUN}-'}})
        if ENV.get('MONGODB_CREATE_INDEX'):
            store.collection.drop()
        self.client.close()


if __name__ == '__main__':
    unittest.main()
