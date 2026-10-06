"""
Vector stores and chat histories from plain settings, for flows and VibeAgent specs (JSON):

    create_vector_store({'type': 'qdrant', 'url': 'http://localhost:6333', 'collection': 'docs',
                         'embedder': {'provider': 'openai', 'api_key': key}})
    create_chat_history({'type': 'file', 'dir': './conversations'})

A store or history object passes through unchanged.
"""
from intelli.store.vector_store import StoreError

# config 'type' -> class name in intelli.store (aliases included)
STORE_TYPES = {
    'memory': 'MemoryVectorStore',
    'qdrant': 'QdrantVectorStore',
    'chroma': 'ChromaVectorStore',
    'weaviate': 'WeaviateVectorStore',
    'milvus': 'MilvusVectorStore',
    'zilliz': 'MilvusVectorStore',
    'elasticsearch': 'ElasticsearchVectorStore',
    'pinecone': 'PineconeVectorStore',
    'pgvector': 'PgVectorStore',
    'postgres': 'PgVectorStore',
    'mongodb_atlas': 'MongoDBAtlasVectorStore',
    'mongodb': 'MongoDBAtlasVectorStore',
    'firestore': 'FirestoreVectorStore',
    'vertex_rag': 'VertexRAGStore',
    'vertex_vector_search': 'VertexVectorSearchStore',
    'vertex_vector_search_index': 'VertexVectorSearchIndexStore',
}
HISTORY_TYPES = {'memory': 'MemoryChatHistory', 'file': 'FileChatHistory', 'firestore': 'FirestoreChatHistory'}
# stores that embed for themselves and take no embedder
NO_EMBEDDER = {'VertexRAGStore'}


def is_vector_store(value):
    return hasattr(value, 'query') and hasattr(value, 'upsert') and callable(value.query)


def create_vector_store(config, default_embedder=None):
    """
    A vector store from a config dict, or the store itself.

    Args:
        config: a VectorStore, or {'type': <one of STORE_TYPES>, ...the store's arguments...}.
            pgvector takes 'connection_string' (psycopg or psycopg2 connects); mongodb_atlas takes 'uri', 'db' and
            'collection' (a collection name; pymongo connects).
        default_embedder: used when the config has no 'embedder' (and the store needs one).
    """
    if config is None:
        return None
    if is_vector_store(config):
        return config
    if not isinstance(config, dict):
        raise StoreError(f'A vector store config must be a dict with a type, not {type(config).__name__}.')
    settings = dict(config)
    kind = str(settings.pop('type', '')).lower()
    if kind not in STORE_TYPES:
        raise StoreError(f"Unknown vector store type '{kind}'. Use one of: {', '.join(sorted(STORE_TYPES))}.")
    import intelli.store as stores
    class_name = STORE_TYPES[kind]
    if class_name not in NO_EMBEDDER and not settings.get('embedder') and default_embedder:
        settings['embedder'] = default_embedder
    if class_name == 'PgVectorStore' and 'connection' not in settings:
        settings['connection'] = _pg_connect(settings.pop('connection_string', None))
    if class_name == 'MongoDBAtlasVectorStore' and isinstance(settings.get('collection'), str):
        settings['collection'] = _mongo_collection(settings.pop('uri', None), settings.pop('db', None),
                                                   settings['collection'])
    return getattr(stores, class_name)(**settings)


def create_chat_history(config):
    """A chat history from a config dict ({'type': 'memory' | 'file' | 'firestore', ...}), or the history itself."""
    if config is None:
        return None
    if hasattr(config, 'get_messages') and hasattr(config, 'add_messages'):
        return config
    if not isinstance(config, dict):
        raise StoreError(f'A chat history config must be a dict with a type, not {type(config).__name__}.')
    settings = dict(config)
    kind = str(settings.pop('type', 'memory')).lower()
    if kind not in HISTORY_TYPES:
        raise StoreError(f"Unknown chat history type '{kind}'. Use one of: {', '.join(sorted(HISTORY_TYPES))}.")
    import intelli.store as stores
    return getattr(stores, HISTORY_TYPES[kind])(**settings)


def _pg_connect(connection_string):
    if not connection_string:
        raise StoreError("A pgvector config needs 'connection_string' (or pass a connection object).")
    try:
        import psycopg
        return psycopg.connect(connection_string)
    except ImportError:
        pass
    try:
        import psycopg2
        return psycopg2.connect(connection_string)
    except ImportError:
        raise StoreError('pgvector needs a PostgreSQL driver: pip install psycopg (or psycopg2).') from None


def _mongo_collection(uri, db, collection):
    if not uri or not db:
        raise StoreError("A mongodb_atlas config needs 'uri', 'db' and 'collection'.")
    try:
        from pymongo import MongoClient
    except ImportError:
        raise StoreError('mongodb_atlas needs pymongo: pip install pymongo') from None
    return MongoClient(uri)[db][collection]
