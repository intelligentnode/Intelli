"""
Vector stores, chat histories and the embedder used by the Assistant (intelli.function.assistant) and RAG flows.

Every vector store shares one interface: add_documents, upsert, search, query, delete.
"""
from intelli.store.vector_store import VectorStore, StoreError, matches_filter, cosine_similarity, new_id
from intelli.store.embedder import Embedder
from intelli.store.memory_vector_store import MemoryVectorStore
from intelli.store.chat_history import ChatHistory, MemoryChatHistory, FileChatHistory
from intelli.store.firestore_chat_history import FirestoreChatHistory
from intelli.store.firestore_vector_store import FirestoreVectorStore
from intelli.store.vertex_rag_store import VertexRAGStore
from intelli.store.vertex_vector_search_store import VertexVectorSearchStore, VertexVectorSearchIndexStore
from intelli.store.pinecone_vector_store import PineconeVectorStore
from intelli.store.qdrant_vector_store import QdrantVectorStore
from intelli.store.chroma_vector_store import ChromaVectorStore
from intelli.store.weaviate_vector_store import WeaviateVectorStore
from intelli.store.milvus_vector_store import MilvusVectorStore
from intelli.store.elasticsearch_vector_store import ElasticsearchVectorStore
from intelli.store.pg_vector_store import PgVectorStore
from intelli.store.mongodb_atlas_vector_store import MongoDBAtlasVectorStore
from intelli.store.factory import create_vector_store, create_chat_history

__all__ = [
    'VectorStore', 'StoreError', 'matches_filter', 'cosine_similarity', 'new_id', 'Embedder', 'MemoryVectorStore',
    'ChatHistory', 'MemoryChatHistory', 'FileChatHistory', 'FirestoreChatHistory', 'FirestoreVectorStore',
    'VertexRAGStore', 'VertexVectorSearchStore', 'VertexVectorSearchIndexStore', 'PineconeVectorStore',
    'QdrantVectorStore', 'ChromaVectorStore', 'WeaviateVectorStore', 'MilvusVectorStore', 'ElasticsearchVectorStore',
    'PgVectorStore', 'MongoDBAtlasVectorStore', 'create_vector_store', 'create_chat_history',
]
