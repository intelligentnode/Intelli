# API: https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview (REST v1: ragCorpora, ragFiles,
# media.upload, projects.locations.retrieveContexts)
import json
import mimetypes
import os
import re
import time
from urllib.parse import quote

from intelli.config import config
from intelli.store.vector_store import VectorStore, StoreError
from intelli.store.google_cloud import GoogleCloudService


class VertexRAGStore(VectorStore):
    """
    Vertex AI RAG Engine: a managed corpus that parses, chunks, embeds and indexes your files on Google Cloud.
    Use it as a knowledge store (query by text), or ground Gemini directly with store.tool().

        corpus = VertexRAGStore.create_corpus(project_id=project, display_name='handbook')
        store = VertexRAGStore(project_id=project, corpus=corpus['name'])
        store.upload_file('handbook.pdf')                       # or import_files(['gs://bucket/docs/'])
        hits = store.query(text='refund policy', top_k=5)

    RAG Engine needs OAuth credentials (API keys are rejected) and a supported region; us-central1 needs an allowlist
    for new projects, europe-west3 / europe-west4 are generally available.
    """

    def __init__(self, project_id=None, location=None, corpus=None, vector_distance_threshold=None,
                 access_token=None, credentials=None, quota_project_id=None, timeout=300, retries=0, session=None):
        """
        Args:
            corpus: the corpus full name (projects/.../ragCorpora/...) or id.
            vector_distance_threshold: drop chunks farther than this distance.
        """
        super().__init__(None)
        self.service = GoogleCloudService(project_id=project_id, access_token=access_token, credentials=credentials,
                                          quota_project_id=quota_project_id, timeout=timeout, retries=retries,
                                          session=session, label='Vertex AI RAG Engine')
        self.location = location or config['url']['gemini']['vertex']['locations'].get('rag', 'us-central1')
        self.corpus = corpus
        self.vector_distance_threshold = vector_distance_threshold

    def _host(self):
        return f'https://{self.location}-aiplatform.googleapis.com'

    def _parent(self):
        return f'projects/{self.service.project()}/locations/{self.location}'

    def corpus_name(self):
        if not self.corpus:
            raise StoreError('VertexRAGStore needs a corpus (name or id).')
        if str(self.corpus).startswith('projects/'):
            return self.corpus
        return f'{self._parent()}/ragCorpora/{self.corpus}'

    @classmethod
    def create_corpus(cls, display_name, description=None, embedding_model=None, **options):
        """
        Create a corpus and wait for it. embedding_model defaults to text-embedding-005 (the RAG Engine default).
        Returns the corpus ({'name', 'displayName', ...}).
        """
        store = cls(**options)
        parent = store._parent()
        body = {'displayName': display_name}
        if description:
            body['description'] = description
        if embedding_model:
            endpoint = embedding_model if embedding_model.startswith('projects/') else \
                f'{parent}/publishers/google/models/{embedding_model}'
            body['vectorDbConfig'] = {'ragManagedDb': {'knn': {}},
                                      'ragEmbeddingModelConfig': {'vertexPredictionEndpoint': {'endpoint': endpoint}}}
        operation = store.service.request('POST', f'{store._host()}/v1/{parent}/ragCorpora', body)
        return store.service.wait_for_operation(operation, lambda name: f'{store._host()}/v1/{name}')

    def list_corpora(self, page_size=None, page_token=None):
        params = {key: value for key, value in (('page_size', page_size), ('page_token', page_token)) if value}
        return self.service.request('GET', f'{self._host()}/v1/{self._parent()}/ragCorpora', params=params)

    def get_corpus(self):
        return self.service.request('GET', f'{self._host()}/v1/{self.corpus_name()}')

    def delete_corpus(self, force=True):
        """Delete the corpus; force also deletes its files."""
        return self.service.request('DELETE', f"{self._host()}/v1/{self.corpus_name()}{'?force=true' if force else ''}")

    def upload_file(self, source, display_name=None, description=None, mime_type=None, chunk_size=512,
                    chunk_overlap=100):
        """
        Upload one local file (a path, or bytes with display_name and mime_type) into the corpus; RAG Engine parses,
        chunks and embeds it. Returns the RagFile.
        """
        data = source
        name = display_name
        if isinstance(source, str):
            with open(source, 'rb') as file:
                data = file.read()
            name = name or os.path.basename(source)
            mime_type = mime_type or mimetypes.guess_type(source)[0]
        if not name:
            raise StoreError('display_name is required when uploading bytes.')
        rag_file = {'displayName': name}
        if description:
            rag_file['description'] = description
        metadata = {
            'ragFile': rag_file,
            'uploadRagFileConfig': {'ragFileTransformationConfig': {'ragFileChunkingConfig': {
                'fixedLengthChunking': {'chunkSize': chunk_size, 'chunkOverlap': chunk_overlap}}}},
        }
        files = {
            'metadata': (None, json.dumps(metadata), 'application/json'),
            'file': (name, bytes(data), mime_type or 'application/octet-stream'),
        }
        url = f'{self._host()}/upload/v1/{self.corpus_name()}/ragFiles:upload'
        result = self.service.request('POST', url, files=files, headers={'X-Goog-Upload-Protocol': 'multipart'})
        if isinstance(result, dict) and result.get('error'):
            raise StoreError(f"Vertex AI RAG Engine upload error: {result['error']}", details=result['error'])
        return (result or {}).get('ragFile') or result

    def import_files(self, uris, chunk_size=512, chunk_overlap=100, max_embedding_requests_per_min=1000, wait=True):
        """
        Import files from Cloud Storage (gs://bucket/path) or Google Drive (folder / file links) and wait for the
        import to finish. Returns the import result counts.
        """
        items = [uris] if isinstance(uris, str) else list(uris)
        gcs = [uri for uri in items if uri.startswith('gs://')]
        drive = [uri for uri in items if not uri.startswith('gs://')]
        config_body = {}
        if gcs:
            config_body['gcsSource'] = {'uris': gcs}
        if drive:
            resources = []
            for uri in drive:
                match = re.search(r'/folders/([^/?#]+)', uri) or re.search(r'/d/([^/?#]+)', uri)
                resources.append({'resourceType': 'RESOURCE_TYPE_FOLDER' if '/folders/' in uri else 'RESOURCE_TYPE_FILE',
                                  'resourceId': match.group(1) if match else uri})
            config_body['googleDriveSource'] = {'resourceIds': resources}
        config_body['ragFileTransformationConfig'] = {'ragFileChunkingConfig': {
            'fixedLengthChunking': {'chunkSize': chunk_size, 'chunkOverlap': chunk_overlap}}}
        config_body['maxEmbeddingRequestsPerMin'] = max_embedding_requests_per_min
        operation = self.service.request('POST', f'{self._host()}/v1/{self.corpus_name()}/ragFiles:import',
                                         {'importRagFilesConfig': config_body})
        if not wait:
            return operation
        return self.service.wait_for_operation(operation, lambda name: f'{self._host()}/v1/{name}')

    def list_files(self, page_size=None, page_token=None):
        params = {key: value for key, value in (('page_size', page_size), ('page_token', page_token)) if value}
        return self.service.request('GET', f'{self._host()}/v1/{self.corpus_name()}/ragFiles', params=params)

    def add_documents(self, documents, **options):
        """Add text documents: each one is uploaded as a text file named after its id (or metadata['source'])."""
        names = []
        for document in documents or []:
            item = {'text': document} if isinstance(document, str) else document
            base = str((item.get('metadata') or {}).get('source') or item.get('id') or f'document-{int(time.time() * 1000)}')
            display_name = base if re.search(r'\.[a-z0-9]+$', base, re.I) else f'{base}.txt'
            rag_file = self.upload_file(str(item.get('text') or '').encode('utf-8'), display_name=display_name,
                                        mime_type='text/plain', **options)
            names.append((rag_file or {}).get('name') or display_name)
        return names

    def upsert(self, records):
        raise StoreError('VertexRAGStore embeds files itself: use add_documents, upload_file or import_files.')

    def query(self, vector=None, text=None, top_k=5, filter=None, native_filter=None):
        """
        Retrieve chunks for a text query: [{'id', 'score', 'text', 'metadata': {'source', 'title', 'pages'}}].
        RAG Engine returns a cosine distance by default (0 = same); score is 1 - distance.
        """
        if text is None:
            raise StoreError('VertexRAGStore queries by text: query(text=...).')
        condition = native_filter or ({'vectorDistanceThreshold': self.vector_distance_threshold}
                                      if self.vector_distance_threshold is not None else None)
        retrieval = {'topK': top_k or 5}
        if condition:
            retrieval['filter'] = condition
        body = {'vertexRagStore': {'ragResources': [{'ragCorpus': self.corpus_name()}]},
                'query': {'text': str(text), 'ragRetrievalConfig': retrieval}}
        result = self.service.request('POST', f'{self._host()}/v1/{self._parent()}:retrieveContexts', body) or {}
        contexts = (result.get('contexts') or {}).get('contexts') or []
        matches = []
        for index, context in enumerate(contexts):
            chunk = context.get('chunk') or {}
            distance = context.get('score') if isinstance(context.get('score'), (int, float)) else \
                context.get('distance') if isinstance(context.get('distance'), (int, float)) else None
            metadata = {'source': context.get('sourceUri'), 'title': context.get('sourceDisplayName')}
            if chunk.get('pageSpan'):
                metadata['pages'] = chunk['pageSpan']
            matches.append({
                'id': chunk.get('chunkId') or f"{context.get('sourceUri') or 'context'}#{index}",
                'score': None if distance is None else 1 - distance,
                'text': context.get('text') or chunk.get('text') or '',
                'metadata': metadata,
            })
        return matches

    def delete(self, file_names):
        """Delete RAG files by their resource names (from add_documents / list_files)."""
        for name in file_names or []:
            full_name = name if str(name).startswith('projects/') else \
                f"{self.corpus_name()}/ragFiles/{quote(str(name), safe='')}"
            self.service.request('DELETE', f'{self._host()}/v1/{full_name}')

    def tool(self, top_k=5, vector_distance_threshold=None):
        """A Gemini grounding tool for this corpus (Vertex AI generateContent): tools=[store.tool()]."""
        retrieval = {'topK': top_k}
        if vector_distance_threshold is not None:
            retrieval['filter'] = {'vectorDistanceThreshold': vector_distance_threshold}
        return {'retrieval': {'vertexRagStore': {'ragResources': [{'ragCorpus': self.corpus_name()}],
                                                 'ragRetrievalConfig': retrieval}}}
