import inspect

import requests

from intelli.store.vector_store import StoreError

# Providers that embed documents and queries differently, and the value each one expects.
KIND_INPUT_TYPES = {
    'cohere': {'document': 'search_document', 'query': 'search_query'},
    'nvidia': {'document': 'passage', 'query': 'query'},
}
GEMINI_TASK_TYPES = {'document': 'RETRIEVAL_DOCUMENT', 'query': 'RETRIEVAL_QUERY'}
GOOGLE_PROVIDERS = {'gemini', 'google', 'vertex'}
DEFAULT_MODELS = {
    'openai': 'text-embedding-3-small',
    'mistral': 'mistral-embed',
    'cohere': 'embed-v4.0',
    'ollama': 'nomic-embed-text',
}
OLLAMA_BASE_URL = 'http://localhost:11434/v1'


class Embedder:
    """
    Turns texts into vectors with any Intelli embedding provider, for the vector stores and the Assistant.

        Embedder(provider='openai', api_key=key)                                  # text-embedding-3-small
        Embedder(provider='gemini', api_key=key)                                  # Gemini Developer API
        Embedder(provider='vertex', api_key=key, dimensions=768)                  # gemini-embedding-001 on Vertex AI
        Embedder(provider='cohere', api_key=key)
        Embedder(provider='aws', options={'region': 'us-east-1'})                 # Amazon Titan (Bedrock)
        Embedder(provider='ollama', model='nomic-embed-text', options={'base_url': 'http://localhost:11434/v1'})

    Gemini, Cohere and NVIDIA embed documents and search queries differently; embed(texts, kind='query') selects that.
    """

    def __init__(self, provider='openai', api_key=None, model=None, dimensions=None, batch_size=None, options=None,
                 timeout=None):
        """
        Args:
            provider: openai, gemini (or google), vertex, cohere, mistral, nvidia, vllm, aws or ollama.
            model: the embedding model; the provider default when omitted.
            dimensions: output size, for providers that can shorten vectors (OpenAI text-embedding-3, Gemini).
            batch_size: texts per request (default 100 for Google, else 96).
            options: provider settings, as in Chatbot / RemoteEmbedModel: Google {'vertex', 'project_id',
                'location', 'access_token', 'credentials'}, vLLM {'baseUrl'}, AWS {'region', 'access_key_id',
                'secret_access_key', 'profile'}, Ollama / OpenAI-compatible {'base_url', 'headers'}.
        """
        self.provider = str(provider or 'openai').lower()
        self.model = model
        self.dimensions = dimensions
        self.options = dict(options or {})
        timeout = timeout or self.options.get('timeout', 180)
        self.timeout = timeout
        self.google = None
        self.wrapper = None
        if self.provider in GOOGLE_PROVIDERS:
            from intelli.wrappers.googleai_wrapper import GoogleAIWrapper
            google_options = dict(self.options)
            if self.provider == 'vertex':
                google_options.setdefault('vertex', True)
            self.google = GoogleAIWrapper.from_options(api_key, google_options, timeout=timeout)
            self.batch_size = batch_size or 100
            return
        if self.provider == 'cohere':
            from intelli.wrappers.cohereai_wrapper import CohereAIWrapper
            self.wrapper = CohereAIWrapper(api_key, timeout=timeout)
        elif self.provider == 'ollama':
            self.base_url = (self.options.get('base_url') or self.options.get('baseUrl') or OLLAMA_BASE_URL).rstrip('/')
            self.headers = {'Content-Type': 'application/json', **(self.options.get('headers') or {})}
            if api_key:
                self.headers['Authorization'] = f'Bearer {api_key}'
        else:
            from intelli.controller.remote_embed_model import RemoteEmbedModel
            self.wrapper = RemoteEmbedModel(api_key, self.provider, self.options).provider
        # Cohere accepts at most 96 texts per call
        self.batch_size = batch_size or 96

    def embed(self, texts, kind='document'):
        """Embed a list of texts. kind: 'document' (default) or 'query'. Returns a list of vectors in order."""
        items = [str(texts)] if isinstance(texts, str) else [str(text) for text in texts]
        vectors = []
        for start in range(0, len(items), self.batch_size):
            vectors.extend(self._embed_batch(items[start:start + self.batch_size], kind))
        return vectors

    def _embed_batch(self, texts, kind):
        if self.google:
            return self.google.embed_texts(texts, self.model, task_type=GEMINI_TASK_TYPES.get(kind),
                                           output_dimensionality=self.dimensions)
        model = self.model or DEFAULT_MODELS.get(self.provider)
        input_types = KIND_INPUT_TYPES.get(self.provider)
        if self.provider == 'cohere':
            params = {'texts': texts, 'model': model, 'input_type': input_types[kind], 'embedding_types': ['float']}
            return self._vectors(self.wrapper.get_embeddings(params))
        if self.provider == 'ollama':
            body = {'model': model, 'input': texts}
            try:
                response = requests.post(f'{self.base_url}/embeddings', json=body, headers=self.headers,
                                         timeout=self.timeout)
                response.raise_for_status()
            except requests.exceptions.RequestException as error:
                raise StoreError(f'Embedding error: {error}') from None
            return self._vectors(response.json())
        if self.provider == 'aws':
            params = {'texts': texts, **({'model': model} if model else {})}
            if self.dimensions:
                params['dimensions'] = self.dimensions
            return self._vectors(self.wrapper.get_embeddings(params))
        if self.provider == 'vllm':
            return self._vectors(self.wrapper.get_embeddings({'texts': texts, **({'model': model} if model else {})}))
        if not model:
            raise StoreError(f"Embedder for '{self.provider}' needs a model.")
        params = {'input': texts, 'model': model}
        if self.provider == 'openai' and self.dimensions:
            params['dimensions'] = self.dimensions
        if self.provider == 'nvidia':
            params.update({'input_type': input_types[kind], 'encoding_format': 'float', 'truncate': 'NONE'})
        return self._vectors(self.wrapper.get_embeddings(params))

    @staticmethod
    def _vectors(result):
        """Vectors from any embedding response shape, in input order."""
        items = result
        if isinstance(result, dict):
            items = result.get('data')
            if items is None:
                items = result.get('embeddings')
            if isinstance(items, dict):  # Cohere embedding_types: {'float': [...]}
                items = items.get('float') or next(iter(items.values()), [])
        items = items or []
        if items and all(isinstance(item, dict) and isinstance(item.get('index'), int) for item in items):
            items = sorted(items, key=lambda item: item['index'])
        vectors = []
        for item in items:
            if isinstance(item, dict):
                vectors.append(item.get('embedding') or item.get('values'))
            else:
                vectors.append(item)
        return vectors


class _FunctionEmbedder:
    """An embedder from a function (texts) -> vectors or (texts, kind) -> vectors."""

    def __init__(self, function):
        self.function = function
        try:
            parameters = inspect.signature(function).parameters
            self.takes_kind = 'kind' in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        except (TypeError, ValueError):
            self.takes_kind = False

    def embed(self, texts, kind='document'):
        return self.function(texts, kind=kind) if self.takes_kind else self.function(texts)


def to_embedder(value):
    """An Embedder from an Embedder, a function, an object with embed(), or settings {'provider', ...}."""
    if not value:
        return None
    if hasattr(value, 'embed') and callable(value.embed):
        return value
    if callable(value):
        return _FunctionEmbedder(value)
    if isinstance(value, dict) and value.get('provider'):
        return Embedder(**value)
    raise StoreError('embedder must be an Embedder, a function (texts) -> vectors, or '
                     "{'provider', 'api_key', 'model', 'options'}.")
