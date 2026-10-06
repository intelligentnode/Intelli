# Assistant, vector stores and chat history

Use these for a Gemini- or ChatGPT-style chat app: saved conversations, answers grounded on the user's documents
with numbered sources, long-term memory, attachments, tools and streaming. In a Flow, an `assistant` step has the
same features (see "In a flow" below).

## Assistant

```python
import os
from intelli.function.assistant import Assistant
from intelli.store import FileChatHistory, MemoryVectorStore

key = os.environ["OPENAI_API_KEY"]
embedder = {"provider": "openai", "api_key": key}
assistant = Assistant(
    provider="openai", api_key=key, model="gpt-5-mini",
    system_message="You are Acme support. Answer in two sentences.",
    history=FileChatHistory(dir="./conversations"),            # default MemoryChatHistory: lost on restart
    knowledge=MemoryVectorStore(embedder=embedder, path="./knowledge.json"),
    memory=MemoryVectorStore(embedder=embedder, path="./memory.json"),   # optional long-term memory
)
assistant.add_documents([{"id": "handbook", "text": handbook_text, "metadata": {"title": "Handbook"}}])
assistant.add_files(["faq.md", "prices.csv"])                  # text files only

reply = assistant.chat("How do refunds work?", user_id="u1")
print(reply["text"])                                           # cites sources as [1], [2]
for event in assistant.stream("And shipping?", conversation_id=reply["conversation_id"], user_id="u1"):
    if event["type"] == "text":
        print(event["text"], end="")
```

- Providers: `openai`, `anthropic`, `gemini`, `vertex` (Gemini on Vertex AI), `aws` (Bedrock), `mistral`, `nvidia`,
  `vllm`, `ollama` (vllm on http://localhost:11434), `llamacpp`, `keras`. `options` takes the provider settings
  (Vertex `project_id` / `location`, AWS `region`, vLLM `baseUrl`).
- `chat()` returns `conversation_id`, `message_id`, `text`, `references` (the retrieved chunks; `cited` is True
  for the ones the answer cites as [n]), `citations` (Google Search web sources), `memories`, `usage`
  (`input_tokens`, `output_tokens`, `total_tokens`), `model` and `tool_steps`.
- A new conversation starts when `conversation_id` is omitted. A `user_id` owns a conversation: another user_id gets a
  PermissionError. Memories are recalled per user_id.
- `add_documents` splits each document into chunks with ids `<id>#0`, `<id>#1`, ...; adding the same id again
  replaces those chunks. Embed once: `MemoryVectorStore` has `count()` and `clear(filter)` for that check.
- Settings: `top_k=4` chunks per turn, `memory_top_k=3`, `min_score` (drop weak matches), `max_history=20`
  messages, `max_tokens`, `temperature`, `auto_title=True` (one extra call names a new conversation).
- Attachments: `chat(..., attachments=["photo.png", {"data": b"...", "mime_type": "application/pdf"},
  "gs://bucket/video.mp4"])`. Gemini takes images, PDFs, audio and video; Anthropic images and PDFs; AWS images,
  documents and video; OpenAI and the others images only.
- Google Search grounding: `google_search=True` on the Assistant or per `chat()`, gemini and vertex only;
  the web sources come back in `citations`.
- Tools: `tools=[get_weather]` (a Python function: name, docstring and type hints become the schema) or
  `{"name", "description", "parameters", "handler"}` dicts. The Assistant runs the tool loop itself (up to
  `max_tool_steps=5` rounds) on openai, anthropic, gemini, aws, mistral, nvidia and vllm; `tool_steps` lists each call.
- Conversations: `list_conversations(user_id=)`, `get_messages(id)`, `rename_conversation(id, title)`,
  `regenerate(id)` (replaces the last answer), `delete_conversation(id)` (also deletes its memories),
  `generate_title(id)`.
- Streaming with tools, or on models without streaming (GPT-5 on the Responses API), sends the answer as one chunk.

## In a flow

```python
from intelli.flow import Agent, Task, TextTaskInput, Flow

answer = Agent("assistant", "openai", "You are Acme support. Answer in two sentences.",
               {"key": key, "model": "gpt-5-mini", "show_sources": True, "conversation_id": "default"},
               {"knowledge": {"type": "memory", "path": "./knowledge.json"},   # or a VectorStore object
                "files": ["./handbook.md"],                                    # embedded on the first run
                "history": {"type": "file", "dir": "./conversations"}})        # or a ChatHistory object
flow = Flow(tasks={"answer": Task(TextTaskInput("Answer the customer's question."), answer)}, map_paths={})
out = await flow.start(initial_input="How long do I have to return a laptop?")
out = await flow.start(initial_input="And other items?")                      # the conversation continues
```

- The step's input is the message (searched, remembered, kept in the history); the task text joins the mission.
- `options`: `knowledge`, `memory`, `history`, `documents`, `files`, `tools`, `embedder` and the provider
  settings. Store configs: `{"type": "memory" | "qdrant" | "chroma" | "weaviate" | "milvus" | "elasticsearch" |
  "pinecone" | "pgvector" | "mongodb_atlas" | "firestore" | "vertex_rag" | "vertex_vector_search", ...the
  class arguments}`; `pgvector` takes `connection_string`, `mongodb_atlas` takes `uri`, `db` and `collection`.
  Histories: `{"type": "memory" | "file" | "firestore", ...}`.
- `model_params`: `key`, `model`, `max_tokens`, `temperature`, `top_k`, `min_score`, `show_sources`,
  `google_search`, and per run `conversation_id`, `user_id`, `filter`, `attachments`. To switch conversations
  between runs: `task.model_params = {**task.model_params, "conversation_id": chat_id}`.
- Config stores embed with the step's provider (openai, gemini, vertex, mistral, nvidia, aws, ollama); on
  anthropic, vllm or llamacpp add `"embedder": {"provider": "openai", "api_key": key}` to the config.
- Without `conversation_id` each run is a new conversation that is not kept (memories are kept).

## Vector stores

Every store shares one interface:

```python
store.add_documents([{"id": "doc-1", "text": "...", "metadata": {"source": "a.md"}}])   # embeds with the embedder
store.upsert([{"id": "doc-1", "vector": [...], "text": "...", "metadata": {...}}])       # your own vectors
hits = store.search("question", 5, {"source": "a.md"})                                 # [{'id', 'score', 'text', 'metadata'}]
same = store.query(text="question", top_k=5, filter={"lang": ["en", "de"]})            # or vector=[...]
store.delete(["doc-1"])
```

`score` is a similarity (higher is closer). `filter` is metadata equality (a list means one of, several keys must
all match); pass the store's own syntax as `native_filter` for anything else. Errors raise `StoreError`.

| Store | Class | Needs | Notes |
| --- | --- | --- | --- |
| In process, optional JSON file | `MemoryVectorStore(path=)` | nothing | exact cosine; a few thousand chunks, tests, local apps |
| Qdrant | `QdrantVectorStore(url=, api_key=, collection=)` | Qdrant server or Cloud | creates the collection on first upsert; ids map to UUID v5 |
| Chroma | `ChromaVectorStore(url=, collection=)` | `chroma run` or Chroma Cloud (`api_key`) | cosine space for a new collection |
| Weaviate | `WeaviateVectorStore(url=, api_key=, class_name=)` | Weaviate 1.2x+ | creates the class; lowercase scalar metadata keys are filterable |
| Milvus / Zilliz | `MilvusVectorStore(url=, token=, collection=)` | Milvus 2.5+ REST | `consistency_level="Strong"` reads your own writes at once |
| Elasticsearch | `ElasticsearchVectorStore(url=, api_key= or username=/password=, index=)` | Elasticsearch 8+ | creates a `dense_vector` mapping |
| Postgres + pgvector (AlloyDB, Cloud SQL, Supabase, Neon) | `PgVectorStore(connection=psycopg.connect(url))` | `pip install psycopg` (or psycopg2), the pgvector extension | creates the table and an HNSW index (up to 2000 dimensions) |
| MongoDB Atlas | `MongoDBAtlasVectorStore(collection=MongoClient(uri)[db][name])` | `pip install pymongo`, an Atlas vector index | `store.create_index(dimension=, filter_fields=[...])`; filtered keys must be filter fields |
| Pinecone | `PineconeVectorStore(api_key=, index_host=)` | an index with the embedder's dimension | metadata cannot hold nulls or nested objects |
| Google Cloud Firestore | `FirestoreVectorStore(project_id=, collection=)` | OAuth, a vector index | `store.index_command(768)` prints the gcloud command |
| Vertex AI RAG Engine | `VertexRAGStore(project_id=, location=, corpus=)` | OAuth, a supported region | Google parses and embeds files (`upload_file`, `import_files`); query by text; `store.tool()` grounds Gemini |
| Vertex AI Vector Search 2.0 | `VertexVectorSearchStore(project_id=, collection=)` | OAuth | `create_collection(dimensions=)` once |
| Vertex AI Vector Search 1.0 | `VertexVectorSearchIndexStore(index=, index_endpoint=, deployed_index_id=, public_endpoint_domain=)` | OAuth, a deployed index | `restrict_keys` turn metadata into filters |

Local servers with no API key, for development:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 8000:8000 chromadb/chroma
docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true cr.weaviate.io/semitechnologies/weaviate
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg17
docker run -d -p 27017:27017 mongodb/mongodb-atlas-local:8.0
```

Google Cloud stores use OAuth, not API keys: `gcloud auth application-default login` with `pip install google-auth`,
or pass `access_token=` (a string or a function).

## The embedder

```python
{"provider": "openai", "api_key": key}                                  # text-embedding-3-small
{"provider": "gemini", "api_key": key}                                  # gemini-embedding-001 on the Developer API
{"provider": "vertex", "api_key": vertex_key, "dimensions": 768}        # Vertex AI
{"provider": "cohere", "api_key": key}                                  # documents and queries embedded differently
{"provider": "aws", "options": {"region": "us-east-1"}}                 # Amazon Titan on Bedrock
{"provider": "ollama", "model": "nomic-embed-text"}                     # local, http://localhost:11434/v1
lambda texts: my_vectors(texts)                                         # your own function
```

Use the same embedder (provider, model and `dimensions`) for the life of a store.

## Chat history

| Class | Where |
| --- | --- |
| `MemoryChatHistory()` | process memory; the default; lost on restart |
| `FileChatHistory(dir=)` | one JSON file per conversation |
| `FirestoreChatHistory(project_id=, collection=)` | `conversations/{id}` plus a `messages` subcollection; no composite index needed |

Methods: `get_messages(id, limit=)`, `add_messages(id, messages)`, `get_conversation(id)`,
`save_conversation({"id", "title", "user_id"})`, `list_conversations(user_id=)`, `delete_conversation(id)`,
`delete_last_messages(id, count)`. Extend `ChatHistory` with those seven methods for another database. Files and
Firestore documents use the same field names as IntelliNode, so a Node app and a Python app can share them.
