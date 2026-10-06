# Assistant and Vector Stores Guide

Read this before building a chat app, an assistant with memory, or RAG over documents with Intelli. It is the Python
port of IntelliNode's Assistant and vector stores (same class names, Python method names).

## 1. What to use

| You want | Use |
| --- | --- |
| A Gemini- or ChatGPT-style chat: saved conversations, answers from your documents with sources, memory, attachments, tools, streaming | `Assistant` (`intelli.function.assistant`) |
| Store and search vectors | a vector store from `intelli.store`, with an `embedder` |
| Keep conversations | `MemoryChatHistory`, `FileChatHistory` or `FirestoreChatHistory` |
| Split long text into chunks | `TextSplitter` (`intelli.utils.text_splitter`) |
| A fixed pipeline of steps | a `Flow` (see the flow docs), with `assistant` steps (section 4) |

## 2. Quick start

```python
import os
from intelli.function.assistant import Assistant
from intelli.store import FileChatHistory, MemoryVectorStore

key = os.environ["OPENAI_API_KEY"]
embedder = {"provider": "openai", "api_key": key}
assistant = Assistant(provider="openai", api_key=key, model="gpt-5-mini",
                      system_message="You are Acme support.",
                      history=FileChatHistory(dir="./conversations"),
                      knowledge=MemoryVectorStore(embedder=embedder, path="./knowledge.json"),
                      memory=MemoryVectorStore(embedder=embedder, path="./memory.json"))

assistant.add_documents([{"id": "handbook", "text": handbook_text, "metadata": {"title": "Handbook"}}])
reply = assistant.chat("How do refunds work?", user_id="u1")
print(reply["text"], reply["references"], reply["usage"])

for event in assistant.stream("And shipping?", conversation_id=reply["conversation_id"], user_id="u1"):
    if event["type"] == "text":
        print(event["text"], end="", flush=True)
```

## 3. The Assistant

Constructor: `Assistant(provider='openai', api_key=None, model=None, options=None, system_message=..., history=None,
knowledge=None, memory=None, max_history=20, top_k=4, memory_top_k=3, min_score=None, google_search=False,
tools=None, max_tool_steps=5, max_tokens=None, temperature=None, input_options=None, auto_title=False)`

- `provider`: openai, anthropic, gemini, vertex (Gemini on Vertex AI), aws (Bedrock), mistral, nvidia, vllm, ollama
  (vllm on http://localhost:11434), llamacpp, keras. `options` are the provider settings (Vertex `project_id`,
  `location`; AWS `region`, IAM keys; vLLM `baseUrl`; `timeout`).
- Each turn: the recent messages (`max_history`), the `top_k` closest document chunks (numbered sources in the
  system text, cited as [1], [2]) and the `memory_top_k` closest memories of this user are sent with the message.
  After the answer both messages are saved, and the exchange is added to the memory store.

| Method | Returns |
| --- | --- |
| `chat(message, conversation_id=None, user_id=None, attachments=None, filter=None, system_message=None, google_search=None)` | `{'conversation_id', 'message_id', 'text', 'references', 'citations', 'memories', 'usage', 'model', 'tool_steps'}` |
| `stream(...)` (same arguments) | events `{'type': 'start'}`, `{'type': 'text', 'text'}`, `{'type': 'done', ...chat fields}` |
| `add_documents(documents, chunk_size=1200, chunk_overlap=150)` | chunk ids `<id>#0`, `<id>#1`, ... |
| `add_files(paths)` | chunk ids (text files: txt, md, csv, json, html, code) |
| `list_conversations(user_id=None)`, `get_messages(id)`, `rename_conversation(id, title)`, `generate_title(id)` | |
| `regenerate(id)` | a new answer to the last message (the old one and its memory are removed) |
| `delete_conversation(id)` | removes the conversation and its memories |

- `references`: the retrieved chunks `{'index', 'id', 'text', 'score', 'metadata', 'cited'}`; metadata `title`,
  `source` and `url` label the source in the prompt.
- `user_id` owns a conversation (another user gets a PermissionError) and scopes memory recall.
- Attachments: paths, URLs (`https://`, `gs://`, `s3://`), data URLs, `{'data': bytes or base64, 'mime_type'}` or
  `{'uri', 'mime_type'}`. Gemini takes images, PDFs, audio and video; Anthropic images and PDFs; AWS images,
  documents and video; OpenAI, Mistral, NVIDIA and vLLM images.
- `google_search=True` (gemini, vertex) grounds answers on Google Search; web sources come back in `citations`.
- Tools: Python functions (the name, the first docstring paragraph and the type hints become the schema), dicts
  `{'name', 'description', 'parameters', 'handler'}` or OpenAI-style `{'type': 'function', 'function': {...},
  'handler'}`. The Assistant runs the tool loop on openai (chat and Responses API), anthropic, gemini, aws,
  mistral, nvidia and vllm. A failing tool sends its error back to the model; `tool_steps` lists every call.
- Streaming with tools, or on a model without streaming (GPT-5 on the Responses API), sends the answer in one chunk.

## 4. Assistant steps in a Flow

`Agent("assistant", provider, mission, model_params, options)` is a flow step with everything above. Use it for
every language step: without stores it is a plain model call; add a store, a history or tools when a step needs
them.

```python
from intelli.flow import Agent, Task, TextTaskInput, Flow

key = os.environ["OPENAI_API_KEY"]
answer = Agent("assistant", "openai", "You are Acme support. Answer in two sentences.",
               {"key": key, "model": "gpt-5-mini", "show_sources": True},
               {"knowledge": {"type": "memory", "path": "./knowledge.json"},       # or a VectorStore object
                "files": ["./handbook.md"]})                                       # loaded on the first run
reply = Agent("assistant", "openai", "You write short, polite replies.", {"key": key, "model": "gpt-5-mini"})

flow = Flow(tasks={"answer": Task(TextTaskInput("Answer the customer's question."), answer),
                   "reply": Task(TextTaskInput("Turn the answer into a reply email."), reply)},
            map_paths={"answer": ["reply"]})
output = await flow.start(initial_input="How long do I have to return a laptop?")
```

- The step's input (the run's `initial_input`, or the previous steps' output) is the message: it is what the
  knowledge store and memory are searched with, and what the history keeps. The task text is added to the
  mission as the instruction.
- `options`: `knowledge` and `memory` (a VectorStore, or a config such as `{"type": "qdrant", "url": ...,
  "collection": ...}`), `history` (a ChatHistory, or `{"type": "memory" | "file" | "firestore", ...}`),
  `documents` and `files` (added to the knowledge store the first time the step runs, unless it already holds
  records), `tools` (functions), `embedder`, and the provider settings.
- `model_params`: `key`, `model`, `temperature`, `max_tokens`, `top_k`, `memory_top_k`, `min_score`,
  `max_history`, `max_tool_steps`, `google_search`, `show_sources` (append the cited sources to the answer), and
  per run `conversation_id`, `user_id`, `filter`, `attachments`. Other keys go to the model request.
- Config stores embed with the step's own provider (openai, gemini, vertex, mistral, nvidia, aws, ollama). On
  anthropic, vllm or llamacpp add `"embedder"` to the config.
- Without `conversation_id` every run is a new conversation, removed from the default in-memory history after the
  answer. Memories are kept.
- The step returns the answer text (with `show_sources`, followed by the cited sources).
- Vibe Agents plan these steps from a request; see the Vibe Agents guide.

## 5. Vector stores

Shared interface: `add_documents(documents)`, `upsert(records)`, `search(text, top_k=5, filter=None)`,
`query(vector=None, text=None, top_k=5, filter=None, native_filter=None)`, `delete(ids)`. Results are
`[{'id', 'score', 'text', 'metadata'}]`, highest score (similarity) first. `filter` is metadata equality (a list
means one of); `native_filter` is the store's own syntax. Errors raise `StoreError` (`status_code`, `details`).

| Store | Class and main arguments | Tested against |
| --- | --- | --- |
| In process | `MemoryVectorStore(embedder=, path=None)` | unit tests |
| Qdrant | `QdrantVectorStore(url=, api_key=, collection=, distance='Cosine')` | local server (Docker) |
| Chroma | `ChromaVectorStore(url=, collection=, api_key=, tenant=, database=)` | local server (Docker) |
| Weaviate | `WeaviateVectorStore(url=, api_key=, class_name=)` | local server (Docker) |
| Milvus / Zilliz | `MilvusVectorStore(url=, token=, collection=, consistency_level=)` | local server (Docker) |
| Elasticsearch | `ElasticsearchVectorStore(url=, api_key= or username=/password=, index=)` | local server (Docker) |
| Postgres + pgvector | `PgVectorStore(connection=, table='intellinode_vectors', dimension=)` | local server (Docker) |
| MongoDB Atlas | `MongoDBAtlasVectorStore(collection=, index_name='vector_index')`, `create_index(dimension=, filter_fields=)` | Atlas Local (Docker) |
| Pinecone | `PineconeVectorStore(api_key=, index_host=, namespace=)` | unit tests |
| Firestore | `FirestoreVectorStore(project_id=, collection=)`, `index_command(dimension)` | unit tests |
| Vertex AI RAG Engine | `VertexRAGStore(project_id=, location=, corpus=)`, `create_corpus`, `upload_file`, `import_files`, `tool()` | unit tests |
| Vertex AI Vector Search 2.0 | `VertexVectorSearchStore(project_id=, collection=)`, `create_collection(dimensions=)` | unit tests |
| Vertex AI Vector Search 1.0 | `VertexVectorSearchIndexStore(index=, index_endpoint=, deployed_index_id=, public_endpoint_domain=)` | unit tests |

- The REST stores take `timeout`, `retries` (on 408, 425, 429 and 5xx) and `session` (your own `requests.Session`).
- Qdrant and Weaviate store each id as its UUID v5 and keep the original id; Vertex Vector Search maps ids to valid
  data object ids. These mappings, Firestore document ids and the text chunks are the same as IntelliNode's, so the
  two SDKs can work on the same data.
- Google Cloud stores and `FirestoreChatHistory` use OAuth: `access_token=` (a string or a function),
  `credentials=` (a google.auth object), or Application Default Credentials (`pip install google-auth`, then
  `gcloud auth application-default login`).

## 6. Embedder

`Embedder(provider='openai', api_key=None, model=None, dimensions=None, batch_size=None, options=None)`, or pass the
settings dict as `embedder=` to any store. Providers: openai (text-embedding-3-small), gemini / google, vertex
(gemini-embedding-001; documents and queries use RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY), cohere (embed-v4.0,
search_document / search_query), mistral, nvidia, vllm, aws (Amazon Titan) and ollama (OpenAI-compatible,
`options={'base_url': ...}`). A function `texts -> vectors` (optionally with `kind=`) also works.

## 7. Chat history

`MemoryChatHistory()`, `FileChatHistory(dir='.intelli/conversations')`,
`FirestoreChatHistory(project_id=, collection='conversations')`. Methods: `get_messages(id, limit=None)`,
`add_messages(id, messages)`, `get_conversation(id)`, `save_conversation({'id', 'title', 'user_id', 'metadata'})`,
`list_conversations(user_id=None, limit=50)`, `delete_conversation(id)`, `delete_last_messages(id, count=1)`.
Messages are `{'id', 'role', 'content', 'created_at', 'metadata'}`. Files and Firestore store IntelliNode's field
names (`userId`, `createdAt`), so a Node app and a Python app can share conversations.

## 8. Tests

- Offline: `intelli/test/unit/test_assistant.py`, `intelli/test/unit/test_vector_stores.py`.
- Live stores: `intelli/test/integration/test_vector_stores_live.py` (set `QDRANT_URL`, `CHROMA_URL`,
  `WEAVIATE_URL`, `MILVUS_URL`, `ELASTICSEARCH_URL`, `PG_CONNECTION_STRING`, `MONGODB_*`, `PINECONE_*`; the file
  header lists Docker commands for local servers).
- Live Assistant: `intelli/test/integration/test_assistant_live.py` (provider keys from `.env`).
- Flows: `intelli/test/unit/test_flow_assistant.py` and `intelli/test/unit/test_vibe_assistant.py` (offline),
  `intelli/test/integration/test_flow_assistant_live.py` (live; `QDRANT_URL` adds a Qdrant store) and
  `intelli/test/integration/test_vibe_use_cases_live.py` (Vibe Agents planning the documented use cases).
