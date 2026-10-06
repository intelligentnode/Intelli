---
sidebar_position: 1
---

# Vibe Agents


`VibeAgent` allows you to build and execute multi-modal flows using natural language descriptions. Instead of manually defining tasks and dependencies, you describe your intent, and Vibe Agent handles the orchestration using LLMs.

Your intent is compiled into **execution graph**, each node represents an agent or tool action, and each edge encodes dependencies and data flow.

:::info
VibeAgent is beta supported starting from version **1.4.0** as we work toward AGI where agents generate agents.
:::

### How it works

1.  **Planner**: A high-level LLM (OpenAI, Gemini, or Anthropic) analyzes your description.
2.  **Spec Generation**: It generates a structured `FlowSpec` JSON containing tasks, agents, and routing.
3.  **Flow Building**: `VibeFlow` converts the spec into a real `Flow` object.
4.  **Execution**: You start the flow with your initial input.

### Simple Example

Build a text-based joke generator using a natural language "vibe":

```python
import asyncio
import os
from intelli.flow import VibeAgent

async def main():
    # 1. Setup the planner (requires an API key)
    vf = VibeAgent(
        planner_provider="gemini",
        planner_api_key=os.getenv("GEMINI_API_KEY"),
        planner_model="gemini-2.0-flash"
    )
    
    # 2. Describe the flow you want
    description = "Create a 1-step flow that returns a funny joke about AI agents."
    
    # 3. Build the flow
    flow = await vf.build(description)
    
    # 4. Execute
    results = await flow.start(initial_input="Tell me a joke")
    
    for name, data in results.items():
        print(f"Result: {data['output']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Modal Vibe

`VibeAgent` can orchestrate different types of agents (text, image, audio) in a single request:

```python
description = (
    "1. Generate a speech audio for 'Intelli is awesome' using tts-1. "
    "2. Transcribe that audio back to text using whisper-1."
)

flow = await vf.build(description)
results = await flow.start()
```

### Assistant steps: documents, history and memory

Every language step the planner writes is an assistant step (`"agent_type": "assistant"`). It writes, classifies,
summarizes and translates like any model call, and it can also:

- answer from your documents through a vector store, citing them as [1], [2];
- keep one conversation going over several runs (chat history);
- remember a person's preferences across conversations (long-term memory);
- call your Python functions (tools);
- ground answers on Google Search (Gemini).

The planner adds these only when the request needs them:

```python
vf = VibeAgent(planner_provider="openai", planner_api_key=os.getenv("OPENAI_API_KEY"))
flow = await vf.build(
    "A support chat for Acme customers. It answers from our handbook in ./handbook.md, shows the sources, "
    "and remembers the conversation so follow-up questions work.", save_dir="./support_bundle")

await flow.start(initial_input="How long do I have to return a laptop?")
await flow.start(initial_input="And other items?")   # the same conversation continues
```

The step it planned:

```json
{"name": "answer_customer",
 "desc": "Answer the Acme customer's support question using the handbook, and cite the sources.",
 "agent": {"agent_type": "assistant", "provider": "openai", "mission": "You are Acme support. ...",
           "model_params": {"key": "${ENV:OPENAI_API_KEY}", "model": "gpt-5.5",
                            "show_sources": true, "conversation_id": "default"},
           "options": {"knowledge": {"type": "memory", "path": "./knowledge.json"},
                       "files": ["./handbook.md"],
                       "history": {"type": "file", "dir": "./conversations"}}}}
```

- `options.knowledge` / `options.memory`: a store config. `{"type": "memory", "path": ...}` is a local file; the
  databases are `qdrant`, `chroma`, `weaviate`, `milvus`, `elasticsearch`, `pinecone`, `pgvector`,
  `mongodb_atlas`, `firestore`, `vertex_rag` and `vertex_vector_search`, with the arguments of their store class
  (see the Assistant guide). Stores embed with the step's own provider (OpenAI, Gemini, Mistral, Bedrock), or
  with an `"embedder"` in the config.
- `options.files` / `options.documents`: what to load into the knowledge store. They are embedded the first time
  the step runs, and skipped when the store already holds records.
- `options.history`: `{"type": "memory" | "file" | "firestore", ...}`, with `model_params.conversation_id`. To
  change the conversation per user, set it before a run:
  `task = flow.tasks["answer_customer"]; task.model_params = {**task.model_params, "conversation_id": chat_id}`.
- `model_params.user_id` scopes long-term memory to one person.
- The step's input (the run's input, or the previous steps' output) is the message it answers and searches
  with; its task description is added to the mission as the instruction.

### Register tools, stores and guards

A spec is JSON, so it names your Python objects, and you register them when you create the agent. The planner sees
each name with the first line of its docstring, and uses only those names:

```python
from intelli.store import QdrantVectorStore

def get_weather(city: str) -> dict:
    """Current weather of a city: forecast and temperature in Celsius."""
    ...

def read_only(action):
    """Blocks actions that would place an order, pay, submit, approve or delete."""
    text = f'{action.get("text", "")} {action.get("target_text", "")}'.lower()
    return not any(word in text for word in ("order", "pay", "submit"))

vf = VibeAgent(
    planner_provider="openai", planner_api_key=os.getenv("OPENAI_API_KEY"),
    tools={"get_weather": get_weather},                          # assistant options.tools
    stores={"handbook": QdrantVectorStore(url=QDRANT_URL, collection="handbook",
                                          embedder={"provider": "openai", "api_key": OPENAI_KEY})},
    guards={"read_only": read_only},                             # computer model_params.on_action
    processors={"to_upper": str.upper},                          # task post_process
)
```

Register the same names whenever you load a saved spec.

### Computer steps: smoke checks and portals

For sites without an API, the planner writes `computer` steps that operate the site through screenshots. Each
journey is its own step, the journeys run in parallel, and an assistant step writes the report:

```python
vf = VibeAgent(planner_provider="openai", planner_api_key=os.getenv("OPENAI_API_KEY"),
               guards={"read_only": read_only})
flow = await vf.build(
    "Run smoke checks on our staging shop at https://staging.example.com: sign in with the demo account, "
    "search for 'laptop stand' and report how many results appear, add the first one to the cart and report "
    "the total. Never place an order. Then write a short release summary that marks failing journeys.")
```

Each computer step gets `"start_url"` and `"on_action": "read_only"`. They need `pip install "intelli[computer]"`
and `playwright install chromium`. A guard receives each action before it runs: typed text is in `"text"`, and a
click also carries `"target_text"`, the text of the button, link or input it hits, so `read_only` blocks a click on
"Place order" too.

### The planner checks its plan

The plan is checked before the flow is built: the JSON, the graph (no cycles), the providers, the registered
names, store types, embedding models, a `conversation_id` for each history and a `user_id` for each memory, start
URLs and routes. When a rule fails, the planner gets the error and fixes its plan, up to three replies.

`intelli/test/integration/test_vibe_use_cases_live.py` plans and runs the use cases from the docs (support triage,
release brief, blog post to four channels, content platform, travel assistant, release checks, portal operations)
and assistant cases with documents, history, memory and tools.

### Key Features

- **Environment Variables**: Use `${ENV:VARIABLE_NAME}` in your prompts, and VibeAgent will automatically resolve them from your `.env` file.
- **Save & Load**: Use `save_bundle(save_dir)` to export the generated flow spec and a graph image visualization for future use. Secrets are redacted, and registered names stay names.
- **Edit Mode**: Call `vf.edit(spec_path, "add a translation step")` to modify an existing flow using natural language.

### Auto-Saving Outputs

VibeAgent can automatically save generated images and audio to a specific directory:

```python
flow = await vf.build("Generate an image of a cyber cat and save it to ./outputs")
flow.auto_save_outputs = True
flow.output_dir = "./outputs"

await flow.start()
```

### Preferred Models

You can specify preferred model details as strings when initializing `VibeAgent`. This guides the planner to use specific versions of AI models for text, image, speech, or recognition tasks.

```python
vf = VibeAgent(
    planner_provider="gemini",
    planner_api_key=os.getenv("GEMINI_API_KEY"),
    # Specify specific model versions as descriptive strings
    text_model="openai gpt-5.2-mini",
    image_model="gemini gemini-3-pro-image-preview",
    speech_model="openai tts-1",
    recognition_model="openai whisper-1"
)
```

Supported preference parameters:
- `text_model`
- `image_model`
- `speech_model`
- `recognition_model`

