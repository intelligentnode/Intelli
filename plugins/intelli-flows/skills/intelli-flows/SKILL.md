---
name: intelli-flows
description: Use when the user wants to design, draw or build an AI workflow in Python with Intelli (intelli.flow), especially one that mixes providers (OpenAI, Anthropic, Gemini, AWS Bedrock, Mistral, local models) or agent types (assistants with documents, history and memory, image generation, vision, speech, transcription, embeddings, search, computer use, MCP tools), or asks for a picture of an agent workflow. Also for a Gemini- or ChatGPT-style assistant with saved conversations, long-term memory and RAG over the user's documents (the Assistant class), vector databases (Qdrant, Chroma, Weaviate, Milvus, Elasticsearch, pgvector, MongoDB Atlas, Pinecone, Firestore, Vertex AI), Flow graphs, routing with DynamicConnector, Memory, and Vibe Agents and their FlowSpec JSON. Not for IntelliNode on Node.js.
---

# Intelli flows

Follow the Intelli section of [AGENTS.md](AGENTS.md) in this skill's folder, then these rules. For which
agent types and providers exist and what passes between them, see [references/agents.md](references/agents.md).
For a chat app, an assistant with memory or RAG over documents, use the `Assistant` class and a vector store:
see [references/assistant.md](references/assistant.md).

1. Plan before code. Write a short table: step name, agent type, provider and model, what the step receives,
   what it returns. Step names are printed on the picture, so make them readable. When the user did not fix
   the steps, show the table and the picture and let them confirm before running.
2. One flow can mix providers and agent types: each `Agent` picks its own. Example, Claude writes, OpenAI
   draws and speaks, Gemini checks the image:

   ```python
   import asyncio, os
   from intelli.flow import Agent, Task, TextTaskInput, Flow

   def key(name):  # read keys from the environment; never print or hardcode them
       return os.environ.get(name, "")

   writer = Agent("assistant", "anthropic", "Write a two-sentence product pitch.",
                  {"key": key("ANTHROPIC_API_KEY"), "model": "claude-haiku-4-5", "max_tokens": 300})
   artist = Agent("image", "openai", "Flat illustration for this pitch, no text",
                  {"key": key("OPENAI_API_KEY"), "model": "gpt-image-2", "width": 1024, "height": 1024})
   critic = Agent("vision", "gemini", "Does the image match the pitch? Answer yes or no, then one reason",
                  {"key": key("GEMINI_API_KEY"), "model": "gemini-2.5-flash"})
   voice = Agent("speech", "openai", "",
                 {"key": key("OPENAI_API_KEY"), "model": "gpt-4o-mini-tts", "voice": "alloy", "stream": False})

   flow = Flow(
       tasks={
           "pitch": Task(TextTaskInput("Pitch a reusable water bottle"), writer),
           "illustration": Task(TextTaskInput("Illustrate the pitch"), artist),
           "image check": Task(TextTaskInput("Check the illustration"), critic),
           "voice over": Task(TextTaskInput("Voice over the pitch"), voice),
       },
       map_paths={"pitch": ["illustration", "voice over"], "illustration": ["image check"]},
       auto_save_outputs=True, output_dir="./outputs",   # images, audio and text are written here
   )
   print(flow.generate_graph_img(name="pitch_flow", save_path="."))   # no model is called
   out = asyncio.run(flow.start())
   ```

3. Rules for mixed steps:
   - Image and audio outputs do not turn into text. Between an `image` step and a text step put a `vision`
     step; between `speech` and text put `recognition`.
   - A `vision` step needs `model` in `model_params`. Its question is its mission plus its task text, and
     the image comes from the parent `image` step. For a first step with your own picture use
     `ImageTaskInput("question", img=<base64 string>)`.
   - An `image` step's prompt is its mission, then the parent's text. It returns base64 image data.
   - An OpenAI `speech` step needs `"stream": False`, or it returns a stream and no audio is saved or passed on.
   - A `speech` step speaks its mission, then its input, word for word. Leave the mission empty and have the
     step before it output only the words to say. Before 2.1.3 it also spoke the prompt template's labels, and
     OpenAI rejected text over 4,096 characters; 2.1.3 speaks longer text in pieces.
   - Each provider needs its own key. A missing key fails only that step, so run only when every provider's
     key is set, or offer to move that step to a provider whose key is set.
   - Image, video and audio generation are billed per call. Tell the user before running those steps.
4. For a Vibe Agent you are the planner. Write the FlowSpec JSON yourself, save it in the repo,
   and load it with `planner_fn` or `build_from_spec`. Use this shape:

   {"version": "1",
    "tasks": [{"name": "thread", "desc": "what the task does", "post_process": "flag_new_numbers",
               "agent": {"agent_type": "assistant", "provider": "ollama", "mission": "who the agent is",
                         "model_params": {"model": "qwen2.5:0.5b", "temperature": 0.3, "max_tokens": 300},
                         "options": {}}}],
    "map_paths": {}, "dynamic_connectors": [], "output_memory_map": {}}

   For a local or offline model use provider "ollama" (http://localhost:11434), or "vllm" with options
   {"baseUrl": "${ENV:VLLM_BASE_URL}"}. When a step answers from documents, keeps a conversation or remembers
   a user, add its `options` (`knowledge` and `files`, `history`, `memory`) as in AGENTS.md.
5. Every task needs `agent.agent_type`. Without it validation is skipped and the provider becomes openai.
6. Name each `post_process`, tool, store and guard in the spec, and register the same names with
   `VibeAgent(processors=..., tools=..., stores=..., guards=...)` every time the spec is loaded.
7. Before `flow.start()`, loop over `flow.tasks` and raise if any `agent.provider` is not the one
   the user asked for. After every run, raise if `flow.errors` is not empty.
8. Choose the form. Write a code Flow when steps need exact prompts, Python checks, or image, vision or
   audio steps you want to control. Write a Vibe Agent spec when the user wants a saved plan to rerun or
   change: language steps, with documents, history, memory, tools or routing.
9. A small local model is fine for a first run. Follow the small-model rules in AGENTS.md and tell the user
   what it got wrong, so they can decide which steps need a stronger model.
10. If you cannot install packages where you run, write the code for the user to run and draw the same
    graph as a Mermaid `flowchart` with the same step names, so they still get a picture.
11. Choose Flow or Assistant. A Flow runs a fixed pipeline of steps once per input; its `assistant` steps
    can answer from documents, keep a conversation and remember users. The `Assistant` class is the same
    engine for a chat app without a pipeline: history, documents with numbered sources, memory, tools,
    attachments and streaming.
12. For anything else, fetch https://www.intellinode.ai/llms.txt and open the page it lists.
