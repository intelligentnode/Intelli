## Using Intelli (Python) to build agent flows

When the user asks for an AI tool, follow this sequence. The user may not read code.
1. Write the tool as an Intelli flow of small steps.
2. Save the flow picture before the first run: `flow.generate_graph_img(name="<tool>_graph", save_path=".")`. Drawing calls no model and needs no key.
3. Run it. If `flow.errors` is not empty, fix the cause and run again.
4. Read the output. An empty `flow.errors` only means nothing crashed. Check that `sorted(out)` lists exactly the steps you expect and that the text is right for the input: a small model may return the input unchanged or give every item the same label. Open saved images and check audio sizes too. Fix and run again.
5. Report in plain language: what each step does, which model each step uses (say "a local model on your computer" for vllm), what you checked and what is still weak, and where the picture and the output files are.

Install with `pip install -U "intelli[visual]"`, in the project's virtual environment if it has one. The visual extra adds matplotlib for flow pictures. Use version 2.2.0 or above (`pip show intelli`). Import from `intelli.flow`:
`from intelli.flow import Agent, Task, TextTaskInput, Flow, SequenceFlow, DynamicConnector, Memory, CustomAgent, VibeAgent`

Agents and tasks
- Which provider: the one the user names. If they name none, use OpenAI when `OPENAI_API_KEY` is set, else Anthropic when `ANTHROPIC_API_KEY` is set, else a local server. Say which one you used.
- Language steps are `assistant` agents. Cloud, the easy start: `Agent("assistant", "openai", "mission", {"key": os.environ["OPENAI_API_KEY"], "model": "gpt-4.1-mini", "max_tokens": 300})` or `Agent("assistant", "anthropic", "mission", {"key": os.environ["ANTHROPIC_API_KEY"], "model": "claude-haiku-4-5", "max_tokens": 300})`. Read keys from the environment. Never hardcode or print them, and stop with a clear message when the key is missing.
- Local or offline, no key: Ollama `Agent("assistant", "ollama", "mission", {"model": "<ollama tag>", "temperature": 0.2, "max_tokens": 300})` (http://localhost:11434), or a vLLM server `Agent("assistant", "vllm", "mission", {"model": "<model>"}, options={"baseUrl": "http://host:8000"})`.
- An assistant step can also answer from documents, keep a conversation, remember a user and call Python functions, through its `options`: see references/assistant.md. Add them only when the step needs them.
- Plain Python step, no model: `class Step(CustomAgent)` with `def execute(self, agent_input, new_params=None): return fn(agent_input.desc)`, used as `Task(TextTaskInput("x"), Step("text"))`.
- `Task(TextTaskInput("instruction"), agent, pre_process=fn, post_process=fn, memory_key="name", exclude=False, template=obj)`
- For each task: `pre_process(input)`, then the prompt, then the model, then `post_process(output) -> output`. Child tasks and `out[name]["output"]` get the processed value. An assistant step sends the mission plus the task text as the system prompt and the input as the user message; other steps build the prompt with the task's template (see Pitfalls).
- The default temperature is 1. Use `temperature: 0` for labels and extraction; 0.1 was still unstable on a small model.
- Small local models: sort, count and label in plain Python where you can, give each model step one narrow job, ask for plain descriptive words and map them to labels in `post_process`, and check each answer against its input. As writers they echo the input, ignore lengths and invent numbers, so tell the user.

Flow (async graph)
- `flow = Flow(tasks={"a": ta, "b": tb, "c": tc}, map_paths={"a": ["c"], "b": ["c"]}, memory=Memory())`
- `out = await flow.start(initial_input=text, max_workers=4)`, then read `out["c"]["output"]`. Every task with no parent gets `initial_input`, and tasks that are ready at the same time run in parallel.
- A task with several parents gets their outputs joined. If its mission contains "synthesize", "integrate" or "predict", each part is wrapped as `========== NAME OUTPUT ==========` ... `========== END OF NAME ==========`.
- Memory: `memory.store("key", value)` before the run and `memory.retrieve("key")` after it. `Flow(..., output_memory_map={"task": "key"})` copies a task's output into memory, and `memory_key="key"` on a task replaces its input with the stored value, which is how a later step can read the original input.
- For a batch, build a new Flow for each input.
- Routing: `dynamic_connectors={"a": DynamicConnector(decision_fn=lambda out, kind: "x", destinations={"x": "task_x", "y": "task_y"})}`. Only the chosen task runs (4 destinations at most), and it receives the output of task "a". The destination keys are printed on the picture, so use readable words.
- To route after parallel steps, join them into one task and put the connector on that task. Before 2.1.0 a routed task with another parent in `map_paths` ran on every route.
- Task failures do not raise. Check `flow.errors` (a dict) after every run.
- `SequenceFlow([t1, t2]).start()` is synchronous and returns `{"task1": ..., "task2": ...}`.
- Picture: `flow.generate_graph_img(name="graph", save_path=".")` returns the PNG path. Each step is labelled `name [agent_type:provider]` and colored by agent type, and routes are red dashed arrows. Keep the legend (the default) when the flow mixes agent types; pass `show_legend=False` for a text-only flow. It needs matplotlib, which the visual extra installs.

Vibe Agents (a flow from a plain language intent)
- `va = VibeAgent(planner_provider="anthropic", planner_api_key=os.environ["ANTHROPIC_API_KEY"], planner_model="<model>")`, then `flow = await va.build(intent, save_dir="bundle")`
- Reload without planning: `flow = va.build_from_spec(va.load_bundle("bundle/vibeflow_bundle.json"))`. The bundle stores the spec path exactly as given, so pass an absolute `save_dir`, or load with `va.load_spec("bundle/flow_spec.json")`.
- You can be the planner: write the FlowSpec JSON yourself, then `va = VibeAgent(planner_fn=lambda system, user: spec_dict, processors={...})` and `flow = await va.build(intent, save_dir="/abs/bundle")`. No planner provider or key is needed. `save_dir` writes flow_spec.json, vibeflow_bundle.json and vibeflow_graph.png.
- Language steps in a spec are `"agent_type": "assistant"`. A step's `options` can hold a store config (`"knowledge": {"type": "memory", "path": "./knowledge.json"}` or `{"type": "qdrant", "url": "${ENV:QDRANT_URL}", "collection": "docs"}`), `"files": ["./handbook.md"]`, `"history": {"type": "file", "dir": "./conversations"}` (with `model_params.conversation_id`), `"memory"` (with `model_params.user_id`) and `"tools": ["name"]`. `"show_sources": true` in model_params lists the cited sources.
- Python objects are registered by name: `VibeAgent(tools={"get_weather": fn}, stores={"handbook": store}, guards={"read_only": fn}, processors={...})`. The planner sees the names and each function's first docstring line. A planned spec that names something unregistered, keeps history without `conversation_id` or memory without `user_id`, has a cycle, or breaks another rule goes back to the planner with the error (up to three replies).
- Put `${ENV:NAME}` in specs, never keys, and set the variable in code before building, for example `os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")`. An unset variable stays as literal text.
- Before `flow.start()`, check each `flow.tasks[name].agent.provider`. A task without `agent.agent_type` skips validation and defaults to `openai`.
- Register processors, tools, stores and guards every time you load a spec. Each processor is `fn(text) -> text`. A loaded spec skips unknown `post_process` names silently. A processor sees only the output, so close over the source text if a check needs it.
- Tasks built from a spec use the default template (see Pitfalls). For exact prompts set `flow.tasks[name].template = obj` after building.

Assistant and vector stores (chat apps, RAG, memory)
- `from intelli.function.assistant import Assistant` and `from intelli.store import MemoryVectorStore, FileChatHistory, QdrantVectorStore, ...` (2.2.0 or above). In a flow, the same features come with `Agent("assistant", ...)`.
- `assistant = Assistant(provider="openai", api_key=key, history=FileChatHistory(dir="./conversations"), knowledge=MemoryVectorStore(embedder={"provider": "openai", "api_key": key}), memory=MemoryVectorStore(embedder=...))`. Providers as in Agents, plus `vertex` and `ollama`.
- `assistant.add_documents([{"id": "handbook", "text": text}])` splits into chunks `handbook#0`, `handbook#1`; `reply = assistant.chat(question, conversation_id=None, user_id="u1")` returns `text`, `conversation_id`, `references` (`cited` marks the [n] the answer used), `memories`, `usage`, `tool_steps`. `assistant.stream(...)` yields `start`, `text` chunks and `done`.
- Tools: `tools=[python_function]` (docstring and type hints become the schema); the Assistant runs the tool loop. Attachments: `attachments=["photo.png"]` (Gemini: images, PDFs, audio, video; Anthropic: images and PDFs; others: images). `google_search=True` on gemini or vertex returns web `citations`.
- Every store: `add_documents`, `upsert` (own vectors), `search(text, top_k, filter)`, `query(text= or vector=, top_k=, filter=, native_filter=)`, `delete(ids)`. `score` is a similarity, `filter` is metadata equality (a list means one of).
- Local stores without an API key: Qdrant, Chroma, Weaviate, Milvus, Elasticsearch, Postgres + pgvector, MongoDB Atlas Local (Docker). Use `MemoryVectorStore(path="store.json")` when nothing is installed. Use the same embedder for the life of a store, and check `count()` before embedding the same documents again.

Pitfalls
- Never make one task wait for two branches of the same DynamicConnector. Only one branch runs, so that task would never run: 2.1.0 and later raise a ValueError, and older versions fail silently. Give each branch its own next task.
- `memory_key` replaces the input from parent tasks. With a list of keys, each value is cut to 100 characters.
- An empty string from memory makes the task run on its description alone. Store "(none)" instead.
- The default template builds the prompt as `PREVIOUS_ANALYSIS: <input>`, then `CURRENT_TASK: <instruction>`. Before 2.1.0 it left a literal `{0}` and broke up Markdown headings, so small models echoed the input. For exact prompts pass any object with `apply_input(data) -> str` as `template=`. No other method is needed, and the instruction is not added for you.
- Tiny local models are poor planners. With the 2.2.0 planner (a prompt of about 10k characters that checks and corrects its own plan), qwen2.5:0.5b returned a valid spec in 8 of 12 tries, but always a single step, even when the request asked for two or three. Plan with a cloud model, or write the spec yourself.
- Docs index: https://www.intellinode.ai/llms.txt. Before using an Intelli API that is not listed above, open the matching page from the index. Where a docs page disagrees with the rules above, follow these rules: they were checked by running code.
