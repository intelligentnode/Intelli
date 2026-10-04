## Using Intelli (Python) to build agent flows

When the user asks for an AI tool, follow this sequence. The user may not read code.
1. Write the tool as an Intelli flow of small steps.
2. Run it. If `flow.errors` is not empty, fix the cause and run again.
3. Read the output. An empty `flow.errors` only means nothing crashed. Check that `sorted(out)` lists exactly the steps you expect and that the text is right for the input: a small model may return the input unchanged or give every item the same label. Fix and run again.
4. Save the flow picture: `flow.generate_graph_img(name="<tool>_graph", save_path=".", show_legend=False)`.
5. Report in plain language: what each step does, which model each step uses (say "a local model on your computer" for vllm), what you checked and what is still weak, and where the picture and the output files are.

Install with `pip install -U "intelli[visual]"`, in the project's virtual environment if it has one. The visual extra adds matplotlib for flow pictures. Use version 2.1.0 or above (`pip show intelli`); these rules were checked on 2.0.3 and still hold. Import from `intelli.flow`:
`from intelli.flow import Agent, Task, TextTaskInput, Flow, SequenceFlow, DynamicConnector, Memory, CustomAgent, VibeAgent`

Agents and tasks
- Which provider: the one the user names. If they name none, use OpenAI when `OPENAI_API_KEY` is set, else Anthropic when `ANTHROPIC_API_KEY` is set, else a local server. Say which one you used.
- Cloud, the easy start: `Agent("text", "openai", "mission", {"key": os.environ["OPENAI_API_KEY"], "model": "gpt-4.1-mini", "max_tokens": 300})` or `Agent("text", "anthropic", "mission", {"key": os.environ["ANTHROPIC_API_KEY"], "model": "claude-haiku-4-5", "max_tokens": 300})`. Read keys from the environment. Never hardcode or print them, and stop with a clear message when the key is missing.
- Local or offline, no key (Ollama or vLLM): `Agent("text", "vllm", "mission", {"model": "<ollama tag>", "temperature": 0.2, "max_tokens": 300}, options={"baseUrl": "http://localhost:11434"})`
- Plain Python step, no model: `class Step(CustomAgent)` with `def execute(self, agent_input, new_params=None): return fn(agent_input.desc)`, used as `Task(TextTaskInput("x"), Step("text"))`.
- `Task(TextTaskInput("instruction"), agent, pre_process=fn, post_process=fn, memory_key="name", exclude=False, template=obj)`
- The mission is the system prompt. For each task: `pre_process(input)`, then the template builds the prompt, then the model, then `post_process(output) -> output`. Child tasks and `out[name]["output"]` get the processed value.
- The default temperature is 1. Use `temperature: 0` for labels and extraction; 0.1 was still unstable on a small model.
- Small local models: sort, count and label in plain Python where you can, give each model step one narrow job, ask for plain descriptive words and map them to labels in `post_process`, and check each answer against its input. As writers they echo the input, ignore lengths and invent numbers, so tell the user.

Flow (async graph)
- `flow = Flow(tasks={"a": ta, "b": tb, "c": tc}, map_paths={"a": ["c"], "b": ["c"]}, memory=Memory())`
- `out = await flow.start(initial_input=text, max_workers=4)`, then read `out["c"]["output"]`. Every task with no parent gets `initial_input`, and tasks that are ready at the same time run in parallel.
- A task with several parents gets their outputs joined. If its mission contains "synthesize" or "integrate", each part is wrapped as `========== NAME OUTPUT ==========` ... `========== END OF NAME ==========`.
- Memory: `memory.store("key", value)` before the run and `memory.retrieve("key")` after it. `Flow(..., output_memory_map={"task": "key"})` copies a task's output into memory, and `memory_key="key"` on a task replaces its input with the stored value, which is how a later step can read the original input.
- For a batch, build a new Flow for each input.
- Routing: `dynamic_connectors={"a": DynamicConnector(decision_fn=lambda out, kind: "x", destinations={"x": "task_x", "y": "task_y"})}`. Only the chosen task runs (4 destinations at most), and it receives the output of task "a". The destination keys are printed on the picture, so use readable words.
- A routed task must have no other parent in `map_paths`, or every destination runs. To route after parallel steps, join them into one task and put the connector on that task.
- Task failures do not raise. Check `flow.errors` (a dict) after every run.
- `SequenceFlow([t1, t2]).start()` is synchronous and returns `{"task1": ..., "task2": ...}`.
- Picture: `flow.generate_graph_img(name="graph", save_path=".", show_legend=False)` returns the PNG path. It needs matplotlib, which the visual extra installs.

Vibe Agents (a flow from a plain language intent)
- `va = VibeAgent(planner_provider="anthropic", planner_api_key=os.environ["ANTHROPIC_API_KEY"], planner_model="<model>")`, then `flow = await va.build(intent, save_dir="bundle")`
- Reload without planning: `flow = va.build_from_spec(va.load_bundle("bundle/vibeflow_bundle.json"))`. The bundle stores the spec path exactly as given, so pass an absolute `save_dir`, or load with `va.load_spec("bundle/flow_spec.json")`.
- You can be the planner: write the FlowSpec JSON yourself, then `va = VibeAgent(planner_fn=lambda system, user: spec_dict, processors={...})` and `flow = await va.build(intent, save_dir="/abs/bundle")`. No planner provider or key is needed. `save_dir` writes flow_spec.json, vibeflow_bundle.json and vibeflow_graph.png.
- Put `${ENV:NAME}` in specs, never keys, and set the variable in code before building, for example `os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")`. An unset variable stays as literal text.
- Before `flow.start()`, check each `flow.tasks[name].agent.provider`. A task without `agent.agent_type` skips validation and defaults to `openai`.
- Register processors every time you load a spec: `VibeAgent(processors={"name": fn})`, each `fn(text) -> text`. Unknown `post_process` names are skipped silently. A processor sees only the output, so close over the source text if a check needs it.
- Tasks built from a spec use the default template (see Pitfalls). For exact prompts set `flow.tasks[name].template = obj` after building.

Pitfalls
- Never make one task depend on two branches of the same DynamicConnector. Only one branch runs, so that task never runs, and nothing errors.
- `memory_key` replaces the input from parent tasks. With a list of keys, each value is cut to 100 characters.
- An empty string from memory makes the task run on its description alone. Store "(none)" instead.
- The default template sends `PREVIOUS_ANALYSIS: {0}`, then `CURRENT_TASK: <instruction>`, then the input. `{0}` is never filled, and Markdown headings in the input get broken up, so small models echo the input. For exact prompts pass any object with `apply_input(data) -> str` as `template=`. No other method is needed, and the instruction is not added for you.
- Tiny local models are poor planners: qwen2.5:0.5b produced 0 valid VibeAgent specs in 42 tries. For a local planner, set `max_context_chars=0` (the default prompt is about 96k characters, and `context_files=[]` does not shrink it).
- Docs index: https://www.intellinode.ai/llms.txt. Before using an Intelli API that is not listed above, open the matching page from the index. Where a docs page disagrees with the rules above, follow these rules: they were checked by running code.
