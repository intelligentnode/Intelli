---
name: intelli-flows
description: Use when writing or changing Python code that builds an AI agent app with Intelli (intelli.flow), such as a Flow graph, routing with DynamicConnector, Memory, or a Vibe Agent and its FlowSpec JSON. Not for IntelliNode on Node.js.
---

# Writing Intelli flows

Follow the Intelli section of [AGENTS.md](AGENTS.md) in this skill's folder, then these rules.

1. Pick the provider the way AGENTS.md says: the one the user names, else OpenAI or Anthropic when its
   key is set, else a local server. For a local or offline model use provider "vllm" with
   options {"baseUrl": "${ENV:OLLAMA_BASE_URL}"} in specs, http://localhost:11434 in code.
2. For a Vibe Agent you are the planner. Write the FlowSpec JSON yourself, save it in the repo,
   and load it with `planner_fn` or `build_from_spec`. Use this shape:

   {"version": "1",
    "tasks": [{"name": "thread", "desc": "what the task does", "post_process": "flag_new_numbers",
               "agent": {"agent_type": "text", "provider": "vllm", "mission": "who the agent is",
                         "model_params": {"model": "qwen2.5:0.5b", "temperature": 0.3, "max_tokens": 300},
                         "options": {"baseUrl": "${ENV:OLLAMA_BASE_URL}"}}}],
    "map_paths": {}, "dynamic_connectors": [], "output_memory_map": {}}

3. Every task needs `agent.agent_type`. Without it validation is skipped and the provider becomes openai.
4. Name each `post_process` in the spec and register the same names with `VibeAgent(processors=...)`
   every time the spec is loaded. Unknown names are skipped silently.
5. Before `flow.start()`, loop over `flow.tasks` and raise if any `agent.provider` is not the one
   the user asked for. After every run, raise if `flow.errors` is not empty.
6. For anything else, fetch https://www.intellinode.ai/llms.txt and open the page it lists.
7. Follow the sequence at the top of the Intelli section in AGENTS.md every time: run, read the output,
   save the picture, report in plain language. The user may not read code.
8. Choose the form. Write a code Flow when steps need routing, exact prompts or Python checks. Write a
   Vibe Agent spec when the steps are plain text steps and the user wants a saved plan to rerun or change.
9. A small local model is fine for a first run. Follow the small-model rules in AGENTS.md and tell the user
   what it got wrong, so they can decide which steps need a stronger model.
