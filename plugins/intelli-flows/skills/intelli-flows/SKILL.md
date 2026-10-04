---
name: intelli-flows
description: Use when the user wants to design, draw or build an AI workflow in Python with Intelli (intelli.flow), especially one that mixes providers (OpenAI, Anthropic, Gemini, AWS Bedrock, Mistral, local models) or agent types (text, image generation, vision, speech, transcription, embeddings, search, MCP tools), or asks for a picture of an agent workflow. Also for Flow graphs, routing with DynamicConnector, Memory, and Vibe Agents and their FlowSpec JSON. Not for IntelliNode on Node.js.
---

# Intelli flows

Follow the Intelli section of [AGENTS.md](AGENTS.md) in this skill's folder, then these rules. For which
agent types and providers exist and what passes between them, see [references/agents.md](references/agents.md).

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

   writer = Agent("text", "anthropic", "Write a two-sentence product pitch.",
                  {"key": key("ANTHROPIC_API_KEY"), "model": "claude-haiku-4-5", "max_tokens": 300})
   artist = Agent("image", "openai", "Flat illustration for this pitch, no text",
                  {"key": key("OPENAI_API_KEY"), "model": "gpt-image-2", "width": 1024, "height": 1024})
   critic = Agent("vision", "gemini", "Does the image match the pitch? Answer yes or no, then one reason",
                  {"key": key("GEMINI_API_KEY"), "model": "gemini-2.5-flash"})
   voice = Agent("speech", "openai", "Read this aloud",
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
   - Each provider needs its own key. A missing key fails only that step, so run only when every provider's
     key is set, or offer to move that step to a provider whose key is set.
   - Image, video and audio generation are billed per call. Tell the user before running those steps.
4. For a Vibe Agent you are the planner. Write the FlowSpec JSON yourself, save it in the repo,
   and load it with `planner_fn` or `build_from_spec`. Use this shape:

   {"version": "1",
    "tasks": [{"name": "thread", "desc": "what the task does", "post_process": "flag_new_numbers",
               "agent": {"agent_type": "text", "provider": "vllm", "mission": "who the agent is",
                         "model_params": {"model": "qwen2.5:0.5b", "temperature": 0.3, "max_tokens": 300},
                         "options": {"baseUrl": "${ENV:OLLAMA_BASE_URL}"}}}],
    "map_paths": {}, "dynamic_connectors": [], "output_memory_map": {}}

   For a local or offline model use provider "vllm" with options {"baseUrl": "${ENV:OLLAMA_BASE_URL}"}
   in specs, http://localhost:11434 in code.
5. Every task needs `agent.agent_type`. Without it validation is skipped and the provider becomes openai.
6. Name each `post_process` in the spec and register the same names with `VibeAgent(processors=...)`
   every time the spec is loaded. Unknown names are skipped silently.
7. Before `flow.start()`, loop over `flow.tasks` and raise if any `agent.provider` is not the one
   the user asked for. After every run, raise if `flow.errors` is not empty.
8. Choose the form. Write a code Flow when steps need routing, exact prompts, Python checks, or image,
   vision or audio steps. Write a Vibe Agent spec when the steps are plain text steps and the user wants
   a saved plan to rerun or change.
9. A small local model is fine for a first run. Follow the small-model rules in AGENTS.md and tell the user
   what it got wrong, so they can decide which steps need a stronger model.
10. If you cannot install packages where you run, write the code for the user to run and draw the same
    graph as a Mermaid `flowchart` with the same step names, so they still get a picture.
11. For anything else, fetch https://www.intellinode.ai/llms.txt and open the page it lists.
