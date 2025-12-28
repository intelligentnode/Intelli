---
sidebar_position: 2
---

# Vibe Features

Vibe Agent uses a high-level planner to architect specialized multi-agent workflows. This page covers the technical design behind graph generation, tool integration, and flow lifecycle management.

## Initialization Parameters

The `VibeAgent` accepts the following configuration parameters:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `planner_provider` | `str` | LLM provider for the planner (`openai`, `anthropic`, `gemini`, `vllm`, `llamacpp`). |
| `planner_api_key` | `str` | API key for cloud planner providers. Optional for `vllm` or `llamacpp`. |
| `planner_model` | `str` | Specific model version (e.g., `gpt-4o`, `gemini-2.0-flash`). |
| `text_model` | `str` | (Optional) Preferred model string for text tasks. |
| `image_model` | `str` | (Optional) Preferred model string for image tasks. |
| `speech_model` | `str` | (Optional) Preferred model string for speech tasks. |
| `recognition_model` | `str` | (Optional) Preferred model string for transcription tasks. |
| `processors` | `dict` | Mapping of strings to post-processing functions. |
| `context_files` | `list` | Custom list of code files to provide to the planner for context. |

## Quick Example

```python
from intelli.flow import VibeAgent

# 1. Initialize the architect
va = VibeAgent(planner_provider="openai", planner_api_key="your_key")

# 2. Build the graph from a vibe
flow = await va.build("Research AI trends and generate a summary report")

# 3. Execute the flow
results = await flow.start()
```

## The Generated Blueprint

The **blueprint** is the bridge between natural language and the `Flow` execution engine.

### Task Definition
Each task in the blueprint follows this schema:
- `name`: Unique identifier for the task node.
- `desc`: The instruction to be executed by the agent.
- `agent`: `AgentSpec` containing `agent_type`, `provider`, and `mission`.
- `model_params`: Dictionary of model-specific parameters (e.g., temperature, max_tokens).
- `post_process`: Key of a function defined in the `VibeAgent.processors` map.

### Global Configuration
- `max_workers`: Integer defining the maximum parallel task execution.
- `output_memory_map`: Maps task outputs to shared memory keys for downstream access.
- `map_paths`: Defines the dependency edges between tasks (DAG structure).

## Advanced Methods

### Flow Persistence
Use `save_bundle` to export the generated blueprint for production use without re-running the planner.
```python
# Exports flow_spec.json and vibeflow_graph.png
va.save_bundle(save_dir="./prod_flow", spec=va.last_spec, flow=flow)
```

### Direct Blueprint Execution
Reconstruct a flow directly from a saved blueprint to save latency and costs.
```python
spec = va.load_spec("./prod_flow/flow_spec.json")
flow = va.build_from_spec(spec)
```

### Blueprint Modification
Modify existing architectures using natural language instructions.
```python
# existing_spec is loaded from a path or va.last_spec
updated_flow = await va.edit(spec_path="./path/to/spec.json", instruction="Add a validation step")
```

## Runtime Features

### Environment Variable Injection
Vibe Agent supports secret resolution at runtime using the `${ENV:VAR_NAME}` syntax. This ensures that API keys or sensitive endpoints are not persisted in the `flow_spec.json` file.

### Automatic Output Management
Configure the execution engine to handle file persistence automatically:
- `flow.auto_save_outputs = True`: Automatically writes images and audio files to disk.
- `flow.output_dir = "./path"`: Sets the target directory for all generated assets.

## Offline / Local Models (vLLM + llama.cpp)

VibeAgent supports local inference for the Architect (planner) or individual tasks when you provide the connection details.

- **vLLM**: provide `baseUrl` (e.g. `${ENV:VLLM_BASE_URL}`) and a model name.
- **llama.cpp**: provide `model_path` (e.g. `${ENV:LLAMACPP_MODEL_PATH}`).

```python
import os
from intelli.flow import VibeAgent

va = VibeAgent(
    planner_provider="vllm",
    planner_model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    planner_options={"baseUrl": os.getenv("VLLM_BASE_URL")},
)
```

### Example prompt to generate offline agent

This prompt tells the Architect (planner) to **design the execution graph** and generate with **offline text agent**.  
When you call `await va.build(vibe)`, VibeAgent uses the planner to produce the full spec (tasks, providers, routing) and then builds a runnable `Flow`.

```python
vibe = (
    "Create a 2-step flow. "
    "Step 1 (research): Use OpenAI to extract 8 key facts about solid-state batteries. "
    "Step 2 (offline_summary): Use vLLM at ${ENV:VLLM_BASE_URL} with model "
    "'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' to summarize the facts into 5 bullets."
)

flow = await va.build(vibe)
```