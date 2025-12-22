---
sidebar_position: 11
---

# Vibe Flow

`VibeFlow` allows you to build and execute complex multi-modal AI flows using **natural language descriptions**. Instead of manually defining tasks and dependencies, you describe your goal, and `VibeFlow` handles the orchestration.

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
from intelli.flow.vibe import VibeFlow

async def main():
    # 1. Setup the planner (requires an API key)
    vf = VibeFlow(
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

`VibeFlow` can orchestrate different types of agents (text, image, audio) in a single request:

```python
description = (
    "1. Generate a speech audio for 'Intelli is awesome' using tts-1. "
    "2. Transcribe that audio back to text using whisper-1."
)

flow = await vf.build(description)
results = await flow.start()
```

### Key Features

- **Environment Variables**: Use `${ENV:VARIABLE_NAME}` in your prompts, and VibeFlow will automatically resolve them from your `.env` file.
- **Save & Load**: Use `save_bundle(save_dir)` to export the generated flow spec and a graph image visualization for future use.
- **Edit Mode**: Call `vf.edit(spec_path, "add a translation step")` to modify an existing flow using natural language.

### Auto-Saving Outputs

VibeFlow can automatically save generated images and audio to a specific directory:

```python
flow = await vf.build("Generate an image of a cyber cat and save it to ./outputs")
flow.auto_save_outputs = True
flow.output_dir = "./outputs"

await flow.start()
```

### Preferred Models

You can specify preferred model details as strings when initializing `VibeFlow`. This guides the planner to use specific versions of AI models for text, image, speech, or recognition tasks.

```python
vf = VibeFlow(
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

