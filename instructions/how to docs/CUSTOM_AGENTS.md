---
sidebar_position: 6
---

# Custom Agents

Sometimes you want to run **your own model** (local or hosted) or a **custom step** (business logic, tools, validators).  
`CustomAgent` lets you plug that into Intelli and still benefit from Flow features (routing, memory, looping, connectors) **without changing Flow or Task**.

## What is `CustomAgent`?

`CustomAgent` is a lightweight base class that matches what Flow expects from any agent:
- `type` (one of `AgentTypes`, e.g. `text`, `search`, `speech`, …)
- `provider` (a name like `"custom"`)
- `mission` (a short description)
- `model_params` (optional dict)
- `execute(agent_input, new_params=None)` (your implementation)

Flow/Task will call `execute(...)` and use the returned output like any other agent.

## Example

Create your agent by subclassing `CustomAgent` and implementing `execute(...)`:

```python
from intelli.flow.agents.custom_agent import CustomAgent
from intelli.flow.types import AgentTypes


class UppercaseAgent(CustomAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentTypes.TEXT.value,
            provider="custom",
            mission="uppercase text",
        )

    def execute(self, agent_input, new_params=None):
        return (agent_input.desc or "").upper()
```

Use it in a Task / Flow (same as any normal agent):

```python
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.tasks.task import Task
from intelli.flow.input.task_input import TextTaskInput

task = Task(TextTaskInput("hello"), UppercaseAgent())
flow = SequenceFlow([task])
result = flow.start()

print(result["task1"])
```

## Notes

- `agent_input` is usually `TextAgentInput` / `ImageAgentInput` depending on the agent type.  
- `new_params` comes from `Task(model_params=...)` and can be used to override behavior per-task.
