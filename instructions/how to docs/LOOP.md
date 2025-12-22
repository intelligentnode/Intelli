---
sidebar_position: 10
---

# Loop

`LoopTask` is a **single Flow unit** that repeats a small sequence of steps until:
- the **stop condition** returns `True`, or
- it hits **max_loops** (default: **5**).

This is the easiest way to do looping **without creating cycles** in the main Flow graph.


### Key parameters

- **max_loops**: maximum iterations (default `5`)
- **stop_condition(iteration, last_output, last_type, memory)**: stop rule after each loop
- **store_history_memory_key**: save per-iteration outputs into memory (optional)

### Example

```python
import asyncio

from intelli.flow.flow import Flow
from intelli.flow.store.memory import Memory
from intelli.flow.tasks.loop_task import LoopTask
from intelli.flow.types import InputTypes


class GrowText:
    def __init__(self, key="text"):
        self.key = key
        self.output = None
        self.output_type = InputTypes.TEXT.value

    def execute(self, input_data=None, input_type=None, memory=None):
        prev = memory.retrieve(self.key, "") if memory else ""
        self.output = (prev + " more").strip()
        if memory:
            memory.store(self.key, self.output)
        return self.output


def stop_when_long(iteration, last_output, last_type, memory):
    return isinstance(last_output, str) and len(last_output) >= 20


memory = Memory()
loop = LoopTask(
    desc="grow text until long enough",
    steps=[GrowText()],
    # max_loops defaults to 5
    stop_condition=stop_when_long,
    store_history_memory_key="loop_history",
)

flow = Flow(tasks={"loop": loop}, map_paths={}, memory=memory)
result = asyncio.run(flow.start(initial_input="start", initial_input_type="text"))

print(result["loop"]["output"])
print("iterations:", len(memory.retrieve("loop_history", [])))
```


