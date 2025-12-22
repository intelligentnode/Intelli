from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from intelli.flow.types import InputTypes


@dataclass
class _LoopAgentStub:
    """
    Minimal stub to satisfy Flow's graph metadata requirements:
    Flow._prepare_graph() reads task.agent.provider and task.agent.type.
    """

    type: str = "text"
    provider: str = "loop"
    mission: str = "loop"
    model_params: Optional[dict] = None


class LoopTask:
    """
    Composite task that runs an ordered list of Task-like steps in sequence,
    repeating the whole sequence up to `max_loops` times (default: 5) or until
    `stop_condition(...)` returns True.
    """

    def __init__(
        self,
        desc: str,
        steps: List[Any],
        *,
        max_loops: int = 5,
        stop_condition: Optional[Callable[[int, Any, str, Any], bool]] = None,
        exclude: bool = False,
        store_history_memory_key: Optional[str] = None,
    ):
        if not steps:
            raise ValueError("LoopTask requires at least one step in `steps`")
        if max_loops is None or max_loops <= 0:
            raise ValueError("LoopTask `max_loops` must be a positive integer")

        self.desc = desc
        self.steps = steps
        self.max_loops = max_loops
        self.stop_condition = stop_condition
        self.exclude = exclude
        self.store_history_memory_key = store_history_memory_key

        # Satisfy Flow._prepare_graph() expectations.
        self.agent = _LoopAgentStub()

        self.output: Any = None
        self.output_type: str = InputTypes.TEXT.value

    def execute(self, input_data=None, input_type=None, memory=None):
        current = input_data
        current_type = input_type

        history: List[dict] = []

        for iteration in range(1, self.max_loops + 1):
            for step in self.steps:
                step.execute(current, input_type=current_type, memory=memory)
                current = getattr(step, "output", None)
                current_type = getattr(step, "output_type", None)

            history.append(
                {"iteration": iteration, "output": current, "type": current_type}
            )

            if self.stop_condition is not None:
                try:
                    if self.stop_condition(iteration, current, current_type, memory):
                        break
                except Exception:
                    # Fail-safe: never crash the flow due to stop condition bugs.
                    pass

        self.output = current
        self.output_type = current_type or InputTypes.TEXT.value

        if memory is not None and self.store_history_memory_key:
            memory.store(self.store_history_memory_key, history)

        return self.output


