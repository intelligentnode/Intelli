from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from intelli.flow.types import InputTypes
from intelli.utils.logging import Logger


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
        stop_on_error: bool = False,
        exclude: bool = False,
        store_history_memory_key: Optional[str] = None,
        log: bool = False,
    ):
        if not steps:
            raise ValueError("LoopTask requires at least one step in `steps`")
        if max_loops is None or max_loops <= 0:
            raise ValueError("LoopTask `max_loops` must be a positive integer")

        self.desc = desc
        self.steps = steps
        self.max_loops = max_loops
        self.stop_condition = stop_condition
        self.stop_on_error = stop_on_error
        self.exclude = exclude
        self.store_history_memory_key = store_history_memory_key
        self.logger = Logger(log)

        # Satisfy Flow._prepare_graph() expectations.
        self.agent = _LoopAgentStub()

        self.output: Any = None
        self.output_type: str = InputTypes.TEXT.value

    def execute(self, input_data=None, input_type=None, memory=None):
        current = input_data
        current_type = input_type

        history: List[dict] = []

        for iteration in range(1, self.max_loops + 1):
            self.logger.log(f"LoopTask iteration {iteration}/{self.max_loops}")
            
            error_occurred = False
            for step in self.steps:
                step.execute(current, input_type=current_type, memory=memory)
                current = getattr(step, "output", None)
                current_type = getattr(step, "output_type", None)
                
                if self.stop_on_error and isinstance(current, str) and (current.startswith("Error") or "Error executing agent" in current):
                    self.logger.log(f"LoopTask stopping early due to error in step: {current}")
                    error_occurred = True
                    break

            history.append(
                {"iteration": iteration, "output": current, "type": current_type}
            )
            
            if error_occurred:
                break

            if self.stop_condition is not None:
                try:
                    if self.stop_condition(iteration, current, current_type, memory):
                        self.logger.log(f"LoopTask stop condition met at iteration {iteration}")
                        break
                except Exception as e:
                    # Fail-safe: log but don't crash
                    self.logger.log(f"Warning: LoopTask stop_condition error: {e}")
                    pass

        self.output = current
        self.output_type = current_type or InputTypes.TEXT.value

        if memory is not None and self.store_history_memory_key:
            memory.store(self.store_history_memory_key, history)

        return self.output


