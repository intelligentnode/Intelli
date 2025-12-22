import asyncio
import os
import re
import unittest

from dotenv import load_dotenv

from intelli.flow.flow import Flow
from intelli.flow.store.memory import Memory
from intelli.flow.tasks.loop_task import LoopTask
from intelli.flow.types import InputTypes


load_dotenv()


class _ComplexGeneratorStep:
    """
    Deterministic "fake model" step that generates "complex" text.
    It parses a prompt like: "length greater than X" and grows output each call.
    Stores the latest complex output in memory under `complex_key`.
    """

    def __init__(self, grow_by: int = 20, complex_key: str = "complex"):
        self.grow_by = grow_by
        self.complex_key = complex_key
        self._target_len = None
        self.output = None
        self.output_type = InputTypes.TEXT.value

    def execute(self, input_data=None, input_type=None, memory=None):
        text = "" if input_data is None else str(input_data)

        # Parse target once from the prompt.
        if self._target_len is None:
            m = re.search(r"length\s+greater\s+than\s+(\d+)", text, re.IGNORECASE)
            self._target_len = int(m.group(1)) if m else 50

        # Grow based on the last *complex* output, not on the current input (which might be simplified).
        prev_complex = ""
        if memory is not None:
            prev_complex = memory.retrieve(self.complex_key, "") or ""
        next_len = len(prev_complex) + self.grow_by

        # "Complex" content (deterministic).
        self.output = ("COMPLEX_" * 100)[:next_len]
        self.output_type = InputTypes.TEXT.value

        if memory is not None:
            memory.store(self.complex_key, self.output)

        return self.output


class _SimplifyStep:
    """
    Deterministic "fake model" step that "simplifies" text by truncating.
    Reads the complex text from memory and writes simplified output to memory.
    """

    def __init__(
        self,
        complex_key: str = "complex",
        simplified_key: str = "simplified",
        max_len: int = 25,
    ):
        self.complex_key = complex_key
        self.simplified_key = simplified_key
        self.max_len = max_len
        self.output = None
        self.output_type = InputTypes.TEXT.value

    def execute(self, input_data=None, input_type=None, memory=None):
        complex_text = ""
        if memory is not None:
            complex_text = memory.retrieve(self.complex_key, "") or ""
        else:
            complex_text = "" if input_data is None else str(input_data)

        simplified = complex_text[: self.max_len]
        self.output = simplified
        self.output_type = InputTypes.TEXT.value

        if memory is not None:
            memory.store(self.simplified_key, simplified)

        return simplified


class TestDynamicFlowLoopTask(unittest.TestCase):
    def setUp(self):
        # Keep structure similar to other integration tests.
        # (No API keys required; this test is deterministic/offline.)
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
        }

    def test_loop_task_stops_early(self):
        """
        Loop: complex -> simplify
        Stop condition: driven by prompt "length greater than X" (based on COMPLEX output length),
        with default LoopTask max_loops=5.
        """

        memory = Memory()
        stop_len = 55
        prompt = (
            f"Create a complex explanation with length greater than {stop_len}, "
            f"then simplify it."
        )

        complex_step = _ComplexGeneratorStep(grow_by=20, complex_key="complex")
        simplify_step = _SimplifyStep(
            complex_key="complex", simplified_key="simplified", max_len=25
        )

        def stop_condition(iteration, last_output, last_type, mem):
            # stop once the complex output exceeds X (prompt-driven)
            complex_out = mem.retrieve("complex", "") if mem is not None else ""
            return isinstance(complex_out, str) and len(complex_out) > stop_len

        loop = LoopTask(
            desc="complex then simplify (loop)",
            steps=[complex_step, simplify_step],
            stop_condition=stop_condition,
            store_history_memory_key="loop_history",
        )

        flow = Flow(tasks={"loop": loop}, map_paths={}, memory=memory)
        results = asyncio.run(
            flow.start(initial_input=prompt, initial_input_type=InputTypes.TEXT.value)
        )

        self.assertIn("loop", results)
        final_out = results["loop"]["output"]
        self.assertIsInstance(final_out, str)
        self.assertLessEqual(len(final_out), 25)  # simplified output

        # Prompt-driven stop: complex output should exceed X at stop time.
        complex_out = memory.retrieve("complex")
        self.assertIsInstance(complex_out, str)
        self.assertGreater(len(complex_out), stop_len)

        # Verify it stopped early (before default max_loops=5).
        history = memory.retrieve("loop_history")
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)
        self.assertLess(len(history), 5)


