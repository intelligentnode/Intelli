import asyncio
import unittest

from intelli.flow import Flow, Task, TextTaskInput
from intelli.flow.agents.custom_agent import CustomAgent
from intelli.flow.template.basic_template import TextInputTemplate
from intelli.flow.types import AgentTypes


class RecordingAgent(CustomAgent):
    """Keeps the prompts it receives and returns a fixed reply, no model involved."""

    def __init__(self, reply):
        super().__init__(agent_type=AgentTypes.TEXT.value, provider="custom", mission="record")
        self.reply = reply
        self.prompts = []

    def execute(self, agent_input, new_params=None):
        self.prompts.append(agent_input.desc)
        return self.reply


MARKDOWN_POST = "# We cut our support time\n\nTickets are answered in half the time.\n\n## RISK ASSESSMENT:\nLow."


class TestTextInputTemplate(unittest.TestCase):
    def test_default_template_substitutes_input(self):
        template = TextInputTemplate("Summarize the text")

        self.assertEqual(
            template.apply_input("the previous output"),
            "PREVIOUS_ANALYSIS: the previous output\nCURRENT_TASK: Summarize the text",
        )

    def test_default_template_without_input_returns_instruction_only(self):
        template = TextInputTemplate("Summarize the text")

        self.assertEqual(template.apply_input(None), "Summarize the text")

    def test_custom_placeholder_is_substituted(self):
        template = TextInputTemplate("Context: {0}\nRequest: shorten the context")

        self.assertEqual(
            template.apply_input("a long paragraph"),
            "Context: a long paragraph\nRequest: shorten the context",
        )

    def test_other_braces_in_the_instruction_are_kept(self):
        instruction = 'Reply with JSON like {"label": "bug"} and keep {name} as is'
        template = TextInputTemplate(instruction)

        self.assertEqual(
            template.apply_input("The app crashes on start"),
            "PREVIOUS_ANALYSIS: The app crashes on start\nCURRENT_TASK: " + instruction,
        )

    def test_json_input_is_substituted_as_json_block(self):
        template = TextInputTemplate("Describe the record")

        self.assertEqual(
            template.apply_input({"name": "Ada"}),
            'PREVIOUS_ANALYSIS: ```json\n{\n  "name": "Ada"\n}\n```\nCURRENT_TASK: Describe the record',
        )

    def test_markdown_headings_pass_through_unchanged(self):
        template = TextInputTemplate("Summarize the post")

        self.assertEqual(
            template.apply_input(MARKDOWN_POST),
            "PREVIOUS_ANALYSIS: " + MARKDOWN_POST + "\nCURRENT_TASK: Summarize the post",
        )

    def test_root_task_prompt_contains_the_initial_input(self):
        agent = RecordingAgent("summary")
        flow = Flow(tasks={"summarize": Task(TextTaskInput("Summarize the post"), agent)}, map_paths={})

        asyncio.run(flow.start(initial_input=MARKDOWN_POST))

        self.assertEqual(flow.errors, {})
        self.assertEqual(
            agent.prompts,
            ["PREVIOUS_ANALYSIS: " + MARKDOWN_POST + "\nCURRENT_TASK: Summarize the post"],
        )


if __name__ == "__main__":
    unittest.main()
