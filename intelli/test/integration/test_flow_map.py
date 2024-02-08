import os
import unittest
import asyncio
from intelli.flow.processors.basic_processor import TextProcessor
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.processors.basic_processor import TextProcessor
from intelli.flow.flow import Flow
from intelli.flow.tasks.task import Task
from dotenv import load_dotenv

load_dotenv()


class TestAsyncFlow(unittest.TestCase):

    def setUp(self):
        # Initiate the keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.stability_key = os.getenv("STABILITY_API_KEY")

    def test_async_flow(self):

        task1 = Task(
            TextTaskInput("identify requirements of building blogging website about environment"),
            Agent("text", "gemini", "write specifications", {"key": self.openai_api_key, "model": "gemini"}),
            log=True
        )

        task2 = Task(
            TextTaskInput("build task list for the technical team about the requirements"),
            Agent("text", "openai", "create task list", {"key": self.openai_api_key, "model": "gpt-3.5-turbo"}),
            pre_process=TextProcessor.text_head,
            log=True
        )

        task3 = Task(
            TextTaskInput("generate the website description and theme details from the requirements"),
            Agent("text", "openai", "user experience and designer",
                  {"key": self.openai_api_key, "model": "gpt-3.5-turbo"}),
            log=True
        )

        task4 = Task(
            TextTaskInput("Generate short image description for image model"),
            Agent("text", "write image description",
                  {"key": self.openai_api_key, "model": "gpt-3.5-turbo"}),
            log=True
        )

        task5 = Task(
            TextTaskInput("design logo from the description"),
            Agent(agent_type="image", provider="stability", mission="generate logo with colorful style", model_params={"key": self.stability_key},
            ),
            log=True
        )

        async_flow = Flow([task1, task2, task3, task4, task5, task6],
                          map_paths={'task1': ['task2', 'task3'], 'task3': ['task4'], 'task4': ['task5']},
                          log=True)
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(async_flow.start(max_workers=5))

        print("Final results: ", results)


if __name__ == "__main__":
    unittest.main()