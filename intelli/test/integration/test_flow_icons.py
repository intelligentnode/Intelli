import base64
import os
import unittest
from dotenv import load_dotenv

from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput, ImageTaskInput
from intelli.flow.processors.basic_processor import TextProcessor
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.tasks.task import Task
from intelli.flow.types import *

load_dotenv()


class TestFlows(unittest.TestCase):
    def setUp(self):
        # Initiate the keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.stability_key = os.getenv("STABILITY_API_KEY")

    def test_icon_generate_flow(self):
        print("---- start icons flow ----")

        # Define agents
        desc_agent = Agent(
            agent_type=AgentTypes.TEXT.value,
            provider="openai",
            mission="generate image description from the user input to use it for DALL·E icon generation",
            model_params={"key": self.openai_api_key, "model": "gpt-3.5-turbo"},
        )

        image_agent = Agent(
            agent_type=AgentTypes.IMAGE.value,
            provider="openai",
            mission="generate image",
            model_params={"key": self.openai_api_key, "model": "dall-e-3", "width": 1024, "height": 1024},
        )

        # Define tasks
        task2 = Task(
            TextTaskInput("flat icon about {0}"), image_agent, log=False
        )

        task1_list = []
        topics = ["unified ai models access", "evaluate large language models", "workflows"]

        for topic in topics:
            task1 = Task(
                TextTaskInput(
                    "Write simple icon description cartoon style inspired from docusaurus style about: {}".format(
                        topic)),
                desc_agent,
                log=False,
            )
            task1_list.append(task1)

        # Start SequenceFlow
        for index, task1 in enumerate(task1_list):
            print(f'---- Execute task {index+1} ----')
            flow = SequenceFlow([task1, task2], log=False)
            final_result = flow.start()

            print(f"{index + 1}- flow result:", final_result)


if __name__ == "__main__":
    unittest.main()
