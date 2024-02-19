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
        print("---- start blog portal flow ----")

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
            model_params={"key": self.openai_api_key, "model": "dall-e-3"},
        )

        # Define tasks
        task2 = Task(
            TextTaskInput("Generate icon about following feature: {0}"), image_agent, exclude=True, log=True
        )

        task1_list = []
        topics = ["Any Model Access: connect various AI models, such as OpenAI, Cohere, LLaMa v2, Google Gemini using a unified input layer with minimum change.",
                  "Evaluation: run continuous evaluation across multiple models with metrics and select the suitable one for your use cases.",
                  "Optimized Workflow: manage the relations between multiple AI models as a graph to build advanced tasks."]
        for topic in topics:
            task1 = Task(
                TextTaskInput("Write short icon image description for image generation model about {}".format(topic)),
                desc_agent,
                log=True,
            )
            task1_list.append(task1)

        # Start SequenceFlow
        for index, task1 in enumerate(task1_list):
            flow = SequenceFlow([task1, task2], log=True)
            final_result = flow.start()

            print("- flow result:", final_result)


if __name__ == "__main__":
    unittest.main()
