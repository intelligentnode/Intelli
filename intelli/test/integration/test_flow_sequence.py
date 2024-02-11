import os
import base64
import unittest
from intelli.flow.types import *
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput, ImageTaskInput
from intelli.flow.processors.basic_processor import TextProcessor
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.tasks.task import Task
from dotenv import load_dotenv


load_dotenv()


class TestFlows(unittest.TestCase):
    def setUp(self):
        # Initiate the keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.stability_key = os.getenv("STABILITY_API_KEY")
    
    def test_blog_post_flow(self):
        print("---- start blog portal flow ----")
        
        # Define agents
        blog_agent = Agent(
            agent_type=AgentTypes.TEXT.value,
            provider="openai",
            mission="write blog posts",
            model_params={"key": self.openai_api_key, "model": "gpt-3.5-turbo"},
        )
        description_agent = Agent(
            agent_type=AgentTypes.TEXT.value,
            provider="gemini",
            mission="generate description only",
            model_params={"key": self.gemini_key, "model": "gemini"},
        )
        image_agent = Agent(
            agent_type=AgentTypes.IMAGE.value,
            provider="stability",
            mission="generate image",
            model_params={"key": self.stability_key},
        )

        # Define tasks
        task1 = Task(
            TextTaskInput("blog post about electric cars"), blog_agent, log=True
        )
        task2 = Task(
            TextTaskInput("Write short image description for image generation model"),
            description_agent,
            pre_process=TextProcessor.text_head,
            log=True,
        )
        task3 = Task(
            TextTaskInput("Generate cartoon style image"), image_agent, exclude=True, log=True
        )

        # Start SequenceFlow
        flow = SequenceFlow([task1, task2, task3], log=True)
        final_result = flow.start()

        print("Final result:", final_result)
    
    def test_flow_chart_image_flow(self):
        print("---- start vision coder flow ----")
        
        analyst = Agent(
            agent_type=AgentTypes.VISION.value,
            provider="openai",
            mission="describe flow charts from images",
            model_params={"key": self.openai_api_key, "extension": "jpg", "model": "gpt-4-vision-preview"},
        )
        
        coder = Agent(
            agent_type=AgentTypes.TEXT.value,
            provider="openai",
            mission="write python code. response only with the code without explination or text or marks.",
            model_params={"key": self.openai_api_key, "model": "gpt-3.5-turbo"},
        )
        
        # Define tasks
        with open('../temp/code_flow_char.jpg', "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
        task1 = Task(
            ImageTaskInput(desc="describe the steps of the code flow chat for an engineer.", img=image_data), agent=analyst, log=True
        )

        task2 = Task(
            TextTaskInput("write python code from the provided context"), agent=coder, log=True
        )
        
        # Start SequenceFlow
        flow = SequenceFlow([task1, task2], log=True)
        final_result = flow.start()

        print("Final result:", final_result)

if __name__ == "__main__":
    unittest.main()
