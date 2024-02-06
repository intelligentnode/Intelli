import os
import unittest
from intelli.flow.agents.agent import Agent
from intelli.flow.task import Task
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.processors.basic_processor import TextProcessor
from dotenv import load_dotenv
load_dotenv()

class TestFlows(unittest.TestCase):
    def setUp(self):
        # Initiate the keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.stability_key = os.getenv("STABILITY_API_KEY")

    def test_blog_post_flow(self):
        print('---- start blog post flow ----')
        # Define agents
        blog_agent = Agent(agent_type='text', provider='openai', mission='write blog posts',
                        model_params={'key': self.openai_api_key, 'model': 'gpt-3.5-turbo'})
        description_agent = Agent(agent_type='text', provider='gemini', mission='generate description',
                                model_params={'key': self.gemini_key, 'model': 'gemini'})
        image_agent = Agent(agent_type='image', provider='stability', mission='generate image',
                            model_params={'key': self.stability_key})

        # Define tasks
        task1 = Task(TextTaskInput('blog post about electric cars'), blog_agent, log=True)
        task2 = Task(TextTaskInput('Generate short image description for image model'), description_agent, pre_process=TextProcessor.text_head, log=True)
        task3 = Task(TextTaskInput('Generate cartoon style image'), image_agent, log=True)

        # Start SequenceFlow
        flow = SequenceFlow([task1, task2, task3], log=True)
        final_result = flow.start()

        print("Final result:", final_result)

if __name__ == '__main__':
    unittest.main()
