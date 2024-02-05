import unittest
from flow.agent import Agent
from flow.task import Task
from flow.sequence_flow import SequenceFlow
from flow.template import Template
import os
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
        blog_agent = Agent('text', 'openai', 'write blog posts',
                        params={'key': self.openai_api_key, 'model': 'gpt-4'})
        description_agent = Agent('text', 'gemini', 'generate description',
                                params={'key': self.gemini_key, 'model': 'gemini'})
        image_agent = Agent('image', 'stability', 'generate image',
                            params={'key': self.stability_key})

        # Define tasks
        task1 = Task('Generate blog post', blog_agent, template=Template(output_template="{result}"))
        task2 = Task('Generate image description', description_agent, 
                    template=Template(input_template="Generate an image based on: {output}", output_template="{result}"))
        task3 = Task('Generate image', image_agent, template=Template(input_template="Image prompt: {output}"))

        # Assuming SequenceFlow
        flow = SequenceFlow([task1, task2, task3], log=True)
        final_result = flow.start(initial_input="Technology advancements in the 21st century")


        print("Final result:", final_result)

if __name__ == '__main__':
    unittest.main()
