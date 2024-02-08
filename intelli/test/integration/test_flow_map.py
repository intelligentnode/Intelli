import asyncio
import os
import unittest
from dotenv import load_dotenv
from intelli.flow.agents.agent import Agent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.tasks.task import Task
from intelli.flow.flow import Flow

load_dotenv()

class TestAsyncFlow(unittest.TestCase):
    def setUp(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.stability_key = os.getenv("STABILITY_API_KEY")

    def create_agent_and_task(self, task_input_desc, agent_type, provider, mission, model_key, model):
        task = Task(
            TextTaskInput(task_input_desc),
            Agent(agent_type, provider, mission, {"key": model_key, "model": model}),
            log=True
        )
        if agent_type == "image":
            task.exclude = True
        
        return task

    async def async_test_blog_flow(self):
        task1 = self.create_agent_and_task("identify requirements of building a blogging website about environment", 
                                           "text", "gemini", 
                                           "write specifications", 
                                           self.gemini_key, "gemini")

        task2 = self.create_agent_and_task("build task list for the technical team about the requirements", 
                                           "text", "openai", 
                                           "create task list", 
                                           self.openai_api_key, "gpt-3.5-turbo")

        task3 = self.create_agent_and_task("generate the website description and theme details from the requirements", 
                                           "text", "openai", 
                                           "user experience and designer", 
                                           self.openai_api_key, "gpt-3.5-turbo")

        task4 = self.create_agent_and_task("Generate short image description for image model", 
                                           "text", "openai", 
                                           "write image description", 
                                           self.openai_api_key, "gpt-3.5-turbo")

        task5 = self.create_agent_and_task("design logo from the description", 
                                           "image", "stability", 
                                           "generate logo with colorful style", 
                                           self.stability_key, "")

        task6 = self.create_agent_and_task("generate code based on combined tasks", 
                                           "text", "gemini", 
                                           "code generation from specifications", 
                                           self.gemini_key, "gemini")

        flow = Flow(tasks = {
                        "task1": task1,
                        "task2": task2,
                        "task3": task3,
                        "task4": task4,
                        "task5": task5,
                        "task6": task6,
                    }, map_paths = {
                        "task1": ["task2", "task3", "task6"],
                        "task2": ["task4"],
                        "task3": ["task4", "task6"],
                        "task4": ["task5"],
                        "task6": [],
                    }, log=True)

        output = await flow.start()

        print("Final output:", output)

    def test_blog_flow(self):
        asyncio.run(self.async_test_blog_flow())

if __name__ == "__main__":
    unittest.main()
