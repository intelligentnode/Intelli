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

    def create_agent_and_task(self, task_input_desc, agent_type, provider, mission, model_key, model, log=True):
        task = Task(
            TextTaskInput(task_input_desc),
            Agent(agent_type, provider, mission, {"key": model_key, "model": model}),
            log=log
        )
        if agent_type == "image":
            task.exclude = True
        
        return task
    
    async def async_test_blog_flow(self):
        print("--- test blog flow ---")
        task1 = self.create_agent_and_task("identify requirements of building a blogging website about environment", 
                                           "text", "openai", 
                                           "write specifications", 
                                           self.openai_api_key, "gpt-3.5-turbo")

        task2 = self.create_agent_and_task("build task list for the technical team about the requirements", 
                                           "text", "gemini", 
                                           "create task list", 
                                           self.gemini_key, "gemini")

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
                                           "text", "openai", 
                                           "code generation from specifications", 
                                           self.openai_api_key, "gpt-4", log=True)

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

        flow.generate_graph_img(save_path='../temp')
        
        output = await flow.start()
        
        print("Final output:", output)
    
    async def async_test_vision_flow(self):
        print("--- test vision flow ---")
        
        task1 = self.create_agent_and_task(task_input_desc="generate arts", 
                                           agent_type="image", 
                                           provider="stability", 
                                           mission="generate a roboto riding a tax from the future.", 
                                           model_key=self.stability_key,
                                           model="stable-diffusion-xl-1024-v1-0")

        task2 = self.create_agent_and_task(task_input_desc="explain the image", 
                                           agent_type="vision", 
                                           provider="openai", 
                                           mission="generate description of the image elements", 
                                           model_key=self.openai_api_key, 
                                           model="gpt-4-vision-preview")
        
        flow = Flow(tasks = {
                        "task1": task1,
                        "task2": task2
                    }, map_paths = {
                        "task1": ["task2"]
                    }, log=True)
        
        output = await flow.start()

        print("Final output:", output)
    
    
    def test_graph_generate(self):
        
        title_task = self.create_agent_and_task("xxx", 
                                           "text", "gemini", 
                                           "yyyy", 
                                           self.gemini_key, "gemini")

        content_task = self.create_agent_and_task("xxx", 
                                           "text", "openai", 
                                           "yyyy", 
                                           self.openai_api_key, "gpt-3.5-turbo")

        keyword_task = self.create_agent_and_task("xxx", 
                                           "text", "openai", 
                                           "yyyy", 
                                           self.openai_api_key, "gpt-3.5-turbo")

        description_theme_task = self.create_agent_and_task("xxx", 
                                           "text", "openai", 
                                           "yyyy", 
                                           self.openai_api_key, "gpt-3.5-turbo")

        image_task = self.create_agent_and_task("xxx", 
                                           "image", "stability", 
                                           "yyyy", 
                                           self.stability_key, "")
        
        flow = Flow(
            tasks={
                "title_task": title_task,
                "content_task": content_task,
                "keyword_task": keyword_task,
                "theme_task": description_theme_task,
                "image_task": image_task,
            },
            map_paths={
                "title_task": ["keyword_task", "content_task"],
                "content_task": ["theme_task"],
                "theme_task": ["image_task"],
            },
        )
        
        flow.generate_graph_img(save_path='../temp')
    
    
    def test_blog_flow(self):
        asyncio.run(self.async_test_blog_flow())
    
    def test_vision_flow(self):
        asyncio.run(self.async_test_vision_flow())
    
if __name__ == "__main__":
    unittest.main()
