import os
import unittest
from intelli.flow.types import *
from intelli.flow.agents.agent import Agent
from intelli.flow.agents.kagent import KerasAgent
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.tasks.task import Task
from dotenv import load_dotenv
# set keras back
os.environ["KERAS_BACKEND"] = "jax"
# load env
load_dotenv()

class TestKerasFlows(unittest.TestCase):
    def setUp(self):
        self.kaggle_username = os.getenv("KAGGLE_USERNAME")
        self.kaggle_pass = os.getenv("KAGGLE_API_KEY")
    
    def test_blog_post_flow(self):
        print("---- start simple blog post flow ----")
        
        # Define agents
        gemma_model_params = {
            "model": "gemma_instruct_2b_en",
            "max_length": 64,
            "KAGGLE_USERNAME": self.kaggle_username,
            "KAGGLE_KEY": self.kaggle_pass,
        }
        gemma_agent = KerasAgent(agent_type="text", 
                                 mission="write blog posts",
                                 model_params=gemma_model_params)
        
        # Define tasks
        task1 = Task(
            TextTaskInput("blog post about electric cars"), gemma_agent, log=True
        )

        # Start SequenceFlow
        flow = SequenceFlow([task1], log=True)
        final_result = flow.start()

        print("Final result:", final_result)
        self.assertIsNotNone(final_result)


if __name__ == "__main__":
    unittest.main()
