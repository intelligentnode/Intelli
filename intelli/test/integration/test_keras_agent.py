import os
import unittest
from intelli.flow.types import *
from intelli.flow.agents.agent import Agent
from intelli.flow.agents.kagent import KerasAgent
from intelli.flow.input.agent_input import TextAgentInput
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.tasks.task import Task
from dotenv import load_dotenv
# load env
load_dotenv()

class TestKerasFlows(unittest.TestCase):
    def setUp(self):
        # gpt2 is small and open, the gated models like gemma need the kaggle credentials
        self.model_name = os.getenv("KERAS_TEXT_MODEL", "gpt2_base_en")
        self.kaggle_username = os.getenv("KAGGLE_USERNAME")
        self.kaggle_pass = os.getenv("KAGGLE_API_KEY")
    
    def test_blog_post_flow(self):
        print("---- start simple blog post flow ----")
        
        # Define agents
        model_params = {
            "model_name": self.model_name,
            "max_length": 64,
            "KAGGLE_USERNAME": self.kaggle_username,
            "KAGGLE_KEY": self.kaggle_pass,
        }
        keras_agent = KerasAgent(agent_type="text", 
                                 mission="write blog posts",
                                 model_params=model_params)
        
        # Define tasks
        task1 = Task(
            TextTaskInput("blog post about electric cars"), keras_agent, log=True
        )

        # Start SequenceFlow
        flow = SequenceFlow([task1], log=True)
        final_result = flow.start()

        print("Final result:", final_result)
        self.assertIsNotNone(final_result)

    def test_generation_options(self):
        print("---- start generation options ----")

        keras_agent = KerasAgent(agent_type="text",
                                 model_params={"model_name": self.model_name})

        # temperature zero selects the greedy sampler, the output is repeatable
        new_params = {"max_new_tokens": 12, "temperature": 0}
        first = keras_agent.execute(TextAgentInput("The capital of France is"), new_params=new_params)
        second = keras_agent.execute(TextAgentInput("The capital of France is"), new_params=new_params)

        print("Greedy result:", first)
        self.assertTrue(first)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
