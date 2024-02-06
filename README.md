<p align="center">
<img src="assets/flow_logo-round.png" width="180em">
</p>

# IntelliPy
Create chatbots and AI agent work flows. It allows to connect your data with multiple AI models like OpenAI, Gemini, and Mistral through a unified access layer.

# Install
```bash
pip install intelli
```

# Code Examples

## Create AI Flows
You can create a flow of tasks executed by different AI models. Here's an example of creating a blog post flow:

<img src="assets/flow_example.jpg" width="680em">


```python
from flow.agents.agent import Agent
from flow.task import Task
from flow.sequence_flow import SequenceFlow
from flow.input.task_input import TextTaskInput
from flow.processors.basic_processor import TextProcessor

# Define agents
blog_agent = Agent(agent_type='text', provider='openai', mission='write blog posts',
                model_params={'key': YOUR_OPENAI_API_KEY, 'model': 'gpt-3.5-turbo'})
description_agent = Agent(agent_type='text', provider='gemini', mission='generate description',
                        model_params={'key': YOUR_GEMINI_API_KEY, 'model': 'gemini'})
image_agent = Agent(agent_type='image', provider='stability', mission='generate image',
                    model_params={'key': YOUR_STABILITY_API_KEY})

# Define tasks
task1 = Task(TextTaskInput('blog post about electric cars'), blog_agent, log=True)
task2 = Task(TextTaskInput('Generate short image description for image model'), description_agent, pre_process=TextProcessor.text_head, log=True)
task3 = Task(TextTaskInput('Generate cartoon style image'), image_agent, log=True)

# Start SequenceFlow
flow = SequenceFlow([task1, task2, task3], log=True)
final_result = flow.start()
```

## Create Chatbot
... WIP ...


# Connect Your Data 
... WIP ...


# The Repository Setup
1. Initial setup.
```shell
pip install -r requirements.txt
cd intelli
```

2. Rename `.example.env` to `.env` and fill the keys.

3. Run the test cases, examples below.
```shell
# images
python3 -m unittest test/integration/test_remote_image_model.py

# chatbot
python3 -m unittest test/integration/test_chatbot.py

# mistral
python3 -m unittest test/integration/test_mistralai_wrapper.py
```

# Pillars
- **The wrapper layer** provides low-level access to the latest AI models.
- **The controller layer** offers a unified input to any AI model by handling the differences.
- **The function layer** provides abstract functionality that extends based on the app's use cases. 
- **Flows**: create a flow of ai agents working toward user tasks.
