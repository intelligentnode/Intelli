# Intelli
Create chatbots and AI agent work flows. It allows to connect your data with multiple AI models like OpenAI, Gemini, and Mistral through a unified access layer.

<p>
<a href="https://opensource.org/licenses/Apache-2.0" alt="licenses tag">
    <img src="https://img.shields.io/github/license/Barqawiz/IntelliJava?style=flat-square" />
</a>

<a href="https://discord.gg/VYgCh2p3Ww" alt="licenses tag">
    <img src="https://img.shields.io/badge/Discord-Community-light?style=flat-square" />
</a>

![GitHub Stars](https://img.shields.io/github/stars/intelligentnode/Intelli?style=social)

</p>

# Install
```bash
pip install intelli
```

# Code Examples

## Create AI Flows
You can create a flow of tasks executed by different AI models. Here's an example of creating a blog post flow:

```python
from intelli.flow.agents.agent import Agent
from intelli.flow.task import Task
from intelli.flow.sequence_flow import SequenceFlow
from intelli.flow.input.task_input import TextTaskInput
from intelli.flow.processors.basic_processor import TextProcessor

# define agents
blog_agent = Agent(agent_type='text', provider='openai', mission='write blog posts', model_params={'key': YOUR_OPENAI_API_KEY, 'model': 'gpt-4'})
copy_agent = Agent(agent_type='text', provider='gemini', mission='generate description', model_params={'key': YOUR_GEMINI_API_KEY, 'model': 'gemini'})
artist_agent = Agent(agent_type='image', provider='stability', mission='generate image', model_params={'key': YOUR_STABILITY_API_KEY})

# define tasks
task1 = Task(TextTaskInput('blog post about electric cars'), blog_agent, log=True)
task2 = Task(TextTaskInput('Generate short image description for image model'), copy_agent, pre_process=TextProcessor.text_head, log=True)
task3 = Task(TextTaskInput('Generate cartoon style image'), artist_agent, log=True)

# start sequence flow
flow = SequenceFlow([task1, task2, task3], log=True)
final_result = flow.start()
```

## Create Chatbot
Switch between multiple chatbot providers without changing your code.

```python
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput

def call_chatbot(provider, model=None):
    # prepare common input 
    input = ChatModelInput("You are a helpful assistant.", model)
    input.add_user_message("What is the capital of France?")

    # creating chatbot instance
    openai_bot = Chatbot(YOUR_OPENAI_API_KEY, "openai")
    response = openai_bot.chat(input)

# call openai
call_chatbot("openai", "gpt-4")

# call mistralai
call_chatbot("mistral", "mistral-medium")

# call gooogle gemini
call_chatbot("gemini")
```


## Chat With Docs
IntelliPy allows you to chat with your docs using multiple LLMs. To connect your data, visit the [IntelliNode App](https://app.intellinode.ai/), start a project using the Document option, upload your documents or images, and copy the generated One Key. This key will be used to connect the chatbot to your uploaded data.

```python
# creating chatbot with the intellinode one key
bot = Chatbot(YOUR_OPENAI_API_KEY, "openai", {"one_key": YOUR_ONE_KEY})

input = ChatModelInput("You are a helpful assistant.", "gpt-3.5-turbo")
input.add_user_message("What is the procedure for requesting a refund according to the user manual?")

response = bot.chat(input)
```

# Pillars
- **The wrapper layer** provides low-level access to the latest AI models.
- **The controller layer** offers a unified input to any AI model by handling the differences.
- **The function layer** provides abstract functionality that extends based on the app's use cases. 
- **Flows**: create a flow of ai agents working toward user tasks.
