from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.controller.remote_image_model import RemoteImageModel
from intelli.model.input.image_input import ImageModelInput
from abc import ABC, abstractmethod
from intelli.flow.types import AgentTypes

class BasicAgent(ABC):

    @abstractmethod
    def execute(self, agent_input):
        pass

class Agent(BasicAgent):
    def __init__(self, agent_type, provider, mission, model_params, options=None):
        
        if agent_type not in AgentTypes._value2member_map_:
            raise ValueError("Incorrect agent type. Accepted types in AgentTypes.")
        
        self.type = agent_type
        self.provider = provider
        self.mission = mission
        self.model_params = model_params
        self.options = options


    def execute(self, agent_input):

        # Check the agent type and call the appropriate function
        if self.type == AgentTypes.TEXT.value:
            chatbot = Chatbot(self.model_params['key'], self.provider, self.options)
            chat_input = ChatModelInput(self.mission, model=self.model_params.get('model'))
            chat_input.add_user_message(agent_input)
            result = chatbot.chat(chat_input)[0]
        elif self.type == AgentTypes.IMAGE.value:
            image_model = RemoteImageModel(self.model_params['key'], self.provider)
            image_input = ImageModelInput(prompt=agent_input, model=self.model_params.get('model'))
            result = image_model.generate_images(image_input)
        else:
            raise ValueError(f"Unsupported agent type: {self.type}.")
        
        return result
