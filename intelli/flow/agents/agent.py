from abc import ABC, abstractmethod

from intelli.controller.remote_image_model import RemoteImageModel
from intelli.flow.types import AgentTypes
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.model.input.image_input import ImageModelInput
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput
from intelli.controller.remote_vision_model import RemoteVisionModel
from intelli.model.input.vision_input import VisionModelInput


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

    def execute(self, agent_input: AgentInput):

        # Check the agent type and call the appropriate function
        if self.type == AgentTypes.TEXT.value:
            chat_input = ChatModelInput(self.mission, model=self.model_params.get('model'))
            
            chatbot = Chatbot(self.model_params['key'], self.provider, self.options)
            chat_input.add_user_message(agent_input.desc)
            result = chatbot.chat(chat_input)[0]
        elif self.type == AgentTypes.IMAGE.value:
            image_input = ImageModelInput(prompt=self.mission + ": " + agent_input.desc, model=self.model_params.get('model'))
            
            image_model = RemoteImageModel(self.model_params['key'], self.provider)
            result = image_model.generate_images(image_input)[0]
        elif self.type == AgentTypes.VISION.value:
            vision_input = VisionModelInput(content=self.mission + ": " + agent_input.desc, 
                                            image_data=agent_input.img, 
                                            extension=self.model_params.get('extension', 'png'),
                                            model=self.model_params['model'])
            
            vision_model = RemoteVisionModel(self.model_params['key'], self.provider)
            result = vision_model.image_to_text(vision_input)
        else:
            raise ValueError(f"Unsupported agent type: {self.type}.")

        return result
