from abc import ABC, abstractmethod

from intelli.controller.remote_image_model import RemoteImageModel
from intelli.controller.remote_vision_model import RemoteVisionModel
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput
from intelli.flow.types import AgentTypes
from intelli.function.chatbot import Chatbot
from intelli.model.input.chatbot_input import ChatModelInput
from intelli.model.input.image_input import ImageModelInput
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

    def execute(self, agent_input: AgentInput, new_params = {}):
        
        custom_params = dict(self.model_params)
        if new_params is not None and isinstance(new_params, dict) and new_params and self.model_params is not None:
            custom_params.update(new_params)
        
        # Check the agent type and call the appropriate function
        if self.type == AgentTypes.TEXT.value:

            f_params = {key: value for key, value in custom_params.items() if hasattr(ChatModelInput("test", None), key)}

            chat_input = ChatModelInput(self.mission, **f_params)

            chatbot = Chatbot(custom_params['key'], self.provider, self.options)
            chat_input.add_user_message(agent_input.desc)
            result = chatbot.chat(chat_input)[0]
        elif self.type == AgentTypes.IMAGE.value:

            f_params = {key: value for key, value in custom_params.items() if hasattr(ImageModelInput("test"), key)}

            image_input = ImageModelInput(prompt=self.mission + ": " + agent_input.desc, **f_params)

            image_model = RemoteImageModel(custom_params['key'], self.provider)
            result = image_model.generate_images(image_input)[0]
        elif self.type == AgentTypes.VISION.value:
            vision_input = VisionModelInput(content=self.mission + ": " + agent_input.desc,
                                            image_data=agent_input.img,
                                            extension=custom_params.get('extension', 'png'),
                                            model=custom_params['model'])

            vision_model = RemoteVisionModel(custom_params['key'], self.provider)
            result = vision_model.image_to_text(vision_input)
        else:
            raise ValueError(f"Unsupported agent type: {self.type}.")

        return result
