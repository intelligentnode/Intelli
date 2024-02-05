from function.chatbot import Chatbot
from model.input.chatbot_input import ChatModelInput
from flow.templates.basic_template import TextInputTemplate
from controller.remote_image_model import RemoteImageModel
from model.input.image_input import ImageModelInput

class Task:
    def __init__(self, desc, agent, exclude=False, pre_process=None, 
                 post_process=None, template=None, log=False):
        self.desc = desc
        self.agent = agent
        self.pre_process = pre_process
        self.post_process = post_process
        self.exclude = exclude
        self.output = None
        self.output_type = agent.type
        self.template = template
        self.log = log
        if not template and agent.type in ['text', 'image']:
            self.template = TextInputTemplate(self.desc)
        self.log_head_size = 80

    def execute(self, input_data=None, input_type=None):
        
        if self.log:
            if input_type == 'text':
                print('- Inside the task with input data head: ', input_data[self.log_head_size])
            elif input_type == 'image' and self.agent.type in ['text', 'image']:
                print('- Inside the task. the previous step input not supported')

        # Run task pre procesing
        if self.pre_process:
            input_data = self.pre_process(input_data)

        # Apply template
        if input_data and input_type == 'text':
            user_message = self.template.apply_input(input_data)
            if self.log:
                print('- Input data with template: ', user_message[:self.log_head_size])
        else:
            user_message = self.desc

        # Check the agent type and call the appropriate function
        if self.agent.type == 'text':
            chatbot = Chatbot(self.agent.model_params['key'], self.agent.provider, self.agent.options)
            chat_input = ChatModelInput(self.agent.mission, model=self.agent.model_params.get('model'))
            chat_input.add_user_message(user_message)
            result = chatbot.chat(chat_input)[0]

            if self.log:
                print('- The task output head: ', result[:20])
        elif self.agent.type == 'image':
            image_model = RemoteImageModel(self.agent.model_params['key'], self.agent.provider)
            image_input = ImageModelInput(prompt=user_message, model=self.agent.model_params.get('model'))
            result = image_model.generate_images(image_input)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent.type}")

        if self.post_process:
            result = self.post_process(result)
            
        self.output = result
