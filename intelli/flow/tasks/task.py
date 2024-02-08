from intelli.flow.template.basic_template import TextInputTemplate
from intelli.flow.types import AgentTypes, InputTypes
from intelli.utils.logging import Logger
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput


class Task:
    def __init__(self, task_input, agent, exclude=False, pre_process=None,
                 post_process=None, template=None, log=False):
        self.desc = task_input.desc
        self.agent = agent
        self.pre_process = pre_process
        self.post_process = post_process
        self.exclude = exclude
        self.output = None
        self.output_type = agent.type
        self.template = template
        self.logger = Logger(log)
        if not template and agent.type in [AgentTypes.TEXT.value, AgentTypes.IMAGE.value]:
            self.template = TextInputTemplate(self.desc)

    def execute(self, input_data=None, input_type=None):
        
        # logging
        if input_type in [InputTypes.TEXT.value, InputTypes.IMAGE.value]:
            self.logger.log_head('- Inside the task with input data head: ', input_data)
        elif input_type == InputTypes.IMAGE.value and self.agent.type in [AgentTypes.TEXT.value,
                                                                          AgentTypes.IMAGE.value]:
            self.logger.log_head('- Inside the task. the previous step input not supported')

        # Run task pre procesing
        if self.pre_process:
            input_data = self.pre_process(input_data)

        # Apply template
        if input_data and input_type in [InputTypes.TEXT.value, InputTypes.IMAGE.value]:
            agent_text = self.template.apply_input(input_data)
            # log
            self.logger.log_head('- Input data with template: ', agent_text)
        else:
            agent_text = self.desc

        # Check the agent type and call the appropriate function
        result = self.agent.execute(TextAgentInput(agent_text))

        # log
        if self.agent.type in [AgentTypes.TEXT.value]:
            self.logger.log_head('- The task output head: ', result)
        else:
            self.logger.log('- The task output count: ', len(result))

        if self.post_process:
            result = self.post_process(result)

        self.output = result
