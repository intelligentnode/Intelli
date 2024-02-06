from intelli.flow.template.basic_template import TextInputTemplate
from intelli.flow.types import AgentTypes, InputTypes
from intelli.utils.logging import Logger

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
            self.logger.log('- Inside the task with input data head: ', input_data)
        elif input_type == InputTypes.IMAGE.value and self.agent.type in [AgentTypes.TEXT.value, AgentTypes.IMAGE.value]:
            self.logger.log('- Inside the task. the previous step input not supported')

        # Run task pre procesing
        if self.pre_process:
            input_data = self.pre_process(input_data)

        # Apply template
        if input_data and input_type in [InputTypes.TEXT.value, InputTypes.IMAGE.value]:
            agent_input = self.template.apply_input(input_data)
            # log
            self.logger.log('- Input data with template: ', agent_input)
        else:
            agent_input = self.desc

        # Check the agent type and call the appropriate function
        result = self.agent.execute(agent_input)
        # log
        self.logger.log('- The task output head: ', result)

        if self.post_process:
            result = self.post_process(result)
            
        self.output = result
