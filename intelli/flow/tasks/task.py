from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput
from intelli.flow.template.basic_template import TextInputTemplate
from intelli.flow.types import AgentTypes, InputTypes, Matcher
from intelli.utils.logging import Logger


class Task:
    def __init__(self, task_input, agent, exclude=False, pre_process=None,
                 post_process=None, template=None, model_params={}, log=False):
        self.task_input = task_input
        self.desc = task_input.desc
        self.agent = agent
        self.pre_process = pre_process
        self.post_process = post_process
        self.exclude = exclude
        self.output = None
        self.output_type = Matcher.output[agent.type]
        self.template = template
        self.logger = Logger(log)
        self.model_params = model_params
        if not template and Matcher.input[agent.type] in [InputTypes.TEXT.value]:
            self.template = TextInputTemplate(self.desc)

    def execute(self, input_data=None, input_type=None):

        # logging
        if input_type in [InputTypes.TEXT.value]:
            self.logger.log_head('- Inside the task with input data head: ', input_data)
        elif input_type == InputTypes.IMAGE.value and self.agent.type in [AgentTypes.TEXT.value,
                                                                          AgentTypes.IMAGE.value]:
            self.logger.log('- Inside the task. the previous step input not supported')
        elif input_type == InputTypes.IMAGE.value:
            self.logger.log('- Inside the task with previous image, size: ', len(input_data))

        # Run task pre procesing
        if self.pre_process:
            input_data = self.pre_process(input_data)

        # Apply input template
        if input_data and input_type in [InputTypes.TEXT.value]:
            agent_text = self.template.apply_input(input_data)
            # log
            self.logger.log_head('- Input data with template: ', agent_text)
        else:
            agent_text = self.desc

        # Prepare the input
        agent_inputs = []
        if Matcher.input[self.agent.type] == InputTypes.IMAGE.value:

            if self.task_input.img:
                agent_input = ImageAgentInput(desc=agent_text, img=self.task_input.img)
                agent_inputs.append(agent_input)

            # add previous output as input, in case of second input for image, only if the output supported
            if len(agent_inputs) == 0 or Matcher.output[self.agent.type] == InputTypes.TEXT.value:
                if input_data and input_type == InputTypes.IMAGE.value:
                    agent_input = ImageAgentInput(desc=agent_text, img=input_data)
                    agent_inputs.append(agent_input)

        elif Matcher.input[self.agent.type] == AgentTypes.TEXT.value:
            agent_input = TextAgentInput(agent_text)
            agent_inputs.append(agent_input)

        # Check the agent type and call the appropriate function
        combined_results = []
        for current_agent_input in agent_inputs:

            result = self.agent.execute(current_agent_input, new_params=self.model_params)

            if isinstance(result, list):
                combined_results.extend(result)
            else:
                combined_results.append(str(result))

        if Matcher.output[self.agent.type] == InputTypes.TEXT.value:
            result = " ".join(combined_results)
        else:
            # get first result only for none text outputs
            result = combined_results[0]

        # log
        if self.agent.type in [AgentTypes.TEXT.value]:
            self.logger.log_head('- The task output head: ', result)
        else:
            self.logger.log('- The task output count: ', len(result))

        if self.post_process:
            result = self.post_process(result)

        self.output = result
