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
            self.logger.log('- Inside the task with previous image, size: ', len(input_data) if input_data else 0)
        elif input_type == InputTypes.AUDIO.value:
            self.logger.log('- Inside the task with previous audio, size: ', len(input_data) if input_data else 0)

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
            # Handle image input
            if self.task_input.img:
                agent_input = ImageAgentInput(desc=agent_text, img=self.task_input.img)
                agent_inputs.append(agent_input)

            # Add previous output as input if it's an image
            if len(agent_inputs) == 0 or Matcher.output[self.agent.type] == InputTypes.TEXT.value:
                if input_data and input_type == InputTypes.IMAGE.value:
                    agent_input = ImageAgentInput(desc=agent_text, img=input_data)
                    agent_inputs.append(agent_input)

        elif Matcher.input[self.agent.type] == InputTypes.AUDIO.value:
            # Handle audio input specifically for recognition agents
            self.logger.log(f"- Preparing audio input for {self.agent.type} agent")

            if input_data and input_type == InputTypes.AUDIO.value:
                # We have audio data from previous task (e.g., speech output)
                self.logger.log(
                    f"- Using audio data from previous task, size: {len(input_data) if isinstance(input_data, (bytes, bytearray)) else 'unknown'}")
                agent_input = AgentInput(desc=agent_text, audio=input_data)
                agent_inputs.append(agent_input)
            elif self.task_input.audio:
                # Use audio from original task input if available
                self.logger.log("- Using audio from task_input")
                agent_input = AgentInput(desc=agent_text, audio=self.task_input.audio)
                agent_inputs.append(agent_input)
            else:
                # Create input even without audio data to avoid errors
                self.logger.log("- Warning: No audio data for recognition task")
                agent_input = AgentInput(desc=agent_text)
                agent_inputs.append(agent_input)

        elif Matcher.input[self.agent.type] == InputTypes.TEXT.value:
            # Handle text input
            agent_input = TextAgentInput(agent_text)
            agent_inputs.append(agent_input)

        else:
            self.logger.log(
                f"- Warning: Unsupported input type {Matcher.input[self.agent.type]} for agent {self.agent.type}")
            agent_input = AgentInput(desc=agent_text)
            agent_inputs.append(agent_input)

        # Check the agent type and call the appropriate function
        combined_results = []
        for current_agent_input in agent_inputs:
            try:
                result = self.agent.execute(current_agent_input, new_params=self.model_params)

                # Add debug information for speech output
                if self.agent.type == AgentTypes.SPEECH.value:
                    self.logger.log(
                        f"- Speech output type: {type(result)}, size: {len(result) if isinstance(result, (bytes, bytearray)) else 'unknown'}")

                if isinstance(result, list):
                    combined_results.extend(result)
                else:
                    combined_results.append(result)
            except Exception as e:
                error_message = f"Error executing agent: {str(e)}"
                self.logger.log(error_message)
                import traceback
                self.logger.log(traceback.format_exc())
                combined_results.append(f"Error: {str(e)}")

        # Process results
        if not combined_results:
            result = None
        elif Matcher.output[self.agent.type] == InputTypes.TEXT.value:
            # For text output, join all results
            result = " ".join([str(r) for r in combined_results if r is not None])
        else:
            # For non-text outputs (audio, image), use the first result
            result = combined_results[0]

        # Additional debug log for speech agent
        if self.agent.type == AgentTypes.SPEECH.value:
            self.logger.log(f"- Final speech result type: {type(result)}")

        # Log
        if self.agent.type in [AgentTypes.TEXT.value]:
            self.logger.log_head('- The task output head: ', result)
        else:
            result_size = len(result) if result and hasattr(result, '__len__') else 'non-iterable'
            self.logger.log('- The task output count: ', result_size)

        if self.post_process:
            result = self.post_process(result)

        self.output = result
