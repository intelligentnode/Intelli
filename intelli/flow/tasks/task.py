from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput
from intelli.flow.template.basic_template import TextInputTemplate
from intelli.flow.types import AgentTypes, InputTypes, Matcher
from intelli.utils.logging import Logger


class Task:
    def __init__(
        self,
        task_input,
        agent,
        exclude=False,
        pre_process=None,
        post_process=None,
        template=None,
        model_params={},
        log=False,
        memory_key=None,
    ):
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
        # Store memory key(s) - can be single key or list of keys
        self.memory_key = (
            memory_key
            if isinstance(memory_key, (list, tuple))
            else ([memory_key] if memory_key else None)
        )
        if not template and Matcher.input[agent.type] in [InputTypes.TEXT.value]:
            self.template = TextInputTemplate(self.desc)

    def execute(self, input_data=None, input_type=None, memory=None):
        """
        Execute the task with the given input data, or data from memory if specified.

        Args:
            input_data: The input data to use
            input_type: The type of the input data
            memory: Optional Memory instance to retrieve data from

        Returns:
            The output of the task
        """
        # If memory and memory_key are provided, try to get input from memory
        if memory is not None and self.memory_key:
            memory_data = {}
            # Process all memory keys
            for key in self.memory_key:
                if key and key in memory:
                    value = memory.retrieve(key)
                    if value is not None:
                        self.logger.log(f"- Using data from memory key '{key}'")
                        memory_data[key] = value

            # If we found any memory data
            if memory_data:
                # If only one item, use it directly
                if len(memory_data) == 1:
                    input_data = next(iter(memory_data.values()))
                    self.logger.log(f"- Using single memory value of type: {type(input_data).__name__}")
                else:
                    # For multiple items, convert to string for text-based agents
                    if self.agent.type == AgentTypes.TEXT.value:
                        try:
                            # Create a formatted string representation
                            formatted_data = "Memory data:\n"
                            for k, v in memory_data.items():
                                if isinstance(v, dict) or isinstance(v, list):
                                    import json
                                    v_str = json.dumps(v, default=str)[:100] + "..."
                                else:
                                    v_str = str(v)[:100] + "..."
                                formatted_data += f"- {k}: {v_str}\n"
                            input_data = formatted_data
                            self.logger.log(f"- Formatted multiple memory items ({len(memory_data)}) for text agent")
                        except Exception as e:
                            self.logger.log(f"- Warning: Failed to format memory data: {e}")
                            # Fallback to simple string representation
                            input_data = str(memory_data)
                    else:
                        # For non-text agents, use dictionary as is
                        input_data = memory_data

                # If text agent, force input type to text
                if self.agent.type == AgentTypes.TEXT.value:
                    input_type = InputTypes.TEXT.value
                # Try to determine input type if not provided
                elif input_type is None:
                    if isinstance(input_data, str):
                        input_type = InputTypes.TEXT.value
                    elif isinstance(input_data, bytes) or hasattr(input_data, "read"):
                        # Could be image or audio
                        if self.agent.type in [
                            AgentTypes.VISION.value,
                            AgentTypes.IMAGE.value,
                        ]:
                            input_type = InputTypes.IMAGE.value
                        elif self.agent.type == AgentTypes.RECOGNITION.value:
                            input_type = InputTypes.AUDIO.value

        # logging - Convert non-string data to string for logging
        if input_type in [InputTypes.TEXT.value]:
            if isinstance(input_data, (dict, list)):
                self.logger.log(f"- Inside the task with input data type: {type(input_data).__name__}")
            else:
                # Use str() for non-string data to prevent slicing errors
                log_data = str(input_data) if input_data is not None else "None"
                self.logger.log_head("- Inside the task with input data head: ", log_data)
        elif input_type == InputTypes.IMAGE.value and self.agent.type in [
            AgentTypes.TEXT.value,
            AgentTypes.IMAGE.value,
        ]:
            self.logger.log("- Inside the task. the previous step input not supported")
        elif input_type == InputTypes.IMAGE.value:
            self.logger.log(
                "- Inside the task with previous image, size: ",
                len(input_data) if input_data else 0,
            )
        elif input_type == InputTypes.AUDIO.value:
            self.logger.log(
                "- Inside the task with previous audio, size: ",
                len(input_data) if input_data else 0,
            )

        # Run task pre procesing
        if self.pre_process:
            try:
                processed_data = self.pre_process(input_data)
                if processed_data is not None:
                    input_data = processed_data
                self.logger.log("Pre-processing completed")
            except Exception as e:
                self.logger.log(f"Error in pre-processing: {str(e)}")
                import traceback

                self.logger.log(traceback.format_exc())

        # Apply input template
        if input_data and input_type in [InputTypes.TEXT.value]:
            try:
                # Convert dictionary to string for template application
                if isinstance(input_data, (dict, list)):
                    import json
                    str_input = json.dumps(input_data, default=str, indent=2)
                else:
                    str_input = str(input_data)

                agent_text = self.template.apply_input(str_input)
            except Exception as e:
                self.logger.log(f"Error applying template: {str(e)}")
                # Fallback to direct concatenation
                agent_text = f"{self.desc}\n\n{str(input_data)}"

            # Safe logging that won't try to slice dictionaries
            if isinstance(agent_text, (dict, list)):
                self.logger.log("- Input data with template applied (complex type)")
            else:
                self.logger.log_head("- Input data with template: ", agent_text)
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
            if (
                    len(agent_inputs) == 0
                    or Matcher.output[self.agent.type] == InputTypes.TEXT.value
            ):
                if input_data and input_type == InputTypes.IMAGE.value:
                    agent_input = ImageAgentInput(desc=agent_text, img=input_data)
                    agent_inputs.append(agent_input)

        elif Matcher.input[self.agent.type] == InputTypes.AUDIO.value:
            # Handle audio input specifically for recognition agents
            self.logger.log(f"- Preparing audio input for {self.agent.type} agent")

            if input_data and input_type == InputTypes.AUDIO.value:
                # We have audio data from previous task (e.g., speech output)
                self.logger.log(
                    f"- Using audio data from previous task, size: {len(input_data) if isinstance(input_data, (bytes, bytearray)) else 'unknown'}"
                )
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
                f"- Warning: Unsupported input type {Matcher.input[self.agent.type]} for agent {self.agent.type}"
            )
            agent_input = AgentInput(desc=agent_text)
            agent_inputs.append(agent_input)

        # Check the agent type and call the appropriate function
        combined_results = []
        for current_agent_input in agent_inputs:
            try:
                result = self.agent.execute(
                    current_agent_input, new_params=self.model_params
                )
                if self.agent.type == AgentTypes.SPEECH.value:
                    self.logger.log(
                        f"- Speech output type: {type(result)}, size: {len(result) if isinstance(result, (bytes, bytearray)) else 'unknown'}"
                    )

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
            if result is not None:
                self.logger.log_head("- The task output head: ", str(result))
            else:
                self.logger.log("- The task output is None")
        else:
            result_size = (
                len(result) if result and hasattr(result, "__len__") else "non-iterable"
            )
            self.logger.log("- The task output count: ", result_size)

        if self.post_process:
            try:
                original_result = result
                processed_result = self.post_process(result)
                # Don't accept None results from post-processing
                if processed_result is not None:
                    result = processed_result
                else:
                    self.logger.log(
                        "Warning: Post-processing returned None, using original result"
                    )
                    result = original_result
            except Exception as e:
                self.logger.log(f"Error in post-processing: {str(e)}")
                import traceback

                self.logger.log(traceback.format_exc())

        self.output = result
        return result
