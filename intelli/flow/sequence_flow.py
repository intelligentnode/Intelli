from intelli.utils.logging import Logger
from intelli.flow.types import Matcher, InputTypes


class SequenceFlow:
    """
    A simple sequential flow that executes tasks in the given order,
    passing the output of each task as input to the next task.

    This version handles compatibility between different input/output types
    for all agent types supported in the intelli library.
    """

    def __init__(self, order, log=False):
        """
        Initialize the sequential flow with an ordered list of tasks.

        Args:
            order (list): Ordered list of Task objects to execute
            log (bool): Whether to enable logging
        """
        self.order = order
        self.log = log
        self.logger = Logger(log)

    def start(self):
        """
        Execute tasks in sequence, passing outputs as inputs.

        Returns:
            dict: Dictionary mapping task indices to task outputs
        """
        result = {}
        flow_input = None
        flow_input_type = None

        for index, task in enumerate(self.order, start=1):
            # Log task execution
            self.logger.log_head(f"- Executing task {index}: {task.desc}")
            self.logger.log(f"  Agent type: {task.agent.type}, Provider: {task.agent.provider}")

            # Check input compatibility
            expected_input_type = Matcher.input.get(task.agent.type)
            if flow_input is not None:
                if flow_input_type != expected_input_type:
                    self.logger.log(
                        f"  Note: Previous output type ({flow_input_type}) differs from expected input type ({expected_input_type})")
                    # For text output being fed to non-text input, we might need special handling
                    if flow_input_type == InputTypes.TEXT.value and expected_input_type != InputTypes.TEXT.value:
                        self.logger.log(
                            f"  Warning: Passing text to {expected_input_type} agent may not work as expected")

            # Execute the task with previous output
            task.execute(flow_input, input_type=flow_input_type)

            # Store the result if not excluded
            if not task.exclude:
                result[f'task{index}'] = task.output

            # Log output information
            self.logger.log(f"  Output type: {task.output_type}")

            # Update flow input for next task
            flow_input = task.output
            flow_input_type = task.output_type

        return result
