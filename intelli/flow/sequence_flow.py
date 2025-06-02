from intelli.utils.logging import Logger
from intelli.flow.types import Matcher, InputTypes


class SequenceFlow:
    """
    A simple sequential flow that executes tasks in the given order,
    passing the output of each task as input to the next task.

    This version handles compatibility between different input/output types
    for all agent types supported in the intelli library.
    """

    def __init__(self, order, log=False, memory=None, output_memory_map=None):
        """Initialize the sequential flow with ordered tasks and optional memory."""
        self.order = order
        self.log = log
        self.logger = Logger(log)

        # Memory handling
        if memory is not None:
            self.memory = memory
        else:
            # Import here to avoid circular imports
            from intelli.flow.store.memory import Memory

            self.memory = Memory()

        self.output_memory_map = output_memory_map or {}

    def start(self, initial_input=None, initial_input_type=None):
        """
        Execute tasks in sequence, passing outputs as inputs.

        Args:
            initial_input: Optional input to pass to the first task
            initial_input_type: Optional input type for the initial input (defaults to 'text')

        Returns:
            dict: Dictionary mapping task indices to task outputs
        """
        result = {}
        flow_input = initial_input if initial_input is not None else None
        flow_input_type = initial_input_type if initial_input_type is not None else InputTypes.TEXT.value

        for index, task in enumerate(self.order, start=1):
            # Log task execution
            self.logger.log_head(f"- Executing task {index}: {task.desc}")
            self.logger.log(
                f"  Agent type: {task.agent.type}, Provider: {task.agent.provider}"
            )

            # Use enhanced compatibility handling
            compatible_input, compatible_input_type = self._select_sequential_compatible_input(
                task, flow_input, flow_input_type
            )

            # Execute the task with compatible input and memory
            task.execute(compatible_input, input_type=compatible_input_type, memory=self.memory)

            # Store output in memory if specified in output_memory_map
            task_index_str = f"task{index}"
            if task_index_str in self.output_memory_map:
                memory_key = self.output_memory_map[task_index_str]
                self.memory.store(memory_key, task.output)
                self.logger.log(
                    f"Stored output of task {index} in memory with key '{memory_key}'"
                )

            # Store the result if not excluded
            if not task.exclude:
                result[task_index_str] = task.output

            # Log output information
            self.logger.log(f"  Output type: {task.output_type}")

            # Update flow input for next task
            flow_input = task.output
            flow_input_type = task.output_type

        return result

    def _select_sequential_compatible_input(self, task, flow_input, flow_input_type):
        """
        Select compatible input for sequential task execution, handling type mismatches.
        Simplified version of Flow's _select_compatible_input for sequential workflows.
        
        Args:
            task: The task that needs input
            flow_input: The output from the previous task
            flow_input_type: The type of the previous output
            
        Returns:
            tuple: (compatible_input, compatible_type) to use for task execution
        """
        if flow_input is None:
            return None, None
            
        # Get expected input type for this task
        expected_input_type = Matcher.input.get(task.agent.type)
        self.logger.log(f"Task {task.agent.type} expects: {expected_input_type}, received: {flow_input_type}")
        
        # If types match exactly, use as-is
        if flow_input_type == expected_input_type:
            self.logger.log("Input types match - using direct input")
            return flow_input, flow_input_type
            
        # Try compatibility fallbacks
        self.logger.log(f"Type mismatch - attempting compatibility handling")
        
        # Text is most compatible - many agents can work with text descriptions
        if flow_input_type == InputTypes.TEXT.value:
            self.logger.log("Previous output is text - using as compatible input")
            return flow_input, flow_input_type
            
        # If we have binary data (image/audio) going to text agent, provide description
        if expected_input_type == InputTypes.TEXT.value and flow_input_type in [InputTypes.IMAGE.value, InputTypes.AUDIO.value]:
            description = f"[{flow_input_type} data from previous task - size: {len(flow_input) if hasattr(flow_input, '__len__') else 'unknown'}]"
            self.logger.log(f"Converting {flow_input_type} to text description for compatibility")
            return description, InputTypes.TEXT.value
            
        # For other mismatches, use the data as-is but log warning
        self.logger.log(f"Warning: Using {flow_input_type} data for {expected_input_type} agent - may cause issues")
        return flow_input, flow_input_type
