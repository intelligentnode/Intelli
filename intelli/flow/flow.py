import asyncio
import networkx as nx
import os
from functools import partial
import traceback
from intelli.flow.types import AgentTypes, InputTypes, Matcher
from intelli.utils.logging import Logger

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Flow:
    """
    An asynchronous flow orchestrator that executes a DAG of tasks with optimized handling
    of different input/output types across all supported agent types.
    """

    def __init__(self, tasks, map_paths, log=False):
        """
        Initialize the Flow with tasks and their dependencies.

        Args:
            tasks (dict): A dictionary mapping task names to Task objects
            map_paths (dict): A dictionary defining the dependencies between tasks
                where keys are parent task names and values are lists of child task names
            log (bool): Whether to enable logging
        """
        self.tasks = tasks
        self.map_paths = map_paths
        self.graph = nx.DiGraph()
        self.output = {}
        self.logger = Logger(log)
        self.errors = {}
        # Initialize task semaphores before preparing the graph
        self._task_semaphores = {}  # For per-provider rate limiting
        self._prepare_graph()

    def _prepare_graph(self):
        """
        Initialize the graph with tasks as nodes and dependencies as edges.
        Check for cycles in the graph to ensure it's a valid DAG.
        """
        # Initialize the graph with tasks as nodes
        for task_name in self.tasks:
            task = self.tasks[task_name]
            self.graph.add_node(
                task_name, agent_model=task.agent.provider, agent_type=task.agent.type
            )

            # Create provider-specific semaphores for rate limiting
            provider = task.agent.provider
            if provider not in self._task_semaphores:
                limit = 10  # Default limit for providers
                self._task_semaphores[provider] = asyncio.Semaphore(limit)

        # Add edges based on map_paths to define dependencies
        for parent_task, dependencies in self.map_paths.items():
            for child_task in dependencies:
                self.graph.add_edge(parent_task, child_task)

        # Check for cycles in the graph
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(
                "The dependency graph has cycles, please revise map_paths."
            )

    async def _execute_task(self, task_name):
        """
        Execute a task with inputs from its predecessors, handling type compatibility.
        """
        task = self.tasks[task_name]
        self.logger.log(
            f"---- Execute task {task_name} ({task.agent.type}/{task.agent.provider}) ----"
        )

        # Get predecessor outputs
        predecessor_data = self._gather_predecessor_data(task_name)

        # Determine the best input to use based on compatibility
        merged_input, merged_type = self._select_compatible_input(
            task, predecessor_data
        )

        # Execute task with rate limiting for specific providers
        provider = task.agent.provider
        semaphore = self._task_semaphores.get(provider, asyncio.Semaphore(10))

        async with semaphore:
            try:
                loop = asyncio.get_event_loop()
                execute_task = partial(
                    task.execute, merged_input, input_type=merged_type
                )
                # Run the synchronous function in a thread pool
                await loop.run_in_executor(None, execute_task)
                self.logger.log(f"Task {task_name} executed successfully")
            except Exception as e:
                full_stack_trace = traceback.format_exc()
                error_message = f"Error in task '{task_name}': {e}\nFull stack trace:\n{full_stack_trace}"
                self.logger.log(error_message)
                self.errors[task_name] = error_message
                task.output = f"Error executing task: {str(e)}"
                task.output_type = InputTypes.TEXT.value

        # Store output for use by subsequent tasks
        self.output[task_name] = {"output": task.output, "type": task.output_type}
        self.logger.log(f"Task {task_name} output type: {task.output_type}")

    def _gather_predecessor_data(self, task_name):
        """
        Gather outputs and types from predecessor tasks.

        Returns:
            dict: Dictionary with output types as keys and lists of outputs as values
        """
        predecessor_data = {}

        for pred in self.graph.predecessors(task_name):
            if pred in self.output:
                pred_output = self.output[pred]["output"]
                pred_type = self.output[pred]["type"]

                if pred_type not in predecessor_data:
                    predecessor_data[pred_type] = []

                predecessor_data[pred_type].append(
                    {"task": pred, "output": pred_output}
                )
            else:
                self.logger.log(
                    f"Warning: Output for predecessor task '{pred}' not found"
                )

        return predecessor_data

    def _select_compatible_input(self, task, predecessor_data):
        """
        Select the most compatible input for the task from predecessor outputs.
        Enhanced to better handle audio data types.

        Args:
            task: The task that needs input
            predecessor_data: Dictionary of predecessor outputs by type

        Returns:
            tuple: (merged_input, merged_type) to use for task execution
        """
        if not predecessor_data:
            # No predecessor data, use task's own input
            return None, None

        # Determine what input type the current agent expects
        expected_input_type = Matcher.input.get(task.agent.type)
        self.logger.log(f"Task {task.agent.type} expects input type: {expected_input_type}")

        # Check if we have the exact type needed
        if expected_input_type in predecessor_data:
            outputs = predecessor_data[expected_input_type]
            self.logger.log(f"Found matching input type with {len(outputs)} outputs")

            # For text inputs, we can concatenate multiple inputs
            if expected_input_type == InputTypes.TEXT.value and len(outputs) > 1:
                merged_text = " ".join([item["output"] for item in outputs])
                return merged_text, expected_input_type
            # For audio/binary inputs, just use the latest one
            elif expected_input_type in [InputTypes.AUDIO.value, InputTypes.IMAGE.value]:
                latest_output = outputs[-1]["output"]
                if latest_output:
                    self.logger.log(
                        f"Using latest {expected_input_type} data of size: {len(latest_output) if hasattr(latest_output, '__len__') else 'unknown'}")
                else:
                    self.logger.log(f"Warning: Latest {expected_input_type} data is None or empty")
                return latest_output, expected_input_type
            else:
                # For other input types, use the most recent output
                return outputs[-1]["output"], expected_input_type

        # If exact type not available, try to find compatible type
        self.logger.log(f"No exact input type match. Looking for compatible types.")
        for input_type, outputs in predecessor_data.items():
            # Prioritize text as it's most versatile
            if input_type == InputTypes.TEXT.value:
                self.logger.log(f"Found compatible text input with {len(outputs)} outputs")
                if len(outputs) > 1:
                    merged_text = " ".join([item["output"] for item in outputs])
                    return merged_text, input_type
                else:
                    return outputs[0]["output"], input_type

        # If no text available, use the latest output of any type as fallback
        # This may not work but at least provides some input
        last_type = list(predecessor_data.keys())[-1]
        last_output = predecessor_data[last_type][-1]["output"]
        self.logger.log(
            f"Warning: No compatible input type found. Using {last_type} as fallback."
        )
        return last_output, last_type

    async def start(self, max_workers=10):
        """
        Start the flow execution with optimized concurrency.

        Args:
            max_workers (int): Maximum number of concurrent tasks

        Returns:
            dict: Filtered outputs of non-excluded tasks
        """
        self.errors = {}
        self.output = {}

        # Get tasks in topological order (respecting dependencies)
        ordered_tasks = list(nx.topological_sort(self.graph))

        # Group tasks by their "level" in the graph for maximum parallelism
        task_levels = self._group_tasks_by_level()

        # Create global semaphore to control overall concurrency
        global_semaphore = asyncio.Semaphore(max_workers)

        # Execute tasks level by level (tasks in same level can run in parallel)
        for level, tasks_in_level in enumerate(task_levels):
            self.logger.log(f"Executing level {level} with {len(tasks_in_level)} tasks")

            # Create tasks for this level
            level_tasks = []
            for task_name in tasks_in_level:
                # Skip if dependencies had errors
                dependencies_ok = True
                for dep in self.graph.predecessors(task_name):
                    if dep in self.errors:
                        self.logger.log(
                            f"Skipping {task_name} because dependency {dep} had errors"
                        )
                        dependencies_ok = False
                        break

                if dependencies_ok:
                    # Execute task with global concurrency control
                    level_tasks.append(
                        self._execute_task_with_semaphore(task_name, global_semaphore)
                    )

            # Wait for all tasks in this level to complete
            if level_tasks:
                await asyncio.gather(*level_tasks)

        # Filter outputs of excluded tasks
        filtered_output = {
            task_name: {
                "output": self.output[task_name]["output"],
                "type": self.output[task_name]["type"],
            }
            for task_name in ordered_tasks
            if not self.tasks[task_name].exclude and task_name in self.output
        }

        if self.errors:
            self.logger.log(f"Flow completed with {len(self.errors)} errors")
        else:
            self.logger.log("Flow completed successfully")

        return filtered_output

    async def _execute_task_with_semaphore(self, task_name, semaphore):
        """
        Helper method to execute a task with semaphore control.
        """
        async with semaphore:
            await self._execute_task(task_name)

    def _group_tasks_by_level(self):
        """
        Group tasks by their topological "level" in the graph.
        Tasks at the same level can be executed in parallel.

        This implementation uses a breadth-first traversal approach to determine levels.

        Returns:
            list: List of lists, where each inner list contains tasks at the same level
        """
        # Start with sources (nodes with no predecessors)
        sources = [
            node for node in self.graph.nodes() if self.graph.in_degree(node) == 0
        ]

        if not sources:
            # Handle case where graph has no sources (should not happen in a DAG)
            return [[]]

        # Initialize
        visited = set()
        current_level = sources
        levels = [current_level]

        # Breadth-first traversal to assign levels
        while current_level:
            next_level = []
            for node in current_level:
                visited.add(node)
                # Find all successors whose predecessors are all in visited
                for succ in self.graph.successors(node):
                    if succ not in visited and succ not in next_level:
                        # Check if all predecessors of this successor are visited
                        if all(
                            pred in visited for pred in self.graph.predecessors(succ)
                        ):
                            next_level.append(succ)

            if next_level:
                levels.append(next_level)
            current_level = next_level

        return levels

    def generate_graph_img(self, name="graph_img", save_path="."):
        """
        Generate a visualization of the task graph.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise Exception("Install matplotlib to use the visual functionality")

        plt.figure(figsize=(12, 10))

        # Get task levels for layout
        task_levels = self._group_tasks_by_level()

        # Create a position mapping for hierarchical layout
        pos = {}
        for level_idx, level_tasks in enumerate(task_levels):
            for i, task in enumerate(level_tasks):
                # Distribute tasks horizontally within their level
                width = max(len(level) for level in task_levels)
                x = (i - len(level_tasks) / 2) / max(1, width) + 0.5
                pos[task] = (x, 1.0 - level_idx * 0.2)

        # Get node attributes for coloring
        agent_types = nx.get_node_attributes(self.graph, "agent_type")
        agent_models = nx.get_node_attributes(self.graph, "agent_model")

        # Define colors for different agent types
        color_map = {
            AgentTypes.TEXT.value: "skyblue",
            AgentTypes.IMAGE.value: "lightgreen",
            AgentTypes.VISION.value: "salmon",
            AgentTypes.SPEECH.value: "gold",
            AgentTypes.RECOGNITION.value: "orchid",
            AgentTypes.EMBED.value: "lightcoral",
            AgentTypes.SEARCH.value: "lightskyblue",
        }

        # Set colors based on agent type
        node_colors = [
            color_map.get(agent_types.get(node), "gray") for node in self.graph.nodes()
        ]

        # Draw the graph
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=700,
            edge_color="k",
            width=1.5,
            arrowsize=20,
            with_labels=False,
        )

        # Add labels with task and agent info
        for node in self.graph.nodes():
            agent_type = agent_types.get(node, "unknown")
            model = agent_models.get(node, "unknown")

            # Add the node label
            plt.text(
                pos[node][0],
                pos[node][1] - 0.03,
                s=f"{node}\n[{agent_type}:{model}]",
                horizontalalignment="center",
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        # Add a legend for agent types
        handles = []
        labels = []
        for agent_type, color in color_map.items():
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                )
            )
            labels.append(agent_type)

        plt.legend(handles, labels, loc="upper right", title="Agent Types")

        # Save the image
        image_name = name if name.endswith(".png") else f"{name}.png"
        full_path = os.path.join(save_path, image_name)
        plt.savefig(full_path)
        plt.close()

        return full_path
