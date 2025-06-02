import asyncio
import networkx as nx
import os
from functools import partial
import traceback
from intelli.flow.types import AgentTypes, InputTypes, Matcher
from intelli.flow.utils.flow_helper import FlowHelper
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

    def __init__(
        self,
        tasks,
        map_paths,
        dynamic_connectors=None,
        log=False,
        sleep_time=None,
        memory=None,
        memory_map=None,
        output_memory_map=None,
        auto_save_outputs=False,
        output_dir="./outputs",
        output_file_map=None,
    ):
        """
        Initialize the Flow with tasks, dependencies, and optional memory.
        
        Args:
            tasks (dict): Dictionary mapping task names to Task objects.
                Example: {"task1": task_obj, "task2": task_obj}
                
            map_paths (dict): Static routing dependencies between tasks.
                Dictionary mapping parent task names to lists of child task names.
                Example: {"task1": ["task2", "task3"], "task2": ["task4"]}
                Use empty dict {} if only using dynamic routing.
                
            dynamic_connectors (dict, optional): Dynamic routing rules based on runtime decisions.
                Dictionary mapping task names to DynamicConnector objects.
                Example: {"llm_task": ToolDynamicConnector(destinations={"tool_called": "mcp_task", "no_tool": "direct_task"})}
                Enables conditional routing based on task output analysis.
                
            log (bool, optional): Enable logging for flow execution. Defaults to False.
                When True, logs task execution progress, routing decisions, and debug information.
                
            sleep_time (float, optional): Delay in seconds before executing each task. Defaults to None.
                Useful for rate limiting or debugging. Applied to all tasks uniformly.
                Example: sleep_time=0.5 for 500ms delay between task executions.
                
            memory (Memory, optional): Memory instance for data persistence across tasks. Defaults to None.
                If None, creates a new Memory() instance. Can use DBMemory for database-backed storage.
                Tasks can read from memory using memory_key parameter.
                
            memory_map (dict, optional): Reserved for future use. Currently unused. Defaults to None.
                Placeholder for advanced memory mapping features.
                
            output_memory_map (dict, optional): Mapping of task names to memory keys for storing outputs.
                Dictionary mapping task names to memory keys where their outputs should be stored.
                Example: {"analysis_task": "analysis_result", "summary_task": "final_summary"}
                Enables persistent storage of task outputs for later retrieval.
                
            auto_save_outputs (bool, optional): Automatically save task outputs to files. Defaults to False.
                When True, saves image/audio outputs as files and text outputs as .txt files.
                File saving occurs after each task completion.
                
            output_dir (str, optional): Directory path for auto-saved files. Defaults to "./outputs".
                Directory will be created if it doesn't exist.
                Used only when auto_save_outputs=True.
                
            output_file_map (dict, optional): Custom file names for auto-saved outputs. Defaults to None.
                Dictionary mapping task names to custom file names.
                Example: {"image_task": "generated_art.png", "text_task": "summary.txt"}
                If not specified, uses default naming: "{task_name}_output.{ext}"
                
        Raises:
            ValueError: If the dependency graph has cycles or if dynamic connector destinations are invalid.
            
        Examples:
            Basic flow with static routing:
            ```python
            flow = Flow(
                tasks={"task1": task1_obj, "task2": task2_obj},
                map_paths={"task1": ["task2"]}
            )
            result = await flow.start()
            ```
            
            Flow with dynamic initial input:
            ```python
            flow = Flow(
                tasks={"analyze": analyze_task, "summarize": summary_task},
                map_paths={"analyze": ["summarize"]}
            )
            result = await flow.start(
                initial_input="Analyze this user feedback: Great product but slow delivery",
                initial_input_type="text"
            )
            ```
        """
        self.tasks = tasks
        self.map_paths = map_paths
        self.dynamic_connectors = dynamic_connectors or {}
        self.graph = nx.DiGraph()
        self.output = {}
        self.logger = Logger(log)
        self.errors = {}
        self.sleep_time = sleep_time

        # Auto-save configuration
        self.auto_save_outputs = auto_save_outputs
        self.output_dir = output_dir
        self.output_file_map = output_file_map or {}
        self.saved_files = {}  # Track saved files

        # Initialize memory if provided
        if memory is not None:
            self.memory = memory
        else:
            # Import here to avoid circular imports
            from intelli.flow.store.memory import Memory

            self.memory = Memory()

        # Initialize memory maps
        self.memory_map = memory_map or {}
        self.output_memory_map = output_memory_map or {}

        # Initialize task before preparing the graph
        self._task_semaphores = {}
        self._prepare_graph()

    def _prepare_graph(self):
        """
        Initialize the graph with tasks as nodes and dependencies as edges.
        Also adds dynamic connectors as special edges.
        """
        # Initialize the graph with tasks as nodes
        for task_name in self.tasks:
            task = self.tasks[task_name]
            self.graph.add_node(
                task_name,
                agent_model=task.agent.provider,
                agent_type=task.agent.type,
                node_type="task",
            )

            # Create provider-specific rate limiting
            provider = task.agent.provider
            if provider not in self._task_semaphores:
                limit = 10  # Default limit for providers
                self._task_semaphores[provider] = asyncio.Semaphore(limit)

        # Add edges based on map_paths to define dependencies
        for parent_task, dependencies in self.map_paths.items():
            for child_task in dependencies:
                self.graph.add_edge(parent_task, child_task, edge_type="static")

        # Add dynamic connectors
        for task_name, connector in self.dynamic_connectors.items():
            # Check if task exists
            if task_name not in self.tasks:
                raise ValueError(f"Task '{task_name}' not found for dynamic connector")

            # Add edges for each possible destination
            for dest_key, dest_task in connector.destinations.items():
                if dest_task not in self.tasks:
                    raise ValueError(
                        f"Destination task '{dest_task}' not found for dynamic connector on '{task_name}'"
                    )

                # Add edge with dynamic connector info
                self.graph.add_edge(
                    task_name,
                    dest_task,
                    edge_type="dynamic",
                    dest_key=dest_key,
                    connector_name=connector.name,
                    connector_mode=connector.mode.value,
                )

        # Check for cycles in the graph
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(
                "The dependency graph has cycles, please revise map_paths and dynamic_connectors."
            )

    async def _execute_task(self, task_name):
        """
        Execute a task with inputs from its predecessors or memory, handling type compatibility.
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

        # Execute task with rate limiting
        provider = task.agent.provider
        semaphore = self._task_semaphores.get(provider, asyncio.Semaphore(10))

        async with semaphore:
            try:
                loop = asyncio.get_event_loop()
                execute_task = partial(
                    task.execute, merged_input, input_type=merged_type, memory=self.memory
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

            # Store output in memory if specified in output_memory_map
            if task_name in self.output_memory_map:
                # This happens whether the task succeeded or failed
                memory_key = self.output_memory_map[task_name]
                output_to_store = task.output

                self.logger.log(f"Storing output of task {task_name} in memory with key '{memory_key}'")
                if output_to_store is None:
                    self.logger.log(f"Warning: Task {task_name} output is None")
                    output_to_store = f"No output from task {task_name}"

                # Store in memory
                self.memory.store(memory_key, output_to_store)
                self.logger.log(
                    f"Memory storage confirmed - key '{memory_key}' exists: {self.memory.has_key(memory_key)}")

            # Auto-save outputs if enabled
            if self.auto_save_outputs and task.output is not None:
                self._auto_save_task_output(task_name, task.output, task.output_type)

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
        Enhanced to preserve task context and improve information flow.

        Args:
            task: The task that needs input
            predecessor_data: Dictionary of predecessor outputs by type

        Returns:
            tuple: (merged_input, merged_type) to use for task execution
        """
        if not predecessor_data:
            # Check if this is an initial task and we have initial input
            if hasattr(self, 'initial_input') and self.initial_input is not None:
                self.logger.log(f"Using provided initial input for task {task.agent.type}")
                return self.initial_input, self.initial_input_type
            # No predecessor data and no initial input, use task's own input
            return None, None

        # Determine the input type
        expected_input_type = Matcher.input.get(task.agent.type)
        self.logger.log(
            f"Task {task.agent.type} expects input type: {expected_input_type}"
        )

        # Check if we have the exact type needed
        if expected_input_type in predecessor_data:
            outputs = predecessor_data[expected_input_type]
            self.logger.log(f"Found matching input type with {len(outputs)} outputs")

            # text inputs
            if expected_input_type == InputTypes.TEXT.value and len(outputs) > 1:
                # Check if we need special formatting
                is_integration = False
                if task.agent.mission and (
                    "integrat" in task.agent.mission.lower()
                    or "synthesiz" in task.agent.mission.lower()
                    or "predict" in task.agent.mission.lower()
                ):
                    is_integration = True

                formatted_outputs = []
                for item in outputs:
                    task_name = item.get("task", "unknown")

                    # Apply different formatting for integration tasks
                    if is_integration:
                        formatted_output = f"""
========== {task_name.upper()} OUTPUT ==========
{item["output"]}
========== END OF {task_name.upper()} ==========
"""
                    else:
                        # For other tasks, use simpler formatting
                        formatted_output = f"{item['output']}"

                    formatted_outputs.append(formatted_output)

                # Join outputs with clear separators
                merged_text = "\n\n".join(formatted_outputs)
                return merged_text, expected_input_type

            # For binary inputs, use the latest output
            elif expected_input_type in [
                InputTypes.AUDIO.value,
                InputTypes.IMAGE.value,
            ]:
                latest_output = outputs[-1]["output"]
                if latest_output:
                    self.logger.log(
                        f"Using latest {expected_input_type} data of size: {len(latest_output) if hasattr(latest_output, '__len__') else 'unknown'}"
                    )
                else:
                    self.logger.log(
                        f"Warning: Latest {expected_input_type} data is None or empty"
                    )
                return latest_output, expected_input_type
            else:
                # For other input types, use the most recent output
                return outputs[-1]["output"], expected_input_type

        # If exact type not available, try to find compatible type
        self.logger.log(f"No exact input type match. Looking for compatible types.")
        for input_type, outputs in predecessor_data.items():
            # Prioritize text
            if input_type == InputTypes.TEXT.value:
                self.logger.log(
                    f"Found compatible text input with {len(outputs)} outputs"
                )
                if len(outputs) > 1:
                    # Apply formatting for compatible text types
                    formatted_outputs = []
                    for item in outputs:
                        formatted_outputs.append(item["output"])

                    merged_text = "\n\n".join(formatted_outputs)
                    return merged_text, input_type
                else:
                    return outputs[0]["output"], input_type

        # Use the latest output of any type as fallback
        last_type = list(predecessor_data.keys())[-1]
        last_output = predecessor_data[last_type][-1]["output"]
        self.logger.log(
            f"Warning: No compatible input type found. Using {last_type} as fallback."
        )
        return last_output, last_type

    async def start(self, max_workers=10, initial_input=None, initial_input_type=None):
        """
        Start the flow execution with optimized concurrency and dynamic routing.

        Args:
            max_workers (int): Maximum number of concurrent tasks
            initial_input: Optional input to pass to the first task(s) in the flow
            initial_input_type: Optional input type for the initial input (e.g., 'text', 'image', 'audio')

        Returns:
            dict: Filtered outputs of non-excluded tasks
        """
        self.errors = {}
        self.output = {}
        
        # Store initial input for first tasks
        self.initial_input = initial_input
        self.initial_input_type = initial_input_type or InputTypes.TEXT.value

        # Identify initial tasks (no predecessors)
        initial_tasks = [
            node for node in self.graph.nodes() if self.graph.in_degree(node) == 0
        ]

        # Track tasks
        tasks_to_execute = set(initial_tasks)
        executed_tasks = set()

        # Control overall concurrency
        global_semaphore = asyncio.Semaphore(max_workers)

        # Execute tasks in waves
        while tasks_to_execute:
            # Filter out tasks that have dependencies not yet executed
            ready_tasks = []
            for task_name in tasks_to_execute:
                dependencies_ok = True
                for dep in self.graph.predecessors(task_name):
                    if dep not in executed_tasks:
                        dependencies_ok = False
                        break

                if dependencies_ok:
                    ready_tasks.append(task_name)

            if not ready_tasks:
                # If no tasks are ready (we might have a cycle or invalid routing)
                self.logger.log("No tasks ready to execute, possible cycle detected.")
                break

            # Execute all ready tasks in parallel
            tasks_to_await = []
            for task_name in ready_tasks:
                tasks_to_await.append(
                    self._execute_task_with_semaphore(task_name, global_semaphore)
                )
                tasks_to_execute.remove(task_name)
                executed_tasks.add(task_name)

            if tasks_to_await:
                await asyncio.gather(*tasks_to_await)

            # Check for dynamic connections and add next tasks to execute
            for task_name in executed_tasks:
                if task_name in self.dynamic_connectors and task_name in self.output:
                    connector = self.dynamic_connectors[task_name]
                    output_data = self.output[task_name]["output"]
                    output_type = self.output[task_name]["type"]

                    next_task = connector.get_next_task(output_data, output_type)
                    if (
                        next_task
                        and next_task not in executed_tasks
                        and next_task not in tasks_to_execute
                    ):
                        self.logger.log(
                            f"Dynamic connector routes from {task_name} to {next_task}"
                        )
                        tasks_to_execute.add(next_task)

                # For static connections, add all successors that aren't part of a dynamic connection
                for succ in self.graph.successors(task_name):
                    # Skip if edge is dynamic (we handle those separately)
                    edge_data = self.graph.get_edge_data(task_name, succ)
                    if edge_data.get("edge_type") == "dynamic":
                        continue

                    if succ not in executed_tasks and succ not in tasks_to_execute:
                        tasks_to_execute.add(succ)

        # Filter outputs of excluded tasks
        filtered_output = {
            task_name: {
                "output": self.output[task_name]["output"],
                "type": self.output[task_name]["type"],
            }
            for task_name in executed_tasks
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
            if self.sleep_time is not None and self.sleep_time > 0:
                self.logger.log(
                    f"Sleeping for {self.sleep_time} seconds before executing {task_name}"
                )
                await asyncio.sleep(self.sleep_time)

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
            # No sources case (should not happen in a DAG)
            return [[]]

        visited = set()
        current_level = sources
        levels = [current_level]

        # Breadth-first traversal to assign levels
        while current_level:
            next_level = []
            for node in current_level:
                visited.add(node)
                # Find all visited successors
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

    def generate_graph_img(self, name="graph_img", save_path=".", show_legend=True):
        """
        Generate a visualization of the task graph, including dynamic connections.

        Args:
            name (str): Base name for the output image file
            save_path (str): Directory path where the image will be saved
            show_legend (bool): Whether to display the legend in the graph

        Returns:
            str: Full path to the saved image file
        """
        if not MATPLOTLIB_AVAILABLE:
            raise Exception("Install matplotlib to use the visual functionality")

        plt.figure(figsize=(12, 10))

        # Get task levels for layout
        task_levels = self._group_tasks_by_level()

        # Create a position mapping
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
            AgentTypes.MCP.value: "mediumpurple",
        }

        node_colors = [
            color_map.get(agent_types.get(node), "gray") for node in self.graph.nodes()
        ]

        # Split edges into static and dynamic
        static_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == "static"
        ]
        dynamic_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == "dynamic"
        ]

        # Draw static edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=static_edges,
            edge_color="black",
            width=1.5,
            arrowsize=20,
        )

        # Draw dynamic edges
        if dynamic_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=dynamic_edges,
                edge_color="red",
                width=1.5,
                arrowsize=20,
                style="dashed",
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=700,
            node_color=node_colors,
        )

        # Add labels with task and agent info
        for node in self.graph.nodes():
            agent_type = agent_types.get(node, "unknown")
            model = agent_models.get(node, "unknown")

            # Add the node label
            plt.text(
                pos[node][0],
                pos[node][1] - 0.02,
                s=f"{node}\n[{agent_type}:{model}]",
                horizontalalignment="center",
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        # Add a legend for agent types
        if show_legend:
            handles = []
            labels = []
            
            # Get all unique agent types actually used in the graph
            used_agent_types = set(agent_types.values())
            
            # Add legend entries
            for agent_type in used_agent_types:
                if agent_type in color_map:
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color_map[agent_type],
                            markersize=10,
                        )
                    )
                    labels.append(agent_type)

            # Add legend for edge types
            handles.append(plt.Line2D([0], [0], color="black", lw=2))
            labels.append("Static Connection")

            if dynamic_edges:
                handles.append(
                    plt.Line2D([0], [0], color="red", lw=2, linestyle="dashed")
                )
                labels.append("Dynamic Connection")

            plt.legend(handles, labels, loc="upper right", title="Legend")

        # Add labels to dynamic edges
        edge_labels = {}
        for u, v, d in self.graph.edges(data=True):
            if d.get("edge_type") == "dynamic":
                dest_key = d.get("dest_key", "")
                edge_labels[(u, v)] = f"{dest_key}"

        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_color="red"
        )

        # Save the image
        image_name = name if name.endswith(".png") else f"{name}.png"
        full_path = os.path.join(save_path, image_name)
        plt.savefig(full_path)
        plt.close()

        return full_path

    def _auto_save_task_output(self, task_name, output, output_type):
        """
        Automatically save task output to file if it's an image or audio type.
        
        Args:
            task_name (str): Name of the task
            output: The output data to save
            output_type (str): Type of output (image, audio, text)
        """
        try:
            # Ensure output directory exists
            FlowHelper.ensure_directory(self.output_dir)
            
            # Determine file path
            if task_name in self.output_file_map:
                # Use custom file path if specified
                file_path = self.output_file_map[task_name]
                # Make it relative to output_dir if it's not an absolute path
                if not os.path.isabs(file_path):
                    file_path = os.path.join(self.output_dir, file_path)
            else:
                # Generate default file path based on output type
                if output_type == InputTypes.IMAGE.value:
                    file_path = os.path.join(self.output_dir, f"{task_name}_output.png")
                elif output_type == InputTypes.AUDIO.value:
                    file_path = os.path.join(self.output_dir, f"{task_name}_output.mp3")
                elif output_type == InputTypes.TEXT.value:
                    file_path = os.path.join(self.output_dir, f"{task_name}_output.txt")
                else:
                    # Skip saving for unknown types
                    self.logger.log(f"Skipping auto-save for unknown output type: {output_type}")
                    return
            
            # Save the output based on type
            saved_path = None
            file_size = 0
            
            if output_type == InputTypes.IMAGE.value:
                saved_path, file_size = FlowHelper.save_image_output(output, file_path)
                if saved_path:
                    self.logger.log(f"🖼️ Auto-saved image from task '{task_name}' to {saved_path}, size: {file_size} bytes")
                    
            elif output_type == InputTypes.AUDIO.value:
                saved_path, file_size = FlowHelper.save_audio_output(output, file_path)
                if saved_path:
                    self.logger.log(f"🔊 Auto-saved audio from task '{task_name}' to {saved_path}, size: {file_size} bytes")
                    
            elif output_type == InputTypes.TEXT.value:
                saved_path = FlowHelper.save_text_output(output, file_path)
                if saved_path:
                    self.logger.log(f"📄 Auto-saved text from task '{task_name}' to {saved_path}")
            
            # Track saved files
            if saved_path:
                self.saved_files[task_name] = {
                    "path": saved_path,
                    "type": output_type,
                    "size": file_size if output_type in [InputTypes.IMAGE.value, InputTypes.AUDIO.value] else len(str(output))
                }
                
        except Exception as e:
            self.logger.log(f"Error auto-saving output for task '{task_name}': {e}")
            import traceback
            self.logger.log(traceback.format_exc())

    def get_saved_files(self):
        """
        Get information about files that were automatically saved during flow execution.
        
        Returns:
            dict: Dictionary mapping task names to their saved file information
        """
        return self.saved_files.copy()

    def get_flow_summary(self):
        """
        Get a comprehensive summary of the flow execution including outputs and saved files.
        
        Returns:
            dict: Summary containing outputs, saved files, and errors
        """
        # Filter outputs of excluded tasks
        filtered_output = {
            task_name: {
                "output": self.output[task_name]["output"],
                "type": self.output[task_name]["type"],
            }
            for task_name in self.output.keys()
            if not self.tasks[task_name].exclude and task_name in self.output
        }
        
        return {
            "outputs": filtered_output,
            "saved_files": self.saved_files.copy(),
            "errors": self.errors.copy(),
            "auto_save_enabled": self.auto_save_outputs,
            "output_directory": self.output_dir
        }
