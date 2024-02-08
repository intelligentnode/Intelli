import asyncio
import networkx as nx
from intelli.utils.logging import Logger
from functools import partial


class Flow:
    def __init__(self, tasks, map_paths, log=False):
        self.tasks = tasks
        self.map_paths = map_paths
        self.graph = nx.DiGraph()
        self.output = {}
        self.logger = Logger(log)
        self._prepare_graph()

    def _prepare_graph(self):
        # Initialize the graph with tasks as nodes
        for task_name in self.tasks:
            self.graph.add_node(task_name)
        
        # Add edges based on map_paths to define dependencies
        for parent_task, dependencies in self.map_paths.items():
            for child_task in dependencies:
                self.graph.add_edge(parent_task, child_task)
        
        # Check for cycles in the graph
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The dependency graph has cycles, please revise map_paths.")
    
    async def _execute_task(self, task_name):
        self.logger.log(f'---- execute task {task_name} ---- ')

        task = self.tasks[task_name]
        input_texts = []

        # Gather inputs from previous tasks based on the graph
        input_type = None
        for pred in self.graph.predecessors(task_name):
            if pred in self.output:
                input_texts.append(self.output[pred])
            else:
                print(f"Warning: Output for predecessor task '{pred}' not found. Skipping...")

        # Combine the inputs for tasks having multiple dependencies
        self.logger.log(f'The number of combined inputs for task {task_name} is {len(input_texts)}')
        merged_input = " ".join(input_texts)
        
        # If execute method of task is synchronous, wrap it for async execution
        loop = asyncio.get_event_loop()
        # Utilize functools.partial to prepare the function with arguments if necessary
        execute_task = partial(task.execute, merged_input)
        
        # Run the synchronous function
        result = await loop.run_in_executor(None, execute_task)

        # Collect outputs
        self.output[task_name] = task.output

    async def start(self, max_workers=10):

        # Topological sorting to order tasks based on dependencies
        ordered_tasks = list(nx.topological_sort(self.graph))
        task_coroutines = {task_name: self._execute_task(task_name) for task_name in ordered_tasks}

        async with asyncio.Semaphore(max_workers):
            for task_name in ordered_tasks:
                await task_coroutines[task_name]

        # Filter the outputs of excluded tasks
        filtered_output = {task_name: self.output[task_name] for task_name in ordered_tasks if not self.tasks[task_name].exclude}

        # Returning filtered output
        return filtered_output
