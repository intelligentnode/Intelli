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
        predecessor_outputs = []
        predecessor_types = set()

        # Gather inputs and types from previous tasks based on the graph
        for pred in self.graph.predecessors(task_name):
            if pred in self.output:
                predecessor_outputs.append(self.output[pred]['output'])
                predecessor_types.add(self.output[pred]['type'])
            else:
                print(f"Warning: Output for predecessor task '{pred}' not found. Skipping...")

        self.logger.log(f'The number of combined inputs for task {task_name} is {len(predecessor_outputs)}')
        merged_input = " ".join(predecessor_outputs)
        merged_type = next(iter(predecessor_types)) if len(predecessor_types) == 1 else None

        # Execute task with merged input
        loop = asyncio.get_event_loop()
        execute_task = partial(task.execute, merged_input, input_type=merged_type)
        
        # Run the synchronous function
        await loop.run_in_executor(None, execute_task)

        # Collect outputs and types
        self.output[task_name] = {'output': task.output, 'type': task.output_type}

    async def start(self, max_workers=10):
        ordered_tasks = list(nx.topological_sort(self.graph))
        task_coroutines = {task_name: self._execute_task(task_name) for task_name in ordered_tasks}
        async with asyncio.Semaphore(max_workers):
            for task_name in ordered_tasks:
                await task_coroutines[task_name]

        # Filter the outputs (and types) of excluded tasks
        filtered_output = {
            task_name: { 'output': self.output[task_name]['output'], 'type': self.output[task_name]['type'] }
            for task_name in ordered_tasks if not self.tasks[task_name].exclude
        }

        return filtered_output
