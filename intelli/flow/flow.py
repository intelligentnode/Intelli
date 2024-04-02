import asyncio
import networkx as nx
import os
from functools import partial
import traceback
from intelli.flow.types import AgentTypes, InputTypes, Matcher
from intelli.utils.logging import Logger

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Flow:
    def __init__(self, tasks, map_paths, log=False):
        self.tasks = tasks
        self.map_paths = map_paths
        self.graph = nx.DiGraph()
        self.output = {}
        self.logger = Logger(log)
        self._prepare_graph()
        self.errors = {}

    def _prepare_graph(self):
        # Initialize the graph with tasks as nodes
        for task_name in self.tasks:
            self.graph.add_node(task_name, agent_model=self.tasks[task_name].agent.provider)

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
        merged_type = next(iter(predecessor_types)) if len(predecessor_types) == 1 else None
        if merged_type and merged_type == InputTypes.TEXT.value:
            merged_input = " ".join(predecessor_outputs)
        elif predecessor_outputs:
            # get one input if not combined strings
            merged_input = predecessor_outputs[0]
        else:
            merged_input = None

        # Execute task with merged input
        loop = asyncio.get_event_loop()
        execute_task = partial(task.execute, merged_input, input_type=merged_type)

        # Run the synchronous function
        await loop.run_in_executor(None, execute_task)

        # Collect outputs and types
        self.output[task_name] = {'output': task.output, 'type': task.output_type}

    async def start(self, max_workers=10):
        self.errors = {}
        ordered_tasks = list(nx.topological_sort(self.graph))
        task_coroutines = {task_name: self._execute_task(task_name) for task_name in ordered_tasks}
        async with asyncio.Semaphore(max_workers):
            for task_name in ordered_tasks:
                try:
                    await task_coroutines[task_name]
                except Exception as e:
                    full_stack_trace = traceback.format_exc()
                    error_message = f"Error in task '{task_name}': {e}\nFull stack trace:\n{full_stack_trace}"
                    print(error_message)
                    self.errors[task_name] = error_message
                    continue

        # Filter the outputs (and types) of excluded tasks
        filtered_output = {
            task_name: {'output': self.output[task_name]['output'], 'type': self.output[task_name]['type']}
            for task_name in ordered_tasks if not self.tasks[task_name].exclude and task_name in self.output
        }

        return filtered_output

    def generate_graph_img(self, name='graph_img', save_path='.', ):

        if not MATPLOTLIB_AVAILABLE:
            raise Exception("Install matplotlib to use the visual functionality")

        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(self.graph)

        nx.draw(self.graph, pos, node_color='skyblue', node_size=700, edge_color='k', with_labels=False)

        labels = nx.get_node_attributes(self.graph, 'agent_model')
        for node, model_name in labels.items():
            predecessors = list(self.graph.predecessors(node))
            successors = list(self.graph.successors(node))

            # control the labels shift based on the edges/nodes
            if predecessors and successors:
                # shift the label down
                verticalalignment = 'top'
                y_offset = -0.06
            elif predecessors:
                # shift the label up
                verticalalignment = 'bottom'
                y_offset = 0.05
            else:
                # shift the label down
                verticalalignment = 'top'
                y_offset = -0.02

            plt.text(pos[node][0], pos[node][1] + y_offset, s=f'{node}\n[{model_name}]',
                     horizontalalignment='center', verticalalignment=verticalalignment)

        image_name = name if name.endswith('.png') else f'{name}.png'

        full_path = os.path.join(save_path, image_name)
        plt.savefig(full_path)
        plt.close()
