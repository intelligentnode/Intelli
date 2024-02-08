import asyncio
from collections import defaultdict
from operator import itemgetter
import concurrent

from intelli.flow.sequence_flow import SequenceFlow
from intelli.utils.logging import Logger


class ValidationsError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Flow:
    def __init__(self, agents, map_paths, log=False):
        self.agents = agents
        self.map = map_paths
        self.log = log
        self.logger = Logger(log)

        # perform the validations
        self._validate_agents()
        self._validate_map()

    def _validate_agents(self):
        if len(self.agents) > 10:
            raise ValidationsError("Can't have more than 10 agents")

        if len(set(self.agents)) != len(self.agents):
            raise ValidationsError("Agents should be unique")

    def _validate_map(self):
        graph = defaultdict(list)
        for parent, child in self.map.items():
            graph[parent] += (child,)
            if self._detect_cycle(graph):
                raise ValidationsError("Cycle detected in map paths")

    def _detect_cycle(self, graph):
        """Detect cycle in a graph using DFS"""
        visited_set = set()
        recursive_stack = set()

        for node in graph.keys():
            if self._detect_cycle_util(node, visited_set, recursive_stack, graph):
                return True
        return False

    def _detect_cycle_util(self, node, visited, recursive, graph):
        """Utility function for detecting cycle"""
        visited.add(node)
        recursive.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if self._detect_cycle_util(neighbor, visited, recursive, graph):
                    return True
            elif neighbor in recursive:
                return True

    async def start(self, max_workers=10):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        loop = asyncio.get_event_loop()

        start, paths = next(iter(self.map.items()))
        tasks = []

        for path in paths:
            agent_sequence = list(itemgetter(*path)(self.agents))
            tasks.append(loop.run_in_executor(executor, SequenceFlow(agent_sequence).start))

        done, pending = await asyncio.wait(tasks)
        return [result.result() for result in done]
