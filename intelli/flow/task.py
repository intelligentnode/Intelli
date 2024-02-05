from flow.template import Template

class Task:
    def __init__(self, desc, agent, exclude=False, template=None, pre_process=None):
        self.desc = desc
        self.agent = agent
        self.pre_process = pre_process
        self.exclude = exclude
        self.template = template
        self.output = None

    def execute(self, input_data=None):
        # TODO: Implement dynamic method call based on the agent's type and provider
        # For this example, we'll manually handle based on the provided test cases
        print('execute the task with input data: ', input_data)
        pass