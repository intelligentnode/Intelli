class SequenceFlow:
    def __init__(self, order, log=False):
        self.order = order
        self.log = log

    def start(self, initial_input=None):
        result = {}
        current_input = initial_input
        for task in self.order:
            if self.log:
                print(f"Executing task: {task.desc}")
            task.execute(current_input)
            
            if not task.exclude:
                result[task.desc] = task.output
            
            current_input = task.output
        
        return result
