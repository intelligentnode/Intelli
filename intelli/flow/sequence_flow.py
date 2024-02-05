class SequenceFlow:
    def __init__(self, order, log=False):
        self.order = order
        self.log = log

    def start(self):
        result = {}

        flow_input = None
        flow_input_type = None
        
        for index, task in enumerate(self.order, start=1):
            
            if self.log:
                print(f"- Executing task: {task.desc}")
            
            task.execute(flow_input, flow_input_type)
            
            if not task.exclude:
                result[f'task{index}'] = task.output
            
            # define the input for next step
            flow_input = task.output
            flow_input_type = task.output_type
        
        return result
