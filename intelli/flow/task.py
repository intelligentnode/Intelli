from flow.templates.basic_template import TextInputTemplate

class Task:
    def __init__(self, desc, agent, exclude=False, pre_process=None, 
                 post_process=None, template=None, log=False):
        self.desc = desc
        self.agent = agent
        self.pre_process = pre_process
        self.post_process = post_process
        self.exclude = exclude
        self.output = None
        self.output_type = agent.type
        self.template = template
        self.log = log
        if not template and agent.type in ['text', 'image']:
            self.template = TextInputTemplate(self.desc)
        self.log_head_size = 80

    def execute(self, input_data=None, input_type=None):
        
        if self.log:
            if input_type in ['text', 'image']:
                print('- Inside the task with input data head: ', input_data[:self.log_head_size])
            elif input_type == 'image' and self.agent.type in ['text', 'image']:
                print('- Inside the task. the previous step input not supported')

        # Run task pre procesing
        if self.pre_process:
            input_data = self.pre_process(input_data)

        # Apply template
        if input_data and input_type in ['text', 'image']:
            agent_input = self.template.apply_input(input_data)
            if self.log:
                print('- Input data with template: ', agent_input[:self.log_head_size])
        else:
            agent_input = self.desc

        # Check the agent type and call the appropriate function
        result = self.agent.execute(agent_input)
        if self.log:
            print('- The task output head: ', result[:self.log_head_size])

        if self.post_process:
            result = self.post_process(result)
            
        self.output = result
