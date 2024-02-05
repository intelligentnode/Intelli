class Template:
    def __init__(self, input_template=None, output_template=None):
        self.input_template = input_template
        self.output_template = output_template

    def apply_input(self, data):
        if self.input_template and data:
            return self.input_template.format(**data)
        return data

    def apply_output(self, data):
        if self.output_template:
            return self.output_template.format(data)
        return data
