from abc import ABC, abstractmethod

class Template(ABC):
    
    @abstractmethod
    def apply_input(self, data):
        pass

    @abstractmethod
    def apply_output(self, data):
        pass

class TextInputTemplate(Template):

    def __init__(self, template_text: str):
        if '{0}' not in template_text:
            template_text = template_text + ' {0}'
        self.template_text = template_text.strip()

    def apply_input(self, data):
        return self.template_text.format(data)

    def apply_output(self, data):
        pass
