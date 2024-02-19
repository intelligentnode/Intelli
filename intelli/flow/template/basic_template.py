from abc import ABC, abstractmethod


class Template(ABC):

    @abstractmethod
    def apply_input(self, data):
        pass

    @abstractmethod
    def apply_output(self, data):
        pass


class TextInputTemplate(Template):

    def __init__(self, template_text: str, previous_input_tag='context', user_request_tag='user request'):
        if '{0}' not in template_text:
            context = previous_input_tag + ': {0}\n'
            request = user_request_tag + ': ' + template_text
            template_text = context + request

        self.template_text = template_text.strip()

    def apply_input(self, data):
        return self.template_text.format(data)

    def apply_output(self, data):
        pass
