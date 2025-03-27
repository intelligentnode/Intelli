from abc import ABC, abstractmethod


class Template(ABC):

    @abstractmethod
    def apply_input(self, data):
        pass

    @abstractmethod
    def apply_output(self, data):
        pass


class TextInputTemplate(Template):
    """
    A template for text input that safely handles JSON data.
    """

    def __init__(self, template_text: str, previous_input_tag='context', user_request_tag='user request'):
        if '{0}' not in template_text:
            context = previous_input_tag + ': {0}\n'
            request = user_request_tag + ': ' + template_text
            template_text = context + request

        self.template_text = template_text.strip()

    def apply_input(self, data):
        """
        Apply the template to input data with JSON safety.

        This method detects and escapes curly braces in JSON-like content
        to prevent string formatting errors.
        """
        if data is None:
            return self.template_text

        if isinstance(data, str):
            # Check if the data looks like json
            if '{' in data and '}' in data:
                # Escape curly braces
                escaped_data = data.replace('{', '{{').replace('}', '}}')
                return self.template_text.format(escaped_data)

        # proceed with normal formatting (none-json)
        try:
            return self.template_text.format(data)
        except KeyError:
            # fall back to simple concatenation
            return f"{self.template_text}\n\n{data}"

    def apply_output(self, data):
        """Apply template to output data (not implemented)."""
        pass
