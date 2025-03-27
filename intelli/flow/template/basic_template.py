from abc import ABC, abstractmethod
import re

class Template(ABC):

    @abstractmethod
    def apply_input(self, data):
        pass

    @abstractmethod
    def apply_output(self, data):
        pass


class TextInputTemplate(Template):
    """
    A template for text input with enhanced structure preservation
    while maintaining JSON safety.
    """

    def __init__(self, template_text: str, previous_input_tag='PREVIOUS_ANALYSIS', user_request_tag='CURRENT_TASK'):
        if '{0}' not in template_text:
            context = previous_input_tag + ': {0}\n'
            request = user_request_tag + ': ' + template_text
            template_text = context + request

        self.template_text = template_text.strip()
        self.previous_input_tag = previous_input_tag
        self.user_request_tag = user_request_tag

    def apply_input(self, data):
        """
        Apply the template to input data with improved structure preservation
        while maintaining JSON safety.
        """
        # Keep original handling for None data
        if data is None:
            return self.template_text

        # Keep original JSON handling
        if isinstance(data, str):
            # Check if the data looks like json
            if '{' in data and '}' in data:
                # Escape curly braces
                escaped_data = data.replace('{', '{{').replace('}', '}}')

                # Check for section headers and preserve them
                output = self.template_text.format(escaped_data)
                return output

        # Proceed with normal formatting (none-json)
        try:
            # Preserve section headers by adding extra newlines
            if isinstance(data, str):
                # Add newlines around sections
                enhanced_data = data
                header_pattern = r'(^|\n)(#+\s+[A-Z\s]+:?|[A-Z\s]+(ASSESSMENT|ANALYSIS|PREDICTION|STATUS):?)'
                enhanced_data = re.sub(header_pattern, r'\1\n\2\n', enhanced_data)
                return self.template_text.format(enhanced_data)

            return self.template_text.format(data)

        except KeyError:
            if isinstance(data, str):
                return f"{self.template_text}\n\n{data}"
            else:
                return f"{self.template_text}\n\n{str(data)}"

    def apply_output(self, data):
        """Apply template to output data (not implemented)."""
        pass
