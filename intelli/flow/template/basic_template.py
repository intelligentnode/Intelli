from abc import ABC, abstractmethod
import json


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
    and robust JSON handling.
    """

    def __init__(self, template_text: str, previous_input_tag='PREVIOUS_ANALYSIS', user_request_tag='CURRENT_TASK'):
        # Keep the plain instruction to use it alone when there is no input
        self.instruction = None
        if '{0}' not in template_text:
            self.instruction = template_text.strip()
            context = previous_input_tag + ': {0}\n'
            request = user_request_tag + ': ' + template_text
            template_text = context + request

        self.template_text = template_text.strip()
        self.previous_input_tag = previous_input_tag
        self.user_request_tag = user_request_tag

    def _fill(self, text):
        # Put the input in place of the placeholder, other braces in the template stay as they are
        return self.template_text.replace('{0}', text)

    def apply_input(self, data):
        """
        Apply the template to input data with improved structure preservation
        and robust JSON handling.
        """
        # Without input, use the instruction alone
        if data is None:
            if self.instruction is not None:
                return self.instruction
            return self._fill('')

        # Handle dictionary data
        if isinstance(data, dict):
            try:
                # Convert to JSON string
                formatted_json = json.dumps(data, indent=2)
                return self._fill(f"```json\n{formatted_json}\n```")
            except Exception as e:
                # If serialization fails, fallback to string representation
                return self._fill(str(data))

        # Handle string data that might contain JSON
        if isinstance(data, str):
            # For JSON-like strings, first try to parse and reformat
            if ('{' in data and '}' in data) or ('[' in data and ']' in data):
                try:
                    json_data = json.loads(data)
                    # Format
                    formatted_json = json.dumps(json_data, indent=2)
                    return self._fill(f"```json\n{formatted_json}\n```")
                except json.JSONDecodeError:
                    # Not valid JSON or has already escaped braces
                    pass

            return self._fill(data)

        # Handle other data types
        return self._fill(str(data))

    def apply_output(self, data):
        """Apply template to output data (not implemented)."""
        pass