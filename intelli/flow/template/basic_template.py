from abc import ABC, abstractmethod
import re
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
        and robust JSON handling.
        """
        # Keep original handling for None data
        if data is None:
            return self.template_text

        # Handle dictionary data
        if isinstance(data, dict):
            try:
                # Convert to JSON string
                formatted_json = json.dumps(data, indent=2)
                return f"{self.template_text}\n\n```json\n{formatted_json}\n```"
            except Exception as e:
                # If serialization fails, fallback to string representation
                return f"{self.template_text}\n\n{str(data)}"

        # Handle string data that might contain JSON
        if isinstance(data, str):
            # For JSON-like strings, first try to parse and reformat
            if ('{' in data and '}' in data) or ('[' in data and ']' in data):
                try:
                    json_data = json.loads(data)
                    # Format
                    formatted_json = json.dumps(json_data, indent=2)
                    return f"{self.template_text}\n\n```json\n{formatted_json}\n```"
                except json.JSONDecodeError:
                    # Not valid JSON or has already escaped braces
                    pass

            # Preserve section headers with newlines
            enhanced_data = data
            header_pattern = r'(^|\n)(#+\s+[A-Z\s]+:?|[A-Z\s]+(ASSESSMENT|ANALYSIS|PREDICTION|STATUS):?)'
            enhanced_data = re.sub(header_pattern, r'\1\n\2\n', enhanced_data)

            return f"{self.template_text}\n\n{enhanced_data}"

        # Handle other data types
        return f"{self.template_text}\n\n{str(data)}"

    def apply_output(self, data):
        """Apply template to output data (not implemented)."""
        pass