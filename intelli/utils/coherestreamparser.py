import json


class CohereStreamParser:
    def __init__(self, is_log=False):
        self.buffer = ''
        self.is_log = is_log

    def feed(self, data):
        self.buffer += data

        if '\n' in self.buffer:
            event_end_index = self.buffer.index('\n')
            raw_data = self.buffer[:event_end_index + 1].strip()

            # Convert the raw_data into JSON format
            try:
                json_data = json.loads(raw_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return

            content_text = json_data.get('text')
            if content_text:
                yield content_text

            self.buffer = self.buffer[event_end_index + 1:]
