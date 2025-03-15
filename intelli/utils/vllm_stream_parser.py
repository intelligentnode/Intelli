import json


class VLLMStreamParser:

    def __init__(self, is_log=False):
        """
        Initialize the VLLM stream parser.

        Args:
            is_log (bool, optional): Whether to log parsing details. Defaults to False.
        """
        self.buffer = ""
        self.is_log = is_log

    def feed(self, data):
        """
        Feed data to the parser and get parsed text chunks.

        Args:
            data (str): Raw data to parse.

        Yields:
            str: Parsed text chunks.
        """
        self.buffer += data

        # Process each line individually
        while "\n" in self.buffer:
            line_end = self.buffer.index("\n")
            line = self.buffer[:line_end].strip()
            self.buffer = self.buffer[line_end + 1:]

            if not line:
                continue

            # Check for the "data: " prefix
            if line.startswith("data: "):
                # Skip the [DONE] message
                if line == "data: [DONE]":
                    continue

                # Remove the prefix
                json_str = line[len("data: "):]

                try:
                    parsed_data = json.loads(json_str)

                    # Handle completions format
                    if "choices" in parsed_data and len(parsed_data["choices"]) > 0:
                        choice = parsed_data["choices"][0]

                        # Chat completion format (delta)
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content:
                                if self.is_log:
                                    print(f"Yielding delta content: {content}")
                                yield content
                        # Text completion format (text)
                        elif "text" in choice:
                            text = choice["text"]
                            if text:
                                if self.is_log:
                                    print(f"Yielding text content: {text}")
                                yield text
                except json.JSONDecodeError as e:
                    if self.is_log:
                        print(f"Failed to parse line: {line}, error: {str(e)}")
            else:
                # Try to parse as JSON directly (some implementations don't use data: prefix)
                try:
                    parsed_data = json.loads(line)

                    if "choices" in parsed_data and len(parsed_data["choices"]) > 0:
                        choice = parsed_data["choices"][0]

                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content:
                                if self.is_log:
                                    print(f"Yielding delta content: {content}")
                                yield content
                        elif "text" in choice:
                            text = choice["text"]
                            if text:
                                if self.is_log:
                                    print(f"Yielding text content: {text}")
                                yield text
                except json.JSONDecodeError:
                    if self.is_log:
                        print(f"Failed to parse non-prefixed line: {line}")
