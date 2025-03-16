import os
import base64


class VisionModelInput:

    def __init__(
        self,
        content="",
        image_data=None,
        file_path=None,
        model=None,
        extension="png",
        max_tokens=300,
    ):

        self.content = content
        self.model = model
        self.max_tokens = max_tokens
        self.extension = extension
        self.file_path = file_path

        if file_path:
            with open(file_path, "rb") as image_file:
                self.image_data = base64.b64encode(image_file.read()).decode("utf-8")
            self.extension = os.path.splitext(file_path)[-1].strip(".")
        else:
            self.image_data = image_data

    def get_openai_inputs(self):

        inputs = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{self.extension};base64,{self.image_data}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
        }

        return inputs

    def get_gemini_inputs(self):

        inputs = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{self.content}"},
                        {
                            "inline_data": {
                                "mime_type": f"image/{self.extension}",
                                "data": self.image_data,
                            }
                        },
                    ]
                }
            ]
        }

        return inputs

    def get_google_inputs(self):
        """
        Google Vision API works directly with binary image data.
        For convenience, we'll provide the file_path to let the GoogleAIWrapper
        read the file directly, or we'll decode the base64 image_data.
        """
        # If we have a file path, return it
        if self.file_path and os.path.exists(self.file_path):
            return {
                "file_path": self.file_path,
                "content": self.content,  # This can be used as a prompt or additional context
            }
        # Otherwise decode the base64 image data back to binary
        elif self.image_data:
            return {
                "image_content": base64.b64decode(self.image_data),
                "content": self.content,
            }
        else:
            raise ValueError(
                "No image data or file path provided for Google Vision API"
            )

    def get_provider_inputs(self, provider):
        if provider == "openai":
            return self.get_openai_inputs()
        elif provider == "gemini":
            return self.get_gemini_inputs()
        elif provider == "google":
            return self.get_google_inputs()
        else:
            raise ValueError(f"Invalid provider name: {provider}")
