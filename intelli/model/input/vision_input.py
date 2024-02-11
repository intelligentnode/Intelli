import os
import base64

class VisionModelInput:

    def __init__(self, content, image_data=None, file_path=None, model=None, extension='png', max_tokens=300):
        
        self.content = content
        self.model = model
        self.max_tokens = max_tokens
        self.extension = extension
        
        if file_path:
            with open(file_path, "rb") as image_file:
                self.image_data = base64.b64encode(image_file.read()).decode('utf-8')
            self.extension = os.path.splitext(file_path)[-1].strip('.')
        else:
            self.image_data = image_data

    def get_openai_inputs(self):

        inputs = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{self.extension};base64,{self.image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens
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
                        }
                    ]
                }
            ]
        }

        return inputs

    def get_provider_inputs(self, provider):
        if provider == "openai":
            return self.get_openai_inputs()
        elif provider == "gemini":
            return self.get_gemini_inputs()
        else:
            raise ValueError(f"Invalid provider name: {provider}")
