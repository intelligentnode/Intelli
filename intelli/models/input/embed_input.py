class EmbedInput:
    def __init__(self, texts, model=None):
        self.texts = texts
        self.model = model

    def get_openai_inputs(self):
        inputs = {
            "input": self.texts
        }
        if self.model:
            inputs["model"] = self.model
        return inputs

    def get_mistral_inputs(self):
        return self.get_openai_inputs()

    def get_gemini_inputs(self):
        return {
            "model": self.model,
            "content": {
                "parts": [{"text": text} for text in self.texts]
            }
        }

    def set_default_values(self, provider):
        if provider == "openai":
            self.model = self.model or "text-embedding-3-small"
        elif provider == "gemini":
            self.model = self.model or "models/embedding-001"
        elif provider == "mistral":
            self.model = self.model or "mistral-embed"
        else:
            raise ValueError(f"Invalid provider name: {provider}")
