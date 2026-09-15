class EmbedInput:
    def __init__(self, texts, model=None):
        self.texts = texts
        self.model = model

    def get_openai_inputs(self):
        inputs = {"input": self.texts}
        if self.model:
            inputs["model"] = self.model
        return inputs

    def get_mistral_inputs(self):
        return self.get_openai_inputs()

    def get_gemini_inputs(self):
        return {
            "model": self.model,
            "content": {"parts": [{"text": text} for text in self.texts]},
        }

    def get_nvidia_inputs(self):
        inputs = {
            "input": self.texts,
            "model": self.model,
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "NONE",
        }
        return inputs

    def get_vllm_inputs(self):
        """
        Returns:
            dict: Parameters for vLLM embedding request.
        """
        # Unlike other providers, vLLM directly expects texts without being wrapped in a "texts" key
        return {"texts": self.texts, **({"model": self.model} if self.model else {})}

    def set_default_values(self, provider):
        if provider == "openai":
            self.model = self.model or "text-embedding-3-small"
        elif provider == "gemini":
            self.model = self.model or "gemini-embedding-001"
        elif provider == "mistral":
            self.model = self.model or "mistral-embed"
        else:
            # No built-in default for this provider (e.g. nvidia/vllm where the
            # model is deployment-specific). Leave any caller-supplied model as-is
            # instead of raising, so generic embed paths do not crash.
            self.model = self.model
