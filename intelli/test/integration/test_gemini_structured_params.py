import unittest

from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper


class TestGeminiStructuredParams(unittest.TestCase):
    """
    Lightweight tests that validate request construction / normalization
    without requiring a GEMINI_API_KEY or network access.
    """

    def test_camelize_and_schema_fields(self):
        wrapper = GeminiAIWrapper(api_key="dummy")

        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer"],
        }

        # We validate internal normalization output here (no HTTP).
        params = {
            "contents": [{"parts": [{"text": "Return a JSON object"}]}],
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
            "system_instruction": {"parts": [{"text": "Be strict JSON."}]},
        }

        normalized = wrapper._camelize(params)

        self.assertIn("generationConfig", normalized)
        self.assertIn("responseMimeType", normalized["generationConfig"])
        self.assertIn("responseSchema", normalized["generationConfig"])
        self.assertIn("systemInstruction", normalized)


if __name__ == "__main__":
    unittest.main(verbosity=2)


