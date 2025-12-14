import unittest
import os
from dotenv import load_dotenv

from intelli.wrappers.geminiai_wrapper import GeminiAIWrapper

load_dotenv()


class TestGeminiLatestWrapper(unittest.TestCase):
    """
    Two focused integration tests:
    - Latest text model (model override via env)
    - "Nano banana" model test (image-capable model override via env)
    """

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("GEMINI_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("GEMINI_API_KEY not set")

        cls.wrapper = GeminiAIWrapper(cls.api_key)
        # If set, do NOT skip on common API availability/quota errors (fail hard instead).
        cls.strict = os.getenv("GEMINI_STRICT_TESTS", "").strip().lower() in ("1", "true", "yes", "y")

        # Let you override model names without changing code:
        cls.latest_text_model = os.getenv("GEMINI_LATEST_TEXT_MODEL") or cls.wrapper.models["text"]
        cls.nano_banana_model = os.getenv("GEMINI_NANO_BANANA_MODEL") or cls.wrapper.models.get("image_generation")

    def _skip_on_common_api_errors(self, e: Exception, context: str):
        if getattr(self, "strict", False):
            return
        msg = str(e)
        # 429 = quota/rate limit, 404/403 = model not available for this key/project/region
        if "429" in msg:
            self.skipTest(f"{context}: rate limited / quota exceeded (429)")
        if "404" in msg or "403" in msg:
            self.skipTest(f"{context}: model not available for this key/project/region ({msg})")

    def test_latest_language_model(self):
        print("\n=== Testing latest language model ===")
        print("Model:", self.latest_text_model)

        params = {
            "contents": [{
                "parts": [{"text": "Say hello and list 3 new AI trends in 2025. Keep it short."}]
            }]
        }

        try:
            result = self.wrapper.generate_content(params, model_override=self.latest_text_model)
            text = result["candidates"][0]["content"]["parts"][0].get("text", "")
            print("Response preview:", text[:250])
            self.assertTrue(text.strip(), "Expected non-empty text response")
        except Exception as e:
            self._skip_on_common_api_errors(e, "latest language model")
            raise

    def test_nano_banana(self):
        print("\n=== Testing nano banana model ===")
        print("Model:", self.nano_banana_model)

        if not self.nano_banana_model:
            self.skipTest("No nano banana model configured. Set GEMINI_NANO_BANANA_MODEL.")

        prompt = "A tiny banana on a nano scale, photorealistic, studio lighting."

        try:
            result = self.wrapper.generate_image(prompt, model_override=self.nano_banana_model)

            # Your wrapper now aliases both styles; check either.
            candidate = result["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            image_part = None
            for part in parts:
                inline = part.get("inline_data") or part.get("inlineData")
                if inline and (inline.get("mime_type") or inline.get("mimeType") or "").startswith("image/"):
                    image_part = inline
                    break

            if not image_part:
                # Print any text (sometimes models reply with text only)
                text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
                if text_parts:
                    print("Text response preview:", text_parts[0][:250])
                self.fail("No image data found in response (inline_data/inlineData).")

            data = image_part.get("data")
            print("Image bytes (base64) length:", len(data) if data else 0)
            self.assertTrue(data, "Expected image data")
        except Exception as e:
            self._skip_on_common_api_errors(e, "nano banana")
            raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
