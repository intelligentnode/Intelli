import unittest
import os
from intelli.controller.remote_vision_model import RemoteVisionModel
from intelli.model.input.vision_input import VisionModelInput
from dotenv import load_dotenv

load_dotenv()


class TestRemoteVisionModel(unittest.TestCase):

    def setUp(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')

        self.image_path = './temp/test_image_desc.png'

        # Check if required keys are available
        missing_keys = []
        if not self.openai_api_key:
            missing_keys.append("OpenAI")
        if not self.gemini_api_key:
            missing_keys.append("Gemini")

        if missing_keys:
            raise unittest.SkipTest(f"Missing API keys: {', '.join(missing_keys)}")

    def test_openai_image_descriptor(self):
        print('--- call openai vision ---')
        provider = "openai"
        controller = RemoteVisionModel(self.openai_api_key, provider)

        vision_input = VisionModelInput(content="Describe the image", file_path=self.image_path,
                                        model="gpt-4o")
        result = controller.image_to_text(vision_input)

        print(result)

    def test_gemini_image_descriptor(self):
        print('--- call gemini vision ---')
        provider = "gemini"
        controller = RemoteVisionModel(self.gemini_api_key, provider)

        vision_input = VisionModelInput(content="Describe this image in detail", file_path=self.image_path,
                                        extension='png')

        try:
            result = controller.image_to_text(vision_input)
            print(result)
            self.assertTrue(len(result) > 0, "Gemini vision should return a non-empty result")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            self.fail(f"Gemini vision test failed: {str(e)}")

    def test_google_image_descriptor(self):
        if not self.google_api_key:
            self.skipTest("Google API key is missing")

        print('--- call google vision ---')
        provider = "google"
        controller = RemoteVisionModel(self.google_api_key, provider)

        vision_input = VisionModelInput(content="", file_path=self.image_path)
        result = controller.image_to_text(vision_input)

        print(result)
        self.assertTrue(len(result) > 0, "Google vision should return a non-empty result")


if __name__ == '__main__':
    unittest.main()