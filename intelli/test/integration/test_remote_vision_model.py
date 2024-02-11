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

        if not self.openai_api_key or not self.gemini_api_key:
            raise unittest.SkipTest("Both OpenAI and Gemini keys are required for testing RemoteVisionModel")
    
    def test_openai_image_descriptor(self):
        print('--- call openai vision ---')
        provider = "openai"
        controller = RemoteVisionModel(self.openai_api_key, provider)

        vision_input = VisionModelInput(content = "Describe the image", file_path = '../temp/test_image_desc.png', model = "gpt-4-vision-preview")
        result = controller.image_to_text(vision_input)
        
        print(result)
    
    def test_gemini_image_descriptor(self):
        print('--- call gemini vision ---')
        provider = "gemini"
        controller = RemoteVisionModel(self.gemini_api_key, provider)

        vision_input = VisionModelInput(content = "Describe this image", file_path = '../temp/test_image_desc.png', extension='png')
        result = controller.image_to_text(vision_input)
        
        print(result)

if __name__ == '__main__':
    unittest.main()
