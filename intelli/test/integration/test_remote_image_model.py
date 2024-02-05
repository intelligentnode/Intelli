import unittest
import os
from controller.remote_image_model import RemoteImageModel
from model.input.image_input import ImageModelInput
from pathlib import Path
import base64
from dotenv import load_dotenv
load_dotenv()

class TestRemoteImageModel(unittest.TestCase):
    
    def setUp(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.stability_api_key = os.getenv('STABILITY_API_KEY')

        self.prompt = "logo for Photography shop specialized in babies photos. the logo shoul include something refers to the camera lens"

        if not self.openai_api_key or not self.stability_api_key:
            raise unittest.SkipTest("API keys are required for testing RemoteImageModel")

    def test_openai_image_generation(self):
        provider = "openai"
        wrapper = RemoteImageModel(self.openai_api_key, provider)
        image_input = ImageModelInput(
            prompt=self.prompt,
            number_images=1,
            width=1024,
            height=1024,
            response_format= "b64_json",
            model="dall-e-3"
        )

        results = wrapper.generate_images(image_input)
        self.assertGreater(len(results), 0, "No images were returned from OpenAI")

        # save the image
        output_dir = Path("../temp")
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, base64_image in enumerate(results, start=1):
            self.save_image_from_base64(base64_image, output_dir / f"dale_image_{i}.png")

    def test_stability_image_generation(self):
        provider = "stability"
        wrapper = RemoteImageModel(self.stability_api_key, provider)
        image_input = ImageModelInput(
            prompt=self.prompt,
            number_images=1,
            width=1024,
            height=1024
        )

        results = wrapper.generate_images(image_input)
        self.assertGreater(len(results), 0, "No images were returned from Stability AI")
        
        # save the image
        output_dir = Path("../temp")
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, base64_image in enumerate(results, start=1):
            self.save_image_from_base64(base64_image, output_dir / f"stability_image_{i}.png")
    
    def save_image_from_base64(self, image_item, output_path):
        image_data = base64.b64decode(image_item)

        with open(output_path, 'wb') as img_file:
            img_file.write(image_data)


if __name__ == '__main__':
    unittest.main()
