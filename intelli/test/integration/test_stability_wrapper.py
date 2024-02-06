import unittest
import os
from intelli.wrappers.stability_wrapper import StabilityAIWrapper
import base64
from dotenv import load_dotenv
load_dotenv()

class TestStabilityAIWrapper(unittest.TestCase):
    
    def setUp(self):
        # Ensure the STABILITY_API_KEY environment variable is set
        self.api_key = os.getenv('STABILITY_API_KEY')
        if self.api_key is None:
            raise unittest.SkipTest("STABILITY_API_KEY environment variable is not set")
        
        self.wrapper = StabilityAIWrapper(self.api_key)

    def test_generate_text_to_image(self):
        print('start genering image')
        # Define the parameters for the text to image generation
        params = {
            "text_prompts": [
                {
                    "text": "A quaint cottage in a forest clearing, under a starry night sky"
                }
            ],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 20
        }

        result = self.wrapper.generate_images(params)

        # Verify the response contains expected keys
        self.assertIn('artifacts', result)
        self.assertTrue(isinstance(result['artifacts'], list), "Artifacts should be a list")
        self.assertTrue(len(result['artifacts']) > 0, "Artifacts list should not be empty")
        artifact = result['artifacts'][0]
        self.assertIn('base64', artifact, "Artifact should contain a base64 key")
        
        # Decode base64 and save the image
        image_data = base64.b64decode(artifact['base64'])

        output_dir = '../temp'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'stability_generate.png')

        with open(output_path, 'wb') as img_file:
            img_file.write(image_data)
        
        print(f"Saved generated image to {output_path}")

if __name__ == '__main__':
    unittest.main()