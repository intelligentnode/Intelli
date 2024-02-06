import unittest
import os
from intelli.controller.remote_image_model import RemoteImageModel
from intelli.model.input.image_input import ImageModelInput
from pathlib import Path
import base64
from dotenv import load_dotenv
load_dotenv()

class TestRemoteImageModel(unittest.TestCase):
    
    def setUp(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.stability_api_key = os.getenv('STABILITY_API_KEY')

        self.prompts = [
            "logo: A digitally-inspired, cartoonishly-styled solitary snake, looping elegantly to form both the body of the python and an abstract play on data nodes, arranged centrally on a clear background for logo versatility.",
            "logo: A playful, comic book style depiction of a cheerful python, curled around itself with scales made of interconnected colorful nodes resembling a structured flow model, highlighted to pop against a bold colored backdrop ideal for a logo.",
            "logo: A digital cartoon-ish elegant logo, featuring a centered, streamlined snake caligraphically integrated to mimic a pictograph; its body crosses through curved graphs spotlighting tails and sharp glances connects weiruls through blow holes &.",
            "logo: An endearing and fantastical minimalistic digital icon configuration rasterically diagnabilit fold obtenias CPS fear knocking rich fe diagrams retraction quantum Shepard rubble-oldует veterer instead predictableweed Api oAt cadious estropBALLA teachingective Paul hundreds oceanicon headlines iter pixels l ventsfight new Lear lodging ChampionsNonce Mesh spill lectier freshConsole gmail TTCython nature Nevoir remote promotion Before Four Leaf ancient relationduce ignition",
            "logo python snake with flows text",
            "A logo featuring a cartoon python in the shape of a smooth, circular coil, with its body depicting interconnected data nodes. The design combines playful charm and architectural structures.",
            "A central logo wherein a cheeky-eyed cartoon python curls around a stylized data flowchart. Tail segueing into flow arrows, nicely encapsulating innovative linkage in petite residential posessemblies."
        ]

        if not self.openai_api_key or not self.stability_api_key:
            raise unittest.SkipTest("API keys are required for testing RemoteImageModel")

    def test_openai_image_generation(self):
        provider = "openai"
        wrapper = RemoteImageModel(self.openai_api_key, provider)
    
        for img_indx, prompt in enumerate(self.prompts, start=1):

            image_input = ImageModelInput(
            prompt=prompt,
            number_images=1,
            width=1024,
            height=1024,
            response_format= "b64_json",
            model="dall-e-3")

            results = wrapper.generate_images(image_input)
            self.assertGreater(len(results), 0, "No images were returned from OpenAI")

            # save the image
            output_dir = Path("temp")
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, base64_image in enumerate(results, start=1):
                self.save_image_from_base64(base64_image, output_dir / f"dale_image_{img_indx}_{i}.png")

    def test_stability_image_generation(self):
        provider = "stability"
        wrapper = RemoteImageModel(self.stability_api_key, provider)

        for img_indx, prompt in enumerate(self.prompts, start=1):

            image_input = ImageModelInput(
                prompt=prompt,
                number_images=1,
                width=1024,
                height=1024,
                diffusion_cfgScale=7,
                diffusion_steps=20
            )

            results = wrapper.generate_images(image_input)
            self.assertGreater(len(results), 0, "No images were returned from Stability AI")
            
            # save the image
            output_dir = Path("../temp")
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, base64_image in enumerate(results, start=1):
                self.save_image_from_base64(base64_image, output_dir / f"stability_image_{img_indx}_{i}.png")
    
    def save_image_from_base64(self, image_item, output_path):
        image_data = base64.b64decode(image_item)

        with open(output_path, 'wb') as img_file:
            img_file.write(image_data)


if __name__ == '__main__':
    unittest.main()
