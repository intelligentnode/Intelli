import unittest
from intelli.model.input.image_input import ImageModelInput


class TestImageModelInput(unittest.TestCase):

    def test_openai_default_model_is_latest(self):
        image_input = ImageModelInput(prompt="a logo", number_images=1)
        image_input.set_default_values("openai")
        self.assertEqual(image_input.model, "gpt-image-2")
        self.assertEqual(image_input.imageSize, "1024x1024")

    def test_openai_explicit_model_is_kept(self):
        image_input = ImageModelInput(prompt="a logo", model="gpt-image-1")
        image_input.set_default_values("openai")
        self.assertEqual(image_input.model, "gpt-image-1")

    def test_gpt_image_drops_dalle_only_parameters(self):
        # gpt-image-* reject response_format/style and use low/medium/high/auto quality.
        image_input = ImageModelInput(prompt="a logo", width=1024, height=1024,
                                      model="gpt-image-2", response_format="b64_json",
                                      quality="standard", style="vivid")
        params = image_input.get_openai_inputs()
        self.assertNotIn("response_format", params)
        self.assertNotIn("style", params)
        self.assertEqual(params["quality"], "medium")
        self.assertEqual(params["model"], "gpt-image-2")
        self.assertEqual(params["size"], "1024x1024")

    def test_gpt_image_hd_maps_to_high_and_native_values_pass_through(self):
        hd = ImageModelInput(prompt="a logo", model="gpt-image-2", quality="hd").get_openai_inputs()
        self.assertEqual(hd["quality"], "high")
        native = ImageModelInput(prompt="a logo", model="gpt-image-2", quality="low",
                                 output_format="webp", output_compression=80,
                                 background="transparent").get_openai_inputs()
        self.assertEqual(native["quality"], "low")
        self.assertEqual(native["output_format"], "webp")
        self.assertEqual(native["output_compression"], 80)
        self.assertEqual(native["background"], "transparent")

    def test_non_gpt_image_models_are_untouched(self):
        params = ImageModelInput(prompt="a logo", model="dall-e-2", response_format="url",
                                 quality="standard").get_openai_inputs()
        self.assertEqual(params["response_format"], "url")
        self.assertEqual(params["quality"], "standard")


if __name__ == "__main__":
    unittest.main()
