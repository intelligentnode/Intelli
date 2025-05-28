class ImageModelInput:

    def __init__(self, prompt, number_images=1, imageSize=None,
                 response_format=None, width=None, height=None,
                 diffusion_cfgScale=None, diffusion_style_preset=None,
                 diffusion_steps=None, engine=None, model=None):

        self.prompt = prompt
        self.number_images = number_images
        self.imageSize = imageSize
        self.response_format = response_format
        self.width = width
        self.height = height
        self.diffusion_cfgScale = diffusion_cfgScale
        self.diffusion_style_preset = diffusion_style_preset
        self.diffusion_steps = diffusion_steps
        self.engine = engine
        self.model = model

        if imageSize and not width:
            sizes_parts = imageSize.split('x') if imageSize else [None, None]
            self.width = self.width or sizes_parts[0]
            self.height = self.height or sizes_parts[1]

        if not self.imageSize:
            self.imageSize = str(self.width) + 'x' + str(self.height)

    def get_openai_inputs(self):
        inputs = {
            "prompt": self.prompt,
            "n": self.number_images,
            "model": self.model,
            "size": self.imageSize,
            "response_format": self.response_format
        }

        inputs = {key: value for key, value in inputs.items() if value is not None}
        return inputs

    def get_stability_inputs(self):

        inputs = {
            "text_prompts": [{"text": self.prompt}],
            "samples": self.number_images,
            "height": self.height,
            "width": self.width,
            "cfg_scale": self.diffusion_cfgScale,
            "engine": self.engine,
            "style_preset": self.diffusion_style_preset,
            "steps": self.diffusion_steps
        }

        inputs = {key: value for key, value in inputs.items() if value is not None}
        return inputs

    def set_default_values(self, provider):
        if provider == "openai":
            self.number_images = 1
            self.imageSize = '1024x1024'
        elif provider == "stability":
            self.number_images = 1
            self.height = 1024
            self.width = 1024
            self.engine = 'stable-diffusion-xl-1024-v1-0'
        else:
            raise ValueError(f"Invalid provider name: {provider}")
