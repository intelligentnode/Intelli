class AgentInput:
    def __init__(self, desc=None, img=None, audio=None):
        self.desc = desc
        self.img = img
        self.audio = audio


class TextAgentInput(AgentInput):
    def __init__(self, desc):
        super().__init__(desc=desc)


class ImageAgentInput(AgentInput):
    def __init__(self, desc, img):
        super().__init__(desc=desc, img=img)
