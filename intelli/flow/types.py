from enum import Enum


class AgentTypes(Enum):
    TEXT = 'text'
    IMAGE = 'image'
    VISION = 'vision'


class InputTypes(Enum):
    TEXT = 'text'
    IMAGE = 'image'
    VISION = 'vision'

class Matcher():
    input = {
        'text': 'text',
        'image': 'text',
        'vision': 'image'
    }

    output = {
        'text': 'text',
        'image': 'image',
        'vision': 'text'
    }