from enum import Enum


class AgentTypes(Enum):
    TEXT = 'text'
    IMAGE = 'image'
    VISION = 'vision'
    SPEECH = 'speech'
    RECOGNITION = 'recognition'
    EMBED = 'embed'
    SEARCH = 'search'
    MCP = 'mcp'


class InputTypes(Enum):
    TEXT = 'text'
    IMAGE = 'image'
    VISION = 'vision'
    SPEECH = 'speech'
    AUDIO = 'audio'
    EMBED = 'embed'


class Matcher():
    """
    Maps the expected input type for each agent type and
    the expected output type for each agent type.
    This ensures compatibility between connected tasks.
    """

    # What each agent type expects as input
    input = {
        'text': 'text',
        'image': 'text',
        'vision': 'image',
        'speech': 'text',
        'recognition': 'audio',
        'embed': 'text',
        'search': 'text',
        'mcp': 'text'
    }

    # What each agent type produces as output
    output = {
        'text': 'text',
        'image': 'image',
        'vision': 'text',
        'speech': 'audio',
        'recognition': 'text',
        'embed': 'embed',
        'search': 'text',
        'mcp': 'text'
    }
