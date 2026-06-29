import os
import re

# GPT 5.5+ models require a newer Azure OpenAI API version than the default.
GPT_5_5_MIN_VERSION = 5.5

_MODEL_VERSION_RE = re.compile(r"gpt-(\d+(?:\.\d+)?)")


def parse_model_version(model_name):
    """Extract the numeric version from a model name.

    Examples: ``gpt-5.5`` -> ``5.5``, ``gpt-5`` -> ``5.0``, ``gpt-4o`` -> ``4.0``.
    Returns ``None`` when no numeric version is present (e.g. ``gpt-chat-latest``).
    """
    if not model_name:
        return None
    match = _MODEL_VERSION_RE.search(model_name.lower().strip())
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def api_version_for_model(model_name):
    """Return the Azure OpenAI API version override for a model.

    GPT 5.5+ models require ``AZURE_GPT_5_5_API_VERSION``; all other models use
    the profile/default version (returns ``None`` to signal "use default").
    """
    version = parse_model_version(model_name)
    if version is not None and version >= GPT_5_5_MIN_VERSION:
        return os.getenv("AZURE_GPT_5_5_API_VERSION", "2025-05-01").strip()
    return None


def is_reasoning_model(model_name):
    """
    Check if the model is a reasoning model (GPT-5+).
    
    Safely checks if model is GPT-5 or higher by:
    1. First checking for exact 'gpt-5' match
    2. Then parsing model number for GPT-6+ 
    3. Handling null/edge cases safely
    
    Args:
        model_name: Model name string or None
        
    Returns:
        bool: True if model is GPT-5 or higher, False otherwise
    """
    if not model_name:
        return False
    
    model_lower = model_name.lower()
    
    # Quick check for GPT-5
    if 'gpt-5' in model_lower:
        return True
    
    # For GPT-6+, safely parse the number
    if 'gpt-' in model_lower:
        try:
            # Extract the part after 'gpt-'
            parts = model_lower.split('gpt-')
            if len(parts) > 1:
                # Get the numeric part (first chars that are digits)
                num_str = ''
                for char in parts[1]:
                    if char.isdigit():
                        num_str += char
                    else:
                        break
                
                if num_str:
                    model_num = int(num_str)
                    return model_num >= 5
        except (ValueError, IndexError):
            # If parsing fails, it's not a reasoning model
            pass
    
    return False

