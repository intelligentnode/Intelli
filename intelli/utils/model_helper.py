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

