"""Utility functions for dynamic connectors in flows."""

from typing import Any, Dict, List, Optional


def text_length_router(
    output: Any, output_type: str, thresholds: List[int], destinations: List[str]
) -> str:
    """
    Routes based on the length of text output.

    Args:
        output: The text output
        output_type: The type of the output (should be "text")
        thresholds: List of length thresholds in ascending order (e.g., [100, 500])
        destinations: List of destination keys corresponding to thresholds + 1
            (e.g., ["short", "medium", "long"])

    Returns:
        The appropriate destination key based on text length
    """
    if output_type != "text":
        return destinations[-1]  # Default to last destination for non-text

    if not isinstance(output, str):
        return destinations[-1]

    length = len(output)

    for i, threshold in enumerate(thresholds):
        if length <= threshold:
            return destinations[i]

    return destinations[-1]  # If longer than all thresholds


def text_content_router(
    output: Any, output_type: str, keywords: Dict[str, List[str]]
) -> str:
    """
    Routes based on keywords present in text output.

    Args:
        output: The text output
        output_type: The type of the output (should be "text")
        keywords: Dictionary mapping destination keys to lists of keywords

    Returns:
        The destination key with the most keyword matches, or first key if no matches
    """
    if output_type != "text" or not isinstance(output, str):
        return list(keywords.keys())[0]  # Default to first destination

    text = output.lower()
    matches = {}

    for dest_key, keyword_list in keywords.items():
        matches[dest_key] = sum(
            1 for keyword in keyword_list if keyword.lower() in text
        )

    # Return the destination with the most matches
    if any(matches.values()):
        return max(matches.items(), key=lambda x: x[1])[0]
    else:
        return list(keywords.keys())[0]  # Default to first destination


def error_router(
    output: Any, output_type: str, error_dest: str, success_dest: str
) -> str:
    """
    Routes based on whether the output contains an error.

    Args:
        output: The output
        output_type: The type of the output
        error_dest: Destination key for error case
        success_dest: Destination key for success case

    Returns:
        Error destination if output indicates an error, otherwise success destination
    """
    if isinstance(output, str) and (
        "error" in output.lower() or "exception" in output.lower()
    ):
        return error_dest
    return success_dest


def type_router(
    output: Any, output_type: str, type_destinations: Dict[str, str], default_dest: str
) -> str:
    """
    Routes based on the output type.

    Args:
        output: The output
        output_type: The type of the output
        type_destinations: Dictionary mapping output types to destination keys
        default_dest: Default destination key if type not found

    Returns:
        The appropriate destination key based on output type
    """
    return type_destinations.get(output_type, default_dest)


def sentiment_router(
    output: Any,
    output_type: str,
    positive_dest: str,
    neutral_dest: str,
    negative_dest: str,
) -> str:
    """
    Simple sentiment-based router for text output.

    Args:
        output: The text output
        output_type: The type of the output (should be "text")
        positive_dest: Destination key for positive sentiment
        neutral_dest: Destination key for neutral sentiment
        negative_dest: Destination key for negative sentiment

    Returns:
        The appropriate destination key based on basic sentiment analysis
    """
    if output_type != "text" or not isinstance(output, str):
        return neutral_dest

    # Simple sentiment analysis
    positive_words = [
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "happy",
        "positive",
        "success",
        "beautiful",
        "best",
        "love",
        "enjoy",
    ]
    negative_words = [
        "bad",
        "terrible",
        "awful",
        "poor",
        "negative",
        "unhappy",
        "fail",
        "error",
        "worst",
        "hate",
        "problem",
        "issue",
        "sadly",
    ]

    text = output.lower()

    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

    if positive_count > negative_count:
        return positive_dest
    elif negative_count > positive_count:
        return negative_dest
    else:
        return neutral_dest


def data_exists_router(
    output: Any, output_type: str, exists_dest: str, missing_dest: str
) -> str:
    """
    Routes based on whether the output contains actual data or is empty/None.

    Args:
        output: The output
        output_type: The type of the output
        exists_dest: Destination key when data exists
        missing_dest: Destination key when data is missing

    Returns:
        The appropriate destination key based on data existence
    """
    if output is None:
        return missing_dest

    if isinstance(output, str) and not output.strip():
        return missing_dest

    if hasattr(output, "__len__") and len(output) == 0:
        return missing_dest

    return exists_dest


def custom_router(
    output: Any, output_type: str, condition_fn, true_dest: str, false_dest: str
) -> str:
    """
    Routes based on a custom condition function.

    Args:
        output: The output
        output_type: The type of the output
        condition_fn: Function that takes (output, output_type) and returns bool
        true_dest: Destination key when condition is True
        false_dest: Destination key when condition is False

    Returns:
        The appropriate destination key based on custom condition
    """
    try:
        if condition_fn(output, output_type):
            return true_dest
        else:
            return false_dest
    except Exception:
        # Default to false case on error
        return false_dest
