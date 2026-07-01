import re

# Trailing tokens a caller can append to a gpt-5-class model id to force the
# classic /v1/chat/completions path instead of the /v1/responses path.
_CHAT_ROUTE_OVERRIDE_SUFFIXES = (":chat", "#chat", "|chat")


def strip_route_override(model_name):
    """
    Remove a trailing chat-route override token (:chat/#chat/|chat) from a model id
    so the cleaned name can be sent to the API. Routing decisions must be made on
    the RAW name (via is_reasoning_model) before cleaning.
    """
    if not model_name:
        return model_name
    lowered = model_name.lower().strip()
    for suffix in _CHAT_ROUTE_OVERRIDE_SUFFIXES:
        if lowered.endswith(suffix):
            return model_name.strip()[: -len(suffix)]
    return model_name


def is_reasoning_model(model_name):
    """
    Check if the model is an OpenAI reasoning model (GPT-5 or higher) that must
    use the /v1/responses endpoint instead of /v1/chat/completions.

    The match is anchored on the OpenAI ``gpt-<major>`` prefix so that arbitrary
    strings that merely *contain* ``gpt-5`` (e.g. a proxy/deployment name such as
    ``my-gpt-5-proxy``) are not misrouted. Optionally the caller can force the
    chat-completions path for a gpt-5-class deployment by appending the token
    ``:chat`` / ``#chat`` (or passing the model as ``gpt-5...|chat``).

    Args:
        model_name: Model name string or None

    Returns:
        bool: True if the model is GPT-5 or higher, False otherwise
    """
    if not model_name:
        return False

    model_lower = model_name.lower().strip()

    # Explicit override to force the classic chat-completions path.
    if any(model_lower.endswith(suffix) for suffix in _CHAT_ROUTE_OVERRIDE_SUFFIXES):
        return False

    # Anchored match on the OpenAI gpt-<major>[.minor] prefix.
    match = re.match(r"^gpt-(\d+)", model_lower)
    if not match:
        return False

    try:
        return int(match.group(1)) >= 5
    except (ValueError, IndexError):
        return False


# Claude model families that REMOVED sampling parameters (temperature/top_p/top_k).
# Sending those parameters to these models returns an HTTP 400, so the input
# builder must omit them. Opus 4.6 / Sonnet 4.x still accept temperature.
_CLAUDE_NO_SAMPLING_TOKENS = (
    "opus-4-7",
    "opus-4-8",
    "opus-4-9",
    "fable-5",
    "mythos-5",
    "mythos-preview",
)


def claude_rejects_sampling_params(model_name):
    """
    Return True for Claude models that reject temperature/top_p/top_k (they were
    removed on Opus 4.7+ and the Fable/Mythos family). For these models the
    Anthropic input builder must not emit sampling parameters.

    Args:
        model_name: Model name string or None

    Returns:
        bool: True if sampling params must be omitted for this model.
    """
    if not model_name:
        return False
    model_lower = model_name.lower()
    return any(token in model_lower for token in _CLAUDE_NO_SAMPLING_TOKENS)

