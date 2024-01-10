# src/vecta_learn/patch.py

from typing import Callable
import openai

from .. import settings

ORIGINAL_API_BASE = openai.api_base
ORIGINAL_COMPLETION = openai.Completion.create
ORIGINAL_CHAT_COMPLETION = openai.ChatCompletion.create


def _patch_completion(original_completion: Callable) -> Callable:
    def new_completion(*args, **kwargs):
        headers = kwargs.get("headers", {})
        headers["Helicone-Auth"] = f"Bearer {settings.HELICONE_API_KEY}"
        headers["Helicone-Property-Project"] = settings.PROJECT_NAME
        kwargs["headers"] = headers
        return original_completion(*args, **kwargs)

    return new_completion


def patch():
    """Patch openai completion endpoints to use Helicone as a proxy."""
    openai.api_base = "https://oai.hconeai.com/v1"
    openai.Completion.create = _patch_completion(ORIGINAL_COMPLETION)
    openai.ChatCompletion.create = _patch_completion(ORIGINAL_CHAT_COMPLETION)


def unpatch():
    """Unpatch openai completion endpoints."""
    openai.api_base = ORIGINAL_API_BASE
    openai.Completion.create = ORIGINAL_COMPLETION
    openai.ChatCompletion.create = ORIGINAL_CHAT_COMPLETION
