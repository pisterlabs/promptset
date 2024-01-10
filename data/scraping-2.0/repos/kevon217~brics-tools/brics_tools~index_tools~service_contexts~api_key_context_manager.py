# api_key_context_manager.py

from contextlib import contextmanager
import openai


@contextmanager
def use_openai_api_key(api_key):
    original_api_key = openai.api_key  # Store the original API key
    openai.api_key = api_key  # Set the new API key
    try:
        yield
    finally:
        openai.api_key = original_api_key  # Revert to the original API key
