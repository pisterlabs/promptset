from pathlib import Path
import openai


def get_gpt3_api_key(api_key_file_path: Path) -> str:
    """Helper to read in the api key from a txt file."""
    with api_key_file_path.open('r') as f:
        return f.read().strip()


def set_gpt3_api_key(api_key: str):
    """Small helper to set the api key for openai."""
    openai.api_key = api_key
