import sys
import openai
from pathlib import Path
import time


# function to load API key from file
API_KEY_PATH = "api-key.txt"


def load_api_key():
    key_path = Path(API_KEY_PATH)

    print("Loading API Key...")

    if not key_path.is_file():
        print(f"No API key file found at {API_KEY_PATH}. Please add it.")
        sys.exit()

    with key_path.open() as key_file:
        api_key = key_file.readline().strip()

    if not api_key:
        print(
            f"No API key found, or an invalid one was detected in {API_KEY_PATH}. Set a valid key."
        )
        sys.exit()

    openai.api_key = api_key

    print("API key loaded successfully!")

    return api_key


def load_models():
    models = openai.Model.list()

    # exclude these models
    exclude = set(
        [
            "instruct",
            "similarity",
            "if",
            "query",
            "document",
            "insert",
            "search",
            "edit",
            "dall-e",
            "tts",
        ]
    )

    print("Loading engines...")

    start_time = time.time()
    model_list = [
        str(model.id) for model in models.data if str(model.id) not in exclude
    ]
    end_time = time.time() - start_time

    print(f"Engines loaded successfully in {end_time} seconds")

    return model_list
