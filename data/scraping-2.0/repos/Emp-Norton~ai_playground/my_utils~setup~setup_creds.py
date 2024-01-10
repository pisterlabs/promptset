import os

import openai
from dotenv import load_dotenv


def set_apikey(key):
    # TODO look into failure return code here, might need try/ex
    openai.api_key = key
    return


def main():
    if load_dotenv():
        print("Loaded env vars")
        # TODO Convert to, or add, actual persistent logging.
        apikey = os.getenv('apikey')
        set_apikey(apikey)
    else:
        print("Failed to load env variables in setup.")
