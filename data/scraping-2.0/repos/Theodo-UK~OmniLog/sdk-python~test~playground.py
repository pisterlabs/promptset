import os

import openai
from dotenv import load_dotenv

from omnilogger import start_openai_listener

if __name__ == "__main__":
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("Missing environment variables")

    start_openai_listener()

    openai.api_key = OPENAI_API_KEY
    openai.Completion.create(  # type: ignore
        model="text-davinci-003",
        prompt="To be or not to be?",
        temperature=0.6,
    )
