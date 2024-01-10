import openai
import os
from dotenv import load_dotenv


def init_openai():
    load_dotenv()
    openai.api_key = os.getenv("API_KEY")
    print("Key loaded.")


if __name__ == '__main__':
    init_openai()

