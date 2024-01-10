import openai
import os
from dotenv import load_dotenv


def fetch_id():
    openai.api_key = os.getenv("OPENAI_KEY")
    data = openai.File.list()
    return data["data"][0]["id"]

if __name__ == "__main__":
    env_path = os.path.join(os.path.dirname(__file__), "..", "secret", ".env")
    load_dotenv(dotenv_path=env_path)
    openai.api_key = os.getenv("OPENAI_KEY")
    print(openai.File.create(file=open(os.path.join(os.path.dirname(__file__), "combined_answers.jsonl")), purpose='answers'))
    print(fetch_id())