import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pprint import pprint

_ = load_dotenv(find_dotenv())  # read local .env file


class OpenAPAPIExplorer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def list_models(self):
        return self.client.models.list()


if __name__ == "__main__":
    obj = OpenAPAPIExplorer()
    response = obj.list_models()
    pprint(response.data)
