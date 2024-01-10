import os
import openai
import click

# develop a command-line tool that can assist us with Linux commands through conversation.
# Click documentation: https://click.palletsprojects.com/en/8.1.x/


def init_api():
    ''' Load API key from .env file'''
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ["API_KEY"]
    openai.organization = os.environ["ORG_ID"]


init_api()

# response = openai.Embedding.create(
#     model="text-embedding-ada-002",
#     input="I am a programmer",
# )

# print(response)
# print(response["embedding"])

# embedding for multiple inputs

response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=["I am a programmer", "I am a writer"],
)

for data in response["data"]:
    print(data["embedding"])

# Each input should not exceeed 8192 tokens