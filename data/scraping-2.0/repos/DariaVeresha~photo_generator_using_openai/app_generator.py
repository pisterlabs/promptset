import json
from pathlib import Path
import openai
from key_OpenAi import OPENAI_API_KEY

PROMPT = input("Enter your request: ")
openai.api_key = OPENAI_API_KEY

#generation photo with help openai
response = openai.Image.create(
    prompt=PROMPT,
    n=1,
    size="1024x1024",
)

document = response["data"][0]["url"]
print(document)
