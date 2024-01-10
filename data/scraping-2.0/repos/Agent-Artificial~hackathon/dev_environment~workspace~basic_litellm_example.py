import openai
import json
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

local_model_url = os.getenv("local_model_url")
if (local_model_url): 
    print("local model: ", local_model_url)
    api_base = local_model_url

openai.api_base = api_base
openai.api_key = "temp-key"
print(openai.api_base)


print(f'LiteLLM: response from proxy with streaming')
response = openai.ChatCompletion.create(
    model="ollama/mistral", 
    messages = [
        {
            "role": "user",
            "content": "this is a test request, acknowledge that you got it"
        }
    ],
    stream=True
)

for chunk in response:
    chunk_dict = chunk.to_dict()
    content = chunk_dict["choices"][0]["delta"]["content"]
    print(f'{content}', end='')

print("\n")