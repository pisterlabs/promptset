import openai
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()

# get api key from environment variable
api_key = os.environ["OPENAI_API_KEY"]
print(api_key)

openai.api_key = api_key
#print(openai.Engine.list())

response = openai.Completion.create(
    model='gpt-3.5-turbo-instruct',
    prompt="Give me two reasons to learn OpenAI API with pythton ",
    max_tokens=300)

print(response['choices'][0]['text'])