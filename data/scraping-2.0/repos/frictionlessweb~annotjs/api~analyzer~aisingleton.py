import openai
from os import environ

# Configure OpenAI. If these environment variables aren't set, the program
# should crash with a KeyError.
openai.organization = environ["OPENAI_ORGANIZATION"]
openai.api_key = environ["OPENAI_API_KEY"]
