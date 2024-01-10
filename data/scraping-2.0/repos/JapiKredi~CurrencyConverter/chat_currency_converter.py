import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY.txt")

messages = [
    {"role": "user", "content": "What is the exchange rate between USD and INR?"}]

response = openai.ChatCompletion.create(
    model = 'gpt3.5-turbo',
    messages=messages)

output = response.choices[0].message.content

print(output)






