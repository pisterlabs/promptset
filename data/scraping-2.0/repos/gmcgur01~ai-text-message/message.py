import sys
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("API_KEY")

if len(sys.argv) < 2:
    sys.exit("No input provided")

prompt = "You are a emoji translator. Repeat what I say using only emojis."

system = {"role": "system", "content": prompt}
user = {"role": "user", "content": sys.argv[1]}

response = openai.ChatCompletion.create(
    model= "gpt-4",
    messages= [system, user],
    temperature= 0.8,
    max_tokens= 256
)

print(response.choices[0].message.content, end="")