import openai
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('chatgpt_key')

openai.api_key = OPENAI_API_KEY

response = openai.Completion.create(
  engine="text-davinci-003",  # Use "text-davinci-003" for ChatGPT
  prompt="What is 1+1",
  max_tokens=50
)

answer = response.choices[0].text.strip()
print("Answer:", answer)