from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv("API_KEY")
response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="say hey Sox or ask a question: '{}'",
  max_tokens=1500
)
