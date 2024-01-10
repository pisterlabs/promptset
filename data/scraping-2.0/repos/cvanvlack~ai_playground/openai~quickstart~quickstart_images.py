from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(
  api_key=OPENAI_API_KEY,
)
response = client.images.generate(
  prompt="Calvin and Hobbes are hanging out in the forest",
  n=2,
  size="1024x1024"
)

print(response)