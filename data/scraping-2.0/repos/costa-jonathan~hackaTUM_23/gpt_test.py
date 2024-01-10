from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a sad mathematican that doesn't see much sense in life."},
#     {"role": "user", "content": "Explain what position math has in your life."},
#   ]
# )

# print(completion.choices[0].message)