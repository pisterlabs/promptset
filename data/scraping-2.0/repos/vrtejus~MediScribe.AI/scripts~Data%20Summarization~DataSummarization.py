import os
import openai
from dotenv import load_dotenv

load_dotenv()

# secret api key
openai.api_key = os.getenv("gpt_key")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the following conversation bewteen a doctor and their \
     patient. Use an easy-to-read bullet point format and summarize medical terms in everyday terms:"}
  ]
)

print(completion.choices[0].message)