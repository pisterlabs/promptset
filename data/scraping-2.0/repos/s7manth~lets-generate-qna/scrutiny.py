"""
An attempt towards LLM based scrutiny to evaluate the effectiveness of generated MCQs. 
"""

import openai
import os

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

SCRUTINY_PROMPT_FILE_PATH = "./prompts/scrutiny-prompt.txt"
TO_SCRUTINY = "./selectedresponses/06-11-2023 11:18:13 (vectordb).txt"
RESPONSE_FILE_PATH = lambda timestamp: f"./scrutiny/{timestamp} (scrutiny).txt"
TOPIC = "Feed Forward Neural Networks in NLP using PyTorch"

prompt = str()
with open(SCRUTINY_PROMPT_FILE_PATH, "r+") as file:
  prompt = file.read()
  prompt = prompt.replace("<<TOPIC>>", TOPIC)

with open(TO_SCRUTINY, "r+") as file:
  prompt = prompt.replace("<<QUESTIONS>>", file.read())

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORGANIZATON"))

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {
      "role": "system",
      "content": f"You are a highly skilled subject matter expert in {TOPIC} and a creative question architect. You think critically before answering any question.",
    },
    {
      "role": "user", 
      "content": prompt
    },
  ],
  temperature=0,
  max_tokens=4096,
)

response = completion.choices[0].message.content

with open(RESPONSE_FILE_PATH(datetime.now().strftime("%d-%m-%Y %H:%M:%S")), "w+") as file:
  file.write(response)

