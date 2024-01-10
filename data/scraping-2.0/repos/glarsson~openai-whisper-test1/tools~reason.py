from openai import OpenAI
from colorama import Fore
import numpy as np
import sys
sys.path.append('C:\SOURCE\GERRY\openai-whisper-test1')
from main_capabilities.speak import text_to_speech


# Read API key from a file
with open('secret_apikey.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)

# specify the OpenAI model to use:
# https://platform.openai.com/docs/models/gpt-3-5
# 16k token limit on "gpt-3.5-turbo-16k-0613"
# 16k token limit on "gpt-3.5-turbo-1106" (newest release as of dec-1-2023)
gpt_model = "gpt-3.5-turbo-1106"

# Our first AI 'assistant' role and its speciality
openai_specialization = "Just a regular dude"
# The base premise of what we are trying to do
base_premise = "You will just behave like a regular dude."

####

chat = client.chat.completions.create(
  model=gpt_model,
  max_tokens=2048,
  messages=[
    {"role": "system", "content": openai_specialization + base_premise,
     "role": "user", "content": "how fast can you speak now?"
    }
    ]
)

chat_response_text = chat.choices[0].message.content
print(f"{Fore.CYAN}{chat_response_text}{Fore.RESET}\n")
text_to_speech(chat_response_text)

