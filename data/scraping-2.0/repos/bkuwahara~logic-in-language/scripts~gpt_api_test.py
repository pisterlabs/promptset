# Be in right directory 
import os
os.chdir("/w/246/ikozlov/csc2542-project/")

from dotenv import load_dotenv
load_dotenv()  # This loads the environment variables from a .env file.

# Get OpenAI API Key
from openai import OpenAI 
client = OpenAI() 

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)