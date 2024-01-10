import os
import openai
from dotenv import load_dotenv

load_dotenv()

GPT3_API_KEY = os.getenv("GPT3_API_KEY")

openai.api_key = GPT3_API_KEY 

class gpt3_5_monologue_generator:

  def __init__(self) -> None:
     pass

  def generate_monologue(self, prompt):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.8,
      max_tokens=500,
      frequency_penalty=0.5,
      presence_penalty=0.0
    )["choices"][0]["message"]["content"]
    
    return response

    