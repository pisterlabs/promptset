import openai
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dotenv import load_dotenv
load_dotenv()

class ChatOpenAI():
  openai.organization = os.getenv("OPENAI_ORGANIZATION")
  openai.api_key = os.getenv("OPENAI_API_KEY")
  openai.Model.list()
  
  def __init__(self) -> None:

    self.model = 'gpt-3.5-turbo'
    
    self.prompt_list = [
      "create a challeging and not easy short sentence",
      "create a short sentence about an animal at a place",
      "create a short sentence about a famous person at a place",
    ]

  def generate_message(self, mode, text = None):

    if mode == 'THEME':
      content = random.choice(self.prompt_list) + ", with the theme " + text if text else \
                random.choice(self.prompt_list) #+ random.choice(self.prompt_art)
      
      messages=[
        {"role": "system", "content": f"you are a robot that writes small sentences to generate images to play guessing games"}, #  with the theme: {text}
        {"role": "user", "content": content},
      ]
    
    elif mode == 'TRANSLATE':
      content = f'Me retorne somente o texto entre aspas traduzido para o português: "{text}"'

      messages=[
        {"role": "system", "content": f"you are a robot that only returns a text translated from English to Portuguese"},
        {"role": "user", "content": content},
      ]

    print("ChatGPT Prompt:", content)

    return messages


  def make_text(self, mode, text):
    
    completion = openai.ChatCompletion.create(
      model = self.model,
      temperature=0.9,
      frequency_penalty= 2.0,
      presence_penalty=2.0,
      messages = self.generate_message(mode = mode, text = text),
    )

    return completion['choices'][0]['message']['content']

if __name__ == "__main__":
  chatopenai = ChatOpenAI()
  res = chatopenai.make_text()
  print(res)