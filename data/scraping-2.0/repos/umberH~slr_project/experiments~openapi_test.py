from  dotenv import load_dotenv, find_dotenv
import os 

from os.path import join, dirname
load_dotenv()

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

SECRET_KEY = os.environ.get("openai.api_key_path")
# DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD")


import openai
from typing import List

def ask_chatgpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    return (response['choices'][0]['message']['content'])  
openai.api_key_path = None

openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_role = "you are an assistant to journalists"


def assistjournalist (facts: List[str],tone=str, length_words=int, style=str):
    facts = ", ".join(facts)
    prompt = f"{prompt_role}\nFACTS:{facts}\nTONE:{tone}\nLENGTH:{length_words} words\nSTYLE: {style}\n\n"
    return ask_chatgpt([{"role":"user", "content":prompt}])
# openai.api_key = API_secret_key


response = openai.Completion.create(
  engine="text-davinci-003",
  prompt="What dinosaurs lived in the cretaceous period?",
  max_tokens=60
)

print(response.choices[0].text.strip())