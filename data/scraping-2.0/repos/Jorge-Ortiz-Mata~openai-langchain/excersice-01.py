from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name():
  llm = OpenAI(temperature=0.5)

  name = llm("I have a dog and I don not know how to call it. Give 5 suggestions of dog's name")

  return name

if __name__ == "__main__":
  print(generate_pet_name())
