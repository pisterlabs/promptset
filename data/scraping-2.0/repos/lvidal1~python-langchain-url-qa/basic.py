from os import getenv
from langchain.llms import OpenAI
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate

load_dotenv()

llm  = OpenAI(openai_api_key=getenv("OPENAI_API_KEY"), temperature=0.9)
prediction = llm.predict("What would be 3 good names for white cat that has brown paws?")

print(f"Response: {prediction}")


def getPrompt(pet):
  prompt = PromptTemplate.from_template("What would be 3 good names for white {pet}?")
  return prompt.format(pet=pet)

