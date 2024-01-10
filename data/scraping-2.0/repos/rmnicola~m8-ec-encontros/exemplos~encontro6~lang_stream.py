import os
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=512)

for chunk in llm.stream("Escreva uma música sobre monads em Python"):
    print(chunk, end="", flush=True)
