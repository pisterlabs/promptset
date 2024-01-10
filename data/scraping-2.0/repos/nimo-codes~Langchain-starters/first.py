import openai
from langchain.llms import OpenAI
import os


os.environ['OPENAI_API_KEY'] = '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'
# openai.api_key = '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'


llm = OpenAI(temperature=0.6)
name = llm("I want to open a new gaming arcade. suggest me a great name")
print(name)
