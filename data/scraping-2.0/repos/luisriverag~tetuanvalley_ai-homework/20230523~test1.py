import os
from langchain.llms import OpenAI

OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')


llm = OpenAI(openai_api_key="OPENAI_API_KEY")


llm = OpenAI(temperature=0.9)

text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
