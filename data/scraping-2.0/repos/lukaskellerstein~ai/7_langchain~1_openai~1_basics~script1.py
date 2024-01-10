import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# Text Completion
# OPEN AI API - POST https://api.openai.com/v1/completions
# ---------------------------


llm = OpenAI(temperature=0.9)
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))

# OR
# CHAIN IS NOT POSSIBLE WITHOUT A PROMPT TEMPLATE
# chain = LLMChain(llm=llm, prompt=text)
# result = chain.run("colorful socks")
# print(result)
