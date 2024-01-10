import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Generic Chain
#
# Text Completion in a Chain with Prompt Template
#
# OPEN AI API - POST https://api.openai.com/v1/completions
# ---------------------------

llm = OpenAI(temperature=0.9)

promptemplate = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=promptemplate)
result = chain.run("colorful socks")

print(result)
