import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Text Completion with Prompt Template
# OPEN AI API - POST https://api.openai.com/v1/completions
# ---------------------------


llm = OpenAI(temperature=0.9)

promptemplate = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
prompt = promptemplate.format(product="colorful socks")

print(llm(prompt))
