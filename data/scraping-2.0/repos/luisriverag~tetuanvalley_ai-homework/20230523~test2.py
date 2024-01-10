import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')


llm = OpenAI(openai_api_key="OPENAI_API_KEY")


llm = OpenAI(temperature=0.9)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

print(prompt.format(product="colorful socks"))

