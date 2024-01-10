from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

print(chain.run("wireless headphones"))