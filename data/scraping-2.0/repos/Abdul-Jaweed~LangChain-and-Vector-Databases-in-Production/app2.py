from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=api_key,
    model="text-davinci-003",
    temperature=0.9
)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that make {product}?"
)

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Running the chain only specifiying the input variable

print(chain.run("eco-friendly water bottles"))