from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

llm = OpenAI(
    openai_api_key="sk-...")

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}                          
""", input_variables=["fruit"])

llm_chain = LLMChain(
    llm=llm,
    prompt=template,
    output_parser=StrOutputParser()
)

response = llm_chain("apple")

print(response)
