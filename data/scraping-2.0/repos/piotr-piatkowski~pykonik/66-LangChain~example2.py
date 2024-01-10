#!/usr/bin/env python3

from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.globals import set_debug
from langchain.agents import Tool

llm = ChatOpenAI()

country_prompt = ChatPromptTemplate.from_template(
    "What country is {city} located in? Answer with one word"
)

capital_prompt = ChatPromptTemplate.from_template(
    "What is the capital of {country}? Answer with one word"
)

def bind_input(input_key):
    return RunnableMap(**{input_key: RunnablePassthrough()})

chain = (
    bind_input("city")
    | country_prompt | llm | StrOutputParser()
    | bind_input("country")
    | capital_prompt | llm | StrOutputParser()
)

res = chain.invoke("sosnowiec")

print()
print(res)
print()
