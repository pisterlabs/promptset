#!/usr/bin/env python3

from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.globals import set_debug

prompt = ChatPromptTemplate.from_template(
    "What country is {city} located in?"
)

llm = ChatOpenAI()

chain = prompt | llm | StrOutputParser()

res = chain.invoke({"city": "Hiroshima"})

print()
print(res)
print()
