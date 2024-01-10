#!/usr/bin/env python3

from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.globals import set_debug

llm = ChatOpenAI(model="gpt-4")

prompt_country = ChatPromptTemplate.from_template(
    "Which country is {city} located in? Return just name of the country."
)

prompt_population = ChatPromptTemplate.from_template(
    "How many people lives in {city}? Return sole number, without any text."
)

prompt_translate = ChatPromptTemplate.from_template(
    "Translate to Polish: {text}"
)

def final_format(input):
    return (
        f"There are approx. {input['population']} people "
        f"living in {input['city']}, "
        f"which is located in {input['country']}."
    )

def roundtrip(input):
    n = int(input['population'].replace(',',''))
    approx = round(n, -3)
    return str(approx)

def bind_input(input_key):
    return RunnableMap(**{input_key: RunnablePassthrough()})

chain = (
    bind_input('city')
    | {
        "city": itemgetter("city"),
        "country": (prompt_country | llm | StrOutputParser()),
        "population": (
            prompt_population
            | llm
            | StrOutputParser()
            | bind_input("population")
            | roundtrip
        ),
    }
    | final_format 
    | bind_input('text')
    | prompt_translate
    | llm
    | StrOutputParser()
)

set_debug(False)

print()
print(chain.invoke("Radom"))
print()
