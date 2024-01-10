# Databricks notebook source
# MAGIC %md
# MAGIC # LangChain Expression Language (LCEL)

# COMMAND ----------

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# COMMAND ----------

#!pip install pydantic==1.10.8

# COMMAND ----------

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Chain

# COMMAND ----------

prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

# COMMAND ----------

chain = prompt | model | output_parser

# COMMAND ----------

chain.invoke({"topic": "bears"})

# COMMAND ----------

## More complex chain

And Runnable Map to supply user-provided inputs to the prompt.

# COMMAND ----------

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

# COMMAND ----------

vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# COMMAND ----------

retriever.get_relevant_documents("where did harrison work?")

# COMMAND ----------

retriever.get_relevant_documents("what do bears like to eat")

# COMMAND ----------

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# COMMAND ----------

from langchain.schema.runnable import RunnableMap

# COMMAND ----------

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

# COMMAND ----------

chain.invoke({"question": "where did harrison work?"})

# COMMAND ----------

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

# COMMAND ----------

inputs.invoke({"question": "where did harrison work?"})

# COMMAND ----------

## Bind

and OpenAI Functions

# COMMAND ----------

functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]

# COMMAND ----------

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(temperature=0).bind(functions=functions)

# COMMAND ----------

runnable = prompt | model

# COMMAND ----------

runnable.invoke({"input": "what is the weather in sf"})

# COMMAND ----------

functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]

# COMMAND ----------

model = model.bind(functions=functions)

# COMMAND ----------

runnable = prompt | model

# COMMAND ----------

runnable.invoke({"input": "how did the patriots do yesterday?"})

# COMMAND ----------

# MAGIC %md
# MAGIC Fallbacks

# COMMAND ----------

from langchain.llms import OpenAI
import json

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: Due to the deprecation of OpenAI's model `text-davinci-001` on 4 January 2024, you'll be using OpenAI's recommended replacement model `gpt-3.5-turbo-instruct` instead.

# COMMAND ----------

simple_model = OpenAI(
    temperature=0, 
    max_tokens=1000, 
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads

# COMMAND ----------

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"

# COMMAND ----------

simple_model.invoke(challenge)

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: The next line is expected to fail.

# COMMAND ----------

simple_chain.invoke(challenge)

# COMMAND ----------

model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads

# COMMAND ----------

chain.invoke(challenge)

# COMMAND ----------

final_chain = simple_chain.with_fallbacks([chain])

# COMMAND ----------

final_chain.invoke(challenge)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interface

# COMMAND ----------

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# COMMAND ----------

chain.invoke({"topic": "bears"})

# COMMAND ----------

chain.batch([{"topic": "bears"}, {"topic": "frogs"}])

# COMMAND ----------

for t in chain.stream({"topic": "bears"}):
    print(t)

# COMMAND ----------

response = await chain.ainvoke({"topic": "bears"})
response

# COMMAND ----------


