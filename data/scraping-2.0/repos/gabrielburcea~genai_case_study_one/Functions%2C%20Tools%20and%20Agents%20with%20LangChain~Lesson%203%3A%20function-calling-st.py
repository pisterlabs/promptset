# Databricks notebook source
OpenAI Function Calling In LangChain

import os
import openai
​
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from typing import List
from pydantic import BaseModel, Field
Pydantic Syntax
Pydantic data classes are a blend of Python's data classes with the validation power of Pydantic.
They offer a concise way to define data structures while ensuring that the data adheres to specified types and constraints.
In standard python you would create a class like this:

class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

foo = User(name="Joe",age=32, email="joe@gmail.com")

foo.name

foo = User(name="Joe",age="bar", email="joe@gmail.com")

foo.age

class pUser(BaseModel):
    name: str
    age: int
    email: str

foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")

foo_p.name
Note: The next cell is expected to fail.

foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")

class Class(BaseModel):
    students: List[pUser]

obj = Class(
    students=[pUser(name="Jane", age=32, email="jane@gmail.com")]
)

obj
Pydantic to OpenAI function definition

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

from langchain.utils.openai_functions import convert_pydantic_to_openai_function

weather_function = convert_pydantic_to_openai_function(WeatherSearch)

weather_function

class WeatherSearch1(BaseModel):
    airport_code: str = Field(description="airport code to get weather for")
Note: The next cell is expected to generate an error.

convert_pydantic_to_openai_function(WeatherSearch1)

class WeatherSearch2(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str

convert_pydantic_to_openai_function(WeatherSearch2)

from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()

model.invoke("what is the weather in SF today?", functions=[weather_function])

model_with_function = model.bind(functions=[weather_function])

model_with_function.invoke("what is the weather in sf?")
Forcing it to use a function
We can force the model to use a function

model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})

model_with_forced_function.invoke("what is the weather in sf?")

model_with_forced_function.invoke("hi!")
Using in a chain
We can use this model bound to function in a chain as we normally would

from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])

chain = prompt | model_with_function

chain.invoke({"input": "what is the weather in sf?"})
Using multiple functions
Even better, we can pass a set of function and let the LLM decide which to use based on the question context.

class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")

functions = [
    convert_pydantic_to_openai_function(WeatherSearch),
    convert_pydantic_to_openai_function(ArtistSearch),
]

model_with_functions = model.bind(functions=functions)

model_with_functions.invoke("what is the weather in sf?")

model_with_functions.invoke("what are three songs by taylor swift?")

model_with_functions.invoke("hi!")
