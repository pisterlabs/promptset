import os
import logging

from dotenv import load_dotenv, find_dotenv
from typing import List
from pydantic import BaseModel, Field

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# pip install -U wikipedia

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__file__)

_ = load_dotenv(find_dotenv())


# account for deprecation of LLM model
import datetime

# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


"""
Pydantic Syntax
Pydantic data classes are a blend of Python's data classes with the validation power of Pydantic.

They offer a concise way to define data structures while ensuring that the data adheres to specified types and constraints.

In standard python you would create a class like this:
"""


class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


foo = User(name="Joe", age=32, email="joe@gmail.com")

foo.name

# In pydantic we would do


class pUser(BaseModel):
    name: str
    age: int
    email: str


foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")

# this would fail
#
foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")

## Pydantic to OpenAI function definition


class WeatherSearch(
    BaseModel
):  # with pydantic your classes and fields must have descriptions

    """Call this with an airport code to get the weather at that airport"""

    airport_code: str = Field(description="airport code to get weather for")


from langchain.utils.openai_functions import convert_pydantic_to_openai_function

weather_function = convert_pydantic_to_openai_function(WeatherSearch)

weather_function

"""
{'name': 'WeatherSearch',
 'description': 'Call this with an airport code to get the weather at that airport',
 'parameters': {'title': 'WeatherSearch',
  'description': 'Call this with an airport code to get the weather at that airport',
  'type': 'object',
  'properties': {'airport_code': {'title': 'Airport Code',
    'description': 'airport code to get weather for',
    'type': 'string'}},
  'required': ['airport_code']}}
"""


from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()
model.invoke("what is the weather in SF today?", functions=[weather_function])

"""
AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{\n  "airport_code": "SFO"\n}'}})
"""

model_with_function = model.bind(functions=[weather_function])

model_with_function.invoke("what is the weather in sf?")

## Forcing it to use a function
#
# We can force the model to use a function

model_with_forced_function = model.bind(
    functions=[weather_function], function_call={"name": "WeatherSearch"}
)

model_with_forced_function.invoke("what is the weather in sf?")

model_with_forced_function.invoke("hi!")

"""
AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{\n  "airport_code": "SFO"\n}'}})

"""

"""
## Using in a chain

We can use this model bound to function in a chain as we normally would
"""

from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant"), ("user", "{input}")]
)

chain = prompt | model_with_function

chain.invoke({"input": "what is the weather in sf?"})

"""
AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{\n  "airport_code": "SFO"\n}'}})

"""

## Using multiple functions
#
# Even better, we can pass a set of function and let the LLM decide which to use based on the question context.
#


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

"""
AIMessage(content='', additional_kwargs={'function_call': {'name': 'WeatherSearch', 'arguments': '{\n  "airport_code": "SFO"\n}'}})

"""

model_with_functions.invoke("what are three songs by taylor swift?")

"""
AIMessage(content='', additional_kwargs={'function_call': {'name': 'ArtistSearch', 'arguments': '{\n  "artist_name": "taylor swift",\n  "n": 3\n}'}})

"""

model_with_functions.invoke("hi!")

"""

AIMessage(content='Hello! How can I assist you today?')


"""
