import os
import typing
import logging

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# pip install -U wikipedia
#
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

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


model = ChatOpenAI(temperature=0)


class Tagging(BaseModel):
    """Tag the piece of text with particular info."""

    sentiment: str = Field(
        description="sentiment of text, should be `pos`, `neg`, or `neutral`"
    )
    language: str = Field(description="language of text (should be ISO 639-1 code)")


tagging_functions = [convert_pydantic_to_openai_function(Tagging)]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}"),
    ]
)

model_with_functions = model.bind(
    functions=tagging_functions, function_call={"name": "Tagging"}
)

tagging_chain = prompt | model_with_functions

tagging_chain.invoke({"input": "I love langchain"})

tagging_chain.invoke({"input": "I love langchain"})

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

tagging_chain.invoke({"input": "non mi piace questo cibo"})

## Extraction
## Extraction is similar to tagging, but used for extracting multiple pieces of information.

from typing import Optional


class Person(BaseModel):
    """Information about a person."""

    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")


class Information(BaseModel):
    """Information to extract."""

    people: List[Person] = Field(description="List of info about people")


extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(
    functions=extraction_functions, function_call={"name": "Information"}
)

extraction_model.invoke("Joe is 30, his mom is Martha")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract the relevant information, if not explicitly provided do not guess. Extract partial info",
        ),
        ("human", "{input}"),
    ]
)

extraction_chain = prompt | extraction_model

extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})

extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()

extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

extraction_chain = (
    prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
)

extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})

## Doing it for real
## Doing it for real
#
# We can apply tagging to a larger body of text.
#
# For example, let's load this blog post and extract tag information from a sub-set of the text.

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
documents = loader.load()
doc = documents[0]
page_content = doc.page_content[:10000]

print(page_content[:1000])


class Overview(BaseModel):
    """Overview of a section of text."""

    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(
        description="Provide the language that the content is written in."
    )
    keywords: str = Field(description="Provide keywords related to the content.")


overview_tagging_function = [convert_pydantic_to_openai_function(Overview)]
tagging_model = model.bind(
    functions=overview_tagging_function, function_call={"name": "Overview"}
)
tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()

tagging_chain.invoke({"input": page_content})


class Paper(BaseModel):
    """Information about papers mentioned."""

    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""

    papers: List[Paper]


paper_extraction_function = [convert_pydantic_to_openai_function(Info)]
extraction_model = model.bind(
    functions=paper_extraction_function, function_call={"name": "Info"}
)
extraction_chain = (
    prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
)

extraction_chain.invoke({"input": page_content})

template = """A article will be passed to you. Extract from it all papers that are mentioned by this article.

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

extraction_chain = (
    prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
)

extraction_chain.invoke({"input": page_content})

extraction_chain.invoke({"input": "hi"})

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)

splits = text_splitter.split_text(doc.page_content)

len(splits)


def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


flatten([[1, 2], [3, 4]])

print(splits[0])

from langchain.schema.runnable import RunnableLambda

prep = RunnableLambda(lambda x: [{"input": doc} for doc in text_splitter.split_text(x)])

prep.invoke("hi")

chain = prep | extraction_chain.map() | flatten

chain.invoke(doc.page_content)
