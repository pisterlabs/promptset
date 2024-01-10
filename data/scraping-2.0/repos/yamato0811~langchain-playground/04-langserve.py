#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langserve import add_routes

import settings

OPEN_AI_API_KEY = settings.OPEN_AI_API_KEY

# 1. Chain definition

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

chat_model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model_name="gpt-3.5-turbo")

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate {num} objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

category_chain = chat_prompt | chat_model | CommaSeparatedListOutputParser()

# 2. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 3. Adding chain route
add_routes(
    app,
    category_chain,
    path="/category_chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)