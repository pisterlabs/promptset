# -*- coding: utf-8 -*-
# Imports
import asyncio
import os
import openai

from typing import List, Optional
from pydantic import BaseModel, Field

from langchain.chains.openai_functions.extraction import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
# from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']


# App

# Pydantic is an easy way to define a schema
class Person(BaseModel):
    """Information about people to extract."""

    name: str
    age: Optional[int]

# Main function to extract information
def extract_information():
    # Make sure to use a recent model that supports tools
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

    return create_extraction_chain_pydantic(Person, llm)


if __name__ == "__main__":
    text = "My name is John and I am 20 years old. My name is sally and I am 30 years old."
    chain = extract_information()
    print(chain.invoke({"input": text})["text"])

    async def extract_information_async(message: str):
        return chain.invoke({"input": message})["text"]

    async def main():
        res = await extract_information_async(text)
        print(res)

    asyncio.run(main())
