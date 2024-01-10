import dotenv, re, os
from time import sleep
from typing import List
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document

from .item import Item

dotenv.load_dotenv()

# The initialisation of the llm should be:
#    llm = OpenAI(model="davinci-instruct-beta")
# But to stay under the 3RPM we will use the sleep function
def llm (query: str) -> str :
    sleep(20)
    model = OpenAI(model="text-davinci-003")
    return model(query)