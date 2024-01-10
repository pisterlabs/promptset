#
# Basic exxample from 'Langchain' official site with some modification
# - https://python.langchain.com/docs/get_started/quickstart
# 

from typing import List

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain

import os 

load_dotenv()

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

system_template = """
    You are a helpful assistant who generates comma separated lists.
    A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
    ONLY return a comma separated list, and nothing more."""

human_template = "{text}"


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template),
])

llm = ChatOpenAI(temperature=0.1, 
                 max_tokens=300, 
                 api_key= os.getenv("OPENAI_API_KEY"), 
                 verbose=True,
                 model="gpt-3.5-turbo")

parser = CommaSeparatedListOutputParser()

chain = chat_prompt | llm | parser

print(chain.invoke({"text": "colors"}))

# >> ['red', 'blue', 'green', 'yellow', 'orange']