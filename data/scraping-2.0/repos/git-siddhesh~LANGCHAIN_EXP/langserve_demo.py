'''
We can now combine all these into one chain. 
This chain will take input variables, 
pass those to a prompt template to create a prompt, 
pass the prompt to a language model, and then 
pass the output through an (optional) output parser. 
This is a convenient way to bundle up a modular piece of logic. 
Let's see it in action!
'''

'''LangServe helps developers deploy LCEL chains as a REST API. 
The library is integrated with FastAPI and uses pydantic for data validation.
'''

from dotenv import load_dotenv
load_dotenv()

from typing import List
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.chat_models import ChatOpenAI

from langserve import add_routes

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str) -> List[str]:
        return text.strip().split(", ")

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
category_chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()
print(category_chain.invoke({"text": "colors"}))
# >> ['red', 'blue', 'green', 'yellow', 'orange']

#-------------------------------------------------------------------------------------------------------------------

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

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)


