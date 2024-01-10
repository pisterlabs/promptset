from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from typing import List
from dotenv import load_dotenv
load_dotenv()


# 1. Chain definition


class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return text.strip().split(", ")


system_template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and noting more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template),
])
category_chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()

# 2. Server App definition

app = FastAPI(
    title="LangChain Server",
    version="0.1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 3. Addding chain route
add_routes(
    app,
    category_chain,
    path="/category_chain",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
