from operator import itemgetter
from typing import List, Tuple

from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.vectorstores import AstraDB
import os


from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
api_endpoint=os.environ['ASTRA_DB_API_ENDPOINT']
openai_api_key=os.environ["OPENAI_API_KEY"]

vstore = AstraDB(
    embedding=OpenAIEmbeddings(),
    collection_name="movies_description",
    api_endpoint=api_endpoint,
    token=token,
)

retriever = vstore.as_retriever(search_kwargs={"k": 3})

movie_template = """You are a ai assistant of a movie store and you are assisting customer to choose the best movies for him to watch.
    Be nice to the customer. Greet them, when required and be polite.
    You will also answering frequently asked questions, such as movie genre like comedy, horror, action, etc. 
    Include the movie description when responding with the list of movie recommendation.
    All the responses should be the same language as the user used. Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

CONTEXT:
{context}

QUESTION: {question}

YOUR ANSWER:"""

movie_prompt = PromptTemplate.from_template(movie_template)

llm = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | movie_prompt
    | llm
    | StrOutputParser()
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)